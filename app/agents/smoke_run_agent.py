from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState, SmokeRunResult, SmokeCommand
from app.tools.command_safety import is_safe_argv
from app.tools.dependency_tool import install_spec_for_package, package_from_missing_dependency
from app.tools.llm import call_llm_json

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a software engineering assistant. Your job is to select the safest possible \
smoke test command for a research code repository.

Given the repository structure and README content, suggest ONE command that:
1. Uses python or pytest as the executable
2. Preferably runs with --help flag for safety
3. Targets the most appropriate entry-point script

Respond with a JSON object:
- "argv": list of strings — the command as a list of arguments (e.g. ["python", "demo.py", "--help"])
- "kind": one of "help", "demo", "pytest"
- "reason": a brief explanation of why this command was chosen

Rules:
- Only use scripts that exist in the provided candidate_scripts list
- Only use "python" or "pytest" as the first argument
- Prefer --help mode for safety
- Do NOT use shell metacharacters, pipes, or redirections"""


def classify_smoke_failure(stderr: str, stdout: str) -> tuple:
    """Classify smoke run failure type and extract key evidence line.

    Returns (failure_type, evidence_line).
    """
    combined = stderr + "\n" + stdout
    text = combined.lower()

    # 1. Module import errors
    if "modulenotfounderror" in text or "no module named" in text:
        match_lines = [l for l in combined.splitlines()
                       if "modulenotfounderror" in l.lower() or "no module named" in l.lower()]
        evidence = match_lines[0].strip() if match_lines else None
        return "missing_dependency", evidence

    # 2. CUDA errors — MUST come before checkpoint check
    cuda_markers = [
        "torch not compiled with cuda enabled",
        "cuda is not available",
        "cuda unavailable",
        "no cuda gpus are available",
        "no nvidia driver",
        "nvidia driver",
        "cuda driver",
        "cuda runtime",
        "cudnn",
        "gpu is not available",
        "no gpu",
        "cuda error",
        "cuda out of memory",
        "assertionerror: torch not compiled",
    ]
    if any(marker in text for marker in cuda_markers):
        match_lines = [l for l in combined.splitlines()
                       if any(marker in l.lower() for marker in cuda_markers)]
        evidence = match_lines[0].strip() if match_lines else None
        return "cuda_unavailable", evidence

    # 3. Checkpoint missing — only with negative indicators
    ckpt_ok, ckpt_evidence = _looks_like_missing_checkpoint(stderr, stdout)
    if ckpt_ok:
        return "missing_checkpoint", ckpt_evidence

    # 4. File not found (non-checkpoint)
    if "filenotfounderror" in text:
        match_lines = [l for l in combined.splitlines()
                       if "filenotfounderror" in l.lower()]
        evidence = match_lines[0].strip() if match_lines else None
        return "file_not_found", evidence

    if ("cannot allocate memory in static tls block" in text
            or ("importerror:" in text and (".so" in text or "libgomp" in text))):
        match_lines = [l for l in combined.splitlines()
                       if "importerror:" in l.lower() or "static tls" in l.lower() or "libgomp" in l.lower()]
        evidence = match_lines[0].strip() if match_lines else None
        return "runtime_linker_error", evidence

    camera_markers = ["could not read camera", "can't open camera", "cannot open camera",
                      "camera index out of range", "webcam", "videoio"]
    if any(marker in text for marker in camera_markers):
        match_lines = [l for l in combined.splitlines()
                       if any(marker in l.lower() for marker in camera_markers)]
        evidence = match_lines[0].strip() if match_lines else None
        return "missing_input_device", evidence

    if "no such file" in text:
        return "file_not_found", None

    if "permission denied" in text:
        return "permission_error", None

    if "timed out" in text:
        return "timeout", None

    if "usage:" in text and "error:" in text:
        match_lines = [l for l in combined.splitlines() if "error:" in l.lower()]
        evidence = match_lines[0].strip() if match_lines else None
        return "argument_error", evidence

    return "unknown", None


def _looks_like_missing_checkpoint(stderr: str, stdout: str) -> tuple[bool, str | None]:
    """Only return True when a negative indicator AND a checkpoint marker are present."""
    combined = stderr + "\n" + stdout
    text = combined.lower()

    checkpoint_markers = ["checkpoint", ".pth", ".ckpt", ".pt", "pretrained", "weights"]

    negative_markers = [
        "no such file",
        "filenotfounderror",
        "file not found",
        "not found",
        "missing",
        "cannot find",
        "could not find",
        "failed to download",
        "http error",
        "connectionerror",
        "download failed",
    ]

    if not any(c in text for c in checkpoint_markers):
        return False, None
    if not any(n in text for n in negative_markers):
        return False, None

    lines = [l for l in combined.splitlines()
             if any(c in l.lower() for c in checkpoint_markers)
             or any(n in l.lower() for n in negative_markers)]
    return True, lines[0].strip() if lines else None


class SmokeRunAgent:
    def __init__(self, timeout_minutes: int = 30, max_repair_attempts: int = 0):
        self.timeout_minutes = timeout_minutes
        self.max_repair_attempts = max(0, max_repair_attempts)

    def run(self, state: TaskState) -> TaskState:
        backend = state.backend

        # backend=none should never reach this agent, but handle gracefully
        if backend == "none":
            emit_progress("Run smoke command", "skipped", level="warning", detail="backend=none")
            state.smoke_run = SmokeRunResult(summary="Skipped (backend=none)")
            state.status = "smoke_ran"
            return state

        # Isolated backends require a successful environment build.
        if backend in {"docker", "venv", "conda"}:
            if not state.env_build or not state.env_build.build_success:
                emit_progress("Run smoke command", "skipped", level="warning", detail=f"{backend} environment did not build")
                state.smoke_run = SmokeRunResult(summary=f"Skipped because {backend} environment build did not succeed")
                state.status = "smoke_ran"
                return state

        if not state.repo_evaluation:
            emit_progress("Run smoke command", "skipped", level="warning", detail="repo was not evaluated")
            state.smoke_run = SmokeRunResult(summary="Skipped because repo was not evaluated")
            state.status = "smoke_ran"
            return state

        task_dir = Path(state.task_dir).resolve()
        run_dir = task_dir / "runs" / "smoke_001"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        command = self._select_smoke_command(state)

        if not command:
            emit_progress("Run smoke command", "no safe command found", level="warning")
            state.smoke_run = SmokeRunResult(summary="No safe smoke command found")
            state.status = "smoke_ran"
            return state

        ok, reason = is_safe_argv(command.argv, state.repo_evaluation.candidate_scripts)
        if not ok:
            emit_progress("Run smoke command", "command blocked", level="warning", detail=reason)
            state.smoke_run = SmokeRunResult(command=command, summary=f"Command blocked: {reason}")
            state.status = "smoke_ran"
            return state

        result = SmokeRunResult(command=command)
        deadline = time.monotonic() + self.timeout_minutes * 60
        repaired_packages: set[str] = set()
        emit_progress("Run smoke command", "selected command", detail=command.display)

        try:
            for attempt_no in range(1, self.max_repair_attempts + 2):
                emit_progress("Run smoke command", f"attempt {attempt_no}", detail=command.display)
                proc = self._run_command(command, state, task_dir, deadline)
                stdout_path = run_dir / "stdout.log"
                stderr_path = run_dir / "stderr.log"
                attempt_dir = run_dir / f"attempt_{attempt_no:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                attempt_stdout_path = attempt_dir / "stdout.log"
                attempt_stderr_path = attempt_dir / "stderr.log"

                stdout_path.write_text(proc.stdout, encoding="utf-8")
                stderr_path.write_text(proc.stderr, encoding="utf-8")
                attempt_stdout_path.write_text(proc.stdout, encoding="utf-8")
                attempt_stderr_path.write_text(proc.stderr, encoding="utf-8")
                (run_dir / "exit_code.txt").write_text(str(proc.returncode), encoding="utf-8")

                failure_type = None
                failure_evidence = None
                if proc.returncode != 0:
                    failure_type, failure_evidence = classify_smoke_failure(proc.stderr, proc.stdout)

                result.attempts.append({
                    "attempt": attempt_no,
                    "exit_code": proc.returncode,
                    "stdout_path": str(attempt_stdout_path),
                    "stderr_path": str(attempt_stderr_path),
                    "failure_type": failure_type,
                    "failure_evidence": failure_evidence,
                })

                result.exit_code = proc.returncode
                result.success = proc.returncode == 0
                result.stdout_path = str(stdout_path)
                result.stderr_path = str(stderr_path)
                result.failure_type = failure_type
                result.failure_evidence = failure_evidence

                if result.success:
                    if result.repair_actions:
                        result.summary = f"Smoke command executed successfully after {len(result.repair_actions)} environment repair(s) (backend={backend})"
                    else:
                        result.summary = f"Smoke command executed successfully (backend={backend})"
                    emit_progress("Run smoke command", f"attempt {attempt_no} passed", level="success")
                    break

                if not self._can_repair_missing_dependency(backend, failure_type, attempt_no):
                    result.summary = f"Smoke command failed with exit code {proc.returncode} (backend={backend})"
                    emit_progress(
                        "Run smoke command",
                        f"attempt {attempt_no} failed",
                        level="warning",
                        detail=failure_type or f"exit {proc.returncode}",
                    )
                    break

                package = package_from_missing_dependency(proc.stderr, proc.stdout)
                if not package or package in repaired_packages:
                    result.summary = f"Smoke command failed with exit code {proc.returncode} (backend={backend})"
                    emit_progress(
                        "Run smoke command",
                        f"attempt {attempt_no} failed",
                        level="warning",
                        detail=failure_type or f"exit {proc.returncode}",
                    )
                    break

                repaired_packages.add(package)
                emit_progress("Run smoke command", "missing dependency", level="warning", detail=package)
                repair_action = self._repair_venv_dependency(package, state, run_dir, deadline)
                result.repair_actions.append(repair_action)
                if not repair_action["success"]:
                    result.summary = f"Smoke command failed and dependency repair failed for {package} (backend={backend})"
                    emit_progress("Run smoke command", "repair failed", level="error", detail=package)
                    break

        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.summary = "Smoke command timed out"
            result.failure_type = "timeout"
            emit_progress("Run smoke command", "timed out", level="error")
        except Exception as e:
            result.summary = f"Smoke execution error: {e}"
            emit_progress("Run smoke command", "execution error", level="error", detail=str(e))

        state.smoke_run = result
        save_json(run_dir / "run_summary.json", result)
        state.status = "smoke_ran"
        return state

    def _run_command(
        self,
        command: SmokeCommand,
        state: TaskState,
        task_dir: Path,
        deadline: float,
    ) -> subprocess.CompletedProcess[str]:
        timeout = self._remaining_timeout(deadline)
        if state.backend == "local":
            return subprocess.run(
                command.argv,
                capture_output=True,
                text=True,
                cwd=state.repo_evaluation.repo_dir,
                timeout=timeout,
            )
        if state.backend in {"venv", "conda"}:
            return subprocess.run(
                self._venv_argv(command.argv, state, task_dir),
                capture_output=True,
                text=True,
                cwd=state.repo_evaluation.repo_dir,
                timeout=timeout,
                env=self._venv_env(state),
            )

        repo_mount = str(task_dir / "repos" / "cloned_repo")
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cpus", "2",
            "--memory", "4g",
            "--pids-limit", "256",
            "-v", f"{repo_mount}:/workspace/repo:ro",
            "-w", "/workspace/repo",
            state.env_build.image_tag,
            *command.argv,
        ]
        return subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _can_repair_missing_dependency(self, backend: str, failure_type: str | None, attempt_no: int) -> bool:
        return (
            backend in {"venv", "conda"}
            and self.max_repair_attempts > 0
            and attempt_no <= self.max_repair_attempts
            and failure_type == "missing_dependency"
        )

    def _repair_venv_dependency(
        self,
        package: str,
        state: TaskState,
        run_dir: Path,
        deadline: float,
    ) -> dict[str, Any]:
        repair_dir = run_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_no = len(list(repair_dir.glob("repair_*.log"))) + 1
        log_path = repair_dir / f"repair_{repair_no:02d}_{package}.log"

        python_bin = self._venv_python(state)
        if python_bin is None:
            action = {
                "package": package,
                "success": False,
                "summary": "No virtualenv python found",
                "log_path": str(log_path),
            }
            save_json(repair_dir / f"repair_{repair_no:02d}.json", action)
            return action

        install_spec = install_spec_for_package(package)
        argv = [python_bin, "-m", "pip", "install", *self._venv_pip_target_args(state), install_spec]
        emit_progress("Run smoke command", "repair dependency", detail=install_spec)
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            cwd=state.repo_evaluation.repo_dir,
            timeout=self._remaining_timeout(deadline),
            env=self._venv_env(state),
        )
        log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        action = {
            "package": package,
            "install_spec": install_spec,
            "argv": argv,
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "summary": (
                f"Installed missing dependency {package}"
                if proc.returncode == 0
                else f"Failed to install missing dependency {package}"
            ),
            "log_path": str(log_path),
        }
        if action["success"]:
            self._ensure_target_site_fixes(state)
            emit_progress("Run smoke command", "repair completed", level="success", detail=install_spec)
        save_json(repair_dir / f"repair_{repair_no:02d}.json", action)
        return action

    def _remaining_timeout(self, deadline: float) -> int:
        return max(1, int(deadline - time.monotonic()))

    def _venv_argv(self, argv: list[str], state: TaskState, task_dir: Path | None = None) -> list[str]:
        python_bin = self._venv_python(state)
        if python_bin is None:
            return argv

        if argv[0] == "python":
            if task_dir and state.env_build and state.env_build.python_paths and len(argv) >= 2:
                wrapper_path = self._write_python_wrapper(task_dir)
                return [python_bin, str(wrapper_path), *argv[1:]]
            return [python_bin, *argv[1:]]
        if argv[0] == "pytest":
            return [python_bin, "-m", "pytest", *argv[1:]]
        return argv

    def _write_python_wrapper(self, task_dir: Path) -> Path:
        run_dir = task_dir / "runs" / "smoke_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = run_dir / "_paper_smoke_run.py"
        wrapper_path.write_text(
            """
import os
import importlib
import runpy
import sys
from pathlib import Path

paths = [p for p in os.environ.get("PAPER_SMOKE_PYTHONPATHS", "").split(os.pathsep) if p]
for path in reversed(paths):
    if path not in sys.path:
        sys.path.insert(0, path)

if len(sys.argv) < 2:
    raise SystemExit("missing target script")

script = Path(sys.argv[1]).resolve()
script_dir = str(script.parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

for name in list(sys.modules):
    if name == "mpl_toolkits" or name.startswith("mpl_toolkits."):
        del sys.modules[name]

for module_name in [m for m in os.environ.get("PAPER_SMOKE_PREIMPORTS", "").split(",") if m]:
    try:
        importlib.import_module(module_name)
    except Exception:
        pass

if os.environ.get("PAPER_SMOKE_PATCH_GRADIO") == "1":
    try:
        import gradio as gr

        def _paper_smoke_launch_noop(self, *args, **kwargs):
            print("PAPER_SMOKE_GRADIO_LAUNCH_PATCHED")
            return None

        for _cls_name in ("Blocks", "Interface"):
            _cls = getattr(gr, _cls_name, None)
            if _cls is not None:
                setattr(_cls, "launch", _paper_smoke_launch_noop)
    except Exception:
        pass

sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
""".lstrip(),
            encoding="utf-8",
        )
        return wrapper_path

    def _venv_python(self, state: TaskState) -> str | None:
        if not state.env_build:
            return None
        if state.env_build.python_executable:
            return state.env_build.python_executable
        if not state.env_build.environment_path:
            return None
        import sys as _sys
        return str(Path(state.env_build.environment_path) / ("python.exe" if _sys.platform == "win32" else "bin/python"))

    def _venv_pip_target_args(self, state: TaskState) -> list[str]:
        if not state.env_build or not state.env_build.python_paths:
            return []
        return ["--target", state.env_build.python_paths[0]]

    def _venv_env(self, state: TaskState | None = None) -> dict[str, str]:
        env = os.environ.copy()
        if state and state.env_build and state.env_build.python_paths:
            paths = list(state.env_build.python_paths)
            existing = env.get("PYTHONPATH")
            if existing:
                paths.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(paths)
            env["PAPER_SMOKE_PYTHONPATHS"] = os.pathsep.join(paths)
            env["PAPER_SMOKE_PREIMPORTS"] = "torch"
            env["PAPER_SMOKE_PATCH_GRADIO"] = "1"
        else:
            env["PYTHONNOUSERSITE"] = "1"
        libgomp = _find_system_libgomp()
        if libgomp:
            existing_preload = env.get("LD_PRELOAD")
            env["LD_PRELOAD"] = (
                f"{libgomp} {existing_preload}"
                if existing_preload
                else libgomp
            )
        return env

    def _ensure_target_site_fixes(self, state: TaskState) -> None:
        if not state.env_build or not state.env_build.python_paths:
            return
        site_path = Path(state.env_build.python_paths[0])
        mpl_toolkits = site_path / "mpl_toolkits"
        if mpl_toolkits.exists():
            init_path = mpl_toolkits / "__init__.py"
            if not init_path.exists():
                init_path.write_text(
                    "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
                    encoding="utf-8",
                )

    def _select_smoke_command(self, state: TaskState) -> SmokeCommand | None:
        scripts = state.repo_evaluation.candidate_scripts if state.repo_evaluation else []
        candidates = []

        # Build heuristic candidates for all scripts
        heuristic_candidates = self._build_heuristic_candidates(scripts)
        argparse_command = self._argparse_select(state, scripts)
        if argparse_command is not None:
            candidates.append({
                "command": argparse_command.display,
                "source": "heuristic",
                "kind": argparse_command.kind,
                "selected": True,
                "reason": "argparse entry point is safer for --help smoke",
            })
            for hc in heuristic_candidates:
                if hc["command"] != argparse_command.display:
                    candidates.append({**hc, "selected": False})
            self._save_command_candidates(state, candidates)
            return argparse_command

        # Try LLM first
        llm_command = self._llm_select(state, scripts)
        if llm_command is not None:
            candidates.append({
                "command": " ".join(llm_command.argv),
                "source": "llm",
                "kind": llm_command.kind,
                "selected": True,
            })
            for hc in heuristic_candidates:
                candidates.append({**hc, "selected": False})
            self._save_command_candidates(state, candidates)
            return llm_command

        # Fallback to heuristic priority list
        logger.info("LLM unavailable or failed, falling back to heuristic command selection")
        selected = self._heuristic_select(scripts)
        for hc in heuristic_candidates:
            is_sel = selected is not None and hc["command"] == selected.display
            candidates.append({**hc, "selected": is_sel})

        self._save_command_candidates(state, candidates)
        return selected

    def _argparse_select(self, state: TaskState, scripts: list[str]) -> SmokeCommand | None:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            return None
        repo_dir = Path(state.repo_evaluation.repo_dir)
        candidates = []
        for script in scripts:
            path = repo_dir / script
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000]
            if "argparse" not in text and "ArgumentParser" not in text:
                continue
            candidates.append(script)

        if not candidates:
            return None

        candidates.sort(key=_argparse_script_sort_key)
        script = candidates[0]
        argv = ["python", script, "--help"]
        return SmokeCommand(argv=argv, display=" ".join(argv), kind="help")

    def _build_heuristic_candidates(self, scripts: list[str]) -> list[dict]:
        """Build all possible heuristic candidates with scores."""
        priority = [
            ("minimal_example.py", ["python", "minimal_example.py"], "demo", 95),
            ("demo.py", ["python", "demo.py", "--help"], "help", 90),
            ("demo.py", ["python", "demo.py"], "demo", 80),
            ("test.py", ["python", "test.py", "--help"], "help", 85),
            ("eval.py", ["python", "eval.py", "--help"], "help", 80),
            ("evaluate.py", ["python", "evaluate.py", "--help"], "help", 80),
            ("main.py", ["python", "main.py", "--help"], "help", 70),
            ("run.py", ["python", "run.py", "--help"], "help", 70),
            ("train.py", ["python", "train.py", "--help"], "help", 60),
        ]

        candidates = []
        for script, argv, kind, score in priority:
            if script in scripts:
                candidates.append({
                    "command": " ".join(argv),
                    "source": "heuristic",
                    "kind": kind,
                    "score": score,
                    "reason": f"{kind}-like script with --help" if "--help" in argv else f"{kind}-like script",
                })

        if scripts:
            script = scripts[0]
            candidates.append({
                "command": f"python {script} --help",
                "source": "heuristic",
                "kind": "help",
                "score": 50,
                "reason": "first candidate script with --help",
            })

        candidates.append({
            "command": "python --version",
            "source": "heuristic",
            "kind": "help",
            "score": 10,
            "reason": "last resort: verify Python environment",
        })

        return candidates

    def _save_command_candidates(self, state: TaskState, candidates: list[dict]) -> None:
        task_dir = Path(state.task_dir).resolve()
        run_dir = task_dir / "runs" / "smoke_001"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "command_candidates.json", candidates)

    def _llm_select(self, state: TaskState, scripts: list[str]) -> SmokeCommand | None:
        if not scripts:
            return None

        # Gather README content for context
        readme_text = ""
        repo_dir = state.repo_evaluation.repo_dir if state.repo_evaluation else None
        if repo_dir:
            repo_path = Path(repo_dir)
            for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")[:3000]
                    break

        user_prompt = (
            f"Repository structure scan:\n"
            f"- candidate_scripts: {scripts}\n"
            f"- has_readme: {state.repo_evaluation.has_readme if state.repo_evaluation else False}\n"
            f"- has_requirements: {state.repo_evaluation.has_requirements if state.repo_evaluation else False}\n"
            f"- has_dockerfile: {state.repo_evaluation.has_dockerfile if state.repo_evaluation else False}\n"
        )
        if readme_text:
            user_prompt += f"\nREADME content (truncated):\n{readme_text}\n"

        result = call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            purpose="smoke_command_selection",
        )
        if result is None:
            return None

        try:
            argv = result.get("argv", [])
            kind = result.get("kind", "help")
            if not isinstance(argv, list) or len(argv) == 0:
                return None
            return SmokeCommand(
                argv=[str(a) for a in argv],
                display=" ".join(str(a) for a in argv),
                kind=kind,
            )
        except Exception as e:
            logger.warning("Failed to parse LLM command result: %s", e)
            return None

    def _heuristic_select(self, scripts: list[str]) -> SmokeCommand | None:
        priority = [
            ("minimal_example.py", ["python", "minimal_example.py"], "demo"),
            ("demo.py", ["python", "demo.py", "--help"], "help"),
            ("demo.py", ["python", "demo.py"], "demo"),
            ("test.py", ["python", "test.py", "--help"], "help"),
            ("eval.py", ["python", "eval.py", "--help"], "help"),
            ("evaluate.py", ["python", "evaluate.py", "--help"], "help"),
            ("main.py", ["python", "main.py", "--help"], "help"),
            ("run.py", ["python", "run.py", "--help"], "help"),
            ("train.py", ["python", "train.py", "--help"], "help"),
        ]

        for script, argv, kind in priority:
            if script in scripts:
                return SmokeCommand(argv=argv, display=" ".join(argv), kind=kind)

        # Try any candidate script with --help
        if scripts:
            script = scripts[0]
            return SmokeCommand(
                argv=["python", script, "--help"],
                display=f"python {script} --help",
                kind="help",
            )

        # Last resort: python --version (always works, confirms environment)
        return SmokeCommand(argv=["python", "--version"], display="python --version", kind="help")


def _find_system_libgomp() -> str | None:
    candidates = [
        "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        "/usr/lib64/libgomp.so.1",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _argparse_script_sort_key(script: str) -> tuple[int, str]:
    path = Path(script)
    name = path.name.lower()
    stem = path.stem.lower()
    if name == "demo.py":
        return (0, script)
    if name == "minimal_example.py":
        return (0, script)
    if stem in {"infer", "inference", "predict", "eval", "evaluate", "amg"}:
        return (1, script)
    if script.startswith("scripts/"):
        return (2, script)
    if stem.startswith(("gradio", "app")) or name == "app.py":
        return (20, script)
    if stem.startswith("train"):
        return (30, script)
    return (10, script)
