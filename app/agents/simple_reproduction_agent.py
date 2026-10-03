from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from app.agents.smoke_run_agent import SmokeRunAgent, classify_smoke_failure
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import ReproductionCommand, ReproductionRunResult, TaskState
from app.tools.llm import call_llm_json


_SYSTEM_PROMPT = """\
You are a research reproduction agent. Select ONE safe, lightweight command that
fully exercises a simple research repository end-to-end.

The command should do real work, not just display help or print a version. Prefer
demo, inference, prediction, evaluation, or example commands that use files already
present in the repository. The goal is a minimal complete reproduction for simple
tasks: run code on a bundled/sample input and produce an observable output or
evaluation result.

Return a JSON object with exactly these keys:
- "run": boolean. false if no safe lightweight reproduction command exists.
- "argv": list of strings. Use only python or pytest as the executable.
- "kind": one of "demo", "example", "evaluation", "pytest".
- "reason": short explanation.

Rules:
- Do not choose --help, -h, or --version commands.
- Do not choose training commands.
- Do not choose commands requiring a large dataset, private files, or manual downloads.
- Do not use shell metacharacters, pipes, redirections, curl, wget, or bash.
- Only use scripts listed in candidate_scripts.
- Do not use bare pytest -q as a reproduction command.
- Prefer commands that use bundled sample files listed in sample_files.
"""


class SimpleReproductionAgent:
    """Run a lightweight end-to-end reproduction command when one is available."""

    def __init__(self, timeout_minutes: int = 30):
        self.timeout_minutes = timeout_minutes
        self._runner = SmokeRunAgent(timeout_minutes=timeout_minutes)

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir).resolve()
        run_dir = task_dir / "runs" / "reproduction_001"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        result = ReproductionRunResult()

        skip_reason = self._skip_reason(state)
        if skip_reason:
            emit_progress("Run simple reproduction", "skipped", level="warning", detail=skip_reason)
            result.skipped = True
            result.skip_reason = skip_reason
            result.summary = f"Skipped simple reproduction: {skip_reason}"
            state.reproduction_run = result
            state.status = "reproduction_ran"
            save_json(run_dir / "reproduction_summary.json", result)
            return state

        result.eligible = True
        command, candidates, recipe = self._select_command(state, run_dir)
        result.command_candidates = candidates
        save_json(run_dir / "command_candidates.json", candidates)

        if command is None:
            emit_progress("Run simple reproduction", "no safe command found", level="warning")
            result.skipped = True
            result.skip_reason = "No safe lightweight reproduction command found"
            result.summary = result.skip_reason
            state.reproduction_run = result
            state.status = "reproduction_ran"
            save_json(run_dir / "reproduction_summary.json", result)
            return state

        ok, reason = _is_safe_reproduction_argv(command.argv, state.repo_evaluation.candidate_scripts)
        if not ok:
            emit_progress("Run simple reproduction", "command blocked", level="warning", detail=reason)
            result.command = command
            result.skipped = True
            result.skip_reason = f"Command blocked: {reason}"
            result.summary = result.skip_reason
            state.reproduction_run = result
            state.status = "reproduction_ran"
            save_json(run_dir / "reproduction_summary.json", result)
            return state

        repo_dir = Path(state.repo_evaluation.repo_dir).resolve()
        before = _snapshot_files(repo_dir)
        deadline = time.monotonic() + self.timeout_minutes * 60
        result.command = command

        emit_progress("Run simple reproduction", "selected command", detail=command.display)
        try:
            proc = subprocess.run(
                self._argv_for_backend(command.argv, state, task_dir),
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=max(1, int(deadline - time.monotonic())),
                env=self._runner._venv_env(state) if state.backend in {"venv", "conda"} else os.environ.copy(),
            )
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")

            result.exit_code = proc.returncode
            result.success = proc.returncode == 0
            result.stdout_path = str(stdout_path)
            result.stderr_path = str(stderr_path)
            result.output_artifacts = _collect_artifacts(repo_dir, before)
            metrics, reference_results, comparisons = _analyze_reproduction_output(
                recipe=recipe,
                stdout=proc.stdout,
                stderr=proc.stderr,
                repo_dir=repo_dir,
                run_dir=run_dir,
            )
            result.metrics = metrics
            result.reference_results = reference_results
            result.comparisons = comparisons

            if result.success:
                artifact_note = (
                    f"; artifacts: {len(result.output_artifacts)}"
                    if result.output_artifacts
                    else ""
                )
                result.summary = f"Simple reproduction command executed successfully{artifact_note}"
                emit_progress("Run simple reproduction", "completed", level="success", detail=command.display)
            else:
                failure_type, evidence = classify_smoke_failure(proc.stderr, proc.stdout)
                result.failure_type = failure_type
                result.failure_evidence = evidence
                result.summary = f"Simple reproduction command failed with exit code {proc.returncode}"
                emit_progress("Run simple reproduction", "failed", level="warning", detail=failure_type)

        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.failure_type = "timeout"
            result.summary = "Simple reproduction command timed out"
            emit_progress("Run simple reproduction", "timed out", level="error")
        except Exception as e:
            result.failure_type = "execution_error"
            result.failure_evidence = str(e)
            result.summary = f"Simple reproduction execution error: {e}"
            emit_progress("Run simple reproduction", "execution error", level="error", detail=str(e))

        state.reproduction_run = result
        state.status = "reproduction_ran"
        save_json(run_dir / "reproduction_summary.json", result)
        return state

    def _skip_reason(self, state: TaskState) -> str | None:
        if state.backend == "none":
            return "backend=none"
        if state.backend in {"venv", "conda", "docker"} and (
            not state.env_build or not state.env_build.build_success
        ):
            return f"{state.backend} environment did not build"
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            return "repo was not evaluated"
        risk_flags = set(state.repo_evaluation.risk_flags)
        if (
            "可能需要大数据集" in risk_flags
            and not self._has_lightweight_surface(state)
        ):
            return "repository appears to require a large dataset"
        if (
            not state.repo_evaluation.candidate_scripts
            and not self._readme_example_command(state, None)
        ):
            return "no runnable scripts or tests found"
        return None

    def _has_lightweight_surface(self, state: TaskState) -> bool:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            return False
        repo_dir = Path(state.repo_evaluation.repo_dir)
        scripts = state.repo_evaluation.candidate_scripts
        if any(_script_looks_lightweight(script) for script in scripts):
            return True
        if _find_sample_files(repo_dir, limit=1):
            return True
        return self._readme_example_command(state, None) is not None

    def _select_command(
        self,
        state: TaskState,
        run_dir: Path,
    ) -> tuple[ReproductionCommand | None, list[dict[str, Any]], str | None]:
        candidates = self._heuristic_candidates(state)
        for candidate in candidates:
            argv = candidate["argv"]
            ok, _ = _is_safe_reproduction_argv(argv, state.repo_evaluation.candidate_scripts)
            if ok:
                command = ReproductionCommand(
                    argv=argv,
                    display=" ".join(argv),
                    kind=candidate["kind"],
                    reason=candidate["reason"],
                )
                return command, [
                    {**c, "command": " ".join(c["argv"]), "selected": c is candidate}
                    for c in candidates
                ], None

        readme_command = self._readme_example_command(state, run_dir)
        if readme_command is not None:
            command, recipe = readme_command
            return command, [
                {
                    "command": command.display,
                    "source": "readme_example",
                    "kind": command.kind,
                    "selected": True,
                    "reason": command.reason,
                    "recipe": recipe,
                },
                *[{**c, "command": " ".join(c["argv"]), "selected": False} for c in candidates],
            ], recipe

        llm_command = self._llm_select(state)
        if llm_command is not None:
            return llm_command, [
                {
                    "command": llm_command.display,
                    "source": "llm",
                    "kind": llm_command.kind,
                    "selected": True,
                    "reason": llm_command.reason,
                },
                *[{**c, "command": " ".join(c["argv"]), "selected": False} for c in candidates],
            ], None

        return None, [{**c, "command": " ".join(c["argv"]), "selected": False} for c in candidates], None

    def _readme_example_command(
        self,
        state: TaskState,
        run_dir: Path | None,
    ) -> tuple[ReproductionCommand, str] | None:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            return None
        repo_dir = Path(state.repo_evaluation.repo_dir)
        readme = _read_readme(repo_dir, limit=16000)
        for block in _extract_python_code_blocks(readme):
            script_body = _sanitize_readme_example_block(block, repo_dir)
            if not script_body:
                continue
            if run_dir is None:
                return ReproductionCommand(
                    argv=["python", "paper_smoke_readme_example.py"],
                    display="python paper_smoke_readme_example.py",
                    kind="example",
                    reason="Run a safe Python usage example extracted from the repository README.",
                ), "readme_python_example"
            script = repo_dir / "paper_smoke_readme_example.py"
            script.write_text(script_body, encoding="utf-8")
            return ReproductionCommand(
                argv=["python", script.name],
                display=f"python {script.name}",
                kind="example",
                reason="Run a safe Python usage example extracted from the repository README.",
            ), "readme_python_example"
        return None

    def _llm_select(self, state: TaskState) -> ReproductionCommand | None:
        if not state.repo_evaluation:
            return None

        repo_dir = Path(state.repo_evaluation.repo_dir)
        readme_text = _read_readme(repo_dir)
        sample_files = _find_sample_files(repo_dir)
        prompt = (
            f"Paper task: {state.reproduction_brief.task if state.reproduction_brief else None}\n"
            f"Datasets: {state.reproduction_brief.datasets if state.reproduction_brief else []}\n"
            f"Metrics: {state.reproduction_brief.metrics if state.reproduction_brief else []}\n"
            f"Risk flags: {state.repo_evaluation.risk_flags}\n"
            f"candidate_scripts: {state.repo_evaluation.candidate_scripts}\n"
            f"candidate_configs: {state.repo_evaluation.candidate_configs}\n"
            f"sample_files: {sample_files}\n\n"
            f"README excerpt:\n{readme_text}\n"
        )
        result = call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            purpose="simple_reproduction_command_selection",
        )
        if not result or not result.get("run", False):
            return None
        argv = result.get("argv") or []
        if not isinstance(argv, list):
            return None
        try:
            return ReproductionCommand(
                argv=[str(part) for part in argv],
                display=" ".join(str(part) for part in argv),
                kind=result.get("kind", "demo"),
                reason=result.get("reason"),
            )
        except Exception:
            return None

    def _heuristic_candidates(self, state: TaskState) -> list[dict[str, Any]]:
        repo_dir = Path(state.repo_evaluation.repo_dir)
        candidates: list[dict[str, Any]] = []
        scripts = state.repo_evaluation.candidate_scripts
        for script in scripts:
            name = Path(script).name.lower()
            stem = Path(script).stem.lower()
            if stem.startswith("train") or "train" in stem:
                continue
            if name in {"app.py", "online_demo.py"} or stem.startswith(("gradio", "server")):
                continue
            if _script_defaults_to_interactive_camera(repo_dir / script):
                continue
            if any(token in stem for token in ("demo", "infer", "inference", "predict", "eval", "evaluate", "benchmark", "example")):
                kind = "evaluation" if any(token in stem for token in ("eval", "evaluate", "benchmark")) else "demo"
                candidates.append({
                    "argv": ["python", script],
                    "source": "heuristic",
                    "kind": kind,
                    "reason": "candidate demo/evaluation script without help flag",
                })
        return candidates

    def _argv_for_backend(self, argv: list[str], state: TaskState, task_dir: Path) -> list[str]:
        if state.backend in {"venv", "conda"}:
            return self._runner._venv_argv(argv, state, task_dir)
        if state.backend == "local":
            return argv
        return argv


def _read_readme(repo_dir: Path, limit: int = 8000) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    return ""


def _find_sample_files(repo_dir: Path, limit: int = 40) -> list[str]:
    roots = [repo_dir / name for name in ("assets", "asset", "examples", "example", "demo", "demos", "data", "samples", "test")]
    roots.append(repo_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".wav", ".mp3", ".flac", ".txt", ".json", ".npy", ".npz", ".mp4", ".avi"}
    files: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if len(files) >= limit:
                return files
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            if any(part.startswith(".") or part in {"__pycache__", ".git"} for part in path.relative_to(repo_dir).parts):
                continue
            try:
                if path.stat().st_size > 20 * 1024 * 1024:
                    continue
            except OSError:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path.relative_to(repo_dir).as_posix())
    return files


def _script_looks_lightweight(script: str) -> bool:
    stem = Path(script).stem.lower()
    if "train" in stem:
        return False
    return any(token in stem for token in ("demo", "infer", "inference", "predict", "eval", "evaluate", "benchmark", "example"))


def _script_defaults_to_interactive_camera(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:12000].lower()
    except OSError:
        return False

    camera_terms = ("webcam", "usb camera", "ip camera", "videostreamer", "cv2.videocapture")
    default_camera_terms = (
        "default='0'",
        'default="0"',
        "default = '0'",
        'default = "0"',
        "default=0",
        "default = 0",
    )
    has_camera_default = "--input" in text and any(term in text for term in default_camera_terms)
    return has_camera_default and any(term in text for term in camera_terms)


def _extract_python_code_blocks(readme: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(readme):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


def _sanitize_readme_example_block(block: str, repo_dir: Path) -> str | None:
    lowered = block.lower()
    dangerous_terms = (
        "subprocess", "os.system", "shutil.rmtree", "rm -rf", "socket",
        "requests.", "urllib", "input(", "fit(", ".fit(", "trainer", "train(",
        ".train(", "download(", "wget", "curl", "open(",
    )
    if any(term in lowered for term in dangerous_terms):
        return None
    if "import " not in lowered and "from " not in lowered:
        return None

    sanitized_lines: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            sanitized_lines.append(line)
            continue
        if stripped.startswith(("!", "%", ">>>", "...")):
            return None
        if any(term in stripped.lower() for term in ("pip install", "conda install", "apt install")):
            return None
        sanitized_lines.append(line)

    code = "\n".join(sanitized_lines)
    sample_files = _find_sample_files(repo_dir)
    audio_sample = next((path for path in sample_files if Path(path).suffix.lower() in {".wav", ".mp3", ".flac"}), None)
    image_sample = next((path for path in sample_files if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}), None)
    if audio_sample:
        for placeholder in ("audio.mp3", "audio.flac", "audio.wav", "japanese.wav"):
            code = code.replace(placeholder, audio_sample)
    if image_sample:
        for placeholder in ("image.jpg", "image.png", "example.jpg", "example.png"):
            code = code.replace(placeholder, image_sample)

    code = re.sub(r'load_model\(["\'](?:turbo|large|medium|small|base)["\']\)', 'load_model("tiny")', code)

    prelude = """
import json
import os
import shutil
from pathlib import Path

try:
    import torch
    if torch.cuda.is_available():
        try:
            import flair
            flair.device = torch.device("cuda")
        except Exception:
            pass
except Exception:
    pass

try:
    import imageio_ffmpeg
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    shim_dir = Path(".paper_smoke_bin")
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.exists():
        try:
            shim.symlink_to(ffmpeg_exe)
        except Exception:
            shutil.copy2(ffmpeg_exe, shim)
            shim.chmod(0o755)
    os.environ["PATH"] = str(shim_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass
""".strip()
    return f"{prelude}\n\n{code}\n\nprint('PAPER_SMOKE_README_EXAMPLE_OK')\n"


def _snapshot_files(repo_dir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if _ignored_artifact_path(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[rel.as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _collect_artifacts(repo_dir: Path, before: dict[str, tuple[int, int]], limit: int = 50) -> list[str]:
    artifacts: list[str] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_dir)
        if _ignored_artifact_path(rel):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        key = rel.as_posix()
        current = (stat.st_mtime_ns, stat.st_size)
        if before.get(key) != current and stat.st_size <= 50 * 1024 * 1024:
            artifacts.append(key)
        if len(artifacts) >= limit:
            break
    return sorted(artifacts)


def _ignored_artifact_path(path: Path) -> bool:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
    return any(part in ignored or part.startswith(".") for part in path.parts)


def _is_safe_reproduction_argv(argv: list[str], candidate_scripts: list[str]) -> tuple[bool, str | None]:
    if not argv:
        return False, "empty command"
    if any(any(char in part for char in [";", "&", "|", "$", "`", ">", "<"]) for part in argv):
        return False, "shell metacharacter blocked"
    lowered = [part.lower() for part in argv]
    blocked_terms = ("train", "wget", "curl", "http://", "https://")
    if any(any(term in part for term in blocked_terms) for part in lowered):
        return False, "training/download/network-looking command blocked"
    if any(part in {"--help", "-h", "--version"} for part in lowered):
        return False, "help/version command is not a reproduction"

    if argv[0] == "pytest":
        return False, "bare pytest is not a lightweight paper reproduction"

    if argv[0] != "python":
        return False, "executable not allowed"
    if len(argv) >= 3 and argv[1] == "-m":
        if argv[:3] == ["python", "-m", "pytest"] and argv[3:] in ([], ["-q"]):
            return True, None
        return False, "only python -m pytest is allowed"
    if len(argv) < 2:
        return False, "missing script"

    script = argv[1]
    if not _is_safe_relative_script(script):
        return False, "unsafe script path"
    if Path(script).stem.lower().startswith("train") or "train" in Path(script).stem.lower():
        return False, "training script blocked"
    if script not in candidate_scripts and not script.startswith("paper_smoke_"):
        return False, "script is not in candidate_scripts"
    for arg in argv[2:]:
        if not _is_safe_arg(arg):
            return False, f"unsafe argument: {arg}"
    return True, None


def _is_safe_relative_script(path: str) -> bool:
    p = Path(path)
    return (
        not p.is_absolute()
        and ".." not in p.parts
        and path.endswith(".py")
        and not any(char in path for char in [";", "&", "|", "$", "`", ">", "<"])
    )


def _is_safe_arg(arg: str) -> bool:
    if not arg or len(arg) > 160:
        return False
    if any(char in arg for char in [";", "&", "|", "$", "`", ">", "<"]):
        return False
    if arg.startswith("-"):
        return bool(re.fullmatch(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_\-]*", arg))
    if Path(arg).is_absolute() or ".." in Path(arg).parts:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_./:=,+@%-]+", arg))


def _analyze_reproduction_output(
    *,
    recipe: str | None,
    stdout: str,
    stderr: str,
    repo_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if recipe == "readme_python_example":
        return {
            "readme_example_completed": "PAPER_SMOKE_README_EXAMPLE_OK" in stdout,
        }, {
            "source": "Repository README Python usage example",
            "scope": "readme_example_not_full_paper_benchmark",
            "metrics": {},
            "note": "The lightweight reproduction executes an official README example on bundled or inline sample input; full paper benchmark numbers may require external datasets.",
        }, []

    return {}, {}, []


def _parse_superglue_metrics(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("AUC@5") and idx + 1 < len(lines):
            values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", lines[idx + 1])]
            keys = ["AUC@5", "AUC@10", "AUC@20", "Prec", "MScore"]
            if len(values) >= len(keys):
                return dict(zip(keys, values[: len(keys)]))
    return {}
