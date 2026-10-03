from __future__ import annotations

import shutil
import re
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from app.agents.docker_build_agent import classify_build_failure
from app.agents.venv_build_agent import relax_requirement_line, pin_requirement_line
from app.core.file_utils import save_json
from app.core.naming import stable_paper_slug
from app.core.progress import emit_progress
from app.core.state import EnvironmentBuildResult, TaskState
from app.tools.repo_tool import (
    extract_pip_requirements_from_environment_file,
    find_requirement_files,
)
from app.tools.conda_env_manager import write_project_env_marker
from app.tools.torch_install_policy import strip_torch_family_requirements


class CondaBuildAgent:
    """Build an isolated conda environment for local smoke testing."""

    def __init__(self, timeout_minutes: int = 30, conda_executable: str | None = None):
        self.timeout_minutes = timeout_minutes
        self.conda_executable = conda_executable

    def run(self, state: TaskState) -> TaskState:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            emit_progress(
                "Build conda env",
                "skipped",
                level="warning",
                detail="repo was not evaluated",
            )
            state.env_build = EnvironmentBuildResult(
                build_success=False,
                failure_summary="Repo not evaluated",
            )
            state.status = "env_built"
            return state

        task_dir = Path(state.task_dir).resolve()
        env_dir = task_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)

        paper_slug = stable_paper_slug(state)
        conda_env_dir = Path(state.workspace_dir).resolve() / "envs" / paper_slug
        build_log_path = env_dir / "conda_build.log"
        repo_dir = Path(state.repo_evaluation.repo_dir).resolve()
        python_bin = conda_env_dir / (
            "python.exe" if sys.platform == "win32" else "bin/python"
        )

        result = EnvironmentBuildResult(
            environment_path=str(conda_env_dir),
            python_executable=str(python_bin),
            build_log_path=str(build_log_path),
        )

        deadline = time.monotonic() + self.timeout_minutes * 60
        log_parts: list[str] = []

        try:
            conda = self._find_conda_executable()
            if conda is None:
                raise BuildStepError("locate conda", 127)

            if python_bin.exists():
                # Reuse existing env — skip conda create, pip upgrade,
                # and torch install, but still install repo deps below.
                result.install_actions.append(
                    {
                        "action": "reuse_conda_env",
                        "paper_slug": paper_slug,
                        "environment_path": str(conda_env_dir),
                    }
                )
                log_parts.append(
                    f"Reusing existing paper-named conda environment: {conda_env_dir}"
                )
                emit_progress(
                    "Build conda env",
                    "reused existing environment",
                    level="success",
                    detail=str(conda_env_dir),
                )
            else:
                if conda_env_dir.exists():
                    shutil.rmtree(conda_env_dir)
                conda_env_dir.parent.mkdir(parents=True, exist_ok=True)

                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                self._run_step(
                    [
                        conda,
                        "create",
                        "-y",
                        "-p",
                        str(conda_env_dir),
                        f"python={python_version}",
                        "pip",
                    ],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name=f"create conda env python={python_version}",
                )
                _bridge_preferred_runtime(conda_env_dir, log_parts, result)

                pip_cmd = [str(python_bin), "-m", "pip"]
                self._run_step(
                    [
                        *pip_cmd,
                        "install",
                        "--upgrade",
                        "pip",
                        "setuptools",
                        "wheel",
                    ],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="upgrade pip tooling",
                )

                # Install PyTorch per runtime decision
                if state.runtime_decision and state.runtime_decision.install_plan:
                    for i, plan_cmd in enumerate(state.runtime_decision.install_plan):
                        resolved = [
                            str(python_bin) if p == "python" else p for p in plan_cmd
                        ]
                        self._run_step(
                            resolved,
                            cwd=repo_dir,
                            deadline=deadline,
                            log_parts=log_parts,
                            step_name=f"install pytorch runtime ({i + 1}/{len(state.runtime_decision.install_plan)})",
                        )

            # ── Common: repo deps + editable install (always run) ──
            pip_cmd = [str(python_bin), "-m", "pip"]

            for requirements_path in find_requirement_files(repo_dir):
                display_req = _safe_display_path(
                    requirements_path, repo_dir=repo_dir, task_dir=task_dir
                )
                # Filter torch family if runtime_decision exists
                if state.runtime_decision:
                    original = requirements_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    filtered = strip_torch_family_requirements(original)
                    if filtered != original:
                        filtered_path = (
                            env_dir / f"{requirements_path.stem}_no_torch.txt"
                        )
                        filtered_path.write_text(
                            "\n".join(filtered) + "\n", encoding="utf-8"
                        )
                        log_parts.append(
                            f"Filtered torch family from {requirements_path.name} per runtime decision"
                        )
                        requirements_path = filtered_path
                        display_req = _safe_display_path(
                            requirements_path, repo_dir=repo_dir, task_dir=task_dir
                        )
                self._install_requirements_with_relax_retry(
                    pip_cmd=pip_cmd,
                    requirements_path=requirements_path,
                    repo_dir=repo_dir,
                    env_dir=env_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name=f"install {display_req}",
                )

            env_requirements = extract_pip_requirements_from_environment_file(repo_dir)
            if env_requirements:
                env_req_path = env_dir / "environment_requirements.txt"
                env_req_path.write_text(
                    "\n".join(env_requirements) + "\n", encoding="utf-8"
                )
                self._install_requirements_with_relax_retry(
                    pip_cmd=pip_cmd,
                    requirements_path=env_req_path,
                    repo_dir=repo_dir,
                    env_dir=env_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="install environment.yml pip requirements",
                )

            if state.repo_evaluation.has_setup_py_or_pyproject:
                # --no-build-isolation: many repos (detectron2, CLIP, etc.)
                # import torch at module level in setup.py, which fails in
                # pip's temporary isolated build environment.
                self._run_step(
                    [
                        *pip_cmd,
                        "install",
                        "-e",
                        ".",
                        "--no-build-isolation",
                    ],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="install editable package",
                )

            if _repo_needs_audio_runtime_helper(repo_dir):
                self._run_step(
                    [*pip_cmd, "install", "imageio-ffmpeg"],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="install audio ffmpeg helper",
                )

            result.build_success = True
            emit_progress("Build conda env", "environment ready", level="success")
            _write_marker(conda_env_dir, state, paper_slug, python_bin)

        except subprocess.TimeoutExpired:
            result.build_success = False
            result.failure_type = "timeout"
            result.failure_summary = "Conda environment build timed out"
            emit_progress("Build conda env", "timed out", level="error")
        except BuildStepError as e:
            result.build_success = False
            result.failure_type = classify_build_failure("\n".join(log_parts))
            if e.returncode == 127 and e.step_name == "locate conda":
                result.failure_type = "conda_not_found"
                result.failure_summary = "Could not find conda executable"
            else:
                result.failure_summary = f"Conda environment build failed during {e.step_name} with exit code {e.returncode}"
            emit_progress(
                "Build conda env",
                "build step failed",
                level="error",
                detail=result.failure_summary,
            )
        except Exception as e:
            result.build_success = False
            result.failure_type = "unknown"
            result.failure_summary = str(e)
            emit_progress(
                "Build conda env", "build error", level="error", detail=str(e)
            )

        build_log_path.write_text("\n".join(log_parts), encoding="utf-8")
        state.env_build = result
        save_json(env_dir / "environment_summary.json", result)
        state.status = "env_built"
        return state

    def _install_requirements_with_relax_retry(
        self,
        *,
        pip_cmd: list[str],
        requirements_path: Path,
        repo_dir: Path,
        env_dir: Path,
        deadline: float,
        log_parts: list[str],
        step_name: str,
    ) -> None:
        # 1) Try with pinned versions (>= → ==, > → bumped)
        pinned_path = env_dir / f"{requirements_path.stem}_conda_pinned.txt"
        original_lines = requirements_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        pinned_lines = [pin_requirement_line(line) for line in original_lines]
        if pinned_lines != original_lines:
            pinned_path.write_text("\n".join(pinned_lines) + "\n", encoding="utf-8")
            log_parts.append(
                f"Pinning versions: {requirements_path.name} \u2192 {pinned_path.name}"
            )
            emit_progress(
                "Build conda env",
                "installing with pinned versions",
                detail=requirements_path.name,
            )
            installed = self._try_run_step(
                [*pip_cmd, "install", "-r", str(pinned_path)],
                cwd=repo_dir,
                deadline=deadline,
                log_parts=log_parts,
                step_name=f"install pinned {requirements_path.name}",
            )
            if installed:
                return

        # 2) Try with original requirements
        installed = self._try_run_step(
            [*pip_cmd, "install", "-r", str(requirements_path)],
            cwd=repo_dir,
            deadline=deadline,
            log_parts=log_parts,
            step_name=step_name,
        )
        if installed:
            return

        # 3) Fall back to relaxed (strip version pins), but cap numpy <2 for API compat
        emit_progress(
            "Build conda env",
            "retrying relaxed requirements",
            level="warning",
            detail=requirements_path.name,
        )

        relaxed_path = env_dir / f"{requirements_path.stem}_conda_relaxed.txt"
        relaxed_lines: list[str] = []
        for line in original_lines:
            relaxed = relax_requirement_line(line)
            # Cap numpy <2 to avoid np.trapz removal in numpy 2.x
            pkg = re.sub(r"\[.*?\]", "", relaxed.strip().lower())
            if pkg == "numpy":
                relaxed = "numpy<2"
            relaxed_lines.append(relaxed)
        relaxed_path.write_text("\n".join(relaxed_lines) + "\n", encoding="utf-8")
        self._run_step(
            [*pip_cmd, "install", "-r", str(relaxed_path)],
            cwd=repo_dir,
            deadline=deadline,
            log_parts=log_parts,
            step_name=f"install relaxed {requirements_path.name}",
        )

    def _find_conda_executable(self) -> str | None:
        candidates = [
            self.conda_executable,
            shutil.which("conda"),
            "/home/duyuan/miniconda3/bin/conda",
            "/opt/conda/bin/conda",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)
        return None

    def _run_step(
        self,
        argv: list[str],
        cwd: Path,
        deadline: float,
        log_parts: list[str],
        step_name: str,
    ) -> subprocess.CompletedProcess[str]:
        proc = self._run_process(argv, cwd, deadline, log_parts, step_name)
        if proc.returncode != 0:
            raise BuildStepError(step_name=step_name, returncode=proc.returncode)
        return proc

    def _try_run_step(
        self,
        argv: list[str],
        cwd: Path,
        deadline: float,
        log_parts: list[str],
        step_name: str,
    ) -> bool:
        proc = self._run_process(argv, cwd, deadline, log_parts, step_name)
        return proc.returncode == 0

    def _run_process(
        self,
        argv: list[str],
        cwd: Path,
        deadline: float,
        log_parts: list[str],
        step_name: str,
    ) -> subprocess.CompletedProcess[str]:
        remaining = max(1, int(deadline - time.monotonic()))
        display_cmd = _display_argv(argv)
        log_parts.append(f"\n$ {' '.join(argv)}")
        emit_progress("Build conda env", step_name, detail=display_cmd)

        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        q: Queue[tuple[str, str]] = Queue()
        if proc.stdout is not None:
            Thread(
                target=_enqueue, args=(proc.stdout, q, "stdout"), daemon=True
            ).start()
        if proc.stderr is not None:
            Thread(
                target=_enqueue, args=(proc.stderr, q, "stderr"), daemon=True
            ).start()

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        last_emit = 0.0
        last_visible: tuple[str, ...] = ()
        pending: list[str] = []

        try:
            while True:
                now = time.monotonic()
                if now > deadline:
                    proc.kill()
                    raise subprocess.TimeoutExpired(argv, remaining)

                got = False
                while True:
                    try:
                        stream, raw = q.get_nowait()
                    except Empty:
                        break
                    got = True
                    if stream == "stdout":
                        stdout_parts.append(raw)
                    else:
                        stderr_parts.append(raw)
                    clean = _clean_cli(raw)
                    if clean:
                        pending.append(clean)

                visible = tuple(_clip_for_tui(x) for x in pending[-5:])
                if visible and visible != last_visible and now - last_emit >= 1.0:
                    last_visible = visible
                    last_emit = now
                    emit_progress(
                        "Build conda env",
                        step_name,
                        detail=None,
                        log_lines=list(visible),
                        cli_stream=True,
                    )
                    pending.clear()

                if proc.poll() is not None:
                    break
                if not got:
                    time.sleep(0.05)

            # Drain
            while True:
                try:
                    stream, raw = q.get_nowait()
                except Empty:
                    break
                if stream == "stdout":
                    stdout_parts.append(raw)
                else:
                    stderr_parts.append(raw)
                clean = _clean_cli(raw)
                if clean:
                    pending.append(clean)

            if pending:
                emit_progress(
                    "Build conda env",
                    step_name,
                    detail=None,
                    log_lines=pending[-5:],
                    cli_stream=True,
                )

            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            if stdout:
                log_parts.append(stdout)
            if stderr:
                log_parts.append(stderr)
            log_parts.append(f"[exit_code] {proc.returncode} ({step_name})")
            if proc.returncode == 0:
                emit_progress("Build conda env", f"{step_name} done", level="success")
            else:
                emit_progress(
                    "Build conda env",
                    f"{step_name} failed",
                    level="warning",
                    detail=f"exit {proc.returncode}",
                )
            return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise


class BuildStepError(Exception):
    def __init__(self, step_name: str, returncode: int):
        super().__init__(step_name)
        self.step_name = step_name
        self.returncode = returncode


def _display_argv(argv: list[str], max_len: int = 140) -> str:
    display = " ".join(argv)
    if len(display) <= max_len:
        return display
    return display[: max_len - 3] + "..."


def _enqueue(pipe, q: Queue[tuple[str, str]], stream_name: str) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if line:
                q.put((stream_name, line))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _clean_cli(line: str) -> str:
    """Handle carriage-return overwrites, strip ANSI/control chars."""
    if not line:
        return ""
    line = line.replace("\r", "\n").splitlines()[-1] if "\r" in line else line
    line = _ANSI_RE.sub("", line)
    line = _OSC_RE.sub("", line)
    line = _CONTROL_RE.sub("", line)
    return line.strip()


def _clip_for_tui(line: str, limit: int = 240) -> str:
    """Clip a log line for TUI display; full content stays in log_parts."""
    if len(line) <= limit:
        return line
    return line[: limit - 3] + "..."


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _repo_needs_audio_runtime_helper(repo_dir: Path) -> bool:
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    has_audio_sample = any(
        path.is_file() and path.suffix.lower() in audio_exts
        for root_name in ("tests", "test", "data", "samples", "examples")
        for root in [repo_dir / root_name]
        if root.exists()
        for path in root.rglob("*")
    )
    if not has_audio_sample:
        return False
    readme_text = ""
    for readme in repo_dir.glob("README*"):
        readme_text += readme.read_text(encoding="utf-8", errors="ignore").lower()
    return any(
        term in readme_text for term in ("ffmpeg", "audio", "transcribe", "speech")
    )


def _bridge_preferred_runtime(
    conda_env_dir: Path, log_parts: list[str], result: EnvironmentBuildResult
) -> None:
    preferred_env = Path("/home/duyuan/miniconda3/envs/torch_py39_env")
    preferred_site = (
        preferred_env
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    target_site = (
        conda_env_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not preferred_site.exists() or not target_site.exists():
        return
    pth_path = target_site / "paper_repro_preferred_runtime.pth"
    pth_path.write_text(str(preferred_site) + "\n", encoding="utf-8")
    message = f"Linked preferred runtime site-packages via {pth_path}: {preferred_site}"
    log_parts.append(message)
    result.install_actions.append(
        {
            "action": "bridge_preferred_runtime",
            "source": str(preferred_site),
            "path": str(pth_path),
        }
    )


def _write_marker(env_dir: Path, state: TaskState, slug: str, python_bin: Path) -> None:
    """Write .paper_reproduct_agent_env.json marker for future discovery."""
    try:
        repo_url = state.selected_repo.url if state.selected_repo else ""
        write_project_env_marker(
            env_dir,
            task_id=state.task_id,
            paper_slug=slug,
            paper_path=state.input_value,
            repo_url=repo_url,
            workspace=state.workspace_dir,
            python_executable=str(python_bin),
        )
    except Exception:
        pass  # non-critical


def _safe_display_path(
    path: Path, *, repo_dir: Path, task_dir: Path | None = None
) -> str:
    """Display a path relative to repo_dir, task_dir, or just the filename."""
    path = Path(path).resolve()
    repo_dir = Path(repo_dir).resolve()

    try:
        return path.relative_to(repo_dir).as_posix()
    except ValueError:
        pass

    if task_dir is not None:
        try:
            return path.relative_to(Path(task_dir).resolve()).as_posix()
        except ValueError:
            pass

    return path.name
