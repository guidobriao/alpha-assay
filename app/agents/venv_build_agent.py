from __future__ import annotations

import re
import subprocess
import sys
import time
import importlib.util
from pathlib import Path
from urllib.parse import urlparse

from app.agents.docker_build_agent import classify_build_failure
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import EnvironmentBuildResult, TaskState
from app.tools.dependency_tool import install_spec_for_package
from app.tools.repo_tool import (
    extract_pip_requirements_from_environment_file,
    find_requirement_files,
)


_TARGET_BOOTSTRAP_PACKAGES = {"gradio"}


class VenvBuildAgent:
    def __init__(self, timeout_minutes: int = 30):
        self.timeout_minutes = timeout_minutes

    def run(self, state: TaskState) -> TaskState:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            emit_progress("Build virtualenv", "skipped", level="warning", detail="repo was not evaluated")
            state.env_build = EnvironmentBuildResult(
                build_success=False,
                failure_summary="Repo not evaluated",
            )
            state.status = "env_built"
            return state

        task_dir = Path(state.task_dir).resolve()
        env_dir = task_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)

        venv_dir = env_dir / "venv"
        target_site = env_dir / "site-packages"
        build_log_path = env_dir / "venv_build.log"
        repo_dir = Path(state.repo_evaluation.repo_dir).resolve()

        result = EnvironmentBuildResult(
            environment_path=str(venv_dir),
            python_executable=str(venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")),
            build_log_path=str(build_log_path),
        )

        deadline = time.monotonic() + self.timeout_minutes * 60
        log_parts: list[str] = []
        emit_progress("Build virtualenv", "preparing environment", detail=str(env_dir))

        try:
            venv_created = self._try_run_step(
                [sys.executable, "-m", "venv", str(venv_dir)],
                cwd=repo_dir,
                deadline=deadline,
                log_parts=log_parts,
                step_name="create venv",
            )

            if venv_created:
                python_bin = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
                pip_cmd = [str(python_bin), "-m", "pip"]
                pip_prefix: list[str] = []
                result.environment_path = str(venv_dir)
                result.python_executable = str(python_bin)
                result.python_paths = []

                self._try_run_step(
                    [*pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="upgrade pip tooling",
                )
            else:
                target_site.mkdir(parents=True, exist_ok=True)
                pip_cmd = [sys.executable, "-m", "pip"]
                pip_prefix = ["--target", str(target_site)]
                pip_install_options = ["--no-deps"]
                result.environment_path = str(env_dir)
                result.python_executable = sys.executable
                result.python_paths = [str(target_site), str(repo_dir)]
                log_parts.append(
                    "python -m venv failed; falling back to pip --target isolated site-packages"
                )
                emit_progress(
                    "Build virtualenv",
                    "venv unavailable; using pip --target fallback",
                    level="warning",
                    detail=str(target_site),
                )
            if venv_created:
                pip_install_options = []

            requirement_files = find_requirement_files(repo_dir)
            for requirements_path in requirement_files:
                install_requirements_path = requirements_path
                if result.python_paths:
                    for spec in _target_bootstrap_specs(requirements_path):
                        emit_progress("Build virtualenv", f"bootstrap {spec}")
                        self._run_step(
                            [*pip_cmd, "install", *pip_prefix, spec],
                            cwd=repo_dir,
                            deadline=deadline,
                            log_parts=log_parts,
                            step_name=f"bootstrap target dependency {spec}",
                        )
                    install_requirements_path = self._write_target_safe_requirements(
                        requirements_path,
                        repo_dir,
                        env_dir,
                    )
                installed = self._try_run_step(
                    [*pip_cmd, "install", *pip_prefix, *pip_install_options, "-r", str(install_requirements_path)],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name=f"install {requirements_path.relative_to(repo_dir).as_posix()}",
                )
                if not installed:
                    emit_progress(
                        "Build virtualenv",
                        "retrying relaxed requirements",
                        level="warning",
                        detail=requirements_path.relative_to(repo_dir).as_posix(),
                    )
                    relaxed_path = env_dir / f"{requirements_path.stem}_relaxed.txt"
                    relaxed_path.write_text(
                        "\n".join(relax_requirement_line(line) for line in install_requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines())
                        + "\n",
                        encoding="utf-8",
                    )
                    self._run_step(
                        [*pip_cmd, "install", *pip_prefix, *pip_install_options, "-r", str(relaxed_path)],
                        cwd=repo_dir,
                        deadline=deadline,
                        log_parts=log_parts,
                        step_name=f"install relaxed {requirements_path.relative_to(repo_dir).as_posix()}",
                    )

            env_requirements = extract_pip_requirements_from_environment_file(repo_dir)
            if env_requirements:
                env_req_path = env_dir / "environment_requirements.txt"
                env_req_path.write_text("\n".join(env_requirements) + "\n", encoding="utf-8")
                install_env_req_path = env_req_path
                if result.python_paths:
                    for spec in _target_bootstrap_specs(env_req_path):
                        emit_progress("Build virtualenv", f"bootstrap {spec}")
                        self._run_step(
                            [*pip_cmd, "install", *pip_prefix, spec],
                            cwd=repo_dir,
                            deadline=deadline,
                            log_parts=log_parts,
                            step_name=f"bootstrap target dependency {spec}",
                        )
                    install_env_req_path = self._write_target_safe_requirements(
                        env_req_path,
                        repo_dir,
                        env_dir,
                    )
                installed = self._try_run_step(
                    [*pip_cmd, "install", *pip_prefix, *pip_install_options, "-r", str(install_env_req_path)],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="install environment.yml pip requirements",
                )
                if not installed:
                    emit_progress(
                        "Build virtualenv",
                        "retrying relaxed environment.yml requirements",
                        level="warning",
                    )
                    relaxed_env_path = env_dir / "environment_requirements_relaxed.txt"
                    relaxed_env_path.write_text(
                        "\n".join(relax_requirement_line(line) for line in install_env_req_path.read_text(encoding="utf-8", errors="ignore").splitlines())
                        + "\n",
                        encoding="utf-8",
                    )
                    self._run_step(
                        [*pip_cmd, "install", *pip_prefix, *pip_install_options, "-r", str(relaxed_env_path)],
                        cwd=repo_dir,
                        deadline=deadline,
                        log_parts=log_parts,
                        step_name="install relaxed environment.yml pip requirements",
                    )

            if state.repo_evaluation.has_setup_py_or_pyproject and not result.python_paths:
                self._run_step(
                    [*pip_cmd, "install", "-e", "."],
                    cwd=repo_dir,
                    deadline=deadline,
                    log_parts=log_parts,
                    step_name="install editable package",
                )

            if result.python_paths:
                _ensure_target_namespace_packages(Path(result.python_paths[0]))

            result.build_success = True
            emit_progress("Build virtualenv", "environment ready", level="success")

        except subprocess.TimeoutExpired:
            result.build_success = False
            result.failure_type = "timeout"
            result.failure_summary = "Virtualenv build timed out"
            emit_progress("Build virtualenv", "timed out", level="error")
        except BuildStepError as e:
            result.build_success = False
            result.failure_type = classify_build_failure("\n".join(log_parts))
            result.failure_summary = f"Virtualenv build failed during {e.step_name} with exit code {e.returncode}"
            emit_progress(
                "Build virtualenv",
                "build step failed",
                level="error",
                detail=f"{e.step_name} exit {e.returncode}",
            )
        except Exception as e:
            result.build_success = False
            result.failure_type = "unknown"
            result.failure_summary = str(e)
            emit_progress("Build virtualenv", "build error", level="error", detail=str(e))

        build_log_path.write_text("\n".join(log_parts), encoding="utf-8")
        state.env_build = result
        save_json(env_dir / "environment_summary.json", result)
        state.status = "env_built"
        return state

    def _write_target_safe_requirements(self, requirements_path: Path, repo_dir: Path, env_dir: Path) -> Path:
        local_names = _local_project_names(repo_dir)
        output_path = env_dir / f"{requirements_path.stem}_target_safe.txt"
        lines = []
        for line in requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if _is_self_vcs_requirement(line, local_names):
                continue
            package_name = _extract_requirement_name(line)
            if package_name and _normalize_project_name(package_name) in _TARGET_BOOTSTRAP_PACKAGES:
                continue
            if _requirement_is_already_importable(line):
                continue
            lines.append(line)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

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
        log_parts.append(f"\n$ {' '.join(argv)}")
        emit_progress("Build virtualenv", step_name, detail=_display_argv(argv))
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=remaining,
        )
        if proc.stdout:
            log_parts.append(proc.stdout)
        if proc.stderr:
            log_parts.append(proc.stderr)
        log_parts.append(f"[exit_code] {proc.returncode} ({step_name})")
        if proc.returncode == 0:
            emit_progress("Build virtualenv", f"{step_name} done", level="success")
        else:
            emit_progress("Build virtualenv", f"{step_name} failed", level="warning", detail=f"exit {proc.returncode}")
        return proc


class BuildStepError(Exception):
    def __init__(self, step_name: str, returncode: int):
        super().__init__(step_name)
        self.step_name = step_name
        self.returncode = returncode


def relax_requirement_line(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return line
    if stripped.startswith(("git+", "http://", "https://")):
        return stripped
    without_marker = stripped.split(";", 1)[0].strip()
    without_extras = re.sub(r"\[.*?\]", "", without_marker)
    return re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|===)\s*", without_extras, maxsplit=1)[0].strip()


def pin_requirement_line(line: str) -> str:
    """Pin loose version constraints to the minimum compatible version.

    ``>=X.Y.Z`` → ``==X.Y.Z`` (pin to lower bound).
    ``>X.Y.Z``  → ``==X.Y.Z+1`` (bump the least-significant component by 1).
    ``<=`` / ``==`` / ``~=`` / ``!=`` → keep as-is.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return line
    if stripped.startswith(("-", "git+", "http://", "https://")):
        return line

    # Preserve trailing markers and comments
    marker = ""
    if ";" in stripped:
        stripped, marker = stripped.split(";", 1)
        stripped, marker = stripped.strip(), " ; " + marker.strip()

    match = re.match(r"^(.+?)\s*(>=|>|<=|==|~=|!=|<)\s*(.+)$", stripped)
    if not match:
        return line

    pkg_part, op, ver = match.group(1), match.group(2), match.group(3).strip()

    if op == ">=":
        return f"{pkg_part}=={ver}{marker}"
    elif op == ">":
        pinned = _bump_version(ver)
        if pinned is None:
            return line
        return f"{pkg_part}=={pinned}{marker}"
    else:
        return line


def _bump_version(ver: str) -> str | None:
    """Bump the least-significant numeric component of a version by 1.

    ``1.18.1`` → ``1.18.2``, ``1.18`` → ``1.19``, ``1`` → ``2``.
    Returns None if the version cannot be parsed.
    """
    parts = ver.split(".")
    for i in range(len(parts) - 1, -1, -1):
        m = re.match(r"^(\d+)(.*)$", parts[i])
        if m:
            num, suffix = int(m.group(1)), m.group(2)
            parts[i] = str(num + 1) + suffix
            return ".".join(parts)
    return None


def _local_project_names(repo_dir: Path) -> set[str]:
    names = {_normalize_project_name(repo_dir.name)}
    for child in repo_dir.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(_normalize_project_name(child.name))
    setup_path = repo_dir / "setup.py"
    if setup_path.exists():
        text = setup_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"name\s*=\s*['\"]([^'\"]+)['\"]", text):
            names.add(_normalize_project_name(match.group(1)))
    return {name for name in names if name}


def _is_self_vcs_requirement(line: str, local_names: set[str]) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    lowered = stripped.lower()
    if not (lowered.startswith("git+") or "git+" in lowered):
        return False

    url_part = stripped
    if url_part.startswith("-e "):
        url_part = url_part[3:].strip()
    if "#egg=" in url_part:
        egg = url_part.split("#egg=", 1)[1].split("&", 1)[0]
        if _normalize_project_name(egg) in local_names:
            return True

    parsed = urlparse(url_part[4:] if url_part.startswith("git+") else url_part)
    repo_name = Path(parsed.path).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return _normalize_project_name(repo_name) in local_names


def _normalize_project_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _requirement_is_already_importable(line: str) -> bool:
    package_name = _extract_requirement_name(line)
    if not package_name:
        return False
    module_name = _module_name_for_package(package_name)
    return importlib.util.find_spec(module_name) is not None


def _target_bootstrap_specs(requirements_path: Path) -> list[str]:
    specs = []
    for line in requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        package_name = _extract_requirement_name(line)
        if not package_name:
            continue
        if _normalize_project_name(package_name) not in _TARGET_BOOTSTRAP_PACKAGES:
            continue
        stripped = line.split("#", 1)[0].strip()
        specs.append(stripped if any(op in stripped for op in ("==", ">=", "<=", "~=", "!=", ">", "<")) else install_spec_for_package(package_name))
    return _dedupe_preserve_order(specs)


def _extract_requirement_name(line: str) -> str | None:
    stripped = line.split("#", 1)[0].strip()
    if not stripped or stripped.startswith("-") or stripped.startswith(("git+", "http://", "https://")):
        return None
    name = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|===)\s*", stripped, maxsplit=1)[0]
    name = name.split(";", 1)[0].strip()
    name = re.sub(r"\[.*?\]", "", name).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return None
    return name


def _module_name_for_package(package_name: str) -> str:
    normalized = package_name.lower().replace("-", "_")
    renames = {
        "opencv_python": "cv2",
        "opencv_python_headless": "cv2",
        "opencv_contrib_python": "cv2",
        "opencv_contrib_python_headless": "cv2",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "scikit_learn": "sklearn",
        "scikit_image": "skimage",
        "open_clip_torch": "open_clip",
    }
    return renames.get(normalized, normalized)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _display_argv(argv: list[str], max_len: int = 140) -> str:
    display = " ".join(argv)
    if len(display) <= max_len:
        return display
    return display[: max_len - 3] + "..."


def _ensure_target_namespace_packages(site_path: Path) -> None:
    mpl_toolkits = site_path / "mpl_toolkits"
    if mpl_toolkits.exists():
        init_path = mpl_toolkits / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
                encoding="utf-8",
            )
