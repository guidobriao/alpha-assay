from __future__ import annotations

from pathlib import Path
import subprocess

import jinja2

from app.core.file_utils import save_json
from app.core.state import TaskState, EnvironmentBuildResult
from app.tools.repo_tool import extract_pip_requirements_from_environment_file, find_requirement_files


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class DockerBuildAgent:
    def __init__(self, timeout_minutes: int = 30):
        self.timeout_minutes = timeout_minutes

    def run(self, state: TaskState) -> TaskState:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            state.env_build = EnvironmentBuildResult(
                build_success=False,
                failure_summary="Repo not evaluated",
            )
            state.status = "env_built"
            return state

        task_dir = Path(state.task_dir).resolve()
        env_dir = task_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = env_dir / "Dockerfile.generated"
        build_log_path = env_dir / "build.log"
        repo_dir = Path(state.repo_evaluation.repo_dir)
        generated_env_requirements = self._write_environment_requirements_file(repo_dir)
        requirement_files = [
            path.relative_to(repo_dir).as_posix()
            for path in find_requirement_files(repo_dir)
        ]

        template_path = _TEMPLATES_DIR / "Dockerfile.python.j2"
        template = jinja2.Template(template_path.read_text(encoding="utf-8"))

        dockerfile_content = template.render(
            requirements_files=requirement_files,
            has_setup=state.repo_evaluation.has_setup_py_or_pyproject,
            has_generated_environment_requirements=generated_env_requirements is not None,
        )
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

        image_tag = f"alpha-assay-{state.task_id}".lower().replace("_", "-")

        cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_tag,
            str(task_dir / "repos"),
        ]

        result = EnvironmentBuildResult(
            dockerfile_path=str(dockerfile_path),
            image_tag=image_tag,
            build_log_path=str(build_log_path),
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_minutes * 60,
            )

            build_log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

            result.build_success = proc.returncode == 0
            if not result.build_success:
                result.failure_type = classify_build_failure(build_log_path.read_text(encoding="utf-8", errors="ignore"))
                result.failure_summary = f"Docker build failed with exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            result.build_success = False
            result.failure_type = "timeout"
            result.failure_summary = "Docker build timed out"
        except Exception as e:
            result.build_success = False
            result.failure_type = "unknown"
            result.failure_summary = str(e)

        state.env_build = result
        save_json(env_dir / "environment_summary.json", result)
        state.status = "env_built"
        return state

    def _write_environment_requirements_file(self, repo_dir: Path) -> Path | None:
        requirements = extract_pip_requirements_from_environment_file(repo_dir)
        if not requirements:
            return None

        output_path = repo_dir / ".paper_smoke_environment_requirements.txt"
        output_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        return output_path


def classify_build_failure(log_text: str) -> str:
    text = log_text.lower()

    if "no matching distribution found" in text:
        return "package_not_found"
    if "could not find a version" in text:
        return "package_not_found"
    if "conflict" in text or "resolutionimpossible" in text:
        return "dependency_conflict"
    if "cuda" in text or "nvidia" in text:
        return "cuda_required"
    if "failed to establish a new connection" in text or "temporary failure" in text:
        return "network_failure"
    if "python_requires" in text or "requires python" in text:
        return "python_version_incompatible"

    return "unknown"
