from __future__ import annotations

"""Discover, inspect, and safely delete conda environments created by this project."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ENV_MARKER = ".paper_reproduct_agent_env.json"


@dataclass
class ProjectCondaEnv:
    index: int
    slug: str
    path: Path
    python_executable: Path | None = None
    exists: bool = True
    has_marker: bool = False
    marker: dict[str, Any] = field(default_factory=dict)
    task_count: int = 0
    last_used_at: float | None = None
    size_bytes: int | None = None


# ── Path helpers ───────────────────────────────────────────

def project_env_root(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / "envs"


def is_relative_to(child: Path, parent: Path) -> bool:
    """Python 3.8 compatible is_relative_to."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def env_python_path(env_path: Path) -> Path:
    if sys.platform == "win32":
        return env_path / "python.exe"
    return env_path / "bin" / "python"


# ── Marker read / write ────────────────────────────────────

def read_env_marker(env_path: Path) -> dict[str, Any]:
    marker_path = env_path / PROJECT_ENV_MARKER
    if not marker_path.exists():
        return {}
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_project_env_marker(
    env_path: Path,
    *,
    task_id: str = "",
    paper_slug: str = "",
    paper_path: str = "",
    repo_url: str = "",
    workspace: str = "",
    python_executable: str = "",
) -> None:
    marker = {
        "created_by": "alpha-assay",
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_id": task_id,
        "paper_slug": paper_slug,
        "paper_path": paper_path,
        "repo_url": repo_url,
        "workspace": str(workspace),
        "backend": "conda",
        "environment_path": str(env_path),
        "python_executable": python_executable,
    }
    marker_path = env_path / PROJECT_ENV_MARKER
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Discovery ───────────────────────────────────────────────

def discover_project_conda_envs(
    workspace: str | Path,
    *,
    include_unmarked_workspace_envs: bool = True,
) -> list[ProjectCondaEnv]:
    root = project_env_root(workspace)
    if not root.exists():
        return []

    results: list[ProjectCondaEnv] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_dir():
            continue

        py = env_python_path(entry)
        has_py = py.exists()
        marker = read_env_marker(entry)
        has_marker = bool(marker)

        # Include if: has marker, OR (in workspace/envs and has python)
        if not has_marker and not (include_unmarked_workspace_envs and has_py):
            continue

        results.append(ProjectCondaEnv(
            index=0,  # filled after sorting
            slug=marker.get("paper_slug", entry.name),
            path=entry,
            python_executable=py if has_py else None,
            exists=entry.exists(),
            has_marker=has_marker,
            marker=marker,
            last_used_at=entry.stat().st_mtime,
        ))

    for i, env in enumerate(results, 1):
        env.index = i

    return results


# ── Size (optional, on-demand) ──────────────────────────────

def calculate_dir_size(path: Path, limit_files: int = 100000) -> int | None:
    if not path.exists():
        return None
    total = 0
    count = 0
    try:
        for entry in path.rglob("*"):
            if count >= limit_files:
                return None
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
            count += 1
    except Exception:
        return None
    return total


# ── Safety ──────────────────────────────────────────────────

def is_safe_project_env_path(env_path: Path, workspace: str | Path) -> tuple[bool, str]:
    """Check whether a path is safe to delete as a project conda env."""
    resolved = env_path.resolve()

    if not resolved.exists():
        return True, "ok"  # already gone

    if not resolved.is_dir():
        return False, "path is not a directory"

    # Allowed: inside <workspace>/envs/
    if is_relative_to(resolved, project_env_root(workspace)):
        return True, "ok"

    # Allowed: has project marker
    if (resolved / PROJECT_ENV_MARKER).exists():
        return True, "ok"

    # Disallowed: looks like a base/system env
    name = resolved.name.lower()
    dangerous = {"base", "miniconda3", "anaconda3", "root"}
    if name in dangerous:
        return False, f"path looks like a base environment: {name}"

    # Disallowed: current running python
    if sys.executable.startswith(str(resolved)):
        return False, "cannot delete the currently running Python environment"

    return False, f"path is not under workspace/envs and has no project marker: {resolved}"


# ── Search ──────────────────────────────────────────────────

def find_env_by_selector(
    selector: str,
    envs: list[ProjectCondaEnv],
) -> ProjectCondaEnv | None:
    selector = selector.strip()
    if not selector:
        return None

    # By index
    try:
        idx = int(selector)
        for env in envs:
            if env.index == idx:
                return env
    except ValueError:
        pass

    # By slug
    for env in envs:
        if env.slug == selector:
            return env

    # By path match
    for env in envs:
        p = str(env.path)
        if selector == p or p.endswith(selector):
            return env

    return None


# ── Removal ─────────────────────────────────────────────────

def remove_conda_env(
    env: ProjectCondaEnv,
    *,
    workspace: str | Path,
    conda_executable: str | None = None,
    force_rmtree: bool = False,
) -> tuple[bool, str]:
    safe, reason = is_safe_project_env_path(env.path, workspace)
    if not safe:
        return False, f"refusing to delete: {reason}"

    if not env.path.exists():
        return True, "environment directory no longer exists"

    conda = conda_executable or shutil.which("conda")

    if conda:
        for cmd in (
            [conda, "env", "remove", "-p", str(env.path), "-y"],
            [conda, "remove", "-p", str(env.path), "--all", "-y"],
        ):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      encoding="utf-8", errors="ignore", timeout=300)
                if proc.returncode == 0 and not env.path.exists():
                    return True, f"deleted via conda: {env.path}"
            except Exception:
                pass

        if force_rmtree:
            shutil.rmtree(env.path, ignore_errors=True)
            return True, f"conda deletion failed, directory force-deleted: {env.path}"
        return False, "conda deletion failed; directory not deleted to avoid accidental removal."

    # No conda: only allow rmtree within workspace/envs
    if is_relative_to(env.path, project_env_root(workspace)):
        shutil.rmtree(env.path, ignore_errors=True)
        return True, f"conda not found, directory deleted: {env.path}"

    return False, "conda not found and path safety insufficient; refusing to delete."


# ── Format ──────────────────────────────────────────────────

def format_env_table(envs: list[ProjectCondaEnv]) -> str:
    lines = [
        "| # | Environment | Status | Marker | Python | Path |",
        "|---:|---|---:|---|---|",
    ]
    for e in envs:
        status = "exists" if e.exists else "missing"
        marker = "yes" if e.has_marker else "legacy"
        py = "yes" if e.python_executable and e.python_executable.exists() else "no"
        path_str = str(e.path)
        if len(path_str) > 60:
            path_str = "..." + path_str[-57:]
        lines.append(f"| {e.index} | `{e.slug[:30]}` | {status} | {marker} | {py} | `{path_str}` |")
    return "\n".join(lines)
