from __future__ import annotations

"""Pre-flight system checks for the TUI splash screen."""

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.paths import find_project_root, project_pdf_dir, default_project_workspace


@dataclass
class CheckItem:
    name: str
    status: str = "pending"  # pending, running, pass, fail
    message: str = ""
    blocking: bool = False


def run_preflight() -> list[CheckItem]:
    results: list[CheckItem] = []

    # 1. Project root
    try:
        root = find_project_root()
        assert root.exists()
        _record(results, "Project root", "pass", str(root), blocking=True)
    except Exception as e:
        _record(results, "Project root", "fail", str(e), blocking=True)

    # 2. logo/logo.png
    logo = find_project_root() / "logo" / "logo.png"
    _record(results, "Logo file", "pass" if logo.exists() else "fail",
            str(logo) if logo.exists() else "not found", blocking=False)

    # 3. Python version
    _record(results, "Python version", "pass", f"Python {sys.version.split()[0]}")

    # 4. Git
    git = shutil.which("git")
    if git:
        try:
            ver = subprocess.run([git, "--version"], capture_output=True, text=True, timeout=5).stdout.strip()
            _record(results, "Git", "pass", ver)
        except Exception:
            _record(results, "Git", "fail", "cannot execute")
    else:
        _record(results, "Git", "fail", "not found")

    # 5. Conda
    conda = shutil.which("conda")
    if conda:
        try:
            ver = subprocess.run([conda, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            _record(results, "Conda", "pass", ver)
        except Exception:
            _record(results, "Conda", "fail", "cannot execute")
    else:
        _record(results, "Conda", "fail", "not found (limited to none/local backends)", blocking=False)

    # 6. Workspace writable
    try:
        ws = default_project_workspace()
        ws.mkdir(parents=True, exist_ok=True)
        test = ws / ".preflight_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        _record(results, "Workspace", "pass", str(ws))
    except Exception as e:
        _record(results, "Workspace", "fail", str(e), blocking=True)

    # 7. PDF dir
    try:
        pdf = project_pdf_dir()
        pdf.mkdir(parents=True, exist_ok=True)
        _record(results, "PDF directory", "pass", str(pdf))
    except Exception as e:
        _record(results, "PDF directory", "fail", str(e), blocking=False)

    # 8. Textual / Rich / Pillow
    for pkg in ("textual", "rich"):
        try:
            __import__(pkg)
            _record(results, f"Dependency {pkg}", "pass", "installed")
        except ImportError:
            _record(results, f"Dependency {pkg}", "fail", "not installed", blocking=True)

    try:
        from PIL import Image as _  # noqa: F401
        _record(results, "Dependency Pillow", "pass", "installed")
    except ImportError:
        _record(results, "Dependency Pillow", "fail", "not installed (emblem logo will not render, non-blocking)", blocking=False)

    # 9. CUDA (non-blocking)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            r = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                _record(results, "NVIDIA CUDA", "pass", r.stdout.strip().split("\n")[0])
            else:
                _record(results, "NVIDIA CUDA", "fail", "nvidia-smi returned an error", blocking=False)
        except Exception:
            _record(results, "NVIDIA CUDA", "fail", "query failed", blocking=False)
    else:
        _record(results, "NVIDIA CUDA", "fail", "not detected (non-blocking)", blocking=False)

    # 10. Network
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)"],
            capture_output=True, timeout=5,
        )
        if proc.returncode == 0:
            _record(results, "Network", "pass", "PyPI reachable")
        else:
            _record(results, "Network", "fail", "PyPI unreachable (may affect auto-install)", blocking=False)
    except Exception:
        _record(results, "Network", "fail", "check timed out (non-blocking)", blocking=False)

    return results


def _record(results: list[CheckItem], name: str, status: str, message: str = "", blocking: bool = False) -> None:
    results.append(CheckItem(name=name, status=status, message=message, blocking=blocking))
