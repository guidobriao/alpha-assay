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
        _record(results, "项目根目录", "pass", str(root), blocking=True)
    except Exception as e:
        _record(results, "项目根目录", "fail", str(e), blocking=True)

    # 2. logo/logo.png
    logo = find_project_root() / "logo" / "logo.png"
    _record(results, "Logo 文件", "pass" if logo.exists() else "fail",
            str(logo) if logo.exists() else "未找到", blocking=False)

    # 3. Python version
    _record(results, "Python 版本", "pass", f"Python {sys.version.split()[0]}")

    # 4. Git
    git = shutil.which("git")
    if git:
        try:
            ver = subprocess.run([git, "--version"], capture_output=True, text=True, timeout=5).stdout.strip()
            _record(results, "Git", "pass", ver)
        except Exception:
            _record(results, "Git", "fail", "无法执行")
    else:
        _record(results, "Git", "fail", "未找到")

    # 5. Conda
    conda = shutil.which("conda")
    if conda:
        try:
            ver = subprocess.run([conda, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            _record(results, "Conda", "pass", ver)
        except Exception:
            _record(results, "Conda", "fail", "无法执行")
    else:
        _record(results, "Conda", "fail", "未找到（将限制为 none/local 后端）", blocking=False)

    # 6. Workspace writable
    try:
        ws = default_project_workspace()
        ws.mkdir(parents=True, exist_ok=True)
        test = ws / ".preflight_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        _record(results, "工作目录", "pass", str(ws))
    except Exception as e:
        _record(results, "工作目录", "fail", str(e), blocking=True)

    # 7. PDF dir
    try:
        pdf = project_pdf_dir()
        pdf.mkdir(parents=True, exist_ok=True)
        _record(results, "PDF 目录", "pass", str(pdf))
    except Exception as e:
        _record(results, "PDF 目录", "fail", str(e), blocking=False)

    # 8. Textual / Rich / Pillow
    for pkg in ("textual", "rich"):
        try:
            __import__(pkg)
            _record(results, f"依赖 {pkg}", "pass", "已安装")
        except ImportError:
            _record(results, f"依赖 {pkg}", "fail", "未安装", blocking=True)

    try:
        from PIL import Image as _  # noqa: F401
        _record(results, "依赖 Pillow", "pass", "已安装")
    except ImportError:
        _record(results, "依赖 Pillow", "fail", "未安装（院徽 logo 无法渲染，非阻塞）", blocking=False)

    # 9. CUDA (non-blocking)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            r = subprocess.run([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                _record(results, "NVIDIA CUDA", "pass", r.stdout.strip().split("\n")[0])
            else:
                _record(results, "NVIDIA CUDA", "fail", "nvidia-smi 返回异常", blocking=False)
        except Exception:
            _record(results, "NVIDIA CUDA", "fail", "无法查询", blocking=False)
    else:
        _record(results, "NVIDIA CUDA", "fail", "未检测到（非阻塞）", blocking=False)

    # 10. Network
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=3)"],
            capture_output=True, timeout=5,
        )
        if proc.returncode == 0:
            _record(results, "网络连接", "pass", "PyPI 可达")
        else:
            _record(results, "网络连接", "fail", "PyPI 不可达（可能影响自动安装）", blocking=False)
    except Exception:
        _record(results, "网络连接", "fail", "检查超时（非阻塞）", blocking=False)

    return results


def _record(results: list[CheckItem], name: str, status: str, message: str = "", blocking: bool = False) -> None:
    results.append(CheckItem(name=name, status=status, message=message, blocking=blocking))
