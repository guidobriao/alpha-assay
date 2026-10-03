from __future__ import annotations

"""CUDA availability detection for benchmark execution."""

import os
import subprocess
from pathlib import Path

from app.core.state import TaskState


def benchmark_python_for_state(state: TaskState) -> str:
    if state.backend in {"conda", "venv"} and state.env_build and state.env_build.python_executable:
        return state.env_build.python_executable
    return "python"


def is_cuda_available_for_state(state: TaskState, cwd: Path | None = None, timeout: int = 20) -> bool:
    python = benchmark_python_for_state(state)
    try:
        proc = subprocess.run(
            [python, "-c", "import torch; print(bool(torch.cuda.is_available()))"],
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=os.environ.copy(),
        )
        return proc.returncode == 0 and proc.stdout.strip().lower() == "true"
    except Exception:
        return False
