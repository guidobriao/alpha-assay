from __future__ import annotations

import shutil
import subprocess

from app.core.state import HostCudaInfo


def detect_host_cuda() -> HostCudaInfo:
    """Detect NVIDIA CUDA capabilities on this host."""
    info = HostCudaInfo()

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        info.error = "nvidia-smi not found"
        return info

    info.has_nvidia_smi = True

    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            if parts:
                info.has_gpu = True
                info.gpu_name = parts[0] if len(parts) >= 1 else None
                info.driver_version = parts[1] if len(parts) >= 2 else None

        proc2 = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10,
        )
        if proc2.returncode == 0:
            for line in proc2.stdout.splitlines():
                if "CUDA Version:" in line:
                    info.cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
                    break
    except Exception:
        info.error = "nvidia-smi query failed"

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            proc3 = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, timeout=5,
            )
            if proc3.returncode == 0:
                for line in proc3.stdout.splitlines():
                    if "release" in line:
                        info.nvcc_version = line.split("release")[-1].strip().rstrip(",")
                        break
        except Exception:
            pass

    return info
