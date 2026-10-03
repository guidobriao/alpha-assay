from __future__ import annotations

from app.core.state import RuntimeDecision


def build_install_plan(decision: RuntimeDecision, python_bin: str = "python") -> list[list[str]]:
    """Generate pip install commands for the selected PyTorch variant."""
    if decision.selected_device == "skip":
        return []

    packages = ["torch", "torchvision", "torchaudio"]
    if decision.torch_version_constraint:
        packages = [f"{p}{decision.torch_version_constraint}" for p in packages]

    if decision.torch_variant == "cpu":
        return [[
            python_bin, "-m", "pip", "install",
            *packages,
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ]]

    if decision.torch_variant == "cuda" and decision.cuda_wheel_tag:
        return [[
            python_bin, "-m", "pip", "install",
            *packages,
            "--index-url", f"https://download.pytorch.org/whl/{decision.cuda_wheel_tag}",
        ]]

    return [[
        python_bin, "-m", "pip", "install",
        *packages,
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]]


TORCH_FAMILY = {"torch", "torchvision", "torchaudio"}


def strip_torch_family_requirements(lines: list[str]) -> list[str]:
    """Remove torch/torchvision/torchaudio lines from requirements."""
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-", "git+", "http")):
            result.append(line)
            continue
        name = stripped.split(";")[0].split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("~")[0].strip()
        if name.lower() in TORCH_FAMILY:
            continue
        result.append(line)
    return result
