from __future__ import annotations

from pathlib import Path

from app.core.progress import emit_progress
from app.core.state import TaskState, RuntimeDecision, HostCudaInfo
from app.core.state import CudaRequirement, RuntimeDevice
from app.tools.cuda_detector import detect_host_cuda
from app.tools.torch_install_policy import build_install_plan

# LightGlue repo markers
_LIGHTGLUE_MARKERS = ["cvg/lightglue", "LightGlue", "lightglue"]


class RuntimeDecisionAgent:
    def run(self, state: TaskState) -> TaskState:
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            state.runtime_decision = RuntimeDecision(reason="repo not evaluated", skip_execution=False)
            state.status = "runtime_decided"
            return state

        repo_dir = Path(state.repo_evaluation.repo_dir)
        emit_progress("Decide runtime", "analysing CUDA requirements from paper and repo",
                       detail="检查论文和仓库的 CUDA 依赖声明")

        # Determine CUDA requirement
        cuda_req, evidence = _infer_cuda_requirement(state, repo_dir)
        emit_progress("Decide runtime", f"CUDA requirement: {cuda_req}",
                       detail=", ".join(evidence) if evidence else "未发现明确 CUDA 依赖声明的证据")

        # Select device
        device, reason = _select_device(cuda_req, repo_dir)
        emit_progress("Decide runtime", f"selected device: {device}", detail=reason)

        # Detect host CUDA
        host_cuda = detect_host_cuda()
        if host_cuda.has_gpu:
            emit_progress("Decide runtime",
                          f"host GPU: {host_cuda.gpu_name or 'unknown'}, CUDA {host_cuda.cuda_version or 'unknown'}",
                          detail=f"driver {host_cuda.driver_version or 'unknown'}")
        else:
            emit_progress("Decide runtime", "no NVIDIA GPU detected on host")

        # Build decision
        compatible = True
        skip = False
        cuda_wheel = None

        if device == "cuda" and not host_cuda.has_gpu:
            skip = True
            compatible = False
            reason += "；但本机未检测到 CUDA 环境，将跳过代码执行阶段"
            emit_progress("Decide runtime", "CUDA required but not available on host",
                          level="warning", detail="跳过代码执行")

        if device == "cuda" and host_cuda.has_gpu:
            cuda_wheel = _cuda_wheel_tag(host_cuda.cuda_version)
            if cuda_wheel:
                emit_progress("Decide runtime", f"CUDA wheel tag: {cuda_wheel}")
            else:
                compatible = False
                skip = True
                reason += "；无法确定 CUDA wheel tag，将跳过"

        decision = RuntimeDecision(
            cuda_requirement=cuda_req,
            selected_device="skip" if skip else device,
            torch_variant="cuda" if (device == "cuda" and compatible) else "cpu",
            cuda_wheel_tag=cuda_wheel,
            compatible=compatible,
            skip_execution=skip,
            reason=reason,
            evidence=evidence,
            host_cuda=host_cuda,
        )

        if not skip and decision.torch_variant != "unknown":
            decision.install_plan = build_install_plan(decision)

        state.runtime_decision = decision
        state.status = "runtime_decided"

        if decision.install_plan:
            emit_progress("Decide runtime", "PyTorch install plan generated",
                          detail=f"{len(decision.install_plan)} step(s)")
        emit_progress("Decide runtime", "runtime decision saved",
                      detail=reason, phase="finish")

        return state


def _infer_cuda_requirement(state: TaskState, repo_dir: Path) -> tuple[CudaRequirement, list[str]]:
    """Infer whether CUDA is required from paper metadata and repo contents."""
    evidence: list[str] = []

    # Check for LightGlue
    repo_url = (state.selected_repo.url or "").lower() if state.selected_repo else ""
    paper_title = (state.paper_metadata.title or "").lower() if state.paper_metadata else ""

    is_lightglue = any(m in repo_url for m in _LIGHTGLUE_MARKERS) or any(
        m in paper_title for m in _LIGHTGLUE_MARKERS
    )
    if is_lightglue:
        evidence.append("LightGlue 仓库/论文，benchmark 支持 CPU/CUDA，默认选择 CPU")
        return "optional", evidence

    # Check README
    readme_paths = sorted(repo_dir.glob("README*"))
    readme_text = ""
    for p in readme_paths[:1]:
        try:
            readme_text = p.read_text(encoding="utf-8", errors="ignore")[:10000].lower()
        except Exception:
            pass

    # Check paper abstract
    paper_text = ""
    if state.paper_metadata and state.paper_metadata.abstract:
        paper_text = state.paper_metadata.abstract.lower()

    combined = readme_text + "\n" + paper_text

    required_markers = [
        "cuda required", "requires cuda", "gpu required", "requires gpu",
        "only gpu", "only cuda", "nvidia gpu required", "tested on cuda only",
    ]
    for marker in required_markers:
        if marker in combined:
            evidence.append(f"发现强制 CUDA 声明：{marker}")
            return "required", evidence

    optional_markers = [
        "cuda optional", "gpu acceleration", "cuda acceleration",
        "use cuda if available", "cuda is available",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"",
        "--device", "device cuda",  # weak signal of CUDA support
    ]
    cpu_markers = ["--device cpu", "cpu benchmark", "runs on cpu", "device cpu"]

    if any(m in combined for m in optional_markers):
        if any(m in combined for m in cpu_markers):
            evidence.append("仓库/README 同时支持 CPU 和 CUDA")
        else:
            evidence.append("仓库/README 提到 CUDA 选项，未强制要求")
        return "optional", evidence

    # Check environment.yml / requirements for cudatoolkit
    req_files = sorted(repo_dir.glob("**/requirements*.txt")) + sorted(repo_dir.glob("**/environment*.yml"))
    for rf in req_files[:3]:
        try:
            text = rf.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if "cudatoolkit" in text or "cupy-cuda" in text:
            evidence.append(f"依赖文件中发现 CUDA 相关包：{rf.name}")
            return "required", evidence

    # Default: not needed / unknown
    if not evidence:
        evidence.append("未发现明确 CUDA 依赖声明，默认选择 CPU")
    return "not_needed", evidence


def _select_device(cuda_req: str, repo_dir: Path) -> tuple[RuntimeDevice, str]:
    """Select CPU/CUDA/skip based on CUDA requirement."""
    if cuda_req == "required":
        return "cuda", "论文/仓库明确要求 CUDA，尝试 CUDA 执行"
    if cuda_req == "optional":
        return "cpu", "论文/仓库支持 CUDA 但未强制要求，自动复现默认使用 CPU"
    return "cpu", "未发现 CUDA 依赖，默认使用 CPU"


def _cuda_wheel_tag(cuda_version: str | None) -> str | None:
    """Map CUDA version to PyTorch wheel tag."""
    if not cuda_version:
        return None
    try:
        ver = float(cuda_version[:4])
    except (ValueError, TypeError):
        return None
    if ver >= 12.4:
        return "cu124"
    if ver >= 12.1:
        return "cu121"
    if ver >= 11.8:
        return "cu118"
    return None
