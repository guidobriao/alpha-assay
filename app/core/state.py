from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.benchmark.schema import BenchmarkRunResult, BenchmarkSpec


BackendType = Literal["none", "local", "venv", "conda", "docker"]
InputType = Literal["arxiv", "pdf", "unknown"]
TaskStatus = Literal[
    "created",
    "paper_ingested",
    "paper_understood",
    "repo_found",
    "repo_evaluated",
    "runtime_decided",
    "env_built",
    "smoke_ran",
    "benchmark_planned",
    "benchmark_ran",
    "reproduction_ran",
    "report_written",
    "cancelled",
    "failed",
]

FinalStatus = Literal[
    "benchmark_success",
    "reproduction_success",
    "success",
    "partial_success_help_only",
    "repo_found_but_env_failed",
    "repo_found_but_smoke_failed",
    "repo_found_but_reproduction_failed",
    "repo_found_but_benchmark_failed",
    "repo_found_reproduction_not_run",
    "repo_found_benchmark_not_run",
    "repo_found_smoke_not_run",
    "repo_not_found",
    "paper_parse_failed",
    "skipped_docker",
    "failed",
]


class PaperInput(BaseModel):
    raw_input: str
    input_type: InputType = "unknown"
    local_pdf_path: Optional[str] = None
    arxiv_id: Optional[str] = None


class PaperMetadata(BaseModel):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_path: Optional[str] = None
    parsed_text_path: Optional[str] = None
    parse_confidence: float = 0.0


class ReproductionBrief(BaseModel):
    task: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    method_keywords: list[str] = Field(default_factory=list)
    github_links_in_paper: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    benchmark_protocol: dict[str, Any] = Field(default_factory=dict)


class RepoCandidate(BaseModel):
    url: str
    owner: Optional[str] = None
    name: Optional[str] = None
    stars: Optional[int] = None
    source: Literal["paper", "github_search", "manual", "local"] = "github_search"
    score: float = 0.0
    confidence: Literal["high", "medium", "low"] = "low"
    reasons: list[str] = Field(default_factory=list)
    local_path: Optional[str] = None


class RepoEvaluation(BaseModel):
    repo_dir: Optional[str] = None

    has_readme: bool = False
    has_requirements: bool = False
    has_environment_yml: bool = False
    has_dockerfile: bool = False
    has_setup_py_or_pyproject: bool = False

    candidate_scripts: list[str] = Field(default_factory=list)
    candidate_configs: list[str] = Field(default_factory=list)

    runnable_score: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    benchmark_surface: dict[str, Any] = Field(default_factory=dict)


class EnvironmentBuildResult(BaseModel):
    dockerfile_path: Optional[str] = None
    image_tag: Optional[str] = None
    environment_path: Optional[str] = None
    python_executable: Optional[str] = None
    python_paths: list[str] = Field(default_factory=list)
    build_success: bool = False
    skipped: bool = False
    failure_type: Optional[str] = None
    failure_summary: Optional[str] = None
    build_log_path: Optional[str] = None
    install_actions: list[dict[str, Any]] = Field(default_factory=list)


class SmokeCommand(BaseModel):
    argv: list[str]
    display: str
    kind: Literal["help", "demo", "pytest"] = "help"


class SmokeRunResult(BaseModel):
    command: Optional[SmokeCommand] = None
    success: bool = False
    exit_code: Optional[int] = None
    timed_out: bool = False
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    summary: Optional[str] = None
    failure_type: Optional[str] = None
    failure_evidence: Optional[str] = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    repair_actions: list[dict[str, Any]] = Field(default_factory=list)


class ReproductionCommand(BaseModel):
    argv: list[str]
    display: str
    kind: Literal["demo", "example", "evaluation", "pytest"] = "demo"
    reason: Optional[str] = None


class ReproductionRunResult(BaseModel):
    command: Optional[ReproductionCommand] = None
    eligible: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    success: bool = False
    exit_code: Optional[int] = None
    timed_out: bool = False
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    summary: Optional[str] = None
    failure_type: Optional[str] = None
    failure_evidence: Optional[str] = None
    output_artifacts: list[str] = Field(default_factory=list)
    command_candidates: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    reference_results: dict[str, Any] = Field(default_factory=dict)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)


RuntimeDevice = Literal["cpu", "cuda", "skip"]
CudaRequirement = Literal["not_needed", "optional", "required", "unknown"]


class HostCudaInfo(BaseModel):
    has_nvidia_smi: bool = False
    has_gpu: bool = False
    gpu_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    nvcc_version: str | None = None
    error: str | None = None


class RuntimeDecision(BaseModel):
    cuda_requirement: CudaRequirement = "unknown"
    selected_device: RuntimeDevice = "cpu"
    torch_variant: Literal["cpu", "cuda", "unknown"] = "cpu"
    torch_version_constraint: str | None = None
    cuda_wheel_tag: str | None = None
    compatible: bool = True
    skip_execution: bool = False
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    host_cuda: HostCudaInfo = Field(default_factory=HostCudaInfo)
    install_plan: list[list[str]] = Field(default_factory=list)


class ReportResult(BaseModel):
    final_status: FinalStatus
    report_markdown_path: str
    report_json_path: Optional[str] = None
    short_conclusion: str


class StepTiming(BaseModel):
    step: str
    started_at: str
    ended_at: str
    duration_ms: int
    success: bool


class ApiCallRecord(BaseModel):
    provider: str
    purpose: str
    success: bool
    duration_ms: Optional[int] = None


class TaskState(BaseModel):
    task_id: str
    input_value: str
    workspace_dir: str
    task_dir: str
    created_at: datetime = Field(default_factory=datetime.now)
    backend: BackendType = "conda"

    paper_input: Optional[PaperInput] = None
    paper_metadata: Optional[PaperMetadata] = None
    reproduction_brief: Optional[ReproductionBrief] = None

    repo_candidates: list[RepoCandidate] = Field(default_factory=list)
    selected_repo: Optional[RepoCandidate] = None
    repo_evaluation: Optional[RepoEvaluation] = None

    env_build: Optional[EnvironmentBuildResult] = None
    runtime_decision: Optional[RuntimeDecision] = None
    smoke_run: Optional[SmokeRunResult] = None
    benchmark_plan: list[BenchmarkSpec] = Field(default_factory=list)
    benchmark_run: Optional[BenchmarkRunResult] = None
    reproduction_run: Optional[ReproductionRunResult] = None
    report: Optional[ReportResult] = None

    step_timings: list[StepTiming] = Field(default_factory=list)
    api_calls: list[ApiCallRecord] = Field(default_factory=list)

    status: TaskStatus = "created"
    errors: list[dict[str, Any]] = Field(default_factory=list)
