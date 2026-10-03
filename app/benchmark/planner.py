from __future__ import annotations

import logging
from pathlib import Path

from app.benchmark.adapters import (
    ASRAdapter,
    AdapterContext,
    LocalFeatureMatchingAdapter,
    SequenceLabelingAdapter,
    ZeroShotClassificationAdapter,
)
from app.benchmark.generic_planner import GenericLLMBenchmarkPlanner
from app.benchmark.ontology import classify_task_ontology
from app.benchmark.plan_validator import BenchmarkPlanValidator
from app.benchmark.repo_affordance_scanner import scan_repo_affordances
from app.benchmark.schema import BenchmarkSpec, ExecutionBudget, KNOWN_TASK_FAMILIES
from app.benchmark.script_miner import mine_script_signals
from app.core.naming import stable_paper_slug
from app.core.state import TaskState

logger = logging.getLogger(__name__)

ADAPTERS = {
    "local_feature_matching": LocalFeatureMatchingAdapter(),
    "zero_shot_classification": ZeroShotClassificationAdapter(),
    "asr": ASRAdapter(),
    "sequence_labeling": SequenceLabelingAdapter(),
}

LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def plan_benchmarks(state: TaskState, budget: ExecutionBudget | None = None) -> list[BenchmarkSpec]:
    if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
        return []

    budget = budget or ExecutionBudget()
    repo_dir = Path(state.repo_evaluation.repo_dir)
    readme = read_readme(repo_dir, limit=30000)
    brief = state.reproduction_brief
    scripts = _expanded_scripts(repo_dir, state.repo_evaluation.candidate_scripts)

    ontology = classify_task_ontology(
        task=brief.task if brief else None,
        datasets=brief.datasets if brief else [],
        metrics=brief.metrics if brief else [],
        keywords=brief.method_keywords if brief else [],
        repo_text=readme,
        scripts=scripts,
    )
    task_family = ontology.family

    context = AdapterContext(
        workspace_dir=Path(state.workspace_dir),
        repo_dir=repo_dir,
        readme_text=readme,
        task=brief.task if brief else None,
        datasets=brief.datasets if brief else [],
        metrics=brief.metrics if brief else [],
        method_keywords=brief.method_keywords if brief else [],
        scripts=scripts,
        budget=budget,
        paper_slug=stable_paper_slug(state),
    )

    # Specialist path — fast, deterministic
    adapter = ADAPTERS.get(task_family)
    if adapter is not None:
        gate = _specialist_adapter_gate(
            task_family=task_family,
            task=brief.task if brief else None,
            datasets=brief.datasets if brief else [],
            metrics=brief.metrics if brief else [],
            keywords=brief.method_keywords if brief else [],
            repo_text=readme,
            scripts=scripts,
            confidence=ontology.confidence,
        )
        if not gate["allowed"]:
            logger.info(
                "Specialist adapter %r blocked by evidence gate: %s",
                task_family,
                gate["reason"],
            )
            return _generic_plan(state, context, repo_dir, readme, scripts, task_family, brief, budget)
        return adapter.propose_benchmarks(context)

    # Generic LLM planner fallback for unknown task families
    logger.info("No specialist adapter for task family %r — using generic planner", task_family)
    return _generic_plan(state, context, repo_dir, readme, scripts, task_family, brief, budget)


def _generic_plan(
    state: TaskState,
    context: AdapterContext,
    repo_dir: Path,
    readme: str,
    scripts: list[str],
    task_family: str,
    brief: object,
    budget: ExecutionBudget,
) -> list[BenchmarkSpec]:
    """Generate benchmark candidates via generic LLM planner + validation."""
    from app.core.state import ReproductionBrief

    brief_typed = brief if isinstance(brief, ReproductionBrief) else None

    affordances = scan_repo_affordances(repo_dir, readme, scripts)
    planner = GenericLLMBenchmarkPlanner(budget=budget)
    raw_specs = planner.propose_benchmarks(context, affordances, task_family, brief_typed)

    if not raw_specs:
        return []

    validator = BenchmarkPlanValidator(repo_dir, state.repo_evaluation.candidate_scripts, budget)
    return validator.validate(raw_specs)


def select_best_benchmark(specs: list[BenchmarkSpec], budget: ExecutionBudget | None = None) -> BenchmarkSpec | None:
    if not specs:
        return None
    budget = budget or ExecutionBudget()
    max_rank = LEVEL_RANK[budget.target_level]
    runnable = [
        spec for spec in specs
        if spec.runnable and LEVEL_RANK[spec.level] <= max_rank and not _dataset_budget_block_reason(spec, budget)
    ]
    if not runnable:
        return None
    return sorted(runnable, key=lambda spec: (LEVEL_RANK[spec.level], _spec_score(spec)), reverse=True)[0]


def downgrade_reasons(specs: list[BenchmarkSpec], selected: BenchmarkSpec | None, budget: ExecutionBudget | None = None) -> list[str]:
    budget = budget or ExecutionBudget()
    reasons: list[str] = []
    target_rank = LEVEL_RANK[budget.target_level]
    achieved_rank = LEVEL_RANK[selected.level] if selected else -1
    if achieved_rank < target_rank:
        blocked = [
            spec for spec in specs
            if LEVEL_RANK[spec.level] > achieved_rank and LEVEL_RANK[spec.level] <= target_rank
        ]
        for spec in blocked:
            reason = _dataset_budget_block_reason(spec, budget) or spec.feasibility.get("reason") or spec.fallback_reason
            if reason:
                reasons.append(f"{spec.level} {spec.title}: {reason}")
    if selected and selected.fallback_reason:
        reasons.append(selected.fallback_reason)
    return _dedupe(reasons)


def read_readme(repo_dir: Path, limit: int = 8000) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    return ""


def _expanded_scripts(repo_dir: Path, candidate_scripts: list[str]) -> list[str]:
    mined = mine_script_signals(repo_dir, candidate_scripts)
    scripts = [signal.path for signal in mined]
    for script in candidate_scripts:
        if script not in scripts:
            scripts.append(script)
    return scripts


def _spec_score(spec: BenchmarkSpec) -> int:
    score = 0
    if spec.reference.get("metrics"):
        score += 5
    if spec.command_kind == "official_script":
        score += 4
    if spec.command_kind == "generated_runner":
        score += 2
    score += len(spec.evidence)
    return score


def _dataset_budget_block_reason(spec: BenchmarkSpec, budget: ExecutionBudget) -> str | None:
    if budget.allow_large_downloads:
        return None

    size_gb = spec.dataset.size_gb
    if size_gb is not None:
        if size_gb > budget.max_dataset_size_gb:
            return (
                f"estimated dataset size {size_gb:.2f}GB exceeds the "
                f"{budget.max_dataset_size_gb:.2f}GB benchmark budget"
            )
        return None

    if spec.level == "L3" and spec.dataset.source not in {"bundled", "readme", "synthetic"}:
        return (
            "dataset size is unknown, so the L3 run is held until the data "
            f"footprint can be verified under {budget.max_dataset_size_gb:.2f}GB"
        )
    return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


_SPECIALIST_STRONG_TERMS: dict[str, list[str]] = {
    "local_feature_matching": [
        "feature matching", "local feature", "keypoint", "homography", "pose estimation",
        "superglue", "super glue", "superpoint", "super point", "loftr", "lightglue",
        "light glue", "xfeat", "scannet", "scan net", "megadepth", "hpatches",
        "phototourism", "photo tourism", "match pairs", "matching",
    ],
    "zero_shot_classification": [
        "clip", "zero shot", "image text", "image-text", "imagenet", "cifar",
        "classification", "openai clip",
    ],
    "asr": [
        "speech recognition", "automatic speech", "transcribe", "transcription",
        "librispeech", "whisper", "audio", "voice", "speech",
    ],
    "sequence_labeling": [
        "sequence labeling", "named entity", "named entities", "ner", "conll",
        "flair", "tagger", "token classification",
    ],
}

_SPECIALIST_METRIC_TERMS: dict[str, list[str]] = {
    "local_feature_matching": ["auc", "auc 5", "auc 10", "auc 20", "matching score", "mscore"],
    "zero_shot_classification": ["top 1", "top 5", "accuracy", "acc"],
    "asr": ["wer", "cer", "word error rate", "character error rate"],
    "sequence_labeling": ["f1", "span f1", "precision", "recall"],
}

_SPECIALIST_CONFLICT_TERMS: dict[str, list[str]] = {
    "local_feature_matching": ["speech recognition", "librispeech", "whisper", "ner", "conll"],
    "zero_shot_classification": ["speech recognition", "librispeech", "wer", "cer", "ner", "conll"],
    "asr": [
        "feature matching", "local feature", "keypoint", "homography", "superglue",
        "super glue", "superpoint", "super point", "loftr", "lightglue", "auc 5",
        "auc 10", "auc 20",
    ],
    "sequence_labeling": ["speech recognition", "librispeech", "feature matching", "homography"],
}


def _specialist_adapter_gate(
    *,
    task_family: str,
    task: str | None,
    datasets: list[str],
    metrics: list[str],
    keywords: list[str],
    repo_text: str,
    scripts: list[str],
    confidence: float,
) -> dict[str, object]:
    """Require positive evidence before running a specialist benchmark adapter.

    Single noisy metric hits from PDF heuristics are common. For example, a
    local-feature paper can be misread as GLUE/WER from broken PDF text. The
    gate keeps specialist adapters evidence-driven and sends weak cases to the
    generic planner instead of executing an unrelated benchmark.
    """
    evidence_text = _normalize_evidence_text(" ".join([
        task or "",
        " ".join(datasets),
        " ".join(metrics),
        " ".join(keywords),
        repo_text[:30000],
        " ".join(scripts),
    ]))

    strong_hits = _hits(evidence_text, _SPECIALIST_STRONG_TERMS.get(task_family, []))
    metric_hits = _hits(evidence_text, _SPECIALIST_METRIC_TERMS.get(task_family, []))
    conflict_hits = _hits(evidence_text, _SPECIALIST_CONFLICT_TERMS.get(task_family, []))

    if strong_hits:
        return {
            "allowed": True,
            "reason": "strong specialist evidence",
            "evidence": strong_hits[:6],
            "confidence": confidence,
        }

    if conflict_hits and not strong_hits:
        return {
            "allowed": False,
            "reason": f"conflicting evidence without direct {task_family} evidence: {conflict_hits[:4]}",
            "evidence": metric_hits[:6],
            "confidence": confidence,
        }

    # Metrics such as WER/F1/accuracy can be false positives in parsed PDFs.
    # Allow metric-only routing only when the classifier was decisive.
    if metric_hits and confidence >= 0.75:
        return {
            "allowed": True,
            "reason": "high-confidence metric evidence",
            "evidence": metric_hits[:6],
            "confidence": confidence,
        }

    return {
        "allowed": False,
        "reason": "insufficient specialist evidence",
        "evidence": [*strong_hits[:4], *metric_hits[:4]],
        "confidence": confidence,
    }


def _hits(text: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    for term in terms:
        normalized = _normalize_evidence_text(term)
        if normalized and normalized in text:
            hits.append(term)
    return hits


def _normalize_evidence_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
