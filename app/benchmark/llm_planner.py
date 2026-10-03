from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.benchmark.schema import BenchmarkSpec
from app.core.state import TaskState
from app.tools.llm import call_llm_json


_SYSTEM_PROMPT = """\
You are a benchmark protocol planning assistant for paper reproduction.

Review the paper/repository context and the deterministic benchmark candidates.
Do not invent paper-specific recipes. Prefer task-family protocols and official
repo benchmark/evaluation surfaces.

Return a JSON object with exactly these keys:
- "selected_spec_id": string or null. Pick one runnable candidate by id if the
  deterministic choice looks reasonable; otherwise choose a better runnable id.
- "task_family_confidence": float between 0 and 1.
- "protocol_notes": list of short strings about protocol alignment, missing data,
  or likely downgrade causes.
- "candidate_notes": object mapping candidate id to a short note.
"""


def llm_review_benchmark_plan(state: TaskState, specs: list[BenchmarkSpec]) -> dict[str, Any] | None:
    if not specs or not state.repo_evaluation:
        return None

    repo_dir = Path(state.repo_evaluation.repo_dir) if state.repo_evaluation.repo_dir else None
    readme_excerpt = _read_readme(repo_dir) if repo_dir else ""
    brief = state.reproduction_brief
    payload = {
        "paper": {
            "title": state.paper_metadata.title if state.paper_metadata else None,
            "arxiv_id": state.paper_metadata.arxiv_id if state.paper_metadata else None,
            "task": brief.task if brief else None,
            "datasets": brief.datasets if brief else [],
            "metrics": brief.metrics if brief else [],
            "method_keywords": brief.method_keywords if brief else [],
        },
        "repo": {
            "url": state.selected_repo.url if state.selected_repo else None,
            "candidate_scripts": state.repo_evaluation.candidate_scripts,
            "candidate_configs": state.repo_evaluation.candidate_configs,
            "risk_flags": state.repo_evaluation.risk_flags,
            "readme_excerpt": readme_excerpt,
        },
        "benchmark_candidates": [_spec_context(spec) for spec in specs],
    }
    return call_llm_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=False),
        purpose="benchmark_protocol_planning",
    )


def apply_llm_review(specs: list[BenchmarkSpec], review: dict[str, Any] | None) -> list[BenchmarkSpec]:
    if not specs or not review:
        return specs

    notes_by_id = review.get("candidate_notes")
    if not isinstance(notes_by_id, dict):
        notes_by_id = {}
    protocol_notes = review.get("protocol_notes")
    if not isinstance(protocol_notes, list):
        protocol_notes = []

    selected_id = review.get("selected_spec_id")
    for spec in specs:
        note = notes_by_id.get(spec.id)
        if isinstance(note, str) and note:
            spec.evidence.append(f"llm_note:{note[:200]}")
        if protocol_notes:
            spec.feasibility.setdefault("llm_protocol_notes", [str(n)[:200] for n in protocol_notes[:5]])

    if isinstance(selected_id, str) and selected_id:
        return sorted(
            specs,
            key=lambda spec: (spec.id == selected_id, spec.runnable, spec.level),
            reverse=True,
        )
    return specs


def _spec_context(spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "task_family": spec.task_family,
        "level": spec.level,
        "title": spec.title,
        "dataset": spec.dataset.model_dump(),
        "model": spec.model.model_dump(),
        "command": spec.command,
        "command_kind": spec.command_kind,
        "runnable": spec.runnable,
        "feasibility": spec.feasibility,
        "evidence": spec.evidence,
        "fallback_reason": spec.fallback_reason,
        "expected_metrics": [m.model_dump() for m in spec.expected_metrics],
    }


def _read_readme(repo_dir: Path | None, limit: int = 5000) -> str:
    if repo_dir is None:
        return ""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    return ""
