"""Generic LLM-based benchmark planner for unknown task families.

When the specialist adapter lookup fails (task family is not one of the four
known families), this planner kicks in:

  1.  Extract a structured task spec from the paper brief.
  2.  Build a repo affordance summary from the scanner.
  3.  Ask the LLM to propose L0–L3 benchmark candidates.
  4.  Parse the LLM response into BenchmarkSpec objects.

The resulting specs are then passed through BenchmarkPlanValidator before
being returned to the caller.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from app.benchmark.adapters.base import AdapterContext
from app.benchmark.plan_validator import BenchmarkPlanValidator
from app.benchmark.repo_affordance_scanner import RepoAffordance, scan_repo_affordances
from app.benchmark.schema import (
    BenchmarkSpec,
    DatasetSpec,
    ExecutionBudget,
    MetricSpec,
    ModelSpec,
)
from app.core.state import ReproductionBrief, TaskState
from app.tools.llm import call_llm_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a benchmark plan generator for research paper reproduction.

Given the paper task, datasets, metrics, benchmark protocol, and repository
affordances, generate concrete benchmark plans at levels L0–L3.

## Output format

Return a JSON object with exactly one key:

"candidates": a list of objects, each with:
- "level": "L0" | "L1" | "L2" | "L3"
- "title": short descriptive string
- "command": list of strings (argv), e.g. ["python", "tools/test.py", "--eval", "bbox"].
  Use [] for non-executable manual protocols.
- "command_kind": "official_script" | "generated_runner" | "readme_example" | "manual_protocol"
- "dataset": {"name": str, "source": str, "size_estimate": str or null, "size_gb": float or null}
- "model": {"name": str or null, "checkpoint_source": str}
- "expected_metrics": list of {"name": str, "direction": "higher_is_better"|"lower_is_better"|"informational", "unit": str or null}
- "parser": {"type": "generic_metrics"} or {"type": "json_file", "path": "..."} or {}
- "feasibility": {"runnable": bool, "reason": str or null}
- "evidence": list of short strings
- "generated_script_name": str or null  (only for generated_runner)
- "generated_script_body": str or null  (only for generated_runner)
- "reference": {} (fill if paper provides reference results)

## Rules
- Only reference files that appear in the repository affordances.
- Never propose training commands.
- L3 must be marked runnable=false unless full dataset paths are confirmed.
- Prefer official repo scripts over generated runners.
- For L1 with sample data, generate a minimal Python script that loads the
  model and runs inference on one sample file.
- Each candidate must have a unique combination of level + title.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GenericLLMBenchmarkPlanner:
    """Generate benchmark candidates for arbitrary task families using LLM + repo affordances."""

    def __init__(self, budget: ExecutionBudget | None = None) -> None:
        self._budget = budget or ExecutionBudget()

    def propose_benchmarks(
        self,
        context: AdapterContext,
        affordances: RepoAffordance,
        task_family: str,
        brief: ReproductionBrief | None,
    ) -> list[BenchmarkSpec]:
        """Main entry point.  Returns 0-N raw benchmark specs (not yet validated)."""
        payload = self._build_payload(context, affordances, task_family, brief)
        raw = call_llm_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            purpose="generic_benchmark_planning",
            max_tokens=4096,
        )
        if raw is None:
            logger.info("Generic planner: LLM returned no response")
            return []
        return self._parse_response(raw, task_family)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        context: AdapterContext,
        affordances: RepoAffordance,
        task_family: str,
        brief: ReproductionBrief | None,
    ) -> dict[str, Any]:
        affordance_json = affordance_summary(affordances)
        payload: dict[str, Any] = {
            "task_family": task_family,
            "paper": {
                "task": brief.task if brief else None,
                "datasets": brief.datasets if brief else [],
                "metrics": brief.metrics if brief else [],
                "method_keywords": brief.method_keywords if brief else [],
                "benchmark_protocol": brief.benchmark_protocol if brief else {},
            },
            "repo": {
                "readme_excerpt": context.readme_text[:4000],
                "candidate_scripts": context.scripts[:50],
                "affordances": affordance_json,
            },
            "budget": {
                "target_level": self._budget.target_level,
                "max_dataset_size_gb": self._budget.max_dataset_size_gb,
            },
        }
        return payload

    def _parse_response(
        self,
        raw: dict[str, Any],
        task_family: str,
    ) -> list[BenchmarkSpec]:
        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            logger.warning("Generic planner: LLM response missing 'candidates' list")
            return []

        specs: list[BenchmarkSpec] = []
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                continue
            spec = _parse_candidate(item, task_family, idx)
            if spec is not None:
                specs.append(spec)
        return specs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_candidate(item: dict[str, Any], task_family: str, idx: int) -> BenchmarkSpec | None:
    try:
        level = item.get("level", "L0")
        title = item.get("title", f"generic_{task_family}_{level}")
        command = item.get("command", [])
        if isinstance(command, str):
            command = command.split()

        dataset_raw = item.get("dataset", {})
        dataset = DatasetSpec(
            name=dataset_raw.get("name", "unknown"),
            source=dataset_raw.get("source", "unknown"),
            size_estimate=dataset_raw.get("size_estimate"),
            size_gb=dataset_raw.get("size_gb"),
        )

        model_raw = item.get("model", {})
        model = ModelSpec(
            name=model_raw.get("name"),
            checkpoint_source=model_raw.get("checkpoint_source", "unknown"),
        )

        metrics_raw = item.get("expected_metrics", [])
        expected_metrics = [
            MetricSpec(
                name=m.get("name", f"metric_{i}"),
                direction=m.get("direction", "informational"),
                unit=m.get("unit"),
            )
            for i, m in enumerate(metrics_raw)
            if isinstance(m, dict)
        ]

        return BenchmarkSpec(
            id=item.get("id") or f"generic_{task_family}_{level}_{idx}",
            task_family=task_family,
            level=level,
            title=title,
            dataset=dataset,
            model=model,
            command=command,
            command_kind=item.get("command_kind", "generated_runner"),
            expected_metrics=expected_metrics,
            parser=item.get("parser", {"type": "generic_metrics"}),
            reference=item.get("reference", {}),
            feasibility=item.get("feasibility", {"runnable": bool(command)}),
            evidence=item.get("evidence", []),
            generated_script_name=item.get("generated_script_name"),
            generated_script_body=item.get("generated_script_body"),
        )
    except Exception as exc:
        logger.warning("Generic planner: failed to parse candidate %d: %s", idx, exc)
        return None


# ---------------------------------------------------------------------------
# Affordance summarisation (for LLM context)
# ---------------------------------------------------------------------------

_MAX_AFFORDANCE_CHARS = 4000


def affordance_summary(affordances: RepoAffordance) -> dict[str, Any]:
    """Serialize RepoAffordance to a compact dict suitable for LLM prompts."""
    entry_summaries = [
        {
            "path": e.path,
            "args": e.cli_args[:15],
            "desc": e.description[:100],
        }
        for e in affordances.entrypoints[:20]
    ]
    config_summaries = [
        {"path": c.path, "format": c.format, "keys": c.keys[:15]}
        for c in affordances.configs[:20]
    ]
    return {
        "entrypoints": entry_summaries,
        "configs": config_summaries,
        "datasets": [
            {"name": d.name, "source": d.source}
            for d in affordances.dataset_mentions[:15]
        ],
        "samples": [
            {"path": s.path, "suffix": s.suffix}
            for s in affordances.sample_files[:15]
        ],
        "checkpoints": affordances.model_checkpoints[:10],
        "frameworks": affordances.framework_signals,
    }
