"""LLM-assisted metric parsing for unknown task families.

When the deterministic parser (parsers.py) returns empty results and the
task family is not one of the four specialist families, this module tries
LLM-based extraction as a last resort.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.benchmark.parsers import parse_generic_metrics
from app.benchmark.schema import BenchmarkSpec
from app.tools.llm import call_llm_json

logger = logging.getLogger(__name__)

_METRIC_EXTRACTION_PROMPT = """\
You are a metric extraction assistant.

Given benchmark command output and expected metric names, extract numerical
values.  Only include values you are confident about.  Return an empty
object if nothing looks like a metric.

Return a JSON object with exactly these keys:
- "metrics": object mapping metric names to their float values
- "evidence": list of short strings copied from the output that support each value
- "confidence": float between 0 and 1

Expected metrics: {expected_metrics}
"""


def parse_with_llm_fallback(
    spec: BenchmarkSpec,
    stdout: str,
    stderr: str,
    repo_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Try deterministic parsers first, then fall back to LLM extraction."""
    # Step 1: generic regex parser
    metrics = parse_generic_metrics(stdout + "\n" + stderr)
    if metrics:
        return metrics

    # Step 2: JSON file parser (if configured)
    parser_cfg = spec.parser
    if parser_cfg.get("type") == "json_file":
        rel_path = parser_cfg.get("path")
        if isinstance(rel_path, str):
            json_path = repo_dir / rel_path
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        numeric = {
                            k: v for k, v in data.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            return numeric
                except Exception:
                    pass

    # Step 3: LLM-based extraction
    metrics = _llm_extract_metrics(spec, stdout, stderr)
    return metrics or {}


def _llm_extract_metrics(
    spec: BenchmarkSpec,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    """Ask the LLM to extract metric values from benchmark output."""
    expected = [m.name for m in spec.expected_metrics]
    if not expected:
        expected = ["accuracy", "f1", "precision", "recall", "loss"]

    combined = stdout[-5000:] + "\n" + stderr[-3000:]
    if not combined.strip():
        return {}

    user_payload = json.dumps({
        "expected_metrics": expected,
        "task_family": spec.task_family,
        "benchmark_title": spec.title,
        "output_tail": combined,
    }, ensure_ascii=False)

    result = call_llm_json(
        system_prompt=_METRIC_EXTRACTION_PROMPT,
        user_prompt=user_payload,
        purpose="metric_extraction",
        max_tokens=1024,
    )
    if not isinstance(result, dict):
        return {}

    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    numeric: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric[key] = float(value)

    if numeric:
        logger.info(
            "LLM metric extraction found %d metric(s) for %s",
            len(numeric), spec.id,
        )
    return numeric
