from __future__ import annotations

import re
from typing import Any

from app.benchmark.finance_schema import FINANCE_TASK_FAMILIES
from app.benchmark.schema import BenchmarkSpec


def compare_metrics(spec: BenchmarkSpec, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    reference = spec.reference.get("metrics") if isinstance(spec.reference, dict) else None
    if not isinstance(reference, dict) or not reference:
        return []

    is_finance = spec.task_family in FINANCE_TASK_FAMILIES
    comparisons: list[dict[str, Any]] = []
    for key, expected in reference.items():
        actual = _metric_value(metrics, key)
        if actual is None and key == "Precision":
            actual = metrics.get("Prec")
        if actual is None and key == "MatchingScore":
            actual = metrics.get("MScore")
        if isinstance(actual, str) or isinstance(expected, str) or isinstance(actual, bool) or isinstance(expected, bool):
            comparisons.append({
                "metric": key,
                "actual": actual,
                "expected": expected,
                "status": "matched" if actual == expected else "different",
            })
            continue
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            comparisons.append({
                "metric": key,
                "actual": actual,
                "expected": expected,
                "status": "missing_actual" if actual is None else "not_numeric",
            })
            continue
        delta = float(actual) - float(expected)
        rel = delta / float(expected) if expected else None
        canonical: str | None = None
        if is_finance:
            canonical = _canonical_metric_key(spec, key)
            tolerance = _finance_tolerance(canonical, float(expected))
        else:
            tolerance = max(0.05 * abs(float(expected)), 0.02)
        status = "matched" if abs(delta) <= tolerance else "different"
        comparison: dict[str, Any] = {
            "metric": key,
            "actual": actual,
            "expected": expected,
            "delta": delta,
            "relative_delta": rel,
            "status": status,
        }
        if (
            is_finance
            and status == "different"
            and canonical in _SIGN_CONSISTENT_METRICS
            and _signs_agree(float(actual), float(expected))
        ):
            comparison["status"] = "sign_matched"
            comparison["note"] = (
                "direction consistent with the paper claim; magnitude differs "
                "beyond tolerance (possible sample or specification drift)"
            )
        comparisons.append(comparison)
    return comparisons


# Metrics whose *sign* is the testable claim: a same-sign different-magnitude
# result still says the paper's direction replicated.
_SIGN_CONSISTENT_METRICS = frozenset({
    "alpha",
    "mean_return_spread",
    "car",
    "factor_mean_return",
})


def _signs_agree(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0) or (a == 0 and b == 0)


def _finance_tolerance(canonical: str, expected: float) -> float:
    """Domain-informed tolerances. Significance statistics compare on an
    absolute scale; per-month return measures get a 10bp/month floor;
    annualized quantities a 1pp floor."""
    if canonical in ("nw_t_stat", "t_stat"):
        return 0.5
    if canonical in ("sharpe_ratio", "information_ratio"):
        return 0.2
    if canonical in ("alpha", "mean_return_spread", "car", "factor_mean_return"):
        return max(0.15 * abs(expected), 0.10)
    if canonical in ("annualized_return", "r2"):
        return max(0.15 * abs(expected), 1.0)
    if canonical == "max_drawdown":
        return max(0.15 * abs(expected), 2.0)
    return max(0.05 * abs(expected), 0.02)


def _canonical_metric_key(spec: BenchmarkSpec, key: str) -> str:
    for metric in spec.expected_metrics:
        if metric.name == key or metric.canonical_name == key:
            return metric.canonical_name or metric.name
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    current: Any = metrics
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def protocol_match(spec: BenchmarkSpec) -> dict[str, Any]:
    if spec.level == "L3":
        status = "target_protocol"
    elif spec.level == "L2":
        status = "official_or_bundled_benchmark"
    elif spec.level == "L1":
        status = "demo_or_readme_protocol"
    else:
        status = "smoke_protocol"
    return {
        "status": status,
        "level": spec.level,
        "task_family": spec.task_family,
        "dataset": spec.dataset.model_dump(),
        "model": spec.model.model_dump(),
        "reference_scope": spec.reference.get("scope") if isinstance(spec.reference, dict) else None,
    }
