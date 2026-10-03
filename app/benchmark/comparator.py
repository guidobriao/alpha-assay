from __future__ import annotations

from typing import Any

from app.benchmark.schema import BenchmarkSpec


def compare_metrics(spec: BenchmarkSpec, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    reference = spec.reference.get("metrics") if isinstance(spec.reference, dict) else None
    if not isinstance(reference, dict) or not reference:
        return []

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
        tolerance = max(0.05 * abs(float(expected)), 0.02)
        comparisons.append({
            "metric": key,
            "actual": actual,
            "expected": expected,
            "delta": delta,
            "relative_delta": rel,
            "status": "matched" if abs(delta) <= tolerance else "different",
        })
    return comparisons


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
