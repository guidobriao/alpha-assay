"""Finance benchmark adapter.

Finance repos rarely ship runnable eval scripts with bundled data, so the
realistic ladder differs from CV/NLP:

L0 static feasibility (handled by the data access agent, not here)
L1 generated stats runner over the open-access Kenneth French library —
   validates the data + statistics pipeline end to end
L2 official repo script on small/bundled data (rare)
L3 full paper protocol with the study's own extracts (usually
   subscription-licensed; flagged by the data access agent)
"""
from __future__ import annotations

from pathlib import Path

from app.benchmark import dataset_registry as registry
from app.benchmark.adapters.base import AdapterContext
from app.benchmark.ontology import metric_specs_for_family
from app.benchmark.schema import BenchmarkSpec, DatasetSpec

_RUNNER_NAME = "finance_l1_french_stats.py"

# Self-contained L1 runner: computes headline statistics (mean, Newey-West
# t-stat, Sharpe, annualized return) for the market factor of a Kenneth
# French monthly factor CSV. Column 0 of the data block is Mkt-RF in all
# standard factor files (FF3, FF5, momentum). No host-package imports:
# the script runs inside any environment with a plain python3.
_RUNNER_BODY = '''#!/usr/bin/env python3
"""Generated L1 finance smoke runner: headline statistics from a Kenneth
French monthly factor CSV (self-contained)."""
import json
import sys
from pathlib import Path


def _mean(xs):
    return sum(xs) / len(xs)


def _nw_t_stat(xs, lags=1):
    n = len(xs)
    if n <= lags + 1:
        return None
    xbar = sum(xs) / n
    d = [v - xbar for v in xs]
    lr = sum(v * v for v in d) / n
    for lag in range(1, lags + 1):
        gl = sum(d[t] * d[t - lag] for t in range(lag, n)) / n
        lr += 2.0 * (1.0 - lag / (lags + 1.0)) * gl
    if lr <= 0:
        return None
    se = (lr / n) ** 0.5
    return xbar / se if se else None


def _sharpe(xs, ppy=12.0):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
    if var <= 0:
        return None
    return (m / (var ** 0.5)) * (ppy ** 0.5)


def _annualized(xs):
    growth = 1.0
    for v in xs:
        growth *= 1.0 + v / 100.0
    years = len(xs) / 12.0
    if growth <= 0 or years <= 0:
        return None
    return (growth ** (1.0 / years) - 1.0) * 100.0


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"metrics": {}, "error": "usage: runner <factors_csv>"}))
        return 1
    rows = []
    for raw in Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in raw.strip().split(",")]
        if not parts or len(parts[0]) != 6 or not parts[0].isdigit():
            if rows:
                break
            continue
        try:
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            if rows:
                break
            continue
        if any(v <= -90.0 for v in vals):
            continue
        if vals:
            rows.append((parts[0], vals))
    if not rows:
        print(json.dumps({"metrics": {}, "error": "no monthly rows parsed"}))
        return 1
    mkt_rf = [vals[0] for _, vals in rows]
    metrics = {
        "n_months": len(mkt_rf),
        "first_month": rows[0][0],
        "last_month": rows[-1][0],
        "mkt_rf_mean_pct_monthly": _mean(mkt_rf),
        "nw_t_stat": _nw_t_stat(mkt_rf, lags=1),
        "sharpe_ratio": _sharpe(mkt_rf),
        "annualized_return": _annualized(mkt_rf),
    }
    print(json.dumps({"metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_FINANCE_SCRIPT_HINTS = (
    "backtest", "sort", "factor", "portfolio", "replicate",
    "replication", "strategy", "signal", "event",
)

_FRENCH_L1_FAMILIES = frozenset({
    "cross_sectional_return_prediction",
    "factor_model_replication",
    "time_series_strategy",
})


class FinanceAdapterBase:
    """Shared proposal logic for the four finance task families."""

    task_family: str = "unknown"

    def propose_benchmarks(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs: list[BenchmarkSpec] = []
        specs.extend(self._propose_french_l1(context))
        specs.extend(self._propose_repo_scripts(context))
        return specs

    def _propose_french_l1(self, context: AdapterContext) -> list[BenchmarkSpec]:
        if self.task_family not in _FRENCH_L1_FAMILIES:
            return []
        csv_path = self._find_french_factors_csv(context)
        if csv_path is None:
            return []
        return [
            BenchmarkSpec(
                id=f"{context.paper_slug}-finance-l1-french-stats",
                task_family=self.task_family,
                level="L1",
                title="French library smoke statistics (generated runner)",
                dataset=DatasetSpec(
                    name="Kenneth French Data Library — factor returns",
                    source="official_repo",
                    size_estimate="a few MB",
                    public=True,
                    notes=[csv_path],
                ),
                command=["python", _RUNNER_NAME, csv_path],
                command_kind="generated_runner",
                expected_metrics=metric_specs_for_family(self.task_family),
                parser={"type": "finance_json"},
                generated_script_name=_RUNNER_NAME,
                generated_script_body=_RUNNER_BODY,
                evidence=[f"French factor CSV available at {csv_path}"],
                feasibility={"runnable": True},
            )
        ]

    def _propose_repo_scripts(self, context: AdapterContext) -> list[BenchmarkSpec]:
        specs: list[BenchmarkSpec] = []
        for script in context.scripts:
            lowered = script.lower()
            if not any(hint in lowered for hint in _FINANCE_SCRIPT_HINTS):
                continue
            if not script.endswith(".py"):
                continue
            if not (context.repo_dir / script).exists():
                continue
            specs.append(
                BenchmarkSpec(
                    id=f"{context.paper_slug}-finance-l2-{Path(script).stem}",
                    task_family=self.task_family,
                    level="L2",
                    title=f"Repository script: {script}",
                    dataset=DatasetSpec(name="repository defaults", source="official_repo"),
                    command=["python", script],
                    command_kind="official_script",
                    expected_metrics=metric_specs_for_family(self.task_family),
                    parser={},
                    evidence=[f"script name matches finance hints: {script}"],
                    feasibility={"runnable": True},
                )
            )
            if len(specs) >= 2:
                break
        return specs

    def _find_french_factors_csv(self, context: AdapterContext) -> str | None:
        root = registry.data_root(
            "french_factors",
            paper_slug=context.paper_slug,
            workspace_dir=str(context.workspace_dir),
        )
        if not root:
            return None
        path = Path(root)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            return None
        preferred = sorted(path.glob("*Research_Data_Factors*.CSV")) + sorted(
            path.glob("*Research_Data_Factors*.csv")
        )
        if preferred:
            return str(preferred[0])
        for pattern in ("*.CSV", "*.csv"):
            matches = sorted(path.glob(pattern))
            if matches:
                return str(matches[0])
        return None


class CrossSectionalReturnPredictionAdapter(FinanceAdapterBase):
    task_family = "cross_sectional_return_prediction"


class EventStudyAdapter(FinanceAdapterBase):
    task_family = "event_study"


class FactorModelReplicationAdapter(FinanceAdapterBase):
    task_family = "factor_model_replication"


class TimeSeriesStrategyAdapter(FinanceAdapterBase):
    task_family = "time_series_strategy"
