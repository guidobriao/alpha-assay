"""Out-of-sample extension for finance replications.

Splits the study timeline into the paper's sample and the period after it
(the Griffin-style "produce the results out of sample"), runs the windowed
statistics runner over both windows, and compares direction and magnitude.

At L1 (open-access factor library) this validates the OOS machinery and the
factor-level direction. A full hypothesis-level OOS test requires the
study's own extracts (L3), which the data access agent flags.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from app.agents.benchmark_reproduction_agent import _is_safe_benchmark_argv
from app.agents.smoke_run_agent import SmokeRunAgent
from app.benchmark.finance_metrics import load_french_monthly
from app.benchmark.parsers import parse_finance_json
from app.benchmark.schema import BenchmarkSpec, DatasetSpec
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState

logger = logging.getLogger(__name__)

WINDOW_RUNNER_NAME = "finance_oos_window.py"

_RUN_TIMEOUT_SECONDS = 120
_MIN_OOS_MONTHS = 12


def _next_month(yyyymm: int) -> int:
    y, m = divmod(yyyymm, 100)
    m += 1
    if m > 12:
        y, m = y + 1, 1
    return y * 100 + m


def _yyyym(value: Any) -> int | None:
    s = str(value or "").strip().replace("-", "")
    return int(s) if len(s) == 6 and s.isdigit() else None


WINDOW_RUNNER_BODY = '''#!/usr/bin/env python3
"""Generated finance runner: windowed monthly statistics from a Kenneth
French library CSV (self-contained; supports date and column windows)."""
import argparse
import json
import sys
from pathlib import Path


def _parse_monthly(path):
    rows = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
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
        rows.append((int(parts[0]), vals))
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


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
    if not xs:
        return None
    growth = 1.0
    for v in xs:
        growth *= 1.0 + v / 100.0
    years = len(xs) / 12.0
    if growth <= 0 or years <= 0:
        return None
    return (growth ** (1.0 / years) - 1.0) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999999)
    ap.add_argument("--col", type=int, default=0)
    args = ap.parse_args()
    rows = _parse_monthly(args.csv)
    window = [(d, vals) for d, vals in rows if args.start <= d <= args.end]
    series = [vals[args.col] for _, vals in window if len(vals) > args.col]
    if not series:
        print(json.dumps({"metrics": {}, "error": "empty window"}))
        return 1
    metrics = {
        "n_months": len(series),
        "first_month": window[0][0],
        "last_month": window[-1][0],
        "column": args.col,
        "mean_pct_monthly": _mean(series),
        "nw_t_stat": _nw_t_stat(series, lags=1),
        "sharpe_ratio": _sharpe(series),
        "annualized_return": _annualized(series),
    }
    print(json.dumps({"metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class _WindowRunner:
    """Backend-aware, safety-gated execution of the windowed finance runner."""

    def __init__(self, state: TaskState, repo_dir: Path, run_dir: Path):
        self.state = state
        self.repo_dir = repo_dir
        self.run_dir = run_dir
        self._smoke = SmokeRunAgent(timeout_minutes=5)

    def write_runner(self) -> None:
        (self.repo_dir / WINDOW_RUNNER_NAME).write_text(
            WINDOW_RUNNER_BODY, encoding="utf-8"
        )

    def _argv_for_backend(self, argv: list[str]) -> list[str]:
        state = self.state
        if state.backend in {"venv", "conda"}:
            return self._smoke._venv_argv(argv, state, Path(state.task_dir))
        return argv

    def run_window(
        self,
        spec: BenchmarkSpec,
        csv_path: str,
        start: int,
        end: int,
        col: int,
        log_stem: str,
    ) -> dict[str, Any]:
        argv = [
            "python", WINDOW_RUNNER_NAME, csv_path,
            "--start", str(start), "--end", str(end), "--col", str(col),
        ]
        candidate_scripts = (
            self.state.repo_evaluation.candidate_scripts
            if self.state.repo_evaluation else []
        )
        ok, reason = _is_safe_benchmark_argv(argv, candidate_scripts, spec)
        if not ok:
            return {"error": f"blocked: {reason}"}
        try:
            proc = subprocess.run(
                self._argv_for_backend(argv),
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_SECONDS,
                env=(
                    self._smoke._venv_env(self.state)
                    if self.state.backend in {"venv", "conda"}
                    else __import__("os").environ.copy()
                ),
            )
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"execution_error: {exc}"}
        (self.run_dir / f"{log_stem}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (self.run_dir / f"{log_stem}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        metrics = parse_finance_json(proc.stdout)
        if not metrics:
            return {
                "error": f"no metrics parsed (exit={proc.returncode})",
                "stderr_tail": proc.stderr[-500:],
            }
        return metrics


def find_french_factors_csv(workspace_dir: str, paper_slug: str) -> str | None:
    """Locate a Kenneth French factor CSV provisioned by the data access agent."""
    from app.benchmark import dataset_registry as registry

    root = registry.data_root(
        "french_factors", paper_slug=paper_slug, workspace_dir=workspace_dir
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


def _sign(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return (a > 0 and b > 0) or (a < 0 and b < 0) or (a == 0 and b == 0)


class OOSAgent:
    """Split at the paper's sample end and re-run the statistics OOS."""

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        out_dir = task_dir / "oos"
        out_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"ran": False}

        brief_path = task_dir / "paper" / "finance_brief.json"
        if not brief_path.exists():
            report["skip_reason"] = "not a finance study"
            state.oos = report
            save_json(out_dir / "oos_report.json", report)
            return state
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            report["skip_reason"] = "repository not evaluated"
            state.oos = report
            save_json(out_dir / "oos_report.json", report)
            return state

        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        period = brief.get("sample_period") or {}
        csv_path = find_french_factors_csv(
            state.workspace_dir, task_dir.name
        )
        if not csv_path:
            report["skip_reason"] = (
                "no open-access factor data provisioned; OOS extension needs "
                "the French library or the study's own extracts"
            )
            state.oos = report
            save_json(out_dir / "oos_report.json", report)
            emit_progress("OOS extension", "skipped", level="warning", detail=report["skip_reason"])
            return state

        rows = load_french_monthly(csv_path)
        if len(rows) < 24:
            report["skip_reason"] = f"insufficient monthly rows ({len(rows)})"
            state.oos = report
            save_json(out_dir / "oos_report.json", report)
            return state

        paper_end = _yyyym(period.get("end"))
        ins_start = _yyyym(period.get("start")) or rows[0][0]
        split_note = None
        if paper_end is not None and paper_end < rows[-1][0]:
            oos_start = _next_month(paper_end)
        else:
            # No exploitable paper end (or data does not extend past it):
            # temporal 70/30 holdout on the available rows.
            k = int(len(rows) * 0.7)
            oos_start = rows[k][0]
            paper_end = rows[k - 1][0]
            split_note = (
                "paper sample end unavailable or beyond data coverage; "
                "approximate 70/30 temporal split applied"
            )

        run_dir = out_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = _WindowRunner(state, Path(state.repo_evaluation.repo_dir), run_dir)
        runner.write_runner()

        def _spec(label: str) -> BenchmarkSpec:
            return BenchmarkSpec(
                id=f"{task_dir.name}-oos-{label}",
                task_family="cross_sectional_return_prediction",
                level="L1",
                title=f"OOS window run: {label}",
                dataset=DatasetSpec(name="Kenneth French factor CSV", source="official_repo"),
                command=["python", WINDOW_RUNNER_NAME],
                command_kind="generated_runner",
                generated_script_name=WINDOW_RUNNER_NAME,
                feasibility={"runnable": True},
            )

        emit_progress(
            "OOS extension",
            "running in-sample and out-of-sample windows",
            detail=f"ins<= {paper_end}, oos>= {oos_start}",
        )
        ins_metrics = runner.run_window(
            _spec("ins"), csv_path, ins_start, paper_end, 0, "oos_ins"
        )
        oos_metrics = runner.run_window(
            _spec("oos"), csv_path, oos_start, 999999, 0, "oos_out"
        )

        comparisons: list[dict[str, Any]] = []
        for metric in ("mean_pct_monthly", "sharpe_ratio", "annualized_return"):
            ins_v = ins_metrics.get(metric) if isinstance(ins_metrics, dict) else None
            oos_v = oos_metrics.get(metric) if isinstance(oos_metrics, dict) else None
            if not isinstance(ins_v, (int, float)) or not isinstance(oos_v, (int, float)):
                comparisons.append({"metric": metric, "status": "missing"})
                continue
            sign_ok = _sign(float(ins_v), float(oos_v))
            delta = float(oos_v) - float(ins_v)
            comparisons.append(
                {
                    "metric": metric,
                    "in_sample": ins_v,
                    "out_of_sample": oos_v,
                    "delta": delta,
                    "sign_consistent": sign_ok,
                    "status": "sign_matched" if sign_ok else "sign_flipped",
                }
            )

        oos_months = oos_metrics.get("n_months") if isinstance(oos_metrics, dict) else None
        ran_ok = not ins_metrics.get("error") and not oos_metrics.get("error")
        if ran_ok and isinstance(oos_months, int) and oos_months >= _MIN_OOS_MONTHS:
            verdict = "oos_machinery_ok"
        elif ran_ok:
            verdict = "oos_window_too_short"
        else:
            verdict = "oos_runs_failed"

        report = {
            "ran": True,
            "verdict": verdict,
            "split": {
                "in_sample": {"start": ins_start, "end": paper_end},
                "out_of_sample": {"start": oos_start, "end": rows[-1][0]},
                "note": split_note,
                "in_sample_end_from_paper": _yyyym(period.get("in_sample_end")),
            },
            "data": {"csv": csv_path, "level": "L1 (factor-library smoke)"},
            "in_sample_metrics": ins_metrics,
            "out_of_sample_metrics": oos_metrics,
            "comparisons": comparisons,
            "caveat": (
                "L1 OOS validates the temporal-split machinery and factor-level "
                "direction; a hypothesis-level OOS test of the paper's signal "
                "requires the study's own data extracts (L3)."
            ),
        }
        state.oos = report
        save_json(out_dir / "oos_report.json", report)
        emit_progress(
            "OOS extension",
            "report written",
            detail=f"verdict={verdict}",
            verdict=verdict,
        )
        return state
