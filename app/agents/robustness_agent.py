"""Robustness matrix for finance replications.

Systematically re-runs the windowed statistics runner across variations —
sub-periods, factor columns, and factor-file definitions — and records the
consistency of direction against the baseline. At L1 this exercises the
robustness machinery over open-access data; specification-level robustness
(universe, weighting, costs) requires the study's own extracts (L3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.agents.oos_agent import (
    WINDOW_RUNNER_NAME,
    OOSAgent,
    _WindowRunner,
    _yyyym,
    find_french_factors_csv,
)
from app.benchmark.schema import BenchmarkSpec, DatasetSpec
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState

logger = logging.getLogger(__name__)


class RobustnessAgent:
    """Build and execute the L1 robustness matrix."""

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        out_dir = task_dir / "robustness"
        out_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"ran": False}

        brief_path = task_dir / "paper" / "finance_brief.json"
        if not brief_path.exists():
            report["skip_reason"] = "not a finance study"
            state.robustness = report
            save_json(out_dir / "robustness_matrix.json", report)
            return state
        if not state.repo_evaluation or not state.repo_evaluation.repo_dir:
            report["skip_reason"] = "repository not evaluated"
            state.robustness = report
            save_json(out_dir / "robustness_matrix.json", report)
            return state

        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        period = brief.get("sample_period") or {}
        factors_root = Path(
            find_french_factors_csv(state.workspace_dir, task_dir.name) or ""
        )
        csv_files = (
            [factors_root] if factors_root.is_file() else self._factor_csvs(factors_root)
        ) if str(factors_root) else []
        if not csv_files:
            report["skip_reason"] = "no open-access factor data provisioned"
            state.robustness = report
            save_json(out_dir / "robustness_matrix.json", report)
            emit_progress("Robustness matrix", "skipped", level="warning", detail=report["skip_reason"])
            return state

        baseline_csv = csv_files[0]
        rows = __import__("app.benchmark.finance_metrics", fromlist=["load_french_monthly"]).load_french_monthly(baseline_csv)
        if len(rows) < 24:
            report["skip_reason"] = f"insufficient monthly rows ({len(rows)})"
            state.robustness = report
            save_json(out_dir / "robustness_matrix.json", report)
            return state

        start = _yyyym(period.get("start")) or rows[0][0]
        end = _yyyym(period.get("end")) or rows[-1][0]
        end = min(end, rows[-1][0])
        k = (start + end) // 2  # rough mid-point for sub-period split

        run_dir = out_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = _WindowRunner(state, Path(state.repo_evaluation.repo_dir), run_dir)
        runner.write_runner()

        def _spec(label: str) -> BenchmarkSpec:
            return BenchmarkSpec(
                id=f"{task_dir.name}-robust-{label}",
                task_family="cross_sectional_return_prediction",
                level="L1",
                title=f"Robustness run: {label}",
                dataset=DatasetSpec(name="Kenneth French factor CSV", source="official_repo"),
                command=["python", WINDOW_RUNNER_NAME],
                command_kind="generated_runner",
                generated_script_name=WINDOW_RUNNER_NAME,
                feasibility={"runnable": True},
            )

        variations: list[dict[str, Any]] = [
            {
                "id": "baseline",
                "label": "Baseline (paper window, market factor)",
                "csv": str(baseline_csv), "start": start, "end": end, "col": 0,
            },
            {
                "id": "subperiod_first_half",
                "label": "First half of paper window",
                "csv": str(baseline_csv), "start": start, "end": k, "col": 0,
            },
            {
                "id": "subperiod_second_half",
                "label": "Second half of paper window",
                "csv": str(baseline_csv), "start": k + 1, "end": end, "col": 0,
            },
            {
                "id": "factor_smb",
                "label": "SMB premium series (col 1) — informational",
                "csv": str(baseline_csv), "start": start, "end": end, "col": 1,
            },
            {
                "id": "factor_hml",
                "label": "HML premium series (col 2) — informational",
                "csv": str(baseline_csv), "start": start, "end": end, "col": 2,
            },
        ]
        # Factor-file definition variants: additional *Research_Data_Factors* files
        for extra in csv_files[1:]:
            variations.append(
                {
                    "id": f"file_{extra.stem[:40]}",
                    "label": f"Definition variant: {extra.name}",
                    "csv": str(extra), "start": start, "end": end, "col": 0,
                }
            )

        emit_progress(
            "Robustness matrix",
            "executing variations",
            detail=f"{len(variations)} run(s) planned",
        )
        results: list[dict[str, Any]] = []
        for variation in variations:
            metrics = runner.run_window(
                _spec(variation["id"]),
                variation["csv"],
                variation["start"],
                variation["end"],
                variation["col"],
                f"robust_{variation['id']}",
            )
            results.append({**variation, "metrics": metrics})

        baseline_metrics = next(
            (r["metrics"] for r in results if r["id"] == "baseline"), {}
        )
        baseline_mean = (
            baseline_metrics.get("mean_pct_monthly")
            if isinstance(baseline_metrics, dict) else None
        )
        for row in results:
            m = row["metrics"]
            if isinstance(m, dict) and not m.get("error") and isinstance(baseline_mean, (int, float)):
                mean_v = m.get("mean_pct_monthly")
                row["baseline_delta_mean"] = (
                    float(mean_v) - float(baseline_mean)
                    if isinstance(mean_v, (int, float)) else None
                )
                from app.agents.oos_agent import _sign
                row["sign_match_vs_baseline"] = _sign(
                    float(mean_v) if isinstance(mean_v, (int, float)) else None,
                    float(baseline_mean),
                )
            else:
                row["baseline_delta_mean"] = None
                row["sign_match_vs_baseline"] = None

        evaluated = [
            r for r in results
            if r["id"] != "baseline" and r["sign_match_vs_baseline"] is not None
        ]
        matches = sum(1 for r in evaluated if r["sign_match_vs_baseline"])
        if len(evaluated) >= 2:
            verdict = "sign_consistent" if matches == len(evaluated) else "mixed"
        else:
            verdict = "insufficient_variations"

        report = {
            "ran": True,
            "verdict": verdict,
            "sign_consistency": f"{matches}/{len(evaluated)}",
            "window": {"start": start, "end": end},
            "variations": results,
            "caveat": (
                "L1 robustness covers sub-periods, factor columns and factor-file "
                "definitions over open-access data. Specification-level robustness "
                "(universe, weighting, transaction costs, winsorization) requires "
                "the study's own extracts (L3)."
            ),
        }
        state.robustness = report
        save_json(out_dir / "robustness_matrix.json", report)
        emit_progress(
            "Robustness matrix",
            "report written",
            detail=f"verdict={verdict} ({report['sign_consistency']})",
            verdict=verdict,
        )
        return state

    def _factor_csvs(self, root: Path) -> list[Path]:
        """All *Research_Data_Factors* CSVs (FF3, FF5); momentum/portfolio files
        have different column layouts and are excluded from file variants."""
        if not root.is_dir():
            return []
        files = sorted(root.glob("*Research_Data_Factors*.CSV")) + sorted(
            root.glob("*Research_Data_Factors*.csv")
        )
        unique: list[Path] = []
        seen: set[str] = set()
        for f in files:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique
