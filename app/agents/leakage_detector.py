"""Static leakage audit for replicated finance studies.

Scans the cloned repository source for look-ahead-bias patterns and audits
the extracted finance brief for methodological red flags. Produces
leakage/leakage_report.json — the input the OOS and report stages use to
qualify replication findings.

This is a first-pass audit, not a formal verifier: every finding carries
the matched line as evidence so a human (or downstream LLM stage) can
confirm or dismiss it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import TaskState

logger = logging.getLogger(__name__)

_SCAN_SUFFIXES = {".py", ".R", ".r"}
_MAX_FILE_BYTES = 200 * 1024
_MAX_FILES = 500
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea"}

_SEVERITY_WEIGHT = {"critical": 3, "warning": 1, "info": 0}

# (rule_id, severity, compiled regex, message, remediation)
_CODE_RULES: list[tuple[str, str, re.Pattern[str], str, str]] = [
    (
        "negative_shift",
        "critical",
        re.compile(r"shift\s*\(\s*-\s*\d"),
        "Negative shift moves future observations into the present row.",
        "Use positive shift (e.g. shift(1)) so each row only sees past data.",
    ),
    (
        "r_lead",
        "critical",
        re.compile(r"\blead\s*\("),
        "dplyr::lead() pulls future observations into the current row.",
        "Replace lead() with lag() and re-align the horizon explicitly.",
    ),
    (
        "backward_fill",
        "warning",
        re.compile(r"\.(?:bfill|backfill)\s*\(|fillna\s*\(\s*(?:method\s*=\s*)?[\"'](?:bfill|backfill)[\"']"),
        "Backward fill propagates future values backward in time.",
        "Use ffill/forward-fill, or fill from lagged data only.",
    ),
    (
        "merge_asof_nearest",
        "critical",
        re.compile(r"merge_asof\s*\([^)]*direction\s*=\s*[\"']nearest[\"']", re.DOTALL),
        "merge_asof with direction='nearest' can match a future observation.",
        "Use direction='backward' and add the documented formation lag.",
    ),
    (
        "merge_asof_no_direction",
        "info",
        re.compile(r"merge_asof\s*\((?![^)]*direction\s*=)", re.DOTALL),
        "merge_asof without explicit direction relies on the pandas default "
        "(backward, safe) — make it explicit to survive refactors.",
        "Add direction='backward' explicitly.",
    ),
    (
        "full_sample_scaler",
        "warning",
        re.compile(r"(?:StandardScaler|MinMaxScaler|RobustScaler)\s*\(\s*\)[^\n]*\n[^\n]*fit(?:_transform)?\s*\(", re.IGNORECASE),
        "Scaler fitted on the full sample before any time-based split "
        "leaks distributional information from the evaluation period.",
        "Fit scalers on the training window only, then transform later periods.",
    ),
    (
        "expanding_over_row",
        "info",
        re.compile(r"expanding\s*\(\s*\)\s*\.\s*(?:mean|std|sum)\s*\(\s*\)"),
        "Expanding-window statistics include the current observation unless "
        "shifted — verify alignment with the signal formation date.",
        "Apply expanding stats to shift(1) data when used as a signal input.",
    ),
]

# Audit rules on the finance brief (not code).
_BRIEF_RULES_SURVIVORSHIP = ("current_constituents",)


class LeakageDetectorAgent:
    """Run the static leakage audit and persist leakage_report.json."""

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        out_dir = task_dir / "leakage"
        out_dir.mkdir(parents=True, exist_ok=True)

        findings: list[dict[str, Any]] = []

        emit_progress("Leakage audit", "scanning repository source")
        repo_dir = state.repo_evaluation.repo_dir if state.repo_evaluation else None
        code_findings = self._scan_repository(Path(repo_dir)) if repo_dir else []
        findings.extend(code_findings)
        emit_progress(
            "Leakage audit",
            "code scan complete",
            detail=f"{len(code_findings)} finding(s) from source scan",
        )

        emit_progress("Leakage audit", "auditing finance brief")
        brief_findings = self._audit_finance_brief(task_dir)
        findings.extend(brief_findings)

        critical = sum(1 for f in findings if f["severity"] == "critical")
        warnings = sum(1 for f in findings if f["severity"] == "warning")
        risk_score = sum(_SEVERITY_WEIGHT[f["severity"]] for f in findings)
        if critical:
            verdict = "high_risk"
        elif warnings:
            verdict = "minor_issues"
        else:
            verdict = "clean"

        report = {
            "verdict": verdict,
            "risk_score": risk_score,
            "counts": {
                "critical": critical,
                "warning": warnings,
                "info": sum(1 for f in findings if f["severity"] == "info"),
                "total": len(findings),
            },
            "code_scan_root": repo_dir,
            "findings": findings,
        }
        save_json(out_dir / "leakage_report.json", report)
        emit_progress(
            "Leakage audit",
            "report written",
            detail=f"verdict={verdict}, critical={critical}, warnings={warnings}",
            verdict=verdict,
        )
        return state

    # -- code scan --------------------------------------------------------

    def _scan_repository(self, repo_dir: Path) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        if not repo_dir.exists():
            return findings
        scanned = 0
        for path in sorted(repo_dir.rglob("*")):
            if scanned >= _MAX_FILES:
                logger.info("Leakage scan file limit reached (%d)", _MAX_FILES)
                break
            if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
                continue
            if any(part in _SKIP_DIRS for part in path.relative_to(repo_dir).parts):
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            lines = text.splitlines()
            for rule_id, severity, pattern, message, remediation in _CODE_RULES:
                for lineno, line in enumerate(lines, start=1):
                    if pattern.search(line):
                        findings.append(
                            {
                                "rule_id": rule_id,
                                "severity": severity,
                                "file": path.relative_to(repo_dir).as_posix(),
                                "line": lineno,
                                "snippet": line.strip()[:200],
                                "message": message,
                                "remediation": remediation,
                            }
                        )
        return findings

    # -- brief audit ------------------------------------------------------

    def _audit_finance_brief(self, task_dir: Path) -> list[dict[str, Any]]:
        brief_path = task_dir / "paper" / "finance_brief.json"
        if not brief_path.exists():
            return []
        try:
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
        except Exception:
            return [
                {
                    "rule_id": "brief_unreadable",
                    "severity": "warning",
                    "file": "paper/finance_brief.json",
                    "line": None,
                    "snippet": None,
                    "message": "Finance brief exists but cannot be parsed; "
                    "methodological audit skipped.",
                    "remediation": "Re-run the paper understanding stage.",
                }
            ]

        findings: list[dict[str, Any]] = []
        universe = brief.get("universe") or {}
        portfolio = brief.get("portfolio") or {}
        period = brief.get("sample_period") or {}
        costs = brief.get("transaction_costs") or {}

        if universe.get("survivorship") in _BRIEF_RULES_SURVIVORSHIP:
            findings.append(
                {
                    "rule_id": "survivorship_bias",
                    "severity": "critical",
                    "file": "paper/finance_brief.json",
                    "line": None,
                    "snippet": f"survivorship={universe.get('survivorship')}",
                    "message": "Universe built from current constituents "
                    "overstates historical performance (survivorship bias).",
                    "remediation": "Rebuild the universe point-in-time and include delisting returns.",
                }
            )
        if not portfolio.get("formation_lag"):
            findings.append(
                {
                    "rule_id": "formation_lag_undocumented",
                    "severity": "warning",
                    "file": "paper/finance_brief.json",
                    "line": None,
                    "snippet": "portfolio.formation_lag is missing",
                    "message": "No formation lag documented; the timing between "
                    "signal measurement and portfolio returns cannot be verified.",
                    "remediation": "Extract the lag from the paper (typically 1 month for monthly sorts).",
                }
            )
        task_family_hint = " ".join(
            str(brief.get(key) or "") for key in ("hypothesis",)
        ).lower()
        portfolio_text = " ".join(
            str(portfolio.get(key) or "") for key in ("sort_method", "signal")
        ).lower()
        looks_like_backtest = any(
            term in task_family_hint + portfolio_text
            for term in ("backtest", "strategy", "trend", "momentum")
        )
        if looks_like_backtest and not costs.get("included"):
            findings.append(
                {
                    "rule_id": "no_transaction_costs",
                    "severity": "warning",
                    "file": "paper/finance_brief.json",
                    "line": None,
                    "snippet": "transaction_costs.included=False",
                    "message": "A strategy-style backtest without transaction "
                    "costs typically overstates net performance.",
                    "remediation": "Add a cost model (e.g. bps per side) or qualify the result as gross-of-costs.",
                }
            )
        if period.get("start") and period.get("start") == period.get("end"):
            findings.append(
                {
                    "rule_id": "degenerate_period",
                    "severity": "info",
                    "file": "paper/finance_brief.json",
                    "line": None,
                    "snippet": f"start==end={period.get('start')}",
                    "message": "Sample period start equals end; the extraction "
                    "likely missed the period.",
                    "remediation": "Re-check the paper's sample description.",
                }
            )
        return findings
