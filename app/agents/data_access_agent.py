"""Provision the finance datasets required by a replication study.

Reads paper/finance_brief.json (written by PaperUnderstandingAgent), matches
the required data sources and factor controls against the dataset registry,
checks local availability, downloads open-access library files (Kenneth
French), and flags subscription-licensed sources (CRSP, Compustat, IBES,
WRDS) as requiring manual provisioning. Subscription data is never fetched.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx

from app.benchmark import dataset_registry as registry
from app.benchmark.dataset_registry import BenchmarkDatasetEntry
from app.core.file_utils import save_json
from app.core.progress import emit_progress
from app.core.state import DataAccessResult, DatasetProvisioning, TaskState

logger = logging.getLogger(__name__)

_FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# Kenneth French library archives relevant to standard factor controls and
# portfolio sorts, keyed by short name.
_FRENCH_LIBRARY_FILES: dict[str, str] = {
    "ff3_factors": "F-F_Research_Data_Factors_CSV.zip",
    "ff5_factors": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum_factor": "F-F_Momentum_Factor_CSV.zip",
    "portfolios_me_beme": "Portfolios_Formed_on_ME_BEME_CSV.zip",
    "portfolios_me_prior_2_12": "Portfolios_Formed_on_ME_PRIOR_2_12_CSV.zip",
    "industry_10": "10_Industry_Portfolios_CSV.zip",
    "industry_48": "48_Industry_Portfolios_CSV.zip",
    "industry_49": "49_Industry_Portfolios_CSV.zip",
}

_OK_STATUSES = {"available", "downloaded", "requires_subscription", "not_matched"}


class DataAccessAgent:
    """Resolve, check, and (where public) provision finance datasets."""

    def run(self, state: TaskState) -> TaskState:
        task_dir = Path(state.task_dir)
        data_dir = task_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        brief_path = task_dir / "paper" / "finance_brief.json"

        if not brief_path.exists():
            emit_progress(
                "Data access",
                "no finance brief — not a finance study, skipping",
                level="info",
            )
            result = DataAccessResult(
                is_finance_study=False,
                ready=False,
                summary="no finance_brief.json found; data access skipped",
            )
            state.data_access = result
            save_json(data_dir / "data_access.json", result)
            return state

        emit_progress("Data access", "loading finance brief", detail=str(brief_path))
        try:
            payload = json.loads(brief_path.read_text(encoding="utf-8"))
        except Exception as exc:
            state.errors.append(
                {"agent": "DataAccessAgent", "error": f"unreadable finance brief: {exc}"}
            )
            result = DataAccessResult(
                is_finance_study=True,
                ready=False,
                summary=f"finance brief unreadable: {exc}",
                brief_path=str(brief_path),
            )
            state.data_access = result
            save_json(data_dir / "data_access.json", result)
            return state

        data_sources = [str(s) for s in payload.get("data_sources", []) if s]
        factor_controls = [str(f) for f in payload.get("factor_controls", []) if f]
        portfolio = payload.get("portfolio") or {}
        portfolio_text = " ".join(
            str(portfolio.get(key) or "")
            for key in ("sort_method", "signal", "long_side", "short_side")
        ).lower()

        emit_progress(
            "Data access",
            "matching required sources against registry",
            detail=f"sources={data_sources}, factors={factor_controls}",
        )
        matched, unmatched_terms = self._match_sources(
            data_sources, factor_controls, portfolio_text
        )

        records: list[DatasetProvisioning] = []
        for term in unmatched_terms:
            records.append(
                DatasetProvisioning(
                    dataset_id=term[:80],
                    name=term[:80],
                    status="not_matched",
                    detail="no registry entry matched this source",
                )
            )

        factor_keys = self._factor_file_keys(factor_controls)
        portfolio_keys = self._portfolio_file_keys(portfolio_text)

        for entry in matched.values():
            record = self._provision_entry(task_dir, state.workspace_dir, entry, factor_keys, portfolio_keys)
            records.append(record)

        failed = [r for r in records if r.status not in _OK_STATUSES]
        result = DataAccessResult(
            is_finance_study=True,
            ready=not failed,
            records=records,
            summary=(
                f"{len(matched)} dataset(s) matched, "
                f"{sum(1 for r in records if r.status == 'available')} available, "
                f"{sum(1 for r in records if r.status == 'downloaded')} downloaded, "
                f"{sum(1 for r in records if r.status == 'requires_subscription')} need subscription extracts, "
                f"{len(failed)} failed/not resolvable"
            ),
            brief_path=str(brief_path),
        )
        state.data_access = result
        save_json(data_dir / "data_access.json", result)
        emit_progress(
            "Data access",
            "data access resolved",
            detail=result.summary,
            ready=result.ready,
        )
        return state

    # -- matching ---------------------------------------------------------

    def _match_sources(
        self,
        data_sources: list[str],
        factor_controls: list[str],
        portfolio_text: str,
    ) -> tuple[dict[str, BenchmarkDatasetEntry], list[str]]:
        matched: dict[str, BenchmarkDatasetEntry] = {}
        unmatched: list[str] = []
        terms = [*data_sources, *factor_controls]
        for term in terms:
            entry = self._match_entry(term)
            if entry is not None:
                matched.setdefault(entry.dataset_id, entry)
            elif term.strip():
                unmatched.append(term.strip())
        # Portfolio-sort text can imply the French portfolio library even
        # when the source list only mentions the factor files.
        if "french_portfolios" not in matched:
            for alias in registry.DATASETS["french_portfolios"].aliases:
                if alias in portfolio_text:
                    matched["french_portfolios"] = registry.DATASETS["french_portfolios"]
                    break
        return matched, unmatched

    def _match_entry(self, term: str) -> BenchmarkDatasetEntry | None:
        t = term.strip().lower()
        if not t:
            return None
        for entry in registry.DATASETS.values():
            for alias in (entry.name, entry.dataset_id, *entry.aliases):
                a = alias.lower()
                if a and (a in t or t in a):
                    return entry
        return None

    # -- file selection ---------------------------------------------------

    def _factor_file_keys(self, factor_controls: list[str]) -> list[str]:
        joined = " ".join(factor_controls).lower()
        keys: list[str] = []
        if any(k in joined for k in ("ff3", "three factor", "3-factor", "3 factor")):
            keys.append("ff3_factors")
        if any(k in joined for k in ("ff5", "five factor", "5-factor", "5 factor")):
            keys.append("ff5_factors")
        if "mom" in joined or "momentum" in joined:
            keys.append("momentum_factor")
        return keys

    def _portfolio_file_keys(self, portfolio_text: str) -> list[str]:
        keys: list[str] = []
        if "25 portfolios" in portfolio_text or "size and book" in portfolio_text or "book to market" in portfolio_text or "book-to-market" in portfolio_text:
            keys.append("portfolios_me_beme")
        if "momentum" in portfolio_text and "portfolio" in portfolio_text:
            keys.append("portfolios_me_prior_2_12")
        for n in ("10", "48", "49"):
            if f"{n} industry" in portfolio_text:
                keys.append(f"industry_{n}")
        return keys

    # -- provisioning -----------------------------------------------------

    def _provision_entry(
        self,
        task_dir: Path,
        workspace_dir: str,
        entry: BenchmarkDatasetEntry,
        factor_keys: list[str],
        portfolio_keys: list[str],
    ) -> DatasetProvisioning:
        local = registry.data_root(entry.dataset_id, workspace_dir=workspace_dir)
        if local and Path(local).exists():
            os.environ.setdefault(entry.env_var, str(local))
            return DatasetProvisioning(
                dataset_id=entry.dataset_id,
                name=entry.name,
                status="available",
                local_path=str(local),
                license=entry.license,
            )

        if entry.access == "open_download" and entry.dataset_id.startswith("french"):
            if entry.dataset_id == "french_factors":
                keys = factor_keys or ["ff3_factors", "ff5_factors"]
            else:
                keys = portfolio_keys or ["portfolios_me_beme"]
            target = self._download_french_files(task_dir, entry, keys)
            if target is not None:
                os.environ.setdefault(entry.env_var, str(target))
                return DatasetProvisioning(
                    dataset_id=entry.dataset_id,
                    name=entry.name,
                    status="downloaded",
                    local_path=str(target),
                    detail=f"downloaded {len(keys)} archive(s) from the French data library",
                    license=entry.license,
                )
            return DatasetProvisioning(
                dataset_id=entry.dataset_id,
                name=entry.name,
                status="failed",
                detail="open-download failed (network error); set "
                f"{entry.env_var} to a local extract to retry",
                license=entry.license,
            )

        if entry.access == "wrds_subscription":
            return DatasetProvisioning(
                dataset_id=entry.dataset_id,
                name=entry.name,
                status="requires_subscription",
                detail=(
                    f"{entry.name} is subscription-licensed ({entry.license}). "
                    f"Provision a local extract and set {entry.env_var}. "
                    f"Coverage: {entry.coverage}."
                ),
                license=entry.license,
            )

        return DatasetProvisioning(
            dataset_id=entry.dataset_id,
            name=entry.name,
            status="needs_download",
            detail=f"set {entry.env_var} or place the dataset under ALPHA_ASSAY_DATA_ROOT/{entry.dataset_id}",
            license=entry.license,
        )

    def _download_french_files(
        self, task_dir: Path, entry: BenchmarkDatasetEntry, keys: list[str]
    ) -> Path | None:
        target_dir = task_dir / "data" / entry.dataset_id
        target_dir.mkdir(parents=True, exist_ok=True)
        successes = 0
        for key in keys:
            filename = _FRENCH_LIBRARY_FILES.get(key)
            if not filename:
                continue
            url = _FRENCH_BASE_URL + filename
            emit_progress("Data access", "downloading", detail=filename)
            try:
                response = httpx.get(url, timeout=120.0, follow_redirects=True)
                response.raise_for_status()
                zip_path = target_dir / filename
                zip_path.write_bytes(response.content)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target_dir)
                successes += 1
            except Exception as exc:
                logger.warning("French library download failed for %s: %s", filename, exc)
                emit_progress(
                    "Data access",
                    f"download failed: {filename}",
                    level="warning",
                    detail=str(exc),
                )
        return target_dir if successes else None
