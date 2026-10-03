"""Finance implementation-contract builder for the contract-guided pipeline.

Converts the finance brief (extracted from the paper by the understanding
stage) into a persistent implementation contract: a list of concrete,
verifiable obligations that generated replication code must satisfy. The
contract follows the same philosophy as the requirement/evidence channels:
obligations stay attached to the artifacts, are validated for coverage, and
can be rendered as a prompt-ready text block for the plan/generate stages.

Standalone usage:
    python -m contractgen.pipeline.finance_contract finance_brief.json [-o finance_contract.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = "finance_contract.v1"


def _opt(brief: dict, *keys: str) -> Any:
    current: Any = brief
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if current not in ("", None, [], {}) else None


def _obligation(
    obligation_id: str,
    category: str,
    requirement: str,
    evidence: Any = None,
) -> dict[str, Any]:
    return {
        "id": obligation_id,
        "category": category,
        "requirement": requirement,
        "evidence": evidence,
        "status": "open",
    }


def build_finance_contract(finance_brief: dict) -> dict[str, Any]:
    """Build the finance implementation contract from a finance brief."""
    obligations: list[dict[str, Any]] = []

    period_start = _opt(finance_brief, "sample_period", "start")
    period_end = _opt(finance_brief, "sample_period", "end")
    frequency = _opt(finance_brief, "sample_period", "frequency") or "unknown"
    obligations.append(
        _obligation(
            "sample-period",
            "data",
            f"Restrict every computation to the sample period "
            f"{period_start or '?'} through {period_end or '?'} "
            f"({frequency} frequency); assert the panel bounds explicitly.",
            evidence={"start": period_start, "end": period_end, "frequency": frequency},
        )
    )

    universe_desc = _opt(finance_brief, "universe", "description")
    exchanges = _opt(finance_brief, "universe", "exchanges") or []
    exclusions = _opt(finance_brief, "universe", "exclusions") or []
    min_price = _opt(finance_brief, "universe", "min_price")
    min_cap = _opt(finance_brief, "universe", "min_market_cap_musd")
    obligations.append(
        _obligation(
            "universe-construction",
            "data",
            "Reconstruct the investment universe point-in-time from the "
            "described filters: "
            f"universe={universe_desc or 'as described in the paper'}; "
            f"exchanges={exchanges or 'all'}; exclusions={exclusions or 'none'}; "
            f"min_price={min_price}; min_market_cap_musd={min_cap}. "
            "Never use current index constituents for historical windows.",
            evidence={
                "description": universe_desc,
                "exchanges": exchanges,
                "exclusions": exclusions,
                "min_price": min_price,
                "min_market_cap_musd": min_cap,
            },
        )
    )

    survivorship = _opt(finance_brief, "universe", "survivorship") or "unknown"
    delisting = _opt(finance_brief, "universe", "delisting_treatment")
    obligations.append(
        _obligation(
            "survivorship-delisting",
            "data",
            f"Handle survivorship ({survivorship}) and delisting returns "
            f"({delisting or 'as specified in the paper'}): include delisted "
            "securities with their delisting returns up to the deletion date.",
            evidence={"survivorship": survivorship, "delisting_treatment": delisting},
        )
    )

    portfolio = finance_brief.get("portfolio") or {}
    if portfolio:
        formation_lag = _opt(portfolio, "formation_lag")
        lag_text = formation_lag or "extracted from the paper; typically one month for monthly sorts"
        obligations.append(
            _obligation(
                "formation-lag",
                "timing",
                f"Apply the documented formation lag between signal measurement "
                f"and portfolio returns ({lag_text}). The signal used at date t "
                "must be computable from information available strictly before "
                "the portfolio formation date.",
                evidence={"formation_lag": formation_lag},
            )
        )

        sort_method = _opt(portfolio, "sort_method")
        n_portfolios = _opt(portfolio, "n_portfolios")
        obligations.append(
            _obligation(
                "portfolio-sort",
                "construction",
                f"Implement the portfolio sort exactly: {sort_method or 'as described'}; "
                f"{n_portfolios or '?'} portfolios; long side {_opt(portfolio, 'long_side') or 'top'}; "
                f"short side {_opt(portfolio, 'short_side') or 'bottom'}.",
                evidence={k: portfolio.get(k) for k in
                          ("sort_method", "n_portfolios", "long_side", "short_side")},
            )
        )

        weighting = _opt(portfolio, "weighting") or "unknown"
        rebalancing = _opt(portfolio, "rebalancing")
        holding = _opt(portfolio, "holding_period")
        weighting_words = weighting.replace("_", " ") if isinstance(weighting, str) else weighting
        obligations.append(
            _obligation(
                "weighting-rebalancing",
                "construction",
                f"Weight portfolios {weighting_words} and rebalance "
                f"{rebalancing or 'as specified in the paper'} "
                f"(holding period: {holding or 'as described'}). "
                "Value-weighting must use lagged market capitalization.",
                evidence={"weighting": weighting, "rebalancing": rebalancing,
                          "holding_period": holding},
            )
        )

    factor_controls = _opt(finance_brief, "factor_controls") or []
    if factor_controls:
        obligations.append(
            _obligation(
                "factor-controls",
                "statistics",
                f"Report alphas and loadings from regressions against the "
                f"declared factor models ({', '.join(map(str, factor_controls))}), "
                "using the paper frequency and Newey-West standard errors with "
                "the paper lag choice.",
                evidence={"factor_controls": factor_controls},
            )
        )

    hypothesis = _opt(finance_brief, "hypothesis")
    backtest_haystack = (
        str(hypothesis or "").lower()
        + str(portfolio.get("signal") or "").lower()
        + str(portfolio.get("sort_method") or "").lower()
    )
    looks_like_backtest = any(
        term in backtest_haystack
        for term in ("backtest", "strategy", "trend", "momentum")
    )
    costs = finance_brief.get("transaction_costs") or {}
    if looks_like_backtest or costs.get("included"):
        bps = _opt(costs, "bps_per_side")
        obligations.append(
            _obligation(
                "transaction-costs",
                "construction",
                f"Apply the transaction-cost model "
                f"({bps or 'as described'} bps per side; "
                f"{_opt(costs, 'model_notes') or 'see paper'}) and report results "
                "both gross and net of costs.",
                evidence={"bps_per_side": bps, "model_notes": _opt(costs, "model_notes")},
            )
        )

    winsorization = _opt(finance_brief, "winsorization")
    if winsorization:
        obligations.append(
            _obligation(
                "winsorization",
                "data",
                f"Winsorize variables at {winsorization} exactly as specified; "
                "apply identically across replications and robustness variants.",
                evidence={"winsorization": winsorization},
            )
        )

    data_sources = _opt(finance_brief, "data_sources") or []
    if data_sources:
        obligations.append(
            _obligation(
                "data-sources",
                "data",
                f"Draw inputs from the declared sources only "
                f"({', '.join(map(str, data_sources))}); record source, table and "
                "vintage for every input used.",
                evidence={"data_sources": data_sources},
            )
        )

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "hypothesis": hypothesis,
        "sample_period": finance_brief.get("sample_period") or {},
        "obligations": obligations,
        "counts": {
            "total": len(obligations),
            "by_category": {
                cat: sum(1 for o in obligations if o["category"] == cat)
                for cat in sorted({o["category"] for o in obligations})
            },
        },
    }


def validate_finance_contract(contract: dict) -> list[str]:
    """Return human-readable coverage gaps in the contract."""
    gaps: list[str] = []
    ids = {o["id"] for o in contract.get("obligations", [])}
    brief_surv = None
    for o in contract.get("obligations", []):
        if o["id"] == "survivorship-delisting":
            brief_surv = (o.get("evidence") or {}).get("survivorship")
    if brief_surv in (None, "unknown"):
        gaps.append("survivorship treatment unknown — flag for human review")
    if "formation-lag" not in ids:
        gaps.append("no formation-lag obligation: portfolio timing cannot be verified")
    if not contract.get("obligations"):
        gaps.append("contract is empty")
    return gaps


def finance_requirements_text(contract: dict) -> str:
    """Render the contract as a prompt-ready requirements block."""
    lines = [
        "Finance implementation requirements (from the paper empirical design):"
    ]
    for o in contract.get("obligations", []):
        lines.append(f"- [{o['category']}] {o['requirement']}")
    gaps = validate_finance_contract(contract)
    if gaps:
        lines.append("Open coverage gaps requiring paper re-reading or human input:")
        lines.extend(f"- {gap}" for gap in gaps)
    return "\n".join(lines)


def augment_ir(ir: dict, contract: dict) -> dict:
    """Attach the finance contract to a canonical-IR-shaped dict (defensive)."""
    out = dict(ir)
    out["finance_contract"] = {
        "schema_version": contract.get("schema_version"),
        "hypothesis": contract.get("hypothesis"),
        "obligations": contract.get("obligations", []),
        "gaps": validate_finance_contract(contract),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a finance implementation contract from a finance brief JSON."
    )
    parser.add_argument("brief", help="path to finance_brief.json")
    parser.add_argument("-o", "--out", help="output path (default: alongside the brief)")
    args = parser.parse_args(argv)

    brief_path = Path(args.brief)
    if not brief_path.exists():
        print(f"brief not found: {brief_path}", file=sys.stderr)
        return 1
    brief = json.loads(brief_path.read_text(encoding="utf-8"))
    contract = build_finance_contract(brief)
    out_path = Path(args.out) if args.out else brief_path.parent / "finance_contract.json"
    out_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"contract written: {out_path} ({contract['counts']['total']} obligations)")
    for gap in validate_finance_contract(contract):
        print(f"  gap: {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
