"""Domain-specific schema for finance replications.

Attached to `BenchmarkSpec.finance` for finance task families only.
`PortfolioSpec.formation_lag` and `SamplePeriod.in_sample_end` are consumed
downstream by the leakage detector and the out-of-sample agent.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


FINANCE_TASK_FAMILIES: frozenset[str] = frozenset({
    "cross_sectional_return_prediction",
    "event_study",
    "factor_model_replication",
    "time_series_strategy",
})


class UniverseSpec(BaseModel):
    """Investment universe of a finance replication study."""
    description: Optional[str] = None
    exchanges: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    min_price: Optional[float] = None
    min_market_cap_musd: Optional[float] = None
    survivorship: Literal["point_in_time", "current_constituents", "unknown"] = "unknown"
    delisting_treatment: Optional[str] = None


class SamplePeriod(BaseModel):
    """Sample period of the study; `in_sample_end` is the OOS split boundary."""
    start: Optional[str] = None
    end: Optional[str] = None
    frequency: Literal["daily", "weekly", "monthly", "quarterly", "annual", "event", "unknown"] = "unknown"
    in_sample_end: Optional[str] = None
    notes: list[str] = Field(default_factory=list)


class PortfolioSpec(BaseModel):
    """Portfolio construction of the replicated sort/strategy."""
    signal: Optional[str] = None
    sort_method: Optional[str] = None
    n_portfolios: Optional[int] = None
    weighting: Literal["value_weighted", "equal_weighted", "unknown"] = "unknown"
    rebalancing: Optional[str] = None
    formation_lag: Optional[str] = None
    holding_period: Optional[str] = None
    long_side: Optional[str] = None
    short_side: Optional[str] = None


class CostSpec(BaseModel):
    """Transaction-cost model of the study."""
    included: bool = False
    bps_per_side: Optional[float] = None
    model_notes: Optional[str] = None


class FinanceSpec(BaseModel):
    """Finance-specific contract attached to a benchmark specification."""
    hypothesis: Optional[str] = None
    universe: Optional[UniverseSpec] = None
    sample_period: Optional[SamplePeriod] = None
    portfolio: Optional[PortfolioSpec] = None
    factor_controls: list[str] = Field(default_factory=list)
    transaction_costs: Optional[CostSpec] = None
    winsorization: Optional[str] = None
    data_sources: list[str] = Field(default_factory=list)
