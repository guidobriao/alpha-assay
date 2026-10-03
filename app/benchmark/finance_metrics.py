"""Pure-Python statistics for finance replication benchmarks.

Deliberately dependency-free (stdlib only): the benchmark framework and its
generated runners must work in minimal environments where pandas/statsmodels
may be absent. All series are ordered oldest-to-newest; returns are in
percent per period (e.g. monthly).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Sequence


def mean(values: Sequence[float]) -> float | None:
    xs = [float(v) for v in values]
    if not xs:
        return None
    return sum(xs) / len(xs)


def t_stat(values: Sequence[float]) -> float | None:
    """Plain t-statistic for the mean of a series (no autocorrelation correction)."""
    x = [float(v) for v in values]
    n = len(x)
    if n < 2:
        return None
    xbar = mean(x)
    var = sum((v - xbar) ** 2 for v in x) / (n - 1)
    if var <= 0:
        return None
    return xbar / math.sqrt(var / n)


def newey_west_t_stat(values: Sequence[float], lags: int = 1) -> float | None:
    """t-statistic for the mean using a Newey-West (Bartlett kernel) long-run
    variance estimator — the standard significance test for monthly portfolio
    sorts in asset pricing."""
    x = [float(v) for v in values]
    n = len(x)
    if n <= lags + 1:
        return None
    xbar = sum(x) / n
    demeaned = [v - xbar for v in x]
    long_run = sum(d * d for d in demeaned) / n
    for lag in range(1, lags + 1):
        gamma_l = sum(demeaned[t] * demeaned[t - lag] for t in range(lag, n)) / n
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma_l
    if long_run <= 0:
        return None
    se = math.sqrt(long_run / n)
    if se == 0:
        return None
    return xbar / se


def sharpe_ratio(
    returns_pct: Sequence[float],
    periods_per_year: float = 12.0,
    rf_annual_pct: float = 0.0,
) -> float | None:
    """Annualized Sharpe ratio from period returns in percent."""
    r = [float(v) for v in returns_pct]
    if len(r) < 2:
        return None
    rf_period = rf_annual_pct / periods_per_year
    excess = [v - rf_period for v in r]
    m = mean(excess)
    if m is None:
        return None
    var = sum((v - m) ** 2 for v in excess) / (len(excess) - 1)
    if var <= 0:
        return None
    return (m / math.sqrt(var)) * math.sqrt(periods_per_year)


def annualized_return(returns_pct: Sequence[float], periods_per_year: float = 12.0) -> float | None:
    """Geometric annualized return (percent) from period returns in percent."""
    r = [float(v) for v in returns_pct]
    if not r:
        return None
    growth = 1.0
    for v in r:
        growth *= 1.0 + v / 100.0
    years = len(r) / periods_per_year
    if growth <= 0 or years <= 0:
        return None
    return (growth ** (1.0 / years) - 1.0) * 100.0


def max_drawdown(returns_pct: Sequence[float]) -> float | None:
    """Maximum peak-to-trough drawdown in percent (negative number,
    e.g. -35.2 for a 35.2% drawdown)."""
    r = [float(v) for v in returns_pct]
    if not r:
        return None
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for v in r:
        equity *= 1.0 + v / 100.0
        peak = max(peak, equity)
        if peak > 0:
            mdd = min(mdd, equity / peak - 1.0)
    return mdd * 100.0


def ols_alpha_beta(
    y_pct: Sequence[float],
    x_pct: Sequence[float],
    annualize_alpha: bool = False,
) -> tuple[float | None, float | None]:
    """OLS regression y = alpha + beta*x on period returns (percent).
    Returns (alpha, beta); alpha is annualized when annualize_alpha is True
    (monthly data assumed)."""
    y = [float(v) for v in y_pct]
    x = [float(v) for v in x_pct]
    n = min(len(y), len(x))
    if n < 3:
        return None, None
    y, x = y[:n], x[:n]
    mx, my = mean(x), mean(y)
    if mx is None or my is None:
        return None, None
    var_x = sum((xi - mx) ** 2 for xi in x)
    if var_x <= 0:
        return None, None
    beta = sum((yi - my) * (xi - mx) for yi, xi in zip(y, x)) / var_x
    alpha = my - beta * mx
    if annualize_alpha:
        alpha = alpha * periods_per_year_for(annualize_alpha)
    return alpha, beta


def periods_per_year_for(_flag: bool) -> float:
    """Placeholder constant for monthly data; kept as a function so callers
    can switch frequency explicitly later without signature churn."""
    return 12.0


_MONTHLY_DATE = re.compile(r"^\d{6}$")


def load_french_monthly(path: str | Path) -> list[tuple[str, list[float]]]:
    """Load the monthly block of a Kenneth French library CSV file.

    French files interleave monthly and annual blocks separated by -99.99
    sentinel rows. Only rows whose first field is a 6-digit YYYYMM date are
    kept, so annual (4-digit) and daily (8-digit) blocks are skipped
    automatically. Rows containing -99.99-style missing markers are dropped.
    """
    rows: list[tuple[str, list[float]]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts or not _MONTHLY_DATE.fullmatch(parts[0]):
            if rows:
                # First non-monthly row after the monthly block: stop.
                break
            continue
        series: list[float] = []
        ok = True
        for value in parts[1:]:
            try:
                x = float(value)
            except ValueError:
                ok = False
                break
            if x <= -90.0:
                ok = False
                break
            series.append(x)
        if ok and series:
            rows.append((parts[0], series))
    return rows
