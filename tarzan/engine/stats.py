"""Pure statistical return/risk math.

No I/O, no network, no global mutable state — every function here is a
pure transform of its inputs. Split out of ``metrics.py`` so the reusable
math can be imported (and unit-tested) without dragging in the pipeline
orchestration or the yfinance/benchmark fetch layer.

Contains:
  * period returns: ``compute_cagr``, ``compute_period_return``,
    ``compute_ytd_return``
  * money-weighted return: ``xnpv``, ``xirr``
  * time-weighted return: ``TwrorResult``, ``twror``
  * risk: ``compute_sharpe``, ``compute_sortino``, ``compute_max_drawdown``,
    ``compute_var``, ``compute_cvar``, ``_compute_beta_alpha``
  * shared constants: ``RISK_FREE_RATE``, ``TRADING_DAYS``, ``DAYS_PER_YEAR``
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tarzan import config as cfg

# ---------------------------------------------------------------------------
# Shared constants — the single home for the annualization conventions used
# across CAGR, XIRR, TWROR and the risk metrics.
# ---------------------------------------------------------------------------
RISK_FREE_RATE = cfg.risk_free_rate() * 100  # e.g. 4.0 = 4%
TRADING_DAYS = cfg.trading_days()
# Calendar days per year used for ALL annualization (CAGR, XIRR, TWROR) so
# the money-weighted and time-weighted figures are directly comparable.
DAYS_PER_YEAR = 365.25


# ======================================================================
# Period returns
# ======================================================================

def compute_cagr(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0:
        return 0.0
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return 0.0
    return ((end / start) ** (1 / (days / DAYS_PER_YEAR)) - 1) * 100


def compute_period_return(
    series: pd.Series, days: int, strict: bool = True,
) -> Optional[float]:
    """Return the % change over the last ``days`` calendar days.

    Args:
        series: Daily price series (datetime-indexed).
        days: Lookback window in calendar days. ``1`` is a special case
            that returns the last-trading-day change.
        strict: When True (default), if the series does not actually
            cover ``days`` of history we return ``None`` instead of
            silently falling back to the full available window. This
            avoids misleading comparisons (e.g. a 2Y portfolio reporting
            "3Y" returns that are actually 2Y returns next to a 3Y
            benchmark return).

    Returns:
        Percentage return over the period, or ``None`` if there is not
        enough data.
    """
    if series.empty or len(series) < 2:
        return None
    if days <= 1:
        start = float(series.iloc[-2])
        return (((float(series.iloc[-1]) / start) - 1) * 100) if start > 0 else None
    if strict:
        # Series must cover the full requested window. We allow a small
        # slack (~7 days) to absorb weekends/holidays at the edges.
        actual_span_days = (series.index[-1] - series.index[0]).days
        if actual_span_days < days - 7:
            return None
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    subset = series[series.index >= cutoff]
    if subset.empty or len(subset) < 2:
        return None
    start = float(subset.iloc[0])
    return (((float(subset.iloc[-1]) / start) - 1) * 100) if start > 0 else None


def compute_ytd_return(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    ytd = series[series.index.year == series.index[-1].year]
    if ytd.empty or len(ytd) < 2:
        return None
    start = float(ytd.iloc[0])
    return (((float(ytd.iloc[-1]) / start) - 1) * 100) if start > 0 else None


# ======================================================================
# Money-weighted (XIRR) and time-weighted (TWROR) returns
# ======================================================================

def xnpv(rate: float, cashflows: list[tuple[datetime.date, float]]) -> float:
    """Net present value of dated ``cashflows`` at a constant annual
    ``rate``, discounting on an actual/365.25 day count from the earliest
    flow (same convention as CAGR/TWROR so the figures are comparable)."""
    if not cashflows:
        return 0.0
    t0 = min(d for d, _ in cashflows)
    return sum(
        amount / (1.0 + rate) ** ((d - t0).days / DAYS_PER_YEAR)
        for d, amount in cashflows
    )


def xirr(cashflows: list[tuple[datetime.date, float]]) -> float:
    """Annualized money-weighted return: the constant rate making
    ``xnpv`` zero, found by bisection on [-0.999, 10].

    Returns NaN when the root cannot be bracketed — typically because
    every cash flow shares the same sign (no realised return), which is
    a legitimate "undefined", not an error.
    """
    if len(cashflows) < 2:
        return float("nan")
    lo, hi = -0.999, 10.0
    try:
        rate = brentq(lambda r: xnpv(r, cashflows), lo, hi, xtol=1e-7)
    except ValueError:
        return float("nan")
    # A solution pinned at the bracket edge is not a well-defined root:
    # near r → -1 (near-total loss) the NPV is ill-conditioned and brentq
    # converges on the rate without the residual actually vanishing.
    # Treat that as "undefined" rather than reporting a misleading rate.
    if rate <= lo + 1e-6 or rate >= hi - 1e-6:
        return float("nan")
    return rate


@dataclass
class TwrorResult:
    """Outcome of a time-weighted return computation.

    Attributes:
        cumulative_pct: chained return over the whole window, in %.
        annualized_pct: the cumulative return annualized over span_days.
        coverage_pct: share of portfolio value (0–100) priced by real
            market data over the window; < 100 means some periods relied
            on the synthetic/carry-flat fallback (disclosed to the user).
        periods: per-period diagnostics, each a dict with date,
            v_before, v_after, r (period return), and source tag.
    """

    cumulative_pct: float
    annualized_pct: float
    coverage_pct: float = 100.0
    periods: list[dict] = field(default_factory=list)


def twror(
    valuations: list[tuple[datetime.date, float]],
    external_flows: dict[datetime.date, float],
    span_days: int,
    coverage_pct: float = 100.0,
) -> TwrorResult:
    """Chained time-weighted return, neutral to deposit timing.

    Args:
        valuations: ``(date, V_after)`` pairs in chronological order,
            where ``V_after`` is the portfolio value at the close of the
            date *with* that date's external flow already applied.
        external_flows: external inflow into the portfolio per date in
            portfolio terms (deposits/buys positive, withdrawals/sells
            negative). ``V_before(d) = V_after(d) - external_flows[d]``.
        span_days: calendar days from first to last valuation, for
            annualization.
        coverage_pct: passthrough disclosure of data coverage.

    Between consecutive valuation dates the market return is
    ``r = V_before(d_i) / V_after(d_{i-1}) - 1``; subtracting the day's
    external flow keeps deposits/withdrawals out of the return (that is
    the whole point of TWROR — a pure deposit yields r = 0).
    """
    chained = 1.0
    prev_v_after = 0.0
    periods: list[dict] = []
    for d, v_after in valuations:
        if prev_v_after > 0:
            v_before = v_after - external_flows.get(d, 0.0)
            if v_before > 0:
                r = v_before / prev_v_after - 1.0
                chained *= 1.0 + r
                periods.append({
                    "date": d,
                    "v_before": v_before,
                    "v_after_prev": prev_v_after,
                    "r": r,
                })
        prev_v_after = v_after

    cumulative_pct = (chained - 1.0) * 100.0
    annualized_pct = (
        (chained ** (DAYS_PER_YEAR / span_days) - 1.0) * 100.0 if span_days > 0 else 0.0
    )
    return TwrorResult(
        cumulative_pct=cumulative_pct,
        annualized_pct=annualized_pct,
        coverage_pct=coverage_pct,
        periods=periods,
    )


# ======================================================================
# Risk
# ======================================================================

def compute_sharpe(annual_return: float, annual_volatility: float) -> float:
    if annual_volatility <= 0:
        return float("nan")
    return (annual_return - RISK_FREE_RATE) / annual_volatility


def compute_sortino(daily_returns: pd.Series, annual_return: float) -> float:
    """Sortino ratio using the textbook *target downside deviation*.

    The downside deviation is the root-mean-square shortfall below the
    (daily) risk-free target, taken over *all* observations — not the
    sample std of the negative-only subset. This is the standard target
    semideviation used by practitioners (Sortino & Price, 1994); the
    negative-only std variant understates the denominator on short or
    upward-skewed windows and inflates the ratio.
    """
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    target_daily = RISK_FREE_RATE / 100.0 / TRADING_DAYS
    shortfall = (daily_returns - target_daily).clip(upper=0.0)
    downside_std = float((shortfall ** 2).mean()) ** 0.5 * np.sqrt(TRADING_DAYS) * 100
    if downside_std <= 0:
        return float("nan")
    return (annual_return - RISK_FREE_RATE) / downside_std


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min())


def compute_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    return float(daily_returns.quantile(1 - confidence))


def compute_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    var = compute_var(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if not tail.empty else var


def _normalize_index(series: pd.Series) -> pd.Series:
    s = series.copy()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None).normalize()
    else:
        s.index = s.index.normalize()
    return s


def _compute_beta_alpha(
    series_or_returns: pd.Series,
    benchmark_history: pd.Series,
    annual_return: float = 0.0,
) -> tuple[float, float]:
    """Compute beta and Jensen's alpha via OLS on weekly returns.

    Both inputs should be **price** series (the function detects returns vs
    prices via a median heuristic). We resample to weekly (Friday close)
    before computing returns and running the regression. Weekly returns
    eliminate the exchange-calendar misalignment that plagues daily cross-
    exchange comparisons (e.g. MSCI index publication dates don't match LSE
    trading days, destroying daily correlation). This is also the standard
    approach used by Morningstar, Bloomberg, and most institutional risk
    systems for β/α.

    Alpha is the OLS intercept annualized (× 52 weeks).
    """
    if benchmark_history is None or len(benchmark_history) < 10:
        return float("nan"), float("nan")
    if series_or_returns is None or len(series_or_returns) < 10:
        return float("nan"), float("nan")

    port_raw = _normalize_index(series_or_returns)
    bench_raw = _normalize_index(benchmark_history)

    # If port is already returns (median abs < 0.5), reconstruct a price
    # index so we can resample to weekly cleanly.
    if port_raw.abs().median() < 0.5:
        port_prices = (1 + port_raw).cumprod()
        port_prices.iloc[0] = 1.0  # normalize start
    else:
        port_prices = port_raw

    # Resample both to weekly (Friday close) — robust to different exchange
    # calendars, public holidays, and index publication lags.
    port_w = port_prices.resample("W-FRI").last().dropna()
    bench_w = bench_raw.resample("W-FRI").last().dropna()

    # Align and compute weekly returns
    aligned = pd.DataFrame({"port": port_w, "bench": bench_w}).dropna()
    if len(aligned) < 5:
        return float("nan"), float("nan")
    rets = aligned.pct_change().dropna()
    if len(rets) < 4:
        return float("nan"), float("nan")

    # Weekly risk-free
    rf_weekly = RISK_FREE_RATE / 100.0 / 52.0

    port_excess = rets["port"] - rf_weekly
    bench_excess = rets["bench"] - rf_weekly

    var_bench = bench_excess.var()
    if var_bench <= 0:
        return float("nan"), float("nan")

    beta = port_excess.cov(bench_excess) / var_bench
    alpha_weekly = port_excess.mean() - beta * bench_excess.mean()
    alpha_annual = alpha_weekly * 52.0 * 100.0  # annualized, in %

    return float(beta), float(alpha_annual)


# ======================================================================
# Small numeric helpers
# ======================================================================

def _scale_or_nan(val: float, factor: float) -> float:
    if val != val:
        return val
    return val * factor


def _cap_to_years(series: pd.Series, years: float) -> pd.Series:
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=int(years * DAYS_PER_YEAR))
    return series[series.index >= cutoff]


def _safe_pct_change(old: float, new: float) -> float:
    if old <= 0 or new <= 0:
        return 0.0
    return (new - old) / old * 100


def _is_nan(value) -> bool:
    """True if value is a float NaN (None counts as not-NaN here)."""
    return isinstance(value, float) and value != value
