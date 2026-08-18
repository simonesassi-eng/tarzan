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

# The single source of truth for return-bucket windows, in *calendar* days.
# Every per-instrument / portfolio period return in Tarzan (benchmarks,
# holding performance, the portfolio performance_full, the newsletter Returns
# tables and the Excel Performance tab) is computed from this mapping, so the
# columns mean exactly the same thing everywhere. ``ytd`` is special-cased
# (since Jan 1) and handled by ``compute_ytd_return``.
PERIOD_DAYS: dict[str, int] = {
    "1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180,
    "1y": 365, "3y": 1095, "5y": 1825,
}

# What each bucket's window actually is, so a Tarzan return can be checked
# against the instrument's own Yahoo Finance page and agree. Yahoo's ranges are
# CALENDAR spans (1mo, 3mo, 6mo, 1y, 5y) and its 5D is five SESSIONS; a fixed
# day count drifts from both, silently. Measured on EXUS.MI on 18 Aug 2026:
# 90 days anchored on 20 May and read +6.45% where the site's 3mo range
# anchored on 18 May and read +7.87% — a 1.4pp gap from arithmetic alone.
# Buckets absent here keep the plain calendar-day window.
_PERIOD_WINDOW: dict[int, tuple[str, int]] = {
    7: ("sessions", 5),
    30: ("months", 1),
    90: ("months", 3),
    180: ("months", 6),
    365: ("years", 1),
    1095: ("years", 3),
    1825: ("years", 5),
}


def window_anchor(series: pd.Series, days: int):
    """The observation a bucket is measured FROM, or None when the series does
    not reach back that far.

    One authority for every window edge: the returns themselves, the
    newsletter's methodology note, and any check against Yahoo all read this,
    so a bucket cannot mean one span in a table and another in its footnote.
    """
    if series is None or len(series) < 2:
        return None
    end = series.index[-1]
    kind, span = _PERIOD_WINDOW.get(days, ("days", days))
    if kind == "sessions":
        # Yahoo's 5D spans the last five sessions and measures across them, so
        # the anchor is the FIRST of those five. Counted on the CALENDAR, not on
        # the series' own rows: a vendor gap (Milan's missing 17 Aug 2026) would
        # otherwise slide the window a session further back than the site's.
        cutoff = end - pd.tseries.offsets.BDay(span - 1)
    elif kind == "months":
        cutoff = end - pd.DateOffset(months=span)
    elif kind == "years":
        cutoff = end - pd.DateOffset(years=span)
    else:
        cutoff = end - pd.Timedelta(days=span)
    # The series must actually reach the window, else the bucket is reported as
    # unavailable rather than silently measuring a shorter span (a 2y book must
    # not print a "5Y" return). ~7 days of slack absorbs a weekend or holiday
    # at the far edge.
    if series.index[0] > cutoff + pd.Timedelta(days=7):
        return None
    inside = series.index[series.index >= cutoff]
    return inside[0] if len(inside) else None


# ======================================================================
# Period returns
# ======================================================================

def compute_cagr(series: pd.Series) -> float:
    # Drop missing observations first: the endpoints drive the whole ratio, so
    # a trailing NaN bar (providers append one for the not-yet-closed session,
    # esp. EU venues queried after close) or a leading gap would null the CAGR
    # while pct_change-based metrics on the same series stay valid. Mirrors the
    # dropna already done in compute_period_return.
    if series is not None:
        series = series.dropna()
    if series.empty or len(series) < 2:
        return 0.0
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0:
        return 0.0
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return 0.0
    return ((end / start) ** (1 / (days / DAYS_PER_YEAR)) - 1) * 100


def compute_period_return(series: pd.Series, days: int) -> Optional[float]:
    """Return the % change over the last ``days`` calendar days.

    If the series does not actually cover ``days`` of history we return
    ``None`` instead of silently falling back to the full available window.
    This avoids misleading comparisons (e.g. a 2Y portfolio reporting "3Y"
    returns that are actually 2Y returns next to a 3Y benchmark return).

    Args:
        series: Daily price series (datetime-indexed).
        days: Lookback window in calendar days. ``1`` is a special case
            that returns the last-trading-day change.

    Returns:
        Percentage return over the period, or ``None`` if there is not
        enough data.
    """
    # Drop missing observations first: a period return must be measured
    # between real closes. Data providers often append a trailing NaN bar
    # for the current (not-yet-closed) session — especially for European
    # instruments queried outside their trading hours — which would
    # otherwise null out every window (last/prev = NaN).
    if series is not None:
        series = series.dropna()
    if series is None or series.empty or len(series) < 2:
        return None
    if days <= 1:
        start = float(series.iloc[-2])
        return (((float(series.iloc[-1]) / start) - 1) * 100) if start > 0 else None
    anchor = window_anchor(series, days)
    if anchor is None or anchor >= series.index[-1]:
        return None
    start = float(series[series.index >= anchor].iloc[0])
    return (((float(series.iloc[-1]) / start) - 1) * 100) if start > 0 else None


def compute_ytd_return(series: pd.Series) -> Optional[float]:
    if series is not None:
        series = series.dropna()
    if series is None or series.empty:
        return None
    current_year = series.index[-1].year
    prior = series[series.index.year < current_year]
    ytd = series[series.index.year == current_year]
    if ytd.empty:
        return None
    # Anchor to the LAST close of the prior year when the series has it, not
    # the first in-year observation: YTD is measured from Dec-31, so a book
    # that rose in early January before its first current-year point (or after
    # a January data gap) would otherwise report ~0% instead of the real move.
    # Fall back to the first in-year point only when no prior-year data exists
    # (mid-year inception) — then at least two in-year points are required.
    if not prior.empty:
        start = float(prior.iloc[-1])
        end = float(ytd.iloc[-1])
    else:
        if len(ytd) < 2:
            return None
        start = float(ytd.iloc[0])
        end = float(ytd.iloc[-1])
    return (((end / start) - 1) * 100) if start > 0 else None


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

def compute_sharpe(annual_return: float, annual_volatility: float,
                   risk_free: float | None = None) -> float:
    rf = RISK_FREE_RATE if risk_free is None else risk_free
    if annual_volatility <= 0:
        return float("nan")
    return (annual_return - rf) / annual_volatility


def _align_rf_daily(daily_returns: pd.Series, rf_daily) -> pd.Series:
    """Align a risk-free input to ``daily_returns``' index, returning a daily
    fraction per row. ``rf_daily`` may be a daily-fraction ``pd.Series`` (a
    time-varying path, aligned by date with forward-fill), a scalar annual
    percent, or ``None`` (falls back to ``RISK_FREE_RATE``)."""
    idx = daily_returns.index
    if isinstance(rf_daily, pd.Series) and not rf_daily.empty:
        def _norm(ix):
            ix = pd.DatetimeIndex(ix)
            if getattr(ix, "tz", None) is not None:
                ix = ix.tz_convert("UTC").tz_localize(None)
            return ix.normalize()
        r = rf_daily.copy()
        r.index = _norm(r.index)
        r = r[~r.index.duplicated(keep="last")]
        aligned = r.reindex(_norm(idx), method="ffill").ffill().bfill()
        aligned.index = idx
        return aligned
    scalar = (RISK_FREE_RATE if rf_daily is None else float(rf_daily))
    return pd.Series(scalar / 100.0 / TRADING_DAYS, index=idx)


def rf_annual_pct(rf_daily) -> float | None:
    """Collapse a time-varying daily risk-free path to a single annualised
    percent — the form Jensen's alpha needs (rf enters the CAPM regression as
    a level, so a per-day path would only perturb beta; the window mean is the
    right scalar and matches ``proxy_data.risk_free_annual``).

    Returns ``None`` when no real series is available, so ``_compute_beta_alpha``
    keeps its documented scalar ``RISK_FREE_RATE`` fallback — the same behaviour
    Sharpe/Sortino use in a pinned run with no cached rate rows.
    """
    if isinstance(rf_daily, pd.Series):
        return float(rf_daily.mean()) * TRADING_DAYS * 100.0 if not rf_daily.empty else None
    if rf_daily is None:
        return None
    return float(rf_daily)


def compute_sharpe_tv(daily_returns: pd.Series, rf_daily) -> float:
    """Annualised Sharpe from a TIME-VARYING daily risk-free path.

    Builds daily excess returns ``r_t − rf_t`` (each day charged its own
    prevailing short rate) and annualises ``mean/std × √252``. This is the
    textbook excess-return Sharpe and is strictly more correct than using a
    single window-average rate when rates move a lot across the window.
    """
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    rf = _align_rf_daily(daily_returns, rf_daily)
    excess = (daily_returns - rf).dropna()
    sd = float(excess.std())
    if excess.empty or sd <= 0:
        return float("nan")
    return float(excess.mean()) / sd * np.sqrt(TRADING_DAYS)


def compute_sortino_tv(daily_returns: pd.Series, rf_daily) -> float:
    """Sortino with a TIME-VARYING daily risk-free target. Downside deviation
    is the RMS shortfall of daily excess returns below zero (i.e. below the
    prevailing daily risk-free), annualised — the target-semideviation form,
    consistent with ``compute_sortino`` but with a per-day target."""
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    rf = _align_rf_daily(daily_returns, rf_daily)
    excess = (daily_returns - rf).dropna()
    if excess.empty:
        return float("nan")
    downside = excess.clip(upper=0.0)
    dd = float((downside ** 2).mean()) ** 0.5 * np.sqrt(TRADING_DAYS)
    if dd <= 0:
        return float("nan")
    return float(excess.mean()) * TRADING_DAYS / dd


def compute_sortino(daily_returns: pd.Series, annual_return: float,
                    risk_free: float | None = None) -> float:
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
    rf = RISK_FREE_RATE if risk_free is None else risk_free
    target_daily = rf / 100.0 / TRADING_DAYS
    shortfall = (daily_returns - target_daily).clip(upper=0.0)
    downside_std = float((shortfall ** 2).mean()) ** 0.5 * np.sqrt(TRADING_DAYS) * 100
    if downside_std <= 0:
        return float("nan")
    return (annual_return - rf) / downside_std


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    # Guard the running-peak division: a leading zero/non-positive value makes
    # cummax==0 and (series-cummax)/cummax a 0/0 NaN (or ±inf against a zero
    # peak). Restrict to the strictly-positive tail so a synthetic/carry-flat
    # NAV that starts at 0 still yields a real drawdown instead of NaN.
    s = series[series > 0]
    if len(s) < 2:
        return 0.0
    cummax = s.cummax()
    drawdown = (s - cummax) / cummax
    return float(drawdown.min())


def compute_ulcer_index(series: pd.Series) -> float:
    """Ulcer Index (Martin & McCann, 1989): the root-mean-square of the
    percentage drawdowns from the running peak, in percent.

    Unlike Max Drawdown (a single worst point) the Ulcer Index captures
    both the *depth* and the *duration* of every drawdown — a portfolio
    that spends long stretches underwater scores worse. Lower is better;
    a value of 0 means the series only ever made new highs. Returned as a
    positive percentage (e.g. 7.3 = 7.3%), comparable to volatility.
    """
    if series is None or series.empty or len(series) < 2:
        return 0.0
    # Same cummax==0 guard as compute_max_drawdown: a leading zero peak makes
    # the percentage drawdown NaN/inf and poisons the RMS.
    series = series[series > 0]
    if len(series) < 2:
        return 0.0
    cummax = series.cummax()
    # Percentage drawdown at each point (≤ 0); square removes the sign so
    # depth and time-underwater both accumulate.
    drawdown_pct = (series - cummax) / cummax * 100.0
    return float(np.sqrt((drawdown_pct ** 2).mean()))


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


def risk_metric_row(series: pd.Series, rf_daily=None) -> dict:
    """The standard risk/return block for a price ``series``.

    cagr, volatility, sharpe, sortino, max_drawdown, ulcer_index, var_95,
    cvar_95 — percent units where applicable. Alpha/beta are left to the caller
    since they need a chosen reference benchmark. Shared by the benchmark metric
    builders and the per-holding performance rows so every table reports the
    same numbers computed the same way.

    When ``rf_daily`` (a daily risk-free path, ``pd.Series`` of daily fractions)
    is supplied, Sharpe/Sortino use the TIME-VARYING excess-return form (each day
    charged its own prevailing short rate); otherwise they fall back to the
    scalar ``RISK_FREE_RATE``. The single source of truth for the risk block, so
    time-varying vs scalar is decided once, here, for every table that shows it.
    """
    cagr = compute_cagr(series)
    daily_ret = series.pct_change().dropna()
    vol = float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100 if len(daily_ret) > 0 else 0.0
    if rf_daily is not None and len(daily_ret) > 0:
        sharpe = compute_sharpe_tv(daily_ret, rf_daily)
        sortino = compute_sortino_tv(daily_ret, rf_daily)
    else:
        sharpe = compute_sharpe(cagr, vol)
        sortino = compute_sortino(daily_ret, cagr) if len(daily_ret) > 0 else float("nan")
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": compute_max_drawdown(series) * 100,
        "ulcer_index": compute_ulcer_index(series),
        "var_95": _scale_or_nan(compute_var(daily_ret, 0.95), 100),
        "cvar_95": _scale_or_nan(compute_cvar(daily_ret, 0.95), 100),
    }


def normalize_index(series: pd.Series, *, drop_duplicates: bool = False) -> pd.Series:
    """Copy of ``series`` with a tz-naive, calendar-day-normalized index.

    Series coming from different exchanges (different timezones) only align
    once their indices are collapsed to naive calendar days. ``drop_duplicates``
    additionally drops the duplicate days that the tz collapse can create
    (last observation wins) — needed when several intraday timestamps fold onto
    the same date.
    """
    s = series.copy()
    idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()
    if drop_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    return s


def _compute_beta_alpha(
    series_or_returns: pd.Series,
    benchmark_history: pd.Series,
    annual_return: float = 0.0,
    risk_free: float | None = None,
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

    port_raw = normalize_index(series_or_returns)
    bench_raw = normalize_index(benchmark_history)

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
    rf_weekly = (RISK_FREE_RATE if risk_free is None else risk_free) / 100.0 / 52.0

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


def to_business_day_series(series: pd.Series) -> pd.Series:
    """Resample a DENSE calendar-day series onto business days (Mon–Fri).

    The order-derived portfolio NAV is built on ``freq="D"`` (every calendar
    day, weekends carried flat), but the risk metrics annualize daily
    volatility with ``sqrt(TRADING_DAYS)`` (252) — the trading-day convention.
    Feeding it calendar-day returns injects ~2/7 exact-zero (weekend) returns,
    understating volatility by ~sqrt(252/365) ≈ 0.83× and polluting the
    VaR/CVaR quantiles with a spike at 0. Collapsing to business days first
    puts the portfolio on the same trading-day basis as the (yfinance,
    trading-day) benchmarks, so ``sqrt(252)`` is then correct and the
    comparison is apples-to-apples.

    Only meant for the calendar-day order NAV. A series that is *already*
    trading-day (yfinance holdings/benchmarks) must NOT be passed here:
    resampling it to "B" would insert NaN rows on exchange holidays and drop
    the real returns around them. Every business day in the calendar-day input
    has a (carried) value, so this never introduces NaN. Empty/short input is
    returned unchanged.
    """
    if series is None or series.empty or len(series) < 2:
        return series
    s = series.copy()
    idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
        s.index = idx
    return s.resample("B").last().dropna()


def _is_nan(value) -> bool:
    """True if value is a float NaN (None counts as not-NaN here)."""
    return isinstance(value, float) and value != value
