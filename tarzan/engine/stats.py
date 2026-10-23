"""Pure statistical return/risk math.

No I/O, no network, no global mutable state — every function here is a
pure transform of its inputs. Split out of ``metrics.py`` so the reusable
math can be imported (and unit-tested) without dragging in the pipeline
orchestration or the yfinance/benchmark fetch layer.

Contains:
  * period returns: ``compute_cagr``, ``compute_period_return``,
    ``compute_ytd_return``
  * money-weighted return: ``xnpv``, ``xirr``
  * time-weighted return: ``TwrResult``, ``twr``
  * risk: ``compute_sharpe``, ``compute_sortino``, ``compute_max_drawdown``,
    ``compute_var``, ``compute_cvar``, ``_compute_beta_alpha``
  * shared constants: ``RISK_FREE_RATE``, ``TRADING_DAYS``, ``DAYS_PER_YEAR``
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from functools import cached_property
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tarzan import config as cfg

# ---------------------------------------------------------------------------
# Shared constants — the single home for the annualization conventions used
# across CAGR, XIRR, TWR and the risk metrics.
# ---------------------------------------------------------------------------
RISK_FREE_RATE = cfg.risk_free_rate() * 100  # e.g. 4.0 = 4%
TRADING_DAYS = cfg.trading_days()
# Calendar days per year used for ALL annualization (CAGR, XIRR, TWR) so
# the money-weighted and time-weighted figures are directly comparable.
DAYS_PER_YEAR = 365.25

# The single source of truth for return-bucket windows: bucket → (unit, span).
# Every per-instrument / portfolio period return in Tarzan (benchmarks, holding
# performance, the portfolio performance_full, the newsletter Returns tables and
# the Excel Performance tab) is measured through this one mapping, so the columns
# mean exactly the same thing everywhere. ``ytd`` is special-cased (since Jan 1)
# and handled by ``compute_ytd_return``.
#
# The units are the ones a published trailing return uses: calendar months and
# years, and sessions for the short buckets. Fixed day counts (90 for "3M", 1825
# for "5Y") were what drifted — 90 days landed two sessions past three calendar
# months and read 3M +6.45% for EXUS.MI on 18 Aug 2026 where three months read
# +7.87%.
# For a "sessions" bucket the number is how many sessions BACK the window is
# anchored — i.e. how many days of change it measures. "5D" is five days of
# change, which needs six closes, and Yahoo's own page agrees: on 24 Aug 2026 it
# read AVWS.DE 5D as -1.56% and RSSY as -1.28%, both anchored on 17 Aug, five
# sessions before the current one. Tarzan stepped back span-1 = four, anchoring
# 18 Aug, so every 5D was one session short: AVWS came out -1.04%. The 1D was
# already right (one step), which is why only the 5D disagreed.
PERIOD_WINDOWS: dict[str, tuple[str, int]] = {
    "1d": ("sessions", 1),
    "5d": ("sessions", 5),
    "1m": ("months", 1),
    "3m": ("months", 3),
    "6m": ("months", 6),
    "1y": ("years", 1),
    "3y": ("years", 3),
    "5y": ("years", 5),
}


def _series_ticker(series) -> str:
    """The provider symbol a series belongs to, for its exchange calendar.

    Both producers name their series after the resolved listing: the enricher
    stamps ``holding.ticker`` onto ``price_history`` and
    ``_fetch_benchmark_history`` sets the selected symbol as the name and in
    ``attrs``. An unnamed or differently-named series resolves to no venue,
    which makes every calendar lookup fall back to the Mon-Fri rule.
    """
    if series is None:
        return ""
    attrs = getattr(series, "attrs", None) or {}
    return str(attrs.get("resolved_ticker") or getattr(series, "name", "") or "")


def statistical_series(series: pd.Series) -> pd.Series:
    """An independent numerical copy for statistics, not for source evidence.

    Provider series may carry full native histories and per-row provenance in
    ``attrs``. Pandas deep-copies those attributes during ordinary arithmetic,
    slicing and resampling, multiplying their cost across every risk metric.
    Constructing from the array avoids copying that payload at this boundary.

    Values, dtype, index, name and duplicate-label policy remain unchanged.
    The resolved listing is retained because return windows use its exchange
    calendar, even when the series is named ``Close``. Original provider
    evidence is neither modified nor replaced: quote selection, FX, coverage
    and publication gates must continue to consume the source series.
    """
    result = pd.Series(
        series.array,
        index=series.index.copy(deep=True),
        name=series.name,
        copy=True,
    )
    result.flags.allows_duplicate_labels = series.flags.allows_duplicate_labels
    ticker = _series_ticker(series)
    if ticker:
        result.attrs["resolved_ticker"] = ticker
    return result


def _roll_to_session(timestamp, ticker: str):
    """Roll a timestamp back to the last SESSION on-or-before it, on
    ``ticker``'s exchange calendar. Falls back to the Mon-Fri rule."""
    from tarzan.data.exchange_calendar import last_session_on_or_before

    rolled = last_session_on_or_before(ticker, timestamp.date())
    out = pd.Timestamp(rolled)
    return out.tz_localize(timestamp.tz) if timestamp.tz is not None else out


def _window_end(series_end, ticker: str = ""):
    """The session a window is measured back FROM: the one the series' LAST
    OBSERVATION belongs to.

    A window's two edges have to be read off the same clock. This used to take
    the start from the run's calendar day while the endpoint came from the
    series, so on a pre-open run the two sat one session apart and every window
    was a session short of the published figure: AVWS.DE's 5D anchored 18 Aug
    against Yahoo's 17 Aug and read -1.04% where the page showed -1.56%.

    What makes the last observation the right end in BOTH session states is the
    stamp (see :mod:`tarzan.data.current_session`), which dates its point by the
    quote's own ``regularMarketTime``:

    * market open — the quote is today's live price, so the last observation IS
      the latest intraday point and the window ends on the current session,
      which is what a broker and Yahoo both show;
    * market closed — the quote is the completed session's close, so the window
      ends there, and a weekend or holiday run measures from the last session
      rather than from a calendar day that never traded.

    A series with no current observation at all (a feed Yahoo does not quote, or
    a quote the sanity gate rejected) therefore measures the window it has data
    for and says so, instead of pairing a start counted from today with an
    endpoint from last week.

    ``series_end`` may be a carried calendar day — the order-derived portfolio
    NAV runs on ``freq="D"`` with weekends flat — so it is rolled back to the
    last real session before being used.
    """
    try:
        return _roll_to_session(series_end, ticker)
    except Exception:  # noqa: BLE001 — a clock must never break a return
        return series_end


def _ytd_anchor(series: pd.Series):
    """The observation "YTD" is measured FROM, or None.

    YTD is the one bucket that is not "N units back from today", so it cannot go
    through the unit arithmetic below. It is also the one bucket that already had
    a return function (:func:`compute_ytd_return`) before this module owned window
    edges, so the rule here is transcribed from it rather than invented: the LAST
    close of the prior year when the series has one, else the first in-year
    observation (mid-year inception) provided at least two exist. Diverging by even
    one session would make a YTD chart disagree with the YTD column beside it.

    The year comes from the series' own last observation, NOT from
    ``_window_end``. Rolling to the last session first is right for every trailing
    bucket and wrong for this one: on Saturday 2 Jan the roll lands on 31 Dec of
    the OLD year, which would anchor "year to date" in the year before last.
    """
    s = series.dropna()
    if s.empty:
        return None
    current_year = s.index[-1].year
    prior = s[s.index.year < current_year]
    in_year = s[s.index.year == current_year]
    if in_year.empty:
        return None
    if not prior.empty:
        return prior.index[-1]
    return in_year.index[0] if len(in_year) >= 2 else None


def window_anchor(series: pd.Series, bucket: str, ticker: Optional[str] = None):
    """The observation a bucket is measured FROM, or None when the series does
    not reach back that far.

    One authority for every window edge: the returns themselves, the
    newsletter's methodology note, and any check against a published figure all
    read this, so a bucket cannot mean one span in a table and another in its
    footnote.

    The anchor is the last close AT OR BEFORE the window's start date, which is
    what "one month ago" means everywhere it is quoted: for XMME.MI on 18 Aug
    2026 that is Friday 17 Jul (76.78, 1M = +2.55%). Taking the first close
    at or *after* it instead skipped to Monday 20 Jul (77.86) and reported
    +1.13% — a whole weekend of the window silently dropped.

    The start date is counted back from the run's today (see
    :func:`_window_end`), so a stale feed shortens the window rather than
    sliding it into the past.
    """
    if series is None or len(series) < 2:
        return None
    if bucket == "ytd":
        return _ytd_anchor(series)
    unit, span = PERIOD_WINDOWS.get(bucket, ("days", 0))
    if ticker is None:
        ticker = _series_ticker(series)
    end = _window_end(series.index[-1], ticker)
    # A window edge is a SESSION DATE, so the whole comparison is done on dates
    # and never on timestamps. A daily bar is stamped at its venue's midnight —
    # the same session reads 00:00-04:00 in New York and 04:00+00:00 once
    # converted to EUR — while ``end`` may come from the run's own clock at
    # midnight. Mixing the two put the cutoff hours before the session it named,
    # so the anchor silently slid one session early: RSSY's 5D measured six
    # sessions and WTIP's reconciled against neither its native nor its EUR
    # return. Dates have no time-of-day to disagree about.
    end_date = end.date()
    if unit == "sessions":
        # "5D" the way Yahoo and the brokers count it: five trading days back on
        # the exchange calendar, not five ROWS of the vendor's series — Milan's
        # 17 Aug 2026 is missing a close, and counting rows would then reach a
        # session further back than the site's own 5D range. The calendar is the
        # venue's own, so the Assumption (Milan shut, Xetra open) shortens one
        # and not the other.
        from tarzan.data.exchange_calendar import sessions_back

        cutoff_date = sessions_back(ticker, end_date, span)
    elif unit == "months":
        cutoff_date = (pd.Timestamp(end_date) - pd.DateOffset(months=span)).date()
    elif unit == "years":
        cutoff_date = (pd.Timestamp(end_date) - pd.DateOffset(years=span)).date()
    else:
        return None
    # ``index.date`` is the venue's own session date for a tz-aware index, which
    # is exactly the key a window is measured on.
    index_dates = series.index.date
    at_or_before = series.index[index_dates <= cutoff_date]
    if len(at_or_before):
        anchor = at_or_before[-1]
        # The whole series sits before the window: there is nothing to measure
        # across, and reporting the last close against itself would print a
        # confident 0.00% (or +€0) for a feed that simply stopped.
        if anchor == series.index[-1]:
            return None
    elif series.index[0].date() <= cutoff_date + datetime.timedelta(days=7):
        # Nothing that old: report the bucket only if the series still covers
        # essentially the whole window (a weekend or holiday of slack), so a 2y
        # book cannot print a "5Y" return off its own first close.
        anchor = series.index[0]
    else:
        return None

    # "1D" names its span exactly, so unlike every longer bucket it may not
    # absorb a missing bar: a month is still a month if a close inside it is
    # absent, but if YESTERDAY's bar is missing (the documented Milan null-close
    # case) the anchor slides one session further back and a TWO-session move
    # gets printed under a "1D" label with nothing marking it. Report nothing
    # instead — the same rule this function already applies above when the
    # anchor collapses onto the series end. Measured on the venue's own
    # calendar, so the session after a holiday is correctly adjacent.
    if bucket == "1d":
        from tarzan.data.exchange_calendar import (
            last_session_on_or_before, previous_session,
        )

        # Compare against the session the series effectively ENDS on, not its
        # last row. The order-derived portfolio NAV is calendar-daily (weekends
        # carried flat), so its last row is routinely a Saturday or Sunday whose
        # value IS Friday's close; demanding an anchor adjacent to the ROW would
        # blank the 1D on every weekend run.
        end_session = last_session_on_or_before(
            ticker, series.index[-1].date())
        if anchor.date() < previous_session(ticker, end_session):
            return None
    return anchor


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


def compute_period_return(series: pd.Series, bucket: str) -> Optional[float]:
    """Return the % change over a ``PERIOD_WINDOWS`` bucket ("5d", "3m", …).

    ``None`` when the series does not cover the window, instead of silently
    measuring a shorter one (a 2Y book must not print a "5Y" return next to a
    benchmark's real one).

    Args:
        series: Daily price series (datetime-indexed).
        bucket: A key of :data:`PERIOD_WINDOWS`.

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
    anchor = window_anchor(series, bucket)
    if anchor is None:
        return None
    start = float(series.loc[:anchor].iloc[-1])
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
# Money-weighted (XIRR) and time-weighted (TWR) returns
# ======================================================================

def xnpv(rate: float, cashflows: list[tuple[datetime.date, float]]) -> float:
    """Net present value of dated ``cashflows`` at a constant annual
    ``rate``, discounting on an actual/365.25 day count from the earliest
    flow (same convention as CAGR/TWR so the figures are comparable)."""
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
class TwrResult:
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


def twr(
    valuations: list[tuple[datetime.date, float]],
    external_flows: dict[datetime.date, float],
    span_days: int,
    coverage_pct: float = 100.0,
) -> TwrResult:
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
    the whole point of TWR — a pure deposit yields r = 0).
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
    return TwrResult(
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


def _risk_free_or_none(rf_daily):
    """Empty RF evidence is absent; nonempty missing evidence is unavailable.

    Normalize before choosing a scalar-CAGR versus time-varying formula.
    A nonempty all-NaN/nullable all-NA path must never become a fallback
    assumption or an invented observed zero-rate path.
    """
    return None if isinstance(rf_daily, pd.Series) and rf_daily.empty else rf_daily


def _align_rf_daily(daily_returns: pd.Series, rf_daily) -> pd.Series:
    """Align a risk-free input to ``daily_returns``' index, returning a daily
    fraction per row. ``rf_daily`` may be a daily-fraction ``pd.Series`` (a
    time-varying path, aligned by date with forward-fill), a scalar annual
    percent, or ``None``/an empty series (falls back to ``RISK_FREE_RATE``).
    Only the last known rate ON OR BEFORE each target session is carried.
    Missing source rows preserve a previously known rate; dates before the
    first known rate stay missing. Neither source order nor consumer order can
    introduce a future rate. This session-date rule does not establish a
    provider's publication time or observed-rate provenance."""
    rf_daily = _risk_free_or_none(rf_daily)
    idx = daily_returns.index
    if isinstance(rf_daily, pd.Series) and not rf_daily.empty:
        def _norm(ix):
            return normalize_session_index(ix)
        r = rf_daily.copy()
        r.index = _norm(r.index)
        r = r[~r.index.duplicated(keep="last")]
        # Last input row wins within a session before sorting. Carry known
        # values on the source timeline, not the consumer's possibly shuffled
        # row order: otherwise a missing left anchor can be replaced by a
        # future observation, or an earlier target by the previous target row.
        r = r.sort_index(kind="stable").ffill()
        aligned = r.reindex(_norm(idx), method="ffill")
        aligned.index = idx
        return aligned
    scalar = (RISK_FREE_RATE if rf_daily is None else float(rf_daily))
    return pd.Series(scalar / 100.0 / TRADING_DAYS, index=idx)


def rf_annual_pct(rf_daily) -> float | None:
    """Collapse a time-varying daily risk-free path to a single annualised
    percent — the form Jensen's alpha needs (rf enters the CAPM regression as
    a level, so a per-day path would only perturb beta; the window mean is the
    right scalar and matches ``proxy_data.risk_free_annual``).

    Returns ``None`` when no series is supplied or it is empty, so ``_compute_beta_alpha``
    keeps its documented scalar ``RISK_FREE_RATE`` fallback — the same behaviour
    Sharpe/Sortino use in a pinned run with no cached rate rows.
    A nonempty path with no valid rate returns NaN, not the scalar fallback.
    """
    if isinstance(rf_daily, pd.Series):
        if rf_daily.empty:
            return None
        mean = rf_daily.mean()
        return float("nan") if pd.isna(mean) else float(mean) * TRADING_DAYS * 100.0
    if rf_daily is None:
        return None
    return float(rf_daily)


def compute_sharpe_tv(daily_returns: pd.Series, rf_daily, *,
                      periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe from a TIME-VARYING risk-free path.

    Builds per-period excess returns ``r_t − rf_t`` (each period charged its own
    prevailing short rate) and annualises ``mean/std × √periods``. This is the
    textbook excess-return Sharpe and is strictly more correct than using a
    single window-average rate when rates move a lot across the window.

    ``periods`` is the number of observations per year: 252 for daily returns,
    12 for monthly. Callers working on a reconstructed series should prefer
    monthly (see :func:`tarzan.engine.robustness.full_metrics`).
    An unknown RF for any requested period makes the ratio unavailable, not a
    ratio over the covered suffix promoted to this full window.
    """
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    rf = _align_rf_daily(daily_returns, rf_daily)
    if rf.isna().any():
        # Dropping unknown excess days would silently report a shorter window
        # next to the return/volatility calculated over the requested window.
        return float("nan")
    excess = (daily_returns - rf).dropna()
    return _sharpe_from_excess(excess, periods)


def _sharpe_from_excess(excess: pd.Series, periods: int) -> float:
    """The scalar Sharpe expression, shared with row-owned preparation."""
    if excess.empty:
        return float("nan")
    sample_sd = excess.std()
    if pd.isna(sample_sd):
        return float("nan")
    sd = float(sample_sd)
    if sd <= 0:
        return float("nan")
    return float(excess.mean()) / sd * np.sqrt(periods)


def compute_sortino_tv(daily_returns: pd.Series, rf_daily, *,
                       periods: int = TRADING_DAYS) -> float:
    """Sortino with a TIME-VARYING risk-free target. Downside deviation is the
    RMS shortfall of per-period excess returns below zero (i.e. below the
    prevailing risk-free), annualised — the target-semideviation form,
    consistent with ``compute_sortino`` but with a per-period target.
    ``periods`` = observations per year (252 daily, 12 monthly). An unknown RF
    for any requested period makes the ratio unavailable for this window."""
    if daily_returns is None or daily_returns.empty:
        return float("nan")
    rf = _align_rf_daily(daily_returns, rf_daily)
    if rf.isna().any():
        return float("nan")
    excess = (daily_returns - rf).dropna()
    return _sortino_from_excess(excess, periods)


def _sortino_from_excess(excess: pd.Series, periods: int) -> float:
    """The scalar Sortino expression, without repeating RF alignment."""
    if excess.empty:
        return float("nan")
    downside = excess.clip(upper=0.0)
    dd = float((downside ** 2).mean()) ** 0.5 * np.sqrt(periods)
    if dd <= 0:
        return float("nan")
    return float(excess.mean()) * periods / dd


def compute_sortino(daily_returns: pd.Series, annual_return: float,
                    risk_free: float | None = None, *,
                    periods: int = TRADING_DAYS) -> float:
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
    target_daily = rf / 100.0 / periods
    shortfall = (daily_returns - target_daily).clip(upper=0.0)
    downside_std = float((shortfall ** 2).mean()) ** 0.5 * np.sqrt(periods) * 100
    if downside_std <= 0:
        return float("nan")
    return (annual_return - rf) / downside_std


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    drawdown = _positive_drawdowns(series)
    return 0.0 if drawdown is None else float(drawdown.min())


def _positive_drawdowns(series: pd.Series) -> Optional[pd.Series]:
    """Fractional drawdowns on the same positive-price tail as both metrics."""
    # Guard the running-peak division: a leading zero/non-positive value makes
    # cummax==0 and (series-cummax)/cummax a 0/0 NaN (or ±inf against a zero
    # peak). Restrict to the strictly-positive tail so a synthetic/carry-flat
    # NAV that starts at 0 still yields a real drawdown instead of NaN.
    s = series[series > 0]
    if len(s) < 2:
        return None
    cummax = s.cummax()
    return (s - cummax) / cummax


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
    drawdown = _positive_drawdowns(series)
    if drawdown is None:
        return 0.0
    # Percentage drawdown at each point (≤ 0); square removes the sign so
    # depth and time-underwater both accumulate.
    drawdown_pct = drawdown * 100.0
    return float(np.sqrt((drawdown_pct ** 2).mean()))


def compute_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    return float(daily_returns.quantile(1 - confidence))


def compute_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    var = compute_var(daily_returns, confidence)
    return _cvar_from_var(daily_returns, var)


def _cvar_from_var(daily_returns: pd.Series, var: float) -> float:
    """Preserve the original inclusive tail and empty-tail fallback."""
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if not tail.empty else var


class PreparedStatistics:
    """Row-owned numerical facts, never a source or a cross-run price cache.

    Prices and a series-valued risk-free input are independent numerical
    copies. A caller must not mutate this preparation while consuming it.
    Each window/native-or-EUR tape gets its own preparation; provider evidence
    stays on the source. Lazy facts are shared only by metrics of that row.
    """

    def __init__(self, series: pd.Series, rf_daily=None):
        self.series = statistical_series(series)
        rf_daily = _risk_free_or_none(rf_daily)
        self.rf_daily = (
            statistical_series(rf_daily)
            if isinstance(rf_daily, pd.Series) else rf_daily
        )

    @cached_property
    def cagr(self) -> float:
        return compute_cagr(self.series)

    @cached_property
    def daily_returns(self) -> pd.Series:
        # Deliberately use the original, not dropna-cleaned prices. Changing
        # that order would change pct_change's missing-observation semantics.
        return self.series.pct_change().dropna()

    @cached_property
    def aligned_risk_free(self) -> pd.Series:
        return _align_rf_daily(self.daily_returns, self.rf_daily)

    @cached_property
    def excess_returns(self) -> pd.Series:
        return (self.daily_returns - self.aligned_risk_free).dropna()

    @cached_property
    def drawdowns(self) -> Optional[pd.Series]:
        return _positive_drawdowns(self.series)

    @cached_property
    def var_95(self) -> float:
        return compute_var(self.daily_returns, 0.95)

    def risk_metrics(self) -> dict:
        """Same formulas/reduction order as the original risk_metric_row."""
        cagr = self.cagr
        daily_ret = self.daily_returns
        vol = (
            float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100
            if len(daily_ret) > 0 else 0.0
        )
        if self.rf_daily is not None and len(daily_ret) > 0:
            if self.aligned_risk_free.isna().any():
                sharpe = sortino = float("nan")
            else:
                excess = self.excess_returns
                sharpe = _sharpe_from_excess(excess, TRADING_DAYS)
                sortino = _sortino_from_excess(excess, TRADING_DAYS)
        else:
            sharpe = compute_sharpe(cagr, vol)
            sortino = (
                compute_sortino(daily_ret, cagr)
                if len(daily_ret) > 0 else float("nan")
            )
        drawdowns = self.drawdowns
        max_drawdown = 0.0 if drawdowns is None else float(drawdowns.min())
        ulcer = (
            0.0 if drawdowns is None
            else float(np.sqrt(((drawdowns * 100.0) ** 2).mean()))
        )
        var = self.var_95
        cvar = (
            float("nan") if daily_ret.empty or len(daily_ret) < 5
            else _cvar_from_var(daily_ret, var)
        )
        return {
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown * 100,
            "ulcer_index": ulcer,
            "var_95": _scale_or_nan(var, 100),
            "cvar_95": _scale_or_nan(cvar, 100),
        }


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
    An empty path means absence and selects the same formula as ``None``;
    a nonempty path missing any requested return day's rate instead makes
    RF-dependent metrics unavailable, without shortening the other metrics.
    """
    return PreparedStatistics(series, rf_daily).risk_metrics()


def normalize_index(series: pd.Series, *, drop_duplicates: bool = False) -> pd.Series:
    """Copy of ``series`` with a tz-naive, calendar-day-normalized index.

    Series coming from different exchanges (different timezones) only align
    once their indices are collapsed to naive calendar days. ``drop_duplicates``
    additionally drops the duplicate days that the tz collapse can create
    (last observation wins) — needed when several intraday timestamps fold onto
    the same date.

    The day is the venue's OWN session date — the same key ``window_anchor``
    measures on (``index.date``) — so the tz is dropped at local wall time
    rather than converted to UTC first. A daily bar is stamped at midnight in
    its venue's tz, so converting Milan's 27 Aug 00:00+02:00 to UTC lands on
    26 Aug 22:00 and normalized to the 26th: every European series slid one
    day into the past, misaligned by a day against the tz-naive order-derived
    NAV and against any US series (whose midnight-04:00 stamp survives the UTC
    trip on the right date). That is what left the 30-day chart measuring
    26 Jul → 26 Aug while its own P&L matrix measured 27 Jul → 27 Aug, and on
    a day with a rebalance in the gap the Unrealized line read +0.00% beside a
    +1.79% table cell.
    """
    s = series.copy()
    s.index = normalize_session_index(s.index)
    if drop_duplicates:
        s = s[~s.index.duplicated(keep="last")]
    return s


def normalize_session_index(index) -> pd.DatetimeIndex:
    """Collapse a daily-bar index to tz-naive VENUE session dates.

    The index-level core of :func:`normalize_index`, extracted so every caller
    that has an index rather than a Series shares the one rule instead of
    re-deriving it. Re-deriving it is what went wrong: four call sites each
    wrote ``tz_convert("UTC").tz_localize(None)``, which is right for an
    INSTANT (a quote's observation time) and wrong for a SESSION DATE, and the
    difference is invisible until a European series slides a day into the past.
    See :func:`normalize_index` for what that cost.
    """
    idx = pd.DatetimeIndex(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


class PreparedWeeklyBenchmark:
    """Owned price snapshot for one exact reference/window.

    Cache weekly PRICE levels, not returns: regression aligns the two weekly
    tapes before pct_change, and missing whole weeks affect that ruler. This
    object is phase-local and must not be mutated by numerical consumers.
    """

    def __init__(self, history: pd.Series):
        self._history = statistical_series(history)
        self._weekly_prices: Optional[pd.Series] = None
        self._lock = Lock()

    @property
    def history(self) -> pd.Series:
        return self._history

    @property
    def weekly_prices(self) -> pd.Series:
        with self._lock:
            if self._weekly_prices is None:
                normalized = normalize_index(self.history)
                self._weekly_prices = (
                    normalized.resample("W-FRI").last().dropna()
                )
            return self._weekly_prices


def _compute_beta_alpha(
    series_or_returns: pd.Series,
    benchmark_history: pd.Series,
    annual_return: float = 0.0,
    risk_free: float | None = None,
    *,
    input_kind: str = "auto",
    prepared_benchmark: Optional[PreparedWeeklyBenchmark] = None,
) -> tuple[float, float]:
    """Compute beta and Jensen's alpha via OLS on weekly returns.

    The benchmark is a price series. ``input_kind="prices"`` declares portfolio
    price levels without a magnitude heuristic; ``"returns"`` declares daily
    fractions and reconstructs their complete cumulative price path. Default
    ``"auto"`` preserves the historical median heuristic and its normalization
    for compatibility. Known price-producing callers should opt into prices:
    a bond's EUR-per-unit price below 0.5 is not a return.
    We resample to weekly (Friday close)
    before computing returns and running the regression. Weekly returns
    eliminate the exchange-calendar misalignment that plagues daily cross-
    exchange comparisons (e.g. MSCI index publication dates don't match LSE
    trading days, destroying daily correlation). This is also the standard
    approach used by Morningstar, Bloomberg, and most institutional risk
    systems for β/α.

    Alpha is the OLS intercept annualized (× 52 weeks).
    """
    if not isinstance(input_kind, str) or input_kind not in ("auto", "prices", "returns"):
        raise ValueError("input_kind must be 'auto', 'prices' or 'returns'")
    if (
        prepared_benchmark is not None
        and prepared_benchmark.history is not benchmark_history
    ):
        raise ValueError("weekly preparation belongs to a different reference")
    if benchmark_history is None or len(benchmark_history) < 10:
        return float("nan"), float("nan")
    if series_or_returns is None or len(series_or_returns) < 10:
        return float("nan"), float("nan")

    port_raw = normalize_index(series_or_returns)
    bench_raw = (
        normalize_index(benchmark_history)
        if prepared_benchmark is None else None
    )

    if input_kind == "returns":
        # Keep the entire compounded path. Overwriting only its first point
        # creates a false jump in the next week when the first close is Friday.
        port_prices = (1 + port_raw).cumprod()
    elif input_kind == "auto" and port_raw.abs().median() < 0.5:
        # Preserve the legacy heuristic/normalization only for legacy callers.
        port_prices = (1 + port_raw).cumprod()
        port_prices.iloc[0] = 1.0  # normalize start
    else:
        port_prices = port_raw

    # Resample both to weekly (Friday close) — robust to different exchange
    # calendars, public holidays, and index publication lags.
    port_w = port_prices.resample("W-FRI").last().dropna()
    if prepared_benchmark is None:
        bench_w = bench_raw.resample("W-FRI").last().dropna()
    else:
        bench_w = prepared_benchmark.weekly_prices

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
    s.index = normalize_session_index(idx)
    return s.resample("B").last().dropna()


def _is_nan(value) -> bool:
    """True if value is a float NaN (None counts as not-NaN here)."""
    return isinstance(value, float) and value != value
