"""Backtest robustness: rolling windows, historical stress, block-bootstrap.

Pure functions (no I/O), like ``engine/stats.py`` — they consume a daily
NAV ``pd.Series`` (or daily returns) and return summary dicts. Used by the
what-if tooling to turn single-point estimates into distributions,
confidence intervals, and stress-tested outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.stats import (
    RISK_FREE_RATE,
    TRADING_DAYS,
    _align_rf_daily,
    _compute_beta_alpha,
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sharpe_tv,
    compute_sortino,
    compute_sortino_tv,
    compute_var,
)

# Below this many monthly observations the monthly standard deviation is noisier
# than the daily one is biased, so the volatility metrics stay on daily returns.
# Two years is the usual floor for a usable monthly moment estimate.
_MIN_MONTHS_FOR_MONTHLY = 24


def full_metrics(nav: pd.Series, bench_nav: pd.Series | None = None,
                 risk_free: float | None = None, rf_daily=None) -> dict:
    """Full return + risk metric set from a single NAV series (one window).

    CAGR / volatility / Sharpe / Sortino / MaxDrawdown / VaR / CVaR, plus
    Beta / Alpha vs an optional benchmark NAV. All in the percent units the
    stats module uses, so it reads like the dashboard's risk block.

    VOLATILITY, SHARPE and SORTINO are measured on MONTHLY returns, not daily
    ones. On a reconstructed series the daily frequency carries non-synchronous
    pricing noise — proxies that close in different time zones, an FX stamp at
    yet another hour, stale historical NAVs — which inflates daily variance and
    cancels on aggregation. Measured on the same portfolio: 15.6% annualised vol
    from daily returns against 11.1% from its own monthly returns. That the gap
    is noise and not risk is settled by restricting to the period where the
    funds have REAL prices (2021+), where the two agree (ratio 1.06 and daily
    autocorrelation -0.02, against 1.41 and -0.19 over the full window).

    The distortion is worse the more diversified the portfolio, because the
    noise is independent across sleeves while the true returns are correlated:
    blending cuts true variance but not noise variance. It ran 1.24x on a
    two-sleeve benchmark against 1.43x on the eight-sleeve candidates, i.e. it
    penalised precisely the portfolios built to diversify. Sharpe moved 0.41 to
    0.59 and Sortino 0.54 to 0.91 on the lead target once measured monthly.

    CAGR is frequency-independent. MAX DRAWDOWN stays on the DAILY path on
    purpose: it is a statement about the worst peak-to-trough actually traversed,
    and a month-end version would erase real intra-month crashes (March 2020).
    BETA and ALPHA also stay daily: portfolio and benchmark are built from the
    same proxies, so their timing noise is common and cancels in the ratio
    (measured beta moves 0.71 to 0.70 monthly), while daily gives 20x the
    observations. VaR/CVaR remain daily-horizon by definition and so keep the
    daily noise — read them as an upper bound.

    When ``rf_daily`` (a daily risk-free path) is supplied, Sharpe and Sortino
    use the TIME-VARYING excess-return form (r_t − rf_t) with the risk-free
    compounded to the same monthly frequency; otherwise they fall back to the
    scalar ``risk_free`` (window-average) form. Windows with fewer than
    ``_MIN_MONTHS_FOR_MONTHLY`` monthly observations keep the daily estimate: a
    standard deviation from 8 points is worse than a noisy one from 170.
    """
    if nav is None or len(nav) < 30:
        return {}
    # A zero/near-zero price surviving ffill produces a -100% then +inf daily
    # return; left in, daily.std() becomes inf and poisons volatility/Sharpe/
    # VaR/CVaR. Restrict to a strictly-positive NAV and finite returns.
    nav = nav[nav > 0]
    daily = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(nav) < 30 or daily.empty:
        return {}
    ann_ret = compute_cagr(nav)                                   # percent
    # Monthly returns for the volatility-based metrics (see the docstring): the
    # aggregation is what cancels the reconstruction's timing noise.
    monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
    use_monthly = len(monthly) >= _MIN_MONTHS_FOR_MONTHLY
    rets = monthly if use_monthly else daily
    per = 12 if use_monthly else TRADING_DAYS
    ann_vol = float(rets.std()) * np.sqrt(per) * 100.0            # percent
    if rf_daily is not None:
        rf_path = rf_daily
        if use_monthly:
            # Charge each month the rate that actually prevailed inside it.
            rf_aligned = _align_rf_daily(daily, rf_daily)
            rf_path = ((1.0 + rf_aligned).resample("ME").prod() - 1.0).dropna()
        sharpe = compute_sharpe_tv(rets, rf_path, periods=per)
        sortino = compute_sortino_tv(rets, rf_path, periods=per)
    else:
        sharpe = compute_sharpe(ann_ret, ann_vol, risk_free=risk_free)
        sortino = compute_sortino(rets, ann_ret, risk_free=risk_free, periods=per)
    m = {
        "cagr": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "risk_free": (RISK_FREE_RATE if risk_free is None else risk_free),
        "max_drawdown": compute_max_drawdown(nav) * 100.0,
        "var_95": compute_var(daily, 0.95) * 100.0,
        "cvar_95": compute_cvar(daily, 0.95) * 100.0,
        "beta": float("nan"),
        "alpha": float("nan"),
    }
    if bench_nav is not None and not bench_nav.empty and len(bench_nav) > 1:
        try:
            beta, alpha = _compute_beta_alpha(nav, bench_nav, ann_ret, risk_free=risk_free)
            m["beta"], m["alpha"] = beta, alpha
        except Exception:  # noqa: BLE001
            pass
    return m


# ---------------------------------------------------------------------------
# NAV construction
# ---------------------------------------------------------------------------

def daily_returns(nav: pd.Series) -> pd.Series:
    if nav is None or len(nav) < 2:
        return pd.Series(dtype=float)
    # Drop non-finite returns from a zero-price tick surviving ffill, so a
    # single bad print does not blow up volatility / bootstrap paths.
    return nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# Rolling-window return / risk distributions
# ---------------------------------------------------------------------------

def rolling_return_distribution(nav: pd.Series, window_days: int = 252) -> dict:
    """Distribution of annualised return over every rolling window.

    Turns one CAGR into "here is the spread of N-year outcomes you'd have
    got starting on any day", exposing regime dependence.
    """
    if nav is None or len(nav) <= window_days:
        return {}
    arr = nav.values
    horizon_years = window_days / TRADING_DAYS
    rets = arr[window_days:] / arr[:-window_days] - 1.0
    ann = np.power(1.0 + rets, 1.0 / horizon_years) - 1.0
    ann = ann[np.isfinite(ann)]
    if ann.size == 0:
        return {}
    return {
        "window_years": round(horizon_years, 2),
        "n": int(ann.size),
        "min": float(ann.min()),
        "p05": float(np.percentile(ann, 5)),
        "p25": float(np.percentile(ann, 25)),
        "median": float(np.median(ann)),
        "p75": float(np.percentile(ann, 75)),
        "p95": float(np.percentile(ann, 95)),
        "max": float(ann.max()),
        "pct_positive": float((ann > 0).mean() * 100.0),
    }


def rolling_sharpe_range(nav: pd.Series, window_days: int = 252, rf_daily=None) -> dict:
    """Min / median / max of the rolling annualised Sharpe ratio.

    Each rolling window is charged the risk-free rate prevailing over it: when
    ``rf_daily`` (the time-varying daily path) is supplied, the Sharpe uses daily
    excess returns ``r_t − rf_t`` (matching ``compute_sharpe_tv``); otherwise the
    per-day rate collapses to the scalar ``RISK_FREE_RATE`` and the result is
    identical to the flat-rate form.
    """
    r = daily_returns(nav)
    if len(r) <= window_days:
        return {}
    # Per-day excess returns; _align_rf_daily forward-fills the path onto r's
    # index, or lays down a flat RISK_FREE_RATE when rf_daily is None.
    excess = r - _align_rf_daily(r, rf_daily)
    roll = excess.rolling(window_days)
    sharpe = ((roll.mean() / roll.std()) * np.sqrt(TRADING_DAYS)
              ).replace([np.inf, -np.inf], np.nan).dropna()
    if sharpe.empty:
        return {}
    a = sharpe.values
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


# ---------------------------------------------------------------------------
# Historical stress scenarios
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "Dot-com 2000-02": ("2000-03-01", "2002-10-09"),
    "GFC 2008": ("2007-10-09", "2009-03-09"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-01", "2022-10-14"),
}


def stress_scenarios(nav: pd.Series, scenarios: dict | None = None) -> dict:
    """Total return and max drawdown of the NAV within each crisis window.

    Windows not covered by the available history are flagged
    ``covered=False`` so the report is honest about what was tested.
    """
    scenarios = scenarios or STRESS_SCENARIOS
    out: dict[str, dict] = {}
    if nav is None or nav.empty:
        return {name: {"covered": False} for name in scenarios}
    hist_start = nav.index.min()
    hist_end = nav.index.max()
    # Slack for a weekend/holiday between the window edge and the first/last
    # available close, so a window whose start falls on a non-trading day is
    # not spuriously flagged partial.
    edge_slack = pd.Timedelta(days=5)
    for name, (start, end) in scenarios.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        seg = nav.loc[(nav.index >= start_ts) & (nav.index <= end_ts)]
        if len(seg) < 5:
            out[name] = {"covered": False}
            continue
        # A segment can have plenty of rows yet still start mid-crisis when the
        # portfolio's history begins after the window's start (or ends before
        # its end). Anchoring the return/drawdown to the first AVAILABLE point
        # then understates the crisis (e.g. entering mid-GFC reports ~-25%
        # instead of the true ~-55% peak-to-trough). Report it as partial so
        # the truncated figure is not presented as a full stress test.
        partial = (hist_start > start_ts + edge_slack) or (hist_end < end_ts - edge_slack)
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        mdd = float(compute_max_drawdown(seg))
        out[name] = {
            "covered": True,
            "partial": partial,
            "return": ret * 100.0,
            "max_drawdown": mdd * 100.0,
            "window_start": start_ts.strftime("%Y-%m-%d"),
            "data_start": hist_start.strftime("%Y-%m-%d"),
        }
    return out


# ---------------------------------------------------------------------------
# Block-bootstrap Monte Carlo → confidence intervals
# ---------------------------------------------------------------------------

def block_bootstrap(nav: pd.Series, *, n_sims: int = 2000, block_months: int = 1,
                    horizon_days: int = 252, seed: int = 42, rf_annual=None,
                    block_days: int | None = None) -> dict:
    """Moving-block bootstrap of MONTHLY returns → CIs on CAGR / Sharpe /
    max drawdown over ``horizon_days``.

    Resampling happens on MONTHLY returns, not daily ones, and that choice is
    load-bearing. A EUR-reported portfolio built from proxies that trade in
    different time zones (and an FX conversion stamped at yet another hour)
    carries non-synchronous pricing noise: it inflates measured DAILY variance
    while cancelling on aggregation. On the levered global target this shows up
    as an annualised vol of 15.6% from daily returns against 11.1% from the
    monthly returns of the very same series — a 1.4x gap no real portfolio has.

    Feeding daily returns to the bootstrap propagates that noise into the
    horizon distribution, and worse, makes the answer depend on an arbitrary
    knob: the 15y CAGR band ran 14.4 / 11.7 / 10.6 points for daily blocks of
    1 / 21 / 126 days, so the old ``block_days=21`` sat on a steep slope. At
    monthly frequency the noise is already gone and the band is FLAT in the
    block length (10.1 / 10.1 / 10.1 / 10.5 for 1 / 3 / 6 / 12-month blocks),
    i.e. there is no residual short-run dependence left to preserve. That
    stability is why monthly with 1-month blocks is the default.

    The cost is that drawdowns become MONTH-END drawdowns: intra-month troughs
    are invisible, so ``max_drawdown`` here is shallower than a daily-path one
    (median -19.5% vs -27.3% on the same portfolio) and is NOT comparable to the
    realised daily max drawdown in the metrics tables. Month-end is the usual
    convention for published fund drawdowns; treat the two as different
    measures rather than reconciling them.

    ``rf_annual`` is the annualised risk-free (percent) charged in the Sharpe
    numerator. Block resampling shuffles the calendar, so a per-day risk-free
    path cannot be aligned; the window-average rate is the right scalar here
    (``proxy_data.risk_free_annual``). ``None`` falls back to the flat
    ``RISK_FREE_RATE``. ``block_days`` is accepted only for backward
    compatibility and is converted to whole months.
    """
    rf = RISK_FREE_RATE if rf_annual is None else float(rf_annual)
    if block_days is not None:                     # legacy callers
        block_months = max(1, int(round(block_days / 21)))
    r_d = daily_returns(nav)
    if r_d.empty:
        return {}
    # Calendar-month compounding: aggregation is what cancels the timing noise.
    r_m = (1.0 + r_d).resample("ME").prod() - 1.0
    r_m = r_m.dropna()
    horizon_months = max(1, int(round(horizon_days / 21)))
    if len(r_m) < max(block_months * 2, horizon_months // 12 + 12, 24):
        return {}
    rng = np.random.default_rng(seed)
    arr = r_m.values
    n = arr.size
    n_blocks = int(np.ceil(horizon_months / block_months))
    horizon_years = horizon_days / TRADING_DAYS

    cagrs = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    mdds = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, max(1, n - block_months), size=n_blocks)
        path = np.concatenate([arr[s:s + block_months] for s in starts])[:horizon_months]
        price = np.cumprod(1.0 + path)
        total = price[-1] - 1.0
        cagr = (1.0 + total) ** (1.0 / horizon_years) - 1.0
        cagrs[i] = cagr
        vol = path.std() * np.sqrt(12.0)           # monthly → annualised
        # Sharpe in consistent percent units (rf is percent).
        sharpes[i] = (cagr * 100.0 - rf) / (vol * 100.0) if vol > 0 else np.nan
        peak = np.maximum.accumulate(price)
        mdds[i] = (price / peak - 1.0).min()

    def _ci(x: np.ndarray) -> dict:
        a = x[np.isfinite(x)]
        return {
            "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
        }

    return {
        "horizon_years": round(horizon_years, 2),
        "n_sims": n_sims,
        "cagr": _ci(cagrs * 100.0),
        "sharpe": _ci(sharpes),
        "max_drawdown": _ci(mdds * 100.0),
        # P(total return < 0 at the horizon), percent of sims.
        "prob_loss": float((cagrs[np.isfinite(cagrs)] < 0).mean() * 100.0),
    }


# Investor-facing horizons (years) shared by the multi-horizon view.
HORIZON_YEARS = (1, 3, 5, 10, 15)


def multi_horizon(nav: pd.Series, *, horizons=HORIZON_YEARS,
                  rf_annual=None) -> dict[int, dict]:
    """Rolling + Monte-Carlo outcome distributions per investor horizon.

    For each horizon (years) returns {"rolling": rolling_return_distribution,
    "mc": block_bootstrap} computed on the SAME NAV — the single entry point
    for "what does N years in this portfolio look like" questions.
    """
    out: dict[int, dict] = {}
    for yrs in horizons:
        days = int(round(yrs * TRADING_DAYS))
        out[yrs] = {
            "rolling": rolling_return_distribution(nav, days),
            "mc": block_bootstrap(nav, horizon_days=days, rf_annual=rf_annual),
        }
    return out
