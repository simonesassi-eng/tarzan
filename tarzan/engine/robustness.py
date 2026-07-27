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


def full_metrics(nav: pd.Series, bench_nav: pd.Series | None = None,
                 risk_free: float | None = None, rf_daily=None) -> dict:
    """Full return + risk metric set from a single NAV series (one window).

    CAGR / volatility / Sharpe / Sortino / MaxDrawdown / VaR / CVaR, plus
    Beta / Alpha vs an optional benchmark NAV. All in the percent units the
    stats module uses, so it reads like the dashboard's risk block.

    When ``rf_daily`` (a daily risk-free path) is supplied, Sharpe and Sortino
    use the TIME-VARYING excess-return form (r_t − rf_t); otherwise they fall
    back to the scalar ``risk_free`` (window-average) form.
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
    ann_vol = float(daily.std()) * np.sqrt(TRADING_DAYS) * 100.0  # percent
    if rf_daily is not None:
        sharpe = compute_sharpe_tv(daily, rf_daily)
        sortino = compute_sortino_tv(daily, rf_daily)
    else:
        sharpe = compute_sharpe(ann_ret, ann_vol, risk_free=risk_free)
        sortino = compute_sortino(daily, ann_ret, risk_free=risk_free)
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

def block_bootstrap(nav: pd.Series, *, n_sims: int = 2000, block_days: int = 21,
                    horizon_days: int = 252, seed: int = 42, rf_annual=None) -> dict:
    """Stationary-block bootstrap of daily returns → CIs on CAGR / Sharpe /
    max drawdown over ``horizon_days``.

    Blocks preserve short-run autocorrelation (momentum / vol clustering)
    that an IID resample would destroy. Returns 5th/50th/95th percentiles.

    ``rf_annual`` is the annualised risk-free (percent) charged in the Sharpe
    numerator. Block resampling shuffles the calendar, so a per-day risk-free
    path cannot be aligned; the window-average real rate is the right scalar
    here (``proxy_data.risk_free_annual``). ``None`` falls back to the flat
    ``RISK_FREE_RATE``.
    """
    rf = RISK_FREE_RATE if rf_annual is None else float(rf_annual)
    r = daily_returns(nav)
    if len(r) < max(block_days * 2, 60):
        return {}
    rng = np.random.default_rng(seed)
    arr = r.values
    n = arr.size
    n_blocks = int(np.ceil(horizon_days / block_days))
    horizon_years = horizon_days / TRADING_DAYS

    cagrs = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    mdds = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, n - block_days, size=n_blocks)
        path = np.concatenate([arr[s:s + block_days] for s in starts])[:horizon_days]
        price = np.cumprod(1.0 + path)
        total = price[-1] - 1.0
        cagr = (1.0 + total) ** (1.0 / horizon_years) - 1.0
        cagrs[i] = cagr
        vol = path.std() * np.sqrt(TRADING_DAYS)
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
