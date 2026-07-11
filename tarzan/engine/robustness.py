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
    _compute_beta_alpha,
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var,
)


def full_metrics(nav: pd.Series, bench_nav: pd.Series | None = None) -> dict:
    """Full return + risk metric set from a single NAV series (one window).

    CAGR / volatility / Sharpe / Sortino / MaxDrawdown / VaR / CVaR, plus
    Beta / Alpha vs an optional benchmark NAV. All in the percent units the
    stats module uses, so it reads like the dashboard's risk block.
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
    m = {
        "cagr": ann_ret,
        "volatility": ann_vol,
        "sharpe": compute_sharpe(ann_ret, ann_vol),
        "sortino": compute_sortino(daily, ann_ret),
        "max_drawdown": compute_max_drawdown(nav) * 100.0,
        "var_95": compute_var(daily, 0.95) * 100.0,
        "cvar_95": compute_cvar(daily, 0.95) * 100.0,
        "beta": float("nan"),
        "alpha": float("nan"),
    }
    if bench_nav is not None and not bench_nav.empty and len(bench_nav) > 1:
        try:
            beta, alpha = _compute_beta_alpha(nav, bench_nav, ann_ret)
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
        "median": float(np.median(ann)),
        "p95": float(np.percentile(ann, 95)),
        "max": float(ann.max()),
        "pct_positive": float((ann > 0).mean() * 100.0),
    }


def rolling_sharpe_range(nav: pd.Series, window_days: int = 252) -> dict:
    """Min / median / max of the rolling annualised Sharpe ratio."""
    r = daily_returns(nav)
    if len(r) <= window_days:
        return {}
    roll = r.rolling(window_days)
    # RISK_FREE_RATE is in percent, so annualised return/vol are ×100 too.
    ann_ret = roll.mean() * TRADING_DAYS * 100.0
    ann_vol = roll.std() * np.sqrt(TRADING_DAYS) * 100.0
    sharpe = ((ann_ret - RISK_FREE_RATE) / ann_vol).replace([np.inf, -np.inf], np.nan).dropna()
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
                    horizon_days: int = 252, seed: int = 42) -> dict:
    """Stationary-block bootstrap of daily returns → CIs on CAGR / Sharpe /
    max drawdown over ``horizon_days``.

    Blocks preserve short-run autocorrelation (momentum / vol clustering)
    that an IID resample would destroy. Returns 5th/50th/95th percentiles.
    """
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
        # Sharpe in consistent percent units (RISK_FREE_RATE is percent).
        sharpes[i] = (cagr * 100.0 - RISK_FREE_RATE) / (vol * 100.0) if vol > 0 else np.nan
        peak = np.maximum.accumulate(price)
        mdds[i] = (price / peak - 1.0).min()

    def _ci(x: np.ndarray) -> dict:
        a = x[np.isfinite(x)]
        return {
            "p05": float(np.percentile(a, 5)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
        }

    return {
        "horizon_years": round(horizon_years, 2),
        "n_sims": n_sims,
        "cagr": _ci(cagrs * 100.0),
        "sharpe": _ci(sharpes),
        "max_drawdown": _ci(mdds * 100.0),
    }
