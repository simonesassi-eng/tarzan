"""Synthetic return construction for long-history robustness.

Pure functions (no I/O): daily-reset leverage with financing + expense
drag, history splicing, and index-proxy replication. Mirrors the model
used by long-history backtesters (leverage applied on a total-return base,
borrowing financed at a short rate plus a spread):

    r_synth = L·r_base − (L−1)·(financing + spread)/252 − expense/252

so volatility/compounding decay of daily-reset leverage is captured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.stats import TRADING_DAYS

# Columns in a factor DataFrame that are estimation CONTROLS (clean market and
# risk-free), not tilt legs applied to the backfill.
_FACTOR_CONTROLS = frozenset({"MKT", "RF"})


def returns_to_price(ret: pd.Series, start: float = 100.0) -> pd.Series:
    """Compound a daily-return series into a price/NAV level series."""
    if ret is None or ret.empty:
        return pd.Series(dtype=float)
    return start * (1.0 + ret).cumprod()


def splice_returns(long_ret: pd.Series, short_ret: pd.Series) -> pd.Series:
    """Use ``short_ret`` where available, ``long_ret`` before its inception.

    The classic proxy-splice: the real instrument's own returns take
    precedence; the longer base/proxy fills the pre-history.
    """
    if short_ret is None or short_ret.empty:
        return long_ret
    if long_ret is None or long_ret.empty:
        return short_ret
    start = short_ret.index.min()
    pre = long_ret.loc[long_ret.index < start]
    return pd.concat([pre, short_ret]).sort_index()


def calibrated_splice(long_ret: pd.Series, short_ret: pd.Series, *,
                      min_overlap: int = 252, beta_bounds=(0.2, 3.0),
                      min_r2: float = 0.10) -> pd.Series:
    """Beta/alpha-calibrated backfill: over the overlap window fit the real
    fund on its proxy basket (``short ≈ a + b·long`` by OLS), then reconstruct
    the pre-inception tail as ``a + b·long`` instead of assuming ``short = long``
    1:1. This corrects a systematic beta/drag mismatch between the fund and its
    proxy composite.

    Falls back to the naive 1:1 splice whenever the calibration is untrustworthy
    (overlap < ``min_overlap`` days, implausible beta, or R² below ``min_r2``),
    so it can never be worse than the classic splice. Note: the reconstructed
    tail omits the regression residual, so its idiosyncratic volatility is a
    lower bound (disclosed in the report).
    """
    if short_ret is None or short_ret.empty:
        return long_ret
    if long_ret is None or long_ret.empty:
        return short_ret
    start = short_ret.index.min()
    pre = long_ret.loc[long_ret.index < start]
    if pre.empty:
        return splice_returns(long_ret, short_ret)
    ov = pd.concat([short_ret.rename("y"), long_ret.rename("x")], axis=1).dropna()
    if len(ov) < min_overlap:
        return splice_returns(long_ret, short_ret)
    x, y = ov["x"].values, ov["y"].values
    vx = x.var()
    if vx <= 0:
        return splice_returns(long_ret, short_ret)
    b = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    a = float(y.mean() - b * x.mean())
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
    if not (beta_bounds[0] <= b <= beta_bounds[1]) or corr * corr < min_r2:
        return splice_returns(long_ret, short_ret)
    backfill = a + b * pre
    return pd.concat([backfill, short_ret]).sort_index()


def factor_loadings(long_ret: pd.Series, short_ret: pd.Series,
                    factors: pd.DataFrame, *, min_overlap_months: int = 24,
                    load_bound: float = 1.5) -> dict:
    """Clipped factor loadings for a fund, estimated on MONTHLY returns.

    Regresses the fund's monthly EXCESS return on ``[MKT-RF, tilt legs]`` (a
    clean market/riskfree control absorbs the market so the tilt loadings —
    SMB/HML/RMW/MOM — come out uncontaminated). MONTHLY (not daily) is
    deliberate: a slow, semi-annually-rebalanced factor ETF (e.g. MSCI Momentum)
    barely co-moves day-to-day with the fast academic factor legs, and a EUR-
    priced global fund vs US-dated factors carries large daily cross-exchange
    timing noise — both wash out the daily loading (momentum came out ~0). At
    monthly frequency the timing noise cancels and the factor exposure is
    recovered cleanly (R² typically ~0.8). Returns ``{factor: loading}`` (empty
    when the overlap is too short). Shared by :func:`factor_splice` and the
    report's simulation map so both describe the SAME discovered tilt."""
    if (short_ret is None or short_ret.empty
            or factors is None or getattr(factors, "empty", True)):
        return {}
    all_cols = list(factors.columns)
    tilt = [c for c in all_cols if c not in _FACTOR_CONTROLS]
    if not tilt:
        return {}

    def _monthly(s):
        return (1.0 + s).resample("ME").prod() - 1.0

    cols = {"y": _monthly(short_ret)}
    for c in all_cols:
        cols[c] = _monthly(factors[c])
    ov = pd.DataFrame(cols).dropna()
    if len(ov) < min_overlap_months:
        return {}
    rf = ov["RF"].values if "RF" in ov else np.zeros(len(ov))
    Y = ov["y"].values - rf
    reg = (["MKT"] if "MKT" in ov else []) + tilt
    X = np.column_stack([np.ones(len(ov))] + [ov[c].values for c in reg])
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return {}
    off = 1 + (1 if "MKT" in ov else 0)          # skip intercept (+ market beta)
    clipped = np.clip(beta[off:off + len(tilt)], -load_bound, load_bound)
    return {c: float(clipped[i]) for i, c in enumerate(tilt)}


def factor_splice(long_ret: pd.Series, short_ret: pd.Series, factors: pd.DataFrame,
                  *, load_bound: float = 1.5) -> pd.Series:
    """Factor-AWARE backfill for factor ETFs (value / momentum / quality / size).

    The tilt loadings are estimated on MONTHLY returns (see
    :func:`factor_loadings`) with a clean market control, then the pre-inception
    tail is reconstructed as ``long + Σ loadingᵢ·factor_legᵢ`` — the geo/market
    base (right regional level, coefficient 1) PLUS the discovered factor tilt.
    No intercept or market beta is extrapolated (the base carries the market),
    and loadings are clipped to ``±load_bound`` to reject overfit extremes.

    Missing factor observations in the pre-inception grid are treated as a zero
    factor return that day (keeps the full base grid, no truncation). Falls back
    to :func:`calibrated_splice` (then the naive splice) when the overlap is too
    short or the factor data is unavailable, so it is never worse than the
    calibrated backfill. The reconstructed tail omits the regression residual,
    so its idiosyncratic volatility is a lower bound (disclosed in the report).
    """
    if short_ret is None or short_ret.empty:
        return long_ret
    if long_ret is None or long_ret.empty:
        return short_ret
    if factors is None or getattr(factors, "empty", True):
        return calibrated_splice(long_ret, short_ret)
    start = short_ret.index.min()
    pre = long_ret.loc[long_ret.index < start]
    if pre.empty:
        return splice_returns(long_ret, short_ret)
    loadings = factor_loadings(long_ret, short_ret, factors, load_bound=load_bound)
    if not loadings:
        return calibrated_splice(long_ret, short_ret)
    # Reconstruct the pre-inception tail: geo/market base + discovered tilt.
    fac_pre = factors.reindex(pre.index).fillna(0.0)
    tilt = pd.Series(0.0, index=pre.index)
    for c, load in loadings.items():
        tilt = tilt + load * fac_pre[c]
    backfill = pre + tilt
    return pd.concat([backfill, short_ret]).sort_index()


def replicate_portfolio_returns(exposures: dict, proxy_returns: dict, *,
                                financing_daily=None,
                                spread_annual: float = 0.0) -> pd.Series:
    """Synthetic daily returns from exposure weights × proxy index returns.

    ``exposures`` maps a proxy key → exposure as a fraction of capital
    (may sum to > 1 with leverage). ``proxy_returns`` maps the same keys →
    daily-return Series. The financed portion (gross − 1) is charged the
    financing rate + spread, so the leverage carries a realistic cost.
    Computed over the window common to all used proxies.
    """
    keys = [k for k in exposures
            if k in proxy_returns and proxy_returns[k] is not None
            and not proxy_returns[k].empty and abs(exposures[k]) > 1e-9]
    if not keys:
        return pd.Series(dtype=float)
    df = pd.DataFrame({k: proxy_returns[k] for k in keys}).dropna(how="any")
    if df.empty:
        return pd.Series(dtype=float)
    w = np.array([exposures[k] for k in keys])
    s = pd.Series(df.values.dot(w), index=df.index)
    gross = float(w.sum())
    if gross > 1.0:
        if financing_daily is None:
            fin = 0.0
        elif np.isscalar(financing_daily):
            fin = float(financing_daily)
        else:
            fin = financing_daily.reindex(df.index).ffill().fillna(0.0)
        s = s - (gross - 1.0) * (fin + spread_annual / TRADING_DAYS)
    return s.dropna()


# ---------------------------------------------------------------------------
# Rebalancing / NAV construction (reusable for ANY portfolio, not just what-if)
# ---------------------------------------------------------------------------

def period_key(ts, freq: str) -> int:
    """Rebalance-period bucket id for a timestamp. A new id → a rebalance at the
    first trading day of that period. ``none`` → one bucket (buy & hold).

    Shared by :func:`combine_returns` and any caller that wants to know which
    rebalance period a date falls in (e.g. a real-portfolio backtest that
    reuses the same policy as the what-if engine).
    """
    y, m = ts.year, ts.month
    if freq == "monthly":
        return y * 100 + m
    if freq == "quarterly":
        return y * 10 + (m - 1) // 3
    if freq == "semiannual":
        return y * 10 + (m - 1) // 6
    if freq == "annual":
        return y
    return 0  # none / buy-and-hold


def combine_returns(df: pd.DataFrame, w: pd.Series, rebalance: str) -> pd.Series:
    """Combine per-instrument daily returns into a portfolio series under a
    given REBALANCE policy.

    ``df`` is a per-instrument daily-return frame (one column per sleeve),
    ``w`` the target weights aligned to ``df.columns`` (need not sum to 1 —
    they are normalised here). ``rebalance`` is one of ``daily`` / ``monthly``
    / ``quarterly`` / ``semiannual`` / ``annual`` / ``none``.

    ``daily`` keeps constant target weights every day (continuous rebalancing).
    Any periodic policy or ``none`` (buy & hold) lets the weights DRIFT with
    each holding's cumulative return between rebalance dates, resetting to
    targets at each period boundary — the realistic, testfol-style model.
    Rebalancing itself is costless (ETF, no commissions); fees/TER are charged
    separately on the base by the caller.
    """
    wv = w.values.astype(float)
    wv = wv / wv.sum() if wv.sum() else wv          # proportional target weights
    if rebalance == "daily":
        return df.mul(wv, axis=1).sum(axis=1)
    R = df.values
    keys = [period_key(ts, rebalance) for ts in df.index]
    out = np.empty(len(R))
    holdings = wv.copy()                              # value per sleeve, Σ = 1
    prev = keys[0] if keys else None
    for t in range(len(R)):
        if keys[t] != prev:                          # period boundary → rebalance
            holdings = wv * holdings.sum()
            prev = keys[t]
        v0 = holdings.sum()
        holdings = holdings * (1.0 + R[t])
        out[t] = holdings.sum() / v0 - 1.0 if v0 else 0.0
    return pd.Series(out, index=df.index)
