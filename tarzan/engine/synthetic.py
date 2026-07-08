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


def leveraged_daily_returns(base_ret: pd.Series, leverage: float, *,
                            expense_annual: float = 0.0,
                            financing_daily=None,
                            spread_annual: float = 0.0) -> pd.Series:
    """Daily returns of a synthetic daily-reset leveraged version of ``base_ret``.

    ``financing_daily`` may be a scalar daily rate, an aligned Series, or
    None (then only the spread is charged on the borrowed portion).
    """
    if base_ret is None or base_ret.empty:
        return pd.Series(dtype=float)
    exp = expense_annual / TRADING_DAYS
    lev = leverage * base_ret
    if leverage > 1.0:
        if financing_daily is None:
            fin = 0.0
        elif np.isscalar(financing_daily):
            fin = float(financing_daily)
        else:
            fin = financing_daily.reindex(base_ret.index).ffill().fillna(0.0)
        borrow = (leverage - 1.0) * (fin + spread_annual / TRADING_DAYS)
        return lev - borrow - exp
    return lev - exp


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
