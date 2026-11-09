"""Two ways the pre-inception tail used to be pessimistic by construction.

1. A proxy that is itself a fund is already net of its own fees, and the
   backtest then charges the target ETF's TER on top: the tail paid two sets of
   fees (up to ~1.5-2%/yr for the managed-futures funds).
2. Before the first real trend fund (2007) the managed-futures bucket was flat
   cash, so a trend sleeve stacked on equity lost its financing cost for seven
   years, the 2000-02 bear market included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.data import manual_proxies, proxy_data
from tarzan.engine.stats import TRADING_DAYS


def test_fund_proxy_is_grossed_up_by_its_own_expense_ratio(monkeypatch):
    idx = pd.bdate_range("2010-01-01", periods=300)
    monkeypatch.setattr(proxy_data, "_returns",
                        lambda sym: pd.Series(0.0, index=idx) if sym == "VFITX" else pd.Series(dtype=float))
    monkeypatch.setattr(proxy_data, "financing_daily", lambda: pd.Series(0.0, index=idx))
    proxy_data.set_target_currency("USD")
    try:
        out, _ = proxy_data.proxy_returns_for({"Fixed Income Intermediate"})
    finally:
        proxy_data.set_target_currency("EUR")
    r = out["Fixed Income Intermediate"]
    assert abs(r.mean() - proxy_data.PROXY_FUND_ER["VFITX"] / TRADING_DAYS) < 1e-12


def test_index_proxy_is_not_grossed_up():
    r = pd.Series(0.001, index=pd.bdate_range("2010-01-01", periods=5))
    assert proxy_data._gross_up(r, "^SP500TR").equals(r)
    assert proxy_data._gross_up(r, None).equals(r)


def _monthly(n, start, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.008, 0.04, n),
                     index=pd.date_range(start, periods=n, freq="BME"))  # AQR dates are business month-ends


def test_tsmom_tail_reproduces_the_calibrated_factor(monkeypatch):
    """Real funds = cash + 0.6 x factor + 0.1%/month, from 2007. Before 2007 the
    tail keeps the factor's timing at that beta, but its LEVEL is the real funds'
    average: the paper factor's own pre-2007 mean is not trusted."""
    factor = _monthly(420, "1990-01-31", seed=1)      # covers every live day
    levels = manual_proxies._monthly_ret_to_daily_levels(factor)
    days = pd.bdate_range("1990-01-01", "2023-12-29")
    fin = pd.Series(0.0001, index=days)
    live_days = days[days >= "2007-01-01"]
    fp = factor.copy()
    fp.index = fp.index.to_period("M")
    m_ex = (0.001 + 0.6 * fp).reindex(live_days.to_period("M")).values
    n = pd.Series(1, index=live_days).groupby(live_days.to_period("M")).transform("size").values
    trend = pd.Series(((1 + m_ex) ** (1 / n)) * 1.0001 - 1.0, index=live_days)

    monkeypatch.setattr(proxy_data.manual_proxies, "get_series",
                        lambda key: levels if key == "TSMOM" else None)
    monkeypatch.setattr(proxy_data, "_returns",
                        lambda sym: trend if sym == proxy_data._MF_FUND_DEEP else pd.Series(dtype=float))
    monkeypatch.setitem(proxy_data.PROXY_FUND_ER, proxy_data._MF_FUND_DEEP, 0.0)

    s, label, start = proxy_data._alt_series(fin)
    assert "TSMOM" in label and start < pd.Timestamp("1991-01-01")
    tail = s["1995-01-01":"2006-11-30"]
    ex = ((1 + tail) / 1.0001).groupby(tail.index.to_period("M")).prod() - 1
    live = (0.001 + 0.6 * fp)["2007-02":"2023-11"]
    pre = fp["1990-03":"2006-12"]    # the daily level series starts a month in
    want = (0.6 * (pre - pre.mean()) + live.mean()).reindex(ex.index)
    assert np.allclose(ex.values, want.values, atol=1e-9)


def test_tsmom_tail_is_not_smoother_than_the_real_funds(monkeypatch):
    """Spreading a month evenly over its days made the tail ~7x too smooth."""
    factor = _monthly(400, "1990-01-31", seed=2)
    levels = manual_proxies._monthly_ret_to_daily_levels(factor)
    days = pd.bdate_range("1990-01-01", "2023-12-29")
    fin = pd.Series(0.0, index=days)
    live_days = days[days >= "2007-01-01"]
    trend = pd.Series(np.random.default_rng(7).normal(0.0002, 0.006, len(live_days)), index=live_days)
    monkeypatch.setattr(proxy_data.manual_proxies, "get_series",
                        lambda key: levels if key == "TSMOM" else None)
    monkeypatch.setattr(proxy_data, "_returns",
                        lambda sym: trend if sym == proxy_data._MF_FUND_DEEP else pd.Series(dtype=float))
    monkeypatch.setitem(proxy_data.PROXY_FUND_ER, proxy_data._MF_FUND_DEEP, 0.0)
    s, _, _ = proxy_data._alt_series(fin)
    ratio = s[:"2006-12-29"].std() / trend.std()
    assert 0.7 < ratio < 1.4, ratio


def test_without_tsmom_the_tail_stays_cash(monkeypatch):
    days = pd.bdate_range("2000-01-03", "2012-12-31")
    fin = pd.Series(0.0001, index=days)
    trend = pd.Series(0.0005, index=days[days >= "2007-01-01"])
    monkeypatch.setattr(proxy_data.manual_proxies, "get_series", lambda key: None)
    monkeypatch.setattr(proxy_data, "_returns",
                        lambda sym: trend if sym == proxy_data._MF_FUND_DEEP else pd.Series(dtype=float))
    s, label, _ = proxy_data._alt_series(fin)
    assert label.endswith("+/cash")
    assert (s[:"2006-12-29"] == 0.0001).all()
