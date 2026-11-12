"""One reconstruction for every simulated sleeve (docs/BACKTEST_METHOD.md).

The what-if backtest and the newsletter's target line used to rebuild a fund
from its composition with two copies of the same code, and only one of them
reached back on a twin share class. Both now go through
``instrument_base_returns``; these tests pin the currency rule it owns.
"""

import numpy as np
import pandas as pd

from tarzan.engine import synthetic as syn


def _flat(value, idx):
    return pd.Series(value, index=idx)


def test_futures_overlay_carries_currency_on_nav_not_on_gross():
    """90/60 fund, proxies already converted to EUR: assets flat in USD, the
    dollar +1%. Converted proxies then read +1% on 150% of exposure, but a
    future's notional is never converted, so the fund gains +1% on its NAV."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    fx = _flat(0.01, idx)
    proxies = {"Equity": _flat(0.01, idx), "Bonds": _flat(0.01, idx)}
    exposures = {"Equity": 0.9, "Bonds": 0.6}
    swap = syn.replicate_portfolio_returns(exposures, proxies)
    overlay = syn.replicate_portfolio_returns(exposures, proxies, financed_fx=fx)
    assert np.allclose(swap.iloc[1:], 0.015)       # currency levered 1.5x
    assert np.allclose(overlay.iloc[1:], (1.015 / 1.005) - 1)
    assert abs(overlay.iloc[1:].mean() - 0.01) < 1e-4


def test_hedged_stacked_fund_keeps_no_currency_after_its_hedge():
    """100/100 EUR-hedged class: the overlay rule leaves one unit of currency
    and the hedge removes it, instead of leaving a full unhedged leg."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    fx = _flat(0.01, idx)
    proxies = {"Equity": _flat(0.01, idx), "Trend": _flat(0.01, idx)}
    out = syn.replicate_portfolio_returns(
        {"Equity": 1.0, "Trend": 1.0}, proxies,
        hedge_fx=fx, hedge_carry=_flat(0.0, idx), financed_fx=fx)
    assert np.allclose(out.iloc[1:], 0.0, atol=2e-4)   # second-order only; one unhedged leg would be 1e-2


def test_target_line_uses_the_backtest_reconstruction(monkeypatch):
    from tarzan.backtest import engine
    from tarzan.engine import target_backfill
    from tarzan.models.holding import Holding

    from tarzan.data import proxy_data

    calls = []
    monkeypatch.setattr(proxy_data, "proxy_returns_for", lambda keys: ({}, None))
    monkeypatch.setattr(proxy_data, "currency_hedge_legs", lambda: (None, None))
    monkeypatch.setattr(engine, "instrument_base_returns",
                        lambda *a, **k: calls.append(a) or pd.Series(dtype=float))
    h = Holding(isin="IE00077IIPQ8", ticker="NTSG", quantity=1.0,
                cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR")
    target_backfill._reconstructed_returns(h)
    assert calls, "target line rebuilt the sleeve without the shared reconstruction"
