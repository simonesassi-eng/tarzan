"""Guards added after the 2026-11 sanity audit: each pins one defect it found."""

from __future__ import annotations


def test_an_unconfirmed_valuation_far_off_the_last_close_is_refused():
    from types import SimpleNamespace
    from tarzan.data import current_session

    h = SimpleNamespace(ticker="SYN.MI", current_price=160.0, quantity=10.0, current_value=1600.0,
                        price_is_fallback=False, data_source="yfinance:regularMarketPrice")
    import tarzan.instruments.valuation as val
    orig = val.select_holding_valuation
    val.select_holding_valuation = lambda holding: "stub"
    try:
        current_session._refuse_incoherent_valuation(h, 100.0)
        assert (h.current_price, h.current_value, h.price_is_fallback) == (100.0, 1000.0, True)
        h.current_price = 104.0
        current_session._refuse_incoherent_valuation(h, 100.0)
        assert h.current_price == 104.0                  # within tolerance: kept
    finally:
        val.select_holding_valuation = orig


def test_alpha_beta_regress_the_benchmark_currency_tape_not_the_native_one():
    import numpy as np
    import pandas as pd
    from tarzan.engine.benchmarks import _populate_perf_row

    idx = pd.bdate_range("2023-01-02", periods=600)
    rng = np.random.default_rng(7)
    eur = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx))), index=idx)
    fx = pd.Series(np.cumprod(1 + rng.normal(0, 0.006, len(idx))), index=idx)
    usd = eur * fx                                      # the same fund, its USD listing
    native_row, fixed_row = {}, {}
    _populate_perf_row(native_row, usd, eur, 0.0)
    _populate_perf_row(fixed_row, usd, eur, 0.0, beta_series=eur)
    assert abs(fixed_row["beta"] - 1.0) < 1e-6 and abs(fixed_row["alpha"]) < 1e-6
    assert abs(native_row["beta"] - 1.0) > 0.02       # what it used to print
    assert fixed_row["1y"] == native_row["1y"]         # returns stay on the own tape


def test_a_buy_only_plan_never_opens_a_position_below_five_percent():
    from tarzan.engine import rebalancer as rb
    from tarzan.models.holding import AssetClass, Holding
    from tarzan.models.investor_config import InvestorConfig

    def _h(ticker, value, target):
        return Holding(isin=f"ID:{ticker}", ticker=ticker, quantity=value / 10.0 if value else 0.0,
                       cost_basis_eur=value, market_value_eur=value, currency="EUR", name=ticker,
                       current_price=10.0, current_value=value, security_type="ETF",
                       asset_class=AssetClass.EQUITIES, gain_pct=0.0, target_portfolio=target)

    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {"Equities": 100.0}
    config.equity_geo_targets_pctg = {}
    config.target_cash_buffer_eur = 0.0
    config.rebalancing_no_sell = True
    config.target_use_per_holding_only = True
    held, seed = _h("HELD.MI", 20600.0, 90.0), _h("SEED.MI", 0.0, 10.0)
    actions, _ = rb.optimize_local_search([held, seed], config, 20600.0, lump_sum=600.0,
                                          params=rb.LSParams(n_restarts=2))
    opened = [a for a in actions if "SEED" in str(a.get("ticker") or a.get("name") or "")]
    assert not opened, actions
