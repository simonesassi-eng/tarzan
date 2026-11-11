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


def test_the_ai_note_cannot_flip_a_sign_and_still_pass_as_grounded():
    from tarzan.export.ai_summary import _numbers_are_grounded as grounded

    digest = {"gap_pp": -4.5, "portfolio_pct": 5.1, "beta": 0.92}
    assert grounded("Behind by 4.5pp; portfolio +5.1%; beta 0.92.", digest)
    assert not grounded("Ahead +4.5pp.", digest)
    assert not grounded("Portfolio -5.1%.", digest)
    assert not grounded("Beta 0.7.", digest)


def test_calibrated_splice_does_not_extrapolate_an_insignificant_intercept():
    import numpy as np
    import pandas as pd
    from tarzan.engine.synthetic import calibrated_splice

    idx = pd.bdate_range("2000-01-03", "2025-12-31")
    rng = np.random.default_rng(3)
    long = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    real_idx = idx[idx >= "2023-01-01"]
    # A lucky three-year overlap: +2.5%/yr over the proxy, buried in tracking
    # noise far too large for it to be distinguishable from zero.
    short = long.loc[real_idx] + rng.normal(0.0001, 0.01, len(real_idx))
    out = calibrated_splice(long, short)
    tail = out.loc[out.index < real_idx[0]]
    _slope, intercept = np.polyfit(long.loc[tail.index].to_numpy(), tail.to_numpy(), 1)
    drift = intercept * 252           # the tail is a + b*long: a is the extrapolated drift
    assert abs(drift) < 0.005, drift   # not 23 years of fabricated drift


def test_a_usd_proxy_converted_to_eur_compounds_to_the_true_eur_ratio(monkeypatch):
    import numpy as np
    import pandas as pd
    from tarzan.data import proxy_data

    days = pd.bdate_range("2024-01-01", periods=300)
    holidays = days[::25]                              # the asset is shut, FX is not
    asset_days = days.difference(holidays)
    rng = np.random.default_rng(1)
    usd_px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(asset_days))), index=asset_days)
    eur_per_usd = pd.Series(0.9 * np.cumprod(1 + rng.normal(0, 0.005, len(days))), index=days)
    monkeypatch.setattr(proxy_data, "_eur_per_usd", lambda: eur_per_usd)
    monkeypatch.setattr(proxy_data, "_TARGET_CCY", "EUR")
    eur_ret = proxy_data._usd_returns_to_eur(usd_px.pct_change().dropna())
    got = float((1 + eur_ret).prod())
    eur_px = usd_px * eur_per_usd.reindex(asset_days)
    assert abs(got - float(eur_px.iloc[-1] / eur_px.iloc[0])) < 1e-9
