"""A young target sleeve is reached back with a MODELED stretch, never altered."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan import config as cfg
from tarzan.engine import target_backfill

US, EU = "US56164V3033", "IE000BS31R00"
_CSV = """name,ticker,isin,kind,asset_class,role,watchlist,is_benchmark_alpha_beta,is_benchmark_geo,usa,emerging_markets,eurozone_emu,japan,dev_ex_usa_ex_emu_ex_jp,exp_equities,exp_fixed_income,exp_gold,exp_commodities,exp_alternative,exp_crypto,ter,description,updated
Man Active Trend Enhanced,MATE,US56164V3033,etf,Equities,Trend Stacked,True,,,100.0,0.0,0.0,0.0,0.0,100.0,,,,100.0,,0.0097,US fund,1/1/26
Man Active Trend Enhanced UCITS ETF EUR Hedged,MATE,IE000BS31R00,etf,Equities,Trend Stacked,True,,,72.0,0.0,9.0,6.0,13.0,100.0,,,,100.0,,0.0096,UCITS,1/1/26
iMGP DBi Managed Futures R EUR,DBMFE,LU2951555403,etf,Alternative,Managed Futures,True,,,,,,,,,,,,,,0.0095,EUR class,1/1/26
iMGP DBi Managed Futures (EUR Hedged),MFEH,LU3359622902,etf,Alternative,Managed Futures,True,,,,,,,,,,,,,,0.0095,EUR hedged class,1/1/26
"""


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    path = tmp_path / "t.csv"
    path.write_text(_CSV)
    monkeypatch.setenv("TARZAN_TAXONOMY_PATH", str(path))
    cfg.reset_input_caches()
    yield
    cfg.reset_input_caches()


def _tape(start, n, seed, scale=100.0):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(scale * np.cumprod(1 + rng.normal(0.0004, 0.01, n)), index=idx)


def test_a_same_named_class_with_another_composition_is_no_twin(catalog, monkeypatch):
    """The two Man Active Trend Enhanced classes share a name, not a portfolio
    (S&P + ~0.84x AHL trend vs MSCI World + ~0.65x): the US tape must never
    stand in for the UCITS, which goes to the composition model instead."""
    from tarzan.data import proxy_data

    sleeve = SimpleNamespace(ticker="MATE.MI", isin=EU, target_portfolio=6.0,
                             price_history=_tape("2026-11-02", 8, 2, scale=10.0))
    twin_native = _tape("2025-01-01", 480, 1)
    record = SimpleNamespace(isin=US, currency="USD", history=twin_native * 0.9,
                             history_native=twin_native)
    monkeypatch.setattr(proxy_data, "currency_hedge_legs",
                        lambda: (pd.Series(dtype=float), pd.Series(dtype=float)))
    monkeypatch.setattr(target_backfill, "_reconstructed_returns", lambda h: (None, ""))
    out, note = target_backfill.extend(sleeve, pd.Timestamp("2025-06-02"), {"MATE": record})
    assert note is None                                  # no model, and no borrowed twin
    pd.testing.assert_series_equal(out, sleeve.price_history, check_names=False)
    row = cfg.catalog().row(EU, None)
    assert target_backfill.twin_rows(row) == []


def test_a_sleeve_that_already_reaches_back_is_untouched(catalog):
    sleeve = SimpleNamespace(ticker="MATE", isin=US, target_portfolio=6.0,
                             price_history=_tape("2025-01-01", 300, 3))
    out, note = target_backfill.extend(sleeve, pd.Timestamp("2025-01-03"), {})
    assert note is None
    pd.testing.assert_series_equal(out, sleeve.price_history, check_names=False)


def test_the_modeled_stretch_of_a_line_is_drawn_dashed_and_meets_the_real_one():
    from tarzan.export._charts import chart_pct_compact

    dates = list(pd.bdate_range("2026-01-01", periods=10))
    svg = chart_pct_compact([{"values": [float(i) for i in range(10)], "color": "#f0f",
                              "dash_until": 4}], dates)
    lines = [l for l in svg.split("<polyline") if 'stroke="#f0f"' in l]
    assert len(lines) == 2
    dashed, solid = lines
    assert 'stroke-dasharray="3,3"' in dashed and "stroke-dasharray" not in solid
    pts = lambda l: l.split('points="')[1].split('"')[0].split()
    assert pts(dashed)[-1] == pts(solid)[0]      # the two pieces meet


def test_a_unhedged_eur_twin_has_its_currency_move_removed_for_a_hedged_sleeve(catalog, monkeypatch):
    from tarzan.data import proxy_data

    days = pd.bdate_range("2025-01-01", periods=480)
    rng = np.random.default_rng(5)
    usd_strategy = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.008, len(days))), index=days)
    eur_per_usd = pd.Series(0.9 * np.cumprod(1 + rng.normal(0, 0.005, len(days))), index=days)
    dbmfe = SimpleNamespace(isin="LU2951555403", price_currency="EUR",
                            price_history=usd_strategy * eur_per_usd, price_history_native=None)
    carry = pd.Series(0.0, index=days)
    monkeypatch.setattr(proxy_data, "currency_hedge_legs",
                        lambda: (eur_per_usd.pct_change().fillna(0.0), carry))
    mfeh = SimpleNamespace(ticker="MFEH.DE", isin="LU3359622902", target_portfolio=7.5,
                           price_history=_tape("2026-10-01", 20, 6, scale=50.0))
    out, note = target_backfill.extend(mfeh, pd.Timestamp("2025-03-03"), {}, [dbmfe])
    assert "twin fund DBMFE" in note["source"] and "currency move removed" in note["source"]
    modeled = out.loc[(out.index >= "2025-03-04") & (out.index < mfeh.price_history.index[0])]
    got = modeled.pct_change().dropna()
    expect = usd_strategy.pct_change().reindex(got.index)   # the dollar strategy, hedged
    assert np.allclose(got.to_numpy(), expect.to_numpy(), atol=1e-10)


def test_the_man_trend_leg_runs_on_the_ahl_program_at_each_funds_scale(monkeypatch):
    """Both Man Active Trend Enhanced classes take AHLT (AHL Trend Core) for
    their trend leg; before AHLT the generic trend proxy is rescaled to AHLT's
    volatility, so a fund's scale means one thing across the splice."""
    from types import SimpleNamespace
    from tarzan.backtest import engine as E
    from tarzan.backtest.model import WhatIfItem, EQ, ALT
    from tarzan.data import proxy_data as P

    h = SimpleNamespace(ter=0.96, geo_breakdown=None, role="Trend Stacked")
    ucits = WhatIfItem("MATE", "MATE.MI", "IE000BS31R00", 6.0, h, {EQ: 100.0, ALT: 65.0}, {}, "")
    other = WhatIfItem("CTA", "CTA", "US82889N6994", 6.0, h, {ALT: 100.0}, {}, "")
    assert E.instrument_exposures(ucits)["Alternative:AHL"] == pytest.approx(0.65)
    assert "Alternative" in E.instrument_exposures(other)            # generic trend

    days = pd.bdate_range("2020-01-01", "2026-11-13")
    rng = np.random.default_rng(4)
    generic = pd.Series(rng.normal(0, 0.006, len(days)), index=days)      # synthetic
    ahlt = pd.Series(rng.normal(0, 0.009, len(days)), index=days).loc["2023-08-30":]
    monkeypatch.setattr(P, "_alt_series", lambda fin: (generic, "generic", days[0]))
    monkeypatch.setattr(P, "_returns", lambda sym: ahlt)
    series, _label, _start = P._ahl_trend_series(None)
    before = series.loc[series.index < ahlt.index[0]]
    assert before.std() == pytest.approx(ahlt.std() * 1.0, rel=0.06)    # vol-matched tail
    assert (series.loc[ahlt.index] - (ahlt + 0.0096 / 252)).abs().max() < 1e-12   # AHLT, fee grossed up
