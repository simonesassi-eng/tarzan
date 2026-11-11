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


def test_the_twin_fund_reaches_a_young_sleeve_back_and_the_real_tape_is_kept(catalog, monkeypatch):
    from tarzan.data import proxy_data

    twin_native = _tape("2025-01-01", 480, 1)
    sleeve = SimpleNamespace(ticker="MATE.MI", isin=EU, target_portfolio=6.0,
                             price_history=_tape("2026-11-02", 8, 2, scale=10.0))
    record = SimpleNamespace(isin=US, currency="USD", history=twin_native * 0.9,
                             history_native=twin_native)
    carry = pd.Series(0.0001, index=pd.bdate_range("2024-01-01", "2027-01-01"))
    monkeypatch.setattr(proxy_data, "currency_hedge_legs", lambda: (None, carry))
    start = pd.Timestamp("2025-06-02")
    out, note = target_backfill.extend(sleeve, start, {"MATE": record})

    real = sleeve.price_history
    pd.testing.assert_series_equal(out.loc[real.index], real, check_names=False)
    assert out.index[0] <= start + pd.Timedelta(days=7)
    assert note["real_from"] == real.index[0] and "twin fund MATE" in note["source"]
    assert "EUR-hedged" in note["source"]
    # The modeled stretch moves like the twin's native tape plus the carry.
    modeled = out.loc[out.index < real.index[0]].pct_change().dropna()
    expect = ((1 + twin_native.pct_change()) * (1 + 0.0001) - 1).reindex(modeled.index)
    assert np.allclose(modeled.to_numpy(), expect.to_numpy(), atol=1e-12)


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
