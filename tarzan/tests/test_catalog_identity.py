"""One instrument, one identity: the ISIN; a symbol only where it is unambiguous.

The curated catalog may list two different funds under one symbol: MATE is a
Nasdaq fund (US56164V3033) and the Milan symbol of a UCITS (IE000BS31R00). Each
test below is a path that once keyed on the symbol and so could hand one fund
the other's name, class, fees, geography, listing or price series.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan import config as cfg
from tarzan.models.instrument_key import (
    instrument_key, is_isin, normalize_ticker, qualify, split_qualified,
)

US, EU = "US56164V3033", "IE000BS31R00"

_CSV = """name,ticker,isin,kind,asset_class,role,watchlist,is_benchmark_alpha_beta,is_benchmark_geo,usa,emerging_markets,eurozone_emu,japan,dev_ex_usa_ex_emu_ex_jp,exp_equities,exp_fixed_income,exp_gold,exp_commodities,exp_alternative,exp_crypto,ter,description,updated
Man Active Trend Enhanced,MATE,US56164V3033,etf,Equities,Efficient Core,True,,,100.0,0.0,0.0,0.0,0.0,100.0,,,,100.0,,0.0097,US fund,1/1/26
Man Active Trend Enhanced UCITS ETF EUR Hedged,MATE,IE000BS31R00,etf,Equities,Equity Trend Stacked,True,,,72.0,0.0,9.0,6.0,13.0,100.0,,,,100.0,,0.0096,UCITS,1/1/26
iShares MSCI ACWI,ISAC,IE00B6R52259,etf,Equities,Equity Broad,True,True,True,63.0,11.0,8.0,5.0,13.0,,,,,,,0.002,ACWI,1/1/26
Amundi MSCI USA (2x) Leveraged,CL2,,etf,Equities,Equity Leveraged,True,,,100.0,0.0,0.0,0.0,0.0,200.0,,,,,,0.005,no ISIN row,1/1/26
"""


@pytest.fixture
def two_mates(tmp_path, monkeypatch):
    path = tmp_path / "instrument_taxonomy.csv"
    path.write_text(_CSV)
    monkeypatch.setenv("TARZAN_TAXONOMY_PATH", str(path))
    from tarzan.data import geo_resolver
    monkeypatch.setattr(geo_resolver, "_ASSET_GEO_DF", None)
    cfg.reset_input_caches()
    yield cfg.catalog()
    cfg.reset_input_caches()
    monkeypatch.setattr(geo_resolver, "_ASSET_GEO_DF", None)


# --- the identity helpers ---------------------------------------------------------

def test_a_qualified_symbol_reads_as_its_symbol_and_its_isin():
    assert qualify("MATE", EU) == f"MATE@{EU}"
    assert split_qualified(f"MATE@{EU.lower()}") == ("MATE", EU)
    assert split_qualified("MATE@not-an-isin") == ("MATE", "")
    assert normalize_ticker(f"MATE@{EU}") == "MATE"          # the reader sees MATE
    assert instrument_key(None, f"MATE@{EU}") == EU          # the code keys the fund
    assert is_isin(US) and not is_isin("MATE")


# --- the catalog index --------------------------------------------------------------

def test_a_shared_symbol_is_keyed_per_fund_and_resolves_to_neither_alone(two_mates):
    index = two_mates
    assert index.shared == {"MATE"}
    keys = {index.isin_of(r): index.key(r) for r in index.rows if index.isin_of(r)}
    assert keys[US] == f"MATE@{US}" and keys[EU] == f"MATE@{EU}"
    assert index.key(index.row(None, "ISAC")) == "ISAC"       # an unshared symbol stays plain
    assert index.row(None, "MATE") is None
    assert index.row(None, "MATE.MI") is None
    assert index.row(EU, "MATE")["role"] == "Equity Trend Stacked"
    assert index.row(None, f"MATE@{US}")["role"] == "Efficient Core"


def test_an_isin_the_catalog_does_not_know_never_borrows_a_same_symbol_row(two_mates):
    assert two_mates.row("XS0000000001", "MATE") is None
    assert two_mates.row("XS0000000001", "ISAC") is None      # ISAC is another ISIN
    # A row the catalog lists WITHOUT an ISIN may still be reached by symbol.
    assert two_mates.row("FR0010755611", "CL2")["ticker"] == "CL2"


def test_the_curated_name_breaks_a_tie_between_two_funds(two_mates):
    row = two_mates.row(None, "MATE", "Man Active Trend Enhanced UCITS ETF EUR Hedged")
    assert two_mates.isin_of(row) == EU


# --- every catalog map --------------------------------------------------------------

@pytest.mark.parametrize("isin, role, ter, eu_share", [(US, "Efficient Core", 0.0097, 0.0),
                                                      (EU, "Equity Trend Stacked", 0.0096, 9.0)])
def test_each_fund_gets_its_own_row_in_every_map(two_mates, isin, role, ter, eu_share):
    from tarzan.data.geo_resolver import _lookup_asset_geo

    assert cfg.catalog_lookup(cfg.instrument_taxonomy(), isin, "MATE")[1] == role
    assert cfg.ter_for(isin, "MATE") == pytest.approx(ter)
    assert cfg.kind_for(isin, "MATE") == "ETF"
    assert cfg.class_breakdown_for(isin, "MATE", "Equities").get("Alternative") == 100.0
    assert cfg.resolve_taxonomy_identity(isin, "") == (isin, "MATE")   # a plain provider hint
    geo, _ = _lookup_asset_geo(isin, "MATE")
    shares = {getattr(k, "value", str(k)): v for k, v in geo.items()}
    assert shares.get("Eurozone EMU", 0.0) == pytest.approx(eu_share)


def test_the_bare_shared_symbol_reads_nothing_from_any_map(two_mates):
    from tarzan.data.geo_resolver import _lookup_asset_geo

    assert cfg.catalog_lookup(cfg.instrument_taxonomy(), None, "MATE") is None
    assert cfg.name_for(None, "MATE") is None
    assert cfg.ter_for(None, "MATE.MI") is None
    assert cfg.kind_for(None, "MATE") is None
    assert cfg.resolve_taxonomy_identity("", "MATE") == ("", "MATE")   # unchanged, not guessed
    assert _lookup_asset_geo("", "MATE") is None
    assert "MATE" not in cfg.instrument_taxonomy()           # never a dict key either
    assert cfg.instrument_taxonomy_has("MATE")               # ...yet still a curated symbol


def test_a_prepared_reference_keeps_the_catalog_it_was_prepared_from(two_mates):
    from tarzan.contracts.reference import ReportReference

    reference = ReportReference.prepare()
    cfg.reset_input_caches()                                 # the global catalog moves on
    assert reference.name_for(EU, "MATE.MI") == "Man Active Trend Enhanced UCITS ETF EUR Hedged"
    assert reference.name_for(None, "MATE") is None


# --- the paths that keyed on the symbol ----------------------------------------------

def test_the_two_funds_are_fetched_under_two_keys_with_their_own_isin(two_mates, monkeypatch):
    from tarzan.data import enricher
    from tarzan.engine import benchmarks

    keys = cfg.benchmarks()
    assert keys["Man Active Trend Enhanced"] == f"MATE@{US}"
    assert keys["Man Active Trend Enhanced UCITS ETF EUR Hedged"] == f"MATE@{EU}"
    asked = []
    monkeypatch.setattr(enricher, "_fetch_benchmark_data",
                        lambda symbol, **kw: asked.append((symbol, kw["isin"], kw["expected_name"])) or {})
    for key in keys.values():
        benchmarks._fetch_benchmark_uncached(key)
    assert ("MATE", EU, "Man Active Trend Enhanced UCITS ETF EUR Hedged") in asked
    assert ("MATE", US, "Man Active Trend Enhanced") in asked
    assert not any("@" in symbol for symbol, _i, _n in asked)   # never a key to the provider


def test_the_plan_keys_the_ucits_by_fund_and_still_merges_share_classes(two_mates):
    from tarzan.engine.target_derivation import derive_class_targets, plan_weights

    plan, notes = plan_weights({
        "a": {"isin": EU, "ticker": "MATE", "target_portfolio": 6.0},
        "b": {"isin": "FR0010755611", "ticker": "CL2", "target_portfolio": 8.0},
        "c": {"isin": "FR0011000000", "ticker": "CL2", "target_portfolio": 8.0},
    })
    assert plan == {f"MATE@{EU}": 6.0, "CL2": 8.0} and not notes
    classes, missing = derive_class_targets(plan)
    assert not missing and classes["Alternative"] == pytest.approx(6.0)


def test_the_backtest_does_not_hand_a_ticker_only_row_either_isin(two_mates, tmp_path, monkeypatch):
    from tarzan.backtest import loader

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "instrument_taxonomy.csv").write_text(_CSV)
    monkeypatch.setattr(loader, "ROOT", tmp_path)
    sym_map = loader.build_symbol_map()
    assert sym_map["MATE"] == (None, "MATE")
    assert sym_map["ISAC"][0] == "IE00B6R52259"
    eu = loader._taxonomy_geo("MATE", EU)
    assert eu is not None and loader._taxonomy_geo("MATE", "") is None


def test_a_bare_symbol_on_the_provider_is_the_us_listing(two_mates):
    from tarzan.data import market_quotes

    assert market_quotes._exchange_for("MATE") == "US"
    assert market_quotes._exchange_for("MATE.MI") == "EU"


def test_the_real_catalog_keys_every_row_once_and_finds_each_by_its_isin():
    cfg.reset_input_caches()
    index = cfg.catalog()
    keys = [index.key(r) for r in index.rows]
    assert len(keys) == len(set(keys)), [k for k in keys if keys.count(k) > 1]
    for row in index.rows:
        isin = index.isin_of(row)
        if isin:
            assert index.row(isin, None) is row
    assert len(set(cfg.benchmarks().values())) == len(cfg.benchmarks())


def test_a_target_whose_isin_contradicts_its_symbol_borrows_nothing(two_mates):
    from tarzan.engine.target_derivation import derive_class_targets, plan_weights

    plan, _ = plan_weights({"a": {"isin": "US0378331005", "ticker": "ISAC",
                                  "target_portfolio": 10.0}})
    assert plan == {"US0378331005": 10.0}               # not ISAC's row
    classes, missing = derive_class_targets(plan)
    assert not classes and missing


def test_a_placeholder_isin_is_no_identity():
    from tarzan.models.instrument_key import normalize_isin

    for junk in ("N/A", "0", "NONE", "nan", "ISAC", "", None):
        assert normalize_isin(junk) == ""
    assert normalize_isin(" ie000bs31r00 ") == EU


def test_the_learned_ticker_isin_map_keeps_each_listing_apart(two_mates, tmp_path, monkeypatch):
    from tarzan.data import price_cache

    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(price_cache, "is_enabled", lambda: True)
    monkeypatch.setattr(price_cache, "_ticker_isin_path", lambda: tmp_path / "xref.json")
    price_cache.store_ticker_isin("MATE", US, observed_at="2026-01-01T00:00:00+00:00")
    price_cache.store_ticker_isin("MATE.MI", EU, observed_at="2026-01-02T00:00:00+00:00")
    assert price_cache.load_ticker_isin("MATE") == US
    assert price_cache.load_ticker_isin("MATE.MI") == EU
    assert price_cache.load_ticker_isin_reverse(US) == "MATE"
    assert price_cache.load_ticker_isin_reverse(EU) == "MATE.MI"


def test_a_promoted_listing_is_never_replayed_for_the_other_fund(two_mates):
    from tarzan.data import enricher

    us = enricher._bare_resolution_cache_key("MATE", US)
    eu = enricher._bare_resolution_cache_key("MATE.MI", EU)
    assert us != eu and US in us and EU in eu
    assert enricher._bare_resolution_cache_key("MATE") == ""  # ambiguous: no key
    assert enricher._bare_resolution_cache_key("ISAC.MI") == "TICKER:ISAC"


def test_the_backtest_keeps_two_funds_in_two_sleeves_and_hedges_the_right_one(two_mates):
    from tarzan.backtest.engine import is_currency_hedged
    from tarzan.backtest.model import WhatIfItem

    us = WhatIfItem("MATE", "MATE", US, 5.0, None, {}, {}, "")
    eu = WhatIfItem("MATE", "MATE.MI", EU, 5.0, None, {}, {}, "")
    isac = WhatIfItem("ISAC", "ISAC.MI", "IE00B6R52259", 5.0, None, {}, {}, "")
    assert {us.key, eu.key, isac.key} == {f"MATE@{US}", f"MATE@{EU}", "ISAC"}
    assert is_currency_hedged(eu) and not is_currency_hedged(us)


def test_the_rebalance_bridge_matches_a_target_by_its_isin_not_its_root(two_mates):
    from tarzan.export.newsletter._sections_alloc import _holding_bridge_rows

    class _M:
        holdings_df = pd.DataFrame([{"isin": US, "ticker": "MATE", "name": "Man Active Trend Enhanced",
                                     "asset_class": "Equities"}])

    class _Ctx:
        metrics = _M()

    items = [{"category": "UCITS", "ticker": "MATE.MI", "isin": EU, "actual_pct": 10.0, "target_pct": 10.0},
             {"category": "US", "ticker": "MATE", "isin": US, "actual_pct": 0.0, "target_pct": 0.0}]
    kept, new, sold = _holding_bridge_rows(_Ctx(), items, base=100.0, weights={US: 12.0},
                                           series=None, series_key="isin")
    assert len(new) == 1 and len(sold) == 1 and not kept

