"""Class and geography targets derived from the per-instrument plan.

The change this covers: when ``targets_per_holding.csv`` carries weights, it IS the
plan, and the asset-class and equity-geography targets are computed from it instead of
read from ``targets.csv``. Three hand-maintained lists could not stay in agreement and
had stopped — on the reference book the plan implied 21% fixed income against a declared
28%, and 15% alternative against 11% — so every drift figure in the issue measured the
book against a portfolio its own plan does not produce.

Weights here are invented. The tickers are real because the taxonomy they resolve
against is committed, but no real plan weight appears in this file.
"""

from __future__ import annotations

import pytest

from tarzan.engine import target_derivation as td


def _rows(*specs):
    """``(isin, ticker, weight)`` triples in the loader's own row shape."""
    return {isin: {"isin": isin, "ticker": ticker, "name": ticker,
                   "target_portfolio": weight}
            for isin, ticker, weight in specs}


class TestThePlanIsDeduplicatedByInstrument:
    """The file is keyed by ISIN, and one fund can hold two of them."""

    def test_two_listings_of_one_fund_count_once(self):
        """The trap this rule exists for.

        The reference plan lists two of its sleeves under two ISINs each, so the raw
        file sums to 115.5% where the plan is 100%. Deriving from the file as loaded
        would have counted a 2x leveraged sleeve at 16% instead of 8% — doubling its
        equity notional and every class target downstream.
        """
        plan, notes = td.plan_weights(_rows(
            ("IE00AAAAAAA1", "CL2", 8.0),
            ("FR00BBBBBBB2", "CL2", 8.0),
            ("IE00CCCCCCC3", "SGLD", 10.0),
        ))
        assert plan == {"CL2": 8.0, "SGLD": 10.0}
        assert notes == []

    def test_the_same_instrument_at_two_weights_is_a_conflict(self):
        """Not averaged: an input that contradicts itself is reported, and the larger
        weight is kept so the plan is never silently under-stated."""
        plan, notes = td.plan_weights(_rows(
            ("IE00AAAAAAA1", "CL2", 8.0),
            ("FR00BBBBBBB2", "CL2", 12.0),
        ))
        assert plan == {"CL2": 12.0}
        assert any("different weights" in n for n in notes)

    def test_a_zero_or_missing_weight_is_not_in_the_plan(self):
        plan, _ = td.plan_weights(_rows(("IE00AAAAAAA1", "SGLD", 0.0)))
        assert plan == {}
        rows = {"X": {"isin": "X", "ticker": "SGLD", "target_portfolio": None}}
        assert td.plan_weights(rows)[0] == {}

    def test_a_row_naming_nothing_is_reported_not_dropped_silently(self):
        rows = {"": {"isin": "", "ticker": "", "target_portfolio": 5.0}}
        plan, notes = td.plan_weights(rows)
        assert plan == {}
        assert notes and "neither a ticker nor an ISIN" in notes[0]

    def test_an_empty_file_yields_an_empty_plan(self):
        assert td.plan_weights({}) == ({}, [])


class TestClassTargetsUseTheNotionalBreakdown:
    def test_a_capital_efficient_sleeve_lands_in_two_classes(self, monkeypatch):
        """A 90/60 efficient core puts 90% of its weight in equities and 60% in bonds,
        which is why the class targets can exceed the capital."""
        monkeypatch.setattr(
            td, "_primary_class", lambda t: "Equities")
        import tarzan.config as cfg
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: {"Equities": 90.0, "Fixed Income": 60.0})
        classes, notes = td.derive_class_targets({"NTSG": 40.0})
        assert classes == {"Equities": 36.0, "Fixed Income": 24.0}
        assert notes == []

    def test_the_totals_are_not_normalised_because_leverage_is_the_point(self,
                                                                        monkeypatch):
        import tarzan.config as cfg
        monkeypatch.setattr(td, "_primary_class", lambda t: "Equities")
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: {"Equities": 200.0})
        classes, _ = td.derive_class_targets({"CL2": 50.0})
        assert classes == {"Equities": 100.0}      # 50% of capital, 2x exposure

    def test_an_instrument_with_no_class_at_all_contributes_nothing_and_is_reported(
            self, monkeypatch):
        """Defaulting it into some class would move a target the reader cannot trace
        back to any instrument."""
        import tarzan.config as cfg
        monkeypatch.setattr(td, "_primary_class", lambda t: None)
        monkeypatch.setattr(cfg, "class_breakdown_for", lambda i, t, c: {})
        classes, notes = td.derive_class_targets({"WAT": 12.0})
        assert classes == {}
        assert notes and "contributes to no class target" in notes[0]


class TestGeographyIsWeightedByEquityNotional:
    def test_a_levered_us_tracker_counts_at_its_notional(self, monkeypatch):
        """Two sleeves of equal weight, one 2x US and one 1x Japan: the US side must
        weigh twice as much in the geography split, not the same."""
        import tarzan.config as cfg
        from tarzan.models.holding import Geography

        breakdowns = {"CL2": {"Equities": 200.0}, "JPN": {"Equities": 100.0}}
        geos = {"CL2": ({Geography.USA: 100.0}, "taxonomy"),
                "JPN": ({Geography.JAPAN: 100.0}, "taxonomy")}
        monkeypatch.setattr(td, "_primary_class", lambda t: "Equities")
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: breakdowns[t])
        monkeypatch.setattr("tarzan.data.geo_resolver._lookup_asset_geo",
                            lambda isin, ticker, index_name="": geos[ticker])
        geo, notes = td.derive_geo_targets({"CL2": 10.0, "JPN": 10.0})
        assert geo == pytest.approx({"USA": 200 / 3, "Japan": 100 / 3})
        assert sum(geo.values()) == pytest.approx(100.0)
        assert notes == []

    def test_a_sleeve_with_no_geography_row_is_excluded_and_reported(self, monkeypatch):
        """Excluding it renormalises the rest, which changes the answer — so it is
        stated rather than absorbed."""
        import tarzan.config as cfg
        from tarzan.models.holding import Geography

        monkeypatch.setattr(td, "_primary_class", lambda t: "Equities")
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: {"Equities": 100.0})
        monkeypatch.setattr(
            "tarzan.data.geo_resolver._lookup_asset_geo",
            lambda isin, ticker, index_name="": (
                ({Geography.USA: 100.0}, "taxonomy") if ticker == "KNOWN" else None))
        geo, notes = td.derive_geo_targets({"KNOWN": 10.0, "MYSTERY": 10.0})
        assert geo == pytest.approx({"USA": 100.0})
        assert notes and "no geography row" in notes[0]

    def test_a_plan_with_no_equity_has_no_geography(self, monkeypatch):
        import tarzan.config as cfg
        monkeypatch.setattr(td, "_primary_class", lambda t: "Gold")
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: {"Gold": 100.0})
        assert td.derive_geo_targets({"SGLD": 100.0}) == ({}, [])


class TestWhatTheConfigEndsUpHolding:
    class _Cfg:
        def __init__(self):
            self.invested_allocation_targets_pctg = {
                "Equities": 65.0, "Fixed Income": 25.0, "Gold": 5.0,
                "Commodities": 0.0, "Crypto": 0.0, "Alternative": 5.0}
            self.equity_geo_targets_pctg = {
                "USA": 50.0, "Japan": 9.0, "Eurozone EMU": 16.0,
                "Dev ex-USA ex-EMU ex-JP": 13.0, "Emerging Markets": 12.0}

    @pytest.fixture
    def patched(self, monkeypatch):
        import tarzan.config as cfg
        from tarzan.models.holding import Geography

        monkeypatch.setattr(td, "_primary_class", lambda t: "Equities")
        monkeypatch.setattr(cfg, "class_breakdown_for",
                            lambda i, t, c: {"Equities": 100.0})
        monkeypatch.setattr("tarzan.data.geo_resolver._lookup_asset_geo",
                            lambda isin, ticker, index_name="": (
                                {Geography.USA: 100.0}, "taxonomy"))

    def test_an_empty_plan_leaves_the_declared_targets_alone(self):
        config = self._Cfg()
        before = dict(config.invested_allocation_targets_pctg)
        assert td.apply_to_config(config, {}) is None
        assert config.invested_allocation_targets_pctg == before

    def test_a_populated_plan_replaces_both_dictionaries(self, patched):
        config = self._Cfg()
        report = td.apply_to_config(config, _rows(("I1", "AVWC", 100.0)))
        assert report["applied"] is True
        assert config.invested_allocation_targets_pctg["Equities"] == 100.0
        assert config.equity_geo_targets_pctg["USA"] == 100.0

    def test_a_class_the_plan_abandons_stays_as_a_zero(self, patched):
        """Dropping the key would hide the class from the table, and a class the plan
        walked away from still has holdings that must be seen drifting to zero."""
        config = self._Cfg()
        td.apply_to_config(config, _rows(("I1", "AVWC", 100.0)))
        assert config.invested_allocation_targets_pctg["Gold"] == 0.0
        assert config.equity_geo_targets_pctg["Japan"] == 0.0

    def test_the_report_carries_both_the_derived_and_the_declared(self, patched):
        config = self._Cfg()
        report = td.apply_to_config(config, _rows(("I1", "AVWC", 100.0)))
        assert report["declared_classes"]["Fixed Income"] == 25.0
        assert report["notional_pct"] == pytest.approx(100.0)

    def test_nothing_derivable_keeps_the_declared_targets(self, monkeypatch):
        import tarzan.config as cfg
        monkeypatch.setattr(td, "_primary_class", lambda t: None)
        monkeypatch.setattr(cfg, "class_breakdown_for", lambda i, t, c: {})
        config = self._Cfg()
        report = td.apply_to_config(config, _rows(("I1", "WAT", 50.0)))
        assert report["applied"] is False
        assert config.invested_allocation_targets_pctg["Equities"] == 65.0


class TestAgainstTheCommittedTaxonomy:
    """One end-to-end pass with no monkeypatching, so the wiring to the real lookups is
    exercised: the taxonomy ships in the repo, only the weights here are invented."""

    def test_an_efficient_core_splits_across_equities_and_bonds(self):
        classes, notes = td.derive_class_targets({"NTSG": 100.0})
        assert classes.get("Equities") == pytest.approx(90.0)
        assert classes.get("Fixed Income") == pytest.approx(60.0)
        assert notes == []

    def test_its_geography_comes_out_of_the_taxonomy_and_sums_to_one_hundred(self):
        geo, notes = td.derive_geo_targets({"NTSG": 100.0})
        assert sum(geo.values()) == pytest.approx(100.0)
        assert geo["USA"] > geo["Japan"] > 0
        assert notes == []
