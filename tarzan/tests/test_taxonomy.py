"""Tests for the asset-class / geography order registry.

These pin each named order variant to its HISTORICAL literal sequence, so an
accidental reorder (which would shift rows in a filed report) fails the suite.
They also confirm every consumer surface still exposes the same order it did
before the registry consolidation.
"""

from __future__ import annotations

from tarzan.models import taxonomy as tx


# The historical literals, captured verbatim from the pre-registry code. Do
# NOT "tidy" these — they are the contract. Changing one is a deliberate,
# sign-off-gated layout change, not a refactor.
_HIST_DASHBOARD = ["Equities", "Fixed Income", "Cash & Cash Equivalents",
                   "Gold", "Commodities", "Crypto", "Alternative"]
_HIST_NEWSLETTER = ["Equities", "Fixed Income", "Gold", "Cash & Cash Equivalents",
                    "Commodities", "Crypto", "Alternative"]
_HIST_PERF = ["Equities", "Fixed Income", "Commodities", "Gold",
              "Alternative", "Cash & Cash Equivalents"]
_HIST_WHATIF = ["Equities", "Fixed Income", "Gold", "Commodities", "Crypto",
                "Alternative", "Cash & Cash Equivalents"]
_HIST_GEO = ["USA", "Japan", "Eurozone EMU", "Dev ex-USA ex-EMU ex-JP",
             "Emerging Markets"]


class TestOrderVariantsPreserved:
    def test_dashboard(self):
        assert list(tx.ORDER_DASHBOARD) == _HIST_DASHBOARD

    def test_newsletter(self):
        assert list(tx.ORDER_NEWSLETTER) == _HIST_NEWSLETTER

    def test_perf(self):
        # Historically omits Crypto (appended via extras). Preserved as-is.
        assert list(tx.ORDER_PERF) == _HIST_PERF
        assert "Crypto" not in tx.ORDER_PERF

    def test_whatif(self):
        assert list(tx.ORDER_WHATIF) == _HIST_WHATIF

    def test_base_matches_dashboard(self):
        assert tx.ORDER_BASE == tx.ORDER_DASHBOARD

    def test_geo(self):
        assert list(tx.GEO_ORDER) == _HIST_GEO

    def test_all_classes_membership(self):
        # Every variant is a subset of the canonical class set (no typos),
        # and the canonical set covers all classes any variant references.
        for variant in (tx.ORDER_DASHBOARD, tx.ORDER_NEWSLETTER,
                        tx.ORDER_WHATIF, tx.ORDER_BASE):
            assert set(variant) == set(tx.ASSET_CLASSES)
        assert set(tx.ORDER_PERF) <= set(tx.ASSET_CLASSES)


class TestConsumersUnchanged:
    """Each surface must still expose the exact order it had pre-registry."""

    def test_metrics_dashboard_order(self):
        # metrics builds class_order from ORDER_DASHBOARD; assert the source.
        from tarzan.models.taxonomy import ORDER_DASHBOARD
        assert list(ORDER_DASHBOARD) == _HIST_DASHBOARD

    def test_format_base_order(self):
        from tarzan.export import _format
        assert _format._ASSET_CLASS_BASE_ORDER == _HIST_DASHBOARD

    def test_newsletter_orders(self):
        from tarzan.export import newsletter as nl
        assert nl._NEWSLETTER_CLASS_ORDER == _HIST_NEWSLETTER
        assert nl._PERF_CLASS_ORDER == _HIST_PERF

    def test_whatif_orders(self):
        from tarzan.export import whatif_excel as we
        assert we._ASSET_ORDER == _HIST_WHATIF
        # The what-if geo order appends an explicit "Other" bucket.
        assert we._GEO_ORDER == _HIST_GEO + ["Other"]


class TestOrderWithExtras:
    def test_no_present_returns_full_order(self):
        assert tx.order_with_extras(tx.ORDER_PERF) == list(tx.ORDER_PERF)

    def test_filters_to_present(self):
        got = tx.order_with_extras(tx.ORDER_DASHBOARD, present=["Gold", "Equities"])
        assert got == ["Equities", "Gold"]  # in canonical order

    def test_unlisted_class_appended_alphabetically_not_dropped(self):
        # A class absent from the variant (e.g. Crypto for PERF) is appended,
        # never silently dropped from a report.
        got = tx.order_with_extras(
            tx.ORDER_PERF, present=["Equities", "Crypto", "Zebras"])
        assert got[0] == "Equities"
        assert set(got) == {"Equities", "Crypto", "Zebras"}
        assert got[-2:] == ["Crypto", "Zebras"]  # extras alphabetical at end
