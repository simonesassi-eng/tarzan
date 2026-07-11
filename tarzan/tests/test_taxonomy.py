"""Tests for the asset-class / geography order registry.

After unification, every surface shares ONE canonical asset-class order
(user-approved "Cash last"). These tests pin that canonical order and assert
each consumer surface exposes it, so an accidental reorder — which would
shift rows in a filed report — fails the suite.
"""

from __future__ import annotations

from tarzan.models import taxonomy as tx


# The single canonical order (user-approved "Cash last"). Changing this is a
# deliberate, sign-off-gated layout change for every report, not a refactor.
_CANONICAL = ["Equities", "Fixed Income", "Gold", "Commodities", "Crypto",
              "Alternative", "Cash & Cash Equivalents"]
_HIST_GEO = ["USA", "Japan", "Eurozone EMU", "Dev ex-USA ex-EMU ex-JP",
             "Emerging Markets"]


class TestCanonicalOrder:
    def test_canonical_is_cash_last_all_seven(self):
        assert list(tx.CANONICAL_ORDER) == _CANONICAL
        # All 7 classes present — Crypto is no longer missing anywhere.
        assert set(tx.CANONICAL_ORDER) == set(tx.ASSET_CLASSES)
        assert tx.CANONICAL_ORDER[-1] == "Cash & Cash Equivalents"
        assert "Crypto" in tx.CANONICAL_ORDER

    def test_all_variants_now_share_the_canonical_order(self):
        # Unification: every named variant is the one canonical order.
        for variant in (tx.ORDER_DASHBOARD, tx.ORDER_NEWSLETTER, tx.ORDER_PERF,
                        tx.ORDER_WHATIF, tx.ORDER_BASE):
            assert list(variant) == _CANONICAL

    def test_geo(self):
        assert list(tx.GEO_ORDER) == _HIST_GEO


class TestConsumersUnified:
    """Every surface must now expose the one canonical order."""

    def test_metrics_dashboard_order(self):
        from tarzan.models.taxonomy import ORDER_DASHBOARD
        assert list(ORDER_DASHBOARD) == _CANONICAL

    def test_format_base_order(self):
        from tarzan.export import _format
        assert _format._ASSET_CLASS_BASE_ORDER == _CANONICAL

    def test_newsletter_orders(self):
        from tarzan.export import newsletter as nl
        assert nl._NEWSLETTER_CLASS_ORDER == _CANONICAL
        assert nl._PERF_CLASS_ORDER == _CANONICAL

    def test_whatif_orders(self):
        from tarzan.export import whatif_excel as we
        assert we._ASSET_ORDER == _CANONICAL
        # The what-if geo order appends an explicit "Other" bucket.
        assert we._GEO_ORDER == _HIST_GEO + ["Other"]


class TestOrderWithExtras:
    def test_no_present_returns_full_order(self):
        assert tx.order_with_extras(tx.CANONICAL_ORDER) == list(tx.CANONICAL_ORDER)

    def test_filters_to_present(self):
        got = tx.order_with_extras(tx.CANONICAL_ORDER, present=["Gold", "Equities"])
        assert got == ["Equities", "Gold"]  # in canonical order

    def test_unlisted_class_appended_alphabetically_not_dropped(self):
        # A class absent from the canonical set (a future/unknown class) is
        # appended, never silently dropped from a report.
        got = tx.order_with_extras(
            tx.CANONICAL_ORDER, present=["Equities", "Zebras", "Aardvark"])
        assert got[0] == "Equities"
        assert set(got) == {"Equities", "Zebras", "Aardvark"}
        assert got[-2:] == ["Aardvark", "Zebras"]  # extras alphabetical at end
