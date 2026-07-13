"""Track A / A3 — order→taxonomy referential-integrity check.

A held instrument with no curated instrument_taxonomy.csv row is a dangling
reference: it silently falls back to a default classification, which can
distort the allocation. ``_check_taxonomy_coverage`` surfaces each such
instrument as a data-quality WARNING instead of leaving it silent.
"""

from __future__ import annotations

from tarzan.runtime import data_quality as dq
from tarzan.models.holding import AssetClass, Holding
from tarzan.orchestrator import _check_taxonomy_coverage


def _h(isin, ticker, ac=AssetClass.ALTERNATIVE, seeded=False):
    return Holding(
        isin=isin, ticker=ticker, quantity=1.0, cost_basis_eur=100.0,
        market_value_eur=100.0, currency="EUR", asset_class=ac,
        is_seeded_target=seeded,
    )


class TestTaxonomyCoverage:
    def teardown_method(self):
        dq.reset()

    def test_unknown_instrument_flagged(self):
        dq.reset()
        _check_taxonomy_coverage([_h("XX9999999999", "ZZZQ")])
        tax = [i for i in dq.issues() if i.source == "taxonomy"]
        assert len(tax) == 1
        assert tax[0].context == "XX9999999999"
        assert "no instrument_taxonomy.csv row" in tax[0].message

    def test_known_instrument_not_flagged(self):
        # IE00077IIPQ8 (NTSG / WisdomTree Global Efficient Core) is in the
        # curated taxonomy, so it must NOT be flagged.
        dq.reset()
        _check_taxonomy_coverage([_h("IE00077IIPQ8", "NTSG", AssetClass.EQUITIES)])
        assert not [i for i in dq.issues() if i.source == "taxonomy"]

    def test_ticker_only_match_not_flagged(self):
        # A raw index / ETF present in the taxonomy by ticker (e.g. CSPX.L)
        # must match on the bare ticker even without an ISIN.
        dq.reset()
        _check_taxonomy_coverage([_h("", "CSPX.L", AssetClass.EQUITIES)])
        assert not [i for i in dq.issues() if i.source == "taxonomy"]

    def test_seeded_targets_skipped(self):
        # Rebalancer seeds aren't part of the real snapshot, so they are not
        # subject to the coverage check.
        dq.reset()
        _check_taxonomy_coverage([_h("XX9999999999", "ZZZQ", seeded=True)])
        assert not [i for i in dq.issues() if i.source == "taxonomy"]

    def test_only_unknown_among_mixed_flagged(self):
        dq.reset()
        _check_taxonomy_coverage([
            _h("XX9999999999", "ZZZQ"),
            _h("IE00077IIPQ8", "NTSG", AssetClass.EQUITIES),
        ])
        tax = [i for i in dq.issues() if i.source == "taxonomy"]
        assert len(tax) == 1 and tax[0].context == "XX9999999999"
