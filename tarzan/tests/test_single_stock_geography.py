"""A single stock's geography is its own country.

The resolver had two paths and neither fitted an ordinary share: the curated
taxonomy, which a user fills for the FUNDS they hold, and a look-through to an
ETF's top holdings. For a stock the second is meaningless — Apple has no top
holdings, it IS one — so every share fell through to "Not Available". A book of
US single stocks reported 95% of its value as geographically unknown, and the
drift column printed a +95pp deviation against a bucket that is not a target.

Everything needed was already present: the provider reports ``country`` on the
instrument itself and the config's country→geography map already knows what to do
with it. Nothing joined them.

Network-free: the provider call is stubbed, so these pin the JOIN rather than
Yahoo's data.
"""

from __future__ import annotations

import pytest

from tarzan.data import geo_resolver as gr
from tarzan.models.holding import Geography


@pytest.fixture()
def _live(monkeypatch):
    """Allow the live-transport branch, which the resolver gates every fetch on."""
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)


def _stub_info(monkeypatch, info: dict):
    """Stub the throttled provider fetch the resolver uses."""
    monkeypatch.setattr(gr, "_lookup_asset_geo", lambda *a, **k: None)
    monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                        lambda fn, **kw: info)


class TestASingleStockGetsItsOwnCountry:
    def test_a_us_share_is_wholly_usa(self, monkeypatch, _live):
        _stub_info(monkeypatch, {"quoteType": "EQUITY", "country": "United States"})

        breakdown, source = gr._geo_from_own_country("AAPL")

        assert breakdown == {Geography.USA: 100.0}
        assert "United States" in source

    def test_the_country_map_places_it_not_a_hardcoded_list(self, monkeypatch, _live):
        """Switzerland is developed-ex-USA/EMU/JP, and the config already says so —
        this rung must read that map rather than restate it."""
        _stub_info(monkeypatch, {"quoteType": "EQUITY", "country": "Switzerland"})

        breakdown, _ = gr._geo_from_own_country("NESN.SW")

        assert breakdown == {Geography.DEVELOPED_EX_USA_EMU_JP: 100.0}

    def test_a_fund_does_not_take_this_path(self, monkeypatch, _live):
        """An ETF's geography IS a look-through question, which the next rung
        answers. Claiming its domicile would put every Irish UCITS in Europe."""
        _stub_info(monkeypatch, {"quoteType": "ETF", "country": "Ireland"})

        assert gr._geo_from_own_country("SWDA.MI") is None

    def test_an_unplaceable_country_claims_nothing(self, monkeypatch, _live):
        """Geography.OTHER is fine for ONE constituent among many, where it is
        diluted. Here it would be the whole answer, so no answer is honest."""
        _stub_info(monkeypatch, {"quoteType": "EQUITY", "country": "Ruritania"})

        assert gr._geo_from_own_country("XXXX") is None

    def test_a_missing_country_claims_nothing(self, monkeypatch, _live):
        _stub_info(monkeypatch, {"quoteType": "EQUITY"})
        assert gr._geo_from_own_country("XXXX") is None

    def test_a_pinned_run_makes_no_live_call(self, monkeypatch):
        """A reproducible run may not reach a provider. It reads a previously
        cached breakdown instead — the same rule the other live rungs follow."""
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("a pinned run must not fetch")

        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf", _boom)
        assert gr._geo_from_own_country("AAPL") is None


class TestTheResolverAsksBeforeTheFundPaths:
    def test_a_stock_is_resolved_without_a_top_holdings_fetch(self, monkeypatch, _live):
        """Ordering is not cosmetic: the look-through costs a scrape plus one fetch
        per constituent to answer None for a share."""
        _stub_info(monkeypatch, {"quoteType": "EQUITY", "country": "United States"})

        def _boom(*a, **k):
            raise AssertionError("the fund look-through was reached for a stock")

        monkeypatch.setattr(gr, "_geo_from_top_holdings", _boom)
        monkeypatch.setattr(gr, "justetf_index_name", lambda isin: None)

        breakdown, _ = gr.resolve_geo("US0378331005", "AAPL")

        assert breakdown == {Geography.USA: 100.0}
