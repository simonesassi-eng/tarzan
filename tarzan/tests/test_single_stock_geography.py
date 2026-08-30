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


class TestASecondaryListingFallsBackToTheDomicile:
    """A secondary listing often publishes no country at all.

    Nestlé's Stuttgart line reports ``quoteType='EQUITY'`` and ``country=None``,
    while its primary Swiss listing says Switzerland. The ISIN's own prefix settles
    it: for a SHARE that prefix is the company's domicile, the same fact the
    provider would have reported. Not a usable signal for a fund, whose domicile
    says nothing about its exposure — and this rung already refuses anything but
    EQUITY.
    """

    def test_the_isin_prefix_places_a_countryless_listing(self, monkeypatch, _live):
        _stub_info(monkeypatch, {"quoteType": "EQUITY"})       # no country

        breakdown, source = gr._geo_from_own_country("NESR.SG", "CH0038863350")

        assert breakdown == {Geography.DEVELOPED_EX_USA_EMU_JP: 100.0}
        assert "CH" in source

    def test_the_reported_country_still_wins_when_present(self, monkeypatch, _live):
        """The prefix is a FALLBACK. A listing that states its country is
        authoritative — a company can be domiciled in Jersey and be British."""
        _stub_info(monkeypatch, {"quoteType": "EQUITY", "country": "United States"})

        breakdown, source = gr._geo_from_own_country("X", "CH0038863350")

        assert breakdown == {Geography.USA: 100.0}
        assert "United States" in source

    def test_an_unplaceable_prefix_claims_nothing(self, monkeypatch, _live):
        """KY (Cayman) and BM (Bermuda) are not in the bucket map, and guessing
        would be worse than declining."""
        _stub_info(monkeypatch, {"quoteType": "EQUITY"})
        assert gr._geo_from_own_country("X", "KY0000000000") is None

    def test_the_map_carries_both_a_name_and_a_code_for_every_country(self):
        """The map already held US and JP in both forms and the rest name-only,
        which is why a Swiss share could not be placed from its ISIN."""
        from tarzan import config as cfg

        gm = cfg.geography_map()
        names = {k: v for k, v in gm.items() if len(k) > 2}
        codes = {k for k in gm if len(k) == 2 and k.isupper()}
        assert len(codes) >= 40, f"only {len(codes)} ISO-2 codes in the map"
        # Every bucket reachable by name must be reachable by a code too.
        assert set(names.values()) == {gm[c] for c in codes}


class TestAnAllOtherLookThroughIsNotAnAnswer:
    """``gm.get(country, Geography.OTHER)`` turns every miss into a positive claim.

    An aggregate that is ENTIRELY Other learned nothing about any constituent, and
    returning it says "geography known, and it is Other" — which blocks the rungs
    below and prints a bucket the reader cannot act on. Measured on a single stock
    whose ISIN the provider would not link: {Other: 100.0} from the fund
    look-through, on an instrument that has no constituents at all.
    """

    def test_an_all_other_aggregate_is_declined(self, monkeypatch, _live):
        import pandas as pd

        holdings = pd.DataFrame({"Holding Percent": [0.6, 0.4]},
                                index=["AAA", "BBB"])

        class _Funds:
            top_holdings = holdings

        class _T:
            funds_data = _Funds()
            info = {"country": "Ruritania"}          # unplaceable

        monkeypatch.setattr("yfinance.Ticker", lambda *a, **k: _T())
        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                            lambda fn, **kw: fn())

        assert gr._geo_from_top_holdings("XXXX") is None

    def test_a_partially_placed_aggregate_is_still_returned(self, monkeypatch, _live):
        """Guards against over-correcting: Other is legitimate for SOME of a real
        fund's constituents, where it is diluted by the ones that placed."""
        import pandas as pd

        holdings = pd.DataFrame({"Holding Percent": [0.7, 0.3]},
                                index=["USCO", "ZZCO"])
        countries = {"USCO": "United States", "ZZCO": "Ruritania"}

        class _Funds:
            top_holdings = holdings

        class _T:
            def __init__(self, sym=""):
                self.symbol = str(sym)
                self.funds_data = _Funds()

            @property
            def info(self):
                return {"country": countries.get(self.symbol, "")}

        monkeypatch.setattr("yfinance.Ticker", lambda sym="", *a, **k: _T(sym))
        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                            lambda fn, **kw: fn())

        result = gr._geo_from_top_holdings("FUND")

        assert result is not None
        breakdown, _ = result
        assert breakdown[Geography.USA] == pytest.approx(70.0)
        assert Geography.OTHER in breakdown
