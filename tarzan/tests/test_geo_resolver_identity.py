"""A taxonomy row keyed by a bare ticker must match the full listing.

instrument_taxonomy.csv stores bare tickers ("CL2"); enrichment resolves
holdings to the traded listing ("CL2.MI"). An exact string compare missed the
row and the holding's whole equity notional landed in a "Not Available"
geography bucket instead of USA.
"""

import pandas as pd

from tarzan.data import geo_resolver as gr
from tarzan.models.holding import Geography


def _taxonomy(monkeypatch, rows):
    monkeypatch.setattr(gr, "_load_asset_geo", lambda: pd.DataFrame(rows))


CL2_ROW = {
    "name": "Amundi MSCI USA (2x) Leveraged",
    "ticker": "CL2",          # bare, as curated
    "isin": "",               # no ISIN in the taxonomy
    "usa": 100.0,
    "emerging_markets": 0.0,
    "eurozone_emu": 0.0,
    "japan": 0.0,
    "dev_ex_usa_ex_emu_ex_jp": 0.0,
}


def test_bare_taxonomy_ticker_matches_full_listing(monkeypatch):
    _taxonomy(monkeypatch, [CL2_ROW])

    # The holding as enrichment sees it: resolved listing + a real ISIN that
    # the taxonomy row does not carry.
    result = gr._lookup_asset_geo("FR0010755611", "CL2.MI")

    assert result is not None, "bare taxonomy ticker did not match CL2.MI"
    breakdown, source = result
    assert breakdown == {Geography.USA: 100.0}
    assert source == "index_geo_allocation (ticker)"


def test_isin_still_wins_and_lowercase_isin_matches(monkeypatch):
    _taxonomy(monkeypatch, [
        {**CL2_ROW, "isin": "fr0010755611"},   # lowercase in the CSV
        {**CL2_ROW, "name": "Other", "ticker": "CL2", "isin": "",
         "usa": 0.0, "japan": 100.0},
    ])

    breakdown, source = gr._lookup_asset_geo("FR0010755611", "CL2.MI")
    assert breakdown == {Geography.USA: 100.0}
    assert source == "index_geo_allocation (isin)"


def test_unknown_instrument_still_returns_none(monkeypatch):
    _taxonomy(monkeypatch, [CL2_ROW])
    assert gr._lookup_asset_geo("IE00NOTHERE", "NOPE.MI") is None
