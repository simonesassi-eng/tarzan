"""Geography must use the same taxonomy location as the instrument registry."""

import pandas as pd
import pytest

from tarzan import config as cfg
from tarzan.data import geo_resolver as geo
from tarzan.models.holding import Geography


@pytest.fixture
def taxonomy(monkeypatch, tmp_path):
    source = tmp_path / "configured" / "instrument_taxonomy.csv"
    source.parent.mkdir()
    pd.DataFrame([{
        "name": "Synthetic Location Fund", "ticker": "SYNLOC",
        "isin": "IE0000000001", "kind": "ETF", "asset_class": "Equities",
        "usa": 100.,
    }]).to_csv(source, index=False)
    monkeypatch.setenv("TARZAN_TAXONOMY_PATH", str(source))
    monkeypatch.setattr(geo, "_ASSET_GEO_DF", None)
    monkeypatch.setattr(geo, "ASSET_GEO_PATH", "input/instrument_taxonomy.csv")
    cfg._load_indexes_csv.cache_clear()
    yield source
    cfg._load_indexes_csv.cache_clear()


def test_configured_taxonomy_works_from_an_unrelated_cwd_without_provider_io(
    taxonomy, monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    for method in ("_geo_from_own_country", "justetf_index_name", "_geo_from_top_holdings"):
        monkeypatch.setattr(geo, method, lambda *a: pytest.fail("local row triggered provider I/O"))
    assert cfg._load_indexes_csv()["ticker"].tolist() == ["SYNLOC"]
    assert geo.resolve_geo("IE0000000001", "SYNLOC.MI") == (
        {Geography.USA: 100.}, "index_geo_allocation (isin)",
    )


def test_explicit_geo_path_override_keeps_its_existing_contract(taxonomy, monkeypatch, tmp_path):
    override = tmp_path / "custom-geo.csv"
    pd.DataFrame([{"ticker": "SYNLOC", "usa": 0., "japan": 100.}]).to_csv(override, index=False)
    monkeypatch.setattr(geo, "ASSET_GEO_PATH", str(override))
    assert geo._lookup_asset_geo("", "SYNLOC.MI") == (
        {Geography.JAPAN: 100.}, "index_geo_allocation (ticker)",
    )


def test_geo_reset_reloads_changed_taxonomy_without_shadowing_user_input(
    taxonomy, monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    assert geo._lookup_asset_geo("IE0000000001", "SYNLOC.MI")[0] == {Geography.USA: 100.}
    frame = pd.read_csv(taxonomy)
    frame["usa"], frame["japan"] = 0., 100.
    frame.to_csv(taxonomy, index=False)
    geo.reset_caches()
    assert geo._lookup_asset_geo("IE0000000001", "SYNLOC.MI")[0] == {Geography.JAPAN: 100.}


@pytest.mark.parametrize("invalid", [
    True, False, float("inf"), float("-inf"), float("nan"), 10 ** 400,
    101, -1, "invalid", None,
])
def test_invalid_percentages_cannot_be_geographic_exposure(invalid):
    row = pd.Series({"usa": invalid}, dtype=object)
    assert geo._parse_geo_row(row) is None


@pytest.mark.parametrize("value", [100, 100., "100", .5])
def test_valid_numeric_geographic_percentages_remain_usable(value):
    assert geo._parse_geo_row(pd.Series({"usa": value}, dtype=object)) == {
        Geography.USA: float(value),
    }
