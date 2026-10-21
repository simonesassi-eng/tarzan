"""Causal identity, immutable target aliases and validated JSON cache listing."""

from dataclasses import FrozenInstanceError
import datetime as dt
import io

import pandas as pd
import pytest

from tarzan import config, orchestrator, runtime
from tarzan.contracts.targets import TargetSetOutcome, TargetSetStatus
from tarzan.data import loader, manual_proxies, price_cache
from tarzan.data.identity_catalog import TargetIdentityCatalog
from tarzan.models.holding import Holding


ISIN = "US0000000001"
OTHER = "US0000000002"


@pytest.fixture(autouse=True)
def private_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("TARZAN_DISABLE_CACHE")
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))


def holding(**changes):
    values = dict(isin=ISIN, ticker=ISIN, quantity=10.0,
                  cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR")
    values.update(changes)
    return Holding(**values)


def test_target_plan_and_its_rows_are_defensive_and_immutable():
    row = {"isin": ISIN, "ticker": "SYNTH", "target_portfolio": 50.0}
    plan = TargetSetOutcome(status=TargetSetStatus.VALID, mapping={ISIN: row, "SYNTH": row})
    assert plan[ISIN] is plan["SYNTH"]
    row["target_portfolio"] = 99.0
    assert plan[ISIN]["target_portfolio"] == 50.0
    with pytest.raises(TypeError):
        plan[ISIN]["target_portfolio"] = 0.0
    with pytest.raises(FrozenInstanceError):
        plan.status = TargetSetStatus.INVALID
    assert plan.planning_eligible


def test_loader_freezing_preserves_single_source_row_under_legacy_aliases():
    plan = loader.load_targets_per_holding(io.BytesIO(
        f"isin,ticker,target_portfolio\n{ISIN},SYNTH,100\n".encode(),
    ))
    assert len({id(row) for row in plan.values()}) == 1
    with pytest.raises(TypeError):
        plan.rows[0].value["target_portfolio"] = 0.0


@pytest.mark.parametrize("seed_phase", [False, True])
def test_future_profile_cannot_attach_a_target_or_suppress_a_real_seed(monkeypatch, seed_phase):
    price_cache.store_instrument_profile(ISIN, {
        "isin": ISIN, "kind": "ETF", "status": "VERIFIED", "source": "OpenFIGI",
        "observed_at": "2025-01-04T00:00:00+00:00",
        "provenance": {"provider": "OpenFIGI"}, "tickers": ["FUTURE"],
    })
    runtime.configure(deterministic=True, as_of=dt.date(2025, 1, 3))
    monkeypatch.setattr(config, "resolve_taxonomy_identity", lambda isin, ticker, *a: (isin, ticker))
    h = holding()
    targets = {"FUTURE": {"isin": "", "ticker": "FUTURE", "target_portfolio": 50.0}}
    if seed_phase:
        assert len(orchestrator._seed_missing_targets([h], targets)) == 1
    else:
        orchestrator._apply_per_holding_targets([h], targets)
        assert h.target_portfolio is None


def test_shared_catalog_reads_each_profile_once_across_join_and_seed_checks(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "resolve_taxonomy_identity", lambda isin, ticker, *a: (isin, ticker))
    monkeypatch.setattr(price_cache, "load_instrument_profile",
                        lambda isin, **kw: calls.append((isin, kw["as_of"])) or {"tickers": ["VISIBLE"]})
    h = holding()
    targets = {"VISIBLE": {"isin": "", "ticker": "VISIBLE", "target_portfolio": 50.0}}
    catalog = TargetIdentityCatalog()
    orchestrator._apply_per_holding_targets([h], targets, catalog=catalog)
    assert orchestrator._seed_missing_targets([h], targets, catalog=catalog) == []
    assert len(calls) == 1 and calls[0][1] == runtime.context().captured_at


def test_future_xrefs_are_rejected_even_by_callers_that_omit_as_of():
    price_cache.store_ticker_isin("FUTURE.MI", ISIN, observed_at="2025-01-04T00:00:00+00:00")
    runtime.configure(deterministic=True, as_of=dt.date(2025, 1, 3))
    assert price_cache.load_ticker_isin("FUTURE.MI") is None
    assert price_cache.load_ticker_isin_reverse(ISIN) is None


def test_refreshed_and_changed_xrefs_preserve_historical_observations():
    price_cache.store_ticker_isin("SYNTH.MI", ISIN, observed_at="2025-01-01T00:00:00+00:00")
    price_cache.store_ticker_isin("SYNTH.MI", ISIN, observed_at="2025-01-10T00:00:00+00:00")
    price_cache.store_ticker_isin("SYNTH.MI", OTHER, observed_at="2025-02-01T00:00:00+00:00")
    assert price_cache.load_ticker_isin("SYNTH.MI", as_of=dt.date(2025, 1, 2)) == ISIN
    assert price_cache.load_ticker_isin("SYNTH.MI", as_of=dt.date(2025, 2, 2)) == OTHER


def test_a_pinned_enrichment_cannot_learn_a_new_current_xref():
    runtime.configure(deterministic=True, as_of=dt.date(2025, 1, 3))
    price_cache.store_ticker_isin("NEW.MI", ISIN)
    assert price_cache.load_ticker_isin("NEW.MI", as_of=dt.date(2026, 10, 21)) is None


def test_similar_names_cannot_suppress_a_different_explicit_target_seed(monkeypatch):
    monkeypatch.setattr(config, "resolve_taxonomy_identity", lambda isin, ticker, *a: (isin, ticker))
    h = holding(ticker="", name="Synthetic ETF Broad World Acc")
    target = {"isin": "", "ticker": "DIFFERENT.MI", "name": "Synthetic ETF Broad World", "target_portfolio": 50.0}
    assert len(orchestrator._seed_missing_targets([h], {"DIFFERENT.MI": target})) == 1


def test_listing_uses_validated_json_and_never_lists_legacy_or_corrupt_entries(tmp_path):
    series = pd.Series([100.0, 101.0], index=pd.bdate_range("2025-01-01", periods=2))
    price_cache.store_history("MANUAL_VISIBLE", series)
    directory = price_cache.cache_dir() / "history"
    (directory / "MANUAL_LEGACY.pkl").write_bytes(b"must not be deserialized")
    (directory / "MANUAL_CORRUPT.json").write_text("{}", encoding="utf-8")
    assert manual_proxies.list_cached() == ["VISIBLE"]
    assert price_cache.list_history_symbols(prefix="MANUAL_") == ["MANUAL_VISIBLE"]


def test_sanitized_filename_collision_does_not_license_another_symbols_history():
    series = pd.Series([100.0, 101.0], index=pd.bdate_range("2025-01-01", periods=2))
    price_cache.store_history("A/B", series)
    assert price_cache.load_history("A_B") is None
