"""Verified ETF identity must survive Yahoo's generic EQUITY metadata."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import enricher, price_cache
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.runtime import data_quality as dq
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunSession, activate_session


MONEY_ISIN = "IE0005PGDJ30"
MONEY_INFO = {
    "quoteType": "EQUITY",
    "typeDisp": "Equity",
    "shortName": "FINECO AM ESTR PLUS UCITS ETF A",
    "longName": "FINECO AM ESTR PLUS UCITS ETF A EUR Acc",
    "currency": "EUR",
}
ETF_MAPPING = [{
    "data": [{
        "securityType": "ETP",
        "securityType2": "Mutual Fund",
        "marketSector": "Equity",
        "name": MONEY_INFO["longName"],
        "ticker": "MONEY",
    }],
}]


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    enricher.reset_run_caches()
    dq.reset()
    yield
    enricher.reset_run_caches()
    dq.reset()


def _holding(**kwargs):
    fields = {
        "isin": MONEY_ISIN,
        "ticker": "MONEY.MI",
        "quantity": 10.0,
        "cost_basis_eur": 100.0,
        "market_value_eur": 73.0,
        "currency": "EUR",
        "security_type": "ETF",
        "instrument_kind_evidence": ("ETF",),
    }
    fields.update(kwargs)
    return Holding(**fields)


def _profile(kind="ETF", *, observed_at=None, status="VERIFIED"):
    return {
        "isin": MONEY_ISIN,
        "kind": kind,
        "status": status,
        "source": "OpenFIGI",
        "observed_at": (
            observed_at or runtime.context().captured_at - dt.timedelta(seconds=1)
        ).isoformat(),
        "provenance": {"provider": "OpenFIGI", "idType": "ID_ISIN"},
        "securityType": ["ETP"] if kind == "ETF" else [],
    }


def _provider(monkeypatch, info=None):
    provider_info = dict(MONEY_INFO if info is None else info)
    history = pd.DataFrame(
        {"Close": [10.0, 10.5], "Price Close": [10.0, 10.5]},
        index=pd.date_range(end=runtime.today(), periods=2, tz="UTC"),
    )
    monkeypatch.setattr(enricher, "_resolve_isin", lambda *a, **k: (
        {"info": provider_info, "history": history}, "MONEY.MI",
    ))
    monkeypatch.setattr(enricher, "_openfigi_raw", lambda isin: [])
    monkeypatch.setattr(enricher, "get_geo_breakdown", lambda *a: None)
    return provider_info


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_money_is_valued_as_a_verified_etf_before_any_live_session_stamp(
    monkeypatch, cache_enabled,
):
    source_info = _provider(monkeypatch)
    if cache_enabled:
        price_cache.store_instrument_profile(MONEY_ISIN, _profile())
    else:
        monkeypatch.setenv("TARZAN_DISABLE_CACHE", "1")
        monkeypatch.setattr(enricher, "_openfigi_raw", lambda isin: ETF_MAPPING)
    ledger = RunLedger("money-metadata")
    session = RunSession(runtime.context(), {}, ledger)

    with activate_session(session):
        holding = enricher._enrich_and_classify(_holding())

    assert holding.instrument_type == "ETF"
    assert holding.security_type == "ETF"
    assert holding.asset_class is AssetClass.CASH_EQUIVALENTS
    assert holding.geography is Geography.OTHER
    assert holding.current_price == pytest.approx(10.5)
    assert holding.current_value == pytest.approx(105.0)
    assert holding.data_source == "yfinance:MONEY.MI"
    assert not ledger.failure_records()
    assert not [issue for issue in dq.issues() if issue.severity == dq.ERROR]
    assert source_info == MONEY_INFO  # never mutate the provider memo


@pytest.mark.parametrize("evidence", [
    ("ETF", "BOND"), ("ETF", "STOCK"), ("ETF", "CASH"),
])
def test_verified_etf_cannot_erase_conflicting_local_mechanics(
    monkeypatch, evidence,
):
    _provider(monkeypatch)
    price_cache.store_instrument_profile(MONEY_ISIN, _profile())
    holding, _ = enricher._enrich_single(
        _holding(instrument_kind_evidence=evidence),
    )

    assert holding.current_value == pytest.approx(73.0)
    assert holding.data_source == "last-known (instrument kind unavailable)"
    assert any(issue.severity == dq.ERROR for issue in dq.issues())


@pytest.mark.parametrize("field,value", [
    ("quoteType", "BOND"),
    ("quoteType", "STOCK"),
    ("instrumentType", "STOCK"),
    ("securityType", "COMMON STOCK"),
    ("typeDisp", "Bond"),
])
def test_verified_etf_does_not_override_specific_provider_kind_conflicts(
    monkeypatch, field, value,
):
    info = {**MONEY_INFO, field: value}
    _provider(monkeypatch, info)
    price_cache.store_instrument_profile(MONEY_ISIN, _profile())

    holding, _ = enricher._enrich_single(_holding())

    assert holding.current_value == pytest.approx(73.0)
    assert holding.data_source == "last-known (instrument kind unavailable)"


def test_yahoo_equity_is_not_overridden_without_verified_isin_evidence(
    monkeypatch,
):
    _provider(monkeypatch)
    holding, _ = enricher._enrich_single(_holding())

    assert holding.current_value == pytest.approx(73.0)
    assert holding.data_source == "last-known (instrument kind unavailable)"


@pytest.mark.parametrize("source", ["manual", "Yahoo"])
def test_another_profile_source_cannot_corroborate_generic_yahoo_equity(
    monkeypatch, source,
):
    _provider(monkeypatch)
    price_cache.store_instrument_profile(
        MONEY_ISIN, {**_profile(), "source": source},
    )

    holding, _ = enricher._enrich_single(_holding())

    assert holding.current_value == pytest.approx(73.0)
    assert holding.data_source == "last-known (instrument kind unavailable)"


@pytest.mark.parametrize("isin", ["", "MONEY", "123456789012"])
def test_invalid_isin_cannot_trigger_etf_metadata_reconciliation(
    monkeypatch, isin,
):
    monkeypatch.setattr(enricher, "_openfigi_raw", lambda value: pytest.fail(
        "an invalid ISIN opened an identity lookup",
    ))

    info = enricher._reconcile_verified_etf_metadata(
        MONEY_INFO, _holding(isin=isin),
    )

    assert info is MONEY_INFO
    assert info["quoteType"] == "EQUITY"


def test_generic_yahoo_equity_remains_a_stock_without_etf_identity(monkeypatch):
    _provider(monkeypatch)
    holding, _ = enricher._enrich_single(_holding(
        security_type="STOCK", instrument_kind_evidence=("STOCK",),
    ))

    assert holding.security_type == "STOCK"
    assert holding.instrument_type == "Stock"
    assert holding.current_value == pytest.approx(105.0)


def test_live_specific_stock_mapping_is_not_hidden_by_an_older_etf_profile(
    monkeypatch,
):
    _provider(monkeypatch)
    price_cache.store_instrument_profile(MONEY_ISIN, _profile())
    monkeypatch.setattr(enricher, "_openfigi_raw", lambda isin: [{
        "data": [{
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
            "marketSector": "Equity",
        }],
    }])

    holding, _ = enricher._enrich_single(_holding())

    assert holding.current_value == pytest.approx(73.0)
    assert holding.data_source == "last-known (instrument kind unavailable)"


@pytest.mark.parametrize("future_profile", [False, True])
def test_pinned_metadata_uses_only_boundary_visible_verified_etf_identity(
    monkeypatch, future_profile,
):
    _provider(monkeypatch)
    boundary = dt.date(2025, 1, 3)
    observed = dt.datetime(
        2025, 1, 4 if future_profile else 2, tzinfo=dt.timezone.utc,
    )
    price_cache.store_instrument_profile(
        MONEY_ISIN, _profile(observed_at=observed),
    )
    runtime.configure(deterministic=True, as_of=boundary)
    monkeypatch.setattr(enricher, "_openfigi_raw", lambda isin: pytest.fail(
        "pinned metadata opened a live transport",
    ))
    monkeypatch.setattr(enricher, "_openfigi_bond_signals", lambda isin: (None, None))
    # The standalone kind reconciliation must obey the same causal boundary
    # even if a past LIVE run left a future ETF observation in memory.
    enricher._openfigi_memo[MONEY_ISIN] = ETF_MAPPING

    info = enricher._reconcile_verified_etf_metadata(MONEY_INFO, _holding())
    kind = enricher._resolve_instrument_kind(info, _holding())

    assert kind is (None if future_profile else InstrumentKind.ETF)
    assert MONEY_INFO["quoteType"] == "EQUITY"
