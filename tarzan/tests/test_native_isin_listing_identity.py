"""Native SDK ISIN search must not lose the actual listing in source evidence."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
from urllib.parse import unquote, urlsplit
from types import SimpleNamespace
from threading import Barrier

from curl_cffi.requests import Response
import pandas as pd
import pytest
import yfinance as yf
from yfinance import cache as sdk_cache
from yfinance.data import YfData

from tarzan import runtime
from tarzan.data import enricher as en
from tarzan.data import _yf_net, price_cache, yahoo_info as yi
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


ISIN = "XS0000000099"
LISTING = "SYNSDK.MI"
CAPTURED = datetime(2026, 10, 26, 6, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def independent_synthetic_mapping():
    # The shared suite already points every SDK database at temporary storage.
    # Each case declares independent source truth; no previous parameter's
    # synthetic mapping or process-wide raw-response LRU may replace that truth.
    sdk_cache.get_isin_cache().store(ISIN, None)
    YfData.cache_get.cache_clear()
    yield
    sdk_cache.get_isin_cache().store(ISIN, None)
    YfData.cache_get.cache_clear()


@contextmanager
def owned(mode=RunMode.LIVE, *, native=True):
    context = RunContext(
        "synthetic-native-isin", mode,
        None if mode is RunMode.LIVE else CAPTURED.date(), CAPTURED,
    )
    budget = ProviderBudget(ProviderTransportSettings.configured())
    owner = RunSession(
        context, {}, RunLedger(context.attempt_id),
        transport=budget if native else SimpleNamespace(yahoo_client=budget.yahoo_client),
    )
    with runtime.bind_context(context), activate_session(owner):
        try:
            yield owner
        finally:
            budget.close()


def install_raw_source(monkeypatch, *, listing=LISTING, currency="EUR", state=None):
    calls = []
    state = {} if state is None else state
    index = pd.to_datetime(["2026-10-22", "2026-10-23"])
    stamps = [int(point.tz_localize("Europe/Rome").timestamp()) for point in index]
    quote = {
        "symbol": listing, "longName": "Synthetic native constructor",
        "shortName": "Synthetic native constructor", "quoteType": "EQUITY",
        "currency": currency, "exchange": "MIL",
        "exchangeTimezoneName": "Europe/Rome",
        "regularMarketPrice": 101.0, "previousClose": 100.0,
        "regularMarketPreviousClose": 100.0,
        "regularMarketTime": int(CAPTURED.timestamp()),
    }

    def quote_for_symbol(symbol):
        if symbol == listing:
            return quote
        alternate = state.get("alternate_quotes", {}).get(symbol)
        assert alternate is not None, "unexpected synthetic listing"
        return {**quote, **alternate, "symbol": symbol}

    def get(client, url, params=None, timeout=30, **kwargs):
        path = urlsplit(url).path
        calls.append(path)
        if path == "/v1/finance/search":
            assert params["q"] == ISIN
            body = {"quotes": [{
                "symbol": listing, "quoteType": "EQUITY", "exchange": "MIL",
                "shortname": quote["shortName"], "longname": quote["longName"],
            }], "news": []}
            if state.get("search_missing"):
                body["quotes"] = []
        elif path.startswith("/v10/finance/quoteSummary/"):
            requested = unquote(path.rsplit("/", 1)[-1])
            metadata = quote_for_symbol(requested)
            if state.get("metadata_raises"):
                raise ValueError("synthetic native metadata is unavailable")
            body = {"quoteSummary": {"result": [{"symbol": requested, "price": metadata}],
                                     "error": None}}
            if state.get("metadata_missing"):
                body["quoteSummary"]["result"] = []
        elif path == "/v7/finance/quote":
            metadata = quote_for_symbol(params["symbols"])
            body = {"quoteResponse": {"result": [metadata], "error": None}}
            if state.get("metadata_missing"):
                body["quoteResponse"]["result"] = []
        elif "/fundamentals-timeseries/" in path:
            body = {"timeseries": {"result": [], "error": None}}
        elif path.startswith("/v8/finance/chart/"):
            metadata = quote_for_symbol(unquote(path.rsplit("/", 1)[-1]))
            body = {"chart": {"result": [{
                "meta": {
                    **metadata, "instrumentType": "EQUITY",
                    "firstTradeDate": stamps[0], "timezone": "CEST",
                    "gmtoffset": 7200, "priceHint": 2,
                    "chartPreviousClose": 100.0, "dataGranularity": "1d",
                    "range": "5y", "validRanges": ["1d", "5y", "max"],
                },
                "timestamp": stamps,
                "indicators": {
                    "quote": [{
                        "open": [100.0, 101.0], "high": [100.0, 101.0],
                        "low": [100.0, 101.0], "close": [100.0, 101.0],
                        "volume": [1000, 1000],
                    }],
                    "adjclose": [{"adjclose": [100.0, 101.0]}],
                },
                "events": {},
            }], "error": None}}
            if state.get("history_missing"):
                body["chart"]["result"] = []
        else:
            raise AssertionError("unexpected synthetic native request shape")
        answer = Response()
        answer.content = json.dumps(body).encode()
        answer.status_code, answer.ok, answer.url = 200, True, url
        answer.encoding = "utf-8"
        return answer

    # Actual native Ticker/Search/Quote/history parsers consume raw responses.
    # No client, financial helper, resolver or output metadata is substituted.
    # This exercises no physical HTTP/cookie/crumb recovery.
    monkeypatch.setattr(YfData, "get", get)
    return calls


def test_native_isin_constructor_records_the_listing_it_actually_consumed(monkeypatch):
    calls = install_raw_source(monkeypatch)
    with owned():
        data = en._fetch_ticker_data(ISIN)
    assert "/v1/finance/search" in calls
    assert data["info"]["symbol"] == LISTING
    assert en._history_has_market_evidence(data["history"])
    assert data[en._TICKER_SYMBOL_KEY] == LISTING
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING


def test_native_isin_history_retains_the_actual_source_listing(monkeypatch):
    install_raw_source(monkeypatch)
    with owned():
        data = en._fetch_ticker_data(ISIN)
    assert data["info"]["symbol"] == LISTING
    assert en._history_has_market_evidence(data["history"])
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING


@pytest.mark.parametrize("listing,currency", [
    ("SYNATIVE.MI", "EUR"), ("SYNATIVE.DE", "EUR"),
    ("SYNATIVE.L", "GBp"), ("SYNATIVE.SW", "CHF"),
    ("SYNATIVE", "USD"),
])
def test_actual_constructor_listing_and_currency_remain_coherent(monkeypatch, listing, currency):
    calls = install_raw_source(monkeypatch, listing=listing, currency=currency)
    with owned():
        data = en._fetch_ticker_data(ISIN)
    assert calls.count("/v1/finance/search") == 1
    assert data[en._TICKER_SYMBOL_KEY] == listing
    assert data["info"]["currency"] == currency
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == listing
    assert en._history_has_market_evidence(data["history"])


def test_candidate_ranking_uses_the_native_actual_listing_without_a_second_search(monkeypatch):
    calls = install_raw_source(monkeypatch)
    with owned():
        candidate = en._fetch_candidate_meta(ISIN)
    assert calls.count("/v1/finance/search") <= 1
    assert candidate.symbol == LISTING
    assert candidate.currency == "EUR"
    assert candidate.price == 101.0
    assert not any(path.startswith("/v8/finance/chart/") for path in calls)


def test_constructor_identity_is_separate_from_metadata_fields_and_serialization(monkeypatch):
    install_raw_source(monkeypatch)
    with owned():
        info = en._fetch_ticker_info(ISIN)
        copied = en._fetch_ticker_info(ISIN)
    assert type(info) is yi._NativeListingInfo
    assert yi.source_symbol_for_info(ISIN, deepcopy(info)) == LISTING
    assert yi.source_symbol_for_info(ISIN, copied) == LISTING
    copied["symbol"] = "CONFLICTING.DE"
    copied["currency"] = "USD"
    assert info["currency"] == "EUR"
    assert yi.source_symbol_for_info(ISIN, copied) == LISTING
    assert yi.source_symbol_for_info(ISIN, dict(info)) == ISIN
    assert json.loads(json.dumps(info)) == dict(info)
    assert set(info) == set(dict(info))
    with pytest.raises(FrozenInstanceError):
        info._source_identity.selected = "CONFLICTING.DE"


@pytest.mark.parametrize("payload", [
    {"symbol": LISTING}, {"_tarzan_selected_ticker": LISTING},
    {"symbol": LISTING, "currency": "EUR", "regularMarketPrice": 101.0},
])
def test_plain_injected_metadata_never_promotes_a_symbol(payload):
    assert yi.source_symbol_for_info(ISIN, payload) == ISIN
    assert yi.source_symbol_for_info("PINNED.MI", payload) == "PINNED.MI"


@pytest.mark.parametrize("identity", [
    None, {"requested": ISIN, "selected": LISTING},
    yi._NativeListingIdentity("XS0000000088", LISTING),
    yi._NativeListingIdentity(ISIN, ""),
    yi._NativeListingIdentity(ISIN, " UNTRIMMED.MI"),
    yi._NativeListingIdentity(ISIN, None),
])
def test_malformed_or_different_request_evidence_cannot_promote(identity):
    info = yi._NativeListingInfo({"symbol": LISTING}, identity)
    assert yi.source_symbol_for_info(ISIN, info) == ISIN


def test_known_mapping_remains_useful_when_current_metadata_is_missing(monkeypatch):
    calls = install_raw_source(monkeypatch, state={"metadata_missing": True})
    with owned():
        data = en._fetch_ticker_data(ISIN)
    assert calls.count("/v1/finance/search") <= 1
    assert data["info"] == {}
    assert data[en._TICKER_SYMBOL_KEY] == LISTING
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING
    assert en._history_has_market_evidence(data["history"])


def test_standalone_native_history_records_the_listing_it_actually_consumed(monkeypatch):
    install_raw_source(monkeypatch)
    with owned():
        history = en._fetch_history(ISIN)
    assert en._history_has_market_evidence(history)
    assert history.attrs[en._HISTORY_SYMBOL_ATTR] == LISTING
    assert "_tarzan_native_constructor_identity" not in history.attrs


@pytest.mark.parametrize("requested", [ISIN, ISIN.lower()])
def test_cached_native_mapping_needs_no_extra_search(monkeypatch, requested):
    sdk_cache.get_isin_cache().store(ISIN, LISTING)
    calls = install_raw_source(monkeypatch)
    with owned():
        data = en._fetch_ticker_data(requested)
    assert "/v1/finance/search" not in calls
    assert data[en._TICKER_SYMBOL_KEY] == LISTING
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING


def test_explicit_qualified_listing_is_never_promoted(monkeypatch):
    calls = install_raw_source(monkeypatch, listing="EXPLICIT.DE")
    with owned():
        data = en._fetch_ticker_data("EXPLICIT.DE", expected_name="Synthetic native constructor")
    assert "/v1/finance/search" not in calls
    assert type(data["info"]) is dict
    assert data[en._TICKER_SYMBOL_KEY] == "EXPLICIT.DE"
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == "EXPLICIT.DE"


@pytest.mark.parametrize("mode,native", [
    (RunMode.POINT_IN_TIME, True), (RunMode.REPRODUCIBLE, True),
    (RunMode.LIVE, False),
])
def test_pinned_and_unowned_metadata_keeps_the_full_original_contract(monkeypatch, mode, native):
    install_raw_source(monkeypatch)
    with owned(mode, native=native):
        info = yi.info_for_enrichment(ISIN)
        assert yi.is_native_isin_request(ISIN) is False
    assert type(info) is dict
    assert info["symbol"] == LISTING
    assert yi.source_symbol_for_info(ISIN, info) == ISIN


@pytest.mark.parametrize("version", ["1.1.0", "1.7.1", None])
def test_unknown_sdk_keeps_plain_metadata_and_original_symbol(monkeypatch, version):
    install_raw_source(monkeypatch)
    monkeypatch.setattr(yf, "__version__", version)
    with owned():
        info = yi.info_for_enrichment(ISIN)
        assert yi.is_native_isin_request(ISIN) is False
    assert type(info) is dict
    assert yi.source_symbol_for_info(ISIN, info) == ISIN


def test_injected_factory_cannot_acquire_native_source_authority(monkeypatch):
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: SimpleNamespace(
        ticker=LISTING, info={"symbol": LISTING, "currency": "EUR"},
    ))
    with owned():
        info = yi.info_for_enrichment(ISIN)
        assert yi.is_native_isin_request(ISIN) is False
    assert type(info) is dict
    assert yi.source_symbol_for_info(ISIN, info) == ISIN


@pytest.mark.parametrize("port", [
    "_fetch_ticker_data", "_fetch_ticker_info", "_fetch_history",
    "_fetch_history_uncached", "_download_price_history", "_space_yf_call", "_retry",
])
def test_modified_enrichment_ports_cannot_skip_the_original_resolution(monkeypatch, port):
    original = getattr(en, port)
    monkeypatch.setattr(en, port, lambda *args, **kwargs: original(*args, **kwargs))
    with owned():
        assert en._owns_native_isin_data_request(ISIN) is False


def test_raw_isin_cache_is_not_mixed_into_a_different_native_listing(monkeypatch, tmp_path):
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "0")
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "synthetic-market-cache"))
    incompatible = pd.DataFrame({
        "Close": [900.0, 901.0], "Price Close": [900.0, 901.0],
    }, index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    price_cache.store_history(ISIN, incompatible)
    install_raw_source(monkeypatch)
    with owned():
        history = en._fetch_history(ISIN)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert history.attrs[en._HISTORY_SYMBOL_ATTR] == LISTING
    assert price_cache.load_history(ISIN)["Close"].tolist() == [900.0, 901.0]
    assert price_cache.load_history(LISTING)["Close"].tolist() == [100.0, 101.0]


def test_missing_native_history_uses_only_the_actual_listing_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "0")
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "synthetic-market-cache"))
    cached = pd.DataFrame({
        "Close": [100.0, 101.0], "Price Close": [100.0, 101.0],
    }, index=pd.to_datetime(["2026-10-22", "2026-10-23"]))
    price_cache.store_history(LISTING, cached)
    sdk_cache.get_tz_cache().store(LISTING, "Europe/Rome")
    install_raw_source(monkeypatch, state={"history_missing": True})
    with owned():
        history = en._fetch_history(ISIN)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert history.attrs[en._HISTORY_SYMBOL_ATTR] == LISTING
    assert price_cache.load_history(ISIN) is None


def test_metadata_failure_does_not_erase_positive_constructor_identity(monkeypatch):
    calls = install_raw_source(monkeypatch, state={"metadata_raises": True})
    with owned():
        data = en._fetch_ticker_data(ISIN)
    assert data["info"] == {}
    assert data[en._TICKER_SYMBOL_KEY] == LISTING
    assert en._history_has_market_evidence(data["history"])
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING
    assert calls.count("/v1/finance/search") == 1


def test_unavailable_search_has_no_positive_or_persisted_negative_identity(monkeypatch):
    calls = install_raw_source(monkeypatch, state={"search_missing": True})
    with owned() as owner:
        data = en._fetch_ticker_data(ISIN)
        assert owner.providers.stores.get(yi._LISTING_FACTS, {}) == {}
        assert yi.completed_constructor_source(ISIN) == ISIN
    assert calls.count("/v1/finance/search") == 1
    assert data[en._TICKER_SYMBOL_KEY] == ISIN
    assert data["info"] == {}
    assert data["history"].empty
    assert sdk_cache.get_isin_cache().lookup(ISIN) is None


def test_completed_identity_does_not_leak_to_a_later_owner(monkeypatch):
    calls = install_raw_source(monkeypatch)
    with owned():
        en._fetch_ticker_info(ISIN)
        assert yi.completed_constructor_source(ISIN) == LISTING
    before = len(calls)
    with owned() as owner:
        assert yi.completed_constructor_source(ISIN) == ISIN
        assert yi._LISTING_FACTS not in owner.providers.stores
    assert len(calls) == before


def test_completed_identity_can_be_used_after_native_io_budget_expiry(monkeypatch):
    calls = install_raw_source(monkeypatch)
    with owned() as owner:
        en._fetch_ticker_info(ISIN)
        before = len(calls)
        owner.transport.deadline = owner.transport.clock() - 1
        assert yi.completed_constructor_source(ISIN) == LISTING
        assert len(calls) == before


@pytest.mark.parametrize("workers", [8, 32])
def test_parallel_same_isin_is_single_flight_with_defensive_source_copies(monkeypatch, workers):
    calls = install_raw_source(monkeypatch)
    barrier = Barrier(workers)

    def fetch():
        barrier.wait(timeout=10)
        return en._fetch_ticker_data(ISIN)

    with owned(), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_context().run, fetch) for _ in range(workers)]
        data = [future.result(timeout=20) for future in futures]
    assert calls.count("/v1/finance/search") == 1
    assert len([path for path in calls if path.startswith("/v10/finance/quoteSummary/")]) == 1
    assert calls.count("/v7/finance/quote") == 1
    assert all(item[en._TICKER_SYMBOL_KEY] == LISTING for item in data)
    assert all(item["history"].attrs[en._HISTORY_SYMBOL_ATTR] == LISTING for item in data)
    data[0]["info"]["currency"] = "USD"
    data[0]["history"].iloc[0, data[0]["history"].columns.get_loc("Close")] = 999.0
    assert data[1]["info"]["currency"] == "EUR"
    assert data[1]["history"]["Close"].iloc[0] == 100.0


@pytest.mark.parametrize("native_search_missing", [False, True])
def test_isin_constructor_does_not_hide_a_better_real_qualified_listing(monkeypatch, native_search_missing):
    qualified = ISIN + ".MI"
    calls = install_raw_source(monkeypatch, listing="NATIVE.DE", currency="USD", state={
        "search_missing": native_search_missing,
        "alternate_quotes": {qualified: {"currency": "EUR"}},
    })
    with owned():
        data = en._fetch_ticker_data(
            ISIN, expected_name="Synthetic native constructor", expected_currency="EUR",
        )
    assert "/v10/finance/quoteSummary/" + qualified in calls
    assert data[en._TICKER_SYMBOL_KEY] == qualified
    assert data["info"]["currency"] == "EUR"
    assert data["history"].attrs[en._HISTORY_SYMBOL_ATTR] == qualified
