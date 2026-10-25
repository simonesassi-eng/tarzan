"""Native SDK history-only benchmark acquisition, without live I/O."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
import json
from types import SimpleNamespace

from curl_cffi.requests import Response
import pandas as pd
import pytest
import yfinance as yf
from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan import runtime
from tarzan.data import _yf_net, enricher, price_cache
from tarzan.engine import benchmarks
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


@contextmanager
def owned(mode=RunMode.LIVE, native=True):
    now = datetime(2026, 10, 25, 20, 30, tzinfo=timezone.utc)
    context = RunContext("benchmark-capability", mode, None if mode is RunMode.LIVE else now.date(), now)
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
    class DelegatingTransport:
        def __getattr__(self, name):
            return getattr(budget, name)

    transport = budget if native else DelegatingTransport()
    session = RunSession(context, {}, ledger, transport=transport)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            budget.close()


@pytest.fixture
def raw_market(monkeypatch):
    calls = []
    state = {"history_status": 200, "currency": "USD"}
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    YfData.cache_get.cache_clear()

    def get(self, url, params=None, *args, **kwargs):
        params = params or {}
        status = 200
        if "/chart/" in url:
            symbol = url.rsplit("/", 1)[-1]
            operation = "history"
            status = state.get("history_status_by_symbol", {}).get(symbol, state["history_status"])
            timestamps = [
                int(datetime(2026, 10, day, 8, tzinfo=timezone.utc).timestamp())
                for day in (21, 22, 23)
            ]
            document = {"chart": {"result": [{
                "meta": {
                    "symbol": symbol, "currency": state["currency"],
                    "exchangeTimezoneName": "Europe/Rome",
                    "instrumentType": "ETF", "dataGranularity": "1d",
                    "validRanges": ["1d", "5y", "max"], "priceHint": 2,
                    "regularMarketTime": timestamps[-1], "regularMarketPrice": 102.0,
                },
                "timestamp": timestamps,
                "indicators": {
                    "quote": [{
                        "open": [100., 101., 102.], "high": [100., 101., 102.],
                        "low": [100., 101., 102.], "close": [100., 101., 102.],
                        "volume": [1, 1, 1],
                    }],
                    "adjclose": [{"adjclose": [100., 101., 102.]}],
                },
            }], "error": None}}
        elif "/quoteSummary/" in url:
            operation = "summary"
            document = {"quoteSummary": {"result": [{
                "quoteType": {"currency": state["currency"], "quoteType": "ETF"},
                "summaryDetail": {"previousClose": 101.0},
            }], "error": None}}
        elif "/v7/finance/quote?" in url:
            operation = "quote"
            document = {"quoteResponse": {"result": [{
                "symbol": params["symbols"], "currency": state["currency"],
                "regularMarketPrice": 102.0, "previousClose": 101.0,
                "shortName": "Synthetic",
            }], "error": None}}
        elif "/fundamentals-timeseries/" in url:
            operation = "peg"
            document = {"timeseries": {"error": None, "result": []}}
        else:
            pytest.fail("unexpected native SDK request: " + url)
        calls.append(operation)
        response = Response()
        response.content = json.dumps(document if status < 400 else {
            "chart": {"result": None, "error": {"description": "synthetic failure"}},
        }).encode()
        response.status_code = status
        response.ok = 200 <= status < 400
        response.encoding = "utf-8"
        response.reason = ""
        response.url = url
        return response

    monkeypatch.setattr(YfData, "get", get)
    yield calls, state
    YfData.cache_get.cache_clear()


@pytest.mark.parametrize("symbol", ["CAPABILITY.MI", "CAPABILITY.DE", "CAPABILITY.PA"])
def test_native_known_currency_keeps_exact_history_without_unused_metadata(raw_market, symbol):
    calls, _state = raw_market
    with owned():
        assert enricher._owns_history_only_benchmark(symbol) is True
        complete = enricher._fetch_ticker_data(symbol, expected_name="Synthetic")
    assert {"summary", "quote", "history"}.issubset(calls)
    calls.clear()
    with owned():
        data = enricher._fetch_benchmark_data(symbol, expected_name="Synthetic")
    assert calls and set(calls) == {"history"}
    pd.testing.assert_frame_equal(data["history"], complete["history"])
    assert data["history"].attrs == complete["history"].attrs
    assert data[enricher._TICKER_SYMBOL_KEY] == symbol
    assert data["info"] == {}


def test_benchmark_preserves_currency_native_values_and_source_identity(raw_market):
    calls, _state = raw_market
    with owned():
        series = benchmarks._fetch_benchmark_uncached("CAPABILITY.MI")
    assert series.tolist() == [100., 101., 102.]
    assert series.name == "CAPABILITY.MI"
    assert series.attrs[benchmarks._CURRENCY_ATTR] == "EUR"
    assert series.attrs["resolved_ticker"] == "CAPABILITY.MI"
    pd.testing.assert_series_equal(
        series.attrs[benchmarks._NATIVE_ATTR], series.rename("Close"), check_names=True,
        check_flags=False,
    )
    assert set(calls) == {"history"}


@pytest.mark.parametrize("symbol", ["CAPABILITY.L", "CAPABILITY.SW", "CAPABILITY.UNKNOWN", "CAPABILITY"])
def test_unknown_currency_or_bare_alias_keeps_complete_metadata(raw_market, symbol):
    calls, _state = raw_market
    with owned():
        assert enricher._owns_history_only_benchmark(symbol) is False
        data = enricher._fetch_benchmark_data(symbol)
    assert {"summary", "quote", "history"}.issubset(calls)
    assert data["info"]["currency"] == "USD"
    assert data[enricher._TICKER_SYMBOL_KEY] == symbol


@pytest.mark.parametrize("name", [
    "_fetch_ticker_data", "_fetch_ticker_info", "_fetch_history", "venue_currency",
    "_is_expandable_bare_ticker", "_annotate_ticker_data", "_retry", "_space_yf_call",
    "_load_bare_resolution", "_history_visible_at_boundary",
    "_resolve_bare_ticker",
])
def test_injected_entry_keeps_complete_contract(monkeypatch, raw_market, name):
    calls, _state = raw_market
    original = getattr(enricher, name)

    @wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(enricher, name, wrapper)
    with owned():
        assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is False
        data = enricher._fetch_benchmark_data("CAPABILITY.MI")
    assert "info" in data
    assert {"summary", "quote", "history"}.issubset(calls)


def test_preimport_wrapper_cannot_be_its_own_native_certificate(monkeypatch, raw_market):
    original = enricher._fetch_ticker_data

    @wraps(original)
    def wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(enricher, "_fetch_ticker_data", wrapper)
    monkeypatch.setattr(enricher, "_NATIVE_BENCHMARK_DATA_PORTS", (
        wrapper, *enricher._NATIVE_BENCHMARK_DATA_PORTS[1:],
    ))
    with owned():
        assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is False
        assert "info" in enricher._fetch_benchmark_data("CAPABILITY.MI")


@pytest.mark.parametrize("native,version", [(False, "1.7.0"), (True, "1.1.0"), (True, "1.7.1")])
def test_unowned_transport_and_unknown_sdk_keep_full_contract(
    monkeypatch, raw_market, native, version,
):
    calls, _state = raw_market
    monkeypatch.setattr(yf, "__version__", version)
    with owned(native=native):
        assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is False
        assert "info" in enricher._fetch_benchmark_data("CAPABILITY.MI")
    assert {"summary", "quote", "peg", "history"}.issubset(calls)


def test_direct_unowned_call_keeps_public_info(raw_market):
    calls, _state = raw_market
    assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is False
    assert "info" in enricher._fetch_benchmark_data("CAPABILITY.MI")
    assert {"summary", "quote", "peg", "history"}.issubset(calls)


def test_pinned_call_remains_cache_only_without_live_metadata(raw_market):
    calls, _state = raw_market
    with owned(mode=RunMode.POINT_IN_TIME):
        assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is False
        data = enricher._fetch_benchmark_data("CAPABILITY.MI")
    assert calls == []
    assert data["info"] == {}
    assert data["history"].empty


def test_missing_history_does_not_turn_a_current_quote_into_a_benchmark(raw_market):
    calls, state = raw_market
    state["history_status"] = 404
    with owned():
        series = benchmarks._fetch_benchmark_uncached("MISSING.MI")
    assert series.empty
    assert series.attrs[benchmarks._CURRENCY_ATTR] == ""
    # A missing timezone may invoke native SDK metadata fallback AFTER the
    # failed chart. Never disable that recovery to manufacture fewer requests.
    assert calls[0] == "history"


def test_history_only_acquisition_does_not_poison_later_required_metadata(raw_market):
    calls, _state = raw_market
    with owned():
        data = enricher._fetch_benchmark_data("CAPABILITY.MI")
        assert data["history"].shape[0] == 3
        assert set(calls) == {"history"}
        calls.clear()
        info = enricher._fetch_ticker_info("CAPABILITY.MI")
        assert info["regularMarketPrice"] == 102.
        assert calls == ["summary", "quote"]


@pytest.mark.parametrize("mode", ["essential", "optional"])
def test_owned_optional_phase_does_not_change_the_capability(raw_market, mode):
    calls, _state = raw_market
    with owned() as session:
        phase = session.transport.optional_work("benchmark-test") if mode == "optional" else nullcontext()
        with phase:
            assert enricher._owns_history_only_benchmark("CAPABILITY.MI") is True
            assert not enricher._fetch_benchmark_data("CAPABILITY.MI")["history"].empty
    assert set(calls) == {"history"}


@pytest.fixture
def resolution(monkeypatch):
    now = datetime(2026, 10, 25, 20, 30, tzinfo=timezone.utc).timestamp()
    entries = {}
    monkeypatch.setattr(price_cache, "_load_resolution_map", lambda: entries)
    monkeypatch.setattr(price_cache, "time", SimpleNamespace(time=lambda: now))
    return entries, now


def test_current_bare_mapping_reuses_exact_history_without_metadata(raw_market, resolution):
    calls, _state = raw_market
    entries, now = resolution
    entries["TICKER:CAPABILITY"] = {"symbol": "CAPABILITY.MI", "ts": now}
    with owned():
        complete = enricher._fetch_ticker_data("CAPABILITY", expected_name="Synthetic")
    assert {"summary", "quote", "history"}.issubset(calls)
    calls.clear()
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert set(calls) == {"history"}
    assert result[enricher._TICKER_SYMBOL_KEY] == "CAPABILITY.MI"
    pd.testing.assert_frame_equal(result["history"], complete["history"])
    assert result["history"].attrs == complete["history"].attrs
    assert entries["TICKER:CAPABILITY"]["ts"] == now


@pytest.mark.parametrize("mapping", [
    None,
    {"symbol": "CAPABILITY.MI", "ts": 0},
    {"symbol": "OTHER.MI", "ts": "current"},
    {"symbol": "", "ts": "current"},
    {"symbol": 12, "ts": "current"},
])
def test_absent_expired_or_malformed_mapping_keeps_discovery(raw_market, resolution, mapping):
    calls, _state = raw_market
    entries, now = resolution
    if mapping is not None:
        entries["TICKER:CAPABILITY"] = {
            **mapping, "ts": now if mapping["ts"] == "current" else mapping["ts"],
        }
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert {"summary", "quote", "history"}.issubset(calls)
    assert result[enricher._TICKER_SYMBOL_KEY].split(".", 1)[0] == "CAPABILITY"
    assert "info" in result
    assert not result["history"].empty


def test_cached_unknown_currency_keeps_metadata_and_currency(raw_market, resolution):
    calls, _state = raw_market
    entries, now = resolution
    entries["TICKER:CAPABILITY"] = {"symbol": "CAPABILITY.L", "ts": now}
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert result[enricher._TICKER_SYMBOL_KEY] == "CAPABILITY.L"
    assert result["info"]["currency"] == "USD"
    assert {"summary", "quote", "history"}.issubset(calls)


def test_cached_alias_without_expected_identity_name_keeps_original_bare_request(
    raw_market, resolution,
):
    calls, _state = raw_market
    entries, now = resolution
    entries["TICKER:CAPABILITY"] = {"symbol": "CAPABILITY.MI", "ts": now}
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY")
    assert result[enricher._TICKER_SYMBOL_KEY] == "CAPABILITY"
    assert {"summary", "quote", "history"}.issubset(calls)


def test_cache_change_does_not_add_a_second_resolution_read(monkeypatch, raw_market):
    reads = []

    def evolving_cache(key):
        reads.append(key)
        return None if len(reads) == 1 else "CAPABILITY.DE"

    monkeypatch.setattr(price_cache, "load_resolution", evolving_cache)
    with owned():
        complete = enricher._fetch_ticker_data("CAPABILITY", expected_name="Synthetic")
    expected_symbol = complete[enricher._TICKER_SYMBOL_KEY]
    expected_reads = len(reads)
    reads.clear()
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert len(reads) == expected_reads == 1
    assert result[enricher._TICKER_SYMBOL_KEY] == expected_symbol


def test_empty_cached_listing_still_discovers_working_alternate_exchange(raw_market, resolution):
    calls, state = raw_market
    entries, now = resolution
    entries["TICKER:CAPABILITY"] = {"symbol": "CAPABILITY.MI", "ts": now}
    state["history_status"] = 404
    state["history_status_by_symbol"] = {"CAPABILITY.DE": 200}
    with owned():
        result = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert result[enricher._TICKER_SYMBOL_KEY] == "CAPABILITY.DE"
    assert not result["history"].empty
    assert "info" in result
    assert {"summary", "quote", "history"}.issubset(calls)


def test_transient_history_absence_does_not_persist_a_negative_capability(raw_market, resolution):
    calls, state = raw_market
    entries, now = resolution
    entries["TICKER:CAPABILITY"] = {"symbol": "CAPABILITY.MI", "ts": now}
    state["history_status"] = 404
    with owned():
        absent = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert absent["history"].empty
    state["history_status"] = 200
    calls.clear()
    YfData.cache_get.cache_clear()
    with owned():
        recovered = enricher._fetch_benchmark_data("CAPABILITY", expected_name="Synthetic")
    assert recovered[enricher._TICKER_SYMBOL_KEY] == "CAPABILITY.MI"
    assert not recovered["history"].empty
    assert set(calls) == {"history"}


def test_ambiguous_listing_retains_native_fx_conversion(monkeypatch, raw_market):
    calls, _state = raw_market
    fx = pd.Series(
        [.9, .9, .9],
        index=pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"])
        .tz_localize("Europe/Rome").tz_convert("UTC"),
    )
    currencies = []

    def rates(currency):
        currencies.append(currency)
        return fx

    monkeypatch.setattr(enricher, "_get_fx_series", rates)
    with owned():
        series = benchmarks._fetch_benchmark_uncached("CAPABILITY.L")
    assert series.tolist() == pytest.approx([90., 90.9, 91.8])
    assert series.attrs[benchmarks._CURRENCY_ATTR] == "USD"
    assert series.attrs[benchmarks._NATIVE_ATTR].tolist() == [100., 101., 102.]
    assert currencies == ["USD"]
    assert {"summary", "quote", "history"}.issubset(calls)
