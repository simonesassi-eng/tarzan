"""Exercise the installed core metadata parser, not a replacement info dict."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Lock
from types import SimpleNamespace

from curl_cffi.requests import Response
import pytest
import yfinance as yf
from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan import runtime
from tarzan.data import _yf_net, enricher, yahoo_info as yi
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


@contextmanager
def owned(name="core-info", mode=RunMode.LIVE, native=True):
    now = datetime.now(timezone.utc)
    context = RunContext(name, mode, None if mode is RunMode.LIVE else now.date(), now)
    ledger = RunLedger(name)
    budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
    transport = budget if native else SimpleNamespace(yahoo_client=budget.yahoo_client)
    session = RunSession(context, {}, ledger, transport=transport)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            budget.close()


@pytest.fixture
def raw_metadata(monkeypatch):
    calls = []
    state = {"currency": "EUR", "type": "ETF", "summary_status": 200, "quote_status": 200,
             "peg_status": 200}
    lock = Lock()
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)
    # The SDK owns a process-wide response LRU; each fault schedule must be
    # independent of whichever parameterized test ran before it.
    YfData.cache_get.cache_clear()

    def get(self, url, params=None, *args, **kwargs):
        if "/quoteSummary/" in url:
            operation, status = "summary", state["summary_status"]
            symbol = url.rsplit("/", 1)[-1]
            document = {"quoteSummary": {"result": [{
                "symbol": symbol,
                "assetProfile": {"country": "Ireland", "companyOfficers": [{"name": "Synthetic"}]},
                "quoteType": {"currency": state["currency"], "quoteType": state["type"]},
                "financialData": {"currentPrice": {"raw": 100.5, "fmt": "100.50"}},
                "summaryDetail": {"yield": {"raw": .03, "fmt": "3%"}, "maxAge": 1},
            }], "error": None}}
        elif "/v7/finance/quote?" in url:
            operation, status = "quote", state["quote_status"]
            symbol = params["symbols"]
            document = {"quoteResponse": {"result": [{
                "symbol": symbol, "currency": state["currency"], "quoteType": state["type"],
                "regularMarketPrice": 101., "regularMarketPreviousClose": 100.,
                "previousClose": 100., "regularMarketTime": 1792749600,
                "exchangeTimezoneName": "Europe/Berlin", "shortName": "Synthetic",
            }], "error": None}}
        elif "/fundamentals-timeseries/" in url:
            operation, status = "peg", state["peg_status"]
            document = {"timeseries": {"error": None, "result": [{
                "trailingPegRatio": [{"reportedValue": {"raw": 2.25}}],
            }]}}
        else:
            pytest.fail("unexpected native metadata request: " + url)
        with lock:
            calls.append(operation)
        if status >= 400:
            document = {"finance": {"result": None, "error": {"description": "synthetic failure"}}}
        response = Response()
        response.content = json.dumps(document).encode()
        response.status_code = status
        response.ok = 200 <= status < 400
        response.encoding = "utf-8"
        response.url = url
        response.reason = ""
        return response

    monkeypatch.setattr(YfData, "get", get)
    yield calls, state
    YfData.cache_get.cache_clear()


@pytest.mark.parametrize("currency,kind", [
    ("EUR", "ETF"), ("USD", "EQUITY"), ("GBp", "ETF"), ("CHF", "BOND"),
])
def test_native_core_preserves_every_noncomplementary_field(raw_metadata, currency, kind):
    calls, state = raw_metadata
    state.update(currency=currency, type=kind)
    complete = _yf_net.ticker("COREPROOF").info
    assert calls == ["summary", "quote", "peg"]
    calls.clear()
    with owned():
        assert yi._native_owned_core(_yf_net.ticker("COREPROOF")) is True
        core = enricher._fetch_ticker_info("COREPROOF")
    assert calls == ["summary", "quote"]
    assert core == {key: value for key, value in complete.items() if key != "trailingPegRatio"}
    assert core["currency"] == currency and core["quoteType"] == kind
    assert core["yield"] == .03 and core["maxAge"] == 86400
    assert core["exchangeTimezoneName"] == "Europe/Berlin"


def test_summary_404_still_recovers_from_the_native_quote_endpoint(raw_metadata):
    calls, state = raw_metadata
    state["summary_status"] = 404
    with owned():
        core = enricher._fetch_ticker_info("RECOVERY.MI")
    assert calls == ["summary", "quote"]
    assert core["regularMarketPrice"] == 101.
    assert core["currency"] == "EUR"


def test_unused_peg_failure_does_not_erase_valid_core_metadata(raw_metadata):
    calls, state = raw_metadata
    state["peg_status"] = 503
    with owned():
        core = enricher._fetch_ticker_info("COREAVAILABLE.MI")
    assert core["regularMarketPrice"] == 101. and core["currency"] == "EUR"
    assert calls == ["summary", "quote"]


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_failed_core_sources_do_not_invent_available_prices(raw_metadata, status):
    calls, state = raw_metadata
    state.update(summary_status=status, quote_status=status)
    with owned():
        assert enricher._fetch_ticker_info("UNAVAILABLE.MI") == {}
    assert calls == ["summary", "quote"]


@pytest.mark.parametrize("version", ["0.2.99", "1.1.1", "1.7.0", None])
def test_uninspected_sdk_version_keeps_the_full_info_contract(monkeypatch, raw_metadata, version):
    calls, _state = raw_metadata
    monkeypatch.setattr(yf, "__version__", version)
    with owned():
        assert yi._native_owned_core(_yf_net.ticker("UNKNOWNVERSION")) is False
        info = enricher._fetch_ticker_info("UNKNOWNVERSION")
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]


@pytest.mark.parametrize("mode,native", [
    (RunMode.LIVE, False), (RunMode.POINT_IN_TIME, True), (RunMode.REPRODUCIBLE, True),
])
def test_unowned_or_pinned_caller_keeps_the_full_info_contract(raw_metadata, mode, native):
    calls, _state = raw_metadata
    with owned(mode=mode, native=native):
        assert yi._native_owned_core(_yf_net.ticker("UNOWNED")) is False
        info = yi.info_for_enrichment("UNOWNED")
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]


def test_direct_helper_and_public_sdk_client_retain_complementary_fields(raw_metadata):
    calls, _state = raw_metadata
    info = yi.info_for_enrichment("DIRECT")
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]
    calls.clear()
    with owned():
        assert yi.info_for_enrichment("OWNED")["regularMarketPrice"] == 101.
        assert _yf_net.ticker("PUBLIC").info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "summary", "quote", "peg"]


def test_injected_factory_preserves_its_entire_info_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: calls.append(symbol) or SimpleNamespace(
        info={"custom": {"required": True}, "trailingPegRatio": 7.},
    ))
    with owned():
        assert yi.info_for_enrichment("INJECTED") == {
            "custom": {"required": True}, "trailingPegRatio": 7.,
        }
    assert calls == ["INJECTED"]


def test_injected_quote_constructor_alias_preserves_required_complementary_fields(monkeypatch, raw_metadata):
    import yfinance.base as sdk_base

    calls, _state = raw_metadata
    original = sdk_base.Quote

    def construct(*args, **kwargs):
        quote = original(*args, **kwargs)
        complement = quote._fetch_complementary

        def required_fields():
            complement()
            quote._info["caller_required_field"] = "preserved"

        quote._fetch_complementary = required_fields
        return quote

    monkeypatch.setattr(sdk_base, "Quote", construct)
    with owned():
        assert yi._native_owned_core(_yf_net.ticker("CONSTRUCTORALIAS")) is False
        info = yi.info_for_enrichment("CONSTRUCTORALIAS")
    assert info["caller_required_field"] == "preserved"
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]


@pytest.mark.parametrize("owner,name,original", yi._NATIVE_SDK_PORTS)
def test_modified_sdk_entry_keeps_the_full_native_path(monkeypatch, raw_metadata, owner, name, original):
    calls, _state = raw_metadata
    if isinstance(original, property):
        replacement = property(lambda self: original.fget(self))
    else:
        replacement = lambda *args, **kwargs: original(*args, **kwargs)
    monkeypatch.setattr(owner, name, replacement)
    with owned():
        assert yi._native_owned_core(_yf_net.ticker("INJECTEDSDK")) is False
        info = yi.info_for_enrichment("INJECTEDSDK")
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]


def test_same_listing_concurrent_miss_is_single_flight_and_isolated(raw_metadata):
    calls, _state = raw_metadata
    barrier = Barrier(8)

    def fetch():
        barrier.wait(timeout=5)
        return enricher._fetch_ticker_info("PARALLEL.MI")

    with owned(), ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result(timeout=5) for future in [
            pool.submit(copy_context().run, fetch) for _ in range(8)
        ]]
        results[0]["companyOfficers"][0]["name"] = "Changed"
        assert results[1]["companyOfficers"][0]["name"] == "Synthetic"
    assert calls == ["summary", "quote"]


def test_distinct_venues_and_subsequent_runs_remain_distinct(raw_metadata):
    calls, _state = raw_metadata
    with owned("first"):
        assert enricher._fetch_ticker_info("VENUE.MI")["symbol"] == "VENUE.MI"
        assert enricher._fetch_ticker_info("VENUE.DE")["symbol"] == "VENUE.DE"
    with owned("second"):
        assert enricher._fetch_ticker_info("VENUE.MI")["symbol"] == "VENUE.MI"
    assert calls == ["summary", "quote"] * 3


@pytest.mark.parametrize("bad", [(), [], None, (None,) * 10])
def test_invalid_manifest_does_not_authorize_a_core_only_request(monkeypatch, raw_metadata, bad):
    calls, _state = raw_metadata
    monkeypatch.setattr(yi, "_NATIVE_SDK_PORTS", bad)
    with owned():
        info = yi.info_for_enrichment("MANIFEST")
    assert info["trailingPegRatio"] == 2.25
    assert calls == ["summary", "quote", "peg"]


@pytest.mark.parametrize("count", [1, 10, 100])
def test_request_work_scales_with_distinct_metadata_not_unused_fields(raw_metadata, count):
    calls, _state = raw_metadata
    with owned():
        for index in range(count):
            symbol = f"CORELOAD{index:04d}.MI"
            assert enricher._fetch_ticker_info(symbol)["symbol"] == symbol
            assert enricher._fetch_ticker_info(symbol)["regularMarketPrice"] == 101.
    assert calls == ["summary", "quote"] * count


@pytest.mark.parametrize("port", ["factory", "quote_constructor", "complementary"])
def test_preimport_wrappers_do_not_certify_themselves_as_native(port):
    script = """
import functools
import importlib
import socket
from datetime import datetime, timezone
from curl_cffi.requests import Session
from yfinance.scrapers.quote import Quote
from tarzan import runtime
from tarzan.data import _yf_net, yahoo_info
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings
def denied(*args, **kwargs):
    raise AssertionError("network denied")
socket.socket.connect = denied
socket.create_connection = denied
Session.request = denied
module = _yf_net if PORT == "factory" else Quote
attribute = {"factory": "ticker", "quote_constructor": "__init__",
             "complementary": "_fetch_complementary"}[PORT]
original = getattr(module, attribute)
@functools.wraps(original)
def wrapper(*args, **kwargs):
    return original(*args, **kwargs)
setattr(module, attribute, wrapper)
importlib.reload(yahoo_info)
context = RunContext("import-order", RunMode.LIVE, None, datetime.now(timezone.utc))
ledger = RunLedger(context.attempt_id)
budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
session = RunSession(context, {}, ledger, transport=budget)
with runtime.bind_context(context), activate_session(session):
    print(yahoo_info._native_owned_core(_yf_net.ticker("PREIMPORT")))
budget.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", "PORT = " + repr(port) + "\n" + script],
        cwd=Path(__file__).resolve().parents[2], check=True,
        capture_output=True, text=True, timeout=20,
    )
    assert result.stdout.strip() == "False"
