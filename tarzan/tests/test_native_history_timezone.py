"""Native timezone fallback must not depend on unused complementary PEG."""

from hashlib import sha256
import json
from urllib.parse import urlsplit

from curl_cffi.requests import Response
import pytest
import yfinance as yf
import yfinance.base as sdk_base
from yfinance import cache as sdk_cache
from yfinance.data import YfData
from yfinance.scrapers.history import PriceHistory

from tarzan.data import _yf_net, enricher as en, yahoo_info as yi
from tarzan.runtime.session import RunMode
from tarzan.tests.test_native_isin_listing_identity import install_raw_source, owned


LISTING = "SYNTZCAPABILITY.MI"
HISTORY_PORTS = (
    (sdk_base.TickerBase, "history", sdk_base.TickerBase.history),
    (sdk_base.TickerBase, "_lazy_load_price_history", sdk_base.TickerBase._lazy_load_price_history),
    (sdk_base.TickerBase, "_get_ticker_tz", sdk_base.TickerBase._get_ticker_tz),
    (sdk_base.TickerBase, "_fetch_ticker_tz", sdk_base.TickerBase._fetch_ticker_tz),
    (PriceHistory, "__init__", PriceHistory.__init__),
    (PriceHistory, "history", PriceHistory.history),
)


@pytest.fixture
def source(monkeypatch, request):
    # Fresh source identity and LRU for every independent fault schedule.
    listing = "SYNTZ" + sha256(request.node.nodeid.encode()).hexdigest()[:12].upper() + ".MI"
    monkeypatch.setitem(globals(), "LISTING", listing)
    sdk_cache.get_tz_cache().store(LISTING, None)
    assert sdk_cache.get_tz_cache().lookup(LISTING) is None
    YfData.cache_get.cache_clear()
    monkeypatch.setattr(sdk_base, "_tz_info_fetch_ctr", 0)
    calls = install_raw_source(monkeypatch, listing=LISTING)
    native = YfData.get
    state = {"bootstrap_timezone_missing": True, "peg": "available"}
    observations = {"peg_calls": 0, "bootstrap_calls": 0}

    def get(instance, url, params=None, *arguments, **keywords):
        path = urlsplit(url).path
        if "/fundamentals-timeseries/" in path:
            observations["peg_calls"] += 1
            if state["peg"] != "available":
                answer = Response()
                answer.status_code = 503 if state["peg"] == "503" else 200
                answer.ok = answer.status_code == 200
                answer.url, answer.encoding = url, "utf-8"
                answer.content = (
                    b"synthetic malformed JSON"
                    if state["peg"] == "malformed-json"
                    else json.dumps({
                        "timeseries": {
                            "result": None,
                            "error": {"description": "synthetic unused PEG failure"},
                        },
                    }).encode()
                )
                return answer
        answer = native(instance, url, params, *arguments, **keywords)
        if path.startswith("/v8/finance/chart/") and (params or {}).get("range") == "1d":
            observations["bootstrap_calls"] += 1
            if state["bootstrap_timezone_missing"]:
                packet = answer.json()
                packet["chart"]["result"][0]["meta"].pop("exchangeTimezoneName", None)
                answer.content = json.dumps(packet).encode()
        return answer

    # No constructor, metadata/history parser, budget or resolver substitution.
    monkeypatch.setattr(YfData, "get", get)
    yield calls, state, observations
    sdk_cache.get_tz_cache().store(LISTING, None)
    YfData.cache_get.cache_clear()


@pytest.mark.parametrize("peg", [
    "available", "malformed-json", "provider-error", "503",
])
def test_timezone_only_history_does_not_purchase_or_depend_on_unused_peg(source, peg):
    calls, state, observations = source
    state["peg"] = peg
    with owned() as session:
        history = en._download_price_history(LISTING)
        assert not session.providers.flight._pending
    assert history["Close"].tolist() == [100.0, 101.0]
    assert history["Price Close"].tolist() == [100.0, 101.0]
    assert str(history.index.tz) == "Europe/Rome"
    assert observations["bootstrap_calls"] == 1
    assert "/v7/finance/quote" in calls
    assert any(path.startswith("/v10/finance/quoteSummary/") for path in calls)
    assert observations["peg_calls"] == 0


def test_native_chart_timezone_needs_no_metadata_fallback(source):
    calls, state, observations = source
    state["bootstrap_timezone_missing"] = False
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert str(history.index.tz) == "Europe/Rome"
    assert "/v7/finance/quote" not in calls
    assert not any(path.startswith("/v10/finance/quoteSummary/") for path in calls)
    assert observations["peg_calls"] == 0


def test_public_sdk_info_retains_complete_complementary_contract(source):
    _calls, _state, observations = source
    with owned():
        info = _yf_net.ticker(LISTING).info
    assert info["exchangeTimezoneName"] == "Europe/Rome"
    assert info["regularMarketPrice"] == 101.0
    assert "trailingPegRatio" in info
    assert observations["peg_calls"] == 1


def test_existing_timezone_fallback_limit_is_not_reopened(source, monkeypatch):
    calls, _state, observations = source
    monkeypatch.setattr(sdk_base, "_tz_info_fetch_ctr", 2)
    with owned():
        history = en._download_price_history(LISTING)
    # Native later 5y chart can still supply its own timezone without info.
    assert history["Close"].tolist() == [100.0, 101.0]
    assert str(history.index.tz) == "Europe/Rome"
    assert "/v7/finance/quote" not in calls
    assert not any(path.startswith("/v10/finance/quoteSummary/") for path in calls)
    assert observations["peg_calls"] == 0


@pytest.mark.parametrize("version", ["1.1.0", "1.7.1", None])
def test_uninspected_sdk_keeps_complete_timezone_fallback(source, monkeypatch, version):
    _calls, _state, observations = source
    monkeypatch.setattr(yf, "__version__", version)
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


@pytest.mark.parametrize("mode", [RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE])
def test_pinned_history_keeps_complete_contract(source, mode):
    _calls, _state, observations = source
    with owned(mode):
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


def test_standalone_history_keeps_complete_contract(source):
    _calls, _state, observations = source
    history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


def test_injected_client_factory_keeps_complete_contract(source, monkeypatch):
    _calls, _state, observations = source
    native_factory = _yf_net.ticker
    monkeypatch.setattr(_yf_net, "ticker", lambda symbol: native_factory(symbol))
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


@pytest.mark.parametrize("owner,name,original", yi._NATIVE_SDK_PORTS)
def test_changed_core_entry_cannot_elide_timezone_complement(
    source, monkeypatch, owner, name, original,
):
    _calls, _state, observations = source
    replacement = (
        property(lambda self: original.fget(self))
        if isinstance(original, property)
        else lambda *args, **kwargs: original(*args, **kwargs)
    )
    monkeypatch.setattr(owner, name, replacement)
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


@pytest.mark.parametrize("owner,name,original", HISTORY_PORTS)
def test_changed_history_entry_retains_full_metadata_contract(
    source, monkeypatch, owner, name, original,
):
    _calls, _state, observations = source
    monkeypatch.setattr(owner, name, lambda *args, **kwargs: original(*args, **kwargs))
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


@pytest.mark.parametrize("manifest", [None, (), [], {}, (None,) * 6])
def test_invalid_history_manifest_never_authorizes_elision(source, monkeypatch, manifest):
    _calls, _state, observations = source
    monkeypatch.setattr(yi, "_NATIVE_HISTORY_PORTS", manifest, raising=False)
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1


def test_injected_native_history_constructor_alias_retains_full_contract(source, monkeypatch):
    _calls, _state, observations = source
    monkeypatch.setattr(sdk_base, "PriceHistory", lambda *a, **kw: PriceHistory(*a, **kw))
    with owned():
        history = en._download_price_history(LISTING)
    assert history["Close"].tolist() == [100.0, 101.0]
    assert observations["peg_calls"] == 1
