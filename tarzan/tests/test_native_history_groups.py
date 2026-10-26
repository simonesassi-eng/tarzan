"""Independent native chart parsing and exact per-child cohort ownership."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import replace
from datetime import datetime
from http.cookiejar import Cookie
import json
from threading import Barrier, Event, Lock
import time
from urllib.parse import unquote, urlencode, urlsplit

from curl_cffi.requests import Session
from curl_cffi.requests.models import Request
import pandas as pd
import pytest
from yfinance import cache as sdk_cache
from yfinance.data import YfData

from tarzan.data import _yf_net, enricher as en
from tarzan.data import yahoo_history_groups as groups
from tarzan.runtime import data_quality as dq
from tarzan.runtime.session import current_session
from tarzan.runtime.transport import _operation
from tarzan.tests.test_native_quote_group_recovery import acquisition_owner


SYMBOLS = ("SYNHISTORYA.MI", "SYNHISTORYB.L", "SYNHISTORYC.JO", "SYNHISTORYD")
ZONES = ("Europe/Rome", "Europe/London", "Africa/Johannesburg", "America/New_York")
MARKER = ContextVar("synthetic_history_child_marker", default=None)


@pytest.fixture
def source(monkeypatch):
    calls, paced = [], []
    guard = Lock()
    state = {"missing": set(), "status": {}, "cancel": set(), "delay": 0}
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: paced.append(1))
    YfData.cache_get.cache_clear()
    for symbol, zone in zip(SYMBOLS, ZONES):
        sdk_cache.get_tz_cache().store(symbol, zone)

    def request(client, method, url, params=None, **keywords):
        route = urlsplit(url).path
        if urlsplit(url).hostname == "fc.yahoo.com":
            client.cookies.jar.set_cookie(Cookie(
                version=0, name="A3", value="SYNTHETIC-NOT-A-CREDENTIAL",
                port=None, port_specified=False, domain=".yahoo.com",
                domain_specified=True, domain_initial_dot=True,
                path="/", path_specified=True, secure=False,
                expires=int(time.time()) + 3600, discard=False,
                comment=None, comment_url=None, rest={},
            ))
            document, status = b"synthetic-cookie", 200
        elif route == "/v1/test/getcrumb":
            document, status = b"synthetic-crumb", 200
        else:
            assert route.startswith("/v8/finance/chart/")
            symbol = unquote(route.rsplit("/", 1)[-1])
            index = SYMBOLS.index(symbol)
            with guard:
                calls.append({
                    "symbol": symbol, "parameters": dict(params),
                    "marker": MARKER.get(), "owner": current_session(),
                    "operation": _operation.get(), "timeout": keywords["timeout"],
                })
            if state["delay"]:
                time.sleep(state["delay"])
            if symbol in state["cancel"]:
                raise KeyboardInterrupt("synthetic chart cancellation")
            zone = ZONES[index]
            dates = pd.date_range("2026-10-22", periods=2, tz=zone)
            raw = [100. + index, 102. + index]
            adjusted = [98. + index, 102. + index]
            stamps = [int(day.timestamp()) for day in dates]
            document = {"chart": {"result": [{
                "meta": {
                    "symbol": symbol, "currency": ("EUR", "GBp", "ZAc", "USD")[index],
                    "exchangeTimezoneName": zone, "instrumentType": "EQUITY",
                    "firstTradeDate": stamps[0], "timezone": "UTC",
                    "gmtoffset": 0, "priceHint": 2, "chartPreviousClose": raw[0],
                    "dataGranularity": "1d", "range": "5y",
                    "validRanges": ["1d", "1y", "5y", "max"],
                },
                "timestamp": stamps,
                "indicators": {
                    "quote": [{
                        "open": raw, "high": raw, "low": raw, "close": raw,
                        "volume": [1000, 1000],
                    }],
                    "adjclose": [{"adjclose": adjusted}],
                },
                "events": {"dividends": {
                    str(stamps[1]): {"date": stamps[1], "amount": 2.},
                }},
            }], "error": None}}
            if symbol in state["missing"]:
                document["chart"]["result"] = []
            status = state["status"].get(symbol, 200)
        response = client.response_class()
        response.url = url.rstrip("?") + ("?" + urlencode(params) if params else "")
        response.request = Request(response.url, {}, method)
        response.content = (
            document if isinstance(document, bytes)
            else json.dumps(document if status < 400 else {
                "finance": {"result": None, "error": {"description": "synthetic"}},
            }).encode()
        )
        response.status_code, response.ok, response.encoding = status, status < 400, "utf-8"
        return response

    monkeypatch.setattr(Session, "request", request)
    yield state, calls, paced
    YfData.cache_get.cache_clear()


def parallel(symbols, callback):
    barrier = Barrier(len(symbols), timeout=10)

    def fetch(symbol):
        barrier.wait()
        return callback(symbol)

    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        pending = {
            symbol: pool.submit(copy_context().run, fetch, symbol) for symbol in symbols
        }
        return {symbol: future.result(timeout=15) for symbol, future in pending.items()}


def test_native_raw_rulers_actions_zones_attrs_and_requests_remain_exact(source):
    _state, calls, paced = source
    with acquisition_owner():
        before = {symbol: en._download_price_history(symbol) for symbol in SYMBOLS}
    original_parameters = {item["symbol"]: item["parameters"] for item in calls}
    calls.clear()
    paced.clear()
    with acquisition_owner() as owner, groups.history_group_scope():
        after = parallel(SYMBOLS, en._download_price_history)
        assert not owner.providers.flight._pending
    assert len(paced) == 2
    assert len(calls) == 4
    for symbol in SYMBOLS:
        pd.testing.assert_frame_equal(before[symbol], after[symbol])
        assert before[symbol].attrs == after[symbol].attrs
        assert after[symbol]["Price Close"].iloc[0] != after[symbol]["Close"].iloc[0]
        assert after[symbol]["Dividends"].sum() == 2
    assert {item["symbol"]: item["parameters"] for item in calls} == original_parameters


def test_each_daily_child_keeps_its_original_owner_context_diagnostics_and_deadline(source):
    _state, calls, _paced = source
    contexts = {}
    buffers = {symbol: [] for symbol in SYMBOLS}

    def fetch(symbol):
        token = MARKER.set(symbol)
        dq.defer_into(buffers[symbol])
        try:
            with owner.transport.operation("yahoo"):
                contexts[symbol] = _operation.get()
                return en._download_price_history(symbol)
        finally:
            MARKER.reset(token)

    with acquisition_owner() as owner, groups.history_group_scope():
        results = parallel(SYMBOLS, fetch)
    assert all(not frame.empty for frame in results.values())
    assert all(item["owner"] is owner and item["marker"] == item["symbol"]
               and item["operation"] is contexts[item["symbol"]]
               and 0 < item["timeout"] <= owner.transport.policies["yahoo"].timeout_seconds
               for item in calls)


def test_singleton_scope_keeps_one_original_pace_and_request(source):
    _state, calls, paced = source
    with acquisition_owner(), groups.history_group_scope():
        result = en._download_price_history(SYMBOLS[0])
    assert not result.empty
    assert len(paced) == len(calls) == 1


def test_completed_identical_history_memo_remains_isolated_and_single_flight(source):
    _state, calls, paced = source
    with acquisition_owner(), groups.history_group_scope():
        first = en._download_price_history(SYMBOLS[0])
        first["Close"] = 777
        second = en._download_price_history(SYMBOLS[0])
    assert second["Close"].tolist() == [98., 102.]
    assert len(calls) == len(paced) == 1


@pytest.mark.parametrize("missing", [True, False])
def test_failure_is_per_child_and_inconclusive_history_remains_retryable(source, missing):
    state, calls, _paced = source
    if missing:
        state["missing"].add(SYMBOLS[0])
    else:
        state["status"][SYMBOLS[0]] = 503
    with acquisition_owner(), groups.history_group_scope():
        first = parallel(SYMBOLS[:2], en._download_price_history)
        assert first[SYMBOLS[0]] is None or first[SYMBOLS[0]].empty
        assert not first[SYMBOLS[1]].empty
        state["missing"].clear()
        state["status"].clear()
        recovered = en._download_price_history(SYMBOLS[0])
        assert not recovered.empty
    assert sum(item["symbol"] == SYMBOLS[1] for item in calls) == 1
    assert sum(item["symbol"] == SYMBOLS[0] for item in calls) >= 2


def test_distinct_optional_epochs_are_not_one_paced_cohort(source):
    _state, calls, paced = source

    def fetch(symbol):
        with owner.transport.optional_work("synthetic individual optional epoch"):
            return en._download_price_history(symbol)

    with acquisition_owner() as owner, groups.history_group_scope():
        results = parallel(SYMBOLS, fetch)
    assert all(not frame.empty for frame in results.values())
    assert len(paced) == len(calls) == 4


def test_expired_original_child_operation_cannot_borrow_another_deadline(source):
    _state, calls, _paced = source

    def fetch(symbol):
        with owner.transport.operation("yahoo"):
            active = _operation.get()
            token = None
            if symbol == SYMBOLS[0]:
                token = _operation.set((active[0], active[1], owner.transport.clock() - 1))
            try:
                return en._download_price_history(symbol)
            finally:
                if token is not None:
                    _operation.reset(token)

    with acquisition_owner() as owner, groups.history_group_scope():
        # The native SDK can mask an expired original operation as empty
        # history. It must still start no native request; a healthy peer
        # retains its own independent authority.
        barrier = Barrier(2, timeout=10)
        def run(symbol):
            barrier.wait()
            return fetch(symbol)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {
                symbol: pool.submit(copy_context().run, run, symbol) for symbol in SYMBOLS[:2]
            }
            unavailable = pending[SYMBOLS[0]].result(timeout=15)
            assert unavailable is None or unavailable.empty
            assert not pending[SYMBOLS[1]].result(timeout=15).empty
    assert not any(item["symbol"] == SYMBOLS[0] for item in calls)


def test_unknown_lower_port_declines_cohort_and_keeps_original_scalar_requests(source, monkeypatch):
    _state, calls, paced = source
    original = YfData.get
    monkeypatch.setattr(YfData, "get", lambda *args, **kwargs: original(*args, **kwargs))
    with acquisition_owner(), groups.history_group_scope():
        results = parallel(SYMBOLS, en._download_price_history)
    assert all(not frame.empty for frame in results.values())
    assert len(paced) == len(calls) == 4


def test_fatal_child_cancellation_drains_peers_and_restores_scope(source):
    state, calls, _paced = source
    state["cancel"].add(SYMBOLS[0])
    with acquisition_owner():
        with pytest.raises(KeyboardInterrupt, match="synthetic chart cancellation"):
            with groups.history_group_scope():
                parallel(SYMBOLS[:2], en._download_price_history)
        assert groups._active.get() is None
        assert any(item["symbol"] == SYMBOLS[1] for item in calls)
        state["cancel"].clear()
        assert not en._download_price_history(SYMBOLS[0]).empty


def test_distinct_period_and_start_arguments_remain_distinct_without_merging(source):
    _state, calls, _paced = source
    with acquisition_owner(), groups.history_group_scope():
        en._download_price_history(SYMBOLS[0])
        en._download_price_history(SYMBOLS[0], datetime(2026, 10, 22))
    assert len(calls) == 2
    assert "range" in calls[0]["parameters"]
    assert "period1" in calls[1]["parameters"]


def test_shorter_pacing_authority_does_not_discard_an_independent_healthy_child():
    from tarzan.runtime.transport import ProviderBudgetExceeded

    queued = Event()
    expired = object()

    def pace():
        if MARKER.get() is expired:
            assert queued.wait(timeout=5)
            raise ProviderBudgetExceeded("synthetic one expired pacing caller")

    with acquisition_owner() as owner:
        cohorts = groups._HistoryGroups(owner.transport)

        def early():
            token = MARKER.set(expired)
            try:
                return cohorts.get(lambda: "must-not-load", pace)
            finally:
                MARKER.reset(token)

        def healthy():
            queued.set()
            return cohorts.get(lambda: "healthy-original-child", pace)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(copy_context().run, early)
            while True:
                with cohorts.condition:
                    accepted = bool(cohorts.pending)
                if accepted:
                    break
                time.sleep(.001)
            second = pool.submit(copy_context().run, healthy)
            with pytest.raises(ProviderBudgetExceeded, match="one expired pacing caller"):
                first.result(timeout=10)
            assert second.result(timeout=10) == "healthy-original-child"
        cohorts.close()
