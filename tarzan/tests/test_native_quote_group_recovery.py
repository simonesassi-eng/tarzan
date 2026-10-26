"""Full native metadata and causally owned group recovery, not warning removal."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from http.cookiejar import Cookie
import json
import time
from threading import Barrier
from urllib.parse import urlencode, urlsplit

from curl_cffi.requests import Session
from curl_cffi.requests.models import Request
import pytest
from yfinance.data import YfData

from tarzan.data import _yf_net, enricher, yahoo_quote_batch as batch
from tarzan.data.yahoo_info_groups import metadata_group_scope
from tarzan.runtime import quote_recovery as recovery
from tarzan.runtime.ledger import LedgerEntryType
from tarzan.tests.test_native_isin_listing_identity import CAPTURED, owned


SYMBOLS = ("SYNRECOVERY.MI", "SYNRECOVERY.L", "SYNRECOVERY.JO", "SYNRECOVERY")
CURRENCIES = ("EUR", "GBp", "ZAc", "USD")
URL = "https://query1.finance.yahoo.com/v7/finance/quote?"


@contextmanager
def acquisition_owner():
    with owned() as owner:
        owner.transport.ledger = owner.ledger
        yield owner
        assert not owner.providers.flight._pending
        state = owner.transport.coordinator.snapshot()
        assert state["active"] == state["queued_essential"] == state["queued_optional"] == 0
        assert state["acquired"] == state["released"]


def quote(symbol):
    index = SYMBOLS.index(symbol)
    return {
        "symbol": symbol, "currency": CURRENCIES[index], "quoteType": "EQUITY",
        "regularMarketPrice": 101. + index, "previousClose": 100. + index,
        "regularMarketTime": int(CAPTURED.timestamp()),
        "shortName": "Synthetic\u00a0name",
        "longName": "Synthetic complete metadata " + symbol,
        "exchangeTimezoneName": "Europe/Rome" if index == 0 else "UTC",
    }


def summary(symbol):
    return {
        "symbol": symbol,
        "price": {**quote(symbol), "regularMarketPrice": 100.},
        "assetProfile": {
            "country": "Ireland", "sector": "Synthetic",
            "companyOfficers": [{"name": "Synthetic\u00a0officer", "age": 40}],
        },
        "quoteType": {"symbol": symbol, "quoteType": "EQUITY"},
        "summaryDetail": {"yield": {"raw": .03, "fmt": "3%"}, "maxAge": 1},
    }


@pytest.fixture
def native_source(monkeypatch):
    state = {
        "batch_status": 200, "summary_status": 200, "single_status": 200,
        "transform": lambda records: records, "batch_error": None,
        "cancel_on_summary": False,
    }
    calls = []
    monkeypatch.setattr(_yf_net, "space_yf_call", lambda: None)
    YfData.cache_get.cache_clear()
    # Each case owns native auth state. The lower response fixture below still
    # exercises the real SDK's request, status and parser path.
    def request(client, method, url, params=None, **_keywords):
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
            document, status, kind, requested = b"synthetic-cookie", 200, "cookie", ()
        elif route == "/v1/test/getcrumb":
            document, status, kind, requested = b"synthetic-crumb", 200, "crumb", ()
        elif "/quoteSummary/" in route:
            symbol = route.rsplit("/", 1)[-1]
            if state["cancel_on_summary"]:
                raise KeyboardInterrupt("synthetic cancellation")
            kind, requested = "summary", (symbol,)
            document = {"quoteSummary": {"result": [summary(symbol)], "error": None}}
        elif route == "/v7/finance/quote":
            requested = tuple(params["symbols"].split(","))
            kind = "batch" if len(requested) > 1 else "single"
            records = [quote(symbol) for symbol in requested]
            if kind == "batch":
                records = state["transform"](records)
            document = {"quoteResponse": {
                "result": records, "error": state["batch_error"] if kind == "batch" else None,
            }}
        else:
            raise AssertionError("undeclared native acquisition")
        if kind not in {"cookie", "crumb"}:
            status = state[kind + "_status"]
            calls.append((kind, requested, status))
        response = client.response_class()
        response.url = url.rstrip("?") + ("?" + urlencode(params) if params else "")
        response.request = Request(response.url, {}, method)
        response.content = (
            document if isinstance(document, bytes)
            else json.dumps(document if status < 400 else {
                "finance": {"result": None, "error": {"description": "synthetic"}},
            }).encode()
        )
        response.status_code, response.ok = status, status < 400
        response.encoding = "utf-8"
        return response

    monkeypatch.setattr(Session, "request", request)
    yield state, calls
    YfData.cache_get.cache_clear()


def failure_codes(owner):
    return sorted(record.stable_code for record in owner.ledger.failure_records())


def compare_original(native_source):
    _state, calls = native_source
    with acquisition_owner() as owner:
        before = {symbol: enricher._fetch_ticker_info(symbol) for symbol in SYMBOLS}
        before_codes = failure_codes(owner)
    before_calls = list(calls)
    calls.clear()
    with acquisition_owner() as owner:
        after = batch.info_many(SYMBOLS)
        after_codes = failure_codes(owner)
        evidence = owner.ledger.entries
    assert before == after, "every native metadata field must remain exact"
    assert before_codes == after_codes, "unresolved native failures must not be hidden"
    return after, before_calls, list(calls), evidence


def test_healthy_batch_preserves_full_parser_and_reduces_actual_native_requests(native_source):
    result, before, after, evidence = compare_original(native_source)
    assert len(before) == 8 and len(after) == 5
    assert [kind for kind, *_rest in after] == ["batch"] + ["summary"] * 4
    assert not any(entry.entry_type is LedgerEntryType.TELEMETRY for entry in evidence)
    for symbol, currency in zip(SYMBOLS, CURRENCIES):
        assert result[symbol]["currency"] == currency
        assert result[symbol]["yield"] == .03
        assert result[symbol]["maxAge"] == 86400
        assert result[symbol]["companyOfficers"][0]["name"] == "Synthetic officer"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_exhausted_group_is_recovered_only_by_all_complete_native_cores(native_source, status):
    state, _calls = native_source
    state["batch_status"] = status
    _result, _before, after, evidence = compare_original(native_source)
    assert sum(kind == "batch" for kind, *_rest in after) == 2
    assert sum(kind == "single" for kind, *_rest in after) == 4
    attempts = [entry.payload for entry in evidence
                if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT]
    assert sum(item["outcome"] == f"HTTP_{status}" for item in attempts) == 2
    recovered = [entry.payload for entry in evidence
                 if entry.entry_type is LedgerEntryType.TELEMETRY]
    assert len(recovered) == 1
    assert recovered[0]["outcome"] == "RECOVERED"
    assert recovered[0]["independently_completed_cores"] == 4


@pytest.mark.parametrize("status", [401, 404])
def test_definitive_group_failure_preserves_independent_native_recovery(native_source, status):
    state, _calls = native_source
    state["batch_status"] = status
    _result, _before, after, evidence = compare_original(native_source)
    assert sum(kind == "single" for kind, *_rest in after) == 4
    assert not any(entry.entry_type is LedgerEntryType.TELEMETRY for entry in evidence)


@pytest.mark.parametrize("case", ["missing", "foreign", "conflicting", "empty", "error"])
def test_inconclusive_member_never_suppresses_native_single_acquisition(native_source, case):
    state, _calls = native_source

    def transform(records):
        if case == "missing":
            return records[1:]
        if case == "foreign":
            return [{"symbol": "FOREIGN.DE", "regularMarketPrice": 999.}] + records[1:]
        if case == "conflicting":
            return records + [{**deepcopy(records[0]), "regularMarketPrice": 777.}]
        if case == "empty":
            return []
        return records

    state["transform"] = transform
    if case == "error":
        state["batch_error"] = {"code": "synthetic"}
    _result, _before, after, _evidence = compare_original(native_source)
    assert any(kind == "single" for kind, *_rest in after)


@pytest.mark.parametrize("status", [404, 429, 503])
def test_independent_summary_failure_keeps_original_warning_and_metadata(native_source, status):
    state, _calls = native_source
    state.update(batch_status=429, summary_status=status)
    _result, _before, _after, evidence = compare_original(native_source)
    # A definitive 404 can still complete the native quote-only contract.
    if status == 404:
        assert any(entry.entry_type is LedgerEntryType.TELEMETRY for entry in evidence)
    else:
        assert not any(entry.entry_type is LedgerEntryType.TELEMETRY for entry in evidence)


def test_native_single_retry_failure_is_never_deferred_by_group_scope(native_source):
    state, _calls = native_source
    state.update(batch_status=429, single_status=429)
    result, _before, _after, evidence = compare_original(native_source)
    assert all(value == {} for value in result.values())
    assert not any(entry.entry_type is LedgerEntryType.TELEMETRY for entry in evidence)
    assert any(entry.entry_type is LedgerEntryType.FAILURE_OPEN for entry in evidence)


def test_unrecovered_group_flushes_failure_on_fatal_cancellation(native_source):
    state, _calls = native_source
    state.update(batch_status=429, cancel_on_summary=True)
    with acquisition_owner() as owner:
        with pytest.raises(KeyboardInterrupt, match="synthetic"):
            batch.info_many(SYMBOLS)
        assert failure_codes(owner) == ["PROVIDER_RETRY_BUDGET_EXHAUSTED"]
        assert recovery._group.get() is recovery._request.get() is None


@pytest.mark.parametrize("native_request", [
    ("POST", URL, {"symbols": ",".join(SYMBOLS)}),
    ("GET", "http://query1.finance.yahoo.com/v7/finance/quote?", {"symbols": ",".join(SYMBOLS)}),
    ("GET", "https://query2.finance.yahoo.com/v7/finance/quote?", {"symbols": ",".join(SYMBOLS)}),
    ("GET", "https://fc.yahoo.com", None),
    ("GET", URL, {"symbols": SYMBOLS[0]}),
    ("GET", URL, {"symbols": ",".join(reversed(SYMBOLS))}),
    ("GET", URL, {"symbols": ",".join(SYMBOLS) + ",FOREIGN.DE"}),
    ("GET", URL + "symbols=" + ",".join(SYMBOLS), {"symbols": ",".join(SYMBOLS)}),
])
def test_unrelated_physical_requests_keep_original_provider_failure(native_request):
    method, url, params = native_request
    with acquisition_owner() as owner:
        with recovery.acquisition(owner.transport, SYMBOLS):
            with recovery.request_scope(owner.transport, method, url, params):
                owner.transport._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
            assert failure_codes(owner) == ["PROVIDER_RETRY_BUDGET_EXHAUSTED"]


@pytest.mark.parametrize("code", ["PROVIDER_BUDGET_EXHAUSTED", "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED"])
def test_deadline_failures_are_never_deferred_or_corrected(code):
    with acquisition_owner() as owner:
        with recovery.acquisition(owner.transport, SYMBOLS):
            with recovery.request_scope(owner.transport, "GET", URL, {"symbols": ",".join(SYMBOLS)}):
                owner.transport._failure("yahoo", code)
            assert failure_codes(owner) == [code]


def test_same_provider_failure_outside_group_remains_after_successful_group_recovery(native_source):
    state, _calls = native_source
    state["batch_status"] = 429
    with acquisition_owner() as owner:
        owner.transport._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
        assert all(batch.info_many(SYMBOLS).values())
        assert failure_codes(owner) == ["PROVIDER_RETRY_BUDGET_EXHAUSTED"]


def test_other_provider_failure_keeps_its_independent_identity(native_source):
    state, _calls = native_source
    state["batch_status"] = 429
    with acquisition_owner() as owner:
        owner.transport._failure("borsa", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
        assert all(batch.info_many(SYMBOLS).values())
        records = owner.ledger.failure_records()
        assert len(records) == 1 and records[0].original_failure["provider"] == "borsa"


def test_new_run_recovers_without_previous_group_state(native_source):
    state, _calls = native_source
    state["batch_status"] = 429
    with acquisition_owner() as owner:
        first = batch.info_many(SYMBOLS)
        assert failure_codes(owner) == []
    state["batch_status"] = 200
    with acquisition_owner() as owner:
        assert batch.info_many(SYMBOLS) == first
        assert failure_codes(owner) == []
        assert not any(entry.entry_type is LedgerEntryType.TELEMETRY
                       for entry in owner.ledger.entries)


def test_nested_recovery_and_foreign_owner_are_rejected():
    with acquisition_owner() as owner:
        with recovery.acquisition(owner.transport, SYMBOLS):
            with pytest.raises(ValueError):
                with recovery.acquisition(owner.transport, SYMBOLS):
                    pass
        with pytest.raises(ValueError):
            with recovery.acquisition(object(), SYMBOLS):
                pass


def test_demanded_enrichment_keys_are_grouped_without_prefetch_or_cross_symbol_cache(native_source):
    _state, calls = native_source
    barrier = Barrier(4)

    def load(symbol):
        barrier.wait(timeout=5)
        return enricher._fetch_ticker_info(symbol)

    with acquisition_owner() as owner, metadata_group_scope():
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {symbol: pool.submit(copy_context().run, load, symbol) for symbol in SYMBOLS}
            values = {symbol: future.result() for symbol, future in pending.items()}
        assert sum(kind == "batch" for kind, *_rest in calls) == 1
        assert sum(kind == "single" for kind, *_rest in calls) == 0
        assert {key for _kind, requested, _status in calls for key in requested} == set(SYMBOLS)
        assert set(owner.providers.store("enricher.info")) == set(SYMBOLS)
        original_calls = len(calls)
        values[SYMBOLS[0]]["country"] = "Mutated by one consumer"
        assert enricher._fetch_ticker_info(SYMBOLS[0])["country"] == "Ireland"
        assert len(calls) == original_calls


def test_single_demand_keeps_original_native_individual_contract(native_source):
    _state, calls = native_source
    with acquisition_owner(), metadata_group_scope():
        info = enricher._fetch_ticker_info(SYMBOLS[0])
    assert info["currency"] == "EUR"
    assert [kind for kind, *_rest in calls] == ["summary", "single"]


def test_distinct_optional_deadlines_cannot_lead_each_others_metadata(native_source):
    _state, calls = native_source
    barrier = Barrier(4)

    def load(owner, symbol):
        with owner.transport.optional_work("synthetic separate discovery"):
            barrier.wait(timeout=5)
            return enricher._fetch_ticker_info(symbol)

    with acquisition_owner() as owner, metadata_group_scope():
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(copy_context().run, load, owner, symbol) for symbol in SYMBOLS]
            assert all(future.result() for future in pending)
    assert not any(kind == "batch" for kind, *_rest in calls)
    assert sum(kind == "single" for kind, *_rest in calls) == 4


def test_context_local_dispatcher_closes_cleanly_after_fatal_worker_exception(native_source):
    state, _calls = native_source
    state.update(batch_status=429, cancel_on_summary=True)
    barrier = Barrier(4)

    def load(symbol):
        barrier.wait(timeout=5)
        return enricher._fetch_ticker_info(symbol)

    with acquisition_owner() as owner:
        with metadata_group_scope():
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = [pool.submit(copy_context().run, load, symbol) for symbol in SYMBOLS]
                for future in pending:
                    with pytest.raises(KeyboardInterrupt, match="synthetic"):
                        future.result()
        assert failure_codes(owner) == ["PROVIDER_RETRY_BUDGET_EXHAUSTED"]
        assert recovery._group.get() is recovery._request.get() is None
        state.update(batch_status=200, cancel_on_summary=False)
        assert enricher._fetch_ticker_info(SYMBOLS[0])["currency"] == "EUR"


def test_unknown_lower_sdk_get_declines_batch_and_keeps_complete_native_info(native_source, monkeypatch):
    _state, calls = native_source
    original = YfData.get

    def injected(data, *arguments, **keywords):
        return original(data, *arguments, **keywords)

    monkeypatch.setattr(YfData, "get", injected)
    with acquisition_owner():
        result = batch.info_many(SYMBOLS)
    assert not any(kind == "batch" for kind, *_rest in calls)
    assert sum(kind == "single" for kind, *_rest in calls) == 4
    assert all(info["country"] == "Ireland" for info in result.values())


def test_locale_changes_do_not_relabel_group_response_as_new_locale(native_source, monkeypatch):
    _state, calls = native_source
    from yfinance.config import YfConfig

    with acquisition_owner():
        data = _yf_net.yahoo_data()
        facts = batch._facts(list(SYMBOLS), data)
        old_locale = facts[SYMBOLS[0]].parameters["lang"]
        monkeypatch.setattr(YfConfig.locale, "lang", "it-IT" if old_locale != "it-IT" else "en-US")
        before = len(calls)
        result = batch._core(SYMBOLS[0], facts[SYMBOLS[0]])
        assert result["currency"] == "EUR"
        assert any(kind == "single" for kind, *_rest in calls[before:])


def test_declared_candidate_info_plan_is_cached_once_without_changing_single_candidate_parser(native_source):
    _state, calls = native_source
    with acquisition_owner() as owner, metadata_group_scope():
        enricher._prepare_native_candidate_infos(list(SYMBOLS))
        assert sum(kind == "batch" for kind, *_rest in calls) == 1
        assert sum(kind == "single" for kind, *_rest in calls) == 0
        assert set(owner.providers.store("enricher.info")) == set(SYMBOLS)
        before = len(calls)
        candidates = [enricher._fetch_candidate_meta(symbol) for symbol in SYMBOLS]
        assert len(calls) == before
        assert [candidate.symbol for candidate in candidates] == list(SYMBOLS)
        assert [candidate.currency for candidate in candidates] == list(CURRENCIES)


def test_four_declared_metadata_plans_drain_same_key_memo_flights(native_source):
    _state, calls = native_source
    barrier = Barrier(4)

    def prepare():
        barrier.wait(timeout=5)
        enricher._prepare_native_candidate_infos(list(SYMBOLS))
        return {symbol: enricher._fetch_ticker_info(symbol) for symbol in SYMBOLS}

    with acquisition_owner() as owner, metadata_group_scope():
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(copy_context().run, prepare) for _ in range(4)]
            answers = [future.result() for future in pending]
        assert all(answer == answers[0] for answer in answers)
        assert sum(kind == "batch" for kind, *_rest in calls) == 1
        assert not owner.providers.flight._pending


def test_injected_candidate_parser_declines_plan_preparation(native_source, monkeypatch):
    _state, calls = native_source
    monkeypatch.setattr(enricher, "_fetch_candidate_meta", lambda symbol: None)
    with acquisition_owner(), metadata_group_scope():
        enricher._prepare_native_candidate_infos(list(SYMBOLS))
    assert calls == []


def test_contract_change_after_memo_reservation_does_not_recurse_or_lose_keys(native_source, monkeypatch):
    _state, calls = native_source
    from tarzan.data import yahoo_info_groups

    monkeypatch.setattr(yahoo_info_groups, "demanded_many", lambda symbols: (False, None))
    with acquisition_owner() as owner, metadata_group_scope():
        enricher._prepare_native_candidate_infos(list(SYMBOLS))
        assert set(owner.providers.store("enricher.info")) == set(SYMBOLS)
        assert all(enricher._fetch_ticker_info(symbol) for symbol in SYMBOLS)
    assert sum(kind == "single" for kind, *_rest in calls) == 4


def test_large_json_price_cannot_discard_valid_previous_close_and_profile(native_source, monkeypatch):
    import sys
    module = sys.modules[__name__]
    original = quote

    def oversized(symbol):
        return {**original(symbol), "regularMarketPrice": 10 ** 1000}

    monkeypatch.setattr(module, "quote", oversized)
    result, _before, after, _evidence = compare_original(native_source)
    assert sum(kind == "batch" for kind, *_rest in after) == 1
    assert all(value["country"] == "Ireland" for value in result.values())


def test_noncanonical_lowercase_key_retains_native_constructor_normalization(native_source):
    _state, calls = native_source
    with acquisition_owner():
        result = batch.info_many([SYMBOLS[0].lower()])
    assert result[SYMBOLS[0].lower()]["symbol"] == SYMBOLS[0]
    assert [kind for kind, *_rest in calls] == ["summary", "single"]
