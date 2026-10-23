"""Fixed endpoint categories observe native I/O without retaining request facts."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tarzan.runtime import request_classes as observed
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, canonical_analysis_id
from tarzan.runtime.transport import (
    ProviderBudget, ProviderBudgetExceeded, ProviderRequestFailure,
    ProviderTransportSettings, _yahoo_request_observation,
)


CANARY = "synthetic-request-private-canary"


def budget(*, retries=0, capacity=4):
    settings = ProviderTransportSettings.configured()
    settings = replace(
        settings, run_budget_seconds=60.0,
        providers=tuple(replace(policy, retry_budget=retries) for policy in settings.providers),
    )
    return ProviderBudget(
        settings, RunLedger("synthetic-request-classes"), capacity=capacity,
        clock=lambda: 0.0, sleep=lambda _: None,
    )


def attempts(owner):
    return [
        entry.payload for entry in owner.ledger.entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    ]


@pytest.mark.parametrize("url,expected", [
    ("https://query1.finance.yahoo.com/v8/finance/chart/SYNTH", "CHART"),
    ("https://QUERY2.FINANCE.YAHOO.COM:443/v8/finance/chart/SYNTH%2EMI", "CHART"),
    ("https://query1.finance.yahoo.com/v7/finance/quote", "QUOTE"),
    ("https://query2.finance.yahoo.com/v6/finance/quote", "QUOTE"),
    ("https://query2.finance.yahoo.com/v10/finance/quoteSummary/SYNTH", "QUOTE_SUMMARY"),
    ("https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/SYNTH", "FUNDAMENTALS"),
    ("https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries", "FUNDAMENTALS"),
    ("https://query1.finance.yahoo.com/v1/test/getcrumb", "CRUMB"),
    ("https://query2.finance.yahoo.com/v1/test/getcrumb", "CRUMB"),
    ("https://fc.yahoo.com", "COOKIE"),
    ("https://fc.yahoo.com/", "COOKIE"),
    ("https://guce.yahoo.com/consent", "CONSENT"),
    ("https://guce.yahoo.com/copyConsent", "CONSENT"),
    ("https://consent.yahoo.com/v2/collectConsent", "CONSENT"),
    ("https://query1.finance.yahoo.com/unknown", "OTHER"),
])
def test_closed_host_path_shapes_ignore_all_query_fragment_and_instrument_facts(url, expected):
    private = url + f"?crumb={CANARY}&symbols={CANARY}#{CANARY}"
    assert observed.yahoo_request_class(private) == expected
    assert observed.yahoo_request_class(private) in observed.REQUEST_CLASSES
    assert CANARY not in observed.yahoo_request_class(private)


@pytest.mark.parametrize("url", [
    None, 0, b"https://fc.yahoo.com/", "",
    "http://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "httpsx://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "//query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com.evil.invalid/v8/finance/chart/SYNTH",
    "https://evil.invalid/?url=https://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com@evil.invalid/v8/finance/chart/SYNTH",
    "https://user:password@query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com:80/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com:bad/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com:65536/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com:0/v8/finance/chart/SYNTH",
    "https://[invalid/v8/finance/chart/SYNTH",
    "https://query1.finance.yаhoo.com/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com./v8/finance/chart/SYNTH",
    "\nhttps://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com\t/v8/finance/chart/SYNTH",
    "https://query1.finance.yahoo.com/v8/finance/chart/",
    "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH/extra",
    "https://query1.finance.yahoo.com/" + "x" * 16384,
])
def test_malformed_similar_protocols_lookalikes_and_untrusted_objects_are_other(url):
    assert observed.yahoo_request_class(url) == "OTHER"


def test_unknown_url_object_is_never_coerced_or_inspected():
    class PrivateURL:
        def __str__(self):
            pytest.fail("private URL object was coerced")

        def __getattr__(self, name):
            pytest.fail("private URL object was inspected")

    class PrivateString(str):
        def strip(self):
            pytest.fail("untrusted string subclass was normalized")

    assert observed.yahoo_request_class(PrivateURL()) == "OTHER"
    assert observed.yahoo_request_class(PrivateString("https://fc.yahoo.com/")) == "OTHER"


def test_classifier_does_not_populate_standard_urlsplit_cache_with_private_addresses():
    before = urlsplit.cache_info()
    for ordinal in range(20):
        assert observed.yahoo_request_class(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{CANARY}"
            f"?crumb={CANARY}-{ordinal}#{CANARY}",
        ) == "CHART"
    # No global cache clearing: unrelated SDK cache entries remain untouched.
    assert urlsplit.cache_info() == before


def test_missing_uncached_parser_is_closed_other_without_cached_fallback(monkeypatch):
    monkeypatch.setattr(observed, "_UNCACHED_URLSPLIT", None)
    before = urlsplit.cache_info()
    assert observed.yahoo_request_class(f"https://fc.yahoo.com/?token={CANARY}") == "OTHER"
    assert urlsplit.cache_info() == before


def test_request_class_observer_can_be_imported_without_external_dependencies():
    path = Path(observed.__file__)
    recipe = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("synthetic_request_classes", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.yahoo_request_class("https://fc.yahoo.com/") == "COOKIE"
assert not any(name.split(".")[0] in {"pandas", "numpy", "yfinance", "curl_cffi", "tarzan"}
               for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", recipe, str(path)], check=True,
        capture_output=True, text=True, timeout=5,
    )
    assert not completed.stdout and not completed.stderr


@pytest.fixture
def native_session(monkeypatch):
    calls = []
    behavior = [lambda: SimpleNamespace(status_code=200)]

    class Session:
        def __init__(self, **kwargs):
            assert kwargs == {"impersonate": "chrome"}

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return behavior[0]()

        def close(self):
            pass

    monkeypatch.setattr("curl_cffi.requests.Session", Session)
    return calls, behavior


def test_native_wire_preserves_original_request_and_never_retains_canary(native_session):
    calls, behavior = native_session
    owner = budget()
    answer = SimpleNamespace(status_code=404)
    behavior[0] = lambda: answer
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{CANARY}?crumb={CANARY}"
    headers = {"Authorization": CANARY}
    params = {"symbols": CANARY}
    assert owner.yahoo_client().request(
        "GET", url, timeout=4, headers=headers, params=params,
    ) is answer
    assert calls == [("GET", url, {"timeout": 4, "headers": headers, "params": params})]
    assert calls[0][2]["headers"] is headers
    assert calls[0][2]["params"] is params
    assert attempts(owner)[0]["native_request_class"] == "CHART"
    assert attempts(owner)[0]["outcome"] == "HTTP_404"
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])
    assert observed.request_class_for(owner, "yahoo") is None
    assert owner.call("yahoo", lambda _: 1) == 1
    assert "native_request_class" not in attempts(owner)[-1]


def test_legacy_call_patch_receives_no_new_keyword_or_changed_arguments(native_session):
    calls, _ = native_session
    owner = budget()
    received = []

    def legacy(provider, callback, *, requested_timeout=None):
        received.append((provider, requested_timeout))
        return callback(requested_timeout)

    owner.call = legacy
    owner.yahoo_client().request("GET", "https://fc.yahoo.com/", timeout=3)
    assert received == [("yahoo", 3)]
    assert calls[0][2] == {"timeout": 3}
    assert str(inspect.signature(ProviderBudget.call)) == (
        "(self, provider, callback, *, requested_timeout=None)"
    )
    assert observed.request_class_for(owner, "yahoo") is None


def test_legacy_budget_subclass_uses_unchanged_call_api_and_observes_native_wire(native_session):
    calls, _ = native_session
    received = []

    class LegacyBudget(ProviderBudget):
        def call(self, provider, callback, *, requested_timeout=None):
            received.append((provider, requested_timeout))
            return super().call(provider, callback, requested_timeout=requested_timeout)

    owner = LegacyBudget(
        budget().settings, RunLedger("synthetic-legacy-budget"),
        clock=lambda: 0.0, sleep=lambda _: None,
    )
    owner.yahoo_client().request("GET", "https://fc.yahoo.com/", timeout=3)
    assert received == [("yahoo", 3)]
    assert calls[0][2] == {"timeout": 3}
    assert attempts(owner)[0]["native_request_class"] == "COOKIE"


def test_unknown_url_object_reaches_original_native_wire_unchanged(native_session):
    calls, _ = native_session
    owner = budget()

    class PrivateURL:
        def __str__(self):
            pytest.fail("private URL object was coerced")

        def __getattr__(self, name):
            pytest.fail("private URL object was inspected")

    private = PrivateURL()
    assert owner.yahoo_client().request("GET", private).status_code == 200
    assert calls[0][1] is private
    assert attempts(owner)[0]["native_request_class"] == "OTHER"
    assert observed.request_class_for(owner, "yahoo") is None


def test_scope_cannot_bless_another_provider_or_foreign_owner():
    first, second = budget(), budget()
    with observed.yahoo_request_scope(first, "https://fc.yahoo.com/"):
        first.call("justetf", lambda _: 1)
        second.call("yahoo", lambda _: 2)
        assert "native_request_class" not in attempts(first)[0]
        assert "native_request_class" not in attempts(second)[0]
        with observed.yahoo_request_scope(second, "https://query2.finance.yahoo.com/v7/finance/quote"):
            second.call("yahoo", lambda _: 3)
            first.call("yahoo", lambda _: 4)
            assert attempts(second)[-1]["native_request_class"] == "QUOTE"
            assert "native_request_class" not in attempts(first)[-1]
        first.call("yahoo", lambda _: 5)
        assert attempts(first)[-1]["native_request_class"] == "COOKIE"
    assert observed.request_class_for(first, "yahoo") is None


def test_parallel_requests_keep_categories_isolated(native_session):
    _, behavior = native_session
    owner = budget(capacity=2)
    barrier = Barrier(2)

    def native():
        barrier.wait(timeout=2)
        return SimpleNamespace(status_code=200)

    behavior[0] = native
    client = owner.yahoo_client()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [
            pool.submit(client.request, "GET", url)
            for url in (
                "https://fc.yahoo.com/",
                "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
            )
        ]
        assert all(future.result(timeout=3).status_code == 200 for future in pending)
    assert sorted(item["native_request_class"] for item in attempts(owner)) == ["CHART", "COOKIE"]
    assert owner.coordinator.snapshot()["active"] == 0
    assert observed.request_class_for(owner, "yahoo") is None


def test_retries_keep_annotation_without_extra_attempts_or_policy_changes(native_session):
    calls, behavior = native_session
    owner = budget(retries=1)
    answers = iter((SimpleNamespace(status_code=500), SimpleNamespace(status_code=200)))
    behavior[0] = lambda: next(answers)
    assert owner.yahoo_client().request("GET", "https://fc.yahoo.com/").status_code == 200
    assert len(calls) == 2
    assert [item["ordinal"] for item in attempts(owner)] == [1, 2]
    assert [item["outcome"] for item in attempts(owner)] == ["HTTP_500", "HTTP_200"]
    assert all(item["native_request_class"] == "COOKIE" for item in attempts(owner))
    assert observed.request_class_for(owner, "yahoo") is None


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit, ProviderBudgetExceeded])
def test_native_failure_or_cancellation_restores_scope_and_original_outcome(native_session, error_type):
    _, behavior = native_session
    owner = budget()
    error = error_type(CANARY)

    def abort():
        raise error

    behavior[0] = abort
    expected = ProviderRequestFailure if error_type is ValueError else error_type
    with pytest.raises(expected) as caught:
        owner.yahoo_client().request("GET", "https://fc.yahoo.com/")
    if error_type is not ValueError:
        assert caught.value is error
    assert attempts(owner)[0]["native_request_class"] == "COOKIE"
    assert observed.request_class_for(owner, "yahoo") is None
    assert owner.coordinator.snapshot()["active"] == 0
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])


@pytest.mark.parametrize("observer", ["classifier", "payload"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_optional_observer_error_cannot_change_primary_response(
    native_session, monkeypatch, observer, error_type,
):
    _, behavior = native_session
    owner = budget()
    answer = SimpleNamespace(status_code=404)
    behavior[0] = lambda: answer

    def broken(*args):
        raise error_type(CANARY)

    monkeypatch.setattr(
        observed, "yahoo_request_class" if observer == "classifier" else "request_class_for", broken,
    )
    assert owner.yahoo_client().request("GET", f"https://fc.yahoo.com/?secret={CANARY}") is answer
    assert attempts(owner)[0]["outcome"] == "HTTP_404"
    assert "native_request_class" not in attempts(owner)[0]
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])


def test_invalid_observer_label_is_not_retained(native_session, monkeypatch):
    owner = budget()
    monkeypatch.setattr(observed, "request_class_for", lambda *args: CANARY)
    owner.yahoo_client().request("GET", "https://fc.yahoo.com/")
    assert "native_request_class" not in attempts(owner)[0]


@pytest.mark.parametrize("phase", ["factory", "enter", "exit", "suppress"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_optional_scope_failure_or_suppression_cannot_replace_primary_native_outcome(
    native_session, monkeypatch, phase, cancelled,
):
    _, behavior = native_session
    owner = budget()
    answer = SimpleNamespace(status_code=404)
    primary = KeyboardInterrupt(CANARY)
    exits = []

    class BrokenObservation:
        def __enter__(self):
            if phase == "enter":
                raise SystemExit(CANARY)

        def __exit__(self, *exception_details):
            exits.append(exception_details)
            if phase == "exit":
                raise KeyboardInterrupt(CANARY)
            return True

    def scope(*args):
        if phase == "factory":
            raise KeyboardInterrupt(CANARY)
        return BrokenObservation()

    def native():
        if cancelled:
            raise primary
        return answer

    monkeypatch.setattr(observed, "yahoo_request_scope", scope)
    behavior[0] = native
    if cancelled:
        with pytest.raises(KeyboardInterrupt) as caught:
            owner.yahoo_client().request("GET", "https://fc.yahoo.com/")
        assert caught.value is primary
        assert attempts(owner)[0]["outcome"] == "CANCELLED"
    else:
        assert owner.yahoo_client().request("GET", "https://fc.yahoo.com/") is answer
        assert attempts(owner)[0]["outcome"] == "HTTP_404"
    assert exits == ([] if phase == "factory" else [(None, None, None)])
    assert owner.coordinator.snapshot()["active"] == 0
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])


@pytest.mark.parametrize("operation", ["set", "get", "reset"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_context_variable_failure_is_optional_and_defensive_cleanup_is_private(
    native_session, monkeypatch, operation, error_type,
):
    owner = budget()
    actual = ContextVar("synthetic-request-failing-scope", default=None)

    class FailingScope:
        def set(self, value):
            if operation == "set":
                raise error_type(CANARY)
            return actual.set(value)

        def get(self):
            if operation == "get":
                raise error_type(CANARY)
            return actual.get()

        def reset(self, token):
            if operation == "reset":
                raise error_type(CANARY)
            return actual.reset(token)

    monkeypatch.setattr(observed, "_scope", FailingScope())
    assert owner.yahoo_client().request("GET", "https://fc.yahoo.com/").status_code == 200
    assert attempts(owner)[0]["outcome"] == "HTTP_200"
    if operation == "reset":
        assert attempts(owner)[0]["native_request_class"] == "COOKIE"
    else:
        assert "native_request_class" not in attempts(owner)[0]
    assert actual.get() is None
    assert observed.request_class_for(owner, "yahoo") is None
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])


@pytest.mark.parametrize("failure", ["factory", "context_set", "all_context_set"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_failed_inner_native_annotation_does_not_inherit_parent_label_and_restores_outer(
    native_session, monkeypatch, failure, error_type,
):
    calls, _ = native_session
    owner = budget()
    original_scope = observed._scope

    class FailingInnerScope:
        def set(self, value):
            if failure == "all_context_set" or (
                type(value) is tuple and value[2] == "COOKIE"
            ):
                raise error_type(CANARY)
            return original_scope.set(value)

        def get(self):
            return original_scope.get()

        def reset(self, state):
            original_scope.reset(state)

    def failed_factory(*args):
        raise error_type(CANARY)

    with observed.yahoo_request_scope(
        owner, "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH",
    ):
        if failure == "factory":
            monkeypatch.setattr(observed, "yahoo_request_scope", failed_factory)
        else:
            monkeypatch.setattr(observed, "_scope", FailingInnerScope())
        assert owner.yahoo_client().request("GET", "https://fc.yahoo.com/").status_code == 200
        assert len(calls) == 1
        assert attempts(owner)[0]["outcome"] == "HTTP_200"
        assert "native_request_class" not in attempts(owner)[0]
        assert observed.request_class_for(owner, "yahoo") == "CHART"
        assert owner.call("yahoo", lambda _: 1) == 1
        assert attempts(owner)[-1]["native_request_class"] == "CHART"
    assert observed.request_class_for(owner, "yahoo") is None
    assert owner.coordinator.snapshot()["active"] == 0
    assert CANARY not in json.dumps([entry.to_dict() for entry in owner.ledger.entries])


@pytest.mark.parametrize("error_type", [None, ValueError, KeyboardInterrupt])
def test_inactive_observer_frames_drop_raw_url_before_native_io(monkeypatch, error_type):
    owner = budget()
    private = f"https://query1.finance.yahoo.com/v8/finance/chart/{CANARY}?crumb={CANARY}"
    if error_type is not None:
        def broken(*args):
            raise error_type(CANARY)

        monkeypatch.setattr(observed, "yahoo_request_class", broken)
    observation = _yahoo_request_observation(owner, private)
    with observation:
        frames = [observation.gen.gi_frame, observation.gen.gi_frame.f_locals["observation"].gen.gi_frame]
        for frame in frames:
            assert frame.f_locals["url"] is None
            assert all(CANARY not in value for value in frame.f_locals.values() if type(value) is str)
    assert observation.gen.gi_frame is None
    assert observed.request_class_for(owner, "yahoo") is None


def test_real_native_ledger_annotation_cannot_change_full_analysis_material_or_id(native_session):
    from datetime import datetime, timezone

    from tarzan.models.portfolio import PortfolioMetrics
    from tarzan.orchestrator import _analysis_evidence

    owner = budget()
    session = RunSession(
        context=RunContext(
            "synthetic-request-classes", RunMode.LIVE, None,
            datetime(2026, 10, 21, 10, tzinfo=timezone.utc),
        ),
        config_snapshot={"target": 60.0}, ledger=owner.ledger,
    )
    valuation = {
        "source": "yahoo", "outcome": "AVAILABLE", "value_eur": 106.0,
        "quote_date": "2026-10-21", "source_observed_at": "2026-10-21T10:00Z",
    }
    session.memo.update({
        "effective_orders": {"digest": "synthetic-visible-orders"},
        "valuation": valuation,
    })
    metrics = PortfolioMetrics(total_value=106.0)
    before = _analysis_evidence(session, metrics)
    before_id = canonical_analysis_id(before)
    owner.yahoo_client().request(
        "GET", f"https://query1.finance.yahoo.com/v8/finance/chart/{CANARY}",
    )
    assert attempts(owner)[0]["native_request_class"] == "CHART"
    after = _analysis_evidence(session, metrics)
    assert after == before
    assert canonical_analysis_id(after) == before_id
    assert after["valuation"] == valuation
    assert after["metrics"]["total_value"] == 106.0
    for field in ("quote_date", "source_observed_at", "outcome", "value_eur"):
        session.memo["valuation"] = valuation | {field: "changed"}
        assert canonical_analysis_id(_analysis_evidence(session, metrics)) != before_id
    # The financial whitelist was not expanded. If any similarly named field
    # were actually introduced into valuation facts it must remain material.
    session.memo["valuation"] = valuation | {"native_request_class": "CHART"}
    assert canonical_analysis_id(_analysis_evidence(session, metrics)) != before_id
