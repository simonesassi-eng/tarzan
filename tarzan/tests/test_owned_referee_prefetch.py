"""Owned early histories preserve native capacity, recovery and shutdown."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
import json
from threading import Event, Lock

import pytest
import requests

from tarzan import runtime
from tarzan.data import justetf
from tarzan.engine import metrics
from tarzan.runtime.acquisition import RequestCoordinator
from tarzan.tests.test_acquisition_coordinator import allowance, until, workers
from tarzan.tests.test_owned_source_preparation import engine, owned


def response(body, status=200):
    result = requests.Response()
    result.status_code = status
    result.headers["content-type"] = "application/json"
    result._content = json.dumps(body).encode("utf-8")
    result._content_consumed = True
    return result


def source_body(url):
    assert url.startswith("https://www.justetf.com/api/etfs/")
    if "/performance-chart?" in url:
        return {"series": [
            {"date": "2026-10-22", "value": {"raw": 101.0}},
            {"date": "2026-10-23", "value": {"raw": 102.0}},
        ]}
    assert "/quote?" in url
    return {
        "latestQuote": {"raw": 102.0}, "previousQuote": {"raw": 101.0},
        "latestQuoteDate": "2026-10-23", "previousQuoteDate": "2026-10-22",
        "quoteTradingVenue": "SYNTHETIC", "currency": "EUR",
    }


@pytest.mark.parametrize("count", [0, -1, True, False, 1.0, "4", None])
def test_protected_work_count_is_an_explicit_positive_integer(count):
    coordinator = RequestCoordinator(4)
    with pytest.raises(ValueError):
        coordinator.protect(count)
    assert coordinator.snapshot()["requested"] == 0
    assert coordinator.protection_snapshot()["pending"] == 0


def test_reservation_is_not_a_permit_and_cannot_close_or_finish_twice():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(2)
    assert coordinator.snapshot()["active"] == 0
    assert coordinator.snapshot()["essential_demand"] == 1
    with pytest.raises(RuntimeError, match="join"):
        protection.close()
    with pytest.raises(RuntimeError, match="more than once"):
        protection.finish(3)
    protection.finish(2)
    protection.close()
    with pytest.raises(RuntimeError, match="more than once"):
        protection.finish()
    assert coordinator.snapshot()["essential_demand"] == 0


def test_reservation_cannot_finish_a_still_active_native_callback():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(1)
    with coordinator.lease(optional=False, remaining=allowance, protection=protection):
        with pytest.raises(RuntimeError, match="native request permits"):
            protection.finish()
        assert coordinator.protection_snapshot()["active"] == 1
    protection.finish()
    with pytest.raises(RuntimeError, match="no longer active"):
        with coordinator.lease(optional=False, remaining=allowance, protection=protection):
            pytest.fail("retired reservation acquired a request")


@pytest.mark.parametrize("optional", [True, False])
def test_a_reservation_cannot_cross_coordinators_or_authorize_optional_work(optional):
    first, second = RequestCoordinator(4), RequestCoordinator(4)
    protection = first.protect(1)
    try:
        with pytest.raises(ValueError):
            with second.lease(
                optional=optional, remaining=allowance, protection=protection,
            ):
                pytest.fail("foreign reservation acquired a request")
        if optional:
            with pytest.raises(ValueError):
                with first.lease(
                    optional=True, remaining=allowance, protection=protection,
                ):
                    pytest.fail("optional work acquired purchased essential authority")
    finally:
        protection.finish()


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
def test_later_phases_cannot_consume_a_purchased_callbacks_retry_slot(capacity):
    coordinator = RequestCoordinator(capacity)
    protection = coordinator.protect(1)
    with workers() as (spawn, errors, event):
        release = event()
        entered = [Event() for _ in range(capacity + 2)]

        def later(index):
            with coordinator.lease(optional=False, remaining=allowance):
                entered[index].set()
                assert release.wait(timeout=3.0)

        for index in range(len(entered)):
            spawn(lambda index=index: later(index))
        until(lambda: sum(item.is_set() for item in entered) == capacity - 1)
        until(lambda: coordinator.snapshot()["queued_essential"] == 3)
        with coordinator.lease(
            optional=False, remaining=allowance, protection=protection,
        ):
            assert coordinator.snapshot()["active"] == capacity
        protection.finish()
        release.set()
    assert not errors
    state = coordinator.snapshot()
    assert state["active"] == state["essential_demand"] == 0
    assert state["peak"] == capacity
    assert state["acquired"] == state["released"] == capacity + 3
    assert coordinator.protection_snapshot()["pending"] == 0


def test_one_purchased_callback_cannot_expand_into_parallel_native_permits():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(1)
    with workers() as (spawn, errors, event):
        release = event()
        first, second = Event(), Event()

        def request(mark):
            with coordinator.lease(
                optional=False, remaining=allowance, protection=protection,
            ):
                mark.set()
                assert release.wait(timeout=3.0)

        spawn(lambda: request(first))
        assert first.wait(timeout=3.0)
        spawn(lambda: request(second))
        until(lambda: coordinator.snapshot()["queued_essential"] == 1)
        assert not second.is_set()
        assert coordinator.snapshot()["active"] == 1
        release.set()
    protection.finish()
    assert not errors
    assert coordinator.protection_snapshot()["acquired"] == 2


def test_quote_becomes_ready_while_early_history_remains_owned_and_protected(monkeypatch):
    history_started, release_history = Event(), Event()
    calls = []
    lock = Lock()

    def get(url, **kwargs):
        operation = "series" if "/performance-chart?" in url else "quote"
        with lock:
            calls.append(operation)
        if operation == "series":
            history_started.set()
            assert release_history.wait(timeout=3.0)
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        with justetf.owned_prefetch_scope():
            try:
                assert justetf._owned_prefetch_available(session)
                justetf.prefetch_current_quotes(["IE0000000001"])
                assert history_started.is_set()
                assert justetf.quote("IE0000000001")["price"] == 102.0
                assert session.transport.coordinator.protection_snapshot()["pending"] == 1
                assert session.transport.coordinator.snapshot()["active"] == 1
                assert not session.memo[justetf._OWNED_PREFETCH_KEY][0].complete
            finally:
                release_history.set()
        assert justetf._OWNED_PREFETCH_KEY not in session.memo
        assert justetf.series("IE0000000001").tolist() == [101.0, 102.0]
        assert sorted(calls) == ["quote", "series"]
        state = session.transport.coordinator.protection_snapshot()
        assert state["active"] == state["pending"] == state["queued"] == 0
        assert state["acquired"] == state["released"] == 1


def test_later_long_consumer_joins_early_series_and_never_purchases_it_twice(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append("series" if "/performance-chart?" in url else "quote")
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session, justetf.owned_prefetch_scope():
        justetf.prefetch_current_quotes(["IE0000000001", "IE0000000001"])
        justetf.prefetch(["IE0000000001"])
        assert session.memo[justetf._OWNED_PREFETCH_KEY][0].complete
        assert justetf.series("IE0000000001").tolist() == [101.0, 102.0]
        assert sorted(calls) == ["quote", "series"]


def test_a_successful_early_native_response_survives_later_run_budget_expiry(monkeypatch):
    now = [0.0]
    calls = []

    def get(url, **kwargs):
        calls.append("series" if "/performance-chart?" in url else "quote")
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session, justetf.owned_prefetch_scope():
        session.transport.clock = lambda: now[0]
        session.transport.deadline = 60.0
        justetf.prefetch_current_quotes(["IE0000000001"])
        until(lambda: session.transport.coordinator.snapshot()["released"] == 2)
        now[0] = 61.0
        assert justetf.series("IE0000000001").tolist() == [101.0, 102.0]
        assert sorted(calls) == ["quote", "series"]


@pytest.mark.parametrize("primary", [ValueError("owner"), KeyboardInterrupt(), SystemExit(2)])
def test_scope_drains_but_does_not_mask_its_body_failure(monkeypatch, primary):
    release = Event()

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            assert release.wait(timeout=3.0)
            raise SystemExit("synthetic child interruption")
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        with pytest.raises(type(primary)) as caught:
            with justetf.owned_prefetch_scope():
                try:
                    justetf.prefetch_current_quotes(["IE0000000001"])
                    raise primary
                finally:
                    release.set()
        assert caught.value is primary
        assert session.transport.coordinator.snapshot()["active"] == 0
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0
        assert justetf._OWNED_PREFETCH_KEY not in session.memo


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_deferred_child_cancellation_prevents_success_and_is_not_an_ambient_error(monkeypatch, kind):
    source = kind("synthetic old history interruption")

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            raise source
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        try:
            raise ValueError("unrelated already-handled outer error")
        except ValueError:
            with pytest.raises(kind) as caught:
                with justetf.owned_prefetch_scope():
                    justetf.prefetch_current_quotes(["IE0000000001"])
                    assert justetf.quote("IE0000000001")["price"] == 102.0
            assert caught.value is source
        assert session.transport.coordinator.snapshot()["active"] == 0
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0


@pytest.mark.parametrize("name", ["prefetch", "series", "quote", "_fetch_series", "_fetch_quote", "_get_json"])
def test_injected_referee_ports_retain_serial_prefetch_contract(monkeypatch, name):
    with owned() as session, justetf.owned_prefetch_scope():
        assert justetf._owned_prefetch_available(session)
        monkeypatch.setattr(justetf, name, lambda *a: None)
        assert not justetf._owned_prefetch_available(session)
        assert session.transport.coordinator.protection_snapshot()["groups_created"] == 0


def test_pinned_and_unowned_callers_never_create_owned_work(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("live request"))
    runtime.configure(deterministic=True, as_of="2026-10-23")
    justetf.prefetch_current_quotes(["IE0000000001"])
    assert not justetf._owned_prefetch_available(None)


def test_ambient_optional_work_keeps_original_scope_and_demand(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: response(source_body(url)))
    with owned() as session, justetf.owned_prefetch_scope():
        with session.transport.optional_work("injected-owner-optional"):
            assert not justetf._owned_prefetch_available(session)
            justetf.prefetch_current_quotes(["IE0000000001"])
        assert session.transport.coordinator.protection_snapshot()["groups_created"] == 0


def test_partial_submission_failure_drains_every_accepted_callback(monkeypatch):
    native_pool = ThreadPoolExecutor
    primary = RuntimeError("synthetic submission failure")

    class PartialPool:
        def __init__(self, **kwargs):
            self.pool = native_pool(**kwargs)
            self.count = 0

        def submit(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                raise primary
            return self.pool.submit(*args, **kwargs)

        def shutdown(self, **kwargs):
            self.pool.shutdown(**kwargs)

    monkeypatch.setattr(justetf, "ThreadPoolExecutor", PartialPool)
    monkeypatch.setattr(requests, "get", lambda url, **k: response(source_body(url)))
    with owned() as session, justetf.owned_prefetch_scope():
        with pytest.raises(RuntimeError) as caught:
            justetf.prefetch_current_quotes(["IE0000000001", "IE0000000002"])
        assert caught.value is primary
        assert session.transport.coordinator.snapshot()["active"] == 0
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0


def test_protection_survives_retry_backoff_without_holding_a_native_permit(monkeypatch):
    backoff, retry, later_release = Event(), Event(), Event()
    history_calls = []

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            history_calls.append(True)
            if len(history_calls) == 1:
                return response({}, status=503)
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        budget = session.transport
        budget.policies["justetf"] = replace(budget.policies["justetf"], retry_budget=1)

        def sleep(seconds):
            backoff.set()
            assert retry.wait(timeout=3.0)

        budget.sleep = sleep
        with justetf.owned_prefetch_scope():
            try:
                justetf.prefetch_current_quotes(["IE0000000001"])
                assert backoff.wait(timeout=3.0)
                assert budget.coordinator.snapshot()["active"] == 0
                assert budget.coordinator.protection_snapshot()["pending"] == 1
                with ThreadPoolExecutor(max_workers=4) as pool:
                    entered = [Event() for _ in range(4)]

                    def later(index):
                        def io(timeout):
                            entered[index].set()
                            assert later_release.wait(timeout=3.0)
                            return 106
                        return budget.call("yahoo", io)

                    futures = [pool.submit(copy_context().run, later, i) for i in range(4)]
                    until(lambda: sum(event.is_set() for event in entered) == 3)
                    retry.set()
                    until(lambda: budget.coordinator.protection_snapshot()["pending"] == 0)
                    assert justetf.series("IE0000000001").tolist() == [101.0, 102.0]
                    later_release.set()
                    assert [future.result(timeout=3.0) for future in futures] == [106] * 4
            finally:
                retry.set()
                later_release.set()
        assert len(history_calls) == 2
        assert budget.coordinator.snapshot()["peak"] == 4
        assert budget.coordinator.snapshot()["acquired"] == budget.coordinator.snapshot()["released"]


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_an_ordinary_early_history_fault_cannot_hide_a_later_child_cancellation(monkeypatch, kind):
    primary = kind("synthetic second early-history cancellation")

    def malformed_json():
        raise RuntimeError("synthetic first early-history parser fault")

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            if "IE0000000002" in url:
                raise primary
            result = response({})
            result.json = malformed_json
            return result
        return response(source_body(url))

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        with pytest.raises(kind) as caught:
            with justetf.owned_prefetch_scope():
                justetf.prefetch_current_quotes(["IE0000000001", "IE0000000002"])
        assert caught.value is primary
        state = session.transport.coordinator.protection_snapshot()
        assert state["active"] == state["pending"] == state["queued"] == 0


@pytest.mark.parametrize("early_kind", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("later_kind", [KeyboardInterrupt, SystemExit])
def test_early_referee_cancellation_precedes_later_source_worker_cancellation(
    monkeypatch, early_kind, later_kind,
):
    release, early_done = Event(), Event()
    early = early_kind("synthetic earlier referee source cancellation")
    later = later_kind("synthetic later order-history source cancellation")

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            assert release.wait(timeout=3.0)
            early_done.set()
            raise early
        return response(source_body(url))

    def history(ctx):
        release.set()
        raise later

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        with pytest.raises(BaseException) as caught:
            with justetf.owned_prefetch_scope():
                justetf.prefetch_current_quotes(["IE0000000001"])
                engine(monkeypatch, history, lambda ctx: ()).compute_all()
        assert caught.value is early
        assert early_done.is_set()
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_owner_wait_interruption_precedes_earlier_and_later_source_worker_cancellation(
    monkeypatch, kind,
):
    release, finished = Event(), Event()
    primary = kind("synthetic actual source-owner wait interruption")
    earlier = SystemExit("synthetic earlier referee worker cancellation")
    later = KeyboardInterrupt("synthetic later history worker cancellation")
    native_pool = ThreadPoolExecutor

    class InterruptedWait:
        def __init__(self, future):
            self.future = future

        def result(self):
            release.set()
            raise primary

        def done(self):
            return self.future.done()

        def cancelled(self):
            return self.future.cancelled()

        def exception(self):
            return self.future.exception()

    class OwnerWaitPool:
        def __init__(self, **kwargs):
            self.pool = native_pool(**kwargs)
            self.count = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            future = self.pool.submit(*args, **kwargs)
            self.count += 1
            return InterruptedWait(future) if self.count == 1 else future

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    def get(url, **kwargs):
        if "/performance-chart?" in url:
            assert release.wait(timeout=3.0)
            raise earlier
        return response(source_body(url))

    def history(ctx):
        assert release.wait(timeout=3.0)
        finished.set()
        raise later

    monkeypatch.setattr(requests, "get", get)
    with owned() as session:
        with pytest.raises(kind) as caught:
            with justetf.owned_prefetch_scope():
                justetf.prefetch_current_quotes(["IE0000000001"])
                subject = engine(monkeypatch, history, lambda ctx: ())
                monkeypatch.setattr(metrics, "ThreadPoolExecutor", OwnerWaitPool)
                subject.compute_all()
        assert caught.value is primary
        assert finished.is_set()
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("activate", [0, 1, None, "", "later"])
def test_deferred_activation_requires_an_explicit_boolean(activate):
    coordinator = RequestCoordinator(4)
    with pytest.raises(ValueError):
        coordinator.protect(1, activate=activate)
    assert coordinator.protection_snapshot()["groups_created"] == 0


def test_deferred_group_retains_normal_fifo_until_its_owner_activates():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(1, activate=False)
    with workers() as (spawn, errors, event):
        release = event()
        entered = [Event() for _ in range(4)]

        def ordinary(index):
            with coordinator.lease(optional=False, remaining=allowance):
                entered[index].set()
                assert release.wait(timeout=3.0)

        for index in range(4):
            spawn(lambda index=index: ordinary(index))
        until(lambda: all(mark.is_set() for mark in entered))
        assert coordinator.snapshot()["active"] == 4
        assert coordinator.protection_snapshot()["reserved_pending"] == 0
        protection.activate()
        protection.activate()
        assert coordinator.protection_snapshot()["activations"] == 1
        assert coordinator.protection_snapshot()["reserved_pending"] == 1
        release.set()
    protection.finish()
    assert errors == []
    assert coordinator.protection_snapshot()["reserved_pending"] == 0


def test_deferred_activation_migrates_a_waiter_without_duplicating_or_losing_its_ticket():
    coordinator = RequestCoordinator(1)
    protection = coordinator.protect(1, activate=False)
    with workers() as (spawn, errors, event):
        called = Event()

        def purchased():
            with coordinator.lease(
                optional=False, remaining=allowance, protection=protection,
            ):
                called.set()

        with coordinator.lease(optional=False, remaining=allowance):
            spawn(purchased)
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            assert coordinator.protection_snapshot()["queued"] == 0
            protection.activate()
            assert coordinator.protection_snapshot()["queued"] == 1
            assert coordinator.snapshot()["queued_essential"] == 1
        assert called.wait(timeout=3.0)
    protection.finish()
    assert errors == []
    assert coordinator.snapshot()["acquired"] == coordinator.snapshot()["released"] == 2
    assert coordinator.protection_snapshot()["queued"] == 0


def test_a_migrated_waiters_deadline_removes_it_from_the_new_queue():
    coordinator = RequestCoordinator(1)
    protection = coordinator.protect(1, activate=False)
    expired = Event()
    primary = KeyboardInterrupt("synthetic migrated waiter interruption")

    def remaining():
        if expired.is_set():
            raise primary
        return 10.0

    with workers() as (spawn, errors, event):
        def purchased():
            with coordinator.lease(
                optional=False, remaining=remaining, protection=protection,
            ):
                pytest.fail("expired migrated waiter acquired capacity")

        with coordinator.lease(optional=False, remaining=allowance):
            spawn(purchased)
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            protection.activate()
            expired.set()
            until(lambda: coordinator.snapshot()["rejected"] == 1)
        assert coordinator.snapshot()["queued_essential"] == 0
        assert coordinator.protection_snapshot()["queued"] == 0
    protection.finish()
    assert errors == [primary]


def test_activating_an_inflight_group_does_not_double_reserve_its_native_slot():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(1, activate=False)
    with workers() as (spawn, errors, event):
        release = event()
        entered = [Event() for _ in range(3)]

        def ordinary(index):
            with coordinator.lease(optional=False, remaining=allowance):
                entered[index].set()
                assert release.wait(timeout=3.0)

        with coordinator.lease(
            optional=False, remaining=allowance, protection=protection,
        ):
            protection.activate()
            for index in range(3):
                spawn(lambda index=index: ordinary(index))
            until(lambda: all(mark.is_set() for mark in entered))
            assert coordinator.snapshot()["active"] == 4
            release.set()
    protection.finish()
    assert errors == []
    assert coordinator.snapshot()["peak"] == 4


def test_series_parsing_cannot_starve_its_own_early_quotes(monkeypatch):
    started, release, quoted = Event(), Event(), Event()
    count = 0
    lock = Lock()

    def get(url, **kwargs):
        nonlocal count
        result = response(source_body(url))
        if "/performance-chart?" in url:
            body = result.json()

            def parse():
                nonlocal count
                with lock:
                    count += 1
                    if count == 4:
                        started.set()
                assert release.wait(timeout=3.0)
                return body

            result.json = parse
        else:
            quoted.set()
        return result

    monkeypatch.setattr(requests, "get", get)
    with owned() as session, justetf.owned_prefetch_scope():
        with workers() as (spawn, errors, event):
            owner_context = copy_context()
            try:
                spawn(lambda: owner_context.run(
                    justetf.prefetch_current_quotes,
                    ["IE0000000001", "IE0000000002", "IE0000000003", "IE0000000004"],
                ))
                assert started.wait(timeout=3.0)
                assert quoted.wait(timeout=3.0)
                assert all(
                    justetf.quote(isin)["price"] == 102.0 for isin in
                    ["IE0000000001", "IE0000000002", "IE0000000003", "IE0000000004"]
                )
            finally:
                release.set()
        assert errors == []


def test_independent_quote_retains_authority_without_certifying_a_failed_long_series(monkeypatch):
    logs = []

    def malformed():
        raise RuntimeError("synthetic series parser fault")

    def get(url, **kwargs):
        result = response(source_body(url))
        if "/performance-chart?" in url:
            result.json = malformed
        return result

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(justetf.logger, "info", lambda *args: logs.append(args))
    with owned() as session, justetf.owned_prefetch_scope():
        justetf.prefetch_current_quotes(["IE0000000001"])
        assert justetf.quote("IE0000000001")["price"] == 102.0
        justetf.finish_owned_prefetch(session)
        justetf.finish_owned_prefetch(session)
        assert len(logs) == 1
        assert "IE0000000001" not in str(logs)
        with pytest.raises(RuntimeError, match="series parser fault"):
            justetf.prefetch(["IE0000000001"])
        assert justetf.quote("IE0000000001")["price"] == 102.0


def test_an_ordinary_log_handler_fault_is_not_independent_quote_authority(monkeypatch):
    def malformed():
        raise RuntimeError("synthetic series parser fault")

    def log(*args):
        raise ValueError("synthetic secondary observer fault")

    def get(url, **kwargs):
        result = response(source_body(url))
        if "/performance-chart?" in url:
            result.json = malformed
        return result

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(justetf.logger, "info", log)
    with owned() as session, justetf.owned_prefetch_scope():
        justetf.prefetch_current_quotes(["IE0000000001"])
        assert justetf.quote("IE0000000001")["price"] == 102.0
        justetf.finish_owned_prefetch(session)
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0


@pytest.mark.parametrize("primary", [ValueError("constructor"), KeyboardInterrupt(), SystemExit(2)])
def test_executor_constructor_failure_releases_every_unsubmitted_obligation(monkeypatch, primary):
    def pool(**kwargs):
        raise primary

    monkeypatch.setattr(justetf, "ThreadPoolExecutor", pool)
    with owned() as session, justetf.owned_prefetch_scope():
        with pytest.raises(type(primary)) as caught:
            justetf.prefetch_current_quotes(["IE0000000001", "IE0000000002"])
        assert caught.value is primary
        assert session.transport.coordinator.protection_snapshot()["pending"] == 0
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        assert session.transport.coordinator.snapshot()["requested"] == 0


def test_an_ambient_protected_owner_keeps_the_original_prefetch_contract(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response(source_body(url)))
    with owned() as session, justetf.owned_prefetch_scope():
        protection = session.transport.coordinator.protect(1)
        try:
            with session.transport.protected_requests(protection):
                assert not justetf._owned_prefetch_available(session)
                justetf.prefetch_current_quotes(["IE0000000001"])
            assert session.transport.protected_work_token() is None
            assert session.memo[justetf._OWNED_PREFETCH_KEY] == []
            assert justetf.quote("IE0000000001")["price"] == 102.0
        finally:
            protection.finish()
