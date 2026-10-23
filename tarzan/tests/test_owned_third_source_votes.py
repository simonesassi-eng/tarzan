"""Synthetic acquisition ownership; no market inputs, provider or email."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import replace
import socket
import threading
import urllib.request

import pytest
import requests

from tarzan import config, runtime
from tarzan.engine import third_source_votes as votes
from tarzan.runtime import data_quality as dq
from tarzan.runtime.acquisition import RequestCoordinator
from tarzan.runtime.memo import memoized
from tarzan.runtime.session import current_session
from tarzan.tests.test_essential_data_priority import owned


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    dq.reset()
    def blocked(*args, **kwargs):
        pytest.fail("a row scheduling test attempted network transport")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    from curl_cffi.requests import Session
    monkeypatch.setattr(Session, "request", blocked)
    yield
    dq.reset()


@contextmanager
def owner(monkeypatch, *, capacity=4, workers=4):
    monkeypatch.setattr(config, "max_workers", lambda: workers)
    with owned() as (session, clock):
        session.transport.coordinator = RequestCoordinator(capacity)
        yield session, clock


def forbid_executor(monkeypatch):
    monkeypatch.setattr(votes, "ThreadPoolExecutor", lambda *a, **kw:
                        pytest.fail("serial fallback created an executor"))


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
@pytest.mark.parametrize("requested", [None, 1, 2, 4, 8])
def test_native_guard_retains_configured_and_aggregate_caps(monkeypatch, capacity, requested):
    with owner(monkeypatch, capacity=capacity, workers=4):
        effective = min(capacity, requested or 4, 4)
        assert votes.can_acquire_together(native=True, workers=requested) is (effective >= 2)
        assert votes.can_acquire_together(native=False, workers=requested) is False
        assert votes.can_acquire_together(workers=requested) is False


@pytest.mark.parametrize("requested", [False, True, 0, -1, "4", 2.0, float("inf")])
def test_invalid_worker_override_never_creates_new_parallel_authority(monkeypatch, requested):
    with owner(monkeypatch):
        assert votes.can_acquire_together(native=True, workers=requested) is False


@pytest.mark.parametrize("mode", ["unowned", "pinned", "foreign_context", "foreign_budget", "injected"])
def test_unowned_pinned_injected_and_mismatched_callers_use_serial(monkeypatch, mode):
    forbid_executor(monkeypatch)
    seen = []
    callbacks = [lambda index=index: seen.append((index, threading.get_ident())) or index
                 for index in range(4)]
    if mode == "unowned":
        batch = votes.acquire_third_source_votes(callbacks, parallel=True)
    else:
        with owner(monkeypatch) as (session, _clock):
            parallel = True
            if mode == "pinned":
                monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)
            elif mode == "foreign_budget":
                session.transport = object()
            elif mode == "foreign_context":
                with runtime.bind_context(replace(session.context, attempt_id="another-attempt")):
                    batch = votes.acquire_third_source_votes(callbacks, parallel=True)
            elif mode == "injected":
                parallel = False
            if mode != "foreign_context":
                batch = votes.acquire_third_source_votes(callbacks, parallel=parallel)
    assert batch.interruption is None
    assert [outcome.result for outcome in batch.outcomes] == list(range(4))
    assert seen == [(index, threading.get_ident()) for index in range(4)]


@pytest.mark.parametrize("callbacks", [[], [lambda: "one"]])
def test_known_zero_or_one_row_needs_no_executor(monkeypatch, callbacks):
    forbid_executor(monkeypatch)
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes(callbacks, parallel=True)
    assert len(batch.outcomes) == len(callbacks)
    assert batch.interruption is None


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
def test_512_rows_bound_submission_active_native_calls_and_keep_order(monkeypatch, capacity):
    native_executor = ThreadPoolExecutor
    pending_count = peak_pending = active = peak_active = 0
    lock = threading.RLock()
    rendezvous = threading.Barrier(capacity)

    class ObservedPool(native_executor):
        def submit(self, *args, **kwargs):
            nonlocal pending_count, peak_pending
            with lock:
                pending_count += 1
                peak_pending = max(peak_pending, pending_count)
            future = super().submit(*args, **kwargs)

            def finished(_future):
                nonlocal pending_count
                with lock:
                    pending_count -= 1

            future.add_done_callback(finished)
            return future

    monkeypatch.setattr(votes, "ThreadPoolExecutor", ObservedPool)
    with owner(monkeypatch, capacity=capacity, workers=8) as (session, _clock):
        def callback(index):
            assert current_session() is session

            def native(_timeout):
                nonlocal active, peak_active
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    if index < capacity and capacity > 1:
                        rendezvous.wait(timeout=3)
                    return index
                finally:
                    with lock:
                        active -= 1

            return session.transport.call("yahoo", native)

        batch = votes.acquire_third_source_votes(
            (lambda index=index: callback(index) for index in range(512)),
            parallel=True, workers=8,
        )
        native_state = session.transport.coordinator.snapshot()
    assert batch.interruption is None
    assert [outcome.result for outcome in batch.outcomes] == list(range(512))
    assert all(outcome.submitted and outcome.started and outcome.completed for outcome in batch.outcomes)
    assert peak_pending <= capacity
    assert peak_active == capacity
    assert active == pending_count == 0
    assert native_state["active"] == 0
    assert native_state["acquired"] == native_state["released"] == 512


def test_fast_later_rows_replenish_window_without_waiting_for_slow_first(monkeypatch):
    later_started = threading.Event()
    completed = []
    with owner(monkeypatch, capacity=2, workers=2):
        def callback(index):
            if index == 0:
                assert later_started.wait(timeout=3)
            if index == 2:
                later_started.set()
            completed.append(index)
            return index

        batch = votes.acquire_third_source_votes(
            [lambda index=index: callback(index) for index in range(4)], parallel=True,
        )
    assert batch.interruption is None
    assert completed.index(2) < completed.index(0)
    assert [outcome.result for outcome in batch.outcomes] == list(range(4))


def test_reversed_completion_keeps_distinct_contexts_and_deferred_dq_order(monkeypatch):
    local = ContextVar("synthetic-row", default="parent")
    outer_buffer = []
    release = [threading.Event() for _ in range(4)]
    release[-1].set()
    rendezvous = threading.Barrier(4)
    completed = []
    contexts = []
    with owner(monkeypatch, capacity=4, workers=4):
        dq.defer_into(outer_buffer)

        def callback(index):
            assert local.get() == "parent"
            local.set(index)
            contexts.append(copy_context())
            dq.warning("synthetic-vote", "synthetic observation", context=str(index))
            rendezvous.wait(timeout=3)
            assert release[index].wait(timeout=3)
            completed.append(index)
            if index:
                release[index - 1].set()
            return index

        batch = votes.acquire_third_source_votes(
            [lambda index=index: callback(index) for index in range(4)], parallel=True,
        )
        assert completed == [3, 2, 1, 0]
        assert local.get() == "parent"
        assert outer_buffer == []
        assert dq.issues() == []
        assert len({id(context) for context in contexts}) == 4
        assert len({id(outcome.diagnostics) for outcome in batch.outcomes}) == 4
        for outcome in batch.outcomes:
            dq.replay(outcome.diagnostics)
        assert len(outer_buffer) == 4
        # Restore a non-deferring adoption context without resetting evidence.
        from tarzan.runtime.data_quality import _deferred
        token = _deferred.set(None)
        try:
            dq.replay(outer_buffer)
            assert [issue.context for issue in dq.issues()] == ["0", "1", "2", "3"]
        finally:
            _deferred.reset(token)


def test_callback_retains_all_candidates_qualification_and_repeated_rows(monkeypatch):
    seen = []
    native_candidates = ("A.MI", "A.DE", "A.PA", "A.AS")
    with owner(monkeypatch):
        def callback(row):
            own = []
            for symbol in native_candidates:
                seen.append((row, symbol))
                # A positive first source does not prune later venues.
                own.append((symbol, {"native": True, "qualified": True}))
            return own

        batch = votes.acquire_third_source_votes(
            [lambda row=row: callback(row) for row in range(3)], parallel=True,
        )
    assert batch.interruption is None
    assert len(seen) == 12
    for row in range(3):
        assert [symbol for index, symbol in seen if index == row] == list(native_candidates)
        assert [symbol for symbol, _facts in batch.outcomes[row].result] == list(native_candidates)


@pytest.mark.parametrize("parallel", [False, True])
def test_no_helper_cache_of_empty_error_or_successful_result(monkeypatch, parallel):
    attempts = []
    error = TimeoutError("synthetic")
    lock = threading.Lock()

    def fetch():
        with lock:
            attempts.append(len(attempts))
            ordinal = len(attempts)
        if ordinal == 1:
            return None
        if ordinal == 2:
            raise error
        return {"recovered": True}

    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes([fetch, fetch, fetch], parallel=parallel)
    assert len(attempts) == 3
    assert sum(outcome.result is None and outcome.error is None for outcome in batch.outcomes) == 1
    assert sum(outcome.error is error for outcome in batch.outcomes) == 1
    assert sum(outcome.result == {"recovered": True} for outcome in batch.outcomes) == 1
    assert batch.interruption is None


def test_children_keep_exact_optional_deadline_and_transport_owner(monkeypatch):
    with owner(monkeypatch) as (session, clock):
        with session.transport.optional_work("synthetic-third-votes"):
            authority = session.transport.optional_work_token()

            def callback():
                assert current_session() is session
                assert session.transport.optional_work_token() == authority
                return session.transport.call("yahoo", lambda timeout: timeout)

            batch = votes.acquire_third_source_votes([callback] * 8, parallel=True)
        assert [outcome.result for outcome in batch.outcomes] == [2] * 8
        assert clock[0] == 0
        assert session.transport.call("yahoo", lambda timeout: timeout) == 20


@pytest.mark.parametrize("capacity", [2, 4, 8])
def test_existing_essential_reserve_and_native_cap_bound_simultaneous_work(monkeypatch, capacity):
    release = threading.Event()
    essential_started = threading.Event()
    active = peak = 0
    lock = threading.Lock()
    errors = []
    with owner(monkeypatch, capacity=capacity, workers=8) as (session, _clock):
        def native(_timeout, essential=False):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == capacity:
                    release.set()
            if essential:
                essential_started.set()
            try:
                assert release.wait(timeout=3)
            finally:
                with lock:
                    active -= 1

        def essential():
            try:
                session.transport.call("yahoo", lambda cap: native(cap, essential=True))
            except BaseException as error:
                errors.append(error)

        with session.transport.essential_work():
            thread = threading.Thread(target=copy_context().run, args=(essential,))
            thread.start()
            try:
                assert essential_started.wait(timeout=3)
                with session.transport.optional_work("synthetic-third-votes"):
                    batch = votes.acquire_third_source_votes(
                        [lambda: session.transport.call("yahoo", native)] * 16, parallel=True,
                    )
            finally:
                release.set()
                thread.join(timeout=3)
        assert not thread.is_alive()
        assert errors == []
        assert batch.interruption is None
        assert all(outcome.error is None for outcome in batch.outcomes)
        assert peak == capacity and active == 0
        assert session.transport.coordinator.snapshot()["active"] == 0


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, TimeoutError, RuntimeError, CancelledError])
def test_ordinary_row_errors_are_local_and_do_not_abort_later_rows(monkeypatch, parallel, error_type):
    error = error_type("synthetic callback failure")
    with owner(monkeypatch):
        def fail():
            raise error

        batch = votes.acquire_third_source_votes([lambda: "before", fail, lambda: "after"],
                                                parallel=parallel)
    assert batch.interruption is None
    assert batch.owner_error is None
    assert [outcome.result for outcome in batch.outcomes] == ["before", None, "after"]
    assert batch.outcomes[1].error is error


def test_canceled_accepted_future_is_owner_fault_not_an_empty_provider_fact(monkeypatch):
    calls = 0
    executed = []

    class CanceledPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                future = Future()
                future.cancel()
                # Match a native queued work item's cancellation notification.
                assert future.set_running_or_notify_cancel() is False
                return future
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(votes, "ThreadPoolExecutor", CanceledPool)
    with owner(monkeypatch, capacity=2, workers=2):
        batch = votes.acquire_third_source_votes([
            lambda: executed.append(0) or "prefix",
            lambda: executed.append(1) or "never",
        ], parallel=True)
    assert executed == [0]
    assert batch.outcomes[0].result == "prefix"
    assert batch.outcomes[1].submitted
    assert not batch.outcomes[1].started and not batch.outcomes[1].completed
    assert isinstance(batch.outcomes[1].error, CancelledError)
    assert batch.interruption is batch.owner_error
    assert isinstance(batch.owner_error, CancelledError)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, CancelledError])
@pytest.mark.parametrize("child_cancellation", [False, True])
def test_failed_future_before_observe_is_owner_fault_and_stops_admission(
    monkeypatch, error_type, child_cancellation,
):
    secondary = error_type("synthetic executor fault before observer")
    primary = SystemExit("synthetic earlier callback cancellation")
    second_submitted = threading.Event()
    calls = 0
    seen = []

    class SetupFaultPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                future = Future()
                future.set_exception(secondary)
                second_submitted.set()
                return future
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(votes, "ThreadPoolExecutor", SetupFaultPool)
    with owner(monkeypatch, capacity=2, workers=2):
        def prefix():
            assert second_submitted.wait(timeout=3)
            seen.append(0)
            if child_cancellation:
                raise primary
            return "prefix"

        batch = votes.acquire_third_source_votes([
            prefix, lambda: seen.append(1), lambda: seen.append(2),
        ], parallel=True)
    assert seen == [0]
    assert len(batch.outcomes) == 2
    assert batch.owner_error is secondary
    assert batch.interruption is (primary if child_cancellation else secondary)
    failed = batch.outcomes[1]
    assert failed.submitted and not failed.started and not failed.completed
    assert failed.error is secondary and failed.result is None


@pytest.mark.parametrize("returned", ["none", "outcome", "foreign"])
@pytest.mark.parametrize("child_cancellation", [False, True])
def test_successful_future_without_observe_is_owner_contract_fault(
    monkeypatch, returned, child_cancellation,
):
    primary = SystemExit("synthetic earlier callback cancellation")
    second_submitted = threading.Event()
    calls = 0
    seen = []

    class MissingObserverPool(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # Native submit receives (copy_context.run, _observe, outcome,
                # callback). Neither _observe nor the callback executes here.
                outcome = args[1]
                value = None if returned == "none" else (
                    outcome if returned == "outcome" else {"foreign": "not a row"}
                )
                future = Future()
                future.set_result(value)
                second_submitted.set()
                return future
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(votes, "ThreadPoolExecutor", MissingObserverPool)
    with owner(monkeypatch, capacity=2, workers=2):
        def prefix():
            assert second_submitted.wait(timeout=3)
            seen.append(0)
            if child_cancellation:
                raise primary
            return "prefix"

        batch = votes.acquire_third_source_votes([
            prefix, lambda: seen.append(1), lambda: seen.append(2),
        ], parallel=True)
    assert seen == [0]
    assert len(batch.outcomes) == 2
    assert isinstance(batch.owner_error, RuntimeError)
    assert batch.interruption is (primary if child_cancellation else batch.owner_error)
    failed = batch.outcomes[1]
    assert failed.submitted and not failed.started and not failed.completed
    assert failed.error is batch.owner_error and failed.result is None


def test_observed_callback_returning_none_is_legitimate_and_not_owner_fault(monkeypatch):
    seen = []
    with owner(monkeypatch, capacity=2, workers=2):
        batch = votes.acquire_third_source_votes([
            lambda: seen.append(0), lambda: seen.append(1), lambda: seen.append(2),
        ], parallel=True)
    assert sorted(seen) == [0, 1, 2]
    assert batch.owner_error is batch.interruption is None
    assert all(outcome.started and outcome.completed for outcome in batch.outcomes)
    assert all(outcome.result is None and outcome.error is None for outcome in batch.outcomes)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_executor_constructor_failure_is_owner_fault_and_calls_no_source(monkeypatch, error_type):
    error = error_type("synthetic executor initialization")
    executed = []

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(votes, "ThreadPoolExecutor", fail)
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes(
            [lambda: executed.append("never")] * 2, parallel=True,
        )
    assert executed == []
    assert batch.outcomes == ()
    assert batch.owner_error is batch.interruption is error


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_iter_initialization_failure_is_owned_before_any_callback(monkeypatch, parallel, error_type):
    error = error_type("synthetic iterable initialization")

    class BrokenIterable:
        def __iter__(self):
            raise error

    forbid_executor(monkeypatch)
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes(BrokenIterable(), parallel=parallel)
    assert batch.outcomes == ()
    assert batch.owner_error is batch.interruption is error


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_serial_cancellation_preserves_prefix_and_never_calls_later_row(monkeypatch, error_type):
    error = error_type("synthetic")
    seen = []
    with owner(monkeypatch):
        def fail():
            seen.append("fail")
            raise error

        batch = votes.acquire_third_source_votes([
            lambda: seen.append("before") or "before", fail, lambda: seen.append("after"),
        ])
    assert seen == ["before", "fail"]
    assert [outcome.result for outcome in batch.outcomes] == ["before", None]
    assert batch.interruption is error


def test_canonical_first_row_cancellation_wins_after_every_accepted_sweep_joins(monkeypatch):
    rendezvous = threading.Barrier(4)
    later_canceled = threading.Event()
    earlier = SystemExit("earlier row")
    later = KeyboardInterrupt("later row")
    exited = []
    with owner(monkeypatch):
        def callback(index):
            try:
                rendezvous.wait(timeout=3)
                if index == 3:
                    later_canceled.set()
                    raise later
                if index == 0:
                    assert later_canceled.wait(timeout=3)
                    raise earlier
                return index
            finally:
                exited.append(index)

        batch = votes.acquire_third_source_votes(
            [lambda index=index: callback(index) for index in range(20)], parallel=True,
        )
    assert sorted(exited) == [0, 1, 2, 3]
    assert batch.interruption is earlier
    assert all(outcome.completed for outcome in batch.outcomes)
    assert len(batch.outcomes) == 4


@pytest.mark.parametrize("owner_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_wait_fault_drains_all_children_and_retains_owner_or_canonical_primary(monkeypatch, owner_type):
    release = threading.Event()
    owner_error = owner_type("synthetic owner wait")
    child_error = KeyboardInterrupt("synthetic child")
    native_wait = votes.wait
    exited = []

    def failing_wait(futures, **kwargs):
        release.set()
        native_wait(futures, **kwargs)
        raise owner_error

    monkeypatch.setattr(votes, "wait", failing_wait)
    with owner(monkeypatch):
        def callback(index):
            try:
                assert release.wait(timeout=3)
                if index == 0:
                    raise child_error
                return index
            finally:
                exited.append(index)

        batch = votes.acquire_third_source_votes(
            [lambda index=index: callback(index) for index in range(4)], parallel=True,
        )
    assert sorted(exited) == [0, 1, 2, 3]
    assert batch.owner_error is owner_error
    primary = child_error if issubclass(owner_type, Exception) else owner_error
    assert batch.interruption is primary
    assert all(outcome.completed for outcome in batch.outcomes)


def test_interruption_of_drain_wait_still_joins_callbacks_and_beats_owner_exception(monkeypatch):
    release = threading.Event()
    owner_error = RuntimeError("synthetic wait observer")
    interruption = KeyboardInterrupt("synthetic owner interruption")
    native_result = Future.result
    injected = False

    def interrupted_result(future, *args, **kwargs):
        nonlocal injected
        if threading.current_thread() is threading.main_thread() and not injected:
            injected = True
            release.set()
            raise interruption
        return native_result(future, *args, **kwargs)

    monkeypatch.setattr(votes, "wait", lambda *a, **kw: (_ for _ in ()).throw(owner_error))
    monkeypatch.setattr(Future, "result", interrupted_result)
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes(
            [lambda: release.wait(timeout=3)] * 4, parallel=True,
        )
    assert batch.owner_error is interruption
    assert batch.interruption is interruption
    assert all(outcome.completed for outcome in batch.outcomes)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_partial_submit_fault_does_not_rerun_or_abandon_accepted_rows(monkeypatch, error_type):
    error = error_type("synthetic submit")
    release = threading.Event()
    seen = []
    calls = 0

    class FailingPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                release.set()
                raise error
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(votes, "ThreadPoolExecutor", FailingPool)
    with owner(monkeypatch):
        def callback(index):
            seen.append(index)
            assert release.wait(timeout=3)
            return index

        batch = votes.acquire_third_source_votes(
            [lambda index=index: callback(index) for index in range(20)], parallel=True,
        )
    assert sorted(seen) == [0, 1]
    assert [outcome.result for outcome in batch.outcomes] == [0, 1, None]
    assert batch.owner_error is error
    assert batch.interruption is error
    assert not batch.outcomes[-1].submitted
    assert all(outcome.completed for outcome in batch.outcomes[:-1])


def test_post_enqueue_submit_and_cleanup_faults_drain_hidden_accepted_callback(monkeypatch):
    release = threading.Event()
    hidden_started = threading.Event()
    owner_error = RuntimeError("synthetic post-enqueue submit")
    observer_error = ValueError("synthetic cleanup observer")
    child_error = SystemExit("synthetic hidden child")
    submit_calls = 0

    class FailingPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal submit_calls
            submit_calls += 1
            future = super().submit(*args, **kwargs)
            if submit_calls == 2:
                assert hidden_started.wait(timeout=3)
                raise owner_error
            return future

        def shutdown(self, **kwargs):
            release.set()
            raise observer_error

    monkeypatch.setattr(votes, "ThreadPoolExecutor", FailingPool)
    with owner(monkeypatch):
        def hidden():
            hidden_started.set()
            assert release.wait(timeout=3)
            raise child_error

        batch = votes.acquire_third_source_votes([lambda: "prefix", hidden], parallel=True)
    assert len(batch.outcomes) == 2
    assert batch.outcomes[0].result == "prefix"
    assert batch.outcomes[1].submitted and batch.outcomes[1].completed
    assert batch.owner_error is owner_error
    assert batch.interruption is child_error


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_iterator_setup_failure_retains_the_completed_prefix(monkeypatch, parallel, error_type):
    release = threading.Event()
    error = error_type("synthetic iterator")

    def callbacks():
        yield lambda: release.wait(timeout=3) or "prefix"
        release.set()
        raise error

    # Serial setup cannot advance the iterator until the first callback returns.
    if not parallel:
        release.set()
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes(callbacks(), parallel=parallel)
    assert len(batch.outcomes) == 1
    assert batch.outcomes[0].completed
    assert batch.owner_error is error
    assert batch.interruption is error


@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_callback_does_not_turn_unsubmitted_work_into_provider_absence(monkeypatch, parallel):
    with owner(monkeypatch):
        batch = votes.acquire_third_source_votes([lambda: "prefix", None], parallel=parallel)
    assert len(batch.outcomes) == 1
    assert batch.outcomes[0].result == "prefix"
    assert isinstance(batch.owner_error, TypeError)


def test_fact_memo_and_diagnostics_remain_isolated_between_fresh_owners(monkeypatch):
    results = []
    sessions = []
    for value in ("first", "second"):
        with owner(monkeypatch) as (session, _clock):
            sessions.append(session)

            def callback():
                assert current_session() is session
                dq.info("synthetic-vote", value)
                return memoized("synthetic-votes", "same", lambda: value, legacy={})

            batch = votes.acquire_third_source_votes([callback] * 4, parallel=True)
            assert dq.issues() == []
            for outcome in batch.outcomes:
                dq.replay(outcome.diagnostics)
            assert len(dq.issues()) == 4
            results.append([outcome.result for outcome in batch.outcomes])
    assert sessions[0] is not sessions[1]
    assert sessions[0].providers is not sessions[1].providers
    assert results == [["first"] * 4, ["second"] * 4]
