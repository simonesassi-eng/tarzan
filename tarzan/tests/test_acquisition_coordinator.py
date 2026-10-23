"""Synthetic admission/fault schedules; no provider or portfolio data is used."""

from contextlib import contextmanager
import threading
import time
from types import SimpleNamespace

import pytest

from tarzan.runtime import acquisition
from tarzan.runtime.acquisition import (
    RequestCoordinator,
    RequestCoordinatorClosed,
    RequestDeadlineExceeded,
)


class Cancelled(BaseException):
    pass


class NativeDeadline(RuntimeError):
    pass


def allowance():
    return 10.0


def until(predicate, *, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("synthetic admission schedule did not reach its boundary")


@contextmanager
def workers():
    """Always unblock synthetic callbacks; a broken coordinator cannot hang pytest."""
    threads = []
    errors = []
    releases = []
    lock = threading.Lock()

    def spawn(callback):
        def invoke():
            try:
                callback()
            except BaseException as error:
                with lock:
                    errors.append(error)
        thread = threading.Thread(target=invoke, daemon=True)
        threads.append(thread)
        thread.start()
        return thread

    def release_event():
        event = threading.Event()
        releases.append(event)
        return event

    try:
        yield spawn, errors, release_event
    finally:
        for event in releases:
            event.set()
        for thread in threads:
            thread.join(timeout=3.0)
        assert not any(thread.is_alive() for thread in threads), "native admission leaked a worker"


@pytest.mark.parametrize("capacity", [0, -1, True, False, 1.0, "4", None])
def test_capacity_is_a_positive_python_integer(capacity):
    with pytest.raises(ValueError):
        RequestCoordinator(capacity)


def test_public_capacity_cannot_be_reassigned_to_bypass_admission():
    coordinator = RequestCoordinator(4)
    with pytest.raises(AttributeError):
        coordinator.capacity = 1000
    assert coordinator.capacity == coordinator.snapshot()["capacity"] == 4


@pytest.mark.parametrize("optional", [0, 1, None, "", "optional"])
def test_priority_requires_an_explicit_boolean(optional):
    coordinator = RequestCoordinator(4)
    with pytest.raises(ValueError):
        with coordinator.lease(optional=optional, remaining=allowance):
            pytest.fail("invalid priority acquired a permit")
    assert coordinator.snapshot()["requested"] == 0


def test_remaining_port_must_be_callable():
    coordinator = RequestCoordinator(1)
    with pytest.raises(TypeError):
        with coordinator.lease(optional=False, remaining=10.0):
            pytest.fail("invalid deadline port acquired a permit")
    assert coordinator.snapshot()["active"] == 0


@pytest.mark.parametrize("capacity", [1, 4, 8])
def test_lease_acquires_and_releases_a_single_native_slot(capacity):
    coordinator = RequestCoordinator(capacity)
    with coordinator.lease(optional=False, remaining=allowance):
        assert coordinator.snapshot()["active"] == 1
    evidence = coordinator.snapshot()
    assert evidence["active"] == 0
    assert evidence["peak"] == 1
    assert evidence["requested"] == evidence["acquired"] == evidence["released"] == 1
    assert evidence["rejected"] == 0


@pytest.mark.parametrize("capacity", [1, 4, 8])
def test_aggregate_native_capacity_is_not_multiplied_by_worker_count(capacity):
    coordinator = RequestCoordinator(capacity)
    entered = [threading.Event() for _ in range(capacity + 3)]
    with workers() as (spawn, errors, event):
        release = event()
        def request(index):
            with coordinator.lease(optional=False, remaining=allowance):
                entered[index].set()
                assert release.wait(timeout=2.0)
        for index in range(capacity + 3):
            spawn(lambda index=index: request(index))
        until(lambda: sum(item.is_set() for item in entered) == capacity)
        until(lambda: coordinator.snapshot()["queued_essential"] == 3)
        assert coordinator.snapshot()["peak"] == capacity
        release.set()
        until(lambda: coordinator.snapshot()["released"] == capacity + 3)
    assert errors == []
    assert coordinator.snapshot()["active"] == 0


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
def test_optional_only_uses_full_capacity_without_a_false_reserve(capacity):
    coordinator = RequestCoordinator(capacity)
    with workers() as (spawn, errors, event):
        release = event()
        def request():
            with coordinator.lease(optional=True, remaining=allowance):
                assert release.wait(timeout=2.0)
        for _ in range(capacity + 2):
            spawn(request)
        until(lambda: coordinator.snapshot()["active_optional"] == capacity)
        until(lambda: coordinator.snapshot()["queued_optional"] == 2)
        assert coordinator.snapshot()["active"] == capacity
        release.set()
        until(lambda: coordinator.snapshot()["released"] == capacity + 2)
    assert errors == []


@pytest.mark.parametrize("capacity", [1, 2, 4, 8])
def test_declared_essential_demand_reserves_one_slot_except_at_capacity_one(capacity):
    coordinator = RequestCoordinator(capacity)
    optional_limit = max(capacity - 1, 1)
    with workers() as (spawn, errors, event):
        release = event()
        def request():
            with coordinator.lease(optional=True, remaining=allowance):
                assert release.wait(timeout=2.0)
        with coordinator.essential_scope():
            assert coordinator.snapshot()["active"] == 0
            assert coordinator.snapshot()["essential_demand"] == 1
            for _ in range(capacity + 2):
                spawn(request)
            until(lambda: coordinator.snapshot()["active_optional"] == optional_limit)
            until(lambda: coordinator.snapshot()["queued_optional"] == capacity + 2 - optional_limit)
            release.set()
            until(lambda: coordinator.snapshot()["released"] == capacity + 2)
    assert errors == []
    assert coordinator.snapshot()["essential_demand"] == 0


def test_reserved_slot_admits_essential_while_optional_queue_is_saturated():
    coordinator = RequestCoordinator(4)
    optional_entered = []
    essential_entered = threading.Event()
    with workers() as (spawn, errors, event):
        release = event()
        guard = threading.Lock()
        def optional():
            with coordinator.lease(optional=True, remaining=allowance):
                with guard:
                    optional_entered.append(True)
                assert release.wait(timeout=2.0)
        def essential():
            with coordinator.lease(optional=False, remaining=allowance):
                essential_entered.set()
                assert release.wait(timeout=2.0)
        with coordinator.essential_scope():
            for _ in range(6):
                spawn(optional)
            until(lambda: coordinator.snapshot()["queued_optional"] == 3)
            spawn(essential)
            assert essential_entered.wait(timeout=1.0)
            assert coordinator.snapshot()["active"] == coordinator.snapshot()["peak"] == 4
            assert len(optional_entered) == 3
            release.set()
            until(lambda: coordinator.snapshot()["released"] == 7)
    assert errors == []


@pytest.mark.parametrize("capacity", [1, 4, 8])
def test_late_essential_demand_does_not_preempt_io_but_gets_the_next_free_slot(capacity):
    coordinator = RequestCoordinator(capacity)
    essential_entered = threading.Event()
    next_optional_entered = threading.Event()
    with workers() as (spawn, errors, event):
        releases = [event() for _ in range(capacity)]
        release_essential = event()
        release_next_optional = event()
        def optional(index):
            with coordinator.lease(optional=True, remaining=allowance):
                assert releases[index].wait(timeout=2.0)
        def essential():
            with coordinator.lease(optional=False, remaining=allowance):
                essential_entered.set()
                assert release_essential.wait(timeout=2.0)
        def next_optional():
            with coordinator.lease(optional=True, remaining=allowance):
                next_optional_entered.set()
                assert release_next_optional.wait(timeout=2.0)
        for index in range(capacity):
            spawn(lambda index=index: optional(index))
        until(lambda: coordinator.snapshot()["active_optional"] == capacity)
        with coordinator.essential_scope():
            spawn(essential)
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            spawn(next_optional)
            until(lambda: coordinator.snapshot()["queued_optional"] == 1)
            assert not essential_entered.is_set()
            assert coordinator.snapshot()["active_optional"] == capacity
            releases[0].set()
            assert essential_entered.wait(timeout=1.0)
            assert not next_optional_entered.is_set()
            assert coordinator.snapshot()["active"] == capacity
            assert coordinator.snapshot()["active_optional"] == capacity - 1
            for release in releases:
                release.set()
            release_essential.set()
            release_next_optional.set()
            until(lambda: coordinator.snapshot()["released"] == capacity + 2)
    assert errors == []
    assert coordinator.snapshot()["essential_demand"] == 0


def test_leaving_essential_scope_unblocks_optional_reserve_without_waiting_for_io():
    coordinator = RequestCoordinator(4)
    fourth_entered = threading.Event()
    with workers() as (spawn, errors, event):
        release = event()
        def request(index):
            with coordinator.lease(optional=True, remaining=allowance):
                if index == 3:
                    fourth_entered.set()
                assert release.wait(timeout=2.0)
        with coordinator.essential_scope():
            for index in range(4):
                spawn(lambda index=index: request(index))
                until(lambda: coordinator.snapshot()["active"] == min(index + 1, 3))
            until(lambda: coordinator.snapshot()["queued_optional"] == 1)
            assert not fourth_entered.is_set()
        assert fourth_entered.wait(timeout=1.0)
        assert coordinator.snapshot()["active_optional"] == 4
        release.set()
        until(lambda: coordinator.snapshot()["released"] == 4)
    assert errors == []


@pytest.mark.parametrize("error", [RuntimeError("scope fault"), Cancelled(), KeyboardInterrupt()])
def test_essential_scope_baseexception_exit_never_leaves_ghost_demand(error):
    coordinator = RequestCoordinator(4)
    with pytest.raises(type(error)):
        with coordinator.essential_scope():
            with coordinator.essential_scope():
                assert coordinator.snapshot()["essential_demand"] == 2
                assert coordinator.snapshot()["active"] == 0
                raise error
    assert coordinator.snapshot()["essential_demand"] == 0
    assert coordinator.snapshot()["requested"] == 0


def test_reopen_also_requires_logical_demand_to_drain():
    coordinator = RequestCoordinator(1)
    with coordinator.essential_scope():
        coordinator.close()
        with pytest.raises(RuntimeError, match="drain"):
            coordinator.reopen()
    coordinator.reopen()
    with coordinator.essential_scope():
        _native_noop(coordinator, allowance)
    assert coordinator.snapshot()["essential_demand"] == 0


def test_closed_coordinator_rejects_new_logical_scopes_without_a_counter_leak():
    coordinator = RequestCoordinator(1)
    coordinator.close()
    with pytest.raises(RequestCoordinatorClosed):
        with coordinator.essential_scope():
            pytest.fail("closed coordinator accepted logical demand")
    assert coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("optional", [False, True])
def test_fifo_within_each_priority_is_queue_admission_order(optional):
    coordinator = RequestCoordinator(1)
    observed = []
    queue_name = "queued_optional" if optional else "queued_essential"
    with workers() as (spawn, errors, _event):
        def request(index):
            with coordinator.lease(optional=optional, remaining=allowance):
                observed.append(index)
        with coordinator.lease(optional=False, remaining=allowance):
            for index in range(8):
                spawn(lambda index=index: request(index))
                until(lambda: coordinator.snapshot()[queue_name] == index + 1)
        until(lambda: len(observed) == 8)
    assert errors == []
    assert observed == list(range(8))


def test_waiting_essential_fifo_precedes_older_optional_tickets():
    coordinator = RequestCoordinator(1)
    observed = []
    schedule = [("O1", True), ("O2", True), ("E1", False), ("E2", False)]
    with workers() as (spawn, errors, _event):
        def request(label, optional):
            with coordinator.lease(optional=optional, remaining=allowance):
                observed.append(label)
        with coordinator.lease(optional=False, remaining=allowance):
            for position, (label, optional) in enumerate(schedule):
                spawn(lambda label=label, optional=optional: request(label, optional))
                until(lambda: coordinator.snapshot()["queued_essential"]
                      + coordinator.snapshot()["queued_optional"] == position + 1)
        until(lambda: len(observed) == 4)
    assert errors == []
    assert observed == ["E1", "E2", "O1", "O2"]


@pytest.mark.parametrize("error", [RuntimeError("synthetic fault"), Cancelled(), KeyboardInterrupt()])
def test_native_callback_baseexception_always_releases_the_permit(error):
    coordinator = RequestCoordinator(1)
    with pytest.raises(type(error)):
        with coordinator.lease(optional=False, remaining=allowance):
            raise error
    with coordinator.lease(optional=False, remaining=allowance):
        assert coordinator.snapshot()["active"] == 1
    assert coordinator.snapshot()["released"] == 2


@pytest.mark.parametrize("capacity", [1, 4])
def test_recursive_native_callback_fails_fast_instead_of_bypassing_capacity(capacity):
    coordinator = RequestCoordinator(capacity)
    with coordinator.lease(optional=False, remaining=allowance):
        with pytest.raises(RuntimeError, match="recursive"):
            with coordinator.lease(optional=True, remaining=allowance):
                pytest.fail("nested callback bypassed aggregate admission")
        assert coordinator.snapshot()["active"] == 1
    assert coordinator.snapshot()["requested"] == 1


def test_deadline_callback_is_not_executed_under_the_condition_lock():
    coordinator = RequestCoordinator(1)
    def remaining():
        assert coordinator.snapshot()["active"] in (0, 1)
        return 10.0
    with workers() as (spawn, errors, _event):
        thread = spawn(lambda: _native_noop(coordinator, remaining))
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "deadline callback ran while admission lock was held"
    assert errors == []


def _native_noop(coordinator, remaining):
    with coordinator.lease(optional=False, remaining=remaining):
        coordinator.snapshot()


def test_recursive_deadline_port_also_fails_fast_and_leaves_no_queue():
    coordinator = RequestCoordinator(1)
    def remaining():
        with coordinator.lease(optional=False, remaining=allowance):
            pytest.fail("recursive deadline port acquired a nested slot")
        return 10.0
    with pytest.raises(RuntimeError, match="recursive"):
        with coordinator.lease(optional=False, remaining=remaining):
            pytest.fail("recursive deadline acquired an outer slot")
    assert coordinator.snapshot()["active"] == coordinator.snapshot()["queued_essential"] == 0
    assert coordinator.snapshot()["rejected"] == 1


@pytest.mark.parametrize(
    "value", [None, True, "10", float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_invalid_remaining_values_fail_closed_without_a_native_callback(value):
    coordinator = RequestCoordinator(1)
    with pytest.raises(ValueError):
        with coordinator.lease(optional=False, remaining=lambda: value):
            pytest.fail("invalid remaining time reached native I/O")
    assert coordinator.snapshot()["active"] == 0
    assert coordinator.snapshot()["rejected"] == 1


@pytest.mark.parametrize("fault_at", [1, 2])
def test_preparatory_clock_cancellation_does_not_leave_a_recursive_guard_or_ticket(
    monkeypatch, fault_at,
):
    coordinator = RequestCoordinator(1)
    calls = []
    error = Cancelled()
    def monotonic():
        calls.append(True)
        if len(calls) == fault_at:
            raise error
        return float(len(calls))
    monkeypatch.setattr(acquisition, "time", SimpleNamespace(monotonic=monotonic))
    with pytest.raises(Cancelled) as raised:
        _native_noop(coordinator, allowance)
    assert raised.value is error
    assert coordinator.snapshot()["active"] == coordinator.snapshot()["queued_essential"] == 0
    assert coordinator.snapshot()["requested"] == int(fault_at == 2)
    assert coordinator.snapshot()["rejected"] == int(fault_at == 2)
    _native_noop(coordinator, allowance)
    assert coordinator.snapshot()["acquired"] == coordinator.snapshot()["released"] == 1


def test_condition_wait_baseexception_removes_its_ticket_before_the_next_attempt(monkeypatch):
    coordinator = RequestCoordinator(1)
    original_wait = coordinator._condition.wait
    error = Cancelled()
    raised = []
    def interrupted_wait(*args, **kwargs):
        if not raised:
            raised.append(True)
            raise error
        return original_wait(*args, **kwargs)
    monkeypatch.setattr(coordinator._condition, "wait", interrupted_wait)
    with workers() as (spawn, errors, _event):
        with coordinator.lease(optional=False, remaining=allowance):
            first = spawn(lambda: _native_noop(coordinator, allowance))
            first.join(timeout=1.0)
            assert not first.is_alive()
            assert coordinator.snapshot()["queued_essential"] == 0
            second = spawn(lambda: _native_noop(coordinator, allowance))
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
        second.join(timeout=1.0)
        assert not second.is_alive()
    assert errors == [error]
    assert coordinator.snapshot()["requested"] == 3
    assert coordinator.snapshot()["released"] == 2
    assert coordinator.snapshot()["rejected"] == 1


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_expired_remaining_value_has_a_specific_admission_timeout(value):
    coordinator = RequestCoordinator(1)
    with pytest.raises(RequestDeadlineExceeded):
        with coordinator.lease(optional=False, remaining=lambda: value):
            pytest.fail("expired admission reached native I/O")
    assert coordinator.snapshot()["acquired"] == 0


def test_deadline_expiring_after_reservation_releases_without_starting_io():
    coordinator = RequestCoordinator(1)
    counts = []
    def remaining():
        counts.append(True)
        if len(counts) > 1:
            raise NativeDeadline("synthetic native deadline")
        return 10.0
    with pytest.raises(NativeDeadline):
        with coordinator.lease(optional=False, remaining=remaining):
            pytest.fail("deadline expired between reservation and native I/O")
    assert coordinator.snapshot()["acquired"] == coordinator.snapshot()["released"] == 1
    assert coordinator.snapshot()["active"] == 0


def test_queued_native_deadline_identity_is_preserved_and_ticket_is_removed():
    coordinator = RequestCoordinator(1)
    expired = threading.Event()
    native_error = NativeDeadline("synthetic original budget exception")
    def remaining():
        if expired.is_set():
            raise native_error
        return 10.0
    with workers() as (spawn, errors, _event):
        with coordinator.lease(optional=False, remaining=allowance):
            thread = spawn(lambda: _native_noop(coordinator, remaining))
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            expired.set()
            thread.join(timeout=1.0)
            assert not thread.is_alive()
            assert coordinator.snapshot()["queued_essential"] == 0
    assert errors == [native_error]
    assert coordinator.snapshot()["rejected"] == 1


def test_cancelled_head_waiter_does_not_poison_following_fifo_tickets():
    coordinator = RequestCoordinator(1)
    cancel = threading.Event()
    completed = []
    def remaining():
        if cancel.is_set():
            raise Cancelled()
        return 10.0
    with workers() as (spawn, errors, _event):
        def next_request():
            with coordinator.lease(optional=False, remaining=allowance):
                completed.append("next")
        with coordinator.lease(optional=False, remaining=allowance):
            spawn(lambda: _native_noop(coordinator, remaining))
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            spawn(next_request)
            until(lambda: coordinator.snapshot()["queued_essential"] == 2)
            cancel.set()
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
        until(lambda: completed == ["next"])
    assert len(errors) == 1 and isinstance(errors[0], Cancelled)


def test_optional_admission_deadline_cannot_infect_independent_essential_work():
    coordinator = RequestCoordinator(1)
    expire_optional = threading.Event()
    essential_done = threading.Event()
    def optional_remaining():
        if expire_optional.is_set():
            raise NativeDeadline("synthetic optional-only deadline")
        return 10.0
    with workers() as (spawn, errors, _event):
        def optional():
            with coordinator.lease(optional=True, remaining=optional_remaining):
                pytest.fail("expired optional worker reached native I/O")
        def essential():
            with coordinator.lease(optional=False, remaining=allowance):
                essential_done.set()
        with coordinator.lease(optional=False, remaining=allowance):
            spawn(optional)
            until(lambda: coordinator.snapshot()["queued_optional"] == 1)
            spawn(essential)
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            expire_optional.set()
            until(lambda: coordinator.snapshot()["queued_optional"] == 0)
        assert essential_done.wait(timeout=1.0)
    assert len(errors) == 1 and isinstance(errors[0], NativeDeadline)
    assert coordinator.snapshot()["released"] == 2


def test_close_wakes_waiters_but_does_not_release_an_active_native_callback():
    coordinator = RequestCoordinator(1)
    with workers() as (spawn, errors, _event):
        with coordinator.lease(optional=False, remaining=allowance):
            thread = spawn(lambda: _native_noop(coordinator, allowance))
            until(lambda: coordinator.snapshot()["queued_essential"] == 1)
            coordinator.close()
            thread.join(timeout=1.0)
            assert not thread.is_alive()
            assert coordinator.snapshot()["active"] == 1
            assert coordinator.snapshot()["closed"] is True
        assert coordinator.snapshot()["active"] == 0
    assert len(errors) == 1 and isinstance(errors[0], RequestCoordinatorClosed)
    with pytest.raises(RequestCoordinatorClosed):
        _native_noop(coordinator, allowance)


def test_reopen_requires_drain_and_can_resume_after_worker_join():
    coordinator = RequestCoordinator(1)
    with coordinator.lease(optional=False, remaining=allowance):
        coordinator.close()
        with pytest.raises(RuntimeError, match="drain"):
            coordinator.reopen()
    coordinator.reopen()
    _native_noop(coordinator, allowance)
    assert coordinator.snapshot()["closed"] is False
    assert coordinator.snapshot()["acquired"] == coordinator.snapshot()["released"] == 2


def test_reopen_rejects_an_unqueued_caller_inside_its_deadline_port():
    coordinator = RequestCoordinator(1)
    inside_remaining = threading.Event()
    with workers() as (spawn, errors, event):
        finish_remaining = event()
        def remaining():
            inside_remaining.set()
            assert finish_remaining.wait(timeout=2.0)
            return 10.0
        caller = spawn(lambda: _native_noop(coordinator, remaining))
        assert inside_remaining.wait(timeout=1.0)
        assert coordinator.snapshot()["requested"] == 1
        assert coordinator.snapshot()["queued_essential"] == 0
        assert coordinator.snapshot()["active"] == 0
        coordinator.close()
        with pytest.raises(RuntimeError, match="drain"):
            coordinator.reopen()
        finish_remaining.set()
        caller.join(timeout=1.0)
        assert not caller.is_alive()
        coordinator.reopen()
    assert len(errors) == 1 and isinstance(errors[0], RequestCoordinatorClosed)
    _native_noop(coordinator, allowance)
    assert coordinator.snapshot()["requested"] == 2
    assert coordinator.snapshot()["rejected"] == coordinator.snapshot()["released"] == 1


def test_retry_backoff_outside_lease_leaves_capacity_for_another_request():
    coordinator = RequestCoordinator(1)
    first_done = threading.Event()
    backoff = threading.Event()
    essential_done = threading.Event()
    with workers() as (spawn, errors, event):
        release_backoff = event()
        def retrying():
            with coordinator.lease(optional=True, remaining=allowance):
                first_done.set()
            backoff.set()
            assert release_backoff.wait(timeout=2.0)
            with coordinator.lease(optional=True, remaining=allowance):
                pass
        def essential():
            assert first_done.wait(timeout=1.0)
            with coordinator.lease(optional=False, remaining=allowance):
                essential_done.set()
        spawn(retrying)
        assert backoff.wait(timeout=1.0)
        spawn(essential)
        assert essential_done.wait(timeout=1.0)
        release_backoff.set()
        until(lambda: coordinator.snapshot()["released"] == 3)
    assert errors == []


def test_parent_waiting_for_child_future_does_not_own_a_native_permit():
    coordinator = RequestCoordinator(1)
    child_done = threading.Event()
    with workers() as (spawn, errors, _event):
        def parent():
            with coordinator.essential_scope():
                assert coordinator.snapshot()["active"] == 0
                child = spawn(lambda: _native_noop(coordinator, allowance))
                child.join(timeout=1.0)  # Logical demand owns no native slot.
                assert not child.is_alive()
                child_done.set()
        spawn(parent)
        assert child_done.wait(timeout=1.5)
    assert errors == []
    assert coordinator.snapshot()["acquired"] == coordinator.snapshot()["released"] == 1


def test_observations_are_fresh_numeric_snapshots_not_request_data():
    coordinator = RequestCoordinator(4)
    first = coordinator.snapshot()
    first["active"] = 100
    assert coordinator.snapshot()["active"] == 0
    _native_noop(coordinator, allowance)
    evidence = coordinator.snapshot()
    assert all(type(value) in (int, float, bool) for value in evidence.values())
    assert evidence["queue_wait_seconds"] >= 0
    assert set(evidence) == {
        "capacity", "active", "active_optional", "essential_demand", "queued_essential",
        "queued_optional", "requested", "acquired", "released", "rejected",
        "peak", "queue_wait_seconds", "closed",
    }
