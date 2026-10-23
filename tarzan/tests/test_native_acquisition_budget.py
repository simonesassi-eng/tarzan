"""Synthetic native requests: admission is global; retries own no waiting slot."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from threading import Event, Lock
import time
from types import SimpleNamespace

import pytest

from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.transport import (
    OptionalProviderBudgetExceeded,
    ProviderBudget,
    ProviderBudgetExceeded,
    ProviderTransportSettings,
)
from tarzan.runtime.workload import build_workload_observation


def budget(*, capacity=4, retries=0, clock=time.monotonic, sleep=time.sleep):
    settings = ProviderTransportSettings.configured()
    settings = replace(
        settings,
        run_budget_seconds=60.0,
        optional_budget_seconds=10.0,
        publication_reserve_seconds=5.0,
        providers=tuple(replace(policy, retry_budget=retries) for policy in settings.providers),
    )
    return ProviderBudget(
        settings, RunLedger("synthetic-native-admission"),
        capacity=capacity, clock=clock, sleep=sleep,
    )


def until(predicate):
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("synthetic native schedule did not reach its boundary")


def attempts(owner):
    return [
        entry for entry in owner.ledger.entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    ]


@pytest.mark.parametrize("capacity", [1, 4, 8])
def test_mixed_providers_share_one_native_cap(capacity):
    owner = budget(capacity=capacity)
    release = Event()
    gate = Lock()
    active = peak = entered = 0

    def native(timeout):
        nonlocal active, peak, entered
        assert timeout > 0
        with gate:
            active += 1
            entered += 1
            peak = max(peak, active)
        try:
            assert release.wait(timeout=2.0)
            return object()
        finally:
            with gate:
                active -= 1

    with ThreadPoolExecutor(max_workers=capacity + 3) as pool:
        pending = [
            pool.submit(owner.call, ("yahoo", "justetf", "borsa")[i % 3], native)
            for i in range(capacity + 3)
        ]
        try:
            until(lambda: entered == capacity)
            until(lambda: owner.coordinator.snapshot()["queued_essential"] == 3)
            assert peak == capacity
        finally:
            release.set()
        assert all(future.result(timeout=3.0) is not None for future in pending)
    assert len(attempts(owner)) == capacity + 3
    assert owner.coordinator.snapshot()["active"] == 0


def test_optional_scope_priority_does_not_depend_on_which_deadline_is_shortest():
    owner = budget()
    seen = []
    original = owner.coordinator.lease

    def lease(**arguments):
        seen.append(arguments["optional"])
        return original(**arguments)

    owner.coordinator.lease = lease
    with owner.optional_work("synthetic"):
        # This operation's timeout is shorter than the optional stage, but
        # its request still belongs to OPTIONAL work for queue admission.
        owner.policies["yahoo"] = replace(owner.policies["yahoo"], timeout_seconds=1.0)
        assert owner.call("yahoo", lambda timeout: 1) == 1
    assert owner.call("yahoo", lambda timeout: 2) == 2
    assert seen == [True, False]


def test_essential_scope_reserves_a_slot_until_all_children_have_joined():
    owner = budget(capacity=4)
    release = Event()
    essential_entered = Event()

    def optional():
        with owner.optional_work("synthetic-optional"):
            return owner.call("justetf", lambda timeout: release.wait(timeout=2.0))

    def essential():
        def native(timeout):
            essential_entered.set()
            return release.wait(timeout=2.0)
        return owner.call("yahoo", native)

    with owner.essential_work(), ThreadPoolExecutor(max_workers=7) as pool:
        pending = [pool.submit(copy_context().run, optional) for _ in range(6)]
        try:
            until(lambda: owner.coordinator.snapshot()["active_optional"] == 3)
            until(lambda: owner.coordinator.snapshot()["queued_optional"] == 3)
            primary = pool.submit(copy_context().run, essential)
            assert essential_entered.wait(timeout=1.0)
            assert owner.coordinator.snapshot()["active"] == 4
        finally:
            release.set()
        assert primary.result(timeout=3.0)
        assert all(future.result(timeout=3.0) for future in pending)
    assert owner.coordinator.snapshot()["essential_demand"] == 0


def test_retry_releases_native_permit_before_backoff():
    owner = budget(capacity=1, retries=1)
    caps = []
    during_backoff = []

    def wait(seconds):
        during_backoff.append(owner.coordinator.snapshot()["active"])
        # Capacity one must permit independent essential progress here.
        assert owner.call("borsa", lambda timeout: "independent") == "independent"

    owner.sleep = wait

    def native(timeout):
        caps.append(timeout)
        if len(caps) == 1:
            raise TimeoutError("synthetic transient transport")
        return "recovered"

    assert owner.call("yahoo", native) == "recovered"
    assert during_backoff == [0]
    assert len(attempts(owner)) == 3
    assert owner.coordinator.snapshot()["released"] == 3


def test_a_recursive_native_callback_cannot_borrow_a_slot_or_deadlock():
    owner = budget(capacity=1)
    with pytest.raises(RuntimeError, match="recursive"):
        owner.call("yahoo", lambda timeout: owner.call("borsa", lambda cap: 1))
    assert owner.coordinator.snapshot()["active"] == 0
    assert len(attempts(owner)) == 1
    assert owner.call("yahoo", lambda timeout: "next") == "next"


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_native_cancellation_releases_slot_and_preserves_primary_error(error_type):
    owner = budget(capacity=1, retries=1)
    error = error_type("synthetic cancellation")

    def native(timeout):
        raise error

    with pytest.raises(error_type) as caught:
        owner.call("yahoo", native)
    assert caught.value is error
    assert owner.coordinator.snapshot()["active"] == 0
    assert len(attempts(owner)) == 1
    assert attempts(owner)[0].payload["outcome"] == "CANCELLED"
    assert owner.call("yahoo", lambda timeout: 1) == 1


def test_native_timeout_is_recalculated_after_queue_admission():
    ticks = [0.0]
    owner = budget(capacity=1, clock=lambda: ticks[0])
    owner.policies["yahoo"] = replace(owner.policies["yahoo"], timeout_seconds=3.0)
    entered = Event()
    release = Event()
    caps = []

    def hold(timeout):
        entered.set()
        assert release.wait(timeout=2.0)
        return "holder"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.call, "yahoo", hold)
        assert entered.wait(timeout=1.0)
        second = pool.submit(owner.call, "yahoo", lambda cap: caps.append(cap) or "second")
        try:
            until(lambda: owner.coordinator.snapshot()["queued_essential"] == 1)
            ticks[0] = 2.0
        finally:
            release.set()
        assert first.result(timeout=3.0) == "holder"
        assert second.result(timeout=3.0) == "second"
    assert caps == [1.0]
    assert owner.coordinator.snapshot()["active"] == 0


@pytest.mark.parametrize("optional", [False, True])
def test_expired_queued_work_is_not_a_native_http_attempt(optional):
    ticks = [0.0]
    owner = budget(capacity=1, clock=lambda: ticks[0])
    owner.policies["borsa"] = replace(owner.policies["borsa"], timeout_seconds=50.0)
    owner.policies["yahoo"] = replace(owner.policies["yahoo"], timeout_seconds=3.0)
    owner.settings = replace(owner.settings, optional_budget_seconds=2.0)
    entered, release = Event(), Event()

    def hold(timeout):
        entered.set()
        assert release.wait(timeout=2.0)
        return "holder"

    def expired():
        if optional:
            with owner.optional_work("synthetic-queue"):
                return owner.call("yahoo", lambda cap: pytest.fail("expired native call"))
        return owner.call("yahoo", lambda cap: pytest.fail("expired native call"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.call, "borsa", hold)
        assert entered.wait(timeout=1.0)
        second = pool.submit(expired)
        try:
            queue = "queued_optional" if optional else "queued_essential"
            until(lambda: owner.coordinator.snapshot()[queue] == 1)
            ticks[0] = 4.0
            expected = OptionalProviderBudgetExceeded if optional else ProviderBudgetExceeded
            with pytest.raises(expected):
                second.result(timeout=1.0)
        finally:
            release.set()
        assert first.result(timeout=3.0) == "holder"
    assert len(attempts(owner)) == 1
    assert owner.coordinator.snapshot()["rejected"] == 1
    assert owner.coordinator.snapshot()["active"] == 0


def test_client_cleanup_keeps_deadline_and_admission_authority_for_output():
    ticks = [0.0]
    owner = budget(capacity=1, clock=lambda: ticks[0])
    closed = []
    owner._yahoo_client = SimpleNamespace(close=lambda: closed.append(True))
    coordinator, deadline = owner.coordinator, owner.deadline
    owner.close()
    assert closed == [True]
    assert owner.coordinator is coordinator
    assert owner.deadline == deadline
    assert not coordinator.snapshot()["closed"]
    ticks[0] = 1.0
    assert owner.call("yahoo", lambda timeout: 1) == 1
    ticks[0] = deadline
    with pytest.raises(ProviderBudgetExceeded):
        owner.call("yahoo", lambda timeout: pytest.fail("reset original deadline"))


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_late_response_is_closed_without_masking_the_original_deadline(cleanup_error):
    ticks = [0.0]
    owner = budget(capacity=1, clock=lambda: ticks[0])
    owner.policies["yahoo"] = replace(owner.policies["yahoo"], timeout_seconds=3.0)
    closed = []

    def close():
        closed.append(True)
        if cleanup_error:
            raise ValueError("synthetic cleanup failure")

    def native(timeout):
        ticks[0] = 4.0
        return SimpleNamespace(status_code=200, close=close)

    with pytest.raises(ProviderBudgetExceeded):
        owner.call("yahoo", native)
    assert closed == [True]
    assert owner.coordinator.snapshot()["active"] == 0
    assert len(attempts(owner)) == 1
    assert attempts(owner)[0].payload["outcome"] == "BUDGET_EXHAUSTED"


def test_configured_worker_cap_is_the_production_coordinator_cap(monkeypatch):
    monkeypatch.setattr("tarzan.config.max_workers", lambda: 7)
    owner = ProviderBudget(ProviderTransportSettings.configured())
    assert owner.coordinator.capacity == 7


@pytest.mark.parametrize("capacity", [0, -1, True, "4", 4.0])
def test_invalid_configured_worker_capacity_is_rejected(monkeypatch, capacity):
    monkeypatch.setattr("tarzan.config.max_workers", lambda: capacity)
    with pytest.raises(ValueError, match="capacity"):
        ProviderBudget(ProviderTransportSettings.configured())


def test_native_telemetry_never_records_callback_arguments_or_payload():
    owner = budget()
    assert owner.call("yahoo", lambda timeout: "PRIVATE-SYNTHETIC-PAYLOAD") == (
        "PRIVATE-SYNTHETIC-PAYLOAD"
    )
    assert "PRIVATE-SYNTHETIC-PAYLOAD" not in repr(owner.coordinator.snapshot())
    assert "PRIVATE-SYNTHETIC-PAYLOAD" not in repr(
        [entry.to_dict() for entry in owner.ledger.entries]
    )


def test_workload_records_only_aggregate_admission_not_request_identity_or_payload():
    owner = budget(capacity=3)
    owner.call("yahoo", lambda timeout: "PRIVATE-SYNTHETIC-PAYLOAD")
    session = SimpleNamespace(ledger=owner.ledger, memo={}, transport=owner)
    observation = build_workload_observation(
        session, None, outcome="SUCCEEDED", duration_ms=1.0,
    )
    native = observation["native_acquisition"]
    assert native["capacity"]["value"] == 3
    assert native["peak"]["value"] == 1
    assert native["acquired"]["value"] == native["released"]["value"] == 1
    assert native["active"]["value"] == native["rejected"]["value"] == 0
    assert observation["counts"]["provider_attempts"]["value"] == 1
    assert observation["native_acquisition_semantics"][
        "queue_wait_is_sum_of_overlapping_waiters"
    ] is True
    assert observation["native_acquisition_semantics"][
        "queue_rejection_is_not_native_http_attempt"
    ] is True
    assert "PRIVATE-SYNTHETIC-PAYLOAD" not in repr(observation)
    assert observation["affects_analysis_id"] is False
    assert observation["affects_financial_decisions"] is False


def test_legacy_workload_has_no_invented_admission_evidence():
    observation = build_workload_observation(
        SimpleNamespace(ledger=RunLedger("synthetic-legacy"), memo={}),
        None, outcome="SUCCEEDED", duration_ms=1.0,
    )
    assert "native_acquisition" not in observation
    assert "native_acquisition_semantics" not in observation
