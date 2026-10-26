"""Measure pacing without changing intervals, transport, ownership or errors."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import pytest

from tarzan import runtime
from tarzan.data import _yf_net as net
from tarzan.runtime import transport, workload
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


@contextmanager
def owned(clock=None, provider_settings=None):
    context = RunContext(
        "synthetic-pacing-observation", RunMode.LIVE, None,
        datetime(2026, 10, 26, 6, tzinfo=timezone.utc),
    )
    options = {} if clock is None else {
        "clock": lambda: clock[0],
        "sleep": lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    }
    budget = transport.ProviderBudget(
        provider_settings or transport.ProviderTransportSettings.configured(), **options,
    )
    session = RunSession(context, {}, RunLedger(context.attempt_id), transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            session.close_transport()


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(net, "_time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(net, "_last_call", [-1.0])
    return now


def test_same_native_spacing_at_zero_and_all_requests_avoid_http_permits(clock):
    with owned(clock) as session:
        assert net._YF_MIN_INTERVAL == 0.2
        instants = []
        for _ in range(3):
            net.space_yf_call()
            instants.append(clock[0])
        assert instants == pytest.approx([0.0, 0.2, 0.4])
        observed = session.transport.yahoo_spacing.snapshot()
        assert observed["calls"] == observed["completed"] == 3
        assert observed["interrupted"] == 0
        assert observed["lock_wait_seconds"] == 0.0
        assert observed["sleep_wait_seconds"] == pytest.approx(0.4)
        assert observed["requested_sleep_seconds"] == pytest.approx(0.4)
        assert not observed["incomplete"] and not observed["truncated"]
        native = session.transport.coordinator.snapshot()
        assert native["requested"] == native["acquired"] == native["active"] == 0


def reference_pacer():
    """Exact previous pacing body, independent of the optional observation."""
    with net._lock:
        wait = net._YF_MIN_INTERVAL - (net._time.monotonic() - net._last_call[0])
        if wait > 0:
            transport.provider_wait(wait, "yahoo")
        net._last_call[0] = net._time.monotonic()


@given(delays=st.lists(st.integers(min_value=0, max_value=700), max_size=40))
@settings(
    max_examples=30, deadline=None,
    # Only fixed clock/last-call ports are installed by the fixture. Both
    # controls reset their mutable clock and create a fresh native owner.
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
def test_generated_call_schedules_match_the_previous_pacing_body(clock, delays):
    def run(pacer):
        clock[0], net._last_call[0] = 0.0, -1.0
        with owned(clock):
            observed = []
            for delay in delays:
                clock[0] += delay / 1000.0
                pacer()
                observed.append((clock[0], net._last_call[0]))
            return observed

    assert run(net.space_yf_call) == run(reference_pacer)


def test_real_native_deadline_matches_previous_pacing_and_never_acquires_http(clock):
    provider_settings = transport.ProviderTransportSettings(
        run_budget_seconds=0.35,
        providers=transport.ProviderTransportSettings.configured().providers,
        optional_budget_seconds=0.1,
        publication_reserve_seconds=0.1,
    )

    def run(pacer):
        clock[0], net._last_call[0] = 0.0, -1.0
        with owned(clock, provider_settings) as session:
            pacer()
            pacer()
            with pytest.raises(transport.ProviderBudgetExceeded) as caught:
                pacer()
            admission = session.transport.coordinator.snapshot()
            assert admission["acquired"] == admission["active"] == 0
            return type(caught.value), caught.value.args, clock[0], net._last_call[0]

    assert run(net.space_yf_call) == run(reference_pacer)


def test_record_port_runs_only_after_shared_pacing_lock_is_released(monkeypatch, clock):
    calls = []

    def record(**values):
        assert net._lock.acquire(blocking=False)
        net._lock.release()
        calls.append(values)

    monkeypatch.setattr(workload, "record_yahoo_spacing", record)
    with owned(clock):
        net.space_yf_call()
    assert len(calls) == 1
    assert calls[0]["completed"] is True


def test_optional_start_clock_error_cannot_change_original_pacer(monkeypatch, clock):
    calls = [0]

    def monotonic():
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("optional clock failure")
        return clock[0]

    monkeypatch.setattr(net, "_time", SimpleNamespace(monotonic=monotonic))
    with owned(clock) as session:
        assert net.space_yf_call() is None
        assert net._last_call[0] == 0.0
        assert session.transport.yahoo_spacing.snapshot()["incomplete"]


def test_optional_end_clock_error_cannot_replace_native_interruption(monkeypatch, clock):
    net._last_call[0] = 0.0
    primary = KeyboardInterrupt("native primary")
    calls = [0]

    def monotonic():
        calls[0] += 1
        if calls[0] > 2:
            raise RuntimeError("optional clock failure")
        return clock[0]

    def wait(seconds, provider):
        raise primary

    monkeypatch.setattr(net, "_time", SimpleNamespace(monotonic=monotonic))
    monkeypatch.setattr(transport, "provider_wait", wait)
    with owned(clock) as session:
        with pytest.raises(KeyboardInterrupt) as caught:
            net.space_yf_call()
        assert caught.value is primary
        assert session.transport.yahoo_spacing.snapshot()["incomplete"]


@pytest.mark.parametrize("error", [
    RuntimeError("private sentinel"),
    KeyboardInterrupt("private sentinel"),
    SystemExit("private sentinel"),
])
def test_observer_errors_never_change_success_and_mark_incomplete(monkeypatch, clock, error):
    def record(**values):
        raise error

    monkeypatch.setattr(workload, "record_yahoo_spacing", record)
    with owned(clock) as session:
        assert net.space_yf_call() is None
        assert net._last_call[0] == 0.0
        assert session.transport.yahoo_spacing.snapshot()["incomplete"] is True


@pytest.mark.parametrize("error", [
    transport.ProviderBudgetExceeded("native primary"),
    TimeoutError("native primary"),
    KeyboardInterrupt("native primary"),
])
def test_native_error_identity_and_old_timestamp_survive_observer(monkeypatch, clock, error):
    net._last_call[0] = 0.0

    def wait(seconds, provider):
        clock[0] += 0.05
        raise error

    monkeypatch.setattr(transport, "provider_wait", wait)
    with owned(clock) as session:
        with pytest.raises(type(error)) as caught:
            net.space_yf_call()
        assert caught.value is error
        assert net._last_call[0] == 0.0
        observed = session.transport.yahoo_spacing.snapshot()
        assert observed["calls"] == observed["interrupted"] == 1
        assert observed["completed"] == 0
        assert observed["sleep_wait_seconds"] == pytest.approx(0.05)


def test_held_telemetry_lock_does_not_block_pacing_or_snapshot(clock):
    with owned(clock) as session:
        observation = session.transport.yahoo_spacing
        observation._lock.acquire()
        try:
            net.space_yf_call()
            assert observation.snapshot() == {
                "available": False, "incomplete": True, "truncated": False,
            }
            payload = workload.build_workload_observation(
                session, None, outcome="SUCCEEDED", duration_ms=0.0,
            )
            assert payload["yahoo_spacing"]["available"] is False
            assert payload["yahoo_spacing"]["incomplete"] is True
            # Lost optional sampling is distinct from a recording bound.
            # It must not pretend that native admission or stage counts cut.
            assert payload["truncated"] is False
        finally:
            observation._lock.release()
        assert observation.snapshot()["incomplete"] is True
        assert observation.snapshot()["calls"] == 0
        assert net._last_call[0] == 0.0


@pytest.mark.parametrize("invalid", [
    float("nan"), float("inf"), -1.0, True, "private text", 1j, None, 10**1000,
])
def test_invalid_measurements_are_not_serialized_as_evidence(invalid):
    observation = workload.YahooSpacingObservation()
    observation.record(
        lock_wait=invalid, sleep_wait=0.0, requested_sleep=0.0, completed=True,
    )
    snapshot = observation.snapshot()
    assert snapshot["calls"] == 0 and snapshot["incomplete"]
    assert "private text" not in json.dumps(snapshot)


def test_recording_bounds_cannot_overflow_or_hide_truncation():
    observation = workload.YahooSpacingObservation()
    observation._counts["calls"] = workload.MAX_RECORDED_INTEGER
    for _ in range(2):
        observation.record(
            lock_wait=1e308, sleep_wait=1e308, requested_sleep=1e308, completed=True,
        )
    snapshot = observation.snapshot()
    assert snapshot["truncated"] and not snapshot["incomplete"]
    assert snapshot["calls"] == workload.MAX_RECORDED_INTEGER
    assert snapshot["lock_wait_seconds"] <= workload.MAX_RECORDED_INTEGER / 1000.0
    json.dumps(snapshot, allow_nan=False)


def test_observation_never_creates_a_run_budget():
    context = RunContext(
        "synthetic-no-pacing-owner", RunMode.LIVE, None,
        datetime(2026, 10, 26, 6, tzinfo=timezone.utc),
    )
    session = RunSession(context, {}, RunLedger(context.attempt_id))
    with runtime.bind_context(context), activate_session(session):
        workload.record_yahoo_spacing(
            lock_wait=0.0, sleep_wait=0.0, requested_sleep=0.0, completed=True,
        )
        assert session.transport is None


def test_no_owner_has_no_new_transport_authority(clock):
    assert net.space_yf_call() is None
    assert runtime.context().attempt_id == "test"


def test_multiple_workers_are_owned_and_do_not_change_native_capacity(monkeypatch):
    # Accelerate only this synthetic test; production remains exactly 0.2 s.
    monkeypatch.setattr(net, "_YF_MIN_INTERVAL", 0.001)
    monkeypatch.setattr(net, "_last_call", [0.0])
    with owned() as session, ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(copy_context().run, net.space_yf_call) for _ in range(32)]
        assert all(future.result(timeout=5) is None for future in futures)
        observed = session.transport.yahoo_spacing.snapshot()
        assert observed["available"]
        assert observed["interrupted"] == 0
        assert observed["calls"] == observed["completed"] <= 32
        assert observed["calls"] == 32 or observed["incomplete"]
        native = session.transport.coordinator.snapshot()
        assert native["capacity"] == 4 and native["requested"] == native["active"] == 0


def test_report_names_units_bounds_and_semantics_are_nonfinancial(clock):
    with owned(clock) as session:
        net.space_yf_call()
        payload = workload.build_workload_observation(
            session, None, outcome="SUCCEEDED", duration_ms=0.0,
        )
        spacing = payload["yahoo_spacing"]
        assert spacing["available"] and not spacing["incomplete"]
        assert spacing["measurements"]["calls"]["value"] == 1
        assert spacing["measurements"]["lock_wait_ms"]["unit"] == "milliseconds"
        assert spacing["semantics"] == "sum_of_caller_intervals_not_stage_critical_path"
        assert spacing["overlapping_lock_waiters_must_not_be_summed_into_delivery_latency"]
        assert spacing["configured_spacing_policy_unchanged"]
        assert spacing["observer_never_blocks_provider_callers"]
        assert payload["affects_analysis_id"] is False
        assert payload["affects_financial_decisions"] is False
        assert payload["truncated"] is False
        assert not session.ledger.entries
