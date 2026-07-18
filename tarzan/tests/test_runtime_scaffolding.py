"""Run-session, ledger, and availability contract tests."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timezone

import pytest

from tarzan.runtime.ledger import Availability, ErrorNormalizer, RunLedger, SectionResult
from tarzan.runtime.session import (
    RunAttemptEnvelope,
    RunContext,
    RunMode,
    SerialExecutionGate,
    canonical_analysis_id,
)


def test_serial_gate_is_fifo_and_has_one_active_lease():
    gate = SerialExecutionGate()
    order: list[int] = []
    first_active = threading.Event()
    release_first = threading.Event()

    def worker(index: int) -> None:
        envelope = RunAttemptEnvelope.create(str(index))
        with gate.acquire(envelope, timeout=2):
            assert gate.active_count == 1
            order.append(index)
            if index == 0:
                first_active.set()
                assert release_first.wait(timeout=2)

    first = threading.Thread(target=worker, args=(0,))
    first.start()
    assert first_active.wait(timeout=2)
    later = [threading.Thread(target=worker, args=(index,)) for index in (1, 2)]
    # Establish the actual gate-arrival order before releasing the active lease.
    # Thread start order alone is not an ordering primitive: the scheduler may
    # run the second thread first, which would correctly make it the first FIFO
    # waiter and turn this test into a race rather than a gate contract check.
    for expected_waiters, thread in enumerate(later, start=1):
        thread.start()
        deadline = time.monotonic() + 2
        while gate.waiting_count < expected_waiters and time.monotonic() < deadline:
            time.sleep(0.001)
        assert gate.waiting_count == expected_waiters
    release_first.set()
    first.join(timeout=2)
    for thread in later:
        thread.join(timeout=2)
    assert order == [0, 1, 2]
    assert gate.active_count == 0


def test_context_modes_and_analysis_identity_are_stable():
    with pytest.raises(ValueError, match="effective date"):
        RunContext(
            attempt_id="a",
            mode=RunMode.REPRODUCIBLE,
            effective_date=None,
            captured_at=datetime.now(timezone.utc),
        )
    context = RunContext(
        attempt_id="a",
        mode=RunMode.POINT_IN_TIME,
        effective_date=date(2025, 6, 30),
        captured_at=datetime.now(timezone.utc),
    )
    assert context.analysis_date == date(2025, 6, 30)
    assert not context.allows_live_transport
    base = canonical_analysis_id({"orders": "abc", "attempt_id": "one", "latency": 1})
    noisy = canonical_analysis_id({"orders": "abc", "attempt_id": "two", "latency": 99})
    changed = canonical_analysis_id({"orders": "def", "attempt_id": "one", "latency": 1})
    assert base == noisy
    assert base != changed


def test_ledger_projects_open_and_closed_failure_lifecycles():
    ledger = RunLedger("attempt")
    failure_id = ledger.open_failure(
        stage="provider",
        stable_code="TIMEOUT",
        severity="ERROR",
        error=TimeoutError("request token=should-not-leak"),
        affected_outputs=["valuation"],
        analytical_impact="valuation degraded",
        publication_impact="DEGRADE",
    )
    opened = ledger.failure_records()[0]
    assert opened.closed is False
    assert opened.availability is Availability.UNAVAILABLE
    assert "should-not-leak" not in repr(opened)

    ledger.remedy(
        failure_id,
        remedy_id="cache",
        action="use validated cache",
        outcome="SUCCEEDED",
        availability=Availability.DEGRADED,
        provenance=["cache:key"],
    )
    ledger.close_failure(
        failure_id,
        automatically_corrected=True,
        selected_resolution="cache",
        availability=Availability.DEGRADED,
        provenance=["cache:key"],
    )
    closed = ledger.failure_records()[0]
    assert closed.closed is True
    assert closed.automatically_corrected is True
    assert closed.availability is Availability.DEGRADED
    assert [item["ordinal"] for item in closed.remedies] == [1]
    assert [entry.sequence for entry in ledger.entries] == [1, 2, 3]


def test_unavailable_is_not_numeric_zero_and_redaction_is_recursive():
    assert SectionResult(Availability.AVAILABLE, 0).to_dict()["value"] == 0
    assert SectionResult(Availability.UNAVAILABLE, None).to_dict()["value"] is None
    with pytest.raises(ValueError):
        SectionResult(Availability.UNAVAILABLE, 0)
    normalized = ErrorNormalizer.normalize({
        "authorization": "Bearer secret",
        "nested": {"url": "https://example.invalid/?token=secret"},
    })
    assert "secret" not in repr(normalized)
