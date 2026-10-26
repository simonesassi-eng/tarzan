"""Observer chronology is independent of the immutable financial run clock."""

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tarzan.runtime import acquisition_receipts as receipts
from tarzan.runtime import diagnostics
from tarzan.runtime.ledger import LedgerEntryType
from tarzan.runtime.session import RunMode
from tarzan.runtime.session import record_last_run_result
from tarzan.tests.test_encrypted_diagnostics import KEY
from tarzan.tests.test_final_acquisition_receipts import native_result, report, request
from tarzan.tests.test_send_diagnostic_lifecycle import shim as send_shim


def sequence(monkeypatch, values):
    iterator = iter(values)
    monkeypatch.setattr(receipts, "_observation_instant", lambda: next(iterator))


@pytest.mark.parametrize("mode", list(RunMode))
def test_actual_observer_uses_its_own_receipt_window_not_financial_clock(
    monkeypatch, tmp_path, mode,
):
    result = native_result()
    capture = datetime(2026, 10, 23, 8, tzinfo=timezone.utc)
    effective = None if mode is RunMode.LIVE else capture.date()
    result.session.context = replace(
        result.session.context, mode=mode, captured_at=capture, effective_date=effective,
    )
    result.ledger.append(LedgerEntryType.DELIVERY, {
        "state": "ACKNOWLEDGED_SUCCESS", "smtp_invoked": True,
    })
    unchanged = [entry.to_dict() for entry in result.ledger.entries]
    started = datetime(2026, 10, 26, 9, tzinfo=timezone.utc)
    completed = started + timedelta(microseconds=5)
    sequence(monkeypatch, (started, completed))
    before = result.session.transport.coordinator.snapshot()
    document = report(result)
    final = document["run"]["final_acquisition"]
    assert final["observation_window"] == {
        "started_at": started.isoformat(timespec="microseconds"),
        "completed_at": completed.isoformat(timespec="microseconds"),
    }
    assert document["captured_at"] == capture.isoformat()
    assert document["ledger"] == unchanged
    assert document["run"]["analysis_id"] == "unchanged-analysis"
    assert result.session.transport.coordinator.snapshot() == before
    diagnostics.write_report(tmp_path / "cipher", document, KEY)
    assert diagnostics.read_report(tmp_path / "cipher" / diagnostics.FILENAME, KEY) == document
    result.session.close_transport()


def test_window_brackets_the_actual_native_observation_not_projection(monkeypatch):
    result = native_result()
    coordinator = result.session.transport.coordinator
    native_observe = coordinator.try_observation
    order = []
    instant = datetime(2026, 10, 26, 9, tzinfo=timezone.utc)

    def clock():
        order.append("clock")
        return instant

    def observe():
        order.append("snapshot")
        with request(coordinator):
            pass
        return native_observe()

    monkeypatch.setattr(receipts, "_observation_instant", clock)
    monkeypatch.setattr(coordinator, "try_observation", observe)
    final = report(result)["run"]["final_acquisition"]
    assert order == ["clock", "snapshot", "clock"]
    assert final["native_acquisition"]["acquired"]["value"] == 1
    assert final["native_acquisition"]["released"]["value"] == 1
    assert final["observation_window"]["started_at"] == final["observation_window"]["completed_at"]
    result.session.close_transport()


@pytest.mark.parametrize("case", [
    "backwards", "naive", "non-utc", "not-datetime", "clock-failed",
    "clock-interrupted",
])
def test_observer_clock_failure_is_unavailable_never_primary_send_failure(monkeypatch, case):
    result = native_result()
    start = datetime(2026, 10, 26, 9, tzinfo=timezone.utc)
    if case in {"clock-failed", "clock-interrupted"}:
        failure = RuntimeError if case == "clock-failed" else KeyboardInterrupt
        monkeypatch.setattr(receipts, "_observation_instant", lambda: (
            (_ for _ in ()).throw(failure("private-clock-canary"))
        ))
    else:
        end = {
            "backwards": start - timedelta(microseconds=1),
            "naive": start.replace(tzinfo=None),
            "non-utc": start.astimezone(timezone(timedelta(hours=1))),
            "not-datetime": True,
        }[case]
        sequence(monkeypatch, (start, end))
    document = report(result)
    assert document["exit_code"] == 0 and document["entrypoint_error_type"] is None
    assert "final_acquisition" not in document["run"]
    assert document["run"]["final_acquisition_error_type"] == receipts.FINAL_ACQUISITION_ERROR
    assert "private-clock-canary" not in str(document)
    assert result.session.transport.coordinator.snapshot()["requested"] == 0
    result.session.close_transport()


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_native_send_shim_preserves_primary_exit_and_never_retries_clock_failure(
    monkeypatch, tmp_path, send_shim, code, failure,
):
    result = native_result()
    invocations = []

    def send():
        invocations.append(1)
        record_last_run_result(result)
        return code

    monkeypatch.setattr(send_shim, "run_and_send", send)
    monkeypatch.setattr(receipts, "_observation_instant", lambda: (
        (_ for _ in ()).throw(failure("private-final-clock-canary"))
    ))
    monkeypatch.setenv(diagnostics.DIRECTORY_ENV, str(tmp_path / "cipher"))
    monkeypatch.setattr("tarzan.runtime.sealed.configured_key", lambda **kwargs: KEY)
    assert send_shim.main() == code and invocations == [1]
    restored = diagnostics.read_report(tmp_path / "cipher" / diagnostics.FILENAME, KEY)
    assert restored["exit_code"] == code and restored["entrypoint_error_type"] is None
    assert restored["run"]["final_acquisition_error_type"] == receipts.FINAL_ACQUISITION_ERROR
    assert "private-final-clock-canary" not in str(restored)
    result.session.close_transport()


@pytest.mark.parametrize("owner", ["absent", "foreign"])
def test_missing_or_foreign_native_owner_does_not_borrow_an_observer_clock(monkeypatch, owner):
    result = native_result()
    original = result
    result = (
        replace(result, session=None) if owner == "absent"
        else replace(result, attempt_id="unowned-attempt")
    )
    monkeypatch.setattr(receipts, "_observation_instant", lambda: (
        (_ for _ in ()).throw(AssertionError("must not observe a foreign clock"))
    ))
    document = report(result)
    assert "final_acquisition" not in document["run"]
    assert "final_acquisition_error_type" not in document["run"]
    original.session.close_transport()


def test_old_final_receipt_without_window_remains_accepted_and_not_rejuvenated(tmp_path):
    result = native_result()
    document = report(result)
    old = copy.deepcopy(document)
    old["run"]["final_acquisition"].pop("observation_window")
    diagnostics.write_report(tmp_path / "old", old, KEY)
    assert diagnostics.read_report(tmp_path / "old" / diagnostics.FILENAME, KEY) == old
    assert "observation_window" not in old["run"]["final_acquisition"]
    result.session.close_transport()


@pytest.mark.parametrize("case", [
    "missing", "extra", "null", "list", "boolean", "naive", "non-utc",
    "relative", "malformed", "backwards", "noncanonical",
])
def test_invalid_window_is_rejected_before_cipher_publication(tmp_path, case):
    result = native_result()
    document = report(result)
    window = document["run"]["final_acquisition"]["observation_window"]
    if case == "missing":
        window.pop("started_at")
    elif case == "extra":
        window["private@example.invalid"] = "not-admitted"
    elif case in {"null", "list"}:
        document["run"]["final_acquisition"]["observation_window"] = None if case == "null" else []
    elif case == "boolean":
        window["started_at"] = False
    elif case == "naive":
        window["started_at"] = "2026-10-26T09:00:00.000000"
    elif case == "non-utc":
        window["started_at"] = "2026-10-26T09:00:00.000000+01:00"
    elif case in {"relative", "malformed"}:
        window["started_at"] = "now" if case == "relative" else "x" * 32
    elif case == "backwards":
        window["started_at"] = "2026-10-26T09:00:01.000000+00:00"
        window["completed_at"] = "2026-10-26T09:00:00.000000+00:00"
    else:
        window["started_at"] = "2026-10-26T09:00:00.000000Z"
    with pytest.raises(ValueError):
        diagnostics.write_report(tmp_path / "invalid", document, KEY)
    assert not (tmp_path / "invalid" / diagnostics.FILENAME).exists()
    result.session.close_transport()
