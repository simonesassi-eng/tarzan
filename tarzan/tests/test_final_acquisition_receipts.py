"""Late native work is independently observed without rewriting analysis facts."""

from dataclasses import replace
from datetime import datetime, timezone
import copy
from types import SimpleNamespace

import pytest

from tarzan.runtime import diagnostics as diag
from tarzan.runtime.acquisition import RequestCoordinator
from tarzan.runtime.acquisition_receipts import FINAL_ACQUISITION_ERROR
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunResult, RunSession
from tarzan.runtime.session import record_last_run_result
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings
from tarzan.runtime.workload import MAX_RECORDED_INTEGER
from tarzan.tests.test_encrypted_diagnostics import KEY
from tarzan.tests.test_send_diagnostic_lifecycle import shim as send_shim


def native_result():
    context = RunContext("final-native-observation", RunMode.LIVE, None,
                         datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
    session = RunSession(context, {}, ledger, transport=budget)
    return RunResult(None, None, context.attempt_id, "unchanged-analysis", ledger, session)


def report(result):
    return diag.build_report(result, operation="send", exit_code=0, duration_seconds=1)


def request(coordinator, *, optional=False):
    return coordinator.lease(optional=optional, remaining=lambda: 100)


def test_late_request_is_observed_without_rewriting_prior_analysis_telemetry(tmp_path):
    result = native_result()
    coordinator = result.session.transport.coordinator
    with request(coordinator):
        pass
    before = coordinator.snapshot()
    result.ledger.append(LedgerEntryType.TELEMETRY, {
        "scenario": "analysis_run", "native_acquired": before["acquired"],
    })
    with request(coordinator, optional=True):
        pass
    result.session.close_transport()
    document = report(result)
    final = document["run"]["final_acquisition"]
    assert document["ledger"][0]["payload"]["native_acquired"] == 1
    assert final["native_acquisition"]["requested"]["value"] == 2
    assert final["native_acquisition"]["acquired"]["value"] == 2
    assert final["native_acquisition"]["released"]["value"] == 2
    # Native client cleanup deliberately preserves the admission authority for
    # later rendering work. Observe it honestly; do not close it for telemetry.
    assert final["admission_closed"] is False
    assert final["scope"] == "native_owner_at_diagnostic_capture"
    assert final["descriptive_only"] and not final["affects_financial_decisions"]
    assert result.analysis_id == "unchanged-analysis"
    diag.write_report(tmp_path / "cipher", document, KEY)
    assert diag.read_report(tmp_path / "cipher" / diag.FILENAME, KEY) == document


def test_nonblocking_observer_cannot_queue_behind_coordinator_or_invent_zeros():
    result = native_result()
    coordinator = result.session.transport.coordinator
    with coordinator._condition:
        assert coordinator.try_observation() is None
        document = report(result)
    assert "final_acquisition" not in document["run"]
    assert document["run"]["final_acquisition_error_type"] == FINAL_ACQUISITION_ERROR
    assert coordinator.snapshot()["requested"] == 0
    result.session.close_transport()


def test_active_native_request_is_reported_active_not_falsely_drained():
    result = native_result()
    coordinator = result.session.transport.coordinator
    with request(coordinator):
        observed = report(result)["run"]["final_acquisition"]
        assert observed["native_acquisition"]["active"]["value"] == 1
        assert observed["native_acquisition"]["released"]["value"] == 0
    assert report(result)["run"]["final_acquisition"]["native_acquisition"]["active"]["value"] == 0
    result.session.close_transport()


def test_only_explicit_native_admission_close_is_reported_closed():
    result = native_result()
    coordinator = result.session.transport.coordinator
    result.session.close_transport()
    assert report(result)["run"]["final_acquisition"]["admission_closed"] is False
    with request(coordinator):
        pass
    coordinator.close()
    assert report(result)["run"]["final_acquisition"]["admission_closed"] is True


@pytest.mark.parametrize("seconds", [
    MAX_RECORDED_INTEGER / 1000.0,
    MAX_RECORDED_INTEGER / 1000.0 + 1,
    float("1e100"),
])
def test_queue_wait_recording_bound_never_rounds_past_json_integer_limit(seconds):
    result = native_result()
    coordinator = result.session.transport.coordinator
    coordinator._queue_wait_seconds = seconds
    document = report(result)
    measure = document["run"]["final_acquisition"]["native_acquisition"]["queue_wait_ms"]
    # Seconds are floating-point native observations; multiplication at the
    # exact boundary can be one ulp below the integer bound, never above it.
    assert MAX_RECORDED_INTEGER - 2 <= measure["value"] <= MAX_RECORDED_INTEGER
    if seconds > MAX_RECORDED_INTEGER / 1000.0:
        assert measure["value"] == MAX_RECORDED_INTEGER
    assert measure["truncated"] is (seconds > MAX_RECORDED_INTEGER / 1000.0)
    diag._validate(document)
    result.session.close_transport()


def test_atomic_groups_equal_original_snapshot_abi_and_observation_does_not_mutate():
    coordinator = RequestCoordinator(4)
    protection = coordinator.protect(1)
    before, protected_before = coordinator.snapshot(), coordinator.protection_snapshot()
    observation = coordinator.try_observation()
    assert observation == {"native_acquisition": before, "protected_acquisition": protected_before}
    observation["native_acquisition"]["requested"] = 999
    assert coordinator.snapshot() == before
    assert coordinator.protection_snapshot() == protected_before
    protection.finish()
    protection.close()


@pytest.mark.parametrize("case", ["no-owner", "unknown-budget", "unknown-coordinator", "foreign"])
def test_uninspected_or_foreign_owners_do_not_become_native_final_evidence(case):
    result = native_result()
    budget = result.session.transport
    if case == "no-owner":
        result = replace(result, session=None)
    elif case == "unknown-budget":
        result.session.transport = SimpleNamespace(coordinator=budget.coordinator)
    elif case == "unknown-coordinator":
        budget.coordinator = SimpleNamespace(try_observation=lambda: {
            "native_acquisition": {"requested": 0},
        })
    else:
        result = replace(result, attempt_id="foreign-attempt")
    try:
        assert "final_acquisition" not in report(result)["run"]
    finally:
        if case == "unknown-coordinator":
            budget.coordinator = RequestCoordinator(4)
        budget.close()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_optional_projection_fault_never_replaces_report_or_primary_outcome(monkeypatch, failure):
    result = native_result()
    monkeypatch.setattr(result.session.transport.coordinator, "try_observation", lambda: (
        (_ for _ in ()).throw(failure("synthetic observer failure"))
    ))
    document = report(result)
    assert document["exit_code"] == 0 and document["entrypoint_error_type"] is None
    assert document["run"]["final_acquisition_error_type"] == FINAL_ACQUISITION_ERROR
    assert "synthetic observer failure" not in str(document)
    result.session.close_transport()


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_actual_shim_keeps_primary_exit_and_one_invocation_when_final_observer_fails(
    monkeypatch, tmp_path, send_shim, code, failure,
):
    result = native_result()
    invocations = []

    def send():
        invocations.append(1)
        record_last_run_result(result)
        return code

    monkeypatch.setattr(send_shim, "run_and_send", send)
    monkeypatch.setattr(result.session.transport.coordinator, "try_observation", lambda: (
        (_ for _ in ()).throw(failure("synthetic-private-observer-canary"))
    ))
    monkeypatch.setenv(diag.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    monkeypatch.setattr("tarzan.runtime.sealed.configured_key", lambda **kwargs: KEY)
    assert send_shim.main() == code and invocations == [1]
    restored = diag.read_report(tmp_path / "encrypted" / diag.FILENAME, KEY)
    assert restored["exit_code"] == code
    assert restored["entrypoint_error_type"] is None
    assert restored["run"]["final_acquisition_error_type"] == FINAL_ACQUISITION_ERROR
    assert "synthetic-private-observer-canary" not in str(restored)
    result.session.close_transport()


@pytest.mark.parametrize("case", [
    "scope", "private-label", "missing-group", "nonfinite", "negative",
    "boolean-count", "unknown-measure", "wrong-unit", "wrong-bound",
    "bad-closed", "financial-authority", "inconsistent-truncation",
])
def test_invalid_final_evidence_is_rejected_before_cipher_publication(tmp_path, case):
    result = native_result()
    document = report(result)
    result.session.close_transport()
    final = document["run"]["final_acquisition"]
    if case == "scope":
        final["scope"] = "arbitrary-source"
    elif case == "private-label":
        final["synthetic@example.invalid"] = 1
    elif case == "missing-group":
        del final["protected_acquisition"]
    elif case == "nonfinite":
        final["native_acquisition"]["queue_wait_ms"]["value"] = float("nan")
    elif case == "negative":
        final["native_acquisition"]["requested"]["value"] = -1
    elif case == "boolean-count":
        final["native_acquisition"]["requested"]["value"] = False
    elif case == "unknown-measure":
        final["native_acquisition"]["requested"]["private"] = "raw-content"
    elif case == "wrong-unit":
        final["native_acquisition"]["requested"]["unit"] = "portfolio-eur"
    elif case == "wrong-bound":
        final["native_acquisition"]["requested"]["maximum_recorded"] += 1
    elif case == "bad-closed":
        final["admission_closed"] = 1
    elif case == "financial-authority":
        final["affects_financial_decisions"] = True
    else:
        final["truncated"] = True
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "invalid", document, KEY)
    assert not (tmp_path / "invalid" / diag.FILENAME).exists()


def test_recording_bound_retains_truncation_not_false_exact_count():
    result = native_result()
    coordinator = result.session.transport.coordinator
    coordinator._requested = MAX_RECORDED_INTEGER + 1
    document = report(result)
    measure = document["run"]["final_acquisition"]["native_acquisition"]["requested"]
    assert measure["value"] == MAX_RECORDED_INTEGER and measure["truncated"] is True
    assert document["run"]["final_acquisition"]["truncated"] is True
    coordinator._requested = 0
    result.session.close_transport()


def test_old_v1_report_and_final_projection_error_remain_compatible(tmp_path):
    result = native_result()
    document = report(result)
    result.session.close_transport()
    old = copy.deepcopy(document)
    old["run"].pop("final_acquisition")
    diag.write_report(tmp_path / "old", old, KEY)
    assert diag.read_report(tmp_path / "old" / diag.FILENAME, KEY) == old
    unavailable = copy.deepcopy(old)
    unavailable["run"]["final_acquisition_error_type"] = FINAL_ACQUISITION_ERROR
    diag.write_report(tmp_path / "unavailable", unavailable, KEY)
    assert diag.read_report(tmp_path / "unavailable" / diag.FILENAME, KEY) == unavailable
    unavailable["run"]["final_acquisition"] = document["run"]["final_acquisition"]
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "ambiguous", unavailable, KEY)


@pytest.mark.parametrize("marker", [None, True, [], {}, "RuntimeError", "private-canary"])
def test_final_projection_error_is_closed_not_arbitrary_metadata(tmp_path, marker):
    result = native_result()
    document = report(result)
    result.session.close_transport()
    document["run"].pop("final_acquisition")
    document["run"]["final_acquisition_error_type"] = marker
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "invalid", document, KEY)
    assert not (tmp_path / "invalid" / diag.FILENAME).exists()
