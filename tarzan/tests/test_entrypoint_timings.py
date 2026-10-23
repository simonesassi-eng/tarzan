"""Operational boundaries cannot change finance, delivery or private schema."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from scripts import verify_newsletter as verify
from tarzan.runtime import diagnostics as diag
from tarzan.runtime.session import record_last_run_result
from tarzan.tests.test_encrypted_diagnostics import KEY, result


def _document(**values):
    return diag.build_report(
        result(), operation="verify", exit_code=1, duration_seconds=3.25, **values,
    )


def test_import_timing_is_additive_without_changing_finance_or_v1_envelope(tmp_path):
    old = _document()
    timings = {"imports_seconds": 2.0, "imports_and_entrypoint_seconds": 5.25}
    new = _document(operational_timings=timings)
    assert set(new) == set(old) == diag._FIELDS
    assert new["schema_version"] == old["schema_version"] == "1"
    assert new["duration_seconds"] == old["duration_seconds"] == 3.25
    assert new["run"]["analysis_id"] == old["run"]["analysis_id"]
    for name in ("kpis", "quality", "provider_counts", "summary"):
        assert new[name] == old[name]
    assert new["run"]["timings"] == timings
    timings["imports_seconds"] = 99
    assert new["run"]["timings"]["imports_seconds"] == 2.0
    diag.write_report(tmp_path / "new", new, KEY)
    assert diag.read_report(tmp_path / "new" / diag.FILENAME, KEY) == new
    diag.write_report(tmp_path / "old", old, KEY)
    assert diag.read_report(tmp_path / "old" / diag.FILENAME, KEY) == old


@pytest.mark.parametrize("values", [
    None, [], "timing",
    {}, {"imports_seconds": 1},
    {"imports_seconds": 1, "imports_and_entrypoint_seconds": 5, "private_label": 1},
    {"imports_seconds": True, "imports_and_entrypoint_seconds": 5},
    {"imports_seconds": 1, "imports_and_entrypoint_seconds": False},
    {"imports_seconds": -1, "imports_and_entrypoint_seconds": 5},
    {"imports_seconds": float("nan"), "imports_and_entrypoint_seconds": 5},
    {"imports_seconds": 1, "imports_and_entrypoint_seconds": float("inf")},
    {"imports_seconds": 1 << 4096, "imports_and_entrypoint_seconds": 1 << 4097},
    {"imports_seconds": 5, "imports_and_entrypoint_seconds": 4},
    {"imports_seconds": "1", "imports_and_entrypoint_seconds": 5},
])
def test_invalid_timing_fields_are_rejected_before_encrypted_publication(
    tmp_path, values,
):
    document = _document()
    document["run"]["timings"] = copy.deepcopy(values)
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "encrypted", document, KEY)
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()


def test_import_inclusive_duration_cannot_be_shorter_than_entrypoint(tmp_path):
    document = _document(operational_timings={
        "imports_seconds": 1, "imports_and_entrypoint_seconds": 2,
    })
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "encrypted", document, KEY)
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()


def test_capture_invalid_timing_is_best_effort_and_never_retries_a_send(
    monkeypatch, tmp_path, caplog,
):
    monkeypatch.setenv(diag.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    captured = diag.capture_configured(
        result(), operation="send", exit_code=0, duration_seconds=1,
        operational_timings={"synthetic@example.invalid": "not an allowed label"},
    )
    assert captured is False
    assert "synthetic@example.invalid" not in caplog.text
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()


@pytest.fixture
def send_shim(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.logging.setup_logging", lambda: None)
    monkeypatch.setattr("tarzan.log_redaction.install", lambda **kwargs: None)
    path = Path(__file__).resolve().parents[2] / "scripts" / "send_newsletter.py"
    spec = importlib.util.spec_from_file_location("tarzan_send_timing_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("code", [0, 1])
def test_send_keeps_old_timer_and_counts_import_once_without_idle_delay(
    send_shim, monkeypatch, code,
):
    completed = result()
    captured = []
    clock = iter((100.0, 103.25))
    monkeypatch.setattr(send_shim.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(send_shim, "_IMPORTS_SECONDS", 2.0)

    def send():
        record_last_run_result(completed)
        return code

    monkeypatch.setattr(send_shim, "run_and_send", send)
    monkeypatch.setattr(
        send_shim, "capture_configured",
        lambda owner, **values: captured.append((owner, values)),
    )
    assert send_shim.main() == code
    owner, values = captured[0]
    assert owner is completed
    assert values["exit_code"] == code
    assert values["duration_seconds"] == 3.25
    assert values["operational_timings"] == {
        "imports_seconds": 2.0, "imports_and_entrypoint_seconds": 5.25,
    }


def test_verification_adds_same_boundaries_without_delivery(monkeypatch):
    completed = result()
    captured = []
    clock = iter((100.0, 103.25))
    monkeypatch.delenv(diag.DIRECTORY_ENV, raising=False)
    monkeypatch.setattr(verify.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(verify, "_IMPORTS_SECONDS", 2.0)
    monkeypatch.setattr(verify, "setup_logging", lambda: None)
    monkeypatch.setattr(verify.log_redaction, "install", lambda **kwargs: None)
    monkeypatch.setattr(verify, "_inputs", lambda _: {})
    monkeypatch.setattr(verify, "_fingerprints", lambda _: {})
    monkeypatch.setattr(verify, "_close_delivery_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        verify, "verify", lambda inputs, previous: (completed, None, 0),
    )
    monkeypatch.setattr(
        diag, "capture_configured",
        lambda owner, **values: captured.append((owner, values)) or True,
    )
    assert verify.main([]) == 0
    owner, values = captured[0]
    assert owner is completed
    assert values["operation"] == "verify"
    assert values["duration_seconds"] == 3.25
    assert values["operational_timings"] == {
        "imports_seconds": 2.0, "imports_and_entrypoint_seconds": 5.25,
    }
