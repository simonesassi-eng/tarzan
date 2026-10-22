"""A durable observer must not turn SMTP acceptance into a retryable failure."""

import importlib.util
from pathlib import Path

import pytest

from tarzan.runtime import diagnostics, sealed
from tarzan.runtime.session import record_last_run_result
from tarzan.tests.test_no_email_verification import result


@pytest.fixture
def shim(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.logging.setup_logging", lambda: None)
    monkeypatch.setattr("tarzan.log_redaction.install", lambda **kwargs: None)
    path = Path(__file__).resolve().parents[2] / "scripts" / "send_newsletter.py"
    spec = importlib.util.spec_from_file_location("tarzan_send_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("code", [0, 1])
def test_send_primary_outcome_is_preserved_when_diagnostic_key_is_unavailable(
    shim, monkeypatch, tmp_path, code,
):
    monkeypatch.setenv(diagnostics.DIRECTORY_ENV, str(tmp_path / "diagnostics"))
    monkeypatch.setenv(sealed.KEY_ENV, "invalid-synthetic-key")
    completed = result()

    def send():
        record_last_run_result(completed)
        return code

    monkeypatch.setattr(shim, "run_and_send", send)
    assert shim.main() == code
    assert not (tmp_path / "diagnostics" / diagnostics.FILENAME).exists()


def test_uncaught_send_error_keeps_the_fresh_analysis_ledger(shim, monkeypatch):
    completed = result(critical=True)
    captured = []

    def send():
        record_last_run_result(completed)
        raise RuntimeError("synthetic delivery error")

    monkeypatch.setattr(shim, "run_and_send", send)
    monkeypatch.setattr(shim, "capture_configured", lambda owner, **values: captured.append((owner, values)))
    assert shim.main() == 1
    assert captured[0][0] is completed
    assert captured[0][1]["entrypoint_error_type"] == "RuntimeError"
    assert captured[0][1]["exit_code"] == 1


def test_early_send_error_never_captures_a_previous_success(shim, monkeypatch):
    record_last_run_result(result())
    captured = []
    monkeypatch.setattr(shim, "run_and_send",
                        lambda: (_ for _ in ()).throw(ValueError("synthetic bootstrap error")))
    monkeypatch.setattr(shim, "capture_configured", lambda owner, **values: captured.append((owner, values)))
    assert shim.main() == 1
    assert captured[0][0] is None
    assert captured[0][1]["entrypoint_error_type"] == "ValueError"
