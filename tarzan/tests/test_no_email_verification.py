"""The real verification entry point has no delivery authority, even on failure."""

import base64
import json
import os
import smtplib

import pandas as pd
import pytest

from scripts import verify_newsletter as verify
from tarzan import delivery
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime import diagnostics as diag, sealed
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import record_last_run_result, RunResult


KEY = b"synthetic-diagnostics-key-000000"


def result(*, critical=False, warning=False):
    ledger = RunLedger("synthetic-verification")
    ledger.append(LedgerEntryType.STAGE, {"stage": "run", "outcome": "SUCCEEDED"})
    if critical or warning:
        ledger.open_failure(
            stage="benchmark_preprocessing" if critical else "optional_reference",
            stable_code="ESSENTIAL_BENCHMARK_UNAVAILABLE" if critical else "OPTIONAL_REFERENCE_UNAVAILABLE",
            severity="CRITICAL" if critical else "WARNING", error={"missing": ["Critical"]},
            affected_outputs=["benchmark"], analytical_impact="source unavailable",
            publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE" if critical else "DEGRADE",
        )
    metrics = PortfolioMetrics(
        total_value=1000, holdings_df=pd.DataFrame([{"cost_basis_eur": 800}]),
    )
    return RunResult(metrics, InvestorConfig(), "synthetic-verification", "synthetic-analysis", ledger)


@pytest.fixture
def exercise(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("order_list.csv", "targets.csv", "targets_per_holding.csv"):
        (inputs / name).write_text("synthetic,private,input\n")
    for name in ("SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL", "GEMINI_API_KEY",
                 "DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(diag.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    monkeypatch.setenv(sealed.KEY_ENV, base64.b64encode(KEY).decode())
    monkeypatch.setattr(verify, "setup_logging", lambda: None)
    monkeypatch.setattr(verify.log_redaction, "install", lambda **k: None)

    def forbidden(*args, **kwargs):
        pytest.fail("verification invoked SMTP, a delivery claim, or run_and_send")

    monkeypatch.setattr(smtplib, "SMTP", forbidden)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbidden)
    monkeypatch.setattr(delivery, "send_email", forbidden)
    monkeypatch.setattr(delivery, "_delivery_claim_store", forbidden)
    monkeypatch.setattr(delivery, "run_and_send", forbidden)
    monkeypatch.setattr(verify, "render_newsletter",
                        lambda **kwargs: "<html>synthetic portfolio verification</html>")
    monkeypatch.setattr(verify, "validate_newsletter_semantics", lambda *args: ())
    return inputs, tmp_path / "local", tmp_path / "encrypted" / diag.FILENAME


def publish_fake(monkeypatch, completed):
    def run(**kwargs):
        assert os.environ["TARZAN_DISABLE_AI"] == "1"
        record_last_run_result(completed)
        return completed.compatibility_tuple()
    monkeypatch.setattr(verify.orchestrator, "run", run)


@pytest.mark.parametrize("warning", [False, True])
def test_real_main_succeeds_without_mail_credentials_and_writes_only_ciphertext_remotely(
    exercise, monkeypatch, warning,
):
    inputs, local, encrypted = exercise
    completed = result(warning=warning)
    publish_fake(monkeypatch, completed)
    assert verify.main(["--input-dir", str(inputs)]) == 0
    assert not local.exists()
    document = diag.read_report(encrypted, KEY)
    assert document["operation"] == "verify" and document["exit_code"] == 0
    assert set(document["input_sha256"]) == {"order_list.csv", "targets.csv", "targets_per_holding.csv"}
    assert document["entrypoint_error_type"] is None
    assert completed.ledger.entries[-1].payload["smtp_invoked"] is False
    assert completed.ledger.entries[-1].payload["claim_invoked"] is False
    assert completed.ledger.entries[-1].payload["state"] == "VERIFICATION_ONLY"
    assert document["summary"]["publication_state"] == (
        "SEND_DEGRADED_NORMAL" if warning else "SEND_NORMAL"
    )


def test_blocked_source_is_not_rendered_as_a_normal_issue(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    completed = result(critical=True)
    publish_fake(monkeypatch, completed)
    monkeypatch.setattr(verify, "render_newsletter",
                        lambda **kwargs: pytest.fail("a blocked normal issue was rendered"))
    assert verify.main(["--input-dir", str(inputs)]) == 1
    document = diag.read_report(encrypted, KEY)
    assert document["summary"]["publication_state"] == "BLOCK_NORMAL_AND_NOTIFY_FAILURE"
    assert document["exit_code"] == 1


def test_failed_semantic_gate_retains_exact_violation_without_delivery(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    completed = result()
    publish_fake(monkeypatch, completed)
    monkeypatch.setattr(verify, "validate_newsletter_semantics",
                        lambda *args: ("PRIVATE.MI critical chart missing source",))
    assert verify.main(["--input-dir", str(inputs)]) == 1
    document = diag.read_report(encrypted, KEY)
    assert document["summary"]["publication_state"] == "BLOCK_ALL_DELIVERY"
    failures = completed.ledger.failure_records()
    assert failures[0].original_failure["violations"] == ("PRIVATE.MI critical chart missing source",)
    assert b"PRIVATE.MI" not in encrypted.read_bytes()


def test_a_failure_before_a_new_result_never_exports_the_previous_portfolio(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    previous = result()
    record_last_run_result(previous)
    monkeypatch.setattr(verify.orchestrator, "run",
                        lambda **kwargs: (_ for _ in ()).throw(ValueError("private bootstrap detail")))
    assert verify.main(["--input-dir", str(inputs)]) == 1
    document = diag.read_report(encrypted, KEY)
    assert document["run"]["attempt_id"] is None
    assert document["kpis"] == {}
    assert document["ledger"] == []
    assert document["entrypoint_error_type"] == "ValueError"
    assert "private bootstrap detail" not in json.dumps(document)


def test_a_fresh_failed_result_keeps_its_ledger_even_when_analysis_raises(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    failed = result(critical=True)
    def run(**kwargs):
        record_last_run_result(failed)
        raise ValueError("private analysis detail")
    monkeypatch.setattr(verify.orchestrator, "run", run)
    assert verify.main(["--input-dir", str(inputs)]) == 1
    document = diag.read_report(encrypted, KEY)
    assert document["run"]["attempt_id"] == failed.attempt_id
    assert document["ledger"][1]["payload"]["stable_code"] == "ESSENTIAL_BENCHMARK_UNAVAILABLE"


def test_an_unchanged_result_is_not_a_successful_verification(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    previous = result()
    record_last_run_result(previous)
    monkeypatch.setattr(verify.orchestrator, "run", lambda **kwargs: previous.compatibility_tuple())
    assert verify.main(["--input-dir", str(inputs)]) == 1
    document = diag.read_report(encrypted, KEY)
    assert document["entrypoint_error_type"] == "RuntimeError"
    assert document["run"]["attempt_id"] is None


def test_local_html_and_json_are_private_and_never_overwrite_a_comparison(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    publish_fake(monkeypatch, result())
    args = ["--input-dir", str(inputs), "--private-output", str(local)]
    assert verify.main(args) == 0
    original = {path.name: path.read_bytes() for path in local.iterdir()}
    assert set(original) == {"report.json", "newsletter.html"}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in local.iterdir())
    assert local.stat().st_mode & 0o777 == 0o700
    # Give the second verification a distinct fresh owner.
    publish_fake(monkeypatch, result())
    assert verify.main(args) == 1
    assert {path.name: path.read_bytes() for path in local.iterdir()} == original


def test_requested_remote_evidence_cannot_silently_disappear_on_a_green_verification(
    exercise, monkeypatch,
):
    inputs, local, encrypted = exercise
    publish_fake(monkeypatch, result())
    monkeypatch.setenv(sealed.KEY_ENV, "invalid-synthetic-key")
    assert verify.main(["--input-dir", str(inputs)]) == 1
    assert not encrypted.exists()


def test_input_cleanup_is_owned_and_does_not_authorize_send(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    paths = verify._inputs(inputs)
    closed = []
    paths.close = lambda: closed.append(True)
    monkeypatch.setattr(verify, "_inputs", lambda directory: paths)
    publish_fake(monkeypatch, result())
    assert verify.main([]) == 0
    assert closed == [True]


def test_ai_environment_is_restored_after_verification(exercise, monkeypatch):
    inputs, local, encrypted = exercise
    monkeypatch.setenv("TARZAN_DISABLE_AI", "caller-setting")
    publish_fake(monkeypatch, result())
    assert verify.main(["--input-dir", str(inputs)]) == 0
    assert os.environ["TARZAN_DISABLE_AI"] == "caller-setting"
