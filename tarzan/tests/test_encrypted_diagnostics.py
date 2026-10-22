"""Exact failure recovery without public finance data, SDK cookies or secrets."""

import base64
from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime import diagnostics as diag, sealed
from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunResult


KEY = b"synthetic-diagnostics-key-000000"
assert len(KEY) == 32


def result():
    ledger = RunLedger("synthetic-diagnostics")
    ledger.open_failure(
        stage="newsletter_semantics", stable_code="NEWSLETTER_SEMANTIC_INVARIANT_FAILED",
        severity="CRITICAL", error={"violations": ["Critical / PRIVATE.MI chart source is missing"]},
        affected_outputs=["newsletter"], analytical_impact="chart has no source",
        publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
    )
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "source": "yahoo", "operation": "native_http", "outcome": "HTTP_404",
    })
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "source": "yahoo", "operation": "history", "outcome": "FAILED",
    })
    return RunResult(
        PortfolioMetrics(total_value=1000, holdings_df=pd.DataFrame([{"cost_basis_eur": 800}])),
        InvestorConfig(), "synthetic-diagnostics", "synthetic-analysis", ledger,
    )


def report():
    return diag.build_report(
        result(), operation="verify", exit_code=1, duration_seconds=2.5,
        input_sha256={"order_list.csv": "a" * 64},
    )


def test_exact_semantic_cause_and_native_outcome_survive_only_in_ciphertext(tmp_path):
    document = report()
    staging = tmp_path / "encrypted"
    diag.write_report(staging, document, KEY)
    path = staging / diag.FILENAME
    assert list(staging.iterdir()) == [path]
    assert path.stat().st_mode & 0o777 == 0o600
    assert staging.stat().st_mode & 0o777 == 0o700
    for private in (b"PRIVATE.MI", b"chart source", b"synthetic-analysis", b"1000"):
        assert private not in path.read_bytes()
    restored = diag.read_report(path, KEY)
    assert restored == document
    assert restored["ledger"][0]["payload"]["original_failure"]["violations"] == [
        "Critical / PRIVATE.MI chart source is missing",
    ]
    assert restored["summary"]["publication_state"] == "BLOCK_ALL_DELIVERY"
    assert restored["provider_counts"] == {
        "native": {"yahoo": {"HTTP_404": 1}},
        "adapters": {"yahoo": {"FAILED": 1}},
    }


@pytest.mark.parametrize("case", ["wrong-key", "tampered", "truncated", "market-purpose"])
def test_unverified_diagnostics_never_create_a_plaintext_report(tmp_path, case, capsys):
    staging = tmp_path / "encrypted"
    diag.write_report(staging, report(), KEY)
    path = staging / diag.FILENAME
    key = KEY
    if case == "wrong-key":
        key = b"x" * 32
    elif case == "tampered":
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    elif case == "truncated":
        path.write_bytes(path.read_bytes()[:15])
    else:
        sealed.write_snapshot(path, canonical_json_bytes(report()), KEY, purpose="tarzan-market-cache-v1")
    recovery = tmp_path / "recovery.b64"
    recovery.write_bytes(base64.b64encode(key))
    output = tmp_path / "private" / "report.json"
    assert diag.main([
        "decrypt", "--snapshot", str(path), "--key-file", str(recovery),
        "--output", str(output),
    ]) == 1
    assert not output.exists()
    assert "PRIVATE" not in capsys.readouterr().out


def test_decryption_creates_private_json_and_preserves_an_existing_output(tmp_path):
    staging = tmp_path / "encrypted"
    diag.write_report(staging, report(), KEY)
    recovery = tmp_path / "recovery.b64"
    recovery.write_bytes(base64.b64encode(KEY))
    output = tmp_path / "private" / "report.json"
    args = ["decrypt", "--snapshot", str(staging / diag.FILENAME),
            "--key-file", str(recovery), "--output", str(output)]
    assert diag.main(args) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    original = output.read_bytes()
    assert diag.main(args) == 1
    assert output.read_bytes() == original


def test_native_secret_values_and_json_private_key_are_excluded(monkeypatch):
    monkeypatch.setenv("SMTP_PASS", "synthetic-mail-password")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-gemini-key")
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_JSON", json.dumps({
        "private_key": "synthetic-private-key", "client_email": "synthetic@example.invalid",
    }))
    owner = result()
    owner.ledger.append(LedgerEntryType.STAGE, {
        "message": "echo synthetic-mail-password synthetic-gemini-key synthetic-private-key",
        "authorization": "native credential header",
    })
    document = diag.build_report(owner, operation="send", exit_code=1, duration_seconds=2)
    content = canonical_json_bytes(document)
    for value in (b"synthetic-mail-password", b"synthetic-gemini-key", b"synthetic-private-key",
                  b"native credential header"):
        assert value not in content
    assert b"[REDACTED]" in content


def test_raw_sdk_capture_and_config_environment_are_not_projections():
    from types import SimpleNamespace

    owner = result()
    owner = RunResult(
        owner.metrics, owner.config, owner.attempt_id, owner.analysis_id, owner.ledger,
        session=SimpleNamespace(
            context=SimpleNamespace(captured_at=datetime.now(timezone.utc)),
            log_records=[{"message": "raw-cookie=do-not-export"}],
            config_snapshot={"SMTP_PASS": "raw-password"},
        ),
    )
    content = canonical_json_bytes(diag.build_report(
        owner, operation="send", exit_code=1, duration_seconds=0,
    ))
    assert b"raw-cookie" not in content
    assert b"raw-password" not in content
    assert b"log_records" not in content


def test_projection_failure_keeps_the_primary_failure_ledger(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.summary.SummaryProjector.project",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("private projection details")))
    document = report()
    assert document["projection_error_type"] == "ValueError"
    assert document["ledger"][0]["payload"]["stable_code"] == "NEWSLETTER_SEMANTIC_INVARIANT_FAILED"
    assert "private projection details" not in canonical_json_bytes(document).decode()


def test_capture_failure_is_best_effort_and_never_prints_key_or_error(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(diag.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    monkeypatch.setenv(sealed.KEY_ENV, "invalid-private-secret")
    assert not diag.capture_configured(result(), operation="send", exit_code=0, duration_seconds=1)
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()
    assert "invalid-private-secret" not in caplog.text
    assert any(getattr(record, "public_code", None) == "DIAGNOSTIC_CAPTURE_FAILED" for record in caplog.records)


@pytest.mark.parametrize("field,value", [
    ("duration_seconds", -1), ("duration_seconds", float("nan")),
    ("exit_code", True), ("exit_code", 2), ("operation", "smtp"),
    ("input_sha256", {"../private-input": "a" * 64}),
])
def test_report_domains_are_validated_before_any_snapshot_write(tmp_path, field, value):
    document = report()
    document[field] = value
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "encrypted", document, KEY)
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()


def test_ciphertext_staging_rejects_symlinks_and_unrelated_files(tmp_path):
    directory = tmp_path / "staging"
    directory.mkdir()
    caller = directory / "caller"
    caller.write_text("preserve")
    with pytest.raises(ValueError):
        diag.write_report(directory, report(), KEY)
    assert caller.read_text() == "preserve"
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError):
        diag.write_report(link, report(), KEY)


def test_duplicate_or_nonfinite_authenticated_json_is_not_a_valid_report(tmp_path):
    directory = tmp_path / "encrypted"
    path = directory / diag.FILENAME
    for content in (b'{"schema_version":"1","schema_version":"1"}', b'{"duration_seconds":NaN}'):
        sealed.write_snapshot(path, content, KEY, purpose=diag.PURPOSE)
        with pytest.raises(ValueError):
            diag.read_report(path, KEY)
