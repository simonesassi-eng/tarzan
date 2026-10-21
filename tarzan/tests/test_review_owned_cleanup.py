"""Private owned staging: retry, primary errors and truthful delivery outcomes."""

from datetime import datetime
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from tarzan import delivery
from tarzan.delivery import drive_loader as drive
from tarzan.delivery.claims import DeliveryState, LocalJsonDeliveryClaimStore
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.session import RunAttemptEnvelope, RunResult, record_last_run_result
from tarzan.tests.test_review_drive_inputs import fake_drive, meta


def staged_inputs(tmp_path):
    directory = tempfile.TemporaryDirectory(prefix="synthetic-owned-", dir=tmp_path)
    paths = {}
    for name in ("order_list.csv", "targets.csv"):
        path = Path(directory.name) / name
        path.write_text("synthetic private input")
        paths[name] = path
    return drive.DownloadedInputSet(directory, paths)


def denied_cleanup(*args, **kwargs):
    raise PermissionError("synthetic cleanup failure")


def test_failed_close_remains_retryable_after_native_finalizer_detaches(monkeypatch, tmp_path):
    inputs = staged_inputs(tmp_path)
    original = inputs._directory._rmtree
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise PermissionError("synthetic temporary permission failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(inputs._directory, "_rmtree", fail_once)
    with pytest.raises(PermissionError):
        inputs.close()
    assert inputs["order_list.csv"].exists()
    assert not inputs._directory._finalizer.alive
    inputs.close()
    assert calls == [True, True]
    assert not inputs.directory.exists()
    with pytest.raises(RuntimeError, match="closed"):
        inputs["order_list.csv"]
    inputs.close()
    assert calls == [True, True]


@pytest.mark.parametrize("primary", [ValueError("synthetic consumer error"), KeyboardInterrupt(), SystemExit(3)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_cleanup_does_not_mask_the_active_consumer_exception(monkeypatch, tmp_path, caplog, primary, wrapped):
    owner = staged_inputs(tmp_path)
    inputs = delivery.ResolvedInputSet(
        {"orders": str(owner["order_list.csv"])}, downloaded=owner,
    ) if wrapped else owner
    with monkeypatch.context() as patch:
        patch.setattr(owner._directory, "_rmtree", denied_cleanup)
        with pytest.raises(type(primary)) as caught:
            with inputs:
                raise primary
        assert caught.value is primary
        assert owner.directory.exists()
        assert any(getattr(record, "public_code", None) == "INPUT_STAGING_CLEANUP_FAILED" for record in caplog.records)
    inputs.close()
    assert not owner.directory.exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_download_keeps_its_original_failure_when_cleanup_also_fails(
    monkeypatch, tmp_path, caplog, interrupted,
):
    fake_drive(
        monkeypatch,
        {None: {"files": [meta(), meta("targets", "targets.csv")]}},
        fail_download="targets",
    )
    original_directory = tempfile.TemporaryDirectory
    created = []

    def create_owned(*args, **kwargs):
        directory = original_directory(*args, **kwargs)
        created.append(directory)
        patch.setattr(directory, "_rmtree", denied_cleanup)
        return directory

    with monkeypatch.context() as patch:
        patch.setattr(drive.tempfile, "TemporaryDirectory", create_owned)
        if interrupted:
            import sys
            downloader = sys.modules["googleapiclient.http"].MediaIoBaseDownload
            original_chunk = downloader.next_chunk

            def chunk(self):
                if self.request == "targets":
                    raise KeyboardInterrupt()
                return original_chunk(self)

            patch.setattr(downloader, "next_chunk", chunk)
        with pytest.raises(KeyboardInterrupt if interrupted else OSError) as caught:
            drive.download_files(
                "folder", "{}", ["order_list.csv", "targets.csv"], dest_dir=tmp_path,
            )
        if not interrupted:
            assert str(caught.value) == "synthetic download failure"
        assert len(created) == 1
        assert any(getattr(record, "public_code", None) == "INPUT_STAGING_CLEANUP_FAILED" for record in caplog.records)
    # Only the test-owned directory, never its parent or caller-owned inputs.
    created[0].cleanup()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("phase", ["success", "listing", "download"])
def test_drive_native_service_connections_close_on_every_exit(monkeypatch, tmp_path, phase):
    fake_drive(
        monkeypatch,
        {None: OSError("synthetic listing failure")} if phase == "listing"
        else {None: {"files": [meta()]}},
        fail_download="orders" if phase == "download" else None,
    )
    service = drive._drive_service("{}")
    closed = []
    service.close = lambda: closed.append(True)
    if phase == "success":
        with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path):
            assert closed == [True]
    else:
        with pytest.raises(OSError):
            drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
        assert closed == [True]
    assert list(tmp_path.iterdir()) == []


def test_drive_close_failure_does_not_replace_a_native_listing_failure(monkeypatch, tmp_path, caplog):
    primary = OSError("synthetic native listing failure")
    fake_drive(monkeypatch, {None: primary})
    service = drive._drive_service("{}")
    service.close = denied_cleanup
    with pytest.raises(OSError) as caught:
        drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    assert caught.value is primary
    assert any(getattr(record, "public_code", None) == "DRIVE_CLIENT_CLEANUP_FAILED" for record in caplog.records)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("build_failure", [False, True])
def test_drive_native_oauth_readonly_scope_timeout_and_build_cleanup(monkeypatch, build_failure):
    from google.oauth2 import service_account
    from googleapiclient import discovery
    import httplib2

    credentials, captured, closed = object(), [], []
    primary = RuntimeError("synthetic discovery construction failure")

    def credential(info, scopes):
        assert info == {"synthetic": True}
        assert scopes == ["https://www.googleapis.com/auth/drive.readonly"]
        return credentials

    class Http:
        def __init__(self, *, timeout):
            assert timeout == 20

        def close(self):
            closed.append(True)

    def build(api, version, *, http, cache_discovery):
        assert (api, version, cache_discovery) == ("drive", "v3", False)
        assert http.credentials is credentials and isinstance(http.http, Http)
        captured.append(http)
        if build_failure:
            raise primary
        return SimpleNamespace(close=http.close)

    monkeypatch.setattr(service_account.Credentials, "from_service_account_info", credential)
    monkeypatch.setattr(httplib2, "Http", Http)
    monkeypatch.setattr(discovery, "build", build)
    if build_failure:
        with pytest.raises(RuntimeError) as caught:
            drive._drive_service('{"synthetic":true}')
        assert caught.value is primary
        assert closed == [True]
    else:
        service = drive._drive_service('{"synthetic":true}')
        assert closed == []
        service.close()
        assert closed == [True]
    assert len(captured) == 1


@pytest.fixture
def owned_delivery(monkeypatch, tmp_path):
    for name in (
        "GITHUB_ACTIONS", "TARZAN_REQUESTED_AT", "TARZAN_WORKFLOW_CREATED_AT",
        "DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN", "DELIVERY_CLAIM_STORE_PATH",
        "AUTHORIZED_RESEND_TOKEN", "DRIVE_FOLDER_ID", "GOOGLE_DRIVE_CREDENTIALS_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "SMTP_USER": "sender@example.invalid", "SMTP_PASS": "synthetic",
        "RECIPIENT_EMAIL": "reader@example.invalid",
        "TARZAN_STABLE_EVENT_ID": "synthetic:owned-cleanup", "DRY_RUN": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "now_local", lambda: datetime(2026, 10, 21, 12))
    monkeypatch.setattr(delivery, "_failed_checks_sha", lambda: None)
    monkeypatch.setattr(delivery, "build_subject", lambda *args: "synthetic subject")
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: "<html>synthetic</html>")
    store = LocalJsonDeliveryClaimStore(tmp_path / "claims.json")
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: store)
    return store


@pytest.mark.parametrize("terminal", ["DRY_RUN", "ACKNOWLEDGED_SUCCESS", "DEFINITE_PRE_SEND_FAILURE", "UNCERTAIN"])
@pytest.mark.parametrize("transient", [False, True])
def test_cleanup_is_recorded_before_finalization_and_cannot_change_smtp_state(
    monkeypatch, tmp_path, owned_delivery, terminal, transient,
):
    owner = staged_inputs(tmp_path)
    inputs = delivery.ResolvedInputSet({
        "orders": str(owner["order_list.csv"]), "config": str(owner["targets.csv"]),
        "targets_per_holding": None,
    }, downloaded=owner)
    cleanup_calls, sent, results = [], [], []
    original = owner._directory._rmtree

    def cleanup(*args, **kwargs):
        cleanup_calls.append(True)
        if not transient or len(cleanup_calls) == 1:
            raise PermissionError("synthetic permission failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner._directory, "_rmtree", cleanup)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: inputs)
    if terminal == "DRY_RUN":
        monkeypatch.setenv("DRY_RUN", "1")

    def analysis(**kwargs):
        assert owner.directory.exists() and cleanup_calls == []
        envelope = RunAttemptEnvelope.create("synthetic-delivery")
        ledger = RunLedger(envelope.attempt_id)
        ledger.append(LedgerEntryType.STAGE, {"stage": "run", "outcome": "COMPLETE"})
        result = RunResult(
            PortfolioMetrics(total_value=100), InvestorConfig(), envelope.attempt_id,
            "synthetic-analysis", ledger,
        )
        results.append(result)
        record_last_run_result(result)
        return result.compatibility_tuple()

    def transport(**kwargs):
        assert cleanup_calls == [True, True]
        if terminal == "DEFINITE_PRE_SEND_FAILURE":
            raise ConnectionError("synthetic connection failure")
        kwargs["before_invoke"]()
        sent.append(True)
        if terminal == "UNCERTAIN":
            raise ConnectionError("synthetic ambiguous response")

    monkeypatch.setattr(delivery, "run", analysis)
    monkeypatch.setattr(delivery, "send_email", transport)
    try:
        assert delivery.run_and_send() == (1 if terminal in {"UNCERTAIN", "DEFINITE_PRE_SEND_FAILURE"} else 0)
        assert cleanup_calls == [True, True]
        manifests = list(tmp_path.glob("output/**/manifest.json"))
        assert len(manifests) == 1
        document = json.loads(manifests[0].read_text())
        assert document["delivery_state"] == terminal
        ledger = (manifests[0].parent / "ledger.jsonl").read_text()
        report = (manifests[0].parent / "report.html").read_text()
        if transient:
            assert not owner.directory.exists()
            assert "CLEANED_AFTER_RETRY" in ledger
            assert "INPUT_STAGING_CLEANUP_FAILED" not in ledger
        else:
            assert owner.directory.exists()
            assert "INPUT_STAGING_CLEANUP_FAILED" in ledger
            assert "INPUT_STAGING_CLEANUP_FAILED" in report
            assert results[0].ledger.failure_records()[0].severity == "WARNING"
        assert sent == ([True] if terminal in {"ACKNOWLEDGED_SUCCESS", "UNCERTAIN"} else [])
        if terminal != "DRY_RUN":
            claim = next(iter(json.loads(owned_delivery.path.read_text())["claims"].values()))
            assert claim["state"] == terminal
    finally:
        monkeypatch.setattr(owner._directory, "_rmtree", original)
        inputs.close()


def test_unrecorded_analysis_error_is_not_replaced_by_cleanup_failure(
    monkeypatch, tmp_path, owned_delivery, caplog,
):
    owner = staged_inputs(tmp_path)
    inputs = delivery.ResolvedInputSet({
        "orders": str(owner["order_list.csv"]), "config": str(owner["targets.csv"]),
        "targets_per_holding": None,
    }, downloaded=owner)
    original = owner._directory._rmtree
    primary = RuntimeError("synthetic unrecorded analysis error")

    def analysis(**kwargs):
        raise primary

    monkeypatch.setattr(delivery, "resolve_inputs", lambda: inputs)
    monkeypatch.setattr(delivery, "run", analysis)
    monkeypatch.setattr(delivery, "send_email", lambda **kwargs: pytest.fail("unexpected SMTP"))
    monkeypatch.setattr(owner._directory, "_rmtree", denied_cleanup)
    try:
        with pytest.raises(RuntimeError) as caught:
            delivery.run_and_send()
        assert caught.value is primary
        assert owner.directory.exists()
        assert any(getattr(record, "public_code", None) == "INPUT_STAGING_CLEANUP_FAILED" for record in caplog.records)
        assert not owned_delivery.path.exists()
    finally:
        monkeypatch.setattr(owner._directory, "_rmtree", original)
        inputs.close()


def test_cleanup_warning_cannot_manufacture_evidence_of_a_zero_value_analysis(
    monkeypatch, tmp_path, owned_delivery,
):
    class Inputs(dict):
        def close(self):
            raise PermissionError("synthetic cleanup failure")

    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: Inputs(
        orders="synthetic", config="synthetic", targets_per_holding=None,
    ))
    result = []

    def analysis(**kwargs):
        envelope = RunAttemptEnvelope.create("synthetic-empty-ledger")
        current = RunResult(
            PortfolioMetrics(total_value=0), InvestorConfig(), envelope.attempt_id,
            "synthetic-analysis", RunLedger(envelope.attempt_id),
        )
        result.append(current)
        record_last_run_result(current)
        return current.compatibility_tuple()

    monkeypatch.setattr(delivery, "run", analysis)
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: pytest.fail("invalid normal newsletter"))
    assert delivery.run_and_send() == 0
    codes = {record.stable_code for record in result[0].ledger.failure_records()}
    assert codes == {"EMPTY_PORTFOLIO_VALUE", "INPUT_STAGING_CLEANUP_FAILED"}
    assert len(list(tmp_path.glob("output/**/manifest.json"))) == 1


def test_cleanup_warning_has_only_a_fixed_public_code():
    from tarzan.log_redaction import RestrictedConsoleFormatter
    record = logging.LogRecord(
        "tarzan.delivery.drive_loader", logging.WARNING, "", 1,
        "private staging at %s", ("/private/user/portfolio-inputs",), None,
    )
    record.public_code = "INPUT_STAGING_CLEANUP_FAILED"
    assert RestrictedConsoleFormatter().format(record) == "[WARNING] delivery: INPUT_STAGING_CLEANUP_FAILED"
    assert record.getMessage() == "private staging at /private/user/portfolio-inputs"
