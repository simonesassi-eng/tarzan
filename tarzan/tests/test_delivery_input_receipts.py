"""Only observed Drive bytes may become exact-attempt diagnostic fingerprints."""

from contextlib import ExitStack
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib

import pytest

from tarzan import delivery
from tarzan.delivery import drive_loader as drive
from tarzan.runtime import diagnostics
from tarzan.runtime.input_receipts import immutable_fingerprints
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import (
    RunContext, RunMode, RunResult, RunSession, record_last_run_result,
)
from tarzan.tests.test_review_drive_inputs import fake_drive, meta
from tarzan.tests.test_send_diagnostic_lifecycle import shim as send_shim
from tarzan.tests.test_encrypted_diagnostics import KEY
from tarzan.tests.test_final_acquisition_receipts import native_result as budgeted_result


def native_result():
    context = RunContext("native-input-receipt", RunMode.LIVE, None, datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    session = RunSession(context, {}, ledger)
    return RunResult(None, None, context.attempt_id, "unchanged-analysis", ledger, session)


def test_validated_drive_bytes_receipt_survives_staging_release(monkeypatch, tmp_path):
    content = b"key,value\nsynthetic,1\n"
    fake_drive(monkeypatch, {None: {"files": [
        meta(size=str(len(content)), md5Checksum=hashlib.md5(content).hexdigest()),
    ]}}, data={"orders": content})
    files = drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    expected = hashlib.sha256(content).hexdigest()
    assert dict(files.fingerprints) == {"order_list.csv": expected}
    directory = files.directory
    files.close()
    assert not directory.exists()
    assert dict(files.fingerprints) == {"order_list.csv": expected}
    with pytest.raises(TypeError):
        files.fingerprints["order_list.csv"] = "b" * 64
    with pytest.raises(AttributeError):
        files.fingerprints = {"order_list.csv": "b" * 64}


def test_receipt_never_re_reads_a_later_file_version(monkeypatch, tmp_path):
    original = b"key,value\noriginal,1\n"
    fake_drive(monkeypatch, {None: {"files": [meta()]}}, data={"orders": original})
    with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path) as files:
        expected = hashlib.sha256(original).hexdigest()
        files["order_list.csv"].write_bytes(b"caller changed later bytes")
        assert files.fingerprints["order_list.csv"] == expected


def test_unrequested_input_and_generic_download_never_create_financial_receipts(
    monkeypatch, tmp_path,
):
    fake_drive(monkeypatch, {None: {"files": [
        meta(), meta("custom", "custom.csv"),
    ]}})
    with drive.download_files("folder", "{}", ["custom.csv"], dest_dir=tmp_path) as files:
        assert list(files) == ["custom.csv"]
        assert dict(files.fingerprints) == {}


def test_hash_observer_failure_keeps_downloaded_input_and_owned_cleanup(monkeypatch, tmp_path):
    content = b"valid downloaded bytes"
    fake_drive(monkeypatch, {None: {"files": [meta()]}}, data={"orders": content})
    monkeypatch.setattr(drive.hashlib, "sha256", lambda data: (
        (_ for _ in ()).throw(RuntimeError("synthetic receipt failure"))
    ))
    with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path) as files:
        assert files["order_list.csv"].read_bytes() == content
        assert dict(files.fingerprints) == {}
        directory = files.directory
    assert not directory.exists()


def test_drive_resolution_preserves_known_filenames_receipts_and_legacy_path_contract(
    monkeypatch, tmp_path,
):
    fake_drive(monkeypatch, {None: {"files": [
        meta(), meta("targets", "targets.csv"), meta("per_holding", "targets_per_holding.csv"),
    ]}})
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "{}")
    with delivery.resolve_inputs() as inputs:
        assert set(inputs) == {"config", "orders", "targets_per_holding"}
        assert all(isinstance(path, str) for path in inputs.values())
        assert set(inputs.fingerprints) == {
            "order_list.csv", "targets.csv", "targets_per_holding.csv",
        }
        expected = dict(inputs.fingerprints)
    assert dict(inputs.fingerprints) == expected


def test_local_and_injected_sources_never_invent_raw_byte_receipts():
    inputs = delivery.ResolvedInputSet({
        "config": "not-read.csv", "orders": "not-read.csv", "targets_per_holding": None,
    })
    assert dict(inputs.fingerprints) == {}
    with pytest.raises(AttributeError):
        inputs.fingerprints = {"order_list.csv": "b" * 64}
    inputs.close()


def test_only_the_copied_mapping_is_validated_not_an_earlier_iteration():
    class ChangingMapping(Mapping):
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(["order_list.csv"] if self.iterations == 1
                        else ["synthetic@example.invalid"])

        def __len__(self):
            return 1

        def __getitem__(self, key):
            return "a" * 64

    receipt = immutable_fingerprints(ChangingMapping())
    assert set(receipt).issubset({"order_list.csv", "targets.csv", "targets_per_holding.csv"})


def test_receipt_failure_and_optional_logging_failure_do_not_lose_download(
    monkeypatch, tmp_path,
):
    content = b"valid downloaded bytes"
    fake_drive(monkeypatch, {None: {"files": [meta()]}}, data={"orders": content})
    monkeypatch.setattr(drive.hashlib, "sha256", lambda data: (
        (_ for _ in ()).throw(RuntimeError("synthetic receipt failure"))
    ))
    monkeypatch.setattr(drive.logger, "warning", lambda *args, **kwargs: (
        (_ for _ in ()).throw(RuntimeError("synthetic observer logging failure"))
    ))
    with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path) as files:
        assert files["order_list.csv"].read_bytes() == content
        assert dict(files.fingerprints) == {}
        directory = files.directory
    assert not directory.exists()


@pytest.mark.parametrize("values", [
    None, [], "invalid",
    {"unobserved.csv": "a" * 64},
    {"order_list.csv": "A" * 64},
    {"order_list.csv": "a" * 63},
    {"order_list.csv": True},
    {"order_list.csv": None},
    {"order_list.csv": b"a" * 64},
])
def test_invalid_receipts_cannot_bind_or_mutate_a_native_attempt(values):
    result = native_result()
    with pytest.raises(ValueError):
        result.session.bind_input_fingerprints(values)
    assert dict(result.session.input_fingerprints) == {}
    assert result.analysis_id == "unchanged-analysis"


def test_bind_copies_receipt_once_without_financial_or_ledger_mutation():
    result = native_result()
    values = {"order_list.csv": "a" * 64}
    before = tuple(result.ledger.entries)
    result.session.bind_input_fingerprints(values)
    values["order_list.csv"] = "b" * 64
    assert dict(result.session.input_fingerprints) == {"order_list.csv": "a" * 64}
    assert result.analysis_id == "unchanged-analysis" and tuple(result.ledger.entries) == before
    with pytest.raises(TypeError):
        result.session.input_fingerprints["order_list.csv"] = "c" * 64
    with pytest.raises(RuntimeError):
        result.session.bind_input_fingerprints(values)


def test_diagnostics_use_exact_native_owner_receipt_and_respect_explicit_override():
    result = native_result()
    result.session.bind_input_fingerprints({"order_list.csv": "a" * 64})
    values = {"operation": "send", "exit_code": 0, "duration_seconds": 1}
    assert diagnostics.build_report(result, **values)["input_sha256"] == {
        "order_list.csv": "a" * 64,
    }
    assert diagnostics.build_report(result, input_sha256={}, **values)["input_sha256"] == {}
    assert diagnostics.build_report(result, input_sha256={
        "targets.csv": "b" * 64,
    }, **values)["input_sha256"] == {"targets.csv": "b" * 64}


def test_foreign_attempt_never_borrows_a_receipt():
    result = native_result()
    result.session.bind_input_fingerprints({"order_list.csv": "a" * 64})
    foreign = RunResult(None, None, "another-attempt", result.analysis_id, result.ledger,
                        result.session)
    report = diagnostics.build_report(foreign, operation="send", exit_code=1, duration_seconds=1)
    assert report["input_sha256"] == {}


def test_same_object_with_foreign_ledger_attempt_never_attests_to_native_receipts():
    result = native_result()
    result.session.bind_input_fingerprints({"order_list.csv": "a" * 64})
    result.ledger.attempt_id = "foreign-ledger-attempt"
    report = diagnostics.build_report(result, operation="send", exit_code=1, duration_seconds=1)
    assert report["input_sha256"] == {}
    assert "final_acquisition" not in report["run"]


@pytest.mark.parametrize("ordinary_publish_failure", [False, True])
def test_actual_send_binding_survives_input_release_and_publication_failure(
    monkeypatch, ordinary_publish_failure,
):
    result = native_result()
    inputs = delivery.ResolvedInputSet({
        "config": "unread-config", "orders": "unread-orders", "targets_per_holding": None,
    }, fingerprints={"order_list.csv": "a" * 64})
    for name in ("SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL"):
        monkeypatch.setenv(name, "synthetic-value")
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: inputs)

    def run(**kwargs):
        record_last_run_result(result)
        return result.compatibility_tuple()

    def publish(*args, release_inputs, **kwargs):
        release_inputs()
        assert dict(result.session.input_fingerprints) == {"order_list.csv": "a" * 64}
        if ordinary_publish_failure:
            raise RuntimeError("synthetic publication failure")
        return 0

    monkeypatch.setattr(delivery, "run", run)
    monkeypatch.setattr(delivery, "_publish_and_deliver", publish)
    with ExitStack() as owner:
        if ordinary_publish_failure:
            with pytest.raises(RuntimeError, match="synthetic publication"):
                delivery._run_and_send_owned(owner)
        else:
            assert delivery._run_and_send_owned(owner) == 0
    assert dict(result.session.input_fingerprints) == {"order_list.csv": "a" * 64}


def test_input_receipt_copy_rejects_private_arbitrary_labels():
    with pytest.raises(ValueError):
        immutable_fingerprints({"synthetic@example.invalid": "a" * 64})


def test_binding_failure_and_optional_log_failure_do_not_abort_publication(monkeypatch):
    result = native_result()
    inputs = delivery.ResolvedInputSet({
        "config": "unread-config", "orders": "unread-orders", "targets_per_holding": None,
    }, fingerprints={"order_list.csv": "a" * 64})
    for name in ("SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL"):
        monkeypatch.setenv(name, "synthetic-value")
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: inputs)

    def run(**kwargs):
        record_last_run_result(result)
        return result.compatibility_tuple()

    monkeypatch.setattr(delivery, "run", run)
    monkeypatch.setattr(result.session, "bind_input_fingerprints", lambda _: (
        (_ for _ in ()).throw(ValueError("synthetic binding failure"))
    ))
    monkeypatch.setattr(delivery.logger, "warning", lambda *args, **kwargs: (
        (_ for _ in ()).throw(RuntimeError("synthetic logging failure"))
    ))
    publications = []
    monkeypatch.setattr(delivery, "_publish_and_deliver",
                        lambda *args, **kwargs: publications.append(1) or 0)
    assert delivery.run_and_send() == 0
    assert publications == [1]
    assert dict(result.session.input_fingerprints) == {}


@pytest.mark.parametrize("primary", [0, 1, "ordinary-error"])
def test_native_drive_to_send_shim_preserves_receipts_after_cleanup(
    monkeypatch, tmp_path, send_shim, primary,
):
    result = budgeted_result()
    content = {
        "orders": b"key,value\nsynthetic-orders,1\n",
        "targets": b"key,value\nsynthetic-targets,2\n",
        "per_holding": b"key,value\nsynthetic-holding,3\n",
    }
    fake_drive(monkeypatch, {None: {"files": [
        meta(), meta("targets", "targets.csv"), meta("per_holding", "targets_per_holding.csv"),
    ]}}, data=content)
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("GOOGLE_DRIVE_CREDENTIALS_JSON", "{}")
    for name in ("SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL"):
        monkeypatch.setenv(name, "synthetic-value")
    monkeypatch.setenv(diagnostics.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    monkeypatch.setattr("tarzan.runtime.sealed.configured_key", lambda **kwargs: KEY)
    publications, directories = [], []
    native_resolve = delivery.resolve_inputs

    def resolve():
        inputs = native_resolve()
        directories.append(inputs.staging_directory)
        return inputs

    def run(**kwargs):
        record_last_run_result(result)
        return result.compatibility_tuple()

    def publish(*args, release_inputs, **kwargs):
        publications.append(1)
        with result.session.transport.coordinator.lease(optional=True, remaining=lambda: 100):
            pass
        release_inputs()
        result.session.close_transport()
        if primary == "ordinary-error":
            raise RuntimeError("synthetic publication failure")
        return primary

    monkeypatch.setattr(delivery, "resolve_inputs", resolve)
    monkeypatch.setattr(delivery, "run", run)
    monkeypatch.setattr(delivery, "_publish_and_deliver", publish)
    assert send_shim.main() == (1 if primary == "ordinary-error" else primary)
    restored = diagnostics.read_report(tmp_path / "encrypted" / diagnostics.FILENAME, KEY)
    expected = {
        name: hashlib.sha256(content[identifier]).hexdigest()
        for name, identifier in (
            ("order_list.csv", "orders"), ("targets.csv", "targets"),
            ("targets_per_holding.csv", "per_holding"),
        )
    }
    assert restored["input_sha256"] == expected
    assert restored["run"]["analysis_id"] == result.analysis_id == "unchanged-analysis"
    final = restored["run"]["final_acquisition"]["native_acquisition"]
    assert final["requested"]["value"] == final["acquired"]["value"] == final["released"]["value"] == 1
    assert final["active"]["value"] == 0
    assert publications == [1] and len(directories) == 1
    assert not directories[0].exists()
    assert (restored["entrypoint_error_type"] == "RuntimeError") is (primary == "ordinary-error")
