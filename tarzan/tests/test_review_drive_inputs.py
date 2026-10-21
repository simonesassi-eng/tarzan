"""Drive selection/auth boundaries and private temporary ownership."""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from tarzan import delivery
from tarzan.delivery import drive_loader as drive
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def fake_drive(monkeypatch, pages, *, data=None, fail_download=None):
    calls, downloads = [], []

    class Request:
        def __init__(self, response):
            self.response = response

        def execute(self):
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    class Files:
        def list(self, **kwargs):
            calls.append(kwargs)
            return Request(pages.get(kwargs.get("pageToken"), {"files": []}))

        def get_media(self, fileId, **kwargs):
            downloads.append(fileId)
            return fileId

    class Downloader:
        def __init__(self, buffer, request, chunksize=None):
            self.buffer, self.request = buffer, request
            assert chunksize == drive.DOWNLOAD_CHUNK_BYTES

        def next_chunk(self):
            if self.request == fail_download:
                raise OSError("synthetic download failure")
            self.buffer.write((data or {}).get(self.request, b"key,value\nx,1\n"))
            return None, True

    files = Files()
    service = types.SimpleNamespace(files=lambda: files)
    client = types.ModuleType("googleapiclient")
    client.__path__ = []
    http = types.ModuleType("googleapiclient.http")
    http.MediaIoBaseDownload = Downloader
    monkeypatch.setitem(sys.modules, "googleapiclient", client)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)
    monkeypatch.setattr(drive, "_drive_service", lambda credentials: service)
    return calls, downloads


def meta(identifier="orders", name="order_list.csv", **kwargs):
    return {"id": identifier, "name": name, **kwargs}


def test_all_pages_are_read_and_owned_staging_has_private_permissions(monkeypatch, tmp_path):
    calls, downloads = fake_drive(monkeypatch, {
        None: {"files": [{"id": "other", "name": "other.txt"}], "nextPageToken": "second"},
        "second": {"files": [meta()]},
    })
    original = tmp_path / "order_list.csv"
    original.write_bytes(b"caller original")
    with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path) as files:
        directory = files.directory
        path = files["order_list.csv"]
        assert path.parent == directory and directory.parent == tmp_path
        assert directory.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.read_bytes().startswith(b"key,value")
    assert len(calls) == 2 and calls[1]["pageToken"] == "second"
    assert "nextPageToken" in calls[0]["fields"]
    assert downloads == ["orders"]
    assert not directory.exists()
    assert original.read_bytes() == b"caller original"


@pytest.mark.parametrize("pages", [
    {None: {"files": [meta("one"), meta("two")]}},
    {None: {"files": [meta("one")], "nextPageToken": "second"},
     "second": {"files": [meta("two")]}},
])
def test_ambiguous_names_fail_before_download_or_staging(monkeypatch, tmp_path, pages):
    _, downloads = fake_drive(monkeypatch, pages)
    with pytest.raises(RuntimeError, match="ambiguous"):
        drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    assert downloads == []
    assert list(tmp_path.iterdir()) == []


def test_same_id_repeated_on_pages_is_not_arbitrarily_replaced(monkeypatch, tmp_path):
    fake_drive(monkeypatch, {
        None: {"files": [meta()], "nextPageToken": "second"},
        "second": {"files": [meta()]},
    })
    with drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path) as files:
        assert set(files) == {"order_list.csv"}


@pytest.mark.parametrize("failure", ["listing", "download"])
def test_failure_leaves_no_private_staging(monkeypatch, tmp_path, failure):
    fake_drive(
        monkeypatch,
        {None: OSError("synthetic authorization/listing failure")}
        if failure == "listing" else {None: {"files": [meta(), meta("targets", "targets.csv")]}},
        fail_download="targets" if failure == "download" else None,
    )
    with pytest.raises(OSError):
        drive.download_files("folder", "{}", ["order_list.csv", "targets.csv"], dest_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("token", ["again", 42])
def test_bad_pagination_fails_closed_without_download(monkeypatch, tmp_path, token):
    _, downloads = fake_drive(monkeypatch, {
        None: {"files": [meta()], "nextPageToken": token},
        "again": {"files": [], "nextPageToken": "again"},
    })
    with pytest.raises(RuntimeError, match="token"):
        drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    assert downloads == []


def test_required_input_selection_finishes_before_any_download(monkeypatch, tmp_path):
    _, downloads = fake_drive(monkeypatch, {None: {"files": [meta()]}})
    with pytest.raises(FileNotFoundError, match="targets.csv"):
        drive.download_files(
            "folder", "{}", drive.KNOWN_INPUT_FILES, dest_dir=tmp_path,
            required_filenames=("order_list.csv", "targets.csv"),
        )
    assert downloads == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("metadata", [
    meta(size=str(drive.MAX_DOWNLOAD_BYTES + 1)),
    meta(size="-1"),
    meta(mimeType="application/vnd.google-apps.spreadsheet"),
    meta(md5Checksum="invalid"),
    meta(identifier="../outside"),
])
def test_invalid_native_metadata_is_rejected_before_staging(monkeypatch, tmp_path, metadata):
    _, downloads = fake_drive(monkeypatch, {None: {"files": [metadata]}})
    with pytest.raises(RuntimeError):
        drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    assert downloads == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mismatch", ["size", "checksum", "bytes"])
def test_changed_or_unbounded_download_is_not_published(monkeypatch, tmp_path, mismatch):
    data = b"synthetic"
    details = {}
    if mismatch == "size":
        details["size"] = str(len(data) + 1)
    elif mismatch == "checksum":
        details["md5Checksum"] = hashlib.md5(b"different", usedforsecurity=False).hexdigest()
    else:
        monkeypatch.setattr(drive, "MAX_DOWNLOAD_BYTES", 4)
    fake_drive(monkeypatch, {None: {"files": [meta(**details)]}}, data={"orders": data})
    with pytest.raises(RuntimeError):
        drive.download_files("folder", "{}", ["order_list.csv"], dest_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_partial_drive_configuration_cannot_select_local_fallback(monkeypatch):
    monkeypatch.setenv("DRIVE_FOLDER_ID", "folder")
    monkeypatch.delenv("GOOGLE_DRIVE_CREDENTIALS_JSON", raising=False)
    with pytest.raises(RuntimeError, match="both"):
        delivery.resolve_inputs()


def test_local_input_lifecycle_never_deletes_caller_files(monkeypatch, tmp_path):
    orders, targets = tmp_path / "orders.csv", tmp_path / "targets.csv"
    orders.write_text("caller orders")
    targets.write_text("caller targets")
    monkeypatch.delenv("DRIVE_FOLDER_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_CREDENTIALS_JSON", raising=False)
    monkeypatch.setenv("ORDERS_PATH", str(orders))
    monkeypatch.setenv("TARGETS_PATH", str(targets))
    with delivery.resolve_inputs() as inputs:
        assert inputs["orders"] == str(orders)
    assert orders.read_text() == "caller orders"
    assert targets.read_text() == "caller targets"


@pytest.mark.parametrize("analysis_failure", [False, True])
def test_delivery_closes_only_its_owned_input_set_even_on_failure(
    monkeypatch, tmp_path, analysis_failure,
):
    for key in ("GITHUB_ACTIONS", "TARZAN_REQUESTED_AT", "TARZAN_WORKFLOW_CREATED_AT"):
        monkeypatch.delenv(key, raising=False)
    closed = []

    class Inputs(dict):
        def close(self):
            closed.append(True)

    for key, value in {
        "SMTP_USER": "owner@example.invalid", "SMTP_PASS": "synthetic",
        "RECIPIENT_EMAIL": "owner@example.invalid", "DRY_RUN": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: Inputs(
        orders="orders.csv", config="targets.csv", targets_per_holding=None,
    ))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **k: "<html>synthetic</html>")

    def run(**kwargs):
        if analysis_failure:
            raise RuntimeError("synthetic analysis error before a result")
        return PortfolioMetrics(total_value=100), InvestorConfig()

    monkeypatch.setattr(delivery, "run", run)
    if analysis_failure:
        with pytest.raises(RuntimeError):
            delivery.run_and_send()
    else:
        assert delivery.run_and_send() == 0
    assert closed == [True]
