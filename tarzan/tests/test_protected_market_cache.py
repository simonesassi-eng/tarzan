"""No plaintext, unauthenticated data, or renewed cache timestamps on restore."""

import base64
import json

import pandas as pd
import pytest

from tarzan.data import price_cache, protected_cache as cache
from tarzan.runtime import sealed
from tarzan.runtime.io_utils import canonical_json_bytes


KEY = b"synthetic-key-is-not-a-secret-00"
assert len(KEY) == 32


@pytest.fixture
def locations(monkeypatch, tmp_path):
    private, staging = tmp_path / "private", tmp_path / "sealed"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(private))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    return private, staging


def prices():
    return pd.DataFrame(
        {"Close": [100.0, 101.0], "Price Close": [100.0, 101.0]},
        index=pd.date_range("2026-10-19", periods=2, tz="Europe/Rome", name="Date"),
    )


def test_roundtrip_preserves_prices_resolution_bytes_and_original_ttl(locations, monkeypatch, tmp_path):
    private, staging = locations
    price_cache.store_history("PRIVATE.MI", prices())
    price_cache.store_resolution("IE0000000001", "PRIVATE.MI")
    price_cache.store_timed("openfigi", "IE0000000001", [{"data": [{"ticker": "PRIVATE"}]}])
    original = {
        str(path.relative_to(private)): path.read_bytes()
        for path in private.rglob("*.json")
    }
    assert cache.export_snapshot(staging, KEY) == 3
    payload = (staging / cache.FILENAME).read_bytes()
    for value in (b"PRIVATE", b"IE0000000001", b"Close", b"resolution"):
        assert value not in payload
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "restored"))
    assert cache.import_snapshot(staging, KEY) == 3
    for name, content in original.items():
        restored = (price_cache.cache_dir() / name).read_bytes()
        assert json.loads(restored) == json.loads(content)
        assert (price_cache.cache_dir() / name).stat().st_mode & 0o777 == 0o600
    pd.testing.assert_frame_equal(price_cache.load_history("PRIVATE.MI"), prices())
    assert price_cache.load_resolution("IE0000000001") == "PRIVATE.MI"
    assert price_cache.cache_dir().stat().st_mode & 0o777 == 0o700
    assert (staging / cache.FILENAME).stat().st_mode & 0o777 == 0o600


def test_snapshots_use_fresh_nonce_and_are_bound_to_their_purpose():
    first = sealed.seal(b"same private content", KEY, purpose=cache.PURPOSE)
    second = sealed.seal(b"same private content", KEY, purpose=cache.PURPOSE)
    assert first != second
    assert sealed.unseal(first, KEY, purpose=cache.PURPOSE) == b"same private content"
    with pytest.raises(ValueError, match="authentication"):
        sealed.unseal(first, KEY, purpose="tarzan-diagnostics-v1")


@pytest.mark.parametrize("case", ["wrong-key", "tampered", "truncated", "unsupported"])
def test_authentication_failure_precedes_all_plaintext_cache_writes(locations, monkeypatch, tmp_path, case):
    _, staging = locations
    price_cache.store_resolution("IE0000000001", "PRIVATE.MI")
    cache.export_snapshot(staging, KEY)
    path = staging / cache.FILENAME
    payload = path.read_bytes()
    key = KEY
    if case == "wrong-key":
        key = b"x" * 32
    elif case == "tampered":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    elif case == "truncated":
        payload = payload[:20]
    else:
        payload = b"not an encrypted snapshot"
    path.write_bytes(payload)
    destination = tmp_path / "restored"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(destination))
    with pytest.raises(ValueError):
        cache.import_snapshot(staging, key)
    assert not list(destination.rglob("*.json"))


def test_authentication_is_checked_before_decompression(monkeypatch):
    payload = sealed.seal(b"secret", KEY, purpose=cache.PURPOSE)
    monkeypatch.setattr(sealed.zlib, "decompressobj", lambda: pytest.fail("unauthenticated decompression"))
    with pytest.raises(ValueError, match="authentication"):
        sealed.unseal(payload, b"x" * 32, purpose=cache.PURPOSE)


def test_restore_does_not_overwrite_newer_run_local_entries(locations):
    _, staging = locations
    price_cache.store_resolution("IE0000000001", "OLDER.MI")
    cache.export_snapshot(staging, KEY)
    price_cache.store_resolution("IE0000000001", "NEWER.DE")
    assert cache.import_snapshot(staging, KEY) == 0
    assert price_cache.load_resolution("IE0000000001") == "NEWER.DE"


def test_export_excludes_inputs_claims_cookies_output_and_manual_or_custom_histories(locations):
    private, staging = locations
    price_cache.store_history("PRIVATE.MI", prices())
    price_cache.store_history("manual:LICENSED", prices())
    custom = prices().assign(quantity=12345)
    price_cache.store_history("ACCOUNT.MI", custom)
    custom_dates = prices().assign(account_opened=pd.Timestamp("2025-01-01"))
    price_cache.store_history("ACCOUNT-DATES.MI", custom_dates)
    for name in ("input", "output", "claims", "yahoo"):
        directory = private / name
        directory.mkdir()
        (directory / "secret.json").write_text('{"private_input":true}')
    (private / "order_list.csv").write_text("private orders")
    assert cache.export_snapshot(staging, KEY) == 1
    document = json.loads(sealed.unseal(
        (staging / cache.FILENAME).read_bytes(), KEY, purpose=cache.PURPOSE,
    ))
    assert set(document["files"]) == {"history/PRIVATE.MI.json"}
    assert list(staging.iterdir()) == [staging / cache.FILENAME]


@pytest.mark.parametrize("relative", [
    "../order_list.csv", "/tmp/private-input.json", "history/../order_list.csv",
    "input/orders.json", "resolution/isin_to_symbol.json/other", "cookies.json",
])
def test_all_authenticated_archive_paths_are_validated_before_the_first_write(locations, monkeypatch, tmp_path, relative):
    _, staging = locations
    price_cache.store_resolution("IE0000000001", "PRIVATE.MI")
    cache.export_snapshot(staging, KEY)
    path = staging / cache.FILENAME
    document = json.loads(sealed.unseal(path.read_bytes(), KEY, purpose=cache.PURPOSE))
    document["files"][relative] = next(iter(document["files"].values()))
    sealed.write_snapshot(path, canonical_json_bytes(document), KEY, purpose=cache.PURPOSE)
    destination = tmp_path / "restored"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(destination))
    with pytest.raises(ValueError):
        cache.import_snapshot(staging, KEY)
    assert not list(destination.rglob("*.json"))


@pytest.mark.parametrize("case", ["private", "child", "parent", "symlink", "symlink-parent", "unrelated"])
def test_ciphertext_staging_never_aliases_plaintext_or_caller_files(locations, tmp_path, case):
    private, staging = locations
    private.mkdir()
    if case == "private":
        staging = private
    elif case == "child":
        staging = private / "sealed"
    elif case == "parent":
        staging = tmp_path
    elif case == "symlink":
        staging.symlink_to(private, target_is_directory=True)
    elif case == "symlink-parent":
        link = tmp_path / "link"
        link.symlink_to(private, target_is_directory=True)
        staging = link / "sealed"
    else:
        staging.mkdir()
        (staging / "caller-file").write_text("preserve")
    with pytest.raises(ValueError):
        cache.export_snapshot(staging, KEY)
    if case == "unrelated":
        assert (staging / "caller-file").read_text() == "preserve"


def test_symlink_history_is_never_archived(locations, tmp_path):
    private, staging = locations
    private.mkdir()
    external = tmp_path / "caller"
    external.mkdir()
    (private / "history").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        cache.export_snapshot(staging, KEY)


def test_corrupt_entry_does_not_prevent_other_validated_entries_from_persisting(locations):
    private, staging = locations
    price_cache.store_resolution("IE0000000001", "PRIVATE.MI")
    history = private / "history"
    history.mkdir()
    (history / "BAD.json").write_text("corrupt private content")
    assert cache.export_snapshot(staging, KEY) == 1


def test_decompression_limit_and_trailing_compressed_payload_are_rejected(monkeypatch):
    payload = sealed.seal(b"a" * 1000, KEY, purpose=cache.PURPOSE)
    monkeypatch.setattr(sealed, "MAX_PLAIN_BYTES", 100)
    with pytest.raises(ValueError, match="bounded contract"):
        sealed.unseal(payload, KEY, purpose=cache.PURPOSE)


def test_key_recovery_file_is_private_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / "recovery" / "key.b64"
    sealed.generate_key(path)
    key = sealed.configured_key(key_file=path)
    assert len(key) == 32
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        sealed.generate_key(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("encoded", ["", "not base64", base64.b64encode(b"short").decode()])
def test_missing_or_invalid_key_fails_closed_without_echoing_it(monkeypatch, encoded, capsys):
    monkeypatch.setenv(sealed.KEY_ENV, encoded)
    assert cache.main(["key-id"]) == 1
    assert not encoded or encoded not in capsys.readouterr().out


def test_workflow_save_output_requires_successfully_encrypted_nonempty_export(locations, monkeypatch, tmp_path):
    _, staging = locations
    output = tmp_path / "actions-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv(sealed.KEY_ENV, base64.b64encode(KEY).decode())
    assert cache.main(["export", "--directory", str(staging)]) == 0
    assert output.read_text() == "nonempty=false\n"
    price_cache.store_resolution("IE0000000001", "PRIVATE.MI")
    assert cache.main(["export", "--directory", str(staging)]) == 0
    assert output.read_text().endswith("nonempty=true\n")
    before = output.read_bytes()
    monkeypatch.setenv(sealed.KEY_ENV, "invalid private secret")
    assert cache.main(["export", "--directory", str(staging)]) == 1
    assert output.read_bytes() == before
