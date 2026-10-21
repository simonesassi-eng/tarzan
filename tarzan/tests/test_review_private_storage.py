"""New application-owned directories/locks are private; caller modes survive."""

import os
from pathlib import Path

import pandas as pd
import pytest

from tarzan.data import price_cache
from tarzan.delivery.claims import DeliveryIntent, DeliveryPurpose, LocalJsonDeliveryClaimStore
from tarzan.runtime.io_utils import private_lock_file


def permissions(path):
    return Path(path).stat().st_mode & 0o777


def test_new_private_cache_namespaces_and_locks_are_not_world_readable(monkeypatch, tmp_path):
    base = tmp_path / "synthetic-cache"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(base))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    frame = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2026-01-01", periods=1))
    price_cache.store_history("SYNTHETIC.X", frame)
    price_cache.store_resolution("XX0000000001", "SYNTHETIC.X")
    assert permissions(base) == 0o700
    assert permissions(base / "history") == 0o700
    files = list(base.rglob("*.json")) + list(base.rglob("*.lock"))
    assert len(files) == 4
    assert all(permissions(path) == 0o600 for path in files)
    pd.testing.assert_frame_equal(price_cache.load_history("SYNTHETIC.X"), frame)


def test_new_local_claim_directory_and_lock_are_private(tmp_path):
    path = tmp_path / "synthetic-claims" / "claims.json"
    store = LocalJsonDeliveryClaimStore(path)
    intent = DeliveryIntent(
        "synthetic-event", DeliveryPurpose.NORMAL_NEWSLETTER,
        "synthetic-recipients", "synthetic-template-v1",
    )
    assert store.claim(intent).created
    assert permissions(path.parent) == 0o700
    assert permissions(path) == 0o600
    assert permissions(path.with_suffix(".json.lock")) == 0o600
    assert store.claim(intent).duplicate


@pytest.mark.parametrize("kind", ["cache", "claims"])
def test_existing_caller_directory_permissions_are_never_changed(monkeypatch, tmp_path, kind):
    existing = tmp_path / "caller-existing"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    if kind == "cache":
        monkeypatch.setenv("TARZAN_CACHE_DIR", str(existing))
        assert price_cache.cache_dir() == existing
        price_cache._subdir("history")
        assert permissions(existing / "history") == 0o700
    else:
        LocalJsonDeliveryClaimStore(existing / "claims.json")
    assert permissions(existing) == 0o755


def test_existing_lock_content_and_permissions_are_preserved(tmp_path):
    path = tmp_path / "caller.lock"
    path.write_bytes(b"caller-owned")
    path.chmod(0o640)
    with private_lock_file(path) as handle:
        handle.seek(0)
        assert handle.read() == b"caller-owned"
    assert permissions(path) == 0o640
    assert path.read_bytes() == b"caller-owned"


def test_private_lock_is_closed_on_body_failure(tmp_path):
    path = tmp_path / "synthetic.lock"
    with pytest.raises(ValueError):
        with private_lock_file(path) as handle:
            descriptor = handle.fileno()
            raise ValueError("synthetic consumer failure")
    with pytest.raises(OSError):
        os.fstat(descriptor)
    assert permissions(path) == 0o600


def test_symlink_lock_cannot_open_or_change_the_target_on_supported_posix(tmp_path):
    # The supported Linux Actions/macOS hosts expose this native flag.
    assert hasattr(os, "O_NOFOLLOW")
    target, alias = tmp_path / "caller-original", tmp_path / "synthetic.lock"
    target.write_bytes(b"caller-original")
    alias.symlink_to(target)
    with pytest.raises(OSError):
        with private_lock_file(alias):
            pytest.fail("symlink target was opened")
    assert target.read_bytes() == b"caller-original"
