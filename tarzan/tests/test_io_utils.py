"""Durability + byte-stability guards for the shared io primitives."""

from __future__ import annotations

import json

import pytest

from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes


def test_canonical_json_is_sorted_compact_and_stable():
    a = canonical_json_bytes({"b": 1, "a": 2})
    b = canonical_json_bytes({"a": 2, "b": 1})
    assert a == b == b'{"a":2,"b":1}'  # key order irrelevant, no spaces


def test_canonical_json_ascii_flag_matches_json_dumps():
    value = {"name": "Éire"}
    assert canonical_json_bytes(value, ascii_only=True) == json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    assert canonical_json_bytes(value, ascii_only=False) == json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def test_canonical_json_rejects_non_finite():
    # A NaN token would break strict downstream parsers — must raise, not emit.
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})


def test_canonical_json_default_stringifies():
    from datetime import date
    assert canonical_json_bytes({"d": date(2026, 7, 23)}, default=str) == b'{"d":"2026-07-23"}'


def test_atomic_write_replaces_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "out.json"
    target.write_bytes(b"OLD")
    atomic_write_bytes(target, b"NEW", fsync_dir=True)
    assert target.read_bytes() == b"NEW"
    # No leftover ".out.json.*.tmp" scratch files.
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
