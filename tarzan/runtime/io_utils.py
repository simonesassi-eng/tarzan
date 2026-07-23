"""Low-level serialization + durable-write primitives (stdlib only).

Two idioms were copy-pasted across the runtime, data and delivery layers and
had already started to drift (an ``allow_nan`` here, a dir-fsync there). They
live here once so the correctness guards — no NaN/Inf tokens in canonical
JSON, tmp-file + fsync + atomic rename on every publish — stay uniform.

Depends on nothing in ``tarzan``, so any layer may import it freely.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def canonical_json_bytes(value, *, ascii_only: bool = False, default=None) -> bytes:
    """Deterministic compact JSON as UTF-8 bytes.

    Sorted keys and ``(",", ":")`` separators make the output byte-stable for
    hashing/checksums; ``allow_nan=False`` rejects NaN/Infinity so a non-finite
    metric raises here instead of emitting a token that breaks strict parsers.

    ``ascii_only`` mirrors ``ensure_ascii`` (True escapes non-ASCII — used where
    the bytes feed an ID/hash that must be transport-safe). ``default`` is passed
    straight to :func:`json.dumps` (pass ``str`` to stringify stray objects;
    omit it to keep the default TypeError-on-unserializable behaviour).
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ascii_only,
        allow_nan=False,
        default=default,
    ).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes, *, fsync_dir: bool = False) -> None:
    """Write ``data`` to ``path`` atomically: same-dir tmp file, fsync, rename.

    The rename is atomic on POSIX, so a reader sees either the old file or the
    complete new one — never a truncated write. The tmp file is cleaned up on
    any failure. ``path.parent`` must already exist (callers own directory
    creation, as their mkdir needs differ).

    ``fsync_dir`` additionally fsyncs the parent directory after the rename so
    the new name survives a crash — needed for a cache that must not resurrect a
    stale entry, skippable for outputs regenerated every run.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if fsync_dir:
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
    finally:
        if temporary.exists():
            temporary.unlink()
