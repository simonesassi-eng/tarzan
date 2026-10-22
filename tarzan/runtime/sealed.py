"""Authenticated private snapshots using the pinned cryptography AES-GCM port.

GitHub caches/artifacts of a public repository are not private storage. Only
ciphertext may leave the runner. Native GitHub Secrets owns the dedicated key;
no SMTP/Drive credential, cookie or GitHub token is reused as a key.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import secrets
import stat
import zlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from tarzan.runtime.io_utils import atomic_write_bytes


KEY_ENV = "TARZAN_CACHE_KEY_B64"
_MAGIC = b"TARZAN-SEALED\x00\x01"
_NONCE_BYTES = 12
MAX_SEALED_BYTES = 64 * 1024 * 1024
MAX_PLAIN_BYTES = 128 * 1024 * 1024


def configured_key(*, key_file: Path | None = None) -> bytes:
    """Read a 256-bit key without accepting it in process arguments or logs."""
    encoded = (
        read_bounded_file(key_file, 1024).decode("ascii").strip()
        if key_file is not None else os.environ.get(KEY_ENV, "").strip()
    )
    try:
        key = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError):
        raise ValueError("private snapshot key is not valid base64") from None
    if len(key) != 32:
        raise ValueError("private snapshots require a dedicated 256-bit key")
    return key


def key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:24]


def owned_directory(path: Path) -> Path:
    """Reject symlinks before reading/writing caller-selected staging."""
    path = Path(path).expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("private snapshot storage cannot use symlinks")
    if path.exists() and not path.is_dir():
        raise ValueError("private snapshot storage must be a directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def read_bounded_file(path: Path, limit: int) -> bytes:
    path = Path(path)
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("private snapshot storage cannot use symlinks")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ValueError("snapshot file exceeds its bounded regular-file contract")
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError("snapshot file exceeds its byte limit")
    return content


def seal(content: bytes, key: bytes, *, purpose: str) -> bytes:
    if len(content) > MAX_PLAIN_BYTES:
        raise ValueError("private snapshot exceeds plaintext limit")
    nonce = secrets.token_bytes(_NONCE_BYTES)
    authenticated = _MAGIC + purpose.encode("ascii")
    payload = _MAGIC + nonce + AESGCM(key).encrypt(
        nonce, zlib.compress(content), authenticated,
    )
    if len(payload) > MAX_SEALED_BYTES:
        raise ValueError("private snapshot exceeds ciphertext limit")
    return payload


def unseal(payload: bytes, key: bytes, *, purpose: str) -> bytes:
    if (
        len(payload) > MAX_SEALED_BYTES
        or len(payload) < len(_MAGIC) + _NONCE_BYTES + 16
        or not payload.startswith(_MAGIC)
    ):
        raise ValueError("unsupported private snapshot envelope")
    boundary = len(_MAGIC) + _NONCE_BYTES
    try:
        # Authenticate BEFORE decompressing or parsing any attacker-supplied data.
        compressed = AESGCM(key).decrypt(
            payload[len(_MAGIC):boundary], payload[boundary:],
            _MAGIC + purpose.encode("ascii"),
        )
    except InvalidTag:
        raise ValueError("private snapshot authentication failed") from None
    decompressor = zlib.decompressobj()
    content = decompressor.decompress(compressed, MAX_PLAIN_BYTES + 1)
    if (
        len(content) > MAX_PLAIN_BYTES or decompressor.unconsumed_tail
        or not decompressor.eof or decompressor.unused_data
    ):
        raise ValueError("private snapshot compression exceeds its bounded contract")
    return content


def write_snapshot(path: Path, content: bytes, key: bytes, *, purpose: str) -> None:
    parent = owned_directory(Path(path).parent)
    target = parent / Path(path).name
    if target.is_symlink():
        raise ValueError("private snapshot destination cannot be a symlink")
    atomic_write_bytes(target, seal(content, key, purpose=purpose), fsync_dir=True)


def generate_key(path: Path) -> None:
    """Create a local recovery copy once; never overwrite an existing key."""
    path = owned_directory(Path(path).parent) / Path(path).name
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(base64.b64encode(AESGCM.generate_key(bit_length=256)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
