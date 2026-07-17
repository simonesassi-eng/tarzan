"""Normalize a generated hash lock to one logical requirement per line.

`uv pip compile --generate-hashes` wraps long requirements with backslash
continuations. Tarzan keeps each complete pinned requirement on one physical
line so the release scope checker can reject every unpinned non-comment line
without implementing a second requirements-file parser.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def normalize_lock(text: str) -> str:
    """Return *text* with continuation blocks collapsed atomically by entry."""
    output: list[str] = []
    current: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if current:
                raise ValueError("unterminated requirement continuation")
            if output and output[-1] != "":
                output.append("")
            continue
        if stripped.startswith("#"):
            if current:
                raise ValueError("comment inside requirement continuation")
            output.append(raw_line.rstrip())
            continue

        continued = stripped.endswith("\\")
        fragment = stripped[:-1].rstrip() if continued else stripped
        current.append(fragment)
        if continued:
            continue

        requirement = " ".join(current)
        current.clear()
        if "==" not in requirement or "--hash=sha256:" not in requirement:
            raise ValueError(f"requirement is not exactly pinned and hashed: {requirement}")
        output.append(requirement)

    if current:
        raise ValueError("unterminated requirement continuation")
    return "\n".join(output).rstrip() + "\n"


def write_normalized(source: Path, destination: Path) -> None:
    normalized = normalize_lock(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    write_normalized(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
