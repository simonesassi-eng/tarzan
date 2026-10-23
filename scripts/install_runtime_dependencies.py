"""Install the runtime projection of the one immutable dependency lock.

Checks keeps the complete lock, including test tools. Publication installs the
same exact lines/hashes except five test-only packages; no second version or
hash authority, executable environment cache, or skipped integrity check.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import subprocess
import sys
import tempfile


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from tarzan import log_redaction  # noqa: E402 — stdlib-only, before dependency installation

TEST_ONLY = frozenset({"hypothesis", "pytest", "iniconfig", "pluggy", "sortedcontainers"})
_REQUIREMENT = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9_.-]*)==[A-Za-z0-9][A-Za-z0-9.!+_-]*"
    r"(?:\s+--hash=sha256:[0-9a-f]{64})+\Z"
)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_requirements(lock: str) -> str:
    """Keep all runtime pins/hashes byte-equivalent; fail closed on loose lines."""
    selected = []
    seen = set()
    for line in lock.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            raise ValueError("runtime installation needs the normalized immutable lock")
        name = canonical_name(match.group(1))
        if name in seen:
            raise ValueError("duplicate dependency in the immutable lock")
        seen.add(name)
        if name not in TEST_ONLY:
            selected.append(line)
    if not selected:
        raise ValueError("runtime dependency projection is empty")
    return "\n".join(selected) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    log_redaction.install()
    try:
        requirements = runtime_requirements((_ROOT / "requirements.txt").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        print("Immutable runtime dependency lock unavailable; installation blocked.", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="tarzan-runtime-dependencies-") as directory:
        path = Path(directory) / "requirements.txt"
        path.write_text(requirements, encoding="utf-8")
        path.chmod(0o600)
        # pip's resolver still checks every dependency. If a future runtime
        # package needs an omitted tool, --require-hashes rejects the unpinned
        # implicit install instead of silently accepting an unchecked package.
        return subprocess.run([
            sys.executable, "-m", "pip", "install", "--require-hashes",
            "--retries", "5", "--timeout", "30", "-r", str(path),
        ], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
