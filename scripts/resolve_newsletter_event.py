"""Resolve a closed event context from native Actions metadata, without secrets in logs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarzan.contracts.events import resolve_workflow_event  # noqa: E402
from tarzan import log_redaction  # noqa: E402


def resolve(environment):
    repository, run_id = environment.get("GITHUB_REPOSITORY", ""), environment.get("GITHUB_RUN_ID", "")
    token = environment.get("GH_TOKEN", "")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or not re.fullmatch(r"[0-9]{1,24}", run_id) or not token
    ):
        raise ValueError("native Actions metadata/authorization is unavailable")
    request = Request(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=10) as response:
        body = response.read(512 * 1024 + 1)
    if len(body) > 512 * 1024:
        raise ValueError("native Actions response exceeds limit")
    metadata = json.loads(body)
    if (
        not isinstance(metadata, dict) or str(metadata.get("id")) != run_id
        or (metadata.get("repository") or {}).get("full_name", "").lower() != repository.lower()
    ):
        raise ValueError("native Actions run identity mismatch")
    event_path = Path(environment["GITHUB_EVENT_PATH"])
    if event_path.stat().st_size > 1024 * 1024:
        raise ValueError("native Actions event exceeds limit")
    event = json.loads(event_path.read_bytes())
    return resolve_workflow_event(
        event, event_name=environment.get("GITHUB_EVENT_NAME"), run_id=run_id,
        run_created_at=metadata.get("created_at"),
    )


def main():
    logging.basicConfig(level=logging.WARNING)
    log_redaction.install()
    try:
        outputs = resolve(os.environ)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            for key, value in outputs.items():
                handle.write(f"{key}={value}\n")
    except Exception:
        # No traceback or native response body can reveal credentials/context.
        print("Newsletter event context unavailable; delivery is blocked.", file=sys.stderr)
        return 1
    print("Native newsletter event context verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
