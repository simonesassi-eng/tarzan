"""Entry point for a newsletter send — a thin shim over ``tarzan.delivery``.

The delivery logic (input resolution, pipeline run, render, SMTP send) lives
in :mod:`tarzan.delivery` so it is importable and unit-testable, and so the
multi-tenant loop (Track B) can call it directly per tenant.

This shim exists only so the invocation path stays stable: GitHub Actions runs
``python scripts/send_newsletter.py``, with all scheduling in the Gmail Apps
Script (``scripts/apps_script/Code.gs``), which fires a ``repository_dispatch``
(event type ``send_now``) at each market slot and for on-demand "Update"
replies. See ``tarzan.delivery`` for the full env-var contract.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the tarzan package importable when invoked as a bare script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarzan.delivery import run_and_send  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


if __name__ == "__main__":
    sys.exit(run_and_send())
