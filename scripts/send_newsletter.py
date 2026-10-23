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
import time
from pathlib import Path

_IMPORTS_STARTED = time.monotonic()

# Make the tarzan package importable when invoked as a bare script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarzan import log_redaction  # noqa: E402
from tarzan.delivery import run_and_send  # noqa: E402

from tarzan.runtime.logging import setup_logging  # noqa: E402
from tarzan.runtime.diagnostics import capture_configured  # noqa: E402
from tarzan.runtime.session import last_run_result  # noqa: E402

setup_logging()
log_redaction.install()


def main():
    previous_result = last_run_result()
    started = time.monotonic()
    exit_code = 1
    error_type = None
    try:
        exit_code = run_and_send()
    except Exception as error:
        error_type = type(error).__name__
        logging.getLogger("tarzan.delivery").exception(
            "Newsletter entry point failed.",
            extra={"public_code": "NEWSLETTER_ENTRYPOINT_FAILED"},
        )
    finally:
        result = last_run_result()
        duration_seconds = time.monotonic() - started
        capture_configured(
            result if result is not previous_result else None,
            operation="send", exit_code=exit_code,
            duration_seconds=duration_seconds,
            entrypoint_error_type=error_type,
            operational_timings={
                "imports_seconds": _IMPORTS_SECONDS,
                "imports_and_entrypoint_seconds": _IMPORTS_SECONDS + duration_seconds,
            },
        )
    return exit_code


_IMPORTS_SECONDS = time.monotonic() - _IMPORTS_STARTED


if __name__ == "__main__":
    sys.exit(main())
