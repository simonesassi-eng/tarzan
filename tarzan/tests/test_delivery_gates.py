"""The delivery path must only be able to fail for delivery reasons.

The 2026-08-18 morning digest was never rendered because a self-imposed
"pins reviewed on/due on" date in the release manifest had passed the day
before. Nothing about the run was broken; a calendar was. These assertions
keep clocks and declaration audits out of the send path for good.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def test_delivery_workflow_gates_behaviour_not_declarations():
    workflow = _text(".github/workflows/newsletter.yml")

    assert "validate_release.py" not in workflow, (
        "the declaration audit is back in the delivery path: a stale manifest, "
        "pin, or README claim can once again withhold a digest"
    )
    assert "python -m pytest tarzan/tests -q" in workflow, (
        "publication is no longer gated on the test suite"
    )
    assert re.search(r"(?m)^    needs:\s*validate\s*$", workflow), (
        "publication no longer depends on validation"
    )


def test_release_declarations_hold_no_clock():
    validator = _text("scripts/validate_release.py")
    manifest = json.loads(_text("tarzan/release_manifest.json"))

    assert "date.today" not in validator and "datetime.now" not in validator, (
        "the release validator reads the wall clock again, so it can start "
        "failing on a day when nothing changed"
    )
    assert "pin_review" not in manifest, (
        "the expiring pin-review window is back in the release manifest"
    )
