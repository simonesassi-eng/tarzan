"""The pin-review clock must never silently become a publication outage."""

from __future__ import annotations

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release", _ROOT / "scripts" / "validate_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overdue_pin_review_warns_inside_grace_then_blocks(capsys):
    module = _validator()
    manifest = json.loads((_ROOT / "tarzan/release_manifest.json").read_text(encoding="utf-8"))
    review = manifest["pin_review"]
    today = date.today()

    review["reviewed_on"] = str(today - timedelta(days=30))
    review["review_due_on"] = str(today - timedelta(days=1))
    module._validate_scope_and_review(manifest)
    assert "Pin review overdue" in capsys.readouterr().out

    review["reviewed_on"] = str(today - timedelta(days=31))
    review["review_due_on"] = str(today - timedelta(days=module._REVIEW_GRACE_DAYS + 1))
    with pytest.raises(module.ReleaseValidationError, match="overdue for review"):
        module._validate_scope_and_review(manifest)
