"""Pre-fix exploration oracle for production-readiness condition C16."""

from __future__ import annotations

from pathlib import Path



# **Validates: Requirements 2.16**
def test_c16_version_and_release_scope_have_single_positive_authorities():
    """One version authority, read by the package and the CLI.

    This used to also read ``.github/workflows/checks.yml`` and assert a release
    manifest file existed — which coupled the suite to the existence of a CI file
    and therefore made deleting that file break the delivery gate, the exact
    coupling the 2026-08-18 split was meant to remove. The version authority is
    the property; the manifest was a second copy of it.
    """
    root = Path(__file__).resolve().parents[2]
    package_init = (root / "tarzan/__init__.py").read_text(encoding="utf-8")
    main_source = (root / "tarzan/main.py").read_text(encoding="utf-8")

    assert "__version__" in package_init
    assert "from tarzan import __version__" in main_source

