"""Pre-fix exploration oracles for production-readiness conditions C16-C17."""

from __future__ import annotations

from pathlib import Path

from tarzan import runtime
from tarzan.export import ai_summary


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


# **Validates: Requirements 2.17**
def test_c17_gemini_attempts_are_structured_without_secret_or_payload_evidence(
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "C17-SECRET-CANARY")
    monkeypatch.setenv("C17_UNRELATED_MACHINE_DATA", "DO-NOT-TRANSMIT")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    runtime.reset()

    captured: list[tuple[str, str, bool]] = []

    def fake_call(system: str, user: str, *, use_search: bool) -> str:
        captured.append((system, user, use_search))
        if use_search:
            raise TimeoutError("grounded attempt timed out")
        return "Grounded fallback summary"

    monkeypatch.setattr(
        ai_summary,
        "build_digest",
        lambda metrics, config: {
            "holdings": [{"ticker": "DOMAIN", "weight_pct": 100.0}],
            "allocations": {"Equities": 100.0},
            "plans": [],
        },
    )
    monkeypatch.setattr(ai_summary, "_call_gemini", fake_call)

    result = ai_summary.generate_summary(object(), object())
    evidence_getter = getattr(ai_summary, "last_provider_result", None)
    evidence = evidence_getter() if callable(evidence_getter) else None
    transmitted = "\n".join(system + "\n" + user for system, user, _ in captured)

    assert result == "Grounded fallback summary"
    assert "C17-SECRET-CANARY" not in transmitted
    assert "DO-NOT-TRANSMIT" not in transmitted
    assert evidence is not None, "Gemini attempts exist only in logs, not structured run evidence"
    attempts = list(getattr(evidence, "attempts", ()))
    assert [getattr(item, "mode", None) for item in attempts] == [
        "GROUNDED",
        "NON_GROUNDED",
    ]
    assert [getattr(item, "outcome", None) for item in attempts] == [
        "FAILED",
        "SUCCEEDED",
    ]
    serialized = repr(evidence)
    assert "C17-SECRET-CANARY" not in serialized
    assert "holdings" not in serialized and "DOMAIN" not in serialized
