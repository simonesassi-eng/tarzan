"""Synthetic publication/evidence oracles; every provider call is stubbed."""

from __future__ import annotations

import io
import json
from datetime import date

import pytest

from tarzan import runtime
from tarzan.export import ai_summary
from tarzan.runtime.ledger import Availability, RunLedger
from tarzan.runtime.session import RunSession, activate_session


@pytest.fixture
def enabled_ai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SYNTHETIC-API-KEY")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)


def _owner() -> RunSession:
    context = runtime.reset(attempt_id="synthetic-ai-publication")
    return RunSession(context, {}, RunLedger(context.attempt_id))


def _fake_native_response(monkeypatch, payload):
    """Exercise the native REST decoding without a socket or real key."""
    requests = []

    def fake_open(request, *, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    def fake_provider_call(provider, callback, *, timeout):
        assert provider == "gemini"
        return callback(timeout)

    monkeypatch.setattr(ai_summary, "urlopen", fake_open)
    monkeypatch.setattr(
        "tarzan.runtime.transport.provider_call", fake_provider_call,
    )
    return requests


@pytest.mark.parametrize("text", [None, "", " \n\t", "***\n###", "` `\n#"])
def test_unpublishable_grounded_text_is_not_available(
    monkeypatch, enabled_ai, text,
):
    calls = []

    def fake_call(system, user, *, use_search):
        calls.append(use_search)
        return text

    monkeypatch.setattr(ai_summary, "_call_gemini", fake_call)
    owner = _owner()
    with activate_session(owner):
        assert ai_summary.generate_summary(object(), object()) is None
        evidence = ai_summary.last_provider_result()

    assert calls == [True]
    assert evidence.selected_mode is None
    assert evidence.summary_availability == "UNAVAILABLE"
    assert len(evidence.attempts) == 1
    assert evidence.attempts[0].outcome == "FAILED"
    assert evidence.attempts[0].summary_availability == "UNAVAILABLE"
    assert evidence.attempts[0].selected_fallback is None
    failures = owner.ledger.failure_records()
    assert len(failures) == 1
    assert not failures[0].automatically_corrected
    assert not failures[0].closed


@pytest.mark.parametrize("text", [None, "", " \n\t", "***\n###", "` `\n#"])
def test_unpublishable_fallback_does_not_claim_a_successful_remedy(
    monkeypatch, enabled_ai, text,
):
    calls = []

    def fake_call(system, user, *, use_search):
        calls.append(use_search)
        if use_search:
            raise TimeoutError("synthetic transport failure")
        return text

    monkeypatch.setattr(ai_summary, "_call_gemini", fake_call)
    owner = _owner()
    with activate_session(owner):
        assert ai_summary.generate_summary(object(), object()) is None
        evidence = ai_summary.last_provider_result()

    assert calls == [True, False]
    assert evidence.selected_mode is None
    assert evidence.summary_availability == "UNAVAILABLE"
    assert [attempt.outcome for attempt in evidence.attempts] == ["FAILED", "FAILED"]
    failures = owner.ledger.failure_records()
    assert len(failures) == 2
    assert all(not failure.automatically_corrected for failure in failures)
    assert all(not failure.closed for failure in failures)
    assert all(not failure.remedies for failure in failures)
    assert "SIGNALS" not in repr(evidence)


@pytest.mark.parametrize("grounded", [True, False])
def test_only_sanitized_publishable_text_gets_success_evidence(
    monkeypatch, enabled_ai, grounded,
):
    calls = []

    def fake_call(system, user, *, use_search):
        calls.append(use_search)
        if use_search and not grounded:
            raise TimeoutError("synthetic transport failure")
        return "  **Markets** moved.\n- Public news item. "

    monkeypatch.setattr(ai_summary, "_call_gemini", fake_call)
    owner = _owner()
    with activate_session(owner):
        assert ai_summary.generate_summary(object(), object()) == (
            "Markets moved.\nPublic news item."
        )
        evidence = ai_summary.last_provider_result()

    assert calls == ([True] if grounded else [True, False])
    assert evidence.selected_mode == ("GROUNDED" if grounded else "NON_GROUNDED")
    assert evidence.summary_availability == ("AVAILABLE" if grounded else "DEGRADED")
    assert evidence.attempts[-1].outcome == "SUCCEEDED"
    failures = owner.ledger.failure_records()
    if grounded:
        assert not failures
    else:
        assert len(failures) == 1
        assert failures[0].automatically_corrected
        assert failures[0].closed
        assert failures[0].availability is Availability.DEGRADED


@pytest.mark.parametrize(
    "response",
    [
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {"promptFeedback": {"blockReason": "UNKNOWN-SYNTHETIC-BLOCK"}},
        *[
            {
                "candidates": [{
                    "finishReason": reason,
                    "content": {"parts": [{"text": "UNPUBLISHABLE-CANARY"}]},
                }],
            }
            for reason in (
                "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT",
                "SPII", "IMAGE_SAFETY",
            )
        ],
        {
            "candidates": [{
                "finishReason": "STOP",
                "safetyRatings": [{"blocked": True}],
                "content": {"parts": [{"text": "UNPUBLISHABLE-CANARY"}]},
            }],
        },
        {
            "promptFeedback": {"safetyRatings": [{"blocked": True}]},
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": "UNPUBLISHABLE-CANARY"}]},
            }],
        },
    ],
)
def test_native_block_is_omitted_without_non_grounded_retry(
    monkeypatch, enabled_ai, caplog, response,
):
    requests = _fake_native_response(monkeypatch, response)
    owner = _owner()
    with activate_session(owner):
        assert ai_summary.generate_summary(object(), object()) is None
        evidence = ai_summary.last_provider_result()

    assert len(requests) == 1
    assert evidence.summary_availability == "UNAVAILABLE"
    assert evidence.selected_mode is None
    assert evidence.attempts[0].outcome == "FAILED"
    rendered_evidence = repr(evidence) + repr(owner.ledger.entries) + caplog.text
    assert "UNPUBLISHABLE-CANARY" not in rendered_evidence
    assert "UNKNOWN-SYNTHETIC-BLOCK" not in rendered_evidence
    assert "SYNTHETIC-API-KEY" not in rendered_evidence


@pytest.mark.parametrize("finish_reason", ["STOP", "MAX_TOKENS"])
def test_native_unblocked_text_remains_publishable(
    monkeypatch, enabled_ai, finish_reason,
):
    response = {
        "promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"},
        "candidates": [{
            "finishReason": finish_reason,
            "safetyRatings": [{"blocked": False}],
            "content": {"parts": [{"text": "Public market news."}]},
        }],
    }
    requests = _fake_native_response(monkeypatch, response)
    assert ai_summary.generate_summary(object(), object()) == "Public market news."
    assert len(requests) == 1
    assert ai_summary.last_provider_result().summary_availability == "AVAILABLE"


@pytest.mark.parametrize(
    "response",
    [
        None, [], {"candidates": None}, {"candidates": [None]},
        {"candidates": [[]]}, {"candidates": [{"content": None}]},
        {"candidates": [{"content": {"parts": [None]}}]},
        {"candidates": [{"content": {"parts": [{"text": 42}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "reasoning", "thought": True}]}}]},
    ],
)
def test_malformed_or_thought_only_native_payload_has_no_publishable_text(response):
    assert ai_summary._extract_text(response) is None


def test_native_text_ignores_thoughts_and_preserves_public_parts():
    response = {
        "candidates": [{
            "content": {"parts": [
                {"text": "INTERNAL-THOUGHT-CANARY", "thought": True},
                {"functionCall": {"name": "unused"}},
                {"text": "Public "},
                {"text": "market news."},
            ]},
        }],
    }
    assert ai_summary._extract_text(response) == "Public market news."


@pytest.mark.parametrize("mode", ["disabled", "no-key", "point-in-time", "offline"])
def test_non_live_or_disabled_summary_performs_zero_provider_calls(
    monkeypatch, enabled_ai, mode,
):
    calls = []
    monkeypatch.setattr(
        ai_summary, "_call_gemini",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    if mode == "disabled":
        monkeypatch.setenv("TARZAN_DISABLE_AI", "yes")
    elif mode == "no-key":
        monkeypatch.delenv("GEMINI_API_KEY")
    else:
        runtime.configure(
            deterministic=mode == "offline",
            as_of=date(2025, 1, 2),
        )

    assert ai_summary.generate_summary(object(), object()) is None
    evidence = ai_summary.last_provider_result()
    assert calls == []
    assert evidence.attempts == ()
    assert evidence.selected_mode is None
    assert evidence.summary_availability == "UNAVAILABLE"


def test_summary_never_builds_or_transmits_a_portfolio_digest(
    monkeypatch, enabled_ai,
):
    def forbidden_digest(*args, **kwargs):
        pytest.fail("general news summary must not inspect portfolio facts")

    class PrivatePortfolio:
        def __getattribute__(self, name):
            pytest.fail("general news summary must not inspect private objects")

    prompts = []

    def fake_call(system, user, *, use_search):
        prompts.append((system, user, use_search))
        return "Public market news."

    monkeypatch.setattr(ai_summary, "build_digest", forbidden_digest)
    monkeypatch.setattr(ai_summary, "_call_gemini", fake_call)
    assert ai_summary.generate_summary(PrivatePortfolio(), PrivatePortfolio()) == (
        "Public market news."
    )
    assert len(prompts) == 1
    assert "SYNTHETIC-API-KEY" not in repr(prompts)
    assert "holdings" not in repr(ai_summary.last_provider_result())
