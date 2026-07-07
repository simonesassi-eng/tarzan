"""Live diagnostic for the AI market-context summary (Gemini).

Runs the REAL Gemini call end-to-end and prints everything useful to explain
why the newsletter's AI block did or didn't appear: key presence (masked),
model, HTTP status + error body, finishReason, token usage, grounding
metadata, and the final sanitized text. Both the grounded (Google Search) and
non-grounded calls are exercised so we can see exactly which path fails.

Usage (the key is read from the environment; it is never printed or stored):

    GEMINI_API_KEY=... python scripts/diagnose_ai_summary.py

Optional: GEMINI_MODEL, AI_SUMMARY_LANGUAGE. This script makes real API calls
and therefore consumes free-tier quota — run it sparingly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tarzan.export import ai_summary as ais


def _mask(key: str) -> str:
    if not key:
        return "(empty)"
    return f"{key[:4]}…{key[-4:]} (len {len(key)})"


def _raw_call(system_prompt: str, user_prompt: str, use_search: bool) -> None:
    """Make one raw Gemini call and dump the full diagnostic picture."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", ais._DEFAULT_MODEL).strip() or ais._DEFAULT_MODEL
    url = ais._GEMINI_ENDPOINT.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": ais._MAX_OUTPUT_TOKENS,
            "topP": 0.9,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    tag = "GROUNDED (google_search)" if use_search else "PLAIN (no search)"
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    print(f"POST {url}")
    req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-goog-api-key", api_key)
    try:
        with urlopen(req, timeout=ais._TIMEOUT_SECONDS) as resp:
            status = resp.status
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        print(f"HTTP ERROR {e.code} {e.reason}")
        print("BODY:", body[:1500])
        return
    except URLError as e:
        print("NETWORK ERROR:", e)
        return

    print(f"HTTP {status}")
    # Finish reason + token usage.
    try:
        cand = data["candidates"][0]
        print("finishReason:", cand.get("finishReason"))
    except (KeyError, IndexError, TypeError):
        cand = None
        print("finishReason: (no candidates)")
    usage = data.get("usageMetadata", {})
    if usage:
        print("usage:", {k: usage.get(k) for k in (
            "promptTokenCount", "candidatesTokenCount",
            "thoughtsTokenCount", "totalTokenCount")})
    # Grounding metadata (present only when search actually fired).
    if cand:
        gm = cand.get("groundingMetadata")
        if gm:
            queries = gm.get("webSearchQueries")
            chunks = gm.get("groundingChunks") or []
            print("grounding: searched =", queries, "| sources =", len(chunks))
        else:
            print("grounding: none (model did not use search)")
        # Safety blocks.
        if cand.get("finishReason") == "SAFETY":
            print("safetyRatings:", cand.get("safetyRatings"))
    # Extracted text.
    text = ais._extract_text(data)
    if text:
        print(f"TEXT ({len(text)} chars):\n{text}")
    else:
        print("TEXT: <none>")
        # Dump top-level keys so an unexpected shape is visible.
        print("response keys:", list(data.keys()))


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    print("GEMINI_API_KEY:", _mask(key))
    print("GEMINI_MODEL:", os.environ.get("GEMINI_MODEL", ais._DEFAULT_MODEL))
    print("AI_SUMMARY_LANGUAGE:", os.environ.get("AI_SUMMARY_LANGUAGE", "English"))
    print("TARZAN_DISABLE_AI:", os.environ.get("TARZAN_DISABLE_AI", "(unset)"))
    print("is_enabled():", ais.is_enabled())
    if not ais.is_enabled():
        print("\nFeature is OFF locally — export GEMINI_API_KEY to run the live call.")
        return 1

    # Build the real digest from the user's actual inputs.
    from tarzan.orchestrator import run
    print("\nBuilding metrics from input/ ...")
    m, c = run(
        config_source="input/targets.csv",
        orders_source="input/order_list.csv",
        targets_per_holding_source="input/targets_per_holding.csv",
    )
    digest = ais.build_digest(m, c)
    language = os.environ.get("AI_SUMMARY_LANGUAGE", "English").strip() or "English"
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    system = ais._system_prompt(language, today_str)
    user = ais._user_prompt(digest)
    print(f"digest chars = {len(json.dumps(digest, ensure_ascii=False))} | "
          f"prompt chars = {len(system) + len(user)}")

    # Exercise both paths, exactly like generate_summary would.
    _raw_call(system, user, use_search=True)
    _raw_call(system, user, use_search=False)

    # End-to-end through the real entry point (with sanitize + fallback logic).
    print(f"\n{'='*70}\nEND-TO-END generate_summary()\n{'='*70}")
    out = ais.generate_summary(m, c)
    print("RESULT:", repr(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
