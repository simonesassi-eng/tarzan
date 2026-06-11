"""AI-generated portfolio summary (free, best-effort, never fatal).

Replaces the rule-based "Signals" block with a short 3-4 sentence narrative
written by an LLM from the *entire* metrics dataset Tarzan computes. The
model only paraphrases figures it is given — it never invents numbers and
never produces personalized financial advice.

Design constraints (in priority order):
  * **Free.** Uses Google Gemini's genuinely-free tier (Flash model, no
    credit card, 1M-token context so the whole dataset fits). Anthropic /
    OpenAI are paid, so they are not the default.
  * **Never fatal.** Any problem (no API key, network error, rate limit,
    bad response) returns None, and the caller falls back to the rule-based
    Signals section. The newsletter send must never fail because of this.
  * **No tokens spent in tests.** The network call only fires when
    ``GEMINI_API_KEY`` is set and ``TARZAN_DISABLE_AI`` is not. Tests leave
    the key unset (and a fixture disables it), so they exercise only the
    deterministic digest builder and the fallback path.

Configuration (environment):
  * ``GEMINI_API_KEY``       — enables the feature (a free key from
    https://aistudio.google.com/apikey). Absent → feature off.
  * ``GEMINI_MODEL``         — model id (default ``gemini-2.5-flash``).
  * ``AI_SUMMARY_LANGUAGE``  — output language (default ``English`` to match
    the newsletter).
  * ``TARZAN_DISABLE_AI``    — set to 1/true to force the feature off.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_MODEL = "gemini-2.5-flash"
_TIMEOUT_SECONDS = 12
_MAX_OUTPUT_TOKENS = 512
_MAX_CHARS = 700  # hard cap on the rendered summary length


def is_enabled() -> bool:
    """True only when an API key is present and the feature is not disabled."""
    if os.environ.get("TARZAN_DISABLE_AI", "").strip().lower() in ("1", "true", "yes"):
        return False
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_summary(metrics, config) -> Optional[str]:
    """Return a short AI portfolio summary, or None to fall back to Signals.

    Best-effort: returns None on any error so the caller degrades to the
    rule-based section. Never raises.
    """
    if not is_enabled():
        return None
    try:
        digest = build_digest(metrics, config)
        language = os.environ.get("AI_SUMMARY_LANGUAGE", "English").strip() or "English"
        text = _call_gemini(_system_prompt(language), _user_prompt(digest))
        return _sanitize(text) if text else None
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.warning("AI summary unavailable (%s); falling back to Signals.", e)
        return None


# ---------------------------------------------------------------------------
# Digest: compact, comprehensive snapshot of the whole metrics dataset
# ---------------------------------------------------------------------------

def build_digest(metrics, config) -> dict:
    """Build a compact JSON-serializable digest of the *entire* dataset.

    Comprehensive (snapshot, per-period TWROR, risk, allocations vs targets,
    geography, every holding, movers, benchmarks, rebalancing actions,
    income) but rounded and trimmed so it stays token-light. Pure function,
    no I/O — safe to unit-test.
    """
    m = metrics
    digest: dict[str, Any] = {"as_of": datetime.now().strftime("%Y-%m-%d")}

    # Snapshot + lifetime figures.
    cost = 0.0
    try:
        if m.holdings_df is not None and not m.holdings_df.empty:
            cost = float(m.holdings_df["cost_basis_eur"].sum())
    except Exception:  # noqa: BLE001
        cost = 0.0
    digest["snapshot"] = _clean({
        "inception": getattr(m, "inception_date", None),
        "value_eur": _num(m.total_value, 0),
        "invested_eur": _num(m.invested_value, 0),
        "cash_eur": _num(m.cash_value, 0),
    })
    digest["since_inception"] = _clean({
        "total_pnl_eur": _num(getattr(m, "pnl_eur", None), 0),
        "total_pnl_pct": _num(getattr(m, "pnl_pct", None)),
        "unrealized_pnl_eur": _num(m.total_value - cost, 0) if cost else None,
        "twror_cumulative_pct": _num(getattr(m, "twror_pct", None)),
        "twror_annualized_pct": _num(getattr(m, "twror_annualized_pct", None)),
        "xirr_pct": _num(getattr(m, "xirr_pct", None)),
        "market_data_coverage_pct": _num(getattr(m, "returns_coverage_pct", None)),
    })

    # Per-period TWROR (short/medium/long-term trend).
    perf = m.performance_full or {}
    periods = ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y"]
    digest["twror_by_period_pct"] = _clean({p: _num(perf.get(p)) for p in periods})

    # Risk.
    risk = m.risk or {}
    digest["risk"] = _clean({
        "volatility_pct": _num(risk.get("volatility")),
        "sharpe": _num(risk.get("sharpe")),
        "sortino": _num(risk.get("sortino")),
        "max_drawdown_pct": _num(risk.get("max_drawdown")),
        "beta": _num(risk.get("beta")),
        "alpha": _num(risk.get("alpha")),
        "var_95_pct": _num(risk.get("var_95")),
    })

    # Allocation by class vs target, and equity geography vs target/ACWI.
    digest["allocation_by_class"] = _allocation_rows(m, "asset_class")
    digest["equity_geography"] = _geo_rows(m)

    # Every holding (compact).
    digest["holdings"] = _holdings_rows(m)

    # Movers this week.
    digest["movers_1w"] = _movers(m)

    # Benchmarks (per-period returns + alpha/beta).
    digest["benchmarks"] = _benchmarks(m)

    # Rebalancing status + the optimizer's concrete actions (to be restated,
    # not invented).
    digest["rebalancing"] = _rebalancing(m)

    # Income / costs.
    digest["income"] = _clean({
        "weighted_yield_pct": _num(getattr(m, "weighted_yield", None)),
        "avg_ter_pct": _num(getattr(m, "avg_ter", None)),
    })

    return _clean(digest)


def _allocation_rows(m, type_filter: str) -> list[dict]:
    gd = getattr(m, "goal_deltas", None)
    rows: list[dict] = []
    try:
        if gd is not None and not gd.empty:
            sub = gd[gd["type"] == type_filter]
            for _, r in sub.iterrows():
                rows.append(_clean({
                    "category": r.get("category"),
                    "actual_pct": _num(r.get("actual_pct")),
                    "target_pct": _num(r.get("target_pct")),
                    "drift_pct": _num(r.get("delta_pct")),
                }))
    except Exception:  # noqa: BLE001
        return []
    return rows


def _geo_rows(m) -> list[dict]:
    gd = getattr(m, "goal_deltas", None)
    rows: list[dict] = []
    try:
        if gd is not None and not gd.empty:
            sub = gd[gd["type"].astype(str).str.startswith("geography")]
            for _, r in sub.iterrows():
                rows.append(_clean({
                    "region": r.get("category"),
                    "actual_pct": _num(r.get("actual_pct")),
                    "target_pct": _num(r.get("target_pct")),
                    "drift_pct": _num(r.get("delta_pct")),
                }))
    except Exception:  # noqa: BLE001
        return []
    return rows


def _holdings_rows(m) -> list[dict]:
    df = getattr(m, "holdings_df", None)
    rows: list[dict] = []
    try:
        if df is not None and not df.empty:
            for _, h in df.iterrows():
                rows.append(_clean({
                    "name": (h.get("name") or h.get("ticker")),
                    "class": h.get("asset_class"),
                    "weight_pct": _num(h.get("weight_pct")),
                    "gain_pct": _num(h.get("gain_pct")),
                    "value_eur": _num(h.get("current_value"), 0),
                }))
    except Exception:  # noqa: BLE001
        return []
    return rows


def _movers(m) -> dict:
    hp = getattr(m, "holding_performance", None)
    try:
        if hp is None or hp.empty or "1w" not in hp.columns:
            return {}
        sub = hp.copy()
        if "type" in sub.columns:
            sub = sub[sub["type"].astype(str).str.contains("portfolio", case=False, na=False)]
        sub = sub.dropna(subset=["1w"])
        if sub.empty:
            return {}
        sub = sub.sort_values("1w", ascending=False)

        def _row(r):
            return _clean({"name": r.get("name") or r.get("ticker"), "ret_1w_pct": _num(r.get("1w"))})

        best = [_row(r) for _, r in sub.head(3).iterrows()]
        worst = [_row(r) for _, r in sub.tail(3).iterrows()]
        return {"best": best, "worst": worst}
    except Exception:  # noqa: BLE001
        return {}


def _benchmarks(m) -> list[dict]:
    bc = getattr(m, "benchmark_comparison", None)
    rows: list[dict] = []
    try:
        if bc is None or bc.empty:
            return []
        keep = [c for c in ("benchmark", "1m", "3m", "ytd", "1y", "cagr", "beta", "alpha")
                if c in bc.columns]
        for _, r in bc.iterrows():
            row = {}
            for c in keep:
                row[c] = r.get(c) if c == "benchmark" else _num(r.get(c))
            rows.append(_clean(row))
    except Exception:  # noqa: BLE001
        return []
    return rows


def _rebalancing(m) -> dict:
    out: dict[str, Any] = {}
    verifs = getattr(m, "rebalancing_verifications", None)
    if verifs:
        if any(v.get("no_solution") for v in verifs):
            out["status"] = "infeasible at configured tolerance"
        elif any(v.get("relaxed") for v in verifs):
            out["status"] = "feasible only at a relaxed tolerance"
        else:
            out["status"] = "feasible"
    sugg = getattr(m, "rebalancing_suggestions", None) or []
    actions: list[dict] = []
    for s in sugg[:12]:
        if isinstance(s, dict):
            actions.append({
                k: (_num(v) if isinstance(v, float) else v)
                for k, v in s.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            })
    out["n_actions"] = len(sugg)
    out["actions"] = actions
    return _clean(out)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _system_prompt(language: str) -> str:
    return (
        "You are a portfolio analyst writing a brief digest for the investor "
        "who owns this portfolio. You are given a JSON snapshot of their "
        "portfolio metrics.\n"
        "RULES:\n"
        "- Use ONLY the figures in the JSON. Never invent or estimate numbers.\n"
        "- Write 3 to 4 sentences of flowing prose. No markdown, no bullet "
        "points, no headings.\n"
        "- Cover, concisely: recent performance (this week and 1-month), the "
        "short/medium/long-term trend (1m vs 3m/6m vs 1y/since inception, "
        "using TWROR), one notable risk or allocation observation, and the "
        "recommended next action.\n"
        "- For the action, ONLY restate the rebalancing actions provided in "
        "the JSON. If there are none, say the allocation is on target. Never "
        "invent trades or give personalized investment advice, predictions, "
        "or guarantees.\n"
        "- Returns are time-weighted (TWROR) unless a figure is labelled "
        "otherwise. Be precise but plain-spoken.\n"
        f"- Write in {language}. Keep it under 85 words."
    )


def _user_prompt(digest: dict) -> str:
    return (
        "Here is the portfolio metrics snapshot as JSON:\n\n"
        + json.dumps(digest, ensure_ascii=False, separators=(",", ":"))
        + "\n\nWrite the summary now."
    )


# ---------------------------------------------------------------------------
# Gemini REST call (urllib — no extra dependency)
# ---------------------------------------------------------------------------

def _call_gemini(system_prompt: str, user_prompt: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    url = _GEMINI_ENDPOINT.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
            "topP": 0.9,
            # Gemini 2.5 Flash is a "thinking" model: reasoning tokens count
            # against maxOutputTokens and would otherwise truncate this short
            # summary mid-sentence. Thinking adds nothing for paraphrasing
            # given figures, so disable it (and keep latency/cost minimal).
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-goog-api-key", api_key)
    with urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _extract_text(data)


def _extract_text(data: dict) -> Optional[str]:
    """Pull the generated text out of a Gemini generateContent response."""
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        return text or None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Output hygiene
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> Optional[str]:
    """Strip markdown noise, collapse whitespace, and cap the length."""
    if not text:
        return None
    cleaned = text.strip()
    # Drop common markdown artifacts so it renders as plain prose.
    for token in ("**", "*", "`", "#", "> "):
        cleaned = cleaned.replace(token, "")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    if len(cleaned) > _MAX_CHARS:
        cleaned = cleaned[: _MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return cleaned


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _num(x, decimals: int = 2):
    """Round to a JSON-friendly number, or None for NaN/None/non-numeric."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return round(f, decimals)


def _clean(obj):
    """Recursively drop None values (and empty containers) to keep the
    digest compact and unambiguous for the model."""
    if isinstance(obj, dict):
        out = {k: _clean(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v is not None and v != [] and v != {}}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj
