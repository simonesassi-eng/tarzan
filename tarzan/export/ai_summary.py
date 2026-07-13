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
import re
from datetime import datetime
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_MODEL = "gemini-2.5-flash"
_TIMEOUT_SECONDS = 20
_MAX_OUTPUT_TOKENS = 1400
_MAX_CHARS = 1600  # generous safety cap for a ~150-word, 6-7 sentence note;
# the trim always ends on a full sentence, never mid-thought or mid-number.


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
    # Deterministic mode: the Gemini call is network-live and non-reproducible,
    # so skip it and let the caller fall back to the rule-based Signals section.
    from tarzan import runtime
    if runtime.is_deterministic():
        logger.info("Deterministic run: skipping the live AI summary.")
        return None
    try:
        digest = build_digest(metrics, config)
        language = os.environ.get("AI_SUMMARY_LANGUAGE", "English").strip() or "English"
        today_str = datetime.now().strftime("%A, %B %d, %Y")
        system, user = _system_prompt(language, today_str), _user_prompt(digest)
        # Try the grounded (Google Search) call first; if the key's tier or
        # model rejects grounding, retry once without it so the macro note
        # still appears (from model knowledge) rather than dropping all the
        # way back to the rule-based Signals.
        try:
            text = _call_gemini(system, user, use_search=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Grounded AI summary failed (%s); retrying without search.", e)
            text = _call_gemini(system, user, use_search=False)
        return _sanitize(text) if text else None
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.warning("AI summary unavailable (%s); falling back to Signals.", e)
        return None


# ---------------------------------------------------------------------------
# Benchmark-divergence note (charts section) — quantitative, always present
# ---------------------------------------------------------------------------

def divergence_note(metrics, config, benchmark_geo: Optional[str] = None) -> Optional[str]:
    """Explain WHY the portfolio is (or isn't) tracking its benchmark, over
    both the last-30-days and the since-inception windows shown in the two
    "You vs the market" charts.

    Unlike :func:`generate_summary` this is quantitative and ALWAYS returns a
    note when the divergence digest can be built: an LLM writes the prose when
    available, otherwise a deterministic rule-based note is computed from the
    same figures (so a golden/offline/no-key run still gets real quantitative
    insight). Returns None only when there is no benchmark to compare against.
    """
    digest = build_divergence_digest(metrics, config, benchmark_geo)
    if digest is None:
        return None

    # Deterministic mode or no key → the rule-based note (no network).
    from tarzan import runtime
    if runtime.is_deterministic() or not is_enabled():
        return _fallback_divergence_note(digest)

    try:
        language = os.environ.get("AI_SUMMARY_LANGUAGE", "English").strip() or "English"
        system = _divergence_system_prompt(language)
        user = _divergence_user_prompt(digest)
        # No Google-Search grounding here: this is a numbers-driven attribution
        # of the investor's own figures, not a macro-news note.
        text = _call_gemini(system, user, use_search=False)
        note = _sanitize(text) if text else None
        # Never leave the section blank: fall back to the quant note if the
        # model returned nothing usable.
        return note or _fallback_divergence_note(digest)
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.warning("Divergence note unavailable (%s); using rule-based note.", e)
        return _fallback_divergence_note(digest)


def build_divergence_digest(metrics, config,
                            benchmark_geo: Optional[str] = None) -> Optional[dict]:
    """Compact, JSON-serialisable inputs for the divergence note.

    Both chart windows' portfolio-vs-benchmark gaps (TWROR − ACWI), the CAPM
    decomposition (beta/alpha → how much of the gap is *taking more/less
    market risk* vs *selection*), allocation drift vs target, and the holdings
    that drove the gap. Pure function, no I/O. None when there is no benchmark
    series to compare against."""
    from tarzan.export._perf_series import _perf_window, _perf_full_series

    bench = benchmark_geo or _bench_name(metrics)
    win30 = _perf_window(metrics, 30, bench)
    full = _perf_full_series(metrics, bench)
    # Need a benchmark line in at least one window to say anything.
    have30 = bool(win30 and win30.get("acwi") and win30.get("twror"))
    havesi = bool(full and full.get("acwi") and full.get("twror"))
    if not (have30 or havesi):
        return None

    def _gap(w):
        # Endpoint TWROR − ACWI over the window (both cumulative from its start).
        if not (w and w.get("twror") and w.get("acwi")):
            return None
        p, b = _num(w["twror"][-1]), _num(w["acwi"][-1])
        if p is None or b is None:
            return None
        return {"portfolio_pct": p, "benchmark_pct": b, "gap_pp": round(p - b, 2)}

    risk = metrics.risk or {}
    beta = _num(risk.get("beta"))
    alpha = _num(risk.get("alpha"))

    digest: dict[str, Any] = {
        "benchmark": bench or "the benchmark",
        "window_30d": _gap(win30),
        "since_inception": _gap(full),
        "capm": _clean({"beta": beta, "alpha_pct": alpha,
                        "volatility_pct": _num(risk.get("volatility"))}),
        # CAPM-implied "expected" gap from taking beta≠1 market risk, on the
        # since-inception benchmark return: (beta − 1) × benchmark. What's left
        # of the actual gap after that is selection/allocation.
        "risk_vs_selection": _risk_vs_selection(_gap(full), beta),
        "allocation_drift": _top_drift(metrics),
        "contributors": _contributors(metrics),
    }
    return _clean(digest)


def _bench_name(m) -> Optional[str]:
    """The configured geo benchmark name (key into benchmark_histories)."""
    bh = getattr(m, "benchmark_histories", None)
    if isinstance(bh, dict) and bh:
        return next(iter(bh))
    return None


def _risk_vs_selection(si_gap: Optional[dict], beta: Optional[float]) -> Optional[dict]:
    """Split the since-inception gap into a market-risk part (beta≠1 on the
    benchmark's return) and a residual selection/allocation part."""
    if not si_gap or beta is None:
        return None
    bench = si_gap["benchmark_pct"]
    risk_part = round((beta - 1.0) * bench, 2)          # expected from beta≠1
    selection_part = round(si_gap["gap_pp"] - risk_part, 2)  # the rest
    return {"beta": beta,
            "market_risk_contribution_pp": risk_part,
            "selection_contribution_pp": selection_part}


def _top_drift(m, k: int = 3) -> list[dict]:
    """The k largest absolute allocation drifts vs target (asset-class + geo)."""
    gd = getattr(m, "goal_deltas", None)
    try:
        if gd is None or gd.empty:
            return []
        sub = gd.dropna(subset=["delta_pct"]).copy()
        sub["absd"] = sub["delta_pct"].abs()
        sub = sub.sort_values("absd", ascending=False).head(k)
        return [_clean({"bucket": r.get("category"),
                        "actual_pct": _num(r.get("actual_pct")),
                        "target_pct": _num(r.get("target_pct")),
                        "drift_pp": _num(r.get("delta_pct"))})
                for _, r in sub.iterrows()]
    except Exception:  # noqa: BLE001
        return []


def _contributors(m, k: int = 3) -> dict:
    """Top contributors / detractors to the portfolio's own return: the
    weight × gain product per holding (a first-order return-contribution proxy
    from the snapshot, no extra computation)."""
    df = getattr(m, "holdings_df", None)
    try:
        if df is None or df.empty or "gain_pct" not in df or "weight_pct" not in df:
            return {}
        d = df.dropna(subset=["gain_pct", "weight_pct"]).copy()
        if d.empty:
            return {}
        d["contrib_pp"] = d["weight_pct"] * d["gain_pct"] / 100.0

        def _row(r):
            return _clean({"name": (r.get("name") or r.get("ticker")),
                           "class": r.get("asset_class"),
                           "weight_pct": _num(r.get("weight_pct")),
                           "gain_pct": _num(r.get("gain_pct")),
                           "contrib_pp": _num(r.get("contrib_pp"))})

        top = [_row(r) for _, r in d.sort_values("contrib_pp", ascending=False).head(k).iterrows()]
        bottom = [_row(r) for _, r in d.sort_values("contrib_pp", ascending=True).head(k).iterrows()]
        return {"top": top, "bottom": bottom}
    except Exception:  # noqa: BLE001
        return {}


def _fallback_divergence_note(d: dict) -> str:
    """A deterministic, quantitative note built from the divergence digest —
    used when the LLM is unavailable (no key / deterministic / error). Same
    figures the model would be given, so the section is never trivial and
    never blank."""
    bench = d.get("benchmark", "the benchmark")
    bits: list[str] = []

    si = d.get("since_inception")
    w30 = d.get("window_30d")
    if si:
        verb = "beat" if si["gap_pp"] >= 0 else "trailed"
        bits.append(
            f"Since inception you returned {_sp(si['portfolio_pct'])} vs {bench} "
            f"{_sp(si['benchmark_pct'])} — you {verb} it by {_pp(si['gap_pp'])}.")
    rvs = d.get("risk_vs_selection")
    if rvs:
        b = rvs["beta"]
        stance = ("more" if b > 1 else "less") if b != 1 else "the same"
        bits.append(
            f"Your beta of {b:.2f} means you carry {abs(b - 1) * 100:.0f}% {stance} "
            f"market risk than {bench}; that risk stance alone accounts for "
            f"{_pp(rvs['market_risk_contribution_pp'])} of the gap, leaving "
            f"{_pp(rvs['selection_contribution_pp'])} from selection and allocation.")
    if w30:
        verb = "ahead of" if w30["gap_pp"] >= 0 else "behind"
        bits.append(
            f"Over the last 30 days you are {verb} {bench} by "
            f"{_pp(abs(w30['gap_pp']))} ({_sp(w30['portfolio_pct'])} vs "
            f"{_sp(w30['benchmark_pct'])}).")
    contrib = d.get("contributors") or {}
    bottom = (contrib.get("bottom") or [])
    top = (contrib.get("top") or [])
    if bottom:
        w = bottom[0]
        # Only frame it as a "drag" when the worst contributor actually detracts.
        if (_num(w.get("contrib_pp")) or 0) < 0:
            drag = (f"The biggest drag is {w.get('name')} ({_sp(w.get('gain_pct'))} on a "
                    f"{_num(w.get('weight_pct'))}% weight, {_pp(w.get('contrib_pp'))} of return)")
        else:
            drag = (f"Every holding is in the black; {w.get('name')} added the least "
                    f"({_pp(w.get('contrib_pp'))})")
        # Name the top contributor only when it's a different holding.
        if top and top[0].get("name") != w.get("name"):
            t = top[0]
            drag += f", while {t.get('name')} led the gains ({_pp(t.get('contrib_pp'))})"
        bits.append(drag + ".")
    drift = d.get("allocation_drift") or []
    if drift:
        big = drift[0]
        direction = "over" if (big.get("drift_pp") or 0) >= 0 else "under"
        rec = (f"Rebalancing your {big.get('bucket')} weight (currently "
               f"{direction} target by {_pp(abs(big.get('drift_pp') or 0))}) would "
               f"move your risk profile back toward {bench}.")
        bits.append(rec)
    return " ".join(b for b in bits if b and b != ".")


def _sp(v) -> str:
    """Signed percent, 1 dp (e.g. +14.5%)."""
    n = _num(v)
    return "n/a" if n is None else f"{n:+.1f}%"


def _pp(v) -> str:
    """Signed percentage points, 1 dp (e.g. -3.2pp)."""
    n = _num(v)
    return "n/a" if n is None else f"{n:+.1f}pp"


def _divergence_system_prompt(language: str) -> str:
    return (
        "You are a portfolio analyst explaining to a retail investor WHY their "
        "portfolio is or is not tracking its benchmark. You are given a JSON "
        "digest with two windows (last 30 days and since inception), each with "
        "the portfolio's cumulative return, the benchmark's, and the gap in "
        "percentage points; a CAPM split (beta, and how much of the "
        "since-inception gap comes from taking more/less market risk vs from "
        "selection/allocation); allocation drift vs target; and the holdings "
        "that contributed most and least to return.\n"
        "WRITE a tight, quantitative explanation (4-6 sentences, ~110 words, "
        "one flowing paragraph):\n"
        "1. State the since-inception gap and the 30-day gap with their exact "
        "figures, and say whether divergence is widening or narrowing.\n"
        "2. Attribute the gap: use the beta / market-risk vs selection split to "
        "say how much is 'you simply took more/less market risk' vs 'your "
        "picks and weights'. Name the specific holdings or sleeves driving it, "
        "with their contribution figures.\n"
        "3. Finish with ONE concrete, non-trivial recommendation tied to the "
        "numbers (e.g. a specific over/underweight to rebalance, or a "
        "concentration/beta observation) — actionable, not generic.\n"
        "RULES: every claim carries a number from the JSON; write percentage "
        "changes with an explicit + or - sign and gaps in 'pp'; never invent "
        "figures beyond the JSON; no preamble, no salutation, no markdown, no "
        "headings, no bullet points; do not restate the whole JSON. This is "
        "analysis and portfolio-construction insight, NOT a solicitation — "
        "phrase the recommendation as an observation about their own "
        f"allocation. Write in {language}."
    )


def _divergence_user_prompt(digest: dict) -> str:
    return (
        "Here is the divergence digest as JSON:\n\n"
        + json.dumps(digest, ensure_ascii=False, separators=(",", ":"))
        + "\n\nWrite the divergence explanation now."
    )


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
    digest: dict[str, Any] = {
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "today": datetime.now().strftime("%A, %B %d, %Y"),
    }

    # Today's major-index levels and daily moves (yfinance-style set,
    # fetched through the cached price layer — warm runs add no network),
    # so the model can cite real figures rather than vague wording.
    try:
        from tarzan.data.market_quotes import fetch_market_quotes
        mk = fetch_market_quotes()
        if mk:
            digest["markets_today"] = [
                {"name": d["name"], "level": _num(d["value"], 2),
                 "change_pct": _num(d["pct"], 2)}
                for d in mk
            ]
    except Exception:  # noqa: BLE001 — best-effort enrichment only
        pass

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

def _system_prompt(language: str, today_str: str) -> str:
    return (
        f"You are a markets commentator writing a 'market context' note "
        f"for a retail investor on {today_str}, explaining the macro backdrop "
        "behind their portfolio's recent moves. Use Google Search to ground "
        "the note in real, recent market news.\n"
        "WEAVE TOGETHER THREE TIME HORIZONS into one flowing note (about two "
        "sentences each). Do NOT label them, number them, or use the words "
        "'breaking', 'previous session' or 'weekly trend' as headings — just "
        "let the note move naturally from the newest to the broader picture:\n"
        "1. The last few hours / today: what is moving world financial markets "
        "right now — the latest headlines, pre-market or intraday moves, and "
        "any news breaking in the last few hours.\n"
        "2. The previous trading session (yesterday): what happened in the last "
        "completed session and the events that drove it.\n"
        "3. The past week's trend: the week's direction — cite the portfolio's "
        "own weekly figures from the JSON (twror_by_period_pct '1w', movers_1w) "
        "alongside the week's macro theme.\n"
        "RULES:\n"
        "- Start directly with the market content. No preamble, no salutation, "
        "no 'Here is your...' opener, no date restatement at the start.\n"
        "- Keep every point relevant to THIS portfolio's exposures (asset "
        "classes, equity geographies, top holdings and recent returns in the "
        "JSON).\n"
        "- Reference the ACTUAL day(s) by name and date (e.g. 'on Thursday the "
        "25th') and cite CONCRETE figures — real index levels and percentage "
        "moves (use the 'markets_today' levels in the JSON plus what you find "
        "in search).\n"
        "- Every market move you mention MUST carry a specific number (level "
        "and/or % change), and write every percentage CHANGE with an explicit "
        "leading + or - sign (e.g. +0.81%, -0.6%). NEVER use vague qualifiers "
        "such as 'slight', 'mixed', 'somewhat', 'a bit', 'modest', 'broadly' "
        "or 'edged' without an attached figure.\n"
        "- Connect the macro drivers (US / Europe / emerging-market equities, "
        "gold, government-bond yields and rates, EUR/USD) to why the portfolio "
        "moved the way the JSON shows.\n"
        "- Refer to real, recent events (rate decisions, inflation prints, "
        "earnings, geopolitics) but NEVER invent figures, quotes or dates; if "
        "unsure, omit that point rather than guessing.\n"
        "- Write 6 to 8 sentences (roughly two per horizon), about 150 words "
        "(never more than 180), as a single flowing paragraph. ALWAYS finish "
        "your final sentence — never stop mid-thought. No markdown, no bullet "
        "points, no headings, no labels. No predictions, no recommendations, "
        "no personalized investment advice.\n"
        f"- Write in {language}."
    )


def _user_prompt(digest: dict) -> str:
    return (
        "Here is the investor's portfolio context as JSON (exposures, recent "
        "returns and today's index levels in 'markets_today'):\n\n"
        + json.dumps(digest, ensure_ascii=False, separators=(",", ":"))
        + "\n\nSearch for the latest market news — covering the last few "
        "hours, the previous trading session, and the past week's trend — and "
        "write the market-context note now."
    )


# ---------------------------------------------------------------------------
# Gemini REST call (urllib — no extra dependency)
# ---------------------------------------------------------------------------

def _call_gemini(system_prompt: str, user_prompt: str, use_search: bool = True) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    url = _GEMINI_ENDPOINT.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
            "topP": 0.9,
            # Disable "thinking": reasoning tokens count against
            # maxOutputTokens and would otherwise truncate this short note to
            # empty. Search grounding still works without thinking.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if use_search:
        # Google Search grounding: pulls the last 24-48h of market news so
        # the context note reflects what actually moved markets. Requires a
        # model/tier that supports it (Gemini 2.x); rejected requests fall
        # back to a non-grounded call by the caller.
        payload["tools"] = [{"google_search": {}}]
    req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-goog-api-key", api_key)
    try:
        with urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        # Surface the API's actual error body — the single most useful clue
        # for why the summary fell back (bad key, grounding not enabled for
        # the tier, quota, unknown field, ...).
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:600]
        except Exception:  # noqa: BLE001
            pass
        logger.warning("Gemini HTTP %s%s: %s", e.code,
                       " [grounded]" if use_search else "", body)
        raise
    text = _extract_text(data)
    if not text:
        # No text despite a 200: log the finish reason so a truncation or
        # safety block is visible rather than silently becoming Signals.
        try:
            fr = data["candidates"][0].get("finishReason")
        except (KeyError, IndexError, TypeError):
            fr = None
        logger.warning("Gemini returned no text%s (finishReason=%s).",
                       " [grounded]" if use_search else "", fr)
    return text


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
        cut = cleaned[:_MAX_CHARS]
        # Prefer ending on a complete sentence so it never looks truncated.
        # A sentence end is .!? (optionally closing quote/paren) FOLLOWED BY
        # whitespace — requiring the trailing space means a decimal point
        # ("3.50%") or an in-word dot is never mistaken for a sentence end,
        # which is what used to leave the note cut mid-number.
        matches = list(re.finditer(r'[.!?]["\')\]]?(?=\s)', cut))
        end = matches[-1].end() if matches else -1
        if end >= _MAX_CHARS * 0.5:
            cleaned = cut[:end].rstrip()
        else:
            cleaned = cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"
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
