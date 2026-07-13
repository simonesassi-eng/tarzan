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

    # Prefer the engine's AUTHORITATIVE figures for the stated numbers so the
    # note quotes the same portfolio/benchmark returns the tables and charts
    # show; fall back to the (canonically-anchored) chart line endpoints only
    # when an authoritative field is unavailable.
    pf = metrics.performance_full or {}
    auth = {
        "port_30d": _num(pf.get("1m")),
        "bench_30d": _num(_bench_period(metrics, bench, "1m")),
        "port_si": _num(getattr(metrics, "twror_pct", None)),
        # No single authoritative "benchmark since inception" scalar exists;
        # the canonically-anchored line endpoint is the source for that leg.
    }

    def _gap(w, port_auth=None, bench_auth=None):
        # TWROR − benchmark over the window. Use authoritative scalars when
        # given, else the chart line endpoints (both anchored the same way).
        if not (w and w.get("twror") and w.get("acwi")):
            return None
        p = port_auth if port_auth is not None else _num(w["twror"][-1])
        b = bench_auth if bench_auth is not None else _num(w["acwi"][-1])
        if p is None or b is None:
            return None
        return {"portfolio_pct": p, "benchmark_pct": b, "gap_pp": round(p - b, 2)}

    gap_30d = _gap(win30, auth["port_30d"], auth["bench_30d"])
    gap_si = _gap(full, auth["port_si"], None)

    risk = metrics.risk or {}
    alpha = _num(risk.get("alpha"))
    # Two DISTINCT, precisely-defined betas, both from the engine's one
    # _compute_beta_alpha primitive against the same benchmark (cfg
    # .benchmark_beta) — so they form a truthful, apples-to-apples trend:
    #   * realized: regression of the REAL order-derived NAV over the actual
    #     holding period (m.risk["beta"]). Since it is measured on the very
    #     returns being attributed, it — not the current-mix beta — is the
    #     correct multiplier for the since-inception gap decomposition.
    #   * current: today's weights held constant (historical_risk portfolio
    #     row) — the beta the Risk-profile table displays, so the note's
    #     "current beta" equals the table by construction.
    trend = _beta_trend(metrics)
    realized_beta = trend["realized_beta"] if trend else _num(risk.get("beta"))

    digest: dict[str, Any] = {
        "benchmark": bench or "the benchmark",
        "window_30d": gap_30d,
        "since_inception": gap_si,
        "capm": _clean({"realized_beta": realized_beta, "alpha_pct": alpha,
                        "volatility_pct": _num(risk.get("volatility"))}),
        # The gap decomposition uses the REALIZED beta (measured on the actual
        # NAV over the period): (realized_beta − 1) × benchmark is the part of
        # the gap that is the mathematical consequence of the risk level;
        # the rest is selection/allocation.
        "risk_vs_selection": _risk_vs_selection(gap_si, realized_beta),
        "beta_trend": trend,
        # NOTE: allocation-vs-target drift is deliberately NOT included. It is a
        # different concept from the benchmark gap (it lives in the optimizer
        # section), and feeding it here made the model suggest incoherent
        # "rebalance to target to close the gap" advice (e.g. buy more bonds,
        # which lowers beta and widens the lag). The gap is explained by
        # beta + selection, full stop.
        "contributors": _contributors(metrics),
    }
    return _clean(digest)


def _current_beta(m) -> Optional[float]:
    """The 'current' portfolio beta = today's weights held constant, i.e. the
    ``historical_risk`` portfolio-row beta the Risk-profile table displays.
    This is what the note calls the *current* beta, so the two agree."""
    hr = getattr(m, "historical_risk", None) or {}
    port = hr.get("portfolio") or {}
    return _num((port.get("metrics") or {}).get("beta"))


def _beta_trend(m) -> Optional[dict]:
    """Both portfolio betas + the direction between them, for the note.

    realized = beta of the ACTUAL order-derived NAV over the holding period
    (``m.risk['beta']``); current = beta of today's weights held constant
    (``historical_risk`` portfolio row). Both come from the engine's single
    ``_compute_beta_alpha`` against the same benchmark, so comparing them is
    apples-to-apples. 'direction' says whether the CURRENT mix tracks the
    benchmark MORE closely than the realized history did — i.e. whether the
    portfolio is set up to CONVERGE toward the benchmark going forward
    (|current − 1| < |realized − 1|)."""
    realized = _num((getattr(m, "risk", None) or {}).get("beta"))
    current = _current_beta(m)
    if realized is None and current is None:
        return None
    direction = None
    if realized is not None and current is not None:
        dr, dc = abs(realized - 1.0), abs(current - 1.0)
        if abs(dc - dr) < 0.02:
            direction = "stable"
        else:
            direction = "converging" if dc < dr else "diverging"
    return _clean({
        "realized_beta": realized,          # actual NAV over the period
        "current_beta": current,            # today's weights (= Risk table)
        "direction": direction,             # converging | diverging | stable
    })


def _bench_period(m, bench_name: Optional[str], period: str):
    """Authoritative period return (e.g. '1m') for a benchmark, from the
    engine's ``holding_performance`` — the single source the Returns tables and
    chart legends read. None when unavailable."""
    if not bench_name:
        return None
    hp = getattr(m, "holding_performance", None)
    if hp is None or getattr(hp, "empty", True):
        return None
    if "name" not in hp.columns or period not in hp.columns:
        return None
    want = bench_name.strip().lower()
    match = hp[hp["name"].astype(str).str.strip().str.lower() == want]
    if match.empty:
        return None
    val = match.iloc[0].get(period)
    return None if (val is None or (isinstance(val, float) and val != val)) else float(val)


def _bench_name(m) -> Optional[str]:
    """The configured GEO benchmark name (the one the 'You vs the market' chart
    compares against), resolved from config — NOT just the first key in
    benchmark_histories (that dict is unordered and its first entry is often an
    unrelated commodities benchmark). Falls back to the first history key only
    if config lookup fails and one exists."""
    bh = getattr(m, "benchmark_histories", None)
    if not (isinstance(bh, dict) and bh):
        return None
    try:
        from tarzan import config as _cfg
        geo = _cfg.benchmark_geo_allocation()
        if geo and geo in bh:
            return geo
    except Exception:  # noqa: BLE001 — config unavailable → best-effort fallback
        pass
    return next(iter(bh))


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

        # Shorten names the same way the newsletter tables do, so the note
        # reads "Xtrackers S&P 500 5C" not the full "... Swap UCITS ETF 5C -
        # EUR Hedged" legal name.
        from tarzan.export._format import short_instrument_name

        def _row(r):
            return _clean({"name": short_instrument_name(r.get("name") or r.get("ticker") or ""),
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
        b = rvs["beta"]                     # realized beta (actual NAV)
        pct = abs(b - 1) * 100
        mkt = rvs["market_risk_contribution_pp"]
        sel = rvs["selection_contribution_pp"]
        if b < 1:
            # Defensive: less market risk → expected to lag a rising benchmark.
            bits.append(
                f"Most of that gap is by design, not bad luck: your realized beta of {b:.2f} "
                f"(how much your portfolio actually moved relative to {bench} over the period) "
                f"means it swung only about {b*100:.0f}% as much, so in a rising market you "
                f"were always going to lag it. That lower risk explains {_pp(mkt)} of the gap; "
                f"the remaining {_pp(sel)} is what your specific fund picks and weights "
                f"{'added on top' if sel >= 0 else 'cost you'}.")
        else:
            bits.append(
                f"Your realized beta of {b:.2f} (how much your portfolio actually moved "
                f"relative to {bench} over the period) means it swung about {pct:.0f}% MORE "
                f"than {bench}, which {'helped' if mkt >= 0 else 'hurt'} by {_pp(mkt)}; the "
                f"remaining {_pp(sel)} is down to your specific fund picks and weights.")

    # Forward-looking beta trend: realized vs today's-weights ("current") beta.
    trend = d.get("beta_trend") or {}
    rb, cb, direction = trend.get("realized_beta"), trend.get("current_beta"), trend.get("direction")
    if rb is not None and cb is not None and direction and abs(cb - rb) >= 0.02:
        if direction == "converging":
            bits.append(
                f"Looking forward, your current mix carries more market risk than your history "
                f"did — beta has risen from {rb:.2f} (realized) to {cb:.2f} (current, today's "
                f"weights) — so the portfolio is now set up to track {bench} more closely than "
                f"it has, and the gap should narrow if that mix holds.")
        else:  # diverging
            bits.append(
                f"Looking forward, your current mix sits further from the benchmark's risk "
                f"level than your history did — beta has moved from {rb:.2f} (realized) to "
                f"{cb:.2f} (current, today's weights) — so it is set up to track {bench} "
                f"less closely going forward, and the gap may widen if that mix holds.")
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
    # Takeaway: about the benchmark GAP and the risk trade-off behind it —
    # deliberately NOT an allocation-drift rebalance suggestion. Drift-to-target
    # is a separate concern (it's in the optimizer section) and mixing it in
    # produces incoherent advice: e.g. "buy more bonds to close the gap" is
    # backwards, since more fixed income LOWERS beta and WIDENS the lag in a
    # rising market. The honest point is the trade-off itself.
    rvs = d.get("risk_vs_selection")
    beta = rvs.get("beta") if rvs else None
    sel = rvs.get("selection_contribution_pp") if rvs else None
    if si and beta is not None and beta < 1 and si["gap_pp"] < 0:
        pick_note = ""
        if sel is not None:
            pick_note = (
                f" Your fund picks {'added' if sel >= 0 else 'cost'} {_pp(abs(sel))} on top, "
                f"so {'they are not the problem' if sel >= 0 else 'they widened it slightly'} — "
                f"the lag is about risk level, not stock-picking.")
        bits.append(
            f"The takeaway is a trade-off, not a mistake: with a beta below 1 this lag is the "
            f"expected cost of a calmer, lower-risk portfolio, and the only way to close it is "
            f"to take on more market risk — which also means bigger drawdowns when markets "
            f"fall.{pick_note} Whether that is worth doing depends on whether tracking {bench} "
            f"matters more to you than a smoother ride.")
    elif si and beta is not None and beta > 1 and si["gap_pp"] < 0:
        # Behind DESPITE more market risk → the lag is selection, not risk.
        bits.append(
            f"Since you carry more market risk than {bench} yet still trail it, this gap is "
            f"about your specific holdings, not your risk level — worth reviewing which "
            f"positions are dragging rather than adding leverage.")
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
        "percentage points; a CAPM split ('capm.realized_beta' and how much of "
        "the since-inception gap comes from taking more/less market risk vs from "
        "selection/allocation in 'risk_vs_selection'); a 'beta_trend' with "
        "realized_beta, current_beta and a direction; allocation drift vs "
        "target; and the holdings that contributed most and least to return.\n"
        "WRITE a tight, quantitative explanation (5-7 sentences, ~140 words, "
        "one flowing paragraph) that a NON-EXPERT can follow — briefly say what "
        "a term means the first time you use it:\n"
        "1. State the since-inception gap and the 30-day gap with their exact "
        "figures, and say whether the divergence is widening or narrowing.\n"
        "2. Attribute the gap using the CAPM split, and EXPLAIN it in plain "
        "words. STATE THE REALIZED BETA VALUE explicitly (from "
        "'risk_vs_selection.beta' / 'beta_trend.realized_beta') and define it: "
        "the realized (performed) beta is how much the portfolio ACTUALLY moved "
        "relative to the benchmark over the period (beta 0.7 = moved ~70% as "
        "much; 1 = in lock-step). A beta below 1 means LESS market risk, so in a "
        "RISING market the portfolio is expected to lag — that is "
        "'market_risk_contribution_pp': the part of the gap that is simply the "
        "mathematical consequence of the chosen risk level, NOT bad fund-picking. "
        "'selection_contribution_pp' is what the specific funds/weights added or "
        "cost ON TOP. Make the cause-and-effect explicit. Name the biggest drag "
        "and the top contributor with their figures.\n"
        "3. Add a FORWARD-LOOKING line from 'beta_trend' ONLY IF both "
        "realized_beta and current_beta are present and they differ: give BOTH "
        "values (label them realized vs current — current = today's holdings' "
        "beta) and, when direction is 'converging' (current beta closer to 1 "
        "than realized), say the portfolio now carries more market risk than it "
        "did on average, so it is set up to track the benchmark MORE closely and "
        "the gap should narrow if that mix holds; when 'diverging', say the "
        "opposite. Never claim a trend when the two betas are equal or one is "
        "missing.\n"
        "4. Finish with ONE takeaway about the GAP and the risk trade-off "
        "behind it. If a beta below 1 is the main cause, say plainly that the "
        "lag is the expected cost of a lower-risk portfolio and the only way to "
        "close it is to take on more market risk (and bigger drawdowns), so it "
        "is a choice about whether tracking the benchmark matters more than a "
        "smoother ride — NOT a mistake to fix. If instead beta is above 1 and "
        "the portfolio still trails, say the gap is about the specific holdings, "
        "not the risk level. DO NOT recommend rebalancing toward allocation "
        "targets or buying a specific asset class (e.g. more bonds/fixed "
        "income) to 'close the gap' — that is a different topic and is usually "
        "BACKWARDS (more fixed income lowers beta and WIDENS the lag in a rising "
        "market). Say nothing about allocation-vs-target drift here.\n"
        "RULES: every claim carries a number from the JSON; write percentage "
        "changes with an explicit + or - sign and gaps in 'pp'; quote beta to two "
        "decimals; never invent figures beyond the JSON; no preamble, no "
        "salutation, no markdown, no headings, no bullet points; do not restate "
        "the whole JSON. This is analysis and portfolio-construction insight, NOT "
        f"a solicitation. Write in {language}."
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
    # Read benchmark period returns + risk from holding_performance — the SAME
    # source the visible Returns/Performance tables and the chart legends use —
    # so the AI note quotes figures that match the tables. (Previously this read
    # benchmark_comparison, whose period returns are computed over a different
    # window (clip-to-portfolio-span vs the tables' 5y cap) and could disagree
    # for the identical benchmark.)
    hp = getattr(m, "holding_performance", None)
    rows: list[dict] = []
    try:
        if hp is None or hp.empty or "type" not in hp.columns:
            return []
        bench = hp[hp["type"].astype(str).str.contains("enchmark", case=False, na=False)]
        keep = ("1m", "3m", "ytd", "1y", "cagr", "beta", "alpha")
        for _, r in bench.iterrows():
            row = {"benchmark": r.get("name") or r.get("ticker")}
            for c in keep:
                if c in bench.columns:
                    row[c] = _num(r.get(c))
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
