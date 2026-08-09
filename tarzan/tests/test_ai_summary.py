"""Tests for the AI portfolio summary — fully network-free / token-free.

The real Gemini call is never made: tests cover the deterministic digest
builder, the disabled/fallback behavior, output sanitization, and the
newsletter rendering path with a mocked summary string.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.export import ai_summary
from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _config() -> InvestorConfig:
    c = InvestorConfig()
    c.invested_allocation_targets_pctg = {"Equities": 100.0}
    return c


def _metrics() -> PortfolioMetrics:
    df = pd.DataFrame([{
        "isin": "US0000000001", "ticker": "AAA", "name": "Alpha ETF",
        "asset_class": "Equities", "current_value": 6000.0,
        "cost_basis_eur": 5000.0, "weight_pct": 100.0, "gain_pct": 20.0,
        "quantity": 100.0, "avg_purchase_price": 50.0, "pct_of_class": 100.0,
        "currency": "EUR",
    }])
    m = PortfolioMetrics(
        total_value=6000.0, invested_value=6000.0, cash_value=0.0,
        holdings_df=df,
        allocation_by_class=pd.DataFrame([{"category": "Equities", "weight_pct": 100.0}]),
        performance_full={"1w": 0.5, "1m": 1.2, "ytd": 8.0, "period_used": "1.0Y"},
    )
    m.pnl_eur = 1000.0
    m.pnl_pct = 20.0
    m.twror_pct = 14.49
    m.inception_date = "2025-12-29"
    m.risk = {"volatility": 12.3, "sharpe": 1.1, "max_drawdown": -8.0}
    return m


# ── Digest builder (deterministic, no network) ──────────────────────────────

def test_digest_is_comprehensive_and_serializable():
    import json
    digest = ai_summary.build_digest(_metrics(), _config())
    # Round-trips as JSON (model input must serialize).
    json.dumps(digest)
    assert digest["snapshot"]["value_eur"] == 6000
    assert digest["since_inception"]["total_pnl_pct"] == 20.0
    assert digest["since_inception"]["twror_cumulative_pct"] == 14.49
    assert "1m" in digest["twror_by_period_pct"]
    assert digest["holdings"][0]["name"] == "Alpha ETF"
    assert "risk" in digest


def test_digest_drops_nan_and_none():
    m = _metrics()
    m.twror_pct = float("nan")
    digest = ai_summary.build_digest(m, _config())
    # NaN values are stripped, not serialized as NaN.
    assert "twror_cumulative_pct" not in digest["since_inception"]


# ── Enable/disable gating ────────────────────────────────────────────────────

def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    assert ai_summary.is_enabled() is False
    # generate_summary short-circuits to None without any network call.
    assert ai_summary.generate_summary(_metrics(), _config()) is None


def test_disabled_flag_overrides_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("TARZAN_DISABLE_AI", "1")
    assert ai_summary.is_enabled() is False


def test_generate_summary_never_raises_on_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ai_summary, "_call_gemini", _boom)
    assert ai_summary.generate_summary(_metrics(), _config()) is None


def test_generate_summary_sanitizes_model_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    monkeypatch.setattr(
        ai_summary, "_call_gemini",
        lambda *a, **k: (
            "**Markets** rallied on rate hopes.\n"
            "- Oil slipped -1.2% on supply data.\n"
        ),
    )
    out = ai_summary.generate_summary(_metrics(), _config())
    assert out is not None
    # Markdown and any leading bullet are stripped per line, but (unlike the
    # single-paragraph _sanitize) newlines are the whole point: one line per
    # news item.
    assert out.splitlines() == [
        "Markets rallied on rate hopes.",
        "Oil slipped -1.2% on supply data.",
    ]


def test_generate_summary_caps_at_seven_lines(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    nine = "\n".join(
        f"Headline {i} moved markets today for a stated reason." for i in range(9)
    )
    monkeypatch.setattr(ai_summary, "_call_gemini", lambda *a, **k: nine)
    out = ai_summary.generate_summary(_metrics(), _config())
    assert len(out.splitlines()) == 7


# ── Output hygiene ───────────────────────────────────────────────────────────

def test_sanitize_caps_length():
    long = "word " * 400
    out = ai_summary._sanitize(long)
    assert len(out) <= ai_summary._MAX_CHARS


def test_extract_text_handles_malformed():
    assert ai_summary._extract_text({}) is None
    assert ai_summary._extract_text({"candidates": []}) is None


# ── Newsletter rendering: AI summary is the only market-context surface ──────

def test_render_shows_ai_summary_when_present():
    html = render_newsletter(
        _metrics(), _config(),
        ai_summary="Your portfolio is up 20% since inception, steady this month.",
    )
    assert "AI-generated news digest" in html
    assert "up 20% since inception" in html
    assert "not financial advice" in html


def test_render_without_ai_has_no_market_context():
    # No AI summary → no market-context block at all (the rule-based Signals
    # block was removed; there is no longer a fallback).
    ctx = build_context(_metrics(), _config(), ai_summary=None)
    assert ctx["ai_summary"] is None
    assert "smart_insights" not in ctx
    html = render_newsletter(_metrics(), _config(), ai_summary=None)
    assert "AI-generated news digest" not in html
    assert "worth your attention" not in html


def test_render_auto_resolves_ai_summary_when_none(monkeypatch):
    # ai_summary=None means "resolve it" (same contract as the benchmark names):
    # render_newsletter must call generate_summary itself, so the CLI and the
    # emailed newsletter render identically without the caller wiring the AI.
    monkeypatch.setattr(
        "tarzan.export.ai_summary.generate_summary",
        lambda m, c: "Auto-resolved market note for the render path.",
    )
    html = render_newsletter(_metrics(), _config())  # ai_summary defaults to None
    assert "AI-generated news digest" in html
    assert "Auto-resolved market note" in html


def test_render_empty_string_forces_block_off(monkeypatch):
    # Escape hatch: "" is falsy but not None, so it must NOT trigger auto-resolve
    # (guards a caller that explicitly wants no market-context block).
    monkeypatch.setattr(
        "tarzan.export.ai_summary.generate_summary",
        lambda m, c: (_ for _ in ()).throw(AssertionError("must not auto-resolve on empty string")),
    )
    html = render_newsletter(_metrics(), _config(), ai_summary="")
    assert "AI-generated news digest" not in html


# ── Benchmark-divergence note (charts section) ───────────────────────────────

def _divergence_metrics() -> PortfolioMetrics:
    """A 2-year portfolio that TRAILS its benchmark, with a beta>1 and a clear
    laggard holding — so the divergence digest has something to attribute."""
    import numpy as np
    idx = pd.date_range("2024-07-01", "2026-07-01", freq="B")
    nav = pd.Series(np.linspace(100, 138, len(idx)), index=idx)    # port +38%
    acwi = pd.Series(np.linspace(200, 288, len(idx)), index=idx)   # bench +44%
    pnl = pd.Series(np.linspace(0, 4500, len(idx)), index=idx)
    hold = pd.DataFrame([
        {"isin": "US0000000001", "ticker": "USA", "name": "USA ETF",
         "asset_class": "Equities", "weight_pct": 45.0, "gain_pct": 30.0,
         "current_value": 6525.0, "cost_basis_eur": 5000.0},
        {"isin": "EM0000000001", "ticker": "EM", "name": "EM ETF",
         "asset_class": "Equities", "weight_pct": 12.0, "gain_pct": -8.0,
         "current_value": 1740.0, "cost_basis_eur": 1900.0},
    ])
    gd = pd.DataFrame([
        {"type": "geography_equity", "category": "USA",
         "actual_pct": 58.0, "target_pct": 50.0, "delta_pct": 8.0},
        {"type": "geography_equity", "category": "Emerging Markets",
         "actual_pct": 8.0, "target_pct": 12.0, "delta_pct": -4.0},
    ])
    m = PortfolioMetrics(total_value=14500.0, invested_value=13500.0,
                         cash_value=1000.0, holdings_df=hold)
    m.actual_value_series = pd.Series(np.linspace(10000, 14500, len(idx)), index=idx)
    m.portfolio_history = nav
    m.pnl_series = pnl
    m.unrealized_series = pnl
    # Lifetime fields the Performance section (which hosts the note) needs to
    # render at all.
    m.pnl_eur = 4500.0
    m.pnl_pct = 45.0
    m.twror_pct = 38.0
    m.twror_annualized_pct = 17.0
    m.xirr_pct = 16.0
    m.benchmark_histories = {"iShares MSCI ACWI": acwi}
    m.goal_deltas = gd
    # Two distinct betas on purpose:
    #   realized (m.risk) = 1.20  → beta of the ACTUAL NAV over the period;
    #                               this is what the gap attribution must use.
    #   current (historical_risk) = 1.40 → today's-weights beta (Risk table);
    #                               |1.40-1| > |1.20-1| ⇒ DIVERGING.
    m.risk = {"beta": 1.20, "alpha": -2.1, "volatility": 16.0}
    m.historical_risk = {
        "available": True,
        "portfolio": {"is_portfolio": True, "label": "Your portfolio",
                      "metrics": {"beta": 1.40, "alpha": -2.1}},
    }
    return m


def test_divergence_digest_has_both_windows_and_capm_split():
    d = ai_summary.build_divergence_digest(_divergence_metrics(), _config())
    assert d is not None
    # Both chart windows carry a portfolio/benchmark/gap.
    assert d["since_inception"]["gap_pp"] == -6.0          # 38 − 44
    assert "gap_pp" in d["window_30d"]
    # CAPM split uses the REALIZED beta (1.20 from m.risk — the actual NAV),
    # NOT the current-weights 1.40, since the gap is a since-inception fact.
    rvs = d["risk_vs_selection"]
    assert rvs["beta"] == 1.20
    assert round(rvs["market_risk_contribution_pp"], 1) == round((1.20 - 1) * 44.0, 1)
    assert round(rvs["market_risk_contribution_pp"] + rvs["selection_contribution_pp"], 1) == -6.0


def test_divergence_digest_carries_beta_trend():
    d = ai_summary.build_divergence_digest(_divergence_metrics(), _config())
    trend = d["beta_trend"]
    assert trend["realized_beta"] == 1.20      # actual NAV
    assert trend["current_beta"] == 1.40       # today's weights (Risk table)
    assert trend["direction"] == "diverging"   # 1.40 is further from 1 than 1.20


def test_divergence_note_states_both_betas_and_trend():
    # The note must quote the realized beta explicitly and both betas for the
    # trend — and must NOT confuse the two (attribution uses realized).
    m = _divergence_metrics()
    note = ai_summary.divergence_note(m, _config())
    assert "1.20" in note                      # realized beta, stated
    assert "1.40" in note                      # current beta, in the trend line
    assert "realized" in note.lower()          # trend line labels realized
    assert "diverging" in note.lower()         # 1.40 further from 1 than 1.20


def test_divergence_digest_none_without_benchmark():
    m = _divergence_metrics()
    m.benchmark_histories = {}          # nothing to compare against
    assert ai_summary.build_divergence_digest(m, _config()) is None


def test_divergence_fallback_is_quantitative_when_ai_off():
    # AI disabled by the autouse fixture → deterministic rule-based note.
    # Fixture beta is 1.20 (>1) and trails → the "gap is about holdings, not
    # risk level" takeaway.
    note = ai_summary.divergence_note(_divergence_metrics(), _config())
    assert note is not None
    assert "-6.0pp" in note                      # the since-inception gap
    assert "beta" in note.lower()                # the CAPM attribution
    assert "EM" in note                          # the laggard, named (short_instrument_name → "EM")
    # Not trivial: several quantitative clauses.
    assert note.count("pp") >= 3
    # The takeaway must NOT be a vague "risk profile" line, and must NOT tell
    # the reader to rebalance toward allocation targets to close the gap
    # (the incoherent, often-backwards advice we removed).
    assert "risk profile" not in note.lower()
    assert "target" not in note.lower()
    assert "rebalanc" not in note.lower()


def test_divergence_fallback_low_beta_frames_tradeoff():
    # A defensive (beta<1) portfolio that trails → the takeaway is the honest
    # risk trade-off, never a "buy more of X to close the gap" suggestion.
    m = _divergence_metrics()
    m.risk = {"beta": 0.70, "alpha": 0.5, "volatility": 10.0}
    m.historical_risk = {"available": True,
                         "portfolio": {"is_portfolio": True,
                                       "metrics": {"beta": 0.80}}}
    note = ai_summary.divergence_note(m, _config())
    assert "more market risk" in note.lower()
    assert "not a fix" in note.lower()
    # No allocation-rebalance advice.
    assert "fixed income" not in note.lower()
    assert "rebalanc" not in note.lower()


def test_divergence_note_is_terse():
    # The whole point of this rewrite: short and data-first, no parentheticals.
    note = ai_summary.divergence_note(_divergence_metrics(), _config())
    assert "(" not in note and ")" not in note          # no parenthetical asides
    assert len(note) <= 700                              # far tighter than the old ~1450
    # Still carries the key data points.
    assert "-6.0pp" in note and "beta" in note.lower()


def test_vs_the_market_draws_the_full_range_chart():
    """The section's wide chart spans the whole history, not 30 days.

    It used to be asserted alongside a "why you're diverging" block that also
    lived here. That block is gone: the gap is in the section's subtitle, the
    three lines carry their end values on the chart, and the beta it quoted is a
    column in RISK -- what it added beyond that was an opinion on whether to
    close the gap, which is advice this issue does not give. The chart itself is
    still worth pinning, so the month-tick evidence stays.
    """
    import re
    from tarzan.export.newsletter import render_newsletter
    html = render_newsletter(_divergence_metrics(), _config(),
                             benchmark_geo="iShares MSCI ACWI")
    assert "Why you" not in html
    # Month ticks across the 2-year span prove it is the full-range chart rather
    # than a 30-day one. (Format is %b %y, e.g. "Jul 24".)
    month_labels = re.findall(r'>([A-Z][a-z]{2}(?: \d{2})?)<', html)
    assert len([m for m in month_labels if re.fullmatch(r"[A-Z][a-z]{2} \d{2}", m)]) >= 2


def test_divergence_note_uses_ai_prose_when_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    monkeypatch.setattr(
        "tarzan.export.ai_summary._call_gemini",
        lambda system, user, use_search=True: "You trail ACWI by -6.0pp; beta 1.15 explains +6.6pp.",
    )
    note = ai_summary.divergence_note(_divergence_metrics(), _config())
    assert "beta 1.15 explains" in note          # the mocked model prose, not the fallback


def test_divergence_note_falls_back_when_ai_returns_nothing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("TARZAN_DISABLE_AI", raising=False)
    monkeypatch.setattr(
        "tarzan.export.ai_summary._call_gemini",
        lambda system, user, use_search=True: None,   # model gave nothing usable
    )
    note = ai_summary.divergence_note(_divergence_metrics(), _config())
    # Never blank: the quant fallback fills in.
    assert note and "-6.0pp" in note
