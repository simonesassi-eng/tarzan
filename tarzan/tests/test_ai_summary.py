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
        lambda *a, **k: "**Your portfolio** is up.\n\n- bullet noise",
    )
    out = ai_summary.generate_summary(_metrics(), _config())
    assert out is not None
    assert "**" not in out and "\n" not in out


# ── Output hygiene ───────────────────────────────────────────────────────────

def test_sanitize_caps_length():
    long = "word " * 400
    out = ai_summary._sanitize(long)
    assert len(out) <= ai_summary._MAX_CHARS


def test_extract_text_handles_malformed():
    assert ai_summary._extract_text({}) is None
    assert ai_summary._extract_text({"candidates": []}) is None


# ── Newsletter rendering: AI summary replaces Signals ────────────────────────

def test_render_shows_ai_summary_when_present():
    html = render_newsletter(
        _metrics(), _config(),
        ai_summary="Your portfolio is up 20% since inception, steady this month.",
    )
    assert "Market context" in html
    assert "up 20% since inception" in html
    assert "not financial advice" in html


def test_render_falls_back_to_signals_without_ai():
    ctx = build_context(_metrics(), _config(), ai_summary=None)
    assert ctx["ai_summary"] is None
    # The rule-based insights are still computed for the fallback.
    assert "smart_insights" in ctx
