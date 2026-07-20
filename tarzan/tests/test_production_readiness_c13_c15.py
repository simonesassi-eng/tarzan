"""Pre-fix exploration oracles for production-readiness conditions C13-C15."""

from __future__ import annotations

from datetime import datetime

from tarzan.data import market_quotes
from tarzan.data.enricher import classify_asset_class
from tarzan.delivery import run_and_send
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


# **Validates: Requirements 2.13**
def test_c13_provider_boundary_returns_structured_attempt_and_policy_evidence():
    """Even an empty request needs a typed result, not an unclassified dict."""
    result = market_quotes.broker_1d([])
    assert hasattr(result, "attempts") and hasattr(result, "availability"), (
        "provider boundary returned a raw mapping with no source, observation/fetch "
        f"time, attempts, fallback rung, coverage, latency, or policy: {result!r}"
    )
    assert getattr(result, "policy", None) is not None


# **Validates: Requirements 2.14**
def test_c14_replayed_logical_event_invokes_smtp_only_once(monkeypatch, tmp_path):
    """Sequential replay is the minimal duplicate-send counterexample."""
    # This test intentionally exercises the host-local claim store. GitHub's
    # validate job has no production claim credentials and must remain isolated.
    for key in (
        "GITHUB_ACTIONS",
        "DELIVERY_CLAIM_ENDPOINT",
        "DELIVERY_CLAIM_TOKEN",
        "DELIVERY_CLAIM_STORE_PATH",
        "TARZAN_STABLE_EVENT_ID",
        "AUTHORIZED_RESEND_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in {
        "SMTP_USER": "sender@example.invalid",
        "SMTP_PASS": "not-a-real-secret",
        "RECIPIENT_EMAIL": "reader@example.invalid",
        "GITHUB_RUN_ID": "424242",
        "ISSUE_NUMBER": "7",
        "DRY_RUN": "0",
    }.items():
        monkeypatch.setenv(key, value)

    metrics = PortfolioMetrics(total_value=100.0)
    config = InvestorConfig()
    sent: list[str] = []

    monkeypatch.setattr("tarzan.delivery.ROOT", tmp_path)
    monkeypatch.setattr(
        "tarzan.delivery.resolve_inputs",
        lambda: {"orders": "orders.csv", "config": "config.csv", "targets_per_holding": None},
    )
    monkeypatch.setattr("tarzan.delivery.run", lambda **kwargs: (metrics, config))
    monkeypatch.setattr("tarzan.delivery.render_newsletter", lambda **kwargs: "<html>ok</html>")
    monkeypatch.setattr("tarzan.delivery.build_subject", lambda *args: "subject")
    monkeypatch.setattr("tarzan.delivery._seed_manual_proxies", lambda: None)
    monkeypatch.setattr("tarzan.backtest.newsletter_portfolios", lambda: [])
    monkeypatch.setattr("tarzan.delivery.now_local", lambda: datetime(2025, 6, 30, 12, 0))
    monkeypatch.setattr(
        "tarzan.delivery.send_email",
        lambda **kwargs: sent.append(kwargs["subject"]),
    )

    assert run_and_send() == 0
    assert run_and_send() == 0
    assert sent == ["subject"], (
        "same frozen logical workflow event reached SMTP more than once; "
        f"invocations={len(sent)}"
    )


# **Validates: Requirements 2.15**
def test_c15_unknown_kind_is_not_heuristically_routed_to_a_tracked_category():
    """Unknown evidence must produce typed unavailable capability results."""
    holding = Holding(
        isin="FUTURE", ticker="FUTURE", quantity=1.0, cost_basis_eur=100.0,
        market_value_eur=100.0, current_price=100.0, current_value=100.0,
        currency="EUR", asset_class=AssetClass.ALTERNATIVE,
    )
    result = classify_asset_class(
        {"quoteType": "FUTURE_KIND", "category": "unknown", "sector": ""},
        "FUTURE",
        holding,
    )

    assert getattr(result, "support", None) == "UNSUPPORTED", (
        "unknown instrument evidence was collapsed into an AssetClass instead of "
        f"a declared unavailable capability result: {result!r}"
    )
    assert getattr(result, "availability", None) == "UNAVAILABLE"
    assert getattr(result, "analytical_impact", None)
