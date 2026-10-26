"""Shared test fixtures for all tests."""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.models.investor_config import InvestorConfig


@pytest.fixture(autouse=True)
def _no_live_credentials(monkeypatch):
    """Tests explicitly inject synthetic credentials when testing auth."""
    for name in (
        "GEMINI_API_KEY", "SMTP_USER", "SMTP_PASS", "RECIPIENT_EMAIL",
        "DRIVE_FOLDER_ID", "GOOGLE_DRIVE_CREDENTIALS_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS", "TARZAN_CACHE_KEY_B64",
        "DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN",
        "AUTHORIZED_RESEND_TOKEN", "GITHUB_TOKEN", "GH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _isolated_yahoo_identity_cache(tmp_path_factory):
    """Never use or populate the developer's SDK identity/cookie cache."""
    import yfinance

    yfinance.set_tz_cache_location(str(tmp_path_factory.mktemp("yahoo-cache")))


@pytest.fixture(autouse=True)
def _reject_unstubbed_transport(monkeypatch):
    """Fail even if an ordinary provider fallback catches the blocked call.

    Tests may replace a transport with a synthetic response explicitly.
    Original acquisition, parser, memo and ownership code stays exercised.
    """
    from tarzan.tests.network_guard import NetworkGuard

    guard = NetworkGuard()
    guard.install(monkeypatch)
    yield
    guard.assert_unused()


@pytest.fixture(autouse=True)
def _disable_market_cache(monkeypatch):
    """Disable the on-disk market-data cache for the whole suite by
    default, so tests never read from or write to the real
    ``~/.cache/tarzan`` (which would leak state across runs and pull the
    network on resolution fast-paths). Tests that specifically exercise
    the cache re-enable it against a temporary directory.
    """
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "1")
    yield


@pytest.fixture(autouse=True)
def _isolate_run_context():
    """Give every test a fresh live run context.

    ``runtime.configure()`` writes into a module-level ContextVar, so a test
    that configures a pinned ``as_of`` (the golden-master runs do) leaves that
    effective date installed for every test that runs after it in the same
    process. Anything reading the ambient context then silently inherits it —
    ``ValuationCompletenessEvaluator`` derives its ``captured_at`` that way, so
    a "2 hours old" holding was measured against a 2025 effective date and
    graded FALLBACK instead of STALE. That made a real freshness assertion
    pass or fail purely on test ORDER, which is exactly the kind of check
    that must not be order-dependent.
    """
    from tarzan import runtime
    runtime.reset(attempt_id="test", invocation_source="test")
    yield
    runtime.reset(attempt_id="test", invocation_source="test")


@pytest.fixture(autouse=True)
def _no_official_quote_batch(monkeypatch):
    """Stub the batched Yahoo quote pair network-free for the whole suite.

    ``intraday_feeds`` measures the 1D against ``regularMarketPrice`` /
    ``regularMarketPreviousClose`` — the pair the instrument's own page shows.
    Left unstubbed, every fixture-driven quote test would reach the live
    endpoint and assert against whatever the market did today. Tests that
    exercise the quote path install their own values.
    """
    monkeypatch.setattr("tarzan.data.market_quotes._fetch_official_quotes",
                        lambda symbols: {}, raising=True)
    yield


@pytest.fixture(autouse=True)
def _disable_ai_summary(monkeypatch):
    """Force the AI portfolio summary off for the whole suite so tests
    never hit the network or consume API tokens. Tests that exercise the
    summary mock the call explicitly."""
    monkeypatch.setenv("TARZAN_DISABLE_AI", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # The digest and the newsletter Markets strip pull live quotes through
    # the price layer; stub them network-free for the suite (tests that
    # exercise quotes mock the fetch explicitly).
    monkeypatch.setattr("tarzan.data.market_quotes.fetch_market_quotes",
                        lambda *a, **k: [], raising=True)
    yield


@pytest.fixture
def unavailable_justetf(monkeypatch):
    """Explicit absent referee for valuation tests that inject primary quotes.

    Keep native prefetch, requests.Response and source handling. Other URLs
    still reach the rejection gate rather than receiving an invented response.
    """
    import requests

    blocked = requests.Session.request

    def request(session, method, url, **kwargs):
        if str(url).startswith("https://www.justetf.com/api/etfs/"):
            response = requests.Response()
            response.status_code = 404
            response.headers["content-type"] = "application/json"
            response._content = b"{}"
            return response
        return blocked(session, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", request)


@pytest.fixture
def unavailable_risk_free(monkeypatch):
    """Explicit missing optional RF tape; preserve the ordinary scalar fallback."""
    monkeypatch.setattr("tarzan.data.proxy_data.risk_free_daily", lambda: None)


@pytest.fixture
def sample_config() -> InvestorConfig:
    """Minimal investor config for testing."""
    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {
        "Equities": 65.0,
        "Fixed Income": 35.0,
    }
    config.equity_geo_targets_pctg = {
        "USA": 50.0,
        "Eurozone EMU": 20.0,
        "Japan": 10.0,
        "Dev ex-USA ex-EMU ex-JP": 10.0,
        "Emerging Markets": 10.0,
    }
    config.target_cash_buffer_eur = 1000.0
    config.rebalancing_target_tolerance_pctg = 2.0
    config.rebalancing_no_sell = False
    config.rebalancing_lump_sum_amount_eur = 0.0
    return config


@pytest.fixture
def sample_holdings() -> list[Holding]:
    """5 holdings: 3 equity (different geos), 1 bond, 1 cash."""
    return [
        Holding(
            isin="US0000000001", ticker="USA_ETF", quantity=100.0,
            cost_basis_eur=5000.0, market_value_eur=6000.0, currency="EUR",
            name="USA ETF", current_price=60.0, current_value=6000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.USA,
            geo_breakdown={Geography.USA: 100.0},
        ),
        Holding(
            isin="EU0000000001", ticker="EU_ETF", quantity=50.0,
            cost_basis_eur=2000.0, market_value_eur=2000.0, currency="EUR",
            name="Eurozone ETF", current_price=40.0, current_value=2000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.EUROZONE_EMU,
            geo_breakdown={Geography.EUROZONE_EMU: 100.0},
        ),
        Holding(
            isin="EM0000000001", ticker="EM_ETF", quantity=30.0,
            cost_basis_eur=1500.0, market_value_eur=2000.0, currency="EUR",
            name="Emerging ETF", current_price=66.66, current_value=2000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.EMERGING_MARKETS,
            geo_breakdown={Geography.EMERGING_MARKETS: 100.0},
        ),
        Holding(
            isin="BOND00000001", ticker="BOND_ETF", quantity=50.0,
            cost_basis_eur=5000.0, market_value_eur=5000.0, currency="EUR",
            name="Bond ETF", current_price=100.0, current_value=5000.0,
            asset_class=AssetClass.FIXED_INCOME,
        ),
        Holding(
            isin="CASH00000001", ticker="CASH_ETF", quantity=10.0,
            cost_basis_eur=1000.0, market_value_eur=1000.0, currency="EUR",
            name="Cash ETF", current_price=100.0, current_value=1000.0,
            # This shared solver fixture intends an account-cash balance.
            # Names/tickers are deliberately not mechanics evidence.
            security_type="CASH",
            asset_class=AssetClass.CASH_EQUIVALENTS,
        ),
    ]


@pytest.fixture
def sample_price_history() -> pd.Series:
    """5-year synthetic daily price series, linearly growing from 100 to 150."""
    dates = pd.date_range(end="2026-03-01", periods=1260, freq="B")
    import numpy as np
    prices = np.linspace(100, 150, len(dates))
    return pd.Series(prices, index=dates, name="TEST")
