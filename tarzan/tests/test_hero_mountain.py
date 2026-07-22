"""Tests for the hero since-inception P&L/TWROR.

Network-free: they build the newsletter context from a hand-made
PortfolioMetrics and assert the hero contract, plus a full-render smoke test.
"""

from __future__ import annotations

import pandas as pd

from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _config() -> InvestorConfig:
    c = InvestorConfig()
    c.invested_allocation_targets_pctg = {"Equities": 100.0}
    return c


def _metrics(*, with_order_returns: bool) -> PortfolioMetrics:
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
        performance_full={"1w": 0.5, "1m": 0.25, "period_used": "1.0Y"},
    )
    if with_order_returns:
        m.pnl_eur = 1200.0          # lifetime realized + unrealized
        m.pnl_pct = 24.0            # on capital deployed
        m.invested_capital_eur = 5000.0
        m.twror_pct = 14.49
        m.actual_value_series = pd.Series(
            [4800.0, 5200.0, 5100.0, 5600.0, 6000.0],
            index=pd.date_range("2025-12-29", periods=5, freq="W"),
        )
        # Cumulative P&L series: real money gained over the window = its
        # delta = 350 − 0 = +€350 (net of contributions).
        m.pnl_series = pd.Series(
            [0.0, 120.0, 90.0, 250.0, 350.0],
            index=pd.date_range("2025-12-29", periods=5, freq="W"),
        )
        # Smooth flow-adjusted NAV index + unrealized series so the
        # Performance section (matrix + charts) renders in the HTML test.
        m.portfolio_history = pd.Series(
            [100.0, 101.5, 101.2, 102.8, 103.5],
            index=pd.date_range("2025-12-29", periods=5, freq="W"),
        )
        m.unrealized_series = pd.Series(
            [0.0, 100.0, 80.0, 200.0, 300.0],
            index=pd.date_range("2025-12-29", periods=5, freq="W"),
        )
        m.inception_date = "2025-12-29"
    return m


class TestHeroSinceInception:
    def test_uses_lifetime_pnl_when_order_path(self):
        hero = build_context(_metrics(with_order_returns=True), _config())["hero"]
        # Total PnL% (24%) on net deposits; Unrealized% = snapshot (20%).
        assert hero["has_total_pnl"] is True
        assert "24.00%" in hero["total_pnl_pct"]
        assert "20.00%" in hero["unrealized_pct"]
        assert hero["twror_pct"] is not None
        assert "14.49%" in hero["twror_pct"]

    def test_inception_label_is_month_year(self):
        hero = build_context(_metrics(with_order_returns=True), _config())["hero"]
        assert hero["inception_label"] == "Dec 2025"

    def test_this_week_has_pnl_and_twror(self):
        hero = build_context(_metrics(with_order_returns=True), _config())["hero"]
        # Weekly money P&L from the cumulative series (last 7 days): the
        # delta from the point ≤ cutoff (250) to the last (350) = +€100.
        assert hero["week_pnl_eur"] is not None
        assert "100" in hero["week_pnl_eur"]
        assert hero["week_pnl_pct"] is not None
        # Weekly TWROR from performance_full['1w'] = 0.5%.
        assert hero["week_twror_pct"] is not None
        assert "0.50%" in hero["week_twror_pct"]

    def test_falls_back_to_snapshot_gain_holdings_only(self):
        hero = build_context(_metrics(with_order_returns=False), _config())["hero"]
        # No order history: Total PnL collapses to the snapshot gain (20%).
        assert hero["has_total_pnl"] is False
        assert "20.00%" in hero["total_pnl_pct"]
        assert hero["twror_pct"] is None


class TestRender:
    def test_renders_without_crash(self):
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        # Lean hero: the portfolio value band (scoreboard + mountain removed).
        assert "Portfolio" in html
        # The Performance section now carries the P&L / TWROR story.
        assert "How your money moved" in html
        assert "Total P&amp;L" in html
        assert "Unrealized P&amp;L" in html
        assert "TWROR" in html
        # The Portfolio value series follows the Markets contract: green above
        # the start baseline, red below it. Unrealized P&L is neutral/dashed.
        from tarzan.export.newsletter import PALETTE
        assert 'clip-path="url(#dg' in html
        assert f'stroke="{PALETTE["green"]}" stroke-width="2.6"' in html
        assert f'stroke="{PALETTE["red"]}" stroke-width="2.6"' in html
        assert (
            f'stroke="{PALETTE["muted"]}" stroke-width="1.8" '
            'stroke-dasharray="4,3"' in html
        )
