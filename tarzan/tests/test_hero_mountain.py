"""Tests for the hero since-inception P&L/TWROR and the mountain chart.

Network-free: they build the newsletter context from a hand-made
PortfolioMetrics and assert the hero and sparkline contracts.
"""

from __future__ import annotations

import pandas as pd
import pytest

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
        performance_full={"1w": 0.5, "period_used": "1.0Y"},
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
    return m


class TestHeroSinceInception:
    def test_uses_lifetime_pnl_when_order_path(self):
        hero = build_context(_metrics(with_order_returns=True), _config())["hero"]
        # PnL% (24%) preferred over the snapshot cost-basis gain (20%).
        assert "24.00%" in hero["gain_pct"]
        assert hero["twror_pct"] is not None
        assert "14.49%" in hero["twror_pct"]

    def test_falls_back_to_snapshot_gain_holdings_only(self):
        hero = build_context(_metrics(with_order_returns=False), _config())["hero"]
        # Snapshot gain = (6000-5000)/5000 = 20%.
        assert "20.00%" in hero["gain_pct"]
        assert hero["twror_pct"] is None


class TestMountainChart:
    def test_plots_actual_value_series(self):
        spark = build_context(_metrics(with_order_returns=True), _config())["sparkline"]
        assert spark["available"] is True
        assert spark["is_mountain"] is True
        # The chart window is the last 30 days (here all 5 points fit).
        assert spark["label"] == "Last 30 days"
        assert len(spark["bars"]) == 5
        # Pills carry the window %-change (PnL/TWROR live in the hero now).
        assert len(spark["pills"]) >= 1

    def test_legacy_line_when_no_order_series(self):
        m = _metrics(with_order_returns=False)
        m.portfolio_history = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        )
        spark = build_context(m, _config())["sparkline"]
        assert spark["is_mountain"] is False
        assert spark["label"].startswith("Last ")

    def test_renders_without_crash(self):
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        assert "Last 30 days" in html
        assert "TWROR" in html  # in the hero line
