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


class TestMountainChart:
    def test_plots_actual_value_series(self):
        spark = build_context(_metrics(with_order_returns=True), _config())["sparkline"]
        assert spark["available"] is True
        assert spark["is_mountain"] is True
        # The chart window is the last 30 days (here all 5 points fit).
        assert spark["label"] == "Last 30 days"
        assert len(spark["bars"]) == 5
        # Order path → a real-money PnL gain pill (€, net of contributions).
        pill_text = " ".join(p["text"] for p in spark["pills"])
        assert "PnL" in pill_text
        assert "350" in pill_text  # +€350 gained over the window

    def test_twr_pill_matches_returns_table_1m(self):
        """The chart TWR pill must equal performance_full['1m'] so it agrees
        with the 'Returns vs benchmarks' total-portfolio 1M cell."""
        spark = build_context(_metrics(with_order_returns=True), _config())["sparkline"]
        pill_text = " ".join(p["text"] for p in spark["pills"])
        assert "TWROR" in pill_text
        assert "+0.25%" in pill_text

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
        assert "Time-Weighted Rate of Return" in html  # scoreboard title
        assert "Profit &amp; Losses" in html or "Profit & Losses" in html
        assert "Unrealized" in html  # hero overlay

    def test_nan_values_do_not_crash(self):
        """A cold cache / vendor outage can leave NaN valuations on some
        days; the area chart must degrade, not raise (regression)."""
        import numpy as np
        m = _metrics(with_order_returns=True)
        m.actual_value_series = pd.Series(
            [4800.0, np.nan, 5100.0, np.nan, 6000.0],
            index=pd.date_range("2025-12-29", periods=5, freq="W"),
        )
        spark = build_context(m, _config())["sparkline"]
        # No NaN heights leak through.
        assert all(isinstance(b["height"], int) for b in spark["bars"])

    def test_all_nan_series_falls_back_flat(self):
        import numpy as np
        m = _metrics(with_order_returns=True)
        m.actual_value_series = pd.Series(
            [np.nan, np.nan, np.nan],
            index=pd.date_range("2025-12-29", periods=3, freq="W"),
        )
        spark = build_context(m, _config())["sparkline"]
        assert spark["available"] is False
        assert all(isinstance(b["height"], int) for b in spark["bars"])
