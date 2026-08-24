"""Tests for the hero since-inception P&L/TWROR.

Network-free: they build the newsletter context from a hand-made
PortfolioMetrics and assert the hero contract, plus a full-render smoke test.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _config() -> InvestorConfig:
    c = InvestorConfig()
    c.invested_allocation_targets_pctg = {"Equities": 100.0}
    return c


@pytest.fixture(autouse=True)
def _pin_clock(monkeypatch):
    """The fixture series end 1 Feb 2026; return windows are measured back from
    the run's today, so the reference date belongs to the fixture."""
    monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2026, 2, 1))


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
        performance_full={"5d": 0.5, "1m": 0.25, "period_used": "1.0Y"},
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
        # Weekly money P&L from the cumulative series. "5D" anchors five
        # sessions back (five days of change, six closes — the span Yahoo's own
        # page uses), so the window reaches one fixture point further back than
        # it did when it stepped four.
        assert hero["week_pnl_eur"] is not None
        assert "260" in hero["week_pnl_eur"]
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


class TestBothPnlMeasuresAreDrawn:
    """Total and Unrealized P&L must BOTH appear, on a shared right axis.

    The two differ whenever anything has been realized (here: lifetime P&L 350
    vs unrealized 300), so a chart carrying one of them answers half the
    question. The axis is the subtle part — fitted to one series it clips the
    other off the plot rather than failing visibly.
    """

    def test_hero_draws_both_and_scales_the_axis_to_the_wider(self):
        import re

        from tarzan.export.newsletter._charts import _hero_value_chart
        from tarzan.export.newsletter import PALETTE

        unreal = [0.0, 2.0, 1.0, 4.0]
        total = [0.0, 9.0, 8.0, 12.0]     # realized included -> much wider
        svg = _hero_value_chart(
            [100.0, 104.0, 102.0, 108.0], unreal,
            ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"], [],
            total_pct=total,
        )
        drawn = dict(
            (color, points) for points, color in
            re.findall(r'<polyline points="([^"]+)" fill="none" stroke="(#[0-9A-Fa-f]{6})"', svg)
        )
        assert PALETTE["unreal"] in drawn and PALETTE["pnl"] in drawn, sorted(drawn)
        ticks = [float(t) for t in re.findall(r'>(−?[\d.]+)%<', svg)]
        assert max(ticks) >= max(total), (
            f"right axis tops out at {max(ticks)}% and clips Total P&L at "
            f"{max(total)}%"
        )

    def test_hero_survives_a_missing_total_series(self):
        from tarzan.export.newsletter._charts import _hero_value_chart

        svg = _hero_value_chart(
            [100.0, 104.0], [0.0, 2.0], ["2026-07-01", "2026-07-02"], [],
            total_pct=None,
        )
        assert svg and "<svg" in svg

    def test_both_return_charts_name_both_measures(self):
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        # One key entry per measure per chart: hero (with axis side), then the
        # two return panels (bare names).
        assert html.count("Total P&amp;L (%, right)") == 1
        assert html.count("Unreal. P&amp;L (%, right)") == 1
        # Three now, not two: the two chart keys plus the hero STATE tile,
        # whose label is escaped at its own markup boundary (it used to reach
        # the document as a bare "&", which mail clients tolerate and no test
        # noticed).
        assert html.count(">Total P&amp;L<") == 3
        assert html.count(">Unreal. P&amp;L<") == 2


class TestRender:
    def test_renders_without_crash(self):
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        # Lean hero: the portfolio value band (scoreboard + mountain removed).
        assert "Portfolio" in html
        # The window matrix sits in the PORTFOLIO section now, under the value
        # chart, and the section heading is the only title it has -- the
        # "How your money moved" line was dropped because the heading plus the
        # matrix say it. Anchor on the matrix's own footer instead.
        assert ">Portfolio</span>" in html
        # The matrix's own first column header. The "Annualized: TWROR / XIRR"
        # footer that used to be the anchor is gone: it repeated the captions of
        # the TWROR and MWR tiles in STATE.
        assert ">Window<" in html
        # Every ampersand reaches the document as an entity. The matrix writes
        # it itself; the tiles go through the template, where autoescape is off
        # because the filename ends in .j2 — so they are escaped at their own
        # markup boundary in _build_hero._tile. Before that they arrived raw,
        # which is invalid HTML that mail clients happen to tolerate.
        assert "P&amp;L \u20ac" in html      # matrix column head
        assert "Unrealized" in html
        assert "TWROR" in html
        assert "Since inception" in html     # matrix row label
        assert "Total P&amp;L" in html       # state tile
        assert "Unrealized P&amp;L" in html  # state tile
        # The Portfolio value series follows the Markets contract: green above
        # the start baseline, red below it. Both P&L measures are dashed
        # secondary series on the right axis, each in the colour it also
        # carries on the return charts -- violet Unrealized, cyan Total -- so
        # the mapping holds across the issue.
        from tarzan.export.newsletter import PALETTE
        assert 'clip-path="url(#dg' in html
        assert f'stroke="{PALETTE["green"]}" stroke-width="2.6"' in html
        assert f'stroke="{PALETTE["red"]}" stroke-width="2.6"' in html
        assert (
            f'stroke="{PALETTE["unreal"]}" stroke-width="1.8" '
            'stroke-dasharray="4,3"' in html
        )
        assert (
            f'stroke="{PALETTE["pnl"]}" stroke-width="1.8" '
            'stroke-dasharray="1,2.5"' in html
        )
        # ...and the key names both, since the right axis can no longer be
        # labelled with one word.
        assert "Unreal. P&amp;L (%, right)" in html
        assert "Total P&amp;L (%, right)" in html
