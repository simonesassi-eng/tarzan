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
        # Unrealized keeps its violet identity. Total P&L is drawn TWICE, green
        # and red, clipped about its own zero: the conditional colour moved off
        # the value line onto the P&L, so cyan is not on this chart any more.
        assert PALETTE["unreal"] in drawn, sorted(drawn)
        assert PALETTE["green"] in drawn and PALETTE["red"] in drawn, sorted(drawn)
        assert drawn[PALETTE["green"]] == drawn[PALETTE["red"]], (
            "the two halves must be the same Total P&L line, differing only in clip"
        )
        assert PALETTE["ink"] in drawn, "the value line is neutral now"
        assert PALETTE["pnl"] not in drawn
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
        # since-inception panel (bare names).
        assert html.count("Total P&amp;L (%, right)") == 1
        assert html.count("Unreal. P&amp;L (%, right)") == 1
        # TWO, not three. The window grid replaced the single 30-day panel, and
        # six 182px cells cannot carry five lines -- so the per-window panels draw
        # TWROR, Target and the benchmark only, and both P&L measures keep their
        # € and % columns in the matrix plus their line on the since-inception
        # chart. The remaining two are that chart's key and the hero STATE tile.
        assert html.count(">Total P&amp;L<") == 2
        assert html.count(">Unreal. P&amp;L<") == 1

    def test_the_window_grid_draws_three_lines_not_five(self):
        """The cost of small multiples, pinned so it cannot drift back.

        Five lines in a 182px cell is not a chart. Should someone re-add Total or
        Unrealized P&L to ``PANEL_LINES``, this fails -- and so does the
        legibility the grid was chosen for.
        """
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        assert "Return · by window" in html
        # The grid's own key names exactly the three lines its cells draw.
        grid = html.split("Return · by window", 1)[1].split("Volatility", 1)[0]
        assert ">TWROR<" in grid
        assert ">Total P&amp;L<" not in grid
        assert ">Unreal. P&amp;L<" not in grid


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
        # The green/red split is on TOTAL P&L about its own break-even, not on
        # the value: value above or below where the window opened is an
        # arbitrary reference that a deposit moves, while the sign of the
        # lifetime P&L is worth colouring. The value line is neutral ink, which
        # also keeps the accent colour free for the cash-flow triangles sitting
        # on it. Unrealized keeps the violet it carries on the return charts.
        from tarzan.export.newsletter import PALETTE
        # The clip ids are the P&L split now (pg/pr), not the value's (dg/dr).
        assert 'clip-path="url(#hg' in html
        assert f'stroke="{PALETTE["green"]}" stroke-width="2.4"' in html
        assert f'stroke="{PALETTE["red"]}" stroke-width="2.4"' in html
        assert f'stroke="{PALETTE["ink"]}" stroke-width="2.6"' in html
        assert (
            f'stroke="{PALETTE["unreal"]}" stroke-width="1.8" '
            'stroke-dasharray="4,3"' in html
        )
        assert f'stroke="{PALETTE["pnl"]}" stroke-width="1.8"' not in html
        # ...and the key names both, since the right axis can no longer be
        # labelled with one word.
        assert "Unreal. P&amp;L (%, right)" in html
        assert "Total P&amp;L (%, right)" in html


class TestClipIdsAreUniqueAcrossCharts:
    """Two charts must never mint the same SVG element id.

    The hero and the intraday sparklines both clip a green/red split, and both
    used a "pg<n>"/"pr<n>" prefix off their OWN counter — so the hero's "pg1" and
    the first sparkline's "pg1" were one id on two elements, and every reference
    resolved to whichever the document happened to put first. The deterministic
    render has no intraday series, which is why no golden ever caught it.
    """

    def test_hero_and_intraday_spark_share_no_ids(self):
        import re

        import pandas as pd

        from tarzan.export.newsletter._charts import (
            _hero_value_chart, _intraday_spark, reset_spark_uids)

        reset_spark_uids()
        ids = lambda svg: set(re.findall(r'id="([^"]+)"', svg))  # noqa: E731
        hero = ids(_hero_value_chart(
            [100.0, 104.0], [0.0, 2.0], ["2026-07-01", "2026-07-02"], [],
            total_pct=[1.0, 3.0]))
        intra = pd.Series(
            [10.0, 10.4], index=pd.date_range("2026-07-02 09:00", periods=2, freq="h"))
        spark = ids(_intraday_spark(intra, 10.2))
        assert hero and spark, (hero, spark)
        assert not hero & spark, f"shared ids: {sorted(hero & spark)}"


class TestTheSplitIsAgainstTheWindowOpen:
    """The green/red boundary is where Total P&L stood when the window opened —
    the same reference the value line used to be coloured against, moved to the
    series that now carries the colour. NOT zero: a P&L that opens the window at
    +8% and dips to +6% is down over the window and must read red, even though it
    never went under break-even.
    """

    def test_the_boundary_is_the_series_own_opening_level(self):
        import re

        from tarzan.export.newsletter._charts import _hero_value_chart
        from tarzan.export.newsletter import PALETTE

        total = [8.0, 11.0, 6.0, 9.5]          # opens +8%, dips BELOW it, recovers
        svg = _hero_value_chart(
            [240000.0, 241000.0, 239500.0, 242000.0], [1.0, 2.0, 0.5, 1.8],
            ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"], [],
            total_pct=total,
        )
        ML, MT, MB, w, h = 52, 12, 26, 580, 196
        ph = h - MT - MB
        ticks = [float(t.replace("−", "-"))
                 for t in re.findall(r'text-anchor="start"[^>]*>(−?[\d.]+)%<', svg)]
        plo, phi = min(ticks), max(ticks)

        def y_of(v):
            return MT + (1 - (v - plo) / ((phi - plo) or 1)) * ph

        dashed = float(re.search(
            r'<line x1="52" y1="([\d.]+)"[^>]*stroke-dasharray="3,3"', svg).group(1))
        assert abs(dashed - y_of(total[0])) < 0.15, (
            f"boundary at y={dashed}, window open is y={y_of(total[0]):.1f}, "
            f"zero is y={y_of(0.0):.1f}"
        )
        assert abs(dashed - y_of(0.0)) > 1.0, "the boundary must not be zero"
        # Both halves exist, so the dip below the opening really reads red.
        assert PALETTE["green"] in svg and PALETTE["red"] in svg

    def test_one_dashed_reference_only(self):
        """There was one dashed line before this change and there is one after:
        the colour boundary. A second one for the value's own opening would say
        nothing, since the value is no longer coloured against it."""
        import re

        from tarzan.export.newsletter._charts import _hero_value_chart

        svg = _hero_value_chart(
            [240000.0, 241000.0, 242000.0], [1.0, 2.0, 1.8],
            ["2026-08-10", "2026-08-17", "2026-08-24"], [],
            total_pct=[8.0, 9.0, 9.5],
        )
        assert len(re.findall(r'stroke-dasharray="3,3"', svg)) == 1
