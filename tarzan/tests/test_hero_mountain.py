"""Tests for the hero since-inception P&L/TWROR.

Network-free: they build the newsletter context from a hand-made
PortfolioMetrics and assert the hero contract, plus a full-render smoke test.
"""

from __future__ import annotations

import datetime
import re

import pandas as pd
import pytest

from tarzan.engine import stats
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


def _right_axis_ticks(svg: str) -> list[float]:
    """The right axis' tick VALUES, in euros, parsed off the rendered labels.

    The axis reads in euros now, so a regex hunting for "%" finds nothing and the
    tests that scale geometry against it die on an empty list rather than on a
    wrong number. ``fmt_eur_tick`` writes "€0", "€2k" and "−€1.5k" (synthetic),
    with U+2212 for the sign like every other figure in the issue.
    """
    import re

    out = []
    for sign, mag, k in re.findall(
            r'text-anchor="start"[^>]*>(−?)€([\d.]+)(k?)</text>', svg):
        v = float(mag) * (1000.0 if k else 1.0)
        out.append(-v if sign else v)
    return out


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
        # The flows XIRR is solved on, so the since-inception panel can draw its
        # MWR line: an opening deposit, one top-up, terminated by today's value.
        # ``xirr_pct`` is SOLVED from them rather than typed in, so the fixture
        # cannot state a rate its own flows do not produce.
        m.xirr_cashflows = [
            (datetime.date(2026, 1, 4), -4800.0),
            (datetime.date(2026, 1, 18), -300.0),
            (datetime.date(2026, 2, 1), 6000.0),
        ]
        m.xirr_pct = stats.xirr(m.xirr_cashflows) * 100.0
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

        # Both P&L series are EUROS now. Synthetic book: €100k of value, unrealized
        # peaking at €4k while total reaches €12k (synthetic), so the axis has to be
        # fitted to the wider one.
        unreal = [0.0, 2000.0, 1000.0, 4000.0]
        total = [0.0, 9000.0, 8000.0, 12000.0]   # realized included -> much wider
        svg = _hero_value_chart(
            [100000.0, 104000.0, 102000.0, 108000.0], unreal,
            ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04"], [],
            total_eur=total,
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
        ticks = _right_axis_ticks(svg)
        assert ticks, "no euro ticks on the right axis"
        assert max(ticks) >= max(total), (
            f"right axis tops out at €{max(ticks):.0f} and clips Total P&L at "
            f"€{max(total):.0f}"
        )

    def test_the_right_axis_carries_the_euro_sign_and_no_percent(self):
        """The unit the axis STATES, not just the numbers it plots.

        The tick formatter stayed on ``fmt_pct_tick`` after the values became euros,
        so a synthetic €1,900 P&L printed as "1900%" — the right number under the
        wrong unit,
        and off by a factor of a hundred read as a percentage of anything.
        """
        from tarzan.export.newsletter import PALETTE
        from tarzan.export.newsletter._charts import _hero_value_chart

        svg = _hero_value_chart(
            [100000.0, 104000.0], [500.0, 1900.0],
            ["2026-07-01", "2026-07-02"], [], total_eur=[600.0, 2500.0],
        )
        # The right axis is the only text drawn in PALETTE["muted"] — the value axis
        # and the date row both use "subtle". Anchoring on the colour rather than on
        # text-anchor keeps the date label ("Jul 01", also start-anchored) out.
        right = re.findall(
            rf'fill="{PALETTE["muted"]}">([^<]+)</text>', svg)
        assert right, "the right axis has no labels"
        assert all("€" in t for t in right), right
        assert not any("%" in t for t in right), right
        # ...and the values are the P&L's own, not a ratio of it.
        assert max(_right_axis_ticks(svg)) >= 2500.0, right

    def test_hero_survives_a_missing_total_series(self):
        from tarzan.export.newsletter._charts import _hero_value_chart

        svg = _hero_value_chart(
            [100.0, 104.0], [0.0, 2.0], ["2026-07-01", "2026-07-02"], [],
            total_eur=None,
        )
        assert svg and "<svg" in svg

    def test_the_hero_is_the_only_chart_that_draws_the_pnl_measures(self):
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        # The hero key names both, with the axis side, because that chart is where a
        # P&L now lives: euros against euros.
        assert html.count("Total P&amp;L (€, right)") == 1
        assert html.count("Unreal. P&amp;L (€, right)") == 1
        # Both P&L lines came off the since-inception panel -- they restated the TWROR
        # line with a different denominator, and MWR took the space to say the thing
        # TWROR cannot. What is left of each name is its STATE tile, and the
        # abbreviated form was ONLY ever the panel's key.
        assert html.count(">Total P&amp;L<") == 1        # the tile
        assert html.count(">Unrealized P&amp;L<") == 1   # the tile
        assert ">Unreal. P&amp;L<" not in html

    def test_the_lifetime_panel_pairs_time_weighted_with_money_weighted(self):
        """The point of the swap: two returns over one span, not one return twice.

        The distance between the lines is whether the timing of contributions helped
        or hurt -- a question neither line answers alone and the P&L lines never
        answered at all.
        """
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        panel = html.split("Return · since inception", 1)[1]
        assert ">TWROR<" in panel
        assert ">MWR (cum.)<" in panel
        assert ">Total P&amp;L<" not in panel
        assert ">Unreal. P&amp;L<" not in panel

    def test_the_window_grid_draws_three_lines_not_five(self):
        """The cost of small multiples, pinned so it cannot drift back.

        Five lines in a 182px cell is not a chart. Should someone re-add Total or
        Unrealized P&L to ``PANEL_LINES``, this fails -- and so does the
        legibility the grid was chosen for.
        """
        html = render_newsletter(_metrics(with_order_returns=True), _config())
        assert "Return · by window" in html
        # The grid's own key names exactly the three lines its cells draw. The
        # boundary is the LIFETIME return panel, which follows the grid directly now
        # that the volatility grid is gone.
        grid = html.split("Return · by window", 1)[1].split(
            "Return · since inception", 1)[0]
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
        assert "Unreal. P&amp;L (€, right)" in html
        assert "Total P&amp;L (€, right)" in html


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
            total_eur=[1.0, 3.0]))
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

        # Synthetic: opens at +€8k, dips BELOW that, recovers.
        total = [8000.0, 11000.0, 6000.0, 9500.0]
        svg = _hero_value_chart(
            [140000.0, 141000.0, 139500.0, 142000.0], [1000.0, 2000.0, 500.0, 1800.0],
            ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24"], [],
            total_eur=total,
        )
        ML, MT, MB, w, h = 52, 12, 26, 580, 196
        ph = h - MT - MB
        ticks = _right_axis_ticks(svg)
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
            [140000.0, 141000.0, 142000.0], [1.0, 2.0, 1.8],
            ["2026-08-10", "2026-08-17", "2026-08-24"], [],
            total_eur=[8.0, 9.0, 9.5],
        )
        assert len(re.findall(r'stroke-dasharray="3,3"', svg)) == 1
