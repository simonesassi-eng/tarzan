"""Section [04] PORTFOLIO MOVERS — what it ranks and what it drops.

The section replaced Attribution, so it inherits the question Attribution answered:
which lines moved the number. The measure is therefore CONTRIBUTION, weight times
return, and not the return on its own — the whole reason the old design was wrong is
that a 4.5% sleeve down 6.6% topped the list while a 12.8% sleeve down 0.8% had moved
the book three times as much.

What is checked here is exactly what a reader would be misled by if it broke:
the ranking measure, which lines are dropped, and that no figure is printed for a
line that was not an extreme.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from tarzan.export.newsletter import _sections_perf as sp


def _universe(rows):
    """``rows`` as ``(bare, weight, {window: return})`` -> the builder's shape."""
    return sp._mover_universe([
        {"ticker": f"{bare}.MI", "bare": bare, "weight": weight, "rets": rets}
        for bare, weight, rets in rows])


class TestTheRankingMeasure:
    def test_weight_times_return_beats_the_bigger_return(self):
        """The defect the section exists to fix.

        Measured on the real book: ranking 1M by return put a 3.3% sleeve up 3.36%
        second, worth +0.110pp, ahead of a 7.7% sleeve up 3.27% worth +0.253pp.
        """
        universe = _universe([
            ("SMALL", 3.3, {"1m": 3.36}),
            ("BIG", 7.7, {"1m": 3.27}),
        ])
        ahead, _behind = sp._mover_rows(universe, "1m")
        assert [r["bare"] for r in ahead] == ["BIG", "SMALL"]

    def test_contribution_is_weight_times_return_over_one_hundred(self):
        universe = _universe([("X", 12.5, {"1d": -0.80})])
        assert universe[0]["contrib"]["1d"] == pytest.approx(-0.10)

    def test_a_line_with_no_return_is_ranked_nowhere(self):
        universe = _universe([
            ("A", 10.0, {"1d": 1.0}),
            ("B", 90.0, {"1d": None}),
        ])
        ahead, behind = sp._mover_rows(universe, "1d")
        assert [r["bare"] for r in ahead] == ["A"]
        assert [r["bare"] for r in behind] == ["A"]

    def test_a_line_with_no_weight_is_not_in_the_universe(self):
        """A closed position carries returns and no weight; it moved nothing today."""
        assert _universe([("GONE", 0.0, {"1d": 9.9})]) == []

    def test_a_line_with_no_returns_at_all_is_not_in_the_universe(self):
        assert _universe([("BOND", 5.0, {"1d": None, "5d": None,
                                         "1m": None, "3m": None})]) == []


class TestWhatTheGridPrints:
    @pytest.fixture
    def html(self):
        # Eight lines so the three-each-side ranking leaves lines out.
        rows = [(f"T{i}", float(i + 1), {"1d": (i - 3) * 0.5, "5d": 0.1,
                                        "1m": 0.2, "3m": 0.3})
                for i in range(8)]
        return sp._mover_grid_html(_universe(rows), "Portfolio", "#E6EDF6",
                                   {"1d": -0.1, "5d": 0.1, "1m": 0.2, "3m": 0.3})

    def test_it_names_the_bare_ticker(self, html):
        """A truncated instrument name printed "Xtr. MSCI Wrld" for two different
        funds; a bare ticker is 3-5 characters and unambiguous, and the venue suffix
        is noise in a grid where every row is one instrument."""
        assert ">T7</td>" in html and "T7.MI" not in html

    def test_only_lines_that_reached_an_extreme_are_shown(self, html):
        shown = set(re.findall(r">(T\d)</td>", html))
        # Ranked on 1d, the middle line of eight is in nobody's top or bottom three.
        assert len(shown) < 8
        assert "T0" in shown and "T7" in shown

    def test_the_header_says_how_many_were_dropped(self, html):
        assert "not top three" in html

    def test_a_figure_is_printed_only_where_a_line_was_an_extreme(self, html):
        """The tint carries the magnitude of the rest.

        Filling every cell was tried on the real book and rejected as too busy, so a
        cell that is merely tinted must stay wordless — and a bold cell must carry a
        figure. Both directions matter: a bold blank would look like a bug.
        """
        # Body cells only. The totals row is also centred and also carries a
        # figure, and it is not an extreme of anything -- only the grid cells,
        # which are the ones with a cell border, are under this rule.
        cells = [c for c in re.findall(r'<td align="center"[^>]*>(?:&nbsp;|[^<]*)</td>',
                                      html) if "border:1px solid" in c]
        assert cells, "no grid cells matched"
        for cell in cells:
            bold = "font-weight:700" in cell
            body = re.sub(r"<[^>]+>", "", cell).replace("&nbsp;", "").strip()
            assert bold == bool(body), cell

    def test_the_universe_total_is_the_last_row(self, html):
        assert html.rindex("All") > html.rindex("T0")


class TestTheSectionOnRealisticMetrics:
    """End to end through the builder, from a metrics object shaped like the engine's."""

    @pytest.fixture
    def ctx(self):
        from tarzan.export.newsletter._constants import _NewsletterContext

        hp = pd.DataFrame([
            {"ticker": "AAA.MI", "type": "In portfolio", "1d": 1.0, "5d": 2.0,
             "1m": 3.0, "3m": 4.0},
            {"ticker": "BBB.DE", "type": "In portfolio", "1d": -1.0, "5d": -2.0,
             "1m": -3.0, "3m": -4.0},
            {"ticker": "CCC.PA", "type": "Target not held", "1d": 0.5, "5d": 0.5,
             "1m": 0.5, "3m": 0.5},
        ])
        holdings = pd.DataFrame([
            {"ticker": "AAA.MI", "weight_pct": 60.0},
            {"ticker": "BBB.DE", "weight_pct": 40.0},
        ])

        class M:
            holding_performance = hp
            holdings_df = holdings
            target_weights = {"AAA.MI": 50.0, "CCC.PA": 50.0}
            performance = {"1d": 0.2, "5d": 0.4, "1m": 0.6, "3m": 0.8}

        class C:
            pass

        return _NewsletterContext(metrics=M(), config=C())

    def test_both_universes_are_rendered(self, ctx):
        out = sp._build_portfolio_movers(ctx)
        assert out["available"] is True
        assert ">Portfolio</span>" in out["html"]
        assert ">Target</span>" in out["html"]

    def test_a_line_held_and_planned_appears_at_both_its_weights(self, ctx):
        """AAA is 60% of the book and 50% of the plan, so its contribution differs
        between the two grids. That is the point of showing both."""
        out = sp._build_portfolio_movers(ctx)
        assert "+0.60" in out["html"]      # 60% x 1.0%
        assert "+0.50" in out["html"]      # 50% x 1.0%

    def test_the_book_total_is_the_measured_return_not_a_reweighting(self, ctx):
        """The portfolio's own 1D is +0.20%, while its two priced lines reweight to
        +0.20% only by coincidence of this fixture — what matters is that the figure
        comes from ``performance``, which every other section prints."""
        ctx.metrics.performance = {"1d": 9.99, "5d": 0.4, "1m": 0.6, "3m": 0.8}
        assert "+9.99%" in sp._build_portfolio_movers(ctx)["html"]

    def test_no_holding_performance_means_no_section(self, ctx):
        ctx.metrics.holding_performance = pd.DataFrame()
        assert sp._build_portfolio_movers(ctx) == {"available": False}

    def test_no_target_still_renders_the_book(self, ctx):
        ctx.metrics.target_weights = {}
        out = sp._build_portfolio_movers(ctx)
        assert out["available"] is True
        assert ">Target</span>" not in out["html"]


class TestTheFigureFormat:
    def test_a_small_contribution_keeps_three_decimals(self):
        """Two decimals round a whole 1D column to 0.00 and the ranking stops being
        legible; the issue tapers precision the same way for percentages."""
        assert sp._mover_pp(-0.0962) == "−0.096"

    def test_a_large_one_takes_two(self):
        assert sp._mover_pp(0.7867) == "+0.79"

    def test_it_uses_the_typographic_minus(self):
        assert sp._mover_pp(-1.0).startswith("−")

    def test_none_prints_nothing(self):
        assert sp._mover_pp(None) == ""


class TestTheTint:
    def test_zero_alpha_is_the_background(self):
        assert sp._mover_tint("#2FBF71", "#111A24", 0.0) == "#111a24"

    def test_full_alpha_is_the_colour(self):
        assert sp._mover_tint("#2FBF71", "#111A24", 1.0) == "#2fbf71"

    def test_it_is_opaque_because_clients_drop_rgba(self):
        assert re.fullmatch(r"#[0-9a-f]{6}",
                            sp._mover_tint("#FF5F52", "#111A24", 0.5))
