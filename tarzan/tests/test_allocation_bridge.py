"""The Allocation section: one butterfly per block, its figures built in.

The colours follow the 5/25 band -- the narrower of ±5 points and ±25% of the
target -- and only the colours: the optimizer keeps its own tolerance.

Network-free: a hand-built PortfolioMetrics. Every amount here is synthetic.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from tarzan.export._palette import PALETTE
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import (
    _alloc_band,
    _band_colour,
    _build_diversification,
)
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

GREEN, RED = PALETTE["green"], PALETTE["red"]


class TestBand:
    @pytest.mark.parametrize("target, band", [
        (5.0, 1.25), (10.0, 2.5), (20.0, 5.0), (35.0, 5.0), (71.5, 5.0), (0.0, 0.0)])
    def test_the_narrower_of_five_points_and_a_quarter_of_the_target(self, target, band):
        assert _alloc_band(target, InvestorConfig()) == pytest.approx(band)

    def test_both_limits_come_from_the_config(self):
        cfg = InvestorConfig.from_dict({"allocation_band_abs_pp": "3",
                                        "allocation_band_rel_pctg": "50"})
        assert _alloc_band(10.0, cfg) == pytest.approx(3.0)
        assert _alloc_band(4.0, cfg) == pytest.approx(2.0)

    def test_two_states_and_the_edge_is_inside(self):
        cfg = InvestorConfig()
        assert _band_colour(1.25, 5.0, cfg) == GREEN
        assert _band_colour(-1.3, 5.0, cfg) == RED
        assert _band_colour(4.9, 35.0, cfg) == GREEN
        assert _band_colour(-5.1, 35.0, cfg) == RED

    def test_a_line_the_plan_sells_is_out_of_band_at_any_weight(self):
        assert _band_colour(0.2, 0.0, InvestorConfig()) == RED

    def test_the_optimizer_keeps_its_own_tolerance(self):
        cfg = InvestorConfig.from_dict({"rebalancing_target_tolerance_pctg": "1.5"})
        assert cfg.rebalancing_target_tolerance_pctg == 1.5
        assert (cfg.allocation_band_abs_pp, cfg.allocation_band_rel_pctg) == (5.0, 25.0)


def _ctx(cash: float = 4000.0) -> _NewsletterContext:
    held = [("IE00000NTSG", "NTSG", "Core", "Equities", 30000.0),
            ("IE00000SGLD", "SGLD", "Gold", "Gold", 10000.0),
            ("IE00000XDEV", "XDEV", "Value", "Equities", 40000.0),
            ("IE00000XMJP", "XMJP", "Japan", "Equities", 20000.0)]
    df = pd.DataFrame([{"isin": i, "ticker": t, "name": n, "asset_class": c,
                        "current_value": v, "cost_basis_eur": v, "weight_pct": v / 1000.0,
                        "gain_pct": 0.0, "quantity": 1.0, "avg_purchase_price": v,
                        "pct_of_class": 0.0, "currency": "EUR"}
                       for i, t, n, c, v in held])
    # The plan's own check is post-trade: XMJP, which a full rebalance sells
    # outright, is not in it although the book still holds it.
    items = [{"category": "Core", "ticker": "NTSG", "actual_pct": 50.0, "target_pct": 50.0},
             {"category": "Gold", "ticker": "SGLD", "actual_pct": 10.0, "target_pct": 10.0},
             {"category": "New", "ticker": "AVWC", "actual_pct": 40.0, "target_pct": 40.0},
             {"category": "Value", "ticker": "XDEV", "actual_pct": 0.0, "target_pct": 0.0}]
    m = PortfolioMetrics(
        total_value=100000.0 + cash, invested_value=100000.0, cash_value=cash,
        holdings_df=df,
        allocation_by_class=pd.DataFrame([{"category": "Equities", "weight_pct": 90.0},
                                          {"category": "Gold", "weight_pct": 10.0}]),
        allocation_by_geo=pd.DataFrame([{"category": "USA", "weight_pct": 100.0}]),
        rebalancing_verifications=[{"kind": "per_holding_portfolio", "items": items}],
    )
    cfg = InvestorConfig()
    cfg.target_use_per_holding_only = True
    cfg.invested_allocation_targets_pctg = {"Equities": 90.0, "Gold": 10.0}
    cfg.equity_geo_targets_pctg = {"USA": 100.0}
    cfg.target_cash_buffer_eur = 4000.0
    return _NewsletterContext(metrics=m, config=cfg, issue_number=1,
                              benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI")


def _block(html: str, title: str) -> str:
    """The block's SVG and the table of tail rows (Total, reserve) under it."""
    start = html.index(f'aria-label="{title}')
    end = html.index("</svg>", start)
    nxt = html.find("aria-label=", end)
    return html[start:nxt if nxt != -1 else len(html)]


_NAME = re.compile(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*fill="(#[0-9A-Fa-f]{6})" '
                   r'text-anchor="middle">([^<]+)</text>')
_FIG = re.compile(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*text-anchor="(end|start)">'
                  r'<tspan font-weight="600">([^<]+)</tspan> <tspan[^>]*>([^<]+)</tspan></text>')
_GAP = re.compile(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*font-weight="700" fill="(#[0-9A-Fa-f]{6})" '
                  r'text-anchor="end">([^<]+)</text>')
_BAR = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" '
                  r'fill="#[0-9A-Fa-f]{6}"( fill-opacity="[\d.]+")?/>')


def _rows(block: str) -> dict:
    """The butterfly read back by row: for each name, its colour, the TODAY and
    PLAN figures (weight, euros) or None, the gap (text, colour) or None, and the
    bars on its line as (x, width) or None."""
    svg = block[:block.index("</svg>")]
    rows: dict = {}
    for x, y, fill, name in _NAME.findall(svg):
        rows[name] = {"x": float(x), "y": float(y), "name_fill": fill,
                      "today": None, "plan": None, "gap": None,
                      "today_bar": None, "plan_bar": None}
    by_y = {r["y"]: r for r in rows.values()}
    for x, y, anchor, pc, eur in _FIG.findall(svg):
        by_y[float(y)]["today" if anchor == "end" else "plan"] = (pc, eur)
    for x, y, fill, text in _GAP.findall(svg):
        if float(x) == 354.0:
            by_y[float(y)]["gap"] = (text, fill)
    for x, y, w, h, faded in _BAR.findall(svg):
        yc = float(y) + float(h) / 2.0
        by_y[yc + 3.5]["plan_bar" if faded else "today_bar"] = (float(x), float(w))
    return rows


class TestBridge:
    def test_every_line_held_today_is_in_today(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        for ticker in ("NTSG", "SGLD", "XDEV", "XMJP"):
            assert f">{ticker}<" in svg, ticker

    def test_the_lines_the_plan_sells_meet_in_one_sale(self):
        block = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        rows = _rows(block)
        # A line the plan sells keeps its red name and has no PLAN bar or figure.
        for t in ("XDEV", "XMJP"):
            assert rows[t]["name_fill"] == RED and rows[t]["plan"] is None, t
        # Their sum is one row under the figure: what the rebalance raises.
        tail = block[block.index("</svg>"):]
        assert ">To sell<" in tail
        assert re.search(r">60\.0% <span[^>]*>\u20ac60k</span>", tail)  # synthetic
        assert re.search(rf'color:{RED};[^"]*">\+60\.0<', tail)

    def test_a_line_bought_from_nothing_is_named_once(self):
        block = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        assert block.count(">AVWC<") == 1
        assert _rows(block)["AVWC"]["today"] is None

    def test_weight_and_euros_side_by_side_and_the_gap_in_points(self):
        block = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        r = _rows(block)["NTSG"]
        # "30.0% €30k" today, left of the name; "50.0% €50k" in the plan (synthetic), right
        # of it; the gap in points at the row's end, coloured by the band.
        assert r["today"] == ("30.0%", "\u20ac30k")  # synthetic
        assert r["plan"] == ("50.0%", "\u20ac50k")  # synthetic
        assert r["gap"] == ("\u221220.0", RED)
        assert _rows(block)["SGLD"]["gap"] == ("0.0", GREEN)

    def test_the_note_states_leverage_as_the_return_stacking_literature_does(self):
        """Gross exposure per euro of capital, the excess as implicit cash borrowed
        at the cash rate, and the funds that stack it — today's from the book,
        the plan's from the config's copy of the plan (a shared symbol ISIN-
        qualified, so MATE's stacked leg is seen). No per-class ratio: capital
        is attributed to no class."""
        ctx = _ctx()
        ctx.config.invested_allocation_targets_pctg = {"Equities": 120.0, "Gold": 10.0}
        ctx.config.target_plan_weights = {"NTSG": 50.0, "SGLD": 10.0, "MATE@IE000BS31R00": 10.0}
        html = _build_diversification(ctx)["html"]
        assert ("Totals past 100% are leverage: 1.00\u00d7 today, 1.30\u00d7 plan; the excess "
                "is implicit cash borrowed at the cash rate (0% today, \u221230% plan).") in html
        # NTSG is held and planned (no side) and comes first; MATE is in the plan only.
        assert ("Stacked funds: NTSG 1.5\u00d7 (0.9 equities + 0.6 fixed income), "
                "MATE 1.65\u00d7 (1 equities + 0.65 alternative, plan only).") in html
        assert "\u00d7\u2192" not in html and "futures" not in html

    def test_an_unlevered_book_prints_no_leverage_line(self):
        html = _build_diversification(_ctx())["html"]
        assert "leverage" not in html
        # The fixture holds NTSG (a 90/60 fund) at a weight that keeps the
        # total under par; the stack is still named, the borrowing is not.
        assert "Stacked funds: NTSG 1.5\u00d7" in html

    def test_cash_is_banded_at_a_quarter_of_its_target(self):
        within = _block(_build_diversification(_ctx(cash=4900.0))["html"], "Asset class")
        beyond = _block(_build_diversification(_ctx(cash=5100.0))["html"], "Asset class")
        # The reserve is a tail row under the stacks; its gap carries the colour.
        assert re.search(rf'color:{GREEN};[^"]*">\+\u20ac900<', within)
        assert re.search(rf'color:{RED};[^"]*">\+\u20ac1.1k<', beyond)  # synthetic




class TestButterfly:
    """One line per row: the name in a centre column, TODAY's bar growing left
    from it, PLAN's bar growing right, both on the block's one scale; the
    figures in fixed outer columns. Nothing is abbreviated, nothing wraps, and
    no bar reaches into a figure column."""

    def _check(self, block: str) -> None:
        from tarzan.export.newsletter import _sections_alloc as sa
        svg = block[:block.index("</svg>")]
        rows = _rows(block)
        assert rows
        # Every name is whole, in one text, on the one centre column.
        assert len({r["x"] for r in rows.values()}) == 1
        assert all("\u2026" not in n for n in rows)
        # Rows sit on the module's pitch, the first under the column heads.
        ys = sorted(r["y"] for r in rows.values())
        assert ys[0] == pytest.approx(sa._TOP + sa._ROW / 2 + 3.5)
        assert all(b - a == pytest.approx(sa._ROW) for a, b in zip(ys, ys[1:]))
        # The figure columns: TODAY ends at one x (anchored end), PLAN starts
        # at one x (anchored start), and the gap ends at the right edge.
        today_x = {float(x) for x, _y, a, _p, _e in _FIG.findall(svg) if a == "end"}
        plan_x = {float(x) for x, _y, a, _p, _e in _FIG.findall(svg) if a == "start"}
        assert len(today_x) == 1 and len(plan_x) == 1
        lf, rf = today_x.pop(), plan_x.pop()
        # Bars: TODAY's end on one edge left of the name, PLAN's start on one
        # edge right of it; each is level with its name and stays in its lane.
        ends = {round(x + w, 0) for r in rows.values() if r["today_bar"] for x, w in [r["today_bar"]]}
        starts = {x for r in rows.values() if r["plan_bar"] for x, _w in [r["plan_bar"]]}
        assert len(ends) == 1 and len(starts) == 1, (ends, starts)
        cx = next(iter(rows.values()))["x"]
        assert max(ends) < cx < min(starts)
        for r in rows.values():
            if r["today_bar"]:
                assert r["today_bar"][0] >= lf + 1.0, r
            if r["plan_bar"]:
                assert sum(r["plan_bar"]) <= rf - 1.0, r
        # One scale on both wings: width per weight point is one number.
        scale = set()
        for r in rows.values():
            for side, bar in (("today", "today_bar"), ("plan", "plan_bar")):
                if r[bar] and r[side]:
                    pct = float(r[side][0].rstrip("%"))
                    if pct > 0:
                        scale.add(round(r[bar][1] / pct, 1))
        assert len(scale) == 1, scale
        # A line the plan sells has a red name and no plan wing; a line bought
        # from nothing has no today wing; every other line has both figures.
        for n, r in rows.items():
            if r["name_fill"] == RED:
                assert r["plan"] is None and r["plan_bar"] is None, n
            else:
                assert r["plan"] is not None and r["plan_bar"] is not None, n
            assert (r["today"] is None) == (r["today_bar"] is None), n
            if r["plan"] is not None:
                assert r["gap"] is not None, n
        # Bands are fills; a stroke tying a label to its slice was an unfilled path.
        assert not re.search(r'<path[^>]*fill="none"', svg), "a leader stroke is back"

    def test_per_holding_card(self):
        self._check(_block(_build_diversification(_ctx())["html"], "Per-holding target"))

    def test_asset_class_card(self):
        self._check(_block(_build_diversification(_ctx())["html"], "Asset class"))

    def test_geography_card(self):
        self._check(_block(_build_diversification(_ctx())["html"], "Equity geography"))
