"""The Allocation section: one bridge per block, its figures built in.

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


class TestBridge:
    def test_every_line_held_today_is_in_today(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        for ticker in ("NTSG", "SGLD", "XDEV", "XMJP"):
            assert f">{ticker}<" in svg, ticker

    def test_the_lines_the_plan_sells_meet_in_one_sale(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        # The sale is one point, named once with its weight and euros.
        assert ">sell 60.0% \u20ac60k<" in svg  # synthetic

    def test_a_line_bought_from_nothing_is_named_once(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        assert svg.count(">AVWC<") == 1

    def test_weight_and_euros_side_by_side_and_the_gap_in_points(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        # "NTSG 30.0%" today, "50.0% NTSG" in the plan; euros in their own column.
        assert re.search(r">NTSG</tspan> 30\.0%<", svg) and ">\u20ac30k<" in svg  # synthetic
        assert re.search(r">50\.0% <tspan[^>]*>NTSG</tspan><", svg) and ">\u20ac50k<" in svg  # synthetic
        assert re.search(rf'fill="{RED}"[^>]*>\u221220.0<', svg)
        assert re.search(rf'fill="{GREEN}"[^>]*>0.0<', svg)

    def test_cash_is_banded_at_a_quarter_of_its_target(self):
        within = _block(_build_diversification(_ctx(cash=4900.0))["html"], "Asset class")
        beyond = _block(_build_diversification(_ctx(cash=5100.0))["html"], "Asset class")
        # The reserve is a tail row under the stacks; its gap carries the colour.
        assert re.search(rf'color:{GREEN};[^"]*">\+\u20ac900<', within)
        assert re.search(rf'color:{RED};[^"]*">\+\u20ac1.1k<', beyond)  # synthetic


class TestLabelsSitOnTheirSlices:
    """Every label stands level with its own slice, however thin: a slice is never
    thinner than one label row, so the stack grows rather than the labels
    drifting off it, and nothing has to tie the two together."""
    _SLICE = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="10" height="([\d.]+)"')
    _TEXT = re.compile(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*text-anchor="(end|start)">'
                       r'(?:<tspan|[\d.]+% <tspan)')

    def _check(self, svg: str, *, two: bool) -> None:
        from tarzan.export.newsletter import _sections_alloc as sa
        slices: dict = {}
        for px, py, ph in self._SLICE.findall(svg):
            slices.setdefault(float(px), []).append((float(py), float(py) + float(ph) + sa._SEAM))
        # The stacks' x is measured per block (the labels decide it), so read
        # it from the drawing: the two columns of 10px rects.
        xl, xr = min(slices), max(slices)
        left, right = slices[xl], slices[xr]
        # One-line labels end 7px before the TODAY stack and start 6px after
        # the PLAN stack (fixed or measured columns alike); two-line ones use
        # the module's fixed columns.
        cname, rname = (sa._CNAME, sa._RNAME) if two else (xl - 7.0, xr + sa._SW + 6.0)
        pitch = sa._PITCH2 if two else sa._PITCH1
        # Contiguous: each slice starts where the one above ended; none thinner
        # than one label row.
        for stack in (left, right):
            for (_a0, a1), (b0, _b1) in zip(stack, stack[1:]):
                assert abs(a1 - b0) < 0.15, stack
            assert all(b - a >= pitch - 0.2 for a, b in stack), stack
        # The name line of each label (two-line mode also prints a figures line
        # 11px lower; only the name line is level with the slice).
        names = [(float(x), float(y), a) for x, y, a in self._TEXT.findall(svg)]
        offset = -1.5 if two else 3.6
        lefts = sorted(y - offset for x, y, a in names if a == "end" and abs(x - cname) < 0.5)
        rights = sorted(y - offset for x, y, a in names if a == "start" and abs(x - rname) < 0.5)
        if two:
            # The left label's second line also ends in a tspan (the move); the
            # right label's second line is plain euros and does not match.
            lefts = lefts[::2]
        assert len(lefts) == len(left) and len(rights) == len(right), (lefts, rights)
        for stack, ys in ((left, lefts), (right, rights)):
            for (a, b), y in zip(stack, ys):
                assert abs((a + b) / 2 - y) < 0.2, ((a, b), y)
        # Bands are filled paths; a stroke tying a label to its slice was an unfilled one.
        assert not re.search(r'<path[^>]*fill="none"', svg), "a leader stroke is back"

    def test_per_holding_card(self):
        self._check(_block(_build_diversification(_ctx())["html"], "Per-holding target"), two=False)

    def test_asset_class_card(self):
        # Class names fit one line beside their figures on this fixture (two
        # short classes), so the block takes the ticker grammar.
        self._check(_block(_build_diversification(_ctx())["html"], "Asset class"), two=False)
