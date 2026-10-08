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
    start = html.index(f'aria-label="{title}')
    return html[start:html.index("</svg>", start)]


class TestBridge:
    def test_every_line_held_today_is_in_today(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        for ticker in ("NTSG", "SGLD", "XDEV", "XMJP"):
            assert f">{ticker}<" in svg, ticker

    def test_the_lines_the_plan_sells_meet_in_one_sale(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        assert ">sell<" in svg
        assert ">60.0%<" in svg and ">\u20ac60k<" in svg  # synthetic

    def test_a_line_bought_from_nothing_is_named_once(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        assert svg.count(">AVWC<") == 1

    def test_weight_and_euros_side_by_side_and_the_gap_in_points(self):
        svg = _block(_build_diversification(_ctx())["html"], "Per-holding target")
        for figure in ("(30.0%)", "\u20ac30k", "(50.0%)", "\u20ac50k"):  # synthetic
            assert f">{figure}<" in svg, figure
        assert re.search(rf'fill="{RED}"[^>]*>\u221220.0<', svg)
        assert re.search(rf'fill="{GREEN}"[^>]*>0.0<', svg)

    def test_cash_is_banded_at_a_quarter_of_its_target(self):
        within = _block(_build_diversification(_ctx(cash=4900.0))["html"], "Asset class")
        beyond = _block(_build_diversification(_ctx(cash=5100.0))["html"], "Asset class")
        assert re.search(rf'fill="{GREEN}"[^>]*>\+\u20ac900<', within)
        assert re.search(rf'fill="{RED}"[^>]*>\+\u20ac1.1k<', beyond)  # synthetic


class TestLabelsSitOnTheirPills:
    """Every label stands level with its own pill, however thin, so nothing has to
    tie the two together: a thin pill gets the room its label needs around it."""

    _PILL = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="8" height="([\d.]+)"')
    _TAG = re.compile(r'<text x="([\d.]+)" y="([\d.]+)"[^>]*text-anchor="(end|start)">'
                      r'<tspan')

    def _check(self, svg: str) -> None:
        pills: dict = {}
        for px, py, ph in self._PILL.findall(svg):
            pills.setdefault(float(px), []).append(float(py) + float(ph) / 2)
        left_x, right_x = min(pills), max(pills)
        tags = self._TAG.findall(svg)
        left = [float(y) - 3.5 for _, y, a in tags if a == "end"]
        right = [float(y) - 3.5 for _, y, a in tags if a == "start"]
        assert len(left) == len(pills[left_x]) and len(right) == len(pills[right_x])
        for centres, labels in ((pills[left_x], left), (pills[right_x], right)):
            for c, y in zip(centres, labels):
                assert abs(c - y) < 0.2, (c, y)
            assert all(b - a >= 15.0 - 0.2 for a, b in zip(centres, centres[1:])), centres
        # Bands are filled paths; a stroke tying a label to its pill was an unfilled one.
        assert not re.search(r'<path[^>]*fill="none"', svg), "a leader stroke is back"

    def test_per_holding_card(self):
        self._check(_block(_build_diversification(_ctx())["html"], "Per-holding target"))

    def test_asset_class_card(self):
        svg = _block(_build_diversification(_ctx())["html"], "Asset class")
        # The tail rows (Total, Cash) carry tags too but no pill: check up to the rule
        # that separates them from the stacks.
        self._check(svg[:svg.index('<line x1="0"')])
