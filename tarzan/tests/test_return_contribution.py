"""Tests for the return-contribution waterfall.

The waterfall's last bar is a total, so the bars have to reconcile with a
figure the reader can find elsewhere in the issue. These tests pin that: the
steps sum to the total bar, and the total equals the portfolio's unrealized
return on cost basis -- the number printed on the STATE tile.

Network-free: contributions are derived from a hand-made holdings frame.
"""

from __future__ import annotations

import re

import pandas as pd

from tarzan.export._charts import waterfall
from tarzan.export.newsletter._sections_alloc import _build_return_contrib


class _Ctx:
    """Minimal stand-in for the newsletter context: the builder reads only
    ``metrics.holdings_df``."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.metrics = type("M", (), {"holdings_df": df})()


def _df(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """(ticker, cost_basis, gain) -> holdings frame."""
    return pd.DataFrame([
        {"isin": f"X{i:011d}", "ticker": t, "name": t,
         "cost_basis_eur": cost, "gain_eur": gain,
         "current_value": cost + gain,
         "weight_pct": 0.0, "gain_pct": (gain / cost * 100.0) if cost else 0.0}
        for i, (t, cost, gain) in enumerate(rows)
    ])


def _bar_pcts(svg: str) -> list[float]:
    """Every signed percentage the waterfall prints on a bar, in order.

    Axis tick labels are excluded: they are the only texts with one decimal,
    bar labels always carry two.
    """
    out = []
    for txt in re.findall(r"<text[^>]*>(.*?)</text>", svg):
        m = re.fullmatch(r"([+\u2212])(\d+\.\d\d)%", txt)
        if m:
            out.append(float(m.group(2)) * (-1 if m.group(1) == "\u2212" else 1))
    return out


class TestWaterfallReconciles:
    def test_steps_sum_to_the_total_bar(self):
        svg = waterfall([("A", 2.0), ("B", 1.0), ("C", -0.5)])
        vals = _bar_pcts(svg)
        assert vals[:3] == [2.0, 1.0, -0.5]
        assert vals[3] == 2.5, "last bar must be the running total"

    def test_one_rect_per_step_plus_the_total(self):
        svg = waterfall([("A", 1.0), ("B", -2.0)])
        assert svg.count("<rect") == 3

    def test_no_items_draws_nothing(self):
        assert waterfall([]) == ""


class TestContributionMeasure:
    def test_total_is_unrealized_return_on_cost(self):
        # 3 winners, 3 laggards and 2 more so the bridge bar is exercised.
        df = _df([("W1", 10_000.0, 900.0), ("W2", 10_000.0, 600.0),
                  ("W3", 10_000.0, 300.0), ("L1", 10_000.0, -100.0),
                  ("L2", 10_000.0, -200.0), ("L3", 10_000.0, -400.0),
                  ("O1", 10_000.0, 50.0), ("O2", 10_000.0, -25.0)])
        out = _build_return_contrib(_Ctx(df))
        expected = df["gain_eur"].sum() / df["cost_basis_eur"].sum() * 100.0
        assert _bar_pcts(out["chart_html"])[-1] == round(expected, 2)

    def test_holdings_outside_the_lists_get_one_bridge_bar(self):
        df = _df([("W1", 10_000.0, 900.0), ("W2", 10_000.0, 600.0),
                  ("W3", 10_000.0, 300.0), ("L1", 10_000.0, -100.0),
                  ("L2", 10_000.0, -200.0), ("L3", 10_000.0, -400.0),
                  ("O1", 10_000.0, 50.0), ("O2", 10_000.0, 70.0)])
        svg = _build_return_contrib(_Ctx(df))["chart_html"]
        assert "+2 more" in svg
        # 6 named movers + the bridge + the total.
        assert svg.count("<rect") == 8

    def test_no_bridge_bar_when_every_holding_is_named(self):
        df = _df([("W1", 10_000.0, 900.0), ("L1", 10_000.0, -100.0)])
        svg = _build_return_contrib(_Ctx(df))["chart_html"]
        assert "more" not in svg
        assert svg.count("<rect") == 3

    def test_bars_are_labelled_with_the_resolved_ticker(self):
        df = _df([("XDEV.MI", 10_000.0, 900.0)])
        svg = _build_return_contrib(_Ctx(df))["chart_html"]
        assert "XDEV.MI" in svg, "the exchange suffix is part of the identity"

    def test_zero_cost_basis_yields_no_section(self):
        df = _df([("A", 0.0, 0.0)])
        out = _build_return_contrib(_Ctx(df))
        assert out == {"winners": [], "laggards": [], "chart_html": ""}

    def test_a_holding_at_break_even_is_neither_winner_nor_laggard(self):
        df = _df([("A", 10_000.0, 900.0), ("B", 10_000.0, 0.0),
                  ("C", 10_000.0, -100.0)])
        out = _build_return_contrib(_Ctx(df))
        assert [w["name"] for w in out["winners"]] == ["A"]
        assert [l["name"] for l in out["laggards"]] == ["C"]
