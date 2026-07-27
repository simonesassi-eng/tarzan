"""Tests for the funding-proof flow diagram.

The diagram claims to be an identity check, so the tests pin the claim: the
last bar is the sum of the ones before it, terms the plan never used are absent
rather than drawn as zero bars, and a proof that does not close says so instead
of rounding itself into agreement.

Network-free: the funding dicts are the shape the rebalancer emits.
"""

from __future__ import annotations

import re

from tarzan.export._charts import funding_flow
from tarzan.export.newsletter._sections_alloc import _funding_flow_html

_OK_FLAGS = {
    "position_invariants_satisfied": True,
    "protected_cash_satisfied": True,
    "frozen_positions_satisfied": True,
}


def _funding(**over) -> dict:
    base = {
        "initial_cash_eur": 9_743.0,
        "protected_cash_eur": 9_700.0,
        "external_contribution_eur": 16_200.0,
        "gross_sales_eur": 0.0,
        "estimated_tax_eur": 0.0,
        "fees_eur": 57.0,
        "gross_purchases_eur": 16_143.0,
        "ending_cash_eur": 9_743.0,
        "residual_eur": 0.0,
        "equation_residual_eur": 0.0,
        **_OK_FLAGS,
    }
    base.update(over)
    return base


def _labels(svg: str) -> list[str]:
    """Bar labels, in draw order (the amounts and the footnote excluded)."""
    texts = re.findall(r"<text[^>]*>(.*?)</text>", svg)
    return [t for t in texts if not t.startswith(("\u20ac", "\u2192"))
            and not t.startswith("Protected")]


class TestFundingFlowPrimitive:
    def test_total_step_is_the_running_sum(self):
        svg = funding_flow([("A", 100.0, "neutral"), ("B", -40.0, "out"),
                            ("= End", 0.0, "total")])
        amounts = re.findall(r"<text[^>]*>(\u20ac[\d,.]+k?)</text>", svg)
        assert amounts[-1] == "\u20ac60"

    def test_no_steps_draws_nothing(self):
        assert funding_flow([]) == ""

    def test_every_step_gets_a_bar(self):
        svg = funding_flow([("A", 1.0, "neutral"), ("B", 2.0, "in"),
                            ("= End", 0.0, "total")])
        assert svg.count("<rect") == 3


class TestFundingFlowBuilder:
    def test_unused_terms_are_absent(self):
        svg = _funding_flow_html(_funding())
        labels = _labels(svg)
        assert labels == ["Cash", "+ Contribution", "\u2212 Purchases",
                          "\u2212 Fees", "= Ending cash"]
        assert "Sales" not in svg and "Tax" not in svg

    def test_sells_add_their_own_terms(self):
        svg = _funding_flow_html(_funding(
            gross_sales_eur=122_300.0, estimated_tax_eur=1_700.0,
            gross_purchases_eur=136_500.0, fees_eur=266.0))
        assert _labels(svg) == ["Cash", "+ Contribution", "+ Sales",
                                "\u2212 Purchases", "\u2212 Tax",
                                "\u2212 Fees", "= Ending cash"]

    def test_ending_bar_equals_the_engine_ending_cash(self):
        svg = _funding_flow_html(_funding())
        amounts = re.findall(r"<text[^>]*>(\u20ac[\d,.]+k?)</text>", svg)
        # 9,743 + 16,200 - 16,143 - 57 = 9,743
        assert amounts[-1] == "\u20ac9.7k"

    def test_an_open_equation_is_drawn_not_hidden(self):
        svg = _funding_flow_html(_funding(equation_residual_eur=-12.5))
        assert "\u00b1 Unexplained" in svg

    def test_a_closed_equation_adds_no_step(self):
        assert "Unexplained" not in _funding_flow_html(_funding())

    def test_residual_is_printed_to_the_cent(self):
        # eur_smart would round this to "€0" and the note would claim a close
        # that did not happen.
        svg = _funding_flow_html(_funding(residual_eur=-0.4))
        assert "residual \u2212\u20ac0.40" in svg

    def test_failed_invariants_are_stated(self):
        svg = _funding_flow_html(_funding(protected_cash_satisfied=False))
        assert "invariants NOT satisfied" in svg

    def test_satisfied_invariants_are_stated(self):
        assert "invariants satisfied" in _funding_flow_html(_funding())
