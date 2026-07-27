"""Tests for the rank-based conditional formatting used by the risk table.

The returns grids colour by sign: positive green, negative red. Risk metrics
cannot use that rule -- a positive volatility is not good news, and a −7%
drawdown beats a −21% one. These tests pin the direction, since getting it
backwards would print a confident green on the worst series in the table.
"""

from __future__ import annotations

from tarzan.export._heat import rank_bg, rank_ink, rank_scale
from tarzan.export._palette import PALETTE


def _closer_to(colour: str, target: str, other: str) -> bool:
    """True when ``colour`` is nearer ``target`` than ``other`` in RGB."""
    def rgb(c):
        return [int(c[1 + 2 * i:3 + 2 * i], 16) for i in range(3)]

    a, b, c = rgb(colour), rgb(target), rgb(other)
    return sum((x - y) ** 2 for x, y in zip(a, b)) < \
        sum((x - y) ** 2 for x, y in zip(a, c))


GREEN, RED = PALETTE["green"], PALETTE["red"]


class TestRankScale:
    def test_returns_the_observed_range(self):
        assert rank_scale([1.0, 5.0, 3.0]) == (1.0, 5.0)

    def test_blanks_are_ignored(self):
        assert rank_scale([None, 2.0, None, 4.0]) == (2.0, 4.0)

    def test_a_single_value_cannot_be_ranked(self):
        assert rank_scale([3.0]) is None
        assert rank_scale([None, 3.0]) is None

    def test_a_flat_column_cannot_be_ranked(self):
        assert rank_scale([2.0, 2.0, 2.0]) is None


class TestPolarity:
    def test_higher_is_better_greens_the_top(self):
        top = rank_bg(10.0, lo=0.0, hi=10.0, higher_is_better=True)
        bottom = rank_bg(0.0, lo=0.0, hi=10.0, higher_is_better=True)
        assert _closer_to(top, GREEN, RED)
        assert _closer_to(bottom, RED, GREEN)

    def test_lower_is_better_greens_the_bottom(self):
        # Volatility: 7% is the good end of a 7..33 column.
        top = rank_bg(33.0, lo=7.0, hi=33.0, higher_is_better=False)
        bottom = rank_bg(7.0, lo=7.0, hi=33.0, higher_is_better=False)
        assert _closer_to(bottom, GREEN, RED)
        assert _closer_to(top, RED, GREEN)

    def test_negative_values_rank_by_position_not_by_sign(self):
        # Max drawdown: every value is negative, −4% is the good end.
        good = rank_bg(-4.0, lo=-51.0, hi=-4.0, higher_is_better=True)
        bad = rank_bg(-51.0, lo=-51.0, hi=-4.0, higher_is_better=True)
        assert _closer_to(good, GREEN, RED)
        assert _closer_to(bad, RED, GREEN)

    def test_the_middle_of_the_range_is_left_unshaded(self):
        assert rank_bg(5.0, lo=0.0, hi=10.0, higher_is_better=True) is None

    def test_a_blank_is_left_unshaded(self):
        assert rank_bg(None, lo=0.0, hi=10.0, higher_is_better=True) is None

    def test_a_flat_range_is_left_unshaded(self):
        assert rank_bg(2.0, lo=2.0, hi=2.0, higher_is_better=True) is None


class TestRankInk:
    def test_ink_flips_at_the_saturated_ends(self):
        assert rank_ink(10.0, lo=0.0, hi=10.0, higher_is_better=True) == PALETTE["ink"]
        assert rank_ink(0.0, lo=0.0, hi=10.0, higher_is_better=True) == PALETTE["ink"]

    def test_ink_is_left_alone_near_the_middle(self):
        assert rank_ink(5.4, lo=0.0, hi=10.0, higher_is_better=True) is None

    def test_blanks_keep_their_colour(self):
        assert rank_ink(None, lo=0.0, hi=10.0, higher_is_better=True) is None
