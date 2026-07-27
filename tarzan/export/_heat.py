"""Diverging cell backgrounds for the returns grids.

Scaled per COLUMN, over the rows of the table being rendered — not over a
global range and not over both grids pooled.

Why per column: the windows differ by an order of magnitude (a 1D move is
±3%, a 5Y cumulative is ±140%). One shared scale saturates on the long
cumulatives and leaves every short window white, which is the opposite of
useful.

Why per table: pooling the holdings grid with the watchlist means neither ever
saturates its own ramp. The best one-month holding prints nearly white because
the pooled maximum belongs to an instrument the reader does not own, and the
worst three-month candidate prints nearly white because the pooled minimum
belongs to a holding. Each table answers "who did best and worst over this
window, here".

The ramp interpolates from the card surface toward a saturated end, so it works
on a dark palette: a light-surface ramp (white → green) inverts into a glowing
block on dark and swamps the figure sitting on it.
"""

from __future__ import annotations

from typing import Iterable, Optional

from tarzan.export._palette import PALETTE

# The 1D cell also carries the session sparkline. A fully saturated background
# there competes with the chart drawn on top of it, so that one column's scale
# is widened, which caps the tint at roughly 45% of the ramp while preserving
# the ranking.
DAY_DAMP = 2.2


def column_scale(values: Iterable[Optional[float]]) -> tuple[float, float]:
    """``(most_negative, most_positive)`` over one column, ignoring blanks."""
    nums = [float(v) for v in values if v is not None]
    neg = min([v for v in nums if v < 0] or [-1.0])
    pos = max([v for v in nums if v > 0] or [1.0])
    return neg, pos


def _mix(base: str, target: str, t: float) -> str:
    def ch(s, i):
        return int(s[1 + 2 * i:3 + 2 * i], 16)
    out = tuple(round(ch(base, i) + t * (ch(target, i) - ch(base, i)))
                for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*out)


# The two ends of the ramp. Deliberately NOT the palette's signal green and red:
# those are the colours a FIGURE is written in, and at full saturation behind a
# figure they fight it. These are the same hues held back, so a saturated cell
# still reads as a surface with a number on it.
_GREEN_END = "#1E8C55"
_RED_END = "#962D28"

# Above this share of the ramp a cell carries enough colour to hold the ink;
# below it the figure steps back to a mid grey. Two values, applied to every
# cell in the grid, so the ONLY thing that varies across the matrix is the
# background. A grid where the text colour also varies has two signals for one
# fact and reads as noise.
_INK_ABOVE = 0.30
_FIGURE_FAINT = "#A9BACD"


def heat(value: Optional[float], *, neg: float, pos: float,
         damp: float = 1.0) -> tuple[Optional[str], str]:
    """``(background, figure_colour)`` for one cell of a returns grid.

    ``value`` is signed percent; ``neg``/``pos`` are that column's extremes, so
    the column's worst value reaches the saturated red end and its best the
    saturated green end.

    Both halves come from here on purpose. When the caller chose the figure's
    colour itself -- green for a gain, red for a loss -- the grid carried the
    sign twice, once in the cell and once in the text, and the text colour
    changed from row to row for a reason unrelated to the tint. One ramp, one
    rule: the background says which way and how far, the figure is either ink or
    stepped back.
    """
    if value is None:
        return None, PALETTE["subtle"]
    span_neg = abs(neg) * damp or 1.0
    span_pos = abs(pos) * damp or 1.0
    v = float(value)
    if v >= 0:
        t = min(1.0, v / span_pos)
        target = _GREEN_END
    else:
        t = min(1.0, -v / span_neg)
        target = _RED_END
    figure = PALETTE["ink"] if t > _INK_ABOVE else _FIGURE_FAINT
    if t <= 0.01:
        return None, figure
    # Linear from the surface to the end of the ramp: the column's extreme is
    # the saturated colour, and everything else sits proportionally between. An
    # eased ramp with a floor made a +0.1% cell visibly tinted and a +8% cell
    # only two thirds of the way to the end.
    return _mix(PALETTE["card_alt"], target, t), figure


def heat_bg(value: Optional[float], *, neg: float, pos: float,
            damp: float = 1.0) -> Optional[str]:
    """Just the background from :func:`heat`, for callers that do not colour a
    figure (the 1D cell, which carries a sparkline)."""
    return heat(value, neg=neg, pos=pos, damp=damp)[0]


# heat_ink is gone: it existed to decide when a SIGN-coloured figure had to give
# way to ink, and the grids no longer sign-colour their figures at all. The
# figure colour comes back from heat() with the background it belongs to.


def rank_scale(values: Iterable[Optional[float]]) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` over one column, or None when there is nothing to rank."""
    nums = [float(v) for v in values if v is not None]
    if len(nums) < 2:
        return None
    lo, hi = min(nums), max(nums)
    return None if hi - lo < 1e-12 else (lo, hi)


def rank_bg(value: Optional[float], *, lo: float, hi: float,
            higher_is_better: bool) -> Optional[str]:
    """Background for a cell whose colour is a rank, not a sign.

    ``heat_bg`` reads the sign: positive is green, negative is red. That is the
    right rule for returns and the wrong one for risk. A volatility of 10% is
    positive and not good; a max drawdown of −7% is negative and better than
    −21%. Here the colour comes from where the value sits inside its column's
    observed range, with the direction stated by the caller — so the greenest
    cell in a column is the best of the series shown, whatever its sign.

    The midpoint of the range is the neutral point. That is a property of the
    values present, not a threshold anyone set: nothing in the engine says what
    a "good" Sharpe is, and inventing one would print an opinion as a fact.
    """
    if value is None:
        return None
    span = hi - lo
    if span < 1e-12:
        return None
    t = (float(value) - lo) / span
    good = t if higher_is_better else 1.0 - t
    # Distance from the middle, so the ends of the range saturate and the
    # middle stays on the card surface.
    strength = abs(good - 0.5) * 2.0
    if strength <= 0.02:
        return None
    target = PALETTE["green"] if good >= 0.5 else PALETTE["red"]
    return _mix(PALETTE["card"], target, 0.08 + 0.52 * (strength ** 0.9))


def rank_ink(value: Optional[float], *, lo: float, hi: float,
             higher_is_better: bool) -> Optional[str]:
    """Palette ink once the rank tint is strong enough to swallow a coloured
    figure, otherwise None so the caller keeps its own colour."""
    if value is None:
        return None
    span = hi - lo
    if span < 1e-12:
        return None
    t = (float(value) - lo) / span
    good = t if higher_is_better else 1.0 - t
    if abs(good - 0.5) * 2.0 >= _INK_ABOVE:
        return PALETTE["ink"]
    return None
