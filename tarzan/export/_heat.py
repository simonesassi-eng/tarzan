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


def heat_bg(value: Optional[float], *, neg: float, pos: float,
            damp: float = 1.0) -> Optional[str]:
    """Background for one cell, or None when there is nothing to shade.

    ``value`` is signed percent. ``neg``/``pos`` are that column's extremes, so
    the column's worst value reaches the saturated red and its best the
    saturated green.
    """
    if value is None:
        return None
    span_neg = abs(neg) * damp or 1.0
    span_pos = abs(pos) * damp or 1.0
    v = float(value)
    if v >= 0:
        t = min(1.0, v / span_pos)
        target = PALETTE["green"]
    else:
        t = min(1.0, -v / span_neg)
        target = PALETTE["red"]
    if t <= 0.01:
        return None
    # Ease the low end so near-zero values stay close to the surface instead of
    # picking up a tint that reads as a signal.
    return _mix(PALETTE["card"], target, 0.10 + 0.62 * (t ** 0.85))


# Above this share of the ramp the background carries enough colour that a
# green figure on green (or red on red) loses contrast, so the text switches to
# the palette ink instead.
_INK_ABOVE = 0.45


def _intensity(value: Optional[float], *, neg: float, pos: float,
               damp: float = 1.0) -> float:
    if value is None:
        return 0.0
    v = float(value)
    span = (abs(pos) if v >= 0 else abs(neg)) * damp or 1.0
    return min(1.0, abs(v) / span)


def heat_ink(value: Optional[float], *, neg: float, pos: float,
             damp: float = 1.0) -> Optional[str]:
    """Palette ink when the tint is strong enough to swallow a coloured figure,
    otherwise None so the caller keeps its own sign colour."""
    if _intensity(value, neg=neg, pos=pos, damp=damp) >= _INK_ABOVE:
        return PALETTE["ink"]
    return None


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
