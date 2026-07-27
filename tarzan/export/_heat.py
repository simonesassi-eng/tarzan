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
