"""Pure text/number formatters for the newsletter (leaf → _constants only)."""

from __future__ import annotations

from html import escape
from typing import Optional

from markupsafe import Markup
import pandas as pd

from tarzan.export._format import base_symbol


def _escaped_text(value) -> Markup:
    """Typed, already-escaped text, never trusted provider/broker markup.

    Preserve the escaped-text context API while making the ownership explicit:
    Jinja must not escape these values twice. Convert to plain str first so a
    supplied __html__ hook/Markup cannot bypass the actual text escape.
    """
    return Markup(escape(str(value)))


def is_missing(value) -> bool:
    """True for an Unavailable display value: ``None`` or a float NaN.

    The one predicate behind every "render '—' when there's nothing to show"
    guard in the newsletter, so a numeric zero is never mistaken for missing
    (the numeric-zero != Unavailable invariant)."""
    return value is None or (isinstance(value, float) and pd.isna(value))


def _eur(amount: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a number as a localised EUR amount: €1,234.56 / +€1,234.56."""
    if is_missing(amount):
        return "—"
    fmt = f",.{decimals}f"
    formatted = f"€{abs(amount):{fmt}}"
    if signed:
        sign = "+" if amount >= 0 else "−"
        return f"{sign}{formatted}"
    if amount < 0:
        return f"−{formatted}"
    return formatted

def _sign_for(value: float, decimals: int) -> str:
    """The sign to print for a value that will be rounded to ``decimals``.

    Empty when the rounded figure is zero. Without it a residual weight of
    -0.004% printed as "\u22120.0%", and a signed zero reads as a real, if tiny,
    move in the negative direction rather than as nothing at all.
    """
    if abs(round(float(value), decimals)) < 10 ** -9:
        return ""
    return "+" if value > 0 else "\u2212"


def _signed(value: float, decimals: int = 2, *, thousands: bool = False) -> str:
    """A signed bare number with the typographic minus.

    Python's ``:+`` format flag emits an ASCII hyphen, which is narrower and
    sits lower than the U+2212 every other figure in the issue uses. This is the
    plain-number counterpart of :func:`_pct`.
    """
    grp = "," if thousands else ""
    sign = _sign_for(value, decimals) or "+"
    return f"{sign}{abs(float(value)):{grp}.{decimals}f}"


def _pct(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a percentage. Already in pp (e.g. 8.59 means 8.59%).

    ``signed=False`` still routes a negative through :func:`_sign_for`, so it
    prints the minus SIGN rather than an ASCII hyphen and a value that rounds to
    zero prints as zero. A position sold to nothing came out as "-0.0%": a
    hyphen where the rest of the issue uses U+2212, in front of a negative zero
    that reads as a small move down rather than as nothing.
    """
    if is_missing(value):
        return "—"
    if signed:
        return f"{_sign_for(value, decimals)}{abs(value):.{decimals}f}%"
    sign = "\u2212" if _sign_for(value, decimals) == "\u2212" else ""
    return f"{sign}{abs(value):.{decimals}f}%"

def _pct_compact(value: Optional[float], signed: bool = True) -> str:
    """Percentage with width-aware precision for the dense returns grids.

    The 8-column returns tables (snapshot + performance) must fit eight
    values inside a 600px email. Two decimals are fine for normal
    returns, but three-digit values like ``+126.17%`` overflow the
    fixed cell width. So we taper precision by magnitude:

        |v| < 100   → 2 decimals   (+8.59%, −1.62%)
        |v| < 1000  → 1 decimal    (+126.2%)
        |v| >= 1000 → 0 decimals   (+1234%)

    This trims width exactly where it's needed without losing
    meaningful precision (a few basis points on a >100% multi-year
    return are noise).
    """
    if is_missing(value):
        return "—"
    v = float(value)
    av = abs(v)
    decimals = 2 if av < 100 else (1 if av < 1000 else 0)
    if signed:
        return f"{_sign_for(v, decimals)}{av:.{decimals}f}%"
    return f"{v:.{decimals}f}%"

def _pct_smart(value: Optional[float], max_decimals: int = 1, signed: bool = False) -> str:
    """Format a percentage with adaptive precision: drop the decimal
    digits when the value is already integer (saves horizontal space).

    Example with ``max_decimals=1``:
      70.0  → "70%"
      71.7  → "71.7%"
      −1.6  → "−1.6%" (or "+1.7%" with signed=True)
    """
    if is_missing(value):
        return "—"
    rounded = round(float(value), max_decimals)
    is_integer = abs(rounded - round(rounded)) < 10 ** (-(max_decimals + 1))
    decimals = 0 if is_integer else max_decimals
    if signed:
        return f"{_sign_for(value, decimals)}{abs(value):.{decimals}f}%"
    return f"{value:.{decimals}f}%"

def _signed_pp(value: Optional[float], decimals: int = 1) -> str:
    """Format a signed delta in percentage points (no % sign)."""
    if is_missing(value):
        return "—"
    return f"{_sign_for(value, decimals)}{abs(value):.{decimals}f}"

def _display_ticker(symbol: Optional[str]) -> Optional[str]:
    """The ticker pin shown in the body: the resolved provider symbol without
    its exchange suffix.

    The suffix is part of the instrument's identity and is never dropped from
    the DATA -- the appendix's instrument reference prints the full canonical
    and intraday symbols, and the semantic gate checks the frames rather than
    this string. In the body it is noise: ".MI" on every pin costs three
    characters of a 9px label that has to sit beside a name, and where two funds
    share a symbol (the two MATEs) the name and currency tell them apart.

    Synthetic portfolio/mix labels stay hidden because they are not provider
    tickers.
    """
    if not symbol:
        return None
    ticker = str(symbol).strip()
    if not ticker or ticker.upper() in ("PORTFOLIO", "\u2014", "NAN"):
        return None
    if "/" in ticker or " " in ticker:
        return None
    return base_symbol(ticker) or ticker
