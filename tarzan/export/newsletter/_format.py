"""Pure text/number formatters for the newsletter (leaf → _constants only)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tarzan.export._format import base_symbol
from tarzan.export.newsletter._constants import PALETTE


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
    this string. In the body it is noise: no two instruments in an issue share a
    base symbol, and ".MI" on every pin costs three characters of a 9px label
    that has to sit beside a name.

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

def _semaphore(delta: Optional[float], tolerance: float) -> str:
    """Return 'green' / 'amber' / 'red' based on |delta| vs tolerance."""
    if is_missing(delta):
        return "muted"
    abs_d = abs(delta)
    if abs_d <= tolerance:
        return "green"
    if abs_d <= 2 * tolerance:
        return "amber"
    return "red"

def _semaphore_color(sema: str) -> str:
    return {"green": PALETTE["green"], "amber": PALETTE["amber"],
            "red": PALETTE["red"], "muted": PALETTE["muted"]}.get(sema, PALETTE["ink"])

def _colorize_pct(text: str) -> str:
    """HTML-escape ``text`` and wrap signed percentages AND percentage-point
    figures (e.g. +0.81%, -1.2%, +0.92pp, -4.53pp) in green/red spans, so both
    the market-context note (uses %) and the divergence note (uses pp) show
    moves in colour. Unsigned percentages (yield levels like 4.38%) are left
    neutral."""
    import html as _html
    import re as _re
    if not text:
        return ""
    esc = _html.escape(text)

    def _wrap(m):
        tok = m.group(0)
        neg = tok[0] in "-\u2212"
        col = PALETTE["red"] if neg else PALETTE["green"]
        return f'<span style="color:{col};font-weight:700;">{tok}</span>'

    # Signed number followed by %, pp, or "percentage point(s)" (bare beta like
    # "0.69" stays neutral). The model sometimes spells out "percentage points"
    # instead of "pp", so match both.
    return _re.sub(
        r"[+\-\u2212]\d+(?:[.,]\d+)?\s?(?:%|pp|percentage points?)", _wrap, esc)


def _colorize_pct_lines(text: str) -> str:
    """Like :func:`_colorize_pct`, but for the news digest's newline-
    separated items. Each line may start with a '[tag]' time marker (e.g.
    '[14:30]' or '[Today]', see _system_prompt); the tag, if present, is
    pulled out and styled distinctly (muted, monospace) from the headline.
    Returns <tr><td> rows, not <li>: the caller wraps this in a <table>,
    which renders far more consistently across email clients than native
    list bullets/spacing (Outlook in particular)."""
    import html as _html
    import re as _re
    if not text:
        return ""
    raw_lines = [ln for ln in text.split("\n") if ln.strip()]
    tag_re = _re.compile(r"^\[([^\]]{1,20})\]\s*")
    rows = []
    for i, line in enumerate(raw_lines):
        m = tag_re.match(line)
        tag, rest = (m.group(1), line[m.end():]) if m else (None, line)
        tag_html = (
            f'<span style="display:inline-block;min-width:52px;'
            f"font-family:'SF Mono',Consolas,monospace;"
            f'color:{PALETTE["subtle"]};font-size:11px;">'
            f'{_html.escape(tag)}</span> '
        ) if tag else ""
        border = (f'border-bottom:1px solid {PALETTE["border"]};'
                  if i < len(raw_lines) - 1 else "")
        rows.append(
            f'<tr><td style="padding:5px 2px;{border}">'
            f'{tag_html}<span style="color:{PALETTE["ink"]};">'
            f"{_colorize_pct(rest)}</span></td></tr>"
        )
    return "".join(rows)

