"""Pure text/number formatters for the newsletter (leaf → _constants only)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tarzan.export.newsletter._constants import PALETTE

def _eur(amount: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a number as a localised EUR amount: €1,234.56 / +€1,234.56."""
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    fmt = f",.{decimals}f"
    formatted = f"€{abs(amount):{fmt}}"
    if signed:
        sign = "+" if amount >= 0 else "−"
        return f"{sign}{formatted}"
    if amount < 0:
        return f"−{formatted}"
    return formatted

def _pct(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a percentage. Already in pp (e.g. 8.59 means 8.59%)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if signed:
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value):.{decimals}f}%"
    return f"{value:.{decimals}f}%"

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
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    v = float(value)
    av = abs(v)
    decimals = 2 if av < 100 else (1 if av < 1000 else 0)
    if signed:
        sign = "+" if v >= 0 else "−"
        return f"{sign}{av:.{decimals}f}%"
    return f"{v:.{decimals}f}%"

def _pct_smart(value: Optional[float], max_decimals: int = 1, signed: bool = False) -> str:
    """Format a percentage with adaptive precision: drop the decimal
    digits when the value is already integer (saves horizontal space).

    Example with ``max_decimals=1``:
      70.0  → "70%"
      71.7  → "71.7%"
      −1.6  → "−1.6%" (or "+1.7%" with signed=True)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    rounded = round(float(value), max_decimals)
    is_integer = abs(rounded - round(rounded)) < 10 ** (-(max_decimals + 1))
    decimals = 0 if is_integer else max_decimals
    if signed:
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value):.{decimals}f}%"
    return f"{value:.{decimals}f}%"

def _signed_pp(value: Optional[float], decimals: int = 1) -> str:
    """Format a signed delta in percentage points (no % sign)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"

def _display_ticker(symbol: Optional[str]) -> Optional[str]:
    """Return the exact resolved provider ticker for human-facing output.

    Exchange suffixes and index markers are part of the instrument identity
    and must not be stripped after preprocessing.  Synthetic portfolio/mix
    labels remain hidden because they are not provider tickers.
    """
    if not symbol:
        return None
    ticker = str(symbol).strip()
    if not ticker or ticker.upper() in ("PORTFOLIO", "—", "NAN"):
        return None
    if "/" in ticker or " " in ticker:
        return None
    return ticker

def _semaphore(delta: Optional[float], tolerance: float) -> str:
    """Return 'green' / 'amber' / 'red' based on |delta| vs tolerance."""
    if delta is None or (isinstance(delta, float) and pd.isna(delta)):
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

