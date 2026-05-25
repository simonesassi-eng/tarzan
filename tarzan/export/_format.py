"""Shared formatting helpers for export surfaces (Excel, newsletter).

Kept tiny on purpose: only utilities that need consistent rendering
between the spreadsheet and the email so end users see the same
notation everywhere.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def eur_smart(amount: Optional[float], signed: bool = False) -> str:
    """Compact EUR formatter: shows ``€9.6k`` / ``€215k`` / ``€1.2M``.

    Rules:
      |amount| < 1,000      → €<int>           (e.g. €356)
      |amount| < 1,000,000  → €<value>k        (1 decimal when non-integer)
      |amount| >= 1,000,000 → €<value>M        (1 decimal when non-integer)

    Always shows the sign with ``signed=True``; otherwise uses a
    leading minus glyph for negative values.
    """
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    abs_amt = abs(float(amount))
    if abs_amt < 1_000:
        body = f"€{abs_amt:,.0f}"
    elif abs_amt < 1_000_000:
        thousands = abs_amt / 1_000
        if abs(thousands - round(thousands)) < 0.05:
            body = f"€{thousands:.0f}k"
        else:
            body = f"€{thousands:.1f}k"
    else:
        millions = abs_amt / 1_000_000
        if abs(millions - round(millions)) < 0.05:
            body = f"€{millions:.0f}M"
        else:
            body = f"€{millions:.1f}M"
    if signed:
        sign = "+" if amount >= 0 else "−"
        return f"{sign}{body}"
    if amount < 0:
        return f"−{body}"
    return body
