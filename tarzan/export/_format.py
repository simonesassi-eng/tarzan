"""Shared formatting helpers for export surfaces (Excel, newsletter).

Kept tiny on purpose: only utilities that need consistent rendering
between the spreadsheet and the email so end users see the same
notation everywhere.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# Boilerplate phrases stripped from instrument display names before
# truncation, so the *distinctive* part of the name survives instead of
# being eaten by fund-structure noise. Order matters: multi-word phrases
# are removed first. All patterns match case-insensitively.
_NAME_NOISE_PATTERNS = [
    r"\bUCITS\b",
    r"\bETF\b",
    r"\bETC\b",
    r"\b(?:EUR|USD|GBP|CHF)\s+Hedged\b",
    r"\bHedged\b",
    r"\(?\bAccumulating\b\)?",
    r"\(?\bDistributing\b\)?",
    r"\(?\bAcc\b\)?",
    r"\(?\bDist\b\)?",
    # Fund-series roman numerals after the issuer ("Xtrackers II ...").
    r"\b(?:II|III|IV)\b",
]

# Issuer name → standardized short form, applied to the leading word(s)
# of an instrument name. Keys are matched case-insensitively and anchored
# at the start. Keep this list the single source of truth so Excel and the
# email abbreviate issuers identically.
_ISSUER_ABBREVIATIONS = {
    "Xtrackers": "Xtr.",
    "Invesco": "Inv.",
    "iShares": "iSh.",
    "Vanguard": "Van.",
    "Amundi": "Amu.",
    "Lyxor": "Lyx.",
    "Franklin": "Frk.",
    "WisdomTree": "WT",
    "VanEck": "VanEck",
    "FINECO AM": "Fineco",
    "Fineco AM": "Fineco",
}


def short_instrument_name(
    name: Optional[str], max_len: int = 40, abbreviate_issuer: bool = True
) -> str:
    """Standardized short form of an instrument display name.

    Strips common fund-structure boilerplate ("UCITS", "ETF", share
    class codes like "1C"/"5C", "EUR Hedged", "(Acc)", fund-series roman
    numerals), optionally abbreviates the issuer (Xtrackers → Xtr.,
    Invesco → Inv.), collapses leftover separators, then truncates to
    ``max_len`` with an ellipsis.

    Used to keep the newsletter Returns-snapshot rows single-line and
    the same height as the Performance table. Kept here (shared) so
    Excel and the email render instrument names identically.

    Returns an empty string for falsy input.
    """
    if not name:
        return ""
    s = str(name)
    if abbreviate_issuer:
        for issuer, abbr in _ISSUER_ABBREVIATIONS.items():
            # Anchor at start, require a word boundary so we only hit the
            # leading issuer token, not a substring elsewhere.
            pattern = r"^\s*" + re.escape(issuer) + r"\b"
            new_s, n = re.subn(pattern, abbr, s, flags=re.IGNORECASE)
            if n:
                s = new_s
                break
    for pat in _NAME_NOISE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    # Drop trailing share-class codes: "1C", "5C", "1Dis", etc.
    s = re.sub(r"\b\d+[A-Za-z]{1,3}\b", " ", s)
    # Normalize separators and whitespace, trim dangling punctuation.
    s = re.sub(r"\s*[-–·]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–·")
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip(" -–·") + "…"
    return s


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
