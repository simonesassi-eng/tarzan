"""Shared formatting helpers for export surfaces (Excel, newsletter).

Kept tiny on purpose: only utilities that need consistent rendering
between the spreadsheet and the email so end users see the same
notation everywhere.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Color taxonomy — single source of truth for asset-class / geography colors
# ---------------------------------------------------------------------------
# Both the Excel dashboard and the HTML newsletter must color the same asset
# class / region identically. Previously each surface kept its own copy and
# they had drifted (Crypto and Alternative rendered different colors in Excel
# vs the email). Define the palette once here, as bare 6-hex codes (the form
# openpyxl wants); use css()/css_map() for the "#RRGGBB" form the email needs.

ASSET_CLASS_COLORS: dict[str, str] = {
    "Equities": "1D4ED8",
    "Fixed Income": "A16207",
    "Cash & Cash Equivalents": "15803D",
    "Gold": "CA8A04",
    "Commodities": "C2410C",
    "Crypto": "7C3AED",
    "Alternative": "475569",
}

# Soft background tints for asset-class chips/rows in the newsletter.
ASSET_CLASS_BG: dict[str, str] = {
    "Equities": "EEF2FF",
    "Fixed Income": "FEF3C7",
    "Cash & Cash Equivalents": "DCFCE7",
    "Gold": "FEF3C7",
    "Commodities": "FEF3C7",
    "Crypto": "EEF2FF",
    "Alternative": "F1F5F9",
}

GEO_COLORS: dict[str, str] = {
    "USA": "1D4ED8",
    "Eurozone EMU": "A16207",
    "Dev ex-USA ex-EMU ex-JP": "15803D",
    "Emerging Markets": "C2410C",
    "Japan": "7C3AED",
}

# Display/iteration order for asset classes across both surfaces. A class not
# listed here still renders — see asset_class_order() — so no holding is ever
# silently dropped from a report.
_ASSET_CLASS_BASE_ORDER: list[str] = [
    "Equities",
    "Fixed Income",
    "Cash & Cash Equivalents",
    "Gold",
    "Commodities",
    "Crypto",
    "Alternative",
]


def css(hex6: Optional[str], default: str = "#5B5BD6") -> str:
    """Return a CSS ``#RRGGBB`` string from a bare 6-hex code."""
    if not hex6:
        return default
    return hex6 if hex6.startswith("#") else f"#{hex6}"


def asset_class_color(name: str, *, css_form: bool = False, default: str = "5B5BD6") -> str:
    """Canonical color for an asset class. Bare hex by default; CSS form
    (``#RRGGBB``) when ``css_form=True``."""
    raw = ASSET_CLASS_COLORS.get(name, default)
    return css(raw) if css_form else raw


def asset_class_bg(name: str, *, css_form: bool = False, default: str = "EEF2FF") -> str:
    """Canonical background tint for an asset class."""
    raw = ASSET_CLASS_BG.get(name, default)
    return css(raw) if css_form else raw


def geo_color(name: str, *, css_form: bool = False, default: str = "5B5BD6") -> str:
    """Canonical color for a geography/region."""
    raw = GEO_COLORS.get(name, default)
    return css(raw) if css_form else raw


def asset_class_order(present: Optional[list[str]] = None) -> list[str]:
    """Asset classes in canonical display order.

    Any class in ``present`` that is not in the base order is appended at
    the end (alphabetically) so a newly added asset class is never silently
    dropped from a report when iterating in order.
    """
    if not present:
        return list(_ASSET_CLASS_BASE_ORDER)
    extra = sorted(set(present) - set(_ASSET_CLASS_BASE_ORDER))
    return [c for c in _ASSET_CLASS_BASE_ORDER if c in present] + extra


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
    original = str(name).strip()
    s = original
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
    # Fund-series roman numeral right after the leading issuer token
    # ("Xtrackers II ...", "iShares III ..."): drop the numeral but keep
    # the issuer. Anchored at the start so a roman numeral elsewhere in
    # the name (rare, but e.g. a real "IV" qualifier) is left untouched.
    s = re.sub(r"^(\S+)\s+(?:II|III|IV)\b", r"\1", s, flags=re.IGNORECASE)
    # Drop a trailing share-class code only at the END of the name
    # ("... 1C", "... 5Dis"): a digit run followed by 1–3 letters. Anchored
    # at the end so a mid-name token like "3M" or "500" is never eaten.
    s = re.sub(r"\s+\d+[A-Za-z]{1,3}\s*$", " ", s)
    # Normalize separators and whitespace, trim dangling punctuation.
    s = re.sub(r"\s*[-–·]\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–·")
    # Fallback: if stripping emptied the name (it was entirely
    # boilerplate/share-class tokens), keep the original rather than
    # returning a blank cell.
    if not s:
        s = original
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
