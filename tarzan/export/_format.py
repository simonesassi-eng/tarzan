"""Shared formatting helpers for the newsletter export surface.

Kept tiny on purpose: the color taxonomy, name shortening and number
formatting the email relies on, in one place so the rules stay consistent.
(Historically these were shared with an Excel dashboard, since removed; the
palette is still stored as bare 6-hex — ``css()`` prefixes it for HTML.)
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Color taxonomy — single source of truth for asset-class / geography colors
# ---------------------------------------------------------------------------
# The newsletter must color every asset class / region consistently. Define
# the palette once here as bare 6-hex codes; use css()/css_map() for the
# "#RRGGBB" form the email needs.

# Both sets are validated against the DARK card surface (#0C131B) in the
# canonical display order of taxonomy.CANONICAL_ORDER / GEO_ORDER, adjacent
# pairs: OKLCH lightness band, chroma floor, CVD separation (deutan/protan/
# tritan ΔE), normal-vision ΔE >= 15, and >= 3:1 contrast. They were originally
# picked for a light surface and never re-derived when the palette flipped to
# dark, which left three defects: Fixed Income (A16207) and Gold (CA8A04) sat
# ΔE 13 apart in normal vision -- two adjacent table rows a reader could not
# separate -- and Equities (1D4ED8) / Alternative (475569) drew at 2.8:1 and
# 2.5:1 against the card. Same story in the geography set, where USA (1D4ED8)
# and Japan (7C3AED) collapsed to ΔE 3.4 under protanopia.
#
# Re-picking any hue here means re-running the six checks in display order, not
# eyeballing the swatch.
ASSET_CLASS_COLORS: dict[str, str] = {
    "Equities": "2563EB",
    "Fixed Income": "0D9488",
    "Cash & Cash Equivalents": "15803D",
    "Gold": "CA8A04",
    "Commodities": "C2410C",
    "Crypto": "7C3AED",
    "Alternative": "64748B",
}

# Soft background tints for asset-class chips/rows in the newsletter.
GEO_COLORS: dict[str, str] = {
    "USA": "2563EB",
    "Eurozone EMU": "A16207",
    "Dev ex-USA ex-EMU ex-JP": "0D9488",
    "Emerging Markets": "C2410C",
    "Japan": "DB2777",
}

# Display/iteration order for asset classes across both surfaces. A class not
# listed here still renders — see asset_class_order() — so no holding is ever
# silently dropped from a report. Sourced from the single taxonomy registry
# (models.taxonomy) so every surface's orders live in one place.
from tarzan.models.taxonomy import ORDER_BASE as _ORDER_BASE

_ASSET_CLASS_BASE_ORDER: list[str] = list(_ORDER_BASE)


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
    # "Hedged" is NOT stripped: it is a real distinction (EUR-hedged vs
    # unhedged) worth keeping, abbreviated to "Hdgd" via _WORD_ABBREVIATIONS.
    r"\(?\bAccumulating\b\)?",
    r"\(?\bDistributing\b\)?",
    r"\(?\bAcc\b\)?",
    r"\(?\bDist\b\)?",
]

# Issuer name → standardized short form, applied to the leading word(s)
# of an instrument name. Keys are matched case-insensitively and anchored
# at the start. Single source of truth so every newsletter table abbreviates
# issuers identically.
_ISSUER_ABBREVIATIONS = {
    # Multi-word brands first — they must match before any single-word key
    # that could be their prefix (none today, but order keeps it robust).
    "Return Stacked": "Ret. Stack.",
    "BNP Paribas": "BNP Par.",
    "Alpha Architect": "Alpha Arch.",
    "Xtrackers": "Xtr.",
    "Invesco": "Inv.",
    "iShares": "iSh.",
    "Vanguard": "Van.",
    "Amundi": "Amu.",
    "Lyxor": "Lyx.",
    "Franklin": "Frk.",
    "WisdomTree": "WT",
    "VanEck": "VanEck",
    "Avantis": "Avant.",
    "Dimensional": "Dim.",
    "KraneShares": "KS",
    "Direxion": "Dir.",
    "FINECO AM": "Fineco",
    "Fineco AM": "Fineco",
}

# Multi-word phrases collapsed to a short form (applied AFTER separator
# normalisation, so a hyphen introduced here survives). Order matters only
# where one phrase is a substring of another; kept longest-first to be safe.
_PHRASE_ABBREVIATIONS = [
    ("Emerging Markets", "EM"),
    ("Minimum Volatility", "Min Vol"),
    ("Small Cap", "Sm Cap"),
    ("Equal Weight", "Eq.Wt"),
    ("Efficient Core", "Eff. Core"),
    ("Managed Futures", "Mgd Fut."),
    ("Market Neutral", "Mkt Neutral"),
    ("Roll Select", "Roll Sel."),
    ("Government Bond", "Govt Bond"),
    ("Mount Lucas", "Mt Lucas"),
    ("Multi Strategy", "Multi-Strat"),
    ("Inflation Linked", "Infl-Linked"),
]

# Single-word abbreviations. Case-insensitive, whole-word.
_WORD_ABBREVIATIONS = {
    "World": "Wrld", "Commodities": "Comm.", "Commodity": "Comm.",
    "Government": "Govt", "Aggregate": "Agg.", "Developed": "Dev.",
    "Enhanced": "Enh.", "Leveraged": "Lev.", "Strategy": "Strat.",
    "International": "Intl", "Momentum": "Mom.", "Quality": "Qual.",
    "Physical": "Phys.", "Bloomberg": "BBG", "Equity": "Eq.",
    "Equities": "Eq.", "Global": "Gl.", "Efficient": "Eff.",
    "Select": "Sel.", "Europe": "Eur.", "European": "Eur.",
    "America": "US", "Stocks": "Stk", "Futures": "Fut.",
    "Bitcoin": "BTC", "Hedged": "Hdgd",
}


def _apply_abbreviations(s: str) -> str:
    """Collapse the phrase then single-word abbreviations in ``s``.

    Kept separate from :func:`short_instrument_name` only for readability; it
    is called once, after issuer/noise/separator cleanup, so the introduced
    tokens (e.g. the hyphen in ``Infl-Linked``) are not re-split."""
    for phrase, short in _PHRASE_ABBREVIATIONS:
        s = re.sub(r"\b" + re.escape(phrase) + r"\b", short, s,
                   flags=re.IGNORECASE)

    def _word(m: "re.Match") -> str:
        w = m.group(0)
        for key, val in _WORD_ABBREVIATIONS.items():
            if key.lower() == w.lower():
                return val
        return w

    return re.sub(r"[A-Za-z][A-Za-z\-]+", _word, s)


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
    the same height as the Performance table. Kept here (shared) so every
    newsletter table renders instrument names identically.

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
    # Separators become spaces, EXCEPT a hyphen between digits: that one is part
    # of the name. "Eurozone Government Bond 7-10" is a maturity band, and
    # collapsing it printed "Bond 7 10", which reads as two separate numbers.
    s = re.sub(r"(?<!\d)\s*[-\u2013\u00b7]\s*(?!\d)", " ", s)
    s = re.sub(r"(?<=\d)\s*[\u2013\u00b7]\s*(?=\d)", "-", s)
    s = re.sub(r"\s+", " ", s).strip(" -–·")
    # Curated word/phrase abbreviations (World → Wrld, Government → Govt,
    # Hedged → Hdgd, …). Runs AFTER separator normalisation so a hyphen it
    # introduces (Infl-Linked) is not immediately re-split into two words.
    s = _apply_abbreviations(s)
    # Drop a trailing all-caps ticker echo like "(LMTH)" — the symbol already
    # leads the row — and any parenthesis left empty by the boilerplate strip.
    s = re.sub(r"\s*\([A-Z]{2,6}\)\s*$", "", s)
    s = re.sub(r"\(\s*\)", "", s)
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


def _taxonomy_names() -> dict:
    """Cached ISIN/ticker → curated name map. Empty when the taxonomy is
    unreadable, so presentation degrades to the broker's string rather than
    failing."""
    global _NAME_CACHE
    if _NAME_CACHE is None:
        try:
            from tarzan import config as _cfg
            _NAME_CACHE = _cfg.name_lookup() or {}
        except Exception:  # noqa: BLE001 — never break a render over a name
            _NAME_CACHE = {}
    return _NAME_CACHE


_NAME_CACHE: Optional[dict] = None


def display_instrument_name(isin: Optional[str], ticker: Optional[str],
                            raw_name: Optional[str] = None,
                            max_len: int = 40) -> str:
    """The instrument's readable name, preferring the curated taxonomy one.

    The holdings frame carries the broker's order-export description, and no
    amount of string cleaning turns "WS GL EFF C USD" into "WisdomTree Global
    Efficient Core" — the words are simply not there. The taxonomy names every
    instrument the portfolio can hold, so that is the source; the broker string
    is the fallback for an instrument the taxonomy has never seen, which is a
    real case worth reading rather than blanking.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    names = _taxonomy_names()
    for key in (normalize_isin(isin), normalize_ticker(ticker)):
        if key and key in names:
            return short_instrument_name(names[key], max_len)
    return short_instrument_name(raw_name, max_len)


def base_symbol(symbol: Optional[str]) -> str:
    """The provider symbol without its exchange suffix: ``XDEV.MI`` → ``XDEV``.

    Used for the ticker pins in the body, where the suffix is noise: no two
    instruments in an issue share a base symbol, and the full symbol — with the
    canonical and intraday feed it came from — is in the appendix's instrument
    reference. Stripping it in the body while keeping it in the audit table is
    the concept's rule, and it is what makes a ticker pin readable at 9px.
    """
    from tarzan.models.instrument_key import normalize_ticker

    return normalize_ticker(symbol) or ""


def greek_safe(label: str) -> str:
    """Wrap Greek letters so a CSS uppercase transform cannot fold them.

    ``text-transform:uppercase`` maps α to Α and β to Β — the Greek capitals,
    which in most fonts are drawn identically to Latin A and B. A tile labelled
    "α" then reads "A", and the reader has no way to know it was alpha. Uppercase
    is the right treatment for the Latin labels beside it, so the exception is
    scoped to the characters that break rather than dropped for the whole label.
    """
    out = []
    for ch in str(label or ""):
        if "\u0370" <= ch <= "\u03ff":
            out.append(f'<span style="text-transform:none;">{ch}</span>')
        else:
            out.append(ch)
    return "".join(out)
