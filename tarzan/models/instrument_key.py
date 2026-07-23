"""Canonical instrument identity — one key, one resolver.

Across the codebase an instrument was matched "by ISIN, else by (uppercased)
ticker" in many places (the loader's per-holding targets, the orchestrator's
target application, the taxonomy lookups, the newsletter's symbol resolution).
That scattered two-step fallback is an ambiguous composite key: easy to get
subtly inconsistent (case, whitespace, suffix handling) and hard to reason
about.

This module defines the single canonical rule:

    instrument_key(isin, ticker) ->
        "<ISIN>"            when a well-formed ISIN is present (uppercased)
        "TICKER:<SYMBOL>"   otherwise, from the bare ticker (suffix-stripped,
                            uppercased)
        ""                  when neither is usable

ISIN is preferred because it is the globally-unique security identifier; the
``TICKER:`` sentinel keeps ticker-only rows (e.g. an index or a not-yet-held
target instrument) addressable without colliding with real ISINs.
"""

from __future__ import annotations

from typing import Optional


def normalize_isin(raw: Optional[str]) -> str:
    """Uppercased, whitespace/hyphen-stripped ISIN; '' for blank/'nan'.

    The single canonical ISIN normalizer; ``contracts.validation`` re-exports it.
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper().replace(" ", "").replace("-", "")
    return "" if s in ("", "NAN") else s


def normalize_ticker(raw: Optional[str]) -> str:
    """Bare ticker: uppercased, exchange suffix stripped (``VWCE.DE`` →
    ``VWCE``); '' for blank/'nan'."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.split(".")[0].strip().upper()


def instrument_key(isin: Optional[str], ticker: Optional[str] = None) -> str:
    """The one canonical instrument key. ISIN when present, else
    ``TICKER:<bare-symbol>``, else ''. See module docstring."""
    i = normalize_isin(isin)
    if i:
        return i
    t = normalize_ticker(ticker)
    return f"TICKER:{t}" if t else ""
