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

A symbol is not an identity: MATE is a fund on Nasdaq (US56164V3033) and the
Milan symbol of a different UCITS (IE000BS31R00). Where the curated catalog
lists two instruments under one symbol, each is carried as a QUALIFIED symbol,
``MATE@IE000BS31R00``, so every place that still keys on the symbol string
(fetch keys, memo keys, the benchmark catalog) gets a unique key, and every
resolver reads the ISIN out of it. ``normalize_ticker`` strips the qualifier
like it strips a venue suffix, so the symbol a reader sees stays ``MATE``.
"""

from __future__ import annotations

import re
from typing import Optional


QUALIFIER = "@"
_ISIN_SHAPE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")


def normalize_isin(raw: Optional[str]) -> str:
    """Uppercased, whitespace/hyphen-stripped ISIN; '' for anything that is not
    ISIN-shaped (blank, 'nan', a placeholder such as 'N/A', a ticker typed into
    the ISIN column). A non-ISIN must not act as an identity: it would
    contradict the holding's catalog row and strip its class, name and TER.

    The single canonical ISIN normalizer; ``contracts.validation`` re-exports it.
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper().replace(" ", "").replace("-", "")
    return s if _ISIN_SHAPE.fullmatch(s) else ""


def is_isin(raw: Optional[str]) -> bool:
    """Whether ``raw`` normalizes to the ISIN shape (2 letters, 9 alnum, 1 digit)."""
    return bool(normalize_isin(raw))


def qualify(ticker: str, isin: Optional[str]) -> str:
    """``TICKER@ISIN`` for a symbol two instruments share; the plain symbol
    when there is no ISIN to tell them apart."""
    t, i = str(ticker or "").strip(), normalize_isin(isin)
    return f"{t}{QUALIFIER}{i}" if t and i else t


def split_qualified(raw: Optional[str]) -> tuple[str, str]:
    """``(symbol, isin)`` of a possibly qualified symbol: ``MATE@IE000BS31R00``
    -> ``("MATE", "IE000BS31R00")``; a plain symbol -> ``(symbol, "")``. A
    malformed qualifier is not an ISIN and is dropped with the separator."""
    s = "" if raw is None else str(raw).strip()
    if s.lower() == "nan":
        return "", ""
    if QUALIFIER not in s:
        return s, ""
    symbol, _, tail = s.partition(QUALIFIER)
    return symbol.strip(), (normalize_isin(tail) if is_isin(tail) else "")


def normalize_ticker(raw: Optional[str]) -> str:
    """Bare ticker: uppercased, ISIN qualifier and exchange suffix stripped
    (``VWCE.DE`` -> ``VWCE``, ``MATE@IE000BS31R00`` -> ``MATE``); '' for
    blank/'nan'."""
    s = split_qualified(raw)[0]
    if not s:
        return ""
    return s.split(".")[0].strip().upper()


def instrument_key(isin: Optional[str], ticker: Optional[str] = None) -> str:
    """The one canonical instrument key. ISIN when present (a qualified
    symbol carries one), else ``TICKER:<bare-symbol>``, else ''. See module
    docstring."""
    i = normalize_isin(isin) or split_qualified(ticker)[1]
    if i:
        return i
    t = normalize_ticker(ticker)
    return f"TICKER:{t}" if t else ""
