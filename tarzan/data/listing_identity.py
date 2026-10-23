"""Run-owned listing signatures already received in Yahoo's batched quotes.

An exact full name, quote type and currency can corroborate a same-root listing.
This is relative evidence, not a verified ISIN mapping. Missing or contradictory
facts never trigger an extra provider request and never become cross-run authority.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Optional


_NAMESPACE = "market.quote_identity"
_GENERIC_NAME_WORDS = frozenset({
    "acc", "accumulating", "bond", "bonds", "class", "dist", "distributing",
    "equity", "etf", "eur", "europe", "european", "fund", "gbp", "global",
    "growth", "hedged", "income", "index", "international", "market", "portfolio",
    "return", "stock", "total", "ucits", "usd", "world",
})
_QUOTE_TYPES = frozenset({"ETF", "EQUITY", "MUTUALFUND", "BOND", "ETC", "ETN"})


def qualified_symbol(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper()
    return symbol if re.fullmatch(r"[A-Z0-9_]+\.[A-Z0-9]+", symbol) else None


@dataclass(frozen=True)
class QuoteIdentity:
    symbol: str
    currency: Optional[str] = None
    full_name: Optional[str] = None
    quote_type: Optional[str] = None
    isin: Optional[str] = None
    malformed: bool = False
    conflicting: bool = False


def _text(record: Mapping, key: str, *, name: bool = False):
    value = record.get(key)
    if value is None:
        return None, False
    if not isinstance(value, str):
        return None, True
    value = " ".join(value.split())
    return (value.casefold() if name else value.upper()) or None, False


def remember_quote_identities(records, requested_symbols) -> None:
    """Retain immutable facts for exact requested symbols in the active owner."""
    from tarzan.runtime.session import current_session

    session = current_session()
    if session is None:
        return
    requested = {symbol for value in requested_symbols if (symbol := qualified_symbol(value))}
    owner = session.providers
    with owner.lock:
        store = owner.store(_NAMESPACE)
        for record in records:
            if not isinstance(record, Mapping):
                continue
            symbol = qualified_symbol(record.get("symbol"))
            if symbol not in requested:
                continue
            values, malformed = {}, False
            for field, key in (
                ("currency", "currency"), ("full_name", "longName"),
                ("quote_type", "quoteType"), ("isin", "isin"),
            ):
                values[field], bad = _text(record, key, name=field == "full_name")
                malformed |= bad
            previous = store.get(symbol)
            conflicting = False
            if previous is not None:
                conflicting = previous.conflicting
                malformed |= previous.malformed
                for field, value in values.items():
                    old = getattr(previous, field)
                    conflicting |= value is not None and old is not None and value != old
                    if value is None:
                        values[field] = old
            store[symbol] = QuoteIdentity(
                symbol=symbol, **values, malformed=malformed, conflicting=conflicting,
            )


def quote_identities(symbols: Iterable[str]) -> dict[str, QuoteIdentity]:
    """Read already-received facts only; no fallback, legacy global or network."""
    from tarzan.runtime.session import current_session

    session = current_session()
    if session is None:
        return {}
    owner = session.providers
    with owner.lock:
        store = owner.store(_NAMESPACE)
        return {
            symbol: store[symbol]
            for value in symbols
            if (symbol := qualified_symbol(value)) and symbol in store
        }


def reset_quote_identities() -> None:
    from tarzan.runtime.session import current_session

    session = current_session()
    if session is not None:
        with session.providers.lock:
            session.providers.store(_NAMESPACE).clear()


def relative_signature(identity: Optional[QuoteIdentity]):
    """All full-name tokens participate: no short name, fuzzy match or token drop."""
    if not isinstance(identity, QuoteIdentity) or identity.malformed or identity.conflicting:
        return None
    if identity.currency != "EUR" or identity.quote_type not in _QUOTE_TYPES:
        return None
    name = identity.full_name
    if not isinstance(name, str):
        return None
    name = " ".join(name.split()).casefold()
    words = re.findall(r"[^\W_]+", name)
    if len(words) < 3 or not any(
        word not in _GENERIC_NAME_WORDS and any(char.isalpha() for char in word)
        for word in words
    ):
        return None
    return name, identity.quote_type, identity.currency


def ambiguous_relative_signatures(subjects, identities: Mapping):
    """Names shared by different known ISINs cannot prove a share class."""
    identities_by_signature: dict[tuple, set[str]] = {}
    for symbol, isin in subjects:
        if not isinstance(isin, str) or not isin.strip():
            continue
        signature = relative_signature(identities.get(qualified_symbol(symbol)))
        if signature is not None:
            identities_by_signature.setdefault(signature, set()).add(isin.strip().upper())
    return {signature for signature, isins in identities_by_signature.items() if len(isins) > 1}
