"""Explicit same-share-class listings, never guesses based on ticker roots.

This small catalog supplements the existing same-root venue candidates. An
entry is usable only after its verification and with matching, causal cached
ISIN evidence. It performs no transport or discovery, and never changes the
canonical ticker or financial units.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ListingGroup:
    isin: str
    symbols: tuple[str, ...]
    currency: str
    share_class_figi: str
    verified_at: datetime


# OpenFIGI's ID_ISIN profile identifies both roots as BBG001SR7X43. Yahoo's
# observed EUR Xetra listing is DBXF.DE, not the guessed X15E.DE. Keep this
# knowledge dated: older frozen analyses must not discover future listings.
_GROUPS = (
    ListingGroup(
        isin="LU0290357507",
        symbols=("X15E.MI", "DBXF.DE"),
        currency="EUR",
        share_class_figi="BBG001SR7X43",
        verified_at=datetime(2026, 10, 22, 11, 52, 27, 909541, tzinfo=timezone.utc),
    ),
)


def _group_for(ticker: str) -> ListingGroup | None:
    symbol = str(ticker or "").strip().upper()
    return next((group for group in _GROUPS if symbol in group.symbols), None)


def _profile_matches(profile, group: ListingGroup, captured_at: datetime) -> bool:
    if not isinstance(profile, dict):
        return False
    if (
        profile.get("isin") != group.isin
        or profile.get("kind") != "ETF"
        or profile.get("status") != "VERIFIED"
        or profile.get("source") != "OpenFIGI"
        or profile.get("confidence") != "HIGH"
    ):
        return False
    provenance = profile.get("provenance")
    identifiers = profile.get("identifiers")
    tickers = profile.get("tickers")
    if (
        not isinstance(provenance, dict)
        or provenance.get("provider") != "OpenFIGI"
        or provenance.get("idType") != "ID_ISIN"
        or not isinstance(identifiers, dict)
        or identifiers.get("shareClassFIGI") != [group.share_class_figi]
        or not isinstance(tickers, list)
    ):
        return False
    roots = {str(ticker).upper().split(".", 1)[0] for ticker in tickers}
    if not {symbol.split(".", 1)[0] for symbol in group.symbols} <= roots:
        return False
    try:
        observed = datetime.fromisoformat(str(profile.get("observed_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return False
    return observed.tzinfo is not None and observed <= captured_at


def verified_siblings(ticker: str, *, isin: str | None = None) -> tuple[str, ...]:
    """Locally authorized alternative roots, visible to this LIVE capture."""
    group = _group_for(ticker)
    if group is None or (isin is not None and str(isin).strip().upper() != group.isin):
        return ()
    from tarzan import runtime

    context = runtime.context()
    captured_at = context.captured_at
    if (
        not runtime.allows_live_transport()
        or not context.allows_live_transport
        or captured_at.tzinfo is None
        or captured_at < group.verified_at
    ):
        return ()
    from tarzan.data import price_cache
    from tarzan.runtime.memo import memoized
    from tarzan.runtime.session import current_session

    def authorized():
        try:
            profile = price_cache.load_instrument_profile(
                group.isin, as_of=captured_at, allow_stale=True,
            )
        except (OSError, TypeError, ValueError):
            return False
        return _profile_matches(profile, group, captured_at)

    # Never retain a compatibility-global decision across independent runs.
    valid = (
        memoized(
            "listing_aliases.authority", (group.isin, captured_at),
            authorized, legacy={},
        )
        if current_session() is not None else authorized()
    )
    symbol = str(ticker).strip().upper()
    return tuple(other for other in group.symbols if other != symbol) if valid else ()


def alias_currency(ticker: str, source: str, *, isin: str) -> str | None:
    """Declared currency of an explicit alternative, not of guessed venues."""
    group = _group_for(ticker)
    if (
        group is None
        or str(isin or "").strip().upper() != group.isin
        or source == str(ticker).strip().upper()
        or source not in group.symbols
    ):
        return None
    return group.currency
