"""Canonical instrument ticker decisions exposed to local run artifacts."""

from __future__ import annotations

from collections.abc import Iterable

from tarzan.models.holding import Holding
from tarzan.models.instrument_key import normalize_isin


def _clean(value: object) -> str:
    return str(value or "").strip()


def _market_ticker(holding: Holding) -> str:
    """Return a real market symbol, excluding an ISIN placeholder."""
    ticker = _clean(holding.ticker)
    isin = normalize_isin(holding.isin)
    return "" if isin and ticker.upper() == isin else ticker


def _scope(holding: Holding) -> str:
    if holding.is_historical_only:
        return "Historical only"
    if holding.is_seeded_target:
        return "Rebalance target"
    return "Current portfolio"


def build_ticker_resolution_records(
    holdings: Iterable[Holding],
    *,
    historical_isins: Iterable[str] = (),
) -> tuple[dict[str, object], ...]:
    """Consolidate per-carrier ticker evidence into deterministic records.

    One row is emitted per ISIN (or per exact ticker when an ISIN is not
    available). ``historical_isins`` independently identifies instruments
    present in the effective order history, so an open holding can correctly
    report ``Current + Historical`` rather than losing its historical
    membership when carriers are deduplicated. Canonical and intraday refresh
    fields retain the latest dated market observations as complete UTC
    timestamps; request time never makes stale evidence appear fresh. Any
    unexpected symbol disagreement remains visible as ``CONFLICT`` instead of
    being hidden by iteration order.
    """
    from datetime import datetime, timezone

    def _as_utc(value: object) -> datetime | None:
        try:
            if hasattr(value, "to_pydatetime"):
                value = value.to_pydatetime()
            if not isinstance(value, datetime):
                return None
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    def _format_utc(value: datetime | None) -> str:
        if value is None:
            return ""
        return value.isoformat(timespec="seconds").replace("+00:00", "Z")

    historical_members = {
        normalized
        for value in historical_isins
        if (normalized := normalize_isin(value))
    }
    grouped: dict[str, dict[str, object]] = {}
    for holding in holdings:
        isin = normalize_isin(holding.isin)
        canonical = _market_ticker(holding)
        key = isin or (f"TICKER:{canonical.upper()}" if canonical else "")
        if not key:
            continue

        record = grouped.setdefault(key, {
            "isin": isin,
            "name": "",
            "scopes": set(),
            "portfolio_memberships": set(),
            "canonical_observation_times": set(),
            "intraday_observation_times": set(),
            "requested_tickers": set(),
            "canonical_tickers": set(),
            "selection_methods": set(),
            "selection_reasons": set(),
            "history_tickers": set(),
            "current_tickers": set(),
            "intraday_tickers": set(),
            "intraday_reasons": set(),
        })
        name = _clean(holding.name)
        if name and not record["name"]:
            record["name"] = name
        record["scopes"].add(_scope(holding))
        if not holding.is_historical_only and not holding.is_seeded_target:
            record["portfolio_memberships"].add("current")
        if holding.is_historical_only or (isin and isin in historical_members):
            record["portfolio_memberships"].add("historical")

        canonical_observation = _as_utc(holding.price_observation_timestamp)
        if canonical_observation is not None:
            record["canonical_observation_times"].add(canonical_observation)
        intraday_observation = _as_utc(
            holding.intraday_observation_timestamp
        )
        if intraday_observation is not None:
            record["intraday_observation_times"].add(intraday_observation)

        for field, value in (
            ("requested_tickers", holding.ticker_requested),
            ("canonical_tickers", canonical),
            ("selection_methods", holding.ticker_selection_method),
            ("selection_reasons", holding.ticker_selection_reason),
            ("history_tickers", holding.history_ticker),
            ("current_tickers", holding.current_ticker),
            ("intraday_tickers", holding.intraday_ticker),
            ("intraday_reasons", holding.intraday_ticker_reason),
        ):
            cleaned = _clean(value)
            if cleaned:
                record[field].add(cleaned)

    rows: list[dict[str, object]] = []
    for key, grouped_record in grouped.items():
        canonical = sorted(grouped_record["canonical_tickers"])
        history = sorted(grouped_record["history_tickers"])
        current = sorted(grouped_record["current_tickers"])
        intraday = sorted(grouped_record["intraday_tickers"])
        canonical_key = canonical[0].casefold() if len(canonical) == 1 else ""
        history_or_current_mismatch = bool(canonical_key) and any(
            ticker.casefold() != canonical_key
            for ticker in (*history, *current)
        )
        intraday_mismatch = bool(canonical_key) and any(
            ticker.casefold() != canonical_key
            for ticker in intraday
        )
        explicit_intraday_fallback = any(
            "sibling fallback" in reason.casefold()
            for reason in grouped_record["intraday_reasons"]
        )
        conflict = (
            any(len(values) > 1 for values in (canonical, history, current, intraday))
            or history_or_current_mismatch
            or (intraday_mismatch and not explicit_intraday_fallback)
        )
        selected = canonical[0] if len(canonical) == 1 else " / ".join(canonical)
        has_feed = bool(history or current)
        status = (
            "CONFLICT" if conflict
            else "RESOLVED" if selected and has_feed
            else "SELECTED_NO_DATA" if selected
            else "UNRESOLVED"
        )
        scopes = sorted(grouped_record["scopes"])
        memberships = grouped_record["portfolio_memberships"]
        if "current" in memberships and "historical" in memberships:
            portfolio_presence = "Current + Historical"
        elif "current" in memberships:
            portfolio_presence = "Current"
        elif "historical" in memberships:
            portfolio_presence = "Historical only"
        elif "Rebalance target" in scopes:
            portfolio_presence = "Rebalance target"
        else:
            portfolio_presence = ""
        canonical_observations = grouped_record["canonical_observation_times"]
        intraday_observations = grouped_record["intraday_observation_times"]
        intraday_default = (
            selected
            if selected and "Current portfolio" in scopes and not conflict
            else ""
        )
        rows.append({
            "identity_key": key,
            "isin": grouped_record["isin"],
            "name": grouped_record["name"] or selected or key,
            "scope": ", ".join(scopes),
            "portfolio_presence": portfolio_presence,
            "canonical_refresh_timestamp": _format_utc(
                max(canonical_observations) if canonical_observations else None
            ),
            "intraday_refresh_timestamp": _format_utc(
                max(intraday_observations) if intraday_observations else None
            ),
            "requested_ticker": " / ".join(sorted(grouped_record["requested_tickers"])),
            "canonical_ticker": selected,
            "selection_method": " / ".join(sorted(grouped_record["selection_methods"])),
            "selection_reason": " | ".join(sorted(grouped_record["selection_reasons"])),
            "history_ticker": " / ".join(history),
            "current_ticker": " / ".join(current),
            "intraday_default_ticker": intraday_default,
            "intraday_effective_ticker": " / ".join(intraday),
            "intraday_reason": " | ".join(sorted(grouped_record["intraday_reasons"])),
            "status": status,
        })

    return tuple(sorted(
        rows,
        key=lambda row: (
            str(row["isin"] or "~"),
            str(row["name"]).casefold(),
            str(row["canonical_ticker"]),
        ),
    ))
