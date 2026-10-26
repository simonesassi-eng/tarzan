"""Admit daily session labels without mistaking them for future trades.

Ordinary daily bars are bounded on their own venue's clock. An overnight
currency/futures session can legitimately carry tomorrow's label before
midnight; only the native chart's already-received trading-period evidence
can establish that exception. No lazy SDK metadata property is accessed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yfinance as yf
from yfinance.scrapers.history import PriceHistory
from yfinance.ticker import Ticker


SESSION_EVIDENCE_ATTR = "tarzan_daily_session_evidence_v1"
_NATIVE_TICKER_HISTORY = Ticker.history
_NATIVE_PRICE_HISTORY = PriceHistory.history
_MAX_SESSION_EVIDENCE = 8


def _instant(value) -> pd.Timestamp | None:
    from tarzan.data.current_session import quote_observed_at

    observed = quote_observed_at({"time": value})
    return pd.Timestamp(observed) if observed is not None else None


def _limit() -> pd.Timestamp:
    from tarzan.data.current_session import observation_time_limit

    return pd.Timestamp(observation_time_limit())


def _key(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def _session_observation(label, evidence, limit) -> pd.Timestamp | None:
    """Validate persisted primitive evidence, not a provider's derived lastTrade."""
    if not isinstance(evidence, dict):
        return None
    if evidence.get("instrument_type") not in ("CURRENCY", "FUTURE"):
        return None
    zone_name = evidence.get("timezone")
    if not isinstance(zone_name, str):
        return None
    try:
        zone = ZoneInfo(zone_name)
        stamp = pd.Timestamp(label)
        if pd.isna(stamp):
            return None
        local = (
            stamp.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
            if stamp.tzinfo is None
            else stamp.tz_convert(zone)
        )
        if pd.isna(local):
            return None
        start = _instant(evidence.get("period_start"))
        end = _instant(evidence.get("period_end"))
        observed = _instant(evidence.get("observed_at"))
        if start is None or end is None or observed is None:
            return None
        if not start <= observed <= limit or not observed < end:
            return None
        if not timedelta(0) < end - start <= timedelta(days=2):
            return None
        local_start, local_end = start.tz_convert(zone), end.tz_convert(zone)
        if local_start.date() >= local_end.date():
            return None
        if local.date() != local_end.date():
            return None
        # This is a session-date label, not an arbitrary later intraday print.
        if any((
            local.hour, local.minute, local.second, local.microsecond, local.nanosecond,
        )) or not start <= local < end:
            return None
        return observed
    except (TypeError, ValueError, OverflowError, ZoneInfoNotFoundError):
        return None


def session_evidence(frame, *, limit=None) -> dict[str, dict[str, str]]:
    """Return only bounded, causal row evidence suitable for private JSON."""
    if not isinstance(frame, (pd.Series, pd.DataFrame)):
        return {}
    declared = frame.attrs.get(SESSION_EVIDENCE_ATTR)
    if not isinstance(declared, dict) or len(declared) > _MAX_SESSION_EVIDENCE:
        return {}
    if not isinstance(frame.index, pd.DatetimeIndex):
        return {}
    limit = _limit() if limit is None else pd.Timestamp(limit)
    result = {}
    for key, evidence in declared.items():
        if not isinstance(key, str):
            continue
        try:
            label = pd.Timestamp(key)
            if pd.isna(label) or _key(label) != key or label not in frame.index:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        if _session_observation(label, evidence, limit) is None:
            continue
        result[key] = {
            field: str(evidence[field])
            for field in (
                "period_start", "period_end", "observed_at",
                "timezone", "instrument_type",
            )
        }
    return result


def attach_native_session_evidence(frame, client):
    """Inspect only the completed pinned SDK response, never a lazy accessor."""
    if (
        not isinstance(frame, pd.DataFrame) or frame.empty
        or not isinstance(frame.index, pd.DatetimeIndex)
        or getattr(yf, "__version__", None) != "1.7.0"
        or type(client) is not Ticker
        or Ticker.history is not _NATIVE_TICKER_HISTORY
        or PriceHistory.history is not _NATIVE_PRICE_HISTORY
    ):
        return frame
    try:
        sdk_root = Path(yf.__file__).resolve().parent
        if not all(
            Path(function.__code__.co_filename).resolve().is_relative_to(sdk_root)
            for function in (_NATIVE_TICKER_HISTORY, _NATIVE_PRICE_HISTORY)
        ):
            return frame
        history = vars(client).get("_price_history")
        if type(history) is not PriceHistory:
            return frame
        metadata = vars(history).get("_history_metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("instrumentType") not in ("CURRENCY", "FUTURE")
        ):
            return frame
        symbol = vars(client).get("ticker")
        if not isinstance(symbol, str) or metadata.get("symbol") != symbol:
            return frame
        kind = metadata["instrumentType"]
        if not symbol.endswith("=X" if kind == "CURRENCY" else "=F"):
            return frame
        period = metadata.get("currentTradingPeriod")
        regular = period.get("regular") if isinstance(period, dict) else None
        if not isinstance(regular, dict) or metadata.get("dataGranularity") != "1d":
            return frame
        evidence = {
            "period_start": regular.get("start"),
            "period_end": regular.get("end"),
            "observed_at": metadata.get("regularMarketTime"),
            "timezone": metadata.get("exchangeTimezoneName"),
            "instrument_type": metadata.get("instrumentType"),
        }
        limit = _limit()
        observations = {}
        for label in frame.index[frame.index > limit]:
            observed = _session_observation(label, evidence, limit)
            if observed is not None:
                observations[_key(label)] = {
                    **evidence,
                    "period_start": _instant(evidence["period_start"]).isoformat(),
                    "period_end": _instant(evidence["period_end"]).isoformat(),
                    "observed_at": observed.isoformat(),
                }
        if observations:
            frame = frame.copy()
            frame.attrs[SESSION_EVIDENCE_ATTR] = observations
        return frame
    except (AttributeError, TypeError, ValueError, OSError, OverflowError):
        return frame


def admit_history(frame, symbol: str = ""):
    """Filter dated future/NaT rows before memo, cache, routing or arithmetic.

    Non-datetime injected ports retain their established undated contract.
    Pinned daily runs retain their existing venue-midnight date boundary.
    A LIVE clock is sampled once per tape, not once per row; capture time is
    not a receipt upper bound while native acquisition is still running.
    """
    if (
        frame is None or frame.empty
        or not isinstance(frame.index, pd.DatetimeIndex)
    ):
        return frame
    from tarzan import runtime

    index = frame.index
    effective_date = runtime.as_of()
    if effective_date is not None:
        cutoff = pd.Timestamp(effective_date)
        labels = index.tz_localize(None) if index.tz is not None else index
        admitted = index.notna() & (labels <= cutoff)
    else:
        limit = _limit()
        if index.tz is not None:
            cutoff = limit
        else:
            from tarzan.data.market_quotes import _exchange_tz

            selected = symbol or frame.attrs.get("tarzan_price_symbol")
            if not selected and isinstance(frame, pd.Series):
                selected = frame.name
            zone = _exchange_tz(selected if isinstance(selected, str) else "")
            cutoff = limit.tz_convert(zone or "UTC").tz_localize(None)
        admitted = index.notna() & (index <= cutoff)
        if not admitted.all():
            evidence = session_evidence(frame, limit=limit)
            for position in (~admitted).nonzero()[0]:
                if not pd.isna(index[position]):
                    admitted[position] = _key(index[position]) in evidence
    if admitted.all():
        return frame
    result = frame.loc[admitted].copy()
    if SESSION_EVIDENCE_ATTR in result.attrs:
        result.attrs[SESSION_EVIDENCE_ATTR] = session_evidence(result)
    return result


def observation_time(frame, label) -> datetime | None:
    """Keep the actual overnight trade time instead of a future session label."""
    if not isinstance(frame.index, pd.DatetimeIndex) or pd.isna(label):
        return None
    declared = frame.attrs.get(SESSION_EVIDENCE_ATTR)
    if isinstance(declared, dict) and len(declared) <= _MAX_SESSION_EVIDENCE:
        evidence = declared.get(_key(label))
        if evidence is not None:
            observed = _session_observation(label, evidence, _limit())
            if observed is not None:
                return observed.to_pydatetime()
    try:
        return pd.Timestamp(label).to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None


def merge_session_evidence(cached, fresh, result):
    """Fresh overlap wins for evidence as well as for prices."""
    if result is None or result.empty:
        return result
    retained = session_evidence(cached)
    new_evidence = session_evidence(fresh)
    if not retained and not new_evidence:
        return result
    if fresh is not None and not fresh.empty:
        if isinstance(fresh.index, pd.DatetimeIndex):
            retained = {
                key: evidence for key, evidence in retained.items()
                if pd.Timestamp(key) not in fresh.index
            }
        retained.update(new_evidence)
    if not retained:
        if SESSION_EVIDENCE_ATTR in result.attrs:
            result = result.copy()
            result.attrs.pop(SESSION_EVIDENCE_ATTR, None)
        return result
    result = result.copy()
    result.attrs[SESSION_EVIDENCE_ATTR] = retained
    result.attrs[SESSION_EVIDENCE_ATTR] = session_evidence(result)
    return result
