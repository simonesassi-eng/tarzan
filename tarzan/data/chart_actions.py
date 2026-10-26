"""Retain typed split rejection evidence beside independent native prices.

This evidence only restricts repair, persistence and reuse; it never grants a
new price, listing, action or source authority. It is not a provider revision
or a substitute for the independently observed OHLC/adjusted-close rulers.
"""

from __future__ import annotations

import math

import pandas as pd
import yfinance as yf
from yfinance.scrapers.history import PriceHistory
from yfinance.ticker import Ticker

from tarzan.runtime.yahoo_response import SPLIT_REJECTIONS_META


REJECTION_ATTR = "tarzan_rejected_split_evidence_v1"
_NATIVE_TICKER_HISTORY = Ticker.history
_NATIVE_PRICE_HISTORY = PriceHistory.history


def has_rejected_splits(frame):
    """An uncertain action history is not a durable or positive-memo fact."""
    return isinstance(frame, (pd.DataFrame, pd.Series)) and REJECTION_ATTR in frame.attrs


def attach_rejected_splits(frame, client):
    """Read only completed native metadata, with no lazy SDK/provider access."""
    if (
        not isinstance(frame, pd.DataFrame) or frame.empty
        or not isinstance(frame.index, pd.DatetimeIndex)
        or getattr(yf, "__version__", None) != "1.7.0"
        or type(client) is not Ticker
        or Ticker.history is not _NATIVE_TICKER_HISTORY
        or PriceHistory.history is not _NATIVE_PRICE_HISTORY
    ):
        return frame
    history = vars(client).get("_price_history")
    metadata = vars(history).get("_history_metadata") if type(history) is PriceHistory else None
    if not isinstance(metadata, dict) or metadata.get("symbol") != vars(client).get("ticker"):
        return frame
    observation = metadata.get(SPLIT_REJECTIONS_META)
    if (
        type(observation) is not dict
        or set(observation) != {
            "count", "dates", "unknown_date", "date_bounds", "dates_truncated",
        }
        or type(observation["count"]) is not int or observation["count"] <= 0
        or type(observation["unknown_date"]) is not bool
        or type(observation["dates"]) is not list or len(observation["dates"]) > 64
        or type(observation["dates_truncated"]) is not bool
        or type(observation["date_bounds"]) is not list
        or len(observation["date_bounds"]) not in (0, 2)
    ):
        return frame
    unknown = observation["unknown_date"]
    dates = []
    lower, upper = frame.index.min().date(), frame.index.max().date()
    zone = frame.index.tz or metadata.get("exchangeTimezoneName") or "UTC"
    # Bounded receipt storage is not an unknown event date. Even when more
    # than 64 events were rejected, their full min/max bounds can prove that
    # every event is outside this window. An overlapping truncated range
    # remains uncertain; it cannot certify the omitted individual dates.
    if observation["dates_truncated"]:
        bounds = observation["date_bounds"]
        try:
            if len(bounds) != 2 or any(
                type(value) not in (int, float)
                or not math.isfinite(float(value)) or not 0 < value < 10_000_000_000
                for value in bounds
            ) or bounds[0] > bounds[1]:
                raise ValueError("unusable split date bounds")
            first, last = (
                pd.Timestamp(value, unit="s", tz="UTC").tz_convert(zone).date()
                for value in bounds
            )
            unknown |= first <= upper and last >= lower
        except (TypeError, ValueError, OverflowError):
            unknown = True
    for value in observation["dates"]:
        try:
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError("unusable split date")
            day = pd.Timestamp(value, unit="s", tz="UTC").tz_convert(zone).date()
        except (TypeError, ValueError, OverflowError):
            unknown = True
            continue
        if lower <= day <= upper:
            dates.append(day.isoformat())
    # Future/out-of-window actions cannot degrade or modify an older tape.
    if not dates and not unknown:
        return frame
    receipt = {
        "symbol": str(metadata["symbol"]),
        "dates": tuple(sorted(set(dates))),
        "unknown_date": unknown,
    }
    result = frame.copy()
    result.attrs[REJECTION_ATTR] = receipt
    from tarzan.runtime import data_quality as dq

    dq.warning(
        "market_data",
        "Malformed provider split factors were rejected. Independent prices "
        "are retained without that split repair; historical action evidence "
        "requires review and this observation is not persisted or reused.",
        context=receipt["symbol"],
    )
    return result


def merge_rejected_splits(cached, fresh, merged):
    """An ordinary pandas concat must not erase an uncertain action receipt."""
    if merged is None or (
        not has_rejected_splits(cached) and not has_rejected_splits(fresh)
    ):
        return merged
    result = merged.copy()
    receipts = [
        frame.attrs[REJECTION_ATTR] for frame in (cached, fresh)
        if has_rejected_splits(frame)
    ]
    result.attrs[REJECTION_ATTR] = _merge_receipts(receipts)
    return result


def attach_batch_rejections(frame, children):
    """Keep uncertainty after concat drops unequal child attrs.

    A projected pandas child inherits batch attrs. Conservatively preventing
    persistence of such a projection is preferable to losing its receipt;
    healthy source children and their own positive memos are not modified.
    """
    receipts = [
        child.attrs[REJECTION_ATTR] for child in children.values()
        if has_rejected_splits(child)
    ]
    if not receipts:
        return frame
    result = frame.copy()
    result.attrs[REJECTION_ATTR] = _merge_receipts(receipts)
    return result


def _merge_receipts(receipts):
    # Merge the bounded uncertainty scope, never an action factor. A repeated
    # merge cannot build an unbounded recursive tuple of prior receipts.
    dates = set()
    symbols = set()
    unknown = False
    for receipt in receipts:
        if not isinstance(receipt, dict):
            unknown = True
            continue
        symbols.add(str(receipt.get("symbol", "")))
        values = receipt.get("dates", ())
        if not isinstance(values, (tuple, list)) or not all(
            isinstance(value, str) for value in values
        ):
            unknown = True
        else:
            dates.update(values)
        unknown |= receipt.get("unknown_date") is not False
    ordered_dates = sorted(dates)
    return {
        "symbol": next(iter(symbols)) if len(symbols) == 1 else "",
        "dates": tuple(ordered_dates[:64]),
        "unknown_date": unknown or len(ordered_dates) > 64 or len(symbols) != 1,
    }
