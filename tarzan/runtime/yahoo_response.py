"""Admit raw chart prices before the native SDK can erase scalar types.

The owned curl client uses its documented response-class extension. Native
bytes, status, authentication, error and cleanup contracts remain unchanged.
Malformed JSON prices in the five chart price arrays become missing cells,
without discarding their independent valid neighbours. Split numeric fields
have a separate admission contract and an explicit rejection receipt; a
Boolean cannot acquire the authority to rescale past prices.
"""

from __future__ import annotations

import math

from curl_cffi.requests import Response


_PRICE_ARRAYS = (
    ("quote", ("open", "high", "low", "close")),
    ("adjclose", ("adjclose",)),
)
SPLIT_REJECTIONS_META = "_tarzan_rejected_splits_v1"
_MAX_REJECTED_SPLIT_DATES = 64


def _valid_split_number(value):
    """Only real positive finite JSON numbers can declare a split factor."""
    kind = type(value)
    if kind in (int, float):
        try:
            return math.isfinite(float(value)) and value > 0
        except (OverflowError, TypeError, ValueError):
            return False
    if kind in (bool, list, dict, str) or value is None:
        return False
    # A caller's custom native JSON decoder retains its original contract.
    return True


def _admit_split_fields(result):
    """Keep valid events; rejected factors stay explicitly uncertain, not zero."""
    events = result.get("events")
    splits = events.get("splits") if type(events) is dict else None
    if type(splits) is not dict:
        return
    rejected = 0
    dates = []
    unknown_date = False
    dates_truncated = False
    lower_date = upper_date = None
    for event_key, event in tuple(splits.items()):
        if type(event) is not dict:
            continue  # Unknown envelopes retain the native parser's failure.
        numerator, denominator = event.get("numerator"), event.get("denominator")
        valid = _valid_split_number(numerator) and _valid_split_number(denominator)
        if valid and type(numerator) in (int, float) and type(denominator) in (int, float):
            try:
                ratio = float(numerator) / float(denominator)
                valid = math.isfinite(ratio) and ratio > 0
            except (OverflowError, TypeError, ValueError, ZeroDivisionError):
                valid = False
        if valid:
            continue
        # Native safe_merge rejects missing action factors as a lost event.
        # Remove only this unusable decoded event and retain its explicit
        # rejection receipt. Never invent a zero/one ratio or reinterpret text.
        del splits[event_key]
        rejected += 1
        stamp = event.get("date")
        try:
            known_date = (
                type(stamp) in (int, float) and math.isfinite(float(stamp))
                and 0 < stamp < 10_000_000_000
            )
        except (OverflowError, TypeError, ValueError):
            known_date = False
        if known_date:
            lower_date = stamp if lower_date is None else min(lower_date, stamp)
            upper_date = stamp if upper_date is None else max(upper_date, stamp)
            if len(dates) < _MAX_REJECTED_SPLIT_DATES:
                dates.append(stamp)
            else:
                dates_truncated = True
        else:
            unknown_date = True
    metadata = result.get("meta")
    if rejected and type(metadata) is dict:
        metadata[SPLIT_REJECTIONS_META] = {
            "count": rejected, "dates": dates, "unknown_date": unknown_date,
            "date_bounds": [] if lower_date is None else [lower_date, upper_date],
            "dates_truncated": dates_truncated,
        }


def _admit_json_price(value):
    """Retain valid raw numeric primitives; do not reinterpret custom decoders."""
    kind = type(value)
    if kind in (bool, list, dict):
        return None
    if kind in (int, float, str):
        try:
            return value if math.isfinite(float(value)) else None
        except (OverflowError, TypeError, ValueError):
            return None
    # None is native missing data. Optional native JSON decoder hooks can
    # supply non-JSON scalar types; leave their original caller contract.
    return value


def _admit_chart_prices(document):
    """Edit only the newly parsed JSON object, never the raw response bytes."""
    if type(document) is not dict:
        return document
    chart = document.get("chart")
    if type(chart) is not dict or type(chart.get("result")) is not list:
        return document
    for result in chart["result"]:
        if type(result) is not dict:
            continue
        _admit_split_fields(result)
        indicators = result.get("indicators")
        if type(indicators) is not dict:
            continue
        for name, fields in _PRICE_ARRAYS:
            arrays = indicators.get(name)
            if type(arrays) is not list:
                continue
            for array in arrays:
                if type(array) is not dict:
                    continue
                for field in fields:
                    values = array.get(field)
                    if type(values) is list:
                        for index, value in enumerate(values):
                            values[index] = _admit_json_price(value)
    return document


class YahooChartResponse(Response):
    """Keep curl's native JSON/errors and independently admit raw price cells."""

    def json(self, **kwargs):
        return _admit_chart_prices(super().json(**kwargs))
