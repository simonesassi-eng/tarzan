"""Admit raw chart prices before the native SDK can erase scalar types.

The owned curl client uses its documented response-class extension. Native
bytes, status, authentication, error and cleanup contracts remain unchanged.
Malformed JSON prices in the five chart price arrays become missing cells,
without discarding their independent valid neighbours. Split numeric fields
have a separate admission contract and an explicit rejection receipt; a
Boolean cannot acquire the authority to rescale past prices.
Unusable timestamp and volume cells in qualified native chart responses
retain their own uncertainty receipts; no epoch or traded volume is invented.
"""

from __future__ import annotations

import math
from urllib.parse import urlsplit

from curl_cffi.requests import Response
from curl_cffi.requests.models import Request


_PRICE_ARRAYS = (
    ("quote", ("open", "high", "low", "close")),
    ("adjclose", ("adjclose",)),
)
SPLIT_REJECTIONS_META = "_tarzan_rejected_splits_v1"
CHART_FIELDS_REJECTIONS_META = "_tarzan_rejected_chart_fields_v1"
_MAX_REJECTED_SPLIT_DATES = 64


def _new_rejection_receipt():
    return {
        "count": 0, "dates": [], "unknown_date": False,
        "date_bounds": [], "dates_truncated": False,
    }


def _record_rejection(receipt, stamp):
    """Bound individual dates without losing the complete known date span."""
    receipt["count"] += 1
    try:
        known_date = (
            type(stamp) in (int, float) and math.isfinite(float(stamp))
            and 0 < stamp < 10_000_000_000
        )
    except (OverflowError, TypeError, ValueError):
        known_date = False
    if not known_date:
        receipt["unknown_date"] = True
        return
    bounds = receipt["date_bounds"]
    if bounds:
        bounds[0], bounds[1] = min(bounds[0], stamp), max(bounds[1], stamp)
    else:
        bounds.extend((stamp, stamp))
    if len(receipt["dates"]) < _MAX_REJECTED_SPLIT_DATES:
        receipt["dates"].append(stamp)
    else:
        receipt["dates_truncated"] = True


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
    rejection = None
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
        if rejection is None:
            rejection = _new_rejection_receipt()
        _record_rejection(rejection, event.get("date"))
    metadata = result.get("meta")
    if rejection is not None and type(metadata) is dict:
        metadata[SPLIT_REJECTIONS_META] = rejection


def _completed_chart_source(response):
    """Scope auxiliary admission to a completed owned native chart response."""
    request = response.request
    if request is not None:
        if type(request) is not Request or request.method != "GET":
            return False
        url = request.url
    else:
        # Native-response replay/primitive fixture supplies its observed URL.
        # Bare JSON decoding without a chart source keeps its prior contract.
        url = response.url
    if type(url) is not str:
        return False
    try:
        parts = urlsplit(url)
        return (
            parts.scheme == "https"
            and parts.hostname in {"query1.finance.yahoo.com", "query2.finance.yahoo.com"}
            and parts.path.startswith("/v8/finance/chart/")
            and bool(parts.path.removeprefix("/v8/finance/chart/"))
        )
    except ValueError:
        return False


def _admit_auxiliary_scalar(value, *, timestamp):
    """Preserve native missing/numeric inputs, without inventing epoch/volume."""
    kind = type(value)
    if value is None:
        return value
    if kind in (bool, list, dict):
        return None
    if kind in (int, float, str):
        try:
            if not timestamp:
                if kind is float:
                    return value if math.isfinite(value) and -(2 ** 63) <= value < 2 ** 63 else None
                integer = int(value)
                return value if -(2 ** 63) <= integer < 2 ** 63 else None
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            # Normal epochs need no parser/import. At the fractional
            # nanosecond-domain boundary, use the same public native pandas
            # conversion rather than rejecting a valid fraction by rounding.
            limit = 9_223_372_036
            if -limit <= numeric <= limit:
                return value
            if abs(numeric) < limit + 1:
                from pandas import to_datetime

                to_datetime([value], unit="s")
                return value
            return None
        except (OverflowError, TypeError, ValueError):
            return None
    return value  # Preserve optional native custom-decoder scalar contracts.


def _drop_rejected_date_rows(result, timestamps):
    """Remove undated rows together, never relocate their independent prices.

    A native NaT is safe without actions but can abort native action merging.
    Pruning is limited to the inspected parallel-array schema. Unrecognized
    shapes retain their failure instead of being silently repaired.
    """
    indicators = result.get("indicators")
    if type(indicators) is not dict:
        return
    arrays = []
    for name, fields in (*_PRICE_ARRAYS, ("quote", ("volume",))):
        entries = indicators.get(name)
        if entries is None:
            continue
        if type(entries) is not list:
            return
        for entry in entries:
            if type(entry) is not dict:
                return
            for field in fields:
                values = entry.get(field)
                if values is None:
                    continue
                if type(values) is not list or len(values) != len(timestamps):
                    return
                arrays.append((entry, field, values))
    retained = [index for index, value in enumerate(timestamps) if value is not None]
    result["timestamp"] = [timestamps[index] for index in retained]
    for entry, field, values in arrays:
        entry[field] = [values[index] for index in retained]


def _admit_chart_auxiliary(document):
    if type(document) is not dict:
        return document
    chart = document.get("chart")
    if type(chart) is not dict or type(chart.get("result")) is not list:
        return document
    for result in chart["result"]:
        if type(result) is not dict or type(result.get("timestamp")) is not list:
            continue
        timestamps = result["timestamp"]
        events = result.get("events")
        has_actions = type(events) is dict and any(
            type(events.get(name)) is dict and bool(events[name])
            for name in ("splits", "dividends", "capitalGains")
        )
        rejected = {}
        for index, value in enumerate(timestamps):
            admitted = _admit_auxiliary_scalar(value, timestamp=True)
            if admitted is None and (value is not None or has_actions):
                receipt = rejected.setdefault("timestamp", _new_rejection_receipt())
                _record_rejection(receipt, None)
                timestamps[index] = None
        indicators = result.get("indicators")
        quotes = indicators.get("quote") if type(indicators) is dict else None
        if type(quotes) is list:
            for quote in quotes:
                volume = quote.get("volume") if type(quote) is dict else None
                if type(volume) is not list or len(volume) != len(timestamps):
                    continue  # Unknown shapes keep the native failure contract.
                for index, value in enumerate(volume):
                    admitted = _admit_auxiliary_scalar(value, timestamp=False)
                    if admitted is None and value is not None:
                        receipt = rejected.setdefault("volume", _new_rejection_receipt())
                        stamp = timestamps[index]
                        if type(stamp) is str:
                            stamp = float(stamp)
                        _record_rejection(receipt, stamp)
                        volume[index] = None
        metadata = result.get("meta")
        if rejected and type(metadata) is dict:
            metadata[CHART_FIELDS_REJECTIONS_META] = rejected
        if "timestamp" in rejected:
            _drop_rejected_date_rows(result, timestamps)
    return document


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
    """Keep native JSON/errors and admit raw chart cells and metadata identity."""

    def json(self, **kwargs):
        from tarzan.runtime.yahoo_metadata import admit_metadata_identity

        document = _admit_chart_prices(
            admit_metadata_identity(super().json(**kwargs), self),
        )
        return _admit_chart_auxiliary(document) if _completed_chart_source(self) else document
