"""Admit raw chart prices before the native SDK can erase scalar types.

The owned curl client uses its documented response-class extension. Native
bytes, status, authentication, error and cleanup contracts remain unchanged.
Only JSON booleans in the five chart price arrays become missing observations;
metadata flags, action events, timestamps and volume are not price arrays.
"""

from __future__ import annotations

from curl_cffi.requests import Response


_PRICE_ARRAYS = (
    ("quote", ("open", "high", "low", "close")),
    ("adjclose", ("adjclose",)),
)


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
                            if type(value) is bool:
                                values[index] = None
    return document


class YahooChartResponse(Response):
    """Preserve curl's own JSON/errors while rejecting typed boolean prices."""

    def json(self, **kwargs):
        return _admit_chart_prices(super().json(**kwargs))
