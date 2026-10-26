"""One malformed raw price cannot erase independent valid chart rows."""

from copy import deepcopy
import json
import math
from decimal import Decimal

import pandas as pd
import pytest

from tarzan.data import _yf_net, enricher
from tarzan.tests.test_native_chart_price_types import chart_document, native_packets
from tarzan.tests.test_yahoo_core_info import owned


INVALID = [
    [101.], {"raw": 101.}, "not a price", 10 ** 400,
    "NaN", "Infinity", "-Infinity",
]
INVALID_IDS = [
    "array", "object", "nonnumeric-string", "overflow-integer",
    "nan-string", "positive-infinite-string", "negative-infinite-string",
]


@pytest.mark.parametrize("invalid", INVALID, ids=INVALID_IDS)
def test_native_malformed_price_does_not_erase_a_prior_valid_observation(native_packets, invalid):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., invalid])
    with owned("native-scalar-prior-recovery"):
        frame = enricher._download_price_history("SYNTH.MI")
    assert frame["Close"].dropna().tolist() == [100.]
    assert frame["Price Close"].dropna().tolist() == [100.]
    assert len(calls) == 1


@pytest.mark.parametrize("invalid", INVALID[:4], ids=INVALID_IDS[:4])
def test_native_invalid_only_scalar_source_keeps_inverse_fx_recovery(native_packets, invalid):
    documents, calls, _responses, stored = native_packets
    documents["USDEUR=X"] = chart_document("USDEUR=X", [invalid, invalid])
    documents["EURUSD=X"] = chart_document("EURUSD=X", [2., 2.])
    with owned("native-scalar-fx-recovery"):
        result = enricher._fetch_fx_pair_uncached("USD")
    assert result.tolist() == [.5, .5]
    assert [symbol for symbol, _ in calls] == ["USDEUR=X", "EURUSD=X"]
    assert len(stored) == 1 and stored[0][1].tolist() == [.5, .5]


@pytest.mark.parametrize("field,column", [
    ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
    ("adjclose", "Adj Close"),
])
def test_each_native_price_column_can_reject_a_malformed_cell_independently(
    native_packets, field, column,
):
    documents, _calls, responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    indicator = "adjclose" if field == "adjclose" else "quote"
    document["chart"]["result"][0]["indicators"][indicator][0][field][1] = {"raw": 101.}
    documents["SYNTH.MI"] = deepcopy(document)
    with owned("native-scalar-field-recovery"):
        frame = _yf_net.ticker("SYNTH.MI").history(
            period="5y", interval="1d", auto_adjust=False, keepna=True,
        )
    assert frame[column].iloc[0] == 100.
    assert pd.isna(frame[column].iloc[1])
    if field != "close":
        assert frame["Close"].tolist() == [100., 101.]
    assert json.loads(responses[0].content) == document
    assert documents["SYNTH.MI"] == document


@pytest.mark.parametrize("value", [
    101., 101, "101", "+1.01e2", "101.0",
    0., -1., 10 ** 308,
], ids=["float", "int", "numeric-string", "exponent-string", "decimal-string",
        "zero", "negative", "large-finite-integer"])
def test_valid_native_numeric_and_repair_inputs_keep_raw_scalar_values(value):
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = chart_document("SYNTH.MI", [100., value])
    response = YahooChartResponse()
    response.content = json.dumps(document).encode()
    parsed = response.json()
    assert parsed == document
    assert type(parsed["chart"]["result"][0]["indicators"]["quote"][0]["close"][1]) is type(value)


@pytest.mark.parametrize("value", [None, 0, 101., "101"])
def test_missing_or_valid_cells_do_not_change_json_or_create_extra_rows(value):
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = chart_document("SYNTH.MI", [100., value])
    response = YahooChartResponse()
    response.content = json.dumps(document).encode()
    assert response.json() == document


def test_native_mixed_invalid_cells_leave_valid_timestamps_and_native_price_scale_intact():
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    values = [100., True, {"raw": 101.}, [102.], "bad", 10 ** 400, "103", 104., None]
    result["timestamp"] = list(range(len(values)))
    for field in ("open", "high", "low", "close"):
        result["indicators"]["quote"][0][field] = list(values)
    result["indicators"]["quote"][0]["volume"] = [0] * len(values)
    result["indicators"]["adjclose"][0]["adjclose"] = list(values)
    response = YahooChartResponse()
    response.content = json.dumps(document).encode()
    admitted = response.json()["chart"]["result"][0]
    assert admitted["timestamp"] == list(range(len(values)))
    assert admitted["indicators"]["quote"][0]["close"] == [
        100., None, None, None, None, None, "103", 104., None,
    ]
    assert admitted["meta"] == result["meta"] and admitted["events"] == result["events"]
    assert json.loads(response.content) == document


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_decodable_native_nonfinite_price_scalars_are_missing(value):
    from curl_cffi.requests import Response
    from tarzan.runtime.yahoo_response import YahooChartResponse

    raw = json.dumps(chart_document("SYNTH.MI", [100., value])).encode()
    native, admitted = Response(), YahooChartResponse()
    native.content = admitted.content = raw
    try:
        original = native.json()
    except ValueError as error:
        # Keep the pinned native decoder's rejection if a backend does not
        # accept these nonstandard JSON literals; never replace that decoder.
        with pytest.raises(type(error)):
            admitted.json()
        return
    assert not math.isfinite(original["chart"]["result"][0]["indicators"]["quote"][0]["close"][1])
    parsed = admitted.json()["chart"]["result"][0]["indicators"]["quote"][0]
    assert parsed["close"] == [100., None]


def test_native_custom_json_decoder_contract_is_not_replaced():
    from curl_cffi.requests import Response
    from tarzan.runtime.yahoo_response import YahooChartResponse

    raw = json.dumps(chart_document("SYNTH.MI", [100., 101.])).encode()
    native, admitted = Response(), YahooChartResponse()
    native.content = admitted.content = raw
    try:
        original = native.json(parse_float=Decimal)
    except TypeError as error:
        with pytest.raises(type(error)):
            admitted.json(parse_float=Decimal)
        return
    parsed = admitted.json(parse_float=Decimal)
    assert parsed == original
    assert type(parsed["chart"]["result"][0]["indicators"]["quote"][0]["close"][0]) is Decimal
