"""Undated rows cannot abort native actions or move a price to another date."""

from copy import deepcopy
import json

import pandas as pd
import pytest
from yfinance.cache import get_tz_cache

from tarzan.data import enricher
from tarzan.runtime.yahoo_response import YahooChartResponse
from tarzan.tests.test_native_chart_price_types import chart_document, native_packets
from tarzan.tests.test_yahoo_core_info import owned


@pytest.mark.parametrize("event", ["split", "dividend", "capital-gain"])
@pytest.mark.parametrize("invalid", [True, {"raw": 100}, None])
def test_rejected_date_matches_a_native_dated_price_with_its_legitimate_action(
    native_packets, event, invalid,
):
    documents, calls, responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    day = result["timestamp"][1]
    result["meta"]["instrumentType"] = "EQUITY"
    if event == "split":
        result["events"]["splits"] = {
            str(day): {"date": day, "numerator": 2., "denominator": 1., "splitRatio": "2:1"},
        }
        column, amount = "Stock Splits", 2.
    elif event == "dividend":
        result["events"]["dividends"] = {str(day): {"date": day, "amount": 1.}}
        column, amount = "Dividends", 1.
    else:
        result["meta"]["instrumentType"] = "ETF"
        result["events"]["capitalGains"] = {str(day): {"date": day, "amount": 1.}}
        column, amount = "Capital Gains", 1.
    original_actions = deepcopy(result["events"])
    control = deepcopy(document)
    counterpart = control["chart"]["result"][0]
    counterpart["meta"]["symbol"] = "CONTROL.MI"
    counterpart["timestamp"] = counterpart["timestamp"][:1]
    for field in ("open", "high", "low", "close", "volume"):
        counterpart["indicators"]["quote"][0][field] = (
            counterpart["indicators"]["quote"][0][field][:1]
        )
    counterpart["indicators"]["adjclose"][0]["adjclose"] = [100.]
    result["timestamp"][1] = invalid
    documents.update({"SYNTH.MI": deepcopy(document), "CONTROL.MI": control})
    get_tz_cache().store("CONTROL.MI", "UTC")
    with owned("native-date-alignment-action-oracle") as session:
        expected = enricher._download_price_history("CONTROL.MI")
        observed = enricher._download_price_history("SYNTH.MI")
        receipt = session.transport.coordinator.snapshot()
    # The SDK's independently dated action-only row is legitimate, but it
    # contains no price. Keep the action; never promote 101 from the bad date.
    assert expected["Price Close"].dropna().tolist() == [100.]
    assert expected.index[-1].date() == pd.Timestamp("2026-10-26").date()
    assert pd.isna(expected["Price Close"].iloc[-1]) and expected[column].iloc[-1] == amount
    assert observed["Price Close"].dropna().tolist() == [100.]
    assert observed[column].tolist() == expected[column].tolist()
    assert "tarzan_rejected_chart_fields_v1" in observed.attrs
    compared = observed.copy()
    compared.attrs.pop("tarzan_rejected_chart_fields_v1")
    pd.testing.assert_frame_equal(compared, expected)
    assert not enricher._reusable_price_observation(observed)
    assert receipt["acquired"] == receipt["released"] == len(calls) == 2
    assert receipt["active"] == 0
    assert json.loads(responses[-1].content) == document == documents["SYNTH.MI"]
    assert responses[-1].json()["chart"]["result"][0]["events"] == original_actions


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("invalid", [True, False, {"raw": 1}, [1], "invalid", 10 ** 400])
def test_undated_price_row_is_removed_from_every_parallel_array(position, invalid):
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"][position] = invalid
    quote = result["indicators"]["quote"][0]
    for number, field in enumerate(("open", "high", "low", "close", "volume")):
        quote[field] = [10 * (number + 1), 10 * (number + 1) + 1]
    original = deepcopy(document)
    response = YahooChartResponse()
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    raw = json.dumps(document).encode()
    response.content = raw
    admitted = response.json()["chart"]["result"][0]
    retained = 1 - position
    assert admitted["timestamp"] == [result["timestamp"][retained]]
    for field in ("open", "high", "low", "close", "volume"):
        assert admitted["indicators"]["quote"][0][field] == [quote[field][retained]]
    assert admitted["indicators"]["adjclose"][0]["adjclose"] == [[100., 101.][retained]]
    assert admitted["meta"]["_tarzan_rejected_chart_fields_v1"]["timestamp"]["count"] == 1
    assert response.content is raw and json.loads(response.content) == original == document


def test_null_native_date_without_a_rejection_keeps_its_original_decoded_contract():
    document = chart_document("SYNTH.MI", [100., 101.])
    document["chart"]["result"][0]["timestamp"][1] = None
    response = YahooChartResponse()
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    assert response.json() == document


@pytest.mark.parametrize("shape", ["short-array", "unknown-array-envelope", "unknown-field-shape"])
def test_unknown_parallel_shape_is_not_repaired_by_relocating_other_prices(shape):
    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"][1] = True
    if shape == "short-array":
        result["indicators"]["quote"][0]["open"] = [100.]
    elif shape == "unknown-array-envelope":
        result["indicators"]["adjclose"] = {"adjclose": [100., 101.]}
    else:
        result["indicators"]["quote"][0]["volume"] = {"raw": 100}
    response = YahooChartResponse()
    response.url = "https://query1.finance.yahoo.com/v8/finance/chart/SYNTH.MI"
    response.content = json.dumps(document).encode()
    admitted = response.json()["chart"]["result"][0]
    assert admitted["timestamp"] == [result["timestamp"][0], None]
    assert admitted["indicators"] == result["indicators"]
