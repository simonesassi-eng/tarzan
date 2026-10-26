"""Typed chart admission before the actual SDK erases JSON booleans."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timedelta, timezone
import json

from curl_cffi.requests import Response, Session
import numpy as np
import pandas as pd
import pytest
from yfinance.cache import get_tz_cache
from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan.data import _yf_net, current_session as cs, enricher, price_cache
from tarzan.models.holding import Holding
from tarzan.tests.test_yahoo_core_info import owned


def chart_document(symbol, values):
    """Independent raw Yahoo schema, not a substituted SDK DataFrame."""
    now = datetime(2026, 10, 26, 22, 30, tzinfo=timezone.utc)
    dates = ["2026-10-23", "2026-10-26"]
    times = [int(pd.Timestamp(value, tz="UTC").timestamp()) for value in dates]
    period = {
        "start": int(pd.Timestamp(now.date(), tz="UTC").timestamp()),
        "end": int((pd.Timestamp(now.date(), tz="UTC") + pd.Timedelta(
            hours=23, minutes=59,
        )).timestamp()),
        "timezone": "UTC", "gmtoffset": 0,
    }
    trading_periods = [{
        **period,
        "start": stamp,
        "end": stamp + 23 * 3600 + 59 * 60,
    } for stamp in times]
    return {"chart": {"result": [{
        "meta": {
            "symbol": symbol, "currency": "EUR",
            "exchangeTimezoneName": "UTC", "timezone": "UTC", "gmtoffset": 0,
            "instrumentType": "CURRENCY",
            "regularMarketTime": int((now - timedelta(minutes=1)).timestamp()),
            "regularMarketPrice": values[-1], "priceHint": 4,
            "firstTradeDate": times[0] - 10 * 365 * 86400,
            "dataGranularity": "1d",
            "validRanges": ["1d", "5d", "1mo", "5y", "max"],
            "currentTradingPeriod": {"regular": period},
            "tradingPeriods": [[value] for value in trading_periods],
            "hasPrePostMarketData": True,
        },
        "timestamp": times,
        "indicators": {
            "quote": [{
                "open": list(values), "high": list(values),
                "low": list(values), "close": list(values),
                "volume": [100, 100],
            }],
            "adjclose": [{"adjclose": list(values)}],
        },
        "events": {},
    }], "error": None}}


@pytest.fixture
def native_packets(monkeypatch):
    """Replace only primitive HTTP; retain the owned native response factory."""
    now = datetime(2026, 10, 26, 22, 30, tzinfo=timezone.utc)

    class ReceiptDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(cs, "datetime", ReceiptDatetime)
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    monkeypatch.setattr(YfData, "_get_cookie_and_crumb", lambda *a, **k: ("synthetic", "basic"))
    monkeypatch.setattr(YfData, "_set_cookie_strategy", lambda *a, **k: None)
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)
    for symbol in ("USDEUR=X", "EURUSD=X", "CHFEUR=X", "SYNTH.MI"):
        get_tz_cache().store(symbol, "UTC")
    documents, calls, responses, stored = {}, [], [], []
    monkeypatch.setattr(price_cache, "store_history", lambda symbol, frame: (
        stored.append((symbol, frame.copy(deep=True)))
    ))
    YfData.cache_get.cache_clear()

    def request(self, method, url, **kwargs):
        assert method == "GET" and kwargs["timeout"] > 0
        assert "/v8/finance/chart/" in url
        symbol = url.rsplit("/", 1)[-1]
        calls.append((symbol, dict(kwargs.get("params") or {})))
        document = documents.get(symbol)
        status = 200 if document is not None else 404
        if document is None:
            document = {"chart": {"result": None, "error": {
                "code": "Not Found", "description": "Synthetic absent pair",
            }}}
        # This is the native documented Response factory, including the
        # candidate's admission. A YfData.get stub would bypass that boundary.
        response = self.response_class()
        response.content = json.dumps(document).encode()
        response.status_code, response.ok = status, status == 200
        response.encoding, response.url, response.reason = "utf-8", url, ""
        responses.append(response)
        return response

    monkeypatch.setattr(Session, "request", request)
    try:
        yield documents, calls, responses, stored
    finally:
        YfData.cache_get.cache_clear()


@pytest.mark.parametrize("value", [101., 101, "101"])
def test_native_positive_price_types_keep_the_exact_financial_value(native_packets, value):
    documents, calls, responses, stored = native_packets
    document = chart_document("USDEUR=X", [100., value])
    documents["USDEUR=X"] = deepcopy(document)
    with owned("native-price-positive") as session:
        result = enricher._fetch_fx_pair_uncached("USD")
        assert session.transport.coordinator.snapshot()["active"] == 0
    assert result.tolist() == [100., 101.]
    assert stored[0][1].tolist() == result.tolist()
    assert documents["USDEUR=X"] == document
    assert json.loads(responses[0].content) == document
    assert len(calls) == 1


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("boolean", [False, True])
def test_native_boolean_fx_cannot_become_an_admissible_rate(native_packets, inverse, boolean):
    documents, calls, responses, stored = native_packets
    symbol = "EURUSD=X" if inverse else "USDEUR=X"
    values = [2., boolean] if inverse else [.5, boolean]
    document = chart_document(symbol, values)
    documents[symbol] = deepcopy(document)
    with owned("native-price-fx") as session:
        result = enricher._fetch_fx_pair_uncached("USD")
        observation = session.transport.coordinator.snapshot()
    assert result.tolist() == [.5]
    assert stored[0][0] == "FX_USD" and stored[0][1].tolist() == [.5]
    assert documents[symbol] == document
    assert json.loads(responses[-1].content) == document
    assert observation["acquired"] == observation["released"] == len(calls)
    assert observation["active"] == 0
    assert {symbol for symbol, _ in calls} == (
        {"USDEUR=X", "EURUSD=X"} if inverse else {"USDEUR=X"}
    )


@pytest.mark.parametrize("boolean", [False, True])
def test_native_boolean_cannot_replace_a_real_holding_price(native_packets, boolean):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., boolean])
    holding = Holding(
        isin="SYNTHETIC", ticker="SYNTH.MI", quantity=2,
        cost_basis_eur=200, market_value_eur=200, currency="EUR",
    )
    with owned("native-price-holding"):
        frame = _yf_net.ticker("SYNTH.MI").history(
            period="5y", interval="1d", auto_adjust=False,
        )
        enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 100.
    assert holding.price_history.tolist() == [100.]
    assert len(calls) == 1


@pytest.mark.parametrize("field,column", [
    ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
    ("adjclose", "Adj Close"),
])
@pytest.mark.parametrize("boolean", [False, True])
def test_native_each_boolean_price_field_is_missing_not_a_zero_or_one(
    native_packets, field, column, boolean,
):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    indicator = "adjclose" if field == "adjclose" else "quote"
    document["chart"]["result"][0]["indicators"][indicator][0][field][1] = boolean
    documents["SYNTH.MI"] = document
    with owned("native-price-field"):
        frame = _yf_net.ticker("SYNTH.MI").history(
            period="5y", interval="1d", auto_adjust=False, keepna=True,
        )
    assert frame[column].iloc[0] == 100.
    assert np.isnan(frame[column].iloc[1])
    if field != "close":
        assert frame["Close"].tolist() == [100., 101.]


@pytest.mark.parametrize("auto_adjust,back_adjust", [(True, False), (False, True)])
def test_native_adjustment_cannot_turn_a_boolean_into_a_price(
    native_packets, auto_adjust, back_adjust,
):
    documents, _calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., True])
    with owned("native-price-adjusted"):
        frame = _yf_net.ticker("SYNTH.MI").history(
            period="5y", interval="1d", auto_adjust=auto_adjust,
            back_adjust=back_adjust, keepna=True,
        )
    assert frame["Close"].iloc[0] == 100.
    assert np.isnan(frame["Close"].iloc[1])


def test_native_invalid_only_direct_rate_still_tries_a_valid_inverse(native_packets):
    documents, calls, _responses, stored = native_packets
    documents["USDEUR=X"] = chart_document("USDEUR=X", [True, True])
    documents["EURUSD=X"] = chart_document("EURUSD=X", [2., 2.])
    with owned("native-price-source-recovery"):
        result = enricher._fetch_fx_pair_uncached("USD")
    assert result.tolist() == [.5, .5]
    assert [symbol for symbol, _ in calls] == ["USDEUR=X", "EURUSD=X"]
    assert len(stored) == 1 and stored[0][1].tolist() == [.5, .5]


def test_native_boolean_failure_does_not_hide_a_later_positive_response(native_packets):
    documents, calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = chart_document("SYNTH.MI", [True, True])
    with owned("native-price-later-recovery"):
        first = enricher._download_price_history("SYNTH.MI")
        documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., 101.])
        second = enricher._download_price_history("SYNTH.MI")
    assert first is None or not enricher._history_has_market_evidence(first)
    assert second["Close"].tolist() == second["Price Close"].tolist() == [100., 101.]
    assert len(calls) == 2


@pytest.mark.parametrize("interval", ["1h", "30m"])
def test_native_intraday_and_resampling_do_not_recover_a_boolean_price(
    native_packets, interval,
):
    documents, _calls, _responses, _stored = native_packets
    documents["SYNTH.MI"] = chart_document("SYNTH.MI", [100., True])
    with owned("native-price-intraday"):
        frame = _yf_net.ticker("SYNTH.MI").history(
            period="1mo", interval=interval, auto_adjust=False, keepna=True,
        )
    close = frame["Close"].dropna()
    assert close.tolist() == [100.]


def test_native_guard_is_installed_on_the_owned_response_factory_only():
    from tarzan.runtime.yahoo_response import YahooChartResponse

    native_parser = Response.json
    with owned("native-price-factory") as session:
        client = session.transport.yahoo_client()
        assert client.response_class is YahooChartResponse
        assert isinstance(client.response_class(), Response)
        assert session.transport.coordinator.snapshot()["requested"] == 0
    with Session() as unowned_client:
        assert unowned_client.response_class is Response
    assert Response.json is native_parser


def test_response_raw_bytes_and_repeated_independent_parses_are_preserved():
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = chart_document("SYNTH.MI", [100., True])
    response = YahooChartResponse()
    raw = json.dumps(document).encode()
    response.content = raw
    original = Response()
    original.content = raw
    first = response.json()
    first["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = 999.
    second = response.json()
    assert second["chart"]["result"][0]["indicators"]["quote"][0]["close"] == [100., None]
    assert response.content is raw and json.loads(response.text) == document
    assert original.json() == document


@pytest.mark.parametrize("document", [
    None, True, [True, 101.], {"quoteResponse": {"result": [{"tradeable": True}]}},
    {"chart": None}, {"chart": {"result": None, "error": {"retryable": True}}},
    {"chart": {"result": [None, True, {"indicators": None}]}},
    {"chart": {"result": [{"indicators": {"quote": [None, True, {"close": True}]}}]}},
])
def test_guard_preserves_nonprice_documents_and_malformed_envelopes(document):
    from tarzan.runtime.yahoo_response import YahooChartResponse

    response = YahooChartResponse()
    response.content = json.dumps(document).encode()
    assert response.json() == document


def test_guard_does_not_reinterpret_timestamps_flags_volume_or_actions():
    from tarzan.runtime.yahoo_response import YahooChartResponse

    document = chart_document("SYNTH.MI", [100., 101.])
    result = document["chart"]["result"][0]
    result["timestamp"] = [False, True]
    result["indicators"]["quote"][0]["volume"] = [False, True]
    result["meta"]["custom"] = {"enabled": True, "price": True}
    result["events"] = {
        "dividends": {"synthetic": {"date": 1792713600, "amount": 1.25}},
        "splits": {"synthetic": {
            "date": 1792713600, "numerator": 2., "denominator": 1., "splitRatio": "2:1",
        }},
    }
    response = YahooChartResponse()
    response.content = json.dumps(document).encode()
    assert response.json() == document


@pytest.mark.parametrize("missing_adjusted", [False, True])
def test_native_valid_adjustments_and_missing_adjusted_array_keep_sdk_contract(
    native_packets, missing_adjusted,
):
    documents, _calls, _responses, _stored = native_packets
    document = chart_document("SYNTH.MI", [100., 101.])
    indicators = document["chart"]["result"][0]["indicators"]
    if missing_adjusted:
        del indicators["adjclose"]
    else:
        indicators["adjclose"][0]["adjclose"] = [90., 95.]
    documents["SYNTH.MI"] = document
    with owned("native-price-adjustment-control"):
        frame = enricher._download_price_history("SYNTH.MI")
    assert frame["Price Close"].tolist() == [100., 101.]
    assert frame["Close"].tolist() == ([100., 101.] if missing_adjusted else [90., 95.])


def test_native_http_status_errors_and_response_identity_are_preserved():
    from curl_cffi.requests.exceptions import HTTPError
    from tarzan.runtime.yahoo_response import YahooChartResponse

    response = YahooChartResponse()
    response.status_code, response.ok, response.reason = 404, False, "Not Found"
    response.content = b'{"chart":{"result":null,"error":{"code":"Not Found"}}}'
    with pytest.raises(HTTPError) as caught:
        response.raise_for_status()
    assert caught.value.response is response
    assert response.json()["chart"]["error"]["code"] == "Not Found"
    response.close()


def test_native_json_parse_errors_are_not_hidden_or_replaced():
    from tarzan.runtime.yahoo_response import YahooChartResponse

    native, owned_response = Response(), YahooChartResponse()
    native.content = owned_response.content = b'{"chart":'
    with pytest.raises(ValueError) as native_error:
        native.json()
    with pytest.raises(type(native_error.value)) as admitted_error:
        owned_response.json()
    assert str(admitted_error.value) == str(native_error.value)


def test_native_concurrent_price_sources_remain_independent_and_drained(native_packets):
    documents, calls, responses, _stored = native_packets
    symbols = [f"SYNTH{number}.MI" for number in range(8)]
    for number, symbol in enumerate(symbols):
        get_tz_cache().store(symbol, "UTC")
        documents[symbol] = chart_document(symbol, [100. + number, True])
    with owned("native-price-concurrent") as session, ThreadPoolExecutor(max_workers=8) as pool:
        pending = [
            pool.submit(copy_context().run, lambda symbol=symbol: (
                _yf_net.ticker(symbol).history(
                    period="5y", interval="1d", auto_adjust=False, keepna=True,
                )
            )) for symbol in symbols
        ]
        frames = [future.result(timeout=10) for future in pending]
        observation = session.transport.coordinator.snapshot()
    assert len(calls) == len(responses) == len(symbols)
    assert observation["capacity"] == 4 and observation["peak"] <= 4
    assert observation["acquired"] == observation["released"] == len(symbols)
    assert observation["active"] == observation["queued_essential"] == 0
    for number, frame in enumerate(frames):
        assert frame["Close"].iloc[0] == 100. + number
        assert np.isnan(frame["Close"].iloc[1])
