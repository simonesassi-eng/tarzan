"""Native chart parsing retains direct/inverse FX and overnight authority."""

from datetime import datetime, timedelta, timezone
import json

from curl_cffi.requests import Response
import pandas as pd
import pytest
from yfinance.config import YfConfig
from yfinance.cache import get_tz_cache
from yfinance.data import YfData

from tarzan import runtime
from tarzan.data import current_session as cs, enricher, history_time as ht, price_cache
from tarzan.models.holding import Holding
from tarzan.runtime.session import RunContext, RunMode
from tarzan.tests.test_yahoo_core_info import owned


@pytest.fixture
def native_chart(monkeypatch):
    now = datetime(2026, 10, 26, 22, 30, tzinfo=timezone.utc)

    class ReceiptDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(cs, "datetime", ReceiptDatetime)
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    # Synthetic private identities make every parameter independent of the
    # SDK's process-wide timezone store. A separate cold case exercises that
    # establishment request; these cases test daily parsing and source choice.
    for pair in ("USDEUR=X", "EURUSD=X"):
        get_tz_cache().store(pair, "UTC")
    answers, calls, stored = {}, [], []
    monkeypatch.setattr(price_cache, "store_history", lambda symbol, frame: (
        stored.append((symbol, frame.copy(deep=True)))
    ))
    YfData.cache_get.cache_clear()

    def get(self, url, params=None, *args, **kwargs):
        if "/v8/finance/chart/" not in url:
            pytest.fail("unexpected non-chart native FX request")
        symbol = url.rsplit("/", 1)[-1]
        calls.append((symbol, dict(params or {})))
        definition = answers.get(symbol)
        if definition is None:
            document = {"chart": {"result": None, "error": {
                "code": "Not Found", "description": "Synthetic absent pair",
            }}}
            status = 404
        else:
            dates, prices, overnight, kind = definition
            period = {
                "start": int((now - timedelta(minutes=30)).timestamp()),
                "end": int((now + timedelta(hours=23, minutes=30)).timestamp()),
                "timezone": "UTC", "gmtoffset": 0,
            } if overnight else {
                "start": int(pd.Timestamp(now.date(), tz="UTC").timestamp()),
                "end": int((pd.Timestamp(now.date(), tz="UTC") + pd.Timedelta(
                    hours=23, minutes=59,
                )).timestamp()),
                "timezone": "UTC", "gmtoffset": 0,
            }
            times = [int(pd.Timestamp(value).tz_localize("UTC").timestamp()) for value in dates]
            document = {"chart": {"result": [{
                "meta": {
                    "symbol": symbol, "currency": "EUR",
                    "exchangeTimezoneName": "UTC", "timezone": "UTC", "gmtoffset": 0,
                    "instrumentType": kind,
                    "regularMarketTime": int((now - timedelta(minutes=1)).timestamp()),
                    "regularMarketPrice": prices[-1], "priceHint": 4,
                    "firstTradeDate": times[0] - 10 * 365 * 86400,
                    "dataGranularity": "1d",
                    "validRanges": ["1d", "5d", "1mo", "5y", "max"],
                    "currentTradingPeriod": {"regular": period},
                    "tradingPeriods": [[period]],
                },
                "timestamp": times,
                "indicators": {
                    "quote": [{
                        "open": prices, "high": prices, "low": prices,
                        "close": prices, "volume": [100] * len(prices),
                    }],
                    "adjclose": [{"adjclose": prices}],
                },
                "events": {},
            }], "error": None}}
            status = 200
        response = Response()
        response.content = json.dumps(document).encode()
        response.status_code, response.ok = status, status == 200
        response.encoding, response.url, response.reason = "utf-8", url, ""
        return response

    monkeypatch.setattr(YfData, "get", get)
    context = RunContext("native-fx-causality", RunMode.LIVE, None, now - timedelta(hours=1))
    with runtime.bind_context(context):
        try:
            yield now, answers, calls, stored
        finally:
            YfData.cache_get.cache_clear()


@pytest.mark.parametrize("inverse", [False, True])
def test_native_fx_rejects_future_rows_before_reciprocal_merge_and_store(native_chart, inverse):
    now, answers, calls, stored = native_chart
    pair = "EURUSD=X" if inverse else "USDEUR=X"
    answers[pair] = (
        ["2026-10-23", "2026-12-07"], [2., .0002] if inverse else [.5, 5000.],
        False, "CURRENCY",
    )
    with owned():
        result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result) == [.5]
    assert result.index[-1].date() < now.date()
    assert stored[0][0] == "FX_USD" and list(stored[0][1]) == [.5]
    assert {symbol for symbol, _params in calls} == (
        {"USDEUR=X", "EURUSD=X"} if inverse else {"USDEUR=X"}
    )


def test_future_only_direct_fx_does_not_hide_a_valid_inverse_source(native_chart):
    _now, answers, calls, stored = native_chart
    answers["USDEUR=X"] = (["2026-10-27"], [5000.], False, "CURRENCY")
    answers["EURUSD=X"] = (["2026-10-23"], [2.], False, "CURRENCY")
    with owned():
        result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result) == [.5]
    assert {symbol for symbol, _params in calls} == {"USDEUR=X", "EURUSD=X"}
    assert len(stored) == 1


@pytest.mark.parametrize("inverse", [False, True])
def test_native_open_overnight_fx_uses_trade_time_not_tomorrow_label(native_chart, inverse):
    now, answers, calls, stored = native_chart
    pair = "EURUSD=X" if inverse else "USDEUR=X"
    answers[pair] = (
        ["2026-10-23", "2026-10-27"], [2., 1.25] if inverse else [.5, .8],
        True, "CURRENCY",
    )
    with owned():
        fx = enricher._fetch_fx_pair("USD")
        assert fx.index[-1] > now
        assert list(fx) == [.5, .8]
        assert ht.observation_time(fx, fx.index[-1]) == now - timedelta(minutes=1)
        holding = Holding(
            isin="SYNTHETIC", ticker="SYNTH.UNKNOWN", quantity=1,
            cost_basis_eur=73, market_value_eur=73,
            currency="USD", security_type="STOCK", instrument_type="Stock",
        )
        enricher._set_price_data(holding, pd.DataFrame(), {
            "regularMarketPrice": 100,
            "regularMarketTime": (now - timedelta(minutes=5)).timestamp(),
        }, "USD")
        assert holding.current_price == 80
        assert holding.price_observation_timestamp == now - timedelta(minutes=5)
        assert not holding.price_is_fallback
    assert stored[0][1].attrs[ht.SESSION_EVIDENCE_ATTR]
    # Timezone establishment + the ordinary history only; no lazy 5d/1h read.
    assert all(params.get("interval") == "1d" for _symbol, params in calls)
    assert len(calls) == (2 if inverse else 1)


def test_native_cash_equity_cannot_borrow_overnight_fx_label_authority(native_chart):
    _now, answers, _calls, stored = native_chart
    answers["USDEUR=X"] = (
        ["2026-10-23", "2026-10-27"], [.5, 5000.], True, "EQUITY",
    )
    with owned():
        result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result) == [.5]
    assert not ht.session_evidence(result)
    assert list(stored[0][1]) == [.5]


def test_cold_native_fx_keeps_timezone_establishment_and_no_lazy_intraday_read(native_chart):
    now, answers, calls, stored = native_chart
    get_tz_cache().store("CHFEUR=X", None)
    answers["CHFEUR=X"] = (
        ["2026-10-23", "2026-10-27"], [.9, 1.], True, "CURRENCY",
    )
    with owned():
        result = enricher._fetch_fx_pair_uncached("CHF")
    assert list(result) == [.9, 1.]
    assert ht.observation_time(result, result.index[-1]) == now - timedelta(minutes=1)
    assert len(calls) == 2
    assert all(params["interval"] == "1d" for _symbol, params in calls)
    assert calls[0][1]["range"] == "1d" and calls[1][1]["range"] == "5y"
    assert len(stored) == 1
