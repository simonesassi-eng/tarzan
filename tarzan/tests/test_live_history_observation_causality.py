"""Native daily-history observations must not gain authority from a future row."""

from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

from curl_cffi.requests import Response
import pandas as pd
import pytest
from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan.data import enricher
from tarzan.models.holding import Holding
from tarzan.tests.test_yahoo_core_info import owned


@pytest.mark.parametrize("include_future", [False, True])
def test_native_daily_future_row_cannot_replace_genuine_current_evidence(
    monkeypatch, include_future,
):
    now = datetime.now(timezone.utc)
    zone = ZoneInfo("Europe/Rome")
    past = pd.Timestamp(now.astimezone(zone).date()) - pd.offsets.BDay(1)
    days = [past]
    if include_future:
        days.append(pd.Timestamp(now.astimezone(zone).date()) + pd.offsets.BDay(30))
    times = [int(day.tz_localize(zone).timestamp()) for day in days]
    prices = [100.0] + ([5000.0] if include_future else [])
    calls = []
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    YfData.cache_get.cache_clear()
    today = pd.Timestamp(now.astimezone(zone).date()).tz_localize(zone)

    def get(self, url, params=None, *arguments, **keywords):
        if "/quoteSummary/" in url:
            calls.append("summary")
            document = {"quoteSummary": {"result": [{
                "symbol": "SYNCHRON.MI",
                "quoteType": {"quoteType": "EQUITY", "currency": "EUR"},
                "summaryDetail": {},
            }], "error": None}}
        elif "/v7/finance/quote?" in url:
            calls.append("quote")
            document = {"quoteResponse": {"result": [{
                "symbol": "SYNCHRON.MI", "currency": "EUR", "quoteType": "EQUITY",
                "regularMarketPrice": 100.0,
                "regularMarketTime": int((now - timedelta(minutes=2)).timestamp()),
                "exchangeTimezoneName": "Europe/Rome", "shortName": "Synthetic",
            }], "error": None}}
        elif "/v8/finance/chart/" in url:
            calls.append("chart")
            period = {
                "start": int((today + pd.Timedelta(hours=9)).timestamp()),
                "end": int((today + pd.Timedelta(hours=17)).timestamp()),
                "timezone": "CET", "gmtoffset": 3600,
            }
            document = {"chart": {"result": [{
                "meta": {
                    "symbol": "SYNCHRON.MI", "currency": "EUR",
                    "instrumentType": "EQUITY", "exchangeTimezoneName": "Europe/Rome",
                    "timezone": "CET", "gmtoffset": 3600, "priceHint": 2,
                    "regularMarketPrice": 100.0,
                    "regularMarketTime": int((now - timedelta(minutes=2)).timestamp()),
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
        elif "/fundamentals-timeseries/" in url:
            calls.append("complementary")
            document = {"timeseries": {"error": None, "result": []}}
        else:
            pytest.fail("unexpected native daily-history boundary")
        response = Response()
        response.content = json.dumps(document).encode()
        response.status_code, response.ok = 200, True
        response.encoding, response.url, response.reason = "utf-8", url, ""
        return response

    monkeypatch.setattr(YfData, "get", get)
    holding = Holding(
        isin="SYNTHETIC", ticker="SYNCHRON.MI", quantity=1.0,
        cost_basis_eur=73.0, market_value_eur=73.0, currency="EUR",
        security_type="STOCK", instrument_type="Stock",
    )
    try:
        with owned():
            holding, metadata = enricher._enrich_single(holding)
        assert "summary" in calls and "quote" in calls and "chart" in calls
        assert metadata["regularMarketPrice"] == 100.0
        assert holding.current_price == 100.0
        assert holding.current_value == 100.0
        assert holding.price_history is not None
        assert holding.price_history.max() == 100.0
        assert holding.price_history.index[-1].date() <= now.astimezone(zone).date()
    finally:
        YfData.cache_get.cache_clear()
