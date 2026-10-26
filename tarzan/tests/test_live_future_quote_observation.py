"""A native parsed quote cannot gain price authority from a future instant."""

from datetime import datetime, timedelta, timezone
import json

from curl_cffi.requests import Response
import pandas as pd
import pytest
from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan.data import enricher
from tarzan.instruments.valuation import ValuationOrigin
from tarzan.models.holding import Holding
from tarzan.tests.test_yahoo_core_info import owned


@pytest.mark.parametrize("observation", [
    "past", "future", "beyond_datetime", "boolean", "infinity", "missing",
])
def test_native_future_quote_is_rejected_without_undating_its_price(monkeypatch, observation):
    observed = datetime.now(timezone.utc) + (
        timedelta(days=30) if observation == "future" else -timedelta(minutes=2)
    )
    raw_time = {
        "past": int(observed.timestamp()), "future": int(observed.timestamp()),
        "beyond_datetime": 10**20, "boolean": True, "infinity": float("inf"),
        "missing": None,
    }[observation]
    calls = []
    monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
    YfData.cache_get.cache_clear()

    def get(self, url, params=None, *arguments, **keywords):
        if "/quoteSummary/" in url:
            calls.append("summary")
            payload = {"quoteSummary": {"result": [{
                "symbol": "SYNFUTURE.MI",
                "quoteType": {"quoteType": "EQUITY", "currency": "EUR"},
                "summaryDetail": {},
            }], "error": None}}
        elif "/v7/finance/quote?" in url:
            calls.append("quote")
            assert params["symbols"] == "SYNFUTURE.MI"
            payload = {"quoteResponse": {"result": [{
                "symbol": "SYNFUTURE.MI", "currency": "EUR",
                "quoteType": "EQUITY", "shortName": "Synthetic future quote",
                "regularMarketPrice": 5000.0,
                "regularMarketTime": raw_time,
                "exchangeTimezoneName": "Europe/Rome",
            }], "error": None}}
        elif "/v8/finance/chart/" in url:
            calls.append("chart")
            payload = {"chart": {"result": None, "error": {
                "code": "Not Found", "description": "Synthetic unavailable history",
            }}}
        elif "/fundamentals-timeseries/" in url:
            calls.append("complementary")
            payload = {"timeseries": {"error": None, "result": []}}
        else:
            pytest.fail("unexpected native source boundary")
        response = Response()
        response.content = json.dumps(payload).encode()
        response.status_code = 404 if "chart" in payload else 200
        response.ok = response.status_code == 200
        response.encoding, response.url, response.reason = "utf-8", url, ""
        return response

    monkeypatch.setattr(YfData, "get", get)
    holding = Holding(
        isin="SYNTHETIC", ticker="SYNFUTURE.MI", quantity=1.0,
        cost_basis_eur=73.0, market_value_eur=73.0, currency="EUR",
        security_type="STOCK", instrument_type="Stock",
    )
    try:
        with owned():
            holding, metadata = enricher._enrich_single(holding)
        assert "summary" in calls and "quote" in calls and "chart" in calls
        assert metadata["regularMarketPrice"] == 5000.0
        assert metadata.get("regularMarketTime") == raw_time
        if observation in {"past", "future"}:
            assert pd.Timestamp(metadata["regularMarketTime"], unit="s", tz="UTC") == (
                pd.Timestamp(observed).floor("s")
            )
        if observation not in {"past", "missing"}:
            assert holding.current_price is None
            assert holding.current_value == 73.0
            assert holding.valuation_result.origin is ValuationOrigin.ANCHOR
        elif observation == "missing":
            assert holding.current_price == 5000.0
            assert holding.price_observation_timestamp is None
            assert holding.price_is_fallback is True
            assert holding.valuation_result.origin is ValuationOrigin.FALLBACK
        else:
            assert holding.current_price == 5000.0
            assert holding.current_value == 5000.0
            assert holding.valuation_result.origin is ValuationOrigin.MARKET
    finally:
        YfData.cache_get.cache_clear()
