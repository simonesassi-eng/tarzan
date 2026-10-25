"""Cache availability is not proved by span alone; young heads stay live."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import enricher, exchange_calendar, price_cache
from tarzan.runtime.session import RunContext, RunMode


NOW = datetime(2026, 10, 25, 20, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clock():
    context = RunContext("cache-coverage", RunMode.LIVE, None, NOW)
    with runtime.bind_context(context):
        yield


def dense():
    return pd.DataFrame(
        {"Close": 100., "Price Close": 100.},
        index=pd.date_range("2021-10-01", "2026-10-23", freq="B"),
    )


def test_distant_endpoints_are_not_a_complete_history():
    frame = dense().iloc[[0, -1]]
    assert not price_cache.covers_period(frame, "5y")


def test_internal_gap_requires_backfill():
    frame = dense().drop(dense().index[100])
    assert not price_cache.covers_period(frame, "5y")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1., 0., True, complex(100., 1.)])
def test_invalid_historical_close_is_not_coverage(invalid):
    frame = dense().astype(object)
    frame.loc[frame.index[100], "Close"] = invalid
    assert not price_cache.covers_period(frame, "5y")


def test_invalid_price_only_close_cannot_certify_the_accounting_head():
    frame = dense()
    frame.loc[frame.index[100], "Price Close"] = float("nan")
    assert not price_cache.covers_period(frame, "5y")


@pytest.mark.parametrize("bad_index", [
    ["old", "new"], [0, 1], pd.to_datetime(["2021-10-01", "2035-01-01"]),
])
def test_invalid_or_synthetic_future_index_fails_closed(bad_index):
    frame = pd.DataFrame({"Close": [100., 101.]}, index=bad_index)
    assert not price_cache.covers_period(frame, "5y")


def test_symbol_calendar_distinguishes_holidays_from_missing_sessions():
    index = pd.date_range("2021-10-01", "2026-10-23", freq="D")
    index = index[[exchange_calendar.is_session("COVER.MI", day.date()) for day in index]]
    frame = pd.DataFrame({"Close": 100., "Price Close": 100.}, index=index)
    assert price_cache.covers_period(frame, "5y", symbol="COVER.MI")
    assert not price_cache.covers_period(frame.drop(frame.index[100]), "5y", symbol="COVER.MI")
    assert not price_cache.covers_period(frame, "5y", symbol="COVER.UNKNOWN")


def test_young_positive_head_still_requires_full_provider_validation():
    frame = dense().iloc[-60:].copy()
    frame.attrs["_tarzan_yahoo_daily_head_v1"] = {
        "symbol": "COVER.MI", "first_trade_at": str(frame.index[0]),
        "observed_at": NOW.isoformat(),
    }
    # Inception is availability evidence, not a provider revision token.
    assert not price_cache.covers_period(frame, "5y")


def test_fetcher_preserves_an_injected_legacy_two_argument_predicate(monkeypatch):
    calls = []
    monkeypatch.setattr(price_cache, "covers_period", lambda frame, period: calls.append(period) or True)
    monkeypatch.setattr(price_cache, "load_history", lambda symbol: dense())
    monkeypatch.setattr(price_cache, "store_history", lambda *args: None)
    monkeypatch.setattr(enricher, "_download_price_history", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kwargs: fn())
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, frame: frame)
    enricher._fetch_history_uncached("COVER.MI")
    assert calls == ["5y"]
