"""A closed current-day request is not a reason to suppress historical data."""

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from yfinance.cache import get_tz_cache
from yfinance.data import YfData

from tarzan import runtime
from tarzan.data import _yf_net, enricher as en, exchange_calendar as ec, market_quotes
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


FRIDAY = date(2026, 10, 23)
SATURDAY = date(2026, 10, 24)
SUNDAY = date(2026, 10, 25)


def frame(last=FRIDAY):
    value = pd.DataFrame(
        {"Close": [100., 101.], "Price Close": [100., 101.]},
        index=pd.to_datetime([last - timedelta(days=1), last]),
    )
    value.attrs["original"] = {"evidence": ["native"]}
    return value


@contextmanager
def owned(day=SUNDAY, *, mode=RunMode.LIVE, native_budget=True):
    context = RunContext(
        "calendar-proof", mode, None if mode is RunMode.LIVE else day,
        datetime.combine(day, datetime.min.time(), timezone.utc),
    )
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
    session = RunSession(
        context, {}, ledger, transport=budget if native_budget else SimpleNamespace(),
    )
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session
        finally:
            budget.close()


@pytest.fixture
def sdk_chart(monkeypatch):
    """Simulate raw Yahoo JSON while exercising the installed SDK parser."""
    calls = []
    answers = {}

    def get(self, url, params=None, *args, **kwargs):
        symbol = url.rsplit("/", 1)[-1]
        calls.append(symbol)
        series = answers.get(symbol, pd.Series([100., 101.], index=pd.to_datetime([
            "2026-10-22", "2026-10-23",
        ])))
        timestamps = [
            int(pd.Timestamp(value).tz_localize("Europe/Berlin").timestamp())
            for value in series.index
        ]
        document = {"chart": {"error": None, "result": [{
            "meta": {
                "symbol": symbol, "currency": "EUR",
                "exchangeTimezoneName": "Europe/Berlin", "instrumentType": "ETF",
                "firstTradeDate": 0, "regularMarketTime": timestamps[-1],
                "dataGranularity": "1d", "validRanges": ["1d", "5d", "1mo", "5y", "max"],
            },
            "timestamp": timestamps,
            "indicators": {
                "quote": [{
                    "open": list(series), "high": list(series),
                    "low": list(series), "close": list(series),
                    "volume": [100] * len(series),
                }],
                "adjclose": [{"adjclose": list(series)}],
            },
        }]}}
        return SimpleNamespace(
            status_code=200, text=json.dumps(document), json=lambda: document,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(YfData, "get", get)

    def prepare(symbols, day=SUNDAY):
        monkeypatch.setattr(en, "_today_local", lambda: day)
        monkeypatch.setattr(market_quotes, "_sibling_symbols", lambda symbol: symbols)
        for symbol in symbols:
            get_tz_cache().store(symbol, "Europe/Berlin")

    return calls, answers, prepare


@pytest.mark.parametrize("symbol,day,closed", [
    ("PROOF.MI", SATURDAY, True),
    ("PROOF.DE", SUNDAY, True),
    ("PROOF.PA", FRIDAY, False),
    ("PROOF.MI", date(2025, 8, 15), True),
    ("PROOF.DE", date(2025, 8, 15), False),
    ("PROOF.L", date(2026, 12, 24), False),  # synthetic future early-close scenario
    ("PROOF.ZZ", SUNDAY, False),
    ("PROOF", SUNDAY, False),
    ("^GDAXI", SUNDAY, False),
    ("EURUSD=X", SUNDAY, False),
    ("PROOF.MI", date(2035, 1, 7), False),  # future, outside bundled coverage
])
def test_closure_requires_explicit_covered_listing_evidence(symbol, day, closed):
    assert ec.daily_session_known_closed(symbol, day) is closed


def test_exchange_group_guess_does_not_authorize_skipping(monkeypatch):
    monkeypatch.setattr(market_quotes, "_exchange_for", lambda symbol: "EU")
    assert not ec.daily_session_known_closed("UNMAPPED.ZZ", SUNDAY)


@pytest.mark.parametrize("table", [
    ({}, {}, {}), ({}, {"XETR": (date(2020, 1, 1), date(2020, 12, 31))}, {}),
])
def test_missing_or_expired_calendar_keeps_the_request(monkeypatch, table):
    monkeypatch.setattr(ec, "_table", lambda: table)
    assert ec.daily_session_known_closed("PROOF.DE", SUNDAY) is False


def test_calendar_error_is_not_closure_evidence(monkeypatch):
    def broken():
        raise ValueError("synthetic corrupt calendar")
    monkeypatch.setattr(ec, "_table", broken)
    assert ec.daily_session_known_closed("PROOF.DE", SUNDAY) is False


def test_actual_corrupt_file_degrades_without_suppressing_data(monkeypatch, tmp_path):
    invalid = tmp_path / "calendar.csv.gz"
    invalid.write_bytes(b"not a gzip document")
    ec._table.cache_clear()
    try:
        with monkeypatch.context() as changes:
            changes.setattr(ec, "_BUNDLED", invalid)
            assert ec.daily_session_known_closed("PROOF.DE", SUNDAY) is False
    finally:
        ec._table.cache_clear()


@pytest.mark.parametrize("bad", [None, "", 0])
def test_invalid_day_keeps_the_request(bad):
    assert ec.daily_session_known_closed("PROOF.DE", bad) is False


@pytest.mark.parametrize("day", [SATURDAY, SUNDAY])
def test_owned_weekend_does_not_acquire_closed_current_day_siblings(sdk_chart, day):
    calls, _answers, prepare = sdk_chart
    prepare(["CALPROOF.DE", "CALPROOF.PA", "CALPROOF.AS", "CALPROOF.F"], day)
    original = frame()
    with owned(day):
        assert en._owns_native_daily_sibling_fill() is True
        result = en._fill_today_from_sibling("CALPROOF.MI", original)
    assert result is original
    assert result.attrs == {"original": {"evidence": ["native"]}}
    assert calls == []


def test_open_monday_keeps_the_return_and_provenance(sdk_chart):
    calls, answers, prepare = sdk_chart
    monday, friday = date(2026, 8, 10), date(2026, 8, 7)
    prepare(["CALMON.DE", "CALMON.PA"], monday)
    answers["CALMON.DE"] = pd.Series([101.2, 102.2], index=pd.to_datetime([friday, monday]))
    original = frame(friday)
    with owned(monday):
        result = en._fill_today_from_sibling("CALMON.MI", original)
    assert calls == ["CALMON.DE"]
    assert result["Close"].iloc[-1] == pytest.approx(101. * 102.2 / 101.2)
    key = en._history_timestamp_key(pd.Timestamp(monday))
    assert result.attrs[en._HISTORY_SYNTHETIC_ATTR][key]["source"] == "CALMON.DE"
    assert len(original) == 2 and len(result) == 3


def test_closed_candidate_does_not_hide_a_later_open_exchange(sdk_chart):
    calls, answers, prepare = sdk_chart
    friday, thursday = date(2025, 8, 15), date(2025, 8, 14)
    prepare(["CALHOL.MI", "CALHOL.DE"], friday)
    answers["CALHOL.DE"] = pd.Series([101., 102.], index=pd.to_datetime([thursday, friday]))
    with owned(friday):
        result = en._fill_today_from_sibling("CALHOL.PA", frame(thursday))
    assert calls == ["CALHOL.DE"]
    assert result["Close"].iloc[-1] == pytest.approx(102.)


def test_owner_holiday_does_not_suppress_an_open_sibling(sdk_chart):
    calls, answers, prepare = sdk_chart
    friday, thursday = date(2025, 8, 15), date(2025, 8, 14)
    prepare(["CALOWNER.DE"], friday)
    answers["CALOWNER.DE"] = pd.Series([101., 102.], index=pd.to_datetime([thursday, friday]))
    with owned(friday):
        result = en._fill_today_from_sibling("CALOWNER.MI", frame(thursday))
    assert calls == ["CALOWNER.DE"]
    assert result["Close"].iloc[-1] == pytest.approx(102.)


def test_unknown_exchange_is_still_asked_and_can_supply_the_bar(sdk_chart):
    calls, answers, prepare = sdk_chart
    prepare(["CALUNKNOWN.DE", "CALUNKNOWN.ZZ"], SUNDAY)
    answers["CALUNKNOWN.ZZ"] = pd.Series([101., 102.], index=pd.to_datetime([FRIDAY, SUNDAY]))
    with owned():
        result = en._fill_today_from_sibling("CALUNKNOWN.MI", frame())
    assert calls == ["CALUNKNOWN.ZZ"]
    assert result["Close"].iloc[-1] == pytest.approx(102.)


def test_historical_sibling_lookup_remains_available_on_a_weekend(sdk_chart):
    calls, _answers, prepare = sdk_chart
    prepare(["CALHISTORY.DE"])
    with owned():
        result = en._sibling_close_series("CALHISTORY.DE")
    assert calls == ["CALHISTORY.DE"]
    assert list(result) == [100., 101.]


def test_existing_current_day_evidence_is_not_discarded_by_the_calendar(monkeypatch, sdk_chart):
    calls, _answers, prepare = sdk_chart
    prepare(["CALCACHED.DE"])
    cached = pd.DataFrame(
        {"Close": [100., 101., 102.], "Price Close": [100., 101., 102.]},
        index=pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-25"]).tz_localize("Europe/Berlin"),
    )
    monkeypatch.setattr(en.price_cache, "load_history", lambda symbol: cached.copy())
    with owned():
        result = en._fill_today_from_sibling("CALCACHED.MI", frame())
    assert calls == ["CALCACHED.DE"]
    assert result["Close"].iloc[-1] == pytest.approx(102.)


@pytest.mark.parametrize("cache", [
    pd.DataFrame({"unrecognized": [100.]}),
    pd.DataFrame({"Close": [100.]}, index=["invalid timestamp"]),
])
def test_uncertain_cache_keeps_the_existing_source_path(monkeypatch, cache):
    monkeypatch.setattr(en.price_cache, "load_history", lambda symbol: cache)
    assert en._closed_sibling_has_observed_today("UNCERTAIN.DE", SUNDAY) is True


def test_cache_reader_error_does_not_authorize_suppressing_data(monkeypatch):
    def broken(symbol):
        raise OSError("synthetic cache failure")
    monkeypatch.setattr(en.price_cache, "load_history", broken)
    assert en._closed_sibling_has_observed_today("UNCERTAIN.DE", SUNDAY) is True


def test_direct_injected_caller_keeps_its_existing_lookup_order(monkeypatch):
    calls = []
    monkeypatch.setattr(en, "_today_local", lambda: SUNDAY)
    monkeypatch.setattr(market_quotes, "_sibling_symbols", lambda symbol: ["DIRECT.DE", "DIRECT.PA"])
    monkeypatch.setattr(en, "_sibling_close_series", lambda symbol: calls.append(symbol))
    original = frame()
    assert en._fill_today_from_sibling("DIRECT.MI", original) is original
    assert calls == ["DIRECT.DE", "DIRECT.PA"]


@pytest.mark.parametrize("module,port", [
    (en, "_sibling_close_series"), (en, "_download_price_history"),
    (en, "_retry"), (en, "_space_yf_call"), (_yf_net, "ticker"),
    (en.yf, "Ticker"), (en._SDK_CANONICAL_TICKER, "history"),
    (en._SDK_PRICE_HISTORY, "history"),
])
def test_injected_entry_port_preserves_the_unoptimized_request(monkeypatch, sdk_chart, module, port):
    calls, _answers, prepare = sdk_chart
    prepare(["CALINJECT.DE"])
    original = getattr(module, port)
    monkeypatch.setattr(module, port, lambda *args, **kwargs: original(*args, **kwargs))
    with owned():
        assert en._owns_native_daily_sibling_fill() is False
        result = en._fill_today_from_sibling("CALINJECT.MI", frame())
    assert calls == ["CALINJECT.DE"]
    assert len(result) == 2


def test_pinned_or_unowned_transport_has_no_planning_authority():
    with owned(mode=RunMode.POINT_IN_TIME):
        assert en._owns_native_daily_sibling_fill() is False
    with owned(native_budget=False):
        assert en._owns_native_daily_sibling_fill() is False
    assert en._owns_native_daily_sibling_fill() is False


@pytest.mark.parametrize("bad", [(), [], None])
def test_invalid_native_port_manifest_cannot_authorize_a_skip(monkeypatch, bad):
    monkeypatch.setattr(en, "_NATIVE_DAILY_SIBLING_PORTS", bad)
    with owned():
        assert en._owns_native_daily_sibling_fill() is False


@pytest.mark.parametrize("port", ["ticker", "price_parser"])
def test_preimport_wrappers_are_not_recognized_as_native(port):
    script = """
import functools
import importlib
from datetime import datetime, timezone
from tarzan import runtime
from tarzan.data import _yf_net, enricher
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings
from yfinance.scrapers.history import PriceHistory
module = _yf_net if PORT == "ticker" else PriceHistory
attribute = "ticker" if PORT == "ticker" else "history"
original = getattr(module, attribute)
@functools.wraps(original)
def wrapper(*args, **kwargs):
    return original(*args, **kwargs)
setattr(module, attribute, wrapper)
importlib.reload(enricher)
context = RunContext("import-order", RunMode.LIVE, None, datetime.now(timezone.utc))
ledger = RunLedger(context.attempt_id)
budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
session = RunSession(context, {}, ledger, transport=budget)
with runtime.bind_context(context), activate_session(session):
    print(enricher._owns_native_daily_sibling_fill())
budget.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", "PORT = " + repr(port) + "\n" + script],
        cwd=Path(__file__).resolve().parents[2], check=True,
        capture_output=True, text=True, timeout=20,
    )
    assert result.stdout.strip() == "False"
