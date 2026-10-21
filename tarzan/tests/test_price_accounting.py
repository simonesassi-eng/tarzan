"""Price rulers and adjustment-cache regressions from the correctness audit."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import current_session, enricher, price_cache
from tarzan.data.market_quotes import _fetch_official_quotes as production_quote_batch
from tarzan.engine.returns_builder import build_order_derived_series
from tarzan.engine.stats import twr
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType


def _holding():
    return Holding(
        isin="US0000000001", ticker="EXAMPLE.MI", quantity=10,
        cost_basis_eur=1000, market_value_eur=980, currency="EUR",
        security_type="STOCK",
    )


def _order(day, kind, quantity, net, price=None):
    return Order(
        date=dt.date(2025, 1, day), trade_date=dt.date(2025, 1, day),
        type=kind, isin="US0000000001", name="Example", ticker="EXAMPLE.MI",
        quantity=quantity, currency="EUR", price_native=price, fx_rate=1,
        gross_eur=net, fees_eur=0, net_eur=net,
        instrument_kind=InstrumentKind.STOCK,
    )


def _provider(monkeypatch, frame, *, full_fails=False):
    requests = []

    def history(**kwargs):
        requests.append(kwargs)
        assert kwargs["auto_adjust"] is False
        if "start" in kwargs:
            return frame.loc[pd.Timestamp(kwargs["start"]):].copy()
        return pd.DataFrame() if full_fails else frame.copy()

    monkeypatch.setattr(enricher.yf, "Ticker", lambda symbol: SimpleNamespace(history=history))
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, result: result)
    monkeypatch.setattr(price_cache, "covers_period", lambda *args: True)
    enricher.reset_run_caches()
    return requests


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    enricher.reset_run_caches()
    yield
    enricher.reset_run_caches()


def test_download_retains_raw_and_total_return_in_one_response(monkeypatch):
    import yfinance as yf

    frame = pd.DataFrame(
        {
            "Close": [100., 98., 99.],
            "Adj Close": [98., 98., 99.],
            "Open": [100., 98., 99.],
            "High": [101., 99., 100.],
            "Low": [99., 97., 98.],
            "Volume": [10, 20, 30],
            "Dividends": [0., 2., 0.],
        },
        index=pd.date_range("2025-01-01", periods=3),
    )
    requests = _provider(monkeypatch, frame)
    result = enricher._download_price_history("EXAMPLE.MI")
    assert result["Price Close"].tolist() == [100, 98, 99]
    assert result["Close"].tolist() == [98, 98, 99]
    assert result["Open"].iloc[0] == 98
    assert result["High"].iloc[0] == pytest.approx(98.98)
    assert result["Volume"].tolist() == [10, 20, 30]
    assert result["Dividends"].tolist() == [0, 2, 0]
    assert "Adj Close" not in result
    assert len(requests) == 1
    assert "period" in requests[0]
    assert frame["Close"].iloc[0] == 100  # no mutation of the source response
    # Exercise the installed dependency's actual implementation as an oracle.
    pd.testing.assert_frame_equal(
        result.drop(columns=["Price Close"]),
        yf.utils.auto_adjust(frame),
        check_like=True,
    )


@pytest.mark.parametrize("factor", [0.98, 0.5, 1.1])
def test_cold_and_warm_cache_have_identical_adjustment_scale(
    monkeypatch, isolated_cache, factor,
):
    days = pd.date_range("2025-01-01", periods=10)
    old = pd.DataFrame(
        {"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9],
    )
    vendor = pd.DataFrame(
        {
            "Close": [100.] * 9 + [100. * factor],
            "Adj Close": [100. * factor] * 10,
        }, index=days,
    )
    requests = _provider(monkeypatch, vendor)
    cold = enricher._fetch_history("EXAMPLE.MI")
    price_cache.store_history("EXAMPLE.MI", old)
    enricher.reset_run_caches()
    requests.clear()

    warm = enricher._fetch_history("EXAMPLE.MI")

    pd.testing.assert_frame_equal(warm, cold)
    assert warm["Close"].iloc[0] == pytest.approx(100 * factor)
    assert warm["Price Close"].iloc[0] == 100
    assert warm["Close"].pct_change().dropna().sum() == pytest.approx(0)
    assert len(requests) == 2  # incremental tail, then one coherent full read
    assert "start" in requests[0]
    assert "period" in requests[1]
    stored = price_cache.load_history("EXAMPLE.MI")
    pd.testing.assert_frame_equal(stored, warm, check_freq=False)


def test_normal_market_move_keeps_the_fast_incremental_path(monkeypatch, isolated_cache):
    days = pd.date_range("2025-01-01", periods=10)
    price_cache.store_history(
        "EXAMPLE.MI",
        pd.DataFrame({"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9]),
    )
    vendor = pd.DataFrame(
        {"Close": [100.] * 9 + [101.], "Adj Close": [100.] * 9 + [101.]},
        index=days,
    )
    requests = _provider(monkeypatch, vendor)
    result = enricher._fetch_history("EXAMPLE.MI")
    assert result["Close"].iloc[0] == 100
    assert result["Close"].iloc[-1] == 101
    assert len(requests) == 1
    assert "start" in requests[0]
    first = enricher._history_timestamp_key(days[0])
    assert result.attrs[enricher._HISTORY_ORIGINS_ATTR][first] == enricher._HISTORY_ORIGIN_CACHE


def test_failed_full_refresh_retains_stale_cache_not_a_mixed_scale(
    monkeypatch, isolated_cache,
):
    days = pd.date_range("2025-01-01", periods=10)
    old = pd.DataFrame(
        {"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9],
    )
    price_cache.store_history("EXAMPLE.MI", old)
    vendor = pd.DataFrame(
        {"Close": [100.] * 9 + [98.], "Adj Close": [98.] * 10}, index=days,
    )
    requests = _provider(monkeypatch, vendor, full_fails=True)
    result = enricher._fetch_history("EXAMPLE.MI")
    pd.testing.assert_frame_equal(result, old)
    assert len(requests) == 2
    assert set(result.attrs[enricher._HISTORY_ORIGINS_ATTR].values()) == {
        enricher._HISTORY_ORIGIN_CACHE,
    }
    holding = _holding()
    enricher._set_price_data(holding, result, {}, "EUR")
    assert holding.current_price == 100
    assert holding.price_is_fallback


def test_legacy_adjusted_only_cache_is_upgraded_with_a_full_raw_head(
    monkeypatch, isolated_cache,
):
    days = pd.date_range("2025-01-01", periods=10)
    price_cache.store_history(
        "EXAMPLE.MI", pd.DataFrame({"Close": [98.] * 9}, index=days[:9]),
    )
    vendor = pd.DataFrame(
        {"Close": [100.] * 9 + [98.], "Adj Close": [98.] * 10}, index=days,
    )
    requests = _provider(monkeypatch, vendor)
    result = enricher._fetch_history("EXAMPLE.MI")
    assert len(requests) == 1
    assert "period" in requests[0]
    assert result["Price Close"].notna().all()
    assert result["Price Close"].iloc[0] == 100
    assert result["Close"].iloc[0] == 98


def test_pinned_valuation_uses_raw_price_without_future_dividend_adjustment(
    monkeypatch, isolated_cache,
):
    days = pd.date_range("2025-01-01", periods=10)
    frame = pd.DataFrame(
        {"Close": [98.] * 10, "Price Close": [100.] * 9 + [98.]}, index=days,
    )
    price_cache.store_history("EXAMPLE.MI", frame)
    runtime.configure(as_of=dt.date(2025, 1, 3), deterministic=True)
    monkeypatch.setattr(enricher.yf, "Ticker", lambda symbol: pytest.fail("pinned run performed I/O"))
    history = enricher._fetch_history("EXAMPLE.MI")
    holding = _holding()
    enricher._set_price_data(
        holding, history, {"regularMarketPrice": 50, "regularMarketTime": 2_000_000_000}, "EUR",
    )
    assert holding.current_price == 100
    assert holding.price_history.iloc[-1] == 98
    assert holding.price_history_price_only.iloc[-1] == 100
    series = build_order_derived_series(
        [_order(1, OrderType.BUY, 10, -1000, 100)],
        {holding.isin: holding}, today=dt.date(2025, 1, 3),
    )
    assert series.valuations[-1][1] == 1000
    pd.testing.assert_frame_equal(price_cache.load_history("EXAMPLE.MI"), frame)


@pytest.mark.parametrize("income_day", [2, 3])
def test_cash_dividend_and_price_only_tape_count_income_exactly_once(income_day):
    days = pd.date_range("2025-01-01", periods=3)
    holding = _holding()
    holding.price_history = pd.Series([98., 98., 98.], index=days)
    holding.price_history_price_only = pd.Series([100., 98., 98.], index=days)
    orders = [
        _order(1, OrderType.BUY, 10, -1000, 100),
        _order(income_day, OrderType.DIVIDEND, 0, 20),
    ]
    result = build_order_derived_series(orders, {holding.isin: holding}, today=dt.date(2025, 1, 3))
    assert result.history_availability.value == "AVAILABLE"
    assert result.valuations[0][1] == 1000
    assert result.valuations[-1][1] == 980
    assert sum(value for _, value in result.xirr_cashflows) == 0
    assert result.pnl_series.iloc[-1] == 0
    assert twr(result.valuations, result.external_flows, result.span_days).cumulative_pct == pytest.approx(0)
    assert len(set(day for day, _ in result.valuations)) == len(result.valuations)


def test_both_tapes_receive_the_same_currency_and_minor_unit_conversion(monkeypatch):
    days = pd.date_range("2025-01-01", periods=3)
    history = pd.DataFrame(
        {"Close": [9800., 9800., 9800.], "Price Close": [10000., 10000., 9800.]},
        index=days,
    )
    fx = pd.Series([1.2, 1.25, 1.3], index=days)
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
    runtime.configure(as_of=dt.date(2025, 1, 3), deterministic=True)
    holding = _holding()
    enricher._set_price_data(holding, history, {}, "GBp")
    assert holding.price_currency == "GBP"
    assert holding.price_history_native.tolist() == [98, 98, 98]
    assert holding.price_history.tolist() == pytest.approx([117.6, 122.5, 127.4])
    assert holding.price_history_price_only.tolist() == pytest.approx([120, 125, 127.4])
    assert holding.current_price == pytest.approx(127.4)


@pytest.mark.parametrize("field", ["regularMarketPrice", "previousClose"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "inf"])
def test_nonfinite_quotes_cannot_become_portfolio_prices(field, value):
    holding = _holding()
    history = pd.DataFrame(
        {"Close": [100., float("inf"), float("nan")]},
        index=pd.date_range("2025-01-01", periods=3),
    )
    enricher._set_price_data(holding, history, {field: value, "regularMarketTime": 2_000_000_000}, "EUR")
    assert holding.current_price == 100
    assert holding.price_history.tolist() == [100]
    empty = _holding()
    enricher._set_price_data(empty, pd.DataFrame(), {field: value}, "EUR")
    assert empty.current_price is None


def test_live_stamp_does_not_undo_a_dividend_adjustment(monkeypatch):
    holding = _holding()
    days = pd.date_range("2025-01-01", periods=3)
    holding.price_history = pd.Series([98., 98., 98.], index=days)
    holding.price_history_native = holding.price_history.copy()
    holding.price_history_price_only = pd.Series([100., 100., 98.], index=days)
    monkeypatch.setattr(runtime, "today", lambda: dt.date(2025, 1, 3))
    quote = {
        "price": 98, "prev_close": 100,
        "time": int(dt.datetime(2025, 1, 3, 16, tzinfo=dt.timezone.utc).timestamp()),
    }
    monkeypatch.setattr("tarzan.data.market_quotes.official_quotes", lambda symbols: {"EXAMPLE.MI": quote})
    monkeypatch.setattr("tarzan.data.justetf.prefetch", lambda *args: None)
    monkeypatch.setattr("tarzan.data.justetf.quote", lambda *args: None)
    assert current_session.apply_to_holdings([holding]) == ("EXAMPLE.MI",)
    assert holding.price_history.tolist() == [98, 98, 98]
    assert holding.price_history_native.tolist() == [98, 98, 98]
    assert holding.price_history_price_only.tolist() == [100, 100, 98]
    assert holding.current_value == 980


def test_split_repair_keeps_the_two_price_rulers_consistent():
    days = pd.date_range("2025-01-01", periods=40)
    frame = pd.DataFrame(
        {"Close": [100.] * 20 + [10.] * 20, "Price Close": [100.] * 20 + [10.] * 20},
        index=days,
    )
    frame["Stock Splits"] = [0.0] * 20 + [10.0] + [0.0] * 19
    repaired = price_cache.repair_split_jumps(frame)
    assert repaired["Close"].tolist() == [10.] * 40
    assert repaired["Price Close"].tolist() == [10.] * 40


def test_sibling_lookup_also_replaces_a_revised_adjusted_head(monkeypatch, isolated_cache):
    days = pd.date_range("2025-01-01", periods=10)
    price_cache.store_history(
        "EXAMPLE.DE",
        pd.DataFrame({"Close": [100.] * 9, "Price Close": [100.] * 9}, index=days[:9]),
    )
    vendor = pd.DataFrame(
        {"Close": [100.] * 9 + [98.], "Adj Close": [98.] * 10}, index=days,
    )
    requests = _provider(monkeypatch, vendor)
    closes = enricher._sibling_close_series("EXAMPLE.DE")
    assert closes.tolist() == [98.] * 10
    assert len(requests) == 2


def _live_currency_holding(monkeypatch, currency, rate, *, kind="STOCK"):
    from tarzan.models.currency import is_minor, major_code

    holding = _holding()
    holding.ticker = "EXAMPLE"
    holding.currency = currency
    holding.price_currency = major_code(currency)
    holding.price_quote_currency = currency
    holding.security_type = kind
    holding.quantity = 1000 if kind == "BOND" else 10
    days = pd.date_range("2025-01-01", periods=3)
    unit = rate / 100 if kind == "BOND" else rate
    holding.price_history_native = pd.Series([100.] * 3, index=days)
    holding.price_history = holding.price_history_native * unit
    holding.price_history_price_only = holding.price_history.copy()
    holding.current_price = 100 * unit
    holding.current_value = holding.quantity * holding.current_price
    holding.price_is_fallback = True
    multiplier = 100 if is_minor(currency) else 1
    quote = {
        "price": 101 * multiplier, "prev_close": 100 * multiplier,
        "currency": currency,
        "time": int(dt.datetime(2025, 1, 3, 16, tzinfo=dt.timezone.utc).timestamp()),
    }
    monkeypatch.setattr(runtime, "today", lambda: dt.date(2025, 1, 3))
    monkeypatch.setattr("tarzan.data.market_quotes.official_quotes", lambda symbols: {"EXAMPLE": quote})
    monkeypatch.setattr("tarzan.data.justetf.prefetch", lambda *args: None)
    monkeypatch.setattr("tarzan.data.justetf.quote", lambda *args: None)
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: pd.Series([rate] * 3, index=days))
    return holding


@pytest.mark.parametrize(
    ("currency", "rate"),
    [("USD", .95), ("JPY", .006), ("GBP", 1.2), ("GBp", 1.2),
     ("GBX", 1.2), ("ZAc", .05), ("ILa", .25)],
)
def test_live_quote_stamps_each_tape_in_its_own_currency(monkeypatch, currency, rate):
    holding = _live_currency_holding(monkeypatch, currency, rate)
    assert current_session.apply_to_holdings([holding]) == ("EXAMPLE",)
    assert holding.price_history_native.iloc[-1] == 101
    assert holding.price_history.iloc[-1] == pytest.approx(101 * rate)
    assert holding.price_history_price_only.iloc[-1] == pytest.approx(101 * rate)
    assert holding.current_price == pytest.approx(101 * rate)
    assert holding.current_value == pytest.approx(1010 * rate)


def test_live_quote_uses_dated_fx_for_each_endpoint_not_a_future_rate(monkeypatch):
    holding = _live_currency_holding(monkeypatch, "USD", .95)
    fx = pd.Series(
        [.95, .95, .90, .50], index=pd.date_range("2025-01-01", periods=4),
    )
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
    assert current_session.apply_to_holdings([holding]) == ("EXAMPLE",)
    assert holding.price_history_native.iloc[-1] == 101
    assert holding.price_history_price_only.loc["2025-01-02"] == 95
    assert holding.current_price == pytest.approx(90.9)
    assert holding.current_value == pytest.approx(909)


def test_live_quote_without_fx_cannot_overwrite_the_eur_valuation(monkeypatch):
    holding = _live_currency_holding(monkeypatch, "USD", .95)
    before = holding.price_history.copy()
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: pd.Series(dtype=float))
    assert current_session.apply_to_holdings([holding]) == ()
    assert holding.current_price == 95
    assert holding.current_value == 950
    assert holding.price_is_fallback
    pd.testing.assert_series_equal(holding.price_history, before)
    assert holding.price_history_native.iloc[-1] == 101


def test_live_quote_retains_the_older_fallback_fx_observation(monkeypatch):
    holding = _live_currency_holding(monkeypatch, "USD", .95)
    fx = pd.Series([.95], index=pd.DatetimeIndex(["2025-01-02"]))
    fx.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
        enricher._history_timestamp_key(fx.index[0]): enricher._HISTORY_ORIGIN_CACHE,
    }
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
    assert current_session.apply_to_holdings([holding]) == ("EXAMPLE",)
    assert holding.current_price == pytest.approx(95.95)
    assert holding.price_is_fallback
    assert holding.price_observation_timestamp == pd.Timestamp("2025-01-02").to_pydatetime()


def test_live_bond_quote_applies_the_per_100_convention_exactly_once(monkeypatch):
    holding = _live_currency_holding(monkeypatch, "USD", .95, kind="BOND")
    assert current_session.apply_to_holdings([holding]) == ("EXAMPLE",)
    assert holding.price_history_native.iloc[-1] == 101
    assert holding.price_history.iloc[-1] == pytest.approx(.9595)
    assert holding.price_history_price_only.iloc[-1] == pytest.approx(.9595)
    assert holding.current_price == pytest.approx(.9595)
    assert holding.current_value == pytest.approx(959.5)


def test_an_older_quote_cannot_replace_a_newer_market_close(monkeypatch):
    holding = _live_currency_holding(monkeypatch, "EUR", 1)
    holding.price_history.iloc[-1] = 105
    holding.price_history_native.iloc[-1] = 105
    holding.price_history_price_only.iloc[-1] = 105
    holding.current_price = 105
    holding.current_value = 1050
    before = holding.price_history.copy()
    quote = {
        "price": 101, "prev_close": 100, "currency": "EUR",
        "time": int(dt.datetime(2025, 1, 2, 16, tzinfo=dt.timezone.utc).timestamp()),
    }
    monkeypatch.setattr("tarzan.data.market_quotes.official_quotes", lambda symbols: {"EXAMPLE": quote})
    assert current_session.apply_to_holdings([holding]) == ()
    assert holding.current_price == 105
    assert holding.current_value == 1050
    pd.testing.assert_series_equal(holding.price_history, before)


def test_official_quote_batch_retains_currency_and_rejects_infinities(monkeypatch):
    from tarzan.data import _yf_net

    payload = {"quoteResponse": {"result": [
        {
            "symbol": "BAD", "regularMarketPrice": float("inf"),
            "regularMarketPreviousClose": 100, "regularMarketTime": float("inf"),
            "regularMarketVolume": float("inf"),
        },
        {
            "symbol": "EXAMPLE.L", "regularMarketPrice": 10100,
            "regularMarketPreviousClose": 10000, "currency": "GBp",
        },
    ]}}
    monkeypatch.setattr("yfinance.data.YfData", lambda: SimpleNamespace())
    monkeypatch.setattr(_yf_net, "fetch_yf", lambda *args, **kwargs: payload)
    quotes = production_quote_batch(["BAD", "EXAMPLE.L"])
    assert "price" not in quotes.get("BAD", {})
    assert "time" not in quotes.get("BAD", {})
    assert "volume" not in quotes.get("BAD", {})
    assert quotes["EXAMPLE.L"]["price"] == 10100
    assert quotes["EXAMPLE.L"]["currency"] == "GBp"


def test_fx_conversion_preserves_the_assets_session_timezone(monkeypatch):
    days = pd.date_range("2025-01-06", periods=3, tz="America/New_York")
    fx_days = pd.date_range("2025-01-06", periods=3, tz="UTC")
    history = pd.DataFrame(
        {"Close": [100., 110., 120.], "Price Close": [100., 110., 120.]},
        index=days,
    )
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: pd.Series([.95] * 3, index=fx_days))
    runtime.configure(as_of=dt.date(2025, 1, 8), deterministic=True)
    holding = _holding()
    holding.ticker = "EXAMPLE"
    holding.currency = "USD"
    enricher._set_price_data(holding, history, {}, "USD")
    assert holding.price_history.index.equals(days)
    assert holding.price_history_price_only.index.equals(days)
    orders = [_order(6, OrderType.BUY, 10, -950, 100)]
    result = build_order_derived_series(
        orders, {holding.isin: holding}, today=dt.date(2025, 1, 8),
    )
    assert result.valuations[0][1] == pytest.approx(950)
    assert result.valuations[-1][1] == pytest.approx(1140)
    assert result.history_availability.value == "AVAILABLE"
    assert twr(result.valuations, result.external_flows, result.span_days).cumulative_pct == pytest.approx(20)
