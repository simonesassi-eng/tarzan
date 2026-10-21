"""Declared actions, preserved losses and causal/non-persisted local repair."""

import datetime as dt

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import enricher, price_cache


def frame(prices, splits=None, start="2025-01-01"):
    values = {column: prices for column in ("Open", "High", "Low", "Close", "Price Close")}
    if splits is not None:
        values["Stock Splits"] = splits
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=len(prices)))


@pytest.mark.parametrize("remaining", [0.8, 0.5, 0.2, 0.1, 0.01])
def test_even_a_round_persistent_real_loss_is_not_a_split(remaining):
    raw = frame([100.0] * 20 + [100.0 * remaining] * 20, [0.0] * 40)
    pd.testing.assert_frame_equal(price_cache.repair_split_jumps(raw), raw)


@pytest.mark.parametrize("ratio", [2.0, 3.0, 10.0, 300.0, 0.5, 0.1])
def test_declared_factor_preserves_the_actual_event_day_market_move(ratio):
    raw = frame([100.0, 103.0 / ratio, 104.0 / ratio], [0.0, ratio, 0.0])
    fixed = price_cache.repair_split_jumps(raw)
    assert fixed["Close"].iloc[0] == pytest.approx(100.0 / ratio)
    assert fixed["Close"].iloc[1] / fixed["Close"].iloc[0] - 1 == pytest.approx(0.03)
    pd.testing.assert_frame_equal(price_cache.repair_split_jumps(fixed), fixed)


@pytest.mark.parametrize("ratio", [2.0, 300.0, 0.1])
def test_an_already_adjusted_provider_ruler_is_not_adjusted_again(ratio):
    raw = frame([100.0, 103.0, 104.0], [0.0, ratio, 0.0])
    pd.testing.assert_frame_equal(price_cache.repair_split_jumps(raw), raw)


def test_total_return_and_price_only_rulers_are_repaired_independently():
    raw = frame([50.0, 51.0, 52.0], [0.0, 2.0, 0.0])
    raw["Price Close"] = [100.0, 51.0, 52.0]
    fixed = price_cache.repair_split_jumps(raw)
    assert fixed["Close"].tolist() == [50.0, 51.0, 52.0]
    assert fixed["Price Close"].tolist() == [50.0, 51.0, 52.0]


def test_curated_cl2_action_is_exact_listing_and_date_scoped():
    raw = frame([4017.15, 13.79, 13.80], start="2023-10-09")
    for symbol in (None, "CL2", "CL2.PA", "DIFFERENT.MI"):
        pd.testing.assert_frame_equal(price_cache.repair_split_jumps(raw, symbol=symbol), raw)
    fixed = price_cache.repair_split_jumps(raw, symbol="CL2.MI")
    assert fixed["Close"].iloc[0] == pytest.approx(4017.15 / 300.0)
    different_day = raw.copy()
    different_day.index += pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(
        price_cache.repair_split_jumps(different_day, symbol="CL2.MI"), different_day,
    )


def test_future_events_do_not_change_pre_boundary_prices():
    raw = frame([4017.15, 13.79, 13.80], start="2023-10-09")
    pd.testing.assert_frame_equal(
        price_cache.repair_split_jumps(raw, symbol="CL2.MI", as_of=dt.date(2023, 10, 9)), raw,
    )


def test_no_future_confirmation_bars_are_needed_or_consumed():
    raw = frame([4017.15, 13.79, 0.001, 10000.0], start="2023-10-09")
    fixed_prefix = price_cache.repair_split_jumps(raw.iloc[:2], symbol="CL2.MI")
    fixed_full = price_cache.repair_split_jumps(raw, symbol="CL2.MI")
    pd.testing.assert_frame_equal(fixed_full.iloc[:2], fixed_prefix)


def test_conflicting_action_declarations_do_not_choose_a_factor():
    raw = frame([4017.15, 13.79], [0.0, 2.0], start="2023-10-09")
    pd.testing.assert_frame_equal(price_cache.repair_split_jumps(raw, symbol="CL2.MI"), raw)


def test_provider_cache_keeps_unrepaired_history_for_later_pinned_runs(monkeypatch):
    raw = frame([4017.15, 13.79], start="2023-10-09")
    stored = {}
    enricher.reset_run_caches()
    monkeypatch.setattr(price_cache, "load_history", lambda symbol: None)
    monkeypatch.setattr(price_cache, "store_history", lambda symbol, value: stored.update({symbol: value.copy()}))
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kw: fn())
    monkeypatch.setattr(enricher, "_download_price_history", lambda *a, **kw: raw.copy())
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, value: value)
    fixed = enricher._fetch_history("CL2.MI")
    assert fixed["Close"].iloc[0] == pytest.approx(4017.15 / 300.0)
    assert stored["CL2.MI"]["Close"].iloc[0] == 4017.15
    runtime.configure(deterministic=True, as_of=dt.date(2023, 10, 9))
    enricher.reset_run_caches()
    monkeypatch.setattr(price_cache, "load_history", lambda symbol: stored[symbol])
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kw: None)
    pinned = enricher._fetch_history("CL2.MI")
    assert pinned["Close"].tolist() == [4017.15]
