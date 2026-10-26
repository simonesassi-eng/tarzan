"""Causal daily tapes preserve venue labels, cache age, and FX recovery."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import (
    current_session as cs, enricher, history_time as ht, price_cache, proxy_data,
)
from tarzan.models.holding import Holding
from tarzan.runtime.session import RunContext, RunMode
from tarzan.tests.test_yahoo_core_info import owned


@pytest.fixture
def receipt_clock(monkeypatch):
    now = datetime(2026, 10, 26, 22, 30, tzinfo=timezone.utc)

    class ReceiptDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(cs, "datetime", ReceiptDatetime)
    context = RunContext("daily-causality", RunMode.LIVE, None, now - timedelta(hours=1))
    with runtime.bind_context(context):
        yield now


def _holding(ticker="SYNTH.MI"):
    return Holding(
        isin="SYNTHETIC", ticker=ticker, quantity=2,
        cost_basis_eur=140, market_value_eur=146, currency="EUR",
        security_type="STOCK", instrument_type="Stock",
    )


def _tape(index, values=(100, 5000)):
    frame = pd.DataFrame({"Close": list(values), "Price Close": list(values)}, index=index)
    frame.attrs["original"] = {"nested": ["unchanged"]}
    return frame


def _session_proof(receipt, *, zone="UTC", kind="CURRENCY"):
    return {
        "period_start": (receipt - timedelta(minutes=30)).isoformat(),
        "period_end": (receipt + timedelta(hours=23, minutes=30)).isoformat(),
        "observed_at": (receipt - timedelta(minutes=1)).isoformat(),
        "timezone": zone,
        "instrument_type": kind,
    }


def _overnight(receipt, *, kind="CURRENCY"):
    label = pd.Timestamp(receipt.date()) + pd.Timedelta(days=1)
    frame = _tape(pd.DatetimeIndex([label - pd.Timedelta(days=1), label], tz="UTC"))
    frame.attrs[ht.SESSION_EVIDENCE_ATTR] = {
        ht._key(frame.index[-1]): _session_proof(receipt, kind=kind),
    }
    return frame


@pytest.mark.parametrize("zone", [None, "UTC", "Europe/Rome", "America/New_York"])
def test_future_rows_never_change_regular_or_total_return_price(receipt_clock, zone):
    frame = _tape(pd.DatetimeIndex(["2026-10-23", "2026-10-27"], tz=zone))
    original = deepcopy(frame)
    holding = _holding()
    enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 100
    # _set_price_data selects EUR/unit; _enrich_single owns position mechanics.
    assert holding.quantity * holding.current_price == 200
    assert list(holding.price_history) == [100]
    assert list(holding.price_history_price_only) == [100]
    assert holding.price_observation_timestamp == frame.index[0].to_pydatetime()
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    assert frame.attrs == original.attrs


@pytest.mark.parametrize("aware", [False, True])
def test_asian_session_ahead_of_utc_date_remains_usable(receipt_clock, aware):
    zone = "Asia/Tokyo" if aware else None
    frame = _tape(pd.DatetimeIndex(["2026-10-26", "2026-10-27"], tz=zone), (99, 100))
    result = ht.admit_history(frame, "SYNTH.T")
    pd.testing.assert_frame_equal(result, frame, check_exact=True)
    holding = _holding("SYNTH.T")
    enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 100
    assert holding.price_history.index[-1].date() > receipt_clock.date()


def test_unknown_naive_venue_does_not_invent_an_asian_clock(receipt_clock):
    frame = _tape(pd.to_datetime(["2026-10-26", "2026-10-27"]))
    assert len(ht.admit_history(frame, "SYNTH.UNKNOWN")) == 1


def test_intraday_future_row_is_not_admitted_as_a_daily_label(receipt_clock):
    index = pd.DatetimeIndex([
        receipt_clock - timedelta(seconds=1), receipt_clock + timedelta(seconds=1),
    ])
    assert list(ht.admit_history(_tape(index)).index) == [index[0]]


def test_after_capture_before_receipt_is_still_genuine_history(receipt_clock):
    index = pd.DatetimeIndex([receipt_clock - timedelta(minutes=5)])
    frame = _tape(index, (100,))
    assert index[0] > runtime.context().captured_at
    assert ht.admit_history(frame) is frame


def test_nat_cannot_be_the_selected_last_price(receipt_clock):
    frame = _tape(pd.DatetimeIndex(["2026-10-23", pd.NaT]))
    holding = _holding()
    enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 100
    assert not holding.price_history.index.hasnans


def test_non_datetime_port_keeps_its_explicit_undated_contract(receipt_clock):
    frame = _tape(pd.RangeIndex(2), (99, 100))
    assert ht.admit_history(frame) is frame
    holding = _holding()
    enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 100
    assert holding.price_observation_timestamp is None


@pytest.mark.parametrize("kind", ["CURRENCY", "FUTURE"])
def test_open_overnight_label_keeps_its_actual_trade_observation(receipt_clock, kind):
    frame = _overnight(receipt_clock, kind=kind)
    result = ht.admit_history(frame)
    assert len(result) == 2
    assert result.index[-1] > receipt_clock
    observed = ht.observation_time(result, result.index[-1])
    assert observed == receipt_clock - timedelta(minutes=1)
    assert observed <= receipt_clock
    holding = _holding()
    enricher._set_price_data(holding, frame, {}, "EUR")
    assert holding.current_price == 5000
    assert holding.price_observation_timestamp == observed


@pytest.mark.parametrize("zone", ["Europe/Rome", "America/New_York", "Asia/Tokyo"])
def test_overnight_evidence_uses_venue_midnight_not_utc_midnight(receipt_clock, zone):
    proof = _session_proof(receipt_clock, zone=zone)
    label = pd.Timestamp(proof["period_end"]).tz_convert(zone).normalize()
    assert label > receipt_clock
    frame = _tape(pd.DatetimeIndex([label]), (100,))
    frame.attrs[ht.SESSION_EVIDENCE_ATTR] = {ht._key(label): proof}
    assert ht.admit_history(frame) is frame
    assert ht.session_evidence(frame)
    assert ht.observation_time(frame, label) == receipt_clock - timedelta(minutes=1)


@pytest.mark.parametrize("field,value", [
    ("observed_at", None), ("observed_at", "now"), ("observed_at", True),
    ("observed_at", "2026-10-26T22:30:01+00:00"),
    ("period_start", "2026-10-26T22:31:00+00:00"),
    ("period_end", "2026-10-26T22:29:00+00:00"),
    ("period_start", "2026-10-20T22:00:00+00:00"),
    ("timezone", "Unknown/Invalid"), ("instrument_type", "EQUITY"),
])
def test_malformed_or_not_yet_open_period_is_not_a_future_label_exception(
    receipt_clock, field, value,
):
    frame = _overnight(receipt_clock)
    proof = next(iter(frame.attrs[ht.SESSION_EVIDENCE_ATTR].values()))
    proof[field] = value
    assert len(ht.admit_history(frame)) == 1
    assert ht.session_evidence(frame) == {}


def test_period_evidence_cannot_authorize_other_dates(receipt_clock):
    frame = _overnight(receipt_clock)
    frame.index = frame.index + pd.Timedelta(days=1)
    result = ht.admit_history(frame)
    assert list(result.index) == [pd.Timestamp("2026-10-27", tz="UTC")]
    assert pd.Timestamp("2026-10-28", tz="UTC") not in result.index


def test_future_fx_is_rejected_before_oldest_composite_can_hide_it(receipt_clock, monkeypatch):
    index = pd.DatetimeIndex([
        receipt_clock - timedelta(hours=2), receipt_clock + timedelta(seconds=1),
    ])
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: pd.Series([.9, 50], index=index))
    holding = _holding()
    quote_time = receipt_clock - timedelta(minutes=5)
    enricher._set_price_data(holding, pd.DataFrame(), {
        "regularMarketPrice": 100, "regularMarketTime": quote_time.timestamp(),
    }, "USD")
    assert holding.current_price == 90
    assert holding.price_observation_timestamp == index[0].to_pydatetime()


def test_no_past_fx_cannot_convert_a_quote_using_future_fx(receipt_clock, monkeypatch):
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: pd.Series(
        [50], index=pd.DatetimeIndex([receipt_clock + timedelta(seconds=1)]),
    ))
    holding = _holding()
    enricher._set_price_data(holding, pd.DataFrame(), {
        "regularMarketPrice": 100,
        "regularMarketTime": (receipt_clock - timedelta(minutes=5)).timestamp(),
    }, "USD")
    assert holding.current_price is None


def test_future_fx_cannot_backfill_or_multiply_historical_prices(receipt_clock, monkeypatch):
    fx = pd.Series([.9, 50], index=pd.to_datetime(["2026-10-23", "2026-10-27"]))
    monkeypatch.setattr(enricher, "_get_fx_series", lambda currency: fx)
    prices = pd.Series([100, 101], index=pd.to_datetime(["2026-10-22", "2026-10-26"]))
    original = deepcopy(fx)
    result = enricher.convert_to_eur(prices, "USD")
    assert list(result.index) == [prices.index[-1]]
    assert list(result) == [90.9]
    assert next(iter(result.attrs[enricher._FX_EVIDENCE_ATTR].values()))[
        "observation_time"
    ] == fx.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
    pd.testing.assert_series_equal(fx, original, check_exact=True)


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "private-history"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)


@pytest.mark.parametrize("series", [False, True])
def test_overnight_cache_roundtrip_retains_actual_old_observation(
    receipt_clock, isolated_cache, series,
):
    frame = _overnight(receipt_clock)
    value = frame["Close"] if series else frame
    before = deepcopy(value)
    price_cache.store_history("SYNTH", value)
    loaded = price_cache.load_history("SYNTH")
    assert len(ht.admit_history(loaded)) == 2
    assert ht.observation_time(loaded, loaded.index[-1]) == receipt_clock - timedelta(minutes=1)
    assert loaded.attrs[ht.SESSION_EVIDENCE_ATTR] == value.attrs[ht.SESSION_EVIDENCE_ATTR]
    if series:
        pd.testing.assert_series_equal(value, before, check_exact=True)
    else:
        pd.testing.assert_frame_equal(value, before, check_exact=True)


def test_fresh_overlap_replaces_old_session_evidence_as_well_as_prices(receipt_clock):
    cached = _overnight(receipt_clock)
    fresh = cached.iloc[-1:].copy()
    fresh.attrs.clear()
    result = ht.merge_session_evidence(cached, fresh, price_cache.merge_history(cached, fresh))
    assert ht.session_evidence(result) == {}
    assert len(ht.admit_history(result)) == 1


def test_pinned_boundary_does_not_adopt_later_overnight_trade(receipt_clock):
    frame = _overnight(receipt_clock)
    context = RunContext(
        "daily-pinned", RunMode.POINT_IN_TIME, receipt_clock.date(), receipt_clock,
    )
    with runtime.bind_context(context):
        result = ht.admit_history(frame)
    assert len(result) == 1
    assert result.index[-1].date() == receipt_clock.date()


@pytest.mark.parametrize("mode", [RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE])
def test_pinned_clipping_survives_ambiguous_venue_midnight(receipt_clock, mode):
    context = RunContext(
        "ambiguous-midnight", mode, pd.Timestamp("2026-11-01").date(), receipt_clock,
    )
    index = pd.date_range("2026-10-31", periods=3, tz="America/Havana", ambiguous=True)
    frame = _tape(index, (99, 100, 5000))
    with runtime.bind_context(context):
        result = enricher._clip_to_as_of(frame)
    assert list(result["Close"]) == [99, 100]
    assert result.index[-1].date() == context.effective_date


def test_future_split_row_cannot_repair_genuine_past_prices(receipt_clock, monkeypatch):
    frame = _tape(pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-27"]), (100, 102, 51))
    frame["Stock Splits"] = [0, 0, 2]
    monkeypatch.setattr(enricher, "_download_price_history", lambda *a, **k: frame)
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kwargs: fn())
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, result: result)
    result = enricher._fetch_history_uncached("SYNTH.MI")
    assert list(result["Close"]) == [100, 102]
    assert list(result["Price Close"]) == [100, 102]
    assert frame["Stock Splits"].iloc[-1] == 2


def test_future_cached_tail_cannot_select_a_future_refresh_window(receipt_clock, monkeypatch):
    cached = _tape(pd.to_datetime(["2026-10-23", "2026-12-01"]))
    starts = []
    stored = []
    monkeypatch.setattr(price_cache, "load_history", lambda symbol: cached)
    monkeypatch.setattr(price_cache, "covers_period", lambda frame, period: True)
    monkeypatch.setattr(price_cache, "store_history", lambda symbol, result: stored.append(result))
    monkeypatch.setattr(enricher, "_download_price_history", lambda symbol, start=None: (
        starts.append(start) or pd.DataFrame()
    ))
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kwargs: fn())
    monkeypatch.setattr(enricher, "_fill_today_from_sibling", lambda symbol, frame: frame)
    result = enricher._fetch_history_uncached("SYNTH.MI")
    assert starts == [datetime(2026, 10, 18)]
    assert result.index[-1] == pd.Timestamp("2026-10-23")
    assert result.attrs[enricher._HISTORY_ORIGINS_ATTR][
        enricher._history_timestamp_key(result.index[-1])
    ] == enricher._HISTORY_ORIGIN_CACHE
    assert len(stored) == 1 and len(stored[0]) == 1
    assert len(cached) == 2


def test_future_only_history_does_not_prove_a_listing_or_return(receipt_clock):
    frame = _tape(pd.to_datetime(["2026-10-27", "2026-10-28"]))
    assert not enricher._history_has_market_evidence(frame)
    assert not enricher._history_supports_a_return(frame)
    assert not enricher._ticker_data_has_market_evidence({"history": frame})


def test_curated_candidate_price_uses_the_same_causal_rows_as_return_evidence(
    receipt_clock, monkeypatch,
):
    frame = _tape(pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-27"]), (99, 100, 5000))
    monkeypatch.setattr(enricher, "_fetch_ticker_info", lambda symbol: {})
    monkeypatch.setattr(enricher, "_fetch_history", lambda symbol: frame)
    monkeypatch.setattr(enricher.cfg, "instrument_taxonomy_has", lambda symbol: True)
    candidate = enricher._fetch_candidate_meta("SYNTH.MI")
    assert candidate.price == 100


def test_proxy_cached_future_tail_cannot_skip_a_real_refresh(receipt_clock, monkeypatch):
    cached = pd.Series([100, 5000], index=pd.to_datetime(["2026-10-01", "2026-12-01"]))
    fresh = _tape(pd.to_datetime(["2026-10-23", "2026-10-27"]))
    calls, stored = [], []
    monkeypatch.setattr(price_cache, "load_history", lambda key: cached)
    monkeypatch.setattr(price_cache, "store_history", lambda key, frame: stored.append(frame))

    class Ticker:
        def history(self, **arguments):
            calls.append(arguments)
            return fresh

    monkeypatch.setattr(enricher._yf_net, "ticker", lambda symbol: Ticker())
    monkeypatch.setattr(enricher._yf_net, "fetch_yf", lambda fn, **kwargs: fn())
    result = proxy_data._fetch_max("SYNTH.MI")
    assert calls == [{"period": "max", "auto_adjust": True}]
    assert list(result) == [100]
    assert list(stored[0]) == [100]
    assert cached.iloc[-1] == 5000


def test_rejected_future_raw_answer_does_not_hide_same_run_recovery(receipt_clock, monkeypatch):
    state = {"frame": _tape(pd.to_datetime(["2026-10-27"]), (5000,))}
    calls = []
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)
    monkeypatch.setattr(enricher._yf_net, "ticker", lambda symbol: SimpleNamespace(
        history=lambda **arguments: calls.append(arguments) or state["frame"],
    ))
    with owned():
        first = enricher._download_price_history("SYNTH.MI")
        assert first.empty
        state["frame"] = _tape(pd.to_datetime(["2026-10-23"]), (100,))
        second = enricher._download_price_history("SYNTH.MI")
        third = enricher._download_price_history("SYNTH.MI")
    assert list(second["Price Close"]) == [100]
    assert list(third["Close"]) == [100]
    assert len(calls) == 2


@pytest.mark.parametrize("size", [64, 640, 6400])
def test_overnight_observation_lookup_does_not_rebuild_the_entire_tape(
    receipt_clock, monkeypatch, size,
):
    label = pd.Timestamp("2026-10-27", tz="UTC")
    frame = pd.DataFrame(
        {"Close": np.ones(size)},
        index=pd.date_range(end=label, periods=size, tz="UTC"),
    )
    frame.attrs[ht.SESSION_EVIDENCE_ATTR] = {
        ht._key(label): _session_proof(receipt_clock),
    }
    calls = []
    original = ht._session_observation
    monkeypatch.setattr(ht, "_session_observation", lambda *args: (
        calls.append(1) or original(*args)
    ))
    for day in frame.index:
        assert ht.observation_time(frame, day) is not None
    assert calls == [1]


@pytest.mark.parametrize("count", [16, 160, 1600])
def test_vector_admission_samples_one_receipt_clock_per_tape(receipt_clock, monkeypatch, count):
    calls = []
    monkeypatch.setattr(cs, "observation_time_limit", lambda: (
        calls.append(1) or receipt_clock
    ))
    index = pd.date_range(end="2026-10-27", periods=count, tz="UTC")
    frame = pd.DataFrame({"Close": np.ones(count)}, index=index)
    result = ht.admit_history(frame)
    assert len(result) == count - 1
    assert calls == [1]
