"""Receipt-time causality is shared without dropping alternate venue discovery."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import current_session as cs, enricher
from tarzan.models.holding import Holding
from tarzan.runtime.session import RunContext, RunMode


@pytest.fixture
def receipt_clock(monkeypatch):
    now = datetime(2026, 10, 26, 10, tzinfo=timezone.utc)

    class ReceiptDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(cs, "datetime", ReceiptDatetime)
    context = RunContext(
        "synthetic-receipt", RunMode.LIVE, None, now - timedelta(hours=1),
    )
    with runtime.bind_context(context):
        yield now


def _holding():
    return Holding(
        isin="SYNTHETIC", ticker="SYNTH.MI", quantity=2.0,
        cost_basis_eur=140.0, market_value_eur=146.0, currency="EUR",
        instrument_type="Stock", security_type="STOCK",
    )


def _info(observed, price=101.0):
    return {
        "regularMarketPrice": price,
        "regularMarketTime": observed.timestamp(),
        "currency": "EUR",
    }


@pytest.mark.parametrize("offset, future", [
    (timedelta(microseconds=-1), False),
    (timedelta(0), False),
    (timedelta(microseconds=1), True),
])
def test_receipt_upper_bound_is_an_instant_not_a_calendar_day(receipt_clock, offset, future):
    assert cs.quote_observation_is_future(receipt_clock + offset) is future


@pytest.mark.parametrize("zone", [
    timezone.utc, timezone(timedelta(hours=14)), timezone(timedelta(hours=-12)),
])
def test_timezone_representation_does_not_change_future_admission(receipt_clock, zone):
    for offset, expected in ((-1, False), (0, False), (1, True)):
        observed = (receipt_clock + timedelta(seconds=offset)).astimezone(zone)
        assert cs.quote_observation_is_future(observed) is expected


def test_naive_parsed_quote_timestamp_keeps_its_established_utc_ruler(receipt_clock):
    assert not cs.quote_observation_is_future(receipt_clock.replace(tzinfo=None))
    assert cs.quote_observation_is_future(
        (receipt_clock + timedelta(seconds=1)).replace(tzinfo=None),
    )


def test_trade_after_capture_but_before_receipt_remains_primary(receipt_clock):
    holding = _holding()
    observed = receipt_clock - timedelta(minutes=5)
    assert observed > runtime.context().captured_at
    enricher._set_price_data(holding, pd.DataFrame(), _info(observed), "EUR")
    assert holding.current_price == 101.0
    assert holding.price_observation_timestamp == observed
    assert holding.price_is_fallback is False
    assert enricher._ticker_data_has_market_evidence({
        "history": pd.DataFrame(), "info": _info(observed),
    })


def test_future_regular_quote_neither_prices_nor_verifies_a_listing(receipt_clock):
    info = _info(receipt_clock + timedelta(seconds=1), price=5000.0)
    holding = _holding()
    before = dict(info)
    enricher._set_price_data(holding, pd.DataFrame(), info, "EUR")
    assert holding.current_price is None
    assert holding.price_observation_timestamp is None
    assert not enricher._ticker_data_has_market_evidence({
        "history": pd.DataFrame(), "info": info,
    })
    assert info == before


def test_rejecting_future_price_preserves_real_history_and_its_provenance(receipt_clock):
    history = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.date_range("2026-10-22", periods=2, tz="UTC"),
    )
    history.attrs[enricher._HISTORY_ORIGINS_ATTR] = {
        enricher._history_timestamp_key(history.index[-1]): enricher._HISTORY_ORIGIN_CACHE,
    }
    history.attrs[enricher._HISTORY_SYMBOL_ATTR] = "SYNTH.MI"
    original = history.copy(deep=True)
    holding = _holding()
    enricher._set_price_data(
        holding, history, _info(receipt_clock + timedelta(days=1)), "EUR",
    )
    assert holding.current_price == 100.0
    assert holding.price_observation_timestamp == history.index[-1].to_pydatetime()
    assert holding.price_is_fallback is True
    assert holding.data_source == "price_cache:SYNTH.MI"
    pd.testing.assert_frame_equal(history, original, check_exact=True)
    assert history.attrs == original.attrs


def test_independent_previous_close_remains_explicit_fallback(receipt_clock):
    info = dict(_info(receipt_clock + timedelta(days=1)), previousClose=99.0)
    holding = _holding()
    enricher._set_price_data(holding, pd.DataFrame(), info, "EUR")
    assert holding.current_price == 99.0
    assert holding.price_observation_timestamp is None
    assert holding.price_is_fallback is True
    assert holding.data_source == "yfinance:previousClose"
    assert enricher._ticker_data_has_market_evidence({
        "history": pd.DataFrame(), "info": info,
    })


def test_missing_clock_remains_undated_not_primary(receipt_clock):
    holding = _holding()
    enricher._set_price_data(
        holding, pd.DataFrame(), {"regularMarketPrice": 101.0}, "EUR",
    )
    assert holding.current_price == 101.0
    assert holding.price_observation_timestamp is None
    assert holding.price_is_fallback is True


@pytest.mark.parametrize("raw_time", [
    10**20, float("inf"), float("nan"), True, np.bool_(True),
    1 + 2j, "not a clock", "now", "today", pd.NaT,
])
def test_malformed_present_clock_never_gains_unknown_age_authority(receipt_clock, raw_time):
    info = {"regularMarketPrice": 5000.0, "regularMarketTime": raw_time}
    holding = _holding()
    enricher._set_price_data(holding, pd.DataFrame(), info, "EUR")
    assert holding.current_price is None
    assert not enricher._ticker_data_has_market_evidence({"info": info})
    quote = {"price": 101.0, "time": raw_time}
    assert cs.quote_observed_at(quote) is None
    assert cs.quote_is_stale(quote)
    assert cs.pick_quote(["SYNTH.MI"], {"SYNTH.MI": quote}, 100.0) == {}
    assert cs.stamp_date(quote, runtime.today(), "SYNTH.MI") is None


@pytest.mark.parametrize("representation", ["datetime", "iso", "numeric-string", "numpy"])
def test_valid_info_timestamp_representation_has_one_price_and_ranking_clock(
    receipt_clock, representation,
):
    observed = receipt_clock - timedelta(minutes=5)
    raw = {
        "datetime": observed, "iso": observed.isoformat(),
        "numeric-string": str(observed.timestamp()),
        "numpy": np.int64(observed.timestamp()),
    }[representation]
    holding = _holding()
    info = {"regularMarketPrice": 101.0, "regularMarketTime": raw}
    enricher._set_price_data(holding, pd.DataFrame(), info, "EUR")
    assert holding.current_price == 101.0
    assert holding.price_observation_timestamp == observed
    assert not holding.price_is_fallback
    candidate = enricher._Candidate("SYNTH.MI", info, 101.0, "EUR", "Synthetic")
    assert enricher._quote_is_fresh(candidate) == 1
    info["regularMarketTime"] = (
        receipt_clock + timedelta(seconds=1)
    ).isoformat()
    assert enricher._quote_is_fresh(candidate) == 0


def test_future_native_price_cannot_be_hidden_by_an_older_fx_observation(receipt_clock, monkeypatch):
    fx_calls = []

    def fx(currency):
        fx_calls.append(currency)
        return pd.Series([0.9], index=pd.DatetimeIndex([receipt_clock - timedelta(hours=2)]))

    monkeypatch.setattr(enricher, "_get_fx_series", fx)
    holding = _holding()
    enricher._set_price_data(
        holding, pd.DataFrame(), _info(receipt_clock + timedelta(seconds=1)), "USD",
    )
    assert holding.current_price is None
    assert fx_calls == []


def test_valid_foreign_quote_still_uses_the_same_conversion_and_oldest_evidence(
    receipt_clock, monkeypatch,
):
    fx_time = receipt_clock - timedelta(hours=2)
    monkeypatch.setattr(
        enricher, "_get_fx_series",
        lambda currency: pd.Series([0.9], index=pd.DatetimeIndex([fx_time])),
    )
    holding = _holding()
    enricher._set_price_data(
        holding, pd.DataFrame(), _info(receipt_clock - timedelta(minutes=5), 100.0), "USD",
    )
    assert holding.current_price == 90.0
    assert holding.price_observation_timestamp == fx_time
    assert holding.price_is_fallback is False


def test_future_preferred_venue_does_not_displace_a_valid_sibling(receipt_clock):
    quotes = {
        "SYNTH.MI": {"price": 101.2, "time": int((receipt_clock + timedelta(seconds=1)).timestamp())},
        "SYNTH.DE": {"price": 101.0, "time": int((receipt_clock - timedelta(minutes=1)).timestamp())},
    }
    assert cs.pick_quote(
        ["SYNTH.MI", "SYNTH.DE"], quotes, reference_price=100.0,
    ) == quotes["SYNTH.DE"]
    assert cs.quote_is_stale(quotes["SYNTH.MI"])
    assert not cs.quote_is_stale(quotes["SYNTH.DE"])


def test_future_quote_cannot_write_an_actual_future_trading_session(receipt_clock):
    quote = {"price": 101.0, "time": int((receipt_clock + timedelta(days=1)).timestamp())}
    assert cs.stamp_date(quote, runtime.today(), "SYNTH.MI") is None


def test_valid_instant_with_next_asian_calendar_date_is_not_future(receipt_clock):
    observed = receipt_clock.astimezone(
        timezone(timedelta(hours=14)),
    )
    assert observed.date() > receipt_clock.date()
    quote = {"price": 101.0, "time": int(observed.timestamp())}
    assert not cs.quote_is_stale(quote)
    assert cs.quote_observed_at(quote) == observed.astimezone(timezone.utc)


def test_metadata_ranking_and_resolution_do_not_promote_future_currency_evidence(
    receipt_clock, monkeypatch,
):
    info = dict(
        _info(receipt_clock + timedelta(days=1)),
        shortName="Synthetic", quoteType="EQUITY",
    )
    monkeypatch.setattr(enricher, "_fetch_ticker_info", lambda symbol: info)
    monkeypatch.setattr(enricher.cfg, "instrument_taxonomy_has", lambda symbol: False)
    assert enricher._fetch_candidate_meta("SYNTH.MI") is None
    candidate = enricher._Candidate(
        symbol="SYNTH.MI", info=info, price=101.0, currency="EUR", name="Synthetic",
    )
    assert enricher._quote_is_fresh(candidate) == 0
    candidate.info = _info(receipt_clock - timedelta(minutes=1))
    assert enricher._quote_is_fresh(candidate) == 1


@pytest.mark.parametrize("mode", [RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE])
def test_pinned_upper_bound_does_not_use_a_later_live_receipt_clock(receipt_clock, mode):
    context = RunContext(
        "synthetic-pinned", mode, date(2026, 10, 22), receipt_clock,
    )
    with runtime.bind_context(context):
        assert not cs.quote_observation_is_future(
            datetime(2026, 10, 22, 23, 59, 59, 999999, tzinfo=timezone.utc),
        )
        assert cs.quote_observation_is_future(
            datetime(2026, 10, 23, tzinfo=timezone.utc),
        )
        holding = _holding()
        history = pd.DataFrame(
            {"Close": [100.0, 5000.0]},
            index=pd.date_range("2026-10-22", periods=2, tz="UTC"),
        )
        enricher._set_price_data(
            holding, history, _info(receipt_clock - timedelta(minutes=1)), "EUR",
        )
        assert holding.current_price == 100.0
        assert holding.price_history.tolist() == [100.0]
