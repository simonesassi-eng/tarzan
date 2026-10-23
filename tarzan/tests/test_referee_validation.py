"""Malformed or incomparable source observations cannot cast a price vote.

All observations are synthetic; these tests never contact a market provider.
"""

from __future__ import annotations

import datetime as dt
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.data import current_session, justetf
from tarzan.data.referee import finite_number
from tarzan.engine import cross_check as xc


TODAY = dt.date(2026, 10, 22)
PREVIOUS = dt.date(2026, 10, 21)
INVALID_PRICES = (None, True, np.bool_(False), 0, -1, np.nan, np.inf, -np.inf, "bad")


def _quote(**changes):
    quote = dict(
        price=101.0, prev_close=100.0, date=TODAY, prev_date=PREVIOUS,
        venue="XETRA", currency="EUR",
    )
    quote.update(changes)
    return quote


def _quote_body():
    return {
        "latestQuote": {"raw": "101"},
        "previousQuote": {"raw": "100"},
        "latestQuoteDate": TODAY.isoformat(),
        "previousQuoteDate": PREVIOUS.isoformat(),
        "quoteTradingVenue": "XETRA",
    }


def _tape():
    return pd.Series(
        [99.0, 100.0, 101.0],
        index=pd.DatetimeIndex(["2026-10-20", "2026-10-21", "2026-10-22"]),
        name="SYNTH.MI",
    )


@pytest.fixture(autouse=True)
def referee_clock(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.today", lambda: TODAY)


@pytest.mark.parametrize("field", ("latestQuote", "previousQuote"))
@pytest.mark.parametrize("value", INVALID_PRICES)
def test_quote_decoder_rejects_non_positive_non_finite_and_boolean_prices(
        monkeypatch, field, value):
    body = _quote_body()
    body[field]["raw"] = value
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


@pytest.mark.parametrize("field", ("latestQuoteDate", "previousQuoteDate"))
@pytest.mark.parametrize("value", (
    None, True, 20261022, "", "NaT", "not-a-date", "today", "now",
    "2026-10-22 now", "2026-10-22Ttoday", "2026-10-22 wrong-suffix",
))
def test_quote_decoder_requires_real_dates(monkeypatch, field, value):
    body = _quote_body()
    body[field] = value
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


@pytest.mark.parametrize("previous", ("2026-10-22", "2026-10-23"))
def test_quote_decoder_does_not_accept_a_rolled_or_reversed_pair(monkeypatch, previous):
    body = _quote_body()
    body["previousQuoteDate"] = previous
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


def test_quote_decoder_does_not_accept_a_future_observation(monkeypatch):
    body = _quote_body()
    body["latestQuoteDate"] = "2026-10-23"
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


def test_quote_decoder_preserves_the_source_pair_and_declares_its_requested_currency(
        monkeypatch):
    monkeypatch.setattr(justetf, "_get_json", lambda _url: _quote_body())
    assert justetf._fetch_quote("XX0000000001") == _quote()


@pytest.mark.parametrize("value", (complex(101, 5), np.complex64(101 + 5j), np.complex128(101 + 5j)))
def test_scalar_complex_never_loses_its_imaginary_component_to_become_a_price(value):
    with warnings.catch_warnings(record=True) as caught:
        assert finite_number(value, positive=True) is None
    assert not caught


@pytest.mark.parametrize("field", ("latestQuote", "previousQuote"))
@pytest.mark.parametrize("value", (np.complex64(101 + 5j), np.complex128(101 + 5j)))
def test_quote_decoder_and_baseline_reject_numpy_complex_prices(monkeypatch, field, value):
    body = _quote_body()
    body[field]["raw"] = value
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    q = _quote()
    q["price" if field == "latestQuote" else "prev_close"] = value
    with warnings.catch_warnings(record=True) as caught:
        assert justetf._fetch_quote("XX0000000001") is None
        assert current_session._independent_baseline(q, TODAY, PREVIOUS) is None
    assert not caught


@pytest.mark.parametrize("declaration", (
    {"currency": "USD"}, {"currency": "GBp"}, {"isin": "XX0000000002"},
))
def test_quote_decoder_rejects_an_explicit_response_identity_or_currency_conflict(
        monkeypatch, declaration):
    body = dict(_quote_body(), **declaration)
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


@pytest.mark.parametrize("field", ("currency", "isin"))
@pytest.mark.parametrize("value", (True, False, 123, [], {}, ["EUR"]))
def test_quote_decoder_rejects_malformed_optional_contract_fields(monkeypatch, field, value):
    body = _quote_body()
    body[field] = value
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") is None


@pytest.mark.parametrize("declaration", (
    {}, {"currency": "EUR", "isin": "XX0000000001"},
    {"currency": "eur", "isin": "xx0000000001"},
    {"currency": None, "isin": None},
    {"fundCurrency": "USD"}, {"baseCurrency": "USD"}, {"tradingCurrency": "GBp"},
))
def test_quote_currency_contract_does_not_confuse_fund_or_trading_currency_with_requested_eur(
        monkeypatch, declaration):
    body = dict(_quote_body(), **declaration)
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_quote("XX0000000001") == _quote()


@pytest.mark.parametrize("body", ([], ["series"], "html", 17, {"series": "bad"}))
def test_series_decoder_does_not_interpret_an_unexpected_body_as_prices(monkeypatch, body):
    monkeypatch.setattr(justetf, "_get_json", lambda _url: body)
    assert justetf._fetch_series("XX0000000001") is None


def test_series_decoder_keeps_only_valid_calendar_observations(monkeypatch):
    points = [
        {"date": "2026-10-20", "value": {"raw": "99"}},
        {"date": "2026-10-21", "value": {"raw": "100"}},
        {"date": "2026-10-21", "value": {"raw": float("inf")}},
        {"date": "2026-10-22T17:00:00+02:00", "value": {"raw": "101"}},
        {"date": "2026-10-23", "value": {"raw": "102"}},
        {"date": "NaT", "value": {"raw": "103"}},
        {"date": 20261022, "value": {"raw": "104"}},
        {"date": "2026-10-19", "value": {"raw": True}},
        {"date": "2026-10-18", "value": {"raw": -1}},
        "bad-point",
    ]
    monkeypatch.setattr(justetf, "_get_json", lambda _url: {"series": points})
    pd.testing.assert_series_equal(justetf._fetch_series("XX0000000001"), _tape().rename(None))


@pytest.mark.parametrize("body", ([], "not-an-object", None, 12))
def test_json_object_contract_rejects_other_valid_json_roots(monkeypatch, body):
    closed = []
    response = SimpleNamespace(
        ok=True, headers={"content-type": "application/json"}, json=lambda: body,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr(
        "tarzan.runtime.transport.provider_call", lambda _name, operation, **kwargs: operation(1)
    )
    assert justetf._get_json("https://provider.example.invalid") is None
    assert closed == [True]


@pytest.mark.parametrize("field", ("price", "prev_close"))
@pytest.mark.parametrize("value", INVALID_PRICES)
def test_one_day_and_baseline_reject_malformed_price_pairs(field, value):
    q = _quote()
    q[field] = value
    assert xc.check_one_day(1.0, q, TODAY, _tape(), ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE
    assert current_session._independent_baseline(q, TODAY, PREVIOUS) is None


@pytest.mark.parametrize("ours", (None, True, np.bool_(True), np.nan, np.inf, -np.inf, "bad"))
def test_one_day_cannot_corroborate_an_invalid_engine_return(ours):
    assert xc.check_one_day(ours, _quote(), TODAY, _tape(), ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE


@pytest.mark.parametrize("previous", (None, TODAY, dt.date(2026, 10, 20), dt.date(2026, 10, 23)))
def test_one_day_and_baseline_require_the_same_previous_observation(previous):
    q = _quote()
    q["prev_date"] = previous
    assert xc.check_one_day(1.0, q, TODAY, _tape(), ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE
    assert current_session._independent_baseline(q, TODAY, PREVIOUS) is None


def test_one_day_accepts_exactly_comparable_tape_and_source_dates():
    v = xc.check_one_day(1.0, _quote(), TODAY, _tape(), ticker="SYNTH.MI")
    assert v["status"] == xc.OK
    assert v["alt"] == pytest.approx(1.0)


def test_a_missing_native_one_day_anchor_is_not_a_multi_session_vote():
    tape = _tape().drop(pd.Timestamp(PREVIOUS))
    assert xc.check_one_day(1.0, _quote(), TODAY, tape, ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE


def test_referee_uses_the_actual_tape_anchor_not_just_its_own_exchange_calendar():
    # Milan traded on 6 January, Helsinki did not. Both see 7 January, but their
    # published "previous closes" span different intervals.
    day = dt.date(2026, 1, 7)
    q = _quote()
    q.update(date=day, prev_date=dt.date(2026, 1, 5), venue="HELSINKI")
    milan = pd.Series([100, 100.5, 101], index=pd.date_range("2026-01-05", periods=3))
    helsinki = milan.drop(pd.Timestamp("2026-01-06"))
    assert xc.check_one_day(1.0, q, day, milan, ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE
    assert xc.check_one_day(1.0, q, day, helsinki, ticker="SYNTH.HE")["status"] == xc.OK


def test_a_native_baseline_observed_on_another_date_cannot_referee_a_stamped_row():
    tape = _tape()
    tape.attrs["current_session_baseline"] = {
        "at": TODAY.isoformat(), "anchor": PREVIOUS.isoformat(),
        "observation_date": "2026-10-20", "symbol": "SYNTH.DE",
    }
    assert xc.check_one_day(1.0, _quote(), TODAY, tape, ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE


def test_stamping_records_native_sibling_dates_without_changing_the_existing_prices():
    day = dt.date(2026, 1, 7)
    tape = pd.Series([99.0, 100.0], index=pd.date_range("2026-01-05", periods=2))
    tape.attrs["resolved_ticker"] = "SYNTH.MI"
    observed = dt.datetime(2026, 1, 7, 12, tzinfo=dt.timezone.utc)
    quote = {
        "symbol": "SYNTH.HE", "price": 101.0, "prev_close": 100.0,
        "time": int(observed.timestamp()),
    }
    result = current_session.stamp_today(tape, pd.Timestamp(day), 101, quote, ticker="SYNTH.MI")
    assert result.to_dict() == {
        pd.Timestamp("2026-01-05"): 99.0,
        pd.Timestamp("2026-01-06"): 100.0,
        pd.Timestamp("2026-01-07"): 101.0,
    }
    assert result.attrs["resolved_ticker"] == "SYNTH.MI"
    assert result.attrs["current_session_baseline"] == {
        "at": "2026-01-07", "anchor": "2026-01-06",
        "observation_date": "2026-01-05", "symbol": "SYNTH.HE",
    }


def test_a_referee_on_a_different_native_interval_does_not_replace_a_previous_close(monkeypatch):
    day = dt.date(2026, 1, 7)
    monkeypatch.setattr("tarzan.runtime.today", lambda: day)
    observed = dt.datetime(2026, 1, 7, 12, tzinfo=dt.timezone.utc)
    quote = {
        "symbol": "SYNTH.HE", "price": 101.0, "prev_close": 103.0,
        "time": int(observed.timestamp()),
    }
    referee = _quote(date=day, prev_date=dt.date(2026, 1, 6))
    # The quote belongs to Helsinki, whose previous session was 5 January, not
    # Milan's 6 January. Even a plausible 100 must not overwrite the native 103.
    selected = current_session.pick_quote(
        ["SYNTH.HE"], {"SYNTH.HE": quote}, 101.0, referee=referee,
    )
    assert selected["prev_close"] == 103.0


def test_a_non_eur_pair_cannot_judge_a_eur_return():
    q = _quote()
    q["currency"] = "USD"
    assert xc.check_one_day(1.0, q, TODAY, _tape(), ticker="SYNTH.MI")["status"] == xc.UNAVAILABLE
    assert current_session._independent_baseline(q, TODAY, PREVIOUS) is None


@pytest.mark.parametrize("value", (True, 0.0, -1.0, np.inf, -np.inf, np.nan))
def test_non_prices_cannot_be_used_as_shared_close_endpoints(value):
    tape = _tape()
    ref = tape.copy()
    ref.iloc[-1] = value
    verdicts = xc.check_windows(tape, ref)
    assert all(
        verdict["alt"] is None or np.isfinite(verdict["alt"])
        for verdict in verdicts.values()
    )
    assert all(
        verdict["at"] is None or verdict["at"] != TODAY
        for verdict in verdicts.values()
    )


@pytest.mark.parametrize("allowance", (True, -1.0, np.inf, np.nan, "bad"))
def test_a_bad_tolerance_cannot_turn_a_dispute_into_an_ok(allowance):
    assert all(
        verdict["status"] == xc.UNAVAILABLE
        for verdict in xc.check_windows(_tape(), _tape(), extra_pp=allowance).values()
    )


@pytest.mark.parametrize("ours,alt", ((np.inf, 1), (1, np.inf), (np.nan, 1), (True, 1)))
def test_invalid_returns_cannot_be_confirmed_or_corrected(ours, alt):
    verdict = {
        "status": xc.DIVERGED, "ours": ours, "alt": alt,
        "anchor": pd.Timestamp(PREVIOUS), "at": TODAY,
    }
    assert xc.settle(verdict, _tape(), [_tape()])["status"] == xc.DIVERGED


def test_an_overflowing_ratio_does_not_cast_a_third_source_vote():
    third = pd.Series([1e-300, 1e300], index=pd.DatetimeIndex([PREVIOUS, TODAY]))
    verdict = {
        "status": xc.DIVERGED, "ours": 10.0, "alt": 1.0,
        "anchor": pd.Timestamp(PREVIOUS), "at": TODAY,
    }
    result = xc.settle(verdict, _tape(), [third])
    assert result["status"] == xc.DIVERGED
    assert result["third"] == []


def test_object_complex_prices_are_not_silently_cast_to_their_real_component():
    series = pd.Series(
        [99.0, complex(100, 5), np.complex64(100 + 3j), "101"],
        index=pd.date_range("2026-10-19", periods=4), dtype=object,
    )
    expected = pd.Series([99.0, 101.0], index=pd.DatetimeIndex(["2026-10-19", "2026-10-22"]))
    pd.testing.assert_series_equal(xc._naive_days(series), expected)


def test_mixed_boolean_and_numeric_strings_keep_only_genuine_prices():
    series = pd.Series(
        ["99", True, np.bool_(True), "100", "bad", "101"],
        index=pd.date_range("2026-10-17", periods=6), dtype=object,
    )
    expected = pd.Series(
        [99.0, 100.0, 101.0],
        index=pd.DatetimeIndex(["2026-10-17", "2026-10-20", "2026-10-22"]),
    )
    pd.testing.assert_series_equal(xc._naive_days(series), expected)


def test_unparsed_relative_index_dates_never_turn_receipt_time_into_source_dates():
    series = pd.Series(
        [99.0, 100.0, 101.0],
        index=pd.Index(["today", "now", "2026-10-22"]),
    )
    expected = pd.Series([101.0], index=pd.DatetimeIndex(["2026-10-22"]))
    pd.testing.assert_series_equal(xc._naive_days(series), expected)


def test_mixed_explicit_index_dates_preserve_native_local_dates():
    series = pd.Series(
        [100.0, 101.0],
        index=pd.Index([pd.Timestamp("2026-10-21T23:00:00-05:00"), "2026-10-22"]),
    )
    expected = pd.Series([100.0, 101.0], index=pd.DatetimeIndex(["2026-10-21", "2026-10-22"]))
    pd.testing.assert_series_equal(xc._naive_days(series), expected)


def test_ordinary_finite_votes_preserve_the_original_arithmetic_exactly():
    third_a = pd.Series([100.0, 103.2], index=pd.DatetimeIndex([PREVIOUS, TODAY]))
    third_b = pd.Series([100.0, 103.3], index=pd.DatetimeIndex([PREVIOUS, TODAY]))
    verdict = {
        "status": xc.DIVERGED, "ours": 10.0, "alt": 3.1,
        "anchor": pd.Timestamp(PREVIOUS), "at": TODAY,
    }
    result = xc.settle(verdict, _tape(), [third_a, third_b])
    assert result["status"] == xc.CORRECTED
    assert result["agreed"] == (verdict["alt"] + sum(result["third"])) / 3


def test_only_an_overflowing_vote_sum_uses_safe_averaging():
    third = pd.Series([1.0, 1e306], index=pd.DatetimeIndex([PREVIOUS, TODAY]))
    alt = (float(third.iloc[-1]) / float(third.iloc[0]) - 1.0) * 100.0
    verdict = {
        "status": xc.DIVERGED, "ours": 0.0, "alt": alt,
        "anchor": pd.Timestamp(PREVIOUS), "at": TODAY,
    }
    result = xc.settle(verdict, _tape(), [third, third])
    assert result["status"] == xc.CORRECTED
    assert np.isfinite(result["agreed"])
    assert np.isfinite(result["corrected_value"])
