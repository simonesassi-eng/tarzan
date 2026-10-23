"""Intraday availability is separate from liquidity, freshness and plotting."""

import pandas as pd
import pytest

from tarzan.data import market_quotes as mq
from tarzan.runtime.ledger import Availability


SYMBOL = "SYNOBS.MI"
NOW = pd.Timestamp("2026-10-23 14:00", tz="UTC")


def bars(values, start="2026-10-23 12:45", *, freq="15min"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq, tz="UTC"))


def batch(symbol, series):
    return pd.DataFrame(
        series.to_numpy(), index=series.index,
        columns=pd.MultiIndex.from_tuples([(symbol, "Close")]),
        dtype=series.dtype,
    )


@pytest.fixture
def sources(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: NOW.tz_convert(tzinfo or "UTC"))
    monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda symbol: pd.DataFrame(
        {"Close": [99., 100.]}, index=pd.to_datetime(["2026-10-21", "2026-10-22"]),
    ))
    monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
        symbol: {"price": 101.5, "prev_close": 100.,
                 "time": int(pd.Timestamp("2026-10-23 13:30", tz="UTC").timestamp())}
        for symbol in symbols
    })


def test_single_current_observation_survives_the_native_ingress(sources, monkeypatch):
    raw = batch(SYMBOL, bars([101.]))
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    result = mq._fetch_intraday([SYMBOL])
    assert result[SYMBOL].equals(bars([101.]))


def test_single_current_observation_does_not_trigger_speculative_venue_queries(sources, monkeypatch):
    calls = []
    series = bars([101.])
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: calls.append(symbols) or {SYMBOL: series})
    result = mq._resolve_intraday([SYMBOL])
    assert result[SYMBOL][1] == SYMBOL
    assert calls == [[SYMBOL]]


def test_single_observation_reaches_the_real_feed_pipeline_and_keeps_actual_timestamp(sources, monkeypatch):
    raw = batch(SYMBOL, bars([101.]))
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    result = mq.intraday_feeds([SYMBOL])
    quote = result[SYMBOL]
    assert quote["intraday_source_ticker"] == SYMBOL
    assert quote["intraday_series"].loc[pd.Timestamp("2026-10-23 12:45", tz="UTC")] == 101.
    assert quote["intraday_observation_timestamp"] == pd.Timestamp("2026-10-23 12:45", tz="UTC")
    assert len(quote["intraday_series"]) >= 2
    assert quote["native_bar_count"] == 1


def test_quote_baseline_alone_is_not_reported_as_complete_intraday_coverage(sources, monkeypatch):
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
    result = mq.intraday_feeds([SYMBOL])
    assert result[SYMBOL]["intraday_series"] is None
    assert result.availability is Availability.DEGRADED
    assert result.policy["drawable_tickers"] == 0
    assert result.policy["missing_tickers"] == (SYMBOL,)


@pytest.mark.parametrize("allow_siblings", [False, True])
def test_lagged_current_session_is_retained_as_lagged_display_evidence_not_fresh_data(
    sources, monkeypatch, allow_siblings,
):
    original = bars([100., 100.2], "2026-10-23 08:00")
    monkeypatch.setattr(
        mq, "_fetch_intraday",
        lambda symbols: {SYMBOL: original} if SYMBOL in symbols else {},
    )
    result = mq._resolve_intraday([SYMBOL], allow_sibling_fallback=allow_siblings)
    assert not mq._has_intraday(original, SYMBOL)
    assert result[SYMBOL][0].equals(original)
    assert result[SYMBOL][1] == SYMBOL
    assert result.policy["evidence"][SYMBOL]["state"] == "LAGGED_CURRENT_SESSION"
    assert result.availability is Availability.DEGRADED


def test_a_previous_session_cannot_be_relabelled_as_current_lagged_evidence(sources, monkeypatch):
    original = bars([100., 100.2], "2026-10-22 08:00")
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {SYMBOL: original})
    result = mq._resolve_intraday([SYMBOL], allow_sibling_fallback=False)
    assert SYMBOL not in result


def test_fresh_verified_candidate_still_wins_over_lagged_canonical_path(sources, monkeypatch):
    stale, fresh = bars([100., 100.2], "2026-10-23 08:00"), bars([100.5, 100.7])
    monkeypatch.setattr(mq, "_sibling_symbols", lambda symbol: ["SYNOBS.DE"])
    monkeypatch.setattr(
        mq, "_fetch_intraday",
        lambda symbols: {SYMBOL: stale} if SYMBOL in symbols else {"SYNOBS.DE": fresh},
    )
    result = mq._resolve_intraday([SYMBOL])
    assert result[SYMBOL][1] == "SYNOBS.DE"


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, "bad", float("nan"), float("inf"), 10 ** 400])
def test_invalid_prices_do_not_count_as_observed_intraday_bars(sources, monkeypatch, bad):
    series = pd.Series([bad, bad], index=bars([101., 102.]).index, dtype=object)
    raw = batch(SYMBOL, series)
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    assert SYMBOL not in mq._fetch_intraday([SYMBOL])


def test_future_observations_cannot_become_source_coverage(sources, monkeypatch):
    raw = batch(SYMBOL, bars([101., 102.], "2026-10-23 15:00"))
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    assert SYMBOL not in mq._fetch_intraday([SYMBOL])


def test_malformed_timestamp_index_fails_closed(sources, monkeypatch):
    raw = pd.DataFrame([101., 102.], index=["bad", "future"],
                       columns=pd.MultiIndex.from_tuples([(SYMBOL, "Close")]))
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    assert SYMBOL not in mq._fetch_intraday([SYMBOL])


def test_conflicting_duplicate_timestamps_cannot_be_arbitrarily_chosen(sources, monkeypatch):
    series = bars([101., 102.])
    series.index = pd.DatetimeIndex([series.index[0], series.index[0]])
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: batch(SYMBOL, series))
    assert SYMBOL not in mq._fetch_intraday([SYMBOL])


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), 10 ** 400])
def test_invalid_published_endpoint_cannot_extend_an_observed_path(sources, bad):
    original = bars([100., 101.])
    quote = {"price": bad, "time": int(pd.Timestamp("2026-10-23 13:30", tz="UTC").timestamp())}
    assert mq._stamp_session_close(original, quote, own_feed=True, ticker=SYMBOL, now=NOW) is original


def test_a_published_endpoint_after_the_capture_cannot_extend_the_path(sources):
    original = bars([100., 101.])
    quote = {"price": 102., "time": int(pd.Timestamp("2026-10-23 15:00", tz="UTC").timestamp())}
    assert mq._stamp_session_close(original, quote, own_feed=True, ticker=SYMBOL, now=NOW) is original


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, "bad", float("nan"), float("inf"), 10 ** 400])
def test_corrupt_baseline_falls_back_to_same_venue_daily_close(sources, monkeypatch, bad):
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {SYMBOL: bars([101.])})
    monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
        symbol: {"price": 101.5, "prev_close": bad} for symbol in symbols
    })
    quote = mq.intraday_feeds([SYMBOL])[SYMBOL]
    assert quote["intraday_baseline"] == 100.
    assert quote["native_bar_count"] == 1
    assert all(pd.notna(quote["intraday_series"]))


@pytest.mark.parametrize("observed", [
    "2026-10-22 13:30", "2026-10-23 15:30",
])
def test_another_session_or_future_quote_cannot_supply_the_chart_baseline(sources, monkeypatch, observed):
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {SYMBOL: bars([101.])})
    monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
        symbol: {"price": 110., "prev_close": 90.,
                 "time": int(pd.Timestamp(observed, tz="UTC").timestamp())}
        for symbol in symbols
    })
    quote = mq.intraday_feeds([SYMBOL])[SYMBOL]
    assert quote["intraday_baseline"] == 100.
    assert 110. not in quote["intraday_series"].values


def test_a_lagged_path_cannot_acquire_a_current_endpoint_from_an_undated_quote(sources, monkeypatch):
    original = bars([100., 100.2], "2026-10-23 08:00")
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {SYMBOL: original} if SYMBOL in symbols else {})
    monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
        symbol: {"price": 101.5, "prev_close": 100.} for symbol in symbols
    })
    quote = mq.intraday_feeds([SYMBOL])[SYMBOL]
    assert quote["intraday_state"] == "LAGGED_CURRENT_SESSION"
    assert quote["intraday_series"].index[-1] == original.index[-1]
    assert quote["intraday_observation_timestamp"] == original.index[-1]


def test_a_closed_session_is_labelled_completed_not_a_fresh_current_quote(sources, monkeypatch):
    now = pd.Timestamp("2026-10-23 20:00", tz="UTC")
    monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: now.tz_convert(tzinfo or "UTC"))
    original = bars([100., 100.2], "2026-10-23 08:00")
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {SYMBOL: original})
    result = mq._resolve_intraday([SYMBOL])
    assert result.policy["evidence"][SYMBOL]["state"] == "LAST_COMPLETED_SESSION"
    assert result.policy["evidence"][SYMBOL]["observed_at"] == original.index[-1].isoformat()


def test_continuous_market_cannot_accept_stale_unknown_session_as_cash_display_evidence(sources, monkeypatch):
    original = bars([100., 100.2], "2026-10-23 08:00")
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {"EURUSD=X": original})
    assert "EURUSD=X" not in mq._resolve_intraday(["EURUSD=X"])


def test_equal_duplicate_timestamps_are_deduplicated_without_inventing_trades(sources, monkeypatch):
    series = bars([101., 101.])
    series.index = pd.DatetimeIndex([series.index[0], series.index[0]])
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: batch(SYMBOL, series))
    result = mq._fetch_intraday([SYMBOL])
    assert len(result[SYMBOL]) == 1
    assert result[SYMBOL].iloc[0] == 101.


def test_unordered_native_bars_are_sorted_without_changing_observations(sources, monkeypatch):
    original = bars([101., 102., 103.])
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: batch(SYMBOL, original.iloc[::-1]))
    result = mq._fetch_intraday([SYMBOL])
    assert result[SYMBOL].equals(original)


def test_a_partial_batch_retains_healthy_symbols_and_reason_codes(sources, monkeypatch):
    raw = batch(SYMBOL, bars([101., 102.]))
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: raw)
    result = mq._fetch_intraday([SYMBOL, "ABSENT.MI"])
    assert set(result) == {SYMBOL}
    assert result.policy["evidence"]["ABSENT.MI"]["reason"] == "NO_SYMBOL_OR_CLOSE"
    assert result.availability is Availability.DEGRADED


def test_provider_timeout_is_unavailability_not_a_successful_empty_catalog(sources, monkeypatch):
    monkeypatch.setattr(
        "tarzan.data._yf_net.fetch_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("synthetic")),
    )
    result = mq._fetch_intraday([SYMBOL])
    assert not result
    assert result.availability is Availability.UNAVAILABLE


def test_disabled_transport_does_not_open_a_new_intraday_request(sources, monkeypatch):
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: False)
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download", lambda *args, **kwargs: pytest.fail("offline I/O"))
    assert not mq._fetch_intraday([SYMBOL])


def test_one_batched_request_scales_without_querying_siblings_for_genuine_single_bars(sources, monkeypatch):
    symbols = [f"SYN{i}.MI" for i in range(512)]
    raw = pd.DataFrame(
        [[100.] * len(symbols)], index=bars([100.]).index,
        columns=pd.MultiIndex.from_tuples((symbol, "Close") for symbol in symbols),
    )
    calls = []
    monkeypatch.setattr("tarzan.data._yf_net.fetch_download",
                        lambda requested, **kwargs: calls.append(tuple(requested)) or raw)
    result = mq._resolve_intraday(symbols + symbols)
    assert len(result) == 512
    assert calls == [tuple(symbols)]
    assert all(fact["native_bar_count"] == 1 for fact in result.policy["evidence"].values())
