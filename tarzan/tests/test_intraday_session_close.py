"""The intraday series must end where the session ended.

Bars are a record of TRADES. On a thinly traded instrument the last one sits well
before the session's end, and reading it as "now" makes every chart drawn from bars
disagree with every figure taken from the official price.

Measured across a 16-instrument portfolio on 1 Sep 2026, one render, so time cannot
explain it: the two readings disagreed for 11 of the 16, up to 0.72pp. The worst three
had their last hourly bar stamped 15:15, 14:30 and 14:00 UTC against a 17:35 Rome
close — two to three hours of session missing from what the chart called "now".

The five that agreed to the cent were exactly the ones whose last bar WAS the closing
price. And the denominator was identical in all sixteen cases, so the previous close
was never the problem: only "now" was.

Network-free: the provider seams are stubbed, so these pin the stamping rule rather
than Yahoo's data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.data import market_quotes as mq


def _bars(values, day="2026-09-01", start="14:00", freq="30min", tz="UTC"):
    idx = pd.date_range(f"{day} {start}", periods=len(values), freq=freq, tz=tz)
    return pd.Series([float(v) for v in values], index=idx)


def _stamp(intra, quote, *, own_feed=True, ticker="AAA.MI", now=None):
    return mq._stamp_session_close(
        intra, quote, own_feed=own_feed, ticker=ticker,
        now=now or pd.Timestamp("2026-09-01 20:00", tz="UTC"))


class TestAThinSessionGetsItsEnd:
    def test_the_official_price_becomes_the_last_point(self):
        """The measured shape: bars stop mid-afternoon, the official close is 31.105."""
        intra = _bars([31.20, 31.05, 30.96])        # 14:00, 14:30, 15:00 UTC
        out = _stamp(intra, {"price": 31.105,
                             "time": int(pd.Timestamp("2026-09-01 15:35",
                                                      tz="UTC").timestamp())})

        assert len(out) == len(intra) + 1
        assert float(out.iloc[-1]) == pytest.approx(31.105)
        assert out.index[-1] == pd.Timestamp("2026-09-01 15:35", tz="UTC")

    def test_the_derived_one_day_then_equals_the_official_pair(self):
        """The whole point. Against a 31.19 previous close, the bars' last print said
        −0.74% while the official pair says −0.27%; after stamping they are one
        number."""
        prev_close = 31.19
        intra = _bars([31.20, 31.05, 30.96])
        out = _stamp(intra, {"price": 31.105})

        from_bars = (float(intra.iloc[-1]) / prev_close - 1) * 100
        from_stamp = (float(out.iloc[-1]) / prev_close - 1) * 100
        official = (31.105 / prev_close - 1) * 100

        assert from_bars == pytest.approx(-0.7374, abs=1e-3)
        assert from_stamp == pytest.approx(official, abs=1e-12)

    def test_the_earlier_bars_are_untouched(self):
        """The bars carry the SHAPE; stamping may not rewrite it."""
        intra = _bars([31.20, 31.05, 30.96])
        out = _stamp(intra, {"price": 31.105})

        assert list(out.iloc[:-1].values) == list(intra.values)
        assert list(out.index[:-1]) == list(intra.index)


class TestASeriesAlreadyCurrentIsLeftAlone:
    def test_a_price_equal_to_the_last_bar_adds_no_duplicate_moment(self):
        """The agreeing shape: the last bar IS the close. A stamp at a LATER instant is
        still appended (it is a real observation) but the level does not move, so the
        line's end is unchanged either way."""
        intra = _bars([123.7, 123.14])
        out = _stamp(intra, {"price": 123.14})

        assert float(out.iloc[-1]) == pytest.approx(123.14)

    def test_a_quote_older_than_the_last_bar_is_skipped(self):
        """A stale quote must not become the end of a fresher series."""
        intra = _bars([31.20, 31.05, 30.96])
        stale = int(pd.Timestamp("2026-09-01 13:00", tz="UTC").timestamp())

        out = _stamp(intra, {"price": 99.0, "time": stale})

        assert out is intra

    def test_a_quote_at_the_last_bars_instant_is_skipped(self):
        intra = _bars([31.20, 30.96])
        same = int(intra.index[-1].timestamp())

        out = _stamp(intra, {"price": 99.0, "time": same})

        assert out is intra


class TestASiblingVenuesBarsAreNotStamped:
    """The shape where a listing has no bars of its own and a sibling venue supplied
    them.

    The baseline is that sibling's previous close too, so the drawn percentage is
    internally consistent within ITS order book. Appending the canonical listing's
    price would splice two order books into one line and draw a step no market made.
    That instrument's remaining disagreement is a different problem — which venue the
    bars belong to — and this rule refuses to hide it.
    """

    def test_a_fallback_series_is_returned_unchanged(self):
        intra = _bars([28.75, 28.715])
        out = _stamp(intra, {"price": 28.650}, own_feed=False)
        assert out is intra


class TestTheStampGoesAtARealInstant:
    def test_the_quotes_own_observation_time_wins(self):
        intra = _bars([31.20, 30.96])
        when = pd.Timestamp("2026-09-01 15:35", tz="UTC")

        out = _stamp(intra, {"price": 31.105, "time": int(when.timestamp())})

        assert out.index[-1] == when

    def test_without_one_a_finished_session_stamps_at_its_close(self, monkeypatch):
        """No observation instant, market shut: the close is where that price belongs.
        Not "now", which would stretch the drawn session hours past the bell."""
        close = pd.Timestamp("2026-09-01 15:35", tz="UTC")
        monkeypatch.setattr(mq, "session_span",
                            lambda t, now=None: (pd.Timestamp("2026-09-01 07:00",
                                                              tz="UTC"), close))
        intra = _bars([31.20, 30.96])

        out = _stamp(intra, {"price": 31.105},
                     now=pd.Timestamp("2026-09-01 20:00", tz="UTC"))

        assert out.index[-1] == close

    def test_without_one_a_running_session_stamps_at_now(self, monkeypatch):
        """Mid-session the close is in the FUTURE; stamping there would draw a session
        that has not happened yet."""
        close = pd.Timestamp("2026-09-01 15:35", tz="UTC")
        monkeypatch.setattr(mq, "session_span",
                            lambda t, now=None: (pd.Timestamp("2026-09-01 07:00",
                                                              tz="UTC"), close))
        now = pd.Timestamp("2026-09-01 14:20", tz="UTC")
        intra = _bars([31.20, 30.96], start="13:00")

        out = _stamp(intra, {"price": 31.105}, now=now)

        assert out.index[-1] == now
        assert out.index[-1] < close

    def test_no_session_and_no_observation_still_stamps_at_now(self, monkeypatch):
        """A continuously traded instrument (FX, futures) has no modelled session."""
        monkeypatch.setattr(mq, "session_span", lambda t, now=None: None)
        now = pd.Timestamp("2026-09-01 20:00", tz="UTC")
        intra = _bars([31.20, 30.96])

        out = _stamp(intra, {"price": 31.105}, now=now)

        assert out.index[-1] == now


class TestNothingToStampIsNotAnError:
    @pytest.mark.parametrize("quote", [None, {}, {"price": None},
                                       {"price": 0.0}, {"price": -3.0},
                                       {"price": "n/a"}])
    def test_an_unusable_quote_leaves_the_series_alone(self, quote):
        intra = _bars([31.20, 30.96])
        assert _stamp(intra, quote) is intra

    def test_no_series_is_no_series(self):
        assert _stamp(None, {"price": 31.105}) is None


class TestTheResolverStampsWhatItHandsOut:
    """End to end through ``intraday_feeds``, because the point of stamping in the
    provider is that EVERY consumer gets it — the 1D window panel, the RETURNS
    table's sparklines and the Markets strip read this one series."""

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        """Same pinning as ``test_yahoo_return_alignment``: Tuesday 08:58 CEST, Milan
        shut, the measured session Monday's. Without a pinned clock the resolver's
        staleness gate discards a fixture dated in the past and hands back no series
        at all — which is what the first draft of these two tests actually measured."""
        pinned = pd.Timestamp("2026-08-18 08:58", tz="Europe/Rome")
        monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: pinned)
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        monkeypatch.setattr(
            "tarzan.runtime.today",
            lambda: __import__("datetime").date(2026, 8, 18))

    def test_the_series_a_consumer_receives_ends_on_the_official_price(self, monkeypatch):
        bars = _bars([40.79, 40.70], day="2026-08-17", start="14:45", tz="UTC")
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {"EXUS.MI": bars})
        monkeypatch.setattr(
            "tarzan.data.enricher._fetch_history",
            lambda s: pd.DataFrame({"Close": [40.96, 40.81, None]},
                                   index=pd.to_datetime(
                                       ["2026-08-13", "2026-08-14", "2026-08-17"])))
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "EXUS.MI": {"price": 40.815, "prev_close": 40.81}})

        got = mq.intraday_feeds(["EXUS.MI"])["EXUS.MI"]
        series = got["intraday_series"]

        assert float(series.iloc[-1]) == pytest.approx(40.815)
        assert len(series) == len(bars) + 1
        # The baseline is untouched: it answers "what was the previous close", which
        # the stamp has no business restating.
        assert float(got["intraday_baseline"]) == pytest.approx(40.81)

    def test_the_observation_timestamp_still_describes_the_bars(self, monkeypatch):
        """``intraday_observation_timestamp`` feeds the valuation freshness policy and
        means "when did the FEED print". Stamping must not move it, or a stamped price
        would relabel the bars' own recency."""
        bars = _bars([40.79, 40.70], day="2026-08-17", start="14:45", tz="UTC")
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {"EXUS.MI": bars})
        monkeypatch.setattr(
            "tarzan.data.enricher._fetch_history",
            lambda s: pd.DataFrame({"Close": [40.96, 40.81, None]},
                                   index=pd.to_datetime(
                                       ["2026-08-13", "2026-08-14", "2026-08-17"])))
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "EXUS.MI": {"price": 40.815, "prev_close": 40.81,
                        "time": int(pd.Timestamp("2026-08-17 15:35",
                                                 tz="UTC").timestamp())}})

        got = mq.intraday_feeds(["EXUS.MI"])["EXUS.MI"]

        assert pd.Timestamp(got["intraday_observation_timestamp"]) == bars.index[-1]


class TestAQuoteFromAnotherSessionIsRefused:
    """The gap the ``stamp <= last_ts`` rule does NOT cover.

    ``_stamp_session_close`` reads the RAW official quote, bypassing
    ``current_session.pick_quote`` — the price-coherence gate that keeps a rotten
    quote out of the valuation. Yahoo keeps dormant records and answers them: on a
    real holding one listing returned ``regularMarketTime`` 326 days old with a
    price 10.9% away from the truth, no error and no flag.

    A quote OLDER than the last bar was already refused, by the rule that a series
    already carrying the latest price is left alone — so the year-stale case was
    covered before this guard existed, and tests asserting it prove nothing.

    What was NOT covered is the opposite pairing: a stale intraday FEED (yesterday's
    bars, which the resolver's own staleness gate can still admit close to a session
    boundary) beside a CURRENT quote. There the quote is newer than the last bar, so
    the old rule waves it through and a two-day "session" gets drawn. The guard asks
    whether both belong to the same session, not merely which came first.
    """

    def test_todays_quote_does_not_extend_yesterdays_bars(self):
        """The case the ordering rule misses: newer, but a different session."""
        intra = _bars([31.20, 30.96], day="2026-09-01", start="08:00")
        today = int(pd.Timestamp("2026-09-02 09:41", tz="UTC").timestamp())

        out = _stamp(intra, {"price": 31.105, "time": today})

        assert out is intra, (
            "a current price was appended to a previous session's bars, drawing a "
            "line that spans two days")

    def test_a_quote_from_the_same_session_still_lands(self):
        """The guard must not refuse the normal case. Mid-session the quote is
        minutes old and belongs to exactly the session being drawn."""
        intra = _bars([31.20, 30.96], day="2026-09-01", start="08:00")
        same = int(pd.Timestamp("2026-09-01 09:41", tz="UTC").timestamp())

        out = _stamp(intra, {"price": 31.105, "time": same})

        assert len(out) == len(intra) + 1
        assert float(out.iloc[-1]) == pytest.approx(31.105)

    def test_a_year_old_quote_is_refused_either_way(self):
        """Belt and braces, and labelled as such: this passes with or without the
        guard, because such a quote is also older than the last bar."""
        intra = _bars([31.20, 31.05, 30.96])
        stale = int(pd.Timestamp("2025-10-10 15:35", tz="UTC").timestamp())

        assert _stamp(intra, {"price": 25.515, "time": stale}) is intra
