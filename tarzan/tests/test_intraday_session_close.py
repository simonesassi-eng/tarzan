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

        # TWO points added, not one: the bell at the front on the previous close, the
        # official price at the back. Both ends of the session, from the two helpers.
        assert len(series) == len(bars) + 2
        assert float(series.iloc[0]) == pytest.approx(40.81)     # opens at 0%
        assert float(series.iloc[-1]) == pytest.approx(40.815)   # ends on the price
        # The baseline is untouched: it answers "what was the previous close", which
        # neither helper has any business restating.
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


class TestASessionChartDrawsOnlyTheElapsedSession:
    """A running session must not fill the width.

    ``chart_pct_compact`` spreads its points evenly BY INDEX, which is right for a
    month of closes and wrong for one trading day: a session 47% done filled the
    whole panel, so the 1D cell said the day was over while it was lunchtime. That
    is precisely the defect ``_intraday_spark`` was written to avoid, and the 1D
    grid cell reused the generic chart without inheriting the cure.

    ``x_span`` places every point at its real position in the venue's session, and
    labels where that session ends so the empty stretch reads as "not over yet"
    rather than "the feed stopped".
    """

    OPEN = "2026-09-02 09:00"
    CLOSE = "2026-09-02 17:30"

    @staticmethod
    def _svg(**kw):
        import pandas as pd

        from tarzan.export import _charts

        dates = pd.date_range("2026-09-02 09:00", "2026-09-02 13:00",
                              freq="30min", tz="Europe/Rome")
        series = [{"values": [0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.15, -0.05],
                   "color": "#E6EDF6", "width": 2.2, "end_label": "-0.05%"}]
        return _charts.chart_pct_compact(
            series, list(dates), w=182, h=116, date_fmt="%H:%M",
            min_day_ticks=3, end_gutter=46, **kw), dates

    @staticmethod
    def _last_x(svg):
        import re

        pts = re.findall(r'points="([^"]+)"', svg)
        assert pts, svg[:200]
        return max(float(p.split(",")[0]) for p in pts[-1].split())

    def _span(self):
        import pandas as pd

        return (pd.Timestamp(self.OPEN, tz="Europe/Rome"),
                pd.Timestamp(self.CLOSE, tz="Europe/Rome"))

    def test_without_a_span_the_line_still_fills_the_width(self):
        """The old behaviour, kept for every other panel: a month of closes SHOULD
        span its plot. This is the contrast the next test is measured against."""
        svg, _ = self._svg()
        plot_right = 182 - 46 - 8          # w - end_gutter - right margin
        assert self._last_x(svg) >= plot_right - 1

    def test_with_a_span_the_line_stops_where_the_day_has_got_to(self):
        svg, _ = self._svg(x_span=self._span())
        plot_right = 182 - 46 - 8
        # 09:00 -> 13:00 of an 09:00-17:30 session is 4h of 8.5h.
        expected = 30 + (4 / 8.5) * (plot_right - 30)
        assert self._last_x(svg) == pytest.approx(expected, abs=1.0)
        assert self._last_x(svg) < plot_right - 20

    def test_the_axis_names_where_the_session_ends(self):
        import re

        svg, _ = self._svg(x_span=self._span())
        assert "17:30" in re.findall(r'>(\d\d:\d\d)<', svg)

    def test_a_point_past_the_close_is_clamped_not_overflowed(self):
        """A venue can quote past its modelled close; the line must end at the edge
        rather than run off the chart."""
        import pandas as pd

        from tarzan.export import _charts

        dates = list(pd.date_range("2026-09-02 09:00", periods=3, freq="4h",
                                   tz="Europe/Rome"))     # 09:00, 13:00, 17:00
        dates.append(pd.Timestamp("2026-09-02 21:00", tz="Europe/Rome"))
        series = [{"values": [0.0, 0.2, -0.1, 0.3], "color": "#E6EDF6"}]
        svg = _charts.chart_pct_compact(
            series, dates, w=182, h=116, date_fmt="%H:%M", end_gutter=46,
            x_span=self._span())

        plot_right = 182 - 46 - 8
        assert self._last_x(svg) == pytest.approx(plot_right, abs=0.5)

    def test_a_degenerate_span_falls_back_to_the_even_spread(self):
        """Zero width, or a span the timestamps cannot be compared against, must not
        collapse every point onto one x."""
        import pandas as pd

        one = pd.Timestamp(self.OPEN, tz="Europe/Rome")
        svg, _ = self._svg(x_span=(one, one))
        plot_right = 182 - 46 - 8
        assert self._last_x(svg) >= plot_right - 1


class TestASessionOpensOnTheBell:
    """One bar is enough, and every line starts at 0%.

    A session line is a path away from the previous close, so at the bell its value
    IS that close. The bars do not say so — the first one already carries the opening
    gap, and a thin instrument's first print can be an hour in — and two things
    followed on the real book:

    *   a sleeve with a SINGLE bar had no drawable line at all, because every
        consumer floors at two points. The RETURNS and Watchlist rows fell back to a
        dashed placeholder, and the 1D panel's target blend (which demands full
        coverage) withheld its line for the first stretch of the day.
    *   a line whose first print is LATE opened mid-morning, missing the first hour of
        the session it claimed to draw.

    It does NOT force every line to 0%: where a bar already sits at the bell — 54 of
    63 series on the reference book — that bar is the opening and carries the gap.
    Nine were opening late and now do not.

    The point added is the series' own denominator, placed at the venue's open — the
    close-to-open convention, under which the overnight gap reads as the first
    segment, where it happened.
    """

    OPEN = pd.Timestamp("2026-09-01 07:00", tz="UTC")     # 09:00 Rome
    CLOSE = pd.Timestamp("2026-09-01 15:30", tz="UTC")    # 17:30 Rome

    @pytest.fixture(autouse=True)
    def _span(self, monkeypatch):
        monkeypatch.setattr(mq, "session_span",
                            lambda t, now=None: (self.OPEN, self.CLOSE))

    @staticmethod
    def _open(intra, baseline, ticker="AAA.MI"):
        return mq._prepend_session_open(
            intra, baseline, ticker=ticker,
            now=pd.Timestamp("2026-09-01 12:00", tz="UTC"))

    def test_a_single_bar_becomes_a_drawable_session(self):
        """The case that produced nothing at all before."""
        intra = _bars([31.05], start="09:00")

        out = self._open(intra, 31.19)

        assert len(out) == 2
        assert out.index[0] == self.OPEN
        assert float(out.iloc[0]) == pytest.approx(31.19)

    def test_a_line_whose_first_print_is_late_opens_on_the_previous_close(self):
        """The bell is 07:00 UTC here and the first bar 09:00, so two hours of session
        were missing from the drawing."""
        intra = _bars([31.05, 30.96], start="09:00")

        out = self._open(intra, 31.19)

        assert (float(out.iloc[0]) / 31.19 - 1) == pytest.approx(0.0, abs=1e-12)

    def test_the_bars_are_untouched(self):
        intra = _bars([31.05, 30.96], start="09:00")

        out = self._open(intra, 31.19)

        assert list(out.iloc[1:].values) == list(intra.values)
        assert list(out.index[1:]) == list(intra.index)

    def test_a_bar_at_or_before_the_open_is_not_displaced(self):
        """Pre-market prints exist. Inserting the bell after them would put the
        opening reference in the middle of the session."""
        intra = _bars([31.10, 31.05], start="06:30")      # first bar before the bell

        out = self._open(intra, 31.19)

        assert out is intra

    def test_no_modelled_session_means_no_bell(self, monkeypatch):
        """A continuously traded instrument (FX, futures) has no open to anchor on,
        and inventing one would draw a gap where trading never stopped."""
        monkeypatch.setattr(mq, "session_span", lambda t, now=None: None)
        intra = _bars([31.05], start="09:00")

        assert self._open(intra, 31.19) is intra

    @pytest.mark.parametrize("baseline", [None, 0.0, -1.0, "n/a"])
    def test_an_unusable_baseline_adds_nothing(self, baseline):
        intra = _bars([31.05], start="09:00")
        assert self._open(intra, baseline) is intra

    def test_an_empty_series_stays_empty(self):
        assert self._open(None, 31.19) is None

    def test_both_ends_together_make_a_full_session(self):
        """The two helpers compose: bell, the bars, then the official price."""
        intra = _bars([31.05], start="09:00")
        opened = self._open(intra, 31.19)

        full = mq._stamp_session_close(
            opened, {"price": 31.105,
                     "time": int(pd.Timestamp("2026-09-01 15:30",
                                              tz="UTC").timestamp())},
            own_feed=True, ticker="AAA.MI",
            now=pd.Timestamp("2026-09-01 18:00", tz="UTC"))

        assert len(full) == 3
        assert float(full.iloc[0]) == pytest.approx(31.19)      # 0% at the bell
        assert float(full.iloc[-1]) == pytest.approx(31.105)    # the close
        assert full.index[0] == self.OPEN and full.index[-1] == self.CLOSE
