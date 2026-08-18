"""A Tarzan return must be the figure the instrument's Yahoo page shows.

Two things had to change for that to hold, and each is pinned here.

1. The window edges are Yahoo's: its ranges are CALENDAR spans (1mo, 3mo, 6mo,
   1y, 3y, 5y) and its 5D spans five sessions. Tarzan measured fixed day counts,
   so 90 days anchored two sessions past three calendar months — verified
   against the site on 18 Aug 2026: EXUS.MI 3M read +6.45% where Yahoo showed
   +7.87%, from 20 May against Yahoo's 18 May.

2. The 1D is measured against the quote pair (``regularMarketPrice`` /
   ``regularMarketPreviousClose``), not against the daily chart. Same day, every
   Milan ETF came back with a NULL close for Monday 17 Aug in the chart feed
   while the quote endpoint carried the official 40.815 — so the chart alone
   could only measure from the Friday.

Network-free: fixture series and an explicit quote pair.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tarzan.data.market_quotes as mq
from tarzan.engine.stats import PERIOD_DAYS, compute_period_return, window_anchor


def _closes(pairs):
    return pd.Series([v for _, v in pairs],
                     index=pd.to_datetime([d for d, _ in pairs]))


def _business_series(start: str, end: str, step: float = 0.1):
    idx = pd.bdate_range(start, end)
    return pd.Series([100.0 + i * step for i in range(len(idx))], index=idx)


class TestWindowEdgesAreYahoosRanges:
    def test_three_months_is_three_calendar_months_not_ninety_days(self):
        # 18 May → 18 Aug 2026 is 92 days: a 90-day window starts two sessions
        # late, which is exactly the gap measured against the site.
        s = _business_series("2026-01-02", "2026-08-18")
        anchor = window_anchor(s, PERIOD_DAYS["3m"])
        assert anchor == pd.Timestamp("2026-05-18")

    def test_five_years_follows_the_calendar_across_leap_days(self):
        s = _business_series("2019-01-01", "2026-08-18")
        assert window_anchor(s, PERIOD_DAYS["5y"]) == pd.Timestamp("2021-08-18")

    def test_one_week_is_five_sessions_counted_on_the_calendar(self):
        """The last five sessions, so a vendor gap cannot widen the window.

        Milan's 17 Aug 2026 is missing from Yahoo's daily feed; counting five
        ROWS back from Tue 18 Aug lands on 11 Aug, a session earlier than the
        site's own 5D range, and printed -0.74% against its -0.95%.
        """
        holed = _closes([
            ("2026-08-10", 100.0), ("2026-08-11", 101.0), ("2026-08-12", 102.0),
            ("2026-08-13", 103.0), ("2026-08-14", 104.0),
            # 17 Aug traded but the vendor has no bar for it.
            ("2026-08-18", 105.0),
        ])
        assert window_anchor(holed, PERIOD_DAYS["1w"]) == pd.Timestamp("2026-08-12")

    def test_a_bucket_the_series_cannot_cover_is_unavailable(self):
        """Yahoo silently shows MAX for a range longer than the history (EXUS.MI
        reported the same +38.04% under 3y and 5y over 2.3 years of data).
        Tarzan reports the bucket as absent instead of relabelling a short
        window."""
        s = _business_series("2024-05-13", "2026-08-18")
        assert window_anchor(s, PERIOD_DAYS["5y"]) is None
        assert compute_period_return(s, PERIOD_DAYS["5y"]) is None


class TestOneDayIsTheQuotePair:
    def _intraday(self, values, day, start, tz="Europe/Rome", freq="15min"):
        idx = pd.date_range(f"{day} {start}", periods=len(values), freq=freq, tz=tz)
        return pd.Series([float(v) for v in values], index=idx)

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        # Tuesday 08:58 CEST: Milan shut, the session being measured is Monday's.
        pinned = pd.Timestamp("2026-08-18 08:58", tz="Europe/Rome")
        monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: pinned)
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)

    def test_a_null_close_in_the_chart_does_not_move_the_baseline(self, monkeypatch):
        """The 18 Aug 2026 shape: Monday absent from the daily feed, so the
        chart's own prior close is FRIDAY's. The quote pair still measures
        Monday against Friday, which is what the page showed."""
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {
            "EXUS.MI": self._intraday([40.79, 40.825], "2026-08-17", "16:45")})
        monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda s: pd.DataFrame(
            {"Close": [40.96, 40.81, None]},
            index=pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-17"])))
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "EXUS.MI": {"price": 40.815, "prev_close": 40.81}})

        selected = mq.broker_1d(["EXUS.MI"])["EXUS.MI"]

        assert round(selected["pct"], 4) == round((40.815 / 40.81 - 1) * 100, 4)
        assert selected["intraday_baseline"] == 40.81
        assert selected["live"] is False          # Milan is not trading at 08:58
        assert selected["source_ticker"] == "EXUS.MI"

    def test_an_instrument_with_no_intraday_feed_still_gets_the_published_move(
            self, monkeypatch):
        """Before, such a row fell back to a close-to-close read of the same
        holed chart; now it carries the quote pair with no sparkline."""
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
        monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda s: None)
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "X710.MI": {"price": 100.5, "prev_close": 100.0}})

        selected = mq.broker_1d(["X710.MI"])["X710.MI"]

        assert round(selected["pct"], 4) == 0.5
        assert selected["intraday_series"] is None
        assert selected["live"] is False
