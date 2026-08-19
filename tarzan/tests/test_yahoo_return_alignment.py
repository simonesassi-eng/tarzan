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
from tarzan.engine.stats import compute_period_return, window_anchor


def _closes(pairs):
    return pd.Series([v for _, v in pairs],
                     index=pd.to_datetime([d for d, _ in pairs]))


def _business_series(start: str, end: str, step: float = 0.1):
    idx = pd.bdate_range(start, end)
    return pd.Series([100.0 + i * step for i in range(len(idx))], index=idx)


class TestTrailingWindowAnchor:
    @pytest.fixture(autouse=True)
    def _pin_today(self, monkeypatch):
        # Every fixture below ends 2026-08-18 and reasons about windows
        # measured back from it (see _window_end's docstring). Without this,
        # window_anchor reads the real run-owned clock (runtime.today()),
        # so a fixture written against one date silently drifts by a day
        # each day the suite runs after it - which is exactly what the
        # three tests below this one, before this fixture existed, did.
        import datetime
        monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2026, 8, 18))

    def test_a_month_snaps_to_the_last_session_on_or_before(self):
        """The bug the user caught: XMME.MI 1M read +1.13% (from Mon 20 Jul)
        where the published figure is +2.55% (from Fri 17 Jul). The window
        starts one calendar month back — 18 Jul, a Saturday — so the anchor is
        the last session AT OR BEFORE it (Friday), not the first one after.
        """
        s = _closes([
            ("2026-07-16", 78.53), ("2026-07-17", 76.78),   # window start = Fri
            ("2026-07-20", 77.86),                           # NOT this Monday
            ("2026-08-14", 80.12), ("2026-08-18", 78.74),
        ])
        assert window_anchor(s, "1m") == pd.Timestamp("2026-07-17")
        assert round(compute_period_return(s, "1m"), 2) == 2.55

    def test_three_months_and_five_years_follow_the_calendar(self):
        s = _business_series("2019-01-01", "2026-08-18")
        assert window_anchor(s, "3m") == pd.Timestamp("2026-05-18")
        assert window_anchor(s, "5y") == pd.Timestamp("2021-08-18")

    def test_five_days_is_five_sessions(self):
        s = _business_series("2026-07-01", "2026-08-18")
        # 18 Aug (Tue) back four business days → 12 Aug (Wed): five sessions.
        assert window_anchor(s, "5d") == pd.Timestamp("2026-08-12")

    def test_a_stale_feed_shortens_the_window_instead_of_sliding_it(self):
        """NTSG.MI on 18 Aug 2026: no close after the 14th. Counting five
        sessions back from its OWN last row reached 8 Aug and reported +0.79%;
        counting back from today gives the +0.60% its page shows over the
        sessions the window really contains.
        """
        stale = _closes([
            ("2026-08-07", 28.8), ("2026-08-10", 29.0), ("2026-08-11", 29.1),
            ("2026-08-12", 29.2), ("2026-08-13", 29.4), ("2026-08-14", 29.6),
        ])
        assert window_anchor(stale, "5d") == pd.Timestamp("2026-08-12")
        assert round(compute_period_return(stale, "5d"), 2) == round((29.6 / 29.2 - 1) * 100, 2)

    def test_the_money_and_percent_columns_share_one_window(self):
        """The matrix row said "7 days" and mixed two spans: the euros walked
        seven CALENDAR days while the TWROR beside them measured five sessions.
        Both now read window_anchor, so 5D bills the same five sessions.
        """
        from tarzan.export._perf_series import _window_money_pnl

        idx = pd.date_range("2026-08-04", "2026-08-18", freq="D")  # ends Tuesday
        pnl = pd.Series(range(len(idx)), index=idx, dtype=float)    # +€1/day
        actual = pd.Series([1000.0] * len(idx), index=idx)

        gain, _pct = _window_money_pnl(pnl, actual, "5d")
        # Tue 18 Aug back four sessions is Wed 12 Aug: six calendar days of P&L,
        # not the seven a "1 week" window used to bill.
        assert gain == 6.0

    def test_a_bucket_the_series_cannot_cover_is_unavailable(self):
        """Yahoo silently shows MAX for a range longer than the history (EXUS.MI
        reported the same +38.04% under 3y and 5y over 2.3 years of data).
        Tarzan reports the bucket as absent instead of relabelling a short
        window."""
        s = _business_series("2024-05-13", "2026-08-18")
        assert window_anchor(s, "5y") is None
        assert compute_period_return(s, "5y") is None


class TestWindowMatrixHasOneSource:
    """Every WINDOW row, including 1D, is derived from the one price series.

    The 1D row used to be overwritten by a separate live-quote move
    (``performance_full["1d"]``, de-compounded against invested_value), so the
    row could print a number the series never contained — -€2.8k beside a +€11
    Session tile on 19 Aug 2026. The override is gone: the row reads the same
    ``pnl_series`` / ``portfolio_history`` as the 5D and 1M rows, whose current
    point the engine has already stamped from the live valuation. A divergent
    ``performance_full`` must not be able to move it.
    """

    def _matrix(self, monkeypatch, injected_1d):
        import datetime
        import re

        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig
        from tarzan.models.portfolio import PortfolioMetrics

        monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2026, 8, 19))
        idx = pd.date_range("2026-06-20", "2026-08-19", freq="D")
        values = []
        for day in idx:
            value = 100_000.0
            if day >= pd.Timestamp("2026-08-18"):
                value = 96_000.0          # yesterday dropped 4%
            if day == pd.Timestamp("2026-08-19"):
                value = 96_010.0          # today is flat: +€10
            values.append(value)
        actual = pd.Series(values, index=idx)
        pnl = actual - 90_000.0
        metrics = PortfolioMetrics(
            total_value=96_010.0, invested_value=96_010.0, cash_value=0.0,
            holdings_df=pd.DataFrame([{"cost_basis_eur": 90_000.0, "ticker": "AAA",
                                       "weight_pct": 100.0}]))
        metrics.actual_value_series = actual
        metrics.pnl_series = pnl
        metrics.unrealized_series = pnl
        metrics.portfolio_history = pd.Series([v / 1000 for v in values], index=idx)
        metrics.pnl_eur, metrics.pnl_pct, metrics.twror_pct = 6_010.0, 6.7, 6.5
        metrics.performance_full = {"1d": injected_1d}

        html = _build_performance30(_NewsletterContext(
            metrics=metrics, config=InvestorConfig()))["matrix_html"]
        cells = {}
        for label in ("1D", "5D", "1M"):
            block = re.search(rf">{label}<.*?</tr>", html, flags=re.S).group(0)
            cells[label] = re.findall(r">([+\-−]?€?[\d.,k]+%?)<", block)
        return cells

    def test_the_one_day_row_is_the_series_last_session_move(self, monkeypatch):
        # 19 Aug (Tue) vs 18 Aug (Mon): +€10, +0.01%. The injected +99% must
        # NOT appear — the row has no path to performance_full any more.
        cells = self._matrix(monkeypatch, injected_1d=99.0)
        joined = " ".join(cells["1D"])
        assert "+€10" in joined, joined
        assert "+0.01%" in joined, joined
        assert "99" not in joined, joined

    def test_the_five_day_row_is_series_derived(self, monkeypatch):
        cells = self._matrix(monkeypatch, injected_1d=99.0)
        assert "−3.99%" in " ".join(cells["5D"])


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
