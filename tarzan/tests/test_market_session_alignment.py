"""One session authority, one ``as_of``: the market-data time contract.

Regression cover for the 09:09 CEST send that reported "market CLOSED" with
Milan and London trading, printed the previous session's close-to-close moves
under an "INTRADAY" heading, and drew forty days of daily closes as if they were
a session that had been running since the open.

The bugs were three readings of one missing fact — which trading session, in the
exchange's own timezone, is current as of the run's reference instant:

* :func:`session_day` is that fact;
* :func:`market_open_now` is a DIFFERENT one (is the venue trading), and the two
  legitimately disagree in the first minutes after an open;
* what the newsletter may CLAIM follows from both, plus whether any bar exists.

Network-free: every test pins the clock and feeds fixture series.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

import tarzan.data.market_quotes as mq
from tarzan.data.market_quotes import fetch_market_quotes  # real impl, bound now


# ── fixtures ────────────────────────────────────────────────────────────────

def _pin(monkeypatch, ts: str, tz: str = "Europe/Rome"):
    """Pin the ONE run reference instant (the seam every session and freshness
    decision reads) to ``ts`` in ``tz``."""
    pinned = pd.Timestamp(ts, tz=tz)
    monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: pinned)
    return pinned


def _bars(values, day, start, tz, freq="15min"):
    idx = pd.date_range(f"{day} {start}", periods=len(values), freq=freq, tz=tz)
    return pd.Series([float(v) for v in values], index=idx)


def _daily(pairs):
    return pd.DataFrame({"Close": [v for _, v in pairs]},
                        index=pd.to_datetime([d for d, _ in pairs]))


def _batch(series_by_symbol):
    """A ``yf.download(group_by="ticker")``-shaped response."""
    frames = {}
    for sym, ser in series_by_symbol.items():
        frames[(sym, "Close")] = ser
        frames[(sym, "Volume")] = pd.Series(1000.0, index=ser.index)
    out = pd.DataFrame(frames)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


# ── the session authority ───────────────────────────────────────────────────

class TestSessionDay:
    """``session_day`` is the exchange-local date of the live-or-last session."""

    def test_after_the_open_it_is_today(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 09:09")  # Thursday, 9 min into Milan
        assert mq.session_day("SGLD.MI") == date(2026, 8, 13)

    def test_before_the_open_it_is_the_previous_trading_day(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 08:30")  # Thursday, Milan not open yet
        assert mq.session_day("SGLD.MI") == date(2026, 8, 12)

    def test_monday_pre_open_walks_back_over_the_weekend(self, monkeypatch):
        _pin(monkeypatch, "2026-08-10 07:00")  # Monday, before any EU open
        assert mq.session_day("SGLD.MI") == date(2026, 8, 7)  # Friday

    def test_each_exchange_uses_its_own_clock(self, monkeypatch):
        # 09:09 CEST: London (08:09 GMT) is 9 min in, New York (03:09 ET) has
        # not opened, so the two disagree about which session is current.
        _pin(monkeypatch, "2026-08-13 09:09")
        assert mq.session_day("^FTSE") == date(2026, 8, 13)
        assert mq.session_day("^GSPC") == date(2026, 8, 12)

    def test_none_for_a_continuously_traded_instrument(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 09:09")
        for sym in ("CL=F", "EURUSD=X", "BTC-USD"):
            assert mq.session_day(sym) is None, sym

    def test_it_is_a_different_question_from_market_open_now(self, monkeypatch):
        """The distinction the bug collapsed: minutes after an open the venue is
        OPEN and the current session still has no data."""
        _pin(monkeypatch, "2026-08-13 09:09")
        assert mq.market_open_now("SGLD.MI") is True
        assert mq.session_day("SGLD.MI") == date(2026, 8, 13)
        _pin(monkeypatch, "2026-08-13 08:30")
        assert mq.market_open_now("SGLD.MI") is False
        assert mq.session_day("SGLD.MI") == date(2026, 8, 12)


class TestEveryReachableVenueHasASession:
    """A suffix the resolver can pick but ``_SUFFIX_EXCHANGE`` does not know has
    no session: ``session_day`` returns None and that instrument silently keeps
    the old "whatever day the data ends on" convention. IS39 (quoted only in
    Munich) is how that gap put the previous session's bars in a live column."""

    def test_every_isin_resolver_suffix_maps_to_a_session(self):
        from tarzan.data.enricher import ISIN_EXCHANGE_SUFFIXES
        missing = [s for s in ISIN_EXCHANGE_SUFFIXES
                   if s.lstrip(".").upper() not in mq._SUFFIX_EXCHANGE]
        assert missing == []

    def test_every_sibling_fallback_suffix_maps_to_a_session(self):
        reachable = set(mq._SIBLING_SUFFIXES)
        for sibs in mq._SIBLING_SUFFIXES.values():
            reachable.update(sibs)
        assert [s for s in sorted(reachable)
                if s not in mq._SUFFIX_EXCHANGE] == []

    def test_every_mapped_group_is_a_modelled_session(self):
        for table in (mq._SUFFIX_EXCHANGE, mq._INDEX_EXCHANGE):
            for key, group in table.items():
                assert group in mq._SESSIONS, f"{key} → {group}"

    def test_a_munich_only_listing_gets_munich_hours(self, monkeypatch):
        """08:30 CEST: Xetra is shut, the German regional venues have traded for
        half an hour, so their session is TODAY, not yesterday."""
        _pin(monkeypatch, "2026-08-13 08:30")
        assert mq.session_day("IS39.MU") == date(2026, 8, 13)
        assert mq.session_day("IS39.MI") == date(2026, 8, 12)   # Milan not open


class TestSessionSpan:
    """``session_span`` is the x-axis a session chart must be drawn on."""

    def test_it_is_the_venues_own_open_and_close(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 09:09")
        start, end = mq.session_span("SGLD.MI")
        assert (start.hour, start.minute) == (9, 0)
        assert (end.hour, end.minute) == (17, 30)
        assert start.date() == date(2026, 8, 13)
        assert (end - start).total_seconds() == 8.5 * 3600

    def test_it_follows_session_day_before_the_open(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 08:00")
        start, _end = mq.session_span("SGLD.MI")
        assert start.date() == date(2026, 8, 12)

    def test_none_when_no_cash_session_is_modelled(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 09:09")
        for sym in ("CL=F", "EURUSD=X", "BTC-USD", "WHATEVER.XYZ"):
            assert mq.session_span(sym) is None, sym


class TestReferenceInstantIsTheRunClock:
    """One tz-aware ``as_of`` for the whole issue, from the run context."""

    def test_live_run_uses_the_run_capture_instant(self):
        from tarzan import runtime
        captured = runtime.configure().captured_at
        try:
            now = mq._intraday_reference_now()
            assert now.tzinfo is not None
            assert now == captured.astimezone(timezone.utc)
        finally:
            runtime.reset()

    def test_point_in_time_run_uses_the_end_of_its_effective_date(self):
        """``--as_of`` bounds the market data too: the reference instant is the
        end of that date, so the invariant is one comparison."""
        from tarzan import runtime
        runtime.configure(as_of=date(2026, 8, 13))
        try:
            now = mq._intraday_reference_now()
            assert now.tzinfo is not None
            assert now.astimezone(timezone.utc).date() == date(2026, 8, 13)
            assert now.astimezone(timezone.utc).hour == 23
        finally:
            runtime.reset()


# ── invariant: no observation later than as_of ──────────────────────────────

class TestAsOfInvariant:
    def test_bars_after_the_reference_instant_are_dropped(self, monkeypatch):
        """A vendor window is requested by period, not by bound, so under
        ``--as_of`` it returns bars from after the effective date."""
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        ser = _bars([100.0, 101.0, 102.0, 103.0], "2026-08-13", "09:00",
                    "Europe/Rome", freq="30min")  # 09:00, 09:30, 10:00, 10:30
        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                            lambda fn, **kw: _batch({"SGLD.MI": ser}))
        now = _pin(monkeypatch, "2026-08-13 09:45")

        out = mq._fetch_intraday(["SGLD.MI"])["SGLD.MI"]

        assert len(out) == 2                       # 09:00 and 09:30 only
        assert out.index.max() <= now
        assert 102.0 not in set(out.values)        # the 10:00 print is unusable

    def test_the_invariant_holds_across_the_whole_strip(self, monkeypatch):
        """Every symbol is clipped against the SAME instant, so two rows of one
        issue cannot straddle ``as_of``."""
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        a = _bars([1.0, 2.0, 3.0], "2026-08-13", "09:00", "Europe/Rome", freq="30min")
        b = _bars([1.0, 2.0, 3.0], "2026-08-13", "09:00", "Europe/London", freq="30min")
        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                            lambda fn, **kw: _batch({"^FCHI": a, "^FTSE": b}))
        now = _pin(monkeypatch, "2026-08-13 10:15")

        out = mq._fetch_intraday(["^FCHI", "^FTSE"])

        for sym, ser in out.items():
            assert ser.index.max() <= now, sym


# ── selection: the current session, in the exchange's timezone ─────────────

class TestSessionSelection:
    def test_a_prior_session_is_not_returned_at_the_open(self, monkeypatch):
        """THE bug. At 09:09 the 15m feed still holds only yesterday's bars;
        returning them made a completed session read as today's."""
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        stale = _bars([7000.0, 7010.0, 7020.0], "2026-08-12", "17:00",
                      "Europe/Rome")
        monkeypatch.setattr("tarzan.data._yf_net.fetch_yf",
                            lambda fn, **kw: _batch({"^FCHI": stale}))
        _pin(monkeypatch, "2026-08-13 09:09")

        assert "^FCHI" not in mq._fetch_intraday(["^FCHI"])

    def test_todays_bars_are_kept_once_they_exist(self, monkeypatch):
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        idx = _bars([7000.0, 7010.0], "2026-08-12", "17:00", "Europe/Rome")
        fresh = _bars([7030.0, 7040.0], "2026-08-13", "09:00", "Europe/Rome")
        monkeypatch.setattr(
            "tarzan.data._yf_net.fetch_yf",
            lambda fn, **kw: _batch({"^FCHI": pd.concat([idx, fresh])}))
        _pin(monkeypatch, "2026-08-13 09:40")

        out = mq._fetch_intraday(["^FCHI"])["^FCHI"]
        assert list(out.values) == [7030.0, 7040.0]

    def test_a_session_straddling_utc_midnight_is_not_split(self, monkeypatch):
        """Sydney's session spans two UTC dates, so a UTC ``.date()`` filter
        kept only the tail of it. The exchange's own date keeps it whole."""
        # January: Sydney is UTC+11, so a 10:00 open is 23:00 UTC the day before.
        ser = _bars([100.0, 101.0, 102.0, 103.0], "2026-01-15", "10:00",
                    "Australia/Sydney", freq="60min")
        assert min(ts.astimezone(timezone.utc).date() for ts in ser.index) != \
            max(ts.astimezone(timezone.utc).date() for ts in ser.index)
        now = _pin(monkeypatch, "2026-01-15 15:00", tz="Australia/Sydney")

        out = mq._select_session_series(ser, "BHP.AX", now)
        assert len(out) == 4

    def test_continuous_markets_keep_the_last_day_present(self, monkeypatch):
        """Futures/FX/crypto have no cash session to be "current"."""
        ser = pd.concat([
            _bars([100.0, 101.0], "2026-08-12", "20:00", "UTC"),
            _bars([102.0, 103.0], "2026-08-13", "01:00", "UTC"),
        ])
        now = _pin(monkeypatch, "2026-08-13 09:09")
        out = mq._select_session_series(ser, "CL=F", now)
        assert list(out.values) == [102.0, 103.0]


# ── the % a level is measured against ──────────────────────────────────────

class TestBaselinePairing:
    """``regularMarketPreviousClose`` is the close before the CURRENT session,
    so pairing it with a level from an earlier one measures nothing."""

    def test_a_prior_session_level_is_not_paired_with_todays_prev_close(self):
        # London is open (Thursday), but the freshest level is Wednesday's
        # close. previousClose is Wednesday's too → the old code printed 0.00%.
        daily = _daily([("2026-08-11", 10_700.0), ("2026-08-12", 10_833.0)])
        q = mq._quote(daily["Close"], None, official_prev=10_833.0,
                      current_session_day=date(2026, 8, 13))
        assert q["observed_day"] == date(2026, 8, 12)
        assert q["stale_session"] is True
        # measured against the close BEFORE its own session, not previousClose
        assert round(q["pct"], 2) == 1.24        # Wednesday's real move
        assert q["pct"] != 0.0

    def test_a_current_session_level_still_uses_the_official_prev_close(self):
        daily = _daily([("2026-08-11", 10_700.0), ("2026-08-12", 10_833.0)])
        q = mq._quote(daily["Close"], None, official_prev=10_800.0,
                      current_session_day=date(2026, 8, 12))
        assert q["stale_session"] is False
        assert round(q["pct"], 2) == 0.31        # 10833 vs the official 10800

    def test_a_level_is_dated_in_the_exchanges_timezone(self):
        """A Sydney bar at 10:00 AEDT is 23:00 UTC the day before, so dating it
        in UTC would call a live session stale and rebase the %."""
        bars = _bars([100.0, 101.0], "2026-01-15", "10:00", "UTC")  # 10:00 UTC…
        bars.index = bars.index.tz_convert("UTC") - pd.Timedelta(hours=11)
        assert bars.index[-1].date() == date(2026, 1, 14)           # …→ 23:00 UTC
        from zoneinfo import ZoneInfo
        q = mq._quote(None, bars, current_session_day=date(2026, 1, 15),
                      session_tz=ZoneInfo("Australia/Sydney"))
        assert q["observed_day"] == date(2026, 1, 15)
        assert q["stale_session"] is False

    def test_a_continuous_market_keeps_the_official_prev_close(self):
        """No cash session → no session to be stale against."""
        daily = _daily([("2026-08-11", 180.0), ("2026-08-12", 200.0)])
        q = mq._quote(daily["Close"], None, official_prev=250.0,
                      current_session_day=None)
        assert q["stale_session"] is False
        assert round(q["pct"], 2) == -20.0       # 200 vs the official 250

    def test_the_strip_reports_the_previous_sessions_real_move_at_the_open(
            self, monkeypatch):
        """End to end: FTSE 100 at 09:09 CEST, no Thursday bars yet."""
        mq._memo = None
        daily = _daily([("2026-08-11", 10_700.0), ("2026-08-12", 10_833.0)])
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
        monkeypatch.setattr(mq, "_fetch_official_prev_closes",
                            lambda symbols: {"^FTSE": 10_833.0})
        monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                            lambda s: daily if s == "^FTSE" else None)
        _pin(monkeypatch, "2026-08-13 09:09")
        try:
            ftse = {d["name"]: d for d in fetch_market_quotes(force=True)}["FTSE 100"]
            assert ftse["value"] == 10_833.0
            assert round(ftse["pct"], 2) == 1.24     # not +0.00%
            assert ftse["stale_session"] is True
            assert ftse["observed_day"] == date(2026, 8, 12)
        finally:
            mq._memo = None


# ── what the newsletter may claim ──────────────────────────────────────────

class TestMarketOpenCaption:
    """The masthead and the State tile state EXCHANGE HOURS, not data basis."""

    def _stamp(self, perf):
        from tarzan.export.newsletter._sections_alloc import _market_is_open
        return _market_is_open(perf)

    def test_open_venue_with_no_intraday_data_still_reads_open(self):
        # The 09:09 send: market_open True, 1d_live False.
        assert self._stamp({"market_open": True, "1d_live": False}) is True

    def test_closed_venue_reads_closed(self):
        assert self._stamp({"market_open": False, "1d_live": False}) is False

    def test_falls_back_to_the_data_basis_when_the_engine_is_silent(self):
        # Point-in-time runs and older projections carry no market_open.
        assert self._stamp({"1d_live": True}) is True
        assert self._stamp({}) is False
        assert self._stamp(None) is False

    def test_engine_projects_exchange_hours_independently_of_the_feed(
            self, monkeypatch):
        """``market_open`` is computed before any fetch, so a provider failure
        cannot silence it — that is what printed "market CLOSED" while Milan
        traded."""
        from tarzan.engine.metrics import MetricsEngine

        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        monkeypatch.setattr(mq, "market_open_now",
                            lambda t, now=None: t.endswith(".MI"))

        def _boom(*a, **kw):
            raise RuntimeError("provider down")

        monkeypatch.setattr(mq, "broker_1d", _boom)
        engine = MetricsEngine.__new__(MetricsEngine)
        engine.holdings = []
        hp = pd.DataFrame({"ticker": ["SGLD.MI"], "1d": [0.78]})
        ctx = {"holding_performance": hp, "performance": {},
               "performance_full": {}}
        engine._live_1d(ctx)

        assert ctx["market_open"] is True
        assert ctx["performance"]["market_open"] is True
        assert not ctx["performance"].get("1d_live")


class TestMarketOpenSpeaksForThePortfolio:
    """The 08:58 CEST send of 18 Aug 2026: "market OPEN" with every holding's
    venue shut, because one Munich-only tracked listing quotes from 08:00."""

    def _holding(self, ticker, value):
        from tarzan.models.holding import Holding

        return Holding(isin=f"TEST{ticker}", ticker=ticker, quantity=1.0,
                       cost_basis_eur=value, market_value_eur=value,
                       currency="EUR", current_value=value)

    def _open_flag(self, monkeypatch, holdings, tickers):
        from tarzan.engine.metrics import MetricsEngine

        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)

        def _boom(*a, **kw):
            raise RuntimeError("provider down")

        monkeypatch.setattr(mq, "broker_1d", _boom)
        engine = MetricsEngine.__new__(MetricsEngine)
        engine.holdings = holdings
        ctx = {"holding_performance": pd.DataFrame({"ticker": tickers}),
               "performance": {}, "performance_full": {}}
        engine._live_1d(ctx)
        return ctx["market_open"]

    def test_an_early_regional_venue_does_not_open_the_portfolio(self, monkeypatch):
        _pin(monkeypatch, "2026-08-18 08:58")     # Munich trading, Milan is not
        assert mq.market_open_now("IS39.MU") is True
        assert self._open_flag(
            monkeypatch,
            [self._holding("EXUS.MI", 100_000.0), self._holding("IS39.MU", 1_000.0)],
            ["EXUS.MI", "IS39.MU"],
        ) is False

    def test_the_venues_carrying_the_value_do_open_it(self, monkeypatch):
        _pin(monkeypatch, "2026-08-18 09:30")     # Milan two minutes in… and on
        assert self._open_flag(
            monkeypatch,
            [self._holding("EXUS.MI", 100_000.0), self._holding("IS39.MU", 1_000.0)],
            ["EXUS.MI", "IS39.MU"],
        ) is True

    def test_cash_and_futures_weigh_on_neither_side(self, monkeypatch):
        """No modelled session → no opinion: a big cash row must not outvote the
        equity sleeve that is actually trading."""
        _pin(monkeypatch, "2026-08-18 09:30")
        assert self._open_flag(
            monkeypatch,
            [self._holding("EXUS.MI", 10_000.0), self._holding("EURUSD=X", 90_000.0)],
            ["EXUS.MI", "EURUSD=X"],
        ) is True


class TestSessionTileNamesItsBasis:
    """A completed session's move must not be captioned as the live one."""

    def _basis(self, perf, last_close="14 Aug"):
        from tarzan.export.newsletter._sections_alloc import _session_basis

        class _M:
            portfolio_history = pd.DataFrame(
                {"value": [1.0, 2.0]},
                index=pd.to_datetime(["2026-08-13", "2026-08-14"]))

        return _session_basis(perf, _M())

    def test_a_non_live_figure_states_the_close_it_measures_from(self):
        # The Tuesday 08:58 shape: a real close-to-close move, no intraday feed.
        assert self._basis({"market_open": True, "1d_live": False}) == \
            "close-to-close vs 14 Aug"

    def test_a_live_figure_states_the_exchange_hours(self):
        assert self._basis({"market_open": True, "1d_live": True}) == "market open"


class TestIntradayColumnHeader:
    """"Intraday" is one claim about a whole column."""

    def test_one_intraday_row_does_not_label_the_column_intraday(self):
        from tarzan.export.newsletter._sections_perf import _intraday_column
        # The real shape: 1 of 40 instruments had bars (a German venue printing
        # before Milan's open) and the header said INTRADAY over 39
        # close-to-close figures.
        assert _intraday_column([True] + [False] * 39) is False

    def test_all_intraday_labels_the_column_intraday(self):
        from tarzan.export.newsletter._sections_perf import _intraday_column
        assert _intraday_column([True, True, True]) is True

    def test_empty_is_not_intraday(self):
        from tarzan.export.newsletter._sections_perf import _intraday_column
        assert _intraday_column([]) is False

    def test_the_header_word_follows(self):
        from tarzan.export.newsletter._charts import day_column_label
        assert day_column_label(None, live=False) == "1D"
        assert day_column_label(None, live=True) == "Intraday"


class TestStripSparklineAtTheOpen:
    """Requirement: a chart drawn minutes after an open must not look like a
    session that has been running all day."""

    def _spark(self, quote):
        from tarzan.export.newsletter import _sections_perf as sp
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.models.portfolio import PortfolioMetrics
        from tarzan.models.investor_config import InvestorConfig
        import tarzan.data.market_quotes as _mq

        ctx = _NewsletterContext(metrics=PortfolioMetrics(),
                                 config=InvestorConfig())
        # _build_markets closes over its own _spark_for; drive it through the
        # section with a single-row snapshot rather than reaching inside.
        _mq_snap = [dict(quote)]
        monkey = _mq.fetch_market_quotes
        try:
            _mq.fetch_market_quotes = lambda *a, **kw: _mq_snap
            sp.fetch_market_quotes = lambda *a, **kw: _mq_snap
            return sp._build_markets(ctx)["html"]
        finally:
            _mq.fetch_market_quotes = monkey

    def _quote(self, **kw):
        base = {"name": "FTSE 100", "symbol": "^FTSE", "category": "Europe",
                "value": 10_833.0, "change": 133.0, "pct": 1.24,
                "spark": [10_500.0 + i * 10 for i in range(40)],
                "baseline": 10_500.0, "spark_series": None,
                "observed_day": date(2026, 8, 12), "stale_session": False}
        base.update(kw)
        return base

    def test_a_stale_session_draws_no_session_path(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 09:09")
        html = self._spark(self._quote(stale_session=True))
        assert "stroke-dasharray" in html     # the dashed placeholder
        assert "Wed close" in html            # and which session the % is
        assert "<polyline" not in html        # NOT a 40-day path stretched wide

    def test_a_completed_session_still_draws_its_daily_context(self, monkeypatch):
        _pin(monkeypatch, "2026-08-13 20:00")  # after the London close
        html = self._spark(self._quote(stale_session=False))
        assert "<polyline" in html

    def test_an_open_venue_with_no_intraday_feed_draws_no_session_path(
            self, monkeypatch):
        """Bund 10Y: Yahoo has a daily row for today but no 15m feed at all, so
        it is not "stale" — and 40 daily closes stretched wide next to an "Op."
        badge is the same lie the stale case was fixed for."""
        _pin(monkeypatch, "2026-08-13 10:30")   # London trading
        html = self._spark(self._quote(stale_session=False))
        assert "stroke-dasharray" in html       # the dashed placeholder
        assert "<polyline" not in html


# ── the chart's drawn extent is the traded extent ──────────────────────────

def _poly_xs(html: str) -> list:
    """X coordinates of the drawn path in a sparkline SVG."""
    import re
    m = re.search(r'<polyline points="([^"]+)"', html)
    assert m is not None, html[:300]
    return [float(p.split(",")[0]) for p in m.group(1).split()]


class TestChartExtentIsTradedExtent:
    """A session chart spread over the full width says "a whole session traded"
    no matter how little of one the bars cover. Drawing on the venue's real
    session window makes the claim true by construction — no flag needed."""

    def test_a_sliver_of_a_session_covers_a_sliver_of_the_width(self, monkeypatch):
        from tarzan.export.newsletter._charts import _intraday_spark
        _pin(monkeypatch, "2026-08-13 09:40")
        ser = _bars([100.0, 101.0, 102.0], "2026-08-13", "09:00", "Europe/Rome")
        html = _intraday_spark(ser, 100.0, w=62, h=22,
                               span=mq.session_span("SGLD.MI"))
        xs = _poly_xs(html)
        # 09:00→09:30 of an 09:00–17:30 session: under a tenth of the width.
        assert max(xs) < 62 * 0.12

    def test_a_completed_session_fills_the_width(self, monkeypatch):
        from tarzan.export.newsletter._charts import _intraday_spark
        _pin(monkeypatch, "2026-08-13 18:00")
        ser = _bars([100.0 + i for i in range(35)], "2026-08-13", "09:00",
                    "Europe/Rome")  # 09:00 → 17:30
        html = _intraday_spark(ser, 100.0, w=62, h=22,
                               span=mq.session_span("SGLD.MI"))
        assert max(_poly_xs(html)) == 62.0

    def test_a_sparse_watchlist_row_is_not_stretched(self, monkeypatch):
        """IS39: three prints, ``live=False`` (its venue was unmodelled, so
        exchange hours read None) — the old axis spread them over a whole
        session. The venue that PRODUCED the bars supplies the window."""
        from tarzan.export.newsletter._sections_perf import _perf_spark_cell
        _pin(monkeypatch, "2026-08-13 09:40", tz="Europe/Berlin")
        ser = _bars([50.0, 50.2, 50.1], "2026-08-13", "09:00", "Europe/Berlin")
        quote = {"intraday_series": ser, "intraday_baseline": 50.0,
                 "intraday_source_ticker": "IS39.MU"}
        _cell, inner = _perf_spark_cell(0.2, "IS39", {"IS39": quote}, live=False)
        xs = _poly_xs(inner)
        # 09:00→09:30 of Munich's 08:00–22:00 quoting day.
        assert max(xs) < 62 * 0.15

    def test_an_unmodelled_venue_keeps_the_elapsed_time_axis(self, monkeypatch):
        """No span → the pre-existing behaviour, unchanged: an open session grows
        from its first bar, a closed one spreads evenly."""
        from tarzan.export.newsletter._charts import _intraday_spark
        ser = _bars([100.0, 101.0, 102.0], "2026-08-13", "09:00", "UTC")
        growing = _poly_xs(_intraday_spark(ser, 100.0, w=62, in_progress=True))
        spread = _poly_xs(_intraday_spark(ser, 100.0, w=62, in_progress=False))
        assert max(growing) < 62 * 0.15
        assert max(spread) == 62.0
