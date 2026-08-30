"""The vendored exchange calendar, against real known closures.

Tarzan decided "the previous session" and "n sessions back" with a Mon-Fri rule
in four places, each carrying its own note that holidays were not modelled. That
slid every window by one session around each exchange holiday — 6 a year on
Milan, 17 on Tokyo. These pin the calendar that replaced it, on dates that can
be checked against any exchange notice.
"""

from __future__ import annotations

import datetime

import pytest

from tarzan.data import exchange_calendar as ec


class TestVenueResolution:
    def test_a_suffix_selects_its_own_exchange(self):
        assert ec.venue_mic("EXUS.MI") == "XMIL"
        assert ec.venue_mic("XDEQ.DE") == "XETR"
        assert ec.venue_mic("NTSG.PA") == "XPAR"
        assert ec.venue_mic("SGLD.L") == "XLON"

    def test_a_german_regional_venue_follows_the_german_calendar(self):
        """Munich keeps its own 08:00-22:00 hours (a market_quotes concern) but
        trades on the days Germany trades."""
        assert ec.venue_mic("IS39.MU") == "XETR"

    def test_an_index_resolves_through_its_exchange_group(self):
        assert ec.venue_mic("^GSPC") == "XNYS"
        assert ec.venue_mic("^FTSE") == "XLON"
        assert ec.venue_mic("^N225") == "XTKS"

    def test_an_unmappable_symbol_has_no_venue(self):
        assert ec.venue_mic("") is None
        assert ec.venue_mic("EURUSD=X") is None
        assert ec.venue_mic("BTC-USD") is None


class TestRealClosures:
    @pytest.mark.parametrize("ticker,day,expected,why", [
        ("EXUS.MI", "2025-08-15", False, "Assumption — Borsa Italiana closed"),
        ("XDEQ.DE", "2025-08-15", True, "same day, Xetra open: no Assumption"),
        ("^GSPC", "2026-04-03", False, "Good Friday — NYSE closed"),
        ("^GSPC", "2026-11-26", False, "Thanksgiving — NYSE closed"),
        ("^GSPC", "2026-10-12", True,
         "Columbus Day — a US FEDERAL holiday the NYSE trades through, which "
         "is why pandas' USFederalHolidayCalendar is the wrong table"),
        ("^N225", "2026-05-04", False, "Golden Week — Tokyo closed"),
        ("EXUS.MI", "2026-01-01", False, "New Year — everyone closed"),
        ("EXUS.MI", "2026-08-14", True, "an ordinary Friday"),
    ])
    def test_known_sessions_and_closures(self, ticker, day, expected, why):
        assert ec.is_session(
            ticker, datetime.date.fromisoformat(day)) is expected, why

    def test_a_weekend_is_never_a_session(self):
        for ticker in ("EXUS.MI", "^GSPC", "^N225", "NOPE.XX"):
            assert ec.is_session(ticker, datetime.date(2026, 8, 22)) is False
            assert ec.is_session(ticker, datetime.date(2026, 8, 23)) is False


class TestSessionArithmetic:
    def test_a_weekend_cutoff_rolls_back_to_the_friday(self):
        """The anchor rule Yahoo's own figures use: XMME.MI's published 1M on
        18 Aug 2026 (+2.55%) is reproducible only from Friday 17 July, the
        session before that window's Saturday cutoff."""
        assert ec.last_session_on_or_before(
            "EXUS.MI", datetime.date(2026, 5, 24)) == datetime.date(2026, 5, 22)

    def test_a_session_is_its_own_anchor(self):
        day = datetime.date(2026, 8, 14)          # an ordinary Friday
        assert ec.last_session_on_or_before("EXUS.MI", day) == day

    def test_previous_session_skips_a_holiday(self):
        """The day after Thanksgiving is a session, and the one before it is the
        Wednesday — not the closed Thursday a Mon-Fri rule would name."""
        assert ec.previous_session(
            "^GSPC", datetime.date(2026, 11, 27)) == datetime.date(2026, 11, 25)

    def test_five_sessions_differ_between_venues_across_a_holiday(self):
        """The window that made this worth building: Milan shut for the
        Assumption on Friday 15 Aug 2025 and Xetra did not, so five sessions
        back from the Monday reaches a different day on each."""
        monday = datetime.date(2025, 8, 18)
        assert ec.sessions_back("EXUS.MI", monday, 4) == datetime.date(2025, 8, 11)
        assert ec.sessions_back("XDEQ.DE", monday, 4) == datetime.date(2025, 8, 12)

    def test_zero_steps_back_is_the_end_session_itself(self):
        sunday = datetime.date(2026, 8, 23)
        assert ec.sessions_back("EXUS.MI", sunday, 0) == datetime.date(2026, 8, 21)


class TestUnknownIsNotOpen:
    def test_beyond_the_horizon_it_degrades_to_the_weekday_rule(self):
        """The source calendars bound themselves about a year ahead, so the
        table does. A far-future weekday must read as a session by FALLBACK, not
        because the table asserts it traded."""
        far = datetime.date(2035, 1, 2)
        assert ec.covers("EXUS.MI", far) is False
        assert ec.is_session("EXUS.MI", far) is True

    def test_an_unmapped_venue_degrades_to_the_weekday_rule(self):
        assert ec.covers("NOPE.XX", datetime.date(2026, 8, 24)) is False
        assert ec.is_session("NOPE.XX", datetime.date(2026, 8, 24)) is True
        assert ec.is_session("NOPE.XX", datetime.date(2026, 8, 22)) is False

    def test_a_missing_table_does_not_break_a_run(self, monkeypatch):
        monkeypatch.setattr(ec, "_table", lambda: ({}, {}, {}))
        assert ec.is_session("EXUS.MI", datetime.date(2025, 8, 15)) is True
        assert ec.last_session_on_or_before(
            "EXUS.MI", datetime.date(2026, 8, 23)) == datetime.date(2026, 8, 21)


class TestEveryReachableVenueHasACalendar:
    """The same discipline as the session-hours coverage test: a venue the
    resolver can settle on but this table does not know silently reverts to the
    Mon-Fri rule, which is the gap the calendar exists to close."""

    def test_every_isin_resolver_suffix_maps_to_a_calendar(self):
        from tarzan.data.enricher import ISIN_EXCHANGE_SUFFIXES
        missing = [s for s in ISIN_EXCHANGE_SUFFIXES
                   if s.lstrip(".").upper() not in ec._SUFFIX_MIC]
        assert missing == []

    def test_every_eur_venue_the_resolver_probes_maps_to_a_calendar(self):
        from tarzan.data.enricher import _EUR_VENUE_SUFFIXES
        missing = [s for s in _EUR_VENUE_SUFFIXES
                   if s.lstrip(".").upper() not in ec._SUFFIX_MIC]
        assert missing == []

    def test_every_sibling_fallback_suffix_maps_to_a_calendar(self):
        from tarzan.data.market_quotes import _SIBLING_SUFFIXES
        reachable = set(_SIBLING_SUFFIXES)
        for siblings in _SIBLING_SUFFIXES.values():
            reachable.update(siblings)
        assert [s for s in sorted(reachable) if s not in ec._SUFFIX_MIC] == []

    def test_every_session_group_maps_to_a_calendar(self):
        from tarzan.data.market_quotes import _SESSIONS
        assert [g for g in _SESSIONS if g not in ec._GROUP_MIC] == []

    def test_every_mapped_mic_is_present_in_the_table(self):
        closed, _, _early = ec._table()
        wanted = set(ec._SUFFIX_MIC.values()) | set(ec._GROUP_MIC.values())
        assert sorted(m for m in wanted if m not in closed) == []


class TestTheHotPathResolvesTheVenueOnce:
    """A session walk must not re-resolve the venue per step.

    Resolving a suffixless symbol reaches the curated taxonomy, which rebuilds a
    record list per call. ``sessions_back`` walks up to five sessions and each
    step probed up to fifteen days, so one five-day window asked for the venue
    dozens of times — that turned a 40-second suite into a 70-minute one while
    every test still passed, which is why this is asserted rather than timed.
    """

    def test_a_five_session_walk_resolves_the_venue_once(self, monkeypatch):
        calls = []
        real = ec.venue_mic.__wrapped__

        def counting(ticker):
            calls.append(ticker)
            return real(ticker)

        monkeypatch.setattr(ec, "venue_mic", counting)
        ec.sessions_back("EXUS.MI", datetime.date(2026, 8, 24), 4)
        assert len(calls) == 1, f"venue resolved {len(calls)} times, expected 1"

    def test_repeated_lookups_are_served_from_the_cache(self):
        ec.venue_mic.cache_clear()
        for _ in range(50):
            ec.venue_mic("EXUS.MI")
        info = ec.venue_mic.cache_info()
        assert info.misses == 1 and info.hits == 49
