"""Today's market point is written ONCE, in the data layer, before the policy.

Every "today" figure reads a price series while the portfolio's own valuation is
selected separately, so "today" used to exist twice. The stamp lived in
``MetricsEngine``, which runs after ``ValuationCompletenessEvaluator`` and after
``_valuation`` — so ``total_value`` and the series terminal could not agree by
construction. Measured on a live run (24 Aug 2026, €242k book) they sat €115
apart: fourteen of sixteen holdings agreed to four decimals, and both that did
not were fallbacks — MONEY.MI's valuation had dropped to its 10.0920 order price
while the market quoted 10.1840.

These pin the contract of the relocated stamp. Network-free: quotes injected.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan.data import current_session as cs
from tarzan.models.holding import Holding


def _holding(ticker: str, closes: list[float], quantity: float,
             *, start: str = "2026-08-17") -> Holding:
    idx = pd.bdate_range(start, periods=len(closes))
    h = Holding(isin=f"XX{ticker:0<10}"[:12], ticker=ticker, quantity=quantity,
                cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR")
    h.price_history = pd.Series(closes, index=idx)
    h.current_price = closes[-1]
    h.current_value = closes[-1] * quantity
    h.price_is_fallback = True          # as a stale enrichment rung would leave it
    h.price_observation_timestamp = None
    return h


@pytest.fixture(autouse=True)
def _live_wednesday(monkeypatch):
    """A live run on Wed 19 Aug 2026, so stamping is allowed."""
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.runtime.today",
                        lambda: datetime.date(2026, 8, 19))


def _quotes(monkeypatch, mapping):
    monkeypatch.setattr(
        "tarzan.data.market_quotes.official_quotes",
        lambda symbols: {s: mapping[s] for s in symbols if s in mapping})


class TestTheValuationAndTheSeriesEndOnOneNumber:
    def test_the_stamp_moves_price_value_and_series_together(self, monkeypatch):
        """The MONEY.MI shape: the series runs on real closes while the
        valuation sat on a staler rung. After the stamp all three — series
        terminal, current_price and current_value — are the quote."""
        h = _holding("MONEY.MI", [10.15, 10.16, 10.092], quantity=100.0)
        _quotes(monkeypatch, {"MONEY.MI": {"price": 10.184, "prev_close": 10.10}})

        stamped = cs.apply_to_holdings([h])

        assert stamped == ("MONEY.MI",)
        assert float(h.price_history.iloc[-1]) == 10.184
        assert h.price_history.index[-1] == pd.Timestamp("2026-08-19")
        assert h.current_price == 10.184
        assert h.current_value == pytest.approx(1018.4)
        # The series terminal and the valuation are now the SAME number, which
        # is the whole point of doing this before the policy runs.
        assert h.current_value == pytest.approx(
            float(h.price_history.iloc[-1]) * h.quantity)

    def test_a_stamped_price_is_primary_dated_evidence(self, monkeypatch):
        """The policy dates freshness on the observation, so a level-validated
        published quote must clear the fallback flag and carry its own time —
        otherwise a live price would be judged as undated evidence."""
        h = _holding("EXUS.MI", [40.5, 40.6, 40.31], quantity=10.0)
        observed = int(datetime.datetime(
            2026, 8, 19, 15, 30, tzinfo=datetime.timezone.utc).timestamp())
        _quotes(monkeypatch, {
            "EXUS.MI": {"price": 40.4, "prev_close": 40.31, "time": observed}})

        cs.apply_to_holdings([h])

        assert h.price_is_fallback is False
        assert h.price_observation_timestamp == datetime.datetime(
            2026, 8, 19, 15, 30, tzinfo=datetime.timezone.utc)

    def test_an_undated_quote_still_stamps_but_carries_no_time(self, monkeypatch):
        h = _holding("XDEQ.MI", [78.0, 78.5, 78.72], quantity=1.0)
        _quotes(monkeypatch, {"XDEQ.MI": {"price": 78.80, "prev_close": 78.72}})

        cs.apply_to_holdings([h])

        assert h.current_price == 78.80
        assert h.price_observation_timestamp is None


class TestTheGateStillProtectsTheValuation:
    def test_a_corrupt_quote_leaves_the_holding_untouched(self, monkeypatch):
        """NTSG.MI's quote priced the fund at 25.5 against a real ~29.4. Since
        the stamp now also writes the VALUATION, letting that through would move
        the portfolio total, not just a chart."""
        h = _holding("NTSG.MI", [29.3, 29.4, 29.45], quantity=1000.0)
        before_value, before_price = h.current_value, h.current_price
        before_series = h.price_history.copy()
        _quotes(monkeypatch, {"NTSG.MI": {"price": 25.515, "prev_close": 25.805}})

        assert cs.apply_to_holdings([h]) == ()
        assert h.current_value == before_value
        assert h.current_price == before_price
        # Neither the 25.515 quote nor its 25.805 prev_close reached the series.
        assert h.price_history.equals(before_series)

    def test_a_clean_sibling_supplies_the_quote(self, monkeypatch):
        h = _holding("NTSG.MI", [29.3, 29.4, 29.45], quantity=1.0)
        _quotes(monkeypatch, {
            "NTSG.MI": {"price": 25.515, "prev_close": 25.805},   # corrupt
            "NTSG.DE": {"price": 29.35, "prev_close": 29.45}})     # clean

        assert cs.apply_to_holdings([h]) == ("NTSG.MI",)
        assert h.current_price == 29.35

    def test_a_holding_with_no_series_is_skipped(self, monkeypatch):
        h = _holding("X.MI", [1.0], quantity=1.0)
        h.price_history = None
        _quotes(monkeypatch, {"X.MI": {"price": 5.0, "prev_close": 4.0}})

        assert cs.apply_to_holdings([h]) == ()


class TestNoLiveObservationEntersAReproducibleRun:
    def test_a_pinned_run_stamps_nothing(self, monkeypatch):
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: False)
        h = _holding("EXUS.MI", [40.5, 40.6, 40.31], quantity=1.0)
        _quotes(monkeypatch, {"EXUS.MI": {"price": 40.4, "prev_close": 40.31}})

        assert cs.apply_to_holdings([h]) == ()
        assert h.current_price == 40.31

    def test_an_undated_weekend_quote_stamps_nothing(self, monkeypatch):
        """Sat 22 Aug 2026 with a quote the provider dated nothing.

        The date falls back to the run's own day, which is not a session, so
        nothing is written: a weekend-dated point would slide window_anchor onto
        the day a month before the WEEKEND rather than the last session.
        """
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 22))
        h = _holding("EXUS.MI", [40.5, 40.6, 40.31], quantity=1.0)
        _quotes(monkeypatch, {"EXUS.MI": {"price": 40.4, "prev_close": 40.31}})

        assert cs.apply_to_holdings([h]) == ()
        assert h.price_history.index[-1] != pd.Timestamp("2026-08-22")

    def test_a_holiday_dated_quote_stamps_nothing(self, monkeypatch):
        """The weekday rule never modelled holidays; the calendar does.

        15 Aug 2026 is a Saturday, so take Milan's Assumption closure of Friday
        15 Aug 2025 — a WEEKDAY the venue did not trade. A quote dated there
        belongs on no session and must not be written.
        """
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2025, 8, 15))
        h = _holding("EXUS.MI", [40.5, 40.6, 40.31], quantity=1.0,
                     start="2025-08-11")
        _quotes(monkeypatch, {"EXUS.MI": {"price": 40.4, "prev_close": 40.31}})

        assert cs.apply_to_holdings([h]) == ()


class TestAWeekendRunStillRecoversTheLastSession:
    """The Saturday digest read THURSDAY.

    Refusing to stamp at the weekend also refused a FRIDAY close that the quote
    endpoint was carrying while Yahoo's daily bars were not: on Sat 29 Aug 2026 the
    28 Aug row came back with a null close for all sixteen holdings, and every
    quote held the real Friday close — XDEQ.MI at 79.73, observed Fri 16:20,
    against a series terminating at its own prev_close of 79.11.

    On a weekday the same gap self-heals through the stamp, which is why this only
    ever surfaced on the Saturday issue. The rule is not "never at the weekend",
    it is "never onto a non-session date".
    """

    FRI = datetime.datetime(2026, 8, 28, 16, 20,
                            tzinfo=datetime.timezone.utc).timestamp()

    def _saturday(self, monkeypatch):
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 29))

    def test_a_friday_dated_quote_lands_on_friday(self, monkeypatch):
        self._saturday(monkeypatch)
        # A series that stops on Thursday 27th, exactly as the vendor left it.
        h = _holding("XDEQ.MI", [78.99, 79.12, 79.11], quantity=100.0,
                     start="2026-08-25")
        _quotes(monkeypatch, {"XDEQ.MI": {
            "price": 79.73, "prev_close": 79.11, "time": self.FRI}})

        assert cs.apply_to_holdings([h]) == ("XDEQ.MI",)
        assert h.price_history.index[-1] == pd.Timestamp("2026-08-28")
        assert float(h.price_history.iloc[-1]) == pytest.approx(79.73)
        # Thursday's settled close is untouched, so the 1D spans one session.
        assert float(h.price_history.loc[pd.Timestamp("2026-08-27")]) == \
            pytest.approx(79.11)

    def test_the_valuation_moves_with_it(self, monkeypatch):
        self._saturday(monkeypatch)
        h = _holding("XDEQ.MI", [78.99, 79.12, 79.11], quantity=100.0,
                     start="2026-08-25")
        _quotes(monkeypatch, {"XDEQ.MI": {
            "price": 79.73, "prev_close": 79.11, "time": self.FRI}})

        cs.apply_to_holdings([h])

        assert h.current_price == pytest.approx(79.73)
        assert h.current_value == pytest.approx(7973.0)
        assert h.price_is_fallback is False

    def test_the_one_day_move_becomes_fridays(self, monkeypatch):
        """The number the reader was shown on Saturday morning.

        Before: the series ended Thursday, so "1D" measured Wed→Thu. After: it
        ends Friday and measures Thu→Fri, which is what a Saturday reader means by
        the last session.
        """
        from tarzan.engine.stats import compute_period_return

        self._saturday(monkeypatch)
        h = _holding("XDEQ.MI", [78.99, 79.12, 79.11], quantity=100.0,
                     start="2026-08-25")
        before = compute_period_return(h.price_history, "1d")
        _quotes(monkeypatch, {"XDEQ.MI": {
            "price": 79.73, "prev_close": 79.11, "time": self.FRI}})

        cs.apply_to_holdings([h])
        after = compute_period_return(h.price_history, "1d")

        assert before == pytest.approx((79.11 / 79.12 - 1) * 100, abs=1e-6)
        assert after == pytest.approx((79.73 / 79.11 - 1) * 100, abs=1e-6)

    def test_a_saturday_dated_quote_is_still_refused(self, monkeypatch):
        """The guard is on the resolved DATE, not on the run day, so a provider
        that stamps a weekend instant is still rejected."""
        self._saturday(monkeypatch)
        sat = datetime.datetime(2026, 8, 29, 10, 0,
                                tzinfo=datetime.timezone.utc).timestamp()
        h = _holding("XDEQ.MI", [78.99, 79.12, 79.11], quantity=100.0,
                     start="2026-08-25")
        _quotes(monkeypatch, {"XDEQ.MI": {
            "price": 79.73, "prev_close": 79.11, "time": sat}})

        assert cs.apply_to_holdings([h]) == ()
        assert h.price_history.index[-1] == pd.Timestamp("2026-08-27")


class TestTheStampBelongsToTheObservedSession:
    """A quote is dated by the session it was observed in, not by the run day.

    Before Xetra opens on a Tuesday, ``regularMarketPrice`` is still Monday's
    closing quote and ``regularMarketTime`` says so. Dating it Tuesday moved
    every window one session forward: AVWS.DE's 5D anchored 19 Aug and read
    −0.55% where its five sessions ending on the observed one anchor 18 Aug.
    """

    def test_a_dated_quote_lands_on_its_own_session(self, monkeypatch):
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 25))   # Tuesday
        s = pd.Series(
            [25.20, 25.36],
            index=pd.DatetimeIndex([
                pd.Timestamp("2026-08-20", tz="Europe/Rome"),
                pd.Timestamp("2026-08-21", tz="Europe/Rome")]))
        observed = int(datetime.datetime(
            2026, 8, 24, 17, 35, tzinfo=datetime.timezone.utc).timestamp())
        out = cs.stamp_today(s, pd.Timestamp("2026-08-25"), 25.30,
                            {"price": 25.30, "prev_close": 25.36,
                             "time": observed}, ticker="AVWS.DE")
        assert out.index[-1].date() == datetime.date(2026, 8, 24), (
            "Monday's closing quote must not be dated Tuesday"
        )
        assert float(out.iloc[-1]) == 25.30

    def test_an_undated_quote_still_falls_back_to_today(self, monkeypatch):
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 25))
        s = pd.Series(
            [25.20, 25.36],
            index=pd.DatetimeIndex([pd.Timestamp("2026-08-20"),
                                    pd.Timestamp("2026-08-21")]))
        out = cs.stamp_today(s, pd.Timestamp("2026-08-25"), 25.30,
                            {"price": 25.30, "prev_close": 25.36},
                            ticker="AVWS.DE")
        assert out.index[-1].date() == datetime.date(2026, 8, 25)


class TestTheRunRecordsWhatDataItHeld:
    """A figure that cannot be traced to its data cannot be diagnosed.

    A 1M column read -0.69% where a recomputation from the same source said -1.61% —
    the same arithmetic on a tape one session behind. Establishing that took two
    attempts and neither succeeded, because the run that printed it kept no record of
    which close each holding's tape ended on. The run's log is retained; this puts the
    answer there.
    """

    @staticmethod
    def _holding(ticker, last_day):
        from tarzan.models.holding import Holding

        h = Holding(isin="IE00" + ticker[:8].ljust(8, "0"), ticker=ticker,
                    quantity=1.0, cost_basis_eur=100.0, market_value_eur=100.0,
                    currency="EUR")
        idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=30)
        h.price_history = pd.Series([100.0] * len(idx), index=idx)
        return h

    def test_it_names_the_holdings_that_lag(self, caplog):
        from tarzan.data import current_session as cs

        with caplog.at_level("INFO"):
            cs._log_tape_vintage([self._holding("AAA.MI", "2026-09-04"),
                                  self._holding("BBB.MI", "2026-09-02")])

        line = next(r.getMessage() for r in caplog.records
                    if "Tape vintage" in r.getMessage())
        assert "newest close 2026-09-04" in line
        assert "BBB.MI@2026-09-02" in line
        assert "AAA.MI" not in line, "a current holding is noise, not signal"

    def test_it_says_so_when_everything_is_current(self, caplog):
        from tarzan.data import current_session as cs

        with caplog.at_level("INFO"):
            cs._log_tape_vintage([self._holding("AAA.MI", "2026-09-04"),
                                  self._holding("BBB.MI", "2026-09-04")])

        line = next(r.getMessage() for r in caplog.records
                    if "Tape vintage" in r.getMessage())
        assert "all 2 holdings current to 2026-09-04" in line

    def test_a_holding_with_no_tape_is_skipped_not_crashed(self, caplog):
        from tarzan.data import current_session as cs

        h = self._holding("CCC.MI", "2026-09-04")
        h.price_history = None
        with caplog.at_level("INFO"):
            cs._log_tape_vintage([h, self._holding("AAA.MI", "2026-09-04")])

        assert any("Tape vintage" in r.getMessage() for r in caplog.records)

    def test_no_holdings_logs_nothing_rather_than_failing(self, caplog):
        from tarzan.data import current_session as cs

        with caplog.at_level("INFO"):
            cs._log_tape_vintage([])
            cs._log_tape_vintage(None)

        assert not [r for r in caplog.records if "Tape vintage" in r.getMessage()]
