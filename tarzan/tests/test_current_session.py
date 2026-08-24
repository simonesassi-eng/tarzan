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

    def test_a_weekend_run_stamps_nothing(self, monkeypatch):
        # Sat 22 Aug 2026: no live session, and a weekend-dated point would
        # slide window_anchor onto the day a month before the WEEKEND.
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 22))
        h = _holding("EXUS.MI", [40.5, 40.6, 40.31], quantity=1.0)
        _quotes(monkeypatch, {"EXUS.MI": {"price": 40.4, "prev_close": 40.31}})

        assert cs.apply_to_holdings([h]) == ()
        assert h.price_history.index[-1] != pd.Timestamp("2026-08-22")
