"""How much of the book the "1D" figure actually covers.

The 1D is the NAV's own move. A holding the price vendor has not published today
is carried forward at its previous close by the series resolver, so it contributes
exactly ZERO to the numerator while its whole value stays in the denominator.
Minutes after an open that pulls the figure toward zero, and nothing said so.

Measured on the 27 Aug 2026 09:12 digest: two instruments out of sixteen had a
27 Aug tick — CL2 up 0.742% at 7.67% of the book, UEQC up 0.148% at 4.61% — and
they rendered as a portfolio "1D +0.06%". At 10:11, with 91% priced, the same book
read +0.18%. The market had not tripled; the book had opened. Worse, CL2 is a 2x
leveraged ETF and among the first to trade, so the early figure leans on the most
volatile sleeve.

The figure is not corrected — it is the real NAV move and every table in the issue
uses the same one, so substituting a different number here would split the
document. It is DISCLOSED.
"""

from __future__ import annotations

import datetime as dt
import html as H

import pandas as pd
import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import (
    _build_hero,
    _priced_today_note,
)
from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

_TODAY = dt.date(2026, 8, 27)
_SESSIONS = pd.bdate_range("2026-08-03", "2026-08-26")   # ends on the 26th


def _holding(ticker, value, *, priced_today, move_pct=0.0):
    """A holding whose series either carries a 27 Aug point or stops on the 26th."""
    closes = [100.0] * (len(_SESSIONS) - 1) + [100.0]
    series = pd.Series(closes, index=_SESSIONS, dtype=float)
    if priced_today:
        series.loc[pd.Timestamp(_TODAY)] = 100.0 * (1 + move_pct / 100.0)
    h = Holding(isin=f"IE{ticker:0<10}", ticker=ticker, quantity=1.0,
                cost_basis_eur=value, market_value_eur=value, currency="EUR")
    h.current_value = value
    h.price_history = series
    return h


def _share(holdings, *, today=_TODAY, monkeypatch=None):
    engine = MetricsEngine(list(holdings), InvestorConfig())
    from tarzan import runtime
    monkeypatch.setattr(runtime, "today", lambda: today)
    return engine._today_priced_share()


class TestCoverageIsWeightedByValue:
    def test_two_small_holdings_out_of_many_read_as_a_small_share(self, monkeypatch):
        """Counting holdings would say 2/16 = 12.5%; what matters is the money."""
        book = [_holding("BIG", 90_000.0, priced_today=False),
                _holding("A", 5_000.0, priced_today=True),
                _holding("B", 5_000.0, priced_today=True)]
        assert _share(book, monkeypatch=monkeypatch) == pytest.approx(10.0)

    def test_one_large_holding_can_carry_most_of_the_coverage(self, monkeypatch):
        book = [_holding("BIG", 90_000.0, priced_today=True),
                _holding("A", 5_000.0, priced_today=False),
                _holding("B", 5_000.0, priced_today=False)]
        assert _share(book, monkeypatch=monkeypatch) == pytest.approx(90.0)

    def test_a_fully_priced_book_is_a_hundred(self, monkeypatch):
        book = [_holding("A", 1_000.0, priced_today=True),
                _holding("B", 3_000.0, priced_today=True)]
        assert _share(book, monkeypatch=monkeypatch) == pytest.approx(100.0)

    def test_nothing_priced_today_is_zero_not_none(self, monkeypatch):
        """Zero and "not measured" are different answers and the caller
        distinguishes them."""
        book = [_holding("A", 1_000.0, priced_today=False)]
        assert _share(book, monkeypatch=monkeypatch) == pytest.approx(0.0)

    def test_an_empty_or_valueless_book_is_none(self, monkeypatch):
        assert _share([], monkeypatch=monkeypatch) is None
        assert _share([_holding("Z", 0.0, priced_today=True)],
                      monkeypatch=monkeypatch) is None

    def test_a_holding_without_a_series_counts_against_coverage(self, monkeypatch):
        """It is in the denominator — it holds value — and it cannot be in the
        numerator, because there is no observation to date."""
        blind = _holding("BLIND", 5_000.0, priced_today=False)
        blind.price_history = None
        assert _share([_holding("A", 5_000.0, priced_today=True), blind],
                      monkeypatch=monkeypatch) == pytest.approx(50.0)


class TestTheTwentySeventhOfAugust:
    """The digest that prompted this, rebuilt from its real weights and moves."""

    BOOK = [                       # ticker, value EUR, priced 27 Aug, move %
        ("NTSG.MI", 31_402, False, 0.0), ("SGLD.MI", 25_048, False, 0.0),
        ("DBMFE.PA", 24_379, False, 0.0), ("CL2.MI", 18_650, True, 0.742),
        ("XMME.MI", 18_526, False, 0.0), ("XSX6.MI", 15_415, False, 0.0),
        ("EXUS.MI", 15_410, False, 0.0), ("XDEQ.MI", 14_645, False, 0.0),
        ("XDEV.MI", 14_355, False, 0.0), ("XDEM.MI", 13_820, False, 0.0),
        ("UEQC.DE", 11_200, True, 0.148), ("MONEY.MI", 9_707, False, 0.0),
        ("XGIN.MI", 9_350, False, 0.0), ("XMJP.MI", 7_739, False, 0.0),
        ("XESC.MI", 6_924, False, 0.0), ("X15E.MI", 6_634, False, 0.0),
    ]

    def _book(self):
        return [_holding(t, float(v), priced_today=p, move_pct=m)
                for t, v, p, m in self.BOOK]

    def test_the_coverage_was_twelve_percent(self, monkeypatch):
        assert _share(self._book(), monkeypatch=monkeypatch) == pytest.approx(
            12.3, abs=0.1)

    def test_the_two_early_ticks_produce_the_published_figure(self):
        """+0.0637% against the +0.06% the subject and the tile printed.

        Not a coincidence to be admired — the point is that the whole reported
        day was two instruments, and the arithmetic says so exactly.
        """
        total = sum(v for _t, v, _p, _m in self.BOOK)
        moved = sum(v / total * m for _t, v, p, m in self.BOOK if p)
        assert moved == pytest.approx(0.0637, abs=0.0005)

    def test_the_same_book_fully_priced_reads_three_times_higher(self):
        """What the figure becomes once the rest of the book opens, holding the
        two known moves and giving the others the day's eventual average."""
        total = sum(v for _t, v, _p, _m in self.BOOK)
        partial = sum(v / total * m for _t, v, p, m in self.BOOK if p)
        assert partial < 0.10
        # The 10:11 reading of the real book was +0.18%.
        assert 0.18 / partial > 2.5


class TestTheEngineActuallyPublishesIt:
    """The seam between the measurement and the caption.

    Both halves were tested and the wiring between them was not: deleting the
    projection left every other test in this file green, which is precisely how a
    feature ships doing nothing. It is also why the projection is its own stage —
    ``_live_1d`` returns early on a provider exception and when no intraday feed
    resolves, so a line inside it would drop the coverage exactly when the run is
    degraded and the reader most needs to know the figure is partial.
    """

    def _ctx(self, book, monkeypatch):
        from tarzan import runtime
        monkeypatch.setattr(runtime, "today", lambda: _TODAY)
        engine = MetricsEngine(list(book), InvestorConfig())
        ctx = {"performance": {"1d": 0.06}, "performance_full": {"1d": 0.06}}
        engine._session_coverage(ctx)
        return ctx

    def test_the_key_reaches_both_projections(self, monkeypatch):
        book = [_holding("A", 5_000.0, priced_today=True),
                _holding("B", 15_000.0, priced_today=False)]
        ctx = self._ctx(book, monkeypatch)
        assert ctx["performance"]["1d_coverage_pct"] == pytest.approx(25.0)
        assert ctx["performance_full"]["1d_coverage_pct"] == pytest.approx(25.0)

    def test_the_stage_is_registered_in_the_pipeline(self):
        engine = MetricsEngine([], InvestorConfig())
        names = [c.__name__ for c in engine._computers]
        assert "_session_coverage" in names
        # After the intraday work, so a stamped series is already in place.
        assert names.index("_session_coverage") > names.index("_live_1d")

    def test_it_survives_a_missing_projection(self, monkeypatch):
        from tarzan import runtime
        monkeypatch.setattr(runtime, "today", lambda: _TODAY)
        MetricsEngine([], InvestorConfig())._session_coverage({})   # must not raise


class TestTheNoteSaysWhatIsCovered:
    def test_a_partial_live_figure_is_disclosed(self):
        note = _priced_today_note({"1d_live": True, "1d_coverage_pct": 12.3})
        assert note == "12% of the book priced today"

    def test_full_coverage_says_nothing(self):
        """At 100% the note is noise; the figure needs no qualification."""
        assert _priced_today_note({"1d_live": True, "1d_coverage_pct": 100.0}) == ""
        assert _priced_today_note({"1d_live": True, "1d_coverage_pct": 99.6}) == ""

    def test_a_completed_session_says_nothing(self):
        """``_session_basis`` already names it "close-to-close vs <date>", and a
        completed session is priced by definition."""
        assert _priced_today_note({"1d_live": False, "1d_coverage_pct": 12.3}) == ""

    def test_an_unmeasured_coverage_says_nothing(self):
        assert _priced_today_note({"1d_live": True}) == ""
        assert _priced_today_note({"1d_live": True, "1d_coverage_pct": None}) == ""
        assert _priced_today_note(None) == ""

    def test_zero_coverage_is_still_disclosed(self):
        assert _priced_today_note({"1d_live": True, "1d_coverage_pct": 0.0}) == \
            "0% of the book priced today"


class TestTheSessionTileCarriesIt:
    @staticmethod
    def _session_tile(perf):
        m = PortfolioMetrics(total_value=243_205.0, invested_value=243_205.0,
                             holdings_df=pd.DataFrame([{"cost_basis_eur": 220_000.0}]))
        m.performance = perf
        tiles = _build_hero(_NewsletterContext(
            metrics=m, config=InvestorConfig()))["tiles"]
        return next(t for t in tiles if t["label"] == "Session")

    def test_the_caption_states_the_coverage(self):
        tile = self._session_tile(
            {"1d": 0.0637, "1d_live": True, "market_open": True,
             "1d_coverage_pct": 12.3})
        caption = H.unescape(tile["caption"])
        assert "market open" in caption
        assert caption.endswith("12% of the book priced today"), caption

    def test_a_fully_priced_session_reads_as_before(self):
        tile = self._session_tile(
            {"1d": 0.18, "1d_live": True, "market_open": True,
             "1d_coverage_pct": 100.0})
        caption = H.unescape(tile["caption"])
        assert "priced today" not in caption
        assert "market open" in caption

    def test_the_value_still_leads_the_caption(self):
        """The euro amount keeps its place; the note is appended, not swapped in."""
        tile = self._session_tile(
            {"1d": 0.0637, "1d_live": True, "market_open": True,
             "1d_coverage_pct": 12.3})
        assert H.unescape(tile["caption"]).startswith("+€")
