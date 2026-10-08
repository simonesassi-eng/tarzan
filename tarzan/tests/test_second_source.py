"""The second, independent source: every printed holding return checked against it.

Network-free. The referee's arithmetic is in ``tarzan.engine.cross_check``; its wiring
into the issue is in ``MetricsEngine._second_source`` and the Returns table.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import cross_check as xc


def _tape(n=1400, seed=3, drift=0.0004, vol=0.009, end="2026-10-07"):
    """Business-day closes, the shape of an engine tape (~5.5 years)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)


def _second_source_of(tape, noise=0.003, seed=5):
    """The same instrument seen on another venue: calendar-daily (the second source
    fills weekends), levels within a few tenths of a percent."""
    rng = np.random.default_rng(seed)
    cal = tape.reindex(pd.date_range(tape.index[0], tape.index[-1], freq="D")).ffill()
    return cal * (1 + rng.normal(0, noise, len(cal)))


class TestTheLongWindows:
    def test_two_honest_venues_agree_on_every_window(self):
        tape = _tape()
        v = xc.check_windows(tape, _second_source_of(tape))
        assert {w: v[w]["status"] for w in xc.LONG_WINDOWS} == \
            {w: xc.OK for w in xc.LONG_WINDOWS}

    def test_a_drifted_source_is_a_divergence(self):
        """The thin fund of 8 Oct 2026 in shape: one source's recent closes run 4%
        away from the other's, so 1M/3M disagree by about that much."""
        tape = _tape()
        ref = _second_source_of(tape, noise=0.0)
        ref[ref.index > ref.index[-25]] *= 0.96
        v = xc.check_windows(tape, ref)
        assert v["1m"]["status"] == xc.DIVERGED
        assert v["1m"]["alt"] == pytest.approx(v["1m"]["ours"] - 4.0, abs=0.6)

    def test_both_sides_end_on_the_same_close(self):
        """The engine tape runs to today; the second source stops at yesterday.
        Comparing them on different end dates manufactured a 2pp 1Y "gap" that was
        only the clock, so both are cut at the last close they share."""
        tape = _tape(end="2026-10-08")
        ref = _second_source_of(tape, noise=0.0)
        ref = ref[ref.index <= "2026-10-07"]
        tape.iloc[-1] *= 0.97                     # today moved; yesterday did not
        v = xc.check_windows(tape, ref)
        assert v["1y"]["at"] == dt.date(2026, 10, 7)
        assert v["1y"]["status"] == xc.OK

    def test_the_tolerance_widens_with_the_size_of_the_return(self):
        """Venue noise is multiplicative: 0.5% of level at each end is ~1pp on a
        +100% five-year figure and ~0.5pp on a flat month. Measured on the reference
        book: the honest gap never exceeded 1.20 x (1 + |r|)."""
        tape = _tape(drift=0.0009)                # a strong five years, ~+200%
        ref = _second_source_of(tape, noise=0.0)
        ref.iloc[-1] *= 1.012                     # 1.2% at the end: ~3.6pp on 5Y
        v = xc.check_windows(tape, ref)
        assert v["5y"]["ours"] > 100
        assert v["5y"]["status"] == xc.OK

    def test_no_second_source_is_not_a_verdict(self):
        v = xc.check_windows(_tape(), None)
        assert all(v[w]["status"] == xc.UNAVAILABLE for w in xc.LONG_WINDOWS)

    def test_a_second_source_too_young_for_the_window_says_so(self):
        tape = _tape()
        ref = _second_source_of(tape)
        ref = ref[ref.index >= ref.index[-400]]   # ~13 months of second source
        v = xc.check_windows(tape, ref)
        assert v["1y"]["status"] == xc.OK
        assert v["3y"]["status"] == xc.UNAVAILABLE
        assert v["5y"]["status"] == xc.UNAVAILABLE


class TestTheOneDay:
    TODAY = dt.date(2026, 10, 8)

    def test_agreeing_quotes(self):
        q = {"price": 99.4, "prev_close": 100.0, "date": self.TODAY}
        assert xc.check_one_day(-0.55, q, self.TODAY)["status"] == xc.OK

    def test_the_8_oct_figure_against_the_second_source(self):
        """-4.38% printed; the second source's Xetra pair for the session said -0.72%."""
        q = {"price": 74.37, "prev_close": 74.91, "date": self.TODAY}
        v = xc.check_one_day(-4.38, q, self.TODAY, _tape())
        assert v["status"] == xc.DIVERGED

    def test_yesterdays_quote_cannot_referee_today(self):
        q = {"price": 99.4, "prev_close": 100.0, "date": self.TODAY - dt.timedelta(days=1)}
        assert xc.check_one_day(-4.38, q, self.TODAY)["status"] == xc.UNAVAILABLE


class TestTheIssueMarksWhatDisagrees:
    def test_the_note_names_both_figures(self):
        from tarzan.export.newsletter._sections_perf import _unverified_note

        note = _unverified_note(
            {"ABC.DE": {"1m": 9.80,
                        "_xc": {"1m": ("diverged", 5.29, 9.44, dt.date(2026, 10, 7)),
                                "1y": ("ok", 12.0, 12.1, dt.date(2026, 10, 7))}}},
            ["1d", "1m", "1y"])
        assert "ABC 1M to 07 Oct" in note
        # The like-for-like pair (to the shared close), NOT the printed +9.80%,
        # which runs to today and would show a gap that is only the clock.
        assert "Yahoo +9.44%" in note and "justETF +5.29%" in note
        assert "9.80" not in note
        assert "1Y" not in note                   # agreeing figures are not listed

    def test_nothing_diverged_means_no_note(self):
        from tarzan.export.newsletter._sections_perf import _unverified_note

        assert _unverified_note({"ABC.DE": {"_xc": {"1m": ("ok", 1.0)}}}, ["1m"]) == ""


class TestTheEngineStage:
    def test_a_pinned_run_never_reaches_the_network_and_marks_nothing(self, monkeypatch):
        import tarzan.runtime as runtime
        from tarzan.data import justetf
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)
        monkeypatch.setattr(justetf, "prefetch",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        hp = pd.DataFrame([{"ticker": "ABC.DE", "type": "In portfolio", "1d": 0.1}])
        ctx = {"holding_performance": hp}
        MetricsEngine([], InvestorConfig())._second_source(ctx)
        assert (ctx["holding_performance"]["xc_1m"] == xc.UNAVAILABLE).all()

    def test_a_failing_source_never_fails_the_run(self, monkeypatch):
        import tarzan.runtime as runtime
        from tarzan.data import justetf
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.holding import Holding
        from tarzan.models.investor_config import InvestorConfig

        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        monkeypatch.setattr(justetf, "prefetch", lambda *a, **k: None)

        def boom(_isin):
            raise RuntimeError("justETF down")

        monkeypatch.setattr(justetf, "series", boom)
        h = Holding(isin="IE0000000001", ticker="ABC.DE", quantity=1.0,
                    cost_basis_eur=1.0, market_value_eur=1.0, currency="EUR")
        h.price_history = _tape()
        hp = pd.DataFrame([{"ticker": "ABC.DE", "type": "In portfolio", "1d": 0.1}])
        ctx = {"holding_performance": hp}
        MetricsEngine([h], InvestorConfig())._second_source(ctx)   # must not raise
        assert "xc_1m" in ctx["holding_performance"].columns
