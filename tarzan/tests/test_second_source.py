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


class TestASplitIsBackAdjustedByItsRoundRatio:
    """CL2.MI, 10 Oct 2023: a 1:300 split that Yahoo did not record.

    The repair used the raw close-to-close jump, 4017.15 -> 13.79 = 1:291.31. That jump is
    the split TIMES the session's real move — the 2x line also rose ~3% that day, +2.69%
    on its Paris listing — so every earlier price came out 3% high and the 5Y return ~10pp
    low. Two other sources (the Paris listing and justETF) agreed with each other and not
    with us; the second-source check is what surfaced it.
    """

    def _cl2_frame(self):
        pre = pd.bdate_range("2023-09-01", "2023-10-09")
        post = pd.bdate_range("2023-10-10", "2023-11-30")
        rng = np.random.default_rng(1)
        pre_px = 4017.15 * np.cumprod(1 + rng.normal(0, 0.006, len(pre)))[::-1] / \
            np.cumprod(1 + rng.normal(0, 0.006, len(pre)))[::-1][0]
        pre_px[-1] = 4017.15
        post_px = 13.79 * np.cumprod(1 + rng.normal(0, 0.006, len(post)))
        post_px[0] = 13.79
        idx = pre.append(post)
        return pd.DataFrame({"Close": np.concatenate([pre_px, post_px])}, index=idx)

    def test_the_factor_is_the_split_not_the_split_times_the_session(self):
        from tarzan.data.price_cache import repair_split_jumps

        raw = self._cl2_frame()
        out = repair_split_jumps(raw)
        before = out.loc["2023-10-09", "Close"]
        assert before == pytest.approx(4017.15 / 300.0, rel=1e-9)
        # ...so the split session keeps its real move instead of being erased.
        assert out.loc["2023-10-10", "Close"] / before - 1 == pytest.approx(0.0298, abs=0.0005)

    def test_a_ratio_that_is_not_round_is_not_forced_onto_one(self):
        from tarzan.data.price_cache import _snap_split_ratio

        assert _snap_split_ratio(1 / 3.4) == pytest.approx(1 / 3.4)
        assert _snap_split_ratio(1 / 291.31) == pytest.approx(1 / 300)
        assert _snap_split_ratio(2.07) == pytest.approx(2.0)

    def test_histories_repaired_the_old_way_are_refetched(self, tmp_path, monkeypatch):
        """A history is cached AFTER repair, so a repaired series has no jump left for a
        better repair to find. Without a version on the payload the 1:291.31 history
        would have kept that factor forever."""
        from tarzan.data import price_cache

        monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(price_cache, "is_enabled", lambda: True)
        frame = self._cl2_frame()
        price_cache.store_history("ABC.MI", frame)
        assert price_cache.load_history("ABC.MI") is not None
        path = price_cache._history_path("ABC.MI")
        payload = price_cache._read_json(path, "history", None)
        payload.pop("repair_version")
        price_cache._atomic_write_json(path, "history", payload)
        assert price_cache.load_history("ABC.MI") is None


class TestADisagreementIsSettledByMajority:
    """Two sources disagreeing is not a result to hand the reader. A third — the same
    instrument on another venue, over the same span — decides."""

    def _setup(self, *, third_follows):
        tape = _tape(end="2026-10-08")
        ref = _second_source_of(tape, noise=0.0)
        ref = ref[ref.index <= "2026-10-07"]
        ref[ref.index > ref.index[-25]] *= 0.95           # justETF reads 5% lower
        if third_follows == "engine":
            third = _second_source_of(tape, noise=0.0, seed=9)
        elif third_follows == "justetf":
            third = ref.copy() * 1.001     # a level offset: same returns as justETF
        else:
            # A third, different story: its RECENT path differs from both. (Scaling the
            # whole series would not do it — returns are scale-free.)
            third = _second_source_of(tape, noise=0.0, seed=9)
            third[third.index > third.index[-25]] *= 1.06
        verdict = xc.check_windows(tape, ref)["1m"]
        assert verdict["status"] == xc.DIVERGED
        return tape, verdict, third

    def test_a_third_source_with_the_engine_confirms_it(self):
        tape, v, third = self._setup(third_follows="engine")
        assert xc.settle(v, tape, [third])["status"] == xc.CONFIRMED

    def test_two_outside_sources_against_the_engine_correct_it(self):
        tape, v, third = self._setup(third_follows="justetf")
        out = xc.settle(v, tape, [third])
        assert out["status"] == xc.CORRECTED
        # The corrected figure is the agreed span x the engine's own move since the
        # shared close: only the disputed part is replaced.
        t = xc._naive_days(tape)
        tail = t.iloc[-1] / t[t.index <= pd.Timestamp(v["at"])].iloc[-1]
        assert out["corrected_value"] == pytest.approx(
            ((1 + out["agreed"] / 100) * tail - 1) * 100)
        assert abs(out["agreed"] - v["alt"]) < 0.5

    def test_three_different_stories_stay_unverified(self):
        """Nobody agrees with anybody: there is no right number to print."""
        tape, v, third = self._setup(third_follows="neither")
        assert xc.settle(v, tape, [third])["status"] == xc.DIVERGED

    def test_no_third_source_leaves_it_unverified(self):
        tape, v, _third = self._setup(third_follows="engine")
        assert xc.settle(v, tape, [])["status"] == xc.DIVERGED

    def test_the_note_discloses_a_correction_with_the_number_it_replaced(self):
        from tarzan.export.newsletter._sections_perf import _unverified_note

        note = _unverified_note(
            {"ABC.MI": {"5y": 164.93,
                        "_xc": {"5y": ("corrected", 169.46, 157.26,
                                       dt.date(2026, 10, 7), 157.26)}}},
            ["5y"])
        assert "Corrected before sending" in note
        assert "+157.26%" in note and "+164.93%" in note
        assert "Not verified" not in note


class TestANavReferee:
    def test_a_premium_drift_is_not_a_divergence_against_a_nav(self):
        """One managed-futures fund trades on Paris only, so the second source is its
        issuer NAV. Its 1Y differed by 2.4pp from the market's — the premium moving, not
        an error. Against a NAV the check catches only gross errors."""
        tape = _tape()
        ref = _second_source_of(tape, noise=0.0)
        ref[ref.index > ref.index[-260]] *= 0.98          # premium drifted ~2.4pp
        v_market = xc.check_windows(tape, ref)["1y"]
        v_nav = xc.check_windows(tape, ref, extra_pp=xc.NAV_PREMIUM_ALLOWANCE_PP)["1y"]
        assert v_market["status"] == xc.DIVERGED
        assert v_nav["status"] == xc.OK

    def test_a_gross_error_still_shows_against_a_nav(self):
        tape = _tape()
        ref = _second_source_of(tape, noise=0.0)
        ref[ref.index > ref.index[-260]] *= 0.90          # 10%: not a premium
        v = xc.check_windows(tape, ref, extra_pp=xc.NAV_PREMIUM_ALLOWANCE_PP)["1y"]
        assert v["status"] == xc.DIVERGED


class TestTheOneDayComparesTheSameSession:
    def test_an_untraded_rows_previous_session_is_not_compared_with_today(self, monkeypatch):
        """9 Oct 10:25: a thin fund untraded that morning showed Thursday's -3.45% — its
        last session, correctly — and was flagged against the second source's quote for
        FRIDAY. Two sessions are not a disagreement."""
        import tarzan.runtime as runtime
        from tarzan.data import justetf
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.holding import Holding
        from tarzan.models.investor_config import InvestorConfig

        today = dt.date(2026, 10, 9)
        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        monkeypatch.setattr(runtime, "today", lambda: today)
        monkeypatch.setattr(justetf, "prefetch", lambda *a, **k: None)
        monkeypatch.setattr(justetf, "series", lambda isin: None)
        monkeypatch.setattr(justetf, "quote", lambda isin: {
            "price": 111.5, "prev_close": 110.88, "date": today, "venue": "XETRA"})
        h = Holding(isin="IE0000000001", ticker="ABC.DE", quantity=1.0,
                    cost_basis_eur=1.0, market_value_eur=1.0, currency="EUR")
        h.price_history = _tape(end="2026-10-08")          # no point for today
        h.price_currency = "EUR"
        hp = pd.DataFrame([{"ticker": "ABC.DE", "type": "In portfolio", "1d": -3.45}])
        ctx = {"holding_performance": hp}
        MetricsEngine([h], InvestorConfig())._second_source(ctx)
        assert ctx["holding_performance"].loc[0, "xc_1d"] == xc.UNAVAILABLE
