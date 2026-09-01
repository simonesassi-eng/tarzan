"""The standard windows as CHART windows, not only as table columns.

The newsletter's "Vs target & benchmark" section draws one panel per standard
window (1D / 5D / 1M / 3M / YTD / 1Y). Two things had to exist for that:

*   a YTD *anchor*. YTD was already a column — ``compute_ytd_return`` computes it
    and the RETURNS tables print it — but ``window_anchor`` had no ``ytd`` branch,
    so nothing could open a WINDOW on it. The risk in adding one is a second
    convention: a chart anchored on 1 Jan beside a column anchored on 31 Dec, both
    labelled YTD. These tests pin them to one number.
*   a ``bucket`` on ``_perf_window``, which had "1m" written into it. The risk
    there is the 30-day FALLBACK: for a book that does not reach back a year,
    falling back would draw one month and label it 1Y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.stats import (
    compute_period_return, compute_ytd_return, window_anchor,
)


def _ramp(start: str, end: str, lo: float = 100.0, hi: float = 130.0):
    idx = pd.date_range(start, end, freq="B")
    return pd.Series(np.linspace(lo, hi, len(idx)), index=idx)


class TestTheYtdAnchorAgreesWithTheYtdColumn:
    """One convention, checked as an identity rather than by inspection.

    ``compute_period_return(s, "ytd")`` goes through the new anchor;
    ``compute_ytd_return(s)`` is the function the tables have always used. They
    must return the same float for every shape, or a YTD chart and the YTD column
    beside it describe different periods.
    """

    def test_with_prior_year_data_it_anchors_on_the_last_close_of_december(self):
        s = _ramp("2025-11-03", "2026-08-26")
        assert window_anchor(s, "ytd") == pd.Timestamp("2025-12-31")
        assert compute_period_return(s, "ytd") == pytest.approx(
            compute_ytd_return(s), abs=1e-12)

    def test_a_mid_year_inception_falls_back_to_its_first_in_year_close(self):
        """The book was born in March; it has no December close to measure from.
        ``compute_ytd_return`` uses the first in-year point, so the anchor must
        too — returning None here would blank a column that already prints."""
        s = _ramp("2026-03-02", "2026-08-26")
        assert window_anchor(s, "ytd") == s.index[0]
        assert compute_period_return(s, "ytd") == pytest.approx(
            compute_ytd_return(s), abs=1e-12)

    def test_a_single_in_year_close_is_not_a_window(self):
        s = pd.Series([100.0, 101.0], index=pd.DatetimeIndex(
            ["2026-06-02", "2026-06-03"]))
        # One in-year point and no prior year: nothing to measure across.
        one = pd.Series([100.0], index=pd.DatetimeIndex(["2026-01-02"]))
        joined = pd.concat([s.rename(None), one])
        joined.index = pd.DatetimeIndex(["2025-06-02", "2025-06-03", "2026-01-02"])
        assert compute_period_return(joined, "ytd") == pytest.approx(
            compute_ytd_return(joined), abs=1e-12)

    def test_new_years_saturday_does_not_anchor_in_the_year_before_last(self):
        """The trap the trailing buckets set for this one.

        Every other bucket rolls the series end back to the last SESSION first,
        which on Saturday 2 Jan lands on 31 Dec of the OLD year — and taking THAT
        year's "prior year" would anchor YTD twelve months too early. The year has
        to come from the series' own last observation.
        """
        idx = list(pd.date_range("2026-11-02", "2026-12-31", freq="B"))
        idx.append(pd.Timestamp("2027-01-02"))          # a Saturday
        s = pd.Series(np.linspace(100, 130, len(idx)), index=pd.DatetimeIndex(idx))

        anchor = window_anchor(s, "ytd")

        assert anchor == pd.Timestamp("2026-12-31"), anchor
        assert compute_period_return(s, "ytd") == pytest.approx(
            compute_ytd_return(s), abs=1e-12)

    def test_a_series_entirely_in_a_past_year_has_no_ytd(self):
        s = _ramp("2025-02-03", "2025-11-28")
        # Last observation is in 2025, so 2025 IS the current year for this
        # series and its prior year is empty -> the first in-year close.
        assert window_anchor(s, "ytd") == s.index[0]


class TestAWindowIsNeverASubstituteForAnother:
    """``_perf_window``'s 30-day fallback, scoped to the window it belongs to."""

    @staticmethod
    def _metrics(start: str, end: str):
        from tarzan.models.portfolio import PortfolioMetrics

        idx = pd.date_range(start, end, freq="B")
        n = len(idx)
        wob = 1 + 0.01 * np.sin(np.arange(n) / 5)
        m = PortfolioMetrics(total_value=6000.0, invested_value=6000.0,
                             holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, n) * wob, index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, n) * wob, index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, n) * wob, index=idx)
        m.benchmark_histories = {"B": pd.Series(
            np.linspace(200, 230, n) * wob, index=idx)}
        return m

    def test_a_short_book_gets_no_one_year_window_rather_than_a_short_one(self):
        """Eight months of history cannot answer a 1Y question. Before ``bucket``
        existed the fallback would have opened 30 days and the panel would have
        been captioned 1Y — a month of data under a year's name."""
        from tarzan.export._perf_series import _perf_window

        m = self._metrics("2026-01-02", "2026-08-26")

        assert _perf_window(m, 30, "B", bucket="1y") is None

    def test_a_long_book_does_get_one_and_it_spans_about_a_year(self):
        from tarzan.export._perf_series import _perf_window

        m = self._metrics("2023-01-02", "2026-08-26")
        win = _perf_window(m, 30, "B", bucket="1y")

        assert win is not None
        span = (win["window_end"] - win["window_start"]).days
        assert 360 <= span <= 372, span

    def test_the_default_window_keeps_its_fallback(self):
        """The fallback is not deleted, only scoped: the 1M window still opens on
        a 30-day slice when no anchor resolves, which is the behaviour every
        existing caller and the semantic gate were built against."""
        from tarzan.export._perf_series import _perf_window

        m = self._metrics("2026-07-01", "2026-08-26")
        # Force the anchor to fail while leaving the series usable.
        m.benchmark_histories = {}

        win = _perf_window(m, 30, "B")

        assert win is not None
        assert win["twror"] is not None

    def test_each_bucket_opens_on_its_own_anchor(self):
        """Distinct buckets must produce distinct windows — a ``bucket`` argument
        that were silently ignored would pass every test above."""
        from tarzan.export._perf_series import _perf_window

        m = self._metrics("2023-01-02", "2026-08-26")
        starts = {}
        for b in ("5d", "1m", "3m", "ytd", "1y"):
            win = _perf_window(m, 30, "B", bucket=b)
            assert win is not None, b
            starts[b] = win["window_start"]

        assert len(set(starts.values())) == 5, starts
        # And they are ordered: a longer window opens earlier.
        ordered = [starts[b] for b in ("1y", "ytd", "3m", "1m", "5d")]
        assert ordered == sorted(ordered), starts


class TestTheVolatilityPanelSharesTheReturnPanelsWindow:
    """Both grids on one vocabulary — the reason ``bucket`` reached the vol
    builder at all. A calendar ``n_days`` slice beside a session-anchored return
    window put the two panels of one row on different spans."""

    @staticmethod
    def _metrics():
        return TestAWindowIsNeverASubstituteForAnother._metrics(
            "2023-01-02", "2026-08-26")

    def test_the_two_panels_of_a_row_open_on_the_same_observation(self):
        from tarzan.export._perf_series import _perf_vol_series, _perf_window

        m = self._metrics()
        for b in ("1m", "3m", "ytd", "1y"):
            ret = _perf_window(m, 30, "B", bucket=b)
            vol = _perf_vol_series(m, "B", bucket=b)
            assert ret is not None and vol is not None, b
            assert vol["dates"][0] == ret["window_start"], (
                f"{b}: volatility opens {vol['dates'][0]}, return opens "
                f"{ret['window_start']}")

    def test_a_bucket_the_book_cannot_reach_yields_no_volatility_panel(self):
        from tarzan.export._perf_series import _perf_vol_series

        m = TestAWindowIsNeverASubstituteForAnother._metrics(
            "2026-01-02", "2026-08-26")

        assert _perf_vol_series(m, "B", bucket="1y") is None

    def test_the_since_inception_panel_still_spans_the_whole_life(self):
        """``bucket=None`` is untouched: it is what the lifetime panel passes."""
        from tarzan.export._perf_series import _perf_vol_series

        m = self._metrics()
        vol = _perf_vol_series(m, "B", n_days=None)

        assert vol is not None
        assert vol["dates"][-1] == m.portfolio_history.index[-1]


class TestTheWindowSigmaIsNotTheLinesEndValue:
    """Why the caption carries a figure at all.

    The plotted line is a ROLLING 21-session σ, so its last point is today's σ —
    the same number in every window, because every window ends today. Six panels
    labelled 6.47% differ in shape and in nothing a reader can quote. The caption
    states σ measured over the window's OWN returns, which does differ.
    """

    @staticmethod
    def _metrics():
        return TestAWindowIsNeverASubstituteForAnother._metrics(
            "2023-01-02", "2026-08-26")

    def test_every_line_ends_on_the_same_rolling_figure(self):
        """The premise. If this ever stops holding, the caption is redundant."""
        from tarzan.export._perf_series import _perf_vol_series

        ends = []
        m = self._metrics()
        for b in ("1m", "3m", "ytd", "1y"):
            vs = _perf_vol_series(m, "B", bucket=b)
            assert vs is not None and vs["port"], b
            ends.append(round(float(vs["port"][-1]), 9))

        assert len(set(ends)) == 1, ends

    def test_the_window_sigma_differs_between_windows(self):
        from tarzan.export._perf_series import _perf_vol_series

        m = self._metrics()
        sigmas = {b: _perf_vol_series(m, "B", bucket=b)["window_sigma"]["port"]
                  for b in ("1m", "3m", "ytd", "1y")}

        assert all(v is not None for v in sigmas.values()), sigmas
        assert len(set(round(v, 9) for v in sigmas.values())) > 1, sigmas

    def test_the_one_year_window_sigma_matches_a_hand_computation(self):
        """Pinned against the estimator's own definition rather than a literal, so
        the test states the arithmetic instead of a number nobody can check."""
        import numpy as np
        from tarzan.engine.stats import TRADING_DAYS
        from tarzan.export._perf_series import _perf_vol_series, _perf_window

        m = self._metrics()
        win = _perf_window(m, 30, "B", bucket="1y")
        vs = _perf_vol_series(m, "B", bucket="1y")

        nav = m.portfolio_history
        clip = nav[(nav.index >= win["window_start"])
                   & (nav.index <= win["window_end"])]
        expected = float(clip.pct_change().dropna().std(ddof=1)
                         * np.sqrt(TRADING_DAYS) * 100.0)

        assert vs["window_sigma"]["port"] == pytest.approx(expected, rel=1e-9)

    def test_a_five_day_window_still_gets_a_figure(self):
        """Five returns is a noisy σ and the floor allows it deliberately: blanking
        the caption on the short windows would blank it on the panels a reader
        checks after a bad week, and the caption names the window it covers."""
        from tarzan.export._perf_series import _perf_vol_series

        vs = _perf_vol_series(self._metrics(), "B", bucket="5d")

        assert vs is not None
        assert vs["window_sigma"]["port"] is not None

    def test_the_caption_prints_it_beside_the_window_name(self):
        import re

        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig

        sec = _build_performance30(_NewsletterContext(
            metrics=self._metrics(), config=InvestorConfig(), benchmark_geo="B"))
        html = sec["vs_market_html"]
        vol_part = html.split("Volatility · by window", 1)[1].split(
            "Return · since inception", 1)[0]
        sigmas = re.findall(r'letter-spacing:0;"> · σ ([\d.]+)%</span>', vol_part)

        assert len(sigmas) >= 4, sigmas
        assert len(set(sigmas)) > 1, f"every window printed the same σ: {sigmas}"


class TestTheOneDayCellDrawsTheSession:
    """1D comes from INTRADAY bars, not from two daily closes.

    Preprocessing already keeps a quote catalog of intraday series — the RETURNS
    table's 1D sparklines draw it — and ``_portfolio_intraday_series`` already
    value-weights the holdings' bars into one portfolio path. The cell reuses both,
    so it costs no fetch and cannot disagree with the 1D column.
    """

    @staticmethod
    def _metrics(*, with_intraday: bool):
        import numpy as np
        from tarzan.models.portfolio import PortfolioMetrics

        idx = pd.date_range("2025-08-01", "2026-08-26", freq="B")
        n = len(idx)
        wob = 1 + 0.01 * np.sin(np.arange(n) / 5)
        m = PortfolioMetrics(
            total_value=6000.0, invested_value=6000.0,
            holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0,
                                       "ticker": "AAA.MI", "weight_pct": 60.0},
                                      {"cost_basis_eur": 3000.0,
                                       "ticker": "BBB.MI", "weight_pct": 40.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, n) * wob, index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, n) * wob, index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, n) * wob, index=idx)
        m.benchmark_histories = {"B": pd.Series(
            np.linspace(200, 230, n) * wob, index=idx)}
        m.benchmark_tickers = {"B": "BENCH.MI"}
        if with_intraday:
            stamps = pd.date_range("2026-08-26 09:05", periods=8, freq="1h",
                                   tz="Europe/Rome")
            def _bars(base, drift):
                # ``intraday_series`` / ``intraday_baseline`` — the shape
                # ``_intraday_quote_parts`` reads, i.e. what MetricsEngine._live_1d
                # emits. A bare series is also accepted but carries no baseline.
                return {"intraday_series": pd.Series(
                    base * (1 + drift * np.arange(len(stamps)) / 100.0),
                    index=stamps), "intraday_baseline": base}
            m.intraday_quotes = {"AAA.MI": _bars(100.0, 0.10),
                                 "BBB.MI": _bars(50.0, -0.05),
                                 "BENCH.MI": _bars(210.0, 0.04)}
        return m

    @staticmethod
    def _cell(m):

        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig

        html = _build_performance30(_NewsletterContext(
            metrics=m, config=InvestorConfig(),
            benchmark_geo="B"))["vs_market_html"]
        # The 1D cell is the first cell of the return grid.
        after = html.split("Return · by window", 1)[1]
        return after.split("5D", 1)[0]

    def test_with_bars_the_cell_is_a_chart_on_a_clock_axis(self):
        cell = self._cell(self._metrics(with_intraday=True))

        assert "<svg" in cell, cell[:300]
        # Times, not dates: the x axis of one session is a clock.
        assert ">09:05<" in cell or ">10:05<" in cell, cell[-600:]

    def test_both_the_portfolio_and_the_benchmark_are_drawn(self):
        from tarzan.export._palette import PALETTE

        cell = self._cell(self._metrics(with_intraday=True))

        assert f'stroke="{PALETTE["port"]}"' in cell
        assert f'stroke="{PALETTE["bench"]}"' in cell

    def test_the_portfolio_path_is_value_weighted_not_the_first_holding(self):
        """60% at +0.10%/bar and 40% at −0.05%/bar is +0.04%/bar, not +0.10%."""
        m = self._metrics(with_intraday=True)
        from tarzan.export.newsletter._sections_perf import (
            _portfolio_intraday_series,
        )

        pf = _portfolio_intraday_series(m)

        assert pf is not None
        # Seven bars after the first, at 0.6*0.10 + 0.4*(-0.05) = +0.04 each.
        assert float(pf.iloc[-1]) - 100.0 == pytest.approx(7 * 0.04, rel=1e-6)

    def test_without_bars_the_cell_states_the_figures_instead(self):
        """A pinned run, an offline one, or a market whose vendor exposes no
        session. The 1D figures still exist, so the cell prints them rather than
        going blank — and says which basis they are on."""
        cell = self._cell(self._metrics(with_intraday=False))

        assert "<svg" not in cell
        assert "session · closes" in cell
        assert ">TWROR<" in cell


class TestTheTargetGetsItsSessionToo:
    """The 1D target line, weighted by the allocation's own policy.

    ``metrics.target_weights`` carries every instrument the target names — seeds
    included — before any availability filtering, because whether the target has a
    daily HISTORY and whether it has a SESSION are different questions. The line is
    drawn only on FULL coverage: a blend over the sleeves that happened to trade is
    a different portfolio wearing the target's name.
    """

    @staticmethod
    def _metrics(*, target_weights, quoted):
        """``quoted`` names the tickers that have intraday bars."""
        import numpy as np
        from tarzan.models.portfolio import PortfolioMetrics

        idx = pd.date_range("2025-08-01", "2026-08-26", freq="B")
        n = len(idx)
        wob = 1 + 0.01 * np.sin(np.arange(n) / 5)
        m = PortfolioMetrics(
            total_value=6000.0, invested_value=6000.0,
            holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0,
                                       "ticker": "AAA.MI", "weight_pct": 100.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, n) * wob, index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, n) * wob, index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, n) * wob, index=idx)
        m.benchmark_histories = {"B": pd.Series(
            np.linspace(200, 230, n) * wob, index=idx)}
        m.benchmark_tickers = {"B": "BENCH.MI"}
        m.target_weights = dict(target_weights)

        stamps = pd.date_range("2026-08-26 09:05", periods=6, freq="1h",
                               tz="Europe/Rome")

        def _bars(base, drift):
            return {"intraday_series": pd.Series(
                base * (1 + drift * np.arange(len(stamps)) / 100.0),
                index=stamps), "intraday_baseline": base}

        drifts = {"AAA.MI": 0.10, "BBB.MI": -0.05, "SEED.MI": 0.20,
                  "BENCH.MI": 0.04}
        m.intraday_quotes = {t: _bars(100.0, drifts.get(t, 0.01))
                             for t in quoted}
        return m

    def _cell(self, m):
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig

        html = _build_performance30(_NewsletterContext(
            metrics=m, config=InvestorConfig(),
            benchmark_geo="B"))["vs_market_html"]
        return html.split("Return · by window", 1)[1].split("5D", 1)[0]

    def test_the_target_is_drawn_when_every_sleeve_has_bars(self):
        from tarzan.export._palette import PALETTE

        cell = self._cell(self._metrics(
            target_weights={"AAA.MI": 60.0, "BBB.MI": 40.0},
            quoted=("AAA.MI", "BBB.MI", "BENCH.MI")))

        assert f'stroke="{PALETTE["target"]}"' in cell, cell[:400]

    def test_a_seed_not_yet_owned_still_counts_toward_the_target(self):
        """Nearly a quarter of a real target can sit in instruments not owned yet.
        They are in ``target_weights``, so their bars are required — and used."""
        from tarzan.export._palette import PALETTE

        weights = {"AAA.MI": 50.0, "SEED.MI": 50.0}
        with_seed = self._cell(self._metrics(
            target_weights=weights, quoted=("AAA.MI", "SEED.MI", "BENCH.MI")))
        without = self._cell(self._metrics(
            target_weights=weights, quoted=("AAA.MI", "BENCH.MI")))

        assert f'stroke="{PALETTE["target"]}"' in with_seed
        assert f'stroke="{PALETTE["target"]}"' not in without, (
            "a target missing half its weight was drawn anyway")

    def test_partial_coverage_draws_no_target_rather_than_a_renormalised_one(self):
        from tarzan.export._palette import PALETTE

        cell = self._cell(self._metrics(
            target_weights={"AAA.MI": 60.0, "BBB.MI": 40.0},
            quoted=("AAA.MI", "BENCH.MI")))          # BBB has no session

        assert f'stroke="{PALETTE["target"]}"' not in cell
        # ...and the portfolio and benchmark still are: one missing reference does
        # not cost the cell its other lines.
        assert f'stroke="{PALETTE["port"]}"' in cell
        assert f'stroke="{PALETTE["bench"]}"' in cell

    def test_the_blend_is_weighted_not_averaged(self):
        """60% at +0.10%/bar and 40% at −0.05%/bar is +0.04%/bar. A plain mean
        would give +0.025%, and equal weights would hide the error."""

        m = self._metrics(target_weights={"AAA.MI": 60.0, "BBB.MI": 40.0},
                          quoted=("AAA.MI", "BBB.MI", "BENCH.MI"))
        cell = self._cell(m)

        # Five bars after the first at +0.04 each.
        assert f'{5 * 0.04:+.2f}%' in cell or "+0.20%" in cell, cell[-700:]

    def test_weights_over_a_sleeve_are_not_forced_to_sum_to_one_hundred(self):
        """A target may be stated over part of the book. The blend divides by the
        weight it carries, so 30/20 behaves like 60/40 — not like 30/20/50-cash."""
        from tarzan.export._palette import PALETTE

        cell = self._cell(self._metrics(
            target_weights={"AAA.MI": 30.0, "BBB.MI": 20.0},
            quoted=("AAA.MI", "BBB.MI", "BENCH.MI")))

        assert f'stroke="{PALETTE["target"]}"' in cell
        assert "+0.20%" in cell, cell[-700:]

    def test_no_target_at_all_is_simply_no_line(self):
        from tarzan.export._palette import PALETTE

        cell = self._cell(self._metrics(target_weights={},
                                        quoted=("AAA.MI", "BENCH.MI")))

        assert f'stroke="{PALETTE["target"]}"' not in cell
        assert f'stroke="{PALETTE["port"]}"' in cell


class TestTheTargetPolicyIsExportedEvenWhenItsHistoryIsNot:
    """``target_weights`` must survive the path that withholds ``target_history``.

    A sleeve with no price history withholds the daily target line — correctly.
    Coupling the POLICY to that verdict would have silently removed the 1D target
    line too, for a reason that has nothing to do with intraday bars.
    """

    def test_the_weights_are_populated_where_the_history_is_withheld(self):
        import numpy as np
        from tarzan.engine.metrics import MetricsEngine

        idx = pd.date_range("2026-06-01", "2026-08-26", freq="B")

        class _H:
            def __init__(self, ticker, isin, weight, history):
                self.ticker, self.isin = ticker, isin
                self.target_portfolio, self.price_history = weight, history

        good = _H("AAA.MI", "IE00AAA", 60.0,
                  pd.Series(np.linspace(100, 110, len(idx)), index=idx))
        blind = _H("BBB.MI", "IE00BBB", 40.0, None)       # no history at all

        engine = MetricsEngine.__new__(MetricsEngine)
        engine.holdings = [good, blind]
        engine.rebalance_seeds = []
        ctx: dict = {}
        engine._target_history(ctx)

        assert ctx.get("target_history") is None, "history must be withheld"
        assert ctx.get("target_weights") == {"AAA.MI": 60.0, "BBB.MI": 40.0}


class TestTheGateVerifiesTheOneDayPanelToo:
    """1D used to be the one drawn panel the semantic gate could not see.

    Its window does not come from ``_perf_window``, so the per-window loop skipped
    it: the three end labels were correct by construction (read off the plotted
    array) but nothing checked them against an independent recomputation, which is
    the property the gate exists to enforce for every other line on the page.

    ``_perf_intraday_window`` now returns the same shape ``_perf_window`` does, so
    the gate recomputes 1D through the identical code path. These tests drive the
    gate and then BREAK the audit, because a check that never fails is not a check.
    """

    @staticmethod
    def _render(monkeypatch, *, with_intraday=True, weights=None):
        import numpy as np

        from tarzan import config as cfg
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig
        from tarzan.models.portfolio import PortfolioMetrics

        # The renderer reads ctx.benchmark_geo; the gate reads config. In production
        # they are one value, so the test makes them agree.
        monkeypatch.setattr(cfg, "benchmark_geo_allocation", lambda: "B")

        idx = pd.date_range("2025-08-01", "2026-08-26", freq="B")
        n = len(idx)
        wob = 1 + 0.01 * np.sin(np.arange(n) / 5)
        m = PortfolioMetrics(
            total_value=6000.0, invested_value=6000.0,
            holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0,
                                       "ticker": "AAA.MI", "weight_pct": 60.0},
                                      {"cost_basis_eur": 3000.0,
                                       "ticker": "BBB.MI", "weight_pct": 40.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, n) * wob, index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, n) * wob, index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, n) * wob, index=idx)
        m.benchmark_histories = {"B": pd.Series(
            np.linspace(200, 230, n) * wob, index=idx)}
        m.benchmark_tickers = {"B": "BENCH.MI"}
        m.target_weights = dict(weights or {"AAA.MI": 60.0, "BBB.MI": 40.0})
        if with_intraday:
            stamps = pd.date_range("2026-08-26 09:05", periods=7, freq="1h",
                                   tz="Europe/Rome")

            def _bars(base, drift):
                return {"intraday_series": pd.Series(
                    base * (1 + drift * np.arange(len(stamps)) / 100.0),
                    index=stamps), "intraday_baseline": base}

            m.intraday_quotes = {"AAA.MI": _bars(100.0, 0.10),
                                 "BBB.MI": _bars(50.0, -0.05),
                                 "BENCH.MI": _bars(210.0, 0.04)}
        audit: dict = {}
        html = _build_performance30(_NewsletterContext(
            metrics=m, config=InvestorConfig(), benchmark_geo="B",
            semantic_audit=audit))["vs_market_html"]
        return m, audit, html

    @staticmethod
    def _errors(m, audit, html):
        from tarzan.export.newsletter._semantic import (
            validate_newsletter_semantics,
        )
        return [e for e in validate_newsletter_semantics(m, audit, html)
                if e.startswith("1d ")]

    def test_a_faithful_render_raises_nothing(self, monkeypatch):
        m, audit, html = self._render(monkeypatch)
        assert audit["performance_windows"]["1d"]["drawn"] == [
            "twror", "target", "acwi"]
        assert self._errors(m, audit, html) == []

    def test_an_endpoint_that_drifts_from_the_recomputation_is_caught(self, monkeypatch):
        m, audit, html = self._render(monkeypatch)
        audit["performance_windows"]["1d"]["endpoints"]["twror"] += 0.5

        errors = self._errors(m, audit, html)

        assert any("line endpoint differs" in e for e in errors), errors

    def test_a_legend_value_that_disagrees_is_caught(self, monkeypatch):
        m, audit, html = self._render(monkeypatch)
        audit["performance_windows"]["1d"]["legend_values"]["acwi"] += 0.5

        errors = self._errors(m, audit, html)

        assert any("legend uses a different endpoint" in e for e in errors), errors

    def test_a_visible_label_that_rounds_to_the_wrong_figure_is_caught(self, monkeypatch):
        m, audit, html = self._render(monkeypatch)
        audit["performance_windows"]["1d"]["legend_labels"]["target"] = "+9.99%"

        errors = self._errors(m, audit, html)

        assert any("disagrees with" in e for e in errors), errors

    def test_a_label_missing_from_the_html_is_caught(self, monkeypatch):
        """The audit may not describe a figure the reader cannot find."""
        m, audit, html = self._render(monkeypatch)
        label = audit["performance_windows"]["1d"]["legend_labels"]["twror"]

        errors = self._errors(m, audit, html.replace(label, "+0.00%"))

        assert any("absent from rendered HTML" in e for e in errors), errors

    def test_a_line_dropped_from_the_panel_is_caught(self, monkeypatch):
        """The teeth that stop a line vanishing quietly: the drawn set must EQUAL
        the set that resolved, not merely be a subset of it."""
        m, audit, html = self._render(monkeypatch)
        audit["performance_windows"]["1d"]["drawn"] = ["twror", "acwi"]

        errors = self._errors(m, audit, html)

        assert any("drew" in e and "resolved" in e for e in errors), errors

    def test_no_session_means_no_audit_and_no_complaint(self, monkeypatch):
        """A pinned or offline run states the figures instead. There is no line, so
        there is nothing to verify — and the gate must not invent a demand."""
        m, audit, html = self._render(monkeypatch, with_intraday=False)

        assert "1d" not in (audit.get("performance_windows") or {})
        assert self._errors(m, audit, html) == []

    def test_an_audit_claiming_a_panel_that_has_no_session_is_caught(self, monkeypatch):
        m, audit, html = self._render(monkeypatch, with_intraday=False)
        audit.setdefault("performance_windows", {})["1d"] = {
            "endpoints": {"twror": 0.21}, "legend_values": {"twror": 0.21},
            "legend_labels": {"twror": "+0.21%"}, "drawn": ["twror"]}

        errors = self._errors(m, audit, html)

        assert any("unavailable window" in e for e in errors), errors

    def test_a_target_withheld_for_partial_coverage_is_not_demanded(self, monkeypatch):
        """The gate recomputes through the same all-or-nothing rule, so a target
        the renderer correctly refused is not reported as a dropped line."""
        m, audit, html = self._render(monkeypatch)
        m.intraday_quotes.pop("BBB.MI")           # half the target loses its bars
        m2, audit2, html2 = self._render(
            monkeypatch, weights={"AAA.MI": 60.0, "ZZZ.MI": 40.0})

        assert audit2["performance_windows"]["1d"]["drawn"] == ["twror", "acwi"]
        assert self._errors(m2, audit2, html2) == []


class TestTheTargetsOwnSleevesAreRequestedIntraday:
    """The gap a real run exposed and no fixture could.

    ``_live_1d`` built its request from ``holding_performance``, which carries what
    is HELD. A target routinely names instruments not owned yet — four of eight on
    the reference book — and the 1D target line demands FULL coverage, so those four
    silently withheld it on every real send. The panel was correct; the bars were
    never asked for.

    Ordering matters here: ``_live_1d`` runs BEFORE ``_target_history``, so the
    weights cannot be read out of ``ctx``. Both computers read
    ``_target_policy_weights`` instead.
    """

    @staticmethod
    def _engine(holdings, seeds):
        from tarzan.engine.metrics import MetricsEngine

        engine = MetricsEngine.__new__(MetricsEngine)
        engine.holdings = list(holdings)
        engine.rebalance_seeds = list(seeds)
        return engine

    @staticmethod
    def _h(ticker, isin, weight):
        class _H:
            pass
        h = _H()
        h.ticker, h.isin, h.target_portfolio = ticker, isin, weight
        h.price_history = None
        return h

    def test_a_seed_that_is_not_held_is_still_requested(self, monkeypatch):
        import tarzan.runtime as rt

        engine = self._engine(
            [self._h("AAA.MI", "IE00AAA", 60.0)],
            [self._h("SEED.MI", "IE00SEED", 40.0)])
        ctx = {"holding_performance": pd.DataFrame([{"ticker": "AAA.MI"}])}
        monkeypatch.setattr(rt, "allows_live_transport", lambda: False)

        engine._live_1d(ctx)

        assert set(ctx["intraday_requested_tickers"]) == {"AAA.MI", "SEED.MI"}

    def test_a_held_instrument_is_not_requested_twice(self, monkeypatch):
        import tarzan.runtime as rt

        engine = self._engine([self._h("AAA.MI", "IE00AAA", 60.0)], [])
        ctx = {"holding_performance": pd.DataFrame([{"ticker": "AAA.MI"}])}
        monkeypatch.setattr(rt, "allows_live_transport", lambda: False)

        engine._live_1d(ctx)

        requested = ctx["intraday_requested_tickers"]
        assert requested == ("AAA.MI",), requested

    def test_an_untargeted_holding_is_unaffected(self, monkeypatch):
        import tarzan.runtime as rt

        engine = self._engine([self._h("AAA.MI", "IE00AAA", 0.0)], [])
        ctx = {"holding_performance": pd.DataFrame([{"ticker": "AAA.MI"}])}
        monkeypatch.setattr(rt, "allows_live_transport", lambda: False)

        engine._live_1d(ctx)

        assert ctx["intraday_requested_tickers"] == ("AAA.MI",)

    def test_the_weights_survive_an_engine_without_target_machinery(self):
        """``_live_1d`` is exercised on engines built with holdings and a clock and
        nothing else — a session-hours probe. Requesting intraday must not depend on
        the target existing."""
        from tarzan.engine.metrics import MetricsEngine

        bare = MetricsEngine.__new__(MetricsEngine)

        assert bare._target_policy_weights() == {}

    def test_the_gate_expects_the_target_sleeves_too(self):
        """The request set is contract-checked. Widening it without widening the
        expectation would fail every run on "candidates differ" — a delivery block,
        not a cosmetic error."""
        from tarzan.export.newsletter._semantic import (
            validate_newsletter_semantics,
        )

        class _M:
            benchmark_tickers = {"B": "BENCH.MI"}
            benchmark_resolution_errors = ()
            holding_performance = pd.DataFrame([{"ticker": "AAA.MI"}])
            holdings_df = pd.DataFrame([{"ticker": "AAA.MI"}])
            target_weights = {"AAA.MI": 60.0, "SEED.MI": 40.0}
            intraday_requested_tickers = ("AAA.MI", "SEED.MI")
            intraday_quotes = {
                "AAA.MI": {"intraday_source_ticker": "AAA.MI",
                           "intraday_series": [1.0, 2.0],
                           "intraday_baseline": 1.0},
                "SEED.MI": {"intraday_source_ticker": "SEED.MI",
                            "intraday_series": [1.0, 2.0],
                            "intraday_baseline": 1.0}}
            actual_value_series = None

        audit = {"performance_intraday": {
            "origin": "metrics_preprocessing",
            "requested_tickers": ("AAA.MI", "SEED.MI"),
            "returned_tickers": ("AAA.MI", "SEED.MI"),
            "source_tickers": {"AAA.MI": "AAA.MI", "SEED.MI": "SEED.MI"}}}

        errors = [e for e in validate_newsletter_semantics(_M(), audit, "")
                  if "candidates differ" in e]

        assert errors == [], errors

    def test_the_gate_keeps_its_teeth_on_a_dropped_target_sleeve(self):
        """Widening the expectation must not turn it into "anything goes"."""
        from tarzan.export.newsletter._semantic import (
            validate_newsletter_semantics,
        )

        class _M:
            benchmark_tickers = {"B": "BENCH.MI"}
            benchmark_resolution_errors = ()
            holding_performance = pd.DataFrame([{"ticker": "AAA.MI"}])
            holdings_df = pd.DataFrame([{"ticker": "AAA.MI"}])
            target_weights = {"AAA.MI": 60.0, "SEED.MI": 40.0}
            intraday_requested_tickers = ("AAA.MI",)      # SEED.MI dropped
            intraday_quotes = {"AAA.MI": {"intraday_source_ticker": "AAA.MI",
                                          "intraday_series": [1.0, 2.0],
                                          "intraday_baseline": 1.0}}
            actual_value_series = None

        audit = {"performance_intraday": {
            "origin": "metrics_preprocessing",
            "requested_tickers": ("AAA.MI",),
            "returned_tickers": ("AAA.MI",),
            "source_tickers": {"AAA.MI": "AAA.MI"}}}

        errors = [e for e in validate_newsletter_semantics(_M(), audit, "")
                  if "candidates differ" in e]

        assert errors, "the gate went blind to a dropped target sleeve"
