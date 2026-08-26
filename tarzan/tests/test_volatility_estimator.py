"""What the volatility lines actually measure, pinned.

The chart says "annualized volatility". That is only true of a specific
estimator, and the three lines on one panel are only comparable if all three use
it over the same lookback. These fix the definition:

  * sample stdev (ddof=1, mean removed) of DAILY SIMPLE returns × √252 × 100 —
    the same estimator ``stats.risk_metric_row`` reports in the RISK section, so
    the chart and the tile cannot drift into two meanings of one word;
  * a FULL window before any number is emitted;
  * ``vol_window`` counts ROWS, so every line must be sampled on trading days or
    one line's 21 rows are three calendar weeks against another's six.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.stats import TRADING_DAYS, risk_metric_row
from tarzan.export._perf_series import _perf_vol_series, _rolling_ann_vol
from tarzan.models.portfolio import PortfolioMetrics

_IDX = pd.date_range("2024-01-01", "2026-08-20", freq="B")


def _wobble(n, amp=0.01, drift=(100.0, 140.0)):
    return np.linspace(*drift, n) * (1 + amp * np.sin(np.arange(n) / 5))


class TestTheEstimatorIsTheStandardOne:
    def test_it_is_stdev_of_daily_returns_times_root_252(self):
        """Asserted against the definition rather than a recorded number."""
        s = pd.Series(_wobble(len(_IDX)), index=_IDX)

        got = _rolling_ann_vol(s, 21)

        r = s.pct_change()
        want = r.iloc[-21:].std(ddof=1) * np.sqrt(252) * 100.0
        assert got.iloc[-1] == pytest.approx(want, rel=1e-12)

    def test_the_annualization_factor_is_the_configured_trading_year(self):
        """√252, not √365: the inputs are sampled on trading days, so scaling by
        calendar days would inflate every figure by ~20%."""
        assert TRADING_DAYS == 252
        s = pd.Series(_wobble(len(_IDX)), index=_IDX)
        daily = s.pct_change().iloc[-21:].std(ddof=1) * 100.0
        assert _rolling_ann_vol(s, 21).iloc[-1] / daily == pytest.approx(
            TRADING_DAYS ** 0.5, rel=1e-12)

    def test_it_is_the_sample_stdev_and_the_mean_is_removed(self):
        """ddof=1 and demeaned. A trending series with ddof=0 or with the mean
        left in reads visibly higher, and neither is what "volatility" means in
        the RISK table beside it."""
        s = pd.Series(_wobble(len(_IDX)), index=_IDX)
        r = s.pct_change().iloc[-21:]

        got = _rolling_ann_vol(s, 21).iloc[-1]

        assert got == pytest.approx(r.std(ddof=1) * np.sqrt(252) * 100, rel=1e-12)
        assert got != pytest.approx(r.std(ddof=0) * np.sqrt(252) * 100, rel=1e-6)
        rms = float(np.sqrt((r ** 2).mean()))          # not demeaned
        assert got != pytest.approx(rms * np.sqrt(252) * 100, rel=1e-6)

    def test_the_chart_and_the_risk_tile_agree_on_a_shared_span(self):
        """One definition of volatility for the whole issue.

        Over the SAME observations the rolling line's estimate must equal the
        figure ``risk_metric_row`` puts in the RISK section. They are read in
        different places by the same reader; two conventions would be a silent
        contradiction.
        """
        s = pd.Series(_wobble(60), index=_IDX[:60])

        rolled = _rolling_ann_vol(s, len(s) - 1).iloc[-1]
        tile = risk_metric_row(s)["volatility"]

        assert rolled == pytest.approx(tile, rel=1e-12)

    def test_a_flat_series_has_no_volatility(self):
        flat = pd.Series(100.0, index=_IDX)
        assert _rolling_ann_vol(flat, 21).iloc[-1] == pytest.approx(0.0)

    def test_a_known_two_state_series_hits_the_closed_form(self):
        """A series alternating +1%/−1% every day has a daily stdev of exactly
        1% (ddof=1 over an even split), so 21 sessions annualize to 1·√252."""
        n = 43                                     # 42 returns: 21 up, 21 down
        px = [100.0]
        for i in range(n - 1):
            px.append(px[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
        s = pd.Series(px, index=_IDX[:n])

        got = _rolling_ann_vol(s, 42).iloc[-1]

        r = s.pct_change().dropna()
        assert got == pytest.approx(r.std(ddof=1) * np.sqrt(252) * 100, rel=1e-12)
        assert 15.0 < got < 17.0                   # ≈ 1% × √252 = 15.9%


class TestNoNumberIsEmittedWithoutAFullWindow:
    def test_the_leading_days_have_no_estimate(self):
        s = pd.Series(_wobble(len(_IDX)), index=_IDX)

        v = _rolling_ann_vol(s, 21)

        # 1 for the first pct_change + 20 more to fill the window.
        assert v.iloc[:21].isna().all()
        assert v.notna().iloc[21]

    def test_half_a_window_is_not_an_estimate(self):
        """It used to emit from ``window // 2`` observations — a σ off ten days,
        a ±24% standard error, printed as a 21-day figure."""
        s = pd.Series(_wobble(len(_IDX)), index=_IDX)
        assert np.isnan(_rolling_ann_vol(s, 21).iloc[10])

    def test_the_since_inception_line_opens_on_its_first_real_estimate(self):
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(_wobble(len(_IDX)), index=_IDX)

        vs = _perf_vol_series(m, None, n_days=None)

        assert vs["dates"][0] == _IDX[21]
        assert not any(v != v for v in vs["port"])
        # And the opening point is a real 21-session estimate, not the value of a
        # later one carried backwards.
        r = m.portfolio_history.pct_change()
        assert vs["port"][0] == pytest.approx(
            r.iloc[1:22].std(ddof=1) * np.sqrt(252) * 100, rel=1e-9)

    def test_the_opening_run_is_not_flat(self):
        """The back-fill's signature: the first ``window`` points all equal.

        A rolling estimate over a wobbling series moves every day, so an opening
        run of identical values means the line was painted rather than measured.
        """
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(_wobble(len(_IDX)), index=_IDX)

        port = _perf_vol_series(m, None, n_days=None)["port"]

        assert len(set(round(v, 9) for v in port[:21])) > 1

    def test_a_history_too_short_for_one_window_draws_nothing(self):
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(_wobble(12), index=_IDX[:12])
        assert _perf_vol_series(m, None, n_days=None) is None


class TestTheLinesAreComparable:
    """21 rows must be 21 sessions on every line."""

    def _metrics(self):
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(_wobble(len(_IDX)), index=_IDX)
        m.target_history = pd.Series(
            _wobble(len(_IDX), amp=0.03, drift=(100.0, 130.0)), index=_IDX)
        m.benchmark_histories = {"B": pd.Series(
            _wobble(len(_IDX), amp=0.02, drift=(200.0, 260.0)), index=_IDX)}
        return m

    def test_every_line_is_sampled_at_trading_day_density(self):
        """A calendar-daily input would put ~15 sessions in one line's 21-row
        window against 21 in another's, and weekend zero-returns would deflate the
        estimate while √252 annualizes as if they were not there.

        Density, not weekday labels: ``normalize_index`` collapses a tz-aware
        index through UTC, so a Milan bar stamped at local midnight lands on the
        previous calendar date and every Monday reads as a Sunday. The real target
        series carries 69 such Sunday labels while still holding exactly one row
        per session — a uniform relabelling, which leaves the return sequence (and
        therefore every σ) untouched. Asserting "no weekend rows" would pass on a
        synthetic fixture and promise something production does not hold.
        """
        m = self._metrics()
        for name, s in [("portfolio", m.portfolio_history),
                        ("target", m.target_history),
                        ("benchmark", m.benchmark_histories["B"])]:
            idx = pd.DatetimeIndex(s.index)
            span = (idx[-1] - idx[0]).days + 1
            assert len(s) / span == pytest.approx(5 / 7, abs=0.05), (
                f"{name} is not sampled one row per session")

    def test_all_three_lines_share_one_index(self):
        vs = _perf_vol_series(self._metrics(), "B", n_days=None)
        n = len(vs["dates"])
        assert len(vs["port"]) == n
        assert len(vs["target"]) == n
        assert len(vs["acwi"]) == n

    def test_a_bumpier_series_reads_bumpier(self):
        vs = _perf_vol_series(self._metrics(), "B", n_days=None)
        # amplitudes 1% / 2% / 3% → portfolio < benchmark < target
        assert vs["port"][-1] < vs["acwi"][-1] < vs["target"][-1]

    def test_a_reference_without_a_full_window_at_the_left_edge_is_dropped(self):
        m = self._metrics()
        m.target_history = m.target_history.iloc[-10:]
        vs = _perf_vol_series(m, "B", n_days=None)
        assert vs["target"] is None
        assert vs["acwi"] is not None


class TestTheThreeLinesAreComparedOverOnePeriod:
    """The like-for-like σ in the since-inception key.

    Each series has its OWN history length — on 26 Aug 2026 the book held 8
    months, the target 16, the benchmark 2 years. Reporting each one's unclipped
    σ put 14.76% (benchmark, two years) beside 10.51% (book, eight months) and
    read as a risk gap when a third of it was just a longer, rougher period.
    """

    @staticmethod
    def _metrics():
        """A book with a SHORT life against two long references, where the extra
        history is deliberately calmer — so an unclipped σ understates the
        reference and a clipped one does not."""
        long_idx = pd.date_range("2024-01-01", "2026-08-20", freq="B")
        # Calm for the first two thirds, rough for the last third.
        n = len(long_idx)
        amp = np.where(np.arange(n) < 2 * n // 3, 0.002, 0.03)
        rough = pd.Series(np.linspace(100, 130, n) * (1 + amp * np.sin(np.arange(n) / 5)),
                          index=long_idx)
        book_idx = long_idx[-120:]                      # only the rough tail
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(_wobble(len(book_idx), amp=0.02), index=book_idx)
        m.target_history = rough
        m.benchmark_histories = {"B": rough * 2.0}
        return m, long_idx, book_idx, rough

    def test_the_references_are_clipped_to_the_books_own_span(self):
        m, _long, book_idx, rough = self._metrics()

        span = _perf_vol_series(m, "B", n_days=None)["span"]

        clipped = rough[(rough.index >= book_idx[0]) & (rough.index <= book_idx[-1])]
        want = clipped.pct_change().dropna().std(ddof=1) * np.sqrt(252) * 100
        assert span["target"] == pytest.approx(want, rel=1e-9)
        assert span["acwi"] == pytest.approx(want, rel=1e-9)   # same shape, ×2 level

    def test_clipping_actually_changes_the_reference(self):
        """Guards the whole point: if the clip were a no-op the test above would
        pass against the unclipped figure too."""
        m, _long, _book, rough = self._metrics()

        span = _perf_vol_series(m, "B", n_days=None)["span"]

        unclipped = rough.pct_change().dropna().std(ddof=1) * np.sqrt(252) * 100
        assert abs(span["target"] - unclipped) > 1.0, (
            f"clipped {span['target']:.2f}% vs whole history {unclipped:.2f}%")

    def test_the_books_own_figure_is_its_navs_whole_life_sigma(self):
        """The book's σ needs no clipping — it IS its whole life — so it equals
        ``risk_metric_row`` on the flow-adjusted NAV.

        Deliberately NOT asserted against the RISK section's volatility. That
        table renders ``historical_risk``, whose portfolio row is a current-weight
        static backtest over the common window of holdings with ≥1Y of history —
        10.77% live on 26 Aug 2026 against this 10.51%. Different construction,
        different span, by design there. An earlier version of this test claimed
        the two were one number.
        """
        m, _long, _book, _rough = self._metrics()

        span = _perf_vol_series(m, "B", n_days=None)["span"]

        assert span["port"] == pytest.approx(
            risk_metric_row(m.portfolio_history)["volatility"], rel=1e-9)

    def test_the_span_is_reported_so_the_reader_can_check_it(self):
        m, _long, book_idx, _rough = self._metrics()
        span = _perf_vol_series(m, "B", n_days=None)["span"]
        assert span["from"] == book_idx[0]
        assert span["to"] == book_idx[-1]

    def test_a_reference_too_short_to_measure_reports_nothing(self):
        m, _long, _book, _rough = self._metrics()
        m.target_history = m.target_history.iloc[-5:]
        assert _perf_vol_series(m, "B", n_days=None)["span"]["target"] is None

    def test_the_key_prints_the_span_figure_beside_every_name(self):
        import re
        html = TestTheSinceInceptionPanelIsRendered._section()["vs_market_html"]
        si_key = re.findall(r'margin:7px 0 0;">(.*?)</div>', html, re.S)[1]
        names = re.findall(r'<span style="color:#8FA3BC;">([^<]+)</span>', si_key)
        assert len(names) == 3
        for n in names:
            assert re.search(r'\d+\.\d+%$', n), f"{n!r} carries no span figure"

    def test_the_thirty_day_key_stays_bare(self):
        """Only the since-inception key carries the span figure. On the 30-day
        panel a whole-life σ would name a period the panel does not draw."""
        import re
        html = TestTheSinceInceptionPanelIsRendered._section()["vs_market_html"]
        keys = re.findall(r'margin:7px 0 0;">(.*?)</div>', html, re.S)
        names = re.findall(r'<span style="color:#8FA3BC;">([^<]+)</span>', keys[3])
        assert names and not any(re.search(r'\d+\.\d+%$', n) for n in names), names


class TestTheSinceInceptionPanelIsRendered:
    @staticmethod
    def _section():
        from tarzan.models.investor_config import InvestorConfig
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30

        n = len(_IDX)
        m = PortfolioMetrics(total_value=6000.0, invested_value=6000.0,
                             holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(_wobble(n, drift=(4800, 6000)), index=_IDX)
        m.pnl_series = pd.Series(_wobble(n, drift=(1, 1000)), index=_IDX)
        m.portfolio_history = pd.Series(_wobble(n), index=_IDX)
        m.target_history = pd.Series(_wobble(n, amp=0.03, drift=(100, 130)), index=_IDX)
        m.benchmark_histories = {"B": pd.Series(
            _wobble(n, amp=0.02, drift=(200, 260)), index=_IDX)}
        return _build_performance30(_NewsletterContext(
            metrics=m, config=InvestorConfig(), benchmark_geo="B"))

    def test_there_are_two_volatility_panels_and_two_return_panels(self):
        import re
        html = self._section()["vs_market_html"]
        caps = re.findall(r'margin-bottom:5px;">([^<]+)</div>', html)
        assert caps == ["Return · since inception",
                        'Volatility · since inception (line: rolling 21 sessions · key: whole span, annualized)',
                        "Return · last 30 days",
                        "Volatility · last 30 days"]

    def test_the_since_inception_volatility_sits_under_its_return_chart(self):
        import re
        html = self._section()["vs_market_html"]
        order = [m.group(1) for m in re.finditer(
            r'margin-bottom:5px;">([^<]+)</div>', html)]
        assert order.index('Volatility · since inception (line: rolling 21 sessions · key: whole span, annualized)') == 1

    def test_the_wide_panel_is_full_width(self):
        import re
        html = self._section()["vs_market_html"]
        widths = re.findall(r'<svg width="100%" viewBox="0 0 (\d+) ', html)
        # since-inception return, since-inception volatility: both 580.
        assert widths[:2] == ["580", "580"]

    def test_the_caption_separates_the_line_from_the_key(self):
        """Two different figures sit on this panel and a reader cannot tell them
        apart from the picture: the LINE is a rolling 21-session estimate, the
        number beside each name in the KEY is one σ over the whole span. The
        caption has to say which is which."""
        html = self._section()["vs_market_html"]
        assert "line: rolling 21 sessions" in html
        assert "key: whole span, annualized" in html
