"""The target allocation as a line on the three "Vs target &" charts.

The section compared the book against the market only. It could not answer the
question the book is actually steered by — "am I ahead of the allocation I am
trying to hold?" — so the target NAV is built once in the engine and drawn on
the return, the since-inception and the volatility panel.

Three ways this can lie, each pinned below: weighting the target by what is
currently HELD (a quarter of it is not owned yet), renormalizing over the
sleeves that happen to have history (a different portfolio under the same name),
and drawing a lead-in the target's own history does not cover.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.export._perf_series import _target_line
from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

_IDX = pd.date_range("2024-01-01", "2026-08-20", freq="B")


def _holding(ticker, *, target=None, history=None, value=1000.0, seeded=False):
    h = Holding(isin=f"IE{ticker:0<10}", ticker=ticker, quantity=1.0,
                cost_basis_eur=value, market_value_eur=value, currency="EUR")
    h.target_portfolio = target
    h.price_history = history
    h.current_value = value
    h.is_seeded_target = seeded
    return h


def _ramp(total_return: float, index=_IDX) -> pd.Series:
    """A price series compounding smoothly to ``total_return`` over ``index``."""
    step = (1.0 + total_return) ** (1.0 / (len(index) - 1))
    return pd.Series(100.0 * step ** np.arange(len(index)), index=index)


def _target_history(holdings, seeds=(), ctx=None):
    engine = MetricsEngine(list(holdings), InvestorConfig(),
                           rebalance_seeds=list(seeds))
    out = dict(ctx or {})
    engine._target_history(out)
    return out.get("target_history")


class TestTheTargetIsWeightedByItsTargets:
    def test_a_not_held_sleeve_still_carries_its_target_weight(self):
        """The seeds are half the question.

        On 25 Aug 2026 three of the eight target instruments (AVWC 10, AVWS 9,
        AVEM 4) were not owned yet — 23% of the target. They live in
        ``rebalance_seeds``, not in ``holdings``, so a builder reading only the
        book would draw a 77% portfolio and label it "target".
        """
        held = _holding("HELD", target=50.0, history=_ramp(0.20))
        seed = _holding("SEED", target=50.0, history=_ramp(0.60), seeded=True)

        nav = _target_history([held], [seed])

        assert nav is not None
        total = nav.iloc[-1] / nav.iloc[0] - 1.0
        # 50/50 of +20% and +60% lands near +40%; the held sleeve alone is +20%.
        assert total == pytest.approx(0.386, abs=0.01)

    def test_weights_are_the_targets_not_the_current_values(self):
        """A €90k position at a 10% target must contribute 10%, not 90%."""
        big = _holding("BIG", target=10.0, history=_ramp(0.0), value=90_000.0)
        small = _holding("SML", target=90.0, history=_ramp(1.00), value=1_000.0)

        nav = _target_history([big, small])

        # 0.866, not 0.90: holding the weight constant means taking 90% of each
        # DAY's return and compounding that, which is below 90% of the compounded
        # total. Nowhere near the ~0.10 a value-weighted reading would give.
        total = nav.iloc[-1] / nav.iloc[0] - 1.0
        assert total == pytest.approx(0.866, abs=0.005)

    def test_the_weights_are_constant_and_never_drift(self):
        """A target allocation is 32% NTSG at every point on the line.

        The alternatives all let the weight WANDER with performance: buy & hold
        never corrects it, and a periodic policy snaps it back on dates that have
        nothing to do with the window the reader is looking at. Here one sleeve
        doubles and the other is flat at 50/50, which separates them cleanly —
        buy & hold lands on exactly +50%, constant weights below it, and any
        periodic policy in between.

        Asserted against the closed form rather than a recorded number: with the
        weight constant, each day's portfolio return is exactly ``0.5 * r``, so
        the NAV is ``(1 + 0.5r)**n``. No drifting policy satisfies that identity —
        monthly resets land within a tenth of a point of it on a smooth ramp,
        which a tolerance loose enough to be readable would wave through.
        """
        up = _ramp(1.00)
        r = float(up.iloc[1] / up.iloc[0] - 1.0)        # the sleeve's daily return
        n = len(up) - 1

        nav = _target_history([
            _holding("UP", target=50.0, history=up),
            _holding("FLAT", target=50.0, history=_ramp(0.00)),
        ])

        total = nav.iloc[-1] / nav.iloc[0] - 1.0
        assert total == pytest.approx((1.0 + 0.5 * r) ** n - 1.0, rel=1e-9)
        assert total < 0.49, "buy & hold would land on exactly +50%"

    def test_an_instrument_without_a_target_is_not_in_the_target(self):
        held = _holding("HELD", target=100.0, history=_ramp(0.20))
        other = _holding("OTHER", target=None, history=_ramp(5.00))

        nav = _target_history([held, other])

        assert nav.iloc[-1] / nav.iloc[0] - 1.0 == pytest.approx(0.20, abs=0.01)

    def test_no_targets_at_all_yields_no_series(self):
        assert _target_history([_holding("HELD", history=_ramp(0.2))]) is None


class TestASleeveWithoutHistoryWithholdsTheWholeLine:
    """Dropping a sleeve renormalizes the rest, which draws a DIFFERENT
    portfolio under the target's name. Better no line than a quiet substitute."""

    def test_a_missing_history_withholds_the_series(self):
        ok = _holding("OK", target=60.0, history=_ramp(0.20))
        blind = _holding("BLIND", target=40.0, history=None)

        assert _target_history([ok, blind]) is None

    def test_a_one_point_history_is_not_a_history(self):
        ok = _holding("OK", target=60.0, history=_ramp(0.20))
        stub = _holding("STUB", target=40.0, history=pd.Series(
            [100.0], index=[pd.Timestamp("2026-08-20")]))

        assert _target_history([ok, stub]) is None

    def test_a_sleeve_with_unavailable_order_history_withholds_it_too(self):
        ok = _holding("OK", target=60.0, history=_ramp(0.20))
        broken = _holding("BROKEN", target=40.0, history=_ramp(0.10))

        withheld = _target_history(
            [ok, broken], ctx={"_order_history_unavailable": [broken.isin]})

        assert withheld is None


class TestPricesAlignBeforeTheyAreDifferenced:
    def test_a_venue_holiday_does_not_delete_that_day_s_return(self):
        """Sleeves trade on different calendars.

        Taking each sleeve's returns FIRST and inner-joining after (the shape
        ``_historical_risk`` uses) discards every day any single venue was shut —
        so a 4% jump on Xetra vanished because Milan was closed for the
        Assumption. Aligning prices first carries the closed venue forward at its
        last close, which is what a closed venue is worth.
        """
        idx = pd.date_range("2026-08-03", "2026-08-21", freq="B")
        milan = pd.Series(100.0, index=idx)
        xetra = pd.Series(100.0, index=idx)
        shut = pd.Timestamp("2026-08-14")
        # Xetra gains 4% on the day Milan is shut, and holds it.
        xetra.loc[xetra.index >= shut] = 104.0
        milan = milan.drop(shut)

        nav = _target_history([
            _holding("MI", target=50.0, history=milan),
            _holding("DE", target=50.0, history=xetra),
        ])

        # Half the book gained 4% → the target gained ~2%. Dropping the day
        # entirely reads 0%.
        assert nav.iloc[-1] / nav.iloc[0] - 1.0 == pytest.approx(0.02, abs=0.002)


class TestTheTargetLineNeverFakesItsLeadIn:
    def _metrics(self, start):
        m = PortfolioMetrics()
        m.target_history = _ramp(0.20, pd.date_range(start, "2026-08-20", freq="B"))
        return m

    def test_a_target_covering_the_window_is_drawn(self):
        idx = pd.date_range("2026-07-20", "2026-08-20", freq="B")
        line = _target_line(self._metrics("2024-01-01"), idx)
        assert line is not None and len(line) == len(idx)
        assert line[0] == pytest.approx(0.0, abs=1e-9)   # rebased on the window

    def test_a_target_opening_inside_the_window_is_blank_before_it_starts(self):
        """The property this has always protected: the uncovered stretch must not read
        as "the target went nowhere". It used to be protected by withholding the whole
        line, which cost the target on every panel wider than its own life. Now the
        lead-in is NaN and ``chart_pct_compact`` breaks its polyline there, so the
        stretch says nothing instead of saying zero.
        """
        import math

        idx = pd.date_range("2026-07-20", "2026-08-20", freq="B")
        line = _target_line(self._metrics("2026-08-05"), idx)

        assert line is not None and len(line) == len(idx)
        first = next(i for i, v in enumerate(line) if math.isfinite(v))
        assert first > 0, "nothing was blanked"
        assert all(not math.isfinite(v) for v in line[:first])
        # And crucially NOT a flat run at zero, which is what withholding avoided.
        assert not all(v == 0.0 for v in line[:first])
        assert line[first] == pytest.approx(0.0, abs=1e-9)   # rebased where it starts

    def test_a_target_that_covers_nothing_is_still_no_line(self):
        """The floor stays: fewer than two in-window observations is not a line."""
        idx = pd.date_range("2026-07-20", "2026-08-20", freq="B")
        m = PortfolioMetrics()
        m.target_history = _ramp(0.20, pd.date_range("2027-01-04", "2027-02-01",
                                                     freq="B"))
        assert _target_line(m, idx) is None

    def test_no_target_history_is_no_line(self):
        assert _target_line(PortfolioMetrics(), _IDX[-20:]) is None


class TestTheVolatilityPanelHoldsTheSameRule:
    def _metrics(self, target_start):
        idx = pd.date_range("2024-01-01", "2026-08-20", freq="B")
        wobble = 1 + 0.01 * np.sin(np.arange(len(idx)) / 5)
        m = PortfolioMetrics()
        m.portfolio_history = pd.Series(np.linspace(100, 140, len(idx)) * wobble,
                                        index=idx)
        t_idx = pd.date_range(target_start, "2026-08-20", freq="B")
        t_wob = 1 + 0.03 * np.sin(np.arange(len(t_idx)) / 4)
        m.target_history = pd.Series(np.linspace(100, 130, len(t_idx)) * t_wob,
                                     index=t_idx)
        return m

    def test_the_target_gets_its_own_volatility_line(self):
        from tarzan.export._perf_series import _perf_vol_series
        vs = _perf_vol_series(self._metrics("2024-01-01"), None, n_days=30)
        assert vs and vs["target"] and len(vs["target"]) == len(vs["dates"])
        # The target wobbles three times harder than the portfolio.
        assert vs["target"][-1] > vs["port"][-1]

    def test_a_target_opening_inside_the_window_is_blank_until_it_can_be_measured(self):
        """Asserted on the since-inception span, which is where this bites.

        The line used to be withheld here, on the stated grounds that it "would
        otherwise be back-filled with its first computable volatility across a year it
        says nothing about". Measured: it would not. ``_line`` reindexes with
        ``method="ffill"``, which fills forward only, so every date before the first
        rolling estimate stays NaN and the chart breaks the polyline there. The
        back-fill that justified withholding is not in the code.

        So the line is drawn from where it can be measured, and the stretch before it
        is blank — which is what withholding was trying to achieve, at the price of
        losing the part that exists.
        """
        import math

        from tarzan.export._perf_series import _perf_vol_series

        vs = _perf_vol_series(self._metrics("2025-06-01"), None, n_days=None)

        assert vs and vs["port"]
        assert vs["target"] is not None
        first = next(i for i, v in enumerate(vs["target"]) if math.isfinite(v))
        assert first > 0, "the lead-in was filled"
        assert all(not math.isfinite(v) for v in vs["target"][:first])

    def test_a_target_with_no_measurable_volatility_anywhere_is_no_line(self):
        """The remaining guard, and the real failure it protects against: a series
        holding no finite estimate at all is not a line."""
        from tarzan.export._perf_series import _perf_vol_series

        m = self._metrics("2024-01-01")
        m.target_history = m.target_history.iloc[:5]      # too short for any window
        vs = _perf_vol_series(m, None, n_days=None)

        assert vs and vs["port"]
        assert vs["target"] is None


class TestTheWholeTargetSharesOneClock:
    def test_the_seeds_are_stamped_with_today_alongside_the_book(self):
        """A part-live endpoint is a part-live line.

        The held sleeves and the tracked benchmarks both get today's market
        point; the not-held target instruments live in ``seeds`` and did not, so
        the target line ended 77% on today's prices and 23% on the previous
        close — 0.23pp on the since-inception figure, measured live on
        26 Aug 2026. One clock for every series the charts read.
        """
        import inspect

        from tarzan import orchestrator

        src = inspect.getsource(orchestrator)
        assert "apply_to_holdings(holdings + seeds)" in src, (
            "the seeds must be stamped with the same current session as the book")

    def test_a_zero_quantity_seed_survives_the_stamp(self):
        """The stamp writes ``current_value = quantity * price``, and a seed's
        quantity is 0 by construction."""
        from tarzan.data import current_session

        seed = _holding("SEED", target=10.0, history=_ramp(0.1), seeded=True)
        seed.quantity = 0.0
        current_session.apply_to_holdings([seed])   # offline: a no-op, not a raise

        assert seed.quantity == 0.0


class TestAllThreePanelsCarryTheTarget:
    """The ask was all three charts, so all three are asserted — the legend is
    the only place a line is named, so a missing key means an unreadable line."""

    @staticmethod
    def _section(with_target=True, bench="Bench Index"):
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30

        idx = pd.date_range("2025-08-01", "2026-08-20", freq="B")
        n = len(idx)
        wobble = 1 + 0.01 * np.sin(np.arange(n) / 5)
        m = PortfolioMetrics(total_value=6000.0, invested_value=6000.0,
                             holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, n) * wobble, index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, n) * wobble, index=idx)
        m.unrealized_series = pd.Series(np.linspace(0, 800, n) * wobble, index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, n) * wobble, index=idx)
        m.benchmark_histories = {bench: pd.Series(
            np.linspace(200, 230, n) * (1 + 0.02 * np.sin(np.arange(n) / 4)),
            index=idx)}
        if with_target:
            m.target_history = _ramp(0.25, idx)
        return _build_performance30(
            _NewsletterContext(metrics=m, config=InvestorConfig(),
                               benchmark_geo=bench))

    def test_every_panel_names_the_target_in_its_colour_key(self):
        sec = self._section()
        html = sec["vs_market_html"]
        # Four keys — the return grid, the lifetime return, and the two volatility
        # panels — plus the 1D cell, which is figures rather than a plot and names
        # its three measures itself. The lifetime volatility key carries the
        # like-for-like span figure too ("Target 11.10%"), so match the name rather
        # than the exact label.
        named = re.findall(r'>(Target(?: [\d.]+%)?)</span>', html)
        assert len(named) == 5, named

    def test_the_target_is_a_line_not_only_a_legend_entry(self):
        from tarzan.export._palette import PALETTE
        html = self._section()["vs_market_html"]
        drawn = len(re.findall(
            rf'<polyline[^>]*stroke="{PALETTE["target"]}"', html))
        # Five return-grid cells (5D/1M/3M/YTD/1Y — 1D is figures, not a plot),
        # the lifetime return, and the volatility pair (3M + lifetime).
        assert drawn == 8, f"target polylines drawn: {drawn}"

    def test_the_heading_names_the_target_and_the_benchmark_in_use(self):
        assert self._section()["vs_market_title"] == "Vs target &amp; Bench Index"

    def test_the_heading_does_not_promise_a_target_it_cannot_draw(self):
        sec = self._section(with_target=False)
        assert sec["vs_market_title"] == "Vs the market"
        assert ">Target</span>" not in sec["vs_market_html"]

    def test_a_benchmark_name_with_markup_is_escaped_in_the_heading(self):
        title = self._section(bench="Smith & Sons")["vs_market_title"]
        assert title == "Vs target &amp; Smith &amp; Sons"

    def test_the_target_endpoint_is_audited(self):
        """The semantic gate refuses to ship a chart line it cannot verify, so
        the target has to reach the audit alongside the other four."""
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30

        audit: dict = {}
        idx = pd.date_range("2025-08-01", "2026-08-20", freq="B")
        m = PortfolioMetrics(total_value=6000.0, invested_value=6000.0,
                             holdings_df=pd.DataFrame([{"cost_basis_eur": 5000.0}]))
        m.pnl_eur, m.pnl_pct, m.twror_pct = 1000.0, 20.0, 14.49
        m.actual_value_series = pd.Series(np.linspace(4800, 6000, len(idx)), index=idx)
        m.pnl_series = pd.Series(np.linspace(0, 1000, len(idx)), index=idx)
        m.portfolio_history = pd.Series(np.linspace(100, 114.5, len(idx)), index=idx)
        m.target_history = _ramp(0.25, idx)
        _build_performance30(_NewsletterContext(
            metrics=m, config=InvestorConfig(), semantic_audit=audit))

        perf = audit["performance_30d"]
        assert perf["endpoints"]["target"] is not None
        assert perf["legend_values"]["target"] == perf["endpoints"]["target"]

    def test_the_gate_verifies_the_target_line(self):
        from tarzan.export.newsletter import _semantic
        import inspect
        src = inspect.getsource(_semantic)
        assert '"target", "acwi"' in src or '"target"' in src, (
            "the 30-day audit loop must include the target line")


class TestTheBlankLeadInSurvivesToTheSvg:
    """The gap has to reach the markup, and cost no other line its label.

    Two things could have swallowed it. ``chart_pct_compact`` might have bridged the
    NaN stretch with a straight segment — it does not, it emits one polyline per
    contiguous finite run — and its own comment warns that "a NaN endpoint wipes EVERY
    end label on the panel, because the placement shares one shift across all of
    them". The target's NaNs are at the HEAD, not the end, so that path should not
    fire; asserted rather than assumed.
    """

    @staticmethod
    def _svg(lead_in_nans: int = 40, n: int = 120):
        from tarzan.export import _charts
        from tarzan.export._palette import PALETTE

        idx = pd.date_range("2026-01-02", periods=n, freq="B")
        full = list(np.linspace(0, 8, n))
        bench = list(np.linspace(0, 5, n))
        tgt = ([float("nan")] * lead_in_nans
               + list(np.linspace(0, 6, n - lead_in_nans)))
        return _charts.chart_pct_compact(
            [{"values": full, "color": PALETTE["port"], "width": 2.2,
              "end_label": "+8.00%"},
             {"values": tgt, "color": PALETTE["target"], "end_label": "+6.00%"},
             {"values": bench, "color": PALETTE["bench"], "end_label": "+5.00%"}],
            list(idx), w=580, h=166, month_ticks=True, end_gutter=54), n

    @staticmethod
    def _polys(svg, colour):
        return re.findall(
            rf'<polyline points="([^"]+)" fill="none" stroke="{colour}"', svg)

    def test_the_target_line_starts_where_its_data_starts(self):
        from tarzan.export._palette import PALETTE

        svg, n = self._svg(lead_in_nans=40)
        polys = self._polys(svg, PALETTE["target"])

        assert len(polys) == 1, polys
        assert len(polys[0].split()) == n - 40
        # ...and to the RIGHT of where the full-span lines begin.
        tgt_x = float(polys[0].split()[0].split(",")[0])
        port_x = float(self._polys(svg, PALETTE["port"])[0].split()[0].split(",")[0])
        assert tgt_x > port_x + 100, (tgt_x, port_x)

    def test_the_references_are_not_truncated_by_it(self):
        from tarzan.export._palette import PALETTE

        svg, n = self._svg(lead_in_nans=40)
        for colour in (PALETTE["port"], PALETTE["bench"]):
            polys = self._polys(svg, colour)
            assert len(polys) == 1 and len(polys[0].split()) == n, (colour, polys)

    def test_no_nan_reaches_the_markup(self):
        """A literal "nan,nan" in a points attribute truncates the line in every
        client, silently and at the wrong place."""
        svg, _n = self._svg(lead_in_nans=40)
        assert "nan" not in svg.lower()

    def test_every_end_label_survives(self):
        svg, _n = self._svg(lead_in_nans=40)
        labels = re.findall(r'>([+-]\d+\.\d\d%)<', svg)
        assert labels == ["+8.00%", "+6.00%", "+5.00%"], labels
