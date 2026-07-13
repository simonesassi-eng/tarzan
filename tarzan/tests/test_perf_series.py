"""Perf-series helpers: benchmark anchoring + the since-inception full window.

Regression cover for the bug where the "You vs the market" chart's benchmark
line disagreed with the Performance section's period return: the chart
reindexed the benchmark onto the portfolio's date index (which can start on a
non-trading day) and ffill'd, taking a STALE pre-window price as the anchor and
overstating the benchmark's move. The fix anchors on the benchmark's own first
in-window observation. Pure math, network-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.export._perf_series import (
    _perf_window,
    _perf_full_series,
    _rebase_benchmark,
)
from tarzan.models.portfolio import PortfolioMetrics


def _bench():
    """A benchmark priced only on business days (no weekend quotes)."""
    idx = pd.date_range("2026-05-01", "2026-07-13", freq="B")
    # A clean ramp so the anchor value is unambiguous.
    return pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)


def test_rebase_anchors_on_first_in_window_observation():
    b = _bench()
    # Window index STARTS on a Saturday (2026-06-13) — the benchmark has no
    # quote that day. A naive ffill would anchor on the Friday (or earlier)
    # price; the fix must anchor on the first benchmark point >= the window
    # start instead.
    idx = pd.date_range("2026-06-13", "2026-07-13", freq="D")
    out = _rebase_benchmark(b, idx)
    assert out is not None
    # First value is exactly 0% (anchored, not a stale pre-window level).
    assert abs(out[0]) < 1e-9
    # Endpoint equals the benchmark's own return from its first in-window
    # observation to its last — NOT from a carried-forward earlier price.
    in_win = b[b.index >= idx[0]]
    expected = (float(in_win.iloc[-1]) / float(in_win.iloc[0]) - 1.0) * 100.0
    assert abs(out[-1] - expected) < 1e-6


def test_chart_benchmark_matches_period_return_definition():
    # The chart's 30-day ACWI endpoint must equal a plain 30-calendar-day
    # return computed on the benchmark's own index (the Performance-section
    # definition) — the two disagreed before the anchoring fix.
    b = _bench()
    val_idx = pd.date_range("2026-06-13", "2026-07-13", freq="D")  # starts on a Sat
    val = pd.Series(np.linspace(10000, 10500, len(val_idx)), index=val_idx)
    nav = pd.Series(np.linspace(100, 105, len(val_idx)), index=val_idx)
    m = PortfolioMetrics(total_value=10500.0, invested_value=10500.0, cash_value=0.0,
                         holdings_df=pd.DataFrame([{"cost_basis_eur": 10000.0}]))
    m.actual_value_series = val
    m.portfolio_history = nav
    m.pnl_series = pd.Series(np.linspace(0, 500, len(val_idx)), index=val_idx)
    m.benchmark_histories = {"ACWI": b}

    win = _perf_window(m, 30, "ACWI")
    chart_acwi = win["acwi"][-1]

    # Section definition: 30 calendar days on the benchmark's own series.
    cutoff = b.index[-1] - pd.Timedelta(days=30)
    sub = b[b.index >= cutoff]
    section_acwi = (float(sub.iloc[-1]) / float(sub.iloc[0]) - 1.0) * 100.0

    assert abs(chart_acwi - section_acwi) < 0.15, (
        f"chart {chart_acwi:.3f}% vs section {section_acwi:.3f}% — anchoring drift"
    )


def test_full_series_spans_inception_and_downsamples():
    idx = pd.date_range("2024-07-01", "2026-07-01", freq="B")  # 2 years
    m = PortfolioMetrics(total_value=14000.0, invested_value=14000.0, cash_value=0.0,
                         holdings_df=pd.DataFrame([{"cost_basis_eur": 10000.0}]))
    m.actual_value_series = pd.Series(np.linspace(10000, 14000, len(idx)), index=idx)
    m.portfolio_history = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
    m.pnl_series = pd.Series(np.linspace(0, 4000, len(idx)), index=idx)
    m.unrealized_series = m.pnl_series
    m.benchmark_histories = {"ACWI": pd.Series(np.linspace(200, 260, len(idx)), index=idx)}

    full = _perf_full_series(m, "ACWI")
    assert full is not None
    # Spans the whole history (starts at inception, not the last 30 days).
    assert full["dates"][0].date() == idx[0].date()
    assert full["dates"][-1].date() == idx[-1].date()
    # Downsampled to keep the SVG light.
    assert len(full["dates"]) <= 180
