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


def test_window_twror_matches_engine_period_return():
    # _window_twror must equal the engine's compute_period_return (single
    # convention): the matrix cell and performance_full must never disagree.
    from tarzan.export._perf_series import _window_twror, _norm_series
    from tarzan.engine.stats import compute_period_return
    idx = pd.date_range("2026-04-01", "2026-07-13", freq="B")
    nav = pd.Series(np.linspace(100, 108, len(idx)), index=idx)
    for days in (7, 30, 90):
        assert _window_twror(_norm_series(nav), days) == compute_period_return(_norm_series(nav), days)


# ── CONTRACT: newsletter numbers == engine authoritative fields ──────────────
# These are the single-source-of-truth guarantee. They build a real
# order-derived PortfolioMetrics (network stubbed) with a benchmark that has
# history, then assert the newsletter's chart/legend numbers equal the engine's
# authoritative scalars. If a future change re-introduces an independent
# computation, one of these fails.

import datetime as _dt  # noqa: E402
import pytest  # noqa: E402
from tarzan import orchestrator  # noqa: E402

_C_AS_OF = _dt.date(2026, 7, 13)
_C_ORDERS = (
    "date,type,isin,quantity,gross_eur,net_eur,currency,price_native,fx_rate\n"
    "2025-07-01,buy,IE00B4L5Y983,100,10000,-10000,EUR,100,1.0\n"
    "2025-07-01,buy,IE00B4WXJJ64,50,5000,-5000,EUR,100,1.0\n"
    "2026-01-05,buy,IE00B4L5Y983,20,2400,-2400,EUR,120,1.0\n"
)


@pytest.fixture
def _contract_metrics(tmp_path, monkeypatch):
    from tarzan.models.holding import AssetClass, Geography
    # Deterministic per-instrument price ramps over a full year.
    hist_idx = pd.date_range("2025-07-01", "2026-07-13", freq="D")

    def _stub_enrich(holdings):
        meta = {"IE00B4L5Y983": ("Equities", "USA", 100.0, 1.25),
                "IE00B4WXJJ64": ("Fixed Income", None, 100.0, 1.06)}
        for h in holdings:
            ac_s, geo_s, p0, mult = meta.get(h.isin, ("Equities", "USA", 100.0, 1.1))
            s = pd.Series([p0 * (1 + (mult - 1) * i / (len(hist_idx) - 1))
                           for i in range(len(hist_idx))], index=hist_idx)
            h.price_history = s
            h.current_price = float(s.iloc[-1])
            h.current_value = h.quantity * h.current_price
            h.asset_class = {a.value: a for a in AssetClass}[ac_s]
            if geo_s:
                g = {gg.value: gg for gg in Geography}[geo_s]
                h.geography = g
                h.geo_breakdown = {g: 100.0}
            h.class_breakdown = {h.asset_class: 100.0}
        return holdings

    # A benchmark WITH history (business days only → weekend-start windows
    # exercise the anchoring), so benchmark_histories + holding_performance
    # both populate. Same series for both engine fetch entry points.
    bench = pd.Series(
        np.linspace(200.0, 230.0, len(pd.date_range("2025-07-01", "2026-07-13", freq="B"))),
        index=pd.date_range("2025-07-01", "2026-07-13", freq="B"))

    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history", lambda *a, **k: bench)
    monkeypatch.setattr("tarzan.engine.metrics._build_benchmark_series", lambda *a, **k: bench)
    # Route the geo benchmark name to our stub so benchmark_histories has it.
    monkeypatch.setattr("tarzan.engine.metrics.BENCHMARKS", {"MSCI ACWI": "ACWI"}, raising=False)
    monkeypatch.setattr("tarzan.engine.metrics.MetricsEngine._live_1d", lambda self, ctx: None)

    orders = tmp_path / "order_list.csv"
    orders.write_text(_C_ORDERS)
    metrics, _ = orchestrator.run(config_source=None, orders_source=str(orders),
                                  targets_per_holding_source=None,
                                  deterministic=True, as_of=_C_AS_OF)
    return metrics


def test_contract_matrix_twror_equals_performance_full(_contract_metrics):
    from tarzan.export._perf_series import _window_twror, _norm_series
    m = _contract_metrics
    pf = m.performance_full or {}
    nav = _norm_series(m.portfolio_history)
    for days, key in ((7, "1w"), (30, "1m")):
        chart = _window_twror(nav, days)
        eng = pf.get(key)
        if chart is not None and eng is not None:
            assert abs(chart - eng) < 1e-6, f"{key}: matrix {chart} != engine {eng}"


def test_contract_chart_twror_line_endpoint_matches_engine(_contract_metrics):
    from tarzan.export._perf_series import _perf_window
    m = _contract_metrics
    pf = m.performance_full or {}
    win = _perf_window(m, 30, None)
    if win and win.get("twror") and pf.get("1m") is not None:
        # The 30-day TWROR chart line's endpoint equals the authoritative 1m.
        assert abs(win["twror"][-1] - pf["1m"]) < 0.05



def test_perf_vol_series_full_and_window():
    from tarzan.export._perf_series import _perf_vol_series
    idx = pd.date_range("2024-07-01", "2026-07-01", freq="B")
    m = PortfolioMetrics(total_value=1.0, invested_value=1.0, cash_value=0.0,
                         holdings_df=pd.DataFrame([{"cost_basis_eur": 1.0}]))
    # A wobbly ramp so volatility is strictly positive.
    m.portfolio_history = pd.Series(
        np.linspace(100, 140, len(idx)) * (1 + 0.01 * np.sin(np.arange(len(idx)) / 5)), index=idx)
    m.benchmark_histories = {"ACWI": pd.Series(
        np.linspace(200, 260, len(idx)) * (1 + 0.02 * np.sin(np.arange(len(idx)) / 4)), index=idx)}

    full = _perf_vol_series(m, "ACWI", n_days=None)
    assert full and full["port"] and full["acwi"]
    assert full["dates"][0].date() == idx[0].date()   # spans inception
    assert len(full["dates"]) <= 180                   # downsampled
    assert all(v >= 0 for v in full["port"])           # vol is non-negative

    w30 = _perf_vol_series(m, "ACWI", n_days=30)
    assert w30 and w30["port"]
    # Bigger benchmark wobble → higher benchmark vol than portfolio.
    assert full["acwi"][-1] > full["port"][-1]


def test_chart_grid_density():
    # Left (month) grid labels every month; right (30d) grid gives >=12 ticks.
    import re
    from tarzan.export._charts import chart_pct_compact
    long_idx = pd.date_range("2025-11-15", "2026-07-13", freq="D")
    svg_l = chart_pct_compact([{"values": list(range(len(long_idx))), "color": "#000"}],
                              list(long_idx), include_zero=False, month_ticks=True)
    months = re.findall(r'>([A-Z][a-z]{2}(?: \d{2})?)<', svg_l)
    assert len(months) >= 8                            # ~9 months labelled

    short_idx = pd.date_range("2026-06-13", periods=23, freq="D")
    svg_r = chart_pct_compact([{"values": list(range(23)), "color": "#000"}],
                              list(short_idx), include_zero=True, min_day_ticks=12)
    days = re.findall(r'>([A-Z][a-z]{2} \d{2})<', svg_r)
    assert len(days) >= 12
