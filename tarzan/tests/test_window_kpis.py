"""The Portfolio section's KPI matrix: measures as rows, Returns windows as columns,
for the book and for the target, computed by the functions every other section
prints with (Vs-target windows, the risk block, the window P&L)."""

import numpy as np
import pandas as pd

from tarzan.export.newsletter import _sections_perf as sp


def test_returns_are_the_vs_target_endpoints_and_risk_uses_the_risk_block(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=120)
    nav = pd.Series(100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 120)), index=idx)

    class M:
        portfolio_history = nav
        target_history = nav * 1.0
        risk_free_daily = None
        pnl_series = None
        actual_value_series = None

    calls = []
    monkeypatch.setattr(sp, "_perf_window", lambda m, n, geo, bucket="1m": {
        "endpoints": {"twr": 1.23, "target": 4.56},
        "window_start": idx[-60], "window_end": idx[-1]})

    def risk(series, rf):
        calls.append((series.index[0], series.index[-1]))
        return {"volatility": 10.0, "sharpe": 0.5, "max_drawdown": -3.0}

    import tarzan.engine.stats as stats
    monkeypatch.setattr(stats, "risk_metric_row", risk)
    k = sp._window_kpis(M(), ["1d", "5d", "1m"], "ACWI")
    assert k["portfolio"]["1m"]["return"] == 1.23 and k["target"]["1m"]["return"] == 4.56
    assert "volatility" not in k["portfolio"]["5d"] and k["portfolio"]["1m"]["sharpe"] == 0.5
    assert calls and calls[0] == (idx[-60], idx[-1])      # the window the chart draws


def test_matrix_rows_and_euro_format():
    # synthetic figures
    kpis = {"portfolio": {"1d": {"pnl_eur": -310.0, "return": -0.1},
                          "1m": {"pnl_eur": 2000.0, "unr_eur": -1440.0, "return": -1.9,
                                 "volatility": 12.4, "sharpe": -1.8, "max_drawdown": -4.1}},
            "target": {"1m": {"return": -1.8, "sharpe": 0.5}}}
    html = sp._kpi_matrix_html(kpis, ["5d", "1m", "3m", "ytd", "1y", "3y", "5y"], "1D")
    # The unit stands in the row label (the grid's notation); cells print bare
    # figures, euros in thousands.
    for text in ("Portfolio", "Target", "Return", "Volatility", "Sharpe", "Max DD", "SI",
                 "−1.80", "−0.3k", "+2.0k", "−1.4k"):  # synthetic
        assert text in html, text
    assert ">vs <" not in html and "P&amp;L %" not in html and "Unr. %" not in html
    # 1Y · 3Y · 5Y open as columns only once the book has lived one of them;
    # until then one line says so.
    assert ">5Y<" not in html and "appear once the book has lived them" in html


def test_the_one_day_ter_is_todays_mix_not_a_lifetime_average():
    """A synthetic book whose TER rose from 0.15% to 0.32%: 1D reads today's
    0.32%, the since-inception column the average."""
    import pandas as pd
    from tarzan.export.newsletter import _sections_perf as sp

    class M:
        portfolio_history = None
        target_history = None
        benchmark_histories = None
        risk_free_daily = None
        pnl_series = None
        actual_value_series = None
        target_ter = None
        ter_timeline = pd.Series([0.15, 0.20, 0.32],
                                 index=pd.to_datetime(["2026-01-02", "2026-06-05", "2026-11-13"]))

    k = sp._window_kpis(M(), ["1d"], None)
    assert k["portfolio"]["1d"]["ter"] == 0.32
