"""The Portfolio section's KPI matrix: measures as rows, Returns windows as columns,
for the book and for the target. The portfolio row left the Returns table for it."""

import numpy as np
import pandas as pd

from tarzan.engine.metrics import MetricsEngine
from tarzan.export.newsletter._sections_perf import _kpi_matrix_html


def _nav(n, drift, vol, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)


def test_risk_figures_start_at_one_month_and_returns_reuse_performance_full():
    engine = object.__new__(MetricsEngine)
    engine._alpha_beta_benchmark = lambda ctx: None
    ctx = {"portfolio_history_full": _nav(300, 0.0004, 0.01, 1),
           "target_history": _nav(300, 0.0003, 0.01, 2),
           "performance_full": {"1d": 0.5, "5d": 1.0, "1m": 2.0, "3m": 3.0, "ytd": 4.0}}
    engine._window_kpis(ctx)
    port = ctx["window_kpis"]["portfolio"]
    assert port["1m"]["return"] == 2.0                      # the Returns row's figure
    assert "volatility" not in port["5d"] and "volatility" in port["1m"]
    assert "1y" not in port                                  # not in performance_full
    assert "1y" in ctx["window_kpis"]["target"]              # 300 sessions reach 1Y


def test_matrix_renders_both_blocks_with_window_columns():
    kpis = {"portfolio": {"1d": {"return": -0.1}, "1m": {"return": -1.9, "volatility": 12.4,
                                                         "sharpe": -1.8, "max_drawdown": -4.1,
                                                         "vs_benchmark": -0.9}},
            "target": {"1m": {"return": -1.8, "sharpe": 0.5}}}
    html = _kpi_matrix_html(kpis, ["5d", "1m", "3m", "ytd", "1y", "3y", "5y"], "1D",
                            bench_name="iShares MSCI ACWI")
    for text in ("Portfolio", "Target", "Volatility", "Sharpe", "Max drawdown",
                 "vs iShares MSCI ACWI", "5Y", "−1.80", "12.4"):
        assert text in html, text
    assert "−" in html and "pp" in html
