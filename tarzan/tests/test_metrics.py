"""Tests for engine/metrics.py core calculations.

Focus on pure math functions that don't require yfinance/network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.metrics import (
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_period_return,
    compute_sharpe,
    compute_sortino,
    compute_var,
    compute_ytd_return,
)


class TestCAGR:
    def test_cagr_simple_doubling_over_2_years(self):
        """Portfolio doubles in 2 years → CAGR ≈ 41.42%."""
        idx = pd.date_range("2024-01-01", "2026-01-01", freq="D")
        values = np.linspace(100, 200, len(idx))
        series = pd.Series(values, index=idx)

        cagr = compute_cagr(series)

        # CAGR = (200/100)^(1/2) - 1 = 41.42%
        assert 40.0 < cagr < 43.0

    def test_cagr_no_growth(self):
        """Flat series → CAGR = 0."""
        idx = pd.date_range("2024-01-01", "2026-01-01", freq="D")
        series = pd.Series([100.0] * len(idx), index=idx)

        cagr = compute_cagr(series)

        assert abs(cagr) < 0.01

    def test_cagr_empty_series_returns_zero(self):
        assert compute_cagr(pd.Series(dtype=float)) == 0.0

    def test_cagr_single_point_returns_zero(self):
        series = pd.Series([100.0], index=[pd.Timestamp("2024-01-01")])
        assert compute_cagr(series) == 0.0

    def test_cagr_negative_start_returns_zero(self):
        idx = pd.date_range("2024-01-01", "2026-01-01", freq="D")
        values = [-100] + [100] * (len(idx) - 1)
        series = pd.Series(values, index=idx)
        assert compute_cagr(series) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        """Peak 100 → trough 75 → recovery → MDD = -25%."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        values = [100, 110, 120, 90, 75, 85, 95, 105, 115, 130]
        series = pd.Series(values, index=idx)

        mdd = compute_max_drawdown(series)

        # Peak 120 → trough 75: drawdown = (75-120)/120 = -0.375
        assert abs(mdd - (-0.375)) < 0.01

    def test_no_drawdown_monotonic_increase(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        series = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190], index=idx)

        mdd = compute_max_drawdown(series)

        assert abs(mdd) < 0.001

    def test_empty_series_returns_zero(self):
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0


class TestSharpe:
    def test_sharpe_with_zero_volatility_returns_nan(self):
        """Division by zero must be guarded."""
        result = compute_sharpe(annual_return=10.0, annual_volatility=0.0)
        assert math.isnan(result)

    def test_sharpe_with_negative_volatility_returns_nan(self):
        result = compute_sharpe(annual_return=10.0, annual_volatility=-5.0)
        assert math.isnan(result)

    def test_sharpe_standard_calculation(self):
        """Return 10%, vol 15%, default RFR — Sharpe is finite."""
        result = compute_sharpe(annual_return=10.0, annual_volatility=15.0)
        assert not math.isnan(result)
        assert isinstance(result, float)


class TestSortino:
    def test_sortino_all_positive_returns_nan(self):
        """No downside → undefined."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = compute_sortino(returns, annual_return=10.0)
        assert math.isnan(result)

    def test_sortino_mixed_returns_computes(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        result = compute_sortino(returns, annual_return=10.0)
        assert isinstance(result, float)


class TestVaR:
    def test_var_insufficient_data_returns_nan(self):
        returns = pd.Series([0.01, 0.02])  # fewer than 5
        assert math.isnan(compute_var(returns))

    def test_var_95_percentile(self):
        """VaR 95% = 5th percentile of returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        var = compute_var(returns, confidence=0.95)

        expected = returns.quantile(0.05)
        assert abs(var - expected) < 0.001

    def test_cvar_lower_than_var(self):
        """CVaR (expected loss in tail) should be more negative than VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        var = compute_var(returns, confidence=0.95)
        cvar = compute_cvar(returns, confidence=0.95)

        assert cvar <= var


class TestPeriodReturn:
    def test_period_return_flat_series_is_zero(self):
        idx = pd.date_range("2024-01-01", periods=365, freq="D")
        series = pd.Series([100.0] * 365, index=idx)
        assert compute_period_return(series, days=30) == 0.0

    def test_period_return_1d_uses_last_two_points(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        series = pd.Series([100, 101, 102, 103, 105], index=idx)
        # 1d: (105/103 - 1) * 100 = 1.94%
        result = compute_period_return(series, days=1)
        assert abs(result - 1.9417) < 0.01

    def test_period_return_empty_returns_none(self):
        assert compute_period_return(pd.Series(dtype=float), days=30) is None


class TestYTD:
    def test_ytd_return_simple(self):
        """YTD from Jan 1 start value to current value."""
        idx = pd.date_range("2026-01-01", "2026-03-01", freq="D")
        values = np.linspace(100, 110, len(idx))
        series = pd.Series(values, index=idx)

        ytd = compute_ytd_return(series)

        # (110 - 100) / 100 * 100 = ~10%
        assert 8 < ytd < 12

    def test_ytd_insufficient_data_returns_none(self):
        idx = pd.date_range("2026-01-01", periods=1, freq="D")
        series = pd.Series([100.0], index=idx)
        assert compute_ytd_return(series) is None
