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
    compute_ulcer_index,
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

    def test_cagr_ignores_edge_nans(self):
        """A leading/trailing NaN (providers append a bar for the not-yet-
        closed session) must not null the CAGR — the endpoints drive the whole
        ratio, so they are dropped first, matching the clean-series result."""
        idx = pd.date_range("2024-01-01", "2026-01-01", freq="D")
        clean = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        expected = compute_cagr(clean)

        lead = clean.copy(); lead.iloc[0] = np.nan
        trail = clean.copy(); trail.iloc[-1] = np.nan
        assert compute_cagr(lead) == pytest.approx(expected, rel=1e-3)
        assert compute_cagr(trail) == pytest.approx(expected, rel=1e-3)


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


class TestUlcerIndex:
    def test_monotonic_increase_is_zero(self):
        # Only ever making new highs → no drawdown → Ulcer = 0.
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        series = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190], index=idx)
        assert compute_ulcer_index(series) == pytest.approx(0.0, abs=1e-9)

    def test_known_single_drawdown(self):
        # Flat at 100 then one day at 90 (−10% drawdown) then back.
        # Drawdowns (%): 0, 0, -10, 0, 0 → RMS = sqrt(100/5) = sqrt(20) ≈ 4.472.
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        series = pd.Series([100.0, 100.0, 90.0, 100.0, 100.0], index=idx)
        assert compute_ulcer_index(series) == pytest.approx(math.sqrt(20.0), abs=1e-6)

    def test_deeper_or_longer_drawdown_scores_higher(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        shallow = pd.Series([100, 100, 95, 100, 100, 100], index=idx)
        deep = pd.Series([100, 100, 70, 70, 70, 100], index=idx)
        assert compute_ulcer_index(deep) > compute_ulcer_index(shallow)

    def test_empty_series_returns_zero(self):
        assert compute_ulcer_index(pd.Series(dtype=float)) == 0.0


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

    def test_sharpe_golden_value(self):
        """Golden: Sharpe = (annual_return − RISK_FREE_RATE) / annual_vol,
        pinned to the exact magnitude so a wrong risk-free rate or a broken
        excess-return definition fails the suite (not just a NaN check).

        With the live RFR (4%): (10 − 4) / 15 = 0.40.
        """
        from tarzan.engine.stats import RISK_FREE_RATE
        result = compute_sharpe(annual_return=10.0, annual_volatility=15.0)
        expected = (10.0 - RISK_FREE_RATE) / 15.0
        assert result == pytest.approx(expected)
        # Guard the RFR itself: if config drifts, this documents the assumption.
        assert RISK_FREE_RATE == pytest.approx(4.0)
        assert result == pytest.approx(0.40)


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
        assert not math.isnan(result)

    def test_sortino_golden_value(self):
        """Golden: Sortino = (annual_return − RISK_FREE_RATE) / downside_dev,
        where downside_dev is the annualized RMS shortfall below the daily
        risk-free target (target semideviation, Sortino & Price 1994). Pin the
        exact value so a wrong annualization factor or a switched-in
        negative-only-std denominator is caught, not silently shipped.
        """
        import numpy as np
        from tarzan.engine.stats import RISK_FREE_RATE, TRADING_DAYS

        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        annual_return = 10.0
        # Recompute the textbook denominator independently.
        target_daily = RISK_FREE_RATE / 100.0 / TRADING_DAYS
        shortfall = (returns - target_daily).clip(upper=0.0)
        downside = float((shortfall ** 2).mean()) ** 0.5 * np.sqrt(TRADING_DAYS) * 100
        expected = (annual_return - RISK_FREE_RATE) / downside

        result = compute_sortino(returns, annual_return=annual_return)
        assert result == pytest.approx(expected)
        # Anchor the magnitude too (guards the whole formula, not just its shape).
        assert result == pytest.approx(0.3744, abs=1e-3)


class TestRiskMetricRowRiskFree:
    """risk_metric_row is the single source of truth for the risk block; it must
    use the time-varying risk-free path when given one, and stay identical to the
    scalar behaviour when not (so pinned/offline runs are unchanged)."""

    def _ramp(self):
        # A price series with genuine daily variation so Sharpe/Sortino are
        # finite and rate-sensitive (a pure linear ramp has ~zero daily vol).
        from tarzan.engine.stats import risk_metric_row
        idx = pd.date_range("2020-01-01", periods=260, freq="B")
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0006, 0.01, len(idx))
        prices = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
        return risk_metric_row, prices

    def test_none_matches_scalar_sharpe_sortino(self):
        risk_metric_row, prices = self._ramp()
        row = risk_metric_row(prices)  # rf_daily default None
        cagr = compute_cagr(prices)
        vol = float(prices.pct_change().dropna().std()) * math.sqrt(252) * 100
        assert row["sharpe"] == pytest.approx(compute_sharpe(cagr, vol))
        assert row["sortino"] == pytest.approx(
            compute_sortino(prices.pct_change().dropna(), cagr))

    def test_timevarying_rf_changes_sharpe(self):
        risk_metric_row, prices = self._ramp()
        daily_ret = prices.pct_change().dropna()
        # A ~0% flat risk-free path (vs the scalar 4%) must lift Sharpe: less
        # is subtracted from every daily return. Proves the tv branch is live.
        rf_zero = pd.Series(0.0, index=daily_ret.index)
        scalar = risk_metric_row(prices)["sharpe"]
        tv = risk_metric_row(prices, rf_zero)["sharpe"]
        assert tv != pytest.approx(scalar)
        assert tv > scalar


class TestAlphaRiskFree:
    """Jensen's alpha must use the real (time-varying) risk-free, not the
    hardcoded 4% scalar. rf enters the CAPM regression as a level, so the
    window-mean annual % (rf_annual_pct) is the correct scalar; None keeps the
    documented RISK_FREE_RATE fallback (pinned/offline runs unchanged)."""

    def test_rf_annual_pct_collapses_series_to_window_mean(self):
        from tarzan.engine.stats import rf_annual_pct, TRADING_DAYS
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        s = pd.Series(0.02 / TRADING_DAYS, index=idx)  # flat 2%/yr daily path
        assert rf_annual_pct(s) == pytest.approx(2.0)
        assert rf_annual_pct(None) is None            # → RISK_FREE_RATE fallback
        assert rf_annual_pct(pd.Series(dtype=float)) is None
        assert rf_annual_pct(3.5) == 3.5              # scalar passthrough

    def test_alpha_shifts_with_risk_free(self):
        # alpha = port.mean − β·bench.mean − rf·(1−β): with β≠1 a different rf
        # moves alpha. A ~0% rf (vs the 4% default) must change the number.
        from tarzan.engine.stats import _compute_beta_alpha
        idx = pd.date_range("2018-01-01", periods=520, freq="B")
        rng = np.random.default_rng(1)
        bench_r = rng.normal(0.0004, 0.008, len(idx))
        bench = pd.Series(100.0 * np.cumprod(1 + bench_r), index=idx)
        # Defensive port (β≈0.5): half the benchmark's moves plus idiosyncratic.
        port_r = 0.5 * bench_r + rng.normal(0.0002, 0.004, len(idx))
        port = pd.Series(100.0 * np.cumprod(1 + port_r), index=idx)
        _, alpha_default = _compute_beta_alpha(port, bench, 5.0)          # rf=4%
        _, alpha_zero = _compute_beta_alpha(port, bench, 5.0, risk_free=0.0)
        assert alpha_zero != pytest.approx(alpha_default)


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
    """Windows are measured back from the run's today, so each fixture pins it
    to its own last observation."""

    def test_period_return_flat_series_is_zero(self, monkeypatch):
        idx = pd.date_range("2024-01-01", periods=365, freq="D")
        series = pd.Series([100.0] * 365, index=idx)
        monkeypatch.setattr("tarzan.runtime.today", lambda: idx[-1].date())
        assert compute_period_return(series, "1m") == 0.0

    def test_period_return_1d_uses_last_two_sessions(self, monkeypatch):
        idx = pd.bdate_range("2024-01-01", periods=5)
        series = pd.Series([100, 101, 102, 103, 105], index=idx)
        monkeypatch.setattr("tarzan.runtime.today", lambda: idx[-1].date())
        # 1d: (105/103 - 1) * 100 = 1.94%
        result = compute_period_return(series, "1d")
        assert abs(result - 1.9417) < 0.01

    def test_period_return_empty_returns_none(self):
        assert compute_period_return(pd.Series(dtype=float), "1m") is None


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

    def test_ytd_anchors_to_prior_year_close(self):
        """YTD is measured from Dec-31, not the first in-year observation.

        A book that rose from the prior-year close before its first
        current-year point must report the full move, not ~0%.
        """
        idx = pd.to_datetime(["2025-12-31", "2026-01-05", "2026-02-01"])
        series = pd.Series([100.0, 105.0, 110.0], index=idx)
        # base = prior-year close 100 → (110/100 - 1) = 10%, NOT (110/105-1).
        assert compute_ytd_return(series) == pytest.approx(10.0)

    def test_ytd_mid_year_inception_uses_first_in_year(self):
        """With no prior-year data (mid-year start) fall back to the first
        in-year observation (needs >=2 points)."""
        idx = pd.to_datetime(["2026-02-10", "2026-03-01"])
        series = pd.Series([100.0, 108.0], index=idx)
        assert compute_ytd_return(series) == pytest.approx(8.0)


class TestBusinessDayAnnualization:
    """The order-path NAV is a dense calendar-day series; volatility is
    annualized with sqrt(252), so it must be resampled to business days
    first or weekend zero-returns understate vol ~17%."""

    def _calendar_nav(self):
        import numpy as np
        rng = np.random.default_rng(0)
        days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        vals = [100.0]
        for d in days[1:]:
            if d.weekday() < 5:  # weekday: a real ~1%/day move
                vals.append(vals[-1] * (1 + rng.normal(0, 0.01)))
            else:                # weekend: carried flat (freq='D' artifact)
                vals.append(vals[-1])
        return pd.Series(vals, index=days)

    def test_resample_drops_weekend_flats(self):
        from tarzan.engine.stats import to_business_day_series
        cal = self._calendar_nav()
        bday = to_business_day_series(cal)
        # ~252 business days, not 366 calendar days.
        assert 250 <= len(bday) <= 262
        # No weekend rows survive.
        assert all(ts.weekday() < 5 for ts in bday.index)

    def test_calendar_days_understate_vol_business_days_fix_it(self):
        import numpy as np
        from tarzan.engine.stats import to_business_day_series, TRADING_DAYS
        cal = self._calendar_nav()

        def annvol(s):
            r = s.pct_change().dropna()
            return float(r.std()) * np.sqrt(TRADING_DAYS) * 100

        vol_cal = annvol(cal)
        vol_bday = annvol(to_business_day_series(cal))
        true_vol = 0.01 * np.sqrt(TRADING_DAYS) * 100  # ~15.87%
        # Calendar-day annualization is materially low; business-day matches.
        assert vol_cal < true_vol * 0.92
        assert abs(vol_bday - true_vol) < true_vol * 0.12
        assert vol_bday > vol_cal


class TestDrawdownZeroPeakGuard:
    def test_leading_zero_series_yields_finite_drawdown(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="D")
        s = pd.Series([0.0, 0.0, 100.0, 90.0, 120.0, 80.0], index=idx)
        dd = compute_max_drawdown(s)
        assert math.isfinite(dd)
        # Worst peak-to-trough among positive points: 120 → 80 = -33.3%.
        assert dd == pytest.approx(-1.0 / 3.0, abs=1e-3)
        assert math.isfinite(compute_ulcer_index(s))


# ---------------------------------------------------------------------------
# Production-readiness bug exploration: C4 canonical exposure
# ---------------------------------------------------------------------------

from hypothesis import given as _given, settings as _settings, strategies as _st  # noqa: E402


# **Validates: Requirements 2.4**
@_given(capital_eur=_st.integers(min_value=100, max_value=100_000))
@_settings(max_examples=5, deadline=None, derandomize=True)
def test_c4_report_optimizer_and_verifier_share_90_60_notional_exposure(capital_eur):
    """Property 1 / C4 exploration across three current consumers.

    Reporting correctly preserves 90% equity + 60% fixed-income notional
    exposure over invested capital (150% total).  The optimizer and verifier
    instead reconstruct 100% equity + 0% fixed income from the primary class.
    """
    from tarzan.engine.metrics import MetricsEngine
    from tarzan.engine.rebalancer import _ObjectiveModel, _verify
    from tarzan.models.holding import AssetClass, Holding
    from tarzan.models.investor_config import InvestorConfig

    capital = float(capital_eur)
    holding = Holding(
        isin="LEV", ticker="LEV", quantity=capital / 100.0,
        cost_basis_eur=capital, market_value_eur=capital, currency="EUR",
        current_price=100.0, current_value=capital,
        asset_class=AssetClass.EQUITIES,
        class_breakdown={
            AssetClass.EQUITIES: 90.0,
            AssetClass.FIXED_INCOME: 60.0,
        },
    )
    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {
        "Equities": 90.0,
        "Fixed Income": 60.0,
    }
    config.equity_geo_targets_pctg = {}
    config.target_cash_buffer_eur = 0.0

    engine = MetricsEngine([holding], config)
    ctx = {}
    engine._valuation(ctx)
    engine._allocations(ctx)
    reporting = {
        str(row.category): float(row.weight_pct)
        for row in ctx["allocation_by_class"].itertuples()
    }

    values = np.array([capital], dtype=float)
    model = _ObjectiveModel([holding], config, values)
    gaps = model.gaps(values)
    optimizer = {
        key: float(target + gap)
        for key, target, gap in zip(model.ac_keys, model.ac_targets, gaps)
    }
    verification = _verify(
        values, [holding], config, model.geo_frac, model.all_geos
    )
    asset_check = next(v for v in verification if v.get("kind") == "asset")
    verifier = {
        item["category"]: float(item["actual_pct"])
        for item in asset_check["items"]
    }

    expected = {"Equities": 90.0, "Fixed Income": 60.0}
    assert reporting == pytest.approx(expected, abs=1e-6)
    assert sum(reporting.values()) == pytest.approx(150.0, abs=1e-6), (
        "150% is intentional notional exposure and must not be normalized or rejected"
    )
    mismatches = {
        name: surface
        for name, surface in (("optimizer", optimizer), ("verifier", verifier))
        if surface != pytest.approx(reporting, abs=1e-6)
    }
    assert mismatches == {}, (
        "same holding has contradictory exposure across consumers; "
        f"reporting={reporting}, mismatches={mismatches}"
    )
