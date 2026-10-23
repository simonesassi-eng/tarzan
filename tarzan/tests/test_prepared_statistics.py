"""Numerical, ownership and work-count gates for row-owned statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import stats
from tarzan.engine.benchmarks import _populate_perf_row


def _reference_risk_row(series, rf_daily=None):
    """Independent pre-preparation expressions from 4da5acf."""
    cagr = stats.compute_cagr(series)
    daily_ret = series.pct_change().dropna()
    vol = (
        float(daily_ret.std()) * np.sqrt(stats.TRADING_DAYS) * 100
        if len(daily_ret) > 0 else 0.0
    )
    if rf_daily is not None and len(daily_ret) > 0:
        aligned = stats._align_rf_daily(daily_ret, rf_daily)
        excess = (daily_ret - aligned).dropna()
        sd = float(excess.std())
        sharpe = (
            float("nan") if excess.empty or sd <= 0
            else float(excess.mean()) / sd * np.sqrt(stats.TRADING_DAYS)
        )
        if excess.empty:
            sortino = float("nan")
        else:
            downside = excess.clip(upper=0.0)
            dd = (
                float((downside ** 2).mean()) ** 0.5
                * np.sqrt(stats.TRADING_DAYS)
            )
            sortino = (
                float("nan") if dd <= 0
                else float(excess.mean()) * stats.TRADING_DAYS / dd
            )
    else:
        sharpe = stats.compute_sharpe(cagr, vol)
        sortino = (
            stats.compute_sortino(daily_ret, cagr)
            if len(daily_ret) > 0 else float("nan")
        )
    max_drawdown = ulcer = 0.0
    if not series.empty and len(series) >= 2:
        positive = series[series > 0]
        if len(positive) >= 2:
            peak = positive.cummax()
            drawdowns = (positive - peak) / peak
            max_drawdown = float(drawdowns.min())
            drawdowns_pct = (positive - peak) / peak * 100.0
            ulcer = float(np.sqrt((drawdowns_pct ** 2).mean()))
    var = stats.compute_var(daily_ret, 0.95)
    # The reference independently computes the VaR again for CVaR, just as
    # the old public function did; never read the prepared cutoff here.
    if daily_ret.empty or len(daily_ret) < 5:
        cvar = float("nan")
    else:
        cvar_var = stats.compute_var(daily_ret, 0.95)
        tail = daily_ret[daily_ret <= cvar_var]
        cvar = float(tail.mean()) if not tail.empty else cvar_var
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown * 100,
        "ulcer_index": ulcer,
        "var_95": stats._scale_or_nan(var, 100),
        "cvar_95": stats._scale_or_nan(cvar, 100),
    }


def _assert_exact_row(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        a, e = actual[key], expected[key]
        if np.isnan(a) and np.isnan(e):
            continue
        dtype = np.complex128 if np.iscomplexobj(e) else np.float64
        assert dtype(a).tobytes() == dtype(e).tobytes(), (key, a, e)


def _prices(case, timezone):
    if case == "random":
        rng = np.random.default_rng(31607)
        values = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 400))
    else:
        values = {
            "empty": [],
            "single": [100.0],
            "two": [100.0, 101.0],
            "flat": [100.0] * 20,
            "missing": [np.nan, 100.0, np.nan, 103.0, 98.0, np.nan, 105.0],
            "all_nan": [np.nan] * 10,
            "nonpositive": [-2.0, 0.0, 100.0, 95.0, np.nan, 101.0],
            "all_nonpositive": [-2.0, 0.0, -1.0, 0.0],
            "inf": [100.0, 105.0, np.inf, 103.0, 100.0, 104.0, 107.0],
            "duplicate_dates": [100.0, 110.0, 105.0, 103.0, 108.0, 101.0],
        }[case]
    index = pd.bdate_range(
        "2025-10-23", periods=len(values), tz=timezone, name="session"
    )
    if case == "duplicate_dates":
        index = index.where(
            np.arange(len(index)) != 1, index[0]
        )
    prices = pd.Series(values, index=index, name="Close", dtype=float)
    prices.attrs["resolved_ticker"] = "EXUS.MI"
    return prices


@pytest.mark.parametrize("case", [
    "empty", "single", "two", "flat", "missing", "all_nan",
    "nonpositive", "all_nonpositive", "inf", "duplicate_dates", "random",
])
@pytest.mark.parametrize("timezone", [
    None, "Europe/Rome", "America/New_York",
])
@pytest.mark.parametrize("rf_kind", ["none", "scalar", "path"])
def test_preparation_matches_original_risk_row_exactly(
    case, timezone, rf_kind,
):
    prices = _prices(case, timezone)
    if rf_kind == "none":
        risk_free = None
    elif rf_kind == "scalar":
        risk_free = 3.75
    else:
        # Different timezone, duplicate session labels and an internal gap:
        # alignment keeps the venue's date, last duplicate and existing fill.
        index = pd.DatetimeIndex([
            "2025-10-20", "2025-10-24", "2025-10-24", "2026-01-12",
        ], tz="Asia/Tokyo")
        risk_free = pd.Series([0.0001, 0.0002, 0.0003, np.nan], index=index)
    before = prices.copy(deep=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = _reference_risk_row(prices, risk_free)
        actual = stats.risk_metric_row(prices, risk_free)
    _assert_exact_row(actual, expected)
    pd.testing.assert_series_equal(prices, before, check_exact=True)


def test_returns_are_computed_before_dropping_missing_prices():
    prices = pd.Series(
        [100.0, np.nan, 110.0], index=pd.bdate_range("2026-08-24", periods=3)
    )
    prepared = stats.PreparedStatistics(prices)
    pd.testing.assert_series_equal(
        prepared.daily_returns, prices.pct_change().dropna(), check_exact=True
    )
    assert len(prepared.daily_returns) == 2
    assert prepared.daily_returns.iloc[0] == 0.0
    assert len(prices.dropna().pct_change().dropna()) == 1


def test_one_alignment_drawdown_and_quantile_per_risk_row(monkeypatch):
    prices = _prices("random", "Europe/Rome")
    rf = pd.Series(0.0001, index=prices.index)
    calls = {"rf": 0, "drawdown": 0, "quantile": 0}
    original_align = stats._align_rf_daily
    original_drawdowns = stats._positive_drawdowns
    original_quantile = pd.Series.quantile

    def align(*args, **kwargs):
        calls["rf"] += 1
        return original_align(*args, **kwargs)

    def drawdowns(*args, **kwargs):
        calls["drawdown"] += 1
        return original_drawdowns(*args, **kwargs)

    def quantile(self, *args, **kwargs):
        calls["quantile"] += 1
        return original_quantile(self, *args, **kwargs)

    monkeypatch.setattr(stats, "_align_rf_daily", align)
    monkeypatch.setattr(stats, "_positive_drawdowns", drawdowns)
    monkeypatch.setattr(pd.Series, "quantile", quantile)
    stats.risk_metric_row(prices, rf)
    assert calls == {"rf": 1, "drawdown": 1, "quantile": 1}


def test_performance_row_reuses_daily_returns(monkeypatch):
    prices = _prices("random", "Europe/Rome")
    calls = []
    original = pd.Series.pct_change

    def observe(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "pct_change", observe)
    row = {}
    _populate_perf_row(row, prices, pd.Series(dtype=float))
    assert len(calls) == 1
    assert row["alpha"] != row["alpha"]


class _EvidenceMustNotBeCopied:
    def __deepcopy__(self, memo):
        raise AssertionError("prepared statistics copied source evidence")


def test_preparation_owns_numeric_values_and_keeps_source_evidence():
    prices = _prices("random", "Europe/Rome").set_flags(
        allows_duplicate_labels=False
    )
    rf = pd.Series(0.0001, index=prices.index)
    evidence = _EvidenceMustNotBeCopied()
    prices.attrs["tarzan_price_origins"] = evidence
    rf.attrs["native_source_evidence"] = evidence
    prepared = stats.PreparedStatistics(prices, rf)
    expected_prices = prepared.series.copy(deep=True)
    expected_rf = prepared.rf_daily.copy(deep=True)
    prices.iloc[0] = -999.0
    rf.iloc[0] = 999.0
    prices.index.values[0] += np.timedelta64(1, "D")
    pd.testing.assert_series_equal(
        prepared.series, expected_prices, check_exact=True
    )
    pd.testing.assert_series_equal(
        prepared.rf_daily, expected_rf, check_exact=True
    )
    assert prepared.series.flags.allows_duplicate_labels is False
    assert prepared.series.attrs == {"resolved_ticker": "EXUS.MI"}
    assert prices.attrs["tarzan_price_origins"] is evidence
    assert rf.attrs["native_source_evidence"] is evidence


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float64", "Int64"])
def test_numeric_dtype_and_missing_values_are_not_coerced(dtype):
    prices = pd.Series(
        [100, None, 103, 97, 100, 102, 101, 104],
        index=pd.bdate_range("2026-03-23", periods=8), dtype=dtype,
    )
    prepared = stats.PreparedStatistics(prices)
    assert prepared.series.dtype == prices.dtype
    _assert_exact_row(
        prepared.risk_metrics(), _reference_risk_row(prices)
    )
