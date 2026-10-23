"""Native risk entrypoint forwards RF evidence without globally averaging it.

All prices/rates are synthetic. The window oracle independently carries rate
events onto the actual consumer sessions; it never calls the RF aligner.
"""

from bisect import bisect_right
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import metrics, stats


def _prices(timezone=None):
    index = pd.bdate_range(end="2024-08-12", periods=161, tz=timezone)
    rng = np.random.default_rng(231024)
    market = rng.normal(0.0005, 0.008, len(index))
    portfolio = 0.0002 + 0.6 * market + rng.normal(0.0, 0.002, len(index))
    return (
        pd.Series(100.0 * np.cumprod(1.0 + portfolio), index=index),
        pd.Series(100.0 * np.cumprod(1.0 + market), index=index),
    )


def _naive(series):
    result = series.copy(deep=True)
    index = pd.DatetimeIndex(result.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result


def _native_ols_oracle(portfolio, benchmark, source):
    prices = _naive(portfolio)
    market = _naive(benchmark)
    weekly = pd.DataFrame({
        "port": prices.resample("W-FRI").last().dropna(),
        "bench": market.resample("W-FRI").last().dropna(),
    }).dropna()
    returns = weekly.pct_change().dropna()
    native = prices.pct_change().dropna()
    end = min(weekly.index[-1], prices.index.max())
    sessions = native.index[(native.index > weekly.index[0]) & (native.index <= end)]
    by_day = dict(zip(_naive(source).index, source.array))
    known = sorted((day, value) for day, value in by_day.items() if not pd.isna(value))
    known_days = [day for day, _ in known]
    carried = []
    for day in sessions:
        position = bisect_right(known_days, day)
        carried.append(known[position - 1][1] if position else np.nan)
    path = pd.Series(carried, index=sessions, dtype=source.dtype)
    if path.isna().any():
        return (
            float(returns["port"].cov(returns["bench"]) / returns["bench"].var()),
            float("nan"),
        )
    annual = float(path.mean()) * stats.TRADING_DAYS * 100.0
    rf_weekly = annual / 100.0 / 52.0
    port_excess = returns["port"] - rf_weekly
    bench_excess = returns["bench"] - rf_weekly
    beta = port_excess.cov(bench_excess) / bench_excess.var()
    alpha = (port_excess.mean() - beta * bench_excess.mean()) * 52.0 * 100.0
    return float(beta), float(alpha)


def _engine(monkeypatch, benchmark, source):
    engine = metrics.MetricsEngine.__new__(metrics.MetricsEngine)
    reads = []
    calls = []

    def get_source(context):
        reads.append(context)
        return source

    original = metrics._compute_beta_alpha

    def inspect(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    engine._rf_daily = get_source
    engine._alpha_beta_benchmark = lambda _: SimpleNamespace(history=benchmark)
    monkeypatch.setattr(metrics, "_compute_beta_alpha", inspect)
    return engine, reads, calls


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
@pytest.mark.parametrize("coverage", ["complete", "missing-left", "all-missing"])
def test_risk_entrypoint_scopes_original_path_to_native_regression_sessions(
    monkeypatch, timezone, dtype, coverage,
):
    portfolio, benchmark = _prices(timezone)
    index = portfolio.index
    source = pd.Series(
        [0.04, 0.00009, 0.000025, 0.3, 0.5],
        index=pd.DatetimeIndex([
            index[0] - pd.Timedelta(days=1500),
            index[0] - pd.Timedelta(days=7),
            index[len(index) // 2],
            index[-1] + pd.Timedelta(days=3),
            index[-1] + pd.Timedelta(days=1000),
        ]),
        dtype=dtype,
    )
    if coverage == "missing-left":
        source = source.iloc[2:].copy()
    elif coverage == "all-missing":
        source.iloc[:] = pd.NA if dtype != "float64" else np.nan
    before = source.copy(deep=True)
    original_prices = portfolio.copy(deep=True)
    engine, reads, calls = _engine(monkeypatch, benchmark, source)
    context = {"portfolio_history": portfolio}
    engine._risk(context)
    assert reads == [context]
    assert len(calls) == 1
    assert calls[0][1]["rf_daily"] is source
    assert calls[0][1]["risk_free"] is None
    assert calls[0][1]["input_kind"] == "prices"
    beta, alpha = _native_ols_oracle(portfolio, benchmark, source)
    assert context["risk"]["beta"] == pytest.approx(beta, rel=1e-12, abs=1e-12)
    if np.isnan(alpha):
        assert np.isnan(context["risk"]["alpha"])
    else:
        assert context["risk"]["alpha"] == pytest.approx(alpha, rel=1e-12, abs=1e-12)
        global_beta, global_alpha = stats._compute_beta_alpha(
            portfolio, benchmark, risk_free=stats.rf_annual_pct(source), input_kind="prices",
        )
        assert beta == pytest.approx(global_beta, abs=1e-12)
        assert abs(alpha - global_alpha) > 1.0
    unchanged = stats.risk_metric_row(portfolio, source)
    for name in ("volatility", "sharpe", "max_drawdown", "sortino", "var_95", "cvar_95"):
        if pd.isna(unchanged[name]):
            assert pd.isna(context["risk"][name])
        else:
            assert context["risk"][name] == unchanged[name]
    pd.testing.assert_series_equal(source, before)
    pd.testing.assert_series_equal(portfolio, original_prices)


@pytest.mark.parametrize("source", [
    None,
    pd.Series(dtype=float),
    pd.Series(dtype="Float64"),
    0.0,
    4.0,
    np.float64(2.5),
    float("nan"),
])
def test_risk_entrypoint_preserves_absent_empty_and_scalar_contracts(monkeypatch, source):
    portfolio, benchmark = _prices()
    expected = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=stats.rf_annual_pct(source), input_kind="prices",
    )
    engine, reads, calls = _engine(monkeypatch, benchmark, source)
    context = {"portfolio_history": portfolio}
    engine._risk(context)
    assert reads == [context]
    assert len(calls) == 1
    assert calls[0][1]["rf_daily"] is source
    for name, value in zip(("beta", "alpha"), expected):
        if np.isnan(value):
            assert np.isnan(context["risk"][name])
        else:
            assert context["risk"][name] == value


@pytest.mark.parametrize("kind", ["unavailable", "empty", "one-point", "flat-empty-return"])
def test_unmeasurable_history_does_not_acquire_rf_or_regression(monkeypatch, kind):
    portfolio, benchmark = _prices()
    engine, reads, calls = _engine(monkeypatch, benchmark, None)
    context = {"portfolio_history": portfolio}
    if kind == "unavailable":
        context["_order_history_unavailable"] = True
    elif kind == "empty":
        context["portfolio_history"] = portfolio.iloc[:0]
    elif kind == "one-point":
        context["portfolio_history"] = portfolio.iloc[:1]
    else:
        context["portfolio_history"] = pd.Series(
            [np.nan, np.nan], index=portfolio.index[:2],
        )
    engine._risk(context)
    assert reads == []
    assert calls == []
    if kind == "unavailable":
        assert context["risk"] is None
    else:
        assert all(np.isnan(value) for value in context["risk"].values())


def test_missing_benchmark_computes_other_risk_with_one_rf_read(monkeypatch):
    portfolio, benchmark = _prices()
    source = pd.Series(0.0001, index=portfolio.index)
    engine, reads, calls = _engine(monkeypatch, benchmark.iloc[:0], source)
    context = {"portfolio_history": portfolio}
    engine._risk(context)
    assert reads == [context]
    assert calls == []
    assert np.isnan(context["risk"]["beta"])
    assert np.isnan(context["risk"]["alpha"])
    assert np.isfinite(context["risk"]["volatility"])
