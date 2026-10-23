"""Absence, unavailable RF evidence and explicit price/return role contracts.

All fixtures are synthetic. The oracles keep the existing frequency, calendar,
alignment and weekly OLS expressions; no provider or external cache is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import benchmarks as bm
from tarzan.engine import robustness as rob
from tarzan.engine import stats


def _prices(*, periods=520, seed=52023, timezone=None, start="2024-01-05"):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods, tz=timezone, name="session")
    return pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.006, periods)),
        index=index, name="SYNTHETIC",
    )


def _normalize(series):
    """Independent venue-session normalization, not a production helper."""
    result = series.copy()
    index = pd.DatetimeIndex(result.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result


def _legacy_rf_align(returns, rf_daily):
    if isinstance(rf_daily, pd.Series) and not rf_daily.empty:
        path = _normalize(rf_daily)
        path = path[~path.index.duplicated(keep="last")]
        aligned = path.reindex(_normalize(returns).index, method="ffill").ffill().bfill()
        aligned.index = returns.index
        return aligned
    scalar = stats.RISK_FREE_RATE if rf_daily is None else float(rf_daily)
    return pd.Series(scalar / 100.0 / stats.TRADING_DAYS, index=returns.index)


def _excess_ratios(returns, rf_daily, periods):
    excess = (returns - _legacy_rf_align(returns, rf_daily)).dropna()
    if excess.empty:
        return float("nan"), float("nan")
    sd = float(excess.std())
    sharpe = (
        float("nan") if sd <= 0
        else float(excess.mean()) / sd * np.sqrt(periods)
    )
    downside = excess.clip(upper=0.0)
    dd = float((downside ** 2).mean()) ** 0.5 * np.sqrt(periods)
    sortino = float("nan") if dd <= 0 else float(excess.mean()) * periods / dd
    return sharpe, sortino


def _ols_oracle(portfolio, benchmark, risk_free=None, *, role="prices"):
    """Weekly price alignment and OLS, including unchanged legacy auto."""
    if benchmark is None or len(benchmark) < 10:
        return float("nan"), float("nan")
    if portfolio is None or len(portfolio) < 10:
        return float("nan"), float("nan")
    prices = _normalize(portfolio)
    if role == "returns":
        prices = (1.0 + prices).cumprod()
    elif role == "auto" and prices.abs().median() < 0.5:
        prices = (1.0 + prices).cumprod()
        prices.iloc[0] = 1.0
    port_weekly = prices.resample("W-FRI").last().dropna()
    bench_weekly = _normalize(benchmark).resample("W-FRI").last().dropna()
    aligned = pd.DataFrame({"port": port_weekly, "bench": bench_weekly}).dropna()
    if len(aligned) < 5:
        return float("nan"), float("nan")
    returns = aligned.pct_change().dropna()
    if len(returns) < 4:
        return float("nan"), float("nan")
    rf_weekly = (
        stats.RISK_FREE_RATE if risk_free is None else risk_free
    ) / 100.0 / 52.0
    port_excess = returns["port"] - rf_weekly
    bench_excess = returns["bench"] - rf_weekly
    var_bench = bench_excess.var()
    if var_bench <= 0:
        return float("nan"), float("nan")
    beta = port_excess.cov(bench_excess) / var_bench
    intercept = port_excess.mean() - beta * bench_excess.mean()
    return float(beta), float(intercept * 52.0 * 100.0)


def _assert_exact_number(actual, expected):
    if np.isnan(actual) and np.isnan(expected):
        return
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


def _assert_exact_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        _assert_exact_number(actual[key], expected[key])


def _assert_exact_pair(actual, expected):
    for a, e in zip(actual, expected):
        _assert_exact_number(a, e)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "object"])
def test_empty_risk_free_uses_scalar_formula_before_preparation(dtype):
    prices = _prices()
    empty = pd.Series(dtype=dtype)
    before = empty.copy(deep=True)
    expected = stats.risk_metric_row(prices, None)
    prepared = stats.PreparedStatistics(prices, empty)
    assert prepared.rf_daily is None
    _assert_exact_metrics(prepared.risk_metrics(), expected)
    _assert_exact_metrics(stats.risk_metric_row(prices, empty), expected)
    assert stats.rf_annual_pct(empty) is None
    pd.testing.assert_series_equal(empty, before, check_exact=True)


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("dtype", ["float64", "Float64", "object"])
def test_empty_risk_free_alignment_and_tv_helpers_match_none(timezone, dtype):
    returns = _prices(timezone=timezone).pct_change().dropna()
    empty = pd.Series(dtype=dtype)
    pd.testing.assert_series_equal(
        stats._align_rf_daily(returns, empty),
        stats._align_rf_daily(returns, None),
        check_exact=True,
    )
    _assert_exact_number(
        stats.compute_sharpe_tv(returns, empty),
        stats.compute_sharpe_tv(returns, None),
    )
    _assert_exact_number(
        stats.compute_sortino_tv(returns, empty),
        stats.compute_sortino_tv(returns, None),
    )
    assert rob.rolling_sharpe_range(_prices(timezone=timezone), 60, empty) == (
        rob.rolling_sharpe_range(_prices(timezone=timezone), 60, None)
    )


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("risk_free", [None, 0.0, 8.0])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_empty_risk_free_full_metrics_respects_explicit_scalar_fallback(
    periods, risk_free, dtype,
):
    prices = _prices(periods=periods)
    benchmark = _prices(periods=periods, seed=42023)
    expected = rob.full_metrics(prices, benchmark, risk_free=risk_free, rf_daily=None)
    actual = rob.full_metrics(
        prices, benchmark, risk_free=risk_free, rf_daily=pd.Series(dtype=dtype),
    )
    _assert_exact_metrics(actual, expected)
    daily = prices.pct_change().dropna()
    monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
    returns = monthly if len(monthly) >= rob._MIN_MONTHS_FOR_MONTHLY else daily
    periods_per_year = 12 if returns is monthly else stats.TRADING_DAYS
    _assert_exact_number(
        actual["sharpe"],
        stats.compute_sharpe(actual["cagr"], actual["volatility"], risk_free=risk_free),
    )
    _assert_exact_number(
        actual["sortino"],
        stats.compute_sortino(
            returns, actual["cagr"], risk_free=risk_free, periods=periods_per_year,
        ),
    )


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32", "Int64"])
def test_nonempty_all_missing_risk_free_remains_unavailable(periods, dtype):
    prices = _prices(periods=periods)
    rf = pd.Series(np.nan, index=prices.index, dtype=dtype)
    before = rf.copy(deep=True)
    prepared = stats.PreparedStatistics(prices, rf)
    assert isinstance(prepared.rf_daily, pd.Series)
    assert prepared.rf_daily is not rf
    assert len(prepared.rf_daily) == periods
    annual = stats.rf_annual_pct(rf)
    assert annual is not None and np.isnan(annual)
    row = prepared.risk_metrics()
    baseline = stats.risk_metric_row(prices)
    for key in row:
        if key in ("sharpe", "sortino"):
            assert np.isnan(row[key])
        else:
            _assert_exact_number(row[key], baseline[key])
    full = rob.full_metrics(prices, risk_free=8.0, rf_daily=rf)
    assert np.isnan(full["sharpe"])
    assert np.isnan(full["sortino"])
    assert full["risk_free"] == 8.0
    assert rob.rolling_sharpe_range(prices, 30, rf_daily=rf) == {}
    pd.testing.assert_series_equal(rf, before, check_exact=True)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32", "Int64"])
def test_monthly_all_na_risk_free_does_not_become_an_observed_zero_rate(dtype):
    prices = _prices(periods=720)
    rf = pd.Series(np.nan, index=prices.index, dtype=dtype)
    invalid = rob.full_metrics(prices, rf_daily=rf)
    zero = rob.full_metrics(prices, rf_daily=pd.Series(0.0, index=prices.index))
    assert np.isnan(invalid["sharpe"])
    assert np.isnan(invalid["sortino"])
    assert np.isfinite(zero["sharpe"])
    assert np.isfinite(zero["sortino"])


def test_nullable_single_excess_return_has_unavailable_sample_sharpe():
    index = pd.DatetimeIndex(["2024-01-05"])
    returns = pd.Series([-0.01], index=index, dtype="Float64")
    rf = pd.Series([0.0], index=index, dtype="Float64")
    assert np.isnan(stats.compute_sharpe_tv(returns, rf))
    assert np.isnan(stats._sharpe_from_excess(returns.iloc[:0], stats.TRADING_DAYS))
    assert stats.compute_sortino_tv(returns, rf) == pytest.approx(-np.sqrt(stats.TRADING_DAYS))


@pytest.mark.parametrize("timezone", [None, "Europe/Rome"])
@pytest.mark.parametrize("rf_kind", ["none", "scalar", "path", "nullable"])
def test_healthy_row_risk_free_reductions_remain_bit_identical(timezone, rf_kind):
    prices = _prices(timezone=timezone)
    daily = prices.pct_change().dropna()
    rf = _healthy_rf(prices, rf_kind)
    actual = stats.risk_metric_row(prices, rf)
    if rf is None:
        expected_sharpe = stats.compute_sharpe(actual["cagr"], actual["volatility"])
        expected_sortino = stats.compute_sortino(daily, actual["cagr"])
    else:
        expected_sharpe, expected_sortino = _excess_ratios(daily, rf, stats.TRADING_DAYS)
    _assert_exact_number(actual["sharpe"], expected_sharpe)
    _assert_exact_number(actual["sortino"], expected_sortino)
    aligned = stats._align_rf_daily(daily, rf)
    pd.testing.assert_series_equal(
        aligned, _legacy_rf_align(daily, rf), check_exact=True,
    )
    if isinstance(rf, pd.Series):
        expected_annual = float(rf.mean()) * stats.TRADING_DAYS * 100.0
        _assert_exact_number(stats.rf_annual_pct(rf), expected_annual)


def _healthy_rf(prices, kind):
    if kind == "none":
        return None
    if kind == "scalar":
        return 8.0
    values = np.linspace(0.00005, 0.0002, len(prices))
    values[::13] = np.nan
    # Include the existing prefix bfill policy in the unchanged-path oracle.
    return pd.Series(
        values, index=prices.index,
        dtype="Float64" if kind == "nullable" else "float64",
    )


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("rf_kind", ["none", "scalar", "path", "nullable"])
def test_healthy_full_metrics_daily_and_monthly_expressions_are_unchanged(periods, rf_kind):
    prices = _prices(periods=periods)
    rf = _healthy_rf(prices, rf_kind)
    actual = rob.full_metrics(prices, risk_free=8.0, rf_daily=rf)
    daily = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
    monthly_used = len(monthly) >= rob._MIN_MONTHS_FOR_MONTHLY
    returns = monthly if monthly_used else daily
    periods_per_year = 12 if monthly_used else stats.TRADING_DAYS
    expected_vol = float(returns.std()) * np.sqrt(periods_per_year) * 100.0
    _assert_exact_number(actual["volatility"], expected_vol)
    if rf is None:
        expected_sharpe = stats.compute_sharpe(actual["cagr"], expected_vol, risk_free=8.0)
        expected_sortino = stats.compute_sortino(
            returns, actual["cagr"], risk_free=8.0, periods=periods_per_year,
        )
    else:
        rf_path = rf
        if monthly_used:
            aligned = _legacy_rf_align(daily, rf)
            rf_path = ((1.0 + aligned).resample("ME").prod() - 1.0).dropna()
        expected_sharpe, expected_sortino = _excess_ratios(returns, rf_path, periods_per_year)
    _assert_exact_number(actual["sharpe"], expected_sharpe)
    _assert_exact_number(actual["sortino"], expected_sortino)


@pytest.mark.parametrize("role", [None, True, False, 0, 1, "", "price", "RETURNS", [], {}])
def test_invalid_input_role_fails_fast_even_with_absent_series(role):
    with pytest.raises(ValueError, match="input_kind"):
        stats._compute_beta_alpha(None, None, input_kind=role)


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("level", [0.00025, 0.25, 100.0, 10000.0])
@pytest.mark.parametrize("prepared", [False, True])
def test_price_role_is_scale_invariant_against_the_same_benchmark(timezone, level, prepared):
    reference = _prices(seed=42023, periods=100)
    portfolio = reference / reference.iloc[0] * level
    if timezone is not None:
        portfolio.index = portfolio.index.tz_localize(timezone)
    weekly = stats.PreparedWeeklyBenchmark(reference) if prepared else None
    benchmark = weekly.history if weekly is not None else reference
    actual = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=0.0, input_kind="prices",
        prepared_benchmark=weekly,
    )
    _assert_exact_pair(actual, _ols_oracle(portfolio, reference, risk_free=0.0))
    assert actual[0] == pytest.approx(1.0, abs=1e-12)
    assert actual[1] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("first_return", [0.2, -0.2, 0.0])
@pytest.mark.parametrize("prepared", [False, True])
def test_explicit_returns_keeps_first_week_without_a_one_point_reset(first_return, prepared):
    rng = np.random.default_rng(62023)
    index = pd.bdate_range("2024-01-05", periods=100)  # First point is Friday.
    returns = pd.Series(rng.normal(0.0004, 0.006, len(index)), index=index)
    returns.iloc[0] = first_return
    benchmark = 100.0 * (1.0 + returns).cumprod()
    before = returns.copy(deep=True)
    weekly = stats.PreparedWeeklyBenchmark(benchmark) if prepared else None
    reference = weekly.history if weekly is not None else benchmark
    actual = stats._compute_beta_alpha(
        returns, reference, risk_free=0.0, input_kind="returns",
        prepared_benchmark=weekly,
    )
    _assert_exact_pair(
        actual, _ols_oracle(returns, benchmark, risk_free=0.0, role="returns"),
    )
    assert actual[0] == pytest.approx(1.0, abs=1e-12)
    assert actual[1] == pytest.approx(0.0, abs=1e-10)
    pd.testing.assert_series_equal(returns, before, check_exact=True)


@pytest.mark.parametrize("role", ["auto", "prices", "returns"])
def test_input_role_is_keyword_only(role):
    prices = _prices(periods=100)
    with pytest.raises(TypeError):
        stats._compute_beta_alpha(prices, prices, 0.0, None, role)


@pytest.mark.parametrize("case", [
    "prices", "low_prices", "returns", "missing_week", "missing_bars",
    "duplicates", "flat_benchmark", "short",
])
@pytest.mark.parametrize("timezone", [None, "Europe/Rome"])
@pytest.mark.parametrize("risk_free", [None, 0.0, 8.0])
def test_legacy_auto_ols_is_bit_identical(case, timezone, risk_free):
    portfolio = _prices(periods=140, timezone=timezone)
    benchmark = _prices(periods=140, seed=42023, timezone="America/New_York")
    if case == "low_prices":
        portfolio = portfolio / 400.0
    elif case == "returns":
        portfolio = portfolio.pct_change()
    elif case == "missing_week":
        portfolio = portfolio[~((portfolio.index.month == 3) & (portfolio.index.day <= 9))]
    elif case == "missing_bars":
        portfolio.iloc[::11] = np.nan
        benchmark.iloc[::13] = np.nan
    elif case == "duplicates":
        portfolio = pd.concat([portfolio.iloc[:20], portfolio.iloc[19:]])
    elif case == "flat_benchmark":
        benchmark[:] = 100.0
    elif case == "short":
        portfolio = portfolio.iloc[:8]
    expected = _ols_oracle(portfolio, benchmark, risk_free, role="auto")
    _assert_exact_pair(
        stats._compute_beta_alpha(portfolio, benchmark, risk_free=risk_free), expected,
    )
    _assert_exact_pair(
        stats._compute_beta_alpha(portfolio, benchmark, risk_free=risk_free, input_kind="auto"),
        expected,
    )


@pytest.mark.parametrize("case", ["prices", "missing_week", "missing_bars", "duplicates"])
@pytest.mark.parametrize("risk_free", [None, 0.0, 8.0])
def test_explicit_healthy_prices_preserve_legacy_ols_exactly(case, risk_free):
    portfolio = _prices(periods=140, timezone="Europe/Rome")
    benchmark = _prices(periods=140, seed=42023, timezone="America/New_York")
    if case == "missing_week":
        portfolio = portfolio[~((portfolio.index.month == 3) & (portfolio.index.day <= 9))]
    elif case == "missing_bars":
        portfolio.iloc[::11] = np.nan
        benchmark.iloc[::13] = np.nan
    elif case == "duplicates":
        portfolio = pd.concat([portfolio.iloc[:20], portfolio.iloc[19:]])
    expected = _ols_oracle(portfolio, benchmark, risk_free, role="auto")
    _assert_exact_pair(
        stats._compute_beta_alpha(portfolio, benchmark, risk_free=risk_free, input_kind="prices"),
        expected,
    )


@pytest.mark.parametrize("consumer", ["benchmark", "performance_row", "robustness"])
@pytest.mark.parametrize("prepared", [False, True])
def test_production_consumers_declare_price_role_for_low_unit_prices(monkeypatch, consumer, prepared):
    reference = _prices(periods=140, seed=42023)
    low_price = reference / 400.0
    observed = []
    original = stats._compute_beta_alpha
    module = rob if consumer == "robustness" else bm

    def record(*args, **kwargs):
        observed.append(kwargs.get("input_kind"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_compute_beta_alpha", record)
    preparation = bm.BenchmarkPreparation(reference) if prepared else None
    if consumer == "benchmark":
        row = bm._compute_single_benchmark_metrics(
            low_price, reference, benchmark_preparation=preparation,
        )
    elif consumer == "performance_row":
        row = {}
        bm._populate_perf_row(
            row, low_price, reference, benchmark_preparation=preparation,
        )
    else:
        row = rob.full_metrics(low_price, reference)
    assert observed == ["prices"]
    assert row["beta"] == pytest.approx(1.0, abs=1e-12)
    assert row["alpha"] == pytest.approx(0.0, abs=1e-10)


def test_returns_contract_and_legacy_auto_remain_distinct_for_first_friday_jump():
    rng = np.random.default_rng(62023)
    index = pd.bdate_range("2024-01-05", periods=100)
    returns = pd.Series(rng.normal(0.0004, 0.006, len(index)), index=index)
    returns.iloc[0] = 0.2
    benchmark = 100.0 * (1.0 + returns).cumprod()
    legacy = stats._compute_beta_alpha(returns, benchmark, risk_free=0.0)
    _assert_exact_pair(legacy, _ols_oracle(returns, benchmark, 0.0, role="auto"))
    assert legacy[0] != pytest.approx(1.0, abs=0.1)
    corrected = stats._compute_beta_alpha(
        returns, benchmark, risk_free=0.0, input_kind="returns",
    )
    assert corrected[0] == pytest.approx(1.0, abs=1e-12)
