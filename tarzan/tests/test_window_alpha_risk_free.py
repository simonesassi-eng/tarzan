"""Weekly OLS RF level uses the native consumer's actual overlapping window.

Synthetic prices and event-rate evidence only; no provider/cache access.
The window oracle derives weekly prices independently and uses a bisected
last-known-event lookup, not the production RF alignment helper.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import benchmarks as bm
from tarzan.engine import robustness as rob
from tarzan.engine import stats


def _prices(*, periods=180, seed=231023, timezone=None, end=None):
    rng = np.random.default_rng(seed)
    index = (
        pd.bdate_range(end=end, periods=periods, tz=timezone, name="session")
        if end is not None
        else pd.bdate_range("2024-01-02", periods=periods, tz=timezone, name="session")
    )
    market_returns = rng.normal(0.0005, 0.009, periods)
    portfolio_returns = 0.0002 + 0.65 * market_returns + rng.normal(0.0, 0.002, periods)
    portfolio = pd.Series(
        100.0 * (1.0 + portfolio_returns).cumprod(), index=index, name="PORT",
    )
    benchmark = pd.Series(
        100.0 * (1.0 + market_returns).cumprod(), index=index, name="BENCH",
    )
    return portfolio, benchmark


def _normalize(series):
    result = series.copy()
    index = pd.DatetimeIndex(result.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result


def _weekly_facts(portfolio, benchmark, *, role="prices"):
    raw = _normalize(portfolio)
    as_returns = role == "returns" or (role == "auto" and raw.abs().median() < 0.5)
    prices = (1.0 + raw).cumprod() if as_returns else raw
    if role == "auto" and as_returns:
        prices.iloc[0] = 1.0
    port_weekly = prices.resample("W-FRI").last().dropna()
    bench_weekly = _normalize(benchmark).resample("W-FRI").last().dropna()
    aligned = pd.DataFrame({"port": port_weekly, "bench": bench_weekly}).dropna()
    if len(aligned) < 5:
        return None
    returns = aligned.pct_change().dropna()
    if len(returns) < 4:
        return None
    native_returns = raw.dropna() if as_returns else prices.pct_change().dropna()
    left = aligned.index[0]
    right = min(aligned.index[-1], prices.dropna().index.max())
    native_window = native_returns.loc[
        (native_returns.index > left) & (native_returns.index <= right)
    ]
    return returns, native_window, left, right


def _causal_event_oracle(consumer, source):
    normalized = _normalize(source)
    by_day = dict(zip(normalized.index, normalized.array))
    known = sorted((day, value) for day, value in by_day.items() if not pd.isna(value))
    known_days = [day for day, _ in known]
    values = []
    missing = pd.NA if pd.api.types.is_extension_array_dtype(source.dtype) else np.nan
    for day in consumer.index:
        position = bisect_right(known_days, day)
        values.append(known[position - 1][1] if position else missing)
    return pd.Series(values, index=consumer.index, dtype=source.dtype)


def _ols_scalar_oracle(returns, annual):
    # A constant explanatory return has no identifiable OLS slope, regardless
    # of roundoff after subtracting a common RF scalar.
    if returns["bench"].min() == returns["bench"].max():
        return float("nan"), float("nan")
    annual = stats.RISK_FREE_RATE if annual is None else annual
    if pd.isna(annual) or not np.isfinite(annual):
        variance = returns["bench"].var()
        if variance <= 0:
            return float("nan"), float("nan")
        beta = returns["port"].cov(returns["bench"]) / variance
        return float(beta), float("nan")
    rf_weekly = annual / 100.0 / 52.0
    port_excess = returns["port"] - rf_weekly
    bench_excess = returns["bench"] - rf_weekly
    variance = bench_excess.var()
    if variance <= 0:
        return float("nan"), float("nan")
    beta = port_excess.cov(bench_excess) / variance
    alpha = (port_excess.mean() - beta * bench_excess.mean()) * 52.0 * 100.0
    return float(beta), float(alpha)


def _window_oracle(portfolio, benchmark, rf_daily=None, risk_free=None, *, role="prices"):
    if portfolio is None or benchmark is None or len(portfolio) < 10 or len(benchmark) < 10:
        return float("nan"), float("nan")
    facts = _weekly_facts(portfolio, benchmark, role=role)
    if facts is None:
        return float("nan"), float("nan")
    returns, native, _, _ = facts
    annual = risk_free
    if annual is None:
        if isinstance(rf_daily, pd.Series) and not rf_daily.empty:
            aligned = _causal_event_oracle(native, rf_daily)
            annual = (
                float("nan") if aligned.empty or aligned.isna().any()
                else float(aligned.mean()) * stats.TRADING_DAYS * 100.0
            )
        elif rf_daily is not None and not isinstance(rf_daily, pd.Series):
            annual = float(rf_daily)
    return _ols_scalar_oracle(returns, annual)


def _assert_exact_number(actual, expected):
    if pd.isna(actual) and pd.isna(expected):
        return
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


def _assert_exact_pair(actual, expected):
    for value, oracle in zip(actual, expected):
        _assert_exact_number(value, oracle)


def _event_path(prices, *, dtype="float64"):
    index = _normalize(prices).index
    return pd.Series(
        [0.0003, 0.00005, 0.0002, 0.0001, 0.03],
        index=pd.DatetimeIndex([
            index[0] - pd.Timedelta(days=2000),
            index[0] - pd.Timedelta(days=7),
            index[len(index) // 3],
            index[2 * len(index) // 3],
            index[-1] + pd.Timedelta(days=400),
        ]),
        dtype=dtype, name="SYNTHETIC_RF",
    )


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
@pytest.mark.parametrize("prepared", [False, True])
def test_alpha_uses_causal_native_return_day_weights_not_all_rate_events(
    timezone, dtype, prepared,
):
    prices, reference = _prices(timezone=timezone)
    source = _event_path(prices, dtype=dtype)
    before = source.copy(deep=True)
    weekly = stats.PreparedWeeklyBenchmark(reference) if prepared else None
    benchmark = reference if weekly is None else weekly.history
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
        prepared_benchmark=weekly,
    )
    _assert_exact_pair(actual, _window_oracle(prices, reference, source))
    returns, native, _, _ = _weekly_facts(prices, reference)
    scoped = _causal_event_oracle(native, source)
    annual = float(scoped.mean()) * stats.TRADING_DAYS * 100.0
    global_annual = float(source.mean()) * stats.TRADING_DAYS * 100.0
    assert abs(annual - global_annual) > 1.0
    native_events = _normalize(source).loc[
        (source.index > native.index.min()) & (source.index <= native.index.max())
    ]
    assert abs(float(native_events.mean()) * stats.TRADING_DAYS * 100.0 - annual) > 0.01
    beta_old, alpha_old = _ols_scalar_oracle(returns, global_annual)
    assert actual[0] == pytest.approx(beta_old, abs=1e-12)
    assert actual[1] - alpha_old == pytest.approx(
        -(1.0 - beta_old) * (annual - global_annual), abs=1e-10,
    )
    pd.testing.assert_series_equal(source, before, check_exact=True)


@pytest.mark.parametrize("timezone", [None, "Europe/Rome"])
@pytest.mark.parametrize("trailing_placeholders", [0, 2])
def test_weekly_friday_label_is_clipped_to_last_native_observed_price(
    timezone, trailing_placeholders,
):
    prices, benchmark = _prices(timezone=timezone, end="2026-10-21")  # Wednesday
    if trailing_placeholders:
        placeholders = pd.bdate_range(
            "2026-10-22", periods=trailing_placeholders, tz=timezone, name="session",
        )
        prices = pd.concat([prices, pd.Series(np.nan, index=placeholders, name=prices.name)])
        benchmark = pd.concat([
            benchmark, pd.Series(np.nan, index=placeholders, name=benchmark.name),
        ])
    source = pd.Series(
        [0.0001, 0.03],
        index=pd.to_datetime(["2020-01-01", "2026-10-23"]), name="FRIDAY_NOT_YET_OBSERVED",
    )
    returns, native, _, right = _weekly_facts(prices, benchmark)
    assert returns.index[-1] == pd.Timestamp("2026-10-23")
    assert right == pd.Timestamp("2026-10-21")
    assert native.index.max() == right
    expected = _ols_scalar_oracle(returns, 0.0001 * stats.TRADING_DAYS * 100.0)
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
    )
    assert actual == pytest.approx(expected, abs=1e-12)
    _assert_exact_pair(actual, _window_oracle(prices, benchmark, source))


def test_first_aligned_weekly_close_is_an_excluded_price_anchor_not_a_return_day():
    prices, benchmark = _prices()
    benchmark = benchmark.iloc[50:]
    _, native, left, _ = _weekly_facts(prices, benchmark)
    assert prices.index.min() < left
    assert (native.index > left).all()
    source = pd.Series(0.0001, index=prices.index)
    source.loc[left] = 0.03
    facts = _weekly_facts(prices, benchmark)
    expected = _ols_scalar_oracle(facts[0], 0.0001 * stats.TRADING_DAYS * 100.0)
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
    )
    assert actual == pytest.approx(expected, abs=1e-12)
    _assert_exact_pair(actual, _window_oracle(prices, benchmark, source))


@pytest.mark.parametrize("role", ["prices", "returns", "auto"])
@pytest.mark.parametrize("risk_free", [None, 0.0, 8.0])
@pytest.mark.parametrize("rf_kind", ["none", "empty", "scalar"])
def test_legacy_scalar_absence_and_explicit_annual_override_keep_exact_ols(
    role, risk_free, rf_kind,
):
    prices, benchmark = _prices()
    portfolio = prices.pct_change().dropna() if role == "returns" else prices
    source = {"none": None, "empty": pd.Series(dtype="Float64"), "scalar": 8.0}[rf_kind]
    expected_annual = risk_free if risk_free is not None else (8.0 if rf_kind == "scalar" else None)
    actual = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=risk_free, input_kind=role, rf_daily=source,
    )
    _assert_exact_pair(
        actual, _window_oracle(portfolio, benchmark, risk_free=expected_annual, role=role),
    )
    legacy = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=expected_annual, input_kind=role,
    )
    _assert_exact_pair(actual, legacy)


@pytest.mark.parametrize("risk_free", [0.0, 8.0])
@pytest.mark.parametrize("rf_kind", ["healthy", "all_na", "future_only"])
def test_explicit_annual_rate_wins_without_aligning_or_inspecting_the_supplied_path(
    monkeypatch, risk_free, rf_kind,
):
    prices, benchmark = _prices()
    if rf_kind == "healthy":
        source = _event_path(prices)
    elif rf_kind == "all_na":
        source = pd.Series(pd.NA, index=prices.index, dtype="Float64")
    else:
        source = pd.Series([0.03], index=prices.index[-1:] + pd.Timedelta(days=400))

    def forbidden(*args, **kwargs):
        pytest.fail("explicit annual override must not depend on RF path coverage")

    monkeypatch.setattr(stats, "_align_rf_daily", forbidden)
    actual = stats._compute_beta_alpha(
        prices, benchmark, risk_free=risk_free, input_kind="prices", rf_daily=source,
    )
    _assert_exact_pair(
        actual, _window_oracle(prices, benchmark, risk_free=risk_free),
    )


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32", "Int64"])
@pytest.mark.parametrize("rf_kind", ["all_na", "late_first", "future_only"])
@pytest.mark.parametrize("prepared", [False, True])
def test_unknown_window_rf_keeps_price_measurable_beta_but_not_alpha(dtype, rf_kind, prepared):
    prices, reference = _prices()
    if rf_kind == "all_na":
        source = pd.Series(np.nan, index=prices.index, dtype=dtype)
    elif rf_kind == "late_first":
        source = pd.Series([0.0], index=prices.index[20:21], dtype=dtype)
    else:
        source = pd.Series(
            [0.0], index=prices.index[-1:] + pd.Timedelta(days=30), dtype=dtype,
        )
    weekly = stats.PreparedWeeklyBenchmark(reference) if prepared else None
    benchmark = reference if weekly is None else weekly.history
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
        prepared_benchmark=weekly,
    )
    expected = _window_oracle(prices, reference, source)
    _assert_exact_pair(actual, expected)
    assert np.isfinite(actual[0])
    assert np.isnan(actual[1])


@pytest.mark.parametrize("unknown", [np.nan, np.inf, -np.inf])
def test_unknown_explicit_annual_rate_also_does_not_discard_price_beta(unknown):
    prices, benchmark = _prices()
    actual = stats._compute_beta_alpha(
        prices, benchmark, risk_free=unknown, input_kind="prices",
    )
    _assert_exact_pair(
        actual, _window_oracle(prices, benchmark, risk_free=unknown),
    )
    assert np.isfinite(actual[0])
    assert np.isnan(actual[1])


@pytest.mark.parametrize("source_shape", ["anchor_only", "anchor_then_na", "unsorted", "duplicate_na"])
def test_pre_window_anchor_survives_empty_event_window_source_order_and_missing_rows(source_shape):
    prices, benchmark = _prices()
    day = prices.index[0] - pd.Timedelta(days=7)
    if source_shape == "anchor_only":
        source = pd.Series([0.0003], index=pd.DatetimeIndex([day]))
    elif source_shape == "anchor_then_na":
        source = pd.Series([0.0003, np.nan], index=pd.DatetimeIndex([day, prices.index[0]]))
    elif source_shape == "unsorted":
        source = pd.Series(
            [0.03, 0.0003, np.nan],
            index=pd.DatetimeIndex([prices.index[-1] + pd.Timedelta(days=400), day, prices.index[0]]),
        )
    else:
        source = pd.Series(
            [0.0003, 0.03, np.nan],
            index=pd.DatetimeIndex([day, prices.index[0], prices.index[0]]),
        )
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
    )
    _assert_exact_pair(actual, _window_oracle(prices, benchmark, source))
    assert actual == pytest.approx(
        _window_oracle(prices, benchmark, risk_free=0.0003 * stats.TRADING_DAYS * 100.0),
        abs=1e-12,
    )


@pytest.mark.parametrize("role", ["prices", "returns", "auto"])
def test_window_rate_is_scale_and_role_consistent_without_changing_the_weekly_prices(role):
    prices, reference = _prices()
    if role == "returns":
        portfolio = prices.pct_change().dropna()
    elif role == "prices":
        portfolio = prices / 400.0
    else:
        portfolio = prices
    source = _event_path(prices)
    actual = stats._compute_beta_alpha(
        portfolio, reference, input_kind=role, rf_daily=source,
    )
    _assert_exact_pair(actual, _window_oracle(portfolio, reference, source, role=role))
    if role == "prices":
        regular = stats._compute_beta_alpha(prices, reference, input_kind="prices", rf_daily=source)
        assert actual == pytest.approx(regular, abs=1e-11)


@pytest.mark.parametrize("consumer", ["benchmark", "performance_row", "robustness"])
@pytest.mark.parametrize("rf_kind", ["none", "empty", "scalar", "path", "all_na"])
@pytest.mark.parametrize("prepared", [False, True])
def test_production_consumers_opt_in_to_windows_without_changing_legacy_scalar_authority(
    consumer, rf_kind, prepared,
):
    prices, reference = _prices()
    source = {
        "none": None,
        "empty": pd.Series(dtype="Float64"),
        "scalar": 8.0,
        "path": _event_path(prices),
        "all_na": pd.Series(pd.NA, index=prices.index, dtype="Float64"),
    }[rf_kind]
    preparation = bm.BenchmarkPreparation(reference) if prepared else None
    if consumer == "benchmark":
        actual = bm._compute_single_benchmark_metrics(
            prices, reference, source, benchmark_preparation=preparation,
        )
    elif consumer == "performance_row":
        actual = {}
        bm._populate_perf_row(
            actual, prices, reference, source, benchmark_preparation=preparation,
        )
    else:
        actual = rob.full_metrics(prices, reference, rf_daily=source)
    if consumer == "robustness" and rf_kind == "scalar":
        # Historical full_metrics rf_daily scalar never selected its alpha
        # annual override. Preserve that contract; risk_free=8 is the override.
        expected = _window_oracle(prices, reference)
    else:
        expected = _window_oracle(prices, reference, source)
    _assert_exact_pair((actual["beta"], actual["alpha"]), expected)
    baseline = (
        bm._compute_single_benchmark_metrics(prices, reference, source)
        if consumer == "benchmark"
        else rob.full_metrics(prices, reference, rf_daily=source)
        if consumer == "robustness"
        else None
    )
    if baseline is not None:
        for key in actual:
            _assert_exact_number(actual[key], baseline[key])


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("risk_free", [0.0, 8.0])
@pytest.mark.parametrize("rf_kind", ["path", "all_na", "scalar"])
def test_full_metrics_manual_annual_override_stays_authoritative(periods, risk_free, rf_kind):
    prices, benchmark = _prices(periods=periods)
    source = {
        "path": _event_path(prices),
        "all_na": pd.Series(pd.NA, index=prices.index, dtype="Float64"),
        "scalar": 3.0,
    }[rf_kind]
    actual = rob.full_metrics(prices, benchmark, risk_free=risk_free, rf_daily=source)
    assert actual["risk_free"] == risk_free
    _assert_exact_pair(
        (actual["beta"], actual["alpha"]),
        _window_oracle(prices, benchmark, risk_free=risk_free),
    )


def test_rf_path_is_keyword_only_and_inputs_keep_their_ownership():
    prices, benchmark = _prices()
    source = _event_path(prices)
    before = [series.copy(deep=True) for series in (prices, benchmark, source)]
    with pytest.raises(TypeError):
        stats._compute_beta_alpha(prices, benchmark, 0.0, None, source)
    stats._compute_beta_alpha(prices, benchmark, input_kind="prices", rf_daily=source)
    for actual, expected in zip((prices, benchmark, source), before):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("case", ["short", "too_few_weeks", "constant_benchmark", "no_overlap"])
@pytest.mark.parametrize("rf_kind", ["path", "all_na"])
def test_rf_window_does_not_relax_existing_ols_minimum_or_reference_variance(case, rf_kind):
    prices, benchmark = _prices()
    if case == "short":
        prices = prices.iloc[:8]
    elif case == "too_few_weeks":
        prices = prices.iloc[:15]
    elif case == "constant_benchmark":
        benchmark[:] = 100.0
    else:
        benchmark.index = benchmark.index + pd.DateOffset(years=2)
    source = (
        _event_path(prices) if rf_kind == "path"
        else pd.Series(pd.NA, index=prices.index, dtype="Float64")
    )
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
    )
    assert np.isnan(actual[0])
    assert np.isnan(actual[1])


@pytest.mark.parametrize("shape", ["flat_prices", "constant_nonzero_weekly_returns"])
@pytest.mark.parametrize("role", ["prices", "returns"])
@pytest.mark.parametrize(
    "rate_kind", ["manual", "window", "no_path", "scalar_rf", "unknown_path", "unknown_annual"],
)
def test_exact_constant_benchmark_returns_are_unidentifiable_for_every_rf_contract(
    shape, role, rate_kind,
):
    prices, benchmark = _prices()
    if shape == "flat_prices":
        benchmark[:] = 100.0
    else:
        index = pd.date_range("2024-01-05", periods=36, freq="W-FRI", name="session")
        rng = np.random.default_rng(231023)
        prices = pd.Series(
            100.0 * (1.0 + rng.normal(0.002, 0.02, len(index))).cumprod(),
            index=index, name="PORT",
        )
        benchmark = pd.Series(100.0 * 2.0 ** np.arange(len(index)), index=index, name="BENCH")
    portfolio = prices.pct_change().dropna() if role == "returns" else prices
    facts = _weekly_facts(portfolio, benchmark, role=role)
    assert facts is not None
    assert facts[0]["bench"].min() == facts[0]["bench"].max()
    assert facts[0]["bench"].var() == 0.0
    source = None
    annual = None
    if rate_kind in ("manual", "window"):
        source = _event_path(prices)
        if rate_kind == "manual":
            annual = 2.9781818181818185  # Legacy finite scalar counterexample.
    elif rate_kind == "scalar_rf":
        source = 8.0
    elif rate_kind == "unknown_path":
        source = pd.Series(pd.NA, index=prices.index, dtype="Float64")
    elif rate_kind == "unknown_annual":
        annual = np.nan
    actual = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=annual, input_kind=role, rf_daily=source,
    )
    assert np.isnan(actual[0])
    assert np.isnan(actual[1])
    _assert_exact_pair(
        actual, _window_oracle(portfolio, benchmark, source, annual, role=role),
    )


@pytest.mark.parametrize("role", ["prices", "returns"])
@pytest.mark.parametrize("rate_kind", ["manual", "window", "no_path", "unknown"])
def test_near_flat_but_nonconstant_benchmark_is_not_rejected_by_an_epsilon(role, rate_kind):
    prices, benchmark = _prices()
    tiny_returns = 1e-9 * np.sin(np.arange(len(benchmark)))
    benchmark[:] = 100.0 * (1.0 + tiny_returns).cumprod()
    portfolio = prices.pct_change().dropna() if role == "returns" else prices
    facts = _weekly_facts(portfolio, benchmark, role=role)
    assert facts[0]["bench"].min() != facts[0]["bench"].max()
    assert 0.0 < facts[0]["bench"].var() < 1e-16
    annual = 2.9781818181818185 if rate_kind == "manual" else None
    source = (
        _event_path(prices) if rate_kind == "window"
        else pd.Series(pd.NA, index=prices.index, dtype="Float64") if rate_kind == "unknown"
        else None
    )
    actual = stats._compute_beta_alpha(
        portfolio, benchmark, risk_free=annual, input_kind=role, rf_daily=source,
    )
    _assert_exact_pair(
        actual, _window_oracle(portfolio, benchmark, source, annual, role=role),
    )
    assert np.isfinite(actual[0])
    assert np.isnan(actual[1]) if rate_kind == "unknown" else np.isfinite(actual[1])


@pytest.mark.parametrize("case", ["holiday_gaps", "missing_week", "missing_bars", "duplicates"])
@pytest.mark.parametrize("prepared", [False, True])
def test_native_calendar_and_missing_price_policy_are_preserved_without_a_new_date_grid(
    case, prepared,
):
    prices, reference = _prices(timezone="Europe/Rome")
    if case == "holiday_gaps":
        prices = prices.loc[prices.index.dayofweek != 0]
    elif case == "missing_week":
        prices = prices.loc[
            ~((prices.index.month == 3) & (prices.index.day >= 11) & (prices.index.day <= 15))
        ]
    elif case == "missing_bars":
        prices.iloc[::11] = np.nan
    else:
        prices = pd.concat([prices.iloc[:30], prices.iloc[29:]])
    source = _event_path(prices)
    before = prices.copy(deep=True)
    weekly = stats.PreparedWeeklyBenchmark(reference) if prepared else None
    benchmark = reference if weekly is None else weekly.history
    actual = stats._compute_beta_alpha(
        prices, benchmark, input_kind="prices", rf_daily=source,
        prepared_benchmark=weekly,
    )
    _assert_exact_pair(actual, _window_oracle(prices, reference, source))
    pd.testing.assert_series_equal(prices, before, check_exact=True)


@pytest.mark.parametrize("first_return", [0.2, -0.2])
@pytest.mark.parametrize("role", ["returns", "auto"])
def test_first_friday_jump_keeps_explicit_return_role_and_legacy_auto_distinct_with_rf_path(
    first_return, role,
):
    rng = np.random.default_rng(231023)
    index = pd.bdate_range("2024-01-05", periods=100)  # First observation is Friday.
    returns = pd.Series(rng.normal(0.0005, 0.009, len(index)), index=index)
    returns.iloc[0] = first_return
    benchmark = 100.0 * (1.0 + returns).cumprod()
    source = _event_path(benchmark)
    actual = stats._compute_beta_alpha(
        returns, benchmark, input_kind=role, rf_daily=source,
    )
    _assert_exact_pair(actual, _window_oracle(returns, benchmark, source, role=role))
    if role == "returns":
        assert actual[0] == pytest.approx(1.0, abs=1e-12)
        assert actual[1] == pytest.approx(0.0, abs=1e-10)
    else:
        assert abs(actual[0] - 1.0) > 0.1


@pytest.mark.parametrize("periods", [80, 720])
def test_full_metrics_legacy_annual_field_is_not_misrepresented_as_window_observed_evidence(periods):
    prices, benchmark = _prices(periods=periods)
    source = _event_path(prices)
    actual = rob.full_metrics(prices, benchmark, rf_daily=source)
    baseline = rob.full_metrics(prices, benchmark)
    assert actual.keys() == baseline.keys()
    assert actual["risk_free"] == baseline["risk_free"] == stats.RISK_FREE_RATE
    for key in actual:
        if key not in ("sharpe", "sortino", "alpha", "beta"):
            _assert_exact_number(actual[key], baseline[key])
    _assert_exact_pair(
        (actual["beta"], actual["alpha"]), _window_oracle(prices, benchmark, source),
    )
