"""Causal session-date RF alignment and complete-window risk contracts.

All inputs are synthetic. The oracle scans source dates independently, rather
than using pandas' fill/reindex implementation or the production helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import robustness as rob
from tarzan.engine import stats


def _prices(*, periods=120, seed=231023, timezone=None, start="2024-03-01"):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods, tz=timezone, name="session")
    return pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0007, 0.008, periods)),
        index=index, name="SYNTHETIC",
    )


def _session(index):
    result = pd.DatetimeIndex(index)
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _causal_oracle(returns, source):
    """Last input row wins per session; only known past values can be used."""
    if source is None or not isinstance(source, pd.Series) or source.empty:
        scalar = (
            stats.RISK_FREE_RATE
            if source is None or isinstance(source, pd.Series) else float(source)
        )
        return pd.Series(
            scalar / 100.0 / stats.TRADING_DAYS, index=returns.index,
        )
    by_day = dict(zip(_session(source.index), source.array))
    observations = sorted(by_day.items())
    values = []
    for day in _session(returns.index):
        value = pd.NA if pd.api.types.is_extension_array_dtype(source.dtype) else np.nan
        for observed_day, rate in observations:
            if observed_day > day:
                break
            if not pd.isna(rate):
                value = rate
        values.append(value)
    return pd.Series(
        values, index=returns.index, name=source.name, dtype=source.dtype,
    )


def _legacy_align(returns, source):
    """Frozen pre-causal formula, used only on healthy compatible histories."""
    if isinstance(source, pd.Series) and not source.empty:
        result = source.copy()
        result.index = _session(result.index)
        result = result[~result.index.duplicated(keep="last")]
        result = result.reindex(_session(returns.index), method="ffill").ffill().bfill()
        result.index = returns.index
        return result
    scalar = (
        stats.RISK_FREE_RATE
        if source is None or isinstance(source, pd.Series) else float(source)
    )
    return pd.Series(scalar / 100.0 / stats.TRADING_DAYS, index=returns.index)


def _ratios(returns, rf, periods):
    excess = (returns - rf).dropna()
    if excess.empty:
        return float("nan"), float("nan")
    sd = excess.std()
    sharpe = (
        float("nan") if pd.isna(sd) or sd <= 0
        else float(excess.mean()) / float(sd) * np.sqrt(periods)
    )
    dd = float((excess.clip(upper=0.0) ** 2).mean()) ** 0.5 * np.sqrt(periods)
    sortino = float("nan") if dd <= 0 else float(excess.mean()) * periods / dd
    return sharpe, sortino


def _assert_exact_number(actual, expected):
    if pd.isna(actual) and pd.isna(expected):
        return
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
@pytest.mark.parametrize(
    "shape", ["unknown_prefix", "leading_na", "all_na", "outside_future"],
)
def test_incomplete_rf_does_not_promote_a_shorter_excess_window(dtype, shape):
    prices = _prices()
    returns = prices.pct_change().dropna()
    if shape == "unknown_prefix":
        source = pd.Series(
            [0.0001, 0.0002], index=returns.index[[5, 30]], dtype=dtype,
        )
    elif shape == "leading_na":
        source = pd.Series(
            [np.nan, 0.0001], index=returns.index[[0, 5]], dtype=dtype,
        )
    elif shape == "all_na":
        source = pd.Series(np.nan, index=returns.index, dtype=dtype)
    else:
        source = pd.Series(
            [0.0001], index=returns.index[-1:] + pd.Timedelta(days=3), dtype=dtype,
        )
    before = source.copy(deep=True)
    aligned = stats._align_rf_daily(returns, source)
    pd.testing.assert_series_equal(
        aligned, _causal_oracle(returns, source), check_exact=True,
    )
    assert aligned.isna().any()
    assert np.isnan(stats.compute_sharpe_tv(returns, source))
    assert np.isnan(stats.compute_sortino_tv(returns, source))
    prepared = stats.PreparedStatistics(prices, source)
    baseline = stats.risk_metric_row(prices)
    for row in (prepared.risk_metrics(), stats.risk_metric_row(prices, source)):
        assert np.isnan(row["sharpe"])
        assert np.isnan(row["sortino"])
        for key in row:
            if key not in ("sharpe", "sortino"):
                _assert_exact_number(row[key], baseline[key])
    pd.testing.assert_series_equal(source, before, check_exact=True)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
@pytest.mark.parametrize("descending", [False, True])
def test_source_nulls_keep_the_last_known_pre_window_anchor(dtype, descending):
    returns = pd.Series(
        [0.01, -0.02, 0.004, -0.002],
        index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]),
    )
    source = pd.Series(
        [0.0001, np.nan, 0.0003, np.nan],
        index=pd.to_datetime(["2025-12-29", "2026-01-04", "2026-01-12", "2026-01-13"]),
        name="SYNTHETIC_RF", dtype=dtype,
    )
    if descending:
        source = source.iloc[::-1]
    before = source.copy(deep=True)
    actual = stats._align_rf_daily(returns, source)
    expected = _causal_oracle(returns, source)
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert float(actual.iloc[0]) == pytest.approx(0.0001)
    assert float(actual.iloc[-1]) == pytest.approx(0.0003)
    for result, oracle in zip(
        (stats.compute_sharpe_tv(returns, source), stats.compute_sortino_tv(returns, source)),
        _ratios(returns, expected, stats.TRADING_DAYS),
    ):
        _assert_exact_number(result, oracle)
    pd.testing.assert_series_equal(source, before, check_exact=True)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
def test_descending_source_never_selects_a_later_rate(dtype):
    returns = _prices(periods=30, start="2026-01-05").pct_change().dropna()
    source = pd.Series(
        [0.0003, 0.0001],
        index=pd.to_datetime(["2026-01-12", "2025-12-29"]),
        dtype=dtype,
    )
    pd.testing.assert_series_equal(
        stats._align_rf_daily(returns, source),
        _causal_oracle(returns, source),
        check_exact=True,
    )


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Float32"])
@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]])
def test_consumer_order_cannot_fill_unknown_past_from_a_later_target(dtype, permutation):
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.04],
        index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"]),
    ).iloc[permutation]
    source = pd.Series([0.0003], index=pd.to_datetime(["2026-01-10"]), dtype=dtype)
    before = returns.copy(deep=True)
    aligned = stats._align_rf_daily(returns, source)
    pd.testing.assert_series_equal(
        aligned, _causal_oracle(returns, source), check_exact=True,
    )
    assert aligned.isna().sum() == 2
    assert np.isnan(stats.compute_sharpe_tv(returns, source))
    assert np.isnan(stats.compute_sortino_tv(returns, source))
    pd.testing.assert_series_equal(returns, before, check_exact=True)


@pytest.mark.parametrize("last_duplicate", [0.0002, 0.0, -0.0001, np.nan])
def test_last_input_row_wins_after_session_normalization_before_date_sort(last_duplicate):
    returns = pd.Series(
        [0.01, -0.01, 0.02],
        index=pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-12"]),
    )
    # The last row is an earlier intraday timestamp, but the documented input
    # row authority is still last-wins once both collapse to the same session.
    source = pd.Series(
        [0.0009, 0.0001, last_duplicate],
        index=pd.to_datetime(
            ["2026-01-08 16:00", "2026-01-05", "2026-01-08 10:00"], format="mixed",
        ),
        name="RF_WITH_DUPLICATES",
    )
    pd.testing.assert_series_equal(
        stats._align_rf_daily(returns, source),
        _causal_oracle(returns, source),
        check_exact=True,
    )


@pytest.mark.parametrize("source_timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("target_timezone", [None, "Europe/Rome", "America/New_York"])
def test_venue_session_dates_survive_timezone_and_dst_normalization(
    source_timezone, target_timezone,
):
    returns = _prices(
        periods=35, timezone=target_timezone, start="2024-03-01",
    ).pct_change().dropna()
    source = pd.Series(
        [0.0001, 0.0002, -0.0001],
        index=pd.DatetimeIndex(
            ["2024-03-01", "2024-03-11", "2024-04-02"], tz=source_timezone,
        ),
        name="RF_DST",
    )
    pd.testing.assert_series_equal(
        stats._align_rf_daily(returns, source),
        _causal_oracle(returns, source),
        check_exact=True,
    )


def test_duplicate_consumer_sessions_keep_each_return_and_the_original_index():
    index = pd.to_datetime(
        ["2026-01-05 09:00", "2026-01-05 16:00", "2026-01-09", "2026-01-12"],
        format="mixed",
    ).tz_localize("Europe/Rome")
    returns = pd.Series([0.01, -0.01, 0.002, -0.001], index=index, name="RETURNS")
    source = pd.Series(
        [0.0001, 0.0002, 0.0008],
        index=pd.to_datetime(["2025-12-31", "2026-01-09", "2026-01-14"]),
    )
    actual = stats._align_rf_daily(returns, source)
    pd.testing.assert_series_equal(actual, _causal_oracle(returns, source), check_exact=True)
    assert actual.index.equals(returns.index)
    assert len(actual) == len(returns)
    assert actual.iloc[2] == actual.iloc[3] == 0.0002


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("rf_kind", ["none", "empty", "scalar", "daily", "sparse", "nullable"])
def test_complete_healthy_paths_keep_exact_existing_metric_reductions(timezone, rf_kind):
    prices = _prices(timezone=timezone)
    returns = prices.pct_change().dropna()
    if rf_kind == "none":
        source = None
    elif rf_kind == "empty":
        source = pd.Series(dtype="float64")
    elif rf_kind == "scalar":
        source = 8.0
    elif rf_kind in ("daily", "nullable"):
        source = pd.Series(
            0.0001 + np.arange(len(prices)) * 0.0000001, index=prices.index,
            dtype="Float64" if rf_kind == "nullable" else "float64",
        )
    else:
        source = pd.Series([0.0001, 0.0002], index=prices.index[[0, 50]])
    aligned = stats._align_rf_daily(returns, source)
    expected = _legacy_align(returns, source)
    pd.testing.assert_series_equal(aligned, expected, check_exact=True)
    expected_sharpe, expected_sortino = _ratios(returns, expected, stats.TRADING_DAYS)
    _assert_exact_number(stats.compute_sharpe_tv(returns, source), expected_sharpe)
    _assert_exact_number(stats.compute_sortino_tv(returns, source), expected_sortino)
    row = stats.risk_metric_row(prices, source)
    if source is None or (isinstance(source, pd.Series) and source.empty):
        _assert_exact_number(
            row["sharpe"], stats.compute_sharpe(row["cagr"], row["volatility"]),
        )
        _assert_exact_number(row["sortino"], stats.compute_sortino(returns, row["cagr"]))
    else:
        _assert_exact_number(row["sharpe"], expected_sharpe)
        _assert_exact_number(row["sortino"], expected_sortino)
    window = 30
    excess = returns - expected
    rolling = excess.rolling(window)
    sharpe = (
        rolling.mean() / rolling.std() * np.sqrt(stats.TRADING_DAYS)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    expected_range = {
        "min": float(sharpe.min()), "median": float(sharpe.median()), "max": float(sharpe.max()),
    }
    for key, value in rob.rolling_sharpe_range(prices, window, source).items():
        _assert_exact_number(value, expected_range[key])


def test_prepared_statistics_reuses_one_alignment_and_owns_its_rate_source(monkeypatch):
    prices = _prices()
    source = pd.Series([0.0001, 0.0002], index=prices.index[[0, 50]])
    prepared = stats.PreparedStatistics(prices, source)
    before = source.copy(deep=True)
    original = stats._align_rf_daily
    calls = []

    def counted(returns, rf):
        calls.append((returns, rf))
        return original(returns, rf)

    monkeypatch.setattr(stats, "_align_rf_daily", counted)
    first = prepared.risk_metrics()
    prepared.excess_returns
    second = prepared.risk_metrics()
    assert len(calls) == 1
    assert prepared.rf_daily is not source
    for key in first:
        _assert_exact_number(first[key], second[key])
    pd.testing.assert_series_equal(source, before, check_exact=True)


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("prefix_kind", ["late_first_rate", "leading_na"])
def test_full_metrics_requires_every_expected_return_day_before_monthly_compounding(
    periods, dtype, prefix_kind,
):
    prices = _prices(periods=periods)
    benchmark = _prices(periods=periods, seed=321023)
    if prefix_kind == "late_first_rate":
        source = pd.Series([0.0001], index=prices.index[5:6], dtype=dtype)
    else:
        source = pd.Series(0.0001, index=prices.index, dtype=dtype)
        source.iloc[:5] = np.nan
    before = source.copy(deep=True)
    baseline = rob.full_metrics(prices, benchmark, risk_free=8.0)
    actual = rob.full_metrics(prices, benchmark, risk_free=8.0, rf_daily=source)
    assert np.isnan(actual["sharpe"])
    assert np.isnan(actual["sortino"])
    for key in actual:
        if key not in ("sharpe", "sortino"):
            _assert_exact_number(actual[key], baseline[key])
    pd.testing.assert_series_equal(source, before, check_exact=True)


@pytest.mark.parametrize("periods", [80, 720])
@pytest.mark.parametrize("timezone", [None, "Europe/Rome"])
@pytest.mark.parametrize("rf_kind", ["scalar", "daily", "nullable", "sparse"])
def test_full_metrics_complete_daily_and_monthly_paths_remain_bit_identical(
    periods, timezone, rf_kind,
):
    prices = _prices(periods=periods, timezone=timezone)
    if rf_kind == "scalar":
        source = 8.0
    elif rf_kind in ("daily", "nullable"):
        source = pd.Series(
            0.0001 + np.arange(len(prices)) * 0.0000001, index=prices.index,
            dtype="Float64" if rf_kind == "nullable" else "float64",
        )
    else:
        source = pd.Series([0.0001, 0.0002], index=prices.index[[0, 50]])
    daily = prices.pct_change().dropna()
    monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
    use_monthly = len(monthly) >= rob._MIN_MONTHS_FOR_MONTHLY
    returns = monthly if use_monthly else daily
    aligned = _legacy_align(daily, source)
    if use_monthly:
        aligned = (1.0 + aligned).resample("ME").prod(min_count=1) - 1.0
    expected_sharpe, expected_sortino = _ratios(
        returns, _legacy_align(returns, aligned),
        12 if use_monthly else stats.TRADING_DAYS,
    )
    actual = rob.full_metrics(prices, risk_free=8.0, rf_daily=source)
    baseline = rob.full_metrics(prices, risk_free=8.0)
    _assert_exact_number(actual["sharpe"], expected_sharpe)
    _assert_exact_number(actual["sortino"], expected_sortino)
    for key in actual:
        if key not in ("sharpe", "sortino"):
            _assert_exact_number(actual[key], baseline[key])


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_zero_return_day_month_is_a_zero_duration_product_not_a_provider_zero_rate(dtype):
    prices = _prices(periods=800, start="2023-01-02")
    # Preserve full_metrics' existing price-return empty-month policy. This
    # tranche neither inserts daily prices nor changes the return calendar.
    prices = prices.loc[prices.index.to_period("M") != pd.Period("2024-03")]
    source = pd.Series(0.0001, index=prices.index, dtype=dtype, name="OBSERVED_NONZERO_RF")
    source.loc[source.index >= pd.Timestamp("2024-04-01")] = 0.0002
    before = source.copy(deep=True)
    daily = prices.pct_change().dropna()
    monthly = ((1.0 + daily).resample("ME").prod() - 1.0).dropna()
    assert len(monthly) >= rob._MIN_MONTHS_FOR_MONTHLY
    daily_rf = _causal_oracle(daily, source)
    # An empty product is 1; hence its zero-duration compounded return is 0.
    # Every actual source row is nonzero: no observed zero-rate fact is added.
    expected_rf = (1.0 + daily_rf).resample("ME").prod() - 1.0
    empty_month = pd.Timestamp("2024-03-31")
    assert daily.resample("ME").size().loc[empty_month] == 0
    assert monthly.loc[empty_month] == expected_rf.loc[empty_month] == 0.0
    assert not (source == 0.0).any()
    expected_sharpe, expected_sortino = _ratios(monthly, expected_rf, 12)
    actual = rob.full_metrics(prices, rf_daily=source)
    _assert_exact_number(actual["sharpe"], expected_sharpe)
    _assert_exact_number(actual["sortino"], expected_sortino)
    baseline = rob.full_metrics(prices)
    for key in actual:
        if key not in ("sharpe", "sortino"):
            _assert_exact_number(actual[key], baseline[key])
    pd.testing.assert_series_equal(source, before, check_exact=True)


def test_rolling_sharpe_uses_only_complete_individual_windows_not_future_filled_prefix():
    prices = _prices()
    source = pd.Series([0.0001], index=prices.index[12:13])
    daily = prices.pct_change().dropna()
    excess = daily - _causal_oracle(daily, source)
    rolling = excess.rolling(20)
    ratios = (
        rolling.mean() / rolling.std() * np.sqrt(stats.TRADING_DAYS)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    assert ratios.index[0] == daily.index[30]
    actual = rob.rolling_sharpe_range(prices, 20, source)
    for key, value in (
        ("min", ratios.min()), ("median", ratios.median()), ("max", ratios.max()),
    ):
        _assert_exact_number(actual[key], value)
