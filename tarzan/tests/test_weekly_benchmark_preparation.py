"""Exact weekly-regression, clipping, ownership and bounded-cache gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import benchmarks as bm
from tarzan.engine import stats


def _reference_beta_alpha(port, benchmark, risk_free=None):
    """The original 4da5acf weekly-price alignment and OLS expressions."""
    if benchmark is None or len(benchmark) < 10:
        return float("nan"), float("nan")
    if port is None or len(port) < 10:
        return float("nan"), float("nan")
    port_raw = stats.normalize_index(port)
    bench_raw = stats.normalize_index(benchmark)
    if port_raw.abs().median() < 0.5:
        port_prices = (1 + port_raw).cumprod()
        port_prices.iloc[0] = 1.0
    else:
        port_prices = port_raw
    port_w = port_prices.resample("W-FRI").last().dropna()
    bench_w = bench_raw.resample("W-FRI").last().dropna()
    aligned = pd.DataFrame({"port": port_w, "bench": bench_w}).dropna()
    if len(aligned) < 5:
        return float("nan"), float("nan")
    rets = aligned.pct_change().dropna()
    if len(rets) < 4:
        return float("nan"), float("nan")
    rf_weekly = (
        stats.RISK_FREE_RATE if risk_free is None else risk_free
    ) / 100.0 / 52.0
    port_excess = rets["port"] - rf_weekly
    bench_excess = rets["bench"] - rf_weekly
    var_bench = bench_excess.var()
    if var_bench <= 0:
        return float("nan"), float("nan")
    beta = port_excess.cov(bench_excess) / var_bench
    alpha_weekly = port_excess.mean() - beta * bench_excess.mean()
    return float(beta), float(alpha_weekly * 52.0 * 100.0)


def _assert_exact_number(actual, expected):
    if np.isnan(actual) and np.isnan(expected):
        return
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


def _assert_exact_pair(actual, expected):
    for a, e in zip(actual, expected):
        _assert_exact_number(a, e)


def _prices(seed, timezone, *, ticker="ASSET.MI", periods=260):
    index = pd.bdate_range(
        "2025-01-02", periods=periods, tz=timezone, name="session"
    )
    rng = np.random.default_rng(seed)
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0002, 0.012, len(index))),
        index=index, name=ticker,
    )
    prices.attrs["resolved_ticker"] = ticker
    return prices


@pytest.mark.parametrize("case", [
    "prices", "returns", "low_prices", "missing_week",
    "missing_bars", "duplicates", "flat", "short",
])
@pytest.mark.parametrize("timezone", [
    None, "Europe/Rome", "America/New_York", "Asia/Tokyo",
])
@pytest.mark.parametrize("risk_free", [None, 0.0, 3.75])
def test_prepared_regression_is_bitwise_equivalent(
    case, timezone, risk_free,
):
    port = _prices(5, timezone)
    benchmark = _prices(6, "America/New_York", ticker="REFERENCE")
    if case == "returns":
        port = port.pct_change()
    elif case == "low_prices":
        # Preserve the existing median heuristic, not a methodological fix
        # concealed inside a performance optimization.
        port = port / 1000.0
    elif case == "missing_week":
        port = port[~((port.index.month == 5) & (port.index.day <= 9))]
        benchmark = benchmark[
            ~((benchmark.index.month == 8) & (benchmark.index.day <= 8))
        ]
    elif case == "missing_bars":
        port.iloc[::7] = np.nan
        benchmark.iloc[::11] = np.nan
    elif case == "duplicates":
        port = pd.concat([port.iloc[:40], port.iloc[39:]])
        benchmark = pd.concat([benchmark.iloc[:30], benchmark.iloc[29:]])
    elif case == "flat":
        benchmark[:] = 100.0
    elif case == "short":
        port = port.iloc[:8]
    prepared = stats.PreparedWeeklyBenchmark(benchmark)
    if case == "flat":
        # A constant explanatory return has no OLS slope. The old scalar
        # subtraction could manufacture tiny positive variance by roundoff;
        # keep the historical bitwise oracle unchanged for every nonflat case.
        raw_weekly = (
            stats.normalize_index(benchmark)
            .resample("W-FRI").last().dropna().pct_change().dropna()
        )
        assert raw_weekly.nunique(dropna=True) <= 1
        assert raw_weekly.var() == 0.0
        expected = float("nan"), float("nan")
    else:
        expected = _reference_beta_alpha(port, benchmark, risk_free)
    actual = stats._compute_beta_alpha(
        port, prepared.history, risk_free=risk_free,
        prepared_benchmark=prepared,
    )
    _assert_exact_pair(actual, expected)


@pytest.mark.parametrize("start,end", [
    ("2025-02-05", "2025-09-08"),  # partial weeks at both edges
    ("2025-02-07", "2025-09-12"),  # Friday boundaries
    ("2025-03-30", "2025-10-27"),  # both DST transitions
    ("2025-01-01", "2026-02-01"),  # full source and external edges
    ("2025-05-01", "2025-05-07"),  # too short for regression
    ("2027-01-01", "2027-01-02"),  # no source observations
])
@pytest.mark.parametrize("timezone", ["Europe/Rome", "America/New_York"])
def test_window_is_clipped_before_weekly_resampling(start, end, timezone):
    reference = _prices(13, timezone, ticker="REFERENCE")
    source_before = reference.copy(deep=True)
    preparation = bm.BenchmarkPreparation(reference)
    prepared = preparation.window(
        pd.Timestamp(start, tz=timezone), pd.Timestamp(end, tz=timezone)
    )
    expected_history = bm._clip_to_window(reference, start, end)
    pd.testing.assert_series_equal(
        prepared.history, expected_history, check_exact=True
    )
    expected_weekly = (
        stats.normalize_index(expected_history)
        .resample("W-FRI").last().dropna()
    )
    pd.testing.assert_series_equal(
        prepared.weekly_prices, expected_weekly, check_exact=True
    )
    pd.testing.assert_series_equal(reference, source_before, check_exact=True)


def test_missing_whole_weeks_align_prices_before_computing_returns():
    port = _prices(17, "Europe/Rome")
    reference = _prices(18, "America/New_York", ticker="REFERENCE")
    reference = reference[
        ~((reference.index.month == 5) & (reference.index.day <= 16))
    ]
    prepared = stats.PreparedWeeklyBenchmark(reference)
    actual = stats._compute_beta_alpha(
        port, prepared.history, prepared_benchmark=prepared
    )
    _assert_exact_pair(actual, _reference_beta_alpha(port, reference))
    # Reference weekly levels, not an independently pct-changed tape, are the
    # exact values the regression must put into the common-window DataFrame.
    pd.testing.assert_series_equal(
        prepared.weekly_prices,
        stats.normalize_index(reference).resample("W-FRI").last().dropna(),
        check_exact=True,
    )


def test_full_reference_resampled_once_across_rows(monkeypatch):
    reference = _prices(23, "America/New_York", ticker="REFERENCE")
    shared = bm.BenchmarkPreparation(reference)
    calls = []
    original = pd.Series.resample

    def observe(self, *args, **kwargs):
        if self.name == "REFERENCE":
            calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "resample", observe)
    for seed in range(5):
        prices = _prices(seed, "Europe/Rome")
        bm._compute_single_benchmark_metrics(
            prices, reference, benchmark_preparation=shared
        )
    assert len(calls) == 1


def test_same_clipped_reference_resampled_once_across_performance_rows(
    monkeypatch,
):
    reference = _prices(26, "America/New_York", ticker="REFERENCE")
    shared = bm.BenchmarkPreparation(reference)
    calls = []
    original = pd.Series.resample

    def observe(self, *args, **kwargs):
        if self.name == "REFERENCE":
            calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "resample", observe)
    for seed in range(5):
        row = {}
        bm._populate_perf_row(
            row, _prices(seed, "Europe/Rome"), reference,
            benchmark_preparation=shared,
        )
    assert len(calls) == 1
    assert len(shared._windows) == 1


@pytest.mark.parametrize("consumer", ["benchmark", "performance"])
@pytest.mark.parametrize("source_ticker", ["SAME.MI", "SAME.DE"])
def test_metric_rows_match_uncached_path_for_exact_listings(
    consumer, source_ticker,
):
    prices = _prices(31, "Europe/Rome", ticker=source_ticker)
    reference = _prices(32, "America/New_York", ticker="REFERENCE")
    risk_free = pd.Series(0.0001, index=reference.index)
    shared = bm.BenchmarkPreparation(reference)
    if consumer == "benchmark":
        expected = bm._compute_single_benchmark_metrics(
            prices, reference, risk_free
        )
        actual = bm._compute_single_benchmark_metrics(
            prices, reference, risk_free, benchmark_preparation=shared
        )
    else:
        expected, actual = {}, {}
        bm._populate_perf_row(expected, prices, reference, risk_free)
        bm._populate_perf_row(
            actual, prices, reference, risk_free,
            benchmark_preparation=shared,
        )
    pd.testing.assert_series_equal(
        pd.Series(actual), pd.Series(expected), check_exact=True
    )


@pytest.mark.parametrize("limit", [0, -1, 1.5, np.nan, np.inf, "3"])
def test_window_cache_rejects_nonpositive_or_nonintegral_limits(limit):
    with pytest.raises((TypeError, ValueError)):
        bm.BenchmarkPreparation(_prices(33, None), max_windows=limit)


def test_window_cache_is_bounded_lru_and_uses_local_session_date_keys():
    reference = _prices(35, "Europe/Rome", ticker="REFERENCE")
    shared = bm.BenchmarkPreparation(reference, max_windows=2)
    a = shared.window("2025-02-03", "2025-10-20")
    b = shared.window("2025-03-03", "2025-10-20")
    assert shared.window(
        pd.Timestamp("2025-02-03 09:30", tz="Europe/Rome"),
        pd.Timestamp("2025-10-20 17:30", tz="Europe/Rome"),
    ) is a
    shared.window("2025-04-03", "2025-10-20")
    assert len(shared._windows) == 2
    assert shared.window("2025-02-03", "2025-10-20") is a
    assert shared.window("2025-03-03", "2025-10-20") is not b
    assert len(shared._windows) == 2


def test_preparations_cannot_be_attributed_to_another_source():
    source = _prices(37, None, ticker="REFERENCE.MI")
    other = _prices(38, None, ticker="REFERENCE.DE")
    shared = bm.BenchmarkPreparation(source)
    with pytest.raises(ValueError, match="different source"):
        bm._compute_single_benchmark_metrics(
            _prices(39, None), other, benchmark_preparation=shared
        )
    with pytest.raises(ValueError, match="different reference"):
        stats._compute_beta_alpha(
            _prices(39, None), other, prepared_benchmark=shared.full
        )


class _EvidenceMustNotBeCopied:
    def __deepcopy__(self, memo):
        raise AssertionError("weekly preparation copied provider evidence")


def test_snapshot_owns_data_without_changing_source_evidence():
    reference = _prices(41, "Europe/Rome", ticker="REFERENCE")
    evidence = _EvidenceMustNotBeCopied()
    reference.attrs["source_evidence"] = evidence
    shared = bm.BenchmarkPreparation(reference)
    expected = shared.full.history.copy(deep=True)
    reference.iloc[0] = -1.0
    reference.index.values[0] += np.timedelta64(1, "D")
    pd.testing.assert_series_equal(
        shared.full.history, expected, check_exact=True
    )
    assert reference.attrs["source_evidence"] is evidence
    assert shared.full.history.attrs == {"resolved_ticker": "REFERENCE"}
    with pytest.raises(AttributeError):
        shared.full.history = reference


def test_short_reference_does_not_force_resampling(monkeypatch):
    reference = pd.Series(np.arange(9.0), name="SHORT")
    prepared = stats.PreparedWeeklyBenchmark(reference)

    def unexpected(*args, **kwargs):
        raise AssertionError("short reference should return unavailable first")

    monkeypatch.setattr(pd.Series, "resample", unexpected)
    actual = stats._compute_beta_alpha(
        _prices(43, None), prepared.history, prepared_benchmark=prepared
    )
    assert all(np.isnan(value) for value in actual)


def test_parallel_readers_share_one_weekly_preparation(monkeypatch):
    reference = _prices(45, "Europe/Rome", ticker="REFERENCE")
    shared = bm.BenchmarkPreparation(reference, max_windows=2)
    calls = []
    original = pd.Series.resample

    def observe(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "resample", observe)

    def read(_):
        return shared.window("2025-02-03", "2025-10-20").weekly_prices

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(read, range(16)))
    assert all(result is results[0] for result in results)
    assert len(calls) == 1
    assert len(shared._windows) == 1


def test_resolved_benchmark_annotations_resolve():
    assert get_type_hints(bm.ResolvedBenchmark)["history"] is pd.Series
