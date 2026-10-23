"""Statistical copies must be cheap without weakening source-evidence contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.benchmarks import (
    _CURRENCY_ATTR,
    _NATIVE_ATTR,
    _compute_single_benchmark_metrics,
    _populate_perf_row,
    preprocess_benchmarks,
)
from tarzan.engine.stats import (
    PERIOD_WINDOWS,
    _series_ticker,
    compute_period_return,
    statistical_series,
)


class _SourceEvidenceMustNotBeCopied:
    def __deepcopy__(self, memo):
        raise AssertionError("statistics attempted to copy provider evidence")


def _prices(*, timezone=None, name="Close", ticker="XDEQ.MI"):
    index = pd.bdate_range("2020-01-02", "2026-08-18", tz=timezone, name="session")
    rng = np.random.default_rng(25)
    series = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(index))),
        index=index,
        name=name,
    )
    series.attrs["resolved_ticker"] = ticker
    return series


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
def test_statistical_copy_preserves_values_index_name_and_listing(timezone):
    source = _prices(timezone=timezone)
    original_values = source.array.copy()
    original_index = source.index.copy(deep=True)
    evidence = _SourceEvidenceMustNotBeCopied()
    source.attrs.update({
        "tarzan_native_series": evidence,
        "tarzan_price_origins": evidence,
        "requested_ticker": "XDEQ",
    })

    result = statistical_series(source)

    assert result is not source
    assert result.dtype == source.dtype
    assert result.name == source.name
    pd.testing.assert_index_equal(result.index, original_index, exact=True)
    np.testing.assert_array_equal(result.array, original_values)
    assert result.attrs == {"resolved_ticker": "XDEQ.MI"}
    assert source.attrs["tarzan_native_series"] is evidence
    assert source.attrs["tarzan_price_origins"] is evidence
    assert source.attrs["requested_ticker"] == "XDEQ"

    # Both directions are isolated, including the array behind the index.
    result.iloc[0] = -1.0
    result.index.values[0] += np.timedelta64(1, "D")
    result.attrs["resolved_ticker"] = "OTHER"
    np.testing.assert_array_equal(source.array, original_values)
    pd.testing.assert_index_equal(source.index, original_index, exact=True)
    assert source.attrs["resolved_ticker"] == "XDEQ.MI"


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64"])
def test_statistical_copy_preserves_missing_values_and_extension_dtype(dtype):
    source = pd.Series([1, None, 3], dtype=dtype, name="PX")
    result = statistical_series(source)
    pd.testing.assert_series_equal(result, source, check_exact=True)
    result.iloc[-1] = 99
    assert source.iloc[-1] == 3


def test_statistical_copy_preserves_duplicate_label_policy():
    source = _prices().set_flags(allows_duplicate_labels=False)
    result = statistical_series(source)
    assert result.flags.allows_duplicate_labels is False


@pytest.mark.parametrize("name", [None, "RSSY", ("price", "RSSY")])
def test_statistical_copy_keeps_name_based_calendar_fallback(name):
    source = pd.Series([1.0, 2.0], name=name)
    result = statistical_series(source)
    assert result.name == name
    assert _series_ticker(result) == _series_ticker(source)


def test_statistical_copy_keeps_empty_series_dtype_and_index():
    source = pd.Series(
        dtype="float64", index=pd.DatetimeIndex([], tz="Europe/Rome"), name="PX",
    )
    result = statistical_series(source)
    pd.testing.assert_series_equal(result, source, check_exact=True)
    assert _series_ticker(result) == "PX"


@pytest.mark.parametrize("bucket", [*PERIOD_WINDOWS, "ytd"])
def test_return_windows_use_the_same_resolved_listing_calendar(bucket, monkeypatch):
    source = _prices()
    result = statistical_series(source)
    tickers = []
    from tarzan.data import exchange_calendar

    original = exchange_calendar.last_session_on_or_before

    def observe(ticker, date):
        tickers.append(ticker)
        return original(ticker, date)

    monkeypatch.setattr(exchange_calendar, "last_session_on_or_before", observe)
    assert compute_period_return(result, bucket) == compute_period_return(source, bucket)
    if bucket != "ytd":
        assert tickers
        assert set(tickers) == {"XDEQ.MI"}


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
@pytest.mark.parametrize("consumer", ["benchmark", "performance"])
def test_metric_rows_ignore_heavy_metadata_without_changing_numbers(timezone, consumer):
    prices = _prices(timezone=timezone)
    reference = _prices(timezone="America/New_York", name="REF", ticker="RSSY")
    reference *= np.linspace(1.0, 1.3, len(reference))
    risk_free = pd.Series(
        np.linspace(0.01, 0.04, len(prices)) / 252,
        index=prices.index,
        name="daily risk-free",
    )

    def calculate():
        if consumer == "benchmark":
            return _compute_single_benchmark_metrics(prices, reference, risk_free)
        row = {}
        _populate_perf_row(row, prices, reference, risk_free)
        return row

    expected = calculate()
    inputs = (prices, reference, risk_free)
    original_values = [series.array.copy() for series in inputs]
    evidence = _SourceEvidenceMustNotBeCopied()
    for series in inputs:
        series.attrs["tarzan_native_series"] = evidence
        series.attrs["tarzan_price_origins"] = evidence
    actual = calculate()

    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected), check_exact=True)
    for source, values in zip(inputs, original_values):
        np.testing.assert_array_equal(source.array, values)
        assert source.attrs["tarzan_native_series"] is evidence
        assert source.attrs["tarzan_price_origins"] is evidence


def test_preprocessing_keeps_one_independent_native_tape_and_all_source_evidence():
    history = _prices(name="EUR tape", ticker="RSSY")
    native = history * 1.2
    native.name = "native tape"
    native.attrs["tarzan_price_origins"] = {"2026-08-18": "NATIVE_PRIMARY"}
    native.attrs["unrecognized_evidence"] = {"provider": ["native"]}
    history.attrs.update({
        _NATIVE_ATTR: native,
        _CURRENCY_ATTR: "USD",
        "tarzan_price_origins": {"2026-08-18": "PRIMARY_WITH_FX"},
        "unrecognized_evidence": {"provider": ["EUR"]},
    })

    catalog, errors = preprocess_benchmarks(
        {"Example": "REQUEST"}, fetch_history=lambda _: history,
    )
    record = catalog["Example"]

    assert errors == ()
    assert record.ticker == "RSSY"
    assert record.currency == "USD"
    assert _NATIVE_ATTR not in record.history.attrs
    assert record.history.attrs["tarzan_price_origins"] == {"2026-08-18": "PRIMARY_WITH_FX"}
    assert record.history_native.attrs["tarzan_price_origins"] == {"2026-08-18": "NATIVE_PRIMARY"}
    np.testing.assert_array_equal(record.history.array, history.array)
    np.testing.assert_array_equal(record.history_native.array, native.array)
    pd.testing.assert_index_equal(record.history.index, history.index)
    pd.testing.assert_index_equal(record.history_native.index, native.index)
    assert record.history_native.name == "RSSY"
    assert history.attrs[_NATIVE_ATTR] is native
    assert native.name == "native tape"

    # A consumer cannot corrupt either the producer's EUR/native prices or its
    # evidence, including metadata unknown to today's implementation.
    record.history.iloc[0] = -1.0
    record.history_native.iloc[0] = -2.0
    record.history.attrs["unrecognized_evidence"]["provider"].append("changed")
    record.history_native.attrs["unrecognized_evidence"]["provider"].append("changed")
    assert history.iloc[0] > 0
    assert native.iloc[0] > 0
    assert history.attrs["unrecognized_evidence"] == {"provider": ["EUR"]}
    assert native.attrs["unrecognized_evidence"] == {"provider": ["native"]}
