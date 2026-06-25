"""Tests for the Markets-strip snapshot derived from benchmark histories."""

from __future__ import annotations

import pandas as pd

from tarzan.export.newsletter import market_snapshot
from tarzan.models.portfolio import PortfolioMetrics


def _series(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_snapshot_computes_level_and_daily_change():
    m = PortfolioMetrics()
    m.benchmark_histories = {
        "S&P 500": _series([100.0, 110.0, 121.0]),   # +10% on the last day
        "Nasdaq 100": _series([200.0, 190.0]),       # -5% on the last day
    }
    snap = {d["name"]: d for d in market_snapshot(m)}
    assert snap["S&P 500"]["value"] == 121.0
    assert snap["S&P 500"]["pct"] == 10.0
    assert snap["Nasdaq 100"]["pct"] == -5.0
    # Spark series carries the points for the mini chart.
    assert snap["S&P 500"]["spark"][-1] == 121.0


def test_snapshot_skips_missing_or_too_short():
    m = PortfolioMetrics()
    m.benchmark_histories = {
        "S&P 500": _series([100.0]),   # only one point → skipped
        # MSCI World absent entirely → skipped
    }
    assert market_snapshot(m) == []


def test_snapshot_empty_without_histories():
    assert market_snapshot(PortfolioMetrics()) == []
