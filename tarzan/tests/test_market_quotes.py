"""Tests for the yfinance-style market-quotes fetcher (network-free).

``fetch_market_quotes`` is bound at import time so the autouse fixture that
stubs the module attribute does not shadow the real implementation here;
the underlying history fetch is monkeypatched instead.
"""

from __future__ import annotations

import pandas as pd

import tarzan.data.market_quotes as mq
from tarzan.data.market_quotes import fetch_market_quotes  # real impl, bound now


def _close(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


def test_builds_quotes_from_history(monkeypatch):
    mq._memo = None
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda symbol: _close([100.0, 102.0]))
    quotes = fetch_market_quotes(force=True)
    try:
        assert len(quotes) == len(mq.MARKETS)
        first = quotes[0]
        assert first["value"] == 102.0
        assert first["pct"] == 2.0
        assert first["category"] in mq.CATEGORY_ORDER
        assert first["spark"][-1] == 102.0
    finally:
        mq._memo = None


def test_skips_symbols_that_fail_or_are_short(monkeypatch):
    mq._memo = None
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})

    def _fetch(symbol):
        if symbol == "^GSPC":
            return _close([100.0, 110.0])      # ok → +10%
        if symbol == "^DJI":
            return _close([100.0])             # too short → skipped
        raise RuntimeError("network down")     # everything else fails

    monkeypatch.setattr("tarzan.data.enricher._fetch_history", _fetch)
    quotes = fetch_market_quotes(force=True)
    try:
        assert [q["name"] for q in quotes] == ["S&P 500"]
        assert quotes[0]["pct"] == 10.0
    finally:
        mq._memo = None


def test_empty_when_fetch_layer_unavailable(monkeypatch):
    mq._memo = None
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda symbol: None)
    try:
        assert fetch_market_quotes(force=True) == []
    finally:
        mq._memo = None


def test_intraday_path_sets_baseline_to_prior_close(monkeypatch):
    mq._memo = None
    import pandas as pd
    # Daily closes end the day BEFORE the intraday day → prior close = 200.
    daily = pd.DataFrame({"Close": [180.0, 200.0]},
                         index=pd.to_datetime(["2026-06-23", "2026-06-24"]))
    intra = pd.Series([202.0, 205.0, 210.0],
                      index=pd.to_datetime(["2026-06-25 09:00", "2026-06-25 12:00",
                                            "2026-06-25 16:00"]))
    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {"^GSPC": intra})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda symbol: daily)
    q = {d["name"]: d for d in fetch_market_quotes(force=True)}
    try:
        sp = q["S&P 500"]
        assert sp["value"] == 210.0          # latest intraday
        assert sp["baseline"] == 200.0       # prior daily close (the 0% line)
        assert round(sp["pct"], 2) == 5.0    # 210 vs 200
    finally:
        mq._memo = None
