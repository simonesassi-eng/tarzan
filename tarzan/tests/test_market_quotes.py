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


# ---------------------------------------------------------------------------
# Intraday-only EUR sibling fallback (.MI → .DE/.PA/...)
# ---------------------------------------------------------------------------
def _intra(values, day="2026-07-10", start="09:00", freq="5min"):
    idx = pd.date_range(f"{day} {start}", periods=len(values), freq=freq, tz="Europe/Rome")
    return pd.Series([float(v) for v in values], index=idx)


def test_sibling_symbols_only_eur_venues():
    assert mq._sibling_symbols("NTSG.MI")[0] == "NTSG.DE"
    assert set(mq._sibling_symbols("NTSG.MI")) == {"NTSG.DE", "NTSG.PA", "NTSG.AS", "NTSG.F"}
    # No siblings for indices, FX, futures, crypto, or bare US tickers.
    assert mq._sibling_symbols("^GSPC") == []
    assert mq._sibling_symbols("EURUSD=X") == []
    assert mq._sibling_symbols("CL=F") == []
    assert mq._sibling_symbols("BTC-USD") == []
    assert mq._sibling_symbols("AAPL") == []


def test_resolve_intraday_falls_back_to_sibling(monkeypatch):
    # Milan empty, Xetra populated → resolver returns the Xetra series keyed
    # under the original .MI symbol, tagged with the source listing.
    sib = _intra([29.2, 29.4, 29.66])

    def fake_fetch(symbols):
        return {"NTSG.DE": sib} if "NTSG.DE" in symbols else {}

    monkeypatch.setattr(mq, "_fetch_intraday", fake_fetch)
    # Primary daily close near the sibling → passes the collision guard.
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _close([29.0, 29.19]))
    resolved = mq._resolve_intraday(["NTSG.MI"])
    assert "NTSG.MI" in resolved
    series, src = resolved["NTSG.MI"]
    assert src == "NTSG.DE"
    assert len(series) == 3


def test_resolve_intraday_rejects_price_collision(monkeypatch):
    # Sibling exists but its price is wildly off the primary close → treated
    # as a different instrument (ticker collision) and rejected.
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": _intra([120.0, 121.0])})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _close([29.0, 29.19]))
    assert mq._resolve_intraday(["NTSG.MI"]) == {}


def test_broker_1d_uses_sibling_prev_close(monkeypatch):
    # cur (Xetra last) and prev (Xetra previous close) must both come from the
    # sibling feed, so the % is the coherent Xetra EUR move (not a Milan/Xetra
    # cross), and the row is keyed under the requested .MI ticker.
    sib = _intra([29.2, 29.66])
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": sib} if "NTSG.DE" in symbols else {})

    def fake_hist(symbol):
        # Xetra prev close 29.18; Milan daily also present but must NOT be used.
        return {"NTSG.DE": _close([29.0, 29.18]),
                "NTSG.MI": _close([28.0, 29.19])}.get(symbol)

    monkeypatch.setattr("tarzan.data.enricher._fetch_history", fake_hist)
    monkeypatch.setattr(mq, "market_open_now", lambda s: True)
    res = mq.broker_1d(["NTSG.MI"])
    assert "NTSG.MI" in res
    # 29.66 / 29.18 - 1 = +1.645% (Xetra-consistent), not 29.66/29.19.
    assert round(res["NTSG.MI"]["pct"], 3) == round((29.66 / 29.18 - 1) * 100, 3)
    assert res["NTSG.MI"]["live"] is True


def test_fetch_intraday_with_fallback_keys_on_original(monkeypatch):
    sib = _intra([29.2, 29.4])
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": sib} if "NTSG.DE" in symbols else {})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _close([29.0, 29.19]))
    out = mq._fetch_intraday_with_fallback(["NTSG.MI"])
    assert "NTSG.MI" in out and "NTSG.DE" not in out
    assert len(out["NTSG.MI"]) == 2


# ---------------------------------------------------------------------------
# Closed-session % uses the official daily close, not the last intraday tick
# ---------------------------------------------------------------------------
def _daily(dates_values):
    idx = pd.to_datetime([d for d, _ in dates_values])
    return pd.DataFrame({"Close": [v for _, v in dates_values]}, index=idx)


def test_broker_1d_closed_uses_official_close(monkeypatch):
    # Intraday last tick is a lone high print (29.66 @ 17:19); the official
    # 17:30 auction close is 29.415. With the session CLOSED, the % must use
    # the official close → +0.81%, not the +1.64% the last tick would give.
    intra = _intra([29.30, 29.66], day="2026-07-10", start="17:14")
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": intra} if "NTSG.DE" in symbols else {})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _daily([("2026-07-09", 29.18), ("2026-07-10", 29.415)]))
    monkeypatch.setattr(mq, "market_open_now", lambda s: False)  # session closed
    res = mq.broker_1d(["NTSG.MI"])
    assert res["NTSG.MI"]["live"] is False
    assert round(res["NTSG.MI"]["pct"], 2) == round((29.415 / 29.18 - 1) * 100, 2)  # +0.81%


def test_broker_1d_live_uses_last_tick(monkeypatch):
    # While the session is LIVE there is no official close yet, so the latest
    # intraday tick is the correct "current" price → +1.64%.
    intra = _intra([29.30, 29.66], day="2026-07-10", start="11:00")
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": intra} if "NTSG.DE" in symbols else {})
    # No same-day daily bar yet (market still open); only the prior close.
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _daily([("2026-07-09", 29.18)]))
    monkeypatch.setattr(mq, "market_open_now", lambda s: True)  # session live
    res = mq.broker_1d(["NTSG.MI"])
    assert res["NTSG.MI"]["live"] is True
    assert round(res["NTSG.MI"]["pct"], 2) == round((29.66 / 29.18 - 1) * 100, 2)  # +1.64%


def test_broker_1d_closed_prefers_primary_listing_over_sibling(monkeypatch):
    # .MI intraday empty → sparkline borrows Xetra; but once CLOSED the % must
    # come from the PRIMARY (Milan) official close, not the Xetra twin.
    # Milan closes 29.19 → 29.385 (+0.67%); Xetra closes 29.18 → 29.415 (+0.81%).
    sib = _intra([29.30, 29.66])
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": sib} if "NTSG.DE" in symbols else {})

    def fake_hist(symbol):
        return {"NTSG.MI": _daily([("2026-07-09", 29.19), ("2026-07-10", 29.385)]),
                "NTSG.DE": _daily([("2026-07-09", 29.18), ("2026-07-10", 29.415)])}.get(symbol)

    monkeypatch.setattr("tarzan.data.enricher._fetch_history", fake_hist)
    monkeypatch.setattr(mq, "market_open_now", lambda s: False)  # closed
    res = mq.broker_1d(["NTSG.MI"])
    assert res["NTSG.MI"]["live"] is False
    assert round(res["NTSG.MI"]["pct"], 2) == round((29.385 / 29.19 - 1) * 100, 2)  # +0.67%


def test_broker_1d_closed_falls_back_to_sibling_close(monkeypatch):
    # Primary listing has no official close for the day (e.g. Milan daily not
    # yet updated) → use the sibling's official close rather than nothing.
    sib = _intra([29.30, 29.66])
    monkeypatch.setattr(mq, "_fetch_intraday",
                        lambda symbols: {"NTSG.DE": sib} if "NTSG.DE" in symbols else {})

    def fake_hist(symbol):
        # NTSG.MI: only the prior day (no 07-10 close). NTSG.DE: full.
        return {"NTSG.MI": _daily([("2026-07-09", 29.19)]),
                "NTSG.DE": _daily([("2026-07-09", 29.18), ("2026-07-10", 29.415)])}.get(symbol)

    monkeypatch.setattr("tarzan.data.enricher._fetch_history", fake_hist)
    monkeypatch.setattr(mq, "market_open_now", lambda s: False)
    res = mq.broker_1d(["NTSG.MI"])
    assert round(res["NTSG.MI"]["pct"], 2) == round((29.415 / 29.18 - 1) * 100, 2)  # +0.81%
