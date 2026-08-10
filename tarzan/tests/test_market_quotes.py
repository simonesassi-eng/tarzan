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


def test_fetch_intraday_logs_symbols_missing_from_batch_response(monkeypatch, caplog):
    """Reproduces the shape of the real 'futures show no intraday' report:
    a batch response that has some symbols but not others should say so
    in the logs, not fail silently - the newsletter's own fallback (daily-
    close chart + a 'no intraday' label) looks intentional either way, so
    this is the only signal that distinguishes a real data gap from a
    one-off Yahoo miss."""
    import logging
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)

    idx = pd.date_range("2026-08-10 09:30", periods=3, freq="15min")
    cols = pd.MultiIndex.from_product([["^GSPC", "ES=F"], ["Close", "Volume"]])
    raw = pd.DataFrame(
        [[100.0, 1000, 101.0, 500],
         [100.5, 1000, 101.5, 500],
         [101.0, 1000, 102.0, 500]],
        index=idx, columns=cols,
    )
    monkeypatch.setattr("tarzan.data._yf_net.fetch_yf", lambda fn, **kw: raw)

    with caplog.at_level(logging.INFO):
        out = mq._fetch_intraday(["^GSPC", "ES=F", "GC=F"])

    assert set(out) == {"^GSPC", "ES=F"}  # present in the batch response
    assert "GC=F" not in out              # absent from it, silently before
    assert any("GC=F" in m for m in caplog.messages)


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


def _pin_now(monkeypatch, ts):
    """Pin _has_intraday's staleness clock to ``ts`` (Europe/Rome), so
    fixture series built on a fixed historical date aren't flagged stale
    just because the real wall clock has since moved on."""
    pinned = pd.Timestamp(ts, tz="Europe/Rome")
    monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: pinned)


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
    _pin_now(monkeypatch, "2026-07-10 09:10")
    resolved = mq._resolve_intraday(["NTSG.MI"])
    assert "NTSG.MI" in resolved
    series, src = resolved["NTSG.MI"]
    assert src == "NTSG.DE"
    assert len(series) == 3


def test_resolve_intraday_routes_around_stale_primary_feed(monkeypatch):
    # Reproduces the reported bug: Milan's intraday feed prints two ticks at
    # the open (09:00/09:05) and then goes silent for the rest of the
    # session — a documented Yahoo .MI behavior. len(ser) >= 2 alone used to
    # treat that as "has intraday" and never try the healthier Xetra sibling,
    # freezing the sparkline/1D% at the stale open-of-day print all day.
    stale_primary = _intra([29.10, 29.12], day="2026-07-10", start="09:00")
    healthy_sibling = _intra(
        [29.10, 29.20, 29.30, 29.45, 29.66], day="2026-07-10", start="09:00", freq="90min"
    )

    def fake_fetch(symbols):
        out = {}
        if "NTSG.MI" in symbols:
            out["NTSG.MI"] = stale_primary
        if "NTSG.DE" in symbols:
            out["NTSG.DE"] = healthy_sibling
        return out

    monkeypatch.setattr(mq, "_fetch_intraday", fake_fetch)
    monkeypatch.setattr("tarzan.data.enricher._fetch_history",
                        lambda s: _close([29.0, 29.19]))
    # It's now mid-afternoon; the primary's last print (09:05) is hours
    # stale, while the sibling's last print (15:00) is still fresh.
    _pin_now(monkeypatch, "2026-07-10 15:05")
    resolved = mq._resolve_intraday(["NTSG.MI"])
    assert "NTSG.MI" in resolved
    series, src = resolved["NTSG.MI"]
    assert src == "NTSG.DE", "stale primary must not block the sibling fallback"
    assert len(series) == 5


def test_resolve_intraday_logs_reason_when_all_candidates_exhausted(monkeypatch, caplog):
    # Mirrors what CL2 likely hit tonight: a stale Milan primary, and every
    # EUR-venue sibling either has no data or prices too far from the
    # canonical close to trust (ticker-root collision on another exchange).
    # Previously these reasons were only logged at DEBUG, invisible in the
    # INFO-level CI log — silent, correct rejection looked identical to a
    # broken fallback. All three reasons should now surface at INFO.
    stale_primary = _intra([94.0, 94.05], day="2026-07-08", start="09:00")
    mismatched_sibling = _intra([250.0, 251.0], day="2026-07-10", start="14:55")

    def fake_fetch(symbols):
        out = {}
        if "NTSG.MI" in symbols:
            out["NTSG.MI"] = stale_primary
        if "NTSG.PA" in symbols:
            out["NTSG.PA"] = mismatched_sibling
        return out

    def fake_hist(ticker):
        return pd.DataFrame({"Close": [95.0]}, index=pd.to_datetime(["2026-07-10"]))

    monkeypatch.setattr(mq, "_fetch_intraday", fake_fetch)
    monkeypatch.setattr("tarzan.data.enricher._fetch_history", fake_hist)
    _pin_now(monkeypatch, "2026-07-10 15:05")
    with caplog.at_level("INFO", logger="tarzan.data.market_quotes"):
        resolved = mq._resolve_intraday(["NTSG.MI"])
    assert resolved == {}
    text = caplog.text
    assert "stale" in text and "NTSG.MI" in text
    assert "NTSG.MI→NTSG.PA rejected" in text
    assert "exhausted for NTSG.MI" in text


def test_broker_1d_closed_stale_primary_anchors_wrong_day(monkeypatch):
    # Reproduces a second symptom of the same bug: broker_1d derives `iday`
    # (which calendar day's official close to look up) from the primary
    # intraday series' *last timestamp*. If that primary is stale from a
    # PRIOR calendar day — not just old-within-today — the old length-only
    # _has_intraday() still accepted it, so the closed-session % got
    # computed for the wrong day entirely: not frozen, but a wrong sign and
    # magnitude, e.g. Tarzan showing +0.53% on a holding that actually
    # closed down.
    stale_primary = _intra([94.0, 94.05], day="2026-07-08", start="09:00")
    # Last print (95.2) close to the canonical last-known close (95.0) used
    # below, so the sibling collision guard (price coherence) accepts it.
    fresh_sibling = _intra(
        [94.0, 94.2, 94.5, 94.8, 95.2], day="2026-07-10", start="09:00", freq="120min"
    )

    def fake_fetch(symbols):
        out = {}
        if "NTSG.MI" in symbols:
            out["NTSG.MI"] = stale_primary
        if "NTSG.DE" in symbols:
            out["NTSG.DE"] = fresh_sibling
        return out

    def fake_hist(ticker):
        return pd.DataFrame(
            {"Close": [90.0, 100.0, 100.0, 95.0]},
            index=pd.to_datetime(["2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"]),
        )

    monkeypatch.setattr(mq, "_fetch_intraday", fake_fetch)
    monkeypatch.setattr("tarzan.data.enricher._fetch_history", fake_hist)
    monkeypatch.setattr(mq, "market_open_now", lambda s: False)  # closed
    _pin_now(monkeypatch, "2026-07-10 17:10")
    res = mq.broker_1d(["NTSG.MI"])
    assert "NTSG.MI" in res
    # Correct: anchored to 2026-07-10 (95 vs prior close 100) = -5.00%.
    # The pre-fix bug anchored to the stale primary's day, 2026-07-08
    # (100 vs prior close 90) = +11.11% — wrong sign, wrong magnitude.
    assert round(res["NTSG.MI"]["pct"], 2) == -5.00


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
    _pin_now(monkeypatch, "2026-07-10 09:05")
    res = mq.broker_1d(["NTSG.MI"])
    assert "NTSG.MI" in res
    selected = res["NTSG.MI"]
    # 29.66 / 29.18 - 1 = +1.645% (Xetra-consistent), not 29.66/29.19.
    assert round(selected["pct"], 3) == round((29.66 / 29.18 - 1) * 100, 3)
    assert selected["live"] is True
    assert selected["source_ticker"] == "NTSG.DE"
    assert selected["intraday_source_ticker"] == "NTSG.DE"
    assert selected["intraday_series"].equals(sib)
    assert selected["intraday_baseline"] == 29.18


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
    _pin_now(monkeypatch, "2026-07-10 17:19")
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
    _pin_now(monkeypatch, "2026-07-10 11:05")
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
    _pin_now(monkeypatch, "2026-07-10 09:05")
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
    _pin_now(monkeypatch, "2026-07-10 09:05")
    res = mq.broker_1d(["NTSG.MI"])
    assert round(res["NTSG.MI"]["pct"], 2) == round((29.415 / 29.18 - 1) * 100, 2)  # +0.81%


# ---------------------------------------------------------------------------
# Memo TTL — a long-running process must not serve stale quotes forever
# ---------------------------------------------------------------------------
def test_memo_serves_within_ttl_and_refetches_after(monkeypatch):
    mq._memo = None
    mq._memo_at = 0.0
    calls = {"n": 0}

    def _fetch(symbol):
        calls["n"] += 1
        return _close([100.0, 101.0])

    monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
    monkeypatch.setattr("tarzan.data.enricher._fetch_history", _fetch)

    # Drive a controllable clock.
    clock = {"t": 1000.0}
    monkeypatch.setattr(mq._time, "monotonic", lambda: clock["t"])
    try:
        fetch_market_quotes()               # cold fill
        after_first = calls["n"]
        assert after_first > 0

        fetch_market_quotes()               # within TTL → served from memo
        assert calls["n"] == after_first

        clock["t"] += mq._MEMO_TTL_SECONDS + 1   # advance past the TTL
        fetch_market_quotes()               # expired → refetch
        assert calls["n"] > after_first
    finally:
        mq._memo = None
        mq._memo_at = 0.0


# ---------------------------------------------------------------------------
# session_caption — trading hours / continuous-market caption
# ---------------------------------------------------------------------------
def test_session_caption_continuous_market_shows_approx_24h():
    for ticker in ("ES=F", "NQ=F", "YM=F", "RTY=F", "EURUSD=X", "BTC-USD"):
        assert mq.session_caption(ticker) == "\u224824h", ticker


def test_session_caption_bounded_session_shows_hours_and_zone():
    assert mq.session_caption("^GSPC") == "09:30\u201316:00 ET"
    assert mq.session_caption("^FTSE") == "08:00\u201316:30 GMT"
    assert mq.session_caption("^N225") == "09:00\u201315:00 JST"


def test_session_caption_unknown_exchange_is_empty():
    assert mq.session_caption("SOME.XX") == ""
    assert mq.session_caption("") == ""


# ---------------------------------------------------------------------------
# New US index futures — must not collide in name with their cash index
# ---------------------------------------------------------------------------
def test_new_index_futures_present_with_correct_ticker_and_category():
    wanted = {"ES=F": "S&P 500 (FUT)", "YM=F": "Dow 30 (FUT)",
              "NQ=F": "Nasdaq 100 (FUT)", "RTY=F": "Russell 2000 (FUT)"}
    by_ticker = {t: n for n, t, c in mq.MARKETS}
    for ticker, name in wanted.items():
        assert by_ticker.get(ticker) == name, ticker

    cats = {t: c for _n, t, c in mq.MARKETS}
    for ticker in wanted:
        assert cats[ticker] == "US"
        assert mq.is_continuous_market(ticker)


def test_nasdaq_composite_and_100_are_both_present_and_distinct():
    # Composite (^IXIC), 100 (^NDX) and the 100's futures (NQ=F) all show
    # up as their own entries -- three genuinely different things, not one
    # collapsed into another.
    by_ticker = {t: n for n, t, c in mq.MARKETS}
    assert by_ticker["^IXIC"] == "Nasdaq Composite"
    assert by_ticker["^NDX"] == "Nasdaq 100"
    assert by_ticker["NQ=F"] == "Nasdaq 100 (FUT)"
    cats = {t: c for _n, t, c in mq.MARKETS}
    assert cats["^NDX"] == "US"
    assert not mq.is_continuous_market("^NDX")


def test_markets_names_are_unique():
    # A duplicate name (e.g. a futures entry sharing its cash index's name)
    # collides in any lookup keyed by name, fetch_market_quotes results
    # included -- this is what test_intraday_path_sets_baseline_to_prior_close
    # would have caught if ES=F had been added as bare "S&P 500".
    names = [n for n, _t, _c in mq.MARKETS]
    assert len(names) == len(set(names)), (
        [n for n in names if names.count(n) > 1]
    )


# ---------------------------------------------------------------------------
# futures_open_now / market_status — real CME Globex schedule, not literal
# 24/7. Reference dates: 2024-01-01 is a known Monday, so 01-05/06/07 are
# Fri/Sat/Sun and 01-08 is the following Monday.
# ---------------------------------------------------------------------------
def _et(y, m, d, hh, mm):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("America/New_York"))


def test_futures_open_now_weekday_daytime_is_open():
    assert mq.futures_open_now(_et(2024, 1, 8, 10, 0)) is True   # Monday


def test_futures_open_now_saturday_is_closed():
    assert mq.futures_open_now(_et(2024, 1, 6, 12, 0)) is False  # Saturday


def test_futures_open_now_sunday_before_reopen_is_closed():
    assert mq.futures_open_now(_et(2024, 1, 7, 17, 59)) is False


def test_futures_open_now_sunday_at_reopen_is_open():
    assert mq.futures_open_now(_et(2024, 1, 7, 18, 0)) is True


def test_futures_open_now_friday_before_close_is_open():
    assert mq.futures_open_now(_et(2024, 1, 5, 16, 59)) is True


def test_futures_open_now_friday_at_close_is_closed():
    assert mq.futures_open_now(_et(2024, 1, 5, 17, 0)) is False


def test_futures_open_now_daily_maintenance_break_is_closed():
    assert mq.futures_open_now(_et(2024, 1, 8, 17, 30)) is False  # Monday
    assert mq.futures_open_now(_et(2024, 1, 8, 18, 0)) is True    # break ends


def test_fx_open_now_weekday_is_open_no_daily_break():
    assert mq.fx_open_now(_et(2024, 1, 8, 10, 0)) is True   # Monday
    assert mq.fx_open_now(_et(2024, 1, 8, 17, 30)) is True  # no CME-style break


def test_fx_open_now_weekend_closure():
    assert mq.fx_open_now(_et(2024, 1, 6, 12, 0)) is False        # Saturday
    assert mq.fx_open_now(_et(2024, 1, 7, 16, 59)) is False       # Sun pre-reopen
    assert mq.fx_open_now(_et(2024, 1, 7, 17, 0)) is True         # Sun reopen
    assert mq.fx_open_now(_et(2024, 1, 5, 16, 59)) is True        # Fri pre-close
    assert mq.fx_open_now(_et(2024, 1, 5, 17, 0)) is False        # Fri close


def test_market_status_continuous_instruments_have_no_day():
    assert mq.market_status("BTC-USD") == (True, "")


def test_market_status_fx_has_real_weekly_status():
    assert mq.market_status("EURUSD=X", _et(2024, 1, 8, 10, 0)) == (True, "Mon")
    assert mq.market_status("EURUSD=X", _et(2024, 1, 6, 12, 0)) == (False, "Fri")


def test_market_status_futures_open_shows_todays_day():
    assert mq.market_status("ES=F", _et(2024, 1, 8, 10, 0)) == (True, "Mon")
    # Reopened Sunday evening: today (Sunday), not Friday.
    assert mq.market_status("ES=F", _et(2024, 1, 7, 19, 0)) == (True, "Sun")


def test_market_status_futures_weekend_closure_shows_friday():
    assert mq.market_status("ES=F", _et(2024, 1, 6, 12, 0)) == (False, "Fri")
    assert mq.market_status("ES=F", _et(2024, 1, 7, 10, 0)) == (False, "Fri")


def test_market_status_futures_daily_break_shows_today_not_friday():
    # Monday 17:30 ET: closed for the hour-long break, but still Monday's
    # session, not the prior week's Friday.
    assert mq.market_status("ES=F", _et(2024, 1, 8, 17, 30)) == (False, "Mon")


def test_market_status_cash_market_open_shows_today():
    is_open, day = mq.market_status("^GSPC", _et(2024, 1, 8, 10, 0))  # Mon
    assert is_open is True and day == "Mon"


def test_market_status_cash_market_before_open_shows_previous_weekday():
    is_open, day = mq.market_status("^GSPC", _et(2024, 1, 8, 7, 0))  # Mon 7am
    assert is_open is False and day == "Fri"  # last session was Friday


def test_market_status_cash_market_weekend_shows_friday():
    is_open, day = mq.market_status("^GSPC", _et(2024, 1, 6, 12, 0))  # Sat
    assert is_open is False and day == "Fri"


def test_market_status_unknown_ticker_is_none():
    assert mq.market_status("SOME.XX") == (None, "")
