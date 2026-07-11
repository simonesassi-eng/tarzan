"""Live quotes for the newsletter "Markets" strip (yfinance-style).

A curated set mirroring the yfinance markets bar (US / Europe / Asia /
Crypto / Rates / Commodities / Currencies). For each instrument we return
the latest level, the change versus the previous close, and an intraday
"day" path (rebased to the previous close) for a two-tone sparkline.

Data sources, both best-effort and graceful:
  * level + previous close: the enricher's cached, throttled daily-history
    helper (reuses the on-disk price cache; no cold re-download on warm
    runs);
  * intraday day path: a single batched ``yfinance`` download (one request
    for all symbols) — if it fails or a symbol is missing, the daily
    history is used as the spark fallback.

Each quote: ``{name, symbol, category, value, change, pct, spark,
baseline}`` where ``baseline`` is the previous close (the 0% line the
sparkline shades green above / red below).
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, time as dtime
from typing import Optional

logger = logging.getLogger(__name__)

# A market is treated as "trading now" (live) when its latest intraday bar is
# no older than this. yfinance intraday lags ~15-30 min, so this keeps a live
# session marked live while flipping to "previous day" within ~1h of the close
# (so an evening newsletter doesn't mislabel a closed market as live).
_MARKET_OPEN_MAX_LAG_MIN = 60

# Regular cash-session hours per exchange group, keyed by a normalized code.
# (IANA timezone, (open_h, open_m), (close_h, close_m)). Holidays and lunch
# breaks are not modelled (best-effort); weekends are handled separately.
_SESSIONS: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "EU": ("Europe/Rome", (9, 0), (17, 30)),      # Milan/Xetra/Paris/Amsterdam
    "L":  ("Europe/London", (8, 0), (16, 30)),    # London
    "US": ("America/New_York", (9, 30), (16, 0)),  # US cash session
    "JP": ("Asia/Tokyo", (9, 0), (15, 0)),        # Tokyo
    "HK": ("Asia/Hong_Kong", (9, 30), (16, 0)),   # Hong Kong
    "CN": ("Asia/Shanghai", (9, 30), (15, 0)),    # Shanghai
    "AU": ("Australia/Sydney", (10, 0), (16, 0)),  # Sydney
    "KR": ("Asia/Seoul", (9, 0), (15, 30)),       # Seoul
}

# Index / bare symbols → exchange group (symbols without a Yahoo suffix).
_INDEX_EXCHANGE: dict[str, str] = {
    "^GSPC": "US", "^DJI": "US", "^IXIC": "US", "^RUT": "US", "^RUI": "US",
    "^VIX": "US", "^NDX": "US", "^SPXEW": "US",
    "^IRX": "US", "^FVX": "US", "^TNX": "US", "^TYX": "US",
    "^FTSE": "L",
    "^FCHI": "EU", "^GDAXI": "EU", "^STOXX50E": "EU", "^N100": "EU",
    "^N225": "JP", "^HSI": "HK", "^AXJO": "AU", "^KS11": "KR",
}

# Yahoo listing suffix → exchange group.
_SUFFIX_EXCHANGE: dict[str, str] = {
    "MI": "EU", "DE": "EU", "PA": "EU", "AS": "EU", "F": "EU",
    "L": "L",
    "SS": "CN", "SZ": "CN", "HK": "HK", "T": "JP",
    "AX": "AU", "KS": "KR",
}


def _exchange_for(ticker: str) -> Optional[str]:
    """Map a Yahoo ticker to its exchange-session group, or None when it has
    no fixed cash session (crypto/FX/futures)."""
    t = (ticker or "").upper()
    if not t:
        return None
    if t.endswith("-USD") or t.endswith("=X") or t.endswith("=F"):
        return None
    if t.startswith("^"):
        return _INDEX_EXCHANGE.get(t)
    if "." in t:
        return _SUFFIX_EXCHANGE.get(t.rsplit(".", 1)[1])
    # Bare ticker (no suffix, no caret) → assume a US listing.
    return "US"


def is_continuous_market(ticker: str) -> bool:
    """Whether the instrument trades ~around the clock with no bounded cash
    session: commodity/index futures (``=F``), FX pairs (``=X``) and crypto
    (``-USD``). These have no equity-style open→close, so their intraday
    sparkline should be drawn full-width (stretched) rather than "growing
    through the session"."""
    t = (ticker or "").upper()
    return t.endswith("=F") or t.endswith("=X") or t.endswith("-USD")


def market_open_now(ticker: str, now: Optional[datetime] = None) -> Optional[bool]:
    """Whether the instrument's primary exchange is in its regular trading
    session right now, judged by exchange hours (NOT bar recency).

    Returns True/False for instruments on a known cash exchange (mapped by
    Yahoo suffix or index symbol); True for 24/7 crypto; and None when the
    session concept doesn't cleanly apply (FX and futures trade nearly around
    the clock) so callers can fall back to recency. Weekends are closed;
    holidays are not modelled. Never raises."""
    t = (ticker or "").upper()
    if t.endswith("-USD"):
        return True  # crypto trades 24/7
    if t.endswith("=X") or t.endswith("=F"):
        return None  # FX / futures → let the caller decide by recency
    ex = _exchange_for(t)
    if ex is None:
        return None
    tzname, (oh, om), (ch, cm) = _SESSIONS[ex]
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
        n = now.astimezone(tz) if now is not None else datetime.now(tz)
    except Exception:  # noqa: BLE001
        return None
    if n.weekday() >= 5:  # Saturday / Sunday
        return False
    return dtime(oh, om) <= n.time() <= dtime(ch, cm)

# (display name, yfinance symbol, category), in display order. The strip
# shows at most 2 rows per category (the newsletter caps it).
MARKETS: list[tuple[str, str, str]] = [
    # US — equity indices + US Treasury yields (^IRX/^FVX/^TNX/^TYX), grouped
    # together since they are all US-market references.
    ("S&P 500", "^GSPC", "US"),
    ("Dow 30", "^DJI", "US"),
    ("Nasdaq", "^IXIC", "US"),
    ("Russell 2000", "^RUT", "US"),
    ("VIX", "^VIX", "US"),
    ("US 13-Wk", "^IRX", "US"),
    ("US 5-Yr", "^FVX", "US"),
    ("US 10-Yr", "^TNX", "US"),
    ("US 30-Yr", "^TYX", "US"),
    # Europe — equity indices + a German 10Y reference. Yahoo exposes no
    # German 10Y yield ticker (à la ^TNX), so "Bund 10Y" is a German
    # government-bond ETF proxy: iShares eb.rexx Government Germany 5.5-10.5yr
    # (EXHD.DE), a EUR PRICE centered on the ~10Y segment (moves inverse to
    # yield), not a yield.
    ("FTSE 100", "^FTSE", "Europe"),
    ("CAC 40", "^FCHI", "Europe"),
    ("DAX", "^GDAXI", "Europe"),
    ("Euronext 100", "^N100", "Europe"),
    ("Euro Stoxx 50", "^STOXX50E", "Europe"),
    ("Bund 10Y", "EXHD.DE", "Europe"),
    # Asia
    ("SSE Composite", "000001.SS", "Asia"),
    ("Nikkei 225", "^N225", "Asia"),
    ("Hang Seng", "^HSI", "Asia"),
    ("ASX 200", "^AXJO", "Asia"),
    ("KOSPI", "^KS11", "Asia"),
    # Commodities. Two global crude benchmarks are shown side by side: WTI
    # (CL=F, NYMEX/CME — the US reference) and Brent (BZ=F, ICE Futures Europe
    # — the international reference used for most of the world incl. Europe).
    ("WTI Crude", "CL=F", "Commodities"),
    ("Gold", "GC=F", "Commodities"),
    ("Silver", "SI=F", "Commodities"),
    ("Copper", "HG=F", "Commodities"),
    ("Natural Gas", "NG=F", "Commodities"),
    ("Brent Crude", "BZ=F", "Commodities"),
    ("Platinum", "PL=F", "Commodities"),
    # Currencies (fiat FX pairs + Bitcoin, both quoted vs USD and traded
    # ~around the clock).
    ("EUR/USD", "EURUSD=X", "Currencies"),
    ("USD/JPY", "JPY=X", "Currencies"),
    ("USD/GBP", "GBP=X", "Currencies"),
    ("Bitcoin", "BTC-USD", "Currencies"),
]

CATEGORY_ORDER = ["US", "Europe", "Asia", "Commodities", "Currencies"]

_memo: Optional[list[dict]] = None
_memo_at: float = 0.0  # monotonic timestamp the memo was filled
# Live quotes go stale within a session. Memoizing forever means a
# long-running process (a persistent worker / server, not the one-shot CLI)
# serves the same quotes across the market close and into the next session.
# A short TTL bounds that: within one CLI run every call is still served from
# the memo, but a process that outlives the TTL re-fetches.
_MEMO_TTL_SECONDS = 900  # 15 minutes (≈ yfinance intraday lag)


# Intraday-only sibling fallback. When a EUR listing has no intraday feed
# (the classic Borsa Italiana ``.MI`` case, where Yahoo's Milan feed is often
# stale/empty), we borrow the intraday series from a sibling listing of the
# SAME instrument on another EUR venue (Xetra/Euronext). Only EUR venues are
# used so the "vs previous close" % stays a faithful EUR proxy — London (.L,
# often USD/GBP) is intentionally excluded to avoid FX-contaminated returns.
# This affects ONLY the intraday sparkline / broker-1D path; EOD/daily history
# (valuation, returns, risk) always stays on the instrument's primary listing.
_SIBLING_SUFFIXES: dict[str, tuple[str, ...]] = {
    "MI": ("DE", "PA", "AS", "F"),   # Milan  → Xetra, Paris, Amsterdam, Frankfurt
    "PA": ("DE", "MI", "AS", "F"),   # Paris  → Xetra, Milan, ...
    "AS": ("DE", "MI", "PA", "F"),   # Amsterdam
    "F":  ("DE", "MI", "PA", "AS"),  # Frankfurt floor → Xetra, ...
    "DE": ("MI", "PA", "AS", "F"),   # Xetra   → Milan, Paris, ...
}

# A sibling listing is accepted only when its latest intraday price is within
# this fraction of the primary listing's last known close. Same-instrument EUR
# listings track within ~1-2%; a wider gap signals a same-root ticker on
# another exchange is a DIFFERENT instrument (collision guard).
_SIBLING_PRICE_TOLERANCE = 0.10


def _has_intraday(ser) -> bool:
    return ser is not None and len(ser) >= 2


def _official_and_prev(fetch_history, ticker: str, iday):
    """From a ticker's daily history, return ``(official_close_on_iday,
    previous_close_before_iday)`` as floats (or ``None`` each when missing).

    ``official_close_on_iday`` is the exchange's settled daily close for the
    session dated ``iday`` (it incorporates the closing auction). Best-effort:
    returns ``(None, None)`` on any error."""
    try:
        hist = fetch_history(ticker)
        if hist is None or not len(hist) or "Close" not in getattr(hist, "columns", []):
            return None, None
        dclose = hist["Close"].dropna()
        same = dclose[[ts.date() == iday for ts in dclose.index]]
        prior = dclose[[ts.date() < iday for ts in dclose.index]]
        oc = float(same.iloc[-1]) if len(same) else None
        pv = float(prior.iloc[-1]) if len(prior) else None
        return oc, pv
    except Exception:  # noqa: BLE001
        return None, None


def _sibling_symbols(ticker: str) -> list[str]:
    """Candidate sibling listings (same root, alternate EUR venue) for a
    ticker, in priority order. Empty for indices (^...), FX/futures (=X/=F),
    crypto (-USD), suffixless US tickers, and non-EUR venues."""
    t = (ticker or "").upper()
    if not t or "." not in t or t.startswith("^") or "=" in t or "-" in t:
        return []
    root, suf = t.rsplit(".", 1)
    return [f"{root}.{s}" for s in _SIBLING_SUFFIXES.get(suf, ())]


def _resolve_intraday(symbols: list[str]) -> dict:
    """Resolve an intraday close series per symbol, with EUR sibling fallback.

    Returns ``{original_symbol: (series, source_symbol)}`` for every symbol
    that has a usable intraday series (>=2 points) — either its own or a
    sibling's. ``source_symbol`` is the listing the series actually came from,
    so the caller can pull a *coherent* previous close from the same feed.
    Best-effort: never raises."""
    prim = _fetch_intraday(symbols)
    out: dict = {s: (prim[s], s) for s in symbols if _has_intraday(prim.get(s))}
    missing = [s for s in symbols if s not in out]
    if not missing:
        return out

    cand_map = {s: _sibling_symbols(s) for s in missing}
    all_cands = list(dict.fromkeys(c for cs in cand_map.values() for c in cs))
    if not all_cands:
        return out

    sib = _fetch_intraday(all_cands)
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception:  # noqa: BLE001
        _fetch_history = None  # type: ignore

    for s in missing:
        # Primary listing's last known close, used as the collision guard.
        prim_close = None
        if _fetch_history is not None:
            try:
                h = _fetch_history(s)
                if h is not None and len(h) and "Close" in getattr(h, "columns", []):
                    cl = h["Close"].dropna()
                    if len(cl):
                        prim_close = float(cl.iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        for c in cand_map[s]:
            ser = sib.get(c)
            if not _has_intraday(ser):
                continue
            if prim_close:
                dev = abs(float(ser.iloc[-1]) / prim_close - 1.0)
                if dev > _SIBLING_PRICE_TOLERANCE:
                    logger.debug("intraday fallback %s→%s rejected (%.1f%% off "
                                 "primary close)", s, c, dev * 100)
                    continue
            out[s] = (ser, c)
            logger.info("intraday fallback: %s → %s (EUR sibling listing)", s, c)
            break
    return out


def _fetch_intraday_with_fallback(symbols: list[str]) -> dict:
    """``{symbol: Close series}`` with EUR sibling fallback for intraday-empty
    listings. The sibling series is keyed under the ORIGINAL symbol so callers
    (e.g. the newsletter sparkline map) need no changes."""
    return {k: ser for k, (ser, _src) in _resolve_intraday(symbols).items()
            if ser is not None}


def _fetch_intraday(symbols: list[str]) -> dict:
    """One batched intraday download → ``{symbol: Close series}``. Empty on
    any failure (the caller falls back to the daily history)."""
    out: dict = {}
    try:
        import warnings
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(symbols, period="1d", interval="15m",
                              group_by="ticker", progress=False, threads=True)
        if raw is None or len(raw) == 0:
            return {}
        level0 = set(raw.columns.get_level_values(0)) if hasattr(raw.columns, "get_level_values") else set()
        for s in symbols:
            try:
                if s in level0 and "Close" in raw[s].columns:
                    cl = raw[s]["Close"].dropna()
                    if len(cl) >= 2:
                        out[s] = cl
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.debug("intraday batch failed: %s", e)
    return out


def _quote(dclose, intra, spark_points: int = 40) -> Optional[dict]:
    """Assemble one quote from the daily close series and (optional)
    intraday close series. Returns None when there is not enough data."""
    spark_series = None
    if intra is not None and len(intra) >= 2:
        cur = float(intra.iloc[-1])
        iday = intra.index[-1].date()
        prev = None
        if dclose is not None and len(dclose):
            prior = dclose[[ts.date() < iday for ts in dclose.index]]
            if len(prior):
                prev = float(prior.iloc[-1])
        if prev is None:
            prev = float(intra.iloc[0])
        spark = [float(x) for x in intra.values]
        baseline = prev
        # Keep the timestamped intraday series so the newsletter can draw it
        # on a full-session time axis (line grows through the day).
        spark_series = intra
    elif dclose is not None and len(dclose) >= 2:
        cur, prev = float(dclose.iloc[-1]), float(dclose.iloc[-2])
        spark = [float(x) for x in dclose.iloc[-spark_points:].values]
        baseline = spark[0]
    else:
        return None
    change = cur - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return {"value": cur, "change": change, "pct": pct,
            "spark": spark, "baseline": baseline, "spark_series": spark_series}


def broker_1d(tickers: list[str]) -> dict:
    """Broker-style 1D return per ticker: the latest intraday price vs the
    previous official close, in the instrument's listing currency.

    This is the "since previous close" figure a broker shows live during the
    session (and the last completed session's change once closed). Returns
    ``{ticker: {"pct": float, "live": bool}}`` only for tickers with a usable
    intraday series (>=2 points); callers fall back to the end-of-day close
    return for the rest. ``live`` is True only when the market is trading
    *now* — i.e. the latest intraday bar is recent. A same-day bar from a
    session that has already closed (e.g. viewed in the evening) is NOT live.
    Best-effort and currency-consistent: both the live price and the previous
    close come from the same native yfinance feed, so for a EUR-listed ETF
    the % is the EUR daily move. Never raises."""
    import pandas as pd
    uniq = [t for t in {t for t in tickers if t}]
    if not uniq:
        return {}
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception:  # noqa: BLE001
        return {}
    # Resolve intraday with EUR sibling fallback: a ``.MI`` holding with no
    # Milan intraday borrows the series from its Xetra/Euronext twin. ``src``
    # is the listing the series came from, so the previous close is pulled
    # from that SAME feed — keeping ``cur`` and ``prev`` currency-consistent.
    resolved = _resolve_intraday(uniq)
    out: dict = {}
    for tk, (intra, src) in resolved.items():
        if intra is None or len(intra) < 2:
            continue
        cur = float(intra.iloc[-1])
        last_ts = intra.index[-1]
        iday = last_ts.date()
        # "live" = the source listing's exchange is in its regular session
        # right now, judged by EXCHANGE HOURS (not bar recency). Uses ``src``
        # so a Milan holding served by its Xetra twin is judged by the venue
        # that actually produced the bars. FX/futures/crypto have no fixed
        # session → market_open_now returns None and we fall back to recency.
        mkt_open = market_open_now(src)
        if mkt_open is None:
            try:
                lt = (last_ts.tz_convert("UTC") if getattr(last_ts, "tzinfo", None)
                      else last_ts.tz_localize("UTC"))
                age_min = (pd.Timestamp.now(tz="UTC") - lt).total_seconds() / 60.0
                is_live = age_min <= _MARKET_OPEN_MAX_LAG_MIN
            except Exception:  # noqa: BLE001
                is_live = False
        else:
            is_live = bool(mkt_open)

        # --- Closed session: the authoritative 1D move is the instrument's
        # OWN primary-listing official daily close (which includes the closing
        # auction) vs its previous close — exactly what a broker shows for the
        # held position, and immune to lone wide intraday prints (e.g. a 17:19
        # tick vs the 17:30 auction). Prefer the primary listing ``tk``; only
        # if it has no official close for the day fall back to the source
        # (sibling) listing. This keeps the % on the venue the user holds,
        # while the sibling is still used for the *live* intraday sparkline. ---
        if not is_live:
            for cand in (tk, src):
                oc, pv = _official_and_prev(_fetch_history, cand, iday)
                if oc is not None and pv:
                    out[tk] = {"pct": (oc / pv - 1.0) * 100.0, "live": False}
                    break
            if tk in out:
                continue

        # --- Live session (or no official close available yet): the latest
        # intraday tick vs the previous close from the SAME feed, so ``cur``
        # and ``prev`` are currency-consistent. ---
        prev = None
        try:
            hist = _fetch_history(src)
            if hist is not None and len(hist) and "Close" in getattr(hist, "columns", []):
                dclose = hist["Close"].dropna()
                prior = dclose[[ts.date() < iday for ts in dclose.index]]
                if len(prior):
                    prev = float(prior.iloc[-1])
        except Exception:  # noqa: BLE001
            pass
        if prev is None:
            prev = float(intra.iloc[0])
        if prev:
            out[tk] = {"pct": (cur / prev - 1.0) * 100.0, "live": bool(is_live)}
    return out


def fetch_market_quotes(force: bool = False) -> list[dict]:
    """Fetch the curated market quotes (memoised per process). Best-effort:
    returns whatever could be fetched; never raises."""
    global _memo, _memo_at
    if _memo is not None and not force and (_time.monotonic() - _memo_at) < _MEMO_TTL_SECONDS:
        return _memo
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception as e:  # noqa: BLE001
        logger.debug("market quotes unavailable (%s)", e)
        return []

    intraday = _fetch_intraday([s for _, s, _ in MARKETS])
    out: list[dict] = []
    for name, symbol, category in MARKETS:
        try:
            hist = _fetch_history(symbol)
            dclose = (hist["Close"].dropna()
                      if hist is not None and len(hist) and "Close" in getattr(hist, "columns", [])
                      else None)
            q = _quote(dclose, intraday.get(symbol))
            if q is None:
                continue
            out.append({"name": name, "symbol": symbol, "category": category, **q})
        except Exception as e:  # noqa: BLE001
            logger.debug("market quote %s failed: %s", symbol, e)
            continue

    _memo = out
    _memo_at = _time.monotonic()
    return out
