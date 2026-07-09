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
from typing import Optional

logger = logging.getLogger(__name__)

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
    # Crypto
    ("Bitcoin", "BTC-USD", "Crypto"),
    # Commodities
    ("Crude Oil", "CL=F", "Commodities"),
    ("Gold", "GC=F", "Commodities"),
    ("Silver", "SI=F", "Commodities"),
    ("Copper", "HG=F", "Commodities"),
    ("Natural Gas", "NG=F", "Commodities"),
    ("Brent Crude", "BZ=F", "Commodities"),
    ("Platinum", "PL=F", "Commodities"),
    # Currencies
    ("EUR/USD", "EURUSD=X", "Currencies"),
    ("USD/JPY", "JPY=X", "Currencies"),
    ("USD/GBP", "GBP=X", "Currencies"),
]

CATEGORY_ORDER = ["US", "Europe", "Asia", "Crypto",
                  "Commodities", "Currencies"]

_memo: Optional[list[dict]] = None


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
    elif dclose is not None and len(dclose) >= 2:
        cur, prev = float(dclose.iloc[-1]), float(dclose.iloc[-2])
        spark = [float(x) for x in dclose.iloc[-spark_points:].values]
        baseline = spark[0]
    else:
        return None
    change = cur - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return {"value": cur, "change": change, "pct": pct,
            "spark": spark, "baseline": baseline}


def broker_1d(tickers: list[str]) -> dict:
    """Broker-style 1D return per ticker: the latest intraday price vs the
    previous official close, in the instrument's listing currency.

    This is the "since previous close" figure a broker shows live during the
    session (and the last completed session's change once closed). Returns
    ``{ticker: {"pct": float, "live": bool}}`` only for tickers with a usable
    intraday series (>=2 points); callers fall back to the end-of-day close
    return for the rest. ``live`` is True when the latest intraday bar belongs
    to *today's* session (its market is open / has traded today) and False
    when the intraday endpoint returned a past completed session (market
    closed). Best-effort and currency-consistent: both the live price and the
    previous close come from the same native yfinance feed, so for a
    EUR-listed ETF the % is the EUR daily move. Never raises."""
    from datetime import datetime
    uniq = [t for t in {t for t in tickers if t}]
    if not uniq:
        return {}
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception:  # noqa: BLE001
        return {}
    intraday = _fetch_intraday(uniq)
    out: dict = {}
    for tk in uniq:
        intra = intraday.get(tk)
        if intra is None or len(intra) < 2:
            continue
        cur = float(intra.iloc[-1])
        last_ts = intra.index[-1]
        iday = last_ts.date()
        # "live" = the latest intraday bar is from today's session in the
        # instrument's own timezone (market open / has traded today). When the
        # market is closed the intraday endpoint returns the last completed
        # session, whose date is in the past → not live.
        try:
            today_local = datetime.now(getattr(last_ts, "tzinfo", None)).date()
        except Exception:  # noqa: BLE001
            today_local = datetime.now().date()
        is_live = iday == today_local
        prev = None
        try:
            hist = _fetch_history(tk)
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
    global _memo
    if _memo is not None and not force:
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
    return out
