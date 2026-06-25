"""Live quotes for the newsletter "Markets" strip (yfinance-style).

A curated set of major indices / commodities / crypto / FX, fetched
through the enricher's cached, throttled history helper so it reuses the
on-disk price cache (no extra cold downloads on warm runs) and degrades
gracefully: any symbol that fails to fetch is simply skipped, and if the
whole fetch fails the caller falls back to the benchmark-derived snapshot.

Each quote: ``{name, symbol, category, value, change, pct, spark}`` where
``change``/``pct`` are versus the prior close and ``spark`` is the recent
close series for a mini chart.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# (display name, yfinance symbol, category) — ordered as shown in the strip.
# Mirrors the yfinance markets bar most relevant to a global EUR investor.
MARKETS: list[tuple[str, str, str]] = [
    ("S&P 500", "^GSPC", "US"),
    ("Nasdaq", "^IXIC", "US"),
    ("Dow 30", "^DJI", "US"),
    ("Russell 2000", "^RUT", "US"),
    ("VIX", "^VIX", "US"),
    ("Euro Stoxx 50", "^STOXX50E", "Europe"),
    ("DAX", "^GDAXI", "Europe"),
    ("CAC 40", "^FCHI", "Europe"),
    ("FTSE 100", "^FTSE", "Europe"),
    ("Nikkei 225", "^N225", "Asia"),
    ("Gold", "GC=F", "Commodities"),
    ("Crude Oil", "CL=F", "Commodities"),
    ("Bitcoin", "BTC-USD", "Crypto"),
    ("EUR/USD", "EURUSD=X", "FX"),
]

CATEGORY_ORDER = ["US", "Europe", "Asia", "Commodities", "Crypto", "FX"]

_memo: Optional[list[dict]] = None


def fetch_market_quotes(spark_points: int = 30, force: bool = False) -> list[dict]:
    """Fetch the curated market quotes (memoised per process).

    Best-effort: returns whatever could be fetched; an empty list when the
    fetch path is unavailable. Never raises.
    """
    global _memo
    if _memo is not None and not force:
        return _memo
    out: list[dict] = []
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception as e:  # noqa: BLE001
        logger.debug("market quotes unavailable (%s)", e)
        return []

    for name, symbol, category in MARKETS:
        try:
            hist = _fetch_history(symbol)
            if hist is None or len(hist) == 0 or "Close" not in getattr(hist, "columns", []):
                continue
            close = hist["Close"].dropna()
            if len(close) < 2:
                continue
            last, prev = float(close.iloc[-1]), float(close.iloc[-2])
            if prev == 0:
                continue
            out.append({
                "name": name,
                "symbol": symbol,
                "category": category,
                "value": last,
                "change": last - prev,
                "pct": (last - prev) / prev * 100.0,
                "spark": list(close.iloc[-spark_points:].astype(float).values),
            })
        except Exception as e:  # noqa: BLE001
            logger.debug("market quote %s failed: %s", symbol, e)
            continue

    _memo = out
    return out
