"""justETF as a SECOND, independent source of ETF market prices.

Everything else in Tarzan prices off Yahoo, so its checks could only ever prove that
Tarzan's arithmetic agrees with Yahoo — never that Yahoo was right. On 8 Oct 2026 Yahoo
itself was wrong twice for one thin fund (a stray trade, and a "previous close" no one
dealt at), and nothing in the run could tell. justETF serves FactSet/Xetra data through
the JSON its own pages read: a daily series from inception, and today's quote.

Used only as a REFEREE. It never supplies a figure the issue prints; it decides whether
a figure the issue prints is corroborated. That keeps its failure modes harmless: when it
is down, slow, rate-limited, or has no data for an ISIN (it returns its HTML page), the
answer is "not cross-checked" and the issue still goes out.

The API is undocumented, so it is used gently: one series call and one quote call per
holding per run, memoized, in a small thread pool, with a short timeout and no retry
storm. Measured: 17 series calls in 2.9 s sequentially, ~245 KB each.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BASE = "https://www.justetf.com/api/etfs/{isin}"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
                   "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"),
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT_S = 10
_WORKERS = 6

_series_memo: dict[str, Optional[pd.Series]] = {}
_quote_memo: dict[str, Optional[dict]] = {}


def reset_memo() -> None:
    _series_memo.clear()
    _quote_memo.clear()


def _get_json(url: str) -> Optional[dict]:
    """The decoded body, or None for anything that is not a JSON answer.

    An ISIN justETF does not cover comes back as its HTML page with HTTP 200, so the
    content type is checked rather than the status.
    """
    import requests

    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — a referee must never break a run
        logger.info("justETF unreachable (%s): %s", type(e).__name__, url.split("?")[0])
        return None
    if not r.ok or "json" not in (r.headers.get("content-type") or ""):
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _fetch_series(isin: str) -> Optional[pd.Series]:
    """Daily EUR closes, dividends reinvested — the same total-return basis as the
    engine's ``auto_adjust=True`` tape — indexed by session date."""
    body = _get_json(
        _BASE.format(isin=isin) + "/performance-chart?locale=en&currency=EUR"
        "&valuesType=MARKET_VALUE&reduceData=false&includeDividends=true")
    points = (body or {}).get("series") or []
    data = {}
    for p in points:
        try:
            data[pd.Timestamp(p["date"])] = float(p["value"]["raw"])
        except (KeyError, TypeError, ValueError):
            continue
    if not data:
        return None
    s = pd.Series(data).sort_index()
    return s[s > 0]


def _fetch_quote(isin: str) -> Optional[dict]:
    """``{price, prev_close, date, prev_date, venue}`` for today's quote, or None."""
    body = _get_json(_BASE.format(isin=isin) + f"/quote?locale=en&currency=EUR&isin={isin}")
    if not body:
        return None
    try:
        return {
            "price": float(body["latestQuote"]["raw"]),
            "prev_close": float(body["previousQuote"]["raw"]),
            "date": pd.Timestamp(body["latestQuoteDate"]).date(),
            "prev_date": pd.Timestamp(body["previousQuoteDate"]).date(),
            "venue": body.get("quoteTradingVenue"),
        }
    except (KeyError, TypeError, ValueError):
        return None


def prefetch(isins) -> None:
    """Fetch every missing series and quote in one small pool. Never raises."""
    from tarzan import runtime

    wanted = [i for i in dict.fromkeys(isins) if i]
    if not wanted or not runtime.allows_live_transport():
        return
    missing_s = [i for i in wanted if i not in _series_memo]
    missing_q = [i for i in wanted if i not in _quote_memo]
    if not (missing_s or missing_q):
        return
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        series = dict(zip(missing_s, pool.map(_fetch_series, missing_s)))
        quotes = dict(zip(missing_q, pool.map(_fetch_quote, missing_q)))
    _series_memo.update(series)
    _quote_memo.update(quotes)


def series(isin: str) -> Optional[pd.Series]:
    if isin not in _series_memo:
        prefetch([isin])
    return _series_memo.get(isin)


def quote(isin: str) -> Optional[dict]:
    if isin not in _quote_memo:
        prefetch([isin])
    return _quote_memo.get(isin)
