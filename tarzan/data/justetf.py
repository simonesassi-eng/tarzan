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
from contextlib import suppress
from contextvars import copy_context
from typing import Optional

import pandas as pd
from tarzan.data.referee import finite_number, observation_date
from tarzan.runtime.memo import memoized, provider_memo, provider_store

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
    with provider_memo().lock:
        provider_store("justetf.series", _series_memo).clear()
        provider_store("justetf.quote", _quote_memo).clear()


def _get_json(url: str) -> Optional[dict]:
    """The decoded body, or None for anything that is not a JSON answer.

    An ISIN justETF does not cover comes back as its HTML page with HTTP 200, so the
    content type is checked rather than the status.
    """
    import requests
    from tarzan.runtime.transport import provider_call

    try:
        r = provider_call(
            "justetf", lambda timeout: requests.get(url, headers=_HEADERS, timeout=timeout),
            timeout=_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 — a referee must never break a run
        logger.info("justETF unreachable (%s): %s", type(e).__name__, url.split("?")[0])
        return None
    try:
        if not r.ok or "json" not in str(r.headers.get("content-type") or "").lower():
            return None
        body = r.json()
        return body if isinstance(body, dict) else None
    except ValueError:
        return None
    finally:
        # requests.get creates a short-lived session. Release a received response on
        # every exit, including an HTML fallback or malformed JSON answer.
        with suppress(Exception):
            r.close()


def _fetch_series(isin: str) -> Optional[pd.Series]:
    """Daily EUR closes, dividends reinvested — the same total-return basis as the
    engine's ``auto_adjust=True`` tape — indexed by session date."""
    body = _get_json(
        _BASE.format(isin=isin) + "/performance-chart?locale=en&currency=EUR"
        "&valuesType=MARKET_VALUE&reduceData=false&includeDividends=true")
    if not isinstance(body, dict):
        return None
    points = body.get("series")
    if not isinstance(points, (list, tuple)):
        return None
    from tarzan import runtime

    today = runtime.today()
    data = {}
    for p in points:
        try:
            day = observation_date(p["date"])
            value = finite_number(p["value"]["raw"], positive=True)
        except (KeyError, TypeError, ValueError):
            continue
        if day is not None and day <= today and value is not None:
            data[pd.Timestamp(day)] = value
    if not data:
        return None
    return pd.Series(data, dtype=float).sort_index()


def _fetch_quote(isin: str) -> Optional[dict]:
    """A finite, dated EUR market pair, or None; freshness is never invented."""
    body = _get_json(_BASE.format(isin=isin) + f"/quote?locale=en&currency=EUR&isin={isin}")
    if not isinstance(body, dict) or not _response_matches_quote_request(body, isin):
        return None
    try:
        price = finite_number(body["latestQuote"]["raw"], positive=True)
        previous = finite_number(body["previousQuote"]["raw"], positive=True)
        day = observation_date(body["latestQuoteDate"])
        previous_day = observation_date(body["previousQuoteDate"])
    except (KeyError, TypeError, ValueError):
        return None
    from tarzan import runtime

    if (
        price is None or previous is None or day is None or previous_day is None
        or previous_day >= day or day > runtime.today()
    ):
        return None
    return {
        "price": price, "prev_close": previous, "date": day, "prev_date": previous_day,
        "venue": body.get("quoteTradingVenue"), "currency": "EUR",
    }


def _response_matches_quote_request(body: dict, isin: str) -> bool:
    """Reject an explicit identity/response-currency contradiction.

    The observed quote schema supplies neither field; ISIN and EUR are the request's
    contract. Check exact optional scalar declarations if a response supplies them,
    never the fund's base currency or a venue's trading currency: those can legitimately
    differ from the requested EUR quote.
    """
    for field, expected in (("isin", str(isin).strip().upper()), ("currency", "EUR")):
        supplied = body.get(field)
        if supplied is None:
            continue
        if not isinstance(supplied, str) or supplied.strip().upper() not in ("", expected):
            return False
    return True


def prefetch(isins) -> None:
    """Fetch every missing series and quote in one small pool. Never raises."""
    from tarzan import runtime

    wanted = [i for i in dict.fromkeys(isins) if i]
    if not wanted or not runtime.allows_live_transport():
        return
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [
            pool.submit(copy_context().run, fetch, isin)
            for fetch in (series, quote) for isin in wanted
        ]
        for future in futures:
            future.result()


def series(isin: str) -> Optional[pd.Series]:
    from tarzan import runtime

    if not isin or not runtime.allows_live_transport():
        return None
    return memoized(
        "justetf.series", isin, lambda: _fetch_series(isin), legacy=_series_memo,
    )


def quote(isin: str) -> Optional[dict]:
    from tarzan import runtime

    if not isin or not runtime.allows_live_transport():
        return None
    return memoized(
        "justetf.quote", isin, lambda: _fetch_quote(isin), legacy=_quote_memo,
    )
