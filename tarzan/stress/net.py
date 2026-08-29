"""Offline seam: make the cache authoritative and prove no socket opened.

The bench needs LIVE run mode (it is the only mode that honours a pinned hour —
see clock.py) but must not make 54 runs' worth of network calls. So every known
fetch boundary is served from a cache snapshot, and a socket guard then PROVES
the run was offline instead of leaving it to hope: anything still reaching for a
socket raises, which is how the remaining boundaries were found in the first
place.

``allow_network()`` opens the guard for the one block that must hit the real tape
(external verification), and counts the calls so the budget can be enforced.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


class NetworkAttempted(RuntimeError):
    """A socket was opened while the bench declared itself offline."""


@dataclass
class NetPolicy:
    allowed: bool = False
    attempts: list[str] = field(default_factory=list)


POLICY = NetPolicy()
_real_socket_connect = socket.socket.connect
_real_create_connection = socket.create_connection


def _guarded_connect(self, address, *a, **kw):
    POLICY.attempts.append(str(address))
    if not POLICY.allowed:
        raise NetworkAttempted(f"socket to {address} while offline")
    return _real_socket_connect(self, address, *a, **kw)


def _guarded_create_connection(address, *a, **kw):
    POLICY.attempts.append(str(address))
    if not POLICY.allowed:
        raise NetworkAttempted(f"connection to {address} while offline")
    return _real_create_connection(address, *a, **kw)


def install_socket_guard() -> None:
    """Guard the Python socket layer AND libcurl.

    The socket patch alone is blind: yfinance 1.1.0 drives curl_cffi 0.13.0, which
    goes through libcurl and never touches socket.socket.connect. The bench ran 12
    cells reporting "0 network attempts" while the log carried 590 real HTTP
    responses — the guard was measuring a layer nothing used. Both are patched
    now, and the curl one is what actually holds.
    """
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
    _guard_curl()


def _guard_curl() -> None:
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:                                # pragma: no cover
        return
    session = curl_requests.Session
    if getattr(session, "_stress_guarded", False):
        return
    original = session.request

    def guarded(self, method, url, *a, **kw):
        POLICY.attempts.append(f"curl {method} {str(url)[:80]}")
        if not POLICY.allowed:
            raise NetworkAttempted(f"curl {method} {url} while offline")
        return original(self, method, url, *a, **kw)

    session.request = guarded
    session._stress_guarded = True

    for name in ("get", "post", "head", "put", "delete"):
        fn = getattr(curl_requests, name, None)
        if fn is None:
            continue

        def module_level(*a, _n=name, **kw):
            POLICY.attempts.append(f"curl {_n} {str(a[0])[:80] if a else '?'}")
            if not POLICY.allowed:
                raise NetworkAttempted(f"curl {_n} while offline")
            return getattr(curl_requests, "_stress_orig_" + _n)(*a, **kw)

        setattr(curl_requests, "_stress_orig_" + name, fn)
        setattr(curl_requests, name, module_level)


def allow_network(flag: bool) -> None:
    POLICY.allowed = flag


def attempts() -> list[str]:
    return list(POLICY.attempts)


def reset_attempts() -> None:
    POLICY.attempts.clear()


class _CachedTicker:
    """A yfinance.Ticker stand-in served from the cache snapshot.

    Stubbing the individual fetch helpers was not enough: the first offline
    matrix reported "0 network attempts" for 12 cells while making 590 real HTTP
    calls, because the ISIN->ticker resolver probes yfinance's quoteSummary per
    suffixed venue candidate and that path was never named. Replacing Ticker
    itself closes every yfinance boundary at once, which is the only way to be
    sure rather than to keep discovering them one host at a time.
    """

    #: symbol (or bare ticker) -> ISO code the venue quotes in. Empty by default,
    #: which is what the first four matrix runs served -- and it meant the product
    #: never received a listing currency at all, so every native-currency check was
    #: really testing the harness' silence. Declaring it here is what lets the mark
    #: and the minor-unit rescale be asserted offline.
    CURRENCIES: dict = {}

    def __init__(self, symbol, *a, **kw):
        self.ticker = str(symbol)

    def history(self, *a, **kw):
        from tarzan.data import price_cache
        frame = price_cache.load_history(self.ticker)
        return frame if frame is not None else pd.DataFrame()

    def _currency(self) -> str:
        base = self.ticker.split(".")[0]
        return self.CURRENCIES.get(self.ticker) or self.CURRENCIES.get(base) or ""

    @property
    def fast_info(self):
        return {}

    @property
    def info(self):
        ccy = self._currency()
        return {"currency": ccy} if ccy else {}

    def get_info(self, *a, **kw):
        return self.info

    def __getattr__(self, name):
        return {}


def serve_from_cache(quotes: Optional[dict] = None,
                     market_strip: Optional[list] = None,
                     currencies: Optional[dict] = None) -> None:
    """Patch every known fetch boundary to a cache-backed or recorded answer.

    ``quotes`` is the recorded official-quote map; absent, quotes are empty,
    which is itself a valid state (a venue with no published quote) and is what
    the degradation block uses.

    ``currencies`` maps symbol -> ISO code and becomes ``Ticker.info["currency"]``,
    which is the ONLY input from which the product learns what a listing is quoted
    in. Left empty, the native-currency columns and their marks cannot be checked
    at all: every instrument arrives currency-less and falls back to EUR.
    """
    from tarzan.data import enricher, geo_resolver, price_cache
    from tarzan.data import market_quotes as mq
    from tarzan.engine import benchmarks

    # History: the cache IS the tape. _fetch_history normally merges a fresh
    # yfinance pull onto the cached frame; serving the cache alone is the same
    # frame minus the network.
    enricher._fetch_history = lambda symbol: (
        price_cache.load_history(symbol) if price_cache.load_history(symbol) is not None
        else pd.DataFrame()
    )
    benchmarks._fetch_benchmark_history = _cached_benchmark_series

    # Quotes / intraday / the markets strip.
    mq._fetch_official_quotes = lambda symbols: dict(quotes or {})
    mq._fetch_intraday = lambda *a, **k: {}
    mq.fetch_market_quotes = lambda *a, **k: list(market_strip or [])

    # Every yfinance boundary at once (history, quoteSummary probing, fast_info).
    import yfinance as yf
    _CachedTicker.CURRENCIES = dict(currencies or {})
    yf.Ticker = _CachedTicker
    enricher.yf = yf

    # The bond scraper reaches www.borsaitaliana.it; the socket guard found it.
    from tarzan.data import bond_fetcher
    bond_fetcher.fetch_bond_price = lambda isin: None

    # Geography and TER enrichment: no fixture needed, their absence is a
    # documented fallback rather than an error.
    geo_resolver._geo_from_top_holdings = lambda ticker: None
    geo_resolver.justetf_index_name = lambda isin: None
    geo_resolver.justetf_ter = lambda isin: None

    # Instrument-profile resolution reaches OpenFIGI/Yahoo; the cache snapshot
    # already carries profiles for the taxonomy, and an unresolved profile is a
    # state the bench deliberately exercises (P10).
    # OpenFIGI identity resolution: the socket guard found 12 attempts here on
    # the first offline run. The cache snapshot already carries the profiles the
    # taxonomy needs; an unresolved one is a state the bench exercises (P10).
    enricher._openfigi_raw_many = lambda isins: {i: [] for i in isins}
    enricher._openfigi_raw = lambda isin: []
    if hasattr(enricher, "_fetch_instrument_profile"):
        enricher._fetch_instrument_profile = lambda *a, **k: None


def _cached_benchmark_series(ticker: str):
    from tarzan.data import price_cache
    from tarzan.engine import benchmarks

    frame = price_cache.load_history(ticker)
    if frame is None or frame.empty or "Close" not in frame:
        return pd.Series(dtype=float)
    series = frame["Close"].dropna().copy()
    series.name = ticker
    # A COPY for the native attr: attaching the series to its own attrs makes it
    # self-referential, and the first deep copy of it blows the recursion limit
    # ("_preprocess_benchmarks failed: maximum recursion depth exceeded").
    native = series.copy()
    native.attrs = {}
    series.attrs.update({"resolved_ticker": ticker, "requested_ticker": ticker,
                         benchmarks._NATIVE_ATTR: native,
                         benchmarks._CURRENCY_ATTR: "EUR"})
    return series


def skip_whatif() -> None:
    """The what-if workbook is the single most expensive stage (31 candidate
    portfolios, 22 proxy series) and no check reads it."""
    from tarzan import backtest

    backtest.newsletter_portfolios = lambda **kw: []
