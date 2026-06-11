"""Fetch market data from yfinance, enrich and classify holdings.

This module handles the Data Enrichment layer:
- Fetching price history and metadata from yfinance
- ISIN resolution via OpenFIGI API
- FX conversion to EUR
- Asset class and geography classification
- Multi-geography breakdown via geo_scraper

Caching policy: only the *immutable* past is cached on disk (see
``tarzan.data.price_cache``). Daily closes up to yesterday, FX history and
the deterministic ISIN→symbol resolution never change, so they are reused
across runs; only the recent tail (last few days, including today) is
re-fetched, so today's price is always fresh and never served stale. The
``info`` blob (which carries today's quote) is never cached.

Architecture note: enrichment is parallelized via ThreadPoolExecutor.
Each holding is enriched independently, with per-holding error isolation
so that a single API failure doesn't block the entire pipeline.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.data import price_cache
from tarzan import config as cfg

logger = logging.getLogger(__name__)

# Configurable backtest period. Set once via set_portfolio_backtest_period()
# before enrichment starts, then only read by worker threads. Guarded by a
# dedicated lock so the write is published safely to the reader threads.
_period_lock = threading.Lock()
BACKTEST_PERIOD = "5y"


def set_portfolio_backtest_period(period: str) -> None:
    """Set the yfinance history period for all subsequent fetches."""
    global BACKTEST_PERIOD
    with _period_lock:
        BACKTEST_PERIOD = period


def _backtest_period() -> str:
    """Thread-safe read of the configured backtest period."""
    with _period_lock:
        return BACKTEST_PERIOD


# ---------------------------------------------------------------------------
# Network layer — retry/backoff + per-run memoization + immutable disk cache
# ---------------------------------------------------------------------------
# Two layers cooperate here:
#
#   * intra-run memoization — within a single enrichment run the same
#     OpenFIGI/yfinance/benchmark request is issued at most once. The
#     stores are reset at the start of every run (reset_run_caches), so
#     this never serves stale data; it only stops the parallel pipeline
#     from hammering the same endpoint (the main source of HTTP 429 noise).
#
#   * cross-run immutable disk cache (tarzan.data.price_cache) — the
#     multi-year price/FX history and the deterministic ISIN→symbol
#     resolution are persisted to ~/.cache/tarzan and reused across runs.
#     Only the recent tail (last few days, incl. today) is re-fetched, so
#     today's close is always fresh. Today's live quote travels in the
#     ``info`` blob, which is NOT cached.
#
# All in-memory stores are guarded by a lock because enrichment runs under
# a ThreadPoolExecutor.

_MAX_FETCH_ATTEMPTS = 3          # total tries per request before giving up
_BACKOFF_BASE_SECONDS = 0.75     # exponential backoff base
_OPENFIGI_MIN_INTERVAL = 0.3     # spacing between OpenFIGI calls (~25/min cap)
_YF_MIN_INTERVAL = 0.2           # min spacing between yfinance calls (anti-429)

_net_lock = threading.Lock()
_openfigi_memo: dict[str, list] = {}
_ticker_info_memo: dict[str, dict] = {}
_history_memo: dict[str, pd.Series] = {}
_benchmark_memo: dict[str, pd.Series] = {}
_openfigi_last_call: list[float] = [0.0]  # mutable single-cell timestamp
_yf_last_call: list[float] = [0.0]        # mutable single-cell timestamp


def reset_run_caches() -> None:
    """Clear all intra-run memoization. Called once at the start of each
    enrichment run so every run starts from fresh network state."""
    with _net_lock:
        _openfigi_memo.clear()
        _ticker_info_memo.clear()
        _history_memo.clear()
        _benchmark_memo.clear()
        _geo_breakdown_memo.clear()
        _geo_source_memo.clear()
        _openfigi_last_call[0] = 0.0
        _yf_last_call[0] = 0.0


def _space_yf_call() -> None:
    """Enforce a minimum interval between yfinance calls across threads.

    yfinance scrapes Yahoo's unofficial endpoints, which rate-limit (HTTP
    429) on bursts. The ThreadPoolExecutor would otherwise fire all
    requests at once; this spreads them out by at least _YF_MIN_INTERVAL
    so we stay under the limit and keep market-data coverage high.
    """
    with _net_lock:
        wait = _YF_MIN_INTERVAL - (_time.monotonic() - _yf_last_call[0])
        if wait > 0:
            _time.sleep(wait)
        _yf_last_call[0] = _time.monotonic()


def _is_transient_error(exc: Exception) -> bool:
    """True when an exception looks like throttling/transient network
    trouble (worth retrying) rather than a definitive 'not found'."""
    msg = str(exc).lower()
    transient = ("429", "too many requests", "rate limit", "timed out",
                 "timeout", "connection", "temporarily", "503", "502")
    return any(t in msg for t in transient)


def _retry(fn, *, what: str):
    """Run ``fn`` with bounded exponential backoff on transient errors.

    Returns ``fn()`` on success, or None if every attempt failed. A
    definitive (non-transient) error is not retried.
    """
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — classified below
            if not _is_transient_error(e) or attempt == _MAX_FETCH_ATTEMPTS:
                logger.debug("%s failed (attempt %d): %s", what, attempt, e)
                return None
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.debug("%s throttled (attempt %d), backing off %.2fs", what, attempt, delay)
            _time.sleep(delay)
    return None


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------

def _get_fx_series(currency: str) -> pd.Series:
    """Get a daily FX rate series for currency→EUR conversion.

    For EUR, returns an empty series (sentinel for no conversion needed).
    Tries direct pair first, then inverse pair as fallback. Fetched fresh
    from yfinance on every call.
    """
    if currency == "EUR":
        return pd.Series(dtype=float)
    return _fetch_fx_pair(currency)


def _fetch_fx_pair(currency: str) -> pd.Series:
    """Fetch FX pair from yfinance, trying direct and inverse.

    FX history is immutable past data, so it is disk-cached per currency
    (``FX_<ccy>``) and only the recent tail is re-fetched on later runs.
    A throttled fetch falls back to the cached series rather than the
    rate=1.0 sentinel, so coverage does not silently degrade.
    """
    cache_key = f"FX_{currency}"
    cached = price_cache.load_history(cache_key)
    start = price_cache.refresh_start(cached)

    for pair, invert in [(f"{currency}EUR=X", False), (f"EUR{currency}=X", True)]:
        def _call(p=pair, s=start):
            _space_yf_call()
            ticker = yf.Ticker(p)
            if s is not None:
                return ticker.history(start=s, interval="1d")
            return ticker.history(period=_backtest_period(), interval="1d")

        hist = _retry(_call, what=f"FX {pair}")
        if hist is not None and not hist.empty:
            fresh = 1.0 / hist["Close"] if invert else hist["Close"]
            merged = price_cache.merge_history(cached, fresh)
            result = merged if merged is not None and not merged.empty else fresh
            price_cache.store_history(cache_key, result)
            return result

    if cached is not None and not cached.empty:
        logger.debug("FX %s fetch failed; using cached history", currency)
        return cached
    logger.warning("No FX data for %s, assuming rate=1.0 (flagged)", currency)
    return pd.Series([1.0], index=[pd.Timestamp.now()])


# ---------------------------------------------------------------------------
# Minor-unit currency normalization (pence, cents, agorot, ...)
# ---------------------------------------------------------------------------
# Yahoo Finance quotes some instruments in the "minor unit" of a currency —
# 1/100 of the major. For example LSE-listed ETFs are often quoted in GBp
# (Great British pence), not GBP. A price of 28450 "GBp" means 284.50 GBP.
# If we skip this step, current_value explodes by 100x.
#
# Convention: the minor-unit code is the 2-letter currency code followed by
# a lowercase letter (e.g. GBp, ZAc, ILa). There is no FX pair for these
# codes on yfinance, so we must rescale to the major unit first.
_MINOR_TO_MAJOR_CURRENCY: dict[str, str] = {
    "GBp": "GBP",   # British pence
    "GBX": "GBP",   # British pence (alternate code, uppercase)
    "ZAc": "ZAR",   # South African cents
    "ZAC": "ZAR",
    "ILa": "ILS",   # Israeli agorot
    "ILA": "ILS",
    "ZWL": "ZWL",   # edge case, keep as-is
}


def _normalize_minor_currency(
    prices: pd.Series, currency: str
) -> tuple[pd.Series, str]:
    """Rescale prices quoted in a minor unit to the major currency.

    Returns (prices, currency) where currency is always the major ISO code.
    If the input currency is not a known minor unit, returns the inputs
    unchanged.
    """
    if currency in _MINOR_TO_MAJOR_CURRENCY and currency != _MINOR_TO_MAJOR_CURRENCY[currency]:
        major = _MINOR_TO_MAJOR_CURRENCY[currency]
        logger.debug("Rescaling %s prices to %s (divide by 100)", currency, major)
        return prices / 100.0, major
    return prices, currency


def convert_to_eur(prices: pd.Series, currency: str) -> pd.Series:
    """Convert a price series to EUR using the FX rate.

    For EUR input, returns prices unchanged. Aligns by date with forward-fill.
    """
    prices, currency = _normalize_minor_currency(prices, currency)
    if currency == "EUR":
        return prices
    fx = _get_fx_series(currency)
    if fx.empty:
        return prices
    combined = pd.DataFrame({"price": prices, "fx": fx})
    combined["fx"] = combined["fx"].ffill().bfill()
    return (combined["price"] * combined["fx"]).dropna()


# ---------------------------------------------------------------------------
# Ticker data fetching
# ---------------------------------------------------------------------------
ISIN_EXCHANGE_SUFFIXES = cfg.isin_exchange_suffixes()


def _fetch_ticker_data(ticker: str) -> dict:
    """Fetch fresh yfinance data for a ticker.

    Falls back to base ticker (without exchange suffix) if initial fetch fails.

    Returns:
        Dict with keys: info, history.
    """
    logger.info("Fetching data for %s", ticker)
    # yfinance raises ValueError("Invalid ISIN number: ...") from the
    # Ticker constructor when the symbol looks like an ISIN but fails
    # its check-digit validation (common for BTP / US Treasury / EIB
    # bonds). The retry/memoized helpers catch this and return empties,
    # so current_price stays None and the caller proceeds to the bond
    # fallback in _enrich_single.
    info = _fetch_ticker_info(ticker)
    history = _fetch_history(ticker)

    # Fallback: strip exchange suffix
    if history.empty and not info.get("regularMarketPrice") and "." in ticker:
        base_ticker = ticker.split(".")[0]
        logger.debug("Trying base ticker %s (stripped from %s)", base_ticker, ticker)
        info2 = _fetch_ticker_info(base_ticker)
        if info2.get("regularMarketPrice") or info2.get("previousClose"):
            history2 = _fetch_history(base_ticker)
            if not history2.empty:
                info, history = info2, history2
                logger.info("Fallback to base ticker %s succeeded", base_ticker)

    return {"info": info, "history": history}


# ---------------------------------------------------------------------------
# ISIN resolution — deterministic ranking
# ---------------------------------------------------------------------------
# An ISIN can map to many yfinance symbols: the same instrument listed on
# several exchanges (.MI / .F / .L …) plus, occasionally, a *different*
# instrument that happens to share the bare ticker (e.g. ISIN IE0006WW1TQ4
# is Xtrackers "MSCI World ex USA", but the bare symbol "EXUS" on Yahoo is
# an unrelated USD Nomura fund). Picking "the first symbol that responds"
# is non-deterministic and can silently select the wrong instrument.
#
# Instead we collect every responding candidate and rank them by objective,
# instrument-independent criteria so the choice is stable and identical
# whichever path (holdings CSV or order list) asks for the ISIN:
#
#   1. name coherence with the OpenFIGI canonical name (rejects collisions)
#   2. currency match to the expected/native currency (minor-unit aware)
#   3. exchange-suffix priority (config-ordered, region-agnostic)
#   4. alphabetical tiebreak (final determinism guarantee)
#   4. alphabetical tiebreak (final determinism guarantee)
#
# No ISINs are hardcoded and no extra input file is needed — the criteria
# are derived from OpenFIGI metadata + config, so this scales globally.

from dataclasses import dataclass, field


@dataclass
class _Candidate:
    """A resolved ISIN candidate with the metadata needed for ranking.

    Ranking uses only the lightweight ``info`` fields, so price history is
    *not* fetched here — it is downloaded once for the winning symbol in
    :func:`_resolve_isin`. ``data`` is filled in only for the winner.
    """

    symbol: str
    info: dict
    price: float
    currency: str
    name: str
    has_history: bool = True
    data: dict = field(default_factory=dict)


# Words that carry no instrument identity — stripped before name comparison.
_NAME_STOPWORDS = frozenset({
    "etf", "ucits", "etc", "fund", "index", "acc", "dist", "inc",
    "the", "of", "and", "class", "shares", "share", "1c", "1d",
    "eur", "usd", "gbp", "chf", "hedged", "accumulating", "distributing",
})


def _name_tokens(name: str) -> set[str]:
    """Normalise an instrument name into a set of identity tokens."""
    if not name:
        return set()
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in name)
    return {tok for tok in cleaned.split() if tok and tok not in _NAME_STOPWORDS}


def _name_match_score(candidate_name: str, canonical_name: str) -> float:
    """Token-overlap score (0..1) between a candidate and the OpenFIGI
    canonical name. Returns a neutral 0.5 when the canonical name is
    unknown so name has no effect on ranking."""
    canon = _name_tokens(canonical_name)
    if not canon:
        return 0.5
    cand = _name_tokens(candidate_name)
    if not cand:
        return 0.0
    return len(canon & cand) / len(canon)


def _suffix_priority(symbol: str) -> int:
    """Deterministic exchange-suffix priority (lower index = preferred).

    Driven by the config-ordered ``isin_exchange_suffixes`` list so the
    preference is region-agnostic and tunable without code changes.
    Symbols with no/unknown suffix rank last.
    """
    for i, suffix in enumerate(ISIN_EXCHANGE_SUFFIXES):
        if symbol.endswith(suffix):
            return i
    return len(ISIN_EXCHANGE_SUFFIXES)


def _rank_key(cand: _Candidate, canonical_name: str, expected_currency: str) -> tuple:
    """Deterministic sort key for a candidate (higher tuple = better).

    Pure function of the candidate metadata + resolution context, so
    ranking the same candidate set always yields the same winner. Only
    lightweight ``info`` fields are used — price history is fetched later,
    for the winner only — so ranking does not depend on a full download.
    """
    name_score = _name_match_score(cand.name, canonical_name)
    # Bucket the name score so tiny float differences don't reorder
    # otherwise-equivalent listings of the same instrument.
    name_bucket = round(name_score * 4)  # 0..4
    currency_match = 1 if _currency_matches(cand.currency, expected_currency) else 0
    # Negate suffix priority so a lower index sorts higher.
    suffix_rank = -_suffix_priority(cand.symbol)
    return (name_bucket, currency_match, suffix_rank)


def _currency_matches(candidate_ccy: str, expected_ccy: str) -> bool:
    """Currency equality that tolerates minor-unit quoting.

    yfinance may quote in a minor unit (GBp, ZAc, ILa) while the declared
    holding currency is the major code (GBP, ZAR, ILS). Normalise both to
    the major unit before comparing so the expected-currency signal is not
    defeated for exactly the instruments it should disambiguate.
    """
    if not expected_ccy or not candidate_ccy:
        return False
    cand = _MINOR_TO_MAJOR_CURRENCY.get(candidate_ccy, candidate_ccy)
    exp = _MINOR_TO_MAJOR_CURRENCY.get(expected_ccy, expected_ccy)
    return cand == exp


def _resolve_isin(
    isin: str, hint_ticker: str = "", expected_currency: str = ""
) -> Optional[tuple[dict, str]]:
    """Resolve an ISIN to the best yfinance symbol, deterministically.

    Collects candidate symbols (CSV/order hint + OpenFIGI mappings +
    brute-force exchange suffixes + raw ISIN), ranks them by
    :func:`_rank_key` using only lightweight ``info`` metadata, and then
    downloads price history **once, for the winner only**. The result is
    stable across runs and identical for both the holdings and order-list
    paths.

    Returns ``(data_dict, winning_symbol)`` or ``None`` if nothing priced.
    """
    clean_isin = isin.replace("-", "")

    # Fast path: a previously-resolved symbol is cached on disk. The
    # resolution is deterministic and stable, so we can skip the whole
    # OpenFIGI + candidate sweep and just price the cached symbol. If it
    # no longer prices (delisted/renamed), fall through to a full
    # re-resolve so the cache self-heals.
    cached_symbol = price_cache.load_resolution(clean_isin)
    if cached_symbol:
        info = _fetch_ticker_info(cached_symbol)
        history = _fetch_history(cached_symbol)
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if (history is not None and not history.empty) or price:
            logger.info("Resolved ISIN %s → %s (from cache)", isin, cached_symbol)
            return {"info": info, "history": history}, cached_symbol
        logger.info(
            "Cached symbol %s for ISIN %s no longer prices; re-resolving",
            cached_symbol, isin,
        )

    canonical_name = _openfigi_name(clean_isin)

    candidates = _collect_candidate_metas(clean_isin, hint_ticker)
    if not candidates:
        return None

    # Best = max by rank key, with alphabetical symbol as the final
    # deterministic tiebreak (smaller symbol wins when keys are equal).
    best = max(
        candidates,
        key=lambda c: (_rank_key(c, canonical_name, expected_currency), _neg_str(c.symbol)),
    )

    # Fetch history only for the winner (ranking never needed it).
    history = _fetch_history(best.symbol)
    best.data = {"info": best.info, "history": history}
    price_cache.store_resolution(clean_isin, best.symbol)
    logger.info(
        "Resolved ISIN %s → %s (price=%.2f %s, name='%s')",
        isin, best.symbol, best.price, best.currency, best.name[:40],
    )
    return best.data, best.symbol


def _neg_str(s: str) -> tuple:
    """Helper to make alphabetical-ascending act as a max() tiebreak:
    returns a key that is *larger* for lexicographically smaller strings."""
    return tuple(-ord(c) for c in s)


def _collect_candidate_metas(clean_isin: str, hint_ticker: str) -> list[_Candidate]:
    """Build the deterministic candidate-symbol list and fetch metadata.

    Symbols are probed in a deterministic order and de-duplicated:

      1. the CSV/order ticker hint (if any);
      2. OpenFIGI-mapped exchange tickers;
      3. every OpenFIGI *bare* ticker combined with each configured
         exchange suffix — this is what lets the order-list path (which
         only knows the ISIN) discover the same local listing, e.g.
         ``EXUS`` + ``.MI`` → ``EXUS.MI``, that the holdings path finds
         via its ticker hint;
      4. the bare ISIN combined with each exchange suffix;
      5. the raw ISIN as-is.

    To bound network cost and rate-limit exposure the number of *fetched*
    candidates is capped; because the order is deterministic the cap does
    not affect reproducibility.
    """
    symbols: list[str] = []

    def _add(sym: str) -> None:
        if sym and sym not in symbols:
            symbols.append(sym)

    if hint_ticker and hint_ticker != clean_isin:
        _add(hint_ticker)

    figi_syms = _openfigi_lookup(clean_isin)
    for sym in figi_syms:
        _add(sym)

    # Bare OpenFIGI tickers (no exchange suffix) get the full suffix sweep,
    # so an ISIN-only caller reaches the local listing the same way.
    for sym in figi_syms:
        if "." not in sym:
            for suffix in ISIN_EXCHANGE_SUFFIXES:
                _add(f"{sym}{suffix}")

    for suffix in ISIN_EXCHANGE_SUFFIXES:
        _add(f"{clean_isin}{suffix}")
    _add(clean_isin)

    metas: list[_Candidate] = []
    for sym in symbols:
        if len(metas) >= _MAX_RESOLVE_FETCHES:
            break
        meta = _fetch_candidate_meta(sym)
        if meta is not None:
            metas.append(meta)
    return metas


# Upper bound on how many candidate symbols we fetch per ISIN. Probed in a
# deterministic order so the cap never changes the resolved result.
_MAX_RESOLVE_FETCHES = 10


def _fetch_candidate_meta(symbol: str) -> Optional[_Candidate]:
    """Fetch a single candidate's lightweight ``info`` for ranking.

    Does NOT download price history — that happens once for the winning
    symbol in :func:`_resolve_isin`. Returns None if the symbol has no
    usable price (so it cannot be a real listing).
    """
    info = _fetch_ticker_info(symbol)
    price = info.get("regularMarketPrice") or info.get("previousClose")
    if not price or price <= 0:
        return None
    name = info.get("longName") or info.get("shortName") or info.get("name") or ""
    return _Candidate(
        symbol=symbol,
        info=info,
        price=float(price),
        currency=info.get("currency", "") or "",
        name=name,
    )


def _fetch_ticker_info(symbol: str) -> dict:
    """yfinance ``info`` for a symbol, retried on throttle and memoized
    for the duration of the run."""
    with _net_lock:
        if symbol in _ticker_info_memo:
            return _ticker_info_memo[symbol]

    def _call():
        _space_yf_call()
        return yf.Ticker(symbol).info or {}

    info = _retry(_call, what=f"info {symbol}") or {}
    with _net_lock:
        _ticker_info_memo[symbol] = info
    return info


def _fetch_history(symbol: str) -> pd.DataFrame:
    """yfinance price history for a symbol, retried on throttle, memoized
    for the run, and backed by the immutable disk cache.

    The heavy multi-year download happens once: on subsequent runs the
    cached history is loaded and only the recent tail (from
    ``price_cache.refresh_start``) is re-fetched and merged, so historical
    closes are reused while the last few sessions stay fresh. A throttled
    tail fetch degrades gracefully to the cached history rather than
    dropping the instrument.
    """
    with _net_lock:
        if symbol in _history_memo:
            return _history_memo[symbol]

    cached = price_cache.load_history(symbol)
    start = price_cache.refresh_start(cached)

    def _call():
        _space_yf_call()
        ticker = yf.Ticker(symbol)
        if start is not None:
            return ticker.history(start=start, interval="1d")
        return ticker.history(period=_backtest_period(), interval="1d")

    fresh = _retry(_call, what=f"history {symbol}")
    if fresh is None:
        fresh = pd.DataFrame()

    merged = price_cache.merge_history(cached, fresh)
    result = merged if merged is not None and not merged.empty else fresh
    if result is not None and not result.empty:
        price_cache.store_history(symbol, result)

    with _net_lock:
        _history_memo[symbol] = result
    return result


# ---------------------------------------------------------------------------
# OpenFIGI API
# ---------------------------------------------------------------------------
_FIGI_EXCHANGE_MAP = cfg.figi_exchange_map()
_FIGI_MIC_MAP = cfg.figi_mic_map()


def _openfigi_raw(isin: str) -> list:
    """Raw OpenFIGI API call, rate-limited, retried on throttle, and
    memoized per run.

    The unauthenticated OpenFIGI endpoint caps at ~25 requests/min, so a
    minimum inter-call spacing is enforced and the result is memoized so
    the three logical lookups per ISIN (name, ticker mapping, classify)
    collapse to a single network call.
    """
    with _net_lock:
        if isin in _openfigi_memo:
            return _openfigi_memo[isin]

    def _call() -> list:
        # Enforce minimum spacing between OpenFIGI calls across threads.
        with _net_lock:
            wait = _OPENFIGI_MIN_INTERVAL - (_time.monotonic() - _openfigi_last_call[0])
            if wait > 0:
                _time.sleep(wait)
            _openfigi_last_call[0] = _time.monotonic()
        url = "https://api.openfigi.com/v3/mapping"
        payload = [{"idType": "ID_ISIN", "idValue": isin}]
        req = Request(url, data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    results = _retry(_call, what=f"OpenFIGI {isin}") or []
    with _net_lock:
        _openfigi_memo[isin] = results
    return results


def _openfigi_name(isin: str) -> str:
    """Return the OpenFIGI canonical instrument name for an ISIN ("" if none).

    This is the authoritative name used to reject ticker collisions: a
    candidate whose yfinance name does not overlap with it is a different
    instrument that merely shares a symbol.
    """
    for result_group in _openfigi_raw(isin):
        for item in result_group.get("data", []):
            name = item.get("name")
            if name:
                return name
    return ""


def _openfigi_bond_signals(isin: str) -> tuple[Optional[str], Optional[str]]:
    """Return (marketSector, securityType2) from OpenFIGI for an ISIN.

    These are OpenFIGI's authoritative instrument-level classification
    fields, used to detect bonds reliably (so the per-100-nominal value
    convention is applied). Reads the per-run memoized OpenFIGI result,
    so this adds no extra network call. Returns (None, None) if unknown.
    """
    clean = isin.replace("-", "")
    for result_group in _openfigi_raw(clean):
        for item in result_group.get("data", []):
            sector = item.get("marketSector")
            sec_type = item.get("securityType2") or item.get("securityType")
            if sector or sec_type:
                return sector, sec_type
    return None, None


def _openfigi_lookup(isin: str) -> list[str]:
    """Query OpenFIGI API to resolve an ISIN to exchange tickers.

    Returns a list of candidate yfinance-style tickers (max 8).
    """
    results = _openfigi_raw(isin)
    tickers: list[str] = []
    if not results:
        return tickers

    for result_group in results:
        for item in result_group.get("data", []):
            ticker = item.get("ticker", "")
            if not ticker:
                continue
            exchange = item.get("exchCode", "")
            mic = item.get("micCode", "")
            yf_ticker = _map_figi_to_yfinance(ticker, exchange, mic)
            if yf_ticker and yf_ticker not in tickers:
                tickers.append(yf_ticker)
            if ticker not in tickers:
                tickers.append(ticker)

    return tickers[:8]


def _map_figi_to_yfinance(ticker: str, exchange: str, mic: str) -> Optional[str]:
    """Map an OpenFIGI ticker + exchange to a yfinance-compatible ticker."""
    suffix = _FIGI_MIC_MAP.get(mic)
    if suffix is not None:
        return f"{ticker}{suffix}" if suffix else ticker
    suffix = _FIGI_EXCHANGE_MAP.get(exchange)
    if suffix is not None:
        return f"{ticker}{suffix}" if suffix else ticker
    return None


def openfigi_classify(isin: str) -> dict:
    """Use OpenFIGI metadata to classify an instrument by ISIN.

    Returns a dict with optional keys: asset_class, instrument_type, name, security_type.
    """
    results = _openfigi_raw(isin)
    if not results:
        return {}

    kw = cfg.classification()
    figi_fi = kw.get("figi_fixed_income", [])
    figi_eq = kw.get("figi_equity", [])
    figi_etf = kw.get("figi_etf", [])

    for result_group in results:
        for item in result_group.get("data", []):
            sec_type = (item.get("securityType2") or "").lower()
            market_sector = (item.get("marketSector") or "").lower()
            name = item.get("name") or ""

            info: dict = {}
            if name:
                info["name"] = name

            # Classify by market sector and security type
            info.update(_classify_figi_item(sec_type, market_sector, name, kw, figi_fi, figi_eq, figi_etf))

            if info.get("asset_class") or info.get("name"):
                logger.debug("OpenFIGI classify %s → %s", isin, info)
                return info

    return {}


def _classify_figi_item(
    sec_type: str, market_sector: str, name: str,
    kw: dict, figi_fi: list, figi_eq: list, figi_etf: list,
) -> dict:
    """Classify a single OpenFIGI item into asset_class and instrument_type."""
    info: dict = {}

    if market_sector == "govt" or sec_type in figi_fi:
        info["asset_class"] = AssetClass.FIXED_INCOME
        if "govt" in sec_type or "sovereign" in sec_type or market_sector == "govt":
            info["instrument_type"] = "Government Bond"
        elif "corp" in sec_type:
            info["instrument_type"] = "Corporate Bond"
        else:
            info["instrument_type"] = "Note"
    elif market_sector == "corp":
        info["asset_class"] = AssetClass.FIXED_INCOME
        info["instrument_type"] = "Corporate Bond"
    elif sec_type in figi_eq:
        info["asset_class"] = AssetClass.EQUITIES
        info["instrument_type"] = "Stock"
    elif sec_type in figi_etf:
        info["asset_class"] = AssetClass.EQUITIES
        info["instrument_type"] = "ETF Equity"
    elif "bond" in sec_type or "fixed" in sec_type:
        info["asset_class"] = AssetClass.FIXED_INCOME
        info["instrument_type"] = "Bond"
    elif "money market" in sec_type:
        info["asset_class"] = AssetClass.CASH_EQUIVALENTS
        info["instrument_type"] = "Money Market"

    # Name-based fallback
    if not info.get("asset_class"):
        name_lower = name.lower()
        if any(k in name_lower for k in kw.get("name_fixed_income", [])):
            info["asset_class"] = AssetClass.FIXED_INCOME
            info["instrument_type"] = info.get("instrument_type", "Bond")
        elif any(k in name_lower for k in kw.get("name_gold", [])):
            info["asset_class"] = AssetClass.GOLD
            info["instrument_type"] = "Gold ETC"
        elif any(k in name_lower for k in kw.get("name_commodities", [])):
            info["asset_class"] = AssetClass.COMMODITIES
            info["instrument_type"] = "ETC"
        elif any(k in name_lower for k in kw.get("name_crypto", [])):
            info["asset_class"] = AssetClass.CRYPTO
            info["instrument_type"] = "Crypto"

    return info


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _infer_instrument_type(info: dict, holding: Holding) -> str:
    """Infer a human-readable instrument type from yfinance info and holding hints."""
    qt = info.get("quoteType", "").upper()
    cat = (info.get("category") or "").lower()
    hint = (holding.asset_type or "").lower()
    name = (info.get("longName") or info.get("shortName") or holding.name or "").lower()
    kw = cfg.classification()

    if qt == "ETF":
        return _classify_etf_subtype(cat, name, kw)
    if qt == "EQUITY":
        if any(k in name for k in ("etf", "ucits")):
            return _classify_etf_subtype(cat, name, kw)
        return "Stock"
    if qt == "ETN":
        return "ETN"
    if qt == "MUTUALFUND":
        return "Fund Bond" if ("bond" in cat or "fixed" in cat) else "Fund"
    if any(k in name for k in kw.get("name_fixed_income", [])):
        return "Bond"
    if any(k in hint for k in kw.get("asset_type_fixed_income", [])):
        return "Bond"
    if "etn" in hint:
        return "ETN"
    if any(k in hint for k in kw.get("asset_type_gold", [])):
        return "Gold ETC"
    if any(k in hint for k in kw.get("asset_type_commodities", [])):
        return "Precious Metals ETF"
    return "Other"


def _classify_etf_subtype(cat: str, name: str, kw: dict) -> str:
    """Classify an ETF into a specific subtype based on category and name."""
    if any(k in cat for k in kw.get("instrument_bond_etf", [])) or any(k in name for k in kw.get("instrument_bond_etf", [])):
        return "ETF Bond"
    if any(k in cat for k in kw.get("instrument_money_market", [])) or any(k in name for k in kw.get("instrument_money_market", [])):
        return "ETF Money Market"
    if any(k in name for k in kw.get("instrument_gold_etc", [])):
        return "Gold ETC"
    if any(k in cat for k in kw.get("instrument_commodities_etf", [])):
        return "ETF Commodities"
    if any(k in name for k in kw.get("instrument_precious_etc", [])):
        return "ETC"
    return "ETF Equity"


def _derive_security_type(holding: Holding) -> str:
    """Derive a standardized security type from asset_class and instrument_type."""
    it = (holding.instrument_type or "").lower()
    ac = holding.asset_class

    type_map = {
        AssetClass.EQUITIES: "Equity ETF" if "etf" in it else "Share",
        AssetClass.FIXED_INCOME: (
            "Bond ETF" if "etf" in it
            else "Government Bond" if "government" in it or "govt" in it
            else "Corporate Bond" if "corporate" in it
            else "Bond"
        ),
        AssetClass.CASH_EQUIVALENTS: "Money Market ETF" if "etf" in it else "Money Market Instrument",
        AssetClass.GOLD: "Gold ETC",
        AssetClass.COMMODITIES: "ETC" if "etc" in it else "Commodity",
        AssetClass.CRYPTO: "Crypto",
    }
    return type_map.get(ac, "Alternative") if ac else "Alternative"


def classify_asset_class(info: dict, ticker: str, holding: Holding) -> AssetClass:
    """Determine asset class from yfinance info and ticker patterns.

    Classification priority:
    1. Explicit hint from input (asset_type column)
    2. yfinance quoteType
    3. Category/sector keywords
    4. Long name keywords
    5. Default to Alternative
    """
    kw = cfg.classification()

    # 1. Explicit hint
    if holding.asset_type:
        result = _classify_from_hint(holding.asset_type, kw)
        if result:
            return result

    qt = info.get("quoteType", "").upper()
    cat = (info.get("category") or "").lower()
    sector = (info.get("sector") or "").lower()
    long_name = (info.get("longName") or "").lower()

    # 2. quoteType-based
    if qt == "EQUITY":
        # Crypto/gold/commodity ETPs are often listed as "EQUITY" on yfinance,
        # so check their long name first to avoid bucketing them as equities.
        if any(k in long_name for k in kw.get("name_crypto", [])):
            return AssetClass.CRYPTO
        if any(k in long_name for k in kw.get("name_gold", [])):
            return AssetClass.GOLD
        if any(k in long_name for k in kw.get("name_commodities", [])):
            return AssetClass.COMMODITIES
        if any(k in long_name for k in kw.get("name_cash", [])):
            return AssetClass.CASH_EQUIVALENTS
        if any(k in long_name for k in kw.get("name_fixed_income", [])):
            return AssetClass.FIXED_INCOME
        return AssetClass.EQUITIES

    # 3. Category/sector keywords
    result = _classify_from_category(cat, sector)
    if result:
        return result

    # 4. Long name keywords
    if any(k in long_name for k in kw.get("name_gold", [])):
        return AssetClass.GOLD
    if any(k in long_name for k in kw.get("name_commodities", [])):
        return AssetClass.COMMODITIES
    if any(k in long_name for k in kw.get("name_crypto", [])):
        return AssetClass.CRYPTO
    if any(k in long_name for k in kw.get("name_fixed_income", [])):
        return AssetClass.FIXED_INCOME

    # 5. ETF with equity-like category
    if qt == "ETF":
        if any(k in cat for k in kw.get("instrument_equity_category", [])):
            return AssetClass.EQUITIES
        if any(k in long_name for k in kw.get("name_equity_etf", [])):
            return AssetClass.EQUITIES
        return AssetClass.EQUITIES

    if qt == "ETN":
        return AssetClass.ALTERNATIVE
    if qt == "CRYPTOCURRENCY":
        return AssetClass.CRYPTO

    return AssetClass.ALTERNATIVE


def _classify_from_hint(hint: str, kw: dict) -> Optional[AssetClass]:
    """Classify from the asset_type hint column."""
    hint_lower = hint.lower()
    mapping = [
        ("asset_type_equities", AssetClass.EQUITIES),
        ("asset_type_fixed_income", AssetClass.FIXED_INCOME),
        ("asset_type_cash", AssetClass.CASH_EQUIVALENTS),
        ("asset_type_gold", AssetClass.GOLD),
        ("asset_type_commodities", AssetClass.COMMODITIES),
        ("asset_type_crypto", AssetClass.CRYPTO),
        ("asset_type_alternative", AssetClass.ALTERNATIVE),
    ]
    for key, ac in mapping:
        if any(k in hint_lower for k in kw.get(key, [])):
            return ac
    return None


def _classify_from_category(cat: str, sector: str) -> Optional[AssetClass]:
    """Classify from yfinance category and sector strings."""
    if any(k in cat for k in ("bond", "fixed income", "treasury", "government")):
        return AssetClass.FIXED_INCOME
    if "money market" in cat:
        return AssetClass.CASH_EQUIVALENTS
    if any(k in cat for k in ("crypto", "bitcoin", "ethereum")) or any(k in sector for k in ("crypto",)):
        return AssetClass.CRYPTO
    if any(k in cat for k in ("gold",)) or any(k in sector for k in ("gold",)):
        return AssetClass.GOLD
    if any(k in cat for k in ("precious metal", "silver", "commodit")):
        return AssetClass.COMMODITIES
    if any(k in sector for k in ("precious metal",)):
        return AssetClass.COMMODITIES
    return None


# ---------------------------------------------------------------------------
# Geography classification
# ---------------------------------------------------------------------------
GEOGRAPHY_MAP: dict[str, Geography] = cfg.geography_map()
EXCHANGE_COUNTRY: dict[str, str] = cfg.exchange_country()

# Per-run geo-breakdown memoization. Written from worker threads during
# enrich_holdings, so all access is guarded by _net_lock; cleared by
# reset_run_caches() at the start of each run so it never persists stale
# breakdowns across runs (consistent with the no-cache guarantee).
_geo_breakdown_memo: dict[str, dict[Geography, float]] = {}
_geo_source_memo: dict[str, str] = {}


def _get_geo_breakdown_cached(ticker: str) -> Optional[dict[Geography, float]]:
    """Thread-safe read of the run-scoped geo-breakdown memo."""
    with _net_lock:
        return _geo_breakdown_memo.get(ticker)


def _store_geo_breakdown(ticker: str, breakdown: dict[Geography, float], source: str) -> None:
    """Thread-safe write to the run-scoped geo-breakdown memo."""
    with _net_lock:
        _geo_breakdown_memo[ticker] = breakdown
        _geo_source_memo[ticker] = source


def classify_geography(info: dict, ticker: str, holding: Holding) -> Geography:
    """Determine geography from yfinance info using MSCI country mapping.

    For ETFs with geo_breakdown, returns the dominant geography.
    For stocks, uses the company's country from yfinance info.
    Falls back to exchange country from ticker suffix.
    """
    # 1. Geo breakdown (scraped)
    breakdown = _get_geo_breakdown_cached(ticker)
    if breakdown:
        return max(breakdown, key=lambda g: breakdown[g])

    # 2. Direct country for non-ETF
    qt = info.get("quoteType", "").upper()
    country = info.get("country") or holding.country
    if country and country in GEOGRAPHY_MAP:
        if qt == "EQUITY" or qt not in ("ETF", "MUTUALFUND"):
            return GEOGRAPHY_MAP[country]

    # 3. Exchange-based fallback for stocks
    if qt == "EQUITY" or qt not in ("ETF", "MUTUALFUND"):
        for suffix, c in EXCHANGE_COUNTRY.items():
            if ticker.endswith(suffix):
                return GEOGRAPHY_MAP.get(c, Geography.USA)

    return Geography.OTHER


def get_geo_breakdown(
    ticker: str, isin: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Get geographic breakdown for an ETF ticker via dynamic scraping.

    Returns (breakdown_dict, source_name) or None. Memoized per run
    (thread-safe) so each ticker is scraped at most once, and backed by
    the immutable disk cache (keyed by the stable ISIN when available) so
    a justETF/scrape outage does not degrade the geography section to
    "Not Available" — once resolved, the breakdown persists across runs.
    """
    cached = _get_geo_breakdown_cached(ticker)
    if cached is not None:
        with _net_lock:
            source = _geo_source_memo.get(ticker, "memory")
        return cached, source

    geo_key = isin or ticker
    disk = price_cache.load_geo(geo_key)
    if disk:
        breakdown = _geo_from_cache(disk["breakdown"])
        if breakdown:
            source = disk.get("source", "cache")
            _store_geo_breakdown(ticker, breakdown, source)
            return breakdown, source

    result = _scrape_geo_breakdown(ticker, isin)
    if result:
        breakdown, source = result
        _store_geo_breakdown(ticker, breakdown, source)
        price_cache.store_geo(
            geo_key, {g.value: v for g, v in breakdown.items()}, source
        )
        return breakdown, source
    return None


def _geo_from_cache(raw: dict) -> dict[Geography, float]:
    """Reconstruct a ``{Geography: pct}`` dict from the cached
    ``{geo_name: pct}`` form, dropping any unknown geography names."""
    out: dict[Geography, float] = {}
    for name, pct in (raw or {}).items():
        try:
            out[Geography(name)] = float(pct)
        except (ValueError, TypeError):
            continue
    return out


def _scrape_geo_breakdown(
    ticker: str, isin: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Get geographic breakdown via the geo resolution chain."""
    long_name = _fetch_ticker_info(ticker).get("longName", "") or ""
    from tarzan.data.geo_resolver import resolve_geo
    return resolve_geo(isin, ticker, long_name)


# ---------------------------------------------------------------------------
# Single holding enrichment
# ---------------------------------------------------------------------------

def _enrich_single(holding: Holding) -> tuple[Holding, dict]:
    """Enrich a single holding with market data from yfinance.

    Fetches price history, metadata, converts to EUR, extracts TER/yield/sector.
    Per-holding errors are caught and logged — the holding is returned partially enriched.

    Returns the enriched holding and the raw yfinance ``info`` dict used,
    so the caller can classify without re-fetching.
    """
    ticker = holding.ticker
    if not ticker:
        logger.warning("No ticker for ISIN=%s, skipping enrichment", holding.isin)
        return holding, {}

    data_source = "yfinance"
    info: dict = {}
    try:
        clean_isin = (holding.isin or "").replace("-", "")
        is_valid_isin = len(clean_isin) == 12 and clean_isin[:2].isalpha()
        data, history = None, pd.DataFrame()

        # ISIN-first resolution: whenever we have a real ISIN we resolve
        # deterministically from it, using the CSV/order ``ticker`` only as
        # a hint and the declared currency as a ranking signal. This makes
        # the holdings path and the order-list path pick the *same* symbol
        # for the same instrument, instead of "first responder wins".
        if is_valid_isin:
            resolved = _resolve_isin(
                clean_isin,
                hint_ticker=ticker,
                expected_currency=(holding.currency or ""),
            )
            if resolved:
                data, resolved_ticker = resolved
                data_source = f"yfinance:{resolved_ticker}"
                info = data.get("info", {})
                history = data.get("history", pd.DataFrame())

        if data is None:
            data = _fetch_ticker_data(ticker)
            info = data.get("info", {})
            history = data.get("history", pd.DataFrame())

        holding.name = info.get("longName") or info.get("shortName") or info.get("name") or ticker
        holding.instrument_type = _infer_instrument_type(info, holding)
        holding.fetch_timestamp = dt.now()

        # Price history and current price
        currency = info.get("currency", holding.currency or "EUR")
        _set_price_data(holding, history, info, currency)

        # Value and gain. For bonds resolved via yfinance the
        # ``current_price`` is a clean quote per 100 of face value
        # (the same convention Borsa Italiana uses for our bond
        # fallback). The holding's ``quantity`` is the bond's
        # nominal amount, not a unit count, so the EUR value is
        # ``quantity × price / 100``. Bond detection and the /100
        # convention are centralized in bond_fetcher so the current
        # and historical valuation paths agree exactly.
        if holding.current_price is not None:
            from tarzan.data.bond_fetcher import is_bond, value_position
            # OpenFIGI's authoritative classification (memoized, no extra
            # network call) reliably catches bonds yfinance mislabels.
            figi_sector, figi_sec_type = (None, None)
            if is_valid_isin:
                figi_sector, figi_sec_type = _openfigi_bond_signals(clean_isin)
            bond = is_bond(
                quote_type=info.get("quoteType"),
                sec_type=info.get("typeDisp"),
                market_sector=figi_sector,
                figi_sec_type=figi_sec_type,
            )
            holding.current_value = value_position(
                holding.quantity, holding.current_price, bond=bond
            )
            if bond:
                # Rescale the price history and unit price the same way
                # so downstream code that multiplies price_history by
                # ``quantity`` (e.g. the rebalancer's drift simulation,
                # the order-derived historical series) reads correct EUR.
                if holding.price_history is not None and len(holding.price_history) > 0:
                    holding.price_history = holding.price_history / 100.0
                holding.current_price = holding.current_price / 100.0
            holding.data_source = data_source
        else:
            # Fallback: try Terrapin Finance API for bonds
            _try_terrapin_fallback(holding)

        if holding.cost_basis_eur > 0 and holding.current_value > 0:
            holding.gain_pct = (
                (holding.current_value - holding.cost_basis_eur)
                / holding.cost_basis_eur * 100
            )

        # Metadata
        holding.ter = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
        holding.yield_pct = (
            info.get("yield") or info.get("dividendYield")
            or info.get("trailingAnnualDividendYield") or info.get("fiveYearAvgDividendYield")
        )
        if holding.sector is None:
            holding.sector = info.get("sector") or info.get("category")
        if holding.country is None:
            holding.country = info.get("country")

    except Exception as e:
        logger.error("Failed to enrich %s: %s", ticker, e)

    return holding, info


def _set_price_data(
    holding: Holding, history: pd.DataFrame, info: dict, currency: str
) -> None:
    """Set price_history and current_price on a holding, with FX conversion."""
    if not history.empty:
        prices = history["Close"].copy()
        # Normalize minor-unit quotes (e.g. GBp → GBP) before FX conversion
        prices, currency = _normalize_minor_currency(prices, currency)
        if currency != "EUR":
            prices = convert_to_eur(prices, currency)
        holding.price_history = prices
        holding.current_price = float(prices.iloc[-1]) if len(prices) > 0 else None
    else:
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if price:
            # Normalize minor-unit quotes first
            scalar = pd.Series([float(price)])
            scalar, currency = _normalize_minor_currency(scalar, currency)
            price = float(scalar.iloc[0])
            if currency != "EUR":
                fx = _get_fx_series(currency)
                if not fx.empty:
                    price = price * float(fx.iloc[-1])
            holding.current_price = float(price)


def _try_terrapin_fallback(holding: Holding) -> None:
    """Try Borsa Italiana scraping as fallback for bonds without yfinance data.

    Borsa Italiana quotes the clean price per 100 of nominal in the bond's
    *native* currency. We convert that to an EUR-per-unit ``current_price``
    (FX-converted, then /100), matching the yfinance bond branch, so every
    downstream consumer reads EUR directly. This is currency-general: a
    USD Treasury, a ZAR EIB note, a GBP gilt are all handled the same way
    via the shared FX machinery — no per-currency special-casing.
    """
    try:
        from tarzan.data.bond_fetcher import fetch_bond_price, value_position

        isin = holding.isin
        if not isin or len(isin.replace("-", "")) != 12:
            holding.current_value = holding.market_value_eur
            holding.data_source = "input_csv (no market data)"
            return

        result = fetch_bond_price(isin)
        if result:
            # Borsa quote: clean price per 100 nominal, in the native currency.
            price_native = result["price"]
            currency = holding.currency or "EUR"

            # Convert the clean price to EUR per 100 nominal. The FX series
            # is EUR-per-native-unit, so a native price is multiplied by it.
            price_eur_per_100 = price_native
            if currency != "EUR":
                fx = _get_fx_series(currency)
                if not fx.empty:
                    price_eur_per_100 = price_native * float(fx.iloc[-1])
                else:
                    logger.warning(
                        "Bond %s: no FX for %s, Borsa price left unconverted (flagged)",
                        isin, currency,
                    )

            value = value_position(holding.quantity, price_eur_per_100, bond=True)

            # Sanity net: if the EUR value is still wildly off the known CSV
            # value (e.g. a non-standard nominal/quantity convention), fall
            # back to the EUR anchor from the CSV rather than a number we
            # cannot reconcile.
            csv_value = holding.market_value_eur
            if csv_value > 0 and abs(value - csv_value) / csv_value > 0.5:
                logger.info(
                    "Bond %s: Borsa value %.2f far from CSV %.2f; using CSV anchor",
                    isin, value, csv_value,
                )
                value = csv_value

            holding.current_price = price_eur_per_100 / 100.0  # EUR per unit
            holding.current_value = value
            holding.data_source = result["source"]
            logger.info(
                "Bond fallback for %s: clean=%.4f %s → %.4f EUR/100, value=%.2f EUR",
                isin, price_native, currency, price_eur_per_100, value,
            )
            return

        # No data from Borsa Italiana either
        holding.current_value = holding.market_value_eur
        holding.data_source = "input_csv (no market data)"

    except Exception as e:
        logger.debug("Bond fallback failed for %s: %s", holding.isin, e)
        holding.current_value = holding.market_value_eur
        holding.data_source = "input_csv (no market data)"


# ---------------------------------------------------------------------------
# Enrichment + classification pipeline
# ---------------------------------------------------------------------------

def _enrich_and_classify(holding: Holding) -> Holding:
    """Enrich a holding with market data and then classify it.

    Classification order:
    1. yfinance metadata → asset class
    2. Enriched name → reclassification signal
    3. OpenFIGI fallback for unclassified instruments
    4. Geography via geo_breakdown or single-country
    """
    holding, info = _enrich_single(holding)

    if holding.asset_class is None:
        holding.asset_class = classify_asset_class(info, holding.ticker, holding)

    # Post-classification: use enriched name as additional signal
    _reclassify_by_name(holding)

    # OpenFIGI fallback
    if holding.asset_class == AssetClass.ALTERNATIVE or holding.instrument_type in (None, "Other"):
        _apply_openfigi_fallback(holding)

    if holding.security_type is None:
        holding.security_type = _derive_security_type(holding)

    if holding.geography is None:
        holding.geography = classify_geography(info, holding.ticker, holding)

    # Multi-geography breakdown
    _apply_geo_breakdown(holding)

    return holding


def _reclassify_by_name(holding: Holding) -> None:
    """Reclassify a holding based on its enriched name if currently ambiguous."""
    if not holding.name or holding.asset_class not in (AssetClass.ALTERNATIVE, AssetClass.EQUITIES):
        return
    name_lower = holding.name.lower()
    kw = cfg.classification()
    if any(k in name_lower for k in kw.get("name_fixed_income", [])):
        holding.asset_class = AssetClass.FIXED_INCOME
    elif any(k in name_lower for k in kw.get("name_cash", [])):
        holding.asset_class = AssetClass.CASH_EQUIVALENTS
    elif any(k in name_lower for k in kw.get("name_gold", [])):
        holding.asset_class = AssetClass.GOLD
    elif any(k in name_lower for k in kw.get("name_commodities", [])):
        holding.asset_class = AssetClass.COMMODITIES
    elif any(k in name_lower for k in kw.get("name_crypto", [])):
        holding.asset_class = AssetClass.CRYPTO


def _apply_openfigi_fallback(holding: Holding) -> None:
    """Apply OpenFIGI classification as a last resort."""
    try:
        figi_info = openfigi_classify(holding.isin)
        if figi_info.get("asset_class") and holding.asset_class == AssetClass.ALTERNATIVE:
            holding.asset_class = figi_info["asset_class"]
            ac_value = holding.asset_class.value if holding.asset_class else "Unknown"
            logger.info("OpenFIGI classified %s → %s", holding.ticker, ac_value)
        if figi_info.get("instrument_type") and holding.instrument_type in (None, "Other"):
            holding.instrument_type = figi_info["instrument_type"]
        if figi_info.get("name") and holding.name in (None, holding.ticker):
            holding.name = figi_info["name"]
    except Exception as e:
        logger.debug("OpenFIGI classify failed for %s: %s", holding.isin, e)


def _apply_geo_breakdown(holding: Holding) -> None:
    """Apply geographic breakdown from index lookup or dynamic scraping."""
    result = get_geo_breakdown(holding.ticker, holding.isin)
    if result:
        breakdown, source = result
        holding.geo_breakdown = breakdown
        holding.geo_source = source
        if breakdown:
            holding.geography = max(breakdown, key=lambda g: breakdown[g])
    else:
        holding.geo_breakdown = None
        holding.geo_source = "not_available"


# ---------------------------------------------------------------------------
# Main enrichment entry point
# ---------------------------------------------------------------------------
MAX_WORKERS = cfg.max_workers()


def enrich_holdings(holdings: list[Holding]) -> list[Holding]:
    """Enrich all holdings in parallel using ThreadPoolExecutor.

    Each holding gets: current_price, price_history, asset_class, geography,
    TER, yield, sector, and computed value/gain.
    Per-holding errors are isolated — the holding is returned partially enriched.

    Args:
        holdings: List of raw Holding objects from the loader.

    Returns:
        List of enriched Holding objects with market data and classifications.
    """
    if not holdings:
        return holdings

    # Reset intra-run memoization so this run starts from fresh network
    # state (no stale data across runs), then de-duplicates concurrent
    # requests within the run.
    reset_run_caches()

    logger.info("Enriching %d holdings (max %d workers)...", len(holdings), MAX_WORKERS)

    enriched: list[Holding] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_enrich_and_classify, h): h for h in holdings}
        for future in as_completed(futures):
            try:
                enriched.append(future.result())
            except Exception as e:
                h = futures[future]
                logger.error("Enrichment failed for %s: %s", h.ticker, e)
                enriched.append(h)

    # Compute weights
    total_value = sum(h.current_value for h in enriched if h.current_value)
    if total_value > 0:
        for h in enriched:
            if h.current_value is not None:
                h.weight_pct = (h.current_value / total_value) * 100

    logger.info("Enrichment complete. Total portfolio value: %.2f EUR", total_value)
    return enriched
