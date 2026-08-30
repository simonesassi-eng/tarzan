"""Fetch market data from yfinance, enrich and classify holdings.

This module handles the Data Enrichment layer:
- Fetching price history and metadata from yfinance
- ISIN resolution via OpenFIGI API
- FX conversion to EUR
- Asset class and geography classification
- Multi-geography breakdown via geo_scraper

Caching policy: historical rows are cached on disk (see
``tarzan.data.price_cache``), while the recent tail is fetched on every run.
When a live tail is unavailable, cached closes remain usable only with their
original observation time and explicit fallback provenance; a recent fetch
attempt never makes an old close fresh.

Architecture note: enrichment is parallelized via ThreadPoolExecutor.
Each holding is enriched independently, with per-holding error isolation
so that a single API failure doesn't block the entire pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from datetime import datetime as dt, timezone
from functools import partial
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.models.instrument_key import normalize_isin, normalize_ticker
from tarzan.data import price_cache
from tarzan.data import _yf_net
from tarzan import config as cfg
from tarzan.runtime import data_quality as dq

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
#   * cross-run disk cache (tarzan.data.price_cache) — multi-year price/FX
#     history and deterministic ISIN→symbol resolution are persisted to
#     ~/.cache/tarzan. The recent tail is re-fetched, while any retained cache
#     rows keep explicit origin and observation time for policy evaluation.
#     The ``info`` blob is not cached.
#
# All in-memory stores are guarded by a lock because enrichment runs under
# a ThreadPoolExecutor.

# yfinance spacing/backoff constants now live in tarzan.data._yf_net (shared).
_OPENFIGI_MIN_INTERVAL = 0.3     # spacing between OpenFIGI calls (~25/min cap)

_net_lock = threading.Lock()
_openfigi_memo: dict[str, list] = {}
_ticker_info_memo: dict[str, dict] = {}
_history_memo: dict[str, pd.DataFrame] = {}
_benchmark_memo: dict[str, pd.Series] = {}
# Per-currency FX history, so N holdings in USD don't each re-read disk and
# re-attempt the network refresh — the series is identical within a run.
_fx_memo: dict[str, pd.Series] = {}
_openfigi_last_call: list[float] = [0.0]  # mutable single-cell timestamp

# DataFrame attrs carry per-run provenance without entering the immutable disk
# cache. Origin is assigned per row so an invalid fresh tail cannot relabel an
# older selected cache close as primary evidence.
_HISTORY_ORIGINS_ATTR = "tarzan_price_origins"
_HISTORY_SYMBOL_ATTR = "tarzan_price_symbol"
_HISTORY_ORIGIN_PRIMARY = "primary"
_HISTORY_ORIGIN_CACHE = "cache"
# A close this venue never printed: today's bar reconstructed by applying a
# price-coherent sibling venue's OWN same-day return to this venue's last
# real close. See _fill_today_from_sibling.
_HISTORY_ORIGIN_SIBLING_RETURN = "sibling_return"
_HISTORY_SYNTHETIC_ATTR = "tarzan_price_synthetic"
_FX_EVIDENCE_ATTR = "tarzan_fx_evidence"


def reset_run_caches() -> None:
    """Clear all intra-run memoization. Called once at the start of each
    enrichment run so every run starts from fresh network state."""
    with _net_lock:
        _openfigi_memo.clear()
        _ticker_info_memo.clear()
        _history_memo.clear()
        _benchmark_memo.clear()
        _fx_memo.clear()
        _geo_breakdown_memo.clear()
        _geo_source_memo.clear()
        _openfigi_last_call[0] = 0.0
    from tarzan.data import market_quotes as _mq
    _mq.reset_quote_memo()
    _yf_net.reset()


# yfinance spacing + retry live in the shared network layer so the enricher,
# the backtest proxy fetch and the geo/ISIN resolver all share ONE throttle
# discipline. Bound the enricher's logger into retry so its debug lines keep
# the enricher origin.
def _space_yf_call() -> None:
    _yf_net.space_yf_call()


def _is_transient_error(exc: Exception) -> bool:
    return _yf_net.is_transient_error(exc)


def _retry(fn, *, what: str):
    """Run a live provider call only when the run contract permits it."""
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    return _yf_net.retry(fn, what=what, log=logger)


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------

def _get_fx_series(currency: str) -> pd.Series:
    """Get a daily FX rate series for currency→EUR conversion.

    For EUR, returns an empty series (sentinel for no conversion needed).
    Tries direct pair first, then inverse pair as fallback. Fetched fresh
    from yfinance on every call.

    An EMPTY currency takes the same sentinel. There is no ``EUR?=X`` pair to
    fetch, so asking for one only burns a request and logs "No FX data for ;
    conversion unavailable" with a blank where the code should be. All three
    callers route through here, which is why the guard belongs here rather than in
    ``convert_to_eur``.
    """
    if not currency or currency == "EUR":
        return pd.Series(dtype=float)
    return _fetch_fx_pair(currency)


def _usable_fx_series(series: Optional[pd.Series]) -> pd.Series:
    """Return finite positive FX rows while preserving transient provenance."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    attrs = dict(series.attrs)
    numeric = pd.to_numeric(series, errors="coerce")
    usable = numeric.replace(
        [float("inf"), float("-inf")],
        float("nan"),
    ).dropna()
    usable = usable[usable > 0]
    usable.attrs.update(attrs)
    return usable


def _fetch_fx_pair(currency: str) -> pd.Series:
    """Fetch FX history with row-level live/cache provenance.

    Memoized per currency for the run (cleared by :func:`reset_run_caches`):
    every holding in a given currency shares one series instead of re-reading
    the disk cache and re-attempting the refresh."""
    with _net_lock:
        if currency in _fx_memo:
            return _fx_memo[currency]

    result = _fetch_fx_pair_uncached(currency)
    with _net_lock:
        _fx_memo[currency] = result
    return result


def _fetch_fx_pair_uncached(currency: str) -> pd.Series:
    cache_key = f"FX_{currency}"
    cached = price_cache.load_history(cache_key)
    start = price_cache.refresh_start(cached)

    for pair, invert in [(f"{currency}EUR=X", False), (f"EUR{currency}=X", True)]:
        def _call(p=pair, s=start):
            _space_yf_call()
            ticker = yf.Ticker(p)
            # auto_adjust pinned for a stable "Close" column across yfinance
            # versions (FX pairs pay no dividends, so the value is unaffected;
            # this only guards the column set the reader below depends on).
            if s is not None:
                return ticker.history(start=s, interval="1d", auto_adjust=True)
            return ticker.history(period=_backtest_period(), interval="1d",
                                  auto_adjust=True)

        history = _retry(_call, what=f"FX {pair}")
        if history is not None and not history.empty:
            fresh = 1.0 / history["Close"] if invert else history["Close"]
            fresh = _usable_fx_series(fresh)
            if fresh.empty:
                continue
            merged = price_cache.merge_history(cached, fresh)
            result = merged if merged is not None and not merged.empty else fresh
            result.attrs[_HISTORY_ORIGINS_ATTR] = _history_origins(
                cached,
                fresh,
                result,
            )
            result.attrs[_HISTORY_SYMBOL_ATTR] = cache_key
            price_cache.store_history(cache_key, result)
            return result

    if cached is not None and not cached.empty:
        logger.debug("FX %s fetch failed; using cached history", currency)
        cached.attrs[_HISTORY_ORIGINS_ATTR] = {
            _history_timestamp_key(index): _HISTORY_ORIGIN_CACHE
            for index in cached.index
        }
        cached.attrs[_HISTORY_SYMBOL_ATTR] = cache_key
        return cached
    # Total FX failure (both pairs throttled AND no disk cache). Return the
    # EUR sentinel (empty series) rather than fabricating a 1.0 rate.
    logger.warning(
        "No FX data for %s; conversion unavailable "
        "(holding valued from last-known EUR anchor)",
        currency,
    )
    dq.warning(
        "enricher",
        f"FX rate for {currency}→EUR unavailable (both pairs failed, no cache); "
        "affected holdings valued from their last-known EUR anchor, not a live quote",
        context=currency,
    )
    return pd.Series(dtype=float)


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
# The list itself lives in tarzan.models.currency, below both this layer and the
# engine, because the seed path in returns_builder needed the same fact and had no
# way to reach it — so it valued a GBp position 100x too high.
from tarzan.models.currency import MINOR_TO_MAJOR as _MINOR_TO_MAJOR_CURRENCY


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


def _as_fraction(value) -> Optional[float]:
    """Normalize a yield/TER to a FRACTION (0.021 == 2.1%), tolerant of
    yfinance's inconsistent fraction-vs-percent field conventions.

    A genuine dividend yield or expense ratio expressed as a fraction is
    always well below 1.0 (1.0 would be 100%). Any value >= 1.0 therefore
    came from a percent-scaled field (e.g. dividendYield=2.4) and is divided
    by 100. None/NaN/non-positive inputs return None so downstream .fillna(0)
    treats them as absent. A value in (0, 1) is assumed already a fraction.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not (v == v) or v <= 0.0:  # NaN or non-positive
        return None
    return v / 100.0 if v >= 1.0 else v


# Exchanges that quote exclusively in EUR. A listing on one of these is EUR by
# definition, so a flaky ``info.currency`` (Yahoo intermittently returns USD or
# None for a Milan line under load) must never trigger an FX conversion of an
# already-EUR series — that corrupted the MSCI ACWI (ISAC.MI) benchmark to
# +1.7% against a real +0.08% on 22 Aug 2026. ``.L`` (GBP/USD), ``.SW`` (CHF)
# and suffixless US tickers are deliberately absent: their currency is genuinely
# ambiguous from the suffix and must be read from the provider.
_EUR_VENUE_SUFFIXES = (
    ".MI", ".SG", ".ETLX", ".DE", ".F", ".PA", ".MU", ".AS",
    ".VI", ".BR", ".LS", ".MC", ".HE", ".IR",
)


def venue_currency(symbol: str) -> Optional[str]:
    """The currency a venue quotes in when the suffix makes it unambiguous
    (EUR for a Eurozone exchange), else None so the caller reads the provider's
    reported currency. Authoritative over ``info.currency`` for EUR venues."""
    text = (symbol or "").upper()
    for suffix in _EUR_VENUE_SUFFIXES:
        if text.endswith(suffix):
            return "EUR"
    return None


def convert_to_eur(prices: pd.Series, currency: str) -> pd.Series:
    """Convert prices to EUR using only contemporaneous-or-earlier FX rows.

    Pinned runs clip FX to the effective boundary before alignment. Forward
    fill is intentionally one-way: a later FX observation must never backfill
    an earlier instrument price.
    """
    prices, currency = _normalize_minor_currency(prices, currency)
    if currency == "EUR":
        return prices
    fx = _usable_fx_series(_get_fx_series(currency))
    fx = _clip_to_as_of(fx)
    if fx.empty:
        return pd.Series(dtype=float)

    combined = pd.DataFrame({"price": prices, "fx": fx}).sort_index()
    combined["fx"] = combined["fx"].ffill()

    fx_origins = fx.attrs.get(_HISTORY_ORIGINS_ATTR, {})
    origin_markers = pd.Series(
        [
            fx_origins.get(
                _history_timestamp_key(index),
                _HISTORY_ORIGIN_PRIMARY,
            )
            for index in fx.index
        ],
        index=fx.index,
        dtype=object,
    ).reindex(combined.index).ffill()
    observation_markers = pd.Series(
        [
            pd.Timestamp(index).isoformat()
            if isinstance(fx.index, pd.DatetimeIndex)
            else None
            for index in fx.index
        ],
        index=fx.index,
        dtype=object,
    ).reindex(combined.index).ffill()

    converted = (combined["price"] * combined["fx"]).dropna()
    from tarzan import runtime

    cache_is_fallback = runtime.allows_live_transport()
    evidence: dict[str, dict[str, object]] = {}
    for index in converted.index:
        observation_value = observation_markers.loc[index]
        observation = _info_observation_time(observation_value)
        origin_value = origin_markers.loc[index]
        origin = (
            str(origin_value)
            if origin_value is not None and not pd.isna(origin_value)
            else _HISTORY_ORIGIN_PRIMARY
        )
        evidence[_history_timestamp_key(index)] = {
            "observation_time": observation,
            "is_fallback": (
                (origin == _HISTORY_ORIGIN_CACHE and cache_is_fallback)
                or observation is None
            ),
            "source": (
                "price_cache:FX"
                if origin == _HISTORY_ORIGIN_CACHE
                else "FX (undated)"
                if observation is None
                else None
            ),
        }
    converted.attrs[_FX_EVIDENCE_ATTR] = evidence
    return converted


# ---------------------------------------------------------------------------
# Ticker data fetching
# ---------------------------------------------------------------------------
ISIN_EXCHANGE_SUFFIXES = cfg.isin_exchange_suffixes()

_TICKER_SYMBOL_KEY = "_tarzan_selected_ticker"
_TICKER_METHOD_KEY = "_tarzan_ticker_selection_method"
_TICKER_REASON_KEY = "_tarzan_ticker_selection_reason"


def _annotate_ticker_data(
    data: dict,
    symbol: str,
    method: str,
    reason: str,
) -> dict:
    """Attach transient symbol-selection evidence to a provider payload."""
    data[_TICKER_SYMBOL_KEY] = symbol
    data[_TICKER_METHOD_KEY] = method
    data[_TICKER_REASON_KEY] = reason
    return data


def _history_has_market_evidence(history: object) -> bool:
    """Whether history contains a finite positive close visible to this run."""
    if history is None or getattr(history, "empty", True):
        return False
    if "Close" not in getattr(history, "columns", ()):
        return False
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    closes = closes[closes.map(math.isfinite) & (closes > 0)]
    try:
        closes = _clip_to_as_of(closes)
    except Exception:  # noqa: BLE001 — evidence checks must degrade closed
        return False
    return not closes.empty


def _positive_market_quote(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def _ticker_data_has_market_evidence(data: dict) -> bool:
    """Whether a selected symbol supplied a price accepted by valuation."""
    if _history_has_market_evidence(data.get("history")):
        return True
    try:
        from tarzan import runtime
        if runtime.as_of() is not None:
            return False
    except Exception:  # noqa: BLE001 — provenance must not break enrichment
        return False
    info = data.get("info") or {}
    return any(
        _positive_market_quote(info.get(field))
        for field in ("regularMarketPrice", "previousClose")
    )


def _is_expandable_bare_ticker(ticker: str) -> bool:
    value = str(ticker or "").strip()
    return bool(
        value
        and "." not in value
        and not value.startswith("^")
        and "=" not in value
        and "-" not in value
    )


def _bare_resolution_cache_key(ticker: str) -> str:
    bare = str(ticker or "").split(".", 1)[0].strip().upper()
    return f"TICKER:{bare}"


def _load_bare_resolution(ticker: str, *, include_expired: bool) -> str:
    key = _bare_resolution_cache_key(ticker)
    cached = price_cache.load_resolution(key)
    if not cached and include_expired:
        try:
            cached = price_cache._load_resolution_map().get(key, {}).get("symbol")
        except Exception:  # noqa: BLE001 — corrupt cache degrades to miss
            cached = None
    bare = key.removeprefix("TICKER:")
    if cached and str(cached).split(".", 1)[0].strip().upper() == bare:
        return str(cached)
    return ""


def _resolve_bare_ticker(
    ticker: str,
    *,
    expected_name: str,
    expected_currency: str,
) -> Optional[dict]:
    """Promote a bare ticker to one full listing with usable daily history.

    A previous live decision is replayable from cache. Discovery itself is
    live-only and ranks the bare symbol plus configured exchange suffixes by
    name coherence, currency, venue priority, and valid history. This keeps
    the input taxonomy venue-neutral while producing one canonical provider
    symbol for every downstream daily-data consumer.
    """
    from tarzan import runtime

    allows_live_transport = runtime.allows_live_transport()
    cached = _load_bare_resolution(
        ticker,
        include_expired=not allows_live_transport,
    )
    if cached:
        info = _fetch_ticker_info(cached)
        history = _fetch_history(cached)
        if _history_visible_at_boundary(history):
            return _annotate_ticker_data(
                {"info": info, "history": history},
                cached,
                "BARE_TICKER_RESOLUTION_CACHE",
                f"Reused the cached full listing previously selected for {ticker}.",
            )

    if not allows_live_transport:
        return None

    bare = str(ticker or "").strip().upper()
    symbols = [bare, *(f"{bare}{suffix}" for suffix in ISIN_EXCHANGE_SUFFIXES)]
    candidates = [
        candidate
        for symbol in dict.fromkeys(symbols)
        if (candidate := _fetch_candidate_meta(symbol)) is not None
        and (
            not expected_name
            or _name_match_score(candidate.name, expected_name) > 0
        )
    ]
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _rank_key(candidate, expected_name, expected_currency),
            _neg_str(candidate.symbol),
        ),
        reverse=True,
    )
    for candidate in ranked:
        history = _fetch_history(candidate.symbol)
        if not _history_visible_at_boundary(history):
            continue
        price_cache.store_resolution(
            _bare_resolution_cache_key(bare),
            candidate.symbol,
        )
        return _annotate_ticker_data(
            {"info": candidate.info, "history": history},
            candidate.symbol,
            "BARE_TICKER_PROMOTION",
            (
                f"Promoted bare ticker {ticker} to {candidate.symbol} after "
                "ranking matching listings with usable daily history."
            ),
        )
    return None


def _fetch_ticker_data(
    ticker: str,
    *,
    expected_name: str = "",
    expected_currency: str = "",
) -> dict:
    """Fetch market data and retain the exact provider symbol consumed.

    A qualified symbol is attempted directly. A bare symbol with known
    instrument name is resolved across configured venues before use, so a
    taxonomy containing only bare tickers still produces one full canonical
    history/current ticker. A qualified symbol is already resolved and is
    never replaced with a bare or sibling listing when data is unavailable.
    """
    logger.info("Fetching data for %s", ticker)
    if _is_expandable_bare_ticker(ticker) and expected_name:
        promoted = _resolve_bare_ticker(
            ticker,
            expected_name=expected_name,
            expected_currency=expected_currency,
        )
        if promoted is not None:
            return promoted
        return _annotate_ticker_data(
            {"info": {}, "history": pd.DataFrame()},
            ticker,
            "BARE_TICKER_UNRESOLVED",
            (
                f"No full listing matching bare ticker {ticker} and the "
                "instrument name supplied usable daily history."
            ),
        )

    selected_ticker = ticker
    # yfinance raises ValueError("Invalid ISIN number: ...") from the
    # Ticker constructor when the symbol looks like an ISIN but fails
    # its check-digit validation. The helpers degrade to empty evidence.
    info = _fetch_ticker_info(ticker)
    history = _fetch_history(ticker)
    if _ticker_data_has_market_evidence({"info": info, "history": history}):
        method = "DIRECT_TICKER"
        reason = "The supplied ticker directly supplied usable market data."
    else:
        method = "DIRECT_TICKER_NO_DATA"
        reason = (
            "The supplied ticker was retained as the canonical request, but "
            "it supplied no usable market data."
        )

    # A qualified ticker is an already-resolved identity.  Never strip its
    # venue and retry a bare symbol: doing so would let current valuation,
    # historical series and intraday charts consume different instruments.
    # Missing evidence remains missing and is handled by the run's typed data-
    # quality policy; only the preprocessing resolver may select a ticker.

    return _annotate_ticker_data(
        {"info": info, "history": history},
        selected_ticker,
        method,
        reason,
    )


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
#
# No ISINs are hardcoded and no extra input file is needed — the criteria
# are derived from OpenFIGI metadata + config, so this scales globally.

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

    ``curated`` is checked first, ahead of name matching: whether the
    candidate's own symbol is independently verifiable against the curated
    taxonomy (not just the hint that got it into the pool). Production
    evidence for FR0010755611 (2026-07-29/30 runs) showed a curated
    candidate (CL2.MI) losing to a non-curated one (18MF.MU) purely on
    name-token overlap -- 18MF.MU's yfinance name happened to mirror the
    OpenFIGI canonical name more closely than the correct venue's own
    metadata did. Name matching is a heuristic for when nothing more
    reliable exists; it should never outrank a candidate we can already
    confirm against ground truth.
    """
    curated = 1 if cfg.instrument_taxonomy_has(cand.symbol) else 0
    name_score = _name_match_score(cand.name, canonical_name)
    # Bucket the name score so tiny float differences don't reorder
    # otherwise-equivalent listings of the same instrument.
    name_bucket = round(name_score * 4)  # 0..4
    currency_match = 1 if _currency_matches(cand.currency, expected_currency) else 0
    # Negate suffix priority so a lower index sorts higher.
    suffix_rank = -_suffix_priority(cand.symbol)
    return (curated, name_bucket, currency_match, suffix_rank)


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


def _history_visible_at_boundary(history: Optional[pd.DataFrame]) -> bool:
    """Whether history contains a usable close visible to this run's clock."""
    return _history_has_market_evidence(history)


def _history_supports_a_return(history: Optional[pd.DataFrame]) -> bool:
    """Whether history has enough closes to compute a single return (>= 2).

    Candidate SELECTION needs a stricter test than
    :func:`_history_visible_at_boundary`, which is satisfied by one close. One
    close is a quote wearing a series' clothes: it cannot produce a return, so
    every downstream consumer (``holding_performance``, ``_holding_histories``,
    the historical-risk reconstruction) drops it on its own ``len < 2`` guard,
    and the instrument silently falls through to the carry-flat/synthetic path.

    That is precisely the failure the candidate walk says it exists to prevent —
    "pick the FIRST candidate that actually serves a usable price history, not
    merely a live quote" — so selection applies the same threshold the consumers
    do. 18MF.MU (one close) was beating CL2.MI (1276) for FR0010755611 on rank
    alone; a listing with real history now wins over one with a lone print.
    """
    if not _history_visible_at_boundary(history):
        return False
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    closes = closes[closes.map(math.isfinite) & (closes > 0)]
    try:
        closes = _clip_to_as_of(closes)
    except Exception:  # noqa: BLE001 — evidence checks must degrade closed
        return False
    return len(closes) >= 2


def _resolve_via_taxonomy_quote(
    clean_isin: str, taxonomy_ticker: str
) -> Optional[tuple[dict, str]]:
    """Last-resort ISIN resolution from curated identity + the v7 quote batch.

    When Yahoo rate-limits hard (CI), every candidate's ``info``/history call
    can fail, so :func:`_collect_candidate_metas` returns nothing and the ISIN
    would otherwise keep its raw form as a "ticker" — which no venue quotes and
    which :func:`_sibling_symbols` cannot expand, so the holding loses its 1D
    and renders as the bare ISIN (LU0328475792 / LU0380865021 on 23 Aug 2026).
    The curated taxonomy is offline authority for the bare ticker, and the v7
    quote endpoint is far more robust than ``quoteSummary``/history under load,
    so confirm a EUR venue through it and resolve there. History comes from the
    local cache when present (the returns series survives a throttled live
    call), and the confirmed venue is cached so it stops re-resolving.
    """
    if not taxonomy_ticker or not _is_expandable_bare_ticker(taxonomy_ticker):
        return None
    from tarzan.data.market_quotes import official_quotes

    venues = [f"{taxonomy_ticker}{suffix}" for suffix in _EUR_VENUE_SUFFIXES]
    quotes = official_quotes(venues)
    for symbol in venues:
        quote = (quotes or {}).get(symbol) or {}
        if not _positive_market_quote(quote.get("price")):
            continue
        history = _fetch_history(symbol)
        price_cache.store_resolution(clean_isin, symbol)
        logger.info(
            "Resolved ISIN %s → %s via curated taxonomy + v7 quote after "
            "candidate metadata was unavailable.", clean_isin, symbol,
        )
        return _annotate_ticker_data(
            {"info": {}, "history": history if history is not None else pd.DataFrame()},
            symbol,
            "INSTRUMENT_TAXONOMY_QUOTE",
            "Curated taxonomy identity confirmed by the v7 quote batch when "
            "candidate metadata was throttled.",
        ), symbol
    return None


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
    from tarzan import runtime
    from tarzan.models.instrument_key import normalize_ticker

    allows_live_transport = runtime.allows_live_transport()
    known_symbols: list[str] = []
    _, taxonomy_ticker = cfg.resolve_taxonomy_identity(clean_isin, "")
    xref_ticker = ""
    promoted_hint = ""
    if not allows_live_transport:
        # Resolve provider-independent identities before consulting the learned
        # resolution cache. A cached symbol from an older live run is accepted
        # only when the current ISIN authority (or, if none exists, the current
        # caller hint) still identifies it.
        xref_ticker = price_cache.load_ticker_isin_reverse(clean_isin)
        authoritative_symbols = [
            symbol for symbol in (taxonomy_ticker, xref_ticker) if symbol
        ]
        if authoritative_symbols:
            known_symbols = list(dict.fromkeys(authoritative_symbols))
            if hint_ticker and hint_ticker not in known_symbols:
                logger.debug(
                    "Pinned run ignored unverified ticker hint %s for ISIN %s",
                    hint_ticker,
                    clean_isin,
                )
        else:
            # No provider-independent ISIN mapping exists. The caller hint is
            # the only available identity evidence; promote a unique bare
            # ticker through taxonomy so ergonomic inputs still work.
            _, promoted_hint = cfg.resolve_taxonomy_identity("", hint_ticker)
            known_symbols = list(dict.fromkeys(
                symbol for symbol in (promoted_hint, hint_ticker) if symbol
            ))

    # Fast path: a previously-resolved symbol is cached on disk. Live runs may
    # reuse it directly. Pinned runs additionally require that it agrees with
    # the authoritative symbol set assembled above and has history visible at
    # the effective boundary; a current quote can never prove point-in-time
    # value.
    cached_symbol = price_cache.load_resolution(clean_isin)
    if not allows_live_transport and not cached_symbol:
        # Live TTLs govern refresh policy, not whether a pinned run may consume
        # an already-known immutable ISIN→symbol mapping. Read the persisted
        # entry even when its live refresh TTL elapsed; compatibility checks
        # below still require current provider-independent identity evidence.
        try:
            cached_entry = price_cache._load_resolution_map().get(clean_isin, {})
            cached_symbol = cached_entry.get("symbol")
        except Exception:  # noqa: BLE001 — cache corruption degrades to miss
            cached_symbol = None

    # A cached entry that maps the ISIN to ITSELF is degenerate poison from a
    # past throttled run: the raw ISIN quotes on no venue and _sibling_symbols
    # cannot expand it, so it costs the holding its 1D and renders as the ISIN.
    # ``instrument_taxonomy_has`` recognises the ISIN (its isin cell), so the
    # compatibility check below would otherwise trust it. When the taxonomy
    # knows a real ticker, drop it here so resolution re-runs and self-heals via
    # the taxonomy + v7 quote fallback — every run, since CI restores the
    # poisoned cache and never re-saves on a same-day cache hit.
    if (cached_symbol and taxonomy_ticker
            and cached_symbol.replace("-", "").casefold() == clean_isin.casefold()):
        logger.info(
            "Dropping degenerate cached resolution %s→itself for ISIN with "
            "curated ticker %s; re-resolving.", clean_isin, taxonomy_ticker,
        )
        cached_symbol = None

    cached_matches_taxonomy = bool(
        cached_symbol
        and taxonomy_ticker
        and (
            cached_symbol.casefold() == taxonomy_ticker.casefold()
            or (
                _is_expandable_bare_ticker(taxonomy_ticker)
                and normalize_ticker(cached_symbol)
                == normalize_ticker(taxonomy_ticker)
            )
        )
    )
    if allows_live_transport:
        # A live run trusts the cache for VENUE choice, but never to the point
        # of keeping a symbol that costs the instrument its curated identity.
        # Asset class and role are looked up by bare ticker, so a cached
        # 18MF.MU normalises to "18MF" — absent from the taxonomy — and the
        # holding lands in OTHER with no class. One unclassified valued holding
        # sets ``classification_available`` False, which blanks the ENTIRE
        # allocation section (every class 0% / €0, "no series", drift −125pp).
        #
        # So: if the taxonomy does not know the cached symbol but DOES know a
        # provider alias of this ISIN, the cached entry is stale relative to
        # curated data and must be re-resolved. A poisoned entry otherwise
        # survives its 30-day TTL, and CI restores the cache every run — which
        # is why the rendered mail stayed broken after the resolution fix.
        cached_is_compatible = True
        if cached_symbol and not cfg.instrument_taxonomy_has(cached_symbol):
            if any(
                cfg.instrument_taxonomy_has(alias)
                for alias in _openfigi_lookup(clean_isin)
            ):
                logger.info(
                    "Cached symbol %s for ISIN %s is unknown to the taxonomy "
                    "while a curated alias exists; re-resolving so the holding "
                    "keeps its asset class.",
                    cached_symbol, isin,
                )
                cached_is_compatible = False
    elif not cached_symbol:
        cached_is_compatible = False
    elif taxonomy_ticker:
        cached_is_compatible = cached_matches_taxonomy
    elif xref_ticker:
        # The learned reverse map is intentionally bare-ticker keyed. A cached
        # exact listing such as XDEM.MI is corroborated by the same bare symbol
        # without pretending that a different bare ticker is equivalent.
        cached_is_compatible = (
            normalize_ticker(cached_symbol) == normalize_ticker(xref_ticker)
        )
    else:
        cached_is_compatible = any(
            cached_symbol.casefold() == symbol.casefold()
            or normalize_ticker(cached_symbol) == normalize_ticker(symbol)
            for symbol in known_symbols
        )
    if cached_symbol and cached_is_compatible:
        info = _fetch_ticker_info(cached_symbol)
        history = _fetch_history(cached_symbol)
        has_quote = any(
            _positive_market_quote(info.get(field))
            for field in ("regularMarketPrice", "previousClose")
        )
        if _history_visible_at_boundary(history) or (
            allows_live_transport and has_quote
        ):
            logger.info("Resolved ISIN %s → %s (from cache)", isin, cached_symbol)
            if cached_matches_taxonomy:
                reason = (
                    "Reused cached ISIN resolution corroborated by instrument "
                    "taxonomy."
                )
            elif xref_ticker and (
                normalize_ticker(cached_symbol) == normalize_ticker(xref_ticker)
            ):
                reason = (
                    "Reused cached ISIN resolution corroborated by the learned "
                    "ISIN cross-reference."
                )
            elif hint_ticker and (
                cached_symbol.casefold() == hint_ticker.casefold()
                or normalize_ticker(cached_symbol) == normalize_ticker(hint_ticker)
            ):
                reason = (
                    "Reused cached ISIN resolution corroborated by the supplied "
                    "ticker."
                )
            else:
                reason = (
                    "Reused the live ISIN-resolution cache after the cached "
                    "symbol supplied usable market evidence."
                )
            return _annotate_ticker_data(
                {"info": info, "history": history},
                cached_symbol,
                "ISIN_RESOLUTION_CACHE",
                reason,
            ), cached_symbol
        logger.info(
            "Cached symbol %s for ISIN %s has no history visible at the run "
            "boundary; re-resolving",
            cached_symbol,
            isin,
        )
    elif cached_symbol:
        logger.info(
            "Pinned run ignored cached symbol %s conflicting with ISIN %s",
            cached_symbol,
            clean_isin,
        )

    if not allows_live_transport:
        # Pinned runs may consume only known symbols and cached history. Never
        # expand an ISIN into the configured Yahoo suffix probe matrix: that
        # sweep is discovery transport, not point-in-time evidence.
        for symbol in known_symbols:
            history = _fetch_history(symbol)
            if _history_visible_at_boundary(history):
                logger.info(
                    "Resolved ISIN %s → %s from pinned cache evidence",
                    isin,
                    symbol,
                )
                if taxonomy_ticker and symbol.casefold() == taxonomy_ticker.casefold():
                    method = "INSTRUMENT_TAXONOMY"
                    reason = "Instrument taxonomy mapped the ISIN to this canonical listing."
                elif xref_ticker:
                    method = "LEARNED_ISIN_XREF"
                    reason = "A learned ISIN-to-ticker cross-reference selected this cached symbol."
                else:
                    method = "SUPPLIED_TICKER_CACHE"
                    reason = "The supplied ticker was the only identity with boundary-visible cached history."
                return _annotate_ticker_data(
                    {"info": {}, "history": history},
                    symbol,
                    method,
                    reason,
                ), symbol
        logger.debug("Pinned run has no cached market history for ISIN %s", isin)
        return None

    canonical_name = _openfigi_name(clean_isin)

    # ISIN-only broker rows commonly carry the ISIN itself as their temporary
    # ticker. When curated taxonomy knows a bare ticker, use that identity as
    # the discovery hint so its qualified venue candidates are probed before
    # the bounded generic sweep. An explicit caller listing remains authoritative.
    hint_is_isin_only = (
        not hint_ticker
        or hint_ticker.replace("-", "").casefold() == clean_isin.casefold()
    )
    resolution_hint = (
        taxonomy_ticker
        if taxonomy_ticker and hint_is_isin_only
        else hint_ticker
    )
    # The taxonomy row for this instrument may be keyed only by ticker (its isin
    # cell empty), in which case the ISIN lookup above found nothing. But the
    # provider already tells us the ISIN's ticker aliases, and one of them is
    # usually the very ticker the taxonomy knows — so cross the two instead of
    # requiring the isin cell to be filled by hand. This is what makes an
    # ISIN-only broker row (Fineco) resolve to the same curated identity a
    # target file reaches by ticker: one resolver, reached from either key.
    if hint_is_isin_only and not taxonomy_ticker:
        for alias in _openfigi_lookup(clean_isin):
            # ``resolve_taxonomy_identity`` echoes its input back when nothing
            # matches, so it cannot answer "is this in the taxonomy?" — ask the
            # curated lookup directly, which only holds real rows.
            if cfg.instrument_taxonomy_has(alias):
                resolution_hint = alias
                logger.info(
                    "ISIN %s matched curated taxonomy via provider alias %s",
                    isin, alias,
                )
                break
    candidates = _collect_candidate_metas(clean_isin, resolution_hint)
    if not candidates:
        # Every candidate's metadata was throttled away. Rather than surrender
        # the curated identity to the raw ISIN (no quote, no siblings, no 1D),
        # confirm a EUR venue through the sturdier v7 quote batch.
        return _resolve_via_taxonomy_quote(clean_isin, taxonomy_ticker)

    # An ISIN may only resolve to a symbol something POSITIVELY identifies. The
    # ranking has no floor, so a candidate that is not curated, whose name matches
    # nothing, and whose currency disagrees could still win on venue priority
    # alone — and be cached as the authoritative mapping for that ISIN.
    #
    # No live instance is on record — this is a guard against a shape the ranking
    # permits, not a fix for an observed wrong answer. (It was written after a
    # suspected one that turned out to be correct: NESR.SG really is Nestle on
    # Stuttgart, and only the bare NESR on Nasdaq is a different company. Reading a
    # symbol without its venue is what made it look wrong.)
    #
    # Refusing leaves the instrument on the carry-flat/synthetic rung, which is
    # DEGRADED and disclosed. A confidently wrong listing would not be. Verified
    # against the real book before shipping: all 35 held ISINs still resolve,
    # including the nine that settle on a non-curated symbol — each of those has a
    # name overlap or a currency agreement.
    identified = [
        c for c in candidates
        if cfg.instrument_taxonomy_has(c.symbol)
        or _name_match_score(c.name, canonical_name) > 0
        or _currency_matches(c.currency, expected_currency)
    ]
    if identified:
        candidates = identified
    elif candidates:
        logger.warning(
            "ISIN %s: no candidate carries any identity evidence (checked %s); "
            "refusing to resolve rather than pick one on venue priority alone",
            clean_isin, ", ".join(c.symbol for c in candidates[:6]),
        )
        dq.warning(
            "instrument_resolution",
            "no provider listing could be positively identified for this ISIN "
            "(no curated match, no name overlap, no currency agreement); the "
            "position falls back to order-price history",
            context=clean_isin,
        )
        return None

    # Rank all candidates deterministically (best first). The alphabetical
    # symbol is the final tiebreak (smaller symbol wins when keys are equal).
    ranked = sorted(
        candidates,
        key=lambda c: (_rank_key(c, canonical_name, expected_currency), _neg_str(c.symbol)),
        reverse=True,
    )

    # Walk in rank order and pick the FIRST candidate that actually serves a
    # usable price *history* — not merely a live quote. A quote-only listing
    # (e.g. a thin Stuttgart ".SG" line that yfinance quotes but has no daily
    # series) would otherwise win on suffix priority and then force the whole
    # returns series onto the carry-flat fallback. Only a symbol with history
    # is cached, so a bad pick can't persist across runs. In the common case
    # the top-ranked symbol has history and we fetch exactly once, as before.
    best = ranked[0]
    for cand in ranked:
        history = _fetch_history(cand.symbol)
        if _history_supports_a_return(history):
            cand.data = _annotate_ticker_data(
                {"info": cand.info, "history": history},
                cand.symbol,
                "PROVIDER_CANDIDATE_RANKING",
                "Provider candidates were ranked by name coherence, currency, venue priority, and usable history.",
            )
            price_cache.store_resolution(clean_isin, cand.symbol)
            logger.info(
                "Resolved ISIN %s → %s (price=%.2f %s, name='%s')",
                isin, cand.symbol, cand.price, cand.currency, cand.name[:40],
            )
            return cand.data, cand.symbol

    # No candidate served a history. Return the top-ranked (best-effort quote)
    # but do NOT cache it, so next run re-resolves and can self-heal once a
    # vendor serves the series.
    history = _fetch_history(best.symbol)
    best.data = _annotate_ticker_data(
        {"info": best.info, "history": history},
        best.symbol,
        "PROVIDER_QUOTE_ONLY",
        "Provider candidate ranking selected this quote, but no candidate supplied usable history.",
    )
    logger.warning(
        "Resolved ISIN %s → %s by quote only (no price history from any "
        "candidate); not cached so it re-resolves next run.",
        isin, best.symbol,
    )
    return best.data, best.symbol


def _neg_str(s: str) -> tuple:
    """Helper to make alphabetical-ascending act as a max() tiebreak:
    returns a key that is *larger* for lexicographically smaller strings."""
    return tuple(-ord(c) for c in s)


def _collect_candidate_metas(clean_isin: str, hint_ticker: str) -> list[_Candidate]:
    """Build the deterministic candidate-symbol list and fetch metadata.

    Symbols are probed in a deterministic order and de-duplicated:

      1. the CSV/order ticker hint (if any), immediately followed by its
         configured exchange variants when the hint is bare;
      2. OpenFIGI-mapped exchange tickers;
      3. every OpenFIGI *bare* ticker combined with each configured
         exchange suffix — this is what lets the order-list path (which
         only knows the ISIN) discover the same local listing, e.g.
         ``EXUS`` + ``.MI`` → ``EXUS.MI``, that the holdings path finds
         via its ticker hint;
      4. the bare ISIN combined with each exchange suffix;
      5. the raw ISIN as-is.

    To bound network cost and rate-limit exposure the number of *fetched*
    candidates is capped; because hint-derived listings precede the generic
    sweep, a curated bare identity cannot be pushed beyond that cap.
    """
    symbols: list[str] = []

    def _add(sym: str) -> None:
        if sym and sym not in symbols:
            symbols.append(sym)

    if hint_ticker and hint_ticker != clean_isin:
        _add(hint_ticker)
        if _is_expandable_bare_ticker(hint_ticker):
            for suffix in ISIN_EXCHANGE_SUFFIXES:
                _add(f"{hint_ticker}{suffix}")
        else:
            # hint_ticker already carries a venue suffix (e.g. "CL2.PA" from
            # an OpenFIGI alias match against the curated taxonomy). Expand
            # its bare root across every configured venue too, the same way
            # a genuinely bare hint is expanded below -- otherwise the one
            # exact suffixed form the alias bridge happened to return is the
            # only qualified candidate tried, and the instrument's actual
            # primary listing (e.g. CL2.MI) stays buried behind the
            # unrelated alias roots in the generic sweep, unreached within
            # the fetch budget -- which is exactly how 18MF.MU won originally.
            bare_root = hint_ticker.split(".", 1)[0]
            if _is_expandable_bare_ticker(bare_root):
                _add(bare_root)
                for suffix in ISIN_EXCHANGE_SUFFIXES:
                    _add(f"{bare_root}{suffix}")

    figi_syms = _openfigi_lookup(clean_isin)
    bare_figi = [sym for sym in figi_syms if "." not in sym]

    # Already-qualified OpenFIGI symbols (they carry a venue) go first.
    for sym in figi_syms:
        if "." in sym:
            _add(sym)

    # Then the bare tickers crossed with the configured venues, SUFFIX-MAJOR:
    # every root on .MI, then every root on .SG, and so on. Ticker-major order
    # (all eight venues of the first root, then the second) buried the primary
    # listing behind the aliases of roots that do not trade.
    #
    # A BARE Yahoo symbol only quotes for suffixless US listings, so for a
    # non-US ISIN every bare candidate is a Bloomberg/venue code that 404s. They
    # are therefore probed LAST, after the qualified listings: OpenFIGI returned
    # ten bare tickers for FR0010755611 (Amundi MSCI USA 2x) — exactly
    # ``_MAX_RESOLVE_FETCHES`` — so the budget was spent entirely on 404s.
    # CL2.MI (1276 closes) sat at candidate 51 and was never probed, and the
    # ISIN resolved to a venue carrying a single close, which left the fund with
    # no usable history at all.
    for suffix in ISIN_EXCHANGE_SUFFIXES:
        for sym in bare_figi:
            _add(f"{sym}{suffix}")

    # US listings are genuinely suffixless, so the bare roots still have to be
    # tried — just not before every qualified venue candidate.
    for sym in bare_figi:
        _add(sym)

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

    Does NOT download price history for an ordinary candidate — that
    happens once for the winning symbol in :func:`_resolve_isin`. The one
    exception is a symbol the curated taxonomy already recognises: Yahoo's
    live ``info`` endpoint is unreliable for thin European listings even
    when the same symbol's daily history is solid, so a curated candidate
    that fails the live-quote check gets one history check before being
    discarded. This is scoped to curated symbols only -- it cannot
    resurrect an arbitrary low-quality candidate, only one we can already
    verify against ground truth -- so it does not reopen the cost the
    quote-only design avoids for the common case.
    """
    info = _fetch_ticker_info(symbol)
    price = info.get("regularMarketPrice") or info.get("previousClose")
    if not price or price <= 0:
        if not cfg.instrument_taxonomy_has(symbol):
            return None
        history = _fetch_history(symbol)
        if not _history_supports_a_return(history):
            return None
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        closes = closes[closes.map(math.isfinite) & (closes > 0)]
        if closes.empty:
            return None
        price = float(closes.iloc[-1])
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


def _history_timestamp_key(value: object) -> str:
    """Normalize an index value for transient row-provenance lookup."""
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    except (TypeError, ValueError):
        return str(value)


def _history_origins(
    cached: Optional[pd.DataFrame],
    fresh: pd.DataFrame,
    result: pd.DataFrame,
) -> dict[str, str]:
    """Classify each merged row by the source that won for its timestamp."""
    fresh_keys = {
        _history_timestamp_key(index)
        for index in fresh.index
    } if fresh is not None else set()
    cached_keys = {
        _history_timestamp_key(index)
        for index in cached.index
    } if cached is not None else set()
    origins: dict[str, str] = {}
    for index in result.index:
        key = _history_timestamp_key(index)
        if key in fresh_keys:
            origins[key] = _HISTORY_ORIGIN_PRIMARY
        elif key in cached_keys:
            origins[key] = _HISTORY_ORIGIN_CACHE
        else:
            origins[key] = _HISTORY_ORIGIN_PRIMARY
    return origins


def _sibling_close_series(symbol: str):
    """Daily closes for ``symbol``, fetched WITHOUT sibling filling.

    Deliberately bypasses :func:`_fetch_history` (and so cannot recurse into
    the fill): a synthesized close must never be derived from another
    synthesized close, or one venue's outage would propagate a chain of
    reconstructed prices across every listing of the fund. Returns ``None``
    on any failure — the fill is best-effort by design.
    """
    try:
        cached = price_cache.load_history(symbol)
        start = price_cache.refresh_start(cached)

        def _call():
            _space_yf_call()
            ticker = yf.Ticker(symbol)
            if start is not None:
                return ticker.history(start=start, interval="1d", auto_adjust=True)
            return ticker.history(period=_backtest_period(), interval="1d",
                                  auto_adjust=True)

        fresh = _retry(_call, what=f"sibling history {symbol}")
        if fresh is None:
            fresh = pd.DataFrame()
        merged = price_cache.merge_history(cached, fresh)
        frame = merged if merged is not None and not merged.empty else fresh
        if frame is None or frame.empty or "Close" not in frame:
            return None
        frame = _clip_to_as_of(frame)
        frame = price_cache.repair_split_jumps(frame)
        if frame is None or frame.empty:
            return None
        closes = frame["Close"].dropna()
        closes = closes[closes > 0]
        return closes if len(closes) else None
    except Exception:  # noqa: BLE001 — a best-effort fill must never break enrichment
        return None


def _fill_today_from_sibling(symbol: str, result: pd.DataFrame) -> pd.DataFrame:
    """Give ``symbol`` today's close by borrowing a sibling venue's own return.

    Yahoo's feed for some European listings (Borsa Italiana ``.MI`` most
    often) publishes today's daily bar late or not at all, while the same
    fund's Xetra/Paris listing already has it. Without this, that holding
    sits a day behind every other one in the book: its 1w/1m/YTD, TWROR,
    XIRR and risk are all measured to yesterday while the rest are measured
    to today.

    The borrowed quantity is a RETURN, never a price. Both endpoints of that
    return come from the sibling, so the venue basis cancels in the division;
    the result is then applied to THIS venue's last real close::

        synthetic = own_last_close * (1 + sibling_today / sibling_prev)

    Splicing the sibling's raw price instead would inject the venue basis as
    a fake return. Measured on NTSG (125 overlapping days) that basis has a
    0.31% standard deviation and swings 0.44% day to day against a 0.70%
    typical daily move — i.e. mostly noise, permanently embedded in the
    return chain and directly inflating volatility. Taking the return keeps
    every step venue-consistent. This mirrors what ``broker_1d`` already
    does for the intraday path, where both the current tick and the previous
    close come from the selected sibling.

    Deliberately narrow:

    * Only the CURRENT day is ever added, and only when it is genuinely
      missing. Older holes are left alone; a gap mid-series is a different
      problem and backfilling it would rewrite settled history run to run.
    * Only when the sibling's own previous close sits on this venue's last
      real close date, so the borrowed return spans exactly the missing step.
    * Only when the two venues' price levels agree within
      ``_SIBLING_PRICE_TOLERANCE`` — the same collision guard the intraday
      path uses to prove the sibling is the same instrument, not an unrelated
      fund sharing a ticker root.
    * Never in a pinned/reproducible run: the caller gates on that, so a
      point-in-time run reflects only what its own venue actually printed.
    """
    from tarzan.data.market_quotes import _sibling_symbols, _SIBLING_PRICE_TOLERANCE

    try:
        closes = result["Close"].dropna()
        closes = closes[closes > 0]
        if len(closes) < 2:
            return result
        own_last_index = closes.index[-1]
        own_last_date = pd.Timestamp(own_last_index).date()
        today = _today_local()
        if own_last_date >= today:
            return result  # already current — nothing to fill
        # Only the immediately-missing session is in scope. If this venue is
        # several sessions behind, that is a deeper outage than a late bar and
        # chaining borrowed returns across it is not something to do silently.
        if not _is_previous_trading_day(symbol, own_last_date, today):
            return result

        own_last_close = float(closes.iloc[-1])
        for candidate in _sibling_symbols(symbol):
            sib = _sibling_close_series(candidate)
            if sib is None or len(sib) < 2:
                continue
            sib_dates = [pd.Timestamp(i).date() for i in sib.index]
            if sib_dates[-1] != today:
                continue  # sibling has no today either
            # The sibling's previous close must sit exactly on this venue's
            # last real close date, so the borrowed return covers precisely
            # the one missing step and not a longer span.
            try:
                prev_pos = sib_dates.index(own_last_date)
            except ValueError:
                continue
            if prev_pos != len(sib_dates) - 2:
                continue
            sib_prev = float(sib.iloc[prev_pos])
            sib_today = float(sib.iloc[-1])
            if sib_prev <= 0 or sib_today <= 0:
                continue
            # Same-instrument guard: levels must be comparable, else an equal
            # ticker root on another exchange is a different fund.
            if abs(sib_prev / own_last_close - 1.0) > _SIBLING_PRICE_TOLERANCE:
                logger.info(
                    "daily tail fill %s→%s rejected (%.1f%% off own last close)",
                    symbol, candidate,
                    abs(sib_prev / own_last_close - 1.0) * 100,
                )
                continue

            sibling_return = sib_today / sib_prev - 1.0
            synthetic = own_last_close * (1.0 + sibling_return)
            new_index = pd.Timestamp(today)
            existing_index = pd.Timestamp(own_last_index)
            if existing_index.tz is not None:
                new_index = new_index.tz_localize(existing_index.tz)

            filled = result.copy()
            row = {col: float("nan") for col in filled.columns}
            row["Close"] = synthetic
            filled.loc[new_index] = row
            filled = filled.sort_index()

            origins = dict(result.attrs.get(_HISTORY_ORIGINS_ATTR, {}))
            origins[_history_timestamp_key(new_index)] = (
                _HISTORY_ORIGIN_SIBLING_RETURN
            )
            filled.attrs.update(result.attrs)
            filled.attrs[_HISTORY_ORIGINS_ATTR] = origins
            synthetic_map = dict(result.attrs.get(_HISTORY_SYNTHETIC_ATTR, {}))
            synthetic_map[_history_timestamp_key(new_index)] = {
                "source": candidate,
                "sibling_return_pct": sibling_return * 100.0,
                "own_last_close": own_last_close,
                "own_last_date": str(own_last_date),
                "synthetic_close": synthetic,
            }
            filled.attrs[_HISTORY_SYNTHETIC_ATTR] = synthetic_map

            logger.info(
                "daily tail fill: %s has no %s close; applied %s's own "
                "%+.4f%% move to %s's last close %.4f → %.4f",
                symbol, today, candidate, sibling_return * 100.0,
                symbol, own_last_close, synthetic,
            )
            dq.info(
                "market_data",
                f"no {today} close from this venue; today's close derived by "
                f"applying {candidate}'s own {sibling_return * 100:+.4f}% move "
                f"to the last real close {own_last_close:.4f} → "
                f"{synthetic:.4f} (venue-consistent return, not a spliced price)",
                context=symbol,
            )
            return filled
    except Exception:  # noqa: BLE001 — a best-effort fill must never break enrichment
        logger.debug("daily tail fill failed for %s", symbol, exc_info=True)
    return result


def _today_local():
    """Today's date in the run's own clock (as_of when pinned)."""
    from tarzan import runtime
    as_of = runtime.as_of()
    if as_of is not None:
        return as_of
    return pd.Timestamp.now().date()


def _is_previous_trading_day(symbol: str, own_last_date, today) -> bool:
    """Whether ``own_last_date`` is the trading day immediately before
    ``today`` for ``symbol``'s exchange — i.e. exactly one session is
    missing, not several.

    Reads the vendored exchange calendar, so the day after a holiday correctly
    treats the session before it as adjacent. Before that this skipped weekends
    only, and a holiday made it decline: the tail fill simply did not happen on
    ~6-17 days a year depending on the venue.
    """
    from tarzan.data.exchange_calendar import previous_session

    return own_last_date == previous_session(symbol, today)


def _fetch_history(symbol: str) -> pd.DataFrame:
    """Fetch and merge daily history while preserving row-level provenance.

    Cached rows remain available when the recent provider tail fails, but each
    row retains cache/live origin in transient DataFrame metadata. Downstream
    selection can therefore classify the actual last valid close rather than
    treating the whole merged frame as fresh because a request just ran.
    """
    with _net_lock:
        if symbol in _history_memo:
            return _history_memo[symbol]

    cached = price_cache.load_history(symbol)
    start = price_cache.refresh_start(cached)
    # A tail refresh can only extend the END; if the cache does not yet span the
    # full backtest window its HEAD may be missing (an instrument cached after
    # its inception), so pull the whole period to backfill it. merge_history
    # then keeps the union. Guards the young-fund YTD/since-inception anchor.
    if start is not None and not price_cache.covers_period(cached, _backtest_period()):
        start = None

    def _call():
        _space_yf_call()
        ticker = yf.Ticker(symbol)
        # auto_adjust pinned EXPLICITLY (the yfinance default flip-flopped
        # across versions — see proxy_data): total-return closes, so a
        # benchmark's dividends live in its price. Holdings are currently all
        # accumulating (no DIVIDEND orders) or bonds priced on the clean price
        # via Borsa, so total-return == price-only for them today. If a
        # DISTRIBUTING holding is ever added AND its dividends are booked as
        # orders (the chosen policy), that holding's price must switch to
        # price-only to avoid double-counting the income — build_order_derived
        # _series guards that case at run time.
        if start is not None:
            return ticker.history(start=start, interval="1d", auto_adjust=True)
        return ticker.history(period=_backtest_period(), interval="1d",
                              auto_adjust=True)

    fresh = _retry(_call, what=f"history {symbol}")
    if fresh is None:
        fresh = pd.DataFrame()

    merged = price_cache.merge_history(cached, fresh)
    result = merged if merged is not None and not merged.empty else fresh
    if fresh.empty and result is not None and not result.empty:
        try:
            last_timestamp = pd.Timestamp(result.index.max())
            if last_timestamp.tz is not None:
                last_timestamp = last_timestamp.tz_convert("UTC").tz_localize(None)
            age_days = (pd.Timestamp.now() - last_timestamp).days
            if age_days > price_cache.REFRESH_TAIL_DAYS:
                logger.warning(
                    "%s: live fetch returned no data; newest close is %d day(s) "
                    "old (%s) — retaining explicitly stale cache evidence.",
                    symbol,
                    age_days,
                    last_timestamp.date(),
                )
                dq.warning(
                    "market_data",
                    f"live fetch returned no data; newest close is {age_days} day(s) "
                    f"old ({last_timestamp.date()}) — retaining explicitly stale "
                    "cache evidence",
                    context=symbol,
                )
        except Exception:  # noqa: BLE001 — diagnostics cannot break fetching
            pass

    origins = _history_origins(cached, fresh, result)
    from tarzan import runtime

    pinned = not runtime.allows_live_transport()
    if pinned:
        # Split detection may use later sessions to confirm a discontinuity and
        # back-adjust all earlier closes. In a point-in-time run that would let
        # a post-boundary split rewrite pre-boundary valuation evidence. Clip
        # first, repair only the visible frame, and leave the full disk cache
        # untouched for future/live runs.
        result = _clip_to_as_of(result)
    result = price_cache.repair_split_jumps(result)
    if result is not None and not result.empty:
        result.attrs[_HISTORY_ORIGINS_ATTR] = origins
        result.attrs[_HISTORY_SYMBOL_ATTR] = symbol
        if not pinned:
            # Store BEFORE the sibling fill: only closes this venue actually
            # printed belong in its own on-disk cache. A synthesized bar is a
            # per-run reconstruction, and persisting it would let it be read
            # back next run as though the venue had printed it — and then
            # become the base another synthetic close is derived from.
            price_cache.store_history(symbol, result)
            # Live runs only. A pinned/reproducible run must reflect exactly
            # what its own venue printed, or the same as_of would yield
            # different history depending on which venues answered.
            result = _fill_today_from_sibling(symbol, result)

    with _net_lock:
        _history_memo[symbol] = result
    return result


# ---------------------------------------------------------------------------
# OpenFIGI API
# ---------------------------------------------------------------------------
_FIGI_EXCHANGE_MAP = cfg.figi_exchange_map()
_FIGI_MIC_MAP = cfg.figi_mic_map()


_OPENFIGI_BATCH_SIZE = 10


def _openfigi_raw_many(isins: list[str]) -> dict[str, list]:
    """Fetch uncached ISIN mappings in bounded OpenFIGI request batches."""
    from tarzan import runtime

    ordered = list(dict.fromkeys(str(isin or "").strip() for isin in isins))
    ordered = [isin for isin in ordered if isin]
    if not ordered:
        return {}
    if not runtime.allows_live_transport():
        logger.debug("Pinned run: skipping %d live OpenFIGI lookup(s)", len(ordered))
        return {isin: [] for isin in ordered}

    with _net_lock:
        missing = [isin for isin in ordered if isin not in _openfigi_memo]

    for offset in range(0, len(missing), _OPENFIGI_BATCH_SIZE):
        batch = missing[offset:offset + _OPENFIGI_BATCH_SIZE]

        def _call(values=batch) -> list:
            # Enforce minimum spacing between OpenFIGI requests across threads.
            with _net_lock:
                wait = _OPENFIGI_MIN_INTERVAL - (
                    _time.monotonic() - _openfigi_last_call[0]
                )
                if wait > 0:
                    _time.sleep(wait)
                _openfigi_last_call[0] = _time.monotonic()
            payload = [
                {"idType": "ID_ISIN", "idValue": value}
                for value in values
            ]
            request = Request(
                "https://api.openfigi.com/v3/mapping",
                data=json.dumps(payload).encode("utf-8"),
            )
            request.add_header("Content-Type", "application/json")
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        response = _retry(
            _call,
            what=f"OpenFIGI batch ({len(batch)} ISINs)",
        )
        valid_response = (
            isinstance(response, list) and len(response) == len(batch)
        )
        if not valid_response:
            logger.debug(
                "OpenFIGI batch returned an invalid shape for %d ISIN(s)",
                len(batch),
            )
        with _net_lock:
            for index, isin in enumerate(batch):
                _openfigi_memo[isin] = (
                    [response[index]] if valid_response else []
                )

    with _net_lock:
        return {isin: _openfigi_memo.get(isin, []) for isin in ordered}


def _openfigi_raw(isin: str) -> list:
    """Return one rate-limited, per-run-memoized OpenFIGI mapping response.

    Pinned runs fail closed before reading a live memo or opening a socket;
    their instrument metadata must come from the as-of-compatible profile
    cache. Live callers share the same bounded batch transport used by the
    effective-order resolver.
    """
    return _openfigi_raw_many([isin]).get(isin, [])


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
    """Return exact OpenFIGI bond evidence for an ISIN, if present.

    OpenFIGI labels exchange-traded products with ``marketSector=Equity``
    even when ``securityType=ETP``.  Passing that exposure sector into the
    mechanics resolver conflicts with the instrument's ETF evidence and can
    make a valid quote unusable.  Only forward fields that jointly resolve to
    bond mechanics; non-bond records contribute no bond assertion.
    """
    from tarzan.instruments.registry import InstrumentKind, TypeEvidenceGateway

    clean = isin.replace("-", "")
    for result_group in _openfigi_raw(clean):
        for item in result_group.get("data", []):
            sector = item.get("marketSector")
            sec_type = item.get("securityType2") or item.get("securityType")
            resolution = TypeEvidenceGateway().resolve(sector, sec_type)
            if resolution.kind is InstrumentKind.BOND:
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
        # OpenFIGI's security type resolves ETF mechanics only. It does not
        # identify the tracked exposure category: fixed-income, commodity,
        # gold, equity, and multi-asset ETFs share the same mechanics kind.
        info["instrument_type"] = "ETF"
    elif sec_type == "money market":
        info["asset_class"] = AssetClass.CASH_EQUIVALENTS
        info["instrument_type"] = "Money Market"

    return info


# ---------------------------------------------------------------------------
# Effective-order instrument profiles and equivalence resolution
# ---------------------------------------------------------------------------

_INSTRUMENT_PROFILE_RESOLVER_VERSION = "2"
_EQUIVALENCE_QUANTITY_EPSILON = 1e-8
_TRANSACTIONAL_EQUIVALENCE_MAX_DAYS = 45


def _normalized_profile_isin(value: object) -> str:
    return "".join(
        character
        for character in str(value or "").strip().upper()
        if character.isalnum()
    )


def _figi_values(items: list[dict], field_name: str) -> list[str]:
    return sorted({
        str(item.get(field_name)).strip()
        for item in items
        if str(item.get(field_name) or "").strip()
    })


def _figi_item_kind(item: dict):
    """Resolve mechanics from OpenFIGI's structured fields, not its name."""
    from tarzan.instruments.registry import InstrumentKind, TypeEvidenceGateway

    # OpenFIGI uses securityType=ETP with securityType2=Mutual Fund for ETFs,
    # bond ETFs, ETCs, and other exchange-traded products. ETP is exact
    # provider mechanics evidence; marketSector=Equity must not turn it into a
    # stock or into an exposure-category assertion.
    provider_security_type = str(item.get("securityType") or "").strip().upper()
    if provider_security_type == "ETP":
        return InstrumentKind.ETF

    kw = cfg.classification()
    security_type = str(item.get("securityType2") or "").strip().lower()
    market_sector = str(item.get("marketSector") or "").strip().lower()
    classified = _classify_figi_item(
        security_type,
        market_sector,
        str(item.get("name") or ""),
        kw,
        kw.get("figi_fixed_income", []),
        kw.get("figi_equity", []),
        kw.get("figi_etf", []),
    )
    declared_type = classified.get("instrument_type")
    if declared_type:
        return TypeEvidenceGateway().resolve(declared_type).kind

    # An equity market sector is not sufficient to distinguish a stock from
    # an ETF. Fixed-income sectors, by contrast, are exact mechanics evidence.
    sector_evidence = (
        item.get("marketSector")
        if market_sector in {"govt", "corp", "mtge", "muni"}
        else None
    )
    return TypeEvidenceGateway().resolve(
        item.get("securityType2"),
        item.get("securityType"),
        sector_evidence,
    ).kind


def _build_openfigi_profile(isin: str, results: list) -> dict:
    """Convert one successful OpenFIGI response into a cache-safe profile."""
    from tarzan import runtime

    items = [
        item
        for group in results
        if isinstance(group, dict)
        for item in group.get("data", [])
        if isinstance(item, dict)
    ]
    kinds = {
        kind
        for kind in (_figi_item_kind(item) for item in items)
        if kind is not None
    }
    if len(kinds) == 1:
        status = "VERIFIED"
        kind_value = next(iter(kinds)).value
        confidence = "HIGH"
    elif len(kinds) > 1:
        status = "CONFLICTING"
        kind_value = None
        confidence = "CONFLICTING"
    else:
        status = "UNRESOLVED"
        kind_value = None
        confidence = "NONE"

    observed_at = runtime.context().captured_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    return {
        "isin": _normalized_profile_isin(isin),
        "kind": kind_value,
        "identifiers": {
            "figi": _figi_values(items, "figi"),
            "compositeFIGI": _figi_values(items, "compositeFIGI"),
            "shareClassFIGI": _figi_values(items, "shareClassFIGI"),
        },
        "securityType": _figi_values(items, "securityType"),
        "securityType2": _figi_values(items, "securityType2"),
        "marketSector": _figi_values(items, "marketSector"),
        "names": _figi_values(items, "name"),
        "tickers": _figi_values(items, "ticker"),
        "source": "OpenFIGI",
        "provenance": {
            "provider": "OpenFIGI",
            "idType": "ID_ISIN",
        },
        "resolver_version": _INSTRUMENT_PROFILE_RESOLVER_VERSION,
        "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
        "status": status,
        "confidence": confidence,
    }


def _reclassify_cached_profile(profile: Optional[dict]) -> Optional[dict]:
    """Apply the current resolver policy to already-observed raw evidence."""
    if profile is None:
        return None
    updated = dict(profile)
    security_types = {
        str(value or "").strip().upper()
        for value in profile.get("securityType", [])
        if str(value or "").strip()
    }
    if security_types == {"ETP"}:
        updated.update({
            "kind": "ETF",
            "status": "VERIFIED",
            "confidence": "HIGH",
        })
    updated["resolver_version"] = _INSTRUMENT_PROFILE_RESOLVER_VERSION
    return updated


def _load_or_fetch_instrument_profile(isin: str) -> tuple[Optional[dict], str]:
    """Resolve one profile cache-first, with live transport only in LIVE mode."""
    from tarzan import runtime

    if runtime.allows_live_transport():
        cached = price_cache.load_instrument_profile(isin)
        if cached is not None:
            reclassified = _reclassify_cached_profile(cached)
            if reclassified != cached:
                price_cache.store_instrument_profile(isin, reclassified)
                return reclassified, "cache_reclassified"
            return cached, "cache"
        stale = price_cache.load_instrument_profile(isin, allow_stale=True)
        results = _openfigi_raw(isin)
        if results:
            profile = _build_openfigi_profile(isin, results)
            price_cache.store_instrument_profile(isin, profile)
            return profile, "openfigi"
        if stale is not None:
            return _reclassify_cached_profile(stale), "stale_cache"
        return None, "unavailable"

    cached = price_cache.load_instrument_profile(
        isin,
        as_of=runtime.as_of(),
        allow_stale=True,
    )
    cached = _reclassify_cached_profile(cached)
    return (cached, "cache_as_of") if cached is not None else (None, "unavailable")


def _profile_kind(profile: Optional[dict]):
    from tarzan.instruments.registry import InstrumentKind

    if not profile or profile.get("status") != "VERIFIED":
        return None
    try:
        return InstrumentKind(str(profile.get("kind") or "").strip().upper())
    except ValueError:
        return None


def _strong_equivalence_components(
    profiles: dict[str, Optional[dict]],
) -> list[tuple[set[str], list[str]]]:
    """Return connected ISIN sets sharing exact provider identity values."""
    claims: dict[tuple[str, str], set[str]] = {}
    for isin, profile in profiles.items():
        identifiers = profile.get("identifiers", {}) if profile else {}
        if not isinstance(identifiers, dict):
            continue
        for field_name in ("figi", "compositeFIGI", "shareClassFIGI"):
            values = identifiers.get(field_name, [])
            if not isinstance(values, list):
                continue
            for value in values:
                normalized = str(value or "").strip().upper()
                if normalized:
                    claims.setdefault((field_name, normalized), set()).add(isin)

    parent = {isin: isin for isin in profiles}

    def find(isin: str) -> str:
        while parent[isin] != isin:
            parent[isin] = parent[parent[isin]]
            isin = parent[isin]
        return isin

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for members in claims.values():
        ordered = sorted(members)
        for member in ordered[1:]:
            union(ordered[0], member)

    components: dict[str, set[str]] = {}
    for isin in profiles:
        components.setdefault(find(isin), set()).add(isin)

    result = []
    for members in components.values():
        if len(members) < 2:
            continue
        evidence = sorted(
            f"{field_name}:{value}"
            for (field_name, value), claim_members in claims.items()
            if len(claim_members & members) >= 2
        )
        if evidence:
            result.append((members, evidence))
    return sorted(result, key=lambda item: sorted(item[0]))


def _name_identity_tokens(names: set[str]) -> tuple[set[str], set[str]]:
    tokens: set[str] = set()
    for name in names:
        normalized = "".join(
            character.upper() if character.isalnum() else " "
            for character in name
        )
        tokens.update(normalized.split())
    issue_tokens = {
        token
        for token in tokens
        if len(token) >= 5
        and any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    }
    issuer_tokens = {
        token for token in tokens if len(token) >= 3 and token.isalpha()
    }
    return issue_tokens, issuer_tokens


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _order_instrument_summaries(orders: list) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for order in orders:
        isin = _normalized_profile_isin(order.isin)
        if not isin:
            continue
        summary = summaries.setdefault(isin, {
            "net_quantity": 0.0,
            "position_quantities": [],
            "position_types": set(),
            "dates": [],
            "currencies": set(),
            "sources": set(),
            "names": set(),
            "prices": [],
        })
        if order.currency:
            summary["currencies"].add(str(order.currency).strip().upper())
        if order.source:
            summary["sources"].add(str(order.source).strip().casefold())
        if order.name:
            summary["names"].add(str(order.name).strip())
        if not order.is_position_change() or math.isclose(
            float(order.quantity), 0.0, abs_tol=_EQUIVALENCE_QUANTITY_EPSILON
        ):
            continue
        quantity = float(order.quantity)
        summary["net_quantity"] += quantity
        summary["position_quantities"].append(quantity)
        summary["position_types"].add(order.type)
        summary["dates"].append(order.trade_date or order.date)
        try:
            price = float(order.price_native)
            if math.isfinite(price) and price > 0:
                summary["prices"].append(price)
        except (TypeError, ValueError):
            pass
    return summaries


def _transactional_equivalence_pairs(
    orders: list,
    kind_by_isin: dict,
    excluded_isins: set[str],
) -> list[tuple[str, str]]:
    """Find mutually unique bond identifier rollovers from ledger evidence.

    This deliberately requires several independent signals. It never compares
    ISIN prefixes, ticker text, or a name by itself.
    """
    from tarzan.instruments.registry import InstrumentKind
    from tarzan.models.order import OrderType

    summaries = _order_instrument_summaries(orders)
    candidates: list[tuple[str, str]] = []
    eligible = sorted(set(summaries) - excluded_isins)
    for index, left_isin in enumerate(eligible):
        for right_isin in eligible[index + 1:]:
            left, right = summaries[left_isin], summaries[right_isin]
            left_quantities = left["position_quantities"]
            right_quantities = right["position_quantities"]
            if not left_quantities or not right_quantities:
                continue
            left_direction = 1 if all(value > 0 for value in left_quantities) else -1 if all(
                value < 0 for value in left_quantities
            ) else 0
            right_direction = 1 if all(value > 0 for value in right_quantities) else -1 if all(
                value < 0 for value in right_quantities
            ) else 0
            if {left_direction, right_direction} != {-1, 1}:
                continue
            positive, negative = (
                (left, right) if left_direction > 0 else (right, left)
            )
            if positive["position_types"] != {OrderType.TRANSFER_IN}:
                continue
            if not negative["position_types"] or not negative["position_types"].issubset(
                {OrderType.SELL, OrderType.TRANSFER_OUT}
            ):
                continue
            if not math.isclose(
                left["net_quantity"],
                -right["net_quantity"],
                rel_tol=1e-10,
                abs_tol=_EQUIVALENCE_QUANTITY_EPSILON,
            ):
                continue
            if (
                len(left["currencies"]) != 1
                or left["currencies"] != right["currencies"]
                or len(right["currencies"]) != 1
                or len(left["sources"]) != 1
                or left["sources"] != right["sources"]
                or len(right["sources"]) != 1
            ):
                continue
            if not positive["dates"] or not negative["dates"]:
                continue
            elapsed_days = (
                min(negative["dates"]) - max(positive["dates"])
            ).days
            if not 0 <= elapsed_days <= _TRANSACTIONAL_EQUIVALENCE_MAX_DAYS:
                continue

            left_issue, left_issuer = _name_identity_tokens(left["names"])
            right_issue, right_issuer = _name_identity_tokens(right["names"])
            if not (left_issue & right_issue) or not (left_issuer & right_issuer):
                continue
            left_price, right_price = _median(left["prices"]), _median(right["prices"])
            if left_price is None or right_price is None:
                continue
            price_ratio = min(left_price, right_price) / max(left_price, right_price)
            if price_ratio < 0.8:
                continue

            kinds = {kind_by_isin.get(left_isin), kind_by_isin.get(right_isin)}
            if kinds != {None, InstrumentKind.BOND}:
                continue
            candidates.append((left_isin, right_isin))

    neighbors: dict[str, set[str]] = {}
    for left_isin, right_isin in candidates:
        neighbors.setdefault(left_isin, set()).add(right_isin)
        neighbors.setdefault(right_isin, set()).add(left_isin)
    return [
        pair
        for pair in candidates
        if len(neighbors.get(pair[0], set())) == 1
        and len(neighbors.get(pair[1], set())) == 1
    ]


def _automatic_group_id(source: str, members: set[str]) -> str:
    digest = hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
    return f"AUTO-{source.upper()}-{digest}"


def resolve_effective_order_instruments(orders: list) -> tuple[list, dict]:
    """Attach generated kind/equivalence evidence to effective-order copies."""
    from tarzan import runtime
    from tarzan.instruments.registry import InstrumentKind

    isins = sorted({
        _normalized_profile_isin(order.isin)
        for order in orders
        if _normalized_profile_isin(order.isin)
    })
    profiles: dict[str, Optional[dict]] = {}
    profile_sources: dict[str, str] = {}
    if runtime.allows_live_transport():
        refresh_isins = [
            isin
            for isin in isins
            if price_cache.load_instrument_profile(isin) is None
        ]
        _openfigi_raw_many(refresh_isins)
    for isin in isins:
        profile, source = _load_or_fetch_instrument_profile(isin)
        profiles[isin] = profile
        profile_sources[isin] = source

    explicit_kinds: dict[str, set] = {}
    explicit_groups: dict[str, set[str]] = {}
    for order in orders:
        isin = _normalized_profile_isin(order.isin)
        if order.instrument_kind is not None:
            explicit_kinds.setdefault(isin, set()).add(order.instrument_kind)
        group = str(order.instrument_equivalence_group or "").strip()
        if group:
            explicit_groups.setdefault(isin, set()).add(group.casefold())

    conflicts: list[str] = []
    blocked_isins: set[str] = set()
    kind_by_isin: dict[str, Optional[InstrumentKind]] = {}
    for isin in isins:
        declared = explicit_kinds.get(isin, set())
        profile_kind = _profile_kind(profiles.get(isin))
        if len(declared) > 1:
            conflicts.append(f"{isin}: conflicting declared kinds")
            blocked_isins.add(isin)
            kind_by_isin[isin] = None
        elif declared:
            declared_kind = next(iter(declared))
            kind_by_isin[isin] = declared_kind
            if profile_kind is not None and profile_kind is not declared_kind:
                conflicts.append(f"{isin}: declared/profile kind conflict")
                blocked_isins.add(isin)
        else:
            kind_by_isin[isin] = profile_kind
        if len(explicit_groups.get(isin, set())) > 1:
            conflicts.append(f"{isin}: conflicting declared equivalence groups")
            blocked_isins.add(isin)

    automatic_group_by_isin: dict[str, str] = {}
    accepted_groups: list[dict] = []
    reserved_isins = set(explicit_groups) | blocked_isins
    for members, evidence in _strong_equivalence_components(profiles):
        if members & reserved_isins:
            continue
        member_kinds = {
            kind_by_isin.get(member)
            for member in members
            if kind_by_isin.get(member) is not None
        }
        if len(member_kinds) > 1:
            conflicts.append(
                f"{', '.join(sorted(members))}: provider identity kind conflict"
            )
            blocked_isins.update(members)
            continue
        group_id = _automatic_group_id("FIGI", members)
        for member in members:
            automatic_group_by_isin[member] = group_id
        accepted_groups.append({
            "group": group_id,
            "source": "strong_provider_identifier",
            "members": sorted(members),
            "evidence": evidence,
        })

    transactional_exclusions = (
        reserved_isins | blocked_isins | set(automatic_group_by_isin)
    )
    for left_isin, right_isin in _transactional_equivalence_pairs(
        orders,
        kind_by_isin,
        transactional_exclusions,
    ):
        members = {left_isin, right_isin}
        group_id = _automatic_group_id("LEDGER", members)
        for member in members:
            automatic_group_by_isin[member] = group_id
        accepted_groups.append({
            "group": group_id,
            "source": "unique_transactional_rollover",
            "members": sorted(members),
            "evidence": [
                "net_flat_opposite_quantity",
                "transfer_in_then_disposal",
                "matching_currency_source_issue_and_price",
            ],
        })

    # Propagate one non-conflicting mechanics kind across every accepted
    # identity component. This is what gives an obsolete cum identifier the
    # verified bond mechanics of its OpenFIGI-resolved ex identifier.
    for group in accepted_groups:
        members = set(group["members"])
        member_kinds = {
            kind_by_isin.get(member)
            for member in members
            if kind_by_isin.get(member) is not None
        }
        if len(member_kinds) == 1:
            resolved_kind = next(iter(member_kinds))
            for member in members:
                if member not in blocked_isins:
                    kind_by_isin[member] = resolved_kind

    resolved_orders = []
    for order in orders:
        isin = _normalized_profile_isin(order.isin)
        resolved_orders.append(replace(
            order,
            instrument_kind=(order.instrument_kind or kind_by_isin.get(isin)),
            instrument_equivalence_group=(
                order.instrument_equivalence_group
                or automatic_group_by_isin.get(isin)
            ),
        ))

    for conflict in conflicts:
        logger.warning("Instrument resolution conflict: %s", conflict)
        dq.warning("instrument_resolution", conflict, context="effective_orders")
    profile_statuses = {
        status: sum(
            1
            for profile in profiles.values()
            if profile and profile.get("status") == status
        )
        for status in ("VERIFIED", "CONFLICTING", "UNRESOLVED")
    }
    report = {
        "network_allowed": runtime.allows_live_transport(),
        "profiles_requested": len(isins),
        "profile_sources": {
            source: sum(1 for value in profile_sources.values() if value == source)
            for source in sorted(set(profile_sources.values()))
        },
        "profile_statuses": profile_statuses,
        "resolved_kind_isins": sorted(
            isin for isin, kind in kind_by_isin.items() if kind is not None
        ),
        "equivalence_groups": accepted_groups,
        "conflicts": sorted(conflicts),
    }
    logger.info(
        "Instrument profiles: %d requested, %d kinds resolved, %d automatic equivalence group(s)",
        len(isins),
        len(report["resolved_kind_isins"]),
        len(accepted_groups),
    )
    return resolved_orders, report


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _infer_instrument_type(info: dict, holding: Holding) -> str:
    """Resolve a display kind from exact declared evidence only."""
    from tarzan.instruments.registry import TypeEvidenceGateway

    resolution = TypeEvidenceGateway().resolve(
        info.get("quoteType"),
        info.get("instrumentType"),
        info.get("securityType"),
        holding.security_type,
        holding.instrument_type,
        *holding.instrument_kind_evidence,
    )
    if resolution.kind is None:
        return "Other"
    return {
        "STOCK": "Stock",
        "ETF": "ETF",
        "BOND": "Bond",
        "CASH": "Cash",
    }[resolution.kind.value]


def _derive_security_type(holding: Holding) -> str:
    """Return the exact registered kind or ``UNKNOWN``; never infer from category."""
    from tarzan.instruments.registry import TypeEvidenceGateway

    resolution = TypeEvidenceGateway().resolve(
        holding.security_type,
        holding.instrument_type,
        *holding.instrument_kind_evidence,
    )
    return resolution.kind.value if resolution.kind is not None else "UNKNOWN"


def _resolve_instrument_kind(
    info: dict,
    holding: Holding,
    *,
    figi_market_sector: Optional[str] = None,
    figi_security_type: Optional[str] = None,
):
    """Resolve exact mechanics from declared provider/input kind evidence."""
    from tarzan.instruments.registry import TypeEvidenceGateway

    return TypeEvidenceGateway().resolve(
        info.get("quoteType"),
        info.get("instrumentType"),
        info.get("securityType"),
        info.get("typeDisp"),
        figi_market_sector,
        figi_security_type,
        holding.security_type,
        holding.instrument_type,
        *holding.instrument_kind_evidence,
    ).kind


def classify_asset_class(info: dict, ticker: str, holding: Holding):
    """Resolve tracked category from exact, provenance-bearing evidence only.

    Ticker/name/price/quantity and substring matches are intentionally absent.
    Stock, Bond, and Cash kinds have intrinsic category declarations; an ETF
    requires an explicit category assertion or curated taxonomy row.
    """
    from tarzan.instruments import CapabilityResult, SupportState
    from tarzan.instruments.registry import (
        InstrumentKind,
        TrackedCategoryEvidenceGateway,
        TypeEvidenceGateway,
        TypeResolutionState,
    )
    from tarzan.runtime.ledger import Availability

    declared = info.get("asset_class")
    if declared is not None:
        try:
            return declared if isinstance(declared, AssetClass) else AssetClass(str(declared))
        except ValueError:
            pass

    quote_type = str(info.get("quoteType") or "").strip().upper()
    provider_category = quote_type if quote_type in {"CRYPTOCURRENCY", "ETN"} else None
    category = TrackedCategoryEvidenceGateway().resolve(
        holding.asset_type,
        info.get("assetClass"),
        info.get("category"),
        provider_category,
    )
    kind = TypeEvidenceGateway().resolve(
        info.get("quoteType"),
        info.get("instrumentType"),
        info.get("securityType"),
        holding.security_type,
        holding.instrument_type,
    )

    if category.state is TypeResolutionState.RESOLVED:
        return AssetClass(category.category)
    if category.state is not TypeResolutionState.AMBIGUOUS:
        intrinsic = {
            InstrumentKind.STOCK: AssetClass.EQUITIES,
            InstrumentKind.BOND: AssetClass.FIXED_INCOME,
            InstrumentKind.CASH: AssetClass.CASH_EQUIVALENTS,
        }
        if kind.kind in intrinsic:
            return intrinsic[kind.kind]

    evidence = tuple(dict.fromkeys(category.evidence + kind.evidence))
    return CapabilityResult(
        support=SupportState.UNSUPPORTED,
        availability=Availability.UNAVAILABLE,
        value=None,
        provenance=evidence,
        analytical_impact=(
            "asset classification and dependent capabilities are unavailable "
            "without one exact tracked-category declaration"
        ),
        publication_impact="DEGRADE",
    )


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
    (thread-safe). The live resolution (instrument_taxonomy.csv + justETF + yfinance)
    runs first so it is always authoritative and fresh; the immutable disk
    cache (keyed by the stable ISIN when available) is consulted only as a
    fallback, so a justETF/scrape outage does not degrade the geography
    section to "Not Available" — while never shadowing an instrument_taxonomy.csv edit.
    """
    cached = _get_geo_breakdown_cached(ticker)
    if cached is not None:
        with _net_lock:
            source = _geo_source_memo.get(ticker, "memory")
        return cached, source

    geo_key = isin or ticker

    # instrument_taxonomy.csv (and the justETF index-name bridge) are authoritative and
    # cheap enough to run fresh on every call, so an edit to instrument_taxonomy.csv
    # takes effect immediately instead of being shadowed for the cache TTL.
    # The on-disk cache is consulted ONLY as a fallback below, when the live
    # resolution is unavailable.
    result = _scrape_geo_breakdown(ticker, isin)
    if result:
        breakdown, source = result
        _store_geo_breakdown(ticker, breakdown, source)
        price_cache.store_geo(
            geo_key, {g.value: v for g, v in breakdown.items()}, source
        )
        return breakdown, source

    # Fallback: a previously-resolved breakdown from the immutable disk
    # cache. This keeps the geography section resilient to a justETF /
    # scrape outage (which would otherwise degrade it to "Not Available")
    # without ever shadowing a fresh instrument_taxonomy.csv edit.
    disk = price_cache.load_geo(geo_key)
    if disk:
        breakdown = _geo_from_cache(disk["breakdown"])
        if breakdown:
            source = disk.get("source", "cache")
            _store_geo_breakdown(ticker, breakdown, source)
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

def _record_ticker_decision(
    holding: Holding,
    original_ticker: str,
    data: dict,
) -> str:
    """Persist one canonical market-symbol decision on ``holding``.

    ``ticker_requested`` preserves user intent while ``holding.ticker`` remains
    the compatibility field consumed by every existing history/current/intraday
    section. The returned symbol is the exact provider feed selected here.
    """
    clean_isin = (holding.isin or "").replace("-", "").strip().upper()
    requested = (holding.ticker_requested or "").strip()
    if not requested and original_ticker and original_ticker.strip().upper() != clean_isin:
        requested = original_ticker.strip()
    selected = str(data.get(_TICKER_SYMBOL_KEY) or holding.ticker or "").strip()
    if clean_isin and selected.upper() == clean_isin:
        selected = ""

    base_method = str(data.get(_TICKER_METHOD_KEY) or "UNRESOLVED")
    base_reason = str(
        data.get(_TICKER_REASON_KEY)
        or "No market ticker with usable provider evidence was resolved."
    )
    method = base_method
    reason = base_reason
    has_market_evidence = _ticker_data_has_market_evidence(data)
    if not selected:
        method = "UNRESOLVED"
        reason = (
            "No canonical ticker with usable market evidence was resolved. "
            f"{base_reason}"
        )
    elif requested:
        requested_upper = requested.upper()
        selected_upper = selected.upper()
        requested_bare = requested_upper.split(".", 1)[0]
        selected_bare = selected_upper.split(".", 1)[0]
        if requested_upper == selected_upper and "." in requested:
            if has_market_evidence:
                reason = (
                    f"User pre-empted the ticker: {requested} was supplied "
                    "explicitly and verified as the selected market listing. "
                    f"Selection evidence: {base_reason}"
                )
            else:
                reason = (
                    f"User pre-empted the ticker choice: {requested} was "
                    "supplied explicitly and retained as the canonical request, "
                    "but no usable market data verified the listing."
                )
        elif (
            requested_upper != selected_upper
            and requested_bare == selected_bare
            and "." not in requested
            and "." in selected
        ):
            reason = (
                f"User supplied the bare ticker {requested}; Tarzan selected "
                f"the canonical listing {selected}. Selection evidence: "
                f"{base_reason}"
            )
        elif requested_upper == selected_upper:
            reason = (
                f"User supplied {requested}, which remained the canonical "
                f"ticker. Selection evidence: {base_reason}"
            )
        else:
            reason = (
                f"User requested {requested}; Tarzan selected {selected} instead. "
                f"Selection evidence: {base_reason}"
            )

    holding.ticker_requested = requested or None
    holding.ticker_selection_method = method
    holding.ticker_selection_reason = reason
    if selected:
        holding.ticker = selected
    return selected


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
                # Write the resolved market symbol back onto the holding. Orders
                # from a broker export often carry no ticker (Fineco is ISIN-
                # only), so without this the holding keeps the ISIN as its
                # "ticker" and every downstream ticker lookup — the curated
                # taxonomy (asset class + role), intraday sparkline resolution,
                # display pin — silently misses. Learn the ISIN↔ticker xref too
                # so the mapping is reusable in both directions.
                if resolved_ticker and resolved_ticker != clean_isin:
                    holding.ticker = resolved_ticker
                    price_cache.store_ticker_isin(resolved_ticker, clean_isin)

        if data is None:
            from tarzan import runtime

            if is_valid_isin and not runtime.allows_live_transport():
                # _resolve_isin already exhausted every symbol corroborated for
                # this ISIN. Preserve any taxonomy-selected canonical listing
                # for reporting, but do not claim that it supplied data.
                _, known_ticker = cfg.resolve_taxonomy_identity(clean_isin, ticker)
                if known_ticker and known_ticker != clean_isin:
                    data = _annotate_ticker_data(
                        {"info": {}, "history": pd.DataFrame()},
                        known_ticker,
                        "INSTRUMENT_TAXONOMY_NO_DATA",
                        "Instrument taxonomy selected this canonical listing, but no boundary-visible market data was available.",
                    )
                else:
                    data = _annotate_ticker_data(
                        {"info": {}, "history": pd.DataFrame()},
                        "",
                        "UNRESOLVED",
                        "No canonical ticker with boundary-visible market data was resolved for this ISIN.",
                    )
            else:
                if _is_expandable_bare_ticker(ticker) and holding.name:
                    data = _fetch_ticker_data(
                        ticker,
                        expected_name=holding.name,
                        expected_currency=holding.currency or "",
                    )
                else:
                    data = _fetch_ticker_data(ticker)
            info = data.get("info", {})
            history = data.get("history", pd.DataFrame())

        selected_ticker = _record_ticker_decision(holding, ticker, data)
        if selected_ticker and is_valid_isin:
            price_cache.store_ticker_isin(selected_ticker, clean_isin)
        if selected_ticker and (history is not None and not history.empty or info):
            data_source = f"yfinance:{selected_ticker}"

        holding.name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("name")
            or holding.name
            or selected_ticker
            or ticker
        )
        holding.instrument_type = _infer_instrument_type(info, holding)
        holding.fetch_timestamp = dt.now(timezone.utc)

        # Price history and current price. Selection records the actual close
        # observation and whether the chosen value came from non-primary data.
        # A EUR venue is EUR regardless of a flaky info.currency (see
        # venue_currency), so an EUR-native listing is never FX-converted.
        currency = (venue_currency(selected_ticker)
                    or info.get("currency", holding.currency or "EUR"))
        _set_price_data(holding, history, info, currency)
        if selected_ticker:
            if holding.price_history is not None and len(holding.price_history) > 0:
                holding.history_ticker = selected_ticker
            if holding.current_price is not None:
                holding.current_ticker = selected_ticker
            holding.intraday_ticker_reason = (
                "Not requested for a historical-only instrument."
                if holding.is_historical_only
                else "The canonical ticker is the default for every intraday consumer; no intraday feed was selected yet."
            )

        # Value and gain use one exact InstrumentKind. Exposure category,
        # display text, names, prices, and quantities never select the bond
        # per-100 convention. Conflicting or absent kind evidence leaves the
        # current valuation on its labeled EUR anchor for the completeness
        # evaluator to reject rather than guessing unit mechanics.
        figi_sector, figi_sec_type = (None, None)
        if is_valid_isin:
            figi_sector, figi_sec_type = _openfigi_bond_signals(clean_isin)
        instrument_kind = _resolve_instrument_kind(
            info,
            holding,
            figi_market_sector=figi_sector,
            figi_security_type=figi_sec_type,
        )
        if instrument_kind is not None:
            holding.security_type = instrument_kind.value

        cp = holding.current_price
        has_price = cp is not None and cp == cp and cp > 0
        if has_price and instrument_kind is not None:
            from tarzan.data.bond_fetcher import value_position
            from tarzan.instruments.registry import InstrumentKind

            holding.current_value = value_position(
                holding.quantity,
                holding.current_price,
                instrument_kind=instrument_kind,
            )
            if instrument_kind is InstrumentKind.BOND:
                # Store all downstream history/current prices as EUR-per-unit;
                # the clean-price /100 conversion is applied exactly once.
                if holding.price_history is not None and len(holding.price_history) > 0:
                    holding.price_history = holding.price_history / 100.0
                holding.current_price = holding.current_price / 100.0
            if not holding.price_is_fallback and not holding.data_source:
                holding.data_source = data_source
            elif holding.price_is_fallback and not holding.data_source:
                holding.data_source = f"{data_source} (fallback price)"
        elif has_price:
            if holding.is_historical_only:
                # Closed-position carriers need the series for return history,
                # but never participate in current valuation. Missing mechanics
                # therefore is not a current instrument-capability failure.
                holding.current_value = 0.0
                holding.data_source = (
                    f"{data_source} (historical-only; instrument kind unavailable)"
                )
            else:
                holding.current_value = holding.market_value_eur or 0.0
                holding.data_source = "last-known (instrument kind unavailable)"
                dq.error(
                    "instrument_capability",
                    "Current price was not valued because exact instrument-kind "
                    "evidence is missing or conflicting.",
                    context=holding.isin or holding.ticker,
                )
        else:
            from tarzan.instruments.registry import InstrumentKind

            if (
                instrument_kind is InstrumentKind.BOND
                and not holding.is_historical_only
            ):
                _try_terrapin_fallback(holding, instrument_kind)
            else:
                holding.current_value = holding.market_value_eur or 0.0
                holding.data_source = (
                    "historical-only (no configured market history)"
                    if holding.is_historical_only
                    else "input_csv (no market data)"
                )

        # Safety net: never let a holding carry a None/NaN current_value into
        # the portfolio total. A single NaN propagates through the sum and
        # collapses the whole portfolio (the ~€13k-instead-of-€221k symptom
        # seen when yfinance throttles the live quote). Seed from the
        # last-known EUR value (CSV/order anchor) so the total stays sane.
        cv = holding.current_value
        if cv is None or cv != cv:  # None or NaN
            holding.current_value = holding.market_value_eur or 0.0
            if not holding.data_source or holding.data_source == "yfinance":
                holding.data_source = "last-known (no live price)"

        if holding.cost_basis_eur > 0 and holding.current_value > 0:
            holding.gain_pct = (
                (holding.current_value - holding.cost_basis_eur)
                / holding.cost_basis_eur * 100
            )

        # Metadata. yield_pct and ter are stored as FRACTIONS (0.021 = 2.1%);
        # the metrics layer multiplies by 100 to display a percentage. yfinance
        # is inconsistent across these fields — 'yield'/'trailingAnnualDividendYield'
        # /'annualReportExpenseRatio' are fractions (0.021) but 'dividendYield'
        # and 'fiveYearAvgDividendYield' are percents (2.4) in modern versions —
        # so a raw fallback chain mixes units and inflates weighted_yield ~100x.
        # _as_fraction normalizes any percent-scaled value (>= 1.0, impossible
        # for a real fraction yield/TER) back to a fraction.
        holding.ter = _as_fraction(
            info.get("annualReportExpenseRatio") or info.get("expenseRatio")
        )
        # Curated TER from instrument_taxonomy.csv (a FRACTION) overrides the
        # yfinance value when present — yfinance rarely carries a correct TER
        # for EU UCITS ETFs, so the hand-verified taxonomy cell is authoritative.
        _tax_ter = cfg.ter_for(holding.isin, holding.ticker)
        if _tax_ter is not None:
            holding.ter = _tax_ter
        holding.yield_pct = _as_fraction(
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


def _clip_to_as_of(prices: pd.Series) -> pd.Series:
    """Drop price observations strictly AFTER the pinned ``as_of`` date, so an
    as-of / deterministic run values the snapshot at the as-of price rather than
    the latest fetched close. Point-in-time correctness: without this, the
    holdings snapshot (current_price → current_value → total_value) would peek at
    prices after the reporting date. No-op in live mode (as_of is None).

    Guarded so a normal live run is completely unaffected."""
    try:
        from tarzan import runtime
        as_of = runtime.as_of()
    except Exception:  # noqa: BLE001 — never let this break enrichment
        as_of = None
    if as_of is None or prices is None or len(prices) == 0:
        return prices
    try:
        cutoff = pd.Timestamp(as_of)
        idx = prices.index
        if getattr(idx, "tz", None) is not None:
            cutoff = cutoff.tz_localize(idx.tz)
        return prices[prices.index <= cutoff]
    except Exception:  # noqa: BLE001 — malformed index → leave as-is
        return prices


def _history_observation_time(
    history: pd.DataFrame,
    index_value: object,
) -> Optional[dt]:
    """Convert a genuine datetime history index value to a Python datetime."""
    if not isinstance(history.index, pd.DatetimeIndex):
        return None
    try:
        return pd.Timestamp(index_value).to_pydatetime()
    except (TypeError, ValueError):
        return None


def _info_observation_time(value: object) -> Optional[dt]:
    """Parse a provider quote timestamp when one accompanies a live quote."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            timestamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp.to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None


def _composite_observation_time(
    price_time: Optional[dt],
    fx_time: Optional[dt],
) -> Optional[dt]:
    """Return the older observation governing a converted price."""
    if price_time is None or fx_time is None:
        return None
    price_timestamp = pd.Timestamp(price_time)
    fx_timestamp = pd.Timestamp(fx_time)
    if price_timestamp.tzinfo is not None:
        price_timestamp = price_timestamp.tz_convert("UTC").tz_localize(None)
    if fx_timestamp.tzinfo is not None:
        fx_timestamp = fx_timestamp.tz_convert("UTC").tz_localize(None)
    return price_time if price_timestamp <= fx_timestamp else fx_time


def _set_price_data(
    holding: Holding, history: pd.DataFrame, info: dict, currency: str
) -> None:
    """Select the best valid EUR price and retain observation provenance.

    In live mode, a timestamped ``regularMarketPrice`` may supersede history
    when it is at least as recent as the selected close. A history close keeps
    the origin of its exact row, so cached evidence remains fallback. Undated
    quotes and ``previousClose`` are fallback-only. Pinned as-of runs use only
    history clipped to the reporting date and never consume a current quote.
    """
    quote_currency = currency
    current_price: Optional[float] = None
    observation_time: Optional[dt] = None
    is_fallback = False
    selected_source: Optional[str] = None
    holding.price_observation_timestamp = None
    holding.price_is_fallback = False

    if history is not None and not history.empty and "Close" in history:
        native = history["Close"].copy()
        native, history_currency = _normalize_minor_currency(native, currency)
        # Clean FIRST, convert LAST, so the native and EUR series are the same
        # series in two currencies rather than two independently cleaned ones.
        # Every step here is either pointwise (dropna, >0) or date-based
        # (_clip_to_as_of), so none of them depends on the unit.
        native = native.dropna()
        native = native[native > 0]
        native = _clip_to_as_of(native)
        fx_evidence: dict[str, dict[str, object]] = {}
        if history_currency != "EUR":
            prices = convert_to_eur(native, history_currency)
            fx_evidence = prices.attrs.get(_FX_EVIDENCE_ATTR, {})
            prices = prices.dropna()
            prices = prices[prices > 0]
        else:
            prices = native
        if len(prices) > 0:
            # Name the series after the listing it came from: that is how every
            # window resolves which exchange calendar governs it (see
            # stats._series_ticker). The benchmark path already does the same.
            prices.name = holding.ticker or prices.name
            holding.price_history = prices
            # The SAME series before the FX step, kept so the per-instrument
            # return columns can be quoted in the currency the instrument
            # actually trades in. A return is a ratio, and FX does not divide out
            # of it: RSSY's five sessions to 28 Aug 2026 read −1.28% on its own
            # Nasdaq tape and −2.18% once each end is converted at its own day's
            # rate. Both are true statements about different things; the tables
            # compare instruments, so they use the instrument's own tape.
            native = native.reindex(prices.index).dropna()
            native.name = prices.name
            holding.price_history_native = native
            holding.price_currency = history_currency
            selected_index = prices.index[-1]
            selected_key = _history_timestamp_key(selected_index)
            current_price = float(prices.iloc[-1])
            observation_time = _history_observation_time(history, selected_index)
            origins = history.attrs.get(_HISTORY_ORIGINS_ATTR, {})
            selected_origin = origins.get(selected_key)
            if selected_origin == _HISTORY_ORIGIN_CACHE:
                # In LIVE mode a retained cache row is a fallback after a failed
                # refresh. In pinned modes the exact as-of-clipped provider row
                # is the only admissible market evidence; freshness, not its
                # storage location, decides whether it is stale.
                from tarzan import runtime

                is_fallback = runtime.allows_live_transport()
                symbol = history.attrs.get(_HISTORY_SYMBOL_ATTR)
                selected_source = (
                    f"price_cache:{symbol}" if symbol else "price_cache"
                )
            selected_fx = fx_evidence.get(selected_key)
            if selected_fx is not None:
                fx_time = selected_fx.get("observation_time")
                observation_time = _composite_observation_time(
                    observation_time,
                    fx_time if isinstance(fx_time, dt) else None,
                )
                if selected_fx.get("is_fallback") or observation_time is None:
                    is_fallback = True
                    selected_source = str(
                        selected_fx.get("source") or "FX (fallback evidence)"
                    )

    def _valid_quote(value: object) -> Optional[float]:
        try:
            numeric = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        if numeric is None or numeric != numeric or numeric <= 0:
            return None
        return numeric

    def _quote_to_eur(
        value: float,
    ) -> tuple[Optional[float], Optional[dict[str, object]]]:
        scalar = pd.Series([value])
        scalar, normalized_currency = _normalize_minor_currency(
            scalar,
            quote_currency,
        )
        converted = float(scalar.iloc[0])
        if normalized_currency == "EUR":
            return converted, None
        fx = _usable_fx_series(_get_fx_series(normalized_currency))
        if fx.empty:
            return None, None
        selected_index = fx.index[-1]
        origins = fx.attrs.get(_HISTORY_ORIGINS_ATTR, {})
        origin = origins.get(
            _history_timestamp_key(selected_index),
            _HISTORY_ORIGIN_PRIMARY,
        )
        fx_time = (
            pd.Timestamp(selected_index).to_pydatetime()
            if isinstance(fx.index, pd.DatetimeIndex)
            else None
        )
        return converted * float(fx.iloc[-1]), {
            "observation_time": fx_time,
            "is_fallback": origin == _HISTORY_ORIGIN_CACHE or fx_time is None,
            "source": (
                "price_cache:FX"
                if origin == _HISTORY_ORIGIN_CACHE
                else "FX (undated)"
                if fx_time is None
                else None
            ),
        }

    def _not_older(candidate: dt, selected: Optional[dt]) -> bool:
        if selected is None:
            return True
        try:
            candidate_ts = pd.Timestamp(candidate)
            selected_ts = pd.Timestamp(selected)
            if candidate_ts.tzinfo is not None:
                candidate_ts = candidate_ts.tz_convert("UTC").tz_localize(None)
            if selected_ts.tzinfo is not None:
                selected_ts = selected_ts.tz_convert("UTC").tz_localize(None)
            return candidate_ts >= selected_ts
        except (TypeError, ValueError):
            return True

    try:
        from tarzan import runtime

        pinned_as_of = runtime.as_of() is not None
    except Exception:  # noqa: BLE001 — provenance must not break enrichment
        pinned_as_of = False

    if not pinned_as_of:
        regular = _valid_quote(info.get("regularMarketPrice"))
        regular_time = _info_observation_time(info.get("regularMarketTime"))
        if regular is not None:
            regular_eur, regular_fx = _quote_to_eur(regular)
            candidate_time = regular_time
            candidate_fallback = regular_time is None
            candidate_source = (
                "yfinance:regularMarketPrice (undated)"
                if regular_time is None
                else None
            )
            if regular_fx is not None:
                fx_time = regular_fx.get("observation_time")
                candidate_time = _composite_observation_time(
                    regular_time,
                    fx_time if isinstance(fx_time, dt) else None,
                )
                if regular_fx.get("is_fallback"):
                    candidate_fallback = True
                    candidate_source = str(
                        regular_fx.get("source") or "FX (fallback evidence)"
                    )
                elif candidate_time is None:
                    candidate_fallback = True
            if (
                regular_eur is not None
                and candidate_time is not None
                and _not_older(candidate_time, observation_time)
            ):
                current_price = regular_eur
                observation_time = candidate_time
                is_fallback = candidate_fallback
                selected_source = candidate_source
            elif regular_eur is not None and current_price is None:
                current_price = regular_eur
                observation_time = candidate_time
                is_fallback = True
                selected_source = candidate_source or (
                    "yfinance:regularMarketPrice (undated)"
                )

        if current_price is None:
            previous_close = _valid_quote(info.get("previousClose"))
            if previous_close is not None:
                previous_eur, previous_fx = _quote_to_eur(previous_close)
                if previous_eur is not None:
                    current_price = previous_eur
                    observation_time = None
                    is_fallback = True
                    selected_source = (
                        str(previous_fx.get("source"))
                        if previous_fx is not None
                        and previous_fx.get("is_fallback")
                        and previous_fx.get("source")
                        else "yfinance:previousClose"
                    )

    holding.current_price = current_price
    holding.price_observation_timestamp = observation_time
    holding.price_is_fallback = is_fallback
    if selected_source is not None:
        holding.data_source = selected_source


def _try_terrapin_fallback(holding: Holding, instrument_kind) -> None:
    """Try Borsa Italiana only for an explicitly resolved bond kind.

    Borsa Italiana quotes the clean price per 100 of nominal in the bond's
    *native* currency. We convert that to an EUR-per-unit ``current_price``
    (FX-converted, then /100), matching the yfinance bond branch, so every
    downstream consumer reads EUR directly. This is currency-general: a
    USD Treasury, a ZAR EIB note, a GBP gilt are all handled the same way
    via the shared FX machinery — no per-currency special-casing.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        holding.current_value = holding.market_value_eur
        holding.data_source = "input_csv (live transport disabled)"
        return

    try:
        from tarzan.data.bond_fetcher import fetch_bond_price, value_position
        from tarzan.instruments.registry import InstrumentKind

        if instrument_kind is not InstrumentKind.BOND:
            raise ValueError("Borsa fallback requires explicit BOND kind")

        isin = holding.isin
        if not isin or len(isin.replace("-", "")) != 12:
            holding.current_value = holding.market_value_eur
            holding.data_source = "input_csv (no market data)"
            return

        result = fetch_bond_price(isin)
        if result:
            # This provider is the explicit secondary rung after yfinance.
            # It remains usable when policy allows fallback, but never becomes
            # primary merely because the scrape happened during this run.
            holding.price_is_fallback = True
            holding.price_observation_timestamp = None
            # Borsa quote: clean price per 100 nominal, in the native currency.
            price_native = result["price"]
            currency = holding.currency or "EUR"

            # Convert the clean price to EUR per 100 nominal. The FX series
            # is EUR-per-native-unit, so a native price is multiplied by it.
            price_eur_per_100 = price_native
            fx_unavailable = False
            if currency != "EUR":
                fx = _usable_fx_series(_get_fx_series(currency))
                if not fx.empty:
                    price_eur_per_100 = price_native * float(fx.iloc[-1])
                else:
                    # No FX rate: a native clean price cannot be converted to
                    # EUR. Do not book it 1:1 — fall back to the CSV/order EUR
                    # anchor below instead of a mislabeled native value.
                    fx_unavailable = True
                    logger.warning(
                        "Bond %s: no FX for %s; using CSV EUR anchor (native price not converted)",
                        isin, currency,
                    )

            if fx_unavailable:
                holding.current_value = holding.market_value_eur
                holding.data_source = "input_csv (FX unavailable)"
                dq.warning(
                    "enricher",
                    f"bond {isin}: FX for {currency} unavailable — valued from "
                    "the CSV/order EUR anchor, not a live quote",
                    context=isin,
                )
                return

            value = value_position(
                holding.quantity,
                price_eur_per_100,
                instrument_kind=instrument_kind,
            )

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

def _apply_taxonomy_override(holding: Holding) -> bool:
    """Pin category, role, and exact mechanics evidence from the taxonomy.

    Looks up by ISIN first, then bare ticker (suffix stripped). On a match,
    sets ``asset_class``/``role`` and contributes the curated ``kind`` without
    erasing other assertions. Conflicts therefore remain ambiguous in the type
    gateway instead of being overwritten by taxonomy precedence.
    """
    lut = cfg.instrument_taxonomy()
    if not lut:
        return False
    keys = []
    if holding.isin:
        keys.append(normalize_isin(holding.isin))
    if holding.ticker:
        keys.append(normalize_ticker(holding.ticker))
    # Bidirectional bridge: a taxonomy row may be keyed by only one of
    # (ISIN, ticker) while the holding knows only the other — e.g. a Fineco
    # ISIN-only order vs a ticker-only UEQC row. Resolve across the learned
    # ISIN↔ticker xref so either identifier reaches the row. (The xref is
    # populated as instruments resolve; ticker→ISIN also lets an ISIN-keyed
    # row match a ticker-only holding.)
    if holding.isin:
        xref_ticker = price_cache.load_ticker_isin_reverse(normalize_isin(holding.isin))
        if xref_ticker:
            keys.append(normalize_ticker(xref_ticker))
    if holding.ticker:
        xref_isin = price_cache.load_ticker_isin(holding.ticker)
        if xref_isin:
            keys.append(normalize_isin(xref_isin))
    for k in keys:
        hit = lut.get(k)
        if not hit:
            continue
        ac_str, role = hit
        try:
            holding.asset_class = AssetClass(ac_str)
        except ValueError:
            logger.warning(
                "Curated asset_class '%s' for %s is not a valid AssetClass; ignoring override",
                ac_str, k,
            )
            return False
        holding.role = role
        curated_kind = cfg.kind_for(holding.isin, holding.ticker)
        if (
            curated_kind
            and curated_kind not in holding.instrument_kind_evidence
        ):
            holding.instrument_kind_evidence = (
                *holding.instrument_kind_evidence,
                curated_kind,
            )
        # Resilient display name: fall back to the curated taxonomy ``name``
        # when enrichment produced none (yfinance unreachable / rate-limited)
        # or left the bare ticker as the name — so a curated instrument always
        # shows a real name, even offline and even when its taxonomy row is
        # keyed only by ticker (no ISIN).
        bare = normalize_ticker(holding.ticker)
        if not holding.name or holding.name.strip().upper() in (
                (holding.ticker or "").strip().upper(), bare):
            tax_name = cfg.name_for(holding.isin, holding.ticker)
            if tax_name:
                holding.name = tax_name
        return True
    return False


def _enrich_and_classify(holding: Holding) -> Holding:
    """Enrich, then resolve category/kind from explicit evidence only.

    Resolution order is curated taxonomy, exact provider/input declarations,
    exact OpenFIGI security-type mappings, then typed unavailability. Names,
    tickers, prices, and quantities never select financial behavior.
    """
    # Curated kind is causal local evidence and must be available before
    # current-value mechanics run. Re-apply after symbol resolution so a row
    # keyed only by the resolved ticker can still provide category/display data.
    curated = _apply_taxonomy_override(holding)
    holding, info = _enrich_single(holding)
    curated = _apply_taxonomy_override(holding) or curated

    # Closed-position carriers exist only to supply exact mechanics and causal
    # price history. Current category, geography, sector, and rebalancing
    # capabilities are neither consumed nor actionable for them.
    if holding.is_historical_only:
        return holding

    if not curated and holding.asset_class is None:
        classification = classify_asset_class(info, holding.ticker, holding)
        if isinstance(classification, AssetClass):
            holding.asset_class = classification
        else:
            # OpenFIGI may supply an exact, configured security-type mapping.
            # It is not allowed to classify from instrument names.
            _apply_openfigi_fallback(holding)
            if holding.asset_class is None:
                dq.error(
                    "instrument_capability",
                    "Explicit instrument kind/category is unknown or ambiguous; asset "
                    "classification, sector, and rebalancing capabilities are unavailable "
                    "and no default adapter was selected.",
                    context=holding.isin or holding.ticker,
                )

        logger.debug(
            "No curated taxonomy for %s / %s → explicit classification %s "
            "(role unset). Add a taxonomy row to pin it.",
            holding.isin, holding.ticker, holding.asset_class,
        )

    if holding.security_type is None:
        holding.security_type = _derive_security_type(holding)

    if holding.geography is None:
        holding.geography = classify_geography(info, holding.ticker, holding)

    _apply_geo_breakdown(holding)
    _apply_class_breakdown(holding)

    # TER gap-fill: curated-taxonomy / yfinance TER (set in _enrich_single) is
    # authoritative; only when it is absent do we fall back to the justETF
    # profile fee, then a per-asset-class default. Same resolver the backtest
    # reads, so the live "average TER" and the backtest charge identical fees
    # instead of the live path counting an unknown fee as 0%. justETF is
    # network-gated (cache-only + None in pinned runs) via allows_live_transport.
    if holding.ter is None:
        from tarzan.data.geo_resolver import resolve_ter
        ac_val = holding.asset_class.value if holding.asset_class else None
        holding.ter = resolve_ter(holding.isin, ac_val)

    return holding


def _apply_class_breakdown(holding: Holding) -> None:
    """Set ``holding.class_breakdown`` (notional asset-class exposure).

    Uses the explicit exp_* taxonomy override when present, otherwise derives
    ``{asset_class: 100}``. Warns when a fund whose role/name signals leverage
    or a multi-asset structure lacks an explicit override, so a future
    capital-efficient instrument is flagged for an exp_* row rather than
    silently counted as a single class."""
    ac_val = holding.asset_class.value if holding.asset_class else None
    holding.class_breakdown = {
        AssetClass(k): v
        for k, v in cfg.class_breakdown_for(holding.isin, holding.ticker, ac_val).items()
        if _is_valid_asset_class(k)
    }
    # Nudge for likely-notional instruments missing an explicit override.
    lut = cfg.class_exposure_lookup()
    keyed = (normalize_isin(holding.isin) in lut or
             normalize_ticker(holding.ticker) in lut)
    if not keyed:
        hint_src = f"{holding.role or ''} {holding.name or ''}".lower()
        if (any(h in (holding.role or "").lower() for h in cfg._NOTIONAL_ROLE_HINTS)
                or any(h in hint_src for h in cfg._NOTIONAL_NAME_HINTS)):
            logger.warning(
                "Instrument %s / %s looks capital-efficient/leveraged but has "
                "no exp_* exposure in instrument_taxonomy.csv — it will be "
                "counted as 100%% %s. Add exp_* columns to capture its notional "
                "split.", holding.isin, holding.ticker, ac_val,
            )


def _is_valid_asset_class(value: str) -> bool:
    try:
        AssetClass(value)
        return True
    except ValueError:
        return False


def _apply_openfigi_fallback(holding: Holding) -> None:
    """Apply OpenFIGI classification as a last resort."""
    try:
        figi_info = openfigi_classify(holding.isin)
        if figi_info.get("asset_class") and holding.asset_class is None:
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


def _enrich_deferring(holding: Holding, diagnostics: list) -> Holding:
    """Worker entry point: buffer this holding's diagnostics for ordered replay.

    ``_enrich_and_classify`` is looked up globally rather than captured, so the
    order tests that monkeypatch it keep working.
    """
    dq.defer_into(diagnostics)
    return _enrich_and_classify(holding)


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

    # The provider memo is run-scoped (see module docstring): one network call
    # per instrument identity per run. An active RunSession means the run has
    # already reset the memo once (orchestrator.py, before instrument
    # resolution), so every enrichment pass in that run — holdings, historical
    # ISINs for returns, backtest candidates — REUSES the resolved symbols,
    # price history and OpenFIGI profiles instead of wiping and re-fetching
    # them. Only standalone callers (tools, tests) with no session start fresh.
    from tarzan.runtime.session import current_session
    if current_session() is None:
        reset_run_caches()

    logger.info("Enriching %d holdings (max %d workers)...", len(holdings), MAX_WORKERS)

    # Results are placed back at their INPUT index, not appended in completion
    # order. Appending as futures complete made this list's order depend on
    # thread scheduling, and that order is the rebalancer's coordinate order:
    # its iterated local search accepts an improvement at 1e-9, so two runs of
    # the same deterministic analysis converged on different local optima and
    # recommended materially different purchases (CL2 and X25E each shifting by
    # thousands of euros) from the same budget. Every other figure matched,
    # because sums and weighted averages do not care about order.
    # Diagnostics are buffered per holding and replayed at the INPUT index, for the
    # same reason the results are: the run ledger records the order it is told about
    # them, and a failure id is a (stage, code, ORDINAL) hash. Two off-taxonomy
    # instruments that complete in either order swapped their failure ids between
    # two REPRODUCIBLE runs of the same book, in 6 of 44 runs.
    slots: list[Optional[Holding]] = [None] * len(holdings)
    diagnostics: list[list] = [[] for _ in holdings]
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Context variables do not propagate to executor threads implicitly.
            # Give each task its own copied context so run mode, effective date,
            # active ledger, and transport policy remain authoritative in workers.
            futures = {
                executor.submit(
                    copy_context().run, _enrich_deferring, h, diagnostics[index]
                ): index
                for index, h in enumerate(holdings)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    slots[index] = future.result()
                except Exception as e:
                    h = holdings[index]
                    logger.error("Enrichment failed for %s: %s", h.ticker, e)
                    diagnostics[index].append(partial(
                        dq.error,
                        "enricher",
                        f"enrichment raised ({e}); holding kept UN-enriched (no live "
                        "price/classification) — it may distort value, allocation "
                        "and returns",
                        context=(h.ticker or h.isin),
                    ))
                    slots[index] = h
    finally:
        # In a finally, not on the happy path: a BaseException escaping the pool
        # (KeyboardInterrupt, or the SIGKILL-adjacent cases) would otherwise
        # discard every diagnostic the workers had already produced — the evidence
        # of what went wrong, lost exactly when it matters.
        for buffered in diagnostics:
            dq.replay(buffered)
    enriched: list[Holding] = [h for h in slots if h is not None]

    # Surface holdings that came through with no usable market price — they
    # fall back to their last-known/CSV EUR value, so their contribution is an
    # anchor, not a live quote. (A NaN/None price was reseeded upstream; this
    # flags the ones still carrying no real current_price.)
    for h in enriched:
        if not h.is_enriched() and not h.is_historical_only:
            dq.warning(
                "enricher",
                "no market price resolved; valued from its last-known/CSV EUR "
                "anchor rather than a live quote",
                context=(h.ticker or h.isin),
            )

    # Compute weights
    total_value = sum(h.current_value for h in enriched if h.current_value)
    if total_value > 0:
        for h in enriched:
            if h.current_value is not None:
                h.weight_pct = (h.current_value / total_value) * 100

    logger.info("Enrichment complete. Total portfolio value: %.2f EUR", total_value)
    return enriched
