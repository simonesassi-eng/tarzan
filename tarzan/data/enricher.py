"""Fetch market data from yfinance, cache results, enrich and classify holdings.

This module handles the Data Enrichment layer:
- Fetching price history and metadata from yfinance
- ISIN resolution via OpenFIGI API
- FX conversion to EUR
- Asset class and geography classification
- Multi-geography breakdown via geo_scraper
- Pickle-based caching with configurable TTL

Architecture note: enrichment is parallelized via ThreadPoolExecutor.
Each holding is enriched independently, with per-holding error isolation
so that a single API failure doesn't block the entire pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt, timedelta
from typing import Any, Optional
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan import config as cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join("input", ".cache")
CACHE_TTL = timedelta(hours=cfg.cache_ttl_hours())

# Configurable backtest period — set via set_portfolio_backtest_period() before enrichment
BACKTEST_PERIOD = "5y"


def set_portfolio_backtest_period(period: str) -> None:
    """Set the yfinance history period for all subsequent fetches."""
    global BACKTEST_PERIOD
    BACKTEST_PERIOD = period


def _cache_path(ticker: str) -> str:
    """Return the pickle cache file path for a given ticker."""
    safe = ticker.replace("^", "_caret_").replace("/", "_slash_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_store(ticker: str, data: Any) -> None:
    """Store data in the pickle cache with a timestamp."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(_cache_path(ticker), "wb") as f:
            pickle.dump({"ts": time.time(), "data": data}, f)
    except Exception as e:
        logger.warning("Failed to write cache for %s: %s", ticker, e)


def cache_load(ticker: str) -> Optional[Any]:
    """Load data from cache if it exists and is within TTL."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        age = time.time() - cached["ts"]
        if age < CACHE_TTL.total_seconds():
            return cached["data"]
        logger.debug("Cache expired for %s (%.0fs old)", ticker, age)
        return None
    except Exception as e:
        logger.warning("Corrupt cache for %s, deleting: %s", ticker, e)
        try:
            os.unlink(path)
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------
_fx_cache: dict[str, pd.Series] = {}


def _get_fx_series(currency: str) -> pd.Series:
    """Get a 5Y daily FX rate series for currency→EUR conversion.

    For EUR, returns an empty series (sentinel for no conversion needed).
    Tries direct pair first, then inverse pair as fallback.
    """
    if currency == "EUR":
        return pd.Series(dtype=float)
    if currency in _fx_cache:
        return _fx_cache[currency]

    series = _fetch_fx_pair(currency)
    _fx_cache[currency] = series
    return series


def _fetch_fx_pair(currency: str) -> pd.Series:
    """Fetch FX pair from yfinance, trying direct and inverse."""
    for pair, invert in [(f"{currency}EUR=X", False), (f"EUR{currency}=X", True)]:
        try:
            hist = yf.Ticker(pair).history(period=BACKTEST_PERIOD, interval="1d")
            if not hist.empty:
                series = 1.0 / hist["Close"] if invert else hist["Close"]
                return series
        except Exception:
            continue
    logger.warning("No FX data for %s, assuming rate=1.0", currency)
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
    """Fetch yfinance data for a ticker, using cache if available.

    Falls back to base ticker (without exchange suffix) if initial fetch fails.

    Returns:
        Dict with keys: info, history.
    """
    cached = cache_load(ticker)
    if cached is not None:
        logger.debug("Cache hit for %s", ticker)
        return cached

    logger.info("Fetching data for %s", ticker)
    # yfinance raises ValueError("Invalid ISIN number: ...") from the
    # Ticker constructor when the symbol looks like an ISIN but fails
    # its check-digit validation (common for BTP / US Treasury / EIB
    # bonds). Left unguarded, that exception aborts the whole
    # enrichment for the holding *before* the Borsa Italiana bond
    # fallback in _enrich_single is ever reached. Catch it here and
    # return an empty result so current_price stays None and the
    # caller proceeds to the bond fallback.
    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        logger.warning("yfinance rejected ticker %s: %s", ticker, e)
        data = {"info": {}, "history": pd.DataFrame()}
        cache_store(ticker, data)
        return data
    try:
        info = t.info or {}
    except Exception:
        info = {}
    try:
        history = t.history(period=BACKTEST_PERIOD, interval="1d")
    except Exception as e:
        logger.warning("History fetch failed for %s: %s", ticker, e)
        history = pd.DataFrame()

    # Fallback: strip exchange suffix
    if history.empty and not info.get("regularMarketPrice") and "." in ticker:
        base_ticker = ticker.split(".")[0]
        logger.debug("Trying base ticker %s (stripped from %s)", base_ticker, ticker)
        try:
            t2 = yf.Ticker(base_ticker)
            info2 = t2.info or {}
            if info2.get("regularMarketPrice") or info2.get("previousClose"):
                history2 = t2.history(period=BACKTEST_PERIOD, interval="1d")
                if not history2.empty:
                    info, history = info2, history2
                    logger.info("Fallback to base ticker %s succeeded", base_ticker)
        except Exception:
            pass

    data = {"info": info, "history": history}
    cache_store(ticker, data)
    return data


# ---------------------------------------------------------------------------
# ISIN resolution
# ---------------------------------------------------------------------------

def _probe_isin_suffixes(isin: str) -> Optional[tuple[dict, str]]:
    """Try fetching data for an ISIN with multiple strategies.

    Strategy order:
    1. OpenFIGI API → real exchange ticker → yfinance
    2. Exchange suffix brute-force (.MI, .SG, etc.)
    3. Raw ISIN as-is

    Returns:
        (data_dict, successful_ticker) or None if all fail.
    """
    clean_isin = isin.replace("-", "")

    # 1. OpenFIGI first
    for candidate in _openfigi_lookup(clean_isin):
        result = _try_fetch_candidate(candidate, isin)
        if result:
            return result

    # 2. Brute-force exchange suffixes
    candidates = [clean_isin] + [f"{clean_isin}{s}" for s in ISIN_EXCHANGE_SUFFIXES]
    for candidate in candidates:
        result = _try_fetch_candidate(candidate, isin)
        if result:
            return result

    return None


def _try_fetch_candidate(candidate: str, isin: str) -> Optional[tuple[dict, str]]:
    """Try fetching a single candidate ticker, returning data if successful."""
    try:
        t = yf.Ticker(candidate)
        info = t.info or {}
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if price and price > 0:
            history = t.history(period=BACKTEST_PERIOD, interval="1d")
            data = {"info": info, "history": history}
            cache_store(candidate, data)
            logger.info("Probe hit: %s → %s (price=%.2f)", isin, candidate, price)
            return data, candidate
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# OpenFIGI API
# ---------------------------------------------------------------------------
_FIGI_EXCHANGE_MAP = cfg.figi_exchange_map()
_FIGI_MIC_MAP = cfg.figi_mic_map()


def _openfigi_raw(isin: str) -> list:
    """Raw OpenFIGI API call. Returns the JSON result list."""
    url = "https://api.openfigi.com/v3/mapping"
    payload = [{"idType": "ID_ISIN", "idValue": isin}]
    try:
        req = Request(url, data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug("OpenFIGI lookup failed for %s: %s", isin, e)
        return []


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
GEO_BREAKDOWN_MAP: dict[str, dict[Geography, float]] = {}
_GEO_SOURCE_MAP: dict[str, str] = {}


def classify_geography(info: dict, ticker: str, holding: Holding) -> Geography:
    """Determine geography from yfinance info using MSCI country mapping.

    For ETFs with geo_breakdown, returns the dominant geography.
    For stocks, uses the company's country from yfinance info.
    Falls back to exchange country from ticker suffix.
    """
    # 1. Geo breakdown (scraped)
    breakdown = GEO_BREAKDOWN_MAP.get(ticker)
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

    Returns (breakdown_dict, source_name) or None.
    """
    if ticker in GEO_BREAKDOWN_MAP:
        return GEO_BREAKDOWN_MAP[ticker], _GEO_SOURCE_MAP.get(ticker, "cache")

    result = _scrape_geo_breakdown(ticker, isin)
    if result:
        breakdown, source = result
        GEO_BREAKDOWN_MAP[ticker] = breakdown
        _GEO_SOURCE_MAP[ticker] = source
        return breakdown, source
    return None


def _scrape_geo_breakdown(
    ticker: str, isin: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Get geographic breakdown via the geo resolution chain."""
    long_name = ""
    try:
        t = yf.Ticker(ticker)
        long_name = (t.info or {}).get("longName", "") or ""
    except Exception:
        pass

    from tarzan.data.geo_resolver import resolve_geo
    return resolve_geo(isin, ticker, long_name)


# ---------------------------------------------------------------------------
# Single holding enrichment
# ---------------------------------------------------------------------------

def _enrich_single(holding: Holding) -> Holding:
    """Enrich a single holding with market data from yfinance.

    Fetches price history, metadata, converts to EUR, extracts TER/yield/sector.
    Per-holding errors are caught and logged — the holding is returned partially enriched.
    """
    ticker = holding.ticker
    if not ticker:
        logger.warning("No ticker for ISIN=%s, skipping enrichment", holding.isin)
        return holding

    data_source = "yfinance"
    try:
        is_isin_ticker = len(ticker.replace("-", "")) == 12 and ticker[:2].isalpha()
        data, info, history = None, {}, pd.DataFrame()

        if is_isin_ticker:
            probe_result = _probe_isin_suffixes(ticker)
            if probe_result:
                data, resolved_ticker = probe_result
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
        # ``quantity × price / 100``. Without this scaling, BTPs
        # come out 100× over-valued.
        if holding.current_price is not None:
            quote_type = (info.get("quoteType") or "").upper()
            sec_type = (info.get("typeDisp") or "").upper()
            is_bond = (
                "BOND" in quote_type
                or "BOND" in sec_type
                or "TREASURY" in sec_type
            )
            if is_bond:
                holding.current_value = holding.quantity * holding.current_price / 100.0
                # Rescale the price history the same way so downstream
                # code that multiplies it by ``quantity`` (e.g. the
                # rebalancer's drift simulation) reads correct EUR.
                if holding.price_history is not None and len(holding.price_history) > 0:
                    holding.price_history = holding.price_history / 100.0
                holding.current_price = holding.current_price / 100.0
            else:
                holding.current_value = holding.quantity * holding.current_price
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

    return holding


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
    """Try Borsa Italiana scraping as fallback for bonds without yfinance data."""
    try:
        from tarzan.data.bond_fetcher import fetch_bond_price, bond_price_to_value

        isin = holding.isin
        if not isin or len(isin.replace("-", "")) != 12:
            holding.current_value = holding.market_value_eur
            holding.data_source = "input_csv (no market data)"
            return

        result = fetch_bond_price(isin)
        if result:
            price = result["price"]
            value = bond_price_to_value(price, holding.quantity)

            # Sanity check: if calculated value is wildly different from CSV value,
            # the quantity might not be standard nominal. Use proportional update instead.
            csv_value = holding.market_value_eur
            if csv_value > 0 and abs(value - csv_value) / csv_value > 0.5:
                # Proportional: assume CSV value was at par (100), scale by current price
                value = csv_value * (price / 100)
                logger.info("Bond %s: using proportional pricing (qty mismatch), value=%.2f", isin, value)

            holding.current_price = price
            holding.current_value = value
            holding.data_source = result["source"]
            logger.info("Bond fallback succeeded for %s: price=%.2f, value=%.2f EUR",
                        isin, price, value)
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
    holding = _enrich_single(holding)

    # Classify using fetched info
    try:
        data = cache_load(holding.ticker)
        info = data.get("info", {}) if data else {}
    except Exception:
        info = {}

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
