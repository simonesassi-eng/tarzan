"""Geographic allocation resolver.

Priority chain for assigning geo exposure to a holding:
1. Lookup in input/instrument_taxonomy.csv by ISIN, ticker, or index name
2. yfinance fund_top_holdings → country of each top holding → aggregated geo

The justETF index name lookup is used to bridge holdings to index names
in the instrument_taxonomy.csv file.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from tarzan.models.holding import Geography

logger = logging.getLogger(__name__)

# Geography map cache
_GEO_MAP: Optional[dict[str, Geography]] = None
_ASSET_GEO_DF: Optional[pd.DataFrame] = None

ASSET_GEO_PATH = os.path.join("input", "instrument_taxonomy.csv")

# Geo column names in the CSV → Geography enum value
_GEO_COLUMNS = {
    "usa": "USA",
    "emerging_markets": "Emerging Markets",
    "eurozone_emu": "Eurozone EMU",
    "japan": "Japan",
    "dev_ex_usa_ex_emu_ex_jp": "Dev ex-USA ex-EMU ex-JP",
}


def _geo_map() -> dict[str, Geography]:
    """Lazy-load the country → Geography mapping from config."""
    global _GEO_MAP
    if _GEO_MAP is None:
        from tarzan import config as cfg
        _GEO_MAP = cfg.geography_map()
    return _GEO_MAP


def reset_caches() -> None:
    """Drop the per-process caches of instrument_taxonomy.csv and the geography map.

    Called at the start of each pipeline run so a user's edited input
    files (instrument_taxonomy.csv, constants) are re-read fresh — the same "inputs
    are never shadowed" guarantee the enricher's run-memos provide. This
    matters when several runs happen in one process (e.g. an interactive
    dashboard refresh, or a multi-tenant service where each user supplies
    their own Drive inputs)."""
    global _GEO_MAP, _ASSET_GEO_DF
    _GEO_MAP = None
    _ASSET_GEO_DF = None


def _load_asset_geo() -> Optional[pd.DataFrame]:
    """Load and cache the instrument_taxonomy.csv file."""
    global _ASSET_GEO_DF
    if _ASSET_GEO_DF is not None:
        return _ASSET_GEO_DF
    if not os.path.exists(ASSET_GEO_PATH):
        logger.debug("No instrument_taxonomy.csv found at %s", ASSET_GEO_PATH)
        return None
    try:
        df = pd.read_csv(ASSET_GEO_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        _ASSET_GEO_DF = df
        logger.info("Loaded %d rows from instrument_taxonomy.csv", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load instrument_taxonomy.csv: %s", e)
        return None


import re


def _normalize_index_str(s: str) -> str:
    """Normalize an index name for fuzzy matching.

    Removes ®, ™, ©, extra whitespace, and lowercases.
    """
    s = re.sub(r"[®™©]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _parse_geo_row(row: pd.Series) -> Optional[dict[Geography, float]]:
    """Parse geo percentage columns from a DataFrame row."""
    geo_lookup = {g.value: g for g in Geography}
    result = {}
    for col_name, geo_value in _GEO_COLUMNS.items():
        if col_name in row.index:
            try:
                pct = float(row[col_name])
                if pct > 0:
                    geo = geo_lookup.get(geo_value)
                    if geo:
                        result[geo] = pct
            except (ValueError, TypeError):
                pass
    return result if result else None


# ---------------------------------------------------------------------------
# Priority 2: Lookup in instrument_taxonomy.csv
# ---------------------------------------------------------------------------

def _lookup_asset_geo(
    isin: str, ticker: str, index_name: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Look up geo exposure in instrument_taxonomy.csv.

    Matches by ISIN, ticker, or index name (in that order). Both sides go
    through the canonical identity normalizers, so a taxonomy row written as
    the bare ``CL2`` still matches a holding resolved to the full listing
    ``CL2.MI``. An exact string compare silently dropped such a holding's whole
    equity notional into a "Not Available" geography bucket.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    df = _load_asset_geo()
    if df is None or df.empty:
        return None

    # Match by ISIN
    if isin and "isin" in df.columns:
        col = df["isin"].map(normalize_isin)
        match = df[col == normalize_isin(isin)]
        if not match.empty:
            geo = _parse_geo_row(match.iloc[0])
            if geo:
                return geo, "index_geo_allocation (isin)"

    # Match by ticker
    if ticker and "ticker" in df.columns:
        col = df["ticker"].map(normalize_ticker)
        match = df[col == normalize_ticker(ticker)]
        if not match.empty:
            geo = _parse_geo_row(match.iloc[0])
            if geo:
                return geo, "index_geo_allocation (ticker)"

    # Match by index name (best match by word overlap, longest wins ties)
    if index_name and "name" in df.columns:
        idx_normalized = _normalize_index_str(index_name)
        idx_words = set(idx_normalized.split())
        candidates = []
        for i, row in df.iterrows():
            row_index = str(row.get("name", "")).strip()
            if not row_index or row_index.lower() == "nan":
                continue
            row_normalized = _normalize_index_str(row_index)
            row_words = set(row_normalized.split())
            if not row_words:
                continue
            # All words in CSV row must be present in justETF name
            if row_words.issubset(idx_words):
                candidates.append((len(row_words), i, row, row_index))

        if candidates:
            candidates.sort(key=lambda x: -x[0])  # most words matched first
            _, _, best_row, best_name = candidates[0]
            geo = _parse_geo_row(best_row)
            if geo:
                return geo, f"index_geo_allocation (index: {best_name})"

    return None


# ---------------------------------------------------------------------------
# Priority 3: a single stock's OWN country
# ---------------------------------------------------------------------------

def _geo_from_own_country(ticker: str, isin: str = "") -> Optional[tuple[dict[Geography, float], str]]:
    """A single stock's geography is its own country. 100% of it, in one bucket.

    The resolver had two paths and neither fitted an ordinary share: the curated
    taxonomy, which a user fills for the FUNDS they hold, and a look-through to an
    ETF's top holdings. For a stock the second is meaningless — Apple has no top
    holdings, it IS one — so ``_geo_from_top_holdings("AAPL")`` returns None and
    every share fell through to "Not Available". A book of US single stocks reported
    95% of its value as geographically unknown, and the drift column then printed a
    +95pp deviation against a bucket that is not a target.

    Everything needed was already here: the provider reports ``country`` on the
    instrument itself, and the config's country→geography map already knows what to
    do with it. Nothing joined them.

    ``country`` is the company's domicile rather than its listing venue, which is
    also how index providers classify a constituent — so it is the right answer for
    an EXPOSURE question, not merely the available one.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    try:
        import yfinance as yf

        from tarzan.data import _yf_net

        info = _yf_net.fetch_yf(lambda: yf.Ticker(str(ticker)).info,
                                what=f"info {ticker}", log=logger) or {}
    except Exception:  # noqa: BLE001 — geography must never break a run
        return None
    # Only for an actual share. A fund reports ETF/MUTUALFUND here, and its
    # geography is a look-through question the next rung answers.
    if str(info.get("quoteType") or "").upper() != "EQUITY":
        return None
    country = str(info.get("country") or "").strip()
    geo = _geo_map().get(country)
    if geo is not None:
        logger.info("single-stock geo for %s: %s (country %s)",
                    ticker, geo.value, country)
        return {geo: 100.0}, f"instrument country ({country})"

    # A SECONDARY listing often carries no country at all: Nestle's Stuttgart line
    # reports quoteType EQUITY and country None, while its primary Swiss listing
    # says Switzerland. The ISIN's own prefix settles it — for a SHARE that prefix
    # is the company's domicile, which is the same fact the provider would have
    # reported. (It is not a usable signal for a FUND, whose domicile says nothing
    # about its exposure, and this rung already refuses anything but EQUITY.)
    code = str(isin or "").strip().upper()[:2]
    geo = _geo_map().get(code) if len(code) == 2 and code.isalpha() else None
    if geo is not None:
        logger.info("single-stock geo for %s: %s (ISIN prefix %s; the listing "
                    "reported no country)", ticker, geo.value, code)
        return {geo: 100.0}, f"instrument domicile ({code})"

    # A country the map does not place is not Geography.OTHER by default: the
    # top-holdings rung uses that fallback for one constituent among many, where it
    # is diluted. Here it would BE the whole answer.
    return None


# ---------------------------------------------------------------------------
# Priority 4: yfinance top holdings → country → geo
# ---------------------------------------------------------------------------

def _geo_from_top_holdings(ticker: str) -> Optional[tuple[dict[Geography, float], str]]:
    """Derive geo exposure from yfinance fund top holdings.

    Fetches the ETF's top holdings, looks up each holding's country
    via yfinance info, and aggregates into Geography buckets weighted
    by holding percentage.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    try:
        import yfinance as yf
        from tarzan.data import _yf_net
        t = yf.Ticker(ticker)

        # Try to get top holdings (spaced+retried against 429 bursts).
        try:
            holdings_df = _yf_net.fetch_yf(lambda: t.funds_data.top_holdings,
                                           what=f"top_holdings {ticker}", log=logger)
        except Exception:
            holdings_df = None

        if holdings_df is None or holdings_df.empty:
            return None

        gm = _geo_map()
        geo_weights: dict[Geography, float] = {}
        total_weight = 0.0

        for holding_ticker, row in holdings_df.iterrows():
            weight = float(row.get("Holding Percent", 0))
            if weight <= 0:
                continue

            # Look up country for this holding (spaced against 429 bursts).
            try:
                h_info = _yf_net.fetch_yf(
                    lambda ht=holding_ticker: yf.Ticker(str(ht)).info,
                    what=f"info {holding_ticker}", log=logger) or {}
                country = h_info.get("country", "")
                geo = gm.get(country, Geography.OTHER)
            except Exception:
                geo = Geography.OTHER

            geo_weights[geo] = geo_weights.get(geo, 0) + weight
            total_weight += weight

        if not geo_weights or total_weight <= 0:
            return None
        # An aggregate that is ENTIRELY Geography.OTHER learned nothing: every
        # constituent's country was absent or unplaceable, and `gm.get(country,
        # OTHER)` turned each of those misses into a positive claim. Returning it
        # says "geography known, and it is Other", which then blocks the rungs
        # below and prints a bucket the reader cannot act on. Measured on a single
        # stock whose ISIN the provider would not link: {Other: 100.0}.
        if set(geo_weights) == {Geography.OTHER}:
            logger.info("top_holdings geo for %s placed no constituent; "
                        "declining rather than claiming Other", ticker)
            return None

        # Normalize to 100%
        result = {
            g: round(w / total_weight * 100, 1)
            for g, w in geo_weights.items()
        }
        logger.info("yfinance top_holdings geo for %s: %s", ticker, {
            g.value: v for g, v in result.items()
        })
        return result, "yfinance_top_holdings"

    except Exception as e:
        logger.debug("yfinance top_holdings failed for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# justETF: get benchmark index name (kept for index matching)
# ---------------------------------------------------------------------------

def justetf_index_name(isin: str) -> Optional[str]:
    """Query justETF to find the benchmark index name for an ISIN.

    Used to bridge a holding's ISIN to an index name for matching
    against instrument_taxonomy.csv rows with an 'index' column.
    """
    if not isin:
        return None
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    try:
        import requests as req
        url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
        resp = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code != 200:
            return None
        import re
        match = re.search(r"seeks to track the (.+?)(?:\s+index\.|\s+Index\.)", resp.text)
        if match:
            name = match.group(1).strip()
            logger.info("justETF: %s → '%s'", isin, name)
            return name
    except Exception as e:
        logger.debug("justETF failed for %s: %s", isin, e)
    return None


def resolve_isin(symbol: str) -> Optional[str]:
    """Best-effort ISIN for a bare/suffixed TICKER, so ticker and ISIN are
    interchangeable inputs (no manual ISIN entry needed).

    Chain: learned ticker↔ISIN cache → yfinance ``.isin`` (reliable for US
    listings; European UCITS listings usually return "-"). Any hit is cached
    (immutable). Returns None when no free source knows it (caller then keeps
    whatever ISIN it already has, or degrades gracefully).
    """
    if not symbol:
        return None
    from tarzan.data import price_cache
    cached = price_cache.load_ticker_isin(symbol)
    if cached:
        return cached
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    try:
        import yfinance as yf
        from tarzan.data import _yf_net
        raw = _yf_net.fetch_yf(lambda: yf.Ticker(symbol).isin,
                               what=f"isin {symbol}", log=logger) or ""
        raw = raw.replace("-", "").strip().upper()
        if len(raw) == 12 and raw[:2].isalpha():
            price_cache.store_ticker_isin(symbol, raw)
            return raw
    except Exception as e:  # noqa: BLE001
        logger.debug("ISIN resolve failed for %s: %s", symbol, e)
    return None


def justetf_ter(isin: str) -> Optional[float]:
    """Total expense ratio (as a FRACTION, e.g. 0.0020 == 0.20%) for an ISIN
    from its justETF profile page. yfinance rarely carries the TER for European
    UCITS ETFs, so this is the automatic source for a new EU ticker's real fee.

    Disk-cached per ISIN (``price_cache`` TER map, TTL-refreshed) — including a
    cached "miss" (None) so an ISIN justETF has no TER for is not re-fetched
    every run. Returns None if unavailable (caller falls back to a class default).
    """
    if not isin:
        return None
    from tarzan.data import price_cache
    if price_cache.has_ter(isin):
        return price_cache.load_ter(isin)
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None
    ter: Optional[float] = None
    try:
        import re

        import requests as req
        url = f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
        resp = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            m = re.search(r"[Tt]otal expense ratio[^0-9]{0,40}?(\d[.,]\d{1,2})\s*%",
                          resp.text)
            if m:
                pct = float(m.group(1).replace(",", "."))
                if 0.0 <= pct < 5.0:                 # sane TER band, in percent
                    ter = pct / 100.0                # → fraction
                    logger.info("justETF TER: %s → %.2f%%", isin, pct)
    except Exception as e:  # noqa: BLE001
        logger.debug("justETF TER failed for %s: %s", isin, e)
    price_cache.store_ter(isin, ter)                 # cache hit OR miss
    return ter


# Per-asset-class default TER (FRACTION) — last resort when neither the curated
# taxonomy, yfinance, nor justETF carry a real fee. Keyed by AssetClass.value so
# the backtest (which keys on the same strings) and the live enricher agree.
_TER_CLASS_DEFAULT = {
    "Equities": 0.0020, "Fixed Income": 0.0015, "Gold": 0.0015,
    "Commodities": 0.0040, "Alternative": 0.0090, "Crypto": 0.0050,
}


def resolve_ter(isin: str, asset_class_value: Optional[str]) -> Optional[float]:
    """Best TER estimate (FRACTION) for an instrument, most-precise source first:
    justETF profile by ISIN → per-asset-class default. Returns ``None`` when the
    asset class is unknown and justETF has nothing — a missing fee stays
    Unavailable (never silently 0%), preserving numeric-zero≠unavailable.

    The curated-taxonomy / yfinance TER is a MORE precise source and is applied
    upstream (``holding.ter``); this only fills the gap when that is absent.
    """
    jt = justetf_ter(isin) if isin else None
    if jt is not None and 0 < jt < 0.05:
        return jt
    return _TER_CLASS_DEFAULT.get(asset_class_value)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def lookup_geo_by_index_name(index_name: str) -> Optional[dict[Geography, float]]:
    """Look up geo exposure from instrument_taxonomy.csv by exact index name.

    Used for benchmark geo comparison (e.g. MSCI ACWI in Allocations tab).
    """
    df = _load_asset_geo()
    if df is None or df.empty or "name" not in df.columns:
        return None

    name_normalized = _normalize_index_str(index_name)
    for _, row in df.iterrows():
        row_index = str(row.get("name", "")).strip()
        if not row_index or row_index.lower() == "nan":
            continue
        if _normalize_index_str(row_index) == name_normalized:
            geo = _parse_geo_row(row)
            if geo:
                return geo
    return None


def resolve_geo(
    isin: str, ticker: str, etf_long_name: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Resolve geographic exposure for a holding.

    Priority:
    1. instrument_taxonomy.csv lookup (by ISIN, ticker, or index name)
    2. a single stock's own country — it has no holdings to look through
    3. yfinance top holdings fallback

    Args:
        isin: Holding ISIN.
        ticker: Holding ticker.
        etf_long_name: ETF long name from yfinance (for context).

    Returns:
        (breakdown_dict, source_name) or None.
    """
    # Priority 2: instrument_taxonomy.csv
    # First try direct ISIN/ticker match
    result = _lookup_asset_geo(isin, ticker)
    if result:
        return result

    # Index-name discovery and top-holdings inspection are live provider
    # transports. Point-in-time/reproducible runs may use only the local
    # taxonomy result above (the caller can still fall back to disk cache).
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return None

    # Priority 3: a single stock's own country. Asked BEFORE the index-name bridge
    # and the top-holdings look-through, both of which are fund questions that cost
    # a scrape and a fetch per constituent to answer "None" for a share.
    result = _geo_from_own_country(ticker, isin)
    if result:
        return result

    # Then try index name match via justETF
    index_name = justetf_index_name(isin)
    if index_name:
        result = _lookup_asset_geo(isin, ticker, index_name)
        if result:
            return result

    # Priority 4: yfinance top holdings
    result = _geo_from_top_holdings(ticker)
    if result:
        return result

    return None
