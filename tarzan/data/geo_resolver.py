"""Geographic allocation resolver.

Priority chain for assigning geo exposure to a holding:
1. Columns in holdings.csv (usa, emerging_markets, etc.) — forced override
2. Lookup in input/indexes.csv by ISIN, ticker, or index name
3. yfinance fund_top_holdings → country of each top holding → aggregated geo

The justETF index name lookup is used to bridge holdings to index names
in the indexes.csv file.
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

ASSET_GEO_PATH = os.path.join("input", "indexes.csv")

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


def _load_asset_geo() -> Optional[pd.DataFrame]:
    """Load and cache the indexes.csv file."""
    global _ASSET_GEO_DF
    if _ASSET_GEO_DF is not None:
        return _ASSET_GEO_DF
    if not os.path.exists(ASSET_GEO_PATH):
        logger.debug("No indexes.csv found at %s", ASSET_GEO_PATH)
        return None
    try:
        df = pd.read_csv(ASSET_GEO_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        _ASSET_GEO_DF = df
        logger.info("Loaded %d rows from indexes.csv", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to load indexes.csv: %s", e)
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
# Priority 2: Lookup in indexes.csv
# ---------------------------------------------------------------------------

def _lookup_asset_geo(
    isin: str, ticker: str, index_name: str = ""
) -> Optional[tuple[dict[Geography, float], str]]:
    """Look up geo exposure in indexes.csv.

    Matches by ISIN, ticker, or index name (in that order).
    """
    df = _load_asset_geo()
    if df is None or df.empty:
        return None

    # Match by ISIN
    if isin and "isin" in df.columns:
        col = df["isin"].astype(str).str.strip().str.upper()
        match = df[col == isin.strip().upper()]
        if not match.empty:
            geo = _parse_geo_row(match.iloc[0])
            if geo:
                return geo, "index_geo_allocation (isin)"

    # Match by ticker
    if ticker and "ticker" in df.columns:
        col = df["ticker"].astype(str).str.strip().str.upper()
        match = df[col == ticker.strip().upper()]
        if not match.empty:
            geo = _parse_geo_row(match.iloc[0])
            if geo:
                return geo, "index_geo_allocation (ticker)"

    # Match by index name (best match by word overlap, longest wins ties)
    if index_name and "index" in df.columns:
        idx_normalized = _normalize_index_str(index_name)
        idx_words = set(idx_normalized.split())
        candidates = []
        for i, row in df.iterrows():
            row_index = str(row.get("index", "")).strip()
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
                geo = _parse_geo_row(row)
                if geo:
                    return geo, f"index_geo_allocation (index: {row_index})"

    return None


# ---------------------------------------------------------------------------
# Priority 3: yfinance top holdings → country → geo
# ---------------------------------------------------------------------------

def _geo_from_top_holdings(ticker: str) -> Optional[tuple[dict[Geography, float], str]]:
    """Derive geo exposure from yfinance fund top holdings.

    Fetches the ETF's top holdings, looks up each holding's country
    via yfinance info, and aggregates into Geography buckets weighted
    by holding percentage.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Try to get top holdings
        try:
            holdings_df = t.funds_data.top_holdings
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

            # Look up country for this holding
            try:
                h_info = yf.Ticker(str(holding_ticker)).info or {}
                country = h_info.get("country", "")
                geo = gm.get(country, Geography.OTHER)
            except Exception:
                geo = Geography.OTHER

            geo_weights[geo] = geo_weights.get(geo, 0) + weight
            total_weight += weight

        if not geo_weights or total_weight <= 0:
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
    against indexes.csv rows with an 'index' column.
    """
    if not isin:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def lookup_geo_by_index_name(index_name: str) -> Optional[dict[Geography, float]]:
    """Look up geo exposure from indexes.csv by exact index name.

    Used for benchmark geo comparison (e.g. MSCI ACWI in Allocations tab).
    """
    df = _load_asset_geo()
    if df is None or df.empty or "index" not in df.columns:
        return None

    name_normalized = _normalize_index_str(index_name)
    for _, row in df.iterrows():
        row_index = str(row.get("index", "")).strip()
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
    1. (holdings.csv columns — handled by caller, not here)
    2. indexes.csv lookup (by ISIN, ticker, or index name)
    3. yfinance top holdings fallback

    Args:
        isin: Holding ISIN.
        ticker: Holding ticker.
        etf_long_name: ETF long name from yfinance (for context).

    Returns:
        (breakdown_dict, source_name) or None.
    """
    # Priority 2: indexes.csv
    # First try direct ISIN/ticker match
    result = _lookup_asset_geo(isin, ticker)
    if result:
        return result

    # Then try index name match via justETF
    index_name = justetf_index_name(isin)
    if index_name:
        result = _lookup_asset_geo(isin, ticker, index_name)
        if result:
            return result

    # Priority 3: yfinance top holdings
    result = _geo_from_top_holdings(ticker)
    if result:
        return result

    return None
