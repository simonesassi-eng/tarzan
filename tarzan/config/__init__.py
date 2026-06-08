"""Configuration loader — reads constants.yaml + static.yaml + indexes.csv.

Two-layer config:
- constants.yaml: investment parameters (classification, metric thresholds, risk-free rate)
- static.yaml: rarely-changed infrastructure mappings (exchanges, FIGI)

Benchmarks and geo references come from indexes.csv (is_benchmark,
is_benchmark_alfa_and_beta, is_benchmark_geo_allocation columns).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import yaml
import pandas as pd

from tarzan.models.holding import Geography

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "constants.yaml")
_STATIC_PATH = os.path.join(os.path.dirname(__file__), "static.yaml")
_INDEXES_CSV_PATH = os.path.join("input", "indexes.csv")


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_static() -> dict:
    with open(_STATIC_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_indexes_csv() -> pd.DataFrame:
    """Load indexes.csv into a DataFrame."""
    if not os.path.exists(_INDEXES_CSV_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(_INDEXES_CSV_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def get(key: str, default=None):
    """Get a config value. Checks constants.yaml → static.yaml."""
    val = _load_raw().get(key)
    if val is not None:
        return val
    return _load_static().get(key, default)


# --- Risk & Performance ---

def risk_free_rate() -> float:
    return get("risk_free_rate", 0.04)

def trading_days() -> int:
    return 252

def benchmark_beta() -> str:
    """Get the ticker for Alpha/Beta calculation from indexes.csv (is_benchmark_alfa_and_beta=true)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_alfa_and_beta" not in df.columns:
        return "^GSPC"
    match = df[df["is_benchmark_alfa_and_beta"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["ticker"]).strip()
    return "^GSPC"


def benchmark_beta_name() -> str:
    """Get the index name for Alpha/Beta calculation (used for column headers)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_alfa_and_beta" not in df.columns:
        return "S&P 500"
    match = df[df["is_benchmark_alfa_and_beta"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["index"]).strip()
    return "S&P 500"

def chart_benchmarks() -> list[str]:
    """Get index names marked as is_benchmark=true for chart overlay."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark" not in df.columns:
        return []
    match = df[df["is_benchmark"].astype(str).str.strip().str.lower() == "true"]
    return match["index"].tolist() if not match.empty else []

def benchmark_geo_allocation() -> str:
    """Get the index name for geo benchmark reference (is_benchmark_geo_allocation=true)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_geo_allocation" not in df.columns:
        return "MSCI ACWI"
    match = df[df["is_benchmark_geo_allocation"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["index"]).strip()
    return "MSCI ACWI"

def mix_60_40() -> dict:
    return get("mix_60_40", {
        "equity_ticker": "^GSPC", "equity_weight": 0.6,
        "bond_ticker": "AGG", "bond_weight": 0.4,
    })


# --- Data fetching ---

def max_workers() -> int:
    return get("max_workers", 8)


# --- Geography ---

def geography_map() -> dict[str, Geography]:
    raw = get("geography_map", {})
    geo_lookup = {g.value: g for g in Geography}
    return {
        country: geo_lookup.get(bucket, Geography.OTHER)
        for country, bucket in raw.items()
    }


# --- Benchmarks (from indexes.csv) ---

def benchmarks() -> dict[str, str]:
    """Get benchmark dict {index_name: ticker} from indexes.csv where is_benchmark=true."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark" not in df.columns:
        return {}
    match = df[df["is_benchmark"].astype(str).str.strip().str.lower() == "true"]
    result = {}
    for _, row in match.iterrows():
        name = str(row.get("index", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if name and ticker:
            result[name] = ticker
    return result

def benchmark_details() -> dict[str, dict]:
    """Get benchmark metadata from indexes.csv where is_benchmark=true."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark" not in df.columns:
        return {}
    match = df[df["is_benchmark"].astype(str).str.strip().str.lower() == "true"]
    result = {}
    for _, row in match.iterrows():
        name = str(row.get("index", "")).strip()
        if name:
            result[name] = {
                "name": name,
                "description": str(row.get("description", "")).strip(),
                "ticker": str(row.get("ticker", "")).strip(),
            }
    return result


# --- Allocation defaults ---

def default_invested_allocation_targets_pctg() -> dict[str, float]:
    """Default allocation within the *invested* portion of the portfolio.

    Must sum to 100. Cash is tracked separately via target_cash_buffer_eur.
    """
    return get("default_invested_allocation_targets_pctg", {
        "Equities": 65.0, "Fixed Income": 25.0,
        "Gold": 5.0, "Commodities": 0.0, "Crypto": 0.0, "Alternative": 5.0,
    })


def default_equity_geo_targets_pctg() -> dict[str, float]:
    """Default geographic allocation within the equity portion. Must sum to 100."""
    return get("default_equity_geo_targets_pctg", {
        "USA": 20.0, "Japan": 20.0, "Eurozone EMU": 20.0,
        "Dev ex-USA ex-EMU ex-JP": 20.0, "Emerging Markets": 20.0,
    })


# --- Classification ---

def classification() -> dict[str, list[str]]:
    return get("classification", {})

def metric_ratings() -> dict:
    return get("metric_ratings", {})


# --- Static mappings ---

def exchange_country() -> dict[str, str]:
    return get("exchange_country", {})

def isin_exchange_suffixes() -> list[str]:
    return get("isin_exchange_suffixes", [])

def figi_exchange_map() -> dict[str, str]:
    return get("figi_exchange_map", {})

def figi_mic_map() -> dict[str, str]:
    return get("figi_mic_map", {})

def sheet_names() -> list[str]:
    return get("sheet_names", [
        "Dashboard", "Optimizer", "Holdings", "Performance",
        "Return Contribution",
    ])


def portfolio_backtest_period() -> str:
    """Default backtest period (5 years, hardcoded)."""
    return "5y"

def portfolio_inception_date() -> str:
    """Default inception date. Can be overridden by targets.csv."""
    return str(get("portfolio_inception_date", ""))
