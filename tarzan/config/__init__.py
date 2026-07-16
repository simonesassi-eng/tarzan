"""Configuration loader — reads constants.yaml + static.yaml + instrument_taxonomy.csv.

Two-layer config:
- constants.yaml: investment parameters (classification, metric thresholds, risk-free rate)
- static.yaml: rarely-changed infrastructure mappings (exchanges, FIGI)

Benchmarks and geo references come from instrument_taxonomy.csv (is_benchmark,
is_benchmark_alpha_beta, is_benchmark_geo columns).
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
# Primary taxonomy location is CWD-relative (the CLI is run from the repo
# root, like its --input_orders default). We also derive a repo-root-anchored
# fallback so a run from another directory still finds the shipped taxonomy
# instead of silently degrading benchmarks / beta reference / notional splits.
_INDEXES_CSV_PATH = os.path.join("input", "instrument_taxonomy.csv")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INDEXES_CSV_FALLBACK = os.path.join(_REPO_ROOT, "input", "instrument_taxonomy.csv")


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
    """Load instrument_taxonomy.csv into a DataFrame.

    Tries the CWD-relative path first, then the repo-root-anchored fallback.
    A missing or unparseable taxonomy is NOT silent: it degrades benchmarks,
    the beta reference and notional asset-class splits, so we record a
    data-quality WARNING pointing at the resolved/attempted path.
    """
    path = (
        _INDEXES_CSV_PATH if os.path.exists(_INDEXES_CSV_PATH)
        else (_INDEXES_CSV_FALLBACK if os.path.exists(_INDEXES_CSV_FALLBACK) else None)
    )
    if path is None:
        _warn_taxonomy(
            f"instrument_taxonomy.csv not found (looked in "
            f"'{_INDEXES_CSV_PATH}' and '{_INDEXES_CSV_FALLBACK}'); "
            "benchmarks, beta reference and notional asset-class splits will "
            "use built-in defaults"
        )
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:  # noqa: BLE001
        _warn_taxonomy(
            f"instrument_taxonomy.csv at '{path}' could not be parsed ({e}); "
            "benchmarks/beta/notional splits fall back to defaults"
        )
        return pd.DataFrame()


def _warn_taxonomy(message: str) -> None:
    """Emit a taxonomy-degradation warning to the data-quality report,
    tolerating the (rare) case where the report module is unavailable."""
    try:
        from tarzan.runtime import data_quality as dq
        dq.warning("config", message, context="instrument_taxonomy.csv")
    except Exception:  # noqa: BLE001
        pass


@lru_cache(maxsize=1)
def _taxonomy_lookup() -> dict:
    """Build the curated instrument taxonomy lookup from instrument_taxonomy.csv.

    Returns a dict keyed by both ISIN (uppercased) and bare ticker (uppercased,
    suffix stripped) → ``(asset_class_str, role_str_or_None)``. ISIN keys take
    precedence at lookup time (the caller checks ISIN before ticker). Rows
    without an ``asset_class`` value are skipped.
    """
    df = _load_indexes_csv()
    if df.empty or "asset_class" not in df.columns:
        return {}
    by_isin: dict[str, tuple] = {}
    by_ticker: dict[str, tuple] = {}
    for _, row in df.iterrows():
        ac = str(row.get("asset_class", "")).strip()
        if not ac or ac.lower() == "nan":
            continue
        role = str(row.get("role", "")).strip()
        val = (ac, role or None)
        isin = str(row.get("isin", "")).strip().upper()
        if isin and isin.lower() != "nan":
            by_isin.setdefault(isin, val)
        tk = str(row.get("ticker", "")).strip()
        bare = tk.split(".")[0].upper() if tk else ""
        if bare and bare.lower() != "nan":
            by_ticker.setdefault(bare, val)
    # ISIN entries win over ticker entries on key collisions.
    merged = dict(by_ticker)
    merged.update(by_isin)
    return merged


def instrument_taxonomy() -> dict:
    """Curated ISIN/ticker → (asset_class, role) lookup (see _taxonomy_lookup)."""
    return _taxonomy_lookup()


# Notional asset-class exposure columns in instrument_taxonomy.csv → AssetClass value.
_EXP_COLUMNS = {
    "exp_equities": "Equities",
    "exp_fixed_income": "Fixed Income",
    "exp_gold": "Gold",
    "exp_commodities": "Commodities",
    "exp_alternative": "Alternative",
    "exp_crypto": "Crypto",
}

# Roles / name fragments that signal a capital-efficient / leveraged / multi-
# asset fund whose true exposure is NOT its single asset_class — used to warn
# when such an instrument lacks explicit exp_* values.
_NOTIONAL_ROLE_HINTS = ("efficient core", "multi-asset", "equity leveraged")
_NOTIONAL_NAME_HINTS = ("leverage", "efficient core", "lifestrategy",
                        "risk parity", "return stack", "90/60", "60/40",
                        "multi-asset", "multi asset", "balanced")


@lru_cache(maxsize=1)
def class_exposure_lookup() -> dict:
    """Explicit notional asset-class exposure overrides from the exp_* columns
    of instrument_taxonomy.csv.

    Returns a dict keyed by ISIN (uppercased) and bare ticker (suffix
    stripped, uppercased) → ``{asset_class_value: pct}``. Only rows with at
    least one non-empty exp_* value are included; everything else is derived
    from the single ``asset_class`` at consumption time (see
    ``class_breakdown_for``). Percentages may sum to >100 (leverage).
    """
    df = _load_indexes_csv()
    if df.empty:
        return {}
    present = [c for c in _EXP_COLUMNS if c in df.columns]
    if not present:
        return {}
    by_isin: dict[str, dict] = {}
    by_ticker: dict[str, dict] = {}
    for _, row in df.iterrows():
        breakdown: dict[str, float] = {}
        for col in present:
            raw = row.get(col)
            if raw is None:
                continue
            s = str(raw).strip()
            if not s or s.lower() == "nan":
                continue
            try:
                v = float(s)
            except (TypeError, ValueError):
                continue
            if v != 0.0:
                breakdown[_EXP_COLUMNS[col]] = v
        if not breakdown:
            continue
        isin = str(row.get("isin", "")).strip().upper()
        if isin and isin.lower() != "nan":
            by_isin.setdefault(isin, breakdown)
        tk = str(row.get("ticker", "")).strip()
        bare = tk.split(".")[0].upper() if tk else ""
        if bare and bare.lower() != "nan":
            by_ticker.setdefault(bare, breakdown)
    merged = dict(by_ticker)
    merged.update(by_isin)  # ISIN wins on collision
    return merged


@lru_cache(maxsize=1)
def ter_lookup() -> dict:
    """Curated TER (total expense ratio, as a FRACTION e.g. 0.0035 == 0.35%)
    from the ``ter`` column of instrument_taxonomy.csv.

    Keyed by ISIN (uppercased) and bare ticker (suffix stripped, uppercased).
    Only rows with a numeric ``ter`` cell are included — the single, editable
    source of truth for per-instrument fees (no hardcoded fee tables in code)."""
    df = _load_indexes_csv()
    if df.empty or "ter" not in df.columns:
        return {}
    by_isin: dict[str, float] = {}
    by_ticker: dict[str, float] = {}
    for _, row in df.iterrows():
        s = str(row.get("ter", "")).strip()
        if not s or s.lower() == "nan":
            continue
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        isin = str(row.get("isin", "")).strip().upper()
        if isin and isin.lower() != "nan":
            by_isin.setdefault(isin, v)
        tk = str(row.get("ticker", "")).strip()
        bare = tk.split(".")[0].upper() if tk else ""
        if bare and bare.lower() != "nan":
            by_ticker.setdefault(bare, v)
    merged = dict(by_ticker)
    merged.update(by_isin)  # ISIN wins on collision
    return merged


@lru_cache(maxsize=1)
def name_lookup() -> dict:
    """Curated instrument display name from the ``name`` column of
    instrument_taxonomy.csv, keyed by ISIN (uppercased) and bare ticker
    (suffix stripped, uppercased). Used as a resilient fallback for
    ``holding.name`` when yfinance is unreachable / carries no name."""
    df = _load_indexes_csv()
    if df.empty or "name" not in df.columns:
        return {}
    by_isin: dict[str, str] = {}
    by_ticker: dict[str, str] = {}
    for _, row in df.iterrows():
        nm = str(row.get("name", "")).strip()
        if not nm or nm.lower() == "nan":
            continue
        isin = str(row.get("isin", "")).strip().upper()
        if isin and isin.lower() != "nan":
            by_isin.setdefault(isin, nm)
        tk = str(row.get("ticker", "")).strip()
        bare = tk.split(".")[0].upper() if tk else ""
        if bare and bare.lower() != "nan":
            by_ticker.setdefault(bare, nm)
    merged = dict(by_ticker)
    merged.update(by_isin)  # ISIN wins on collision
    return merged


def name_for(isin: Optional[str], ticker: Optional[str]) -> Optional[str]:
    """Curated display name for one instrument by ISIN then bare ticker, or
    None when the taxonomy has no ``name`` for it."""
    lut = name_lookup()
    for key in (
        (isin or "").strip().upper(),
        (ticker or "").split(".")[0].strip().upper(),
    ):
        if key and key in lut:
            return lut[key]
    return None


def ter_for(isin: Optional[str], ticker: Optional[str]) -> Optional[float]:
    """Curated TER (FRACTION) for one instrument by ISIN then bare ticker, or
    None when the taxonomy has no ``ter`` for it."""
    lut = ter_lookup()
    for key in (
        (isin or "").strip().upper(),
        (ticker or "").split(".")[0].strip().upper(),
    ):
        if key and key in lut:
            return lut[key]
    return None


def class_breakdown_for(isin: Optional[str], ticker: Optional[str],
                        asset_class_value: Optional[str]) -> dict:
    """Notional asset-class breakdown for one instrument.

    Explicit exp_* override (by ISIN then bare ticker) when present, else a
    derived ``{asset_class_value: 100.0}`` so every instrument — including any
    added in the future — always has a valid breakdown with no manual work.
    Returns ``{}`` only when there is neither an override nor an asset class.
    """
    lut = class_exposure_lookup()
    for key in (
        (isin or "").strip().upper(),
        (ticker or "").split(".")[0].strip().upper(),
    ):
        if key and key in lut:
            return dict(lut[key])
    if asset_class_value and str(asset_class_value).strip():
        return {str(asset_class_value).strip(): 100.0}
    return {}


def get(key: str, default=None):
    """Get a config value. Checks constants.yaml → static.yaml."""
    val = _load_raw().get(key)
    if val is not None:
        return val
    return _load_static().get(key, default)


def reset_input_caches() -> None:
    """Drop the per-process caches of the input/config files.

    constants.yaml and static.yaml ship with the code, but instrument_taxonomy.csv is
    a user-supplied input (downloaded from each user's Google Drive), so it
    must be re-read fresh on every run. Clearing all three keeps the rule
    simple and consistent: a run never serves a previous run's inputs. The
    immutable market-data disk cache (price_cache) is intentionally left
    untouched — it holds universal data, safe to reuse across runs/users."""
    _load_raw.cache_clear()
    _load_static.cache_clear()
    _load_indexes_csv.cache_clear()
    _taxonomy_lookup.cache_clear()
    class_exposure_lookup.cache_clear()
    ter_lookup.cache_clear()
    name_lookup.cache_clear()


# --- Risk & Performance ---

def risk_free_rate() -> float:
    return get("risk_free_rate", 0.04)

def trading_days() -> int:
    return 252

def benchmark_beta() -> str:
    """Get the ticker for Alpha/Beta calculation from instrument_taxonomy.csv (is_benchmark_alpha_beta=true)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_alpha_beta" not in df.columns:
        return "^GSPC"
    match = df[df["is_benchmark_alpha_beta"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["ticker"]).strip()
    return "^GSPC"


def benchmark_beta_name() -> str:
    """Get the index name for Alpha/Beta calculation (used for column headers)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_alpha_beta" not in df.columns:
        return "S&P 500"
    match = df[df["is_benchmark_alpha_beta"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["name"]).strip()
    return "S&P 500"

def chart_benchmarks() -> list[str]:
    """Get index names marked as is_benchmark=true for chart overlay."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark" not in df.columns:
        return []
    match = df[df["is_benchmark"].astype(str).str.strip().str.lower() == "true"]
    return match["name"].tolist() if not match.empty else []

def benchmark_geo_allocation() -> str:
    """Get the index name for geo benchmark reference (is_benchmark_geo=true)."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark_geo" not in df.columns:
        return "MSCI ACWI"
    match = df[df["is_benchmark_geo"].astype(str).str.strip().str.lower() == "true"]
    if not match.empty:
        return str(match.iloc[0]["name"]).strip()
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


# --- Benchmarks (from instrument_taxonomy.csv) ---

def benchmarks() -> dict[str, str]:
    """Get benchmark dict {index_name: ticker} from instrument_taxonomy.csv where is_benchmark=true."""
    df = _load_indexes_csv()
    if df.empty or "is_benchmark" not in df.columns:
        return {}
    match = df[df["is_benchmark"].astype(str).str.strip().str.lower() == "true"]
    result = {}
    for _, row in match.iterrows():
        name = str(row.get("name", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if name and ticker:
            result[name] = ticker
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
