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


def _by_isin_and_ticker(df: pd.DataFrame, extract) -> dict:
    """Build a taxonomy lookup keyed by both ISIN and bare ticker.

    ``extract(row) -> value`` returns the cell value for a row, or ``None`` to
    skip it. Keys are the canonical identity normalizers (``normalize_isin`` /
    ``normalize_ticker``); ISIN entries win over ticker entries on collision.
    One place to keep the taxonomy build rule and the identity convention.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    by_isin: dict[str, object] = {}
    by_ticker: dict[str, object] = {}
    for _, row in df.iterrows():
        value = extract(row)
        if value is None:
            continue
        isin = normalize_isin(row.get("isin"))
        if isin:
            by_isin.setdefault(isin, value)
        ticker = normalize_ticker(row.get("ticker"))
        if ticker:
            by_ticker.setdefault(ticker, value)
    merged = dict(by_ticker)
    merged.update(by_isin)  # ISIN wins on collision
    return merged


def _lookup_by_identity(lut: dict, isin: Optional[str], ticker: Optional[str]):
    """Read a ``_by_isin_and_ticker`` map by ISIN then bare ticker, or None."""
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    for key in (normalize_isin(isin), normalize_ticker(ticker)):
        if key and key in lut:
            return lut[key]
    return None


def _flagged_rows(col: str) -> pd.DataFrame:
    """Taxonomy rows whose boolean flag column ``col`` is truthy ("true")."""
    df = _load_indexes_csv()
    if df.empty or col not in df.columns:
        return df.iloc[0:0] if not df.empty else df
    return df[df[col].astype(str).str.strip().str.lower() == "true"]


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

    def extract(row):
        ac = str(row.get("asset_class", "")).strip()
        if not ac or ac.lower() == "nan":
            return None
        role = str(row.get("role", "")).strip()
        return (ac, role or None)

    return _by_isin_and_ticker(df, extract)


def instrument_taxonomy() -> dict:
    """Curated ISIN/ticker → (asset_class, role) lookup (see _taxonomy_lookup)."""
    return _taxonomy_lookup()


def resolve_taxonomy_identity(
    isin: Optional[str],
    ticker: Optional[str],
    name: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve a partial identity to one unambiguous taxonomy row.

    Target files intentionally may use a bare ticker such as ``X710``. This
    resolver preserves that curated identity so rows remain matchable; market
    enrichment is responsible for selecting a provider-compatible listing
    such as ``X710.MI``. ISIN wins when supplied; otherwise an exact ticker or
    a unique bare ticker is used. Name is only an ambiguity breaker. Ambiguous
    or absent taxonomy evidence is returned unchanged rather than guessed.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    normalized_isin = normalize_isin(isin)
    raw_ticker = str(ticker or "").strip()
    if raw_ticker.casefold() == "nan":
        raw_ticker = ""
    bare_ticker = normalize_ticker(raw_ticker)
    raw_name = str(name or "").strip()
    if raw_name.casefold() == "nan":
        raw_name = ""

    frame = _load_indexes_csv()
    if frame.empty:
        return normalized_isin, raw_ticker

    def _cell(row, column: str) -> str:
        value = row.get(column)
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    rows = list(frame.to_dict("records"))
    candidates = rows
    if normalized_isin:
        candidates = [
            row for row in rows
            if normalize_isin(_cell(row, "isin")) == normalized_isin
        ]
    elif bare_ticker:
        exact = [
            row for row in rows
            if _cell(row, "ticker").casefold() == raw_ticker.casefold()
        ]
        # A suffix supplied by the user is an explicit venue choice. If that
        # full listing is absent from the taxonomy, preserve it rather than
        # replacing it with a same-symbol listing from another exchange.
        if exact:
            candidates = exact
        elif "." in raw_ticker:
            return normalized_isin, raw_ticker
        else:
            candidates = [
                row for row in rows
                if normalize_ticker(_cell(row, "ticker")) == bare_ticker
            ]
    elif raw_name:
        candidates = [
            row for row in rows
            if _cell(row, "name").casefold() == raw_name.casefold()
        ]
    else:
        return normalized_isin, raw_ticker

    if len(candidates) > 1 and raw_name:
        exact_name = [
            row for row in candidates
            if _cell(row, "name").casefold() == raw_name.casefold()
        ]
        if exact_name:
            candidates = exact_name

    identities = {
        (
            normalize_isin(_cell(row, "isin")),
            _cell(row, "ticker"),
        )
        for row in candidates
        if normalize_isin(_cell(row, "isin")) or _cell(row, "ticker")
    }
    if len(identities) != 1:
        return normalized_isin, raw_ticker

    taxonomy_isin, taxonomy_ticker = next(iter(identities))
    return (
        normalized_isin or taxonomy_isin,
        taxonomy_ticker or raw_ticker,
    )


def kind_for(isin: Optional[str], ticker: Optional[str]) -> Optional[str]:
    """Return one supported mechanics kind from curated taxonomy evidence.

    ISIN evidence wins when the taxonomy contains it. If the supplied ISIN is
    absent, an unambiguous exact/resolved ticker may still identify a curated
    row; this supports broker placeholders without allowing a conflicting ISIN
    row to be overridden. ETC/ETN listings use ETF per-unit mechanics.
    """
    from tarzan.models.instrument_key import normalize_isin

    normalized_isin = normalize_isin(isin)
    raw_ticker = str(ticker or "").strip()
    frame = _load_indexes_csv()
    if frame.empty or "kind" not in frame.columns:
        return None

    def _clean(value) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    rows = list(frame.to_dict("records"))
    candidates = [
        row for row in rows
        if normalized_isin
        and normalize_isin(_clean(row.get("isin"))) == normalized_isin
    ]
    if not candidates and raw_ticker:
        resolved_isin, resolved_ticker = resolve_taxonomy_identity(
            "", raw_ticker
        )
        if resolved_isin:
            candidates = [
                row for row in rows
                if normalize_isin(_clean(row.get("isin"))) == resolved_isin
            ]
        if not candidates:
            expected = str(resolved_ticker or raw_ticker).strip().casefold()
            candidates = [
                row for row in rows
                if _clean(row.get("ticker")).casefold() == expected
            ]

    mapping = {
        "stock": "STOCK",
        "equity": "STOCK",
        "etf": "ETF",
        "etc": "ETF",
        "etn": "ETF",
        "bond": "BOND",
        "cash": "CASH",
    }
    kinds = {
        mapping[value]
        for value in (
            _clean(row.get("kind")).casefold()
            for row in candidates
        )
        if value in mapping
    }
    return next(iter(kinds)) if len(kinds) == 1 else None


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

    def extract(row):
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
        return breakdown or None

    return _by_isin_and_ticker(df, extract)


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

    def extract(row):
        s = str(row.get("ter", "")).strip()
        if not s or s.lower() == "nan":
            return None
        try:
            v = float(s)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    return _by_isin_and_ticker(df, extract)


@lru_cache(maxsize=1)
def name_lookup() -> dict:
    """Curated instrument display name from the ``name`` column of
    instrument_taxonomy.csv, keyed by ISIN (uppercased) and bare ticker
    (suffix stripped, uppercased). Used as a resilient fallback for
    ``holding.name`` when yfinance is unreachable / carries no name."""
    df = _load_indexes_csv()
    if df.empty or "name" not in df.columns:
        return {}

    def extract(row):
        nm = str(row.get("name", "")).strip()
        return nm if nm and nm.lower() != "nan" else None

    return _by_isin_and_ticker(df, extract)


def name_for(isin: Optional[str], ticker: Optional[str]) -> Optional[str]:
    """Curated display name for one instrument by ISIN then bare ticker, or
    None when the taxonomy has no ``name`` for it."""
    return _lookup_by_identity(name_lookup(), isin, ticker)


def ter_for(isin: Optional[str], ticker: Optional[str]) -> Optional[float]:
    """Curated TER (FRACTION) for one instrument by ISIN then bare ticker, or
    None when the taxonomy has no ``ter`` for it."""
    return _lookup_by_identity(ter_lookup(), isin, ticker)


def class_breakdown_for(isin: Optional[str], ticker: Optional[str],
                        asset_class_value: Optional[str]) -> dict:
    """Notional asset-class breakdown for one instrument.

    Explicit exp_* override (by ISIN then bare ticker) when present, else a
    derived ``{asset_class_value: 100.0}`` so every instrument — including any
    added in the future — always has a valid breakdown with no manual work.
    Returns ``{}`` only when there is neither an override nor an asset class.
    """
    override = _lookup_by_identity(class_exposure_lookup(), isin, ticker)
    if override is not None:
        return dict(override)
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

def default_benchmarks() -> dict[str, str]:
    """Fallback benchmark universe for an uncurated taxonomy.

    A new user's ``instrument_taxonomy.csv`` flags no ``is_benchmark`` row, so
    every benchmark accessor below used to return nothing and alpha/beta came
    out ``None`` while the α/β footnote still named a hardcoded index the run
    never computed. Falling back to one shipped default keeps the three
    accessors, the engine catalog and the semantic gate naming the same
    instrument. A curated row always wins.
    """
    return get("default_benchmarks", {"iShares MSCI ACWI": "ISAC.MI"})


def _default_benchmark_name() -> str:
    """Display name of the first shipped default benchmark."""
    return next(iter(default_benchmarks()), "iShares MSCI ACWI")


def benchmark_beta_name() -> str:
    """Get the index name for Alpha/Beta calculation (used for column headers)."""
    match = _flagged_rows("is_benchmark_alpha_beta")
    if not match.empty:
        return str(match.iloc[0]["name"]).strip()
    return _default_benchmark_name()

def chart_benchmarks() -> list[str]:
    """Get index names marked as is_benchmark=true for chart overlay."""
    match = _flagged_rows("is_benchmark")
    return match["name"].tolist() if not match.empty else list(default_benchmarks())

def benchmark_geo_allocation() -> str:
    """Get the index name for geo benchmark reference (is_benchmark_geo=true)."""
    match = _flagged_rows("is_benchmark_geo")
    if not match.empty:
        return str(match.iloc[0]["name"]).strip()
    return _default_benchmark_name()

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
    """Get benchmark dict {index_name: ticker} from instrument_taxonomy.csv where is_benchmark=true.

    Falls back to :func:`default_benchmarks` when the taxonomy flags none, so a
    first run still computes alpha/beta instead of reporting ``None`` under a
    footnote naming an index it never used.
    """
    match = _flagged_rows("is_benchmark")
    result = {}
    for _, row in match.iterrows():
        name = str(row.get("name", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        if name and ticker:
            result[name] = ticker
    return result or dict(default_benchmarks())


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

def portfolio_backtest_period() -> str:
    """Default backtest period (5 years, hardcoded)."""
    return "5y"


def instrument_display_names() -> dict:
    """Curated ISIN/ticker → display name from ``instrument_taxonomy.csv``.

    The holdings frame carries whatever the broker's order export called the
    instrument ("WS GL EFF C USD", "XTR W VAL USD-1C-AC"). Those strings cannot
    be cleaned into a readable name, because the information is not in them —
    it is in the taxonomy, which already names every instrument the portfolio
    can hold. This is the lookup that lets presentation prefer the curated name
    and fall back to the broker's string only for something the taxonomy has
    never seen.
    """

    def extract(row):
        name = str(row.get("name", "")).strip()
        return name if name and name.lower() != "nan" else None

    return _by_isin_and_ticker(_load_indexes_csv(), extract)
