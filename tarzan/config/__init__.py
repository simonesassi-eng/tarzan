"""Configuration loader — reads constants.yaml + static.yaml + instrument_taxonomy.csv.

Two-layer config:
- constants.yaml: investment parameters (classification, metric thresholds, risk-free rate)
- static.yaml: rarely-changed infrastructure mappings (exchanges, FIGI)

The tracked instrument universe and the two reference roles come from
instrument_taxonomy.csv (``watchlist``, ``is_benchmark_alpha_beta``,
``is_benchmark_geo`` columns).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Optional

import yaml
import pandas as pd

from tarzan.models.holding import Geography

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "constants.yaml")
_STATIC_PATH = os.path.join(os.path.dirname(__file__), "static.yaml")
# The taxonomy is located by _load_indexes_csv, which walks up from the cwd and
# from this module (and honours TARZAN_TAXONOMY_PATH). No fixed path constants:
# the two that used to live here described a cwd-relative + repo-root scheme
# that the ancestor walk replaced, and nothing read them.


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_static() -> dict:
    with open(_STATIC_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _taxonomy_path():
    """One taxonomy location for registry, classification and geography.

    The taxonomy is runtime data, not optional configuration. Resolve it from
    an explicit environment override first, then search upward from the
    current working directory and from this module's repository ancestors.
    This makes CLI, pytest and installed-package execution agree on the same
    repository data instead of silently depending on the caller's cwd.
    """
    from pathlib import Path

    candidates: list[Path] = []
    override = os.environ.get("TARZAN_TAXONOMY_PATH")
    if override:
        candidates.append(Path(override).expanduser())

    def add_ancestors(start: Path) -> None:
        start = start.resolve()
        for parent in (start, *start.parents):
            candidates.append(parent / "input" / "instrument_taxonomy.csv")

    add_ancestors(Path.cwd())
    add_ancestors(Path(__file__).resolve())

    seen: set[str] = set()
    path = None
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            path = candidate
            break

    if path is None:
        _warn_taxonomy(
            "instrument_taxonomy.csv not found; looked from cwd/module ancestors; "
            "benchmarks, beta reference and notional asset-class splits will use "
            "built-in defaults"
        )
        return None
    return path


@lru_cache(maxsize=1)
def _load_indexes_csv() -> pd.DataFrame:
    """Load the curated instrument taxonomy from its shared resolved location."""
    path = _taxonomy_path()
    if path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"name", "ticker", "isin", "kind", "asset_class"}
        missing = required.difference(df.columns)
        if df.empty or missing:
            raise ValueError(
                f"missing required columns {sorted(missing)} or empty file"
            )
        return df
    except Exception as e:  # noqa: BLE001
        _warn_taxonomy(
            f"instrument_taxonomy.csv at '{path}' could not be parsed ({e}); "
            "benchmarks/beta/notional splits fall back to defaults"
        )
        return pd.DataFrame()


def _cell(row, column: str) -> str:
    value = row.get(column)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    value = str(value).strip()
    return "" if value.casefold() == "nan" else value


class _CatalogIndex:
    """The curated catalog, indexed once, with ONE identity rule.

    An instrument is its ISIN. A symbol identifies a catalog row only when it
    is unambiguous: exactly one row carries it, and that row's ISIN does not
    contradict the ISIN the caller holds. Two rows sharing a symbol (MATE on
    Nasdaq, US56164V3033, and the Milan UCITS IE000BS31R00) are each keyed by a
    qualified symbol (``MATE@IE000BS31R00``) wherever code keys on the symbol;
    a bare ``MATE`` resolves to neither rather than to whichever row came
    first, which is how the UCITS was once priced, named and classed as the US
    fund. An ISIN the catalog does not know never borrows a same-symbol row.
    """

    def __init__(self, frame: pd.DataFrame):
        from collections import defaultdict
        from tarzan.models.instrument_key import normalize_isin, normalize_ticker

        self.rows = [] if frame.empty else list(frame.to_dict("records"))
        self.by_isin: dict[str, dict] = {}
        self.by_exact: dict[str, list] = defaultdict(list)
        self.by_bare: dict[str, list] = defaultdict(list)
        duplicate_isins = set()
        for row in self.rows:
            isin = normalize_isin(_cell(row, "isin"))
            ticker = _cell(row, "ticker")
            if isin:
                if isin in self.by_isin:
                    duplicate_isins.add(isin)
                self.by_isin.setdefault(isin, row)
            if ticker:
                self.by_exact[ticker.casefold()].append(row)
                self.by_bare[normalize_ticker(ticker)].append(row)
        self.shared = {bare for bare, rows in self.by_bare.items() if len(rows) > 1}
        unresolvable = sorted(
            bare for bare in self.shared
            if sum(1 for r in self.by_bare[bare] if not normalize_isin(_cell(r, "isin"))) > 0
        )
        if duplicate_isins:
            _warn_taxonomy(f"ISIN listed on more than one row (first wins): {sorted(duplicate_isins)}")
        if unresolvable:
            _warn_taxonomy(
                "symbol shared by several rows, at least one without an ISIN, so those "
                f"rows cannot be told apart and resolve to none: {unresolvable}")

    def isin_of(self, row) -> str:
        from tarzan.models.instrument_key import normalize_isin
        return normalize_isin(_cell(row, "isin"))

    def key(self, row) -> str:
        """The symbol code keys this row on: its ticker, qualified with its ISIN
        when another row shares the symbol."""
        from tarzan.models.instrument_key import normalize_ticker, qualify

        ticker = _cell(row, "ticker")
        if normalize_ticker(ticker) in self.shared:
            return qualify(ticker, self.isin_of(row))
        return ticker

    def row(self, isin=None, ticker=None, name=None, *, venue_strict: bool = False):
        """The one catalog row this identity names, or None.

        ``ticker`` may be qualified (``MATE@IE000BS31R00``) or carry a venue
        suffix. ``venue_strict``: a suffixed symbol the catalog does not list
        is an explicit venue choice and matches no bare row. ``name`` only
        breaks a tie between rows the symbol leaves ambiguous.
        """
        from tarzan.models.instrument_key import normalize_isin, normalize_ticker, split_qualified

        symbol, qualified_isin = split_qualified(ticker)
        isin = normalize_isin(isin) or qualified_isin
        if isin and isin in self.by_isin:
            return self.by_isin[isin]
        candidates = []
        if symbol:
            candidates = self.by_exact.get(symbol.casefold(), [])
            if not candidates and not ("." in symbol and venue_strict):
                candidates = self.by_bare.get(normalize_ticker(symbol), [])
        if isin:
            # A row with ANOTHER ISIN is another instrument, whatever its symbol.
            candidates = [r for r in candidates if self.isin_of(r) in ("", isin)]
        if not candidates and name:
            # Last resort, for a row the catalog lists without an ISIN (CL2):
            # its exact curated name, under the same no-contradiction rule.
            wanted = str(name).strip().casefold()
            candidates = [r for r in self.rows if _cell(r, "name").casefold() == wanted
                          and (not isin or self.isin_of(r) in ("", isin))]
        if len(candidates) > 1 and name:
            wanted = str(name).strip().casefold()
            named = [r for r in candidates if _cell(r, "name").casefold() == wanted]
            candidates = named or candidates
        if len(candidates) > 1:
            unique = {id(r) for r in candidates}
            if len(unique) > 1:
                return None
        return candidates[0] if candidates else None


_CATALOG: list = [None, None]  # (frame the index was built from, index)


def catalog() -> _CatalogIndex:
    """The indexed catalog for the current taxonomy file (rebuilt when the
    loader's cache is cleared, so tests and ``reset_input_caches`` see it)."""
    frame = _load_indexes_csv()
    if _CATALOG[0] is not frame:
        _CATALOG[0], _CATALOG[1] = frame, _CatalogIndex(frame)
    return _CATALOG[1]


def catalog_key(isin: Optional[str], ticker: Optional[str]) -> str:
    """The catalog's own key for an instrument (see :meth:`_CatalogIndex.key`),
    or the ticker unchanged when the catalog does not know it."""
    row = catalog().row(isin, ticker)
    return catalog().key(row) if row is not None else str(ticker or "").strip()


class IdentityMap(dict):
    """A catalog lookup that remembers the catalog index it was built from, so
    :func:`catalog_lookup` applies that catalog's identity rule even after the
    global taxonomy has changed (a prepared report reference keeps its own)."""

    index: Optional["_CatalogIndex"] = None


class FrozenIdentityMap(Mapping):
    """Read-only view of an :class:`IdentityMap` that keeps its index."""

    def __init__(self, data: Mapping, index: Optional["_CatalogIndex"]):
        self._data, self.index = data, index

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"FrozenIdentityMap({dict(self._data)!r})"


def _by_isin_and_ticker(df: pd.DataFrame, extract) -> dict:
    """Build a taxonomy lookup keyed by ISIN, by the row's catalog key, and by
    the bare ticker where the bare ticker names one row only.

    ``extract(row) -> value`` returns the cell value for a row, or ``None`` to
    skip it. A symbol two rows share is deliberately NOT a key: a dict lookup
    on it would silently return whichever row came first. Read these maps with
    :func:`catalog_lookup`, which applies the catalog's identity rule.
    """
    from tarzan.models.instrument_key import normalize_ticker

    index = catalog()
    out = IdentityMap()
    out.index = index
    for row in index.rows:
        value = extract(row)
        if value is None:
            continue
        isin = index.isin_of(row)
        bare = normalize_ticker(_cell(row, "ticker"))
        keys = [isin, index.key(row)]
        if bare and bare not in index.shared:
            keys.append(bare)
        for key in keys:
            if key:
                out.setdefault(key, value)
    return out


def catalog_lookup(lut: Mapping, isin: Optional[str], ticker: Optional[str],
                   name: Optional[str] = None):
    """Read a :func:`_by_isin_and_ticker` map for one instrument, or None.

    The instrument is first identified as a catalog row by the rule of the
    catalog the map was built from (ISIN; else an unambiguous, non-contradicting
    symbol); the map is then read at THAT row's key. An instrument the catalog
    cannot identify gets nothing, never a same-symbol neighbour's values.

    A map built by hand (no index) is read the simple way: ISIN, then bare
    symbol. It holds whatever its author put in it, which is the authority.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker, split_qualified

    index = getattr(lut, "index", None)
    if index is None:
        symbol, qualified_isin = split_qualified(ticker)
        for key in (normalize_isin(isin) or qualified_isin, normalize_ticker(symbol)):
            if key and key in lut:
                return lut[key]
        return None
    row = index.row(isin, ticker, name)
    if row is None:
        return None
    for key in (index.isin_of(row), index.key(row)):
        if key and key in lut:
            return lut[key]
    return None


_lookup_by_identity = catalog_lookup  # the name existing callers use


# The taxonomy column that says "track this instrument". It names the WATCHLIST,
# not a benchmark role: the two reference roles have their own columns
# (``is_benchmark_alpha_beta``, ``is_benchmark_geo``), and a tracked instrument is
# usually neither — it is something the investor watches, or a target the book
# does not hold yet. The digest decides where a tracked row is PRINTED from
# whether it is held (Returns) or targeted (Target instruments); this flag only
# says the series is worth fetching, so removing an instrument from the target
# leaves it tracked and it falls back into the watchlist on its own.
WATCHLIST_FLAG = "watchlist"


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


def instrument_taxonomy_has(identity: Optional[str]) -> bool:
    """Whether the curated taxonomy actually holds this ISIN or symbol.

    ``resolve_taxonomy_identity`` echoes unknown input straight back — by
    design, so partial identities survive — which makes it useless for asking
    "does the taxonomy know this?". This answers that question against the
    catalog's real rows: an ISIN, an exact or bare ticker (a symbol two rows
    share is still a curated symbol), or a qualified symbol.
    """
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker, split_qualified

    if not identity:
        return False
    index = catalog()
    raw = str(identity).strip()
    symbol, qualified_isin = split_qualified(raw)
    return bool(
        normalize_isin(raw) in index.by_isin
        or (qualified_isin and qualified_isin in index.by_isin)
        or (symbol and symbol.casefold() in index.by_exact)
        or (normalize_ticker(symbol) and normalize_ticker(symbol) in index.by_bare)
    )


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

    Returns ``(isin, ticker)``: the row's ISIN (or the caller's) and its PLAIN
    ticker — a provider hint, never a qualified key (see :func:`catalog_key`).
    A qualified input (``MATE@IE000BS31R00``) is read for its ISIN.
    """
    from tarzan.models.instrument_key import normalize_isin, split_qualified

    symbol, qualified_isin = split_qualified(ticker)
    normalized_isin = normalize_isin(isin) or qualified_isin
    index = catalog()
    row = index.row(normalized_isin, symbol, name, venue_strict=True)
    if row is None:
        return normalized_isin, symbol
    return normalized_isin or index.isin_of(row), _cell(row, "ticker") or symbol


def kind_for(isin: Optional[str], ticker: Optional[str]) -> Optional[str]:
    """Return one supported mechanics kind from curated taxonomy evidence.

    The catalog row is identified by the catalog's rule (ISIN; else an
    unambiguous symbol that does not contradict the ISIN), so a broker
    placeholder still finds its row while a conflicting ISIN row is never
    borrowed. ETC/ETN listings use ETF per-unit mechanics.
    """
    mapping = {
        "stock": "STOCK",
        "equity": "STOCK",
        "etf": "ETF",
        "etc": "ETF",
        "etn": "ETF",
        "bond": "BOND",
        "cash": "CASH",
    }
    row = catalog().row(isin, ticker)
    if row is None:
        return None
    return mapping.get(_cell(row, "kind").casefold())


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
_NOTIONAL_ROLE_HINTS = ("efficient core", "stacked", "multi-asset", "equity leveraged")
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
    (suffix stripped, uppercased).

    Two consumers, one lookup: a resilient fallback for ``holding.name`` when
    yfinance is unreachable / carries no name (via :func:`name_for`), and the
    presentation layer's preferred display name over whatever the broker's order
    export called the instrument ("WS GL EFF C USD", "XTR W VAL USD-1C-AC").
    Those strings cannot be cleaned into a readable name because the information
    is not in them — it is in the taxonomy, which already names every instrument
    the portfolio can hold."""
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

    A new user's ``instrument_taxonomy.csv`` flags no ``watchlist`` row, so
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

# No separate ``chart_benchmarks()``. It answered "whose history is published for
# the charts" off the watchlist flag alone while ``benchmarks()`` answered "whose
# history is fetched", and the moment the two sets stopped being identical the
# geo benchmark was fetched and then withheld from ``benchmark_histories`` — the
# ACWI line vanished from three panels and the gap tile with it. One set, one
# accessor: if it is fetched, the charts can draw it.

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


# --- Tracked instruments (from instrument_taxonomy.csv) ---

# Every flag that earns an instrument a fetched price series. ``watchlist`` is
# the investor's own tracking list; the other two are the reference ROLES, and a
# role needs its history whether or not the investor also wants the row printed
# in the watchlist table — alpha/beta, the geo comparison, the benchmark gap
# tile, the vs-market charts and the risk table all read that series. Keeping
# the roles out of the fetch set unless separately watchlisted made two
# independent flags silently interdependent: ``watchlist=FALSE`` on the geo row
# dropped the ACWI line off three charts and the gap tile.
_TRACKED_FLAGS = (WATCHLIST_FLAG, "is_benchmark_alpha_beta", "is_benchmark_geo")


def _named_rows(flag: str) -> "list[tuple[str, pd.Series]]":
    """``(name, row)`` for each taxonomy row whose ``flag`` is true and which
    carries a name."""
    return [
        (name, row) for _, row in _flagged_rows(flag).iterrows()
        if (name := str(row.get("name", "")).strip())
    ]


def watchlist_names() -> frozenset[str]:
    """Lowercased curated names the investor tracks (``watchlist=true``).

    What the WATCHLIST table prints, which is NOT the same question as what gets
    a price series (:func:`benchmarks`): a reference role is fetched regardless,
    and a tracked instrument that is held or targeted is printed by Returns or
    Target instruments instead.

    Falls back to the shipped default alongside :func:`benchmarks`, so an
    uncurated taxonomy still shows the one instrument it computes alpha/beta
    against rather than an empty table under a populated benchmark gap.
    """
    named = {name.lower() for name, _ in _named_rows(WATCHLIST_FLAG)}
    return frozenset(named or {n.lower() for n in default_benchmarks()})


def benchmarks() -> dict[str, str]:
    """``{name: ticker}`` for every instrument that needs a fetched series.

    The union of ``_TRACKED_FLAGS``: the watchlist plus the two reference roles.
    Kept under the ``benchmarks`` name because that is what the engine catalog
    and the semantic gate call it.

    Falls back to :func:`default_benchmarks` when the taxonomy flags none, so a
    first run still computes alpha/beta instead of reporting ``None`` under a
    footnote naming an index it never used.
    """
    result = {}
    index = catalog()
    for flag in _TRACKED_FLAGS:
        for name, row in _named_rows(flag):
            # The catalog key, qualified when another row shares the symbol:
            # it is the fetch and memo key of the instrument's series.
            ticker = index.key(row.to_dict())
            if ticker:
                result.setdefault(name, ticker)
    return result or dict(default_benchmarks())


def benchmark_identities() -> tuple[tuple[str, str, str], ...]:
    """``(name, requested ticker, ISIN)`` per tracked taxonomy row.

    :func:`benchmarks` drops the ISIN because alpha/beta only ever needs a
    symbol. The appendix's feed audit prints the ISIN of every instrument it
    lists, and a tracked instrument is not a holding — its taxonomy row is the
    only place that ISIN exists. It lists whatever is FETCHED, so a reference
    role appears here even when it is not watchlisted; the run pulled its feed
    and the audit says so. Empty when the taxonomy flags none: there is then
    nothing to list, unlike ``benchmarks``, which substitutes defaults so
    alpha/beta still computes.
    """
    from tarzan.models.instrument_key import normalize_isin

    seen: dict[str, tuple[str, str, str]] = {}
    index = catalog()
    for flag in _TRACKED_FLAGS:
        for name, row in _named_rows(flag):
            seen.setdefault(name, (
                name,
                index.key(row.to_dict()),
                normalize_isin(row.get("isin")),
            ))
    return tuple(seen.values())


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
