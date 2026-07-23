"""On-disk cache for *immutable* historical market data.

The design reconciles two goals that look contradictory:

  * "I always want fresh data on every run."
  * "But the multi-year price history for my instruments only needs to be
    downloaded once."

The reconciliation: daily closes up to *yesterday* never change, so they
are cached and reused. Only the recent tail (the last few days, including
today) is re-fetched on every run, so today's price is always fresh — it
is never served stale. We cache the immutable past, not the present.

Three things are cached, all stable:
  * per-symbol daily price history (the heavy multi-year download);
  * FX pair history (currency→EUR series);
  * the deterministic ISIN→symbol resolution (skips the OpenFIGI + probe
    sweep entirely on subsequent runs);
  * the geographic breakdown per instrument (an ETF's geo allocation is
    near-immutable, so caching it makes the equity-geography section
    resilient to justETF/scrape outages).

Caching is intentionally best-effort: any read/write error degrades to a
live fetch, never breaks the pipeline.

Location
--------
* Local: ``~/.cache/tarzan/`` (override with the ``TARZAN_CACHE_DIR`` env
  var). It lives outside the repo and is git-ignored.
* GitHub Actions (newsletter): the same directory, persisted across runs
  by ``actions/cache`` — it is per-repository and isolated per fork, so a
  user cloning the repo gets their own cache with zero configuration. It
  is NOT stored in Google Drive (Drive only holds the input files) and is
  never committed to the repo. The cached data is public market data, so
  it is safe even for a public repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from tarzan.models.instrument_key import normalize_ticker

logger = logging.getLogger(__name__)

# How many trailing days to always re-fetch so the latest close (and any
# vendor revision of the last sessions) is fresh on every run.
REFRESH_TAIL_DAYS = 5

# Resolution entries older than this are re-validated by a fresh probe, so
# a delisted/renamed symbol self-heals over time.
RESOLUTION_TTL_DAYS = 30

# Instrument-profile observations retain history for point-in-time reads. Live
# runs refresh an old observation, while pinned runs may only consume evidence
# observed on or before their effective date.
INSTRUMENT_PROFILE_TTL_DAYS = 30
INSTRUMENT_PROFILE_HISTORY_LIMIT = 64
_INSTRUMENT_PROFILE_NAMESPACE = "instrument_profiles_v1"

_DISABLED_ENV = "TARZAN_DISABLE_CACHE"
_CACHE_SCHEMA_VERSION = "1"
_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _file_lock(path: Path):
    """Synchronize cache transactions across supported threads/processes."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(path):
        with open(lock_path, "a+b") as handle:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                yield
            finally:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_json_value(value) -> bool:
    if isinstance(value, dict):
        return all(isinstance(key, str) and _validate_json_value(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_validate_json_value(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, int, bool))


def _envelope(namespace: str, entries) -> dict:
    body = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "entries": entries,
    }
    body["checksum"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _read_json(path: Path, namespace: str, default):
    if not path.exists():
        return default
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        checksum = document.pop("checksum", None)
        if document.get("schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError("incompatible cache schema")
        if document.get("namespace") != namespace:
            raise ValueError("cache namespace mismatch")
        if checksum != hashlib.sha256(_canonical_json(document)).hexdigest():
            raise ValueError("cache checksum mismatch")
        entries = document.get("entries")
        if not _validate_json_value(entries):
            raise ValueError("invalid or non-finite cache value")
        return entries
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ignoring corrupt/incompatible %s cache at %s: %s", namespace, path, exc)
        return default


def _atomic_write_json(path: Path, namespace: str, entries) -> None:
    if not _validate_json_value(entries):
        raise ValueError(f"invalid or non-finite {namespace} cache value")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(_envelope(namespace, entries))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_map(path: Path, namespace: str) -> dict:
    value = _read_json(path, namespace, {})
    return value if isinstance(value, dict) else {}


def _update_map(path: Path, namespace: str, key: str, entry: dict | str) -> None:
    with _file_lock(path):
        current = _read_map(path, namespace)
        current[key] = entry
        _atomic_write_json(path, namespace, current)


def is_enabled() -> bool:
    """Cache on by default; set TARZAN_DISABLE_CACHE=1 to force live."""
    return os.environ.get(_DISABLED_ENV, "").strip() not in ("1", "true", "yes")


def cache_dir() -> Path:
    """Base cache directory, created on first use."""
    override = os.environ.get("TARZAN_CACHE_DIR")
    base = Path(override).expanduser() if override else Path.home() / ".cache" / "tarzan"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))


def _subdir(name: str) -> Path:
    d = cache_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Daily price / FX history
# ---------------------------------------------------------------------------

def _history_path(symbol: str) -> Path:
    return _subdir("history") / f"{_safe(symbol)}.json"


def load_history(symbol: str) -> Optional[pd.DataFrame]:
    """Return validated non-executable cached DataFrame/Series data."""
    if not is_enabled():
        return None
    path = _history_path(symbol)
    payload = _read_json(path, "history", None)
    if not isinstance(payload, dict) or payload.get("kind") not in ("dataframe", "series"):
        return None
    try:
        table = payload.get("table")
        frame = pd.read_json(io.StringIO(json.dumps(table)), orient="table")
        index_freq = payload.get("index_freq")
        if index_freq and isinstance(frame.index, pd.DatetimeIndex):
            try:
                frame.index = pd.DatetimeIndex(frame.index, freq=index_freq)
            except ValueError:
                pass
        if frame.empty:
            return None
        if payload["kind"] == "series":
            series = frame.iloc[:, 0]
            series.name = payload.get("name")
            return series
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ignoring invalid history cache for %s: %s", symbol, exc)
        return None


def store_history(symbol: str, df: pd.DataFrame) -> None:
    """Transactionally persist daily history using pandas table JSON."""
    if not is_enabled() or df is None or df.empty:
        return
    try:
        is_series = isinstance(df, pd.Series)
        frame = df.to_frame(name="__value__") if is_series else df
        table = json.loads(frame.to_json(
            orient="table", date_format="iso", double_precision=15
        ))
        payload = {
            "kind": "series" if is_series else "dataframe",
            "name": str(df.name) if is_series and df.name is not None else None,
            "index_freq": getattr(frame.index, "freqstr", None),
            "table": table,
        }
        path = _history_path(symbol)
        with _file_lock(path):
            _atomic_write_json(path, "history", payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Price cache write failed for %s: %s", symbol, exc)


def merge_history(cached: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine cached history with a freshly fetched tail.

    The fresh rows win on overlapping dates (to absorb vendor revisions of
    the most recent sessions), and the result is de-duplicated and sorted.
    """
    if cached is None or cached.empty:
        return fresh
    if fresh is None or fresh.empty:
        return cached
    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


# Daily Close ratios outside this band are not real market moves for the
# diversified ETFs/indices/funds we track (even a 2x-leveraged fund would
# need its underlying to move >±50% in one session). A *persistent* jump
# past it is an unadjusted split/denomination change, which we back-adjust.
SPLIT_JUMP_LO = 0.5
SPLIT_JUMP_HI = 2.0


def repair_split_jumps(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Back-adjust unadjusted split/denomination discontinuities in a daily
    OHLC history so multi-period returns stay correct.

    Yahoo sometimes fails to split-adjust non-US listings: the series then
    carries a single huge day-over-day jump (e.g. CL2.MI fell ~291x on
    2023-10-09), which makes a 5Y return read −99% instead of the true
    +150%+. We detect such a jump on Close, confirm it *persists* for ~10
    sessions — distinguishing a real split from a transient bad print —
    then multiply every earlier row by the jump factor so the series is
    continuous (the standard back-adjustment). Healthy series are returned
    untouched, and the operation is idempotent (an already-adjusted series
    has no jump left to fix).
    """
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return df
    valid = df["Close"].astype(float).dropna()
    if len(valid) < 5:
        return df
    vals = list(valid.values)
    n = len(vals)
    factor = [1.0] * n
    cum = 1.0
    splits = 0
    # A real split/denomination boundary sits between two runs of *genuine*
    # prices. A transient bad print (e.g. a lone 0.0001 tick) also produces a
    # huge day-over-day ratio, but the median of the sessions on the low side
    # then reflects the bad tick, not a new price level. Requiring both the
    # before- and after-medians to be well clear of zero rejects a spurious
    # split triggered by an aberrant near-zero print, which would otherwise
    # back-adjust the entire earlier series and corrupt all multi-period
    # returns for that instrument.
    ref = float(pd.Series([v for v in vals if v > 0]).median()) if any(v > 0 for v in vals) else 0.0
    min_level = ref * 1e-3  # 0.1% of the typical level
    for i in range(n - 2, -1, -1):
        prev = vals[i]
        r = (vals[i + 1] / prev) if prev else 1.0
        if r < SPLIT_JUMP_LO or r > SPLIT_JUMP_HI:
            before = pd.Series(vals[max(0, i - 9):i + 1]).median()
            after = pd.Series(vals[i + 1:i + 11]).median()
            expected = before * r if before else 0.0
            # Persistent split: the new level holds near before×ratio, AND
            # both surrounding levels are real prices (not a near-zero tick).
            if (expected > 0 and after > 0 and before > min_level and after > min_level
                    and abs(after - expected) / expected <= 0.35):
                cum *= r
                splits += 1
        factor[i] = cum
    if splits == 0:
        return df
    fseries = (pd.Series(factor, index=valid.index)
               .reindex(df.index).ffill().bfill().fillna(1.0))
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col].astype(float) * fseries
    logger.info("repair_split_jumps: back-adjusted %d split(s) in a price series", splits)
    return out


def refresh_start(cached: Optional[pd.DataFrame]) -> Optional[datetime]:
    """The date from which to re-fetch given a cached series: a few days
    before the last cached date. None means "no cache → full fetch"."""
    if cached is None or cached.empty:
        return None
    last = cached.index.max()
    try:
        last = last.tz_localize(None) if last.tzinfo else last
    except (AttributeError, TypeError):
        pass
    return last.to_pydatetime() - timedelta(days=REFRESH_TAIL_DAYS)


# ---------------------------------------------------------------------------
# ISIN → resolved symbol
# ---------------------------------------------------------------------------

def _resolution_path() -> Path:
    return _subdir("resolution") / "isin_to_symbol.json"


def _load_resolution_map() -> dict:
    if not is_enabled():
        return {}
    return _read_map(_resolution_path(), "resolution")


def load_resolution(isin: str) -> Optional[str]:
    """Return the cached resolved symbol for an ISIN if present and not
    expired, else None."""
    entry = _load_resolution_map().get(isin)
    if not entry:
        return None
    symbol, ts = entry.get("symbol"), entry.get("ts", 0)
    if not symbol:
        return None
    if time.time() - ts > RESOLUTION_TTL_DAYS * 86400:
        return None
    return symbol


def store_resolution(isin: str, symbol: str) -> None:
    """Persist an ISIN→symbol resolution transactionally."""
    if not is_enabled() or not isin or not symbol:
        return
    try:
        # Preserve the observable read seam used by concurrency tests; the
        # transactional update re-reads while holding the lock.
        _load_resolution_map()
        _update_map(
            _resolution_path(),
            "resolution",
            isin,
            {"symbol": symbol, "ts": time.time()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Resolution cache write failed for %s: %s", isin, exc)


# ---------------------------------------------------------------------------
# Versioned instrument profiles (ISIN → observed OpenFIGI evidence history)
# ---------------------------------------------------------------------------

def _instrument_profile_path() -> Path:
    return _subdir("instrument_profiles") / "profiles_v1.json"


def _load_instrument_profile_map() -> dict:
    if not is_enabled():
        return {}
    return _read_map(_instrument_profile_path(), _INSTRUMENT_PROFILE_NAMESPACE)


def _normalize_profile_isin(isin: str) -> str:
    return "".join(
        character
        for character in str(isin or "").strip().upper()
        if character.isalnum()
    )


def _parse_profile_observed_at(value: object) -> Optional[datetime]:
    try:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        observed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed.astimezone(timezone.utc)


def _profile_string_list(value: object) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item).strip() for item in values if str(item or "").strip()})


def _normalize_instrument_profile(isin: str, profile: object) -> Optional[dict]:
    """Return the canonical, non-executable profile representation."""
    if not isinstance(profile, dict):
        return None
    normalized_isin = _normalize_profile_isin(isin)
    profile_isin = _normalize_profile_isin(profile.get("isin", ""))
    if not normalized_isin or (profile_isin and profile_isin != normalized_isin):
        return None

    observed = _parse_profile_observed_at(profile.get("observed_at"))
    source = str(profile.get("source") or "").strip()
    status = str(profile.get("status") or "").strip().upper()
    kind_value = profile.get("kind")
    kind = str(kind_value).strip().upper() if kind_value is not None else None
    if observed is None or not source:
        return None
    if status not in {"VERIFIED", "CONFLICTING", "UNRESOLVED"}:
        return None
    if kind not in {None, "STOCK", "ETF", "BOND", "CASH"}:
        return None
    if (status == "VERIFIED") != (kind is not None):
        return None

    raw_identifiers = profile.get("identifiers")
    if not isinstance(raw_identifiers, dict):
        raw_identifiers = {}
    identifiers = {
        field: _profile_string_list(raw_identifiers.get(field))
        for field in ("figi", "compositeFIGI", "shareClassFIGI")
    }
    provenance = profile.get("provenance")
    if not isinstance(provenance, dict) or not _validate_json_value(provenance):
        return None

    return {
        "isin": normalized_isin,
        "kind": kind,
        "identifiers": identifiers,
        "securityType": _profile_string_list(profile.get("securityType")),
        "securityType2": _profile_string_list(profile.get("securityType2")),
        "marketSector": _profile_string_list(profile.get("marketSector")),
        "names": _profile_string_list(profile.get("names")),
        "tickers": _profile_string_list(profile.get("tickers")),
        "source": source,
        "provenance": provenance,
        "resolver_version": str(profile.get("resolver_version") or "1"),
        "observed_at": observed.isoformat(),
        "status": status,
        "confidence": str(profile.get("confidence") or "NONE").strip().upper(),
    }


def _profile_cutoff(as_of: Optional[date | datetime]) -> Optional[datetime]:
    if as_of is None:
        return datetime.fromtimestamp(time.time(), timezone.utc)
    if isinstance(as_of, datetime):
        cutoff = as_of
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return cutoff.astimezone(timezone.utc)
    if isinstance(as_of, date):
        return datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
    return None


def load_instrument_profile(
    isin: str,
    *,
    as_of: Optional[date | datetime] = None,
    allow_stale: bool = False,
) -> Optional[dict]:
    """Load the latest profile observation visible at ``as_of``.

    A live read applies the refresh TTL unless ``allow_stale`` is true. Pinned
    reads ignore wall-clock TTL but reject every observation made after the
    effective boundary, preventing future metadata from leaking into a replay.
    """
    normalized_isin = _normalize_profile_isin(isin)
    cutoff = _profile_cutoff(as_of)
    if not normalized_isin or cutoff is None:
        return None
    raw_history = _load_instrument_profile_map().get(normalized_isin, [])
    if isinstance(raw_history, dict):
        raw_history = [raw_history]
    if not isinstance(raw_history, list):
        return None

    visible: list[dict] = []
    for raw_profile in raw_history:
        profile = _normalize_instrument_profile(normalized_isin, raw_profile)
        if profile is None:
            continue
        observed = _parse_profile_observed_at(profile["observed_at"])
        if observed is not None and observed <= cutoff:
            visible.append(profile)
    if not visible:
        return None

    latest = max(
        visible,
        key=lambda profile: _parse_profile_observed_at(profile["observed_at"]),
    )
    observed = _parse_profile_observed_at(latest["observed_at"])
    if (
        as_of is None
        and not allow_stale
        and observed is not None
        and time.time() - observed.timestamp() > INSTRUMENT_PROFILE_TTL_DAYS * 86400
    ):
        return None
    return json.loads(_canonical_json(latest).decode("utf-8"))


def store_instrument_profile(isin: str, profile: dict) -> None:
    """Append one validated profile observation transactionally."""
    if not is_enabled():
        return
    normalized_isin = _normalize_profile_isin(isin)
    normalized = _normalize_instrument_profile(normalized_isin, profile)
    if normalized is None:
        logger.debug("Ignoring invalid instrument profile for %s", isin)
        return
    try:
        path = _instrument_profile_path()
        # Preserve the established observable read seam before the locked
        # transaction; the transaction itself always re-reads current state.
        _load_instrument_profile_map()
        with _file_lock(path):
            current = _read_map(path, _INSTRUMENT_PROFILE_NAMESPACE)
            raw_history = current.get(normalized_isin, [])
            if isinstance(raw_history, dict):
                raw_history = [raw_history]
            history = []
            if isinstance(raw_history, list):
                for candidate in raw_history:
                    existing = _normalize_instrument_profile(
                        normalized_isin, candidate
                    )
                    if existing is not None:
                        history.append(existing)
            history = [
                candidate
                for candidate in history
                if candidate["observed_at"] != normalized["observed_at"]
            ]
            history.append(normalized)
            history.sort(key=lambda candidate: candidate["observed_at"])
            current[normalized_isin] = history[-INSTRUMENT_PROFILE_HISTORY_LIMIT:]
            _atomic_write_json(path, _INSTRUMENT_PROFILE_NAMESPACE, current)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Instrument-profile cache write failed for %s: %s", isin, exc)


# ---------------------------------------------------------------------------
# Geographic breakdown (ISIN/ticker → {geo_name: pct})
# ---------------------------------------------------------------------------
# An ETF's geographic allocation is near-immutable (it drifts only as slowly
# as the underlying index reconstitutes), so it is cached like the ISIN
# resolution — with a TTL so it self-heals over time. This makes the geo
# section resilient to justETF/scrape outages, which otherwise silently
# degrade the whole equity-geography breakdown to "Not Available".

def _geo_path() -> Path:
    return _subdir("geo") / "geo_breakdown.json"


def _load_geo_map() -> dict:
    if not is_enabled():
        return {}
    return _read_map(_geo_path(), "geo")


def load_geo(key: str) -> Optional[dict]:
    """Return ``{"breakdown": {geo_name: pct}, "source": str}`` for a key
    (ISIN or ticker) if present and not expired, else None."""
    if not key:
        return None
    entry = _load_geo_map().get(key)
    if not entry:
        return None
    breakdown = entry.get("breakdown")
    if not breakdown:
        return None
    if time.time() - entry.get("ts", 0) > RESOLUTION_TTL_DAYS * 86400:
        return None
    return {"breakdown": breakdown, "source": entry.get("source", "cache")}


def store_geo(key: str, breakdown: dict, source: str) -> None:
    """Transactionally persist a geographic breakdown."""
    if not is_enabled() or not key or not breakdown:
        return
    try:
        _update_map(
            _geo_path(),
            "geo",
            key,
            {"breakdown": dict(breakdown), "source": source, "ts": time.time()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Geo cache write failed for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# TER (ISIN → total expense ratio fraction)
# ---------------------------------------------------------------------------
# An ETF's TER is near-immutable (it changes at most once a year), so it is
# cached like the geo breakdown — with a TTL so a rare fee change self-heals.
# This makes the fee drag in the what-if backtest resilient to justETF outages
# and, more importantly, fetched only once per ISIN across runs.

def _ter_path() -> Path:
    return _subdir("ter") / "ter_by_isin.json"


def _load_ter_map() -> dict:
    if not is_enabled():
        return {}
    return _read_map(_ter_path(), "ter")


def load_ter(key: str) -> Optional[float]:
    """Return the cached TER FRACTION for a key (ISIN/ticker) if present and
    not expired, else None. A cached "miss" (None value) is honoured within the
    TTL so a fund justETF has no TER for isn't re-fetched every run."""
    if not key:
        return None
    entry = _load_ter_map().get(key)
    if not entry:
        return None
    if time.time() - entry.get("ts", 0) > RESOLUTION_TTL_DAYS * 86400:
        return None
    return entry.get("ter")  # may be None (a cached miss)


def has_ter(key: str) -> bool:
    """True if a (non-expired) TER entry exists for the key, even if its cached
    value is None — so callers can skip a re-fetch on a known miss."""
    if not key:
        return False
    entry = _load_ter_map().get(key)
    if not entry:
        return False
    return time.time() - entry.get("ts", 0) <= RESOLUTION_TTL_DAYS * 86400


def store_ter(key: str, ter: Optional[float]) -> None:
    """Transactionally persist a TER fraction (or a cached miss)."""
    if not is_enabled() or not key:
        return
    try:
        if ter is not None and not math.isfinite(float(ter)):
            raise ValueError("TER must be finite or null")
        _update_map(_ter_path(), "ter", key, {"ter": ter, "ts": time.time()})
    except Exception as exc:  # noqa: BLE001
        logger.debug("TER cache write failed for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Ticker → ISIN cross-reference (bidirectional, learned over time)
# ---------------------------------------------------------------------------
# ISIN↔ticker is immutable, so once learned (from any ISIN-keyed use, from
# yfinance for US tickers, or from a resolver) it is cached forever. This lets
# the what-if tool accept a bare TICKER or an ISIN interchangeably: whichever
# is supplied, the other is filled in automatically on later runs.

def _ticker_isin_path() -> Path:
    return _subdir("xref") / "ticker_isin.json"


def _load_ticker_isin_map() -> dict:
    if not is_enabled():
        return {}
    return _read_map(_ticker_isin_path(), "ticker_isin")


def load_ticker_isin(ticker: str) -> Optional[str]:
    """Cached ISIN for a bare ticker (immutable; no TTL), or None."""
    if not ticker:
        return None
    return _load_ticker_isin_map().get(normalize_ticker(ticker))


def load_ticker_isin_reverse(isin: str) -> Optional[str]:
    """Cached bare ticker for an ISIN — the reverse of :func:`load_ticker_isin`
    — or None. The stored map is ticker→ISIN, so this scans its items; the map
    holds at most a few hundred instruments, so a linear scan is fine and keeps
    a single source of truth (no second on-disk map to keep in sync)."""
    if not isin:
        return None
    want = isin.replace("-", "").strip().upper()
    for ticker, mapped_isin in _load_ticker_isin_map().items():
        if mapped_isin == want:
            return ticker
    return None


def store_ticker_isin(ticker: str, isin: str) -> None:
    """Transactionally learn a ticker→ISIN mapping."""
    if not is_enabled() or not ticker or not isin:
        return
    clean = isin.replace("-", "").strip().upper()
    if len(clean) != 12 or not clean[:2].isalpha():
        return
    try:
        key = normalize_ticker(ticker)
        _update_map(_ticker_isin_path(), "ticker_isin", key, clean)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ticker↔ISIN cache write failed for %s: %s", ticker, exc)
