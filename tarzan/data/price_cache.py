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
    sweep entirely on subsequent runs).

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

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# How many trailing days to always re-fetch so the latest close (and any
# vendor revision of the last sessions) is fresh on every run.
REFRESH_TAIL_DAYS = 5

# Resolution entries older than this are re-validated by a fresh probe, so
# a delisted/renamed symbol self-heals over time.
RESOLUTION_TTL_DAYS = 30

_DISABLED_ENV = "TARZAN_DISABLE_CACHE"


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
    return _subdir("history") / f"{_safe(symbol)}.pkl"


def load_history(symbol: str) -> Optional[pd.DataFrame]:
    """Return the cached daily history for a symbol, or None.

    Accepts both a DataFrame (per-symbol OHLCV history) and a Series
    (an FX rate series), since both are immutable past data cached the
    same way.
    """
    if not is_enabled():
        return None
    path = _history_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
        ok = isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty
        return df if ok else None
    except Exception as e:  # noqa: BLE001
        logger.debug("Price cache read failed for %s: %s", symbol, e)
        return None


def store_history(symbol: str, df: pd.DataFrame) -> None:
    """Persist the daily history for a symbol (best-effort)."""
    if not is_enabled() or df is None or df.empty:
        return
    try:
        with open(_history_path(symbol), "wb") as f:
            pickle.dump(df, f)
    except Exception as e:  # noqa: BLE001
        logger.debug("Price cache write failed for %s: %s", symbol, e)


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
    return _subdir("resolution") / "isin_to_symbol.pkl"


def _load_resolution_map() -> dict:
    if not is_enabled():
        return {}
    path = _resolution_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.debug("Resolution cache read failed: %s", e)
        return {}


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
    """Persist an ISIN→symbol resolution (best-effort)."""
    if not is_enabled() or not isin or not symbol:
        return
    try:
        data = _load_resolution_map()
        data[isin] = {"symbol": symbol, "ts": time.time()}
        with open(_resolution_path(), "wb") as f:
            pickle.dump(data, f)
    except Exception as e:  # noqa: BLE001
        logger.debug("Resolution cache write failed for %s: %s", isin, e)
