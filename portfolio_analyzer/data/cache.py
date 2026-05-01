"""Pickle-based cache with configurable TTL for market data.

Stores yfinance responses as timestamped pickle files to avoid
redundant API calls within the TTL window.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("input", ".cache")


class DataCache:
    """Simple file-based cache with TTL expiry."""

    def __init__(self, cache_dir: str = CACHE_DIR, ttl_hours: float = 24.0):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)

    def _path(self, key: str) -> str:
        safe = key.replace("^", "_caret_").replace("/", "_slash_")
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def store(self, key: str, data: Any) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self._path(key), "wb") as f:
                pickle.dump({"ts": time.time(), "data": data}, f)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key, e)

    def load(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            age = time.time() - cached["ts"]
            if age < self.ttl.total_seconds():
                return cached["data"]
            logger.debug("Cache expired for %s (%.0fs old)", key, age)
            return None
        except Exception as e:
            logger.warning("Corrupt cache for %s, deleting: %s", key, e)
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
