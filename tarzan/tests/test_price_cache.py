"""Tests for the immutable on-disk market-data cache.

All deterministic and network-free: they exercise the cache primitives
directly against a temporary directory (via the ``TARZAN_CACHE_DIR``
override) so the heavy yfinance download is never touched.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tarzan.data import price_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a throwaway dir and ensure it is enabled."""
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    yield


def _history(start: str, periods: int, value: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame({"Close": [value] * periods}, index=idx)


# ── History round-trip ─────────────────────────────────────────────────────

def test_store_then_load_roundtrip():
    df = _history("2024-01-01", 10)
    assert price_cache.load_history("XDEM.MI") is None  # cold cache
    price_cache.store_history("XDEM.MI", df)
    loaded = price_cache.load_history("XDEM.MI")
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df)


def test_load_history_accepts_series():
    """FX history is a Series, not a DataFrame — both must round-trip."""
    s = pd.Series([1.1, 1.2, 1.3], index=pd.date_range("2024-01-01", periods=3))
    price_cache.store_history("FX_USD", s)
    loaded = price_cache.load_history("FX_USD")
    assert isinstance(loaded, pd.Series)
    pd.testing.assert_series_equal(loaded, s)


def test_empty_history_not_stored():
    price_cache.store_history("EMPTY", pd.DataFrame())
    assert price_cache.load_history("EMPTY") is None


def test_symbol_names_are_filesystem_safe():
    """A symbol with path separators must not escape the cache dir."""
    df = _history("2024-01-01", 3)
    price_cache.store_history("../../evil/SYM", df)
    # It still round-trips through the sanitized name.
    assert price_cache.load_history("../../evil/SYM") is not None


# ── Merge semantics (fresh tail wins on overlap) ────────────────────────────

def test_merge_appends_fresh_tail():
    cached = _history("2024-01-01", 5, value=100.0)
    fresh = _history("2024-01-04", 4, value=200.0)  # overlaps last 2 days
    merged = price_cache.merge_history(cached, fresh)
    # Union of dates (01-01..01-07), de-duplicated.
    assert len(merged) == 7
    # Fresh wins on the overlapping dates.
    assert merged.loc["2024-01-05", "Close"] == 200.0
    # Old-only dates retained.
    assert merged.loc["2024-01-01", "Close"] == 100.0
    # Sorted ascending.
    assert merged.index.is_monotonic_increasing


def test_merge_with_no_cache_returns_fresh():
    fresh = _history("2024-01-01", 3)
    assert price_cache.merge_history(None, fresh) is fresh


def test_merge_with_no_fresh_returns_cached():
    cached = _history("2024-01-01", 3)
    assert price_cache.merge_history(cached, pd.DataFrame()) is cached


# ── refresh_start: re-fetch only the recent tail ────────────────────────────

def test_refresh_start_none_for_cold_cache():
    assert price_cache.refresh_start(None) is None
    assert price_cache.refresh_start(pd.DataFrame()) is None


def test_refresh_start_is_tail_before_last_date():
    cached = _history("2024-01-01", 30)
    start = price_cache.refresh_start(cached)
    last = cached.index.max().to_pydatetime()
    assert start == last - timedelta(days=price_cache.REFRESH_TAIL_DAYS)


# ── Resolution cache (with TTL + self-heal) ─────────────────────────────────

def test_resolution_roundtrip():
    assert price_cache.load_resolution("IE00BL25JP72") is None
    price_cache.store_resolution("IE00BL25JP72", "XDEM.MI")
    assert price_cache.load_resolution("IE00BL25JP72") == "XDEM.MI"


def test_resolution_expires_after_ttl(monkeypatch):
    price_cache.store_resolution("IE00BL25JP72", "XDEM.MI")
    # Simulate an entry older than the TTL by advancing "now".
    future = time.time() + (price_cache.RESOLUTION_TTL_DAYS + 1) * 86400
    monkeypatch.setattr(price_cache.time, "time", lambda: future)
    assert price_cache.load_resolution("IE00BL25JP72") is None


def test_resolution_blank_inputs_ignored():
    price_cache.store_resolution("", "XDEM.MI")
    price_cache.store_resolution("ISIN", "")
    assert price_cache.load_resolution("") is None
    assert price_cache.load_resolution("ISIN") is None


# ── Geo breakdown cache (with TTL) ──────────────────────────────────────────

def test_geo_roundtrip():
    assert price_cache.load_geo("IE00BL25JP72") is None
    price_cache.store_geo("IE00BL25JP72", {"USA": 79.0, "Japan": 4.0}, "justetf")
    got = price_cache.load_geo("IE00BL25JP72")
    assert got is not None
    assert got["breakdown"] == {"USA": 79.0, "Japan": 4.0}
    assert got["source"] == "justetf"


def test_geo_empty_not_stored():
    price_cache.store_geo("ISIN", {}, "justetf")
    assert price_cache.load_geo("ISIN") is None


def test_geo_expires_after_ttl(monkeypatch):
    price_cache.store_geo("ISIN", {"USA": 100.0}, "justetf")
    future = time.time() + (price_cache.RESOLUTION_TTL_DAYS + 1) * 86400
    monkeypatch.setattr(price_cache.time, "time", lambda: future)
    assert price_cache.load_geo("ISIN") is None


# ── Disable switch ──────────────────────────────────────────────────────────

def test_disable_env_forces_cold(monkeypatch):
    df = _history("2024-01-01", 5)
    price_cache.store_history("XDEM.MI", df)
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "1")
    assert not price_cache.is_enabled()
    assert price_cache.load_history("XDEM.MI") is None
    price_cache.store_resolution("ISIN", "SYM")  # no-op while disabled
    assert price_cache.load_resolution("ISIN") is None
