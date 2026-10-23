"""Positive identity caching and recovery from legacy absence observations."""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.data import enricher as en
from tarzan.data import price_cache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path))
    import tarzan.runtime as runtime
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
    return tmp_path


def test_a_timed_entry_expires(cache, monkeypatch):
    price_cache.store_timed("m", "k", [1])
    assert price_cache.load_timed("m", "k", 7) == [1]
    real = price_cache.time.time
    monkeypatch.setattr(price_cache.time, "time", lambda: real() + 8 * 86400)
    assert price_cache.load_timed("m", "k", 7) is None


@pytest.mark.parametrize("stamp", [
    None, True, False, "1", [], {}, 0, -1, 10 ** 400,
])
def test_malformed_timestamps_degrade_to_a_miss_without_crashing(cache, stamp):
    price_cache._atomic_write_json(
        price_cache._timed_path("m"), "m", {"k": {"ts": stamp, "value": [1]}},
    )
    assert price_cache.load_timed("m", "k", 7) is None


@pytest.mark.parametrize("ttl", [
    None, True, False, "1", [], {}, 0, -1, float("nan"), float("inf"),
    10 ** 400, 1e308,
])
def test_malformed_ttls_degrade_to_a_miss_without_crashing(cache, ttl):
    price_cache.store_timed("m", "k", [1])
    assert price_cache.load_timed("m", "k", ttl) is None


def test_future_receipt_cannot_renew_an_observation(cache, monkeypatch):
    price_cache.store_timed("m", "k", [1])
    real_now = price_cache.time.time()
    monkeypatch.setattr(price_cache.time, "time", lambda: real_now - 60)
    assert price_cache.load_timed("m", "k", 7) is None


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timestamp_is_rejected_by_writer_and_reader(cache, monkeypatch, stamp):
    entries = {"k": {"ts": stamp, "value": [1]}}
    with pytest.raises(ValueError):
        price_cache._atomic_write_json(price_cache._timed_path("m"), "m", entries)
    monkeypatch.setattr(price_cache, "_read_map", lambda *args: entries)
    assert price_cache.load_timed("m", "k", 7) is None


def _probe_counting(monkeypatch, kind):
    calls = []
    monkeypatch.setattr(price_cache, "load_instrument_profile",
                        lambda isin, **k: {"status": "VERIFIED", "kind": kind})
    monkeypatch.setattr(en, "_openfigi_name", lambda isin: "")
    monkeypatch.setattr(en, "_openfigi_lookup", lambda isin: [])
    monkeypatch.setattr(en, "_collect_candidate_metas",
                        lambda isin, hint: calls.append(isin) or [])
    monkeypatch.setattr(en, "_resolve_via_taxonomy_quote", lambda isin, t: None)
    return calls


def test_inconclusive_bond_discovery_does_not_poison_the_isin_cache(cache, monkeypatch):
    calls = _probe_counting(monkeypatch, "BOND")
    assert en._resolve_isin("IT0000000001", "") is None
    assert en._resolve_isin("IT0000000001", "") is None
    assert calls == ["IT0000000001", "IT0000000001"]
    assert price_cache.load_timed("no_listing", "IT0000000001", 7) is None


def test_legacy_bond_absence_never_suppresses_live_discovery_or_renews_its_ttl(cache, monkeypatch):
    calls = _probe_counting(monkeypatch, "BOND")
    price_cache.store_timed("no_listing", "IT0000000001", True)
    original = price_cache._timed_path("no_listing").read_bytes()
    assert en._resolve_isin("IT0000000001", "") is None
    assert calls == ["IT0000000001"]
    assert en._resolve_isin("IT0000000001", "") is None
    assert calls == ["IT0000000001", "IT0000000001"]
    assert price_cache._timed_path("no_listing").read_bytes() == original


def test_an_etf_that_fails_to_resolve_keeps_being_retried(cache, monkeypatch):
    calls = _probe_counting(monkeypatch, "ETF")
    en._resolve_isin("IE0000000001", "")
    en._resolve_isin("IE0000000001", "")
    assert calls == ["IE0000000001", "IE0000000001"]


class _Ticker:
    fetches = 0

    def __init__(self, symbol):
        pass

    def history(self, **kw):
        _Ticker.fetches += 1
        return pd.DataFrame()


def test_an_empty_sdk_answer_does_not_claim_a_missing_venue(cache, monkeypatch):
    _Ticker.fetches = 0
    monkeypatch.setattr(en.yf, "Ticker", _Ticker)
    monkeypatch.setattr(en, "_space_yf_call", lambda: None)
    assert en._sibling_close_series("ABC.DE") is None
    assert en._sibling_close_series("ABC.DE") is None
    assert _Ticker.fetches == 2
    assert price_cache.load_timed("no_listing", "ABC.DE", 7) is None


def test_a_throttled_venue_is_not_recorded_as_missing(cache, monkeypatch):
    monkeypatch.setattr(en, "_retry", lambda fn, what: None)   # retries exhausted
    en._sibling_close_series("ABC.DE")
    assert price_cache.load_timed("no_listing", "ABC.DE", 7) is None


def test_an_openfigi_mapping_is_reused_across_runs(cache, monkeypatch):
    sent = []

    def fake_retry(fn, what, *, provider="yahoo"):
        assert provider == "openfigi"
        sent.append(what)
        return [{"data": [{"name": "SYNTHETIC FUND", "ticker": "ABC"}]}]

    monkeypatch.setattr(en, "_retry", fake_retry)
    en._openfigi_memo.clear()
    first = en._openfigi_raw("IE0000000001")
    en._openfigi_memo.clear()                          # a new run
    assert en._openfigi_raw("IE0000000001") == first
    assert len(sent) == 1
