"""Provider answers that do not change between issues are not re-asked every run.

Measured on a warm run, 9 Oct 2026: eight bonds probed on nine Yahoo venues each, all
404, ~100s of request time, every run; 25 of 36 third-source venue fetches empty.
Network-free.
"""

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


def test_a_bond_listed_nowhere_is_not_reprobed(cache, monkeypatch):
    calls = _probe_counting(monkeypatch, "BOND")
    assert en._resolve_isin("IT0000000001", "") is None
    assert en._resolve_isin("IT0000000001", "") is None
    assert calls == ["IT0000000001"]


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


def test_a_venue_with_no_data_is_not_refetched(cache, monkeypatch):
    _Ticker.fetches = 0
    monkeypatch.setattr(en.yf, "Ticker", _Ticker)
    monkeypatch.setattr(en, "_space_yf_call", lambda: None)
    assert en._sibling_close_series("ABC.DE") is None
    assert en._sibling_close_series("ABC.DE") is None
    assert _Ticker.fetches == 1


def test_a_throttled_venue_is_not_recorded_as_missing(cache, monkeypatch):
    monkeypatch.setattr(en, "_retry", lambda fn, what: None)   # retries exhausted
    en._sibling_close_series("ABC.DE")
    assert price_cache.load_timed(en._NO_LISTING, "ABC.DE", 7) is None


def test_an_openfigi_mapping_is_reused_across_runs(cache, monkeypatch):
    sent = []

    def fake_retry(fn, what):
        sent.append(what)
        return [{"data": [{"name": "SYNTHETIC FUND", "ticker": "ABC"}]}]

    monkeypatch.setattr(en, "_retry", fake_retry)
    en._openfigi_memo.clear()
    first = en._openfigi_raw("IE0000000001")
    en._openfigi_memo.clear()                          # a new run
    assert en._openfigi_raw("IE0000000001") == first
    assert len(sent) == 1
