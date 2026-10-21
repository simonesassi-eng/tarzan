"""Provider ownership, concurrent misses and cancellation regressions."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextvars import copy_context
from datetime import date, datetime, timezone
from threading import Barrier, Event, Lock
import time

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import bond_fetcher, enricher, justetf, market_quotes
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.memo import SingleFlight, memoized, memoized_many
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


def session(name="synthetic"):
    context = RunContext(name, RunMode.LIVE, None, datetime.now(timezone.utc))
    return RunSession(context, {}, RunLedger(name))


def parallel(fn, count=8):
    barrier = Barrier(count)

    def task():
        barrier.wait(timeout=5)
        return fn()

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(copy_context().run, task) for _ in range(count)]
        return [future.result(timeout=5) for future in futures]


@pytest.mark.parametrize("negative", [False, True])
def test_parallel_info_miss_calls_provider_once(monkeypatch, negative):
    calls = []
    enricher.reset_run_caches()
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def info(self):
            calls.append(self.symbol)
            time.sleep(0.03)
            return {} if negative else {"quoteType": "ETF", "nested": {"x": 1}}

    monkeypatch.setattr(enricher.yf, "Ticker", Ticker)
    results = parallel(lambda: enricher._fetch_ticker_info("SYNTH.MI"))
    assert calls == ["SYNTH.MI"]
    assert all(result == results[0] for result in results)
    if not negative:
        results[0]["nested"]["x"] = 99
        assert results[1]["nested"]["x"] == 1
        assert enricher._fetch_ticker_info("SYNTH.MI")["nested"]["x"] == 1


@pytest.mark.parametrize("error", [ValueError("synthetic"), CancelledError()])
def test_leader_exception_reaches_waiters_and_allows_next_attempt(error):
    flight = SingleFlight()
    started = Event()
    release = Event()
    calls = []

    def loader():
        calls.append(1)
        started.set()
        assert release.wait(5)
        raise error

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(flight.one, "key", loader)
        assert started.wait(5)
        second = pool.submit(flight.one, "key", loader)
        # A short controlled wait lets the waiter reserve the pending future.
        time.sleep(0.02)
        release.set()
        for future in (first, second):
            with pytest.raises(type(error)):
                future.result(timeout=5)
    assert calls == [1]
    assert flight.one("key", lambda: 7) == 7


def test_recursion_fails_without_deadlocking_or_orphaning_other_keys():
    flight = SingleFlight()
    with pytest.raises(RuntimeError, match="recursive"):
        flight.one("a", lambda: flight.many(["b", "a"], lambda keys: {}))
    assert flight.one("a", lambda: 1) == 1
    assert flight.one("b", lambda: 2) == 2


def test_different_keys_are_not_serialized():
    flight = SingleFlight()
    barrier = Barrier(2)

    def load(key):
        barrier.wait(timeout=5)
        return key

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(flight.one, key, lambda k=key: load(k)) for key in ("a", "b")]
        assert [f.result(timeout=5) for f in futures] == ["a", "b"]


def test_overlapping_batches_only_fetch_new_keys():
    started = Event()
    release = Event()
    batches = []
    legacy = {}

    def load(keys):
        batches.append(tuple(keys))
        if "a" in keys:
            started.set()
            assert release.wait(5)
        return {key: {"key": key} for key in keys}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(memoized_many, "batch-test", ["a", "b"], load, legacy=legacy)
        assert started.wait(5)
        second = pool.submit(memoized_many, "batch-test", ["b", "c"], load, legacy=legacy)
        time.sleep(0.02)
        release.set()
        assert set(first.result(timeout=5)) == {"a", "b"}
        assert set(second.result(timeout=5)) == {"b", "c"}
    assert sorted(key for batch in batches for key in batch) == ["a", "b", "c"]


@pytest.mark.parametrize("namespace", ["enricher.info", "enricher.fx", "bond.price", "justetf.quote"])
def test_completed_runs_keep_separate_cache_values(namespace):
    legacy = {"same": "legacy"}
    first, second = session("first"), session("second")
    with activate_session(first):
        assert memoized(namespace, "same", lambda: "first", legacy=legacy) == "first"
    with activate_session(second):
        assert memoized(namespace, "same", lambda: "second", legacy=legacy) == "second"
    with activate_session(first):
        assert memoized(namespace, "same", lambda: pytest.fail("refetch"), legacy=legacy) == "first"
    assert legacy == {"same": "legacy"}


def test_reset_does_not_clear_another_run_or_shared_legacy_cache():
    first, second = session("first"), session("second")
    enricher._ticker_info_memo["SYNTH"] = {"owner": "legacy"}
    with activate_session(first):
        memoized("enricher.info", "SYNTH", lambda: {"owner": "first"}, legacy={})
    with activate_session(second):
        memoized("enricher.info", "SYNTH", lambda: {"owner": "second"}, legacy={})
        enricher.reset_run_caches()
        assert second.providers.store("enricher.info") == {}
    assert first.providers.store("enricher.info")["SYNTH"] == {"owner": "first"}
    assert enricher._ticker_info_memo.pop("SYNTH") == {"owner": "legacy"}


def test_history_fx_and_geo_coalesce_and_do_not_share_mutable_frames(monkeypatch):
    current = session()
    counts = {}
    lock = Lock()
    frame = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-01-02"]))

    def load(name, value):
        with lock:
            counts[name] = counts.get(name, 0) + 1
        time.sleep(0.02)
        return value

    monkeypatch.setattr(enricher, "_fetch_history_uncached", lambda s: load("history", frame))
    monkeypatch.setattr(enricher, "_fetch_fx_pair_uncached", lambda s: load("fx", frame["Close"]))
    monkeypatch.setattr(enricher, "_get_geo_breakdown_uncached", lambda *a: load("geo", None))
    with activate_session(current):
        histories = parallel(lambda: enricher._fetch_history("SYNTH"))
        parallel(lambda: enricher._fetch_fx_pair("USD"))
        parallel(lambda: enricher.get_geo_breakdown("SYNTH", "IE0000000001"))
    assert counts == {"history": 1, "fx": 1, "geo": 1}
    histories[0].iloc[0, 0] = 99
    assert histories[1].iloc[0, 0] == 1


def test_justetf_workers_inherit_run_and_negative_cache(monkeypatch):
    current = session()
    owners = []
    from tarzan.runtime.session import current_session

    def fetch(isin):
        owners.append((isin, current_session()))
        return None

    monkeypatch.setattr(justetf, "_fetch_series", fetch)
    monkeypatch.setattr(justetf, "_fetch_quote", fetch)
    with activate_session(current):
        justetf.prefetch(["IE0000000001", "IE0000000001"])
        justetf.prefetch(["IE0000000001"])
    assert len(owners) == 2
    assert all(owner is current for _, owner in owners)


def test_pinned_referees_and_bonds_do_not_consume_live_memo(monkeypatch):
    justetf._quote_memo["IE0000000001"] = {"price": 100}
    runtime.configure(as_of=date(2026, 1, 2))
    monkeypatch.setattr(justetf, "_fetch_quote", lambda i: pytest.fail("live transport"))
    monkeypatch.setattr(bond_fetcher, "_try_segment", lambda *a: pytest.fail("live transport"))
    assert justetf.quote("IE0000000001") is None
    assert bond_fetcher.fetch_bond_price("IT0000000001") is None
    justetf._quote_memo.pop("IE0000000001")


def test_official_quote_batch_misses_are_single_flight(monkeypatch):
    calls = []

    def fetch(symbols):
        calls.append(tuple(symbols))
        time.sleep(0.03)
        return {s: {"price": 10} for s in symbols if s != "MISSING"}

    monkeypatch.setattr(market_quotes, "_fetch_official_quotes", fetch)
    with activate_session(session()):
        results = parallel(lambda: market_quotes.official_quotes(["SYNTH", "MISSING"]))
        assert market_quotes.official_quotes(["MISSING"]) == {}
    assert calls == [("SYNTH", "MISSING")]
    assert all(result == {"SYNTH": {"price": 10}} for result in results)


def test_period_setting_is_run_owned():
    first, second = session("first"), session("second")
    with activate_session(first):
        enricher.set_portfolio_backtest_period("3y")
    with activate_session(second):
        assert enricher._backtest_period() == "5y"
        enricher.set_portfolio_backtest_period("1y")
    with activate_session(first):
        assert enricher._backtest_period() == "3y"
