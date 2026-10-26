"""Real local SDK SQLite initialization; no network and no persistent cache."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import threading

import pytest

from tarzan import runtime
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunSession, activate_session
from tarzan.runtime.transport import (
    ProviderBudget, ProviderTransportPolicy, ProviderTransportSettings,
    _prepare_owned_yahoo_cache,
)


@pytest.fixture
def cold_timezone_cache(monkeypatch, tmp_path):
    from yfinance import cache

    monkeypatch.setattr(cache._TzCacheManager, "_tz_cache", None)
    monkeypatch.setattr(cache._TzDBManager, "_db", None)
    monkeypatch.setattr(cache._TzDBManager, "_cache_dir", str(tmp_path))
    yield cache
    database = cache._TzDBManager._db
    if database is not None:
        database.close()


def _budget():
    settings = ProviderTransportSettings(
        300,
        tuple(
            ProviderTransportPolicy(name, 20, 1)
            for name in ("yahoo", "openfigi", "borsa", "justetf", "manual", "gemini")
        ),
    )
    return ProviderBudget(settings, capacity=4)


def test_native_client_prepares_real_empty_timezone_store_before_lookup(
    cold_timezone_cache,
):
    cache = cold_timezone_cache
    require_empty = cache.get_tz_cache()
    assert require_empty.initialised == -1
    budget = _budget()
    try:
        budget.yahoo_client()
        assert cache.get_tz_cache() is require_empty
        assert require_empty.initialised == 1
        assert require_empty.lookup("SYNTHETIC-NO-TIMEZONE") is None
        assert list(cache._TZ_KV.select()) == []
        assert budget.coordinator.snapshot()["requested"] == 0
    finally:
        budget.close()


@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_owned_concurrent_clients_see_one_real_initialized_store(
    cold_timezone_cache, workers,
):
    cache = cold_timezone_cache
    budget = _budget()
    barrier = threading.Barrier(workers)

    def client_and_lookup(number):
        barrier.wait(timeout=10)
        client = budget.yahoo_client()
        store = cache.get_tz_cache()
        assert store.initialised == 1
        assert store.lookup(f"SYNTHETIC-{number}") is None
        return client, store, store.db

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(client_and_lookup, range(workers)))
        assert len({id(row[0]) for row in results}) == 1
        assert len({id(row[1]) for row in results}) == 1
        assert len({id(row[2]) for row in results}) == 1
        assert list(cache._TZ_KV.select()) == []
        assert budget.coordinator.snapshot()["requested"] == 0
    finally:
        budget.close()


def test_existing_positive_timezone_is_preserved_across_client_generations(
    cold_timezone_cache,
):
    store = cold_timezone_cache.get_tz_cache()
    store.store("SYNTHETIC.MI", "Europe/Rome")
    budget = _budget()
    try:
        first = budget.yahoo_client()
        budget.close()
        second = budget.yahoo_client()
        assert first is not second
        assert cold_timezone_cache.get_tz_cache() is store
        assert store.lookup("SYNTHETIC.MI") == "Europe/Rome"
        assert list(cold_timezone_cache._TZ_KV.select().dicts()) == [
            {"key": "SYNTHETIC.MI", "value": "Europe/Rome"},
        ]
    finally:
        budget.close()


def test_unowned_injected_factory_does_not_prepare_sdk_cache(cold_timezone_cache):
    def injected_factory():
        raise AssertionError("not constructed by cache preparation")

    _prepare_owned_yahoo_cache(injected_factory)
    assert cold_timezone_cache._TzCacheManager._tz_cache is None
    assert cold_timezone_cache._TzDBManager._db is None


def test_injected_both_session_aliases_keep_original_cache_contract(
    monkeypatch, cold_timezone_cache,
):
    import curl_cffi.requests
    from curl_cffi.requests import session

    class InjectedSession:
        pass

    monkeypatch.setattr(curl_cffi.requests, "Session", InjectedSession)
    monkeypatch.setattr(session, "Session", InjectedSession)
    _prepare_owned_yahoo_cache(InjectedSession)
    assert cold_timezone_cache._TzCacheManager._tz_cache is None


def test_unsupported_sdk_keeps_the_original_cache_contract(
    monkeypatch, cold_timezone_cache,
):
    import yfinance
    from curl_cffi.requests.session import Session

    monkeypatch.setattr(yfinance, "__version__", "UNINSPECTED-SDK")
    _prepare_owned_yahoo_cache(Session)
    assert cold_timezone_cache._TzCacheManager._tz_cache is None


def test_injected_cache_getter_is_not_invoked(monkeypatch, cold_timezone_cache):
    from curl_cffi.requests.session import Session

    def injected():
        raise AssertionError("must preserve the injected cache port")

    monkeypatch.setattr(cold_timezone_cache, "get_tz_cache", injected)
    _prepare_owned_yahoo_cache(Session)
    assert cold_timezone_cache._TzDBManager._db is None


def test_injected_store_initializer_is_not_invoked(
    monkeypatch, cold_timezone_cache,
):
    from curl_cffi.requests.session import Session

    def injected(store):
        raise AssertionError("must preserve the injected initializer")

    monkeypatch.setattr(cold_timezone_cache._TzCache, "initialise", injected)
    _prepare_owned_yahoo_cache(Session)
    assert cold_timezone_cache._TzDBManager._db is None


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, MemoryError])
def test_cache_failure_does_not_block_native_client_or_leak_error_text(
    monkeypatch, cold_timezone_cache, caplog, error_type,
):
    import peewee

    def failed_connect(database, *arguments, **keywords):
        raise error_type("SYNTHETIC-PRIVATE-CACHE-PATH-CANARY")

    monkeypatch.setattr(peewee.SqliteDatabase, "connect", failed_connect)
    budget = _budget()
    try:
        assert budget.yahoo_client() is not None
        assert budget.coordinator.snapshot()["requested"] == 0
    finally:
        budget.close()
    assert "SYNTHETIC-PRIVATE-CACHE-PATH-CANARY" not in caplog.text
    warnings = [
        record for record in caplog.records
        if getattr(record, "public_code", None)
        == "YAHOO_NATIVE_CACHE_PREPARATION_UNAVAILABLE"
    ]
    assert len(warnings) == 1
    assert warnings[0].args == (error_type.__name__,)
    assert warnings[0].exc_info is None
    assert "SYNTHETIC-PRIVATE-CACHE-PATH-CANARY" not in repr(vars(warnings[0]))


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_actual_owner_cancellation_is_preserved_before_client_construction(
    monkeypatch, cold_timezone_cache, error_type,
):
    import peewee

    error = error_type("synthetic owner cancellation")

    def interrupted_connect(database, *arguments, **keywords):
        raise error

    monkeypatch.setattr(peewee.SqliteDatabase, "connect", interrupted_connect)
    budget = _budget()
    with pytest.raises(error_type) as caught:
        budget.yahoo_client()
    assert caught.value is error
    assert budget._yahoo_client is None
    assert budget.coordinator.snapshot()["requested"] == 0
    budget.close()


def test_failed_local_observer_does_not_mask_optional_cache_failure(
    monkeypatch, cold_timezone_cache,
):
    import peewee
    from tarzan.runtime import transport

    def failed_connect(database, *arguments, **keywords):
        raise RuntimeError("synthetic cache failure")

    def failed_log(*arguments, **keywords):
        raise ValueError("synthetic observer failure")

    monkeypatch.setattr(peewee.SqliteDatabase, "connect", failed_connect)
    monkeypatch.setattr(transport.logger, "warning", failed_log)
    budget = _budget()
    try:
        assert budget.yahoo_client() is not None
    finally:
        budget.close()


def test_preparation_and_local_lookup_hold_no_native_http_permit(
    cold_timezone_cache,
):
    context = runtime.context()
    budget = _budget()
    session = RunSession(context, {}, RunLedger(context.attempt_id))
    session.transport = budget
    try:
        with activate_session(session):
            budget.yahoo_client()
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = [
                    pool.submit(copy_context().run,
                                cold_timezone_cache.get_tz_cache().lookup,
                                f"SYNTHETIC-{number}")
                    for number in range(16)
                ]
                assert [result.result() for result in results] == [None] * 16
        admission = budget.coordinator.snapshot()
        assert admission["acquired"] == admission["released"] == 0
        assert admission["active"] == admission["active_optional"] == 0
        assert admission["queued_essential"] == admission["queued_optional"] == 0
    finally:
        session.close_transport()
