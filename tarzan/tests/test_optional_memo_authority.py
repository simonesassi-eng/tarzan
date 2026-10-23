"""Optional discoveries cannot poison or lead essential provider work."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from datetime import datetime, timezone
from threading import Event

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.memo import clear_provider_stores, memoized, memoized_many
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


@pytest.fixture
def owner():
    ticks = [0.0]
    context = RunContext("synthetic-memo-owner", RunMode.LIVE, None, datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    settings = replace(
        ProviderTransportSettings.configured(),
        run_budget_seconds=100.0, optional_budget_seconds=2.0,
        publication_reserve_seconds=5.0,
    )
    budget = ProviderBudget(settings, ledger, clock=lambda: ticks[0])
    session = RunSession(context, {}, ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session, ticks
        finally:
            budget.close()


@pytest.mark.parametrize("miss", [None, {}, [], pd.Series(dtype=float), pd.DataFrame()])
def test_optional_miss_is_not_a_completed_essential_fact(owner, miss):
    session, _ = owner
    calls = []
    with session.transport.optional_work("synthetic"):
        first = memoized("fixture", "key", lambda: calls.append("optional") or miss, legacy={})
        second = memoized("fixture", "key", lambda: pytest.fail("same optional phase refetch"), legacy={})
        if isinstance(miss, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        elif isinstance(miss, pd.Series):
            pd.testing.assert_series_equal(first, second)
        else:
            assert first == second == miss
    assert session.providers.store("fixture") == {}
    assert memoized("fixture", "key", lambda: calls.append("essential") or 106.0, legacy={}) == 106.0
    assert calls == ["optional", "essential"]


@pytest.mark.parametrize("error", [TimeoutError("synthetic timeout"), CancelledError()])
def test_essential_caller_never_follows_an_optional_leader_error(owner, error):
    session, _ = owner
    started, release = Event(), Event()
    calls = []

    def optional_loader():
        calls.append("optional")
        started.set()
        assert release.wait(timeout=2.0)
        raise error

    def optional():
        with session.transport.optional_work("synthetic-optional"):
            return memoized("fixture", "key", optional_loader, legacy={})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(copy_context().run, optional)
        assert started.wait(timeout=1.0)
        second = pool.submit(
            copy_context().run, memoized, "fixture", "key",
            lambda: calls.append("essential") or {"price": 106.0}, legacy={},
        )
        try:
            assert second.result(timeout=1.0) == {"price": 106.0}
        finally:
            release.set()
        with pytest.raises(type(error)) as caught:
            first.result(timeout=3.0)
        assert caught.value is error
    assert calls == ["optional", "essential"]
    assert memoized("fixture", "key", lambda: pytest.fail("lost essential fact"), legacy={}) == {
        "price": 106.0,
    }


def test_short_optional_deadline_does_not_lead_another_optional_deadline(owner):
    session, ticks = owner
    started, release = Event(), Event()
    calls = []

    def first():
        with session.transport.optional_work("first"):
            def load():
                calls.append("first")
                started.set()
                assert release.wait(timeout=2.0)
                return None
            return memoized("fixture", "key", load, legacy={})

    def second():
        with session.transport.optional_work("second"):
            return memoized("fixture", "key", lambda: calls.append("second") or 106.0, legacy={})

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(copy_context().run, first)
        assert started.wait(timeout=1.0)
        ticks[0] = 1.0  # second optional scope has a different deadline
        two = pool.submit(copy_context().run, second)
        try:
            assert two.result(timeout=1.0) == 106.0
        finally:
            release.set()
        # The later completed payload is now legitimately reusable. The
        # first missing response must neither erase it nor poison its cache.
        assert one.result(timeout=3.0) == 106.0
    assert calls == ["first", "second"]
    assert session.providers.store("fixture") == {"key": 106.0}


def test_optional_phase_can_reuse_completed_essential_fact_after_io_budget_expiry(owner):
    session, ticks = owner
    assert memoized("fixture", "key", lambda: {"price": 106.0}, legacy={}) == {"price": 106.0}
    ticks[0] = 101.0
    with session.transport.optional_work("expired"):
        answer = memoized("fixture", "key", lambda: pytest.fail("unnecessary native authority"), legacy={})
    answer["price"] = 999.0
    assert session.providers.store("fixture")["key"] == {"price": 106.0}


def test_optional_batch_partial_admission_remains_atomic_and_deadline_scoped(owner):
    session, _ = owner
    with session.transport.optional_work("synthetic"):
        with pytest.raises(ValueError, match="synthetic admission"):
            memoized_many(
                "fixture", ["a", "b"], lambda keys: {"a": 1, "b": 2},
                legacy={},
                cache_if=lambda value: True if value == 1 else (
                    (_ for _ in ()).throw(ValueError("synthetic admission"))
                ),
            )
        assert session.providers.store("fixture") == {}
        assert all(not store for store in session.providers.optional_stores.values())
        assert memoized("fixture", "a", lambda: 106.0, legacy={}) == 106.0
    assert session.providers.store("fixture") == {"a": 106.0}


def test_optional_recursion_fails_without_poisoning_essential_next_attempt(owner):
    session, _ = owner
    with session.transport.optional_work("synthetic"):
        with pytest.raises(RuntimeError, match="recursive"):
            memoized(
                "fixture", "key",
                lambda: memoized("fixture", "key", lambda: 2, legacy={}), legacy={},
            )
    assert memoized("fixture", "key", lambda: 106.0, legacy={}) == 106.0


def test_explicit_reset_drops_all_deadline_scopes_for_only_the_named_namespace(owner):
    session, ticks = owner
    for time in (0.0, 1.0):
        ticks[0] = time
        with session.transport.optional_work("synthetic"):
            memoized("first", "key", lambda: 1, legacy={})
            memoized("second", "key", lambda: 2, legacy={})
    clear_provider_stores("first")
    assert not any(namespace == "first" for namespace, _ in session.providers.optional_stores)
    assert len([namespace for namespace, _ in session.providers.optional_stores if namespace == "second"]) == 2
    with session.transport.optional_work("synthetic"):
        assert memoized("first", "key", lambda: 106.0, legacy={}) == 106.0


def test_unhashable_descriptive_optional_label_is_not_cache_authority(owner):
    session, _ = owner
    with session.transport.optional_work(["synthetic-label"]):
        assert memoized("fixture", "key", lambda: 106.0, legacy={}) == 106.0
        assert memoized("fixture", "key", lambda: pytest.fail("lost cache"), legacy={}) == 106.0


def test_sequential_essential_misses_keep_established_same_run_negative_cache(owner):
    calls = []
    assert memoized("fixture", "key", lambda: calls.append(1) or None, legacy={}) is None
    assert memoized("fixture", "key", lambda: pytest.fail("changed essential negative policy"), legacy={}) is None
    assert calls == [1]


def test_optional_scope_is_not_leaked_into_a_later_essential_call(owner):
    session, _ = owner
    with pytest.raises(ValueError, match="synthetic primary"):
        with session.transport.optional_work("synthetic"):
            memoized("fixture", "key", lambda: None, legacy={})
            raise ValueError("synthetic primary")
    assert session.transport.optional_work_token() is None
    assert memoized("fixture", "key", lambda: 106.0, legacy={}) == 106.0


def test_legacy_transport_without_optional_scope_keeps_ordinary_memo_contract(owner):
    session, _ = owner
    original = session.transport
    session.transport = object()
    try:
        assert memoized("fixture", "key", lambda: 106.0, legacy={}) == 106.0
        assert memoized(
            "fixture", "key", lambda: pytest.fail("lost legacy cached fact"), legacy={},
        ) == 106.0
    finally:
        session.transport = original


@pytest.mark.parametrize("payload", [
    0, False, {"price": 106.0}, [106.0],
    pd.Series([106.0]), pd.DataFrame({"Close": [106.0]}),
])
def test_completed_optional_payload_is_reused_without_rescheduling_provider_io(owner, payload):
    session, _ = owner
    with session.transport.optional_work("synthetic"):
        first = memoized("fixture", "key", lambda: payload, legacy={})
    second = memoized("fixture", "key", lambda: pytest.fail("refetched a completed fact"), legacy={})
    if isinstance(payload, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second)
        first.iloc[0, 0] = 999.0
        assert second.iloc[0, 0] == 106.0
    elif isinstance(payload, pd.Series):
        pd.testing.assert_series_equal(first, second)
        first.iloc[0] = 999.0
        assert second.iloc[0] == 106.0
    elif isinstance(payload, dict):
        assert first == second == payload
        first["price"] = 999.0
        assert second["price"] == 106.0
    elif isinstance(payload, list):
        assert first == second == payload
        first[0] = 999.0
        assert second[0] == 106.0
    else:
        assert first == second == payload
