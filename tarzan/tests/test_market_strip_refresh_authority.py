"""Exercise the real strip API against native run-owned cache authority.

Bind the API before the suite's autouse fixture replaces its module attribute.
Only acquisition of the synthetic strip is injected; optional scopes, cache
generations, pending single flights and run ownership remain native.
"""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from datetime import datetime, timezone
from threading import Event

import pytest

from tarzan import runtime
from tarzan.data import market_quotes as mq
from tarzan.data.market_quotes import fetch_market_quotes as fetch_strip
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.memo import memoized
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


def _strip(value):
    return [{"name": "Synthetic market", "value": value, "spark": [value]}]


@contextmanager
def _owned_session(attempt="synthetic-strip-refresh"):
    ticks = [0.0]
    context = RunContext(
        attempt, RunMode.LIVE, None, datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(
        ProviderTransportSettings.configured(), ledger, clock=lambda: ticks[0],
    )
    session = RunSession(context, {}, ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session, ticks
        finally:
            budget.close()


@pytest.fixture
def owner():
    with _owned_session() as owned:
        yield owned


@pytest.mark.parametrize("force", [False, True])
def test_completed_strip_is_reused_unless_refresh_is_explicit(owner, monkeypatch, force):
    session, _ = owner
    calls = []

    def load():
        calls.append(len(calls) + 1)
        return _strip(calls[-1])

    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", load)
    assert fetch_strip() == _strip(1)
    expected = 2 if force else 1
    assert fetch_strip(force=force) == _strip(expected)
    assert fetch_strip() == _strip(expected)
    assert len(calls) == expected
    assert session.providers.flight._pending == {}


@pytest.mark.parametrize("force", [False, True])
def test_same_optional_deadline_miss_can_be_explicitly_refreshed(
    owner, monkeypatch, force,
):
    session, _ = owner
    responses = iter([[], _strip(207.0)])
    calls = []

    def load():
        calls.append("acquire")
        return next(responses)

    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", load)
    with session.transport.optional_work("same-deadline"):
        assert fetch_strip() == []
        assert fetch_strip() == []
        assert calls == ["acquire"]
        assert session.providers.store("market.strip") == {}
        expected = _strip(207.0) if force else []
        assert fetch_strip(force=force) == expected
        assert fetch_strip() == expected
    assert len(calls) == (2 if force else 1)
    assert session.providers.flight._pending == {}


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("old_payload", [[], _strip(106.0)])
def test_refresh_does_not_follow_or_republish_a_pre_refresh_pending_strip(
    owner, monkeypatch, optional, old_payload,
):
    session, _ = owner
    started, release = Event(), Event()
    calls = []
    fresh = _strip(207.0)

    def load():
        calls.append("acquire")
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=10.0)
            return old_payload
        return fresh

    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", load)
    scope = session.transport.optional_work("same-deadline") if optional else nullcontext()
    with scope, ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(copy_context().run, fetch_strip)
        try:
            assert started.wait(timeout=2.0)
            new = pool.submit(copy_context().run, fetch_strip, force=True)
            # The post-refresh read must finish while the old acquisition is
            # still pending, not join it and subsequently return its old fact.
            assert new.result(timeout=2.0) == fresh
        finally:
            release.set()
        assert old.result(timeout=3.0) == old_payload
    assert calls == ["acquire", "acquire"]
    assert fetch_strip() == fresh
    assert session.providers.store("market.strip") == {"curated": fresh}
    assert session.providers.flight._pending == {}


def test_refresh_invalidates_only_strip_authority_not_individual_source_facts(
    owner, monkeypatch,
):
    session, ticks = owner
    official = {"price": 106.0, "currency": "EUR"}
    history = [105.0, 106.0]
    assert memoized("market.official", "synthetic", lambda: official, legacy={}) == official
    assert memoized("enricher.history", "synthetic", lambda: history, legacy={}) == history
    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: [])
    for tick in (0.0, 1.0):
        ticks[0] = tick
        with session.transport.optional_work("synthetic-optional"):
            assert fetch_strip() == []
            assert memoized("other.optional", "missing", lambda: None, legacy={}) is None

    other_optional = {
        key: dict(value)
        for key, value in session.providers.optional_stores.items()
        if key[0] == "other.optional"
    }
    other_generations = {
        name: session.providers.generations.get(name, 0)
        for name in ("market.official", "enricher.history", "other.optional")
    }
    fresh = _strip(207.0)
    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: fresh)
    with session.transport.optional_work("synthetic-optional"):
        assert fetch_strip(force=True) == fresh

    strip_optional = [
        value for key, value in session.providers.optional_stores.items()
        if key[0] == "market.strip"
    ]
    assert all(not value for value in strip_optional)
    assert {
        key: value for key, value in session.providers.optional_stores.items()
        if key[0] == "other.optional"
    } == other_optional
    assert session.providers.store("market.official") == {"synthetic": official}
    assert session.providers.store("enricher.history") == {"synthetic": history}
    assert {
        name: session.providers.generations.get(name, 0)
        for name in other_generations
    } == other_generations
    assert session.providers.flight._pending == {}


@pytest.mark.parametrize("error", [ValueError("synthetic acquisition"), CancelledError()])
def test_failed_refresh_cannot_resurrect_the_previous_completed_strip(
    owner, monkeypatch, error,
):
    session, _ = owner
    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: _strip(106.0))
    assert fetch_strip() == _strip(106.0)

    def fail():
        raise error

    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", fail)
    with pytest.raises(type(error)) as caught:
        fetch_strip(force=True)
    assert caught.value is error
    assert session.providers.store("market.strip") == {}
    assert session.providers.flight._pending == {}
    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: _strip(207.0))
    assert fetch_strip() == _strip(207.0)


def test_refreshed_strip_and_its_nested_values_remain_defensive_copies(owner, monkeypatch):
    session, _ = owner
    source = _strip(207.0)
    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: source)
    first = fetch_strip(force=True)
    first[0]["value"] = 999.0
    first[0]["spark"].append(999.0)
    source[0]["spark"].append(888.0)
    assert fetch_strip() == _strip(207.0)
    assert session.providers.store("market.strip") == {"curated": _strip(207.0)}


def test_refresh_in_one_run_does_not_change_another_runs_strip(monkeypatch):
    first = None
    with _owned_session("synthetic-first") as (first, _):
        monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: _strip(106.0))
        assert fetch_strip() == _strip(106.0)
    with _owned_session("synthetic-second") as (second, _):
        monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", lambda: _strip(207.0))
        assert fetch_strip(force=True) == _strip(207.0)
        assert second.providers is not first.providers
    assert first.providers.store("market.strip") == {"curated": _strip(106.0)}


def test_standalone_force_and_ttl_contract_are_unchanged(monkeypatch):
    ticks = [1000.0]
    calls = []
    monkeypatch.setattr(mq, "_memo", None)
    monkeypatch.setattr(mq, "_memo_at", 0.0)
    monkeypatch.setattr(mq._time, "monotonic", lambda: ticks[0])

    def load():
        calls.append(len(calls) + 1)
        return _strip(calls[-1])

    monkeypatch.setattr(mq, "_fetch_market_quotes_uncached", load)
    assert fetch_strip() == _strip(1)
    assert fetch_strip() == _strip(1)
    assert fetch_strip(force=True) == _strip(2)
    assert fetch_strip() == _strip(2)
    ticks[0] += mq._MEMO_TTL_SECONDS + 1
    assert fetch_strip() == _strip(3)
    assert calls == [1, 2, 3]
