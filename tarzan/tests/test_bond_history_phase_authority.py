"""Incomplete native bond packets cannot own a later acquisition phase."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from datetime import date
from threading import Barrier, Lock

import pytest

from tarzan.data import bond_history as bonds, price_cache
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunSession, activate_session
from tarzan.runtime import transport
from tarzan.tests.test_official_bond_history import (
    END, ISIN, START, entry, live, packet,
)


@pytest.mark.parametrize("owned", [False, True])
def test_complete_cached_fact_needs_neither_transport_nor_identity_authority(monkeypatch, live, owned):
    observed = entry()
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: deepcopy(observed))
    monkeypatch.setattr(transport, "current_budget",
                        lambda: pytest.fail("completed history requested new transport authority"))
    monkeypatch.setattr(price_cache, "load_instrument_profile",
                        lambda *args, **kwargs: pytest.fail("completed history requested identity"))
    if not owned:
        assert bonds.raw_history(ISIN, START, END) == observed
    else:
        owner = RunSession(live[0], {}, None)
        with activate_session(owner):
            assert bonds.raw_history(ISIN, START, END) == observed
            assert owner.transport is None


@pytest.mark.parametrize("early", ["absence", "partial"])
def test_later_optional_phase_retries_native_inconclusive_leaf_and_outer_result(
    monkeypatch, live, early,
):
    context, budget, _writes = live
    calls = []
    phase = {"late": False}

    def request(isin, market="MOTX"):
        calls.append((phase["late"], market))
        raw = packet()
        if not phase["late"]:
            if early == "absence" or market == "ETLX":
                raw["payload"].update(status=2304, history={"historyDt": []}, transco=None)
                if market == "ETLX":
                    raw["exchange"] = "ETLX"
            else:
                raw["payload"]["history"]["historyDt"] = raw["payload"]["history"]["historyDt"][:3]
        return raw

    monkeypatch.setattr(bonds, "_request_raw", request)
    owner = RunSession(context, {}, RunLedger(context.attempt_id), transport=budget)
    with activate_session(owner):
        with budget.optional_work("first-source-phase"):
            first = bonds.raw_history(ISIN, START, END)
        assert first is None or len(first["prices"]) == 3
        phase["late"] = True
        with budget.optional_work("second-source-phase"):
            second = bonds.raw_history(ISIN, START, END)
        assert second == entry()
        assert calls == [(False, "MOTX"), (False, "ETLX"), (True, "MOTX")]
        assert not owner.providers.flight._pending


def test_complete_fact_is_reused_across_optional_phases_with_defensive_copies(monkeypatch, live):
    context, budget, writes = live
    calls = []
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or packet())
    owner = RunSession(context, {}, RunLedger(context.attempt_id), transport=budget)
    with activate_session(owner):
        with budget.optional_work("complete-first"):
            first = bonds.raw_history(ISIN, START, END)
        first["prices"][0]["close"] = 777
        first["currency"] = "USD"
        with budget.optional_work("complete-later"):
            later = bonds.raw_history(ISIN, START, END)
        assert later == entry()
    assert calls == [ISIN]
    assert len(writes) == 1


def test_same_phase_inconclusive_packets_reuse_across_requested_windows(monkeypatch, live):
    context, budget, writes = live
    calls = []

    def request(isin, market="MOTX"):
        calls.append(market)
        raw = packet()
        raw["exchange"] = "XMIL" if market == "MOTX" else "ETLX"
        raw["payload"].update(status=2304, history={"historyDt": []}, transco=None)
        return raw

    monkeypatch.setattr(bonds, "_request_raw", request)
    with activate_session(RunSession(context, {}, None, transport=budget)):
        with budget.optional_work("same-phase"):
            assert bonds.raw_history(ISIN, START, END) is None
            assert bonds.raw_history(ISIN, date(2026, 1, 6), END) is None
    assert calls == ["MOTX", "ETLX"]
    assert writes == []


def test_optional_partial_cannot_suppress_independent_essential_recovery(monkeypatch, live):
    context, budget, _writes = live
    calls = []
    phase = {"late": False}

    def request(isin, market="MOTX"):
        calls.append((phase["late"], market))
        raw = packet()
        if not phase["late"]:
            if market == "MOTX":
                raw["payload"]["history"]["historyDt"] = raw["payload"]["history"]["historyDt"][:3]
            else:
                raw["exchange"] = "ETLX"
                raw["payload"].update(status=2304, history={"historyDt": []}, transco=None)
        return raw

    monkeypatch.setattr(bonds, "_request_raw", request)
    with activate_session(RunSession(context, {}, None, transport=budget)):
        with budget.optional_work("tentative-partial"):
            assert len(bonds.raw_history(ISIN, START, END)["prices"]) == 3
        phase["late"] = True
        assert bonds.raw_history(ISIN, START, END) == entry()
    assert calls == [(False, "MOTX"), (False, "ETLX"), (True, "MOTX")]


def test_same_phase_concurrent_packet_acquisition_is_single_flight(monkeypatch, live):
    context, budget, _writes = live
    barrier = Barrier(4, timeout=5)
    guard = Lock()
    calls = []

    def request(isin):
        with guard:
            calls.append(isin)
        return packet()

    def fetch():
        barrier.wait()
        return bonds.raw_history(ISIN, START, END)

    monkeypatch.setattr(bonds, "_request_raw", request)
    owner = RunSession(context, {}, None, transport=budget)
    with activate_session(owner), budget.optional_work("one-parallel-phase"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(copy_context().run, fetch) for _ in range(4)]
            results = [future.result(timeout=10) for future in futures]
        assert not owner.providers.flight._pending
    assert calls == [ISIN]
    assert all(value == entry() for value in results)
    results[0]["prices"][0]["close"] = 777
    assert results[1] == entry()


def test_completed_observation_cannot_satisfy_a_longer_uncovered_window(monkeypatch, live):
    context, budget, _writes = live
    phase = {"late": False}
    calls = []

    def request(isin, market="MOTX"):
        calls.append((phase["late"], market))
        raw = packet()
        if phase["late"]:
            raw["payload"]["history"]["historyDt"].append({"dt": "20260112", "closePx": 103.})
        return raw

    monkeypatch.setattr(bonds, "_request_raw", request)
    with activate_session(RunSession(context, {}, None, transport=budget)):
        with budget.optional_work("covered-shorter"):
            assert bonds.raw_history(ISIN, START, END) == entry()
        phase["late"] = True
        with budget.optional_work("uncovered-longer"):
            longer = bonds.raw_history(ISIN, START, date(2026, 1, 12))
        assert longer["prices"][-1] == {"date": "2026-01-12", "close": 103.}
    assert calls == [(False, "MOTX"), (True, "MOTX")]


def test_new_runs_do_not_borrow_a_previous_completed_observation(monkeypatch, live):
    context, budget, _writes = live
    calls = []
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or packet())
    for _ in range(2):
        with activate_session(RunSession(context, {}, None, transport=budget)):
            assert bonds.raw_history(ISIN, START, END) == entry()
    assert calls == [ISIN, ISIN]


def test_fatal_packet_cancellation_never_leaves_a_complete_fact_or_pending_flight(monkeypatch, live):
    context, budget, _writes = live
    owner = RunSession(context, {}, None, transport=budget)

    def cancelled(isin):
        raise KeyboardInterrupt("synthetic bond packet cancellation")

    monkeypatch.setattr(bonds, "_request_raw", cancelled)
    with activate_session(owner), budget.optional_work("cancelled-source"):
        with pytest.raises(KeyboardInterrupt, match="synthetic bond packet cancellation"):
            bonds.raw_history(ISIN, START, END)
        assert owner.providers.store("bond_history.complete") == {}
        assert not owner.providers.flight._pending
