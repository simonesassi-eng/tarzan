"""Known zero admission counts are evidence, not absent owner observations."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings
from tarzan.runtime.workload import MAX_RECORDED_INTEGER, build_workload_observation


@pytest.fixture
def native_session():
    context = RunContext(
        "synthetic-protected-workload", RunMode.LIVE, None,
        datetime.now(timezone.utc),
    )
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(ProviderTransportSettings.configured(), ledger)
    session = RunSession(context, {}, ledger, transport=budget)
    try:
        yield session
    finally:
        budget.close()


def observe_native(session):
    coordinator = session.transport.coordinator
    native_before = coordinator.snapshot()
    protected_before = coordinator.protection_snapshot()
    metrics = PortfolioMetrics(total_value=100.0, invested_value=100.0)
    financial_before = metrics.to_summary_dict()
    ledger_before = tuple(session.ledger.entries)
    memo_before = dict(session.memo)

    observation = build_workload_observation(
        session, metrics, outcome="SUCCEEDED", duration_ms=0,
    )

    assert coordinator.snapshot() == native_before
    assert coordinator.protection_snapshot() == protected_before
    assert metrics.to_summary_dict() == financial_before
    assert tuple(session.ledger.entries) == ledger_before
    assert session.memo == memo_before
    assert observation["descriptive_only"] is True
    assert observation["affects_analysis_id"] is False
    assert observation["affects_financial_decisions"] is False
    assert set(observation["protected_acquisition"]) == set(protected_before)
    for name, value in protected_before.items():
        assert observation["protected_acquisition"][name] == {
            "value": min(value, MAX_RECORDED_INTEGER),
            "unit": "count",
            "minimum": 0,
            "maximum_recorded": MAX_RECORDED_INTEGER,
            "truncated": value > MAX_RECORDED_INTEGER,
        }
    for name, measure in observation["native_acquisition"].items():
        if name != "queue_wait_ms":
            assert measure["value"] == native_before[name]
    semantics = observation["native_acquisition_semantics"]
    if protected_before["groups_created"]:
        assert semantics["priority"] == (
            "purchased_early_fifo_before_essential_fifo_before_optional"
        )
        assert semantics["reservation"] == (
            "purchased_callback_slots_within_same_aggregate_cap"
        )
        assert semantics["owned_referee_reservation_activation"] == (
            "after_all_current_quotes_join"
        )
    else:
        assert semantics["priority"] == "essential_fifo_before_optional"
        assert semantics["reservation"] == (
            "one_slot_only_while_essential_demand_exists"
        )
        assert "owned_referee_reservation_activation" not in semantics
    return observation


@pytest.mark.parametrize("state", [
    "fresh", "essential-completed", "optional-completed", "explicitly-closed",
])
def test_native_owner_reports_actual_zero_protection_counts(native_session, state):
    coordinator = native_session.transport.coordinator
    if state.endswith("-completed"):
        with coordinator.lease(
            optional=state == "optional-completed", remaining=lambda: 100,
        ):
            pass
    elif state == "explicitly-closed":
        coordinator.close()

    observed = observe_native(native_session)
    assert all(
        item["value"] == 0 for item in observed["protected_acquisition"].values()
    )
    assert observed["truncated"] is False


@pytest.mark.parametrize("activate", [False, True])
def test_positive_reserved_active_and_drained_counts_remain_native(
    native_session, activate,
):
    coordinator = native_session.transport.coordinator
    protection = coordinator.protect(2, activate=activate)
    try:
        before = observe_native(native_session)
        assert before["protected_acquisition"]["pending"]["value"] == 2
        assert before["protected_acquisition"]["reserved_pending"]["value"] == (
            2 if activate else 0
        )
        with coordinator.lease(
            optional=False, remaining=lambda: 100, protection=protection,
        ):
            active = observe_native(native_session)
            assert active["protected_acquisition"]["active"]["value"] == 1
            assert active["protected_acquisition"]["acquired"]["value"] == 1
            assert active["protected_acquisition"]["released"]["value"] == 0
        protection.finish(2)
        after = observe_native(native_session)
        assert after["protected_acquisition"]["pending"]["value"] == 0
        assert after["protected_acquisition"]["active"]["value"] == 0
        assert after["protected_acquisition"]["released"]["value"] == 1
        assert after["protected_acquisition"]["groups_created"]["value"] == 1
    finally:
        pending = coordinator.protection_snapshot()["pending"]
        if pending:
            protection.finish(pending)
        protection.close()


def test_recording_bound_still_marks_actual_large_protection_counts(native_session):
    coordinator = native_session.transport.coordinator
    coordinator._protected_groups_created = MAX_RECORDED_INTEGER + 1
    observed = observe_native(native_session)
    assert observed["protected_acquisition"]["groups_created"]["truncated"] is True
    assert observed["truncated"] is True


@pytest.mark.parametrize("state", ["no-transport", "no-coordinator", "foreign"])
def test_missing_or_unknown_owner_does_not_become_known_zero_counts(state):
    calls = []

    def foreign_snapshot():
        calls.append("foreign")
        pytest.fail("an unknown admission implementation must not be observed")

    session = SimpleNamespace(ledger=RunLedger("synthetic-unowned"), memo={})
    if state == "no-coordinator":
        session.transport = SimpleNamespace()
    elif state == "foreign":
        session.transport = SimpleNamespace(coordinator=SimpleNamespace(
            snapshot=foreign_snapshot, protection_snapshot=foreign_snapshot,
        ))
    observed = build_workload_observation(
        session, None, outcome="SUCCEEDED", duration_ms=0,
    )
    assert "native_acquisition" not in observed
    assert "protected_acquisition" not in observed
    assert not calls
