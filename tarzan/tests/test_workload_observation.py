"""Bounded descriptive workload evidence and network-free harness tests."""

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.workload import (
    MAX_RECORDED_INTEGER,
    build_workload_observation,
    run_network_free_workload_harness,
)


def _count(observation: dict, name: str) -> int:
    return int(observation["counts"][name]["value"])


# **Validates: Requirements 2.15, 2.16, 3.15, 3.16**
def test_network_free_harness_exercises_explicit_kinds_and_categories(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("workload harness attempted external network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    first = run_network_free_workload_harness()
    second = run_network_free_workload_harness()

    assert first == second
    assert first["network_access"] == "NOT_USED"
    assert first["drive_access"] == "NOT_USED"
    assert first["gemini_access"] == "NOT_USED"
    assert first["delivery_access"] == "NOT_USED"
    assert first["instrument_kinds"] == ["STOCK", "ETF", "BOND", "CASH"]
    assert first["tracked_categories"] == [
        "Equities",
        "Fixed Income",
        "Cash & Cash Equivalents",
        "Gold",
        "Commodities",
        "Crypto",
        "Alternative",
    ]
    observation = first["observation"]
    assert _count(observation, "holdings") == 4
    assert _count(observation, "capability_events") == 35
    assert observation["descriptive_only"] is True
    assert observation["scale_gate"] is False
    assert observation["affects_analysis_id"] is False
    assert observation["affects_financial_decisions"] is False
    json.dumps(first, allow_nan=False)


# **Validates: Requirements 2.15, 3.15**
def test_workload_counts_are_bounded_and_do_not_change_financial_output():
    ledger = RunLedger("attempt")
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "provider": "preferred",
        "operation": "current_quote",
        "outcome": "FAILED",
    })
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "provider": "validated_cache",
        "operation": "current_quote",
        "outcome": "SUCCEEDED",
        "selected_fallback": "cache",
    })
    ledger.append(LedgerEntryType.CAPABILITY, {
        "kind": "ETF",
        "capability": "PRICING_VALUATION",
        "availability": "AVAILABLE",
    })
    ledger.append(LedgerEntryType.PLAN, {
        "plan": "Buy only",
        "actions": [{"side": "BUY"}, {"side": "BUY"}],
    })
    ledger.append(LedgerEntryType.STAGE, {
        "stage": "configuration",
        "outcome": "SUCCEEDED",
    })
    metrics = PortfolioMetrics(total_value=100.0, invested_value=100.0)
    metrics.portfolio_history = tuple(range(3))
    metrics.holding_histories = {"ETF": {"history": tuple(range(5))}}
    metrics.benchmark_histories = {"ACWI": tuple(range(7))}
    before = metrics.to_summary_dict()
    session = SimpleNamespace(
        ledger=ledger,
        memo={
            "workload": {
                "orders": MAX_RECORDED_INTEGER + 100,
                "holdings": 1,
                "rebalance_seeds": 0,
                "diagnostics": 2,
            }
        },
    )

    observation = build_workload_observation(
        session,
        metrics,
        outcome="SUCCEEDED",
        duration_ms=12.5,
    )

    assert metrics.to_summary_dict() == before
    assert _count(observation, "orders") == MAX_RECORDED_INTEGER
    assert observation["counts"]["orders"]["truncated"] is True
    assert _count(observation, "portfolio_history_points") == 3
    assert _count(observation, "holding_history_points") == 5
    assert _count(observation, "benchmark_history_points") == 7
    assert _count(observation, "provider_attempts") == 2
    assert _count(observation, "cache_attempts") == 1
    assert _count(observation, "capability_events") == 1
    assert _count(observation, "plan_searches") == 1
    assert _count(observation, "plan_actions") == 2
    assert observation["truncated"] is True
    assert observation["stages"][-1]["duration_ms"]["value"] == 12.5
