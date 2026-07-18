"""Bounded descriptive workload evidence with no product scale gate.

The observations in this module are operational evidence only. Recording bounds
protect JSON consumers; they are not supported portfolio-size, latency,
throughput, memory, or availability thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from tarzan.runtime.ledger import LedgerEntryType, RunLedger


WORKLOAD_TELEMETRY_SCHEMA_VERSION = "1.0"
# Largest integer represented exactly by interoperable JSON/JavaScript readers.
MAX_RECORDED_INTEGER = (1 << 53) - 1
MAX_STAGE_OBSERVATIONS = 64
MAX_LABEL_CHARACTERS = 96


@dataclass(frozen=True)
class BoundedMeasure:
    """One nonnegative observation bounded for interoperable serialization."""

    value: int | float
    unit: str
    maximum_recorded: int = MAX_RECORDED_INTEGER
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "minimum": 0,
            "maximum_recorded": self.maximum_recorded,
            "truncated": self.truncated,
        }


def _bounded(value: int | float, unit: str, *, integral: bool) -> BoundedMeasure:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{unit} observation must be finite and nonnegative")
    truncated = numeric > MAX_RECORDED_INTEGER
    recorded = min(numeric, float(MAX_RECORDED_INTEGER))
    return BoundedMeasure(
        value=int(recorded) if integral else round(recorded, 3),
        unit=unit,
        truncated=truncated,
    )


def _bounded_count(value: object) -> BoundedMeasure:
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        numeric = 0
    return _bounded(max(numeric, 0), "count", integral=True)


def _safe_length(value: object) -> int:
    if value is None:
        return 0
    try:
        return max(int(len(value)), 0)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0


def _history_lengths(metrics: object) -> tuple[int, int, int]:
    if metrics is None:
        return 0, 0, 0
    portfolio = _safe_length(getattr(metrics, "portfolio_history", None))
    holdings = 0
    for item in (getattr(metrics, "holding_histories", None) or {}).values():
        history = item.get("history") if isinstance(item, Mapping) else item
        holdings += _safe_length(history)
    benchmarks = sum(
        _safe_length(item)
        for item in (getattr(metrics, "benchmark_histories", None) or {}).values()
    )
    return portfolio, holdings, benchmarks


def _contains_cache_label(payload: Mapping[str, Any]) -> bool:
    for key in ("source", "provider", "fallback_rung", "selected_fallback", "operation"):
        value = payload.get(key)
        if value is not None and "cache" in str(value).casefold():
            return True
    return False


def _bounded_label(value: object) -> tuple[str, bool]:
    label = str(value or "UNKNOWN")
    return label[:MAX_LABEL_CHARACTERS], len(label) > MAX_LABEL_CHARACTERS


def build_workload_observation(
    session: object,
    metrics: object,
    *,
    outcome: str,
    duration_ms: float,
    scenario: str = "analysis_run",
) -> dict[str, Any]:
    """Build one bounded, non-secret, descriptive ledger payload.

    This function only observes existing memo and ledger projections. It does
    not mutate metrics, choose providers, alter optimization, or participate in
    Analysis ID construction.
    """

    ledger = getattr(session, "ledger")
    entries = tuple(ledger.entries)
    memo = dict(getattr(session, "memo", {}).get("workload", {}))
    provider_entries = tuple(
        entry for entry in entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    )
    capability_entries = tuple(
        entry for entry in entries
        if entry.entry_type is LedgerEntryType.CAPABILITY
    )
    plan_entries = tuple(
        entry for entry in entries
        if entry.entry_type is LedgerEntryType.PLAN
    )
    stage_entries = tuple(
        entry for entry in entries
        if entry.entry_type is LedgerEntryType.STAGE
    )

    portfolio_points, holding_points, benchmark_points = _history_lengths(metrics)
    plan_actions = sum(_safe_length(entry.payload.get("actions")) for entry in plan_entries)
    counts = {
        "orders": memo.get("orders", 0),
        "holdings": memo.get("holdings", 0),
        "rebalance_seeds": memo.get("rebalance_seeds", 0),
        "diagnostics": memo.get("diagnostics", 0),
        "portfolio_history_points": portfolio_points,
        "holding_history_points": holding_points,
        "benchmark_history_points": benchmark_points,
        "provider_attempts": len(provider_entries),
        "cache_attempts": sum(
            1 for entry in provider_entries if _contains_cache_label(entry.payload)
        ),
        "capability_events": len(capability_entries),
        "stage_events": len(stage_entries) + 1,
        "plan_searches": len(plan_entries),
        "plan_actions": plan_actions,
    }
    bounded_counts = {
        name: _bounded_count(value).to_dict()
        for name, value in sorted(counts.items())
    }

    stage_observations: list[dict[str, Any]] = []
    label_truncated = False
    for entry in stage_entries[: MAX_STAGE_OBSERVATIONS - 1]:
        stage, stage_cut = _bounded_label(entry.payload.get("stage"))
        stage_outcome, outcome_cut = _bounded_label(entry.payload.get("outcome"))
        label_truncated = label_truncated or stage_cut or outcome_cut
        raw_duration = entry.payload.get("duration_ms")
        stage_observations.append({
            "stage": stage,
            "outcome": stage_outcome,
            "duration_ms": (
                _bounded(raw_duration, "milliseconds", integral=False).to_dict()
                if raw_duration is not None else None
            ),
        })
    terminal_stage, terminal_stage_cut = _bounded_label("orchestration")
    terminal_outcome, terminal_outcome_cut = _bounded_label(outcome)
    label_truncated = label_truncated or terminal_stage_cut or terminal_outcome_cut
    stage_observations.append({
        "stage": terminal_stage,
        "outcome": terminal_outcome,
        "duration_ms": _bounded(duration_ms, "milliseconds", integral=False).to_dict(),
    })

    scenario_label, scenario_cut = _bounded_label(scenario)
    truncated = (
        len(stage_entries) + 1 > MAX_STAGE_OBSERVATIONS
        or label_truncated
        or scenario_cut
        or any(item["truncated"] for item in bounded_counts.values())
        or any(
            stage["duration_ms"] is not None and stage["duration_ms"]["truncated"]
            for stage in stage_observations
        )
    )
    return {
        "schema_version": WORKLOAD_TELEMETRY_SCHEMA_VERSION,
        "event": "DESCRIPTIVE_WORKLOAD_OBSERVATION",
        "scenario": scenario_label,
        "outcome": terminal_outcome,
        "counts": bounded_counts,
        "stages": stage_observations,
        "recording_bounds": {
            "maximum_recorded_integer": MAX_RECORDED_INTEGER,
            "maximum_stage_observations": MAX_STAGE_OBSERVATIONS,
            "maximum_label_characters": MAX_LABEL_CHARACTERS,
        },
        "truncated": truncated,
        "descriptive_only": True,
        "scale_gate": False,
        "affects_analysis_id": False,
        "affects_financial_decisions": False,
    }


def run_network_free_workload_harness() -> dict[str, Any]:
    """Exercise explicit kind/category declarations with no external adapters.

    The fixed fixture records what was exercised; it does not infer support at
    larger dimensions and does not establish a pass/fail performance target.
    """

    from tarzan.instruments.registry import (
        InstrumentCapability,
        InstrumentKind,
        TypeEvidenceGateway,
        default_instrument_registry,
        default_tracked_category_registry,
    )

    gateway = TypeEvidenceGateway()
    instruments = default_instrument_registry()
    categories = default_tracked_category_registry()
    ledger = RunLedger("network-free-workload-harness")

    exercised_kinds: list[str] = []
    for kind in InstrumentKind:
        profile = instruments.resolve(gateway.resolve(kind.value))
        exercised_kinds.append(kind.value)
        for capability in InstrumentCapability:
            result = profile.capability(capability)
            ledger.append(LedgerEntryType.CAPABILITY, {
                "fixture": "network_free",
                "kind": kind.value,
                "capability": capability.value,
                "support": result.support.value,
                "availability": result.availability.value,
            })

    exercised_categories = list(categories.names)
    for category in exercised_categories:
        profile = categories.get(category)
        ledger.append(LedgerEntryType.CAPABILITY, {
            "fixture": "network_free",
            "tracked_category": category,
            "sector_support": profile.sector_support.value if profile else "UNAVAILABLE",
            "rebalancing_support": (
                profile.rebalancing_support.value if profile else "UNAVAILABLE"
            ),
        })

    ledger.append(LedgerEntryType.STAGE, {
        "stage": "explicit_registry_fixture",
        "outcome": "SUCCEEDED",
    })
    fixture_metrics = SimpleNamespace(
        portfolio_history=tuple(range(8)),
        holding_histories={
            kind: {"history": tuple(range(4))} for kind in exercised_kinds
        },
        benchmark_histories={},
    )
    fixture_session = SimpleNamespace(
        ledger=ledger,
        memo={
            "workload": {
                "orders": 8,
                "holdings": len(exercised_kinds),
                "rebalance_seeds": 0,
                "diagnostics": 0,
            }
        },
    )
    observation = build_workload_observation(
        fixture_session,
        fixture_metrics,
        outcome="SUCCEEDED",
        duration_ms=0.0,
        scenario="network_free_explicit_kinds_and_categories",
    )
    return {
        "harness_schema_version": WORKLOAD_TELEMETRY_SCHEMA_VERSION,
        "network_access": "NOT_USED",
        "drive_access": "NOT_USED",
        "gemini_access": "NOT_USED",
        "delivery_access": "NOT_USED",
        "instrument_kinds": exercised_kinds,
        "tracked_categories": exercised_categories,
        "observation": observation,
        "claim": "DESCRIPTIVE_ONLY_NO_SCALE_INFERENCE",
    }
