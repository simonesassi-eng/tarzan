"""Clock-specific observations are descriptive, inclusive and bounded."""

from types import SimpleNamespace

from tarzan.runtime import workload
from tarzan.runtime.ledger import LedgerEntryType, RunLedger


def test_stage_clock_keeps_wall_owner_thread_and_process_cpu_distinct(monkeypatch):
    wall, thread, process = iter((10.0, 13.0)), iter((4.0, 4.5)), iter((7.0, 8.75))
    monkeypatch.setattr(workload.time, "perf_counter", lambda: next(wall))
    monkeypatch.setattr(workload.time, "thread_time", lambda: next(thread))
    monkeypatch.setattr(workload.time, "process_time", lambda: next(process))
    assert workload.StageClock.capture().fields() == {
        "duration_ms": 3000.0, "thread_cpu_ms": 500.0, "process_cpu_ms": 1750.0,
    }


def test_explicit_clock_does_not_install_profiler_or_mutate_financial_facts(monkeypatch):
    import sys
    import threading

    monkeypatch.setattr(sys, "setprofile", lambda _: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(threading, "setprofile", lambda _: (_ for _ in ()).throw(AssertionError()))
    fields = workload.StageClock.capture().fields()
    assert set(fields) == {"duration_ms", "thread_cpu_ms", "process_cpu_ms"}
    assert all(value >= 0 for value in fields.values())


def test_cpu_observation_is_bounded_and_declares_inclusive_worker_scope():
    ledger = RunLedger("synthetic-clocks")
    ledger.append(LedgerEntryType.STAGE, {
        "stage": "synthetic_stage", "outcome": "SUCCEEDED", "duration_ms": 12.5,
        "thread_cpu_ms": 5.5, "process_cpu_ms": workload.MAX_RECORDED_INTEGER + 1024,
    })
    session = SimpleNamespace(ledger=ledger, memo={})
    observation = workload.build_workload_observation(
        session, None, outcome="SUCCEEDED", duration_ms=20,
    )
    stage = observation["stages"][0]
    assert stage["duration_ms"]["value"] == 12.5
    assert stage["thread_cpu_ms"]["value"] == 5.5
    assert stage["process_cpu_ms"]["value"] == workload.MAX_RECORDED_INTEGER
    assert stage["process_cpu_ms"]["truncated"] is True
    assert observation["truncated"] is True
    assert observation["cpu_clock_semantics"] == {
        "thread_cpu_ms": "stage_owner_thread",
        "process_cpu_ms": "all_process_threads_in_stage_wall_interval",
        "overlapping_measurements_are_inclusive": True,
        "exclusive_attribution": False,
    }
    assert observation["affects_analysis_id"] is False
    assert observation["affects_financial_decisions"] is False


def test_old_workload_without_cpu_fields_retains_optional_field_contract():
    ledger = RunLedger("synthetic-old-clocks")
    ledger.append(LedgerEntryType.STAGE, {
        "stage": "synthetic_stage", "outcome": "SUCCEEDED", "duration_ms": 12.5,
    })
    observation = workload.build_workload_observation(
        SimpleNamespace(ledger=ledger, memo={}), None,
        outcome="SUCCEEDED", duration_ms=20,
    )
    assert "cpu_clock_semantics" not in observation
    assert all(
        not {"thread_cpu_ms", "process_cpu_ms"} & set(stage)
        for stage in observation["stages"]
    )
