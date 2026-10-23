"""Independent source preparation retains one owner and canonical evidence."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import replace
from datetime import date, datetime, timezone
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import time

import pytest

from tarzan import config, runtime
from tarzan.engine import metrics
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime import data_quality as dq
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session, current_session
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


@contextmanager
def owned(*, mode=RunMode.LIVE, capacity=4):
    context = RunContext(
        "synthetic-source-preparation", mode,
        None if mode is RunMode.LIVE else date(2026, 10, 23),
        datetime(2026, 10, 23, 15, tzinfo=timezone.utc),
    )
    ledger = RunLedger(context.attempt_id)
    settings = replace(
        ProviderTransportSettings.configured(), run_budget_seconds=60.0,
        optional_budget_seconds=10.0, publication_reserve_seconds=5.0,
    )
    budget = ProviderBudget(settings, ledger, capacity=capacity)
    session = RunSession(context, {}, ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        dq.reset()
        try:
            yield session
        finally:
            budget.close()


def engine(monkeypatch, history, benchmarks):
    def history_port(self, ctx):
        return history(ctx)

    history_port.__name__ = "_portfolio_history_from_orders"
    monkeypatch.setattr(metrics.MetricsEngine, "_portfolio_history_from_orders", history_port)
    monkeypatch.setattr(
        metrics.MetricsEngine, "_prepare_benchmark_context",
        lambda self, ctx: benchmarks(ctx),
    )
    monkeypatch.setattr(config, "max_workers", lambda: 4)
    result = metrics.MetricsEngine([], InvestorConfig(), orders=[SimpleNamespace()])
    result._computers = [
        lambda ctx: ctx.update(valuation=10),
        result._portfolio_history_from_orders,
        result._preprocess_benchmarks,
    ]
    result._build_result = lambda ctx: ctx
    return result


def stages(session):
    return [
        entry.payload for entry in session.ledger.entries
        if entry.entry_type is LedgerEntryType.STAGE
        and entry.payload.get("parallel_group") == "essential_source_preparation"
    ]


def test_independent_sources_overlap_but_dependents_read_only_joined_facts(monkeypatch):
    meeting = Barrier(2)
    contexts = []
    with owned() as session:
        def history(ctx):
            assert current_session() is session
            assert runtime.context() is session.context
            assert session.transport.coordinator.snapshot()["essential_demand"] == 1
            contexts.append(ctx)
            meeting.wait(timeout=2.0)
            ctx["history"] = 106

        def benchmarks(ctx):
            assert current_session() is session
            contexts.append(ctx)
            meeting.wait(timeout=2.0)
            ctx["benchmarks"] = 207
            return ()

        subject = engine(monkeypatch, history, benchmarks)
        subject._computers.append(lambda ctx: ctx.update(
            dependent=ctx["history"] + ctx["benchmarks"] + ctx["valuation"],
        ))
        result = subject.compute_all()
        assert result["dependent"] == 323
        assert contexts[0] is not contexts[1]
        assert all("valuation" not in ctx for ctx in contexts)
        assert [entry["stage"] for entry in stages(session)] == [
            "metrics:_portfolio_history_from_orders", "metrics:_preprocess_benchmarks",
        ]
        assert all(entry["outcome"] == "SUCCEEDED" for entry in stages(session))
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("capacity", [2, 4, 8])
def test_nested_native_work_never_multiplies_the_aggregate_cap(monkeypatch, capacity):
    lock = Lock()
    active = peak = 0
    with owned(capacity=capacity) as session:
        def native(timeout):
            nonlocal active, peak
            assert timeout > 0
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.002)
                return 106
            finally:
                with lock:
                    active -= 1

        def branch(ctx):
            with ThreadPoolExecutor(max_workers=6) as pool:
                pending = [
                    pool.submit(copy_context().run, session.transport.call, "yahoo", native)
                    for _ in range(6)
                ]
                assert sum(item.result(timeout=3.0) for item in pending) == 636

        result = engine(monkeypatch, branch, lambda ctx: (branch(ctx), ())[1]).compute_all()
        assert result["_degraded"] == []
        assert 2 <= peak <= capacity
        counters = session.transport.coordinator.snapshot()
        assert counters["active"] == counters["essential_demand"] == 0
        assert counters["acquired"] == counters["released"] == 12


@pytest.mark.parametrize("mode", [RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE])
def test_historical_modes_keep_the_serial_pipeline_without_live_requests(monkeypatch, mode):
    seen = []
    with owned(mode=mode) as session:
        subject = engine(
            monkeypatch, lambda ctx: seen.append("history"),
            lambda ctx: seen.append("benchmarks") or (),
        )
        subject.compute_all()
        assert seen == ["history", "benchmarks"]
        assert stages(session) == []
        assert session.transport.coordinator.snapshot()["requested"] == 0


def test_capacity_one_keeps_the_established_serial_pipeline(monkeypatch):
    seen = []
    with owned(capacity=1) as session:
        subject = engine(
            monkeypatch, lambda ctx: seen.append("history"),
            lambda ctx: seen.append("benchmarks") or (),
        )
        subject.compute_all()
        assert seen == ["history", "benchmarks"]
        assert stages(session) == []


def test_no_owner_and_injected_computers_are_not_reclassified_as_a_dag(monkeypatch):
    seen = []
    subject = engine(
        monkeypatch, lambda ctx: seen.append("history"),
        lambda ctx: seen.append("benchmarks") or (),
    )
    subject.compute_all()
    assert seen == ["history", "benchmarks"]
    with owned() as session:
        original = subject._computers[1]
        subject._computers[1] = lambda ctx: original(ctx)
        seen.clear()
        subject.compute_all()
        assert seen == ["history", "benchmarks"]
        assert stages(session) == []


def test_pure_holdings_and_ambient_optional_work_remain_serial(monkeypatch):
    with owned() as session:
        pure = metrics.MetricsEngine([], InvestorConfig())
        assert not any(pure._can_prepare_sources_together(i) for i in range(len(pure._computers)))
        subject = engine(monkeypatch, lambda ctx: None, lambda ctx: ())
        with session.transport.optional_work("caller-owned-optional"):
            subject.compute_all()
        assert stages(session) == []
        assert session.transport.optional_work_token() is None


@pytest.mark.parametrize("fast_branch", ["history", "benchmarks"])
def test_financial_diagnostics_are_adopted_in_pipeline_not_completion_order(
    monkeypatch, fast_branch,
):
    fast_finished = Event()
    with owned() as session:
        def emit(which, ctx):
            if which != fast_branch:
                assert fast_finished.wait(timeout=2.0)
            dq.warning("same-synthetic-source", which, context=which)
            ctx[which] = which
            if which == fast_branch:
                fast_finished.set()

        subject = engine(
            monkeypatch, lambda ctx: emit("history", ctx),
            lambda ctx: (emit("benchmarks", ctx), ())[1],
        )
        subject.compute_all()
        assert [issue.context for issue in session.diagnostics] == ["history", "benchmarks"]
        assert [record.original_failure["context"] for record in session.ledger.failure_records()] == [
            "history", "benchmarks",
        ]


def test_nested_worker_diagnostic_buffers_reenter_the_parent_phase_buffer(monkeypatch):
    with owned() as session:
        def history(ctx):
            deferred = []

            def child():
                dq.defer_into(deferred)
                dq.warning("synthetic-nested", "history", context="history")

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(copy_context().run, child).result(timeout=2.0)
            dq.replay(deferred)

        subject = engine(
            monkeypatch, history,
            lambda ctx: (dq.warning("synthetic-nested", "benchmarks", context="benchmarks"), ())[1],
        )
        subject.compute_all()
        assert [issue.context for issue in session.diagnostics] == ["history", "benchmarks"]


def test_distinct_context_copies_cannot_leak_local_mutation_into_owner(monkeypatch):
    local = ContextVar("synthetic-source-context", default="owner")
    meeting = Barrier(2)
    with owned() as session:
        def history(ctx):
            local.set("history")
            meeting.wait(timeout=2.0)
            assert local.get() == "history"
            ctx["history"] = current_session().context.attempt_id

        def benchmarks(ctx):
            local.set("benchmarks")
            meeting.wait(timeout=2.0)
            assert local.get() == "benchmarks"
            ctx["benchmarks"] = current_session().context.attempt_id
            return ()

        result = engine(monkeypatch, history, benchmarks).compute_all()
        assert result["history"] == result["benchmarks"] == session.context.attempt_id
        assert local.get() == "owner"


@pytest.mark.parametrize("failed", ["history", "benchmarks", "both"])
def test_ordinary_computer_errors_retain_partial_facts_and_canonical_degradation(monkeypatch, failed):
    with owned() as session:
        def history(ctx):
            ctx["history"] = 106
            if failed in ("history", "both"):
                raise ValueError("synthetic history fault")

        def benchmarks(ctx):
            ctx["benchmarks"] = 207
            if failed in ("benchmarks", "both"):
                raise TypeError("synthetic benchmark fault")
            return ()

        result = engine(monkeypatch, history, benchmarks).compute_all()
        expected = (
            ["_portfolio_history_from_orders", "_preprocess_benchmarks"]
            if failed == "both" else [
                "_portfolio_history_from_orders" if failed == "history" else "_preprocess_benchmarks",
            ]
        )
        assert result["_degraded"] == expected
        assert result["history"] == 106 and result["benchmarks"] == 207
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("failed", ["history", "benchmarks", "both"])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_canonical_interruption_is_raised_only_after_all_children_and_evidence_join(
    monkeypatch, failed, kind,
):
    meeting = Barrier(2)
    exited = []
    history_error = kind("synthetic history interruption")
    benchmark_error = kind("synthetic benchmark interruption")
    with owned() as session:
        def history(ctx):
            meeting.wait(timeout=2.0)
            exited.append("history")
            if failed in ("history", "both"):
                raise history_error

        def benchmarks(ctx):
            meeting.wait(timeout=2.0)
            exited.append("benchmarks")
            if failed in ("benchmarks", "both"):
                raise benchmark_error
            return ()

        with pytest.raises(kind) as caught:
            engine(monkeypatch, history, benchmarks).compute_all()
        assert caught.value is (benchmark_error if failed == "benchmarks" else history_error)
        assert set(exited) == {"history", "benchmarks"}
        assert len(stages(session)) == 2
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


def test_missing_essential_reference_failure_is_published_after_history_diagnostics(monkeypatch):
    with owned() as session:
        def history(ctx):
            dq.warning("synthetic-history", "history evidence", context="history")

        result = engine(
            monkeypatch, history,
            lambda ctx: ctx.update(_degraded=["_preprocess_benchmarks"]) or ("Missing core",),
        ).compute_all()
        assert result["_degraded"] == ["_preprocess_benchmarks"]
        failures = session.ledger.failure_records()
        assert [item.stable_code for item in failures] == [
            "DATA_QUALITY_WARNING", "ESSENTIAL_BENCHMARK_UNAVAILABLE",
        ]
        assert PublicationEvaluator.evaluate(failures).decision is \
            PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE


def test_overlapping_context_fields_fail_closed_without_last_writer_wins(monkeypatch):
    with owned() as session:
        result = engine(
            monkeypatch, lambda ctx: ctx.update(valuation=999),
            lambda ctx: ctx.update(benchmarks=207) or (),
        ).compute_all()
        assert result["valuation"] == 10
        assert result["benchmarks"] == 207
        assert result["_degraded"] == ["_portfolio_history_from_orders"]
        assert any(item.stable_code == "METRIC_PHASE_CONTEXT_CONFLICT"
                   for item in session.ledger.failure_records())
        assert PublicationEvaluator.evaluate(session.ledger.failure_records()).decision is \
            PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE


def test_context_collision_cannot_replace_a_worker_primary_interruption(monkeypatch):
    error = KeyboardInterrupt("synthetic original cancellation")
    with owned():
        def history(ctx):
            ctx["valuation"] = 999
            raise error

        with pytest.raises(KeyboardInterrupt) as caught:
            engine(monkeypatch, history, lambda ctx: ()).compute_all()
        assert caught.value is error


def test_custom_preparer_and_subclass_keep_the_legacy_context_contract(monkeypatch):
    from types import MethodType

    with owned() as session:
        subject = engine(monkeypatch, lambda ctx: None, lambda ctx: ())

        def needs_valuation(self, ctx):
            ctx["benchmarks"] = ctx["valuation"]
            return ()

        subject._prepare_benchmark_context = MethodType(needs_valuation, subject)
        assert not subject._can_prepare_sources_together(1)
        assert subject.compute_all()["benchmarks"] == 10

        class Extended(metrics.MetricsEngine):
            _prepare_benchmark_context = needs_valuation

        extended = Extended([], InvestorConfig(), orders=[SimpleNamespace()])
        extended._computers = [
            lambda ctx: ctx.update(valuation=10),
            extended._portfolio_history_from_orders, extended._preprocess_benchmarks,
        ]
        extended._build_result = lambda ctx: ctx
        assert not extended._can_prepare_sources_together(1)
        assert extended.compute_all()["benchmarks"] == 10
        assert stages(session) == []


@pytest.mark.parametrize("fault", ["timing", "adoption"])
def test_observation_failure_cannot_mask_the_joined_source_primary(monkeypatch, fault):
    from tarzan.runtime.workload import StageClock

    error = KeyboardInterrupt("synthetic source primary")
    finished = Event()
    with owned() as session:
        def history(ctx):
            raise error

        def benchmarks(ctx):
            finished.set()
            return ()

        subject = engine(monkeypatch, history, benchmarks)
        subject._computers = subject._computers[1:]
        if fault == "timing":
            monkeypatch.setattr(
                StageClock, "fields",
                lambda self: (_ for _ in ()).throw(ValueError("synthetic timing failure")),
            )
        else:
            original = session.ledger.append

            def append(kind, payload):
                if payload.get("parallel_group") == "essential_source_preparation":
                    raise ValueError("synthetic adoption observation failure")
                return original(kind, payload)

            monkeypatch.setattr(session.ledger, "append", append)
        with pytest.raises(KeyboardInterrupt) as caught:
            subject.compute_all()
        assert caught.value is error
        assert finished.is_set()
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


def test_optional_branches_cannot_take_the_declared_essential_slot(monkeypatch):
    optional_started, essential_started, release = Event(), Event(), Event()
    lock = Lock()
    count = 0
    with owned(capacity=4) as session:
        def optional_native(timeout):
            nonlocal count
            with lock:
                count += 1
                if count == 3:
                    optional_started.set()
            assert release.wait(timeout=3.0)
            return 106

        def history(ctx):
            assert optional_started.wait(timeout=2.0)

            def essential_native(timeout):
                essential_started.set()
                assert release.wait(timeout=3.0)
                return 207

            ctx["history"] = session.transport.call("borsa", essential_native)

        def benchmarks(ctx):
            with session.transport.optional_work("synthetic-reference-extra"):
                with ThreadPoolExecutor(max_workers=6) as children:
                    pending = [
                        children.submit(
                            copy_context().run, session.transport.call, "justetf", optional_native,
                        )
                        for _ in range(6)
                    ]
                    ctx["benchmarks"] = sum(item.result(timeout=4.0) for item in pending)
            return ()

        subject = engine(monkeypatch, history, benchmarks)
        with ThreadPoolExecutor(max_workers=1) as parent:
            pending = parent.submit(copy_context().run, subject.compute_all)
            try:
                assert essential_started.wait(timeout=2.0)
                observed = session.transport.coordinator.snapshot()
                assert observed["active_optional"] == 3
                assert observed["active"] == observed["peak"] == 4
                assert observed["essential_demand"] == 1
            finally:
                release.set()
            result = pending.result(timeout=5.0)
        assert result["history"] == 207 and result["benchmarks"] == 636
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("fault", ["submit", "canceled"])
def test_executor_fault_joins_every_accepted_callback_before_scope_release(monkeypatch, fault):
    release, finished = Event(), Event()
    original_pool = ThreadPoolExecutor
    error = RuntimeError("synthetic second submission failed")

    class FaultPool:
        def __init__(self, **kwargs):
            self.pool = original_pool(**kwargs)
            self.count = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                release.set()
                if fault == "submit":
                    raise error
                canceled = Future()
                assert canceled.cancel()
                return canceled
            return self.pool.submit(*args, **kwargs)

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    with owned() as session:
        def history(ctx):
            assert release.wait(timeout=2.0)
            finished.set()
            ctx["history"] = 106

        subject = engine(monkeypatch, history, lambda ctx: pytest.fail("unstarted branch"))
        monkeypatch.setattr(metrics, "ThreadPoolExecutor", FaultPool)
        with pytest.raises(RuntimeError if fault == "submit" else CancelledError) as caught:
            subject.compute_all()
        if fault == "submit":
            assert caught.value is error
        assert finished.is_set()
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        assert stages(session)[0]["outcome"] == "SUCCEEDED"


@pytest.mark.parametrize("fault", ["submit", "canceled", "failed_future"])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_ordinary_executor_fault_cannot_mask_joined_source_cancellation(monkeypatch, fault, kind):
    original_pool = ThreadPoolExecutor
    release, finished = Event(), Event()
    primary = kind("synthetic accepted-source cancellation")

    class FaultPool:
        def __init__(self, **kwargs):
            self.pool = original_pool(**kwargs)
            self.count = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                release.set()
                if fault == "submit":
                    raise ValueError("synthetic second submission defect")
                failed = Future()
                if fault == "canceled":
                    assert failed.cancel()
                else:
                    failed.set_exception(ValueError("synthetic setup defect"))
                return failed
            return self.pool.submit(*args, **kwargs)

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    with owned() as session:
        def history(ctx):
            assert release.wait(timeout=2.0)
            finished.set()
            raise primary

        subject = engine(monkeypatch, history, lambda ctx: pytest.fail("unstarted branch"))
        monkeypatch.setattr(metrics, "ThreadPoolExecutor", FaultPool)
        with pytest.raises(kind) as caught:
            subject.compute_all()
        assert caught.value is primary
        assert finished.is_set()
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        assert stages(session)[0]["outcome"] == "INTERRUPTED"


@pytest.mark.parametrize("observer_kind", [ValueError, KeyboardInterrupt])
def test_secondary_worker_observer_cannot_replace_source_primary(monkeypatch, observer_kind):
    from tarzan.runtime.workload import StageClock

    primary = SystemExit("synthetic source primary")
    secondary = observer_kind("synthetic secondary log-handler failure")
    finished = Event()
    with owned() as session:
        def history(ctx):
            raise primary

        def benchmarks(ctx):
            finished.set()
            return ()

        subject = engine(monkeypatch, history, benchmarks)
        subject._computers = subject._computers[1:]
        monkeypatch.setattr(
            StageClock, "fields",
            lambda self: (_ for _ in ()).throw(ValueError("synthetic timing error")),
        )
        monkeypatch.setattr(
            metrics.logger, "warning",
            lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
        )
        with pytest.raises(SystemExit) as caught:
            subject.compute_all()
        assert caught.value is primary
        assert finished.is_set()
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        assert stages(session)[0]["outcome"] == "INTERRUPTED"


def test_new_attempt_does_not_reuse_prior_context_degradation_or_owner(monkeypatch):
    owners = []
    subject = engine(
        monkeypatch, lambda ctx: owners.append(current_session()) or ctx.update(history=106),
        lambda ctx: ctx.update(benchmarks=207) or (),
    )
    for _ in range(2):
        with owned() as session:
            assert subject.compute_all()["_degraded"] == []
            assert session.transport.coordinator.snapshot()["essential_demand"] == 0
    assert owners[0] is not owners[1]
