"""Native speculative plans never acquire source or publication authority."""

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import fields, replace
from pathlib import Path
from struct import unpack
from threading import Barrier, Event, current_thread
from types import FunctionType, MethodType, SimpleNamespace

import pandas as pd
import pytest

from tarzan.engine import allocations, metrics, owned_planning as planning, rebalancer
from tarzan.instruments.valuation import holding_kind_resolution
from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime import audit
from tarzan.runtime.ledger import LedgerEntryType
from tarzan.runtime.session import current_session
from tarzan.tests.test_owned_source_preparation import engine, owned, stages


def book():
    return [
        Holding(
            "SYNTHETIC-EQUITY", "EQ", 6.0, 500.0, 600.0, "EUR",
            name="Synthetic equity", instrument_type="ETF",
            current_price=100.0, current_value=600.0, gain_pct=20.0,
            asset_class=AssetClass.EQUITIES, geography=Geography.USA,
            geo_breakdown={Geography.USA: 100.0},
            class_breakdown={AssetClass.EQUITIES: 100.0}, target_portfolio=50.0,
        ),
        Holding(
            "SYNTHETIC-BOND", "BD", 200.0, 190.0, 200.0, "EUR",
            name="Synthetic government bond", instrument_type="GOVERNMENT_BOND",
            current_price=1.0, current_value=200.0, gain_pct=5.0,
            asset_class=AssetClass.FIXED_INCOME, target_portfolio=30.0,
        ),
        Holding(
            "SYNTHETIC-CASH", "CASH", 200.0, 200.0, 200.0, "EUR",
            instrument_type="CASH", current_price=1.0, current_value=200.0,
            asset_class=AssetClass.CASH_EQUIVALENTS,
        ),
    ]


def inputs():
    holdings = book()
    config = InvestorConfig(
        target_use_per_holding_only=True,
        rebalancing_lump_sum_amount_eur=100.0,
        rebalancing_transaction_fee_buy_eur=1.0,
        rebalancing_transaction_fee_sell_eur=2.0,
        rebalancing_capital_gains_tax_standard_pctg=26.0,
        rebalancing_capital_gains_tax_government_pctg=12.5,
        target_cash_buffer_eur=100.0,
    )
    result = planning.snapshot_inputs(holdings, config, 1000.0, 100.0)
    assert result is not None
    return holdings, config, result


def test_real_native_generated_methods_are_eligible():
    assert planning.native_solver_available()
    assert any(
        isinstance(port[2], FunctionType) and port[2].__code__.co_filename == "<string>"
        for port in planning._NATIVE_PORTS
    )


@pytest.mark.parametrize("module,name", [
    (rebalancer, "compute_unified_rebalancing"),
    (rebalancer, "optimize_local_search"),
    (rebalancer._Cost, "__call__"),
    (rebalancer._CashMechanics, "prepare"),
    (rebalancer, "LSParams"),
    (allocations, "holding_class_breakdown"),
])
def test_injected_native_dependency_preserves_serial_contract(monkeypatch, module, name):
    monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail("foreign call"))
    assert not planning.native_solver_available()
    assert planning.snapshot_inputs(book(), InvestorConfig(), 1000.0, None) is None


@pytest.mark.parametrize("ports", [(), [], None, (("malformed",),)])
def test_bad_native_manifest_fails_closed(monkeypatch, ports):
    monkeypatch.setattr(planning, "_NATIVE_PORTS", ports)
    assert not planning.native_solver_available()


def test_consumed_native_holding_fields_remain_in_explicit_snapshot():
    nodes = [
        ast.parse(Path(module.__file__).read_text())
        for module in (rebalancer, allocations)
    ]
    import inspect
    nodes.append(ast.parse(inspect.getsource(holding_kind_resolution)))
    consumed = set()
    for tree in nodes:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in {"h", "holding"}:
                    consumed.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "getattr" and len(node.args) >= 2:
                    subject, field = node.args[:2]
                    if (
                        isinstance(subject, ast.Name) and subject.id in {"h", "holding"}
                        and isinstance(field, ast.Constant) and isinstance(field.value, str)
                    ):
                        consumed.add(field.value)
    assert consumed
    assert consumed <= set(planning.HOLDING_FIELDS)


def test_snapshot_isolated_and_does_not_copy_unused_history():
    holdings, config, snapshot = inputs()
    holdings[0].price_history = pd.Series([100.0, 105.0])
    holdings[0].geo_breakdown[Geography.JAPAN] = 20.0
    config.invested_allocation_targets_pctg["Equities"] = 7.0
    assert snapshot.holdings[0] is not holdings[0]
    assert snapshot.holdings[0].price_history is None
    assert snapshot.holdings[0].geo_breakdown == {Geography.USA: 100.0}
    assert snapshot.config is not config
    assert snapshot.config.invested_allocation_targets_pctg != config.invested_allocation_targets_pctg


def test_unconsumed_dynamic_config_state_is_not_copied_or_executed():
    class DoNotCopy:
        def __deepcopy__(self, _memo):
            raise RuntimeError("unused external state must not be invoked")

    config = InvestorConfig()
    config.external_diagnostic = DoNotCopy()
    result = planning.snapshot_inputs(book(), config, 1000.0, None)
    assert result is not None
    assert not hasattr(result.config, "external_diagnostic")


def test_native_policy_copy_does_not_execute_custom_copy_hook(monkeypatch):
    config = InvestorConfig()
    monkeypatch.setattr(
        planning.ProviderQualityPolicy, "__deepcopy__",
        lambda *a: (_ for _ in ()).throw(RuntimeError("copy hook must not execute")),
        raising=False,
    )
    assert planning.snapshot_inputs(book(), config, 1000.0, None) is not None


def test_recursive_input_keeps_the_original_owner_computation():
    holdings = book()
    cycle = {}
    cycle["self"] = cycle
    holdings[0].class_breakdown = cycle
    assert planning.snapshot_inputs(holdings, InvestorConfig(), 1000.0, None) is None


@pytest.mark.parametrize("value", [object(), {1, 2}, SimpleNamespace(), pd.Series([1.0])])
def test_unsupported_facts_keep_original_computation(value):
    holdings = book()
    holdings[0].name = value
    assert planning.snapshot_inputs(holdings, InvestorConfig(), 1000.0, None) is None


def test_signature_preserves_types_float_bits_and_order():
    assert planning._freeze(1) != planning._freeze(1.0)
    assert planning._freeze(True) != planning._freeze(1)
    assert planning._freeze(0.0) != planning._freeze(-0.0)
    nan1 = unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    nan2 = unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    assert planning._freeze(nan1) != planning._freeze(nan2)
    assert planning._freeze({"a": 1.0, "b": 2.0}) != planning._freeze({"b": 2.0, "a": 1.0})


def test_native_two_plan_outputs_are_identical_and_consumed_once():
    holdings, config, snapshot = inputs()
    result = planning.prepare_plans(snapshot)
    assert result is not None
    for no_sell in (True, False):
        variant = replace(config, rebalancing_no_sell=no_sell)
        expected = rebalancer.compute_unified_rebalancing(
            holdings, variant, 1000.0, lump_sum=100.0,
        )
        actual = result.take(holdings, variant, 1000.0, 100.0)
        assert actual == expected
        assert result.take(holdings, variant, 1000.0, 100.0) is None
    assert not result.variants


@pytest.mark.parametrize("field", planning.HOLDING_FIELDS)
def test_every_consumed_field_change_forces_owner_recomputation(field):
    holdings, config, snapshot = inputs()
    signature = planning._signature(snapshot.holdings, replace(config, rebalancing_no_sell=True), 1000.0, 100.0)
    prepared = planning.PreparedPlans({True: (signature, ("synthetic", "result"))})
    setattr(holdings[0], field, object())
    assert prepared.take(holdings, replace(config, rebalancing_no_sell=True), 1000.0, 100.0) is None


@pytest.mark.parametrize("changed", ["total", "lump", "holding_order", "config", "mapping_order"])
def test_changed_complete_inputs_never_adopt_stale_plan(changed):
    holdings, config, snapshot = inputs()
    config = replace(config, rebalancing_no_sell=True)
    signature = planning._signature(snapshot.holdings, config, 1000.0, 100.0)
    result = planning.PreparedPlans({True: (signature, ("synthetic", "result"))})
    total, lump = 1000.0, 100.0
    if changed == "total":
        total += 1.0
    elif changed == "lump":
        lump += 1.0
    elif changed == "holding_order":
        holdings.reverse()
    elif changed == "config":
        config = replace(config, rebalancing_transaction_fee_buy_eur=2.0)
    else:
        config = replace(config, invested_allocation_targets_pctg=dict(
            reversed(list(config.invested_allocation_targets_pctg.items())),
        ))
    assert result.take(holdings, config, total, lump) is None


@pytest.mark.parametrize("changed", ["mechanics_mapping", "search_defaults", "government_pattern"])
def test_changed_native_policy_or_search_facts_cannot_reuse_old_results(monkeypatch, changed):
    holdings, config, snapshot = inputs()
    config = replace(config, rebalancing_no_sell=True)
    key = planning._signature(snapshot.holdings, config, 1000.0, 100.0)
    prepared = planning.PreparedPlans({True: (key, ("synthetic", "result"))})
    if changed == "mechanics_mapping":
        monkeypatch.setattr(
            planning.registry.TypeEvidenceGateway, "_MAPPING",
            {**planning.registry.TypeEvidenceGateway._MAPPING, "SYNTHETIC": planning.registry.InstrumentKind.BOND},
        )
    elif changed == "search_defaults":
        init = rebalancer.LSParams.__init__
        defaults = init.__defaults__
        assert defaults
        updated = list(defaults)
        index = [field.name for field in fields(rebalancer.LSParams)].index("seed")
        updated[index] += 1
        monkeypatch.setattr(init, "__defaults__", tuple(updated))
    else:
        import re
        monkeypatch.setattr(planning.tax, "_GOV_BOND_NAME", re.compile("SYNTHETIC"))
    assert prepared.take(holdings, config, 1000.0, 100.0) is None


def test_worker_timing_is_adopted_by_owner_only_and_is_not_financial_authority(monkeypatch):
    holdings, config, snapshot = inputs()
    with owned() as session:
        subject = metrics.MetricsEngine(holdings, config)
        subject._prepared_plans = planning.prepare_plans(snapshot)
        assert not any(
            entry.payload.get("stage") == "metrics:_prepare_rebalancing"
            for entry in session.ledger.entries
        )
        subject._record_prepared_plan_timing()
        rows = [
            entry.payload for entry in session.ledger.entries
            if entry.payload.get("stage") == "metrics:_prepare_rebalancing"
        ]
        assert len(rows) == 1
        assert rows[0]["duration_ms"] + 1 >= rows[0]["thread_cpu_ms"] >= 0
        monkeypatch.setattr(
            session.ledger, "append",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic telemetry defect")),
        )
        subject._record_prepared_plan_timing()
        assert subject._prepared_plans.take(
            holdings, replace(config, rebalancing_no_sell=True), 1000.0, 100.0,
        ) is not None


def test_owner_stage_retains_exact_audit_and_native_plans(monkeypatch):
    holdings, config, snapshot = inputs()
    prepared = planning.prepare_plans(snapshot)
    recorded = []
    monkeypatch.setattr(audit, "record_rebalancing_plan", lambda *a, **k: recorded.append((a, k)))
    serial = metrics.MetricsEngine(holdings, config)
    expected = {"total_value": 1000.0}
    serial._rebalancing(expected)
    recorded.clear()
    candidate = metrics.MetricsEngine(holdings, config)
    candidate._prepared_plans = prepared
    actual = {"total_value": 1000.0}
    candidate._rebalancing(actual)
    assert actual == expected
    assert [item[1]["no_sell"] for item in recorded] == [True, False]
    assert not prepared.variants


@pytest.mark.parametrize("gate", ["mechanics", "valuation", "classification", "invalid_target", "no_config"])
def test_late_unavailability_never_publishes_speculative_plan(monkeypatch, gate):
    holdings, config, _snapshot = inputs()
    subject = metrics.MetricsEngine(holdings, config)
    subject._prepared_plans = SimpleNamespace(take=lambda *a: pytest.fail("gate bypassed"))
    monkeypatch.setattr(audit, "record_rebalancing_plan", lambda *a, **k: pytest.fail("unavailable audit"))
    context = {"total_value": 1000.0}
    if gate == "mechanics":
        context["_order_mechanics_unavailable"] = ["SYNTHETIC"]
    elif gate == "valuation":
        subject.planning_eligible = False
    elif gate == "classification":
        context["classification_available"] = False
    elif gate == "invalid_target":
        subject.target_plan = SimpleNamespace(
            planning_eligible=False, status=SimpleNamespace(value="INVALID"),
        )
    else:
        subject.config = None
    subject._rebalancing(context)
    assert context["planning_availability"] == "UNAVAILABLE"
    assert context["rebalancing_plans"] is None


@pytest.mark.parametrize("name", [
    "_current_valuation_holdings", "_rebalancing", "_portfolio_history_from_orders",
    "_prepare_benchmark_context", "_snapshot_planning_inputs",
])
def test_injected_engine_methods_are_not_speculated(monkeypatch, name):
    subject = metrics.MetricsEngine(book(), InvestorConfig())
    original = metrics.MetricsEngine._snapshot_planning_inputs
    monkeypatch.setattr(subject, name, MethodType(
        lambda *a: pytest.fail("foreign engine method"), subject,
    ))
    assert original(subject, {"total_value": 1000.0}) is None


@pytest.mark.parametrize("manifest", [None, [], (), (lambda: None,)])
def test_truncated_engine_manifest_is_not_authority(monkeypatch, manifest):
    monkeypatch.setattr(metrics, "_NATIVE_PLANNING_METHODS", manifest)
    subject = metrics.MetricsEngine(book(), InvestorConfig())
    assert subject._snapshot_planning_inputs({"total_value": 1000.0}) is None


def test_owned_cpu_and_both_sources_overlap_without_native_permit(monkeypatch):
    meeting = Barrier(3)
    joined = []
    token = object()
    with owned() as session:
        def history(ctx):
            meeting.wait(timeout=3.0)
            joined.append("history")

        def benchmarks(ctx):
            meeting.wait(timeout=3.0)
            joined.append("benchmarks")
            return ()

        def prepare(_inputs):
            assert current_session() is session
            assert session.transport.coordinator.snapshot()["active"] == 0
            assert current_thread().name.startswith("tarzan-sources")
            meeting.wait(timeout=3.0)
            joined.append("plans")
            return token

        monkeypatch.setattr(planning, "prepare_plans_owned", prepare)
        subject = engine(monkeypatch, history, benchmarks)
        subject._planning_inputs = object()
        phases, error = subject._prepare_sources_together()
        assert error is None
        assert len(phases) == 2
        assert subject._prepared_plans is token
        assert set(joined) == {"history", "benchmarks", "plans"}
        counters = session.transport.coordinator.snapshot()
        assert counters["active"] == counters["essential_demand"] == counters["acquired"] == 0


@pytest.mark.parametrize("fault", ["worker", "submit", "canceled"])
def test_ordinary_speculative_failure_never_fails_original_sources(monkeypatch, fault):
    original_pool = ThreadPoolExecutor

    class FaultPool:
        def __init__(self, **kwargs):
            self.pool = original_pool(**kwargs)
            self.count = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            self.count += 1
            if self.count == 3 and fault == "submit":
                raise RuntimeError("synthetic CPU submission")
            if self.count == 3 and fault == "canceled":
                result = Future()
                result.cancel()
                return result
            return self.pool.submit(*args, **kwargs)

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    def prepare(_inputs):
        if fault == "worker":
            raise ValueError("synthetic CPU fault")
        return planning.PreparedPlans({})

    monkeypatch.setattr(planning, "prepare_plans_owned", prepare)
    monkeypatch.setattr(metrics, "ThreadPoolExecutor", FaultPool)
    finished = []
    with owned() as session:
        subject = engine(
            monkeypatch, lambda ctx: finished.append("history"),
            lambda ctx: finished.append("benchmarks") or (),
        )
        subject._planning_inputs = object()
        phases, error = subject._prepare_sources_together()
        assert error is None
        assert all(phase.error is None for phase in phases)
        assert subject._prepared_plans is None
        assert set(finished) == {"history", "benchmarks"}
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_cancellation_drains_all_children_and_retains_canonical_primary(
    monkeypatch, source_cancel, kind,
):
    primary = kind("synthetic source primary")
    speculative = kind("synthetic CPU interruption")
    meeting = Barrier(3)
    finished = []

    def prepare(_inputs):
        try:
            meeting.wait(timeout=3.0)
            raise speculative
        finally:
            finished.append("plans")

    monkeypatch.setattr(planning, "prepare_plans_owned", prepare)
    with owned() as session:
        def history(ctx):
            try:
                meeting.wait(timeout=3.0)
                if source_cancel:
                    raise primary
            finally:
                finished.append("history")

        def benchmarks(ctx):
            meeting.wait(timeout=3.0)
            finished.append("benchmarks")
            return ()

        subject = engine(monkeypatch, history, benchmarks)
        subject._planning_inputs = object()
        phases, error = subject._prepare_sources_together()
        assert error is (primary if source_cancel else speculative)
        assert set(finished) == {"plans", "history", "benchmarks"}
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0
        if source_cancel:
            assert phases[0].error is primary


def test_compute_all_clears_previous_attempt_and_requires_native_snapshot(monkeypatch):
    with owned() as session:
        subject = engine(monkeypatch, lambda ctx: None, lambda ctx: ())
        subject._prepared_plans = object()
        subject._planning_inputs = object()
        monkeypatch.setattr(planning, "prepare_plans_owned", lambda *a: pytest.fail("stale inputs"))
        result = subject.compute_all()
        assert result["_degraded"] == []
        assert subject._prepared_plans is None
        assert subject._planning_inputs is None
        assert len(stages(session)) == 2


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_owner_cpu_wait_interruption_retains_precedence_and_drains_worker(monkeypatch, kind):
    original_pool = ThreadPoolExecutor
    owner_error = kind("synthetic owner wait")
    child_error = kind("synthetic source interruption")
    release, finished = Event(), Event()

    class InterruptedWait:
        def __init__(self, future):
            self.future = future

        def result(self):
            release.set()
            raise owner_error

        def done(self):
            return self.future.done()

        def cancelled(self):
            return self.future.cancelled()

        def exception(self):
            return self.future.exception()

    class FaultPool:
        def __init__(self, **kwargs):
            self.pool = original_pool(**kwargs)
            self.count = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            self.count += 1
            future = self.pool.submit(*args, **kwargs)
            return InterruptedWait(future) if self.count == 3 else future

        def __exit__(self, *args):
            self.pool.shutdown(wait=True)

    def prepare(_inputs):
        assert release.wait(timeout=3.0)
        finished.set()
        return planning.PreparedPlans({})

    monkeypatch.setattr(planning, "prepare_plans_owned", prepare)
    monkeypatch.setattr(metrics, "ThreadPoolExecutor", FaultPool)
    with owned() as session:
        subject = engine(
            monkeypatch,
            lambda ctx: (_ for _ in ()).throw(child_error),
            lambda ctx: (),
        )
        subject._planning_inputs = object()
        phases, error = subject._prepare_sources_together()
        assert error is owner_error
        assert phases[0].error is child_error
        assert finished.is_set()
        assert session.transport.coordinator.snapshot()["essential_demand"] == 0


def test_actual_spawned_native_plans_match_and_process_is_owned():
    import multiprocessing
    holdings, config, snapshot = inputs()
    children_before = {child.pid for child in multiprocessing.active_children()}
    prepared = planning.prepare_plans_owned(snapshot)
    assert prepared is not None
    assert prepared.worker_evidence == {
        "native_solver_invocations": 2, "separate_process": True,
    }
    assert prepared.timing["worker_process_cpu_ms"] > 0
    for flag in (True, False):
        variant = replace(config, rebalancing_no_sell=flag)
        assert prepared.take(holdings, variant, 1000.0, 100.0) == rebalancer.compute_unified_rebalancing(
            holdings, variant, 1000.0, lump_sum=100.0,
        )
    assert {child.pid for child in multiprocessing.active_children()} <= children_before


@pytest.mark.parametrize("value", [
    None, True, 1, 1.0, -0.0, "native",
    {AssetClass.EQUITIES: [1.0, (Geography.USA, 2)]},
])
def test_builtins_payload_preserves_native_fact_types(value):
    encoded = planning._freeze(value)
    decoded = planning._thaw(encoded)
    assert planning._freeze(decoded) == encoded
    assert type(decoded) is type(value)


def test_payload_never_executes_model_reduction_hooks(monkeypatch):
    _holdings, _config, snapshot = inputs()
    for cls in (Holding, InvestorConfig, planning.ProviderQualityPolicy):
        monkeypatch.setattr(
            cls, "__reduce_ex__",
            lambda *a: pytest.fail("arbitrary model reduction"), raising=False,
        )
    prepared = planning.prepare_plans_owned(snapshot)
    assert prepared is not None


def _worker_safety_probe():
    import os
    import requests
    import smtplib
    import socket
    import urllib.request
    from curl_cffi import requests as curl_requests

    def raw_tcp():
        with socket.socket() as client:
            return client.connect_ex(("127.0.0.1", 9))

    def raw_udp():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            return client.sendto(b"synthetic", ("127.0.0.1", 9))

    operations = [
        lambda: socket.create_connection(("127.0.0.1", 9)),
        raw_tcp, raw_udp,
        lambda: urllib.request.urlopen("https://example.invalid"),
        lambda: requests.Session().get("https://example.invalid"),
        lambda: curl_requests.Session().get("https://example.invalid"),
        lambda: smtplib.SMTP("127.0.0.1", 9),
    ]
    rejected = []
    for operation in operations:
        try:
            operation()
        except RuntimeError as error:
            rejected.append(str(error) == "native planning worker has no network or email authority")
        else:
            rejected.append(False)
    return {
        "rejected": rejected,
        "secret_keys_present": any(name in os.environ for name in (
            "GEMINI_API_KEY", "SMTP_PASS", "GITHUB_TOKEN", "TARZAN_CACHE_KEY_B64",
        )),
        "AI_disabled": os.environ.get("TARZAN_DISABLE_AI") == "1",
    }


def test_spawn_worker_rejects_native_io_and_removes_credentials(monkeypatch):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    for name in ("GEMINI_API_KEY", "SMTP_PASS", "GITHUB_TOKEN", "TARZAN_CACHE_KEY_B64"):
        monkeypatch.setenv(name, "synthetic-unused-credential")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=planning._initialize_worker,
    ) as pool:
        evidence = pool.submit(_worker_safety_probe).result(timeout=15.0)
    assert evidence["rejected"] == [True] * 7
    assert not evidence["secret_keys_present"]
    assert evidence["AI_disabled"]
