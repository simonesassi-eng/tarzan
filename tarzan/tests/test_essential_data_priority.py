"""Optional reference discovery cannot starve portfolio or critical benchmark data."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import config, runtime
from tarzan.engine import benchmarks, metrics
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime.transport import ProviderBudget, ProviderBudgetExceeded, ProviderTransportSettings


def prices(ticker):
    series = pd.Series([100.0, 101.0], index=pd.date_range("2026-10-19", periods=2))
    series.attrs["resolved_ticker"] = ticker + ".MI"
    return series


@contextmanager
def owned(*, total=30, optional=2, reserve=5):
    clock = [0.0]
    settings = replace(
        ProviderTransportSettings.configured(), run_budget_seconds=total,
        optional_budget_seconds=optional, publication_reserve_seconds=reserve,
    )
    context = RunContext("synthetic-priority", RunMode.LIVE, None, datetime.now(timezone.utc))
    ledger = RunLedger(context.attempt_id)
    budget = ProviderBudget(settings, ledger, clock=lambda: clock[0], sleep=lambda seconds: None)
    session = RunSession(context, {}, ledger, transport=budget)
    with runtime.bind_context(context), activate_session(session):
        try:
            yield session, clock
        finally:
            budget.close()


def test_closed_order_history_has_canonical_adoption_precedence_over_reference_preparation():
    engine = metrics.MetricsEngine([], InvestorConfig(), orders=[SimpleNamespace()])
    stages = [computer.__name__ for computer in engine._computers]
    assert stages.index("_portfolio_history_from_orders") < stages.index("_preprocess_benchmarks")
    assert stages.index("_preprocess_benchmarks") < stages.index("_current_prices") < stages.index("_performance")


def test_critical_reference_is_fetched_first_but_output_order_remains_deterministic():
    seen = []
    definitions = {"Extra A": "A", "Extra B": "B", "Critical": "CORE"}
    catalog, errors = benchmarks.preprocess_benchmarks(
        definitions, required_names=("Critical",),
        fetch_history=lambda ticker: seen.append(ticker) or prices(ticker),
    )
    assert seen[0] == "CORE"
    assert list(catalog) == list(definitions)
    assert errors == ()


def test_optional_budget_does_not_consume_the_publication_reserve(monkeypatch):
    monkeypatch.setattr(config, "max_workers", lambda: 1)
    seen = []
    with owned(total=10, optional=20, reserve=5) as (session, clock):
        def fetch(ticker):
            seen.append(ticker)
            clock[0] += 1
            return prices(ticker)

        catalog, errors = benchmarks.preprocess_benchmarks(
            {"Critical": "CORE", **{f"Extra {i}": f"X{i}" for i in range(10)}},
            required_names=("Critical",), fetch_history=fetch, workers=1,
        )
        assert seen == ["CORE", "X0", "X1", "X2", "X3"]
        assert "Critical" in catalog
        assert clock[0] == 5
        assert len(errors) == 6
        # Native authority is still available to the essential next phase.
        assert session.transport.call("yahoo", lambda timeout: "available") == "available"
        assert any(f.stable_code == "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED"
                   for f in session.ledger.failure_records())


def test_native_requests_in_optional_workers_inherit_the_optional_deadline():
    with owned(total=20, optional=1.5, reserve=5) as (session, clock):
        caps = []
        def fetch(ticker):
            def request(timeout):
                caps.append(timeout)
                clock[0] += 2
                return prices(ticker)
            return session.transport.call("yahoo", request)

        catalog, errors = benchmarks.preprocess_benchmarks(
            {"Extra": "X"}, fetch_history=fetch, workers=2,
        )
        assert catalog == {}
        assert len(errors) == 1
        assert caps == [1.5]
        assert session.transport.call("yahoo", lambda timeout: "core still allowed") == "core still allowed"


def test_optional_pacing_cannot_sleep_through_the_reserve():
    with owned(total=10, optional=1, reserve=5) as (session, clock):
        with session.transport.optional_work("optional_benchmarks"):
            with pytest.raises(ProviderBudgetExceeded):
                session.transport.wait(2, "yahoo")
        assert clock[0] == 0
        codes = {f.stable_code for f in session.ledger.failure_records()}
        assert codes == {"OPTIONAL_PROVIDER_BUDGET_EXHAUSTED"}


def test_required_native_requests_are_not_subject_to_the_optional_ceiling():
    with owned(total=30, optional=1, reserve=5) as (session, clock):
        caps = []

        def fetch(ticker):
            def request(timeout):
                caps.append((ticker, timeout))
                clock[0] += 2 if ticker == "CORE" else 0
                return prices(ticker)
            return session.transport.call("yahoo", request)

        catalog, errors = benchmarks.preprocess_benchmarks(
            {"Extra": "X", "Critical": "CORE"}, required_names=("Critical",),
            fetch_history=fetch, workers=1,
        )
        assert list(catalog) == ["Extra", "Critical"]
        assert errors == ()
        assert caps == [("CORE", 20), ("X", 1)]


def test_nested_optional_scopes_cannot_extend_the_outer_deadline():
    with owned(total=30, optional=2, reserve=5) as (session, clock):
        with session.transport.optional_work("outer"):
            clock[0] = 1
            with session.transport.optional_work("inner"):
                assert session.transport.call("yahoo", lambda timeout: timeout) == 1
                clock[0] = 2
                assert session.transport.optional_request_allowed("yahoo") is False
        assert session.transport.call("yahoo", lambda timeout: "still owned") == "still owned"
        assert {record.stable_code for record in session.ledger.failure_records()} == {
            "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED",
        }


def test_optional_workers_run_in_parallel_and_retain_the_native_owner():
    import threading
    from tarzan.runtime.session import current_session

    rendezvous = threading.Barrier(3)
    with owned() as (session, clock):
        def fetch(ticker):
            assert current_session() is session
            rendezvous.wait(timeout=2)
            return prices(ticker)

        catalog, errors = benchmarks.preprocess_benchmarks(
            {"A": "A", "B": "B", "C": "C"}, fetch_history=fetch, workers=3,
        )
        assert list(catalog) == ["A", "B", "C"]
        assert errors == ()


def test_missing_essential_benchmark_is_source_unavailability_not_a_fake_success(monkeypatch):
    monkeypatch.setattr(metrics, "BENCHMARKS", {"Critical": "CORE", "Extra": "EXTRA"})
    monkeypatch.setattr(config, "benchmark_beta_name", lambda: "Critical")
    monkeypatch.setattr(config, "benchmark_geo_allocation", lambda: "Critical")
    monkeypatch.setattr(metrics, "_fetch_benchmark_history", lambda ticker: prices(ticker) if ticker == "EXTRA" else None)
    with owned() as (session, clock):
        engine = metrics.MetricsEngine([], InvestorConfig())
        context = {}
        engine._preprocess_benchmarks(context)
        assert "Critical" not in context["benchmark_tickers"]
        assert "_preprocess_benchmarks" in context["_degraded"]
        failures = session.ledger.failure_records()
        assert any(f.stable_code == "ESSENTIAL_BENCHMARK_UNAVAILABLE" for f in failures)
        assert PublicationEvaluator.evaluate(failures).decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE


def test_missing_optional_benchmark_does_not_block_valid_portfolio_publication(monkeypatch):
    monkeypatch.setattr(metrics, "BENCHMARKS", {"Extra": "EXTRA", "Critical": "CORE"})
    monkeypatch.setattr(config, "benchmark_beta_name", lambda: "Critical")
    monkeypatch.setattr(config, "benchmark_geo_allocation", lambda: "Critical")
    monkeypatch.setattr(metrics, "_fetch_benchmark_history", lambda ticker: prices(ticker) if ticker == "CORE" else None)
    with owned() as (session, clock):
        context = {}
        metrics.MetricsEngine([], InvestorConfig())._preprocess_benchmarks(context)
        assert "Critical" in context["benchmark_tickers"]
        assert "_preprocess_benchmarks" not in context.get("_degraded", [])
        assert not any(f.severity == "CRITICAL" for f in session.ledger.failure_records())


@pytest.mark.parametrize("field", ["optional_budget_seconds", "publication_reserve_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, "5", float("nan"), float("inf")])
def test_optional_budget_configuration_rejects_invalid_domains(field, value):
    with pytest.raises(ValueError):
        replace(ProviderTransportSettings.configured(), **{field: value})
