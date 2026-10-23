"""Configured transport and acceptance policies change real adapter/gate behavior."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import _yf_net, bond_fetcher, enricher, justetf, price_cache
from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import Availability, LedgerEntryType, RunLedger
from tarzan.runtime.provider import ProviderQualityPolicy, ValuationCompletenessEvaluator
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
from tarzan.runtime.session import RunContext, RunMode, RunResult, RunSession, activate_session
from tarzan.runtime.summary import SummaryProjector
from tarzan.runtime.transport import (
    ProviderBudget, ProviderBudgetExceeded, ProviderRequestFailure,
    ProviderTransportPolicy, ProviderTransportSettings, provider_call, provider_wait,
)


class Clock:
    def __init__(self):
        self.value = 0.0
        self.waits = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


def settings(*, timeout=3.0, retries=1, total=20.0, provider="yahoo"):
    configured = ProviderTransportSettings.configured()
    return replace(
        configured, run_budget_seconds=total,
        providers=tuple(
            replace(policy, timeout_seconds=timeout, retry_budget=retries)
            if policy.provider == provider else policy
            for policy in configured.providers
        ),
    )


@contextmanager
def owned(budget):
    context = RunContext("synthetic-policy", RunMode.LIVE, None, datetime.now(timezone.utc))
    owner = RunSession(context, {}, budget.ledger or RunLedger(context.attempt_id), transport=budget)
    with runtime.bind_context(context), activate_session(owner):
        try:
            yield owner
        finally:
            budget.close()


def budget(**kwargs):
    clock = Clock()
    ledger = RunLedger("synthetic-policy")
    return ProviderBudget(settings(**kwargs), ledger, clock=clock.now, sleep=clock.sleep), clock


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "5"])
def test_transport_timeout_rejects_invalid_domains(value):
    with pytest.raises(ValueError):
        ProviderTransportPolicy("yahoo", value, 1)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1", 6])
def test_transport_retry_budget_is_a_closed_integer_domain(value):
    with pytest.raises(ValueError):
        ProviderTransportPolicy("yahoo", 5, value)


def test_provider_settings_reject_duplicate_or_omitted_transports():
    policies = ProviderTransportSettings.configured().providers
    with pytest.raises(ValueError):
        ProviderTransportSettings(10, policies[:-1])
    with pytest.raises(ValueError):
        ProviderTransportSettings(10, (*policies[:-1], policies[0]))


@pytest.mark.parametrize("retries", [0, 2])
def test_retry_configuration_controls_actual_native_attempt_count(retries):
    owner, clock = budget(timeout=10, retries=retries)
    calls = []

    def failed(timeout):
        calls.append(timeout)
        clock.value += 0.1
        raise TimeoutError("synthetic private request information")

    with pytest.raises(ProviderRequestFailure):
        owner.call("yahoo", failed)
    assert len(calls) == retries + 1
    assert clock.waits == [0.25 * 2 ** attempt for attempt in range(retries)]
    assert all("private request" not in str(item.to_dict()) for item in owner.ledger.entries)


def test_retry_and_pacing_consume_the_same_operation_deadline():
    owner, clock = budget(timeout=3, retries=1)
    caps = []

    def callback(timeout):
        caps.append(timeout)
        clock.value += 1
        if len(caps) == 1:
            raise TimeoutError("synthetic timeout")
        return "value"

    assert owner.call("yahoo", callback) == "value"
    assert caps == pytest.approx([3, 1.75])
    assert clock.waits == [0.25]


def test_run_deadline_caps_native_timeout_and_blocks_further_requests():
    owner, clock = budget(timeout=5, retries=2, total=2)
    caps = []

    def callback(timeout):
        caps.append(timeout)
        clock.value += 1
        if len(caps) == 1:
            raise TimeoutError("synthetic timeout")
        return "late response"

    with pytest.raises(ProviderBudgetExceeded):
        owner.call("yahoo", callback)
    assert caps == pytest.approx([2, 0.75])
    for _ in range(3):
        with pytest.raises(ProviderBudgetExceeded):
            owner.call("yahoo", lambda timeout: pytest.fail("a request started after exhaustion"))
    failures = [f for f in owner.ledger.failure_records() if f.stable_code == "PROVIDER_BUDGET_EXHAUSTED"]
    assert len(failures) == 1


@pytest.mark.parametrize(("requested", "expected"), [(1.5, 1.5), (10., 3.), ((1., 2.), 1.)])
def test_shared_adapter_respects_the_callers_native_timeout_ceiling(requested, expected):
    owner, _ = budget(timeout=3.)
    caps = []
    answer = object()
    with owned(owner):
        result = provider_call(
            "yahoo", lambda timeout: caps.append(timeout) or answer, timeout=requested,
        )
    assert result is answer
    assert caps == [expected]
    attempts = [
        entry.payload for entry in owner.ledger.entries
        if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT
    ]
    assert attempts[0]["timeout_seconds"] == expected


def test_pacing_cannot_sleep_past_remaining_provider_budget():
    owner, clock = budget(timeout=1, total=0.1)
    with owned(owner), pytest.raises(ProviderBudgetExceeded):
        provider_wait(0.2, "yahoo")
    assert clock.waits == []


@pytest.mark.parametrize("status,expected_calls", [(404, 1), (429, 2), (503, 2)])
def test_only_transient_http_statuses_retry(status, expected_calls):
    owner, _ = budget()
    calls, closed = [], []

    def callback(timeout):
        calls.append(timeout)
        code = status if len(calls) == 1 else 200
        return SimpleNamespace(status_code=code, close=lambda: closed.append(code))

    assert owner.call("yahoo", callback).status_code == (404 if status == 404 else 200)
    assert len(calls) == expected_calls
    assert closed == ([] if status == 404 else [status])


def test_cancellation_is_not_swallowed_or_retried():
    class Cancelled(BaseException):
        pass
    owner, clock = budget()
    def cancel(timeout):
        raise Cancelled()
    with pytest.raises(Cancelled):
        owner.call("yahoo", cancel)
    assert clock.waits == []
    attempts = [entry for entry in owner.ledger.entries if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT]
    assert attempts[0].payload["outcome"] == "CANCELLED"


@pytest.mark.parametrize("retries", [0, 1])
def test_real_yfinance_http_port_obeys_policy_without_sdk_retry_multiplication(monkeypatch, retries):
    from curl_cffi.requests import Session
    from yfinance.data import YfData
    owner, _ = budget(timeout=2, retries=retries)
    caps = []
    def request(self, method, url, **kwargs):
        caps.append(kwargs["timeout"])
        raise TimeoutError("synthetic transport timeout")
    monkeypatch.setattr(Session, "request", request)
    monkeypatch.setattr(YfData, "_get_cookie_and_crumb", lambda *args, **kwargs: ("synthetic", "basic"))
    with owned(owner):
        ticker = _yf_net.ticker("SYNTHETIC.X")
        with pytest.raises(ProviderRequestFailure):
            ticker._data.get("https://provider.example.invalid/quote", timeout=30)
    assert len(caps) == retries + 1
    assert max(caps) == 2


def test_owned_batch_retains_the_native_owner_without_shared_sdk_batch_state(monkeypatch):
    import yfinance
    from tarzan.runtime.session import current_session
    owner, _ = budget()
    observed = []

    class Ticker:
        def __init__(self, symbol, **kwargs):
            observed.append((kwargs["session"], current_session()))

        def history(self, **kwargs):
            assert "progress" not in kwargs
            return pd.DataFrame()

    monkeypatch.setattr(yfinance, "Ticker", Ticker)
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pytest.fail("shared SDK coordinator"))
    with owned(owner), owner.operation("yahoo"):
        _yf_net.download(["SYNTHETIC.X"], threads=True, progress=True)
        assert observed[0][0] is owner.yahoo_client()
        assert observed[0][1].transport is owner


@pytest.mark.parametrize("provider", ["justetf", "openfigi", "borsa"])
def test_production_non_yahoo_adapters_pass_configured_native_timeout(monkeypatch, provider):
    owner, _ = budget(timeout=1.5, retries=0, provider=provider)
    caps = []
    def urlopen(request, timeout):
        caps.append(timeout)
        if provider == "openfigi":
            response = io.BytesIO(json.dumps([{"data": [{"ticker": "SYNTHETIC"}]}]).encode())
        else:
            response = io.BytesIO((("<html>Prezzo ufficiale 99,50 " + "x" * 6000 + "</html>")).encode())
            response.status = 200
        return response
    monkeypatch.setattr(enricher, "urlopen", urlopen)
    monkeypatch.setattr(bond_fetcher, "urlopen", urlopen)
    monkeypatch.setattr(price_cache, "load_timed", lambda *args: None)
    monkeypatch.setattr(price_cache, "store_timed", lambda *args: None)
    monkeypatch.setattr(enricher, "_openfigi_last_call", [0.0])
    def get(url, **kwargs):
        caps.append(kwargs["timeout"])
        return SimpleNamespace(status_code=200, ok=True, headers={"content-type": "application/json"}, json=lambda: {})
    monkeypatch.setattr("requests.get", get)
    with owned(owner):
        if provider == "justetf":
            assert justetf._get_json("https://provider.example.invalid") == {}
        elif provider == "openfigi":
            assert enricher._openfigi_load_many(["IE0000000001"])["IE0000000001"]
        else:
            assert bond_fetcher._try_segment("IT0000000001", "mot")["price"] == 99.5
    assert caps == [1.5]


def quality(*, coverage=0.0, valuation=100.0, publication=100.0, fallback=False):
    return ProviderQualityPolicy(
        "synthetic-acceptance", 3600.0, coverage, fallback, valuation, publication,
    )


def holding(key, basis, *, known=False):
    return Holding(
        isin=key, ticker=key, quantity=1, cost_basis_eur=basis,
        market_value_eur=0.0, currency="EUR", security_type="STOCK",
        current_price=basis if known else None, current_value=basis if known else None,
        data_source="synthetic-primary" if known else None,
    )


def assess(policy, holdings=None):
    ledger = RunLedger("synthetic-acceptance")
    assessment = ValuationCompletenessEvaluator({"STOCK:current_valuation": policy}).evaluate(
        holdings or [holding("KNOWN", 94, known=True), holding("MISSING", 6)], ledger,
    )
    return assessment, ledger


@pytest.mark.parametrize("minimum,eligible", [(90, True), (94, True), (95, False), (100, False)])
def test_minimum_accepted_coverage_changes_total_and_planning_gate(minimum, eligible):
    result, ledger = assess(quality(coverage=minimum))
    assert result.coverage[0].coverage_pct == 94
    assert result.planning_eligible is eligible
    assert result.trustworthy_total_eur == (94 if eligible else None)
    assert PublicationEvaluator.evaluate(ledger.failure_records()).decision is PublicationDecision.SEND_DEGRADED_NORMAL


def test_accepted_anchor_fallback_counts_as_valuation_not_market_history_coverage():
    missing = holding("ANCHOR", 6)
    missing.market_value_eur = 6
    result, _ = assess(quality(coverage=100, fallback=True), [holding("KNOWN", 94, known=True), missing])
    assert result.coverage[0].coverage_pct == 100
    assert result.planning_eligible
    assert result.availability is Availability.DEGRADED


def test_publication_threshold_is_independent_of_valuation_threshold():
    permissive, permissive_ledger = assess(quality(valuation=5, publication=10))
    strict, strict_ledger = assess(quality(valuation=10, publication=5))
    assert not permissive.planning_eligible and permissive.trustworthy_total_eur is None
    assert PublicationEvaluator.evaluate(permissive_ledger.failure_records()).decision is PublicationDecision.SEND_DEGRADED_NORMAL
    assert strict.planning_eligible and strict.trustworthy_total_eur == 94
    assert PublicationEvaluator.evaluate(strict_ledger.failure_records()).decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE


def test_insufficient_coverage_summary_is_unavailable_even_without_critical_publication():
    result, ledger = assess(quality(coverage=95))
    metrics = PortfolioMetrics(
        total_value=94, valuation_availability=result.availability.value,
        known_valuation_subtotal_eur=94, planning_availability="UNAVAILABLE",
    )
    summary = SummaryProjector.project(RunResult(metrics, InvestorConfig(), "attempt", "analysis", ledger))
    assert summary.sections["portfolio"]["availability"] == "UNAVAILABLE"
    assert summary.sections["portfolio"]["value"] is None
    assert summary.metrics["total_value_eur"] is None
    assert summary.sections["planning"]["value"] is None
    assert summary.publication_state == "SEND_DEGRADED_NORMAL"


@pytest.mark.parametrize("field,value", [
    ("minimum_coverage_pct", True), ("minimum_coverage_pct", float("nan")),
    ("minimum_coverage_pct", 101), ("publication_materiality_pct", -1),
    ("allow_fallback", "false"), ("policy_id", ""),
])
def test_acceptance_policy_rejects_ambiguous_or_nonfinite_domains(field, value):
    with pytest.raises(ValueError):
        replace(quality(), **{field: value})
