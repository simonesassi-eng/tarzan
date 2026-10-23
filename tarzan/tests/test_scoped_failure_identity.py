"""Synthetic failure identities are independent of native completion order."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from threading import Barrier

import pytest

from tarzan.runtime import ledger as ledger_module, transport
from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.ledger import (
    Availability, LedgerEntryType, RunLedger,
)
from tarzan.runtime.transport import ProviderBudget, ProviderTransportSettings


PROVIDERS = ("yahoo", "openfigi", "borsa", "justetf", "manual", "gemini")


class Cancelled(BaseException):
    pass


class UntrustedScope:
    def __str__(self):
        raise AssertionError("untrusted scope must not be converted to text")


class StringSubclass(str):
    pass


def failure(ledger, *, stage="synthetic", code="SYNTHETIC_FAULT", **arguments):
    return ledger.open_failure(
        stage=stage, stable_code=code, severity="WARNING",
        error={"kind": "synthetic"}, affected_outputs=["synthetic"],
        analytical_impact="synthetic data unavailable",
        publication_impact="DEGRADE", **arguments,
    )


def legacy_id(stage, code, ordinal):
    return hashlib.sha256(
        f"failure-v1|{stage}|{code}|{ordinal}".encode("utf-8"),
    ).hexdigest()[:24]


def scoped_id(stage, code, scope, ordinal):
    return hashlib.sha256(canonical_json_bytes({
        "version": "failure-v2", "stage": stage, "stable_code": code,
        "identity_scope": scope, "ordinal": ordinal,
    }, ascii_only=True)).hexdigest()[:24]


def owner(name="synthetic-provider-failure"):
    configured = ProviderTransportSettings.configured()
    settings = replace(
        configured,
        run_budget_seconds=60.0, optional_budget_seconds=10.0,
        publication_reserve_seconds=5.0,
        providers=tuple(replace(policy, retry_budget=0) for policy in configured.providers),
    )
    return ProviderBudget(settings, RunLedger(name), capacity=4)


@pytest.mark.parametrize("stage,code", [
    ("synthetic", "SYNTHETIC_FAULT"), ("stage|with|separators", "FAULT|CODE"),
    ("synthetic-é", "SYNTHETIC_日本語"),
])
def test_unscoped_ids_remain_byte_identical_to_v1(stage, code):
    ledger = RunLedger("synthetic-legacy")
    for ordinal in range(1, 11):
        assert failure(ledger, stage=stage, code=code) == legacy_id(stage, code, ordinal)
    assert all("identity_scope" not in entry.payload for entry in ledger.entries)


def test_explicit_none_scope_retains_the_identical_legacy_payload():
    first, second = RunLedger("synthetic-a"), RunLedger("synthetic-b")
    assert failure(first) == failure(second, identity_scope=None)
    assert first.entries[0].payload == second.entries[0].payload
    assert first.entries[0].entry_id == second.entries[0].entry_id


def test_scoped_calls_do_not_advance_any_existing_unscoped_counter():
    ledger = RunLedger("synthetic-mixed-identities")
    for provider in PROVIDERS:
        failure(ledger, identity_scope=f"provider:{provider}")
    assert failure(ledger) == legacy_id("synthetic", "SYNTHETIC_FAULT", 1)
    failure(ledger, identity_scope="provider:yahoo")
    assert failure(ledger) == legacy_id("synthetic", "SYNTHETIC_FAULT", 2)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_scope_owns_its_exact_structured_identity_and_ordinal(provider):
    ledger = RunLedger("synthetic-scoped")
    scope = f"provider:{provider}"
    for ordinal in range(1, 6):
        assert failure(ledger, identity_scope=scope) == scoped_id(
            "synthetic", "SYNTHETIC_FAULT", scope, ordinal,
        )
    assert len({entry.payload["failure_id"] for entry in ledger.entries}) == 5
    assert all(entry.payload["identity_scope"] == scope for entry in ledger.entries)
    assert ledger_module.LEDGER_SCHEMA_VERSION == "1.0"


def test_scoped_structured_fields_cannot_collide_at_text_separators():
    ledger = RunLedger("synthetic-field-boundaries")
    first = failure(
        ledger, stage="synthetic|left", code="right", identity_scope="provider:yahoo",
    )
    second = failure(
        ledger, stage="synthetic", code="left|right", identity_scope="provider:yahoo",
    )
    assert first != second


def test_scoped_ids_are_invariant_to_provider_completion_order():
    def record(providers):
        ledger = RunLedger("synthetic-order")
        return {
            provider: failure(ledger, identity_scope=f"provider:{provider}")
            for provider in providers
        }
    assert record(PROVIDERS) == record(reversed(PROVIDERS))


def test_scoped_ordinals_remain_independent_of_stages_codes_and_other_scopes():
    ledger = RunLedger("synthetic-independent-identities")
    first = failure(ledger, stage="a", code="ONE", identity_scope="provider:yahoo")
    for stage, code, scope in (
        ("b", "ONE", "provider:yahoo"),
        ("a", "TWO", "provider:yahoo"),
        ("a", "ONE", "provider:borsa"),
    ):
        assert failure(ledger, stage=stage, code=code, identity_scope=scope) == scoped_id(
            stage, code, scope, 1,
        )
    second = failure(ledger, stage="a", code="ONE", identity_scope="provider:yahoo")
    assert first != second
    assert second == scoped_id("a", "ONE", "provider:yahoo", 2)


def test_one_hundred_concurrent_same_scope_failures_allocate_unique_ids():
    ledger = RunLedger("synthetic-concurrent-identities")
    meeting = Barrier(10)
    def batch():
        meeting.wait(timeout=2.0)
        return [failure(ledger, identity_scope="provider:yahoo") for _ in range(10)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(batch) for _ in range(10)]
        results = [value for item in futures for value in item.result(timeout=3.0)]
    assert len(results) == len(set(results)) == 100
    assert set(results) == {
        scoped_id("synthetic", "SYNTHETIC_FAULT", "provider:yahoo", ordinal)
        for ordinal in range(1, 101)
    }
    assert len(ledger.failure_records()) == 100


def test_one_hundred_failures_per_provider_have_no_cross_scope_counter_effect():
    ledger = RunLedger("synthetic-concurrent-providers")
    meeting = Barrier(len(PROVIDERS))
    def batch(provider):
        meeting.wait(timeout=2.0)
        return provider, [
            failure(ledger, identity_scope=f"provider:{provider}")
            for _ in range(100)
        ]
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        futures = [pool.submit(batch, provider) for provider in reversed(PROVIDERS)]
        results = dict(item.result(timeout=3.0) for item in futures)
    all_ids = [value for values in results.values() for value in values]
    assert len(all_ids) == len(set(all_ids)) == 600
    for provider, values in results.items():
        assert set(values) == {
            scoped_id("synthetic", "SYNTHETIC_FAULT", f"provider:{provider}", ordinal)
            for ordinal in range(1, 101)
        }


@pytest.mark.parametrize("scope", [
    "", "arbitrary-label", "provider:unknown", "provider:YAHOO",
    "provider:yahoo?token=SYNTHETIC-SECRET-CANARY",
    "https://example.invalid/?token=SYNTHETIC-SECRET-CANARY",
    True, 1, [], {}, ("provider", "yahoo"), b"provider:yahoo",
    StringSubclass("provider:yahoo"),
    pytest.param(UntrustedScope(), id="untrusted-object"),
])
def test_invalid_scope_is_rejected_before_allocation_without_echoing_its_value(scope):
    ledger = RunLedger("synthetic-invalid-scope")
    with pytest.raises(ValueError) as raised:
        failure(ledger, identity_scope=scope)
    assert str(raised.value) == "unsupported failure identity scope"
    assert ledger.entries == ()
    assert ledger._failure_ordinals == {}
    assert failure(ledger) == legacy_id("synthetic", "SYNTHETIC_FAULT", 1)


def test_scoped_remedy_and_close_retain_the_existing_projection_contract():
    ledger = RunLedger("synthetic-scoped-resolution")
    failure_id = failure(ledger, identity_scope="provider:yahoo")
    ledger.remedy(
        failure_id, remedy_id="synthetic-remedy", action="synthetic fallback",
        outcome="SUCCEEDED", availability=Availability.DEGRADED,
        provenance=["synthetic-provider-fact"],
    )
    ledger.close_failure(
        failure_id, automatically_corrected=True, selected_resolution="synthetic-remedy",
        availability=Availability.DEGRADED, provenance=["synthetic-provider-fact"],
    )
    record = ledger.failure_records()[0]
    assert record.failure_id == failure_id
    assert record.closed and record.automatically_corrected
    assert record.selected_resolution == "synthetic-remedy"
    assert record.availability is Availability.DEGRADED
    assert record.remedies[0]["ordinal"] == 1


def test_rejected_scope_does_not_pollute_logs_or_ledger_with_secret_data(caplog):
    ledger = RunLedger("synthetic-private-scope")
    with pytest.raises(ValueError):
        failure(ledger, identity_scope="provider:SYNTHETIC-SECRET-CANARY")
    assert "SYNTHETIC-SECRET-CANARY" not in caplog.text
    assert "SYNTHETIC-SECRET-CANARY" not in repr(ledger.entries)
    assert "SYNTHETIC-SECRET-CANARY" not in repr(ledger._failure_ordinals)


def test_provider_scope_allowlist_matches_declared_transport_settings_exactly():
    settings = ProviderTransportSettings.configured()
    assert {policy.provider for policy in settings.providers} == set(PROVIDERS)
    assert ledger_module._FAILURE_IDENTITY_SCOPES == {
        f"provider:{provider}" for provider in PROVIDERS
    }


def test_transport_provider_failures_use_stable_scopes_regardless_of_completion_order():
    def record(providers):
        budget = owner()
        for provider in providers:
            budget._failure(provider, "PROVIDER_RETRY_BUDGET_EXHAUSTED")
        return {
            entry.payload["original_failure"]["provider"]: entry.payload["failure_id"]
            for entry in budget.ledger.entries
            if entry.entry_type is LedgerEntryType.FAILURE_OPEN
        }
    assert record(PROVIDERS) == record(reversed(PROVIDERS))


def test_concurrent_transport_duplicates_emit_one_scoped_failure():
    budget = owner()
    meeting = Barrier(10)
    def batch():
        meeting.wait(timeout=2.0)
        for _ in range(10):
            budget._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(batch) for _ in range(10)]
        for item in futures:
            item.result(timeout=3.0)
    assert len(budget.ledger.failure_records()) == 1
    assert budget.ledger.entries[0].payload["identity_scope"] == "provider:yahoo"


@pytest.mark.parametrize("provider", [
    "", "unknown", "SYNTHETIC-SECRET-CANARY", True, 1, [],
    StringSubclass("yahoo"), pytest.param(UntrustedScope(), id="untrusted-object"),
])
def test_transport_rejects_untrusted_provider_before_recording_or_logging(provider, caplog):
    budget = owner()
    with pytest.raises(ValueError) as raised:
        budget._failure(provider, "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    assert str(raised.value) == "unknown provider transport"
    assert budget.ledger.entries == ()
    assert budget._reported == set()
    assert "SYNTHETIC-SECRET-CANARY" not in caplog.text


def test_run_owners_keep_independent_scope_ordinals_and_deduplication():
    first, second = owner("synthetic-run-a"), owner("synthetic-run-b")
    first._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    second._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    assert first.ledger.failure_records()[0].failure_id == second.ledger.failure_records()[0].failure_id
    assert first.ledger._failure_ordinals == second.ledger._failure_ordinals
    assert first.ledger._failure_ordinals is not second.ledger._failure_ordinals
    first._failure("borsa", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    assert len(second.ledger.failure_records()) == 1


def test_callback_cancellation_leaves_no_identity_scope_or_native_permit():
    budget = owner()
    error = Cancelled()
    def native(timeout):
        raise error
    with pytest.raises(Cancelled) as raised:
        budget.call("yahoo", native)
    assert raised.value is error
    assert budget.coordinator.snapshot()["active"] == 0
    assert budget.ledger.failure_records() == ()
    budget._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    assert budget.ledger.failure_records()[0].failure_id == scoped_id(
        "provider_transport", "PROVIDER_RETRY_BUDGET_EXHAUSTED", "provider:yahoo", 1,
    )
    assert failure(budget.ledger) == legacy_id("synthetic", "SYNTHETIC_FAULT", 1)


def test_optional_scope_cancellation_cannot_change_later_failure_identity():
    budget = owner()
    error = Cancelled()
    with pytest.raises(Cancelled) as raised:
        with budget.optional_work("synthetic-optional"):
            budget._failure("yahoo", "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED")
            raise error
    assert raised.value is error
    assert budget.optional_work_token() is None
    budget._failure("yahoo", "PROVIDER_RETRY_BUDGET_EXHAUSTED")
    assert len(budget.ledger.failure_records()) == 2
    assert budget.ledger.failure_records()[-1].failure_id == scoped_id(
        "provider_transport", "PROVIDER_RETRY_BUDGET_EXHAUSTED", "provider:yahoo", 1,
    )
