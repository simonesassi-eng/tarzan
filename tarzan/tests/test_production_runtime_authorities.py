"""Focused integration tests for production runtime authorities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tarzan.delivery.claims import (
    AppsScriptPropertiesDeliveryClaimStore,
    DeliveryIntent,
    DeliveryPurpose,
    DeliveryState,
    LocalJsonDeliveryClaimStore,
    recipient_set_digest,
)
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.artifacts import LocalArtifactWriter, LocalOnlyWorkbook, StorageDescriptor
from tarzan.runtime.ledger import Availability, RunLedger
from tarzan.runtime.provider import ProviderQualityPolicy, ValuationCompletenessEvaluator
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
from tarzan.runtime.session import RunResult
from tarzan.runtime.summary import SummaryProjector


def _result(metrics: PortfolioMetrics, ledger: RunLedger, attempt: str = "attempt-a") -> RunResult:
    return RunResult(
        metrics=metrics,
        config=InvestorConfig(),
        attempt_id=attempt,
        analysis_id="analysis-stable",
        ledger=ledger,
    )


def _policy(policy_id: str, materiality_pct: float) -> ProviderQualityPolicy:
    return ProviderQualityPolicy(
        policy_id=policy_id,
        freshness_seconds=3600.0,
        minimum_coverage_pct=100.0,
        timeout_seconds=5.0,
        retry_budget=0,
        allow_fallback=False,
        valuation_materiality_pct=materiality_pct,
        publication_materiality_pct=materiality_pct,
    )


# **Validates: Requirements 2.6, 2.7, 2.11, 3.6, 3.11**
def test_summary_is_strict_immutable_and_attempt_independent():
    metrics = PortfolioMetrics(total_value=0.0, invested_value=0.0, cash_value=0.0)
    metrics.trustworthy_total_value_eur = 0.0
    metrics.known_valuation_subtotal_eur = 0.0
    first = SummaryProjector.project(_result(metrics, RunLedger("attempt-a"), "attempt-a"))
    second = SummaryProjector.project(_result(metrics, RunLedger("attempt-b"), "attempt-b"))

    assert first.canonical_bytes() == second.canonical_bytes()
    assert b"attempt-a" not in first.canonical_bytes()
    assert first.sections["portfolio"]["availability"] == "AVAILABLE"
    assert first.sections["portfolio"]["trustworthy_total_eur"] == 0.0
    json.loads(first.canonical_bytes())
    with pytest.raises(TypeError):
        first.sections["portfolio"] = {}  # type: ignore[index]


# **Validates: Requirements 2.7, 2.11, 2.13, 3.7, 3.11, 3.13**
def test_critical_valuation_gap_blocks_summary_planning_and_publication():
    policies = {
        "STOCK:current_valuation": _policy("stock-current", 10.0),
        "BOND:current_valuation": _policy("bond-current", 10.0),
    }
    known = Holding(
        isin="KNOWN", ticker="KNOWN", quantity=1.0, cost_basis_eur=90.0,
        market_value_eur=90.0, currency="EUR", security_type="STOCK",
        current_price=90.0, current_value=90.0, data_source="fixture-primary",
    )
    missing = Holding(
        isin="MISSING", ticker="MISSING", quantity=1.0, cost_basis_eur=20.0,
        market_value_eur=20.0, currency="EUR", security_type="BOND",
        current_price=None, current_value=None, data_source=None,
    )
    ledger = RunLedger("valuation-attempt")
    assessment = ValuationCompletenessEvaluator(policies).evaluate(
        [known, missing], ledger
    )

    assert assessment.availability is Availability.UNAVAILABLE
    assert assessment.trustworthy_total_eur is None
    assert assessment.known_subtotal_eur == 90.0
    assert assessment.planning_eligible is False
    failures = ledger.failure_records()
    assert any(record.stable_code == "MATERIAL_VALUATION_GAP" for record in failures)
    publication = PublicationEvaluator.evaluate(failures)
    assert publication.decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE

    metrics = PortfolioMetrics(total_value=90.0, invested_value=90.0)
    metrics.trustworthy_total_value_eur = assessment.trustworthy_total_eur
    metrics.known_valuation_subtotal_eur = assessment.known_subtotal_eur
    summary = SummaryProjector.project(_result(metrics, ledger), publication)
    assert summary.sections["portfolio"]["value"] is None
    assert summary.sections["portfolio"]["known_subtotal_eur"] == 90.0
    assert summary.sections["planning"]["value"] is None
    assert summary.publication_state == "BLOCK_NORMAL_AND_NOTIFY_FAILURE"


# **Validates: Requirements 2.11, 2.14, 3.11, 3.14**
def test_local_artifact_manifest_is_last_and_verifies_every_committed_file(
    monkeypatch, tmp_path,
):
    writer = LocalArtifactWriter(
        tmp_path,
        "attempt",
        storage=StorageDescriptor(
            storage_scope="local",
            automation_local_ephemeral=True,
            retention_guarantee="none",
            execution_environment="test-automation",
        ),
    )
    calls: list[str] = []
    real_atomic_write = writer._atomic_write

    def track(name: str, content: bytes):
        calls.append(name)
        return real_atomic_write(name, content)

    monkeypatch.setattr(writer, "_atomic_write", track)
    manifest_path = writer.finalize(
        analysis_id="analysis",
        summary={"schema_version": "1", "value": 0},
        ledger_entries=[{"sequence": 1, "entry_type": "STAGE"}],
        report_html="<html>report</html>",
        publication_state="SEND_NORMAL",
        newsletter_html="<html>newsletter</html>",
        what_if=LocalOnlyWorkbook(b"local workbook"),
        delivery_state="ACKNOWLEDGED_SUCCESS",
    )

    assert calls[-1] == "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["storage_scope"] == "local"
    assert manifest["automation_local_ephemeral"] is True
    assert manifest["retention_guarantee"] == "none"
    assert "what_if.xlsx" in manifest["files"]
    for name, expected in manifest["files"].items():
        payload = (manifest_path.parent / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
    assert not list(manifest_path.parent.glob("*.tmp"))


# **Validates: Requirements 2.14, 3.14**
def test_uncertain_delivery_is_consumed_until_an_authorized_resend(tmp_path):
    store = LocalJsonDeliveryClaimStore(tmp_path / "delivery-claims.json")
    base = DeliveryIntent(
        stable_event_id="scheduled:2026-07-17:close",
        purpose=DeliveryPurpose.NORMAL_NEWSLETTER,
        recipient_set_digest=recipient_set_digest(["reader@example.invalid"]),
        template_schema_version="1",
    )
    assert store.claim(base).created is True
    assert store.claim(base).duplicate is True
    store.transition(
        base.logical_id,
        (DeliveryState.CLAIMED,),
        DeliveryState.SMTP_INVOCATION_STARTED,
    )
    store.transition(
        base.logical_id,
        (DeliveryState.SMTP_INVOCATION_STARTED,),
        DeliveryState.UNCERTAIN,
    )
    replay = store.claim(base)
    assert replay.duplicate is True and replay.state is DeliveryState.UNCERTAIN
    with pytest.raises(ValueError, match="invalid delivery transition"):
        store.transition(
            base.logical_id,
            (DeliveryState.CLAIMED,),
            DeliveryState.SMTP_INVOCATION_STARTED,
        )

    resend = DeliveryIntent(
        stable_event_id=base.stable_event_id,
        purpose=base.purpose,
        recipient_set_digest=base.recipient_set_digest,
        template_schema_version=base.template_schema_version,
        authorized_resend_token="audited-resend-1",
    )
    assert resend.logical_id != base.logical_id
    assert store.claim(resend).created is True


# **Validates: Requirements 2.10, 2.14, 2.16, 3.10, 3.14, 3.16**
def test_apps_script_claim_adapter_and_workflow_share_the_durable_contract(
    monkeypatch,
):
    requests: list[dict] = []

    class Response:
        def __init__(self, document: dict):
            self._payload = json.dumps(document).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit: int) -> bytes:
            return self._payload[:limit]

    def fake_urlopen(request, timeout):
        document = json.loads(request.data.decode("utf-8"))
        requests.append(document)
        if document["action"] == "claim":
            return Response({
                "ok": True,
                "retention_days": 45,
                "created": True,
                "duplicate": False,
                "conflict": False,
                "state": "CLAIMED",
            })
        return Response({
            "ok": True,
            "retention_days": 45,
            "state": document["target"],
        })

    monkeypatch.setattr("tarzan.delivery.claims.urlopen", fake_urlopen)
    store = AppsScriptPropertiesDeliveryClaimStore(
        "https://claims.example.invalid/exec",
        "claim-secret",
        minimum_retention_days=30,
    )
    intent = DeliveryIntent(
        stable_event_id="update:thread:message",
        purpose=DeliveryPurpose.CRITICAL_FAILURE_NOTIFICATION,
        recipient_set_digest=recipient_set_digest(["reader@example.invalid"]),
        template_schema_version="1",
    )
    assert store.claim(intent).state is DeliveryState.CLAIMED
    assert store.transition(
        intent.logical_id,
        (DeliveryState.CLAIMED,),
        DeliveryState.SMTP_INVOCATION_STARTED,
    ) is DeliveryState.SMTP_INVOCATION_STARTED
    assert requests[0]["auth_token"] == "claim-secret"
    assert "reader@example.invalid" not in repr(requests)

    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/apps_script/Code.gs").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/newsletter.yml").read_text(encoding="utf-8")
    assert "function doPost(e)" in script
    assert "LockService.getScriptLock()" in script
    assert "DELIVERY_CLAIM_PREFIX = 'delivery_claim:'" in script
    assert "SENT_MARKER_PREFIX = 'sent:'" in script
    assert "SMTP_INVOCATION_STARTED: ['ACKNOWLEDGED_SUCCESS', 'UNCERTAIN']" in script
    assert "needs: validate" in workflow
    assert "TARZAN_STABLE_EVENT_ID:" in workflow
    assert "DELIVERY_CLAIM_ENDPOINT:" in workflow
    assert "DELIVERY_CLAIM_TOKEN:" in workflow


# **Validates: Requirements 2.7, 2.13, 3.7, 3.13**
def test_non_material_valuation_gap_is_degraded_and_fully_recorded():
    policies = {
        "STOCK:current_valuation": _policy("stock-current", 10.0),
        "BOND:current_valuation": _policy("bond-current", 10.0),
    }
    known = Holding(
        isin="KNOWN-BOUNDARY", ticker="KNOWN", quantity=1.0,
        cost_basis_eur=90.0, market_value_eur=90.0, currency="EUR",
        security_type="STOCK", current_price=90.0, current_value=90.0,
    )
    missing = Holding(
        isin="MISSING-BOUNDARY", ticker="MISSING", quantity=1.0,
        cost_basis_eur=10.0, market_value_eur=10.0, currency="EUR",
        security_type="BOND", current_price=None, current_value=None,
    )
    ledger = RunLedger("valuation-boundary")

    assessment = ValuationCompletenessEvaluator(policies).evaluate(
        [known, missing], ledger
    )

    assert assessment.availability is Availability.DEGRADED
    assert assessment.trustworthy_total_eur == 90.0
    assert assessment.known_subtotal_eur == 90.0
    assert assessment.missing_materiality_pct == pytest.approx(10.0)
    assert assessment.planning_eligible is True
    records = ledger.failure_records()
    assert [record.stable_code for record in records] == [
        "FALLBACK_VALUATION_UNAVAILABLE"
    ]
    assert records[0].closed is False
    assert records[0].availability is Availability.UNAVAILABLE
    assert records[0].analytical_impact
    assert records[0].publication_impact == "DEGRADE"
    assert PublicationEvaluator.evaluate(records).decision is PublicationDecision.SEND_DEGRADED_NORMAL


# **Validates: Requirements 2.15, 3.15**
def test_names_tickers_and_partial_categories_cannot_select_financial_behavior():
    from tarzan.data.enricher import classify_asset_class
    from tarzan.instruments import SupportState

    unresolved = Holding(
        isin="NO-GUESS", ticker="GOLD-BOND-CRYPTO", quantity=1.0,
        cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR",
        instrument_type="ETF",
    )
    first = classify_asset_class(
        {
            "quoteType": "ETF",
            "category": "large gold and bond blend",
            "sector": "crypto technology",
            "longName": "Gold Bond Crypto Fund",
        },
        unresolved.ticker,
        unresolved,
    )
    second = classify_asset_class(
        {
            "quoteType": "ETF",
            "category": "unmapped",
            "sector": "unmapped",
            "longName": "Completely Different Name",
        },
        "DIFFERENT-TICKER",
        unresolved,
    )
    assert first.support is SupportState.UNSUPPORTED
    assert second.support is SupportState.UNSUPPORTED
    assert first.availability is Availability.UNAVAILABLE
    assert second.availability is Availability.UNAVAILABLE

    explicit_gold = Holding(
        isin="EXPLICIT-GOLD", ticker="ANY", quantity=1.0,
        cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR",
        instrument_type="ETF", asset_type="Gold",
    )
    assert classify_asset_class(
        {"quoteType": "ETF", "longName": "Name is not evidence"},
        explicit_gold.ticker,
        explicit_gold,
    ).value == "Gold"

    exact_fixed_income = Holding(
        isin="EXPLICIT-FI", ticker="ANY", quantity=1.0,
        cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR",
        instrument_type="ETF",
    )
    assert classify_asset_class(
        {"quoteType": "ETF", "category": "Fixed Income"},
        exact_fixed_income.ticker,
        exact_fixed_income,
    ).value == "Fixed Income"


# **Validates: Requirements 2.2, 2.6, 2.11, 3.2, 3.6, 3.11**
def test_analysis_id_is_invariant_to_telemetry_schema_and_payload():
    from datetime import datetime, timezone

    from tarzan.orchestrator import _analysis_evidence
    from tarzan.runtime.session import (
        RunContext,
        RunMode,
        RunSession,
        canonical_analysis_id,
    )

    def session(telemetry_version: str, telemetry_payload: dict) -> RunSession:
        value = RunSession(
            context=RunContext(
                attempt_id=f"attempt-{telemetry_version}",
                mode=RunMode.LIVE,
                effective_date=None,
                captured_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
                schema_versions={"input": "2", "telemetry": telemetry_version},
                policy_versions={"release": "1.0"},
            ),
            config_snapshot={"target": 60.0},
            ledger=RunLedger(f"attempt-{telemetry_version}"),
        )
        value.memo["effective_orders"] = {"digest": "stable-orders"}
        value.memo["valuation"] = {"trustworthy_total_eur": 90.0}
        value.memo["telemetry"] = telemetry_payload
        value.memo["workload"] = {"telemetry": telemetry_payload}
        return value

    first = session("1.0", {"orders": 1, "duration_ms": 2.0})
    second = session("999.0", {"orders": 1_000_000, "duration_ms": 999_999.0})

    assert canonical_analysis_id(_analysis_evidence(first)) == canonical_analysis_id(
        _analysis_evidence(second)
    )
    assert "telemetry" not in _analysis_evidence(first)["schema_versions"]


# **Validates: Requirements 2.7, 2.13, 3.7, 3.13**
def test_materiality_aggregates_missing_rows_by_kind_and_policy():
    policies = {
        "STOCK:current_valuation": _policy("stock-current", 10.0),
        "BOND:current_valuation": _policy("bond-current", 10.0),
    }
    holdings = [
        Holding(
            isin="KNOWN-88", ticker="KNOWN", quantity=1.0,
            cost_basis_eur=88.0, market_value_eur=88.0, currency="EUR",
            security_type="STOCK", current_price=88.0, current_value=88.0,
        ),
        Holding(
            isin="MISSING-6-A", ticker="MISSING-A", quantity=1.0,
            cost_basis_eur=6.0, market_value_eur=6.0, currency="EUR",
            security_type="BOND", current_price=None, current_value=None,
        ),
        Holding(
            isin="MISSING-6-B", ticker="MISSING-B", quantity=1.0,
            cost_basis_eur=6.0, market_value_eur=6.0, currency="EUR",
            security_type="BOND", current_price=None, current_value=None,
        ),
    ]
    ledger = RunLedger("aggregate-materiality")

    assessment = ValuationCompletenessEvaluator(policies).evaluate(holdings, ledger)

    assert assessment.missing_materiality_pct == pytest.approx(12.0)
    assert assessment.availability is Availability.UNAVAILABLE
    assert assessment.trustworthy_total_eur is None
    assert assessment.known_subtotal_eur == 88.0
    assert assessment.planning_eligible is False
    critical = next(
        record for record in ledger.failure_records()
        if record.stable_code == "MATERIAL_VALUATION_GAP"
    )
    critical_entry = next(
        entry for entry in ledger.entries
        if entry.payload.get("failure_id") == critical.failure_id
        and entry.entry_type.value == "FAILURE_OPEN"
    )
    assert critical_entry.payload["context"]["material_groups"] == [{
        "instrument_kind": "BOND",
        "policy_id": "bond-current",
        "missing_count": 2,
        "missing_basis_eur": 12.0,
        "missing_materiality_pct": 12.0,
        "threshold_pct": 10.0,
    }]


# **Validates: Requirements 2.4, 2.7, 2.13, 3.4, 3.7, 3.13**
def test_non_material_assessment_is_authoritative_for_metrics_and_planning(monkeypatch):
    from tarzan.engine.metrics import MetricsEngine

    policies = {
        "STOCK:current_valuation": _policy("stock-current", 10.0),
        "BOND:current_valuation": _policy("bond-current", 10.0),
    }
    known = Holding(
        isin="KNOWN-90", ticker="KNOWN", quantity=1.0,
        cost_basis_eur=90.0, market_value_eur=90.0, currency="EUR",
        security_type="STOCK", asset_class=AssetClass.EQUITIES,
        current_price=90.0, current_value=90.0,
    )
    rejected = Holding(
        isin="REJECTED-10", ticker="REJECTED", quantity=1.0,
        cost_basis_eur=10.0, market_value_eur=10.0, currency="EUR",
        security_type="BOND", current_price=None, current_value=None,
    )
    ledger = RunLedger("authoritative-valuation")
    assessment = ValuationCompletenessEvaluator(policies).evaluate(
        [known, rejected], ledger
    )
    assert assessment.planning_eligible is True

    planning_calls: list[tuple[list[Holding], float]] = []

    def fake_rebalancing(holdings, config, total_value, lump_sum=None):
        planning_calls.append((list(holdings), float(total_value)))
        return [], []

    monkeypatch.setattr(
        "tarzan.engine.rebalancer.compute_unified_rebalancing",
        fake_rebalancing,
    )
    monkeypatch.setattr(
        "tarzan.engine.rebalancer.plan_cost",
        lambda *args, **kwargs: {"cgt_eur": 0.0, "fees_eur": 0.0},
    )
    monkeypatch.setattr(
        "tarzan.runtime.audit.record_rebalancing_plan",
        lambda *args, **kwargs: None,
    )

    engine = MetricsEngine(
        [known, rejected],
        InvestorConfig(),
        planning_eligible=assessment.planning_eligible,
        valuation_assessment=assessment,
    )
    context: dict = {}
    engine._valuation(context)
    engine._allocations(context)
    engine._rebalancing(context)
    metrics = engine._build_result(context)
    summary = SummaryProjector.project(_result(metrics, ledger))

    assert metrics.total_value == 90.0
    assert metrics.trustworthy_total_value_eur == 90.0
    assert metrics.known_valuation_subtotal_eur == 90.0
    assert list(metrics.holdings_df["isin"]) == ["KNOWN-90"]
    assert metrics.to_summary_dict()["total_value_eur"] == 90.0
    assert summary.metrics["total_value_eur"] == 90.0
    assert summary.sections["portfolio"]["trustworthy_total_eur"] == 90.0
    assert summary.sections["portfolio"]["known_subtotal_eur"] == 90.0
    assert len(planning_calls) == 2
    for projected_holdings, total in planning_calls:
        assert total == 90.0
        assert [holding.isin for holding in projected_holdings] == ["KNOWN-90"]
        assert projected_holdings[0].current_value == 90.0


# **Validates: Requirements 2.14, 3.14**
def test_local_delivery_claim_store_rejects_every_prohibited_edge(tmp_path):
    store = LocalJsonDeliveryClaimStore(tmp_path / "delivery-graph.json")
    allowed = {
        DeliveryState.CLAIMED: {
            DeliveryState.SMTP_INVOCATION_STARTED,
            DeliveryState.DEFINITE_PRE_SEND_FAILURE,
        },
        DeliveryState.SMTP_INVOCATION_STARTED: {
            DeliveryState.ACKNOWLEDGED_SUCCESS,
            DeliveryState.UNCERTAIN,
        },
        DeliveryState.ACKNOWLEDGED_SUCCESS: set(),
        DeliveryState.DEFINITE_PRE_SEND_FAILURE: set(),
        DeliveryState.UNCERTAIN: set(),
    }
    paths = {
        DeliveryState.CLAIMED: (),
        DeliveryState.SMTP_INVOCATION_STARTED: (
            DeliveryState.SMTP_INVOCATION_STARTED,
        ),
        DeliveryState.ACKNOWLEDGED_SUCCESS: (
            DeliveryState.SMTP_INVOCATION_STARTED,
            DeliveryState.ACKNOWLEDGED_SUCCESS,
        ),
        DeliveryState.DEFINITE_PRE_SEND_FAILURE: (
            DeliveryState.DEFINITE_PRE_SEND_FAILURE,
        ),
        DeliveryState.UNCERTAIN: (
            DeliveryState.SMTP_INVOCATION_STARTED,
            DeliveryState.UNCERTAIN,
        ),
    }

    for source, path in paths.items():
        intent = DeliveryIntent(
            stable_event_id=f"graph:{source.value}",
            purpose=DeliveryPurpose.NORMAL_NEWSLETTER,
            recipient_set_digest=recipient_set_digest(["reader@example.invalid"]),
            template_schema_version="1",
        )
        store.claim(intent)
        current = DeliveryState.CLAIMED
        for target in path:
            store.transition(intent.logical_id, (current,), target)
            current = target
        assert current is source

        for target in DeliveryState:
            if target in allowed[source]:
                continue
            with pytest.raises(
                ValueError,
                match=f"invalid delivery transition {source.value} -> {target.value}",
            ):
                store.transition(intent.logical_id, (source,), target)
            assert store.claim(intent).state is source


# **Validates: Requirements 2.7, 2.13, 3.7, 3.13**
def test_indeterminate_valuation_basis_is_critical_even_with_known_denominator():
    from types import SimpleNamespace

    policies = {
        "STOCK:current_valuation": _policy("stock-current", 100.0),
        "BOND:current_valuation": _policy("bond-current", 100.0),
    }
    known = Holding(
        isin="KNOWN-BASIS", ticker="KNOWN", quantity=1.0,
        cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR",
        security_type="STOCK", current_price=100.0, current_value=100.0,
    )
    indeterminate = SimpleNamespace(
        isin="UNKNOWN-BASIS",
        ticker="UNKNOWN-BASIS",
        security_type="BOND",
        instrument_type=None,
        instrument_kind_evidence=(),
        current_price=None,
        current_value=None,
        market_value_eur=None,
        cost_basis_eur=None,
        fetch_timestamp=None,
        data_source=None,
        is_seeded_target=False,
    )
    ledger = RunLedger("indeterminate-materiality")

    assessment = ValuationCompletenessEvaluator(policies).evaluate(
        [known, indeterminate], ledger
    )

    assert assessment.availability is Availability.UNAVAILABLE
    assert assessment.missing_materiality_pct is None
    assert assessment.trustworthy_total_eur is None
    assert assessment.planning_eligible is False
    critical = next(
        record for record in ledger.failure_records()
        if record.stable_code == "MATERIAL_VALUATION_GAP"
    )
    critical_entry = next(
        entry for entry in ledger.entries
        if entry.payload.get("failure_id") == critical.failure_id
        and entry.entry_type.value == "FAILURE_OPEN"
    )
    assert critical_entry.payload["context"]["material_groups"] == [{
        "instrument_kind": "BOND",
        "policy_id": "bond-current",
        "missing_count": 1,
        "missing_basis_eur": 0,
        "missing_materiality_pct": None,
        "threshold_pct": 100.0,
        "materiality_indeterminate": True,
    }]


# **Validates: Requirements 2.14, 3.14**
def test_remote_claim_store_enforces_graph_before_io_and_exact_target(monkeypatch):
    store = AppsScriptPropertiesDeliveryClaimStore(
        "https://claims.example.invalid/exec",
        "claim-secret",
    )
    requests: list[dict] = []

    def permissive_call(request):
        requests.append(request)
        return {"state": DeliveryState.UNCERTAIN.value}

    monkeypatch.setattr(store, "_call", permissive_call)

    with pytest.raises(
        ValueError,
        match="invalid delivery transition CLAIMED -> ACKNOWLEDGED_SUCCESS",
    ):
        store.transition(
            "logical-id",
            (DeliveryState.CLAIMED,),
            DeliveryState.ACKNOWLEDGED_SUCCESS,
        )
    assert requests == []

    with pytest.raises(RuntimeError, match="did not apply the requested transition"):
        store.transition(
            "logical-id",
            (DeliveryState.CLAIMED,),
            DeliveryState.SMTP_INVOCATION_STARTED,
        )
    assert len(requests) == 1


# **Validates: Requirements 2.4, 2.12, 2.15, 3.4, 3.12, 3.15**
def test_unclassified_etf_suppresses_category_outputs_and_planning(monkeypatch):
    from tarzan.engine.metrics import MetricsEngine

    holding = Holding(
        isin="UNCLASSIFIED-ETF",
        ticker="UNCLASSIFIED-ETF",
        quantity=10.0,
        cost_basis_eur=1000.0,
        market_value_eur=1050.0,
        currency="EUR",
        security_type="ETF",
        instrument_type="ETF",
        current_price=105.0,
        current_value=1050.0,
    )
    engine = MetricsEngine([holding], InvestorConfig())
    context: dict = {}
    engine._valuation(context)
    engine._allocations(context)

    assert context["holdings_df"].iloc[0]["asset_class"] is None
    assert context["classification_available"] is False
    assert context["classification_unavailable_instruments"] == [
        "UNCLASSIFIED-ETF"
    ]
    assert context["allocation_by_class"].empty
    assert context["allocation_by_geo"].empty
    assert context["allocation_by_sector"].empty

    engine._goals(context)
    assert context["goal_deltas"] is None

    monkeypatch.setattr(
        "tarzan.engine.rebalancer.compute_unified_rebalancing",
        lambda *args, **kwargs: pytest.fail(
            "unclassified ETF reached category-dependent planning"
        ),
    )
    engine._rebalancing(context)
    assert context["rebalancing_suggestions"] is None
    assert context["rebalancing_verifications"] is None
    assert context["rebalancing_plans"] is None


# **Validates: Requirements 2.7, 2.13, 3.7, 3.13**
def test_order_derived_zero_anchor_has_indeterminate_materiality():
    import datetime

    from tarzan.engine.returns_builder import build_holdings_from_orders
    from tarzan.models.order import Order, OrderType

    unresolved = build_holdings_from_orders([
        Order(
            date=datetime.date(2025, 1, 1),
            trade_date=datetime.date(2025, 1, 1),
            type=OrderType.BUY,
            isin="UNRESOLVED-ZERO-ANCHOR",
            name="Unresolved",
            ticker="",
            quantity=10.0,
            currency="EUR",
            price_native=100.0,
            fx_rate=1.0,
            gross_eur=0.0,
            fees_eur=0.0,
            net_eur=0.0,
            instrument_kind=None,
        )
    ])[0]
    assert unresolved.quantity == 10.0
    assert unresolved.market_value_eur == 0.0
    assert unresolved.cost_basis_eur == 0.0

    known = Holding(
        isin="KNOWN-DENOMINATOR", ticker="KNOWN", quantity=1.0,
        cost_basis_eur=100.0, market_value_eur=100.0, currency="EUR",
        security_type="STOCK", current_price=100.0, current_value=100.0,
    )
    policies = {
        "STOCK:current_valuation": _policy("stock-current", 100.0),
        "UNKNOWN:current_valuation": _policy("unknown-current", 100.0),
    }
    assessment = ValuationCompletenessEvaluator(policies).evaluate(
        [known, unresolved],
        RunLedger("order-zero-anchor"),
    )

    unresolved_evidence = next(
        evidence for evidence in assessment.evidence
        if evidence.instrument_kind == "UNKNOWN"
    )
    assert unresolved_evidence.materiality_basis_eur is None
    assert assessment.missing_materiality_pct is None
    assert assessment.availability is Availability.UNAVAILABLE
    assert assessment.trustworthy_total_eur is None
    assert assessment.planning_eligible is False
