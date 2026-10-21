"""Immutable boundary, complete analysis identity and concurrent log projection."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
import datetime as dt

import pandas as pd
import pytest

from tarzan import orchestrator, runtime
from tarzan.contracts.material import material_metrics
from tarzan.models.order import Order, OrderFact, OrderType
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.effective_orders import EffectiveOrderSnapshot
from tarzan.runtime.ledger import Availability, ErrorNormalizer, LedgerEntryType, RunLedger
from tarzan.runtime.session import RunSession, canonical_analysis_id


def order(**changes):
    values = dict(
        date=dt.date(2025, 1, 2), trade_date=dt.date(2025, 1, 2),
        type=OrderType.BUY, isin="SYNTH", name="Synthetic", ticker="SYNTH",
        quantity=10.0, currency="EUR", price_native=100.0, fx_rate=1.0,
        gross_eur=1000.0, fees_eur=0.0, net_eur=-1000.0, instrument_kind="ETF",
    )
    values.update(changes)
    return Order(**values)


def failure(ledger):
    return ledger.open_failure(
        stage="fixture", stable_code="SYNTHETIC", severity="WARNING",
        error={"details": {"values": [10.0]}}, affected_outputs=["portfolio"],
        analytical_impact="fixture", publication_impact="DEGRADE",
    )


def test_effective_order_fact_has_no_source_alias_or_mutable_field():
    source = order()
    snapshot = EffectiveOrderSnapshot.build([source, source], None)
    assert snapshot.accepted_count == 2  # legitimate equal broker fills are retained
    assert snapshot.orders == (source, source)
    assert all(isinstance(item, OrderFact) for item in snapshot.orders)
    source.quantity = 99.0
    assert snapshot.orders[0].quantity == 10.0
    with pytest.raises(FrozenInstanceError):
        snapshot.orders[0].net_eur = 999.0
    assert snapshot.orders[0].acquisition_cost_eur() == 1000.0
    assert replace(snapshot.orders[0], instrument_kind=None).instrument_kind is None


@pytest.mark.parametrize("change", [
    {"fees_eur": 1.0}, {"fx_rate": 0.9}, {"currency": "USD"},
    {"price_native": 101.0}, {"instrument_kind": "BOND"},
    {"instrument_equivalence_group": "different"}, {"name": "Different issuer"},
])
def test_effective_digest_covers_all_financial_and_identity_authority(change):
    assert EffectiveOrderSnapshot.build([order()], None).digest != (
        EffectiveOrderSnapshot.build([order(**change)], None).digest
    )


def test_order_fact_contract_tracks_the_ingestion_record_fields():
    assert [item.name for item in fields(OrderFact)] == [item.name for item in fields(Order)]


def test_deep_ledger_entry_and_failure_projection_cannot_mutate_authority():
    ledger = RunLedger("fixture")
    fid = failure(ledger)
    ledger.remedy(fid, remedy_id="one", action="fixture", outcome="SUCCEEDED",
                  availability=Availability.DEGRADED, provenance=["fixture"])
    entry = ledger.entries[0]
    with pytest.raises(TypeError):
        entry.payload["original_failure"]["details"]["values"][0] = 999.0
    projection = entry.to_dict()
    projection["payload"]["original_failure"]["details"]["values"][0] = 999.0
    record = ledger.failure_records()[0]
    assert record.original_failure["details"]["values"] == (10.0,)
    with pytest.raises(TypeError):
        record.remedies[0]["action"] = "rewritten"
    # Normalization supports frozen mappings/dataclasses without deepcopy.
    assert ErrorNormalizer.normalize(record)["original_failure"]["details"]["values"] == [10.0]


def test_concurrent_remedies_have_atomic_contiguous_ordinals():
    ledger = RunLedger("concurrency")
    fid = failure(ledger)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(
            lambda i: ledger.remedy(fid, remedy_id=str(i), action="fixture",
                                    outcome="SUCCEEDED", availability=Availability.DEGRADED),
            range(200),
        ))
    records = ledger.failure_records()
    assert [item["ordinal"] for item in records[0].remedies] == list(range(1, 201))
    assert len({entry.sequence for entry in ledger.entries}) == 201


def test_failure_projection_is_reused_until_failure_evidence_changes():
    ledger = RunLedger("cache")
    fid = failure(ledger)
    first = ledger.failure_records()
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {"provider": "fixture"})
    assert ledger.failure_records() is first
    ledger.close_failure(fid, automatically_corrected=True, selected_resolution="fixture",
                         availability=Availability.AVAILABLE)
    second = ledger.failure_records()
    assert second is not first and second[0].closed
    assert first[0].closed is False


def analysis(metrics, memo=None):
    session = RunSession(runtime.context(), {}, RunLedger("fixture"), memo=memo or {})
    return canonical_analysis_id(orchestrator._analysis_evidence(session, metrics))


@pytest.mark.parametrize("field", [
    "holding_performance", "holdings_df", "allocation_by_class",
    "benchmark_comparison", "portfolio_history", "actual_value_series",
    "pnl_series", "unrealized_series", "target_history",
])
def test_every_published_financial_table_or_full_series_affects_identity(field):
    a = PortfolioMetrics(total_value=1000.0)
    if field.endswith("history") or field.endswith("series"):
        before = pd.Series(range(1000), index=pd.bdate_range("2020-01-01", periods=1000), dtype=float)
        after = before.copy()
        after.iloc[500] += 1.0  # absent from pandas' truncated display
        assert str(before) == str(after)
    else:
        before = pd.DataFrame([{"ticker": "SYNTH", "value": 1.0}])
        after = pd.DataFrame([{"ticker": "SYNTH", "value": 5.0}])
    setattr(a, field, before)
    b = replace(a, **{field: after})
    assert analysis(a) != analysis(b)


def test_target_and_resolved_identity_evidence_are_material():
    metrics = PortfolioMetrics()
    assert analysis(metrics, {"target_plan": {"rows": {"SYNTH": {"target_portfolio": 50}}}}) != (
        analysis(metrics, {"target_plan": {"rows": {"SYNTH": {"target_portfolio": 60}}}})
    )
    assert material_metrics(metrics)["planning_availability"] == "AVAILABLE"


def test_excluded_future_order_count_does_not_change_causal_identity():
    evidence = {"digest": "same-visible-orders", "accepted_count": 10, "boundary": "2025-01-02"}
    metrics = PortfolioMetrics()
    assert analysis(metrics, {"effective_orders": {**evidence, "excluded_count": 0}}) == (
        analysis(metrics, {"effective_orders": {**evidence, "excluded_count": 100}})
    )
