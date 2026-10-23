"""Synthetic fidelity and work-count tests for narrow report projections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import product

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts import report
from tarzan.contracts.immutable import thaw_json
from tarzan.contracts.report import ReportView
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import Availability, RunLedger
from tarzan.runtime.publication import (
    DeliveryPurpose, PublicationDecision, PublicationOutcome,
)
from tarzan.runtime.session import RunResult
from tarzan.runtime.summary import SummaryProjector


def _metrics(**changes):
    metrics = PortfolioMetrics(
        total_value=1111.987654,
        invested_value=1000.0,
        cash_value=111.987654,
        holdings_df=pd.DataFrame({
            "ticker": ["SYNTH"],
            "cost_basis_eur": [1000.123456],
            "nested": [{"items": [1]}],
        }),
        performance={"1m": 2.12345678},
        risk={"sharpe": float("nan"), "alpha": 0.12345678},
        rebalancing_suggestions=[{"ticker": "SYNTH", "quantity": np.int64(2)}],
        rebalancing_verifications=[{"valid": True, "nested": {"items": [7]}}],
    )
    for name, value in changes.items():
        setattr(metrics, name, value)
    return metrics


def _result(metrics=None):
    metrics = _metrics() if metrics is None else metrics
    config = InvestorConfig()
    return RunResult(
        metrics, config, "synthetic-attempt", "synthetic-analysis",
        RunLedger("synthetic-attempt"), report_view=ReportView.build(metrics, config),
    )


def _fail(result, *, stage, severity="WARNING", affected=("market_context_summary",)):
    return result.ledger.open_failure(
        stage=stage, stable_code="SYNTHETIC_FAILURE", severity=severity,
        error="synthetic evidence", affected_outputs=affected,
        analytical_impact="synthetic impact",
        publication_impact=(
            "BLOCK_NORMAL_AND_DO_NOT_SEND"
            if stage == "newsletter_semantics" else "DEGRADE"
        ),
    )


def _legacy_projection(result):
    metrics, _ = result.report_view.project()
    return replace(result, metrics=metrics, report_view=None)


def test_summary_uses_no_full_pandas_or_config_projection(monkeypatch):
    result = _result()

    def full_projection_forbidden(*args, **kwargs):
        pytest.fail("summary must use immutable narrow facts, not a full cube")

    monkeypatch.setattr(ReportView, "project", full_projection_forbidden)
    assert SummaryProjector.project(result).metrics["total_value_eur"] == 1111.99


def test_machine_serializer_is_used_once_at_report_build_not_per_summary(monkeypatch):
    calls = []
    native = PortfolioMetrics.to_summary_dict

    def counted(metrics):
        calls.append(metrics)
        return native(metrics)

    monkeypatch.setattr(PortfolioMetrics, "to_summary_dict", counted)
    result = _result()
    assert len(calls) == 1
    first = SummaryProjector.project(result)
    for _ in range(3):
        assert SummaryProjector.project(result).canonical_bytes() == first.canonical_bytes()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "valuation,history,planning,with_returns",
    list(product(
        ("AVAILABLE", "DEGRADED", "UNAVAILABLE"),
        ("AVAILABLE", "DEGRADED", "UNAVAILABLE"),
        ("AVAILABLE", "UNAVAILABLE"),
        (True, False),
    )),
)
def test_narrow_summary_has_exact_legacy_machine_and_section_bytes(
    valuation, history, planning, with_returns,
):
    metrics = _metrics(
        valuation_availability=valuation,
        history_availability=history,
        planning_availability=planning,
        trustworthy_total_value_eur=0.0 if valuation == "AVAILABLE" else None,
        known_valuation_subtotal_eur=123.456789,
        xirr_pct=0.0 if with_returns else None,
        twr_pct=-0.123456789 if with_returns else None,
        twr_annualized_pct=float("inf") if with_returns else None,
        returns_coverage_pct=0.0 if with_returns else None,
    )
    result = _result(metrics)
    legacy = _legacy_projection(result)
    assert SummaryProjector.project(result).canonical_bytes() == (
        SummaryProjector.project(legacy).canonical_bytes()
    )


def test_summary_facts_are_deeply_immutable_and_render_mutation_is_isolated():
    result = _result()
    before = SummaryProjector.project(result).canonical_bytes()
    result.metrics.total_value = 999.0
    result.metrics.rebalancing_suggestions[0]["quantity"] = 99
    result.metrics.rebalancing_verifications[0]["nested"]["items"].append(99)
    projected, _ = result.report_view.project()
    projected.total_value = 777.0
    projected.performance["1m"] = 777.0
    projected.rebalancing_verifications[0]["nested"]["items"].append(77)
    facts = result.report_view.project_summary_facts()
    with pytest.raises(TypeError):
        facts.machine_metrics["total_value_eur"] = 555.0
    with pytest.raises(TypeError):
        facts.planning_value["actions"][0]["quantity"] = 55
    assert SummaryProjector.project(result).canonical_bytes() == before
    summary_dict = SummaryProjector.project(result).to_dict()
    assert summary_dict["sections"]["planning"]["value"]["actions"][0]["quantity"] == 2


@pytest.mark.parametrize(
    "stage,severity,affected,decision",
    [
        ("gemini", "WARNING", ("market_context_summary",), "SEND_DEGRADED_NORMAL"),
        ("newsletter_semantics", "CRITICAL", ("newsletter",), "BLOCK_ALL_DELIVERY"),
        ("delivery_reconciliation", "CRITICAL", ("delivery", "publication"),
         "BLOCK_NORMAL_AND_NOTIFY_FAILURE"),
        ("valuation", "CRITICAL", ("portfolio", "planning"),
         "BLOCK_NORMAL_AND_NOTIFY_FAILURE"),
    ],
)
def test_new_ai_semantic_or_delivery_evidence_changes_a_later_summary(
    stage, severity, affected, decision,
):
    result = _result()
    initial = SummaryProjector.project(result)
    assert initial.publication_state == "SEND_NORMAL"
    _fail(result, stage=stage, severity=severity, affected=affected)
    final = SummaryProjector.project(result)
    assert final.publication_state == decision
    assert final.canonical_bytes() == SummaryProjector.project(
        _legacy_projection(result),
    ).canonical_bytes()
    assert initial.publication_state == "SEND_NORMAL"


def test_new_remedy_and_explicit_publication_outcome_are_not_cached():
    result = _result()
    failure_id = _fail(
        result, stage="valuation", severity="CRITICAL",
        affected=("portfolio", "planning"),
    )
    first = SummaryProjector.project(result)
    assert first.sections["portfolio"]["availability"] == "UNAVAILABLE"
    result.ledger.close_failure(
        failure_id, automatically_corrected=True,
        selected_resolution="synthetic-remedy", availability=Availability.DEGRADED,
    )
    corrected = SummaryProjector.project(result)
    assert corrected.publication_state == "SEND_DEGRADED_NORMAL"
    assert corrected.sections["portfolio"]["availability"] == "DEGRADED"
    explicit = PublicationOutcome(
        PublicationDecision.BLOCK_ALL_DELIVERY, DeliveryPurpose.NO_DELIVERY,
    )
    assert SummaryProjector.project(result, explicit).publication_state == (
        "BLOCK_ALL_DELIVERY"
    )


def test_field_projection_is_exact_and_defensive_for_typed_nested_pandas():
    index = pd.MultiIndex.from_tuples([("a", 1)], names=["letter", "ordinal"])
    frame = pd.DataFrame([[1.0, 2.0]], columns=["same", "same"], index=index)
    series = pd.Series(
        [1, 2], dtype="Int64", name="synthetic",
        index=pd.date_range("2025-01-01", periods=2, tz="Europe/Rome"),
    )
    series.attrs["nested"] = {"items": [7]}
    metrics = _metrics(holdings_df=frame, portfolio_history=series)
    view = ReportView.build(metrics, InvestorConfig())
    selected = view.project_metrics_fields(
        ("holdings_df", "portfolio_history", "rebalancing_verifications"),
    )
    assert tuple(selected) == (
        "holdings_df", "portfolio_history", "rebalancing_verifications",
    )
    pd.testing.assert_frame_equal(selected["holdings_df"], frame)
    pd.testing.assert_series_equal(selected["portfolio_history"], series)
    selected["holdings_df"].iloc[0, 0] = 999.0
    selected["portfolio_history"].attrs["nested"]["items"].append(99)
    selected["rebalancing_verifications"][0]["nested"]["items"].append(99)
    fresh = view.project_metrics_fields(tuple(selected))
    assert fresh["holdings_df"].iloc[0, 0] == 1.0
    assert fresh["portfolio_history"].attrs == {"nested": {"items": [7]}}
    assert fresh["rebalancing_verifications"][0]["nested"]["items"] == [7]


def test_empty_or_scalar_field_request_projects_no_frame(monkeypatch):
    view = ReportView.build(_metrics(), InvestorConfig())
    native = report._project

    def frame_projection_forbidden(value):
        if isinstance(value, (report._FrameFact, report._SeriesFact)):
            pytest.fail("unrequested pandas facts must not be reconstructed")
        return native(value)

    monkeypatch.setattr(report, "_project", frame_projection_forbidden)
    assert view.project_metrics_fields(()) == {}
    assert view.project_metrics_fields(("total_value",)) == {
        "total_value": 1111.987654,
    }
    assert view.project_metric_properties(()) == {}


@pytest.mark.parametrize(
    "names",
    [
        ("unknown",), ("total_value", "unknown"), ("total_value", "total_value"),
        "total_value", (42,), ([],), None,
    ],
)
def test_field_names_are_validated_before_any_projection(monkeypatch, names):
    view = ReportView.build(_metrics(), InvestorConfig())

    def projection_forbidden(*args):
        pytest.fail("invalid selection must fail before reconstructing any fact")

    monkeypatch.setattr(report, "_project", projection_forbidden)
    with pytest.raises(ValueError):
        view.project_metrics_fields(names)


@pytest.mark.parametrize(
    "names", [("unknown",), ("total_value",), ("unrealized_pnl_eur",) * 2, "unrealized_pnl_eur"],
)
def test_only_whitelisted_metric_properties_can_be_requested(names):
    view = ReportView.build(_metrics(), InvestorConfig())
    with pytest.raises(ValueError):
        view.project_metric_properties(names)


def test_properties_use_raw_original_getters_with_one_minimal_projection(monkeypatch):
    metrics = _metrics()
    expected = {
        "unrealized_pnl_eur": metrics.unrealized_pnl_eur,
        "unrealized_pnl_pct": metrics.unrealized_pnl_pct,
    }
    view = ReportView.build(metrics, InvestorConfig())
    metrics.total_value = 999.0
    metrics.holdings_df.loc[0, "cost_basis_eur"] = 999.0
    projected_fields = []
    native = ReportView.project_metrics_fields

    def counted(self, names):
        projected_fields.append(tuple(names))
        return native(self, names)

    monkeypatch.setattr(ReportView, "project_metrics_fields", counted)
    actual = view.project_metric_properties(tuple(expected))
    assert actual == expected
    assert projected_fields == [("holdings_df", "total_value")]
    assert actual["unrealized_pnl_eur"] != round(actual["unrealized_pnl_eur"], 2)


def test_absent_property_dependencies_do_not_add_an_eager_build_failure():
    view = ReportView.build(
        _metrics(holdings_df=pd.DataFrame({"same": [1]})), InvestorConfig(),
    )
    assert view.project_metrics_fields(("total_value",))["total_value"] == 1111.987654
    with pytest.raises(KeyError, match="cost_basis_eur"):
        view.project_metric_properties(("unrealized_pnl_eur",))


def test_zero_cost_property_null_semantics_match_the_original_authority():
    metrics = _metrics(holdings_df=pd.DataFrame({"cost_basis_eur": [0.0]}))
    view = ReportView.build(metrics, InvestorConfig())
    assert view.project_metric_properties(
        ("unrealized_pnl_eur", "unrealized_pnl_pct"),
    ) == {
        "unrealized_pnl_eur": metrics.unrealized_pnl_eur,
        "unrealized_pnl_pct": None,
    }


def test_generic_dataclass_field_projection_preserves_legacy_report_build():
    @dataclass
    class SyntheticRecord:
        nested: dict
        unused: list

    view = ReportView.build(
        SyntheticRecord({"items": [1]}, [7]), InvestorConfig(),
    )
    copied = view.project_metrics_fields(("nested",))
    copied["nested"]["items"].append(99)
    assert view.project_metrics_fields(("nested",)) == {"nested": {"items": [1]}}


def test_summary_serialization_error_keeps_legacy_build_and_failure_boundary(monkeypatch):
    def invalid_serializer(metrics):
        raise ValueError("synthetic invalid summary")

    monkeypatch.setattr(PortfolioMetrics, "to_summary_dict", invalid_serializer)
    result = _result()
    assert result.report_view.project_metrics_fields(("total_value",))
    with pytest.raises(ValueError, match="synthetic invalid summary"):
        SummaryProjector.project(result)


def test_legacy_result_without_report_preserves_its_existing_numpy_plan_contract():
    metrics = _metrics()
    result = RunResult(
        metrics, InvestorConfig(), "synthetic-attempt", "synthetic-analysis",
        RunLedger("synthetic-attempt"),
    )
    summary = SummaryProjector.project(result).to_dict()
    assert summary["sections"]["planning"]["value"]["actions"][0]["quantity"] == "2"


def test_legacy_four_argument_report_constructor_retains_summary_parity():
    prepared = _result()
    view = prepared.report_view
    legacy_view = ReportView(
        view._metrics, view._config, view.benchmark_alpha_beta, view.benchmark_geo,
    )
    assert SummaryProjector.project(
        replace(prepared, report_view=legacy_view),
    ).canonical_bytes() == SummaryProjector.project(prepared).canonical_bytes()


@pytest.mark.parametrize("with_report", [True, False])
def test_critical_unavailable_portfolio_does_not_request_broken_machine_values(
    monkeypatch, with_report,
):
    calls = []

    def invalid_serializer(metrics):
        calls.append(metrics)
        raise ValueError("synthetic invalid summary")

    monkeypatch.setattr(PortfolioMetrics, "to_summary_dict", invalid_serializer)
    result = _result()
    if not with_report:
        result = replace(result, report_view=None)
    before = len(calls)
    _fail(result, stage="valuation", severity="CRITICAL", affected=("portfolio",))
    summary = SummaryProjector.project(result)
    assert summary.sections["portfolio"]["availability"] == "UNAVAILABLE"
    assert summary.metrics == {}
    assert len(calls) == before


def test_absent_metrics_keeps_legacy_unavailable_sections():
    result = RunResult(
        None, None, "synthetic-attempt", "synthetic-analysis",
        RunLedger("synthetic-attempt"),
    )
    summary = SummaryProjector.project(result)
    assert summary.sections["portfolio"]["availability"] == "UNAVAILABLE"
    assert summary.sections["planning"]["availability"] == "UNAVAILABLE"
    assert summary.metrics == {}


@pytest.mark.parametrize("with_report", [True, False])
def test_unavailable_plan_does_not_serialize_its_unused_nested_values(with_report):
    class SyntheticUnavailableValue(Enum):
        VALUE = "synthetic"

        def __str__(self):
            raise ValueError("unavailable synthetic value must not be serialized")

    metrics = _metrics(
        planning_availability="UNAVAILABLE",
        rebalancing_suggestions=[{"unavailable": SyntheticUnavailableValue.VALUE}],
    )
    result = _result(metrics)
    if not with_report:
        result = replace(result, report_view=None)
    summary = SummaryProjector.project(result)
    assert summary.sections["planning"]["availability"] == "UNAVAILABLE"
    assert summary.sections["planning"]["value"] is None


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
@pytest.mark.parametrize("duplicate", [True, False])
def test_positional_snapshot_preserves_wide_integers_and_nullable_columns(dtype, duplicate):
    largest = 2**64 - 1 if dtype.lower().startswith("u") else 2**63 - 1
    exact = pd.Series([2**53 + 1, largest, 0], dtype=dtype)
    if dtype[0].isupper():
        exact.iloc[2] = pd.NA
    frame = pd.concat([
        exact.rename("same" if duplicate else "exact"),
        pd.Series([1.25, 1.75, 0.0], name="same" if duplicate else "float"),
        pd.Series([100.0, 200.0, 0.0], name="cost_basis_eur"),
    ], axis=1)
    metrics = _metrics(holdings_df=frame)
    view = ReportView.build(metrics, InvestorConfig())
    pd.testing.assert_frame_equal(view.project()[0].holdings_df, frame)
    pd.testing.assert_frame_equal(
        view.project_metrics_fields(("holdings_df",))["holdings_df"], frame,
    )
    assert view.project_metrics_fields(("holdings_df",))["holdings_df"].iloc[0, 0] == (
        2**53 + 1
    )
    assert thaw_json(view.project_summary_facts().machine_metrics) == (
        view.project()[0].to_summary_dict()
    )


@pytest.mark.parametrize("source_mutation", [True, False])
def test_categorical_dtype_metadata_is_not_shared_with_source_or_projection(source_mutation):
    categories = pd.Index(["class-a", "class-b"], dtype=object)
    series = pd.Series(
        pd.Categorical(["class-a", "class-b"], categories=categories, ordered=True),
        name="synthetic-categories",
    )
    view = ReportView.build(
        _metrics(portfolio_history=series), InvestorConfig(),
    )
    mutable = series if source_mutation else view.project()[0].portfolio_history
    mutable.cat.categories.values[0] = "tampered-category"
    fresh = view.project_metrics_fields(("portfolio_history",))["portfolio_history"]
    assert fresh.cat.categories.tolist() == ["class-a", "class-b"]
    assert fresh.tolist() == ["class-a", "class-b"]
    assert fresh.cat.ordered


@pytest.mark.parametrize("position", ["column-values", "row-index", "column-index"])
def test_categorical_metadata_is_defensive_in_every_frame_position(position):
    frame = pd.DataFrame({
        "cost_basis_eur": [100.0, 200.0],
        "kind": pd.Categorical(["class-a", "class-b"], ordered=True),
    })
    if position == "row-index":
        frame.index = pd.CategoricalIndex(
            ["a", "b"], categories=pd.Index(["a", "b"], dtype=object),
            ordered=True, name="synthetic-index",
        )
    elif position == "column-index":
        frame.columns = pd.CategoricalIndex(
            frame.columns, categories=pd.Index(frame.columns, dtype=object),
            ordered=True, name="synthetic-columns",
        )
    view = ReportView.build(_metrics(holdings_df=frame), InvestorConfig())
    projected = view.project_metrics_fields(("holdings_df",))["holdings_df"]
    if position == "column-values":
        projected["kind"].cat.categories.values[0] = "tampered"
    elif position == "row-index":
        projected.index.categories.values[0] = "tampered"
    else:
        projected.columns.categories.values[0] = "tampered"
    fresh = view.project_metrics_fields(("holdings_df",))["holdings_df"]
    pd.testing.assert_frame_equal(fresh, frame)


def test_native_machine_snapshot_keeps_prior_typed_record_normalization():
    metrics = _metrics(
        schema_version=np.int64(3),
        history_unavailable_instruments=[np.int64(23), np.str_("SYNTH")],
        trustworthy_total_value_eur=np.float64(1111.987654),
    )
    result = _result(metrics)
    assert SummaryProjector.project(result).canonical_bytes() == (
        SummaryProjector.project(_legacy_projection(result)).canonical_bytes()
    )
