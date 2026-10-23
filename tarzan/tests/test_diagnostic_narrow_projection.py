"""Synthetic whole-report fidelity for exact, narrow diagnostic projections."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, dataclass, fields, make_dataclass, replace
from datetime import date, datetime, timezone
from itertools import product
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.reference import ReportReference, reference_for
from tarzan.contracts import report as report_contract
from tarzan.contracts.report import ReportView
from tarzan.data.market_quotes import MARKETS
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime import diagnostics as diag
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.publication import PublicationEvaluator
from tarzan.runtime.session import RunResult
from tarzan.runtime.summary import SummaryProjector
from tarzan.verification import portfolio


_SCALAR_NAMES = (
    "total_value", "trustworthy_total_value_eur", "known_valuation_subtotal_eur",
    "invested_value", "cash_value", "cash_target_eur",
    "pnl_eur", "pnl_pct", "invested_capital_eur",
    "estimated_cgt_eur", "pnl_eur_net_tax", "pnl_pct_net_tax",
    "xirr_pct", "xirr_net_tax_pct", "twr_pct", "twr_annualized_pct",
    "returns_coverage_pct", "weighted_yield", "avg_ter",
    "valuation_availability", "history_availability",
    "unrealized_pnl_eur", "unrealized_pnl_pct",
)
_STRUCTURED_NAMES = (
    "performance", "performance_full", "risk", "historical_risk",
    "allocation_by_class", "allocation_by_geo", "allocation_by_sector",
    "holding_performance", "goal_deltas", "target_weights",
    "rebalancing_plans", "degraded_computers", "benchmark_comparison",
    "portfolio_history", "actual_value_series", "pnl_series", "unrealized_series",
    "target_history", "holding_histories", "benchmark_histories",
    "allocation_timeline", "external_flows", "xirr_cashflows",
)
_PROPERTY_NAMES = ("unrealized_pnl_eur", "unrealized_pnl_pct")
_HOLDING_COLUMNS = (
    "isin", "ticker", "quantity", "current_price", "current_value",
    "cost_basis_eur", "weight_pct", "gain_eur", "gain_pct",
    "value_eur", "portfolio_weight_pct",
)
_QUALITY_FIELDS = (
    "report_reference", "holdings_df", "intraday_requested_tickers",
    "valuation_availability", "history_availability",
    "history_unavailable_instruments", "returns_coverage_pct",
    "benchmark_tickers", "benchmark_resolution_errors", "intraday_quotes",
    "intraday_evidence", "market_quotes", "degraded_computers",
)
_CAPTURED_AT = datetime(2026, 10, 21, 10, 15, tzinfo=timezone.utc)
_VALUES = {
    "operation": "verify", "exit_code": 0, "duration_seconds": 1.23456789,
    "input_sha256": {"order_list.csv": "b" * 64},
}


@pytest.fixture(autouse=True)
def _synthetic_environment(monkeypatch):
    for name in diag._CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA"):
        monkeypatch.setenv(name, "synthetic-" + name.lower())
    monkeypatch.setenv("TARZAN_DISABLE_AI", "1")


def _reference():
    return ReportReference(
        names={"SYNTH.MI": "Synthetic"},
        benchmarks={"Synthetic index": {"ticker": "SYNTH-INDEX"}},
        benchmark_alpha_beta="Synthetic index", benchmark_geo="Synthetic geography",
    )


def _metrics(**changes):
    dates = pd.date_range("2026-10-19", periods=3, freq="B")
    metrics = PortfolioMetrics(
        total_value=1111.987654321,
        invested_value=1000.123456789,
        cash_value=111.864197532,
        holdings_df=pd.DataFrame({
            "isin": ["XX0000000001"], "ticker": ["SYNTH.MI"], "quantity": [2],
            "current_price": [555.9938271605], "current_value": [1111.987654321],
            "cost_basis_eur": [1000.123456789], "weight_pct": [100.0],
            "gain_eur": [111.864197532], "gain_pct": [11.185038891],
            "unused_annotation": [{"nested": [1]}],
        }),
        performance={"1d": 0.123456789, "1m": -2.123456789},
        performance_full={"1d": 0.123456789, "1d_coverage_pct": 99.123456789},
        risk={"sharpe": float("nan"), "alpha": 0.123456789},
        historical_risk={"available": False, "nested": [None, 0]},
        holding_performance=pd.DataFrame({"ticker": ["SYNTH.MI"], "1d": [0.123456789]}),
        target_weights={"SYNTH.MI": 100.0},
        rebalancing_suggestions=[{"ticker": "SYNTH.MI", "quantity": 2}],
        rebalancing_verifications=[{"kind": "funding", "valid": True}],
        rebalancing_plans=[{"label": "Synthetic", "suggestions": [], "nested": [1]}],
        portfolio_history=pd.Series([1000.0, 1001.0, 1002.0], index=dates),
        actual_value_series=pd.Series([1100.0, 1101.0, 1111.987654321], index=dates),
        pnl_series=pd.Series([100.0, 101.0, 111.864197532], index=dates),
        unrealized_series=pd.Series([100.0, 101.0, 111.864197532], index=dates),
        holding_histories={"SYNTH.MI": {"name": "Synthetic", "history": pd.Series(
            [5.0, 6.0, 7.0], index=dates,
        )}},
        benchmark_histories={"Synthetic index": pd.Series([9.0, 10.0, 11.0], index=dates)},
        benchmark_tickers={"Synthetic index": "SYNTH-INDEX"},
        benchmark_resolution_errors=("Synthetic missing benchmark",),
        intraday_requested_tickers=("SYNTH.MI", "WATCH.MI"),
        intraday_quotes={"SYNTH.MI": {"intraday_series": pd.Series(
            [5.0, 6.0], index=pd.date_range("2026-10-21T09:00", periods=2, freq="min"),
        )}},
        intraday_evidence={
            "SYNTH.MI": {"state": "LAGGED_CURRENT_SESSION", "native_bar_count": 1,
                         "nested": {"items": [3]}},
            "WATCH.MI": {"state": "MISSING", "native_bar_count": 0},
        },
        market_quotes=({"ticker": "SYNTH-MACRO", "price": 3.0},),
        report_reference=_reference(),
        external_flows={date(2026, 10, 19): 1000.123456789},
        xirr_cashflows=[(date(2026, 10, 19), -1000.123456789),
                       (date(2026, 10, 21), 1111.987654321)],
        xirr_pct=0.123456789, twr_pct=-0.123456789,
        twr_annualized_pct=float("inf"), returns_coverage_pct=99.123456789,
    )
    for name, value in changes.items():
        setattr(metrics, name, value)
    return metrics


def _result(metrics=None, *, prepared=True):
    metrics = _metrics() if metrics is None else metrics
    config = InvestorConfig()
    ledger = RunLedger("synthetic-diagnostic-attempt")
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "source": "synthetic-provider", "operation": "native_http", "outcome": "HTTP_200",
    })
    ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, {
        "source": "synthetic-provider", "operation": "history", "outcome": "SUCCEEDED",
    })
    return RunResult(
        metrics, config, "synthetic-diagnostic-attempt", "synthetic-diagnostic-analysis",
        ledger, session=SimpleNamespace(context=SimpleNamespace(captured_at=_CAPTURED_AT)),
        report_view=ReportView.build(metrics, config) if prepared else None,
    )


def _legacy_quality(metrics):
    """The pre-tranche quality projection, independent of new field metadata."""
    if metrics is None:
        return {}
    reference = reference_for(metrics)
    held = set(metrics.holdings_df.get("ticker", ())) & set(metrics.intraday_requested_tickers)
    return {
        "valuation_availability": metrics.valuation_availability,
        "history_availability": metrics.history_availability,
        "history_unavailable_instruments": list(metrics.history_unavailable_instruments),
        "returns_coverage_pct": metrics.returns_coverage_pct,
        "required_benchmarks": {
            "alpha_beta": reference.benchmark_alpha_beta,
            "geography": reference.benchmark_geo,
        },
        "benchmark_tickers": dict(metrics.benchmark_tickers),
        "benchmarks_requested": len(reference.benchmarks),
        "benchmarks_resolved": len(metrics.benchmark_tickers),
        "benchmark_resolution_errors": list(metrics.benchmark_resolution_errors),
        "intraday_requested": len(metrics.intraday_requested_tickers),
        "intraday_with_bars": len(metrics.intraday_quotes),
        "intraday_missing": [
            ticker for ticker in metrics.intraday_requested_tickers
            if ticker not in metrics.intraday_quotes
        ],
        "intraday_lagged": [
            ticker for ticker, fact in metrics.intraday_evidence.items()
            if fact.get("state") == "LAGGED_CURRENT_SESSION"
        ],
        "intraday_single_observation": [
            ticker for ticker, fact in metrics.intraday_evidence.items()
            if fact.get("native_bar_count") == 1
        ],
        "intraday_evidence": dict(metrics.intraday_evidence),
        "intraday_current_holdings_missing": sorted(held - set(metrics.intraday_quotes)),
        "intraday_current_holdings_lagged": sorted(
            ticker for ticker in held
            if metrics.intraday_evidence.get(ticker, {}).get("state") == "LAGGED_CURRENT_SESSION"
        ),
        "macro_requested": len(MARKETS),
        "macro_available": len(metrics.market_quotes) if metrics.market_quotes is not None else None,
        "degraded_computers": list(metrics.degraded_computers),
    }


def _legacy_kpis(metrics):
    result = {
        name: portfolio._financial_value(getattr(metrics, name, None))
        for name in _SCALAR_NAMES
    }
    for name in _STRUCTURED_NAMES:
        result[name] = portfolio._financial_value(getattr(metrics, name, None))
    holdings = metrics.holdings_df
    result["holdings"] = portfolio._financial_value(
        holdings[[name for name in _HOLDING_COLUMNS if name in holdings]],
    )
    return result


def _legacy_report(result, **values):
    """Original full-projection implementation, retaining its error boundaries."""
    entries = [entry.to_dict() for entry in result.ledger.entries] if result is not None else []
    native, adapters = defaultdict(Counter), defaultdict(Counter)
    if result is not None:
        for entry in result.ledger.entries:
            if entry.entry_type is LedgerEntryType.PROVIDER_ATTEMPT:
                counts = native if entry.payload.get("operation") == "native_http" else adapters
                counts[str(entry.payload.get("source", "UNKNOWN"))][
                    str(entry.payload.get("outcome", "UNKNOWN"))
                ] += 1
    document = {
        "schema_version": diag.SCHEMA_VERSION,
        "operation": values["operation"], "exit_code": values["exit_code"],
        "duration_seconds": values["duration_seconds"],
        "captured_at": (
            result.session.context.captured_at.isoformat()
            if result is not None and result.session is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "run": {
            "attempt_id": result.attempt_id if result is not None else None,
            "analysis_id": result.analysis_id if result is not None else None,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "revision": os.environ.get("GITHUB_SHA"),
        },
        "input_sha256": dict(values.get("input_sha256") or {}),
        "entrypoint_error_type": values.get("entrypoint_error_type"),
        "provider_counts": {"native": dict(native), "adapters": dict(adapters)},
        "quality": {}, "summary": None, "kpis": {}, "ledger": entries,
        "projection_error_type": None,
    }
    if result is not None:
        try:
            metrics = result.report_view.project()[0] if result.report_view is not None else result.metrics
            document["quality"] = _legacy_quality(metrics)
            document["summary"] = SummaryProjector.project(
                result, PublicationEvaluator.evaluate(result.ledger.failure_records()),
            ).to_dict()
            if metrics is not None:
                document["kpis"] = _legacy_kpis(metrics)
        except Exception as error:
            document["projection_error_type"] = type(error).__name__
    return diag._redact_credentials(document)


def _document(result, **changes):
    return diag.build_report(result, **(_VALUES | changes))


def _assert_legacy(result, **changes):
    values = _VALUES | changes
    before = _legacy_report(result, **values)
    after = diag.build_report(result, **values)
    if result is None or result.session is None:
        before["captured_at"] = after["captured_at"]
    assert after == before
    return after


def test_happy_path_never_reconstructs_the_full_cube_or_config(monkeypatch):
    owner = _result()
    expected = _legacy_report(owner, **_VALUES)

    def forbidden(*args, **kwargs):
        pytest.fail("prepared native diagnostics must not project the whole report")

    monkeypatch.setattr(ReportView, "project", forbidden)
    assert _document(owner) == expected


def test_field_metadata_pins_original_kpi_order_without_new_formulas():
    assert portfolio.PORTFOLIO_KPI_SCALAR_NAMES == _SCALAR_NAMES
    assert portfolio.PORTFOLIO_KPI_STRUCTURED_NAMES == _STRUCTURED_NAMES
    assert portfolio.PORTFOLIO_KPI_PROPERTY_NAMES == _PROPERTY_NAMES
    assert portfolio.PORTFOLIO_KPI_HOLDING_COLUMNS == _HOLDING_COLUMNS
    assert portfolio.PORTFOLIO_KPI_FIELD_NAMES == (
        tuple(name for name in _SCALAR_NAMES if name not in _PROPERTY_NAMES)
        + _STRUCTURED_NAMES + ("holdings_df",)
    )
    assert list(portfolio.portfolio_kpis(_metrics())) == (
        list(_SCALAR_NAMES) + list(_STRUCTURED_NAMES) + ["holdings"]
    )


def test_only_required_frozen_fields_and_original_properties_are_projected(monkeypatch):
    owner = _result()
    field_calls, property_calls = [], []
    native_fields = ReportView.project_metrics_fields
    native_properties = ReportView.project_metric_properties

    def selected(view, names):
        names = tuple(names)
        field_calls.append(names)
        return native_fields(view, names)

    def properties(view, names):
        names = tuple(names)
        property_calls.append(names)
        return native_properties(view, names)

    monkeypatch.setattr(ReportView, "project_metrics_fields", selected)
    monkeypatch.setattr(ReportView, "project_metric_properties", properties)
    assert _document(owner)["projection_error_type"] is None
    expected_fields = tuple(dict.fromkeys(
        _QUALITY_FIELDS
        + tuple(name for name in _SCALAR_NAMES if name not in _PROPERTY_NAMES)
        + _STRUCTURED_NAMES + ("holdings_df",),
    ))
    assert field_calls == [expected_fields, ("holdings_df", "total_value")]
    assert property_calls == [_PROPERTY_NAMES]


@pytest.mark.parametrize(
    "valuation,history,planning,macro",
    list(product(
        ("AVAILABLE", "DEGRADED", "UNAVAILABLE"),
        ("AVAILABLE", "DEGRADED", "UNAVAILABLE"),
        ("AVAILABLE", "UNAVAILABLE"),
        (None, ()),
    )),
)
def test_every_report_field_matches_legacy_for_availability_states(
    valuation, history, planning, macro,
):
    _assert_legacy(_result(_metrics(
        valuation_availability=valuation, history_availability=history,
        planning_availability=planning, market_quotes=macro,
        trustworthy_total_value_eur=0.0 if valuation == "AVAILABLE" else None,
        known_valuation_subtotal_eur=123.4567890123,
        history_unavailable_instruments=("XX0000000002",),
        degraded_computers=[] if history == "AVAILABLE" else ["_synthetic_history"],
    )))


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("total,cost", [
    (1111.987654321, 1000.123456789),
    (0.0, 0.0), (1111.987654321, 0.0), (1111.987654321, -1.0),
    (np.int64(2**53 + 37), 1000.123456789),
])
def test_raw_properties_are_not_rounded_or_replaced_by_summary_values(prepared, total, cost):
    metrics = _metrics(
        total_value=total,
        holdings_df=pd.DataFrame({"ticker": ["SYNTH.MI"], "cost_basis_eur": [cost]}),
    )
    owner = _result(metrics, prepared=prepared)
    document = _assert_legacy(owner)
    assert document["kpis"]["unrealized_pnl_eur"] == metrics.unrealized_pnl_eur
    assert document["kpis"]["unrealized_pnl_pct"] == metrics.unrealized_pnl_pct


@pytest.mark.parametrize("dtype,wide", [
    ("int64", 2**53 + 37), ("int64", 2**63 - 1),
    ("uint64", 2**64 - 1), ("Int64", 2**53 + 37),
    ("UInt64", 2**64 - 1),
])
def test_wide_integer_and_nullable_holdings_survive_whole_report_projection(dtype, wide):
    quantity = pd.Series([wide, pd.NA] if dtype[0].isupper() else [wide, 1], dtype=dtype)
    frame = pd.DataFrame({
        "ticker": pd.Categorical(["SYNTH.MI", "SECOND.MI"], ordered=True),
        "quantity": quantity, "current_price": [555.0, 556.0],
        "cost_basis_eur": pd.Series([1000.123456789, pd.NA], dtype="Float64"),
    })
    owner = _result(_metrics(holdings_df=frame))
    document = _assert_legacy(owner)
    assert document["kpis"]["holdings"][0]["quantity"] == wide
    if dtype[0].isupper():
        assert document["kpis"]["holdings"][1]["quantity"] is None


def test_category_series_nested_attrs_and_native_missing_values_match_legacy():
    metrics = _metrics()
    metrics.holdings_df["ticker"] = pd.Categorical(["SYNTH.MI"], ordered=True)
    metrics.portfolio_history = pd.Series(
        pd.Categorical(["low", None, "high"], categories=["low", "high"], ordered=True),
        index=metrics.portfolio_history.index,
    )
    metrics.actual_value_series = pd.Series(
        [1111.123456789, pd.NA, 1111.987654321],
        index=metrics.actual_value_series.index, dtype="Float64",
    )
    metrics.actual_value_series.attrs = {"native": {"observations": [1, 2]}}
    metrics.performance["synthetic_wide_int"] = np.int64(2**53 + 37)
    document = _assert_legacy(_result(metrics))
    assert document["kpis"]["performance"]["synthetic_wide_int"] == 2**53 + 37


def test_frozen_display_reference_is_used_without_new_configuration_lookup(monkeypatch):
    owner = _result()

    def forbidden():
        pytest.fail("prepared diagnostic reference must not be prepared again")

    monkeypatch.setattr(ReportReference, "prepare", forbidden)
    document = _assert_legacy(owner)
    assert document["quality"]["required_benchmarks"] == {
        "alpha_beta": "Synthetic index", "geography": "Synthetic geography",
    }
    assert document["quality"]["benchmarks_requested"] == 1


def test_source_mutation_and_previous_output_mutation_cannot_change_frozen_evidence():
    owner = _result()
    before = _document(owner)
    owner.metrics.total_value = -9999.0
    owner.metrics.holdings_df.loc[0, "cost_basis_eur"] = 0.0
    owner.metrics.portfolio_history.iloc[0] = -9999.0
    owner.metrics.intraday_evidence["SYNTH.MI"]["nested"]["items"].append(9)
    owner.config.cash_target_eur = 9999.0
    assert _document(owner) == before
    returned = _document(owner)
    returned["quality"]["intraday_evidence"]["SYNTH.MI"]["nested"]["items"].append(8)
    returned["kpis"]["rebalancing_plans"][0]["nested"].append(8)
    returned["kpis"]["portfolio_history"]["values"].append(8)
    assert _document(owner) == before


@pytest.mark.parametrize("stage,severity,affected", [
    ("gemini", "WARNING", ["market_context_summary"]),
    ("newsletter_semantics", "CRITICAL", ["newsletter"]),
    ("valuation", "CRITICAL", ["portfolio", "planning"]),
    ("delivery_reconciliation", "CRITICAL", ["delivery", "publication"]),
])
def test_late_failure_publication_and_smtp_evidence_remain_fresh(stage, severity, affected):
    owner = _result()
    initial = _document(owner)
    owner.ledger.open_failure(
        stage=stage, stable_code="SYNTHETIC_LATE_FAILURE", severity=severity,
        error="synthetic late evidence", affected_outputs=affected,
        analytical_impact="synthetic impact",
        publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND" if severity == "CRITICAL" else "DEGRADE",
    )
    owner.ledger.append(LedgerEntryType.PUBLICATION, {"decision": "synthetic decision"})
    owner.ledger.append(LedgerEntryType.DELIVERY, {
        "operation": "smtp", "outcome": "synthetic accepted",
    })
    document = _assert_legacy(owner, operation="send", exit_code=1)
    assert document["ledger"] != initial["ledger"]
    assert document["summary"] != initial["summary"]
    assert document["kpis"] == initial["kpis"]
    assert document["ledger"][-1]["payload"]["outcome"] == "synthetic accepted"


class _LegacyView:
    """A pre-narrow adapter still supplying its original full view."""

    def __init__(self, native):
        self.native = native
        self.calls = 0

    def project(self):
        self.calls += 1
        return self.native.project()

    def project_summary_facts(self):
        return self.native.project_summary_facts()


def test_genuinely_legacy_view_keeps_full_projection_compatibility():
    owner = _result()
    legacy_view = _LegacyView(owner.report_view)
    owner = replace(owner, report_view=legacy_view)
    expected = _legacy_report(owner, **_VALUES)
    legacy_view.calls = 0
    assert _document(owner) == expected
    assert legacy_view.calls == 1


def test_original_four_argument_view_preserves_its_machine_and_report_contract():
    owner = _result()
    native = owner.report_view
    owner = replace(owner, report_view=ReportView(
        native._metrics, native._config, native.benchmark_alpha_beta, native.benchmark_geo,
    ))
    _assert_legacy(owner)


def test_failed_genuine_legacy_full_projection_keeps_its_primary_error(monkeypatch):
    owner = _result()
    legacy_view = _LegacyView(owner.report_view)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic private legacy projection failure")

    monkeypatch.setattr(legacy_view, "project", broken)
    document = _assert_legacy(replace(owner, report_view=legacy_view))
    assert document["projection_error_type"] == "RuntimeError"
    assert document["quality"] == {} and document["summary"] is None
    assert document["ledger"]


@dataclass
class _GenericMetrics:
    total_value: object
    report_reference: object
    holdings_df: object
    intraday_requested_tickers: object
    valuation_availability: object
    history_availability: object
    history_unavailable_instruments: object
    returns_coverage_pct: object
    benchmark_tickers: object
    benchmark_resolution_errors: object
    intraday_quotes: object
    intraday_evidence: object
    market_quotes: object
    degraded_computers: object

    def to_summary_dict(self):
        return {"generic_total_eur": self.total_value}


def test_genuine_generic_cube_keeps_original_missing_field_and_property_semantics():
    native = _metrics()
    generic = _GenericMetrics(
        total_value=native.total_value,
        **{name: getattr(native, name) for name in _QUALITY_FIELDS},
    )
    document = _assert_legacy(_result(generic))
    assert document["projection_error_type"] is None
    assert document["kpis"]["unrealized_pnl_eur"] is None
    assert document["kpis"]["invested_value"] is None
    assert document["quality"]["intraday_requested"] == 2


@dataclass
class _IncompleteGenericMetrics:
    total_value: float

    def to_summary_dict(self):
        return {"generic_total_eur": self.total_value}


def test_generic_cube_failure_keeps_original_exception_type_and_primary_ledger():
    document = _assert_legacy(_result(_IncompleteGenericMetrics(1000.0)))
    assert document["projection_error_type"] == "AttributeError"
    assert document["ledger"]
    assert document["quality"] == {}


def test_complete_generic_record_with_required_constructor_keeps_full_compatibility():
    native = _metrics()
    metric_fields = tuple(item for item in fields(PortfolioMetrics) if item.init)
    generic_type = make_dataclass(
        "SyntheticCompleteGenericMetrics",
        [(item.name, object) for item in metric_fields],
        namespace={
            "to_summary_dict": PortfolioMetrics.to_summary_dict,
            "unrealized_pnl_eur": PortfolioMetrics.unrealized_pnl_eur,
            "unrealized_pnl_pct": PortfolioMetrics.unrealized_pnl_pct,
        },
    )
    generic = generic_type(**{
        item.name: getattr(native, item.name) for item in metric_fields
    })
    document = _assert_legacy(_result(generic))
    assert document["projection_error_type"] is None
    assert document["kpis"]["unrealized_pnl_eur"] == native.unrealized_pnl_eur


@dataclass(kw_only=True)
class _CustomMetrics(PortfolioMetrics):
    synthetic_extra_authority: object


def test_custom_native_subclass_with_required_field_keeps_its_original_constructor():
    native = _metrics()
    custom = _CustomMetrics(
        synthetic_extra_authority={"synthetic": 1},
        **{item.name: getattr(native, item.name) for item in fields(native) if item.init},
    )
    document = _assert_legacy(_result(custom))
    assert document["projection_error_type"] is None
    assert document["kpis"]["unrealized_pnl_eur"] == native.unrealized_pnl_eur


def test_native_authority_metadata_is_exact_readonly_and_evaluates_no_values(monkeypatch):
    native = _result().report_view
    generic = _result(_IncompleteGenericMetrics(1000.0)).report_view
    custom = _result(_CustomMetrics(
        synthetic_extra_authority=1,
        holdings_df=pd.DataFrame({"cost_basis_eur": [1.0]}),
    )).report_view
    absent = ReportView.build(None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("authority metadata must not project, instantiate or compute values")

    monkeypatch.setattr(report_contract, "_project", forbidden)
    monkeypatch.setattr(PortfolioMetrics, "__init__", forbidden)
    monkeypatch.setattr(PortfolioMetrics, "unrealized_pnl_eur", property(forbidden))
    monkeypatch.setattr(ReportReference, "prepare", forbidden)
    assert native.is_native_portfolio_metrics is True
    assert generic.is_native_portfolio_metrics is False
    assert custom.is_native_portfolio_metrics is False
    assert absent.is_native_portfolio_metrics is False
    with pytest.raises(FrozenInstanceError):
        native.is_native_portfolio_metrics = False


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_native_field_projection_errors_never_fallback_to_full_view(monkeypatch, error_type):
    owner = _result()

    def broken(*args, **kwargs):
        raise error_type("synthetic private projection failure")

    def forbidden(*args, **kwargs):
        pytest.fail("a broken native narrow API is not a legacy fallback")

    monkeypatch.setattr(ReportView, "project_metrics_fields", broken)
    monkeypatch.setattr(ReportView, "project", forbidden)
    document = _document(owner)
    assert document["projection_error_type"] == error_type.__name__
    assert document["quality"] == {} and document["summary"] is None
    assert document["ledger"] == [entry.to_dict() for entry in owner.ledger.entries]
    assert "synthetic private projection failure" not in repr(document)


def test_unknown_native_field_is_an_explicit_projection_error_not_fake_defaults(monkeypatch):
    owner = _result()
    monkeypatch.setattr(
        portfolio, "PORTFOLIO_KPI_FIELD_NAMES",
        portfolio.PORTFOLIO_KPI_FIELD_NAMES + ("synthetic_deleted_field",),
    )
    monkeypatch.setattr(ReportView, "project",
                        lambda *args: pytest.fail("do not conceal native field drift"))
    document = _document(owner)
    assert document["projection_error_type"] == "ValueError"
    assert document["kpis"] == {}
    assert document["ledger"]


@pytest.mark.parametrize("which", ["fields", "properties"])
@pytest.mark.parametrize("shape", ["missing", "extra", "not_dict"])
def test_malformed_native_projection_mapping_is_explicit_not_fake_values(
    monkeypatch, which, shape,
):
    owner = _result()
    method = "project_metrics_fields" if which == "fields" else "project_metric_properties"
    native = getattr(ReportView, method)

    def malformed(view, names):
        facts = native(view, names)
        if shape == "missing":
            facts.pop(next(iter(facts)))
        elif shape == "extra":
            facts["synthetic_unknown"] = 0.0
        else:
            facts = tuple(facts.items())
        return facts

    monkeypatch.setattr(ReportView, method, malformed)
    document = _document(owner)
    assert document["projection_error_type"] == "ValueError"
    assert document["ledger"]
    assert document["kpis"] == {}
    if which == "properties":
        assert document["quality"] and document["summary"] is not None
    else:
        assert document["quality"] == {} and document["summary"] is None


@pytest.mark.parametrize("method", ["project_metrics_fields", "project_metric_properties"])
def test_present_noncallable_native_api_is_not_mistaken_for_absent_legacy_api(monkeypatch, method):
    owner = _result()
    monkeypatch.setattr(ReportView, method, None)
    document = _document(owner)
    assert document["projection_error_type"] == "TypeError"
    assert document["ledger"]


def test_property_failure_keeps_already_projected_quality_and_summary():
    owner = _result(_metrics(
        holdings_df=pd.DataFrame({"ticker": ["SYNTH.MI"], "quantity": [2]}),
    ))
    document = _assert_legacy(owner)
    assert document["projection_error_type"] == "KeyError"
    assert document["quality"]["intraday_requested"] == 2
    assert document["summary"]["analysis_id"] == owner.analysis_id
    assert document["kpis"] == {}


@pytest.mark.parametrize("stage", ["quality", "summary"])
def test_properties_are_not_eagerly_evaluated_before_prior_report_failures(monkeypatch, stage):
    owner = _result()
    calls = []
    native_properties = ReportView.project_metric_properties

    def counted(*args, **kwargs):
        calls.append(True)
        return native_properties(*args, **kwargs)

    def broken(*args, **kwargs):
        raise ValueError("synthetic earlier projection error")

    monkeypatch.setattr(ReportView, "project_metric_properties", counted)
    if stage == "quality":
        monkeypatch.setattr(diag, "_quality", broken)
    else:
        monkeypatch.setattr(SummaryProjector, "project", broken)
    document = _document(owner)
    assert document["projection_error_type"] == "ValueError"
    assert calls == []


def test_deferred_machine_serializer_failure_retains_original_partial_report(monkeypatch):
    def malformed(*args, **kwargs):
        raise ValueError("synthetic malformed machine contract")

    monkeypatch.setattr(PortfolioMetrics, "to_summary_dict", malformed)
    document = _assert_legacy(_result())
    assert document["projection_error_type"] == "ValueError"
    assert document["quality"]["intraday_requested"] == 2
    assert document["summary"] is None
    assert document["kpis"] == {}


@pytest.mark.parametrize("prepared", [False, True])
def test_empty_and_unavailable_metrics_do_not_gain_fictitious_values(prepared):
    metrics = _metrics(
        total_value=0.0, holdings_df=pd.DataFrame(), performance=None, performance_full=None,
        risk=None, rebalancing_suggestions=None, rebalancing_verifications=None,
        valuation_availability="UNAVAILABLE", history_availability="UNAVAILABLE",
        planning_availability="UNAVAILABLE", trustworthy_total_value_eur=None,
        known_valuation_subtotal_eur=None,
    )
    document = _assert_legacy(_result(metrics, prepared=prepared))
    assert document["kpis"]["unrealized_pnl_pct"] is None
    assert document["summary"]["sections"]["portfolio"]["value"] is None


def test_none_result_preserves_schema_and_does_not_attempt_any_projection(monkeypatch):
    monkeypatch.setattr(ReportView, "project_metrics_fields",
                        lambda *args: pytest.fail("there is no result to project"))
    document = _assert_legacy(None, exit_code=1, entrypoint_error_type="ValueError")
    assert document["summary"] is None and document["ledger"] == []
    assert document["projection_error_type"] is None


def test_prepared_absent_metrics_preserve_original_none_result_values():
    owner = _result()
    owner = replace(owner, metrics=None, report_view=ReportView.build(None, owner.config))
    document = _assert_legacy(owner)
    assert document["quality"] == {} and document["kpis"] == {}
    assert document["summary"]["sections"]["portfolio"]["value"] is None


def test_frozen_native_authority_survives_absent_source_reference():
    owner = _result()
    expected = _legacy_report(owner, **_VALUES)
    assert _document(replace(owner, metrics=None)) == expected


def test_native_field_drift_still_fails_when_source_metrics_reference_is_absent(monkeypatch):
    owner = replace(_result(), metrics=None)
    monkeypatch.setattr(
        portfolio, "PORTFOLIO_KPI_FIELD_NAMES",
        portfolio.PORTFOLIO_KPI_FIELD_NAMES + ("synthetic_deleted_field",),
    )
    document = _document(owner)
    assert document["projection_error_type"] == "ValueError"
    assert document["kpis"] == {}
    assert document["ledger"]


@pytest.mark.parametrize("metadata", [None, 1, "native"])
def test_invalid_native_metadata_is_not_an_implicit_full_fallback(monkeypatch, metadata):
    owner = _result()
    monkeypatch.setattr(ReportView, "is_native_portfolio_metrics", property(lambda view: metadata))
    monkeypatch.setattr(ReportView, "project",
                        lambda *args: pytest.fail("invalid metadata must not hide a defect"))
    document = _document(owner)
    assert document["projection_error_type"] == "ValueError"
    assert document["ledger"] and document["kpis"] == {}


def test_original_direct_legacy_metrics_path_is_not_replaced_by_rounded_snapshot():
    owner = _result(prepared=False)
    initial = _assert_legacy(owner)
    owner.metrics.total_value = 4321.123456789
    fresh = _assert_legacy(owner)
    assert fresh["kpis"]["total_value"] == 4321.123456789
    assert fresh["kpis"] != initial["kpis"]


def test_native_secret_redaction_and_primary_error_payload_are_preserved(monkeypatch):
    owner = _result()
    secret = "synthetic-diagnostic-credential"
    monkeypatch.setenv("SMTP_PASS", secret)
    owner.ledger.open_failure(
        stage="newsletter_semantics", stable_code="SYNTHETIC_SEMANTIC_FAILURE",
        severity="CRITICAL", error={"violations": ["synthetic finance " + secret]},
        affected_outputs=["newsletter"], analytical_impact="synthetic private failure",
        publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
    )
    document = _assert_legacy(owner)
    assert secret not in repr(document)
    assert "[REDACTED]" in repr(document)
    assert document["summary"]["publication_state"] == "BLOCK_ALL_DELIVERY"
    assert document["ledger"][-1]["payload"]["stable_code"] == "SYNTHETIC_SEMANTIC_FAILURE"
