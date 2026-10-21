"""Prepared reference facts are immutable and never re-read during rendering."""

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from tarzan import config
from tarzan.contracts.material import material_metrics
from tarzan.contracts.reference import ReportReference
from tarzan.contracts.report import ReportView
from tarzan.export._format import display_instrument_name
from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.session import canonical_analysis_id
from tarzan.tests.test_newsletter_golden_html import rendered  # noqa: F401 — fixture


def _forbid_global_reference_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a prepared report read a later global reference")

    for name in (
        "name_lookup", "name_for", "instrument_taxonomy", "benchmarks",
        "benchmark_identities", "watchlist_names", "metric_ratings",
        "benchmark_beta_name", "benchmark_geo_allocation", "_default_benchmark_name",
    ):
        monkeypatch.setattr(config, name, forbidden)
    monkeypatch.setattr("tarzan.export._format._taxonomy_names", forbidden)


@pytest.mark.parametrize("field", ["names", "taxonomy", "benchmarks", "metric_ratings"])
def test_reference_maps_defensively_freeze_every_nested_value(field):
    source = {"key": {"nested": [1]}}
    reference = ReportReference(**{field: source})
    source["key"]["nested"].append(2)
    assert getattr(reference, field)["key"]["nested"] == (1,)
    with pytest.raises(TypeError):
        getattr(reference, field)["key"]["nested"] = (3,)


def test_reference_sequences_and_role_labels_cannot_mutate():
    identities = [["Reference", "REF.X", "IE0000000001"]]
    watchlist = ["WATCH", "watch", "other"]
    reference = ReportReference(benchmark_identities=identities, watchlist_names=watchlist)
    identities[0][0] = "Changed"
    watchlist.clear()
    assert reference.benchmark_identities == (("Reference", "REF.X", "IE0000000001"),)
    assert reference.watchlist_names == ("other", "watch")
    with pytest.raises(FrozenInstanceError):
        reference.benchmark_geo = "Changed"


def test_preparation_reads_each_authority_once_and_retains_exact_identity(monkeypatch):
    answers = {
        "name_lookup": {"IE0000000001": "Exact", "REF": "Fallback"},
        "instrument_taxonomy": {"REF": ("Equities", "Equity Broad")},
        "benchmarks": {"Reference": "REF.X"},
        "benchmark_identities": (("Reference", "REF.X", "IE0000000001"),),
        "watchlist_names": frozenset({"reference"}),
        "metric_ratings": {"beta": {"thresholds": [1, 2]}},
        "benchmark_beta_name": "Reference",
        "benchmark_geo_allocation": "Reference",
    }
    calls = []
    for name, value in answers.items():
        def accessor(name=name, value=value):
            calls.append(name)
            return value
        monkeypatch.setattr(config, name, accessor)
    reference = ReportReference.prepare()
    assert sorted(calls) == sorted(answers)
    assert reference.name_for("ie0000000001", "REF.X") == "Exact"
    assert reference.name_for(None, "ref.x") == "Fallback"
    assert reference.name_for(None, "UNLISTED.X") is None
    assert reference.taxonomy["REF"] == ("Equities", "Equity Broad")


def test_empty_explicit_names_do_not_fall_back_to_the_global_cache(monkeypatch):
    _forbid_global_reference_reads(monkeypatch)
    assert display_instrument_name(None, "REF.X", "Broker name", names={}) == "Broker name"


def test_legacy_cube_is_prepared_once_at_the_context_boundary(monkeypatch):
    calls = []
    reference = ReportReference(benchmark_alpha_beta="Reference", benchmark_geo="Reference")
    monkeypatch.setattr(ReportReference, "prepare", classmethod(lambda cls: calls.append(1) or reference))
    context = build_context(PortfolioMetrics(), InvestorConfig(), ai_summary="")
    assert calls == [1]
    assert context["risk_legend"]


def test_completed_render_keeps_its_reference_when_globals_change(rendered, monkeypatch):
    original_html, metrics, investor = rendered
    assert metrics.report_reference is not None
    _forbid_global_reference_reads(monkeypatch)
    assert render_newsletter(metrics, investor, issue_number=31, ai_summary="") == original_html
    projected, _ = ReportView.build(metrics, investor).project()
    assert projected.report_reference == metrics.report_reference
    with pytest.raises(TypeError):
        projected.report_reference.names["NEW"] = "Changed"


def test_semantic_gate_checks_the_prepared_benchmark_catalog(monkeypatch):
    reference = ReportReference(
        benchmarks={"Reference": "REF.X"}, benchmark_alpha_beta="Reference",
        benchmark_geo="Reference",
    )
    series = pd.Series(
        [100.0, 101.0], index=pd.date_range("2025-01-01", periods=2), name="REF.X",
    )
    series.attrs.update(resolved_ticker="REF.X", requested_ticker="REF.X")
    metrics = PortfolioMetrics(
        report_reference=reference, benchmark_tickers={"Reference": "REF.X"},
        benchmark_histories={"Reference": series},
        holding_performance=pd.DataFrame([{
            "name": "Reference", "ticker": "REF.X", "type": "Benchmark",
        }]),
        benchmark_comparison=pd.DataFrame([{"benchmark": "Reference", "ticker": "REF.X"}]),
        historical_risk={"instruments": [{"label": "Reference", "ticker": "REF.X"}]},
        intraday_requested_tickers=("REF.X",),
    )
    audit = {"performance_intraday": {
        "origin": "metrics_preprocessing", "requested_tickers": ("REF.X",),
        "returned_tickers": (), "source_tickers": {},
    }}
    _forbid_global_reference_reads(monkeypatch)
    assert validate_newsletter_semantics(metrics, audit, "") == ()
    metrics.report_reference = replace(reference, benchmarks={"Other": "OTHER.X"})
    assert any("unexpected entries" in error for error in validate_newsletter_semantics(metrics, audit, ""))


def test_analysis_identity_includes_reference_facts_with_canonical_watchlist_order():
    first = PortfolioMetrics(report_reference=ReportReference(
        watchlist_names=("second", "first"), names={"REF": "First name"},
    ))
    reordered = PortfolioMetrics(report_reference=ReportReference(
        watchlist_names=("first", "second"), names={"REF": "First name"},
    ))
    changed = PortfolioMetrics(report_reference=replace(first.report_reference, names={"REF": "New name"}))
    digest = lambda metrics: canonical_analysis_id(material_metrics(metrics))
    assert digest(first) == digest(reordered)
    assert digest(first) != digest(changed)
