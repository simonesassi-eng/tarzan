"""Completed-run ownership, immutable rendering and truthful finalization."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import date, datetime, timezone
import hashlib
import json
import logging
from threading import Event
import time

import pandas as pd
import pytest

from tarzan import main, orchestrator, runtime
from tarzan.contracts.report import ReportView
from tarzan.export import ai_summary
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime import data_quality
from tarzan.runtime.artifacts import LocalArtifactWriter
from tarzan.runtime.ledger import LedgerEntryType, RunLedger
from tarzan.runtime.logging import LOG_CAPTURE
from tarzan.runtime.session import (
    RunContext, RunMode, RunResult, RunSession, SerialExecutionGate,
    activate_session, completed_run_scope, current_session, last_run_result,
    record_last_run_result,
)
from tarzan.runtime.summary import SummaryProjector


def result(name="synthetic", day=date(2025, 1, 2), gate=None):
    context = RunContext(
        name, RunMode.POINT_IN_TIME, day, datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    ledger = RunLedger(name)
    owner = RunSession(context, {}, ledger, gate=gate)
    metrics = PortfolioMetrics(total_value=100.0, invested_value=100.0)
    config = InvestorConfig()
    return RunResult(
        metrics, config, name, "synthetic-analysis", ledger, owner,
        ReportView.build(metrics, config),
    )


def test_report_view_is_deeply_defensive_and_preserves_typed_pandas_facts():
    index = pd.date_range("2025-01-01", periods=2, tz="Europe/Rome", name="day")
    metrics = PortfolioMetrics(
        holdings_df=pd.DataFrame({
            "cost_basis_eur": [50.0, 60.0],
            "nested": [{"items": [1]}, {"items": [2]}],
        }, index=index),
        portfolio_history=pd.Series([100.0, 105.0], index=index, name="NAV"),
        market_quotes=({"name": "Index", "spark": [10.0, 11.0]},),
    )
    metrics.portfolio_history.attrs["nested"] = {"items": [7]}
    config = InvestorConfig()
    view = ReportView.build(metrics, config)
    metrics.holdings_df.iloc[0, 1]["items"].append(99)
    metrics.portfolio_history.iloc[0] = 999
    metrics.portfolio_history.attrs["nested"]["items"].append(99)
    config.invested_allocation_targets_pctg.clear()
    first, first_config = view.project()
    first.holdings_df.iloc[0, 1]["items"].append(77)
    first.market_quotes[0]["spark"][0] = 999
    first_config.invested_allocation_targets_pctg.clear()
    second, second_config = view.project()
    assert second.holdings_df.iloc[0, 1] == {"items": [1]}
    assert second.portfolio_history.iloc[0] == 100
    assert second.portfolio_history.attrs == {"nested": {"items": [7]}}
    assert second.market_quotes[0]["spark"] == [10.0, 11.0]
    assert second_config.invested_allocation_targets_pctg
    assert second.portfolio_history.index.equals(index)


def test_report_projection_preserves_duplicate_columns_and_multiindex():
    frame = pd.DataFrame(
        [[1.0, 2.0]], columns=["same", "same"],
        index=pd.MultiIndex.from_tuples([("a", 1)], names=["letter", "ordinal"]),
    )
    view = ReportView.build(PortfolioMetrics(holdings_df=frame), InvestorConfig())
    pd.testing.assert_frame_equal(view.project()[0].holdings_df, frame)


def test_summary_reads_the_completed_facts_not_mutated_source_cube():
    completed = result()
    completed.metrics.total_value = 999
    summary = SummaryProjector.project(completed)
    assert summary.metrics["total_value_eur"] == 100.0


def test_scope_binds_old_clock_diagnostics_and_owner_then_restores_caller():
    first, second = result("first"), result("second", date(2025, 2, 3))
    first.session.diagnostics.append(data_quality.Issue("WARNING", "first", "first only"))
    second.session.diagnostics.append(data_quality.Issue("WARNING", "second", "second only"))
    record_last_run_result(second)
    with runtime.bind_context(second.session.context):
        with completed_run_scope(first):
            assert current_session() is first.session
            assert last_run_result() is first
            assert runtime.today() == date(2025, 1, 2)
            assert [issue.message for issue in data_quality.issues()] == ["first only"]
        assert current_session() is None
        assert last_run_result() is second
        assert runtime.today() == date(2025, 2, 3)
        assert [issue.message for issue in data_quality.issues()] == ["second only"]


def test_another_analysis_waits_until_output_lease_is_released():
    gate = SerialExecutionGate()
    completed = result(gate=gate)
    began, entered = Event(), Event()

    def other():
        began.set()
        from tarzan.runtime.session import RunAttemptEnvelope
        with gate.acquire(RunAttemptEnvelope.create("other")):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with completed_run_scope(completed):
            task = pool.submit(other)
            assert began.wait(5)
            time.sleep(0.02)
            assert not entered.is_set()
            assert gate.active_count == 1
        task.result(timeout=5)
        assert entered.is_set()


def test_cross_run_nested_scope_fails_and_releases_without_deadlock():
    first, second = result("first"), result("second")
    with activate_session(first.session):
        with pytest.raises(RuntimeError, match="another run"):
            with completed_run_scope(second):
                pass
    with completed_run_scope(second):
        assert current_session() is second.session


def test_ai_attempt_mirrors_only_to_matching_completed_ledger():
    first = result()
    record_last_run_result(first)
    evidence = ai_summary.GeminiProviderResult(
        (ai_summary.GeminiAttemptEvidence(
            "Gemini", "synthetic", "GROUNDED", 1, "FAILED", selected_fallback="SIGNALS",
        ),), "SIGNALS", "UNAVAILABLE",
    )
    with runtime.bind_context(first.session.context):
        ai_summary._store_provider_result(evidence)
    attempts = [e for e in first.ledger.entries if e.entry_type is LedgerEntryType.PROVIDER_ATTEMPT]
    assert len(attempts) == 1
    runtime.reset(attempt_id="other")
    ai_summary._store_provider_result(evidence)
    assert len([e for e in first.ledger.entries if e.entry_type is LedgerEntryType.PROVIDER_ATTEMPT]) == 1


def test_logging_setup_preserves_foreign_handlers_and_is_idempotent(tmp_path):
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        main.setup_logging(str(tmp_path))
        before = tuple(root.handlers)
        main.setup_logging(str(tmp_path))
        assert tuple(root.handlers) == before
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_logs_stay_with_owner_and_setup_does_not_erase_completed_records(tmp_path):
    first, second = result("first"), result("second")
    main.setup_logging(str(tmp_path))
    with completed_run_scope(first):
        logging.getLogger("tarzan.synthetic").warning("first only")
    main.setup_logging(str(tmp_path))
    with completed_run_scope(second):
        logging.getLogger("tarzan.synthetic").warning("second only")
    with completed_run_scope(first):
        assert [r["message"] for r in LOG_CAPTURE.records] == ["first only"]
    assert [r["message"] for r in second.session.log_records] == ["second only"]


def fake_cli(monkeypatch):
    completed = result()

    def run(**kwargs):
        record_last_run_result(completed)
        return completed.compatibility_tuple()

    monkeypatch.setattr(orchestrator, "run", run)
    monkeypatch.setattr("tarzan.export.newsletter.render_newsletter", lambda **k: "<html>core</html>")
    return completed


def test_cli_default_does_not_run_optional_candidate_backtest(monkeypatch, tmp_path):
    fake_cli(monkeypatch)
    monkeypatch.setattr(
        "tarzan.backtest.newsletter_portfolios",
        lambda **k: pytest.fail("default CLI must not run the optional backtest"),
    )
    assert main.main(["--output", str(tmp_path)]) == 0
    manifests = list(tmp_path.glob("**/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["files"]["newsletter.html"]


def test_optional_failure_happens_after_core_commit_and_cannot_overwrite_it(
    monkeypatch, tmp_path,
):
    completed = fake_cli(monkeypatch)
    committed = []

    def optional(**kwargs):
        core = list(tmp_path.glob("*/synthetic/manifest.json"))
        assert len(core) == 1
        committed.append(core[0].read_bytes())
        raise RuntimeError("synthetic optional failure")

    monkeypatch.setattr("tarzan.backtest.newsletter_portfolios", optional)
    assert main.main(["--output", str(tmp_path), "--what-if"]) == 0
    core = list(tmp_path.glob("*/synthetic/manifest.json"))[0]
    assert core.read_bytes() == committed[0]
    manifest = json.loads(core.read_text())
    for filename, checksum in manifest["files"].items():
        assert hashlib.sha256((core.parent / filename).read_bytes()).hexdigest() == checksum
    assert any(f.stable_code == "OPTIONAL_WHAT_IF_FAILED" for f in completed.ledger.failure_records())
    assert list(tmp_path.glob("*/what-if/synthetic/manifest.json"))


def test_core_disk_failure_returns_failure_without_claiming_success(monkeypatch, tmp_path, caplog):
    completed = fake_cli(monkeypatch)
    monkeypatch.setattr(
        LocalArtifactWriter, "finalize",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic disk full")),
    )
    assert main.main(["--output", str(tmp_path)]) == 1
    assert not any("committed successfully" in record.message for record in caplog.records)
    assert any(f.stable_code == "LOCAL_ARTIFACT_FINALIZATION_FAILED" for f in completed.ledger.failure_records())


def test_core_render_failure_retains_financial_facts_but_blocks_output(monkeypatch, tmp_path):
    completed = fake_cli(monkeypatch)
    monkeypatch.setattr(
        "tarzan.export.newsletter.render_newsletter",
        lambda **k: (_ for _ in ()).throw(RuntimeError("synthetic renderer failure")),
    )
    assert main.main(["--output", str(tmp_path)]) == 1
    assert completed.metrics.total_value == 100
    assert any(f.stable_code == "NEWSLETTER_RENDER_FAILED" for f in completed.ledger.failure_records())
    manifest = json.loads(next(tmp_path.glob("**/manifest.json")).read_text())
    assert "newsletter.html" not in manifest["files"]
    assert manifest["publication_state"] == "BLOCK_ALL_DELIVERY"


def test_semantic_failure_cannot_finalize_normal_newsletter(monkeypatch, tmp_path):
    fake_cli(monkeypatch)
    monkeypatch.setattr(
        "tarzan.export.newsletter._semantic.validate_newsletter_semantics",
        lambda *a, **k: ("synthetic wrong chart identity",),
    )
    assert main.main(["--output", str(tmp_path)]) == 1
    manifest = json.loads(next(tmp_path.glob("**/manifest.json")).read_text())
    assert manifest["publication_state"] == "BLOCK_ALL_DELIVERY"
    assert "newsletter.html" not in manifest["files"]


def test_presentation_uses_prepared_market_facts_without_provider_io(monkeypatch):
    from tarzan.export.newsletter._sections_perf import _build_markets
    from tarzan.export.newsletter._constants import _NewsletterContext
    from tarzan.data import market_quotes

    monkeypatch.setattr(market_quotes, "fetch_market_quotes", lambda: pytest.fail("presentation IO"))
    metrics = PortfolioMetrics(market_quotes=())
    assert _build_markets(_NewsletterContext(metrics, InvestorConfig()))["available"] is False
    assert "markets_today" not in ai_summary.build_digest(metrics, InvestorConfig())
