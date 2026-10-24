"""Original NAV, immutable report, HTML and no-email publication boundaries."""

from dataclasses import replace
from datetime import date, datetime, time as day_time, timezone
import os
from pathlib import Path
import smtplib
import socket
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import verify_newsletter as verify
from tarzan import delivery, orchestrator, runtime
from tarzan.contracts.report import ReportView
from tarzan.engine.metrics import MetricsEngine
from tarzan.engine.returns_builder import _build_daily_series, _trim_carried_tail
from tarzan.engine.stats import to_business_day_series
from tarzan.export.newsletter import render_newsletter
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime import artifacts, diagnostics, sealed
from tarzan.runtime.ledger import Availability, LedgerEntryType, RunLedger
from tarzan.runtime.publication import PublicationEvaluator
from tarzan.runtime.session import (
    RunContext, RunMode, RunResult, RunSession, completed_run_scope,
    record_last_run_result,
)


CASES = [
    pytest.param(date(2026, 10, 23), False, id="friday-control"),
    pytest.param(date(2026, 10, 24), True, id="saturday-accounting-tail"),
    pytest.param(date(2026, 10, 18), True, id="sunday-accounting-tail"),
    pytest.param(date(2026, 10, 19), True, id="monday-preopen-accounting-tail"),
]


@pytest.fixture(autouse=True)
def strictly_offline(monkeypatch):
    import requests
    from curl_cffi.requests import Session

    def forbidden(*args, **kwargs):
        pytest.fail("offline NAV integration invoked transport, claim or SMTP")

    previous_timezone = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "UTC")
    if hasattr(time, "tzset"):
        time.tzset()
    monkeypatch.setenv("TARZAN_DISABLE_AI", "1")
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "1")
    monkeypatch.delenv(diagnostics.DIRECTORY_ENV, raising=False)
    monkeypatch.delenv(sealed.KEY_ENV, raising=False)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(Session, "request", forbidden)
    monkeypatch.setattr(smtplib, "SMTP", forbidden)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbidden)
    monkeypatch.setattr(delivery, "send_email", forbidden)
    monkeypatch.setattr(delivery, "_delivery_claim_store", forbidden)
    monkeypatch.setattr(delivery, "_failed_checks_sha", lambda: None)
    monkeypatch.setattr("tarzan.export.ai_summary.generate_summary", lambda *a, **k: None)
    monkeypatch.setattr("tarzan.data.market_quotes.fetch_market_quotes", lambda *a, **k: [])
    runtime.reset(attempt_id="synthetic-nav-integration", invocation_source="test")
    yield
    if previous_timezone is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous_timezone
    if hasattr(time, "tzset"):
        time.tzset()
    runtime.reset(attempt_id="synthetic-nav-integration", invocation_source="test")


def source(day):
    friday = day - pd.Timedelta(days=(day.weekday() - 4) % 7)
    start = friday - pd.Timedelta(days=1)
    prices = {
        stamp.date(): (1000.0 if stamp.date() == start else 1100.0)
        for stamp in pd.date_range(start, day)
    }

    class Timeline:
        def isins(self):
            return ("SYNTHETIC",)

        def qty_at(self, isin, when):
            assert isin == "SYNTHETIC"
            return 1.0

    class Resolver:
        def price_on(self, isin, when):
            assert isin == "SYNTHETIC"
            return prices[when], "yfinance"

    steps = []
    full, actual = _build_daily_series(
        Timeline(), Resolver(), {}, [start], day,
        closed_identity_groups=lambda _: set(), step_evidence=steps,
    )
    measured = _trim_carried_tail(full)
    context = {
        "portfolio_history": to_business_day_series(measured),
        "_portfolio_history_basis": "ORDER_FLOW_ADJUSTED_NAV",
        "_order_series": SimpleNamespace(
            daily_series=measured, daily_return_steps=steps,
            history_availability=Availability.AVAILABLE,
        ),
    }
    MetricsEngine._performance(SimpleNamespace(), context)
    metrics = PortfolioMetrics(
        total_value=1100.0, invested_value=1100.0,
        portfolio_history=context["portfolio_history"],
        actual_value_series=actual, external_flows={},
        performance=context["performance"],
        session_move_evidence=context["session_move_evidence"],
        return_nav_series=context["return_nav_series"],
        history_source_basis="ORDER_FLOW_ADJUSTED_NAV",
    )
    assert metrics.return_nav_series is measured
    return metrics, measured


def owned(metrics, day, name):
    context = RunContext(
        name, RunMode.LIVE, None,
        datetime.combine(day, day_time(4, 21), tzinfo=timezone.utc),
    )
    assert context.analysis_date == day
    ledger = RunLedger(name)
    ledger.append(LedgerEntryType.STAGE, {"stage": "run", "outcome": "SUCCEEDED"})
    session = RunSession(context, {}, ledger)
    config = InvestorConfig()
    return RunResult(
        metrics, config, name, "synthetic-nav-analysis", ledger, session,
        ReportView.build(metrics, config),
    )


def real_render(result):
    with completed_run_scope(result):
        metrics, config = result.report_view.project()
        audit = {}
        html = render_newsletter(
            metrics, config, ai_summary="", semantic_audit=audit, run_result=result,
        )
        return html, validate_newsletter_semantics(metrics, audit, html)


def isolated_publish(result, monkeypatch):
    writes = []

    class NoFiles:
        def __init__(self, *args, **kwargs):
            self.directory = Path("/synthetic-no-files")

    monkeypatch.setattr(artifacts, "LocalArtifactWriter", NoFiles)
    monkeypatch.setattr(
        delivery, "_write_delivery_artifacts",
        lambda writer, owner, html, state: writes.append((owner, html, state)),
    )
    with completed_run_scope(result):
        status = delivery._publish_and_deliver(
            result, result.metrics, result.config,
            "synthetic-user", "synthetic-password", "synthetic@example.invalid",
            "smtp.invalid", 465, 1, "Synthetic", "offline-test", True,
        )
    return status, writes


def verification_main(result, monkeypatch):
    capture = []

    def fake_acquisition(**kwargs):
        record_last_run_result(result)
        return result.compatibility_tuple()

    monkeypatch.setattr(orchestrator, "run", fake_acquisition)
    monkeypatch.setattr(verify, "_inputs", lambda _: {
        "orders": "synthetic-no-file", "config": "synthetic-no-file",
        "targets_per_holding": None,
    })
    monkeypatch.setattr(verify, "_fingerprints", lambda _: {})
    monkeypatch.setattr(verify, "_close_delivery_inputs", lambda *a, **k: None)
    monkeypatch.setattr(verify, "setup_logging", lambda: None)
    monkeypatch.setattr(verify.log_redaction, "install", lambda **k: None)
    monkeypatch.setattr(
        diagnostics, "capture_configured",
        lambda owner, **values: capture.append((owner, values)) or True,
    )
    return verify.main([]), capture


@pytest.mark.parametrize("day,carried", CASES)
def test_original_nav_report_render_and_guard_accept_the_same_live_clock(day, carried):
    metrics, measured = source(day)
    assert metrics.actual_value_series.index[-1].date() == day
    assert bool(metrics.actual_value_series.index[-1] > measured.index[-1]) is carried
    assert metrics.session_move_evidence.source_end_date == measured.index[-1].date()
    html, errors = real_render(owned(metrics, day, "render-source"))
    assert html and errors == ()


@pytest.mark.parametrize("day,carried", CASES)
def test_real_verification_entrypoint_checks_the_weekend_render_without_sending(
    day, carried, monkeypatch,
):
    metrics, _ = source(day)
    result = owned(metrics, day, "verify-source")
    status, capture = verification_main(result, monkeypatch)
    assert status == 0
    assert capture[0][0] is result and capture[0][1]["exit_code"] == 0
    assert PublicationEvaluator.evaluate(result.ledger.failure_records()).decision.value == "SEND_NORMAL"
    events = [
        entry.payload for entry in result.ledger.entries
        if entry.entry_type is LedgerEntryType.DELIVERY
    ]
    assert events[-1]["state"] == "VERIFICATION_ONLY"
    assert events[-1]["smtp_invoked"] is False and events[-1]["claim_invoked"] is False


@pytest.mark.parametrize("day,carried", CASES)
def test_real_publisher_dry_run_accepts_the_weekend_source_without_claim_or_smtp(
    day, carried, monkeypatch,
):
    metrics, _ = source(day)
    result = owned(metrics, day, "publish-source")
    status, writes = isolated_publish(result, monkeypatch)
    assert status == 0
    assert writes and writes[-1][-1] == "DRY_RUN"
    assert PublicationEvaluator.evaluate(result.ledger.failure_records()).decision.value == "SEND_NORMAL"
    events = [
        entry.payload for entry in result.ledger.entries
        if entry.entry_type is LedgerEntryType.DELIVERY
    ]
    assert events[-1]["state"] == "DRY_RUN" and events[-1]["smtp_invoked"] is False


@pytest.mark.parametrize("boundary", ["verify", "publish"])
def test_tampered_source_dates_still_block_both_real_no_email_boundaries(boundary, monkeypatch):
    metrics, _ = source(date(2026, 10, 23))
    metrics.session_move_evidence = replace(
        metrics.session_move_evidence, source_end_date=date(2026, 10, 24),
    )
    result = owned(metrics, date(2026, 10, 24), "tampered-native-date")
    if boundary == "verify":
        status, _ = verification_main(result, monkeypatch)
    else:
        status, writes = isolated_publish(result, monkeypatch)
        assert writes[-1][-1] == "SEMANTIC_VALIDATION_FAILED"
    assert status == 1
    assert PublicationEvaluator.evaluate(result.ledger.failure_records()).decision.value == "BLOCK_ALL_DELIVERY"
    assert any(
        failure.stable_code == "NEWSLETTER_SEMANTIC_INVARIANT_FAILED"
        for failure in result.ledger.failure_records()
    )
    events = [
        entry.payload for entry in result.ledger.entries
        if entry.entry_type is LedgerEntryType.DELIVERY
    ]
    assert events and events[-1]["smtp_invoked"] is False
