"""Pre-fix exploration oracles for production-readiness conditions C9-C12."""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path

import pytest

from tarzan import runtime
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime import report_html


# **Validates: Requirements 2.9**
def test_c9_overlapping_attempts_cannot_replace_run_owned_clock_state():
    first_date = date(2024, 1, 31)
    second_date = date(2025, 6, 30)
    first_configured = threading.Event()
    second_configured = threading.Event()
    observed: dict[str, object] = {}

    def first_run() -> None:
        runtime.configure(deterministic=True, as_of=first_date)
        first_configured.set()
        assert second_configured.wait(timeout=5)
        observed["first"] = runtime.today()

    def second_run() -> None:
        assert first_configured.wait(timeout=5)
        runtime.configure(deterministic=True, as_of=second_date)
        observed["second"] = runtime.today()
        second_configured.set()

    threads = [
        threading.Thread(target=first_run, name="c9-first"),
        threading.Thread(target=second_run, name="c9-second"),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert not [thread.name for thread in threads if thread.is_alive()]
    finally:
        runtime.reset()

    assert observed == {"first": first_date, "second": second_date}, (
        f"run-owned clocks crossed attempt boundaries: {observed}"
    )


# **Validates: Requirements 2.10**
def test_c10_workflow_validates_without_credentials_before_publication():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/newsletter.yml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    mutable_actions = re.findall(r"uses:\s*[^\s#]+@(?![0-9a-f]{40}(?:\s|$))[^\s#]+", workflow)
    open_dependencies = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        and (">=" in line or "~=" in line or "==" not in line or "--hash=sha256:" not in line)
    ]
    has_validation_job = bool(re.search(r"(?m)^\s{2}validate:\s*$", workflow))
    publication_depends_on_validation = bool(re.search(r"(?m)^\s+needs:\s*validate\s*$", workflow))
    job_scoped_secrets = bool(re.search(r"(?ms)^\s{4}env:\s*\n(?:\s{6}.+secrets\.)", workflow))

    assert {
        "has_validation_job": has_validation_job,
        "publication_depends_on_validation": publication_depends_on_validation,
        "job_scoped_secrets": job_scoped_secrets,
        "mutable_actions": mutable_actions,
        "open_dependencies": open_dependencies,
    } == {
        "has_validation_job": True,
        "publication_depends_on_validation": True,
        "job_scoped_secrets": False,
        "mutable_actions": [],
        "open_dependencies": [],
    }


# **Validates: Requirements 2.11**
def test_c11_interrupted_report_publish_preserves_last_committed_evidence(
    monkeypatch, tmp_path,
):
    report = tmp_path / "report.html"
    previous = "<html>last committed report</html>"
    report.write_text(previous, encoding="utf-8")
    replace_calls: list[tuple[object, object]] = []

    def fail_replace(source, destination) -> None:
        replace_calls.append((source, destination))
        raise OSError("injected finalization failure")

    monkeypatch.setattr(report_html.os, "replace", fail_replace)
    published = report_html.write_report(
        str(tmp_path),
        generated_at="2025-06-30 00:00",
        log_records=[],
    )

    assert replace_calls and published is None
    assert report.read_text(encoding="utf-8") == previous


# **Validates: Requirements 2.12**
def test_c12_known_sector_reaches_allocation_and_reconciles_with_unknown():
    known = Holding(
        isin="KNOWN", ticker="KNOWN", quantity=1.0, cost_basis_eur=60.0,
        market_value_eur=60.0, current_price=60.0, current_value=60.0,
        currency="EUR", asset_class=AssetClass.EQUITIES, sector="Technology",
    )
    missing = Holding(
        isin="UNKNOWN", ticker="UNKNOWN", quantity=1.0, cost_basis_eur=40.0,
        market_value_eur=40.0, current_price=40.0, current_value=40.0,
        currency="EUR", asset_class=AssetClass.EQUITIES, sector=None,
    )
    engine = MetricsEngine([known, missing], InvestorConfig())
    ctx: dict = {}
    engine._valuation(ctx)
    engine._allocations(ctx)

    sector = {
        str(row.category): float(row.weight_pct)
        for row in ctx["allocation_by_sector"].itertuples()
    }
    assert sector == pytest.approx({"Technology": 60.0, "Unknown": 40.0}, abs=1e-6)
    assert sum(sector.values()) == pytest.approx(100.0, abs=1e-6)
