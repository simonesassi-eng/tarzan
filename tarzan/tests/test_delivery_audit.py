"""Offline delivery races and failure notification regressions."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tarzan import delivery
from tarzan.delivery.claims import (
    AppsScriptPropertiesDeliveryClaimStore,
    DeliveryState,
    LocalJsonDeliveryClaimStore,
)
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.orchestrator import run
from tarzan.runtime.session import last_run_result


@pytest.fixture
def isolated_delivery(monkeypatch, tmp_path):
    for name in (
        "GITHUB_ACTIONS", "DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN",
        "DELIVERY_CLAIM_STORE_PATH", "AUTHORIZED_RESEND_TOKEN",
        "DRIVE_FOLDER_ID", "GOOGLE_DRIVE_CREDENTIALS_JSON",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {
        "SMTP_USER": "sender@example.invalid",
        "SMTP_PASS": "test-only-credential",
        "RECIPIENT_EMAIL": "reader@example.invalid",
        "TARZAN_STABLE_EVENT_ID": "audit:same-event",
        "ISSUE_NUMBER": "1", "DRY_RUN": "0",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "now_local", lambda: dt.datetime(2025, 1, 3, 12))
    monkeypatch.setattr(delivery, "_failed_checks_sha", lambda: None)
    yield


@pytest.mark.parametrize("transport_mode", ["callback", "smtp"])
def test_run_and_send_concurrent_claimers_invoke_the_transport_once(
    monkeypatch, tmp_path, isolated_delivery, transport_mode,
):
    barrier = Barrier(2)

    class ConcurrentStore(LocalJsonDeliveryClaimStore):
        def claim(self, intent):
            result = super().claim(intent)
            assert result.state is DeliveryState.CLAIMED
            barrier.wait(timeout=10)
            return result

    store = ConcurrentStore(tmp_path / "claims.json")
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: store)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: {
        "orders": "orders.csv", "config": "targets.csv", "targets_per_holding": None,
    })
    monkeypatch.setattr(delivery, "run", lambda **kwargs: (PortfolioMetrics(total_value=100), InvestorConfig()))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: "<html>ok</html>")
    monkeypatch.setattr(delivery, "build_subject", lambda *args: "subject")
    sent = []

    def transport(**kwargs):
        kwargs["before_invoke"]()
        sent.append(kwargs["subject"])

    if transport_mode == "callback":
        monkeypatch.setattr(delivery, "send_email", transport)
    else:
        class SMTP:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def login(self, *args):
                pass

            def send_message(self, message):
                assert message["Auto-Submitted"] == "auto-generated"
                assert message["X-Tarzan-Message-Type"] == "newsletter"
                sent.append(message["Subject"])

        monkeypatch.setattr(delivery.smtplib, "SMTP_SSL", SMTP)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(delivery.run_and_send) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results == [0, 0]
    assert sent == ["subject"]
    claims = json.loads(store.path.read_text())["claims"]
    assert len(claims) == 1
    assert next(iter(claims.values()))["state"] == "ACKNOWLEDGED_SUCCESS"
    manifests = list(tmp_path.glob("output/**/manifest.json"))
    states = {json.loads(path.read_text())["delivery_state"] for path in manifests}
    assert states == {"ACKNOWLEDGED_SUCCESS", "SUPPRESSED_CONCURRENT"}


@pytest.mark.parametrize("source_kind", ["empty", "missing", "malformed"])
def test_failed_analysis_sends_only_a_sanitized_failure_notification(
    monkeypatch, tmp_path, isolated_delivery, source_kind,
):
    sources = {
        "empty": io.BytesIO(b"date,type,isin,quantity,gross_eur,net_eur\n"),
        "missing": str(tmp_path / "not-found.csv"),
        "malformed": io.BytesIO(b"date,type\n2025-01-01,buy\n"),
    }
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: {
        "orders": sources[source_kind], "config": None, "targets_per_holding": None,
    })
    monkeypatch.setattr(delivery, "run", lambda **kwargs: run(
        orders_source=kwargs["orders_source"], orders_filename="order_list.csv",
        deterministic=True, as_of=dt.date(2025, 1, 3),
    ))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: pytest.fail("normal newsletter rendered"))
    sent = []

    def transport(**kwargs):
        kwargs["before_invoke"]()
        sent.append(kwargs)

    monkeypatch.setattr(delivery, "send_email", transport)
    assert delivery.run_and_send() == 0
    assert len(sent) == 1
    assert sent[0]["subject"].endswith("Analysis failure")
    assert "UNHANDLED_RUN_FAILURE" in sent[0]["html"]
    assert "test-only-credential" not in sent[0]["html"]
    assert "not-found.csv" not in sent[0]["html"]
    assert last_run_result().metrics is None


def test_input_resolution_failure_is_notified_without_reusing_old_metrics(
    monkeypatch, isolated_delivery,
):
    def missing_inputs():
        raise FileNotFoundError("private-directory/secret-order-list.csv")

    monkeypatch.setattr(delivery, "resolve_inputs", missing_inputs)
    monkeypatch.setattr(delivery, "run", lambda **kwargs: pytest.fail("analysis should not start"))
    sent = []

    def transport(**kwargs):
        kwargs["before_invoke"]()
        sent.append(kwargs)

    monkeypatch.setattr(delivery, "send_email", transport)
    assert delivery.run_and_send() == 0
    assert len(sent) == 1
    assert "INPUT_RESOLUTION_FAILED" in sent[0]["html"]
    assert "secret-order-list" not in sent[0]["html"]
    assert last_run_result().metrics is None


def test_remote_transition_requires_a_matching_operation_acknowledgment(monkeypatch):
    store = AppsScriptPropertiesDeliveryClaimStore("https://claims.example.invalid/exec", "test-only-credential")
    monkeypatch.setattr(store, "_call", lambda request: {"state": request["target"]})
    with pytest.raises(RuntimeError, match="did not confirm the transition operation"):
        store.transition("logical-id", (DeliveryState.CLAIMED,), DeliveryState.SMTP_INVOCATION_STARTED)


def test_distinct_remote_calls_have_distinct_operations(monkeypatch):
    store = AppsScriptPropertiesDeliveryClaimStore("https://claims.example.invalid/exec", "test-only-credential")
    operations = []

    def successful(request):
        operations.append(request["operation_id"])
        return {"state": request["target"], "operation_id": request["operation_id"]}

    monkeypatch.setattr(store, "_call", successful)
    for _ in range(2):
        store.transition("logical-id", (DeliveryState.CLAIMED,), DeliveryState.SMTP_INVOCATION_STARTED)
    assert operations[0] != operations[1]
    assert all(len(operation) == 32 for operation in operations)


@pytest.mark.parametrize("failure_at", ["cleanup", "acknowledgment"])
def test_delivery_storage_failure_is_finalized_without_an_automatic_resend(
    monkeypatch, tmp_path, isolated_delivery, failure_at,
):
    class FailingStore(LocalJsonDeliveryClaimStore):
        def transition(self, logical_id, expected, target):
            failed_target = (
                DeliveryState.DEFINITE_PRE_SEND_FAILURE
                if failure_at == "cleanup" else DeliveryState.ACKNOWLEDGED_SUCCESS
            )
            if target is failed_target:
                raise RuntimeError("claim storage unavailable")
            return super().transition(logical_id, expected, target)

    store = FailingStore(tmp_path / "claims.json")
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: store)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: {
        "orders": "orders.csv", "config": "targets.csv", "targets_per_holding": None,
    })
    monkeypatch.setattr(delivery, "run", lambda **kwargs: (PortfolioMetrics(total_value=100), InvestorConfig()))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: "<html>ok</html>")
    monkeypatch.setattr(delivery, "build_subject", lambda *args: "subject")
    sent = []

    def transport(**kwargs):
        if failure_at == "cleanup":
            raise ConnectionError("SMTP connection failed before send")
        kwargs["before_invoke"]()
        sent.append(kwargs["subject"])

    monkeypatch.setattr(delivery, "send_email", transport)
    assert delivery.run_and_send() == 1
    expected_state = "DEFINITE_PRE_SEND_FAILURE" if failure_at == "cleanup" else "UNCERTAIN"
    manifests = list(tmp_path.glob("output/**/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["delivery_state"] == expected_state
    assert sent == ([] if failure_at == "cleanup" else ["subject"])
    if failure_at == "acknowledgment":
        assert delivery.run_and_send() == 0
        assert sent == ["subject"]


def test_production_apps_script_delivery_protocol_in_node():
    node = shutil.which(os.environ.get("TARZAN_TEST_NODE", "node"))
    if node is None:
        pytest.skip("Node.js is required to execute the Apps Script protocol tests")
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [node, str(Path(__file__).with_name("apps_script_delivery_claim_test.js")), str(root / "scripts/apps_script/Code.gs")],
        check=False, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "corruption checks passed" in completed.stdout
