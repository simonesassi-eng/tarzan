"""Native event metadata, output injection and last-moment delivery age guards."""

from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tarzan import delivery
from tarzan.contracts.events import (
    FreshnessState, MAX_DELIVERY_AGE_SECONDS, assess_freshness, resolve_workflow_event,
)
from tarzan.delivery.claims import DeliveryState, LocalJsonDeliveryClaimStore
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.session import last_run_result


NOW = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)


def stamp(age=0):
    return (NOW - timedelta(seconds=age)).isoformat()


@pytest.mark.parametrize("requested,created,automated,state", [
    (None, None, False, FreshnessState.ELIGIBLE),
    (None, None, True, FreshnessState.BLOCKED_INVALID_REQUEST),
    (stamp(), None, True, FreshnessState.BLOCKED_INVALID_REQUEST),
    (None, stamp(), True, FreshnessState.ELIGIBLE),
    (stamp(10), stamp(20), True, FreshnessState.ELIGIBLE),
    (stamp(2700), stamp(), True, FreshnessState.SKIPPED_STALE),
    (stamp(), stamp(2701), True, FreshnessState.SKIPPED_STALE),
    ("not-a-clock", stamp(), True, FreshnessState.BLOCKED_INVALID_REQUEST),
    ("2026-10-21T12:00:00", stamp(), True, FreshnessState.BLOCKED_INVALID_REQUEST),
    (stamp(-121), stamp(), True, FreshnessState.BLOCKED_INVALID_REQUEST),
    (stamp(-120), stamp(), True, FreshnessState.ELIGIBLE),
    (stamp(2699), stamp(), True, FreshnessState.ELIGIBLE),
])
def test_freshness_uses_oldest_operational_timestamp_not_the_portfolio_clock(requested, created, automated, state):
    result = assess_freshness(requested, created, now=NOW, automated=automated)
    assert result.state is state


def test_naive_operational_clock_fails_closed():
    assert not assess_freshness(stamp(), stamp(), now=NOW.replace(tzinfo=None), automated=True).eligible


@pytest.mark.parametrize("label", ["manual", "08:00", "official close", "on-demand", "23:00"])
def test_native_event_outputs_are_closed_single_line_values(label):
    result = resolve_workflow_event(
        {"client_payload": {"label": label, "event_id": "update:" + "a" * 64, "timestamp": stamp(30)}},
        event_name="repository_dispatch", run_id="123", run_created_at=stamp(20),
    )
    assert result == {
        "label": label, "event_id": "update:" + "a" * 64,
        "requested_at": stamp(30), "run_created_at": stamp(20),
    }


@pytest.mark.parametrize("label", ["ok\ninjected=true", "ok\rsecret", "x" * 81, "$bad", [], True])
def test_event_output_newline_shell_or_type_injection_is_rejected(label):
    with pytest.raises(ValueError):
        resolve_workflow_event(
            {"client_payload": {"label": label}}, event_name="repository_dispatch",
            run_id="123", run_created_at=stamp(),
        )


def test_arbitrary_event_context_becomes_an_opaque_stable_identity():
    payload = {"client_payload": {"event_id": "owner@example.invalid:private-account"}}
    first = resolve_workflow_event(payload, event_name="repository_dispatch", run_id="123", run_created_at=stamp())
    second = resolve_workflow_event(payload, event_name="repository_dispatch", run_id="456", run_created_at=stamp())
    assert first["event_id"] == second["event_id"]
    assert first["event_id"].startswith("dispatch:") and len(first["event_id"]) == 73
    assert "private" not in first["event_id"] and "owner" not in first["event_id"]


def test_manual_and_legacy_dispatch_use_native_run_creation_not_job_start():
    for event_name, event, label in (
        ("workflow_dispatch", {"inputs": {"trigger_label": "manual"}}, "manual"),
        ("repository_dispatch", {}, "on-demand"),
    ):
        result = resolve_workflow_event(event, event_name=event_name, run_id="123", run_created_at=stamp(2800))
        assert result["requested_at"] == stamp(2800)
        assert result["event_id"] == "workflow:123" and result["label"] == label
        assert not assess_freshness(result["requested_at"], result["run_created_at"], now=NOW, automated=True).eligible


@pytest.fixture
def resolver():
    path = Path(__file__).resolve().parents[2] / "scripts/resolve_newsletter_event.py"
    spec = importlib.util.spec_from_file_location("review_event_resolver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_environment(tmp_path):
    event = tmp_path / "native-event.json"
    event.write_text('{"inputs":{"trigger_label":"manual"}}')
    return {
        "GITHUB_REPOSITORY": "example/tarzan", "GITHUB_RUN_ID": "123",
        "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_EVENT_PATH": str(event),
        "GH_TOKEN": "synthetic-native-token", "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }


def native_response(*, run_id=123, repository="example/tarzan"):
    return {"id": run_id, "repository": {"full_name": repository}, "created_at": stamp(30)}


def test_resolver_uses_native_scoped_authorization_and_bounded_run_metadata(resolver, monkeypatch, tmp_path):
    environment = native_environment(tmp_path)
    calls = []

    def open_metadata(request, *, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        return io.BytesIO(json.dumps(native_response()).encode())

    monkeypatch.setattr(resolver, "urlopen", open_metadata)
    assert resolver.resolve(environment)["run_created_at"] == stamp(30)
    assert calls == [("https://api.github.com/repos/example/tarzan/actions/runs/123", "Bearer synthetic-native-token", 10)]


@pytest.mark.parametrize("case", ["different-run", "different-repo", "missing-time", "oversized", "no-token", "bad-repo", "bad-run"])
def test_native_metadata_mismatch_or_missing_authorization_cannot_start_delivery(resolver, monkeypatch, tmp_path, case):
    environment = native_environment(tmp_path)
    metadata = native_response()
    if case == "different-run":
        metadata["id"] = 999
    elif case == "different-repo":
        metadata["repository"]["full_name"] = "other/repo"
    elif case == "missing-time":
        metadata.pop("created_at")
    elif case == "no-token":
        environment["GH_TOKEN"] = ""
    elif case == "bad-repo":
        environment["GITHUB_REPOSITORY"] = "../outside"
    elif case == "bad-run":
        environment["GITHUB_RUN_ID"] = "123?secret"
    body = b"x" * (512 * 1024 + 1) if case == "oversized" else json.dumps(metadata).encode()
    monkeypatch.setattr(resolver, "urlopen", lambda *args, **kwargs: io.BytesIO(body))
    with pytest.raises(ValueError):
        resolver.resolve(environment)


def test_resolver_failure_does_not_print_native_body_context_or_credential(resolver, monkeypatch, tmp_path, capsys):
    for key, value in native_environment(tmp_path).items():
        monkeypatch.setenv(key, value)

    def fail(*args, **kwargs):
        raise OSError("synthetic-native-token owner@example.invalid private-ledger")

    monkeypatch.setattr(resolver, "urlopen", fail)
    assert resolver.main() == 1
    output = capsys.readouterr()
    assert output.err == "Newsletter event context unavailable; delivery is blocked.\n"
    assert not (tmp_path / "outputs").exists()


def test_native_event_preflight_needs_no_installed_third_party_dependencies():
    path = Path(__file__).resolve().parents[2] / "scripts/resolve_newsletter_event.py"
    environment = dict(os.environ)
    environment.pop("GH_TOKEN", None)
    # -S disables site packages. Missing native auth must stop before any I/O,
    # with a fixed diagnostic rather than an import failure or traceback.
    completed = subprocess.run(
        [sys.executable, "-S", str(path)], env=environment,
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 1
    assert completed.stderr == "Newsletter event context unavailable; delivery is blocked.\n"


@pytest.fixture
def live_delivery(monkeypatch, tmp_path):
    for key in ("DELIVERY_CLAIM_ENDPOINT", "DELIVERY_CLAIM_TOKEN", "AUTHORIZED_RESEND_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "SMTP_USER": "sender@example.invalid", "SMTP_PASS": "synthetic",
        "RECIPIENT_EMAIL": "reader@example.invalid", "DRY_RUN": "0",
        "GITHUB_ACTIONS": "true", "TARZAN_REQUESTED_AT": stamp(),
        "TARZAN_WORKFLOW_CREATED_AT": stamp(), "TARZAN_STABLE_EVENT_ID": "workflow:123",
        "ISSUE_NUMBER": "1",
    }.items():
        monkeypatch.setenv(key, value)
    clock = [NOW]
    monkeypatch.setattr(delivery, "_delivery_now_utc", lambda: clock[0])
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "now_local", lambda: NOW)
    monkeypatch.setattr(delivery, "_failed_checks_sha", lambda: None)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: {
        "orders": "synthetic-orders", "config": "synthetic-targets", "targets_per_holding": None,
    })
    monkeypatch.setattr(delivery, "run", lambda **kwargs: (PortfolioMetrics(total_value=100), InvestorConfig()))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: "<html>synthetic</html>")
    monkeypatch.setattr(delivery, "build_subject", lambda *args: "synthetic subject")
    store = LocalJsonDeliveryClaimStore(tmp_path / "claims.json")
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: store)
    return clock, store


def manifest(tmp_path):
    paths = list(tmp_path.glob("output/**/manifest.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


@pytest.mark.parametrize("invalid", [False, True])
def test_aged_or_invalid_request_stops_before_configuration_inputs_claims_or_smtp(monkeypatch, live_delivery, invalid):
    monkeypatch.setenv("TARZAN_REQUESTED_AT", "private-invalid-time" if invalid else stamp(2700))
    monkeypatch.delenv("SMTP_USER")
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: pytest.fail("private inputs opened"))
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: pytest.fail("claim created"))
    monkeypatch.setattr(delivery, "send_email", lambda **kwargs: pytest.fail("SMTP invoked"))
    assert delivery.run_and_send() == (1 if invalid else 0)
    result = last_run_result()
    assert result.metrics is None
    assert result.ledger.entries[-1].to_dict()["payload"]["smtp_invoked"] is False


def test_expiry_during_render_preserves_artifacts_but_creates_no_claim(monkeypatch, tmp_path, live_delivery):
    clock, store = live_delivery

    def render(**kwargs):
        clock[0] += timedelta(seconds=MAX_DELIVERY_AGE_SECONDS)
        return "<html>synthetic</html>"

    monkeypatch.setattr(delivery, "render_newsletter", render)
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: pytest.fail("stale claim created"))
    monkeypatch.setattr(delivery, "send_email", lambda **kwargs: pytest.fail("stale SMTP"))
    assert delivery.run_and_send() == 0
    assert manifest(tmp_path)["delivery_state"] == "SKIPPED_STALE"
    assert not store.path.exists()


@pytest.mark.parametrize("stage", ["checkpoint", "login", "cas"])
def test_last_moment_expiry_never_invokes_native_send_message(monkeypatch, tmp_path, live_delivery, stage):
    clock, store = live_delivery
    events = []
    if stage == "checkpoint":
        original = delivery._write_delivery_artifacts

        def checkpoint(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("checkpoint"):
                clock[0] += timedelta(seconds=MAX_DELIVERY_AGE_SECONDS)
            return result

        monkeypatch.setattr(delivery, "_write_delivery_artifacts", checkpoint)
    if stage == "cas":
        original = store.transition

        def transition(logical_id, expected, target):
            result = original(logical_id, expected, target)
            if target is DeliveryState.SMTP_INVOCATION_STARTED:
                clock[0] += timedelta(seconds=MAX_DELIVERY_AGE_SECONDS)
            return result

        monkeypatch.setattr(store, "transition", transition)

    class SMTP:
        def __init__(self, *args, **kwargs):
            events.append("connect")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, *args):
            events.append("login")
            if stage == "login":
                clock[0] += timedelta(seconds=MAX_DELIVERY_AGE_SECONDS)

        def send_message(self, message):
            pytest.fail("obsolete request invoked SMTP send_message")

    monkeypatch.setattr(delivery.smtplib, "SMTP_SSL", SMTP)
    assert delivery.run_and_send() == 0
    assert events == ([] if stage == "checkpoint" else ["connect", "login"])
    state = next(iter(json.loads(store.path.read_bytes())["claims"].values()))["state"]
    assert state == ("UNCERTAIN" if stage == "cas" else "CLAIMED")
    assert manifest(tmp_path)["delivery_state"] == "SKIPPED_STALE"


def test_fresh_request_preserves_delivery_and_never_seeds_optional_manual_series(monkeypatch, tmp_path, live_delivery):
    _, store = live_delivery
    sent = []
    from tarzan.data import manual_proxies
    monkeypatch.setattr(manual_proxies, "ingest", lambda *a, **k: pytest.fail("optional manual source ingested"))
    monkeypatch.setattr(manual_proxies, "get_series", lambda *a, **k: pytest.fail("optional manual source read"))

    def transport(**kwargs):
        kwargs["before_invoke"]()
        sent.append(kwargs["subject"])

    monkeypatch.setattr(delivery, "send_email", transport)
    assert delivery.run_and_send() == 0
    assert sent == ["synthetic subject"]
    assert manifest(tmp_path)["delivery_state"] == "ACKNOWLEDGED_SUCCESS"
    assert next(iter(json.loads(store.path.read_bytes())["claims"].values()))["state"] == "ACKNOWLEDGED_SUCCESS"
