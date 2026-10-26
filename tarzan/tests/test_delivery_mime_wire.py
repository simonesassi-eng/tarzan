"""Native email serialization with only the SMTP wire boundary replaced."""

import json
import logging
from email import policy
from email.parser import BytesParser
from smtplib import SMTP_SSL as NativeSMTPSSL

import pytest

from tarzan import delivery
from tarzan.delivery.claims import LocalJsonDeliveryClaimStore
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.tests.test_delivery_audit import isolated_delivery


@pytest.fixture
def smtp_wire(monkeypatch):
    events, messages = [], []
    failures = {}

    class SMTP(NativeSMTPSSL):
        def __init__(self, host, port, *, timeout):
            assert (host, port, timeout) == ("smtp.example.invalid", 465, 30)
            self.esmtp_features = {}
            events.append("connect")

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            events.append("close")
            return False

        def login(self, sender, password):
            assert (sender, password) == (
                "sender@example.invalid", "synthetic-not-a-credential",
            )
            events.append("login")
            if "login" in failures:
                raise failures["login"]

        def ehlo_or_helo_if_needed(self):
            events.append("ehlo")

        def sendmail(self, sender, recipients, message, mail_options=(), rcpt_options=()):
            events.append("wire")
            messages.append((sender, recipients, message))
            if "wire" in failures:
                raise failures["wire"]
            return {}

    # send_message is inherited unchanged: the real stdlib envelope selection
    # and BytesGenerator serialize the application's real EmailMessage.
    assert SMTP.send_message is NativeSMTPSSL.send_message
    monkeypatch.setattr(delivery.smtplib, "SMTP_SSL", SMTP)
    return events, messages, failures


def send(html, subject, before_invoke):
    delivery.send_email(
        html=html, subject=subject,
        sender="sender@example.invalid", recipient="reader@example.invalid",
        smtp_host="smtp.example.invalid", smtp_port=465,
        smtp_pass="synthetic-not-a-credential", before_invoke=before_invoke,
    )


@pytest.mark.parametrize("html", [
    '<html><body>€ 1.234,56 — qualità Σ ☕ &amp; &lt;ETF&gt;</body></html>',
    "<html>\r\n<body>\n<p>Line endings</p>\r\n</body></html>\n",
    "<html><body>" + (
        '<svg viewBox="0 0 100 100"><path d="M0 0L100 100"/></svg>'
        '<p>€ 1.234,56 — qualità &amp; benchmark</p>'
    ) * 8000 + "</body></html>",
], ids=["unicode-and-escaping", "mixed-line-endings", "large-inline-svg"])
def test_native_mime_roundtrip_preserves_the_html_subject_and_envelope(smtp_wire, html):
    events, messages, _ = smtp_wire
    subject = "Tarzan — qualità, € +1,23%"
    send(html, subject, lambda: events.append("before"))
    assert events == ["connect", "login", "before", "ehlo", "wire", "close"]
    assert len(messages) == 1
    sender, recipients, wire = messages[0]
    assert sender == "sender@example.invalid"
    assert recipients == ["reader@example.invalid"]
    assert isinstance(wire, bytes)
    assert b"\n" not in wire.replace(b"\r\n", b"")
    assert all(len(line) <= 998 for line in wire.split(b"\r\n"))
    message = BytesParser(policy=policy.default).parsebytes(wire)
    assert message.defects == []
    assert str(message["Subject"]) == subject
    assert str(message["From"]) == sender
    assert str(message["To"]) == recipients[0]
    assert message["Auto-Submitted"] == "auto-generated"
    assert message["X-Tarzan-Message-Type"] == "newsletter"
    assert message["Message-ID"].endswith("@tarzan.local>")
    assert message["Date"].datetime.tzinfo is not None
    assert message.get_content_type() == "multipart/alternative"
    parts = list(message.iter_parts())
    assert [part.get_content_type() for part in parts] == ["text/plain", "text/html"]
    assert all(part.defects == [] for part in parts)
    assert subject in parts[0].get_content()
    expected = html.replace("\r\n", "\n")
    if not expected.endswith("\n"):
        expected += "\n"  # The native text MIME API adds a final newline.
    assert parts[1].get_content().replace("\r\n", "\n") == expected


@pytest.mark.parametrize("field", ["subject", "sender", "recipient"])
def test_native_header_injection_is_rejected_before_any_smtp_or_claim(monkeypatch, field):
    events = []
    monkeypatch.setattr(
        delivery.smtplib, "SMTP_SSL",
        lambda *args, **kwargs: events.append("forbidden-connect"),
    )
    arguments = {
        "html": "<html>safe</html>", "subject": "safe",
        "sender": "sender@example.invalid", "recipient": "reader@example.invalid",
        "smtp_host": "smtp.example.invalid", "smtp_port": 465,
        "smtp_pass": "synthetic-not-a-credential",
        "before_invoke": lambda: events.append("forbidden-claim"),
    }
    arguments[field] += "\r\nBcc: intruder@example.invalid"
    with pytest.raises(ValueError):
        delivery.send_email(**arguments)
    assert events == []


@pytest.mark.parametrize("phase", ["login", "before", "wire"])
def test_native_send_keeps_the_original_failure_closes_and_never_retries(smtp_wire, phase):
    events, messages, failures = smtp_wire
    original = RuntimeError("synthetic-primary")
    if phase != "before":
        failures[phase] = original

    def before():
        events.append("before")
        if phase == "before":
            raise original

    with pytest.raises(RuntimeError) as caught:
        send("<html>safe</html>", "safe", before)
    assert caught.value is original
    assert events[-1] == "close"
    assert events.count("connect") == 1
    assert events.count("before") == int(phase != "login")
    assert len(messages) == int(phase == "wire")


@pytest.fixture
def smtp_native_exit(monkeypatch):
    """Retain the real SMTP context-manager teardown as well as serialization."""
    events, failures = [], {}

    class SMTP(NativeSMTPSSL):
        def __init__(self, host, port, *, timeout):
            self.esmtp_features = {}
            events.append("connect")

        def __enter__(self):
            return self

        def login(self, sender, password):
            events.append("login")
            if "login" in failures:
                raise failures["login"]

        def ehlo_or_helo_if_needed(self):
            events.append("ehlo")

        def sendmail(self, sender, recipients, message, mail_options=(), rcpt_options=()):
            events.append("wire")
            if "wire" in failures:
                raise failures["wire"]
            events.append("ack")
            return {}

        def docmd(self, command, *args):
            assert command == "QUIT"
            events.append("quit")
            if isinstance(failures["quit"], BaseException):
                raise failures["quit"]
            return failures["quit"], b"synthetic-quit-reply"

        def close(self):
            events.append("close")

    assert SMTP.__exit__ is NativeSMTPSSL.__exit__
    assert SMTP.send_message is NativeSMTPSSL.send_message
    monkeypatch.setattr(delivery.smtplib, "SMTP_SSL", SMTP)
    return events, failures


@pytest.mark.parametrize("quit_failure", [503, TimeoutError("synthetic-quit-timeout")])
def test_native_quit_failure_after_ack_does_not_turn_delivery_into_failure(
    smtp_native_exit, caplog, quit_failure,
):
    events, failures = smtp_native_exit
    failures["quit"] = quit_failure
    with caplog.at_level(logging.INFO, logger="tarzan.newsletter"):
        send("<html>safe</html>", "safe", lambda: events.append("before"))
    assert events == ["connect", "login", "before", "ehlo", "wire", "ack", "quit", "close"]
    assert sum(
        getattr(record, "public_code", None) == "SMTP_ACKNOWLEDGED"
        for record in caplog.records
    ) == 1
    cleanup = [
        record for record in caplog.records
        if getattr(record, "public_code", None) == "SMTP_CLEANUP_AFTER_ACK"
    ]
    assert len(cleanup) == 1
    assert "synthetic-quit" not in cleanup[0].getMessage()


@pytest.mark.parametrize("phase", ["login", "before", "wire"])
@pytest.mark.parametrize("quit_failure", [503, TimeoutError("synthetic-quit-timeout")])
def test_native_quit_failure_never_masks_the_primary_delivery_error(
    smtp_native_exit, phase, quit_failure,
):
    events, failures = smtp_native_exit
    original = RuntimeError("synthetic-primary")
    failures["quit"] = quit_failure
    if phase != "before":
        failures[phase] = original

    def before():
        events.append("before")
        if phase == "before":
            raise original

    with pytest.raises(RuntimeError) as caught:
        send("<html>safe</html>", "safe", before)
    assert caught.value is original
    assert events[-2:] == ["quit", "close"]
    assert events.count("wire") == int(phase == "wire")
    assert "ack" not in events


def test_native_quit_failure_never_masks_primary_cancellation(smtp_native_exit):
    events, failures = smtp_native_exit
    original = KeyboardInterrupt("synthetic-cancellation")
    failures.update(login=original, quit=503)
    with pytest.raises(KeyboardInterrupt) as caught:
        send("<html>safe</html>", "safe", lambda: events.append("before"))
    assert caught.value is original
    assert events == ["connect", "login", "quit", "close"]


def test_post_ack_cancellation_is_not_suppressed(smtp_native_exit):
    events, failures = smtp_native_exit
    original = KeyboardInterrupt("synthetic-post-ack-cancellation")
    failures["quit"] = original
    with pytest.raises(KeyboardInterrupt) as caught:
        send("<html>safe</html>", "safe", lambda: events.append("before"))
    assert caught.value is original
    assert events == ["connect", "login", "before", "ehlo", "wire", "ack", "quit", "close"]


@pytest.mark.parametrize("quit_failure", [503, TimeoutError("synthetic-quit-timeout")])
def test_native_post_ack_quit_failure_persists_success_and_suppresses_resend(
    smtp_native_exit, isolated_delivery, monkeypatch, tmp_path, quit_failure,
):
    events, failures = smtp_native_exit
    failures["quit"] = quit_failure
    store = LocalJsonDeliveryClaimStore(tmp_path / "claims.json")
    monkeypatch.setattr(delivery, "_delivery_claim_store", lambda: store)
    monkeypatch.setattr(delivery, "resolve_inputs", lambda: {
        "orders": "orders.csv", "config": "targets.csv", "targets_per_holding": None,
    })
    monkeypatch.setattr(delivery, "run", lambda **kwargs: (
        PortfolioMetrics(total_value=100), InvestorConfig(),
    ))
    monkeypatch.setattr(delivery, "render_newsletter", lambda **kwargs: "<html>safe</html>")
    monkeypatch.setattr(delivery, "build_subject", lambda *args: "safe")
    assert delivery.run_and_send() == 0
    assert events.count("wire") == events.count("ack") == 1
    claims = json.loads(store.path.read_text())["claims"]
    assert len(claims) == 1
    assert next(iter(claims.values()))["state"] == "ACKNOWLEDGED_SUCCESS"
    assert delivery.run_and_send() == 0
    assert events.count("wire") == events.count("ack") == 1
    states = {
        json.loads(path.read_text())["delivery_state"]
        for path in tmp_path.glob("output/**/manifest.json")
    }
    assert states == {"ACKNOWLEDGED_SUCCESS", "SUPPRESSED_DUPLICATE"}
