"""Newsletter delivery service — run the pipeline and email the report.

This is the importable, testable home for what used to live in
``scripts/send_newsletter.py``. The script is now a thin shim that calls
:func:`run_and_send`; CI invokes the shim unchanged.

Kept here (not in ``scripts/``) so the multi-tenant loop (Track B) can call
``run_and_send`` per tenant rather than shelling out, and so the subject/
input-resolution logic is unit-testable.

Input CSVs are resolved in priority order:
    1. Drive mode — if GOOGLE_DRIVE_CREDENTIALS_JSON and DRIVE_FOLDER_ID are
       set, download the known input files from the Drive folder (public-repo
       friendly: personal data never lands in git).
    2. Local mode — fall back to ORDERS_PATH / TARGETS_PATH /
       TARGETS_PER_HOLDING_PATH (default .private/*.csv).

Environment variables (provided by GitHub Actions secrets in CI):
    SMTP_USER       (required) Gmail account that sends the newsletter
    SMTP_PASS       (required) Gmail App Password (NOT the account password)
    RECIPIENT_EMAIL (required) Inbox where the newsletter is delivered
    SMTP_HOST                  Default smtp.gmail.com
    SMTP_PORT                  Default 465 (SSL)
    ORDERS_PATH                Default .private/order_list.csv
    TARGETS_PATH               Default .private/targets.csv
    TARGETS_PER_HOLDING_PATH   Default .private/targets_per_holding.csv
    DRIVE_FOLDER_ID            Drive folder ID (no slashes)
    GOOGLE_DRIVE_CREDENTIALS_JSON  Service-account JSON key
    ISSUE_NUMBER               Default 1
    SUBJECT_PREFIX             Default "Portfolio Digest"
    DRY_RUN                    If "1", render only, do not send
    TRIGGER_LABEL              Free-form tag (currently not appended to subject)

The order list is the single source of truth: the snapshot (positions,
valuation, allocations) is derived from it, and it also drives the historical
value series and XIRR/TWROR.
"""

from __future__ import annotations

import html as html_lib
import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from tarzan.export.newsletter import render_newsletter
from tarzan.orchestrator import run
from tarzan.delivery.claims import (
    AppsScriptPropertiesDeliveryClaimStore,
    DeliveryClaimStore,
    DeliveryIntent,
    DeliveryPurpose,
    DeliveryState,
    LocalJsonDeliveryClaimStore,
    recipient_set_digest,
)

logger = logging.getLogger("tarzan.newsletter")

# Repo root (…/tarzan/delivery.py → parents[1]). Used to place local HTML copies
# under output/<date>/ the same way the CLI does.
ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    """Read an env var, optionally enforcing presence."""
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Set it in GitHub Secrets or your local environment."
        )
    return value or ""


def now_local() -> datetime:
    """Return the current time in Europe/Rome.

    GitHub Actions runners are in UTC, so ``datetime.now()`` there is 1–2 hours
    behind Italian local time (DST-dependent). The subject line, output
    filenames and run logs the user reads make more sense in wall-clock time.
    """
    return datetime.now(ZoneInfo("Europe/Rome"))


def build_subject(metrics, prefix: str, trigger_label: str = "") -> str:
    """Build the newsletter subject line.

    Example: "Portfolio Digest - 19:35 - uP&L +6.68%"

    The percentage is the unrealized P&L on current holdings
    ((total value − cost basis) / cost basis), the same figure the Hero shows
    as "Unrealized PnL".
    """
    cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - cost
    gain_pct = (total_gain / cost * 100) if cost > 0 else 0.0
    generated_at = now_local().strftime("%H:%M")
    sign = "+" if gain_pct >= 0 else "−"

    # Subject is exactly "<prefix> - HH:MM - uP&L ±X.XX%". The trigger label is
    # intentionally NOT appended: the scheduler's slot label already carries the
    # time, which duplicated the HH:MM in the subject.
    parts = [prefix or "Portfolio Digest", generated_at, f"uP&L {sign}{abs(gain_pct):.2f}%"]
    return " - ".join(parts)


def send_email(
    html: str,
    subject: str,
    sender: str,
    recipient: str,
    smtp_host: str,
    smtp_port: int,
    smtp_pass: str,
    before_invoke: Optional[Callable[[], None]] = None,
) -> None:
    """Send a single HTML message via Gmail SMTP over SSL.

    Plain-text fallback is generated automatically so non-HTML clients still
    see something readable.
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="tarzan.local")

    # Plain-text fallback
    msg.set_content(
        f"This is your Tarzan portfolio digest — {subject}. "
        "View this email in an HTML-capable client to see the full dashboard."
    )
    msg.add_alternative(html, subtype="html")

    logger.info("Connecting to %s:%d (SSL)...", smtp_host, smtp_port)
    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as smtp:
        smtp.login(sender, smtp_pass)
        if before_invoke is not None:
            before_invoke()
        smtp.send_message(msg)
    logger.info("Sent newsletter to %s with subject: %s", recipient, subject)


def resolve_inputs() -> dict[str, str | None]:
    """Resolve the pipeline input paths.

    Returns a dict with keys ``config``, ``orders`` and ``targets_per_holding``
    (values are absolute paths or None). The order list is the single source of
    truth — the snapshot is derived from it.

    Drive mode (credentials present) downloads the known input files that exist
    in the folder; it requires ``order_list.csv`` and ``targets.csv``. Local
    mode mirrors this via the *_PATH env vars.
    """
    drive_folder = _env("DRIVE_FOLDER_ID")
    drive_creds = _env("GOOGLE_DRIVE_CREDENTIALS_JSON")
    if drive_folder and drive_creds:
        from tarzan.delivery.drive_loader import KNOWN_INPUT_FILES, download_files
        logger.info("Loading inputs from Google Drive folder %s", drive_folder)
        files = download_files(
            folder_id=drive_folder,
            credentials_json=drive_creds,
            filenames=KNOWN_INPUT_FILES,
        )
        if "targets.csv" not in files:
            raise FileNotFoundError(
                "Drive folder is missing targets.csv (the config file)."
            )
        if "order_list.csv" not in files:
            raise FileNotFoundError(
                "Drive folder is missing order_list.csv (the order list that "
                "drives the whole report)."
            )
        return {
            "config": str(files["targets.csv"]),
            "orders": str(files["order_list.csv"]),
            "targets_per_holding": (
                str(files["targets_per_holding.csv"])
                if "targets_per_holding.csv" in files else None
            ),
        }

    # Local mode.
    targets_path = _env("TARGETS_PATH", ".private/targets.csv")
    orders_path = _env("ORDERS_PATH", ".private/order_list.csv")
    tph_path = _env("TARGETS_PER_HOLDING_PATH", ".private/targets_per_holding.csv")

    if not Path(orders_path).exists():
        raise FileNotFoundError(
            f"No order list found at {orders_path!r}, or set DRIVE_FOLDER_ID + "
            "GOOGLE_DRIVE_CREDENTIALS_JSON to load from Drive."
        )
    if not Path(targets_path).exists():
        raise FileNotFoundError(f"Config/targets file not found at {targets_path!r}.")

    logger.info(
        "Local inputs — orders=%s targets=%s per_holding=%s",
        orders_path, targets_path,
        tph_path if Path(tph_path).exists() else "(none)",
    )
    return {
        "config": targets_path,
        "orders": orders_path,
        "targets_per_holding": tph_path if Path(tph_path).exists() else None,
    }


# Manual-proxy source files (third-party index levels) that feed the carry /
# CTA backtest sleeves. Never committed to the repo; carried in the SAME private
# Drive folder as the personal inputs. filename → manual_proxies cache key.
_MANUAL_PROXY_SOURCES = {
    "US_BNPIF73P.xlsx": "CRRYSIM",   # BNP Enhanced Commodity Carry (UEQC/CRRY)
    "NHIndexMonthly.csv": "NHCTA",   # NilssonHedge CTA index
}


def _seed_manual_proxies() -> None:
    """Populate the manual-proxy cache (carry / CTA index levels) for the
    backtest, WITHOUT committing third-party data to the repo.

    Idempotent and fully guarded: a series already in the cache (warm via
    actions/cache) is left alone; a missing one is sourced — first from the
    private Drive folder (same channel as the personal inputs), then from a
    local path — and ingested. If neither is available the backtest simply
    uses its generic fallback for that sleeve. Never raises.
    """
    try:
        from tarzan.data import manual_proxies as mp
        missing = {fn: key for fn, key in _MANUAL_PROXY_SOURCES.items()
                   if mp.get_series(key) is None}
        if not missing:
            return  # all warm (e.g. restored from actions/cache)

        downloaded: dict[str, Path] = {}
        drive_folder = _env("DRIVE_FOLDER_ID")
        drive_creds = _env("GOOGLE_DRIVE_CREDENTIALS_JSON")
        if drive_folder and drive_creds:
            try:
                from tarzan.delivery.drive_loader import download_files
                downloaded = download_files(
                    folder_id=drive_folder, credentials_json=drive_creds,
                    filenames=list(missing),
                )
            except Exception as e:  # noqa: BLE001
                logger.info("Manual-proxy Drive fetch skipped (%s): %s",
                            type(e).__name__, e)

        for fn, key in missing.items():
            path = downloaded.get(fn)
            if path is None:
                candidate = ROOT / ".private" / fn
                if candidate.exists():
                    path = candidate
            if path is None:
                logger.info("Manual proxy %s: source %s not available — "
                            "backtest uses the generic fallback.", key, fn)
                continue
            try:
                mp.ingest(key, str(path))
            except Exception as e:  # noqa: BLE001
                logger.warning("Manual proxy ingest failed for %s: %s", key, e)
    except Exception as e:  # noqa: BLE001
        logger.info("Manual-proxy seeding skipped (%s): %s", type(e).__name__, e)


def _delivery_claim_store() -> DeliveryClaimStore:
    """Select cross-run durable control state; GitHub requires Apps Script."""
    endpoint = _env("DELIVERY_CLAIM_ENDPOINT")
    token = _env("DELIVERY_CLAIM_TOKEN")
    if bool(endpoint) != bool(token):
        raise RuntimeError("delivery claim endpoint and credential must be configured together")
    if endpoint and token:
        return AppsScriptPropertiesDeliveryClaimStore(endpoint, token)
    if _env("GITHUB_ACTIONS", "").lower() == "true":
        raise RuntimeError("GitHub publication requires the durable Apps Script claim service")
    # Local/manual operation retains a transactional host-local implementation.
    # It is never selected by the ephemeral GitHub runner.
    path = Path(_env(
        "DELIVERY_CLAIM_STORE_PATH",
        str(ROOT / "output" / "delivery_claims.json"),
    ))
    return LocalJsonDeliveryClaimStore(path)


def _failure_notification_html(result) -> str:
    """Render a minimal sanitized notification with no portfolio payload."""
    critical = [
        record for record in result.ledger.failure_records()
        if record.severity.upper() == "CRITICAL" and not record.automatically_corrected
    ]
    items = "".join(
        "<li>"
        + html_lib.escape(f"{record.stage}: {record.stable_code}")
        + "</li>"
        for record in critical
    ) or "<li>Run failed before critical details were finalized.</li>"
    return (
        "<!doctype html><html><body>"
        "<h1>Tarzan analysis could not be published</h1>"
        f"<p>Analysis ID: <code>{html_lib.escape(result.analysis_id)}</code></p>"
        f"<ul>{items}</ul>"
        "<p>The normal portfolio newsletter was blocked. Local diagnostics may "
        "be ephemeral on automation runners and have no retention guarantee.</p>"
        "</body></html>"
    )


def _write_delivery_artifacts(
    writer,
    result,
    newsletter_html: str,
    delivery_state: str,
    *,
    checkpoint: bool = False,
):
    """Project and atomically commit the current ledger at a delivery boundary."""
    from tarzan import runtime
    from tarzan.runtime import report_html
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.publication import PublicationEvaluator
    from tarzan.runtime.summary import SummaryProjector

    publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
    result.ledger.append(LedgerEntryType.ARTIFACT, {
        "artifact_set": "local",
        "state": "CHECKPOINT_REQUESTED" if checkpoint else "FINALIZATION_REQUESTED",
        "delivery_state": delivery_state,
        "storage_scope": writer.storage.storage_scope,
        "automation_local_ephemeral": writer.storage.automation_local_ephemeral,
        "retention_guarantee": writer.storage.retention_guarantee,
    })
    summary = SummaryProjector.project(result, publication)
    stamp = runtime.now_stamp("%Y-%m-%d %H:%M")
    rendered_report = report_html.render(
        generated_at=stamp,
        ledger=result.ledger,
        publication_state=publication.decision.value,
        storage_scope=writer.storage.storage_scope,
        automation_local_ephemeral=writer.storage.automation_local_ephemeral,
        retention_guarantee=writer.storage.retention_guarantee,
    )
    arguments = {
        "analysis_id": result.analysis_id,
        "summary": summary.to_dict(),
        "ledger_entries": (entry.to_dict() for entry in result.ledger.entries),
        "report_html": rendered_report,
        "publication_state": publication.decision.value,
        "newsletter_html": newsletter_html,
        "delivery_state": delivery_state,
    }
    if checkpoint:
        return writer.checkpoint(**arguments)
    return writer.finalize(**arguments)


def run_and_send() -> int:
    """Run, evaluate publication, claim durably, checkpoint, and invoke SMTP once."""
    from tarzan.runtime.artifacts import LocalArtifactWriter, StorageDescriptor
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
    from tarzan.runtime.session import last_run_result

    smtp_user = _env("SMTP_USER", required=True)
    smtp_pass = _env("SMTP_PASS", required=True)
    recipient = _env("RECIPIENT_EMAIL", required=True)
    smtp_host = _env("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_env("SMTP_PORT", "465"))
    issue_number = int(_env("ISSUE_NUMBER", "1"))
    subject_prefix = _env("SUBJECT_PREFIX", "Portfolio Digest")
    trigger_label = _env("TRIGGER_LABEL", "")
    dry_run = _env("DRY_RUN", "0") == "1"

    inputs = resolve_inputs()
    logger.info("Tarzan newsletter — trigger=%r, issue=%d", trigger_label, issue_number)
    logger.info(
        "Inputs (order-only) — orders=%s | targets=%s | per_holding=%s",
        inputs["orders"], inputs["config"], inputs["targets_per_holding"] or "(none)",
    )

    previous_result = last_run_result()
    metrics, config = run(
        config_source=inputs["config"],
        orders_source=inputs["orders"],
        targets_per_holding_source=inputs["targets_per_holding"],
    )
    result = last_run_result()
    if result is None or result is previous_result:
        # Compatibility adapter for injected/test orchestration callables. The
        # production orchestrator always records a fresh authoritative
        # RunResult; an unchanged projection belongs to an earlier run.
        from tarzan.runtime.ledger import RunLedger
        from tarzan.runtime.session import RunAttemptEnvelope, RunResult, canonical_analysis_id
        envelope = RunAttemptEnvelope.create("delivery-compatibility")
        result = RunResult(
            metrics=metrics,
            config=config,
            attempt_id=envelope.attempt_id,
            analysis_id=canonical_analysis_id({"summary": metrics.to_summary_dict()}),
            ledger=RunLedger(envelope.attempt_id),
        )

    if metrics.total_value == 0 and not result.ledger.failure_records():
        result.ledger.open_failure(
            stage="valuation",
            stable_code="EMPTY_PORTFOLIO_VALUE",
            severity="CRITICAL",
            error="pipeline produced no portfolio value",
            affected_outputs=["portfolio", "total", "planning", "publication"],
            analytical_impact="portfolio total and planning are unavailable",
            publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
        )

    publication = PublicationEvaluator.evaluate(result.ledger.failure_records())

    semantic_errors: tuple[str, ...] = ()
    if publication.decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE:
        html = _failure_notification_html(result)
        subject = f"{subject_prefix} - Analysis failure"
        logger.critical("Normal newsletter blocked; preparing failure notification.")
    else:
        from tarzan.export.ai_summary import is_enabled as _ai_on
        logger.info(
            "AI summary %s.",
            "enabled (market-context block resolved at render)"
            if _ai_on() else "disabled (no GEMINI_API_KEY / pinned mode)",
        )
        _seed_manual_proxies()
        from tarzan.backtest import newsletter_portfolios
        backtest_portfolios = newsletter_portfolios()
        semantic_audit: dict = {}
        html = render_newsletter(
            metrics=metrics,
            config=config,
            issue_number=issue_number,
            backtest_portfolios=backtest_portfolios,
            semantic_audit=semantic_audit,
        )
        from tarzan.export.newsletter._semantic import validate_newsletter_semantics
        semantic_errors = validate_newsletter_semantics(
            metrics,
            semantic_audit,
            html,
        )
        if semantic_errors:
            result.ledger.open_failure(
                stage="newsletter_semantics",
                stable_code="NEWSLETTER_SEMANTIC_INVARIANT_FAILED",
                severity="CRITICAL",
                error="; ".join(semantic_errors),
                affected_outputs=[
                    "newsletter",
                    "benchmark_tables",
                    "performance_charts",
                    "delivery",
                ],
                analytical_impact=(
                    "rendered market identities or chart labels are inconsistent "
                    "with preprocessed analytical data"
                ),
                publication_impact="BLOCK_NORMAL_AND_DO_NOT_SEND",
            )
            logger.critical(
                "Newsletter semantic gate failed with %d violation(s); claim and "
                "SMTP are blocked.",
                len(semantic_errors),
            )
        subject = build_subject(metrics, subject_prefix, trigger_label)

    # Rendering can add a fail-closed semantic failure. Evaluate and record the
    # single authoritative publication outcome only after that gate completes.
    publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
    result.ledger.append(LedgerEntryType.PUBLICATION, {
        "decision": publication.decision.value,
        "delivery_purpose": publication.delivery_purpose.value,
        "critical_failure_refs": list(publication.critical_failure_ids),
    })

    now = now_local()
    output_root = ROOT / "output" / now.strftime("%Y-%m-%d")
    automated = _env("GITHUB_ACTIONS", "").lower() == "true"
    writer = LocalArtifactWriter(
        output_root,
        result.attempt_id,
        storage=StorageDescriptor(
            storage_scope="local",
            automation_local_ephemeral=automated,
            retention_guarantee="none",
            execution_environment="github-actions" if automated else "email-local",
        ),
    )

    if semantic_errors:
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "SEMANTIC_VALIDATION_FAILED",
            "purpose": publication.delivery_purpose.value,
            "smtp_invoked": False,
            "violations": list(semantic_errors),
        })
        _write_delivery_artifacts(
            writer,
            result,
            html,
            "SEMANTIC_VALIDATION_FAILED",
        )
        return 1

    if dry_run:
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "DRY_RUN",
            "purpose": publication.delivery_purpose.value,
            "smtp_invoked": False,
        })
        _write_delivery_artifacts(writer, result, html, "DRY_RUN")
        logger.warning("DRY_RUN=1 — skipping claim and SMTP send.")
        return 0

    stable_event_id = _env("TARZAN_STABLE_EVENT_ID")
    if not stable_event_id:
        workflow_run = _env("GITHUB_RUN_ID")
        stable_event_id = (
            f"workflow:{workflow_run}" if workflow_run
            else f"manual:{now.date().isoformat()}:issue-{issue_number}:{trigger_label or 'default'}"
        )
    intent = DeliveryIntent(
        stable_event_id=stable_event_id,
        purpose=DeliveryPurpose(publication.delivery_purpose.value),
        recipient_set_digest=recipient_set_digest([recipient]),
        template_schema_version="newsletter-v1",
        authorized_resend_token=_env("AUTHORIZED_RESEND_TOKEN") or None,
    )

    try:
        store = _delivery_claim_store()
        claim = store.claim(intent)
    except Exception as error:  # noqa: BLE001
        result.ledger.open_failure(
            stage="delivery_claim",
            stable_code="DURABLE_CLAIM_UNAVAILABLE",
            severity="CRITICAL",
            error=error,
            affected_outputs=["delivery", "publication"],
            analytical_impact="analysis is complete but email delivery is blocked",
            publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
        )
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "CLAIM_STORE_UNAVAILABLE",
            "purpose": intent.purpose.value,
            "smtp_invoked": False,
        })
        logger.critical(
            "Delivery claim storage unavailable (%s); SMTP is blocked.",
            type(error).__name__,
        )
        _write_delivery_artifacts(writer, result, html, "CLAIM_STORE_UNAVAILABLE")
        return 1

    if claim.conflict:
        result.ledger.open_failure(
            stage="delivery_claim",
            stable_code="DELIVERY_INTENT_CONFLICT",
            severity="CRITICAL",
            error="logical delivery identity has a differing intent digest",
            affected_outputs=["delivery"],
            analytical_impact="SMTP is blocked to avoid sending ambiguous content",
            publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
        )
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "CONFLICT",
            "logical_id": intent.logical_id,
            "smtp_invoked": False,
        })
        _write_delivery_artifacts(writer, result, html, "CONFLICT")
        return 1
    if claim.duplicate:
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": "SUPPRESSED_DUPLICATE",
            "logical_id": intent.logical_id,
            "original_state": claim.state.value,
            "smtp_invoked": False,
        })
        _write_delivery_artifacts(writer, result, html, "SUPPRESSED_DUPLICATE")
        logger.warning(
            "Delivery %s already claimed in state %s; suppressing duplicate SMTP.",
            intent.logical_id,
            claim.state.value,
        )
        return 0

    result.ledger.append(LedgerEntryType.DELIVERY, {
        "state": DeliveryState.CLAIMED.value,
        "logical_id": intent.logical_id,
        "purpose": intent.purpose.value,
        "smtp_invoked": False,
    })
    try:
        _write_delivery_artifacts(
            writer,
            result,
            html,
            DeliveryState.CLAIMED.value,
            checkpoint=True,
        )
    except Exception as error:  # noqa: BLE001
        try:
            store.transition(
                intent.logical_id,
                (DeliveryState.CLAIMED,),
                DeliveryState.DEFINITE_PRE_SEND_FAILURE,
            )
        except Exception:
            pass
        result.ledger.open_failure(
            stage="local_artifacts",
            stable_code="PRE_DELIVERY_CHECKPOINT_FAILED",
            severity="CRITICAL",
            error=error,
            affected_outputs=["manifest", "summary", "ledger", "report", "delivery"],
            analytical_impact="local checkpoint is incomplete; SMTP was not invoked",
            publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
        )
        logger.critical("Pre-delivery local checkpoint failed; SMTP is blocked.")
        return 1

    invocation_started = False

    def mark_invocation_started() -> None:
        nonlocal invocation_started
        store.transition(
            intent.logical_id,
            (DeliveryState.CLAIMED,),
            DeliveryState.SMTP_INVOCATION_STARTED,
        )
        invocation_started = True
        result.ledger.append(LedgerEntryType.DELIVERY, {
            "state": DeliveryState.SMTP_INVOCATION_STARTED.value,
            "logical_id": intent.logical_id,
            "smtp_invoked": True,
        })

    try:
        send_email(
            html=html,
            subject=subject,
            sender=smtp_user,
            recipient=recipient,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_pass=smtp_pass,
            before_invoke=mark_invocation_started,
        )
        # Test/future transports may not expose the callback seam. Marking here
        # is safe only after such a transport has returned definitively.
        if not invocation_started:
            mark_invocation_started()
    except Exception as error:  # noqa: BLE001
        target = (
            DeliveryState.UNCERTAIN
            if invocation_started
            else DeliveryState.DEFINITE_PRE_SEND_FAILURE
        )
        expected = (
            (DeliveryState.SMTP_INVOCATION_STARTED,)
            if invocation_started
            else (DeliveryState.CLAIMED,)
        )
        try:
            store.transition(intent.logical_id, expected, target)
        finally:
            result.ledger.append(LedgerEntryType.DELIVERY, {
                "state": target.value,
                "logical_id": intent.logical_id,
                "smtp_invoked": invocation_started,
                "automatic_retry": False,
            })
            _write_delivery_artifacts(writer, result, html, target.value)
        logger.error("SMTP delivery ended in %s (%s).", target.value, type(error).__name__)
        return 1

    store.transition(
        intent.logical_id,
        (DeliveryState.SMTP_INVOCATION_STARTED,),
        DeliveryState.ACKNOWLEDGED_SUCCESS,
    )
    result.ledger.append(LedgerEntryType.DELIVERY, {
        "state": DeliveryState.ACKNOWLEDGED_SUCCESS.value,
        "logical_id": intent.logical_id,
        "smtp_invoked": True,
    })
    _write_delivery_artifacts(
        writer,
        result,
        html,
        DeliveryState.ACKNOWLEDGED_SUCCESS.value,
    )
    return 0
