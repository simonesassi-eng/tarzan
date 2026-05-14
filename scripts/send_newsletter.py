"""Run the Tarzan pipeline and email the newsletter as inline HTML.

This script is the entry point for both scheduled (cron) runs and
on-demand runs triggered by replying "Update" to a previous newsletter.

Required environment variables (provided by GitHub Actions secrets):
    SMTP_USER       Gmail account that sends the newsletter
    SMTP_PASS       Gmail App Password (NOT the regular account password)
    RECIPIENT_EMAIL Inbox where the newsletter is delivered

Input CSVs are loaded in this priority order:
    1. If GOOGLE_DRIVE_CREDENTIALS_JSON and DRIVE_FOLDER_ID are set, the
       script downloads holdings.csv and targets.csv from the configured
       Drive folder. Use this for public repos so personal data never
       lands in git.
    2. Otherwise, falls back to local files at HOLDINGS_PATH and
       TARGETS_PATH (defaults to .private/*.csv).

Optional:
    SMTP_HOST                       Default smtp.gmail.com
    SMTP_PORT                       Default 465 (SSL)
    HOLDINGS_PATH                   Default .private/holdings.csv
    TARGETS_PATH                    Default .private/targets.csv
    DRIVE_FOLDER_ID                 Drive folder ID (no slashes)
    GOOGLE_DRIVE_CREDENTIALS_JSON   Service-account JSON key
    ISSUE_NUMBER                    Default 1
    SUBJECT_PREFIX                  Default "[Tarzan]"
    DRY_RUN                         If "1", render only, do not send
    TRIGGER_LABEL                   Free-form tag added to the subject
"""

from __future__ import annotations

import logging
import os
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

# Ensure tarzan package is importable when invoked from repo root, plus
# scripts/ itself so we can import drive_loader as a sibling module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tarzan.export.newsletter import render_newsletter  # noqa: E402
from tarzan.orchestrator import run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tarzan.newsletter")


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    """Read an env var, optionally enforcing presence."""
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Set it in GitHub Secrets or your local environment."
        )
    return value or ""


def _build_subject(metrics, prefix: str, trigger_label: str) -> str:
    """Build a concise informative subject line.

    Example: "[Tarzan] morning · €58,790 (+8.59%) · 1 action"
    """
    cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - cost
    gain_pct = (total_gain / cost * 100) if cost > 0 else 0.0
    parts = [prefix]
    if trigger_label:
        parts.append(trigger_label)
    parts.append(f"€{metrics.total_value:,.0f}")
    sign = "+" if gain_pct >= 0 else "−"
    parts.append(f"({sign}{abs(gain_pct):.2f}%)")
    n_actions = len(metrics.rebalancing_suggestions or [])
    if n_actions > 0:
        s = "s" if n_actions != 1 else ""
        parts.append(f"{n_actions} action{s}")
    return " · ".join(parts)


def _send_email(html: str, subject: str, sender: str, recipient: str,
                smtp_host: str, smtp_port: int, smtp_pass: str) -> None:
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
        smtp.send_message(msg)
    logger.info("Sent newsletter to %s with subject: %s", recipient, subject)


def _resolve_inputs() -> tuple[str, str]:
    """Resolve the holdings and targets paths.

    Tries Google Drive first if credentials are present; otherwise falls
    back to local paths. Returns absolute string paths suitable for the
    Tarzan orchestrator.
    """
    drive_folder = _env("DRIVE_FOLDER_ID")
    drive_creds = _env("GOOGLE_DRIVE_CREDENTIALS_JSON")
    if drive_folder and drive_creds:
        from drive_loader import download_inputs  # type: ignore[import-not-found]
        logger.info("Loading inputs from Google Drive folder %s", drive_folder)
        files = download_inputs(folder_id=drive_folder, credentials_json=drive_creds)
        return str(files["holdings.csv"]), str(files["targets.csv"])

    holdings_path = _env("HOLDINGS_PATH", ".private/holdings.csv")
    targets_path = _env("TARGETS_PATH", ".private/targets.csv")
    logger.info("Loading inputs from local paths: %s / %s", holdings_path, targets_path)

    if not Path(holdings_path).exists():
        raise FileNotFoundError(
            f"Holdings file not found at {holdings_path!r}. "
            "Either commit it under .private/ in a private repo, or set "
            "DRIVE_FOLDER_ID and GOOGLE_DRIVE_CREDENTIALS_JSON to load "
            "from Google Drive."
        )
    if not Path(targets_path).exists():
        raise FileNotFoundError(
            f"Targets file not found at {targets_path!r}. "
            "See above for input loading options."
        )
    return holdings_path, targets_path


def main() -> int:
    smtp_user = _env("SMTP_USER", required=True)
    smtp_pass = _env("SMTP_PASS", required=True)
    recipient = _env("RECIPIENT_EMAIL", required=True)
    smtp_host = _env("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_env("SMTP_PORT", "465"))
    issue_number = int(_env("ISSUE_NUMBER", "1"))
    subject_prefix = _env("SUBJECT_PREFIX", "[Tarzan]")
    trigger_label = _env("TRIGGER_LABEL", "")
    dry_run = _env("DRY_RUN", "0") == "1"

    holdings_path, targets_path = _resolve_inputs()

    logger.info("Tarzan newsletter — trigger=%r, issue=%d", trigger_label, issue_number)
    logger.info("Holdings: %s | Targets: %s", holdings_path, targets_path)

    # 1. Run the full pipeline (load → enrich → compute)
    metrics, config = run(holdings_source=holdings_path, config_source=targets_path)
    if metrics.total_value == 0:
        logger.error("Pipeline produced empty metrics. Aborting send.")
        return 1

    # 2. Render newsletter HTML
    html = render_newsletter(
        metrics=metrics,
        config=config,
        issue_number=issue_number,
        benchmark_alpha_beta="S&P 500",
        benchmark_geo="MSCI ACWI",
    )

    subject = _build_subject(metrics, subject_prefix, trigger_label)

    # 3. Optionally write a local copy for traceability (CI artifacts)
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    artifact = output_dir / f"newsletter_{timestamp}.html"
    artifact.write_text(html, encoding="utf-8")
    logger.info("Saved local copy: %s", artifact)

    if dry_run:
        logger.warning("DRY_RUN=1 — skipping SMTP send.")
        return 0

    # 4. Send via SMTP
    _send_email(
        html=html,
        subject=subject,
        sender=smtp_user,
        recipient=recipient,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_pass=smtp_pass,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
