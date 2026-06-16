"""Run the Tarzan pipeline and email the newsletter as inline HTML.

This script is the entry point for every newsletter send. It is invoked
by GitHub Actions, which acts purely as a runner: all scheduling lives
in the Gmail Apps Script, which fires a `repository_dispatch` (event
type ``send_now``) at each market slot and for on-demand "Update"
replies. A manual ``workflow_dispatch`` run also lands here.

Required environment variables (provided by GitHub Actions secrets):
    SMTP_USER       Gmail account that sends the newsletter
    SMTP_PASS       Gmail App Password (NOT the regular account password)
    RECIPIENT_EMAIL Inbox where the newsletter is delivered

Input CSVs are loaded in this priority order:
    1. If GOOGLE_DRIVE_CREDENTIALS_JSON and DRIVE_FOLDER_ID are set, the
       script downloads the known input files from the Drive folder:
       order_list.csv (the order list that drives the report), targets.csv
       (config) and the optional targets_per_holding.csv. Use this for
       public repos so personal data never lands in git.
    2. Otherwise, falls back to local files at ORDERS_PATH / TARGETS_PATH /
       TARGETS_PER_HOLDING_PATH (default .private/*.csv).

The order list is the single source of truth: the snapshot (positions,
valuation, allocations) is derived from it, and it also drives the
historical value series and XIRR/TWROR.

Optional:
    SMTP_HOST                       Default smtp.gmail.com
    SMTP_PORT                       Default 465 (SSL)
    ORDERS_PATH                     Default .private/order_list.csv
    TARGETS_PATH                    Default .private/targets.csv
    TARGETS_PER_HOLDING_PATH        Default .private/targets_per_holding.csv
    DRIVE_FOLDER_ID                 Drive folder ID (no slashes)
    GOOGLE_DRIVE_CREDENTIALS_JSON   Service-account JSON key
    ISSUE_NUMBER                    Default 1
    SUBJECT_PREFIX                  Default "Tarzan Portfolio Digest"
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
from zoneinfo import ZoneInfo

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


def _now_local() -> datetime:
    """Return the current time in Europe/Rome.

    GitHub Actions runners are in UTC, so calling ``datetime.now()``
    directly produces times that are 1–2 hours behind Italian local
    time depending on DST. The whole pipeline (subject line, output
    filenames, run logs the user reads) makes more sense in the
    user's wall-clock time, not the runner's.
    """
    return datetime.now(ZoneInfo("Europe/Rome"))


def _build_subject(metrics, prefix: str, trigger_label: str) -> str:
    """Build the newsletter subject line.

    Example: "Tarzan Portfolio Digest · 15/05/2026 18:42 · RTD +8.59%"
    """
    cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - cost
    gain_pct = (total_gain / cost * 100) if cost > 0 else 0.0
    generated_at = _now_local().strftime("%d/%m/%Y %H:%M")
    sign = "+" if gain_pct >= 0 else "−"

    parts = [prefix or "Tarzan Portfolio Digest", generated_at, f"RTD {sign}{abs(gain_pct):.2f}%"]
    if trigger_label:
        parts.append(trigger_label)
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


def _resolve_inputs() -> dict[str, str | None]:
    """Resolve the pipeline input paths.

    Returns a dict with keys ``config``, ``orders`` and
    ``targets_per_holding`` (values are absolute paths or None). The order
    list is the single source of truth — the snapshot is derived from it.

    Drive mode (credentials present) downloads the known input files that
    exist in the folder; it requires ``order_list.csv`` and ``targets.csv``.
    Local mode mirrors this via the *_PATH env vars.
    """
    drive_folder = _env("DRIVE_FOLDER_ID")
    drive_creds = _env("GOOGLE_DRIVE_CREDENTIALS_JSON")
    if drive_folder and drive_creds:
        from drive_loader import KNOWN_INPUT_FILES, download_files  # type: ignore[import-not-found]
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


def main() -> int:
    smtp_user = _env("SMTP_USER", required=True)
    smtp_pass = _env("SMTP_PASS", required=True)
    recipient = _env("RECIPIENT_EMAIL", required=True)
    smtp_host = _env("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_env("SMTP_PORT", "465"))
    issue_number = int(_env("ISSUE_NUMBER", "1"))
    subject_prefix = _env("SUBJECT_PREFIX", "Tarzan Portfolio Digest")
    trigger_label = _env("TRIGGER_LABEL", "")
    dry_run = _env("DRY_RUN", "0") == "1"

    inputs = _resolve_inputs()

    logger.info("Tarzan newsletter — trigger=%r, issue=%d", trigger_label, issue_number)
    logger.info(
        "Inputs (order-only) — orders=%s | targets=%s | per_holding=%s",
        inputs["orders"], inputs["config"], inputs["targets_per_holding"] or "(none)",
    )

    # 1. Run the full pipeline (load → enrich → compute). The order list is
    #    the single source of truth: the snapshot is derived from it and it
    #    drives the historical series + XIRR/TWROR.
    metrics, config = run(
        config_source=inputs["config"],
        orders_source=inputs["orders"],
        targets_per_holding_source=inputs["targets_per_holding"],
    )
    if metrics.total_value == 0:
        logger.error("Pipeline produced empty metrics. Aborting send.")
        return 1

    # 2. Render newsletter HTML.
    # The α/β and geo benchmark names are read from configuration
    # (indexes.csv: is_benchmark_alfa_and_beta / is_benchmark_geo_allocation)
    # rather than hardcoded, so the labels and the cells they color match
    # the benchmark the engine actually computed against.
    from tarzan import config as tarzan_config

    benchmark_alpha_beta = tarzan_config.benchmark_beta_name()
    benchmark_geo = tarzan_config.benchmark_geo_allocation()
    logger.info(
        "Benchmarks — α/β: %s | geo: %s", benchmark_alpha_beta, benchmark_geo
    )

    # Optional AI portfolio summary (free Gemini tier). Best-effort: when no
    # GEMINI_API_KEY is set, or the call fails, this returns None and the
    # newsletter falls back to the rule-based Signals block.
    from tarzan.export.ai_summary import generate_summary, is_enabled as _ai_on
    ai_summary = None
    if _ai_on():
        logger.info("Generating AI portfolio summary...")
        ai_summary = generate_summary(metrics, config)
        logger.info("AI summary %s", "generated" if ai_summary else "unavailable (using Signals)")
    else:
        logger.info("AI summary disabled (no GEMINI_API_KEY) — using Signals.")

    html = render_newsletter(
        metrics=metrics,
        config=config,
        issue_number=issue_number,
        benchmark_alpha_beta=benchmark_alpha_beta,
        benchmark_geo=benchmark_geo,
        ai_summary=ai_summary,
    )

    subject = _build_subject(metrics, subject_prefix, trigger_label)

    # 3. Optionally write a local copy for traceability (CI artifacts)
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    timestamp = _now_local().strftime("%Y%m%d_%H%M")
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
