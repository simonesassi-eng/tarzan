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

import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from zoneinfo import ZoneInfo

from tarzan.export.newsletter import render_newsletter
from tarzan.orchestrator import run

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


def send_email(html: str, subject: str, sender: str, recipient: str,
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
                for base in (ROOT / "tbtf-analisi", ROOT / ".private"):
                    cand = base / fn
                    if cand.exists():
                        path = cand
                        break
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


def run_and_send() -> int:
    """Run the full pipeline, render the newsletter, and email it.

    Reads its configuration from the environment (see module docstring).
    Returns a process exit code: 0 on success (or DRY_RUN), 1 on empty metrics.
    """
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

    # 1. Run the full pipeline (load → enrich → compute). The order list is the
    #    single source of truth: the snapshot is derived from it and it drives
    #    the historical series + XIRR/TWROR.
    metrics, config = run(
        config_source=inputs["config"],
        orders_source=inputs["orders"],
        targets_per_holding_source=inputs["targets_per_holding"],
    )
    if metrics.total_value == 0:
        logger.error("Pipeline produced empty metrics. Aborting send.")
        return 1

    # 2. Render newsletter HTML.
    # render_newsletter is the single render path (CLI + email produce the same
    # HTML): it resolves the α/β and geo benchmark names from configuration, and
    # the optional AI market-context summary, internally when not passed. Leaving
    # them as their defaults here keeps this and the CLI byte-identical.
    from tarzan.export.ai_summary import is_enabled as _ai_on
    logger.info("AI summary %s.",
                "enabled (market-context block resolved at render)"
                if _ai_on() else "disabled (no GEMINI_API_KEY / deterministic)")

    # Long-history backtest of the candidate portfolios, computed once and
    # rendered into the newsletter's "Backtesting" section. Guarded: a failure
    # (or a missing weights file) simply omits the section, never blocks the send.
    # Seed the carry/CTA manual proxies from the private Drive first (idempotent)
    # so the section models those sleeves with the real indices in CI too.
    _seed_manual_proxies()
    from tarzan.backtest import newsletter_portfolios
    backtest_portfolios = newsletter_portfolios()

    html = render_newsletter(
        metrics=metrics,
        config=config,
        issue_number=issue_number,
        backtest_portfolios=backtest_portfolios,
    )

    subject = build_subject(metrics, subject_prefix, trigger_label)

    # 3. Optionally write a local copy for traceability (CI artifacts). Group
    #    per run date (output/<YYYY-MM-DD>/) so copies don't pile up flat,
    #    matching the CLI's output layout.
    now = now_local()
    output_dir = ROOT / "output" / now.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M")
    artifact = output_dir / f"newsletter_{timestamp}.html"
    artifact.write_text(html, encoding="utf-8")
    logger.info("Saved local copy: %s", artifact)

    if dry_run:
        logger.warning("DRY_RUN=1 — skipping SMTP send.")
        return 0

    # 4. Send via SMTP.
    send_email(
        html=html,
        subject=subject,
        sender=smtp_user,
        recipient=recipient,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_pass=smtp_pass,
    )
    return 0
