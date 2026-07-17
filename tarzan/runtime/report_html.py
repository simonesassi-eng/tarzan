"""The single run log — one compact, self-contained HTML file.

``output/report.html`` is the ONE log Tarzan writes per run (there is no
separate analyzer.log). It has two parts:

  1. A top **summary** of what actually needed a human's attention this run —
     Tarzan's own data-quality events (skips / coercions / fallbacks) WITH how
     each was handled, plus a short note explaining any third-party
     (yfinance) probe chatter so its red log lines aren't mistaken for real
     Tarzan errors.
  2. The **run log** itself, as a lean table. To keep the file small, only
     Tarzan's own records and any WARNING/ERROR from any source are shown;
     third-party DEBUG/INFO chatter (yfinance/peewee/urllib3 — the vast bulk
     of raw log volume) is dropped.

Colour is used sparingly: only the level word is tinted; rows are plain.
Best-effort — rendering/writing must never raise into the pipeline. Inline
CSS only, no JS, so it opens anywhere offline.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import tempfile
from typing import Optional

from tarzan.runtime import data_quality

logger = logging.getLogger(__name__)

# Sparse level tint — applied only to the level word, not whole rows.
_LEVEL_COLOR = {
    "CRITICAL": "#B91C1C",
    "ERROR": "#DC2626",
    "WARNING": "#B45309",
    "INFO": "#334155",
    "DEBUG": "#64748B",
}
_SEV_COLOR = {"ERROR": "#DC2626", "WARNING": "#B45309", "INFO": "#334155"}


def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def _color(level: str) -> str:
    return _LEVEL_COLOR.get((level or "").upper(), "#334155")


# ---------------------------------------------------------------------------
# Log filtering (keep the file small + relevant)
# ---------------------------------------------------------------------------

def _keep(rec: dict) -> bool:
    """Keep Tarzan's own records and any WARNING/ERROR from any source; drop
    third-party DEBUG/INFO noise (yfinance/peewee/urllib3)."""
    origin = (rec.get("origin") or "")
    level = (rec.get("level") or "").upper()
    if origin == "tarzan" or origin.startswith("tarzan."):
        return True
    return level in ("WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# Top summary: real issues + how handled
# ---------------------------------------------------------------------------

def _issues_html() -> str:
    """Project every legacy issue into a complete, stable failure record."""
    issues = data_quality.issues()
    if not issues:
        return ('<p class="ok">No data-quality issues — every input parsed and '
                "priced cleanly this run.</p>")
    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    records = []
    for ordinal, issue in enumerate(sorted(issues, key=lambda i: order.get(i.severity, 9)), 1):
        raw_id = f"failure-v1|{issue.source}|{issue.context or ''}|{issue.message}|{ordinal}"
        failure_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        lower = issue.message.lower()
        fallback = issue.message if any(word in lower for word in ("fallback", "selected", "cached", "using")) else "None"
        corrected = fallback != "None" and issue.severity != "ERROR"
        availability = "Degraded" if corrected or issue.severity == "WARNING" else (
            "Unavailable" if issue.severity == "ERROR" else "Available"
        )
        publication = "Block normal and notify failure" if issue.severity == "ERROR" else "Degrade/disclose"
        records.append(
            '<section class="failure-record">'
            f'<h3>Failure ID: <span class="mono">{_esc(failure_id)}</span></h3>'
            f'<p><b>Stage:</b> {_esc(issue.source)} · <b>Severity:</b> {_esc(issue.severity)}</p>'
            f'<p><b>Original failure:</b> {_esc(issue.message)}</p>'
            f'<p><b>Remedies:</b> {_esc(issue.message if corrected else "No successful automatic remedy recorded")}</p>'
            f'<p><b>Selected fallback:</b> {_esc(fallback)}</p>'
            f'<p><b>Provenance:</b> {_esc(issue.source)} / {_esc(issue.context or "run")}</p>'
            f'<p><b>Automatically corrected:</b> {"Yes" if corrected else "No"}</p>'
            f'<p><b>Availability:</b> {_esc(availability)}</p>'
            f'<p><b>Affected section:</b> {_esc(issue.source)}</p>'
            f'<p><b>Analytical impact:</b> {_esc("Result may use fallback evidence" if corrected else "Affected result is not trustworthy")}</p>'
            f'<p><b>Publication impact:</b> {_esc(publication)}</p>'
            '</section>'
        )
    return "".join(records)


def _ledger_issues_html(ledger) -> str:
    """Render the authoritative append-only failure projection when present."""
    failures = tuple(ledger.failure_records()) if ledger is not None else ()
    if not failures:
        return _issues_html()

    corrected = sum(1 for item in failures if item.automatically_corrected)
    unavailable = sum(1 for item in failures if item.availability.value == "UNAVAILABLE")
    uncorrected = sum(1 for item in failures if not item.automatically_corrected)
    records = [
        '<p class="note"><b>Failure summary:</b> '
        f'{len(failures)} total · {corrected} automatically corrected · '
        f'{unavailable} unavailable · {uncorrected} uncorrected.</p>'
    ]
    for failure in failures:
        remedy_rows = []
        for remedy in failure.remedies:
            remedy_rows.append(
                f"#{_esc(remedy.get('ordinal'))} "
                f"{_esc(remedy.get('remedy_id'))}: "
                f"{_esc(remedy.get('action'))} → {_esc(remedy.get('outcome'))} "
                f"({_esc(remedy.get('availability'))})"
            )
        remedies = "<br>".join(remedy_rows) if remedy_rows else "No remedy completed"
        original = json.dumps(
            failure.original_failure,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        records.append(
            '<section class="failure-record">'
            f'<h3>Failure ID: <span class="mono">{_esc(failure.failure_id)}</span></h3>'
            f'<p><b>Stage:</b> {_esc(failure.stage)} · '
            f'<b>Code:</b> {_esc(failure.stable_code)} · '
            f'<b>Severity:</b> {_esc(failure.severity)}</p>'
            f'<p><b>Original failure:</b> {_esc(original)}</p>'
            f'<p><b>Remedies:</b> {remedies}</p>'
            f'<p><b>Selected resolution:</b> {_esc(failure.selected_resolution or "None")}</p>'
            f'<p><b>Provenance:</b> {_esc(", ".join(failure.provenance) or "None")}</p>'
            f'<p><b>Automatically corrected:</b> {"Yes" if failure.automatically_corrected else "No"}</p>'
            f'<p><b>Lifecycle closed:</b> {"Yes" if failure.closed else "No"}</p>'
            f'<p><b>Availability:</b> {_esc(failure.availability.value.title())}</p>'
            f'<p><b>Affected sections:</b> {_esc(", ".join(failure.affected_outputs) or "None")}</p>'
            f'<p><b>Analytical impact:</b> {_esc(failure.analytical_impact)}</p>'
            f'<p><b>Publication impact:</b> {_esc(failure.publication_impact)}</p>'
            '</section>'
        )
    return "".join(records)


def _thirdparty_note(records) -> str:
    """Explain third-party probe chatter so its red lines don't alarm: count
    the yfinance 'possibly delisted' probe messages (expected — the enricher
    tries exchange-suffix variants and uses the first that resolves)."""
    probe = [
        r for r in records or []
        if (r.get("origin") or "").startswith("yfinance")
        and (r.get("level") or "").upper() in ("ERROR", "WARNING")
    ]
    if not probe:
        return ""
    return (
        f'<p class="note">Note: yfinance emitted {len(probe)} expected '
        "ticker-probe message(s) while resolving instruments (it tries "
        "exchange-suffix variants — .DE / .F / .AS / .PA — and the enricher "
        "uses the first that resolves). These are <b>not</b> Tarzan errors and "
        "were handled automatically; they appear in the log below for "
        "completeness.</p>"
    )


# ---------------------------------------------------------------------------
# Log table
# ---------------------------------------------------------------------------

def _log_table_html(records) -> str:
    kept = [r for r in (records or []) if _keep(r)]
    if not kept:
        return '<p class="muted">No log entries.</p>'
    rows = "".join(
        f"<tr>"
        f'<td class="lvl" style="color:{_color(r.get("level"))}">{_esc((r.get("level") or "").upper())}</td>'
        f'<td class="mono">{_esc(r.get("time"))}</td>'
        f'<td class="org">{_esc(r.get("origin"))}</td>'
        f'<td class="msg">{_esc(r.get("message"))}</td>'
        "</tr>"
        for r in kept
    )
    return (
        f'<p class="muted">{len(kept)} of {len(records)} log entries shown '
        "(Tarzan records + any warning/error; third-party debug noise "
        "hidden).</p>"
        '<table class="tbl log"><thead><tr>'
        "<th style='width:7%'>Level</th><th style='width:9%'>Time</th>"
        "<th style='width:16%'>Origin</th><th>Message</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1E293B;
  margin:0;padding:18px 22px;font-size:13px;line-height:1.45;}
h1{font-size:18px;margin:0 0 2px;} h2{font-size:14px;margin:22px 0 8px;
  color:#334155;} .sub{color:#64748B;font-size:12px;margin:0 0 4px;}
.ok{color:#15803D;font-weight:600;} .muted{color:#94A3B8;font-size:12px;}
.note{color:#64748B;font-size:12px;background:#F8FAFC;border-left:3px solid #CBD5E1;
  padding:6px 10px;margin:8px 0;}
table.tbl{border-collapse:collapse;width:100%;font-size:12.5px;}
.tbl th{text-align:left;background:#F1F5F9;color:#475569;font-weight:600;
  font-size:11px;text-transform:uppercase;letter-spacing:.03em;padding:5px 8px;
  border-bottom:1px solid #E2E8F0;}
.tbl td{padding:4px 8px;border-bottom:1px solid #F1F5F9;vertical-align:top;}
.sev,.lvl{font-weight:700;white-space:nowrap;}
.mono,.org{white-space:nowrap;color:#64748B;font-size:11.5px;}
.msg{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;
  word-break:break-word;}
.ctx{color:#94A3B8;white-space:nowrap;}
.log th{position:sticky;top:0;}
footer{color:#94A3B8;font-size:11px;margin-top:16px;}
.status{padding:8px 10px;background:#F8FAFC;border-left:4px solid #475569;
  font-weight:700;margin:8px 0 12px;}
.failure-record{border:1px solid #E2E8F0;border-radius:4px;padding:8px 10px;
  margin:8px 0;background:#FFF;}.failure-record h3{font-size:12px;margin:0 0 5px;}
.failure-record p{margin:3px 0;}
"""


def render(
    generated_at: str,
    log_records: Optional[list] = None,
    *,
    ledger=None,
    publication_state: Optional[str] = None,
    storage_scope: str = "local",
    automation_local_ephemeral: bool = False,
    retention_guarantee: str = "none",
) -> str:
    """Render the complete ledger-backed run report as one HTML string."""
    records = log_records or []
    state = publication_state or "NOT_EVALUATED"
    storage_note = (
        f"storage_scope={storage_scope}; "
        f"automation_local_ephemeral={str(automation_local_ephemeral).lower()}; "
        f"retention_guarantee={retention_guarantee}"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Tarzan — Run Report</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)}.</p>"
        f"<div class='status'>Publication status: {_esc(state)}</div>"
        "<h2>Issues &amp; how they were handled</h2>"
        f"{_ledger_issues_html(ledger)}"
        f"{_thirdparty_note(records)}"
        "<h2>Run log</h2>"
        f"{_log_table_html(records)}"
        f"<footer>Tarzan run report · {_esc(storage_note)}.</footer>"
        "</body></html>"
    )


def write_report(output_dir: str, generated_at: str,
                 log_records: Optional[list] = None,
                 filename: str = "report.html") -> Optional[str]:
    """Atomically publish the report; retain the last committed file on error."""
    temporary = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=output_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(render(generated_at, log_records))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    except Exception as e:  # noqa: BLE001
        logger.error("Run report write failed: %s", e)
        return None
    finally:
        if temporary and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
