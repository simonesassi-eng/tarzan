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
    """Project legacy issues into compact records with complete evidence."""
    issues = data_quality.issues()
    if not issues:
        return ('<p class="ok">No data-quality issues — every input parsed and '
                "priced cleanly this run.</p>")
    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    records = []
    for ordinal, issue in enumerate(sorted(issues, key=lambda i: order.get(i.severity, 9)), 1):
        raw_id = f"failure-v1|{issue.source}|{issue.context or ''}|{issue.message}|{ordinal}"
        failure_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        # Legacy issues carry no structured remedy or selection evidence.
        # Prose such as "cached" or "using" must never be promoted into an
        # accepted fallback; without a ledger, warnings remain reviewable.
        fallback = "None"
        corrected = False
        needs_review = issue.severity in ("ERROR", "WARNING")
        availability = (
            "Unavailable" if issue.severity == "ERROR"
            else "Degraded" if issue.severity == "WARNING"
            else "Available"
        )
        publication = "Block normal and notify failure" if issue.severity == "ERROR" else "Degrade/disclose"
        analytical_impact = (
            "Affected result requires review" if needs_review
            else "Informational only"
        )
        records.append(
            '<details class="failure-record">'
            '<summary>'
            f'<span class="severity" style="color:{_SEV_COLOR.get(issue.severity, "#334155")}">'
            f'{_esc(issue.severity)}</span>'
            f'<span class="record-title">{_esc(issue.source)}</span>'
            f'<span class="record-hint">{_esc(issue.context or issue.message)}</span>'
            '</summary><div class="record-body">'
            f'<p><b>Failure ID:</b> <span class="mono">{_esc(failure_id)}</span></p>'
            f'<p><b>Stage:</b> {_esc(issue.source)} · <b>Severity:</b> {_esc(issue.severity)}</p>'
            f'<p><b>Original failure:</b> {_esc(issue.message)}</p>'
            '<p><b>Remedies:</b> No successful automatic remedy recorded</p>'
            f'<p><b>Selected fallback:</b> {_esc(fallback)} · <b>Selected resolution:</b> {_esc(fallback)}</p>'
            f'<p><b>Provenance:</b> {_esc(issue.source)} / {_esc(issue.context or "run")}</p>'
            f'<p><b>Automatically corrected:</b> {"Yes" if corrected else "No"} · '
            f'<b>Lifecycle closed:</b> {"No" if needs_review else "Yes"} · '
            f'<b>Needs review:</b> {"Yes" if needs_review else "No"}</p>'
            f'<p><b>Availability:</b> {_esc(availability)}</p>'
            f'<p><b>Affected section:</b> {_esc(issue.source)} · '
            f'<b>Affected sections:</b> {_esc(issue.source)}</p>'
            f'<p><b>Analytical impact:</b> {_esc(analytical_impact)}</p>'
            f'<p><b>Publication impact:</b> {_esc(publication)}</p>'
            '</div></details>'
        )
    return (
        f'<p class="failure-help"><b>Failure summary:</b> {len(records)} record(s). '
        'Expand a record for its complete lifecycle.</p>'
        '<div class="failure-list">' + "".join(records) + '</div>'
    )


def _failure_family(failure) -> str:
    """Stable grouping label with per-instrument identity removed."""
    original = failure.original_failure
    if hasattr(original, "get"):
        message = original.get("message")
        if message:
            family = str(message)
            for key in ("context", "instrument", "isin", "ticker"):
                identity = original.get(key)
                if identity not in (None, ""):
                    family = family.replace(str(identity), "{identifier}")
            return " ".join(family.split())
        descriptors = [
            f"{key}={original[key]}"
            for key in sorted(original)
            if key not in {"context", "instrument", "isin", "ticker"}
        ]
        if descriptors:
            return ", ".join(descriptors)
    return str(failure.stable_code or "Failure")


def _clip(value, limit: int = 180) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _failure_hint(failure) -> str:
    original = failure.original_failure
    if hasattr(original, "get"):
        for key in ("context", "instrument", "isin", "ticker"):
            value = original.get(key)
            if value not in (None, ""):
                return _clip(value, 72)
    return _clip(failure.selected_resolution or "details", 72)


def _failure_detail_html(failure) -> str:
    remedy_rows = []
    for remedy in failure.remedies:
        remedy_rows.append(
            '<li>'
            f"#{_esc(remedy.get('ordinal'))} "
            f"<span class='mono'>{_esc(remedy.get('remedy_id'))}</span>: "
            f"{_esc(remedy.get('action'))} → {_esc(remedy.get('outcome'))} "
            f"({_esc(remedy.get('availability'))})"
            '</li>'
        )
    remedies = (
        '<ol class="remedies">' + "".join(remedy_rows) + '</ol>'
        if remedy_rows else "No remedy completed"
    )
    original = json.dumps(
        failure.original_failure,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    selected = failure.selected_resolution or "None"
    affected = ", ".join(failure.affected_outputs) or "None"
    return (
        '<details class="failure-record">'
        '<summary>'
        f'<span class="severity" style="color:{_SEV_COLOR.get(failure.severity, "#334155")}">'
        f'{_esc(failure.severity)}</span>'
        f'<span class="record-title mono">Failure ID: {_esc(failure.failure_id)}</span>'
        f'<span class="record-hint">{_esc(_failure_hint(failure))}</span>'
        '</summary><div class="record-body">'
        f'<p><b>Failure ID:</b> <span class="mono">{_esc(failure.failure_id)}</span></p>'
        f'<p><b>Stage:</b> {_esc(failure.stage)} · '
        f'<b>Code:</b> {_esc(failure.stable_code)} · '
        f'<b>Severity:</b> {_esc(failure.severity)}</p>'
        f'<p><b>Original failure:</b> <span class="payload">{_esc(original)}</span></p>'
        f'<div class="record-row"><b>Remedies:</b> {remedies}</div>'
        f'<p><b>Selected fallback:</b> {_esc(selected)} · '
        f'<b>Selected resolution:</b> {_esc(selected)}</p>'
        f'<p><b>Provenance:</b> {_esc(", ".join(failure.provenance) or "None")}</p>'
        f'<p><b>Automatically corrected:</b> {"Yes" if failure.automatically_corrected else "No"} · '
        f'<b>Lifecycle closed:</b> {"Yes" if failure.closed else "No"}</p>'
        f'<p><b>Availability:</b> {_esc(failure.availability.value.title())}</p>'
        f'<p><b>Affected section:</b> {_esc(affected)} · '
        f'<b>Affected sections:</b> {_esc(affected)}</p>'
        f'<p><b>Analytical impact:</b> {_esc(failure.analytical_impact)}</p>'
        f'<p><b>Publication impact:</b> {_esc(failure.publication_impact)}</p>'
        '</div></details>'
    )


def _ledger_issues_html(ledger) -> str:
    """Group lifecycle evidence and separate explicit acceptance from review."""
    failures = tuple(ledger.failure_records()) if ledger is not None else ()
    if not failures:
        return _issues_html()

    def _accepted_degraded(item) -> bool:
        return (
            item.closed
            and item.availability.value == "DEGRADED"
            and not item.automatically_corrected
            and bool(item.selected_resolution)
        )

    def _needs_review(item) -> bool:
        return (
            not item.closed
            or item.availability.value == "UNAVAILABLE"
            or (
                not item.automatically_corrected
                and not _accepted_degraded(item)
            )
        )

    corrected = sum(
        1 for item in failures
        if item.closed and item.automatically_corrected
    )
    accepted_degraded = sum(
        1 for item in failures if _accepted_degraded(item)
    )
    unavailable = sum(
        1 for item in failures
        if item.availability.value == "UNAVAILABLE"
    )
    needs_review = sum(1 for item in failures if _needs_review(item))

    groups: dict[tuple[str, str, str], list] = {}
    for failure in failures:
        key = (failure.stage, failure.stable_code, _failure_family(failure))
        groups.setdefault(key, []).append(failure)

    severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}

    def _group_sort(item):
        (stage, code, family), members = item
        rank = min(severity_order.get(member.severity, 9) for member in members)
        return rank, stage, code, family

    group_cards = []
    for (stage, code, family), members in sorted(groups.items(), key=_group_sort):
        severity = min(
            (member.severity for member in members),
            key=lambda value: severity_order.get(value, 9),
        )
        availabilities = sorted({member.availability.value.title() for member in members})
        corrected_in_group = sum(
            1 for member in members
            if member.closed and member.automatically_corrected
        )
        accepted_in_group = sum(
            1 for member in members if _accepted_degraded(member)
        )
        review_in_group = sum(1 for member in members if _needs_review(member))
        open_attr = " open" if review_in_group else ""
        group_cards.append(
            f'<details class="failure-group"{open_attr}>'
            '<summary>'
            f'<span class="severity" style="color:{_SEV_COLOR.get(severity, "#334155")}">'
            f'{_esc(severity)}</span>'
            '<span class="group-copy">'
            f'<span class="group-title">{_esc(stage)} · <span class="mono">{_esc(code)}</span></span>'
            f'<span class="group-family">{_esc(_clip(family))}</span>'
            '</span>'
            f'<span class="group-count">{len(members)} record{"s" if len(members) != 1 else ""}</span>'
            '</summary>'
            '<div class="group-body">'
            f'<p class="group-meta">Availability: {_esc(", ".join(availabilities))} · '
            f'{corrected_in_group} auto-resolved · {accepted_in_group} policy-accepted degraded · '
            f'{review_in_group} need review. Expand a record for complete identity, remedies and impacts.</p>'
            + "".join(_failure_detail_html(member) for member in members)
            + '</div></details>'
        )

    return (
        '<div class="metric-grid">'
        f'<div class="metric"><strong>{len(failures)}</strong><span>Lifecycle records</span></div>'
        f'<div class="metric"><strong>{len(groups)}</strong><span>Families</span></div>'
        f'<div class="metric good"><strong>{corrected}</strong><span>Auto-resolved</span></div>'
        f'<div class="metric warn"><strong>{accepted_degraded}</strong><span>Accepted degraded</span></div>'
        f'<div class="metric bad"><strong>{needs_review}</strong><span>Needs review</span></div>'
        '</div>'
        f'<p class="failure-help"><b>Lifecycle summary:</b> {len(failures)} records grouped into '
        f'{len(groups)} families; {unavailable} unavailable. Only fallbacks with an explicit '
        'selected resolution are policy-accepted; other uncorrected warnings remain reviewable.</p>'
        '<div class="failure-groups">' + "".join(group_cards) + '</div>'
    )


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
        '<div class="table-wrap"><table class="tbl log"><thead><tr>'
        "<th style='width:7%'>Level</th><th style='width:9%'>Time</th>"
        "<th style='width:16%'>Origin</th><th>Message</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light;}*{box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  color:#1E293B;background:#F8FAFC;margin:0;padding:24px;font-size:13px;line-height:1.45;}
body>header,body>main,body>footer{max-width:1180px;margin-left:auto;margin-right:auto;}
h1{font-size:20px;margin:0 0 2px;letter-spacing:-.01em;}h2{font-size:14px;margin:22px 0 8px;
  color:#334155;}.sub{color:#64748B;font-size:12px;margin:0;}.ok{color:#15803D;font-weight:600;
  background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:10px 12px;}
.muted{color:#64748B;font-size:12px;}.note{color:#64748B;font-size:12px;background:#FFF;
  border:1px solid #E2E8F0;border-left:3px solid #CBD5E1;border-radius:6px;padding:7px 10px;margin:8px 0;}
.status{padding:9px 12px;background:#FFF;border:1px solid #E2E8F0;border-left:4px solid #475569;
  border-radius:7px;font-weight:700;margin:12px 0 18px;}.mono,.org{font-family:ui-monospace,
  SFMono-Regular,Menlo,Consolas,monospace;color:#64748B;font-size:11.5px;}.severity{font-weight:800;
  white-space:nowrap;font-size:10px;letter-spacing:.04em;}.metric-grid{display:grid;
  grid-template-columns:repeat(5,minmax(100px,1fr));gap:8px;margin:8px 0;}.metric{display:flex;
  align-items:baseline;gap:7px;background:#FFF;border:1px solid #E2E8F0;border-radius:8px;padding:9px 11px;}
.metric strong{font-size:18px;color:#334155;}.metric span{color:#64748B;font-size:11px;}.metric.good strong{color:#15803D;}
.metric.warn strong{color:#B45309;}.metric.bad strong{color:#DC2626;}.failure-help{color:#64748B;
  font-size:12px;margin:8px 1px;}.failure-group{background:#FFF;border:1px solid #E2E8F0;border-radius:9px;
  margin:7px 0;overflow:hidden;}.failure-group>summary,.failure-record>summary,.run-log>summary{list-style:none;
  cursor:pointer;}.failure-group>summary::-webkit-details-marker,.failure-record>summary::-webkit-details-marker,
.run-log>summary::-webkit-details-marker{display:none;}.failure-group>summary{display:flex;align-items:center;gap:10px;
  padding:10px 12px;}.failure-group>summary:before,.run-log>summary:before{content:"›";color:#94A3B8;
  font-size:18px;line-height:1;transition:transform .1s;}.failure-group[open]>summary:before,
.run-log[open]>summary:before{transform:rotate(90deg);}.group-copy{display:flex;flex:1;min-width:0;
  flex-direction:column;gap:1px;}.group-title{font-weight:700;color:#334155;}.group-family{font-size:12px;
  color:#64748B;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}.group-count{color:#475569;
  background:#F1F5F9;border-radius:999px;padding:2px 8px;font-size:11px;white-space:nowrap;}.group-body{
  border-top:1px solid #E2E8F0;background:#F8FAFC;padding:8px 10px 10px;}.group-meta{margin:0 2px 7px;
  color:#64748B;font-size:11px;}.failure-record{background:#FFF;border:1px solid #E2E8F0;border-radius:6px;
  margin:5px 0;}.failure-record>summary{display:flex;align-items:center;gap:9px;padding:7px 9px;}.failure-record>summary:before{
  content:"+";color:#94A3B8;font-weight:700;}.failure-record[open]>summary:before{content:"−";}.record-title{
  color:#334155;font-weight:600;}.record-hint{margin-left:auto;color:#94A3B8;font-size:11px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;max-width:45%;}.record-body{border-top:1px solid #F1F5F9;padding:7px 10px 9px;
  color:#475569;font-size:11.5px;}.record-body p,.record-row{margin:3px 0;}.payload{font-family:ui-monospace,
  SFMono-Regular,Menlo,Consolas,monospace;word-break:break-word;}.remedies{display:inline-block;margin:0 0 0 20px;
  padding:0;vertical-align:top;}.run-log{max-width:1180px;margin:22px auto 0;background:#FFF;border:1px solid #E2E8F0;
  border-radius:9px;overflow:hidden;}.run-log>summary{display:flex;align-items:center;gap:8px;padding:10px 12px;
  font-weight:700;color:#334155;}.run-log .summary-meta{margin-left:auto;font-weight:400;color:#94A3B8;font-size:11px;}
.log-body{border-top:1px solid #E2E8F0;padding:8px 10px 10px;}.table-wrap{overflow:auto;max-height:70vh;}
table.tbl{border-collapse:collapse;width:100%;font-size:12px;}.tbl th{text-align:left;background:#F1F5F9;
  color:#475569;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.03em;padding:5px 7px;
  border-bottom:1px solid #E2E8F0;}.tbl td{padding:4px 7px;border-bottom:1px solid #F1F5F9;vertical-align:top;}
.sev,.lvl{font-weight:700;white-space:nowrap;}.org{white-space:nowrap;}.msg{font-family:ui-monospace,
  SFMono-Regular,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;}.ctx{color:#94A3B8;
  white-space:nowrap;}.log th{position:sticky;top:0;z-index:1;}footer{color:#94A3B8;font-size:11px;margin-top:16px;}
@media(max-width:720px){body{padding:14px;}.metric-grid{grid-template-columns:repeat(2,minmax(100px,1fr));}
  .record-hint{display:none;}.group-count{font-size:10px;}.failure-group>summary{gap:7px;}}
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
    kept_count = sum(1 for record in records if _keep(record))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header><h1>Tarzan — Run Report</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)}.</p></header>"
        "<main>"
        f"<div class='status'>Publication status: {_esc(state)}</div>"
        "<section><h2>Issues &amp; how they were handled</h2>"
        f"{_ledger_issues_html(ledger)}"
        f"{_thirdparty_note(records)}</section>"
        "</main>"
        "<details class='run-log'>"
        "<summary><span>Run log</span>"
        f"<span class='summary-meta'>{kept_count} relevant of {len(records)} entries</span>"
        "</summary><div class='log-body'>"
        f"{_log_table_html(records)}"
        "</div></details>"
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
