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

import html
import logging
import os
from typing import Optional

from tarzan import data_quality

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
    """Tarzan data-quality issues, each with the actionable message that says
    how it was handled (the messages already carry the resolution)."""
    issues = data_quality.issues()
    if not issues:
        return ('<p class="ok">No data-quality issues — every input parsed and '
                "priced cleanly this run.</p>")
    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    rows = "".join(
        f"<tr>"
        f'<td class="sev" style="color:{_SEV_COLOR.get(i.severity, "#334155")}">{_esc(i.severity)}</td>'
        f"<td>{_esc(i.source)}</td>"
        f"<td>{_esc(i.message)}</td>"
        f'<td class="ctx">{_esc(i.context)}</td>'
        "</tr>"
        for i in sorted(issues, key=lambda i: order.get(i.severity, 9))
    )
    return (
        '<table class="tbl"><thead><tr><th>Severity</th><th>Where</th>'
        "<th>What happened &amp; how it was handled</th><th>Context</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
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
"""


def render(generated_at: str, log_records: Optional[list] = None) -> str:
    """Render the run report (summary + lean log table) as one HTML string."""
    records = log_records or []
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Tarzan — Run Report</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)}.</p>"
        "<h2>Issues &amp; how they were handled</h2>"
        f"{_issues_html()}"
        f"{_thirdparty_note(records)}"
        "<h2>Run log</h2>"
        f"{_log_table_html(records)}"
        "<footer>Tarzan run report · regenerated every run.</footer>"
        "</body></html>"
    )


def write_report(output_dir: str, generated_at: str,
                 log_records: Optional[list] = None,
                 filename: str = "report.html") -> Optional[str]:
    """Write the single HTML run report. Best-effort — None on I/O error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(generated_at, log_records))
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Run report write failed: %s", e)
        return None
