"""The single run log — one self-contained, color-coded HTML file.

``output/report.html`` is the ONE log Tarzan writes per run (there is no
separate analyzer.log). It renders the whole run's log as a structured,
color-coded table — one row per log entry — modelled on the reference
DetailedRunLog: a fixed column layout with the row colored by log level.

Columns (the fields Tarzan's logging actually provides):
    Log level · Time · Origin (logger name) · Message

Best-effort, like the collectors it reads: rendering/writing must never raise
into the pipeline. Inline CSS only, no JS / external resources, so it opens
anywhere offline (Chrome, email, etc.).
"""

from __future__ import annotations

import html
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Per-level row color (mirrors the reference log's palette + Tarzan levels).
#   ERROR/CRITICAL → red, WARNING → amber, INFO/notice → ink, DEBUG → teal.
_LEVEL_COLOR = {
    "CRITICAL": "#B91C1C",
    "ERROR": "#DC2626",
    "WARNING": "#D28004",   # amber, as in the reference
    "INFO": "#1E293B",
    "DEBUG": "#579FA8",     # teal, as in the reference
    "NOTSET": "#64748B",
}
# Soft row background tint per level (subtle zebra-by-severity).
_LEVEL_BG = {
    "CRITICAL": "#FEF2F2",
    "ERROR": "#FEF2F2",
    "WARNING": "#FFFBEB",
    "DEBUG": "#F0FDFA",
}


def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def _color(level: str) -> str:
    return _LEVEL_COLOR.get((level or "").upper(), "#1E293B")


def _counts(records) -> dict:
    out: dict[str, int] = {}
    for r in records or []:
        lv = (r.get("level") or "").upper()
        out[lv] = out.get(lv, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _rows_html(records) -> str:
    rows = []
    for r in records or []:
        lv = (r.get("level") or "").upper()
        color = _color(lv)
        bg = _LEVEL_BG.get(lv, "")
        style = f"color:{color};" + (f"background:{bg};" if bg else "")
        # Message keeps whitespace (ASCII banners / indented traces) readable.
        rows.append(
            f'<tr style="{style}">'
            f'<td class="lvl">{_esc(lv)}</td>'
            f'<td class="mono">{_esc(r.get("time"))}</td>'
            f'<td class="org">{_esc(r.get("origin"))}</td>'
            f'<td class="msg">{_esc(r.get("message"))}</td>'
            "</tr>"
        )
    return "".join(rows)


def _summary_chips(records) -> str:
    counts = _counts(records)
    order = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
    chips = []
    for lv in order:
        if counts.get(lv):
            c = _color(lv)
            chips.append(
                f'<span class="chip" style="color:{c};border-color:{c}55;">'
                f"{counts[lv]} {lv}</span>"
            )
    # Any level not in the standard order (defensive).
    for lv, n in counts.items():
        if lv not in order:
            chips.append(f'<span class="chip">{n} {_esc(lv)}</span>')
    return " ".join(chips) or '<span class="chip">no log entries</span>'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1E293B;
  margin:0;padding:20px 24px;background:#F8FAFC;}
h1{font-size:20px;margin:0 0 2px;} .sub{color:#64748B;font-size:13px;margin:0 0 12px;}
.chips{margin:0 0 14px;} .chip{display:inline-block;font-size:12px;font-weight:700;
  padding:2px 10px;border:1px solid #CBD5E1;border-radius:999px;margin:0 6px 6px 0;
  background:#fff;}
table.log{border-collapse:collapse;width:100%;background:#fff;
  box-shadow:0 1px 2px rgba(15,23,42,.06);border-radius:8px;overflow:hidden;}
.log th{position:sticky;top:0;text-align:left;background:#0F172A;color:#E2E8F0;
  font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
  padding:8px 10px;}
.log td{padding:5px 10px;border-bottom:1px solid #EEF2F7;font-size:12.5px;
  vertical-align:top;}
.log tr:hover td{background:#F1F5F9 !important;}
td.lvl{font-weight:700;white-space:nowrap;}
td.mono{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:nowrap;color:#64748B;}
td.org{color:#64748B;white-space:nowrap;font-size:11.5px;}
td.msg{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;
  word-break:break-word;line-height:1.4;}
footer{color:#94A3B8;font-size:12px;margin-top:14px;}
"""


def render(generated_at: str, log_records: Optional[list] = None) -> str:
    """Render the whole run log as a single color-coded HTML table string."""
    records = log_records or []
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Log</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Tarzan — Run Log</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)} · {len(records)} entries. "
        "The complete, color-coded log of this run.</p>"
        f"<div class='chips'>{_summary_chips(records)}</div>"
        "<table class='log'><thead><tr>"
        "<th style='width:6%'>Level</th><th style='width:8%'>Time</th>"
        "<th style='width:14%'>Origin</th><th>Message</th>"
        "</tr></thead><tbody>"
        f"{_rows_html(records)}"
        "</tbody></table>"
        "<footer>Tarzan run log · regenerated every run.</footer>"
        "</body></html>"
    )


def write_report(output_dir: str, generated_at: str,
                 log_records: Optional[list] = None,
                 filename: str = "report.html") -> Optional[str]:
    """Write the single HTML run log. Best-effort — returns None on I/O error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(generated_at, log_records))
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Run log write failed: %s", e)
        return None
