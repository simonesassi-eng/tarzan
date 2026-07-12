"""Unified, human-readable run report (single self-contained HTML file).

One Chrome-openable file per run (``output/report.html``) describing THIS
Tarzan run:

  * Run summary — headline figures (total / invested / cash value, XIRR,
    TWROR, coverage %, holdings count, generated timestamp);
  * Data quality — everything the run skipped, coerced, or fell back on
    (from :mod:`tarzan.data_quality`).

The verbose ``analyzer.log`` stays separate (it is the raw debug trace).
Best-effort, like the collectors it reads: rendering/writing must never raise
into the pipeline. Inline CSS only, no JS / external resources, so it opens
anywhere offline.
"""

from __future__ import annotations

import html
import logging
import math
import os
from typing import Any, Optional

from tarzan import data_quality

logger = logging.getLogger(__name__)

_SEV_COLOR = {"ERROR": "#DC2626", "WARNING": "#D97706", "INFO": "#2563EB"}
_SEV_BG = {"ERROR": "#FEF2F2", "WARNING": "#FFF7ED", "INFO": "#EFF6FF"}


def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def _eur(v) -> str:
    try:
        return f"€{float(v):,.2f}"
    except (TypeError, ValueError):
        return _esc(v)


# ---------------------------------------------------------------------------
# Data-quality section
# ---------------------------------------------------------------------------

def _data_quality_html() -> str:
    issues = data_quality.issues()
    if not issues:
        return (
            '<p class="ok">No issues this run — every input parsed and priced '
            "cleanly. ✅</p>"
        )
    counts = data_quality.counts()
    chips = " ".join(
        f'<span class="chip" style="background:{_SEV_BG[s]};color:{_SEV_COLOR[s]};'
        f'border:1px solid {_SEV_COLOR[s]}33;">{counts[s]} {s}</span>'
        for s in ("ERROR", "WARNING", "INFO") if counts.get(s)
    )
    rows = []
    # Group by source, most-serious severity first within each group.
    order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    for src in sorted({i.source for i in issues}):
        group = sorted((i for i in issues if i.source == src),
                       key=lambda i: order.get(i.severity, 99))
        rows.append(f'<tr><td class="src" rowspan="{len(group)}">{_esc(src)}</td>'
                    + _issue_cells(group[0]) + "</tr>")
        for it in group[1:]:
            rows.append(f"<tr>{_issue_cells(it)}</tr>")
    return (
        f'<div class="chips">{chips}</div>'
        '<table class="grid"><thead><tr>'
        "<th>Section</th><th>Severity</th><th>Detail</th><th>Context</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _issue_cells(it) -> str:
    color = _SEV_COLOR.get(it.severity, "#334155")
    return (
        f'<td><span class="sev" style="color:{color};">{_esc(it.severity)}</span></td>'
        f"<td>{_esc(it.message)}</td>"
        f'<td class="ctx">{_esc(it.context)}</td>'
    )


# ---------------------------------------------------------------------------
# Run-summary section (headline figures for THIS run)
# ---------------------------------------------------------------------------

def _pct(v) -> str:
    try:
        f = float(v)
        return "—" if not math.isfinite(f) else f"{f:,.2f}%"
    except (TypeError, ValueError):
        return "—"


def _run_summary_html(metrics: Any) -> str:
    if metrics is None:
        return '<p class="muted">No metrics available for this run.</p>'
    m = metrics

    def g(attr, default=None):
        return getattr(m, attr, default)

    n_holdings = 0
    try:
        hdf = g("holdings_df")
        n_holdings = 0 if hdf is None else len(hdf)
    except Exception:  # noqa: BLE001
        pass

    # (label, value) tiles. Order-path fields (XIRR/TWROR/coverage) show "—"
    # on a holdings-only run where they are None.
    tiles = [
        ("Total value", _eur(g("total_value"))),
        ("Invested", _eur(g("invested_value"))),
        ("Cash", _eur(g("cash_value"))),
        ("Holdings", str(n_holdings)),
        ("XIRR (money-weighted, ann.)", _pct(g("xirr_pct")) if g("xirr_pct") is not None else "—"),
        ("TWROR (time-weighted, cum.)", _pct(g("twror_pct")) if g("twror_pct") is not None else "—"),
        ("Real-data coverage", _pct(g("returns_coverage_pct")) if g("returns_coverage_pct") is not None else "—"),
    ]
    # Surface degraded computers (a section computed on defaults, not real data).
    degraded = g("degraded_computers") or []
    tiles_html = "".join(
        f'<div class="tile"><div class="tk">{_esc(label)}</div>'
        f'<div class="tv">{_esc(value)}</div></div>'
        for label, value in tiles
    )
    warn = ""
    if degraded:
        warn = (f'<p class="degraded">⚠ {len(degraded)} metric section(s) fell '
                f"back to defaults (not real results): {_esc(', '.join(degraded))}."
                "</p>")
    return f'<div class="tiles">{tiles_html}</div>{warn}'


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1E293B;
  max-width:900px;margin:24px auto;padding:0 20px;line-height:1.5;}
h1{font-size:22px;margin:0 0 2px;} h2{font-size:16px;margin:28px 0 10px;
  border-bottom:2px solid #EEF2FF;padding-bottom:6px;} h3{font-size:14px;margin:16px 0 6px;}
.sub{color:#64748B;font-size:13px;margin:0 0 4px;}
.chips{margin:8px 0;} .chip{display:inline-block;font-size:12px;font-weight:700;
  padding:2px 9px;border-radius:999px;margin-right:6px;}
table.grid{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 4px;}
.grid th{text-align:left;background:#F8FAFF;color:#475569;font-weight:700;
  padding:6px 8px;border:1px solid #E5E7EF;} .grid td{padding:6px 8px;
  border:1px solid #E5E7EF;vertical-align:top;} .grid td.num{text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap;}
.src{font-weight:700;background:#FCFCFF;} .sev{font-weight:700;} .ctx{color:#64748B;}
.ok{color:#15803D;font-weight:600;} .muted{color:#94A3B8;}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;}
.tile{flex:1 1 150px;border:1px solid #E5E7EF;border-radius:10px;padding:10px 14px;
  background:#FCFCFF;} .tk{color:#64748B;font-size:11px;font-weight:700;
  text-transform:uppercase;letter-spacing:.02em;} .tv{font-size:18px;font-weight:700;
  color:#1E293B;font-variant-numeric:tabular-nums;margin-top:2px;}
.degraded{color:#D97706;font-size:13px;font-weight:600;margin:8px 0 0;}
details{margin:8px 0;} summary{cursor:pointer;color:#5B5BD6;font-size:13px;font-weight:600;}
pre.log{background:#0F172A;color:#E2E8F0;font-family:ui-monospace,Menlo,Consolas,monospace;
  font-size:11.5px;line-height:1.45;padding:12px 14px;border-radius:8px;overflow-x:auto;
  white-space:pre-wrap;word-break:break-word;max-height:520px;overflow-y:auto;margin:8px 0;}
footer{color:#94A3B8;font-size:12px;margin-top:32px;border-top:1px solid #EEF2FF;padding-top:8px;}
"""


def _full_log_html(log_text: Optional[str]) -> str:
    """The complete DEBUG trace, embedded inline in a collapsible block. This
    is what used to be the separate analyzer.log — now part of the one file."""
    if not log_text:
        return '<p class="muted">No log trace captured for this run.</p>'
    return (
        "<details><summary>Show full run log "
        f"({log_text.count(chr(10)) + 1} lines)</summary>"
        f"<pre class='log'>{_esc(log_text)}</pre></details>"
    )


def render(generated_at: str, metrics: Any = None,
           log_text: Optional[str] = None) -> str:
    """Render the whole unified report as a single HTML string.

    ``metrics`` is the run's PortfolioMetrics (for the summary tiles) and
    ``log_text`` is the full captured DEBUG trace (embedded inline). Both are
    optional; None renders a muted placeholder.
    """
    dq_summary = data_quality.summary_line()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Tarzan — Run Report</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)}. The complete record of "
        "this run — headline figures, anything skipped or estimated, and the "
        "full log trace — in one file.</p>"
        "<h2>Run summary</h2>"
        f"{_run_summary_html(metrics)}"
        "<h2>Data quality</h2>"
        f"<p class='sub'>{_esc(dq_summary)}</p>"
        f"{_data_quality_html()}"
        "<h2>Full run log</h2>"
        f"{_full_log_html(log_text)}"
        "<footer>Tarzan run report · this file is regenerated every run.</footer>"
        "</body></html>"
    )


def write_report(output_dir: str, generated_at: str, metrics: Any = None,
                 log_text: Optional[str] = None,
                 filename: str = "report.html") -> Optional[str]:
    """Write the unified HTML report. Best-effort — returns None on I/O error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(generated_at, metrics, log_text))
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Unified report write failed: %s", e)
        return None
