"""Unified, human-readable run report (single self-contained HTML file).

Merges the two per-run side reports into ONE Chrome-openable file
(``output/report.html``):

  * Data quality — everything the run skipped, coerced, or fell back on
    (from :mod:`tarzan.data_quality`);
  * Rebalancing audit — why each suggested trade was proposed, as readable
    prose (from :mod:`tarzan.audit`), not JSON.

The verbose ``analyzer.log`` stays separate (it is the raw debug trace).
Best-effort, like the collectors it reads: rendering/writing must never raise
into the pipeline. Inline CSS only, no JS / external resources, so it opens
anywhere offline.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Optional

from tarzan import audit, data_quality

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
# Rebalancing-audit section (readable prose)
# ---------------------------------------------------------------------------

def _audit_html() -> str:
    records = audit.records()
    if not records:
        return '<p class="muted">No rebalancing plans were produced this run.</p>'
    blocks = []
    for rec in records:
        blocks.append(_audit_plan_html(rec))
    return "".join(blocks)


def _audit_plan_html(rec: dict) -> str:
    cfg = rec.get("config", {}) or {}
    actions = rec.get("actions", []) or []
    holdings = rec.get("holdings", []) or []
    verifs = rec.get("verifications", []) or []

    # Config line, in prose.
    cfg_bits = [
        f"tolerance ±{cfg.get('target_tolerance_pctg')}%",
        f"lump sum {_eur(cfg.get('lump_sum_eur'))}",
        f"cash buffer {_eur(cfg.get('cash_buffer_eur'))}",
    ]
    if cfg.get("cgt_standard_pctg"):
        cfg_bits.append(f"CGT {cfg.get('cgt_standard_pctg')}%/{cfg.get('cgt_government_pctg')}% gov")
    if cfg.get("fee_buy_eur") or cfg.get("fee_sell_eur"):
        cfg_bits.append(f"fees {_eur(cfg.get('fee_buy_eur'))} buy / {_eur(cfg.get('fee_sell_eur'))} sell")

    # Actions as readable lines.
    if actions:
        act_rows = "".join(
            f"<tr><td>{_esc(a.get('direction','').upper())}</td>"
            f"<td>{_esc(a.get('name') or a.get('ticker'))}</td>"
            f'<td class="num">{_eur(a.get("amount_eur"))}</td>'
            f"<td>{_esc(a.get('reason'))}</td></tr>"
            for a in actions
        )
        actions_html = (
            '<table class="grid"><thead><tr><th>Action</th><th>Instrument</th>'
            "<th>Amount</th><th>Why</th></tr></thead><tbody>"
            + act_rows + "</tbody></table>"
        )
    else:
        actions_html = '<p class="ok">Already balanced — no trades suggested.</p>'

    # Post-trade verification per ambit.
    if verifs:
        vrows = "".join(
            f"<tr><td>{_esc(v.get('check'))}</td>"
            f"<td>{_esc(v.get('status'))}</td>"
            f'<td class="ctx">{_esc(v.get("detail"))}</td></tr>'
            for v in verifs
        )
        verif_html = (
            '<details><summary>Post-trade check ({} ambit(s))</summary>'
            '<table class="grid"><thead><tr><th>Ambit</th><th>Status</th>'
            "<th>Detail</th></tr></thead><tbody>{}</tbody></table></details>"
        ).format(len(verifs), vrows)
    else:
        verif_html = ""

    # Inputs the solver saw (collapsed by default — it's the long part).
    hrows = "".join(
        f"<tr><td>{_esc(h.get('ticker') or h.get('isin'))}</td>"
        f"<td>{_esc(h.get('asset_class'))}</td>"
        f'<td class="num">{_eur(h.get("value_eur"))}</td>'
        f'<td class="num">{_esc(h.get("target_portfolio"))}</td></tr>'
        for h in holdings
    )
    holdings_html = (
        '<details><summary>Inputs the optimizer saw ({} holding(s), '
        "total {})</summary>"
        '<table class="grid"><thead><tr><th>Instrument</th><th>Class</th>'
        "<th>Value</th><th>Target %</th></tr></thead><tbody>{}</tbody>"
        "</table></details>"
    ).format(len(holdings), _eur(rec.get("total_value_eur")), hrows)

    return (
        f'<div class="plan"><h3>{_esc(rec.get("plan"))}</h3>'
        f'<p class="cfg">{_esc(" · ".join(cfg_bits))}</p>'
        f"{actions_html}{verif_html}{holdings_html}</div>"
    )


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
.plan{border:1px solid #E5E7EF;border-radius:10px;padding:12px 16px;margin:12px 0;
  background:#FCFCFF;} .cfg{color:#64748B;font-size:12px;margin:0 0 8px;}
details{margin:8px 0;} summary{cursor:pointer;color:#5B5BD6;font-size:13px;font-weight:600;}
footer{color:#94A3B8;font-size:12px;margin-top:32px;border-top:1px solid #EEF2FF;padding-top:8px;}
"""


def render(generated_at: str) -> str:
    """Render the whole unified report as a single HTML string."""
    dq_counts = data_quality.counts()
    dq_summary = data_quality.summary_line()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tarzan — Run Report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Tarzan — Run Report</h1>"
        f"<p class='sub'>Generated {_esc(generated_at)}. Human-readable summary of "
        "this run — what was skipped or estimated, and why each rebalancing "
        "trade was suggested. (Full debug trace: <code>analyzer.log</code>.)</p>"
        "<h2>Data quality</h2>"
        f"<p class='sub'>{_esc(dq_summary)}</p>"
        f"{_data_quality_html()}"
        "<h2>Rebalancing audit</h2>"
        f"{_audit_html()}"
        "<footer>Tarzan run report · this file is regenerated every run.</footer>"
        "</body></html>"
    )


def write_report(output_dir: str, generated_at: str,
                 filename: str = "report.html") -> Optional[str]:
    """Write the unified HTML report. Best-effort — returns None on I/O error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(generated_at))
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Unified report write failed: %s", e)
        return None
