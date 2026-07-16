"""Newsletter "Backtesting" section (rendered under the Optimizer block).

Long-history synthetic comparison of the candidate portfolios in
``input/portfolio_test.csv``. Compute is delegated entirely to
:mod:`tarzan.backtest` (which reuses Tarzan's shared stats/robustness/synthetic
engine with a fine, currency-matched, time-varying risk-free); this module only
renders the result with the newsletter palette/label primitives.

Performance & safety:
  * the backtest is network-bound and can take ~1 minute, so it runs only when
    explicitly enabled (``TARZAN_BACKTEST`` env truthy) or when a same-day cache
    fragment exists — normal/deterministic renders and tests skip it and the
    section reports ``available=False``;
  * the whole build is wrapped in try/except, so a backtest failure can never
    break the newsletter;
  * the rendered HTML fragment is cached on disk keyed by date + weights-file
    mtime, so at most one backtest runs per day.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from tarzan.export.newsletter._constants import (
    ASSET_COLORS, GEO_COLORS, PALETTE, _NewsletterContext, group_by_class_role,
)
from tarzan.export.newsletter._format import _semaphore, _semaphore_color
from tarzan.export.newsletter._sections_alloc import _div_label

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
_WEIGHTS = ROOT / "input" / "portfolio_test.csv"
_CACHE_DIR = ROOT / "output" / ".backtest_cache"

P = PALETTE
_SERIES_COLORS = ["#4F46E5", "#DC2626", "#16A34A", "#EA580C",
                  "#0891B2", "#C026D3", "#65A30D", "#334155"]
_BB = f'border-bottom:1px solid {P["border"]};'


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------

def _present(order, portfolios, attr) -> list[str]:
    return [c for c in order
            if any(getattr(p, attr).get(c, 0.0) for p in portfolios)]


def _kicker(text):
    return (f'<div style="font-size:13px;font-weight:700;letter-spacing:0.08em;'
            f'color:{P["accent"]};text-transform:uppercase;margin-top:28px;">{text}</div>')


def _sub(text):
    return f'<div style="margin-top:4px;font-size:12px;color:{P["muted"]};">{text}</div>'


def _subhead(text):
    return (f'<div style="margin-top:20px;font-size:11px;font-weight:700;letter-spacing:0.06em;'
            f'color:{P["muted"]};text-transform:uppercase;">{text}</div>')


def _cellval(v):
    return v if isinstance(v, tuple) else (v, P["ink"])


def _grid_table(first_col_label, headers, rows, *, first_w=150):
    """Generic comparison grid used by every block: muted uppercase headers,
    tight rows, NON-bold numbers, fixed layout. ``headers`` = (label, align[,
    width]); ``rows`` are data rows {label_html, values, bg?, pad?, bar?} or
    class headers {group:(cls, color, role)} with a coloured left bar."""
    ncols = 1 + len(headers)
    cols = [f'<col style="width:{first_w}px;">']
    for hd in headers:
        wd = hd[2] if len(hd) > 2 else None
        cols.append(f'<col style="width:{wd}px;">' if wd else "<col>")
    # A header may set a 4th element ``nowrap=True`` to keep its column (header
    # + data cells) on a single line — used for short date columns that must
    # not wrap (e.g. "Real from" / "2024-05").
    nowrap = [bool(hd[3]) if len(hd) > 3 else False for hd in headers]
    hcell = (f'padding:5px 6px;font-size:10px;font-weight:700;letter-spacing:0.03em;'
             f'text-transform:uppercase;color:{P["subtle"]};vertical-align:bottom;')
    out = [f'<table cellpadding="0" cellspacing="0" border="0" style="table-layout:fixed;'
           f'width:100%;margin-top:8px;background:{P["card_alt"]};border:1px solid {P["border"]};'
           f'border-radius:10px;border-collapse:collapse;overflow:hidden;">',
           f'<colgroup>{"".join(cols)}</colgroup>',
           f'<tr><td style="{hcell}word-break:break-word;">{first_col_label}</td>']
    for j, hd in enumerate(headers):
        ws_h = "white-space:nowrap;" if nowrap[j] else "word-break:break-word;"
        out.append(f'<td align="{hd[1]}" style="{hcell}{ws_h}">{hd[0]}</td>')
    out.append("</tr>")
    for r in rows:
        if r.get("group"):
            cls, color, role = r["group"]
            role_html = (f'&nbsp;&middot;&nbsp;<span style="color:{P["muted"]};font-weight:700;">'
                         f'{role}</span>' if role and role != "—" else "")
            out.append(f'<tr><td colspan="{ncols}" style="padding:5px 8px 5px 10px;'
                       f'background:#FFFFFF;{_BB}border-left:4px solid {color};font-size:10px;'
                       f'letter-spacing:0.04em;text-transform:uppercase;">'
                       f'<span style="color:{color};font-weight:700;">{cls}</span>{role_html}</td></tr>')
            continue
        bgs = f'background:{r["bg"]};' if r.get("bg") else ""
        pad = r.get("pad", "4px 8px")
        lb = f'border-left:4px solid {r["bar"]};' if r.get("bar") else ""
        line = [f'<tr><td style="padding:{pad};{_BB}{bgs}{lb}font-size:12px;color:{P["ink"]};'
                f'white-space:nowrap;">{r["label_html"]}</td>']
        for j, v in enumerate(r["values"]):
            txt, col = _cellval(v)
            ws_c = "white-space:nowrap;" if nowrap[j] else ""
            line.append(f'<td align="{headers[j][1]}" style="padding:{pad};{_BB}{bgs}'
                        f'font-size:12px;font-weight:400;color:{col};'
                        f'font-variant-numeric:tabular-nums;{ws_c}">{txt}</td>')
        line.append("</tr>")
        out.append("".join(line))
    out.append("</table>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 1. Instruments x portfolios — grouped by asset class + role (sub-class)
# ---------------------------------------------------------------------------

def _instrument_matrix(portfolios):
    reps = {}
    for p in portfolios:
        for it in p.items:
            reps.setdefault(it.bare, it)
    wmaps = {p.name: {it.bare: it.weight for it in p.items} for p in portfolios}
    headers = [(p.name, "right") for p in portfolios]

    raw_groups = group_by_class_role(
        reps.values(),
        asset_class=lambda it: (it.holding.asset_class.value
                                if getattr(it.holding, "asset_class", None) else "Other"),
        role=lambda it: (getattr(it.holding, "role", "") or "—"),
    )
    rows = []
    for cls, col, role_list in raw_groups:
        for role, its in role_list:
            rows.append({"group": (cls, col, role)})
            for it in sorted(its, key=lambda x: x.bare):
                vals = []
                for p in portfolios:
                    v = wmaps[p.name].get(it.bare)
                    vals.append((f"{v:.1f}%", P["ink"]) if v else ("&mdash;", P["subtle"]))
                rows.append({
                    "label_html": f'<span style="font-weight:600;color:{P["ink"]};">{it.bare}</span>',
                    "values": vals, "pad": "3px 8px 3px 10px", "bar": col})
    return _grid_table("Instrument", headers, rows, first_w=110)


# ---------------------------------------------------------------------------
# 2. Diversification — NOTIONAL (not normalised), drift-coloured vs target
# ---------------------------------------------------------------------------

def _per_class_lev(p, cls):
    cap = p.cap.get(cls, 0.0)
    return (p.notl_gross.get(cls, 0.0) / cap) if cap > 0 else None


def _diversification(portfolios, asset_target, geo_target, tol, asset_order, geo_order):
    headers = [(p.name, "right") for p in portfolios] + [("Target", "right", 56)]

    def _cell(v, tgt):
        if not v:
            return ("&mdash;", P["subtle"])
        col = _semaphore_color(_semaphore((v - tgt) if tgt is not None else None, tol))
        return (f"{v:.1f}%", col)

    def _pct(v):
        return "&mdash;" if not v else f"{v:.1f}%"

    asset_labels = _present(asset_order, portfolios, "notl_gross")
    a_rows = []
    for cls in asset_labels:
        tgt = asset_target.get(cls)
        a_rows.append({
            "label_html": _div_label(cls, ASSET_COLORS.get(cls, P["accent"])),
            "values": [_cell(p.notl_gross.get(cls), tgt) for p in portfolios] + [(_pct(tgt), P["muted"])],
        })
    tgt_gross = sum(v for v in asset_target.values() if v) or None
    a_rows.append({
        "label_html": f'<span style="color:{P["ink"]};">Gross exposure</span>',
        "values": [f"{p.gross:.0f}%" for p in portfolios]
                  + [(f"{tgt_gross:.0f}%" if tgt_gross else "&mdash;", P["muted"])], "bg": P["accent_bg"]})
    a_rows.append({
        "label_html": f'<span style="color:{P["ink"]};">Leverage</span>',
        "values": [f"{p.leverage:.2f}x" for p in portfolios]
                  + [(f"{tgt_gross / 100.0:.2f}x" if tgt_gross else "&mdash;", P["muted"])], "bg": P["accent_bg"]})
    asset_tbl = _grid_table("Asset class", headers, a_rows)

    lev_labels = [c for c in asset_labels
                  if any((_per_class_lev(p, c) or 0) > 1.001 for p in portfolios)]
    lev_headers = [(p.name, "right") for p in portfolios]
    lev_rows = []
    for cls in lev_labels:
        vals = [("&mdash;" if _per_class_lev(p, cls) is None
                 else f"{_per_class_lev(p, cls):.2f}x") for p in portfolios]
        lev_rows.append({"label_html": _div_label(cls, ASSET_COLORS.get(cls, P["accent"])),
                         "values": vals})
    lev_tbl = _grid_table("Leverage by class", lev_headers, lev_rows) if lev_rows else ""

    geo_labels = _present(geo_order, portfolios, "geo_notl")
    g_rows = []
    for reg in geo_labels:
        tgt = geo_target.get(reg)
        g_rows.append({
            "label_html": _div_label(reg, GEO_COLORS.get(reg, P["accent"])),
            "values": [_cell(p.alloc.get("geo_notl", {}).get(reg), tgt) for p in portfolios]
                      + [(_pct(tgt), P["muted"])]})
    geo_tbl = _grid_table("Region", headers, g_rows)

    lev_block = ""
    if lev_tbl:
        lev_block = (_subhead("By leverage class")
                     + _sub("Notional exposure / funded capital in each class "
                            "(&gt;1.00x = partly synthetic, e.g. an efficient-core overlay).")
                     + lev_tbl)
    return (_kicker("Diversification")
            + _subhead("By asset class")
            + _sub("NOTIONAL exposure (leverage-aware, not normalised), % of capital, "
                   "coloured by drift vs target.")
            + asset_tbl + lev_block
            + _subhead("By geography")
            + _sub("NOTIONAL equity exposure by MSCI region (not normalised), vs target.")
            + geo_tbl)


# ---------------------------------------------------------------------------
# 3. Portfolio risk metrics — metrics on ROWS, portfolios on COLUMNS, EUR/USD
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    ("CAGR", "cagr", "{:.2f}%"), ("Vol", "volatility", "{:.1f}%"),
    ("Sharpe", "sharpe", "{:.2f}"), ("Sortino", "sortino", "{:.2f}"),
    ("Max DD", "max_drawdown", "{:.1f}%"), ("Ulcer", "ulcer_index", "{:.1f}%"),
    ("VaR", "var_95", "{:.2f}%"), ("CVaR", "cvar_95", "{:.2f}%"),
    ("\u03b1", "alpha", "{:.2f}%"), ("\u03b2", "beta", "{:.2f}"),
]


def _risk_block(portfolios, attr):
    from tarzan.engine.stats import compute_ulcer_index
    headers = [(p.name, "right") for p in portfolios]
    cache = {}
    for p in portfolios:
        m = dict(getattr(p, attr, {}) or {})
        if "ulcer_index" not in m and getattr(p, "nav", None) is not None:
            try:
                m["ulcer_index"] = compute_ulcer_index(p.nav)
            except Exception:  # noqa: BLE001
                m["ulcer_index"] = None
        cache[p.name] = m
    rows = []
    for label, key, fmt in _METRIC_COLS:
        vals = []
        for p in portfolios:
            v = cache[p.name].get(key)
            ok = v is not None and not (isinstance(v, float) and v != v)
            vals.append(fmt.format(v) if ok else "&mdash;")
        rows.append({"label_html": f'<span style="color:{P["ink"]};">{label}</span>',
                     "values": vals})
    return _grid_table("Metric", headers, rows, first_w=120)


def _risk_metrics(portfolios):
    rf_e = next(((p.metrics_aligned_eur or {}).get("risk_free") for p in portfolios
                 if p.metrics_aligned_eur), None)
    rf_u = next(((p.metrics_aligned_usd or {}).get("risk_free") for p in portfolios
                 if p.metrics_aligned_usd), None)
    win = next((p.window for p in portfolios if p.window), None)
    wtxt = f"{win[0]:%Y-%m} &rarr; {win[1]:%Y-%m}" if win else "aligned"
    return (_kicker("Portfolio risk metrics")
            + _sub(f"Single aligned history ({wtxt}). &alpha;/&beta; vs the S&amp;P 500.")
            + f'<div style="margin-top:12px;font-size:11px;font-weight:700;color:{P["ink"]};">'
              f'EUR numeraire (unhedged)' + (f' &middot; risk-free {rf_e:.2f}%' if rf_e else '') + '</div>'
            + _risk_block(portfolios, "metrics_aligned_eur")
            + f'<div style="margin-top:16px;font-size:11px;font-weight:700;color:{P["ink"]};">'
              f'USD numeraire' + (f' &middot; risk-free {rf_u:.2f}%' if rf_u else '') + '</div>'
            + _risk_block(portfolios, "metrics_aligned_usd"))


def _robustness_metrics(portfolios):
    headers = [(p.name, "right") for p in portfolios]

    def roll(p, key, sub, fmt="{:.1f}%"):
        v = (p.rob.get(key) or {}).get(sub)
        return fmt.format(v) if v is not None else "&mdash;"

    def stress(p, scen):
        v = ((p.rob.get("stress") or {}).get(scen) or {}).get("max_drawdown")
        return f"{v:.1f}%" if v is not None else "&mdash;"

    def mc(p, metric, sub):
        v = ((p.rob.get("bootstrap") or {}).get(metric) or {}).get(sub)
        return f"{v:.0f}%" if v is not None else "&mdash;"

    def sharpe_range(p):
        d = p.rob.get("sharpe") or {}
        return (f'{d["min"]:.2f}&ndash;{d["max"]:.2f}' if d.get("min") is not None else "&mdash;")

    specs = [
        ("Roll 1Y ret p05", lambda p: roll(p, "rolling1y", "p05")),
        ("Roll 1Y ret median", lambda p: roll(p, "rolling1y", "median")),
        ("Roll 1Y ret p95", lambda p: roll(p, "rolling1y", "p95")),
        ("1Y windows positive", lambda p: roll(p, "rolling1y", "pct_positive", "{:.0f}%")),
        ("Roll 3Y ret p05", lambda p: roll(p, "rolling3y", "p05")),
        ("Roll 3Y ret median", lambda p: roll(p, "rolling3y", "median")),
        ("Roll 1Y Sharpe min&ndash;max", sharpe_range),
        ("MC CAGR 1Y p05", lambda p: mc(p, "cagr", "p05")),
        ("MC CAGR 1Y p95", lambda p: mc(p, "cagr", "p95")),
        ("MC MaxDD p05 (worst)", lambda p: mc(p, "max_drawdown", "p05")),
        ("Dot-com 2000-02 maxDD", lambda p: stress(p, "Dot-com 2000-02")),
        ("GFC 2008 maxDD", lambda p: stress(p, "GFC 2008")),
        ("COVID 2020 maxDD", lambda p: stress(p, "COVID 2020")),
        ("2022 rate shock maxDD", lambda p: stress(p, "2022 rate shock")),
    ]
    rows = [{"label_html": f'<span style="color:{P["ink"]};">{lbl}</span>',
             "values": [fn(p) for p in portfolios]} for lbl, fn in specs]
    return (_kicker("Robustness &middot; stress &amp; distribution metrics")
            + _sub("Rolling-window percentiles, Monte-Carlo (block bootstrap) bands, "
                   "and historical crisis drawdowns.")
            + _grid_table("Metric", headers, rows, first_w=150))


# ---------------------------------------------------------------------------
# 4. Robustness charts — multi-line SVG with a per-chart legend
# ---------------------------------------------------------------------------

def _chart_legend(names, colors):
    out = [f'<div style="margin-top:4px;font-size:10px;color:{P["muted"]};line-height:1.8;">']
    for i, name in enumerate(names):
        out.append(f'<span style="display:inline-block;margin-right:11px;white-space:nowrap;">'
                   f'<span style="display:inline-block;width:11px;height:3px;border-radius:2px;'
                   f'background:{colors[i % len(colors)]};vertical-align:middle;margin-right:4px;">'
                   f'</span>{name}</span>')
    out.append("</div>")
    return "".join(out)


def _multiline(frame, colors, *, w=580, h=210, pct=False, log=False, fmt="{:.0f}"):
    import math
    frame = frame.dropna(how="all")
    if frame.empty or len(frame) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 46, 10, 10, 22
    x0, x1 = frame.index.min().value, frame.index.max().value
    vmin, vmax = float(frame.min().min()), float(frame.max().max())
    use_log = log and vmin > 0
    if use_log:
        lmin, lmax = math.log10(vmin), math.log10(vmax)
        if lmax <= lmin:
            lmax = lmin + 0.1
        pad = (lmax - lmin) * 0.06
        lmin -= pad
        lmax += pad
    else:
        if vmax <= vmin:
            vmax = vmin + 1.0
        rng = vmax - vmin
        vmin -= rng * 0.06
        vmax += rng * 0.06

    def X(ts):
        return pad_l + (w - pad_l - pad_r) * ((ts.value - x0) / (x1 - x0) if x1 > x0 else 0.0)

    def Y(v):
        if use_log:
            return pad_t + (h - pad_t - pad_b) * (1.0 - (math.log10(v) - lmin) / (lmax - lmin))
        return pad_t + (h - pad_t - pad_b) * (1.0 - (v - vmin) / (vmax - vmin))

    parts = [f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
             f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;margin-top:6px;'
             f'font-family:-apple-system,Helvetica,Arial,sans-serif;">']
    for k in range(5):
        val = (10 ** (lmin + (lmax - lmin) * k / 4.0)) if use_log else (vmin + (vmax - vmin) * k / 4.0)
        yy = Y(val)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w - pad_r}" y2="{yy:.1f}" '
                     f'stroke="{P["border"]}" stroke-width="0.75"/>')
        parts.append(f'<text x="{pad_l - 5}" y="{yy + 3:.1f}" text-anchor="end" font-size="9" '
                     f'fill="{P["subtle"]}">{fmt.format(val)}{"%" if pct else ""}</text>')
    if pct and not use_log and vmin < 0 < vmax:
        y0 = Y(0.0)
        parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{w - pad_r}" y2="{y0:.1f}" '
                     f'stroke="{P["subtle"]}" stroke-width="0.9" stroke-dasharray="2,2"/>')
    years = sorted({ts.year for ts in frame.index})
    for yr in years[::max(1, len(years) // 7)]:
        ts = pd.Timestamp(f"{yr}-06-30")
        if ts < frame.index.min() or ts > frame.index.max():
            continue
        parts.append(f'<text x="{X(ts):.1f}" y="{h - 6}" text-anchor="middle" font-size="9" '
                     f'fill="{P["subtle"]}">{yr}</text>')
    for i, c in enumerate(frame.columns):
        s = frame[c].dropna()
        if use_log:
            s = s[s > 0]
        if len(s) < 2:
            continue
        pts = " ".join(f"{X(ts):.1f},{Y(v):.1f}" for ts, v in s.items())
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % len(colors)]}" '
                     f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append("</svg>")
    return "".join(parts)


def _robustness(portfolios):
    navs = {p.name: p.nav for p in portfolios
            if getattr(p, "nav", None) is not None and len(p.nav) >= 60}
    if not navs:
        return ""
    df = pd.DataFrame(navs).dropna(how="all").ffill()
    colors = _SERIES_COLORS
    names = list(df.columns)

    def samp(x):
        return x.iloc[::21]

    growth = samp(df / df.iloc[0] * 100.0)
    dd = samp((df / df.cummax() - 1.0) * 100.0)
    r5 = samp(((df / df.shift(5 * 252)) ** (252.0 / (5 * 252)) - 1.0) * 100.0).dropna(how="all")
    r10 = samp(((df / df.shift(10 * 252)) ** (252.0 / (10 * 252)) - 1.0) * 100.0).dropna(how="all")

    html = (_kicker("Robustness")
            + _sub("Long-history behaviour on the single aligned window (monthly)."))
    for frame, ttl, pct, log in [
            (growth, "Growth of 100 (log scale, start = 100)", False, True),
            (dd, "Drawdown (underwater) %", True, False),
            (r5, "Rolling 5-year return, annualised %", True, False),
            (r10, "Rolling 10-year return, annualised %", True, False)]:
        html += (f'<div style="margin-top:18px;font-size:11px;font-weight:700;'
                 f'color:{P["ink"]};">{ttl}</div>'
                 + _multiline(frame, colors, pct=pct, log=log)
                 + _chart_legend(names, colors))
    return html


# ---------------------------------------------------------------------------
# 5. Simulation map
# ---------------------------------------------------------------------------

def _simulation_map(portfolios):
    from tarzan.backtest import simulation_rows
    headers = [("Real from", "center", 72, True), ("Base from", "center", 72, True),
               ("Simulation base (proxy x exposure)", "left")]
    rows = []
    for r in simulation_rows(portfolios):
        rows.append({
            "label_html": f'<span style="font-weight:600;color:{P["ink"]};">{r["ticker"]}</span>',
            "pad": "3px 8px",
            "values": [
                (r["real_from"], P["muted"]), (r["base_from"], P["muted"]),
                (f'<span style="font-size:10px;color:{P["muted"]};">{r["base"]}</span>', P["muted"]),
            ],
        })
    return (_kicker("Simulation map")
            + _sub("Per-instrument history reconstruction: real fund returns from "
                   "&lsquo;Real from&rsquo;, proxy base before, factor tilt on factor ETFs.")
            + _grid_table("Ticker", headers, rows, first_w=90))


# ---------------------------------------------------------------------------
# Assembly + guarded entry point
# ---------------------------------------------------------------------------

def _title():
    return (f'<div style="margin-top:32px;font-size:13px;font-weight:700;letter-spacing:0.08em;'
            f'color:{P["accent"]};text-transform:uppercase;">Backtesting</div>'
            f'<div style="margin-top:2px;font-size:12px;color:{P["muted"]};">'
            f'Long-history synthetic comparison of the candidate portfolios '
            f'(reconstructed pre-inception from index proxies, net of TER).</div>')


def _render(portfolios, asset_target, geo_target, tol) -> str:
    from tarzan.backtest import ASSET_ORDER, GEO_ORDER
    return (_title()
            + _kicker("Instruments &times; portfolios")
            + _sub("Target weight of each instrument in each candidate portfolio.")
            + _instrument_matrix(portfolios)
            + _diversification(portfolios, asset_target, geo_target, tol,
                               ASSET_ORDER, GEO_ORDER)
            + _risk_metrics(portfolios)
            + _robustness_metrics(portfolios)
            + _robustness(portfolios)
            + _simulation_map(portfolios))


def _cache_key() -> Optional[str]:
    """Date + weights-file mtime hash, so the backtest runs at most once/day."""
    try:
        from datetime import date
        mtime = int(_WEIGHTS.stat().st_mtime)
        raw = f"{date.today().isoformat()}:{mtime}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def _enabled() -> bool:
    return str(os.environ.get("TARZAN_BACKTEST", "")).strip().lower() in {"1", "true", "yes", "on"}


def _render_and_cache(portfolios, cfg, cache_file) -> dict:
    tol = float(getattr(cfg, "rebalancing_target_tolerance_pctg", 1.5) or 1.5)
    html = _render(portfolios,
                   cfg.invested_allocation_targets_pctg or {},
                   cfg.equity_geo_targets_pctg or {}, tol)
    if cache_file:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(html, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return {"available": True, "html": html}


def _build_backtesting(ctx: _NewsletterContext, portfolios=None) -> dict:
    """Build the Backtesting section. Returns {"available", "html"}.

    If ``portfolios`` is supplied (the CLI computes the backtest once and shares
    it with the Excel export), it is rendered directly. Otherwise the section
    runs only when explicitly enabled (``TARZAN_BACKTEST``) or a same-day cache
    fragment exists, so normal/test renders are unaffected. Never raises.
    """
    off = {"available": False, "html": ""}
    cfg = ctx.config
    key = _cache_key()
    cache_file = (_CACHE_DIR / f"{key}.html") if key else None

    # Fast path: caller already computed the portfolios — just render.
    if portfolios:
        try:
            return _render_and_cache(portfolios, cfg, cache_file)
        except Exception as e:  # noqa: BLE001
            logger.warning("Backtesting section skipped (%s): %s", type(e).__name__, e)
            return off

    if not _WEIGHTS.exists():
        return off
    if cache_file and cache_file.exists():
        try:
            html = cache_file.read_text(encoding="utf-8")
            if html.strip():
                return {"available": True, "html": html}
        except Exception:  # noqa: BLE001
            pass
    if not _enabled():
        return off
    try:
        from tarzan.backtest import run_backtest
        portfolios = run_backtest(_WEIGHTS, currency="eur",
                                  backfill="factor", rebalance="quarterly")
        if not portfolios:
            return off
        return _render_and_cache(portfolios, cfg, cache_file)
    except Exception as e:  # noqa: BLE001
        logger.warning("Backtesting section skipped (%s): %s", type(e).__name__, e)
        return off
