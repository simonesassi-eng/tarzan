"""Excel export for the ad-hoc what-if comparison (``scripts/whatif.py``).

Single sheet, **portfolios as columns**: a specs block (summary + per-
instrument weights) followed by matrices for funded-capital allocation,
notional exposure, equity geography, and returns & risk. Any number of
portfolios (the real "Current" one plus each column of the weights CSV)
render side by side, so adding a portfolio just adds a column.

Colors come from the shared taxonomy in ``tarzan.export._format``.
"""

from __future__ import annotations

import logging
from datetime import datetime

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from tarzan.export._format import (
    asset_class_color,
    eur_smart,
    geo_color,
    short_instrument_name,
)

logger = logging.getLogger(__name__)

_C = {
    "header": "1E293B",
    "band": "5B5BD6",
    "alt": "F8FAFF",
    "white": "FFFFFF",
    "text": "1E293B",
    "muted": "64748B",
    "green": "16A34A",
    "red": "DC2626",
    "amber": "D97706",
    "border": "CBD5E1",
}

_ASSET_ORDER = ["Equities", "Fixed Income", "Gold", "Commodities", "Crypto",
                "Alternative", "Cash & Cash Equivalents"]
_GEO_ORDER = ["USA", "Japan", "Eurozone EMU", "Dev ex-USA ex-EMU ex-JP",
              "Emerging Markets", "Other"]
_RISK_ROWS = [
    ("CAGR", "cagr", "perf", "%"),
    ("Return 1Y", "1y", "perf", "%"),
    ("Return 3Y", "3y", "perf", "%"),
    ("Volatility (ann.)", "volatility", "risk", "%"),
    ("Sharpe", "sharpe", "risk", ""),
    ("Sortino", "sortino", "risk", ""),
    ("Max Drawdown", "max_drawdown", "risk", "%"),
    ("VaR 95% (daily)", "var_95", "risk", "%"),
    ("CVaR 95% (daily)", "cvar_95", "risk", "%"),
    ("Beta vs S&P 500", "beta", "risk", ""),
    ("Alpha (ann.)", "alpha", "risk", "%"),
]

_PCOL0 = 3  # portfolios start at column C (A = ticker/category, B = description)


def _fill(hex6):
    return PatternFill("solid", fgColor=hex6)


def _font(size=10, bold=False, color="1E293B", italic=False):
    return Font(name="Calibri", size=size, bold=bold, color=color, italic=italic)


def _align(h="left", v="center"):
    return Alignment(horizontal=h, vertical=v)


def _border():
    s = Side(style="thin", color=_C["border"])
    return Border(left=s, right=s, top=s, bottom=s)


def _renorm(d):
    total = sum(d.values())
    return {k: v * 100.0 / total for k, v in d.items()} if total > 0 else dict(d)


def _dev_color(delta, tol):
    if delta is None or delta != delta or tol <= 0:
        return _C["text"]
    a = abs(delta)
    if a <= tol:
        return _C["green"]
    if a <= 2 * tol:
        return _C["amber"]
    return _C["red"]


def _risk_value(metrics, key, src, unit):
    d = (metrics.performance if src == "perf" else metrics.risk) or {}
    v = d.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.2f}{unit}"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _section_header(ws, row, text, span):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    c = ws.cell(row=row, column=1, value=text)
    c.font = _font(11, bold=True, color=_C["white"])
    c.fill = _fill(_C["band"])
    return row + 1


def _merge_label(ws, row, text, *, bold=True, color=None, bg=None, italic=False):
    """Write a row label merged across columns A:B (so tables whose data
    starts at column C don't show an empty B)."""
    for col in (1, 2):
        c = ws.cell(row=row, column=col)
        if bg:
            c.fill = _fill(bg)
            c.border = _border()
    a = ws.cell(row=row, column=1, value=text)
    a.font = _font(9, bold=bold, color=color or _C["text"], italic=italic)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    return a


def _col_headers(ws, row, portfolios, label_hdr, with_target, second_hdr=None):
    """Write the label (+ optional second label) + one-column-per-portfolio
    (+ optional Target) header."""
    h = ws.cell(row=row, column=1, value=label_hdr)
    h.font = _font(9, bold=True, color=_C["white"])
    h.fill = _fill(_C["header"])
    h.border = _border()
    if second_hdr is not None:
        h2 = ws.cell(row=row, column=2, value=second_hdr)
        h2.font = _font(9, bold=True, color=_C["white"])
        h2.fill = _fill(_C["header"])
        h2.border = _border()
    else:
        # No second column in use → merge A:B so B isn't an empty column.
        c2 = ws.cell(row=row, column=2)
        c2.fill = _fill(_C["header"])
        c2.border = _border()
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
    for j, p in enumerate(portfolios):
        c = ws.cell(row=row, column=_PCOL0 + j, value=p.name)
        c.font = _font(9, bold=True, color=_C["white"])
        c.fill = _fill(_C["header"])
        c.alignment = _align("center")
        c.border = _border()
    tcol = None
    if with_target:
        tcol = _PCOL0 + len(portfolios)
        c = ws.cell(row=row, column=tcol, value="Target")
        c.font = _font(9, bold=True, color=_C["white"])
        c.fill = _fill(_C["header"])
        c.alignment = _align("center")
        c.border = _border()
    return tcol


def _alloc_matrix(ws, row, title, labels, portfolios, value_of, *,
                  target=None, tolerance=1.5, swatch=None) -> dict:
    """Allocation/geo matrix: rows = labels, cols = portfolios (+ Target).

    ``value_of(p, label)`` returns a percentage (0..100+). Portfolio cells
    are traffic-light coloured by their deviation from target when a target
    is supplied.
    """
    ncols = _PCOL0 + len(portfolios) + (1 if target is not None else 0) - 1
    row = _section_header(ws, row, title, ncols)
    tcol = _col_headers(ws, row, portfolios, "Category", target is not None)
    header_row = row
    row += 1
    first = row
    for i, lbl in enumerate(labels):
        bg = _C["alt"] if i % 2 else _C["white"]
        _merge_label(ws, row, lbl, bold=True,
                     color=(swatch(lbl) if swatch else _C["text"]), bg=bg)
        t = target.get(lbl, 0.0) if target is not None else None
        for j, p in enumerate(portfolios):
            v = value_of(p, lbl)
            c = ws.cell(row=row, column=_PCOL0 + j, value=v / 100.0)
            c.number_format = "0.0%"
            c.alignment = _align("center")
            c.fill = _fill(bg)
            c.border = _border()
            c.font = _font(9, color=(_dev_color(v - t, tolerance) if t is not None else _C["text"]))
        if tcol is not None:
            c = ws.cell(row=row, column=tcol, value=t / 100.0)
            c.number_format = "0.0%"
            c.alignment = _align("center")
            c.fill = _fill(bg)
            c.border = _border()
            c.font = _font(9, bold=True, color=_C["muted"])
        row += 1
    return {"header_row": header_row, "first": first, "last": row - 1, "tcol": tcol}


def _plain_row(ws, row, label, portfolios, text_of, color=None):
    """A summary row (label + one text value per portfolio), no target."""
    _merge_label(ws, row, label, bold=True, color=color)
    for j, p in enumerate(portfolios):
        c = ws.cell(row=row, column=_PCOL0 + j, value=text_of(p))
        c.alignment = _align("center")
        c.font = _font(9, color=color or _C["text"])
    return row + 1


def _specs_block(ws, row, portfolios, anchor) -> int:
    """Summary rows + per-instrument weight matrix (portfolios as columns)."""
    ncols = _PCOL0 + len(portfolios) - 1
    row = _section_header(ws, row, "Portfolio specs", ncols)
    _col_headers(ws, row, portfolios, "", with_target=False)
    row += 1
    row = _plain_row(ws, row, "Instruments", portfolios, lambda p: len(p.items))
    row = _plain_row(ws, row, "Gross exposure", portfolios, lambda p: f"{p.gross:.0f}%")
    row = _plain_row(ws, row, "Leverage", portfolios, lambda p: f"{p.leverage:.2f}x", color=_C["amber"])
    row = _plain_row(ws, row, "Notional EUR", portfolios,
                     lambda p: eur_smart(anchor * p.gross / 100.0), color=_C["muted"])
    row += 1

    # Weight matrix: Ticker (col A) + Description (col B), portfolios from C.
    _col_headers(ws, row, portfolios, "Ticker", with_target=False, second_hdr="Description")
    row += 1
    tickers = sorted({it.bare for p in portfolios for it in p.items})
    name_by = {}
    for p in portfolios:
        for it in p.items:
            name_by.setdefault(it.bare, short_instrument_name(it.holding.name or it.bare, 46))
    wmaps = [p.weights() for p in portfolios]
    for i, tk in enumerate(tickers):
        bg = _C["alt"] if i % 2 else _C["white"]
        a = ws.cell(row=row, column=1, value=tk)
        a.font = _font(9, bold=True)
        a.fill = _fill(bg)
        a.border = _border()
        b = ws.cell(row=row, column=2, value=name_by.get(tk, ""))
        b.font = _font(9, color=_C["muted"])
        b.fill = _fill(bg)
        b.border = _border()
        for j, wm in enumerate(wmaps):
            c = ws.cell(row=row, column=_PCOL0 + j,
                        value=(wm[tk] / 100.0 if tk in wm else None))
            c.number_format = "0.0%"
            c.alignment = _align("center")
            c.fill = _fill(bg)
            c.border = _border()
        row += 1
    return row + 1


_METRIC_ROWS = [
    ("CAGR", "cagr", "%"), ("Volatility (ann.)", "volatility", "%"),
    ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""),
    ("Max Drawdown", "max_drawdown", "%"),
    ("VaR 95% (daily)", "var_95", "%"), ("CVaR 95% (daily)", "cvar_95", "%"),
    ("Beta vs S&P 500", "beta", ""), ("Alpha (ann.)", "alpha", "%"),
]


def _metric_value(metrics, key, unit):
    v = (metrics or {}).get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.2f}{unit}"


def _risk_matrix(ws, row, portfolios) -> int:
    ncols = _PCOL0 + len(portfolios) - 1
    w = next((p.window for p in portfolios if getattr(p, "window", None)), None)
    win = f"{w[0]:%Y-%m} → {w[1]:%Y-%m}" if w else "aligned"
    row = _section_header(ws, row, f"Portfolio metrics — single aligned history ({win})", ncols)
    _col_headers(ws, row, portfolios, "Metric", with_target=False)
    row += 1
    for i, (label, key, unit) in enumerate(_METRIC_ROWS):
        bg = _C["alt"] if i % 2 else _C["white"]
        _merge_label(ws, row, label, bold=True, bg=bg)
        for j, p in enumerate(portfolios):
            c = ws.cell(row=row, column=_PCOL0 + j,
                        value=_metric_value(p.metrics_aligned, key, unit))
            c.alignment = _align("center")
            c.fill = _fill(bg)
            c.border = _border()
            c.font = _font(9)
        row += 1
    return row + 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_whatif_excel(path, portfolios, asset_target, geo_target, anchor,
                        tolerance=1.5, sim_rows=None, testfol=None,
                        testfol_byinst=None) -> str:
    """Write the single-sheet, column-per-portfolio comparison workbook."""
    wb = Workbook()
    ws = wb.active
    ws.title = "What-If"
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = _C["band"]

    ncol = _PCOL0 + len(portfolios)  # incl. a possible Target column
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 48
    for j in range(len(portfolios) + 1):
        ws.column_dimensions[get_column_letter(_PCOL0 + j)].width = 14

    # Title.
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    t = ws.cell(row=1, column=1, value="WHAT-IF PORTFOLIO COMPARISON")
    t.font = _font(16, bold=True, color=_C["white"])
    t.fill = _fill(_C["header"])
    ws.row_dimensions[1].height = 28
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
    st = ws.cell(row=2, column=1, value=(
        f"Anchor (real invested, ex-cash): {eur_smart(anchor)}  ·  "
        f"generated {datetime.now():%Y-%m-%d %H:%M}"))
    st.font = _font(9, italic=True, color=_C["muted"])

    row = 4
    row = _specs_block(ws, row, portfolios, anchor)

    funded_labels = [c for c in _ASSET_ORDER if any(p.cap.get(c, 0.0) for p in portfolios)]
    cap_tbl = _alloc_matrix(
        ws, row, "Asset allocation — FUNDED CAPITAL (where euros sit, incl. cash)",
        funded_labels, portfolios, lambda p, l: p.cap.get(l, 0.0),
        target=asset_target, tolerance=tolerance, swatch=asset_class_color)
    row = cap_tbl["last"] + 2

    notl_labels = [c for c in _ASSET_ORDER if any(p.notl_mix.get(c, 0.0) for p in portfolios)]
    notl_tbl = _alloc_matrix(
        ws, row, "Asset allocation — NOTIONAL EXPOSURE (mix, normalised) vs target",
        notl_labels, portfolios, lambda p, l: p.notl_mix.get(l, 0.0),
        target=asset_target, tolerance=tolerance, swatch=asset_class_color)
    row = notl_tbl["last"] + 2

    # Gross exposure per class (leveraged, NOT normalised — sums to > 100%).
    gross_labels = [c for c in _ASSET_ORDER if any(p.notl_gross.get(c, 0.0) for p in portfolios)]
    gross_tbl = _alloc_matrix(
        ws, row, "Asset allocation — NOTIONAL EXPOSURE (gross % of capital, leveraged)",
        gross_labels, portfolios, lambda p, l: p.notl_gross.get(l, 0.0),
        target=None, swatch=asset_class_color)
    row = gross_tbl["last"] + 1
    row = _plain_row(ws, row, "Gross / leverage", portfolios,
                     lambda p: f"{p.gross:.0f}% · {p.leverage:.2f}x", color=_C["amber"])
    row += 2

    # Leverage applied per class = notional − funded (isolates where leverage
    # sits; unlevered legs are 0). Only shown when some portfolio is levered.
    lev_labels = [c for c in _ASSET_ORDER
                  if any(abs(p.lev_by_class.get(c, 0.0)) > 0.05 for p in portfolios)]
    if lev_labels:
        lev_tbl = _alloc_matrix(
            ws, row, "Leverage by class (notional − funded, pp of capital)",
            lev_labels, portfolios, lambda p, l: p.lev_by_class.get(l, 0.0),
            target=None, swatch=asset_class_color)
        row = lev_tbl["last"] + 2

    geo_labels = [c for c in _GEO_ORDER if any(p.geo_notl.get(c, 0.0) for p in portfolios)]
    geo_tbl = _alloc_matrix(
        ws, row, "Equity geography — NOTIONAL (% of equity sleeve) vs target",
        geo_labels, portfolios, lambda p, l: p.geo_notl.get(l, 0.0),
        target=geo_target, tolerance=tolerance, swatch=geo_color)
    row = geo_tbl["last"] + 2

    row = _risk_matrix(ws, row, portfolios)

    _add_chart(ws, cap_tbl, portfolios)
    _robustness_sheet(wb, portfolios)
    if sim_rows:
        _simulation_sheet(wb, sim_rows)
    if testfol:
        _testfol_sheet(wb, testfol, testfol_byinst)

    wb.save(path)
    logger.info("What-if workbook saved to %s", path)
    return path


def _add_chart(ws, cap_tbl, portfolios) -> None:
    """Clustered bar of funded-capital allocation across portfolios + target."""
    cats = Reference(ws, min_col=1, min_row=cap_tbl["first"], max_row=cap_tbl["last"])
    max_col = cap_tbl["tcol"] or (_PCOL0 + len(portfolios) - 1)
    bar = BarChart()
    bar.type = "col"
    bar.title = "Funded capital allocation"
    bar.height = 8
    bar.width = 16
    bar.y_axis.numFmt = "0%"
    data = Reference(ws, min_col=_PCOL0, max_col=max_col,
                     min_row=cap_tbl["header_row"], max_row=cap_tbl["last"])
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)
    anchor_col = get_column_letter(max_col + 2)
    ws.add_chart(bar, f"{anchor_col}4")


def _robustness_sheet(wb, portfolios) -> None:
    """Second sheet: rolling / stress / Monte-Carlo robustness, portfolios as
    columns, for both the actual ~5y history and the synthetic long history."""
    ws = wb.create_sheet("Robustness")
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = _C["amber"]
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 26
    for j in range(len(portfolios)):
        ws.column_dimensions[get_column_letter(_PCOL0 + j)].width = 16
    ncol = _PCOL0 + len(portfolios) - 1

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
    t = ws.cell(row=1, column=1, value="BACKTEST ROBUSTNESS")
    t.font = _font(15, bold=True, color=_C["white"])
    t.fill = _fill(_C["header"])
    ws.row_dimensions[1].height = 24

    def _roll(p, sub, key):
        v = p.rob.get(sub, {}).get(key)
        return "—" if v is None else f"{v * 100:.1f}%"

    def _stress(p, scen, field):
        d = p.rob.get("stress", {}).get(scen, {})
        return "—" if not d.get("covered") else f"{d[field]:.1f}%"

    def _ci(p, metric, dec=0):
        d = p.rob.get("bootstrap", {}).get(metric, {})
        return "—" if not d else f"{d['p05']:.{dec}f} / {d['p95']:.{dec}f}%"

    def block(row, title, rows):
        row = _section_header(ws, row, title, ncol)
        _col_headers(ws, row, portfolios, "Metric", with_target=False)
        row += 1
        for i, (label, fn) in enumerate(rows):
            bg = _C["alt"] if i % 2 else _C["white"]
            _merge_label(ws, row, label, bold=True, bg=bg)
            for j, p in enumerate(portfolios):
                c = ws.cell(row=row, column=_PCOL0 + j, value=fn(p))
                c.alignment = _align("center")
                c.fill = _fill(bg)
                c.border = _border()
                c.font = _font(9)
            row += 1
        return row + 1

    rolling_rows = [
        ("Roll 1Y ret p05", lambda p: _roll(p, "rolling1y", "p05")),
        ("Roll 1Y ret median", lambda p: _roll(p, "rolling1y", "median")),
        ("Roll 1Y ret p95", lambda p: _roll(p, "rolling1y", "p95")),
        ("1Y windows positive", lambda p: (lambda d: "—" if not d else f"{d['pct_positive']:.0f}%")(p.rob.get("rolling1y", {}))),
        ("Roll 3Y ret p05", lambda p: _roll(p, "rolling3y", "p05")),
        ("Roll 3Y ret median", lambda p: _roll(p, "rolling3y", "median")),
        ("Roll 1Y Sharpe min–max", lambda p: (lambda d: "—" if not d else f"{d['min']:.2f}–{d['max']:.2f}")(p.rob.get("sharpe", {}))),
        ("MC CAGR 1Y [p05/p95]", lambda p: _ci(p, "cagr")),
        ("MC MaxDD p05 (worst)", lambda p: (lambda d: "—" if not d else f"{d['p05']:.1f}%")(p.rob.get("bootstrap", {}).get("max_drawdown", {}))),
    ]
    stress_rows = [
        ("Dot-com maxDD", lambda p: _stress(p, "Dot-com 2000-02", "max_drawdown")),
        ("GFC'08 return", lambda p: _stress(p, "GFC 2008", "return")),
        ("GFC'08 maxDD", lambda p: _stress(p, "GFC 2008", "max_drawdown")),
        ("COVID'20 maxDD", lambda p: _stress(p, "COVID 2020", "max_drawdown")),
        ("2022 maxDD", lambda p: _stress(p, "2022 rate shock", "max_drawdown")),
    ]

    row = 3
    row = block(row, "Rolling & Monte-Carlo (aligned history)", rolling_rows)
    row = block(row, "Historical stress scenarios", stress_rows)
    note = ws.cell(row=row, column=1, value=(
        "Single aligned history = per-instrument splice: real fund returns where available, "
        "proxy-reconstructed (geo + leverage financing) before inception. Modeled, USD-based."))
    note.font = _font(8, italic=True, color=_C["muted"])
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncol)


def _simulation_sheet(wb, sim_rows) -> None:
    """Sheet listing, per instrument, the proxy 'simulation' used to backfill
    its pre-inception history (real fund returns take over from 'Real from')."""
    ws = wb.create_sheet("Simulation")
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = _C["green"]
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 60

    ws.merge_cells("A1:D1")
    t = ws.cell(row=1, column=1, value="SIMULATION MAP — per-instrument history reconstruction")
    t.font = _font(14, bold=True, color=_C["white"])
    t.fill = _fill(_C["header"])
    ws.row_dimensions[1].height = 22

    headers = ["Ticker", "Real from", "Base from", "Simulation base (proxy × exposure)"]
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=3, column=j, value=h)
        c.font = _font(9, bold=True, color=_C["white"])
        c.fill = _fill(_C["header"])
        c.border = _border()
    row = 4
    for i, r in enumerate(sim_rows):
        bg = _C["alt"] if i % 2 else _C["white"]
        for j, key in enumerate(("ticker", "real_from", "base_from", "base"), start=1):
            c = ws.cell(row=row, column=j, value=r[key])
            c.fill = _fill(bg)
            c.border = _border()
            c.font = _font(9, bold=(j == 1))
            c.alignment = _align("left" if j in (1, 4) else "center")
        row += 1
    note = ws.cell(row=row + 1, column=1, value=(
        "Recent period uses REAL fund returns from 'Real from'; earlier history is the proxy "
        "base (geo/asset index × exposure, leverage financed at ^IRR+0.5%). Modeled, USD-based."))
    note.font = _font(8, italic=True, color=_C["muted"])
    ws.merge_cells(start_row=row + 1, start_column=1, end_row=row + 1, end_column=4)


def _testfol_sheet(wb, testfol, byinst=None) -> None:
    """Sheet with the testfol.io lines (ticker + allocation %) to paste per
    portfolio. Proxies/leverage syntax follow the KB (SIM tickers, ?L=, CASHX
    for the leverage funding). For each portfolio we show:
      1. a per-instrument breakdown (each fund → the testfol code(s) it maps to,
         so one instrument may expand into several codes, e.g. an efficient-core
         fund → equity + bond − CASHX);
      2. the aggregated lines (codes summed across instruments) + a copy-paste
         one-liner ready to paste into the site."""
    byinst = byinst or {}
    ws = wb.create_sheet("Testfolio")
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = _C["green"]
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 26
    ws.column_dimensions["E"].width = 10

    ws.merge_cells("A1:E1")
    t = ws.cell(row=1, column=1, value="TESTFOL.IO — proxies & weights per portfolio")
    t.font = _font(14, bold=True, color=_C["white"])
    t.fill = _fill(_C["header"])
    ws.row_dimensions[1].height = 22
    ws.merge_cells("A2:E2")
    ws.cell(row=2, column=1, value=(
        "USD proxies (SIM = long backfilled history). Efficient-core funds (NTSG/NTSZ/NTSX) "
        "expand into equity + bond − CASHX (leverage funding); 2x funds (CL2/LVWC) use ?L=. "
        "TER baked as ?E=. One instrument may map to several codes.")
        ).font = _font(8, italic=True, color=_C["muted"])
    ws.merge_cells("A3:E3")
    ws.cell(row=3, column=1, value=(
        "Managed futures (MFEH/DBMFE): blend 50% DBMFSIM + 50% KMLMSIM (the two trend SIMs "
        "diverge a lot — half-and-half cuts model risk; both ~1992+). "
        "Carry (UEQC/CRRY): NO testfol ticker exists (carry ≠ trend); DBMFSIM shown is only a "
        "rough stand-in — the correct method is a custom CSV: CRRY = BNP index BNPIF73P + CASHX − 0.66% "
        "(0.34 TER + 0.32 swap), UEQC = 2.5x BCOMF3T − 2.5x BCOM + CASHX then UBS index from 2015.")
        ).font = _font(8, italic=True, color=_C["amber"])
    ws.merge_cells("A4:E4")
    ws.cell(row=4, column=1, value=(
        "Inception limiters: factor SIMs MTUM/QUAL/VLUE start ~2013 → factor-tilt portfolios "
        "limited to ~2013 on testfol. Leverage cost: CL2 (USA 2x) ≈ TER; LVWC (World 2x) real cost "
        "≈1.6% once div drag is doubled (per Rational Reminder). Always keep CASHX/SP financing.")
        ).font = _font(8, italic=True, color=_C["amber"])

    row = 5
    for name, lines in testfol.items():
        row = _section_header(ws, row, f"{name}", 5)

        # --- 1. Per-instrument breakdown -----------------------------------
        inst_rows = byinst.get(name)
        if inst_rows:
            for c, txt in ((1, "Ticker"), (2, "Instrument"), (3, "Weight"),
                           (4, "testfol code(s)"), (5, "Code wt")):
                h = ws.cell(row=row, column=c, value=txt)
                h.font = _font(9, bold=True, color=_C["white"])
                h.fill = _fill(_C["header"])
                h.border = _border()
            row += 1
            for i, ir in enumerate(inst_rows):
                bg = _C["alt"] if i % 2 else _C["white"]
                codes = ir["codes"] or [("—", 0.0)]
                first = row
                for j, (expr, cw) in enumerate(codes):
                    if j == 0:
                        tk = ws.cell(row=row, column=1, value=ir["ticker"])
                        tk.font = _font(9, bold=True)
                        tk.fill = _fill(bg); tk.border = _border()
                        nm = ws.cell(row=row, column=2, value=ir["name"])
                        nm.font = _font(9); nm.fill = _fill(bg); nm.border = _border()
                        wc = ws.cell(row=row, column=3, value=ir["weight"] / 100.0)
                        wc.number_format = "0.0%"; wc.alignment = _align("center")
                        wc.font = _font(9, bold=True); wc.fill = _fill(bg)
                        wc.border = _border()
                    else:
                        for cc in (1, 2, 3):
                            fillc = ws.cell(row=row, column=cc)
                            fillc.fill = _fill(bg); fillc.border = _border()
                    ec = ws.cell(row=row, column=4, value=expr)
                    ec.font = _font(9, color=_C["green"]); ec.fill = _fill(bg)
                    ec.border = _border()
                    cwc = ws.cell(row=row, column=5, value=cw / 100.0)
                    cwc.number_format = "0.0%"; cwc.alignment = _align("center")
                    cwc.font = _font(9); cwc.fill = _fill(bg); cwc.border = _border()
                    row += 1
                if len(codes) > 1:
                    ws.merge_cells(start_row=first, start_column=1,
                                   end_row=row - 1, end_column=1)
                    ws.merge_cells(start_row=first, start_column=2,
                                   end_row=row - 1, end_column=2)
                    ws.merge_cells(start_row=first, start_column=3,
                                   end_row=row - 1, end_column=3)
            row += 1

        # --- 2. Aggregated lines (paste block) -----------------------------
        sub = ws.cell(row=row, column=1, value="Aggregated (paste-ready)")
        sub.font = _font(9, bold=True, italic=True, color=_C["muted"])
        row += 1
        h1 = ws.cell(row=row, column=1, value="testfol code")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        h2 = ws.cell(row=row, column=5, value="Alloc")
        for c in (h1, h2):
            c.font = _font(9, bold=True, color=_C["white"])
            c.fill = _fill(_C["header"])
            c.border = _border()
        row += 1
        total = 0.0
        for i, (tk, w) in enumerate(lines):
            bg = _C["alt"] if i % 2 else _C["white"]
            a = ws.cell(row=row, column=1, value=tk)
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
            a.font = _font(9, bold=True)
            a.fill = _fill(bg)
            a.border = _border()
            b = ws.cell(row=row, column=5, value=w / 100.0)
            b.number_format = "0.0%"
            b.alignment = _align("center")
            b.fill = _fill(bg)
            b.border = _border()
            total += w
            row += 1
        tot = ws.cell(row=row, column=1, value="Total")
        tot.font = _font(9, bold=True, color=_C["muted"])
        tc = ws.cell(row=row, column=5, value=total / 100.0)
        tc.number_format = "0.0%"
        tc.alignment = _align("center")
        tc.font = _font(9, bold=True, color=_C["muted"])
        row += 1
        # Copy-paste one-liner.
        oneliner = "  ".join(f"{tk} {w:g}" for tk, w in lines)
        lbl = ws.cell(row=row, column=1, value="Paste string")
        lbl.font = _font(8, bold=True, color=_C["muted"])
        cp = ws.cell(row=row, column=2, value=oneliner)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=5)
        cp.font = _font(8, color=_C["muted"])
        cp.alignment = _align("left")
        row += 2