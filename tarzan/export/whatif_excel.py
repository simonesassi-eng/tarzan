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
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
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

# High-contrast, colour-blind-friendlier series palette for charts (indigo,
# red, green, amber, cyan, pink, violet, slate). Distinct hues + brightness so
# adjacent portfolio lines/bars are easy to tell apart.
_CHART_COLORS = ["5B5BD6", "DC2626", "16A34A", "D97706",
                 "0891B2", "DB2777", "7C3AED", "475569"]


def _style_line_series(chart) -> None:
    """Give each series a distinct solid colour and a readable line width."""
    for i, s in enumerate(chart.series):
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        s.graphicalProperties = GraphicalProperties()
        s.graphicalProperties.line = LineProperties(solidFill=color, w=26000)  # ~2pt
        s.smooth = False


def _style_bar_series(chart) -> None:
    """Give each bar series a distinct solid fill from the palette."""
    for i, s in enumerate(chart.series):
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        s.graphicalProperties = GraphicalProperties(solidFill=color)


def _show_axes(chart, x_title: str, y_title: str) -> None:
    """Force both axes (and their titles/tick labels) to render — openpyxl
    leaves ``delete`` unset, which makes Excel hide the axes entirely so the
    chart looks empty. Setting delete=False restores the scale and labels."""
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.x_axis.tickLblPos = "low"
    chart.y_axis.majorGridlines = chart.y_axis.majorGridlines  # keep y gridlines
    if chart.legend is not None:
        chart.legend.position = "b"                            # legend at bottom

from tarzan.models.taxonomy import ORDER_WHATIF as _ORDER_WHATIF, GEO_ORDER as _GEO_REG

_ASSET_ORDER = list(_ORDER_WHATIF)
# The what-if workbook also renders an explicit "Other" geo bucket at the end.
_GEO_ORDER = list(_GEO_REG) + ["Other"]

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
    # Shared primitive — same normalisation the metrics engine and backtest use.
    from tarzan.engine.allocations import renorm
    return renorm(d)


def _dev_color(delta, tol):
    if delta is None or delta != delta or tol <= 0:
        return _C["text"]
    a = abs(delta)
    if a <= tol:
        return _C["green"]
    if a <= 2 * tol:
        return _C["amber"]
    return _C["red"]


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


def _instrument_matrix_block(ws, row, portfolios) -> int:
    """Per-instrument weight matrix, grouped by asset class + role (sub-class) —
    the same layout as the newsletter's "Instruments × portfolios" block."""
    ncols = _PCOL0 + len(portfolios) - 1
    row = _section_header(ws, row, "Instruments \u00d7 portfolios", ncols)
    _col_headers(ws, row, portfolios, "Ticker", with_target=False, second_hdr="Description")
    row += 1

    reps: dict = {}
    for p in portfolios:
        for it in p.items:
            reps.setdefault(it.bare, it)

    def _cls(it):
        return (it.holding.asset_class.value
                if getattr(it.holding, "asset_class", None) else "Other")

    def _role(it):
        return getattr(it.holding, "role", "") or "\u2014"

    by_cls: dict = {}
    for it in reps.values():
        by_cls.setdefault(_cls(it), []).append(it)
    ordered = [c for c in _ASSET_ORDER if c in by_cls] + \
              [c for c in by_cls if c not in _ASSET_ORDER]
    name_by = {b: short_instrument_name(it.holding.name or b, 46)
               for b, it in reps.items()}
    wmaps = [p.weights() for p in portfolios]

    i = 0
    for cls in ordered:
        color = asset_class_color(cls)
        # Class header spanning the full width, coloured swatch via font colour.
        ch = _merge_label(ws, row, cls, bold=True, color=color, bg=_C["alt"])
        for j in range(len(portfolios)):
            cc = ws.cell(row=row, column=_PCOL0 + j)
            cc.fill = _fill(_C["alt"])
            cc.border = _border()
        row += 1
        roles: dict = {}
        for it in by_cls[cls]:
            roles.setdefault(_role(it), []).append(it)
        for role, its in roles.items():
            if role and role != "\u2014":
                r = ws.cell(row=row, column=1, value=f"   {role}")
                r.font = _font(8, italic=True, color=_C["muted"])
                ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
                row += 1
            for it in sorted(its, key=lambda x: x.bare):
                bg = _C["alt"] if i % 2 else _C["white"]
                i += 1
                a = ws.cell(row=row, column=1, value=it.bare)
                a.font = _font(9, bold=True)
                a.fill = _fill(bg)
                a.border = _border()
                b = ws.cell(row=row, column=2, value=name_by.get(it.bare, ""))
                b.font = _font(9, color=_C["muted"])
                b.fill = _fill(bg)
                b.border = _border()
                for j, wm in enumerate(wmaps):
                    c = ws.cell(row=row, column=_PCOL0 + j,
                                value=(wm[it.bare] / 100.0 if it.bare in wm else None))
                    c.number_format = "0.0%"
                    c.alignment = _align("center")
                    c.fill = _fill(bg)
                    c.border = _border()
                row += 1
    return row + 1


# Same metric set as the newsletter "Portfolio risk metrics" block (incl. Ulcer).
_METRIC_ROWS = [
    ("CAGR", "cagr", "%"), ("Volatility (ann.)", "volatility", "%"),
    ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""),
    ("Max Drawdown", "max_drawdown", "%"), ("Ulcer index", "ulcer_index", "%"),
    ("VaR 95% (daily)", "var_95", "%"), ("CVaR 95% (daily)", "cvar_95", "%"),
    ("Beta vs S&P 500", "beta", ""), ("Alpha (ann.)", "alpha", "%"),
]


def _inject_ulcer(portfolios) -> None:
    """Ensure each portfolio's aligned metrics carry an Ulcer index (computed
    from its NAV, currency-independent) — mirrors the newsletter block."""
    from tarzan.engine.stats import compute_ulcer_index
    for p in portfolios:
        nav = getattr(p, "nav", None)
        if nav is None:
            continue
        for attr in ("metrics_aligned_eur", "metrics_aligned_usd"):
            m = getattr(p, attr, None)
            if isinstance(m, dict) and "ulcer_index" not in m:
                try:
                    m["ulcer_index"] = compute_ulcer_index(nav)
                except Exception:  # noqa: BLE001
                    m["ulcer_index"] = None


def _metric_value(metrics, key, unit):
    v = (metrics or {}).get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.2f}{unit}"


def _metrics_block(ws, row, portfolios, attr, ccy_label) -> int:
    """Render one currency block of the metrics matrix (EUR or USD)."""
    ncols = _PCOL0 + len(portfolios) - 1
    rf = next(((getattr(p, attr, {}) or {}).get("risk_free")
               for p in portfolios if getattr(p, attr, None)), None)
    rf_s = f"  ·  risk-free {rf:.2f}%" if isinstance(rf, (int, float)) else ""
    _merge_label(ws, row, f"{ccy_label}{rf_s}", bold=True, color=_C["band"])
    row += 1
    for i, (label, key, unit) in enumerate(_METRIC_ROWS):
        bg = _C["alt"] if i % 2 else _C["white"]
        _merge_label(ws, row, label, bold=True, bg=bg)
        for j, p in enumerate(portfolios):
            c = ws.cell(row=row, column=_PCOL0 + j,
                        value=_metric_value(getattr(p, attr, {}), key, unit))
            c.alignment = _align("center")
            c.fill = _fill(bg)
            c.border = _border()
            c.font = _font(9)
        row += 1
    return row


def _risk_matrix(ws, row, portfolios) -> int:
    _inject_ulcer(portfolios)
    ncols = _PCOL0 + len(portfolios) - 1
    w = next((p.window for p in portfolios if getattr(p, "window", None)), None)
    win = f"{w[0]:%Y-%m} → {w[1]:%Y-%m}" if w else "aligned"
    row = _section_header(ws, row, f"Portfolio metrics — single aligned history ({win})", ncols)
    _col_headers(ws, row, portfolios, "Metric", with_target=False)
    row += 1
    row = _metrics_block(ws, row, portfolios, "metrics_aligned_eur", "EUR numeraire (unhedged)")
    row += 1
    row = _metrics_block(ws, row, portfolios, "metrics_aligned_usd", "USD numeraire")
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
        f"Notional anchor (display only): {eur_smart(anchor)}  ·  "
        f"generated {datetime.now():%Y-%m-%d %H:%M}"))
    st.font = _font(9, italic=True, color=_C["muted"])

    # The workbook mirrors the newsletter "Backtesting" section — same
    # components, no more, no less (plus the testfol tab): (1) instruments ×
    # portfolios, (2) diversification (notional asset class + gross/leverage +
    # leverage-by-class + geography), (3) portfolio risk metrics EUR/USD,
    # (4) robustness sheet, (5) simulation sheet.
    row = 4
    row = _instrument_matrix_block(ws, row, portfolios)

    # --- Diversification: asset class NOTIONAL exposure (gross, leverage-aware),
    # coloured by drift vs target — like the newsletter's "By asset class".
    gross_labels = [c for c in _ASSET_ORDER if any(p.notl_gross.get(c, 0.0) for p in portfolios)]
    gross_tbl = _alloc_matrix(
        ws, row, "Diversification — by asset class (NOTIONAL, % of capital) vs target",
        gross_labels, portfolios, lambda p, l: p.notl_gross.get(l, 0.0),
        target=asset_target, tolerance=tolerance, swatch=asset_class_color)
    row = gross_tbl["last"] + 1
    tgt_gross = sum(v for v in (asset_target or {}).values() if v) or None
    row = _plain_row(ws, row, "Gross exposure", portfolios,
                     lambda p: f"{p.gross:.0f}%")
    row = _plain_row(ws, row, "Leverage", portfolios,
                     lambda p: f"{p.leverage:.2f}x", color=_C["amber"])
    row += 1

    # Leverage by class = notional / funded capital in each class (>1.00x =
    # partly synthetic). Shown as a ratio (like the newsletter), only when some
    # class is levered.
    def _lev(p, c):
        return (p.notl_gross.get(c, 0.0) / p.cap[c]) if p.cap.get(c, 0.0) > 0 else None
    lev_labels = [c for c in _ASSET_ORDER
                  if any((_lev(p, c) or 0) > 1.001 for p in portfolios)]
    if lev_labels:
        ncols = _PCOL0 + len(portfolios) - 1
        row = _section_header(ws, row, "Diversification — leverage by class (notional / funded)", ncols)
        _col_headers(ws, row, portfolios, "Category", with_target=False)
        row += 1
        for k, cls in enumerate(lev_labels):
            bg = _C["alt"] if k % 2 else _C["white"]
            _merge_label(ws, row, cls, bold=True, color=asset_class_color(cls), bg=bg)
            for j, p in enumerate(portfolios):
                lv = _lev(p, cls)
                c = ws.cell(row=row, column=_PCOL0 + j,
                            value=("\u2014" if lv is None else f"{lv:.2f}x"))
                c.alignment = _align("center")
                c.fill = _fill(bg)
                c.border = _border()
                c.font = _font(9)
            row += 1
        row += 1

    geo_labels = [c for c in _GEO_ORDER if any(p.geo_notl.get(c, 0.0) for p in portfolios)]
    geo_tbl = _alloc_matrix(
        ws, row, "Diversification — by geography (NOTIONAL, % of equity sleeve) vs target",
        geo_labels, portfolios, lambda p, l: p.geo_notl.get(l, 0.0),
        target=geo_target, tolerance=tolerance, swatch=geo_color)
    row = geo_tbl["last"] + 2

    row = _risk_matrix(ws, row, portfolios)

    _robustness_sheet(wb, portfolios)
    if sim_rows:
        _simulation_sheet(wb, sim_rows)
    if testfol:
        _testfol_sheet(wb, testfol, testfol_byinst)

    wb.save(path)
    logger.info("What-if workbook saved to %s", path)
    return path


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

    # Plain-language legend so the metrics are self-explanatory.
    row = _section_header(ws, row, "How to read these metrics", ncol)
    legend = [
        ("Rolling 1Y / 3Y return (p05 · median · p95)",
         "Annualised return over EVERY rolling 1- or 3-year window in the history. "
         "p05 = unlucky start (5th percentile), median = typical, p95 = lucky start. "
         "A wide spread means the outcome depends a lot on WHEN you start."),
        ("1Y windows positive",
         "Share of all rolling 1-year windows that ended in positive territory."),
        ("Rolling 1Y Sharpe min–max",
         "Worst-to-best annualised Sharpe over any rolling 1-year window (risk-adjusted consistency)."),
        ("MC CAGR 1Y [p05 / p95]",
         "Monte-Carlo (block bootstrap, 2000 paths): 90% confidence band for next-year CAGR. "
         "Blocks keep momentum / volatility-clustering, unlike a naive random resample."),
        ("MC MaxDD p05 (worst)",
         "Bad-case (5th-percentile) max drawdown across the Monte-Carlo paths — a plausible worst year."),
        ("Historical stress maxDD / return",
         "Actual peak-to-trough drawdown (and total return) the portfolio WOULD have suffered in each "
         "real crisis window, using the reconstructed history."),
    ]
    for i, (term, desc) in enumerate(legend):
        bg = _C["alt"] if i % 2 else _C["white"]
        a = ws.cell(row=row, column=1, value=term)
        a.font = _font(9, bold=True); a.fill = _fill(bg); a.border = _border()
        a.alignment = _align("left", "top")
        b = ws.cell(row=row, column=2, value=desc)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=ncol)
        b.font = _font(9, color=_C["muted"]); b.fill = _fill(bg); b.border = _border()
        b.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        ws.row_dimensions[row].height = 42
        row += 1
    row += 1

    row = _robustness_charts(ws, portfolios, row, ncol)

    try:
        from tarzan.data import proxy_data as _pd
        _ccy = getattr(_pd, "_TARGET_CCY", "EUR")
    except Exception:  # noqa: BLE001
        _ccy = "EUR"
    note = ws.cell(row=row, column=1, value=(
        "Single aligned history = per-instrument splice: real fund returns where available, "
        f"proxy-reconstructed (geo + leverage financing) before inception. Modeled, {_ccy}-based; "
        "equity proxies are total-return (^SP500TR), commodities collateralised, "
        "managed futures = RYMFX/AQMNX or custom SG-CTA CSV. Net of each fund's TER on "
        "the modeled base (real returns already net); periodic rebalancing (CLI "
        "--rebalance, default quarterly), costless. Sharpe/Sortino use a currency-matched, "
        "TIME-VARYING daily risk-free (USD = ^IRX T-bill; EUR = ECB AAA-govt 3M spot via "
        "SDMX); the header % is the window average of that path."))
    note.font = _font(8, italic=True, color=_C["muted"])
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncol)


def _robustness_charts(ws, portfolios, start_row, ncol) -> int:
    """Line charts from the aligned monthly NAV: growth-of-100, drawdown, and
    rolling 5Y / 10Y annualised return. Source data lives on a hidden helper
    SHEET (not hidden columns — Excel refuses to plot hidden cells, which is why
    the earlier version showed blank charts)."""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return start_row
    navs = {p.name: p.nav[p.nav > 0] for p in portfolios
            if getattr(p, "nav", None) is not None and len(p.nav) >= 30}
    if not navs:
        return start_row
    daily = pd.DataFrame(navs).dropna(how="all").ffill()
    if daily.empty or daily.shape[0] < 30:
        return start_row
    cols = list(daily.columns)
    n = len(cols)

    def _roll_ann(df, win):
        r = (df / df.shift(win)) ** (252.0 / win) - 1.0
        return r * 100.0

    growth = (daily / daily.iloc[0] * 100.0).iloc[::21]
    dd = ((daily / daily.cummax() - 1.0) * 100.0).iloc[::21]
    r5 = _roll_ann(daily, 5 * 252).iloc[::21].dropna(how="all")
    r10 = _roll_ann(daily, 10 * 252).iloc[::21].dropna(how="all")

    # Dedicated hidden data sheet (charts still plot from a hidden sheet).
    wb = ws.parent
    data_title = "ChartData"
    dws = wb[data_title] if data_title in wb.sheetnames else wb.create_sheet(data_title)

    anchor_col = 1

    def _write_block(frame, title):
        """Write [Date | names...] starting at the next free column; return the
        (first_data_col, n, header_row, last_row) needed to build References."""
        nonlocal anchor_col
        c0 = anchor_col
        dws.cell(row=1, column=c0, value=title)
        dws.cell(row=2, column=c0, value="Date")
        for j, name in enumerate(cols):
            dws.cell(row=2, column=c0 + 1 + j, value=name)
        for i, (idx, r) in enumerate(frame.iterrows(), start=3):
            dws.cell(row=i, column=c0, value=idx.to_pydatetime().date())
            for j, name in enumerate(cols):
                v = r[name]
                dws.cell(row=i, column=c0 + 1 + j,
                         value=None if pd.isna(v) else round(float(v), 2))
        last = frame.shape[0] + 2
        anchor_col = c0 + n + 2          # leave a gap before the next block
        return c0, last

    def _line(frame, title, y_title, at_cell, logscale=False):
        c0, last = _write_block(frame, title)
        ch = LineChart()
        ch.title = title
        ch.height, ch.width = 9, 22
        ch.x_axis.number_format = "yyyy"
        ch.x_axis.majorTimeUnit = "years"
        if logscale:
            ch.y_axis.scaling.logBase = 10
        data = Reference(dws, min_col=c0 + 1, max_col=c0 + n, min_row=2, max_row=last)
        cats = Reference(dws, min_col=c0, min_row=3, max_row=last)
        ch.add_data(data, titles_from_data=True)
        ch.set_categories(cats)
        _style_line_series(ch)                 # distinct colours + line width
        _show_axes(ch, "Year", y_title)        # force axes/titles to render
        ws.add_chart(ch, at_cell)

    _line(growth, "Growth of 100 — aligned history (monthly)", "Index (start = 100)",
          f"A{start_row}")
    _line(dd, "Drawdown (underwater), monthly", "Drawdown %", f"A{start_row + 19}")
    _line(r5, "Rolling 5-year return (annualised)", "Ann. return %", f"A{start_row + 38}")
    _line(r10, "Rolling 10-year return (annualised)", "Ann. return %", f"A{start_row + 57}")

    dws.sheet_state = "hidden"
    return start_row + 76


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
        "base (geo/asset index × exposure, leverage financed at ^IRX+0.5%, net of TER). Proxies "
        "are total-return, converted to the reporting currency; modeled, not an exact replication. "
        "Factor ETFs additionally carry a factor tilt (Developed FF SMB/HML/RMW/MOM loadings, shown in the "
        "base column) so their value/momentum/quality tilt is preserved pre-inception."))
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