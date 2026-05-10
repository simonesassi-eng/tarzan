"""Generate a professional multi-sheet Excel dashboard using openpyxl.

This module is the Reporting layer. It consumes a PortfolioMetrics object
and produces a formatted Excel workbook with charts, tables, and documentation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import PieChart, BarChart, LineChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.series import DataPoint, SeriesLabel
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan import config as cfg

logger = logging.getLogger(__name__)

ASSET_CLASS_COLORS = cfg.asset_class_colors()
SHEET_NAMES = cfg.sheet_names()


# ── STYLING ───────────────────────────────────────────

C = {
    'bg_page':   'F7F8FC',
    'bg_header': '1E293B',
    'bg_card':   'FFFFFF',
    'bg_alt':    'F8FAFF',
    'text_pri':  '1E293B',
    'text_sec':  '64748B',
    'text_wht':  'FFFFFF',
    'accent':    '5B5BD6',
    'green':     '16A34A',
    'red':       'DC2626',
    'amber':     'D97706',
    'border':    'CBD5E1',
    'border_dk': '94A3B8',
}

ASSET_COLORS = {
    'Equities': '1D4ED8', 'Fixed Income': 'A16207',
    'Cash & Cash Equivalents': '15803D', 'Gold': 'CA8A04', 'Commodities': 'C2410C', 'Alternative': '7C3AED',
}

GEO_COLORS = {
    'USA': '1D4ED8', 'Eurozone EMU': 'A16207', 'Dev ex-USA ex-EMU ex-JP': '15803D',
    'Emerging Markets': 'C2410C', 'Japan': '7C3AED',
}

TAB_COLORS = {
    'Dashboard': '5B5BD6', 'Holdings': '1E293B', 'Optimizer': '16A34A',
    'Performance': '2563EB', 'Return Contribution': 'D97706',
}

# KPI value color map
KPI_COLORS = {
    'Total Value': C['accent'], 'Sharpe': C['accent'], 'Sortino': C['accent'], 'Beta': C['accent'],
    'Total Gain': C['green'], 'CAGR': C['green'], 'HPR': C['green'], 'Alpha': C['green'],
    'Max Drawdown': C['red'], 'VaR': C['red'], 'CVaR': C['red'],
    'Volatility': C['amber'],
}


def px_fill(c):
    return PatternFill('solid', fgColor=c)


def px_font(size=10, bold=False, color='1E293B', italic=False):
    return Font(name='Calibri', size=size, bold=bold, color=color, italic=italic)


def px_align(h='left', v='center', wrap=False):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def px_border():
    s = Side(style='thin', color='CBD5E1')
    return Border(left=s, right=s, top=s, bottom=s)


def px_border_hdr():
    s = Side(style='thin', color='94A3B8')
    return Border(left=s, right=s, top=s, bottom=s)


def px_no_border():
    return Border()


def _kpi_color(label):
    """Return the hex color for a KPI label based on KPI_COLORS map."""
    for key, color in KPI_COLORS.items():
        if key.lower() in label.lower():
            return color
    return C['accent']


def _num_color(val):
    """Return green or red hex color based on sign."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return C['text_pri']
    return C['green'] if val >= 0 else C['red']


def _deviation_color(delta_pct, tolerance):
    """Return a traffic-light color based on how far Actual% deviates from Target%.

    Args:
        delta_pct: Actual - Target (percentage points, signed).
        tolerance: Alert threshold in percentage points (from config.rebalancing_threshold).

    Returns:
        Green if |delta| <= tolerance, amber if within 2× tolerance, red beyond.
        Neutral text color if delta or tolerance are unavailable.
    """
    if delta_pct is None or (isinstance(delta_pct, float) and pd.isna(delta_pct)):
        return C['text_pri']
    if tolerance is None or tolerance <= 0:
        return C['text_pri']
    abs_delta = abs(delta_pct)
    if abs_delta <= tolerance:
        return C['green']
    if abs_delta <= 2 * tolerance:
        return C['amber']
    return C['red']


def _apply_title(ws, row, col, text):
    """LIVELLO 1 — Sheet title."""
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = px_font(size=16, bold=True, color=C['text_pri'])
    cell.fill = px_fill(C['bg_page'])
    cell.alignment = px_align(h='left')
    cell.border = px_no_border()


def _apply_subtitle(ws, row, col, text):
    """LIVELLO 2 — Section subtitle."""
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = px_font(size=12, bold=True, color=C['text_pri'])
    cell.fill = px_fill(C['bg_page'])
    cell.alignment = px_align(h='left')
    cell.border = px_no_border()


def _apply_header(ws, row, col, text):
    """LIVELLO 3 — Table header cell."""
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = px_font(size=10, bold=True, color=C['text_wht'])
    cell.fill = px_fill(C['bg_header'])
    cell.alignment = px_align(h='center')
    cell.border = px_border_hdr()


def _data_fill(table_idx):
    """Alternating row fill based on table-relative index (0-based)."""
    return px_fill(C['bg_card']) if table_idx % 2 == 0 else px_fill(C['bg_alt'])


def _write_data_cell(ws, row, col, value, table_idx, is_number=False, bold=False,
                     asset_class=None, geography=None, num_fmt=None, font_color=None):
    """Write a data cell with proper Clean Premium styling."""
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = _data_fill(table_idx)
    cell.border = px_border()

    if font_color is not None:
        cell.font = px_font(size=10, bold=bold or is_number, color=font_color)
        cell.alignment = px_align(h='center' if is_number else 'left')
    elif asset_class and asset_class in ASSET_COLORS:
        cell.font = px_font(size=10, bold=True, color=ASSET_COLORS[asset_class])
        cell.alignment = px_align(h='left')
    elif geography and geography in GEO_COLORS:
        cell.font = px_font(size=10, bold=True, color=GEO_COLORS[geography])
        cell.alignment = px_align(h='left')
    elif is_number:
        color = _num_color(value)
        cell.font = px_font(size=10, bold=bold, color=color)
        cell.alignment = px_align(h='center')
    else:
        cell.font = px_font(size=10, bold=bold, color=C['text_pri'])
        cell.alignment = px_align(h='left')

    # Apply number format: explicit > auto-detect for numeric cells
    if num_fmt:
        cell.number_format = num_fmt
    elif is_number and isinstance(value, (int, float)):
        cell.number_format = '0.00'
    return cell


def _write_portfolio_row(ws, row, col_start, label, values, table_idx, num_fmt='0.00'):
    """Write a TOTAL PORTFOLIO / YOUR PORTFOLIO styled row."""
    portfolio_fill = px_fill('EEF2FF')
    cell = ws.cell(row=row, column=col_start, value=label)
    cell.font = px_font(size=10, bold=True, color=C['accent'])
    cell.fill = portfolio_fill
    cell.border = px_border()
    cell.alignment = px_align(h='left')
    for i, val in enumerate(values):
        c = ws.cell(row=row, column=col_start + 1 + i, value=val)
        c.fill = portfolio_fill
        c.border = px_border()
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            color = _num_color(val) if isinstance(val, (int, float)) else C['text_pri']
            c.font = px_font(size=10, bold=True, color=color)
            if isinstance(val, (int, float)) and num_fmt:
                c.number_format = num_fmt
        else:
            c.font = px_font(size=10, bold=True, color=C['text_pri'])
        c.alignment = px_align(h='center')


def _write_kpi_pair(ws, row, col, label, value, kpi_label_hint=None):
    """Write a KPI card pair: label cell + value cell."""
    # Label cell
    lc = ws.cell(row=row, column=col, value=label)
    lc.font = px_font(size=10, bold=True, color=C['text_sec'])
    lc.fill = px_fill(C['bg_alt'])
    lc.alignment = px_align(h='left')
    lc.border = px_border()
    # Value cell
    vc = ws.cell(row=row, column=col + 1, value=value)
    kpi_color = _kpi_color(kpi_label_hint or label)
    vc.font = px_font(size=14, bold=True, color=kpi_color)
    vc.fill = px_fill(C['bg_card'])
    vc.alignment = px_align(h='center')
    vc.border = px_border()
    if isinstance(value, (int, float)):
        # EUR values get comma format, percentages/ratios get 0.00
        if 'EUR' in label or 'Value' in label or 'Gain' in label:
            vc.number_format = '#,##0.00'
        else:
            vc.number_format = '0.00'


def _write_footer(ws, row, col):
    """Write the generation timestamp footer."""
    cell = ws.cell(row=row, column=col,
                   value=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} v2.0")
    cell.font = px_font(size=8, italic=True, color=C['text_sec'])


def _format_number(val, is_pct=False):
    """Format a number for display, return string or None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    if is_pct:
        return round(val, 2)
    return round(val, 2) if isinstance(val, float) else val


# ── END STYLING ───────────────────────────────────────


def _make_pie(title, sheet, cat_col, val_col, start_row, end_row, color_map, width=15, height=10):
    """Create a bulletproof pie chart with % labels and proper sizing."""
    chart = PieChart()
    chart.title = title
    chart.width = width
    chart.height = height
    chart.style = 10

    cats = Reference(sheet, min_col=cat_col, min_row=start_row, max_row=end_row)
    vals = Reference(sheet, min_col=val_col, min_row=start_row, max_row=end_row)
    chart.add_data(vals, titles_from_data=False)
    chart.set_categories(cats)

    # Data labels: show category name + percentage, no value
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showCatName = True
    chart.dataLabels.showPercent = True
    chart.dataLabels.showVal = False
    chart.dataLabels.showSerName = False
    chart.dataLabels.showLeaderLines = True
    chart.dataLabels.separator = "\n"

    # Hide legend (labels on chart are enough)
    chart.legend = None

    # Apply colors from map
    n_rows = end_row - start_row + 1
    for i in range(n_rows):
        cat_cell = sheet.cell(row=start_row + i, column=cat_col).value
        color = color_map.get(cat_cell, '808080')
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = color
        chart.series[0].data_points.append(pt)

    return chart


def _make_bar(title, sheet, cat_col, series_defs, start_row, end_row, width=18, height=12):
    """Create a bulletproof bar chart. series_defs = [(col, label), ...]."""
    chart = BarChart()
    chart.type = "bar"
    chart.title = title
    chart.width = width
    chart.height = height
    chart.style = 10

    cats = Reference(sheet, min_col=cat_col, min_row=start_row, max_row=end_row)
    for col, label in series_defs:
        vals = Reference(sheet, min_col=col, min_row=start_row, max_row=end_row)
        chart.add_data(vals, titles_from_data=False)
        chart.series[-1].tx = SeriesLabel(v=label)
    chart.set_categories(cats)

    return chart



def generate_excel(
    metrics: PortfolioMetrics,
    holdings: list[Holding],
    config: InvestorConfig,
    output_dir: str,
) -> str:
    """Generate the multi-sheet Excel dashboard.

    Args:
        metrics: Computed portfolio metrics.
        holdings: List of enriched holdings.
        config: Investor configuration.
        output_dir: Directory for the output file.

    Returns:
        Path to the generated Excel file.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = os.path.join(output_dir, f"portfolio_dashboard_{date_str}.xlsx")

    workbook = Workbook()
    try:
        # Create sheets (first sheet is created by default)
        sheets = {}
        for i, name in enumerate(SHEET_NAMES):
            if i == 0:
                ws = workbook.active
                ws.title = name
            else:
                ws = workbook.create_sheet(title=name)
            sheets[name] = ws
            # Global settings per sheet
            ws.sheet_view.showGridLines = False
            if name in TAB_COLORS:
                ws.sheet_properties.tabColor = TAB_COLORS[name]

        _write_dashboard(workbook, sheets["Dashboard"], metrics, config)
        _write_allocations(workbook, sheets["Optimizer"], metrics, config)
        _write_holdings(workbook, sheets["Holdings"], metrics)
        _write_performance(workbook, sheets["Performance"], metrics)
        _write_analysis(workbook, sheets["Return Contribution"], metrics)

        _set_column_widths(sheets)

        workbook.save(filepath)
        logger.info("Excel dashboard written to %s", filepath)
    except Exception as e:
        logger.error("Error generating Excel: %s", e)
        raise

    return filepath


def _set_column_widths(sheets: dict) -> None:
    """Set consistent column widths for all sheets."""
    s = sheets

    s["Dashboard"].column_dimensions['A'].width = 40
    s["Dashboard"].column_dimensions['B'].width = 16
    s["Dashboard"].column_dimensions['C'].width = 14
    s["Dashboard"].column_dimensions['D'].width = 3
    s["Dashboard"].column_dimensions['E'].width = 22
    s["Dashboard"].column_dimensions['F'].width = 16
    s["Dashboard"].column_dimensions['G'].width = 14
    for col_letter in ['H', 'I', 'J', 'K']:
        s["Dashboard"].column_dimensions[col_letter].width = 14

    s["Holdings"].column_dimensions['A'].width = 35
    s["Holdings"].column_dimensions['B'].width = 12
    s["Holdings"].column_dimensions['C'].width = 14
    s["Holdings"].column_dimensions['D'].width = 22
    s["Holdings"].column_dimensions['E'].width = 16
    s["Holdings"].column_dimensions['F'].width = 8
    for col_letter in ['G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']:
        s["Holdings"].column_dimensions[col_letter].width = 12
    s["Holdings"].column_dimensions['O'].width = 10
    for col_letter in ['P', 'Q', 'R']:
        s["Holdings"].column_dimensions[col_letter].width = 16

    s["Optimizer"].column_dimensions['A'].width = 45
    s["Optimizer"].column_dimensions['B'].width = 12
    s["Optimizer"].column_dimensions['C'].width = 14
    s["Optimizer"].column_dimensions['D'].width = 12
    s["Optimizer"].column_dimensions['E'].width = 22

    s["Performance"].column_dimensions['A'].width = 35
    for i in range(2, 15):
        s["Performance"].column_dimensions[get_column_letter(i)].width = 10

    s["Return Contribution"].column_dimensions['A'].width = 35
    for i in range(2, 9):
        s["Return Contribution"].column_dimensions[get_column_letter(i)].width = 14


# ---------------------------------------------------------------------------
# Sheet writers
# ---------------------------------------------------------------------------



def _write_dashboard(workbook, sheet, metrics: PortfolioMetrics, config: InvestorConfig):
    """Dashboard: Hero (Value/Gain/RTD) + Allocation + Top 5 + Rebalancing alert."""

    # --- Dates ---
    inception_str = config.portfolio_inception or ""
    as_of = datetime.now().strftime("%d %b %Y")

    # --- KPI values ---
    total_cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - total_cost
    rtd = (total_gain / total_cost * 100) if total_cost > 0 else 0.0

    # --- Column widths ---
    sheet.column_dimensions['A'].width = 22
    sheet.column_dimensions['B'].width = 16
    sheet.column_dimensions['C'].width = 14
    sheet.column_dimensions['D'].width = 3
    sheet.column_dimensions['E'].width = 22
    sheet.column_dimensions['F'].width = 16
    sheet.column_dimensions['G'].width = 14
    sheet.column_dimensions['H'].width = 12

    # --- Title ---
    _apply_title(sheet, 1, 1, "Portfolio Dashboard")
    sub = sheet.cell(row=2, column=1, value=f"As of: {as_of}")
    sub.font = px_font(size=10, italic=True, color=C['text_sec'])
    sub.fill = px_fill(C['bg_page'])
    sub.border = px_no_border()

    # HERO - Portfolio Status
    row = 4
    inception_label = f" (since {inception_str})" if inception_str else ""
    _write_area_header(sheet, row, 1, 8, f"PORTFOLIO STATUS{inception_label}")
    row += 1

    hero_data = [
        ("Total Value (EUR)", metrics.total_value, None, "number"),
        ("Total Gain (EUR)", total_gain, total_gain, "number_signed"),
        ("RTD (%)", rtd, total_gain, "number_signed"),
    ]
    for ti, (label, value, gain_for_color, kind) in enumerate(hero_data):
        lcell = sheet.cell(row=row, column=1, value=label)
        lcell.font = px_font(size=10, color=C['text_sec'])
        lcell.fill = _data_fill(ti)
        lcell.border = px_border()
        lcell.alignment = px_align(h='left')
        color = C['text_pri']
        if gain_for_color is not None and isinstance(gain_for_color, (int, float)):
            color = C['green'] if gain_for_color >= 0 else C['red']
        vcell = sheet.cell(row=row, column=2, value=value)
        vcell.font = px_font(size=11, bold=True, color=color)
        vcell.fill = _data_fill(ti)
        vcell.border = px_border()
        vcell.alignment = px_align(h='right')
        if kind == "number":
            vcell.number_format = '#,##0.00'
        elif kind == "number_signed":
            vcell.number_format = '+#,##0.00;-#,##0.00;0.00'
        row += 1

    # ALLOCATION
    row += 2
    _write_area_header(sheet, row, 1, 8, "ALLOCATION")
    row += 1

    ac_targets = {}
    geo_targets = {}
    if metrics.goal_deltas is not None and not metrics.goal_deltas.empty:
        for _, gd in metrics.goal_deltas.iterrows():
            if gd["type"] == "asset_class":
                ac_targets[gd["category"]] = (gd["target_pct"], gd["delta_pct"])
            elif gd["type"] == "geography (equity only)":
                geo_targets[gd["category"]] = (gd["target_pct"], gd["delta_pct"])

    header_row = row
    for c, h in enumerate(["Asset Class", "Actual %", "Target %"], 1):
        _apply_header(sheet, header_row, c, h)
    row += 1

    tol = config.rebalancing_threshold if config else 5.0
    if not metrics.allocation_by_class.empty:
        sorted_ac = metrics.allocation_by_class.sort_values("weight_pct", ascending=False)
        for ti, (_, rd) in enumerate(sorted_ac.iterrows()):
            cat = rd["category"]
            target_pct, delta_pct = ac_targets.get(cat, (None, None))
            dev_color = _deviation_color(delta_pct, tol)
            _write_data_cell(sheet, row, 1, cat, ti, asset_class=cat)
            _write_data_cell(sheet, row, 2, rd["weight_pct"], ti, is_number=True,
                             font_color=dev_color)
            _write_data_cell(sheet, row, 3,
                             target_pct if target_pct is not None else "",
                             ti, is_number=target_pct is not None,
                             font_color=C['text_pri'] if target_pct is not None else None)
            row += 1
    ac_end_row = row

    row = header_row
    for c, h in enumerate(["Geography (Equity)", "Actual %", "Target %"], 5):
        _apply_header(sheet, row, c, h)
    row += 1

    if not metrics.allocation_by_geo.empty:
        sorted_geo = metrics.allocation_by_geo.sort_values("weight_pct", ascending=False)
        for ti, (_, rd) in enumerate(sorted_geo.iterrows()):
            cat = rd["category"]
            target_pct, delta_pct = geo_targets.get(cat, (None, None))
            dev_color = _deviation_color(delta_pct, tol)
            _write_data_cell(sheet, row, 5, cat, ti, geography=cat)
            _write_data_cell(sheet, row, 6, rd["weight_pct"], ti, is_number=True,
                             font_color=dev_color)
            _write_data_cell(sheet, row, 7,
                             target_pct if target_pct is not None else "",
                             ti, is_number=target_pct is not None,
                             font_color=C['text_pri'] if target_pct is not None else None)
            row += 1

    row = max(row, ac_end_row)

    # TOP 5 HOLDINGS
    row += 2
    _write_area_header(sheet, row, 1, 8, "TOP 5 HOLDINGS")
    row += 1

    for c, h in enumerate(["Name", "Value \u20ac", "Weight %", "Gain %", "Class"], 1):
        _apply_header(sheet, row, c, h)
    row += 1

    if not metrics.holdings_df.empty:
        top5 = metrics.holdings_df.nlargest(5, "weight_pct")
        for ti, (_, hr) in enumerate(top5.iterrows()):
            _write_data_cell(sheet, row, 1, hr.get("name", ""), ti)
            _write_data_cell(sheet, row, 2, hr.get("current_value", 0), ti, is_number=True, num_fmt='#,##0.00')
            _write_data_cell(sheet, row, 3, hr.get("weight_pct", 0), ti, is_number=True)
            _write_data_cell(sheet, row, 4, hr.get("gain_pct", 0), ti, is_number=True)
            _write_data_cell(sheet, row, 5, hr.get("asset_class", ""), ti, asset_class=hr.get("asset_class"))
            row += 1

    # REBALANCING ALERT
    if metrics.rebalancing_suggestions:
        row += 2
        n = len(metrics.rebalancing_suggestions)
        s_str = "s" if n > 1 else ""
        alert = f"\u26a0  {n} rebalancing action{s_str} suggested. See the Optimizer tab for details."
        cell = sheet.cell(row=row, column=1, value=alert)
        cell.font = px_font(size=11, bold=True, color=C['amber'])
        cell.fill = px_fill(C['bg_page'])
        cell.border = px_no_border()
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        row += 1

    # Clear column D (spacer) for all dashboard rows
    from openpyxl.cell.cell import MergedCell
    from openpyxl.styles import PatternFill
    white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    for clear_r in range(1, row + 1):
        dc = sheet.cell(row=clear_r, column=4)
        if not isinstance(dc, MergedCell):
            dc.fill = white_fill
            dc.border = px_no_border()
            dc.value = None

    _write_footer(sheet, row + 1, 1)


def _write_area_header(ws, row, col_start, col_end, title):
    """Write a section header — clean subtitle style, consistent across all tabs."""
    ws.merge_cells(start_row=row, start_column=col_start, end_row=row, end_column=col_end)
    cell = ws.cell(row=row, column=col_start, value=title)
    cell.font = px_font(size=12, bold=True, color=C['text_pri'])
    cell.fill = px_fill(C['bg_page'])
    cell.alignment = px_align(h='left', v='center')
    cell.border = px_no_border()
    # Bottom border accent
    from openpyxl.styles import Border, Side
    accent_border = Border(bottom=Side(style='medium', color=C['bg_header']))
    cell.border = accent_border
    for c in range(col_start + 1, col_end + 1):
        mc = ws.cell(row=row, column=c)
        mc.fill = px_fill(C['bg_page'])
        mc.border = accent_border









def _rate_metric(metric_key, value):
    """Return (color_hex, label) for a metric value based on config thresholds."""
    ratings_cfg = cfg.metric_ratings()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (C['amber'], "N/A")
    rc = ratings_cfg.get(metric_key, {})
    thresholds = rc.get("thresholds", [])
    labels = rc.get("labels", ["Strong", "Fair", "Weak"])
    invert = rc.get("invert", False)
    if len(thresholds) < 2:
        return (C['text_sec'], "")
    good_t, warn_t = thresholds
    if invert:
        # "Lower is better" metrics. Use absolute values for comparison
        # to handle both positive (volatility: 7%) and negative (max_dd: -19%) correctly.
        av = abs(value)
        ag = abs(good_t)
        aw = abs(warn_t)
        if av <= ag:
            return (C['green'], labels[0])
        elif av <= aw:
            return (C['amber'], labels[1])
        else:
            return (C['red'], labels[2])
    else:
        if value > good_t:
            return (C['green'], labels[0])
        elif value > warn_t:
            return (C['amber'], labels[1])
        else:
            return (C['red'], labels[2])


def _write_card_header(ws, row, col_start, col_end, title):
    """Write a card section header. If spanning across col D (spacer), split into two merges."""
    spacer_col = 4  # column D
    if col_start < spacer_col < col_end:
        # Left half
        ws.merge_cells(start_row=row, start_column=col_start, end_row=row, end_column=spacer_col - 1)
        cell = ws.cell(row=row, column=col_start, value=title)
        cell.font = px_font(size=11, bold=True, color=C['text_wht'])
        cell.fill = px_fill(C['bg_header'])
        cell.alignment = px_align(h='left', v='center')
        cell.border = px_border_hdr()
        for c in range(col_start + 1, spacer_col):
            mc = ws.cell(row=row, column=c)
            mc.fill = px_fill(C['bg_header'])
            mc.border = px_border_hdr()
        # Spacer D — page background
        dc = ws.cell(row=row, column=spacer_col)
        dc.fill = px_fill(C['bg_page'])
        dc.border = px_no_border()
        # Right half
        ws.merge_cells(start_row=row, start_column=spacer_col + 1, end_row=row, end_column=col_end)
        rc = ws.cell(row=row, column=spacer_col + 1)
        rc.fill = px_fill(C['bg_header'])
        rc.border = px_border_hdr()
        for c in range(spacer_col + 2, col_end + 1):
            mc = ws.cell(row=row, column=c)
            mc.fill = px_fill(C['bg_header'])
            mc.border = px_border_hdr()
    else:
        ws.merge_cells(start_row=row, start_column=col_start, end_row=row, end_column=col_end)
        cell = ws.cell(row=row, column=col_start, value=title)
        cell.font = px_font(size=11, bold=True, color=C['text_wht'])
        cell.fill = px_fill(C['bg_header'])
        cell.alignment = px_align(h='left', v='center')
        cell.border = px_border_hdr()
        for c in range(col_start + 1, col_end + 1):
            mc = ws.cell(row=row, column=c)
            mc.fill = px_fill(C['bg_header'])
            mc.border = px_border_hdr()


def _write_card_metric(ws, row, col_label, col_value, col_rating, label, value,
                       metric_key=None, num_fmt='0.00', is_eur=False):
    """Write a single metric row inside a card: label | value | rating badge."""
    lc = ws.cell(row=row, column=col_label, value=label)
    lc.font = px_font(size=10, bold=False, color=C['text_sec'])
    lc.fill = px_fill(C['bg_card'])
    lc.alignment = px_align(h='left')
    lc.border = px_border()

    vc = ws.cell(row=row, column=col_value, value=value)
    vc.fill = px_fill(C['bg_card'])
    vc.alignment = px_align(h='right')
    vc.border = px_border()
    if isinstance(value, (int, float)):
        vc.number_format = num_fmt
        if metric_key:
            color, _ = _rate_metric(metric_key, value)
            vc.font = px_font(size=11, bold=True, color=color)
        else:
            vc.font = px_font(size=11, bold=True, color=_num_color(value))
    else:
        vc.font = px_font(size=11, bold=True, color=C['text_pri'])

    if metric_key and value is not None and not (isinstance(value, float) and pd.isna(value)):
        color, rating_label = _rate_metric(metric_key, value)
        rc = ws.cell(row=row, column=col_rating, value=f"\u25cf {rating_label}")
        rc.font = px_font(size=9, bold=True, color=color, italic=True)
    else:
        rc = ws.cell(row=row, column=col_rating, value="")
        rc.font = px_font(size=9, color=C['text_sec'])
    rc.fill = px_fill(C['bg_card'])
    rc.alignment = px_align(h='left')
    rc.border = px_border()





def _abs_eur(pct_val, total_value):
    """Convert a percentage risk metric to absolute EUR value."""
    if pct_val is None or (isinstance(pct_val, float) and pd.isna(pct_val)):
        return None
    return abs(pct_val / 100) * total_value


def _write_holdings(workbook, sheet, metrics: PortfolioMetrics):
    """Holdings: full enriched table with instrument type, data source, timestamp."""
    df = metrics.holdings_df
    if df.empty:
        _write_data_cell(sheet, 1, 1, "No holdings data", 0)
        return

    _apply_title(sheet, 1, 1, "Holdings Detail")

    columns = [
        ("Name", "name", False, None),
        ("Ticker", "ticker", False, None),
        ("ISIN", "isin", False, None),
        ("Asset Class", "asset_class", False, None),
        ("Security Type", "security_type", False, None),
        ("Currency", "currency", False, None),
        ("Quantity", "quantity", True, '#,##0.00'),
        ("Avg Price", "avg_purchase_price", True, '#,##0.00'),
        ("Current Price", "current_price", True, '#,##0.00'),
        ("Cost Basis (EUR)", "cost_basis_eur", True, '#,##0.00'),
        ("Value (EUR)", "current_value", True, '#,##0.00'),
        ("% of Portfolio", "weight_pct", True, '0.00'),
        ("% of Asset Class", "pct_of_class", True, '0.00'),
        ("Gain (EUR)", "gain_eur", True, '#,##0.00'),
        ("Gain %", "gain_pct", True, '0.00'),
        ("Geography", "geography", False, None),
        ("Geo Source", "geo_source", False, None),
        ("Data Source", "data_source", False, None),
        ("Fetch Time", "fetch_timestamp", False, None),
    ]

    row = 3
    for c, (header, _, _, _) in enumerate(columns):
        _apply_header(sheet, row, c + 1, header)

    for idx, (_, data_row) in enumerate(df.iterrows()):
        row = idx + 4
        for c, (_, col_key, is_num, nf) in enumerate(columns):
            val = data_row.get(col_key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                _write_data_cell(sheet, row, c + 1, "", idx)
            else:
                ac = data_row.get("asset_class") if col_key == "asset_class" else None
                _write_data_cell(sheet, row, c + 1, val, idx, is_number=is_num,
                                 asset_class=ac, num_fmt=nf)

    _write_footer(sheet, row + 2, 1)



def _write_allocations(workbook, sheet, metrics: PortfolioMetrics, config: InvestorConfig):
    """Optimizer: status overview, rebalancing actions, consolidated deviations table."""
    _apply_title(sheet, 1, 1, "Portfolio Optimizer")

    # Column widths
    sheet.column_dimensions['A'].width = 36   # Category / Holding
    sheet.column_dimensions['B'].width = 13   # Current %
    sheet.column_dimensions['C'].width = 13   # Target %
    sheet.column_dimensions['D'].width = 16   # Post-rebal %
    sheet.column_dimensions['E'].width = 16   # Delta after rebal
    sheet.column_dimensions['F'].width = 12   # Status
    sheet.column_dimensions['G'].width = 45   # Reason / Notes

    tol = config.rebalancing_threshold if config else 5.0

    # =====================================================================
    # OVERVIEW banner — traffic-light status based on largest deviation
    # =====================================================================
    max_abs_delta = 0.0
    if metrics.goal_deltas is not None and not metrics.goal_deltas.empty:
        max_abs_delta = float(metrics.goal_deltas["delta_pct"].abs().max())
    n_actions = len(metrics.rebalancing_suggestions) if metrics.rebalancing_suggestions else 0

    if max_abs_delta <= tol:
        banner_color = C['green']
        banner_icon = "\u25cf"  # filled circle
        banner_text = f"Aligned — all allocations within \u00b1{tol:.1f}%"
    elif max_abs_delta <= 2 * tol:
        banner_color = C['amber']
        banner_icon = "\u25cf"
        banner_text = (
            f"Minor drift — largest deviation {max_abs_delta:.1f}%"
            f" (tolerance \u00b1{tol:.1f}%). Rebalancing optional."
        )
    else:
        banner_color = C['red']
        banner_icon = "\u25cf"
        banner_text = (
            f"Action needed — largest deviation {max_abs_delta:.1f}%"
            f" (>{2 * tol:.1f}%). Rebalancing recommended."
        )
    if n_actions > 0:
        banner_text += f"  \u00b7  {n_actions} action{'s' if n_actions != 1 else ''} suggested below"

    row = 3
    bcell = sheet.cell(row=row, column=1, value=f"{banner_icon}  {banner_text}")
    bcell.font = px_font(size=11, bold=True, color=banner_color)
    bcell.fill = px_fill(C['bg_page'])
    bcell.alignment = px_align(h='left', v='center')
    bcell.border = px_no_border()
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    sheet.row_dimensions[row].height = 22
    row += 2

    # =====================================================================
    # REBALANCING ACTIONS
    # =====================================================================
    header_parts = ["REBALANCING ACTIONS"]
    if config.rebalancing_lump_sum_amount > 0:
        header_parts.append(
            f"lump sum {_format_number(config.rebalancing_lump_sum_amount)} EUR"
        )
    if config.rebalancing_no_sell:
        header_parts.append("no-sell mode")
    _write_area_header(sheet, row, 1, 7, " \u00b7 ".join(header_parts))
    row += 1

    if metrics.rebalancing_suggestions:
        total_buy = sum(s["amount_eur"] for s in metrics.rebalancing_suggestions if s["direction"] == "buy")
        total_sell = sum(s["amount_eur"] for s in metrics.rebalancing_suggestions if s["direction"] == "sell")
        summary = (
            f"Total BUY: {_format_number(total_buy)} EUR"
            f"  \u00b7  Total SELL: {_format_number(total_sell)} EUR"
            f"  \u00b7  Net: {_format_number(total_buy - total_sell)} EUR"
        )
        scell = sheet.cell(row=row, column=1, value=summary)
        scell.font = px_font(size=10, bold=True, color=C['text_sec'])
        scell.fill = px_fill(C['bg_page'])
        scell.alignment = px_align(h='left')
        scell.border = px_no_border()
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 2

        for c, h in enumerate(
            ["Holding", "Direction", "Amount (EUR)", "% of Portfolio", "", "", "Reason"], 1
        ):
            if h:
                _apply_header(sheet, row, c, h)
        row += 1
        total_value = metrics.total_value or 1.0
        for ti, s in enumerate(metrics.rebalancing_suggestions):
            direction = s["direction"].upper()
            pct_of_port = (s["amount_eur"] / total_value) * 100 if total_value > 0 else 0
            dir_color = C['green'] if direction == "BUY" else C['red']
            _write_data_cell(sheet, row, 1, s.get("name", ""), ti)
            _write_data_cell(sheet, row, 2, direction, ti, bold=True, font_color=dir_color)
            _write_data_cell(sheet, row, 3, s["amount_eur"], ti, is_number=True,
                             num_fmt='#,##0.00', font_color=C['text_pri'])
            _write_data_cell(sheet, row, 4, pct_of_port, ti, is_number=True,
                             num_fmt='0.00', font_color=C['text_pri'])
            _write_data_cell(sheet, row, 7, s.get("reason", ""), ti)
            row += 1
    else:
        nocell = sheet.cell(
            row=row, column=1,
            value="No actions within the current tolerance and no-sell / min-transaction constraints.",
        )
        nocell.font = px_font(size=10, italic=True, color=C['text_sec'])
        nocell.fill = px_fill(C['bg_page'])
        nocell.border = px_no_border()
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1

    # =====================================================================
    # ALLOCATION DEVIATIONS — grouped by type, with post-rebalancing view
    # =====================================================================
    row += 2
    _write_area_header(sheet, row, 1, 7, "ALLOCATION DEVIATIONS")
    row += 1
    subcell = sheet.cell(
        row=row, column=1,
        value=(
            f"Status color based on delta after rebalancing vs target. "
            f"Threshold \u00b1{tol:.1f}% (green), \u00b1{2 * tol:.1f}% (amber), beyond (red)."
        ),
    )
    subcell.font = px_font(size=9, italic=True, color=C['text_sec'])
    subcell.fill = px_fill(C['bg_page'])
    subcell.border = px_no_border()
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 2

    # Build current (goal_deltas) lookup by (type, category)
    current_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if metrics.goal_deltas is not None and not metrics.goal_deltas.empty:
        for _, gd in metrics.goal_deltas.iterrows():
            tp = "asset" if gd["type"] == "asset_class" else "geography"
            current_lookup[(tp, gd["category"])] = (
                float(gd["actual_pct"]), float(gd["target_pct"]),
            )

    # Build post-rebalancing lookup by (kind, category)
    post_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if metrics.rebalancing_verifications:
        for v in metrics.rebalancing_verifications:
            for it in v.get("items", []) or []:
                post_lookup[(v["kind"], it["category"])] = (
                    float(it["actual_pct"]), float(it["target_pct"]),
                )

    # Define the four groups in the desired order
    groups = [
        ("asset", "Asset Allocation", current_lookup),
        ("geography", "Geography (equity only)", current_lookup),
        ("per_holding_equity", "Per-Holding Equity Targets", None),
        ("per_holding_fi", "Per-Holding Fixed Income Targets", None),
    ]

    for kind, title, current_source in groups:
        # Collect categories: current_lookup keys for asset/geo, verifications items for per-holding
        categories: list[str] = []
        if kind in ("asset", "geography"):
            categories = [cat for (tp, cat) in current_lookup if tp == kind]
        else:
            # per-holding targets only come from the verifications pass
            for (k, cat) in post_lookup:
                if k == kind:
                    categories.append(cat)

        if not categories:
            continue

        # Group header
        hdr = sheet.cell(row=row, column=1, value=title)
        hdr.font = px_font(size=10, bold=True, color=C['text_pri'])
        hdr.fill = px_fill(C['bg_page'])
        hdr.alignment = px_align(h='left')
        hdr.border = px_no_border()
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1

        # Column headers
        col_headers = [
            "Category", "Current %", "Target %",
            "Post-rebal %", "Delta after rebal (pp)", "Status",
        ]
        for c, h in enumerate(col_headers, 1):
            _apply_header(sheet, row, c, h)
        row += 1

        # Sort categories by |post-rebal delta|, then by |current delta|, descending
        def _sort_key(cat: str) -> float:
            post = post_lookup.get((kind, cat))
            if post:
                return abs(post[0] - post[1])
            curr = current_source.get((kind, cat)) if current_source else None
            return abs(curr[0] - curr[1]) if curr else 0.0

        categories.sort(key=_sort_key, reverse=True)

        for ti, cat in enumerate(categories):
            # Current (pre-rebalancing)
            curr_tuple = current_source.get((kind, cat)) if current_source else None
            current_pct = curr_tuple[0] if curr_tuple else None
            # Target + post from verifications (authoritative for per-holding)
            post_tuple = post_lookup.get((kind, cat))
            if post_tuple:
                post_pct, target_pct = post_tuple
            else:
                post_pct = None
                target_pct = curr_tuple[1] if curr_tuple else None

            # When current is not in goal_deltas but we do have a post value
            # (typical for per-holding targets), use post as current baseline so
            # the Current column is not empty and confusing.
            if current_pct is None and post_pct is not None:
                current_pct = post_pct

            # Delta after rebal drives color and status
            if post_pct is not None and target_pct is not None:
                delta_after = post_pct - target_pct
            elif current_pct is not None and target_pct is not None:
                delta_after = current_pct - target_pct
            else:
                delta_after = 0.0

            color = _deviation_color(delta_after, tol)
            abs_d = abs(delta_after)
            if abs_d <= tol:
                status = "\u25cf Aligned"
            elif abs_d <= 2 * tol:
                status = "\u25cf Drift"
            else:
                status = "\u25cf Action"

            _write_data_cell(sheet, row, 1, cat, ti)
            _write_data_cell(sheet, row, 2,
                             current_pct if current_pct is not None else "",
                             ti, is_number=current_pct is not None,
                             num_fmt='0.00' if current_pct is not None else None,
                             font_color=C['text_pri'])
            _write_data_cell(sheet, row, 3,
                             target_pct if target_pct is not None else "",
                             ti, is_number=target_pct is not None,
                             num_fmt='0.00' if target_pct is not None else None,
                             font_color=C['text_pri'])
            _write_data_cell(sheet, row, 4,
                             post_pct if post_pct is not None else "",
                             ti, is_number=post_pct is not None,
                             num_fmt='0.00' if post_pct is not None else None,
                             font_color=C['text_pri'])
            _write_data_cell(sheet, row, 5, delta_after, ti, is_number=True,
                             num_fmt='+0.00;-0.00;0.00', font_color=color)
            _write_data_cell(sheet, row, 6, status, ti, bold=True, font_color=color)
            row += 1

        row += 1  # spacer between groups

    if not current_lookup and not post_lookup:
        nocell = sheet.cell(row=row, column=1, value="No targets configured.")
        nocell.font = px_font(size=10, italic=True, color=C['text_sec'])
        nocell.fill = px_fill(C['bg_page'])
        nocell.border = px_no_border()
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        row += 1

    # =====================================================================
    # SOLVER INFO
    # =====================================================================
    row += 2
    _write_area_header(sheet, row, 1, 7, "SOLVER PARAMETERS")
    row += 1

    tol_used = None
    if metrics.rebalancing_verifications:
        tol_used = metrics.rebalancing_verifications[0].get("tolerance")

    info_rows = [
        ("Alert threshold",
         f"\u00b1{tol:.1f}%",
         "Deviation at which a category turns amber (green below, red beyond 2\u00d7)"),
        ("Solver tolerance",
         (f"\u00b1{tol_used:.1f}%" if tol_used is not None else "n/a"),
         "Actual tolerance the optimizer converged at (progressive up to max)"),
        ("Max tolerance",
         f"\u00b1{config.rebalancing_max_tolerance:.1f}%",
         "Cap on solver tolerance (from config.rebalancing_max_tolerance)"),
        ("Min transaction",
         f"{_format_number(config.rebalancing_min_transaction_eur)} EUR",
         "Trades below this amount are skipped"),
        ("Lump sum",
         (f"{_format_number(config.rebalancing_lump_sum_amount)} EUR"
          if config.rebalancing_lump_sum_amount > 0 else "—"),
         "Additional cash to deploy in the rebalance"),
        ("No-sell mode",
         ("enabled" if config.rebalancing_no_sell else "disabled"),
         "If enabled, the solver can only buy, never sell"),
    ]

    for c, h in enumerate(["Parameter", "Value", "", "", "", "", "Description"], 1):
        if h:
            _apply_header(sheet, row, c, h)
    row += 1
    for ti, (param, value, desc) in enumerate(info_rows):
        _write_data_cell(sheet, row, 1, param, ti, bold=True)
        _write_data_cell(sheet, row, 2, value, ti)
        _write_data_cell(sheet, row, 7, desc, ti)
        row += 1

    row += 1
    _write_footer(sheet, row, 1)





def _write_performance(workbook, sheet, metrics: PortfolioMetrics):
    """Performance: unified table with TOTAL PORTFOLIO + holdings + benchmarks.

    All risk metrics (CAGR, Vol, Sharpe, Sortino, Max DD, Alpha, Beta) are computed
    on the full available history per instrument, capped at 5 years. The Period Used
    column shows the actual window used.

    Alpha and Beta are computed vs the benchmark marked is_benchmark_alfa_and_beta=true
    in indexes.csv (column headers are dynamic).
    """
    _apply_title(sheet, 1, 1, "Performance Analysis")

    # Dynamic header for Alpha/Beta based on configured benchmark
    bench_beta_name = cfg.benchmark_beta_name()
    alpha_label = f"\u03b1 (vs {bench_beta_name})"
    beta_label = f"\u03b2 (vs {bench_beta_name})"

    # Info note
    row = 2
    note = (f"Period returns (1D–5Y) and risk metrics calculated on available history "
            f"per instrument (max 5 years). \u03b1/\u03b2 computed vs {bench_beta_name}.")
    cell = sheet.cell(row=row, column=1, value=note)
    cell.font = px_font(size=9, italic=True, color=C['text_sec'])
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=21)
    row = 4

    # Columns
    all_cols = ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y",
                "cagr", "volatility", "sharpe", "sortino", "max_drawdown",
                "var_95", "cvar_95", "alpha", "beta", "period_used"]
    all_labels = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y",
                  "CAGR", "Volatility", "Sharpe", "Sortino", "Max DD",
                  "VaR 95%", "CVaR 95%", alpha_label, beta_label, "Period Used"]

    # Set column widths
    sheet.column_dimensions['A'].width = 42
    sheet.column_dimensions['B'].width = 14
    for i in range(3, 21):
        sheet.column_dimensions[get_column_letter(i)].width = 11
    sheet.column_dimensions[get_column_letter(21)].width = 12

    # Header row
    _apply_header(sheet, row, 1, "Name")
    _apply_header(sheet, row, 2, "Type")
    for c, label in enumerate(all_labels, 3):
        _apply_header(sheet, row, c, label)
    row += 1

    # --- TOTAL PORTFOLIO row ---
    port_full = metrics.performance_full or {}
    port_vals = ["Portfolio"]
    for key in all_cols:
        val = port_full.get(key)
        port_vals.append(_format_number(val, True) if key != "period_used" else (val or "—"))
    _write_portfolio_row(sheet, row, 1, "** TOTAL PORTFOLIO **", port_vals, 0)
    row += 1

    # --- Holdings rows + Benchmarks rows ---
    if not metrics.holding_performance.empty:
        hp = metrics.holding_performance
        # Sort: In portfolio first, then Benchmark index
        hp_sorted = hp.sort_values(by="type", ascending=True, kind="stable")
        for ti, (_, hr) in enumerate(hp_sorted.iterrows()):
            _write_data_cell(sheet, row, 1, hr.get("name", hr.get("ticker", "")), ti)
            _write_data_cell(sheet, row, 2, hr.get("type", ""), ti)
            for c, key in enumerate(all_cols, 3):
                val = hr.get(key)
                if key == "period_used":
                    _write_data_cell(sheet, row, c, val or "—", ti)
                else:
                    display = _format_number(val, True)
                    _write_data_cell(sheet, row, c, display, ti,
                                     is_number=isinstance(display, (int, float)))
            row += 1

    row += 2

    # --- LEGEND: Rating Thresholds ---
    _apply_subtitle(sheet, row, 1, "Legend — Rating Thresholds")
    row += 1

    legend_headers = ["Metric", "Description", "\u25cf Strong", "\u25cf Fair", "\u25cf Weak", "Source"]
    # Column widths tuned for readability of Description
    sheet.column_dimensions['A'].width = 22  # Metric
    sheet.column_dimensions['B'].width = 80  # Description
    for i in range(3, 6):
        sheet.column_dimensions[get_column_letter(i)].width = 16
    sheet.column_dimensions['F'].width = 34  # Source
    for c, h in enumerate(legend_headers, 1):
        _apply_header(sheet, row, c, h)
    row += 1

    ratings = cfg.metric_ratings() or {}
    legend_rows = [
        ("CAGR", "cagr",
         "Compound Annual Growth Rate. The single yearly return that, if repeated every year, "
         "would grow your portfolio from start to end value. Accounts for compounding. "
         "~7% is the long-term global equity average.",
         "Equity risk premium (Dimson et al.)"),
        (alpha_label, "alpha",
         "Extra annual return vs the benchmark, after adjusting for how risky the portfolio is (CAPM). "
         "Positive = you beat the market beyond what your risk alone justified. "
         "Negative = you underperformed after fees and noise.",
         "Jensen's Alpha (CAPM)"),
        (beta_label, "beta",
         "How much your portfolio moves when the benchmark moves 1%. β=1 in line, β=0.5 half as "
         "reactive, β=1.5 amplifies by 50%, β≈0 uncorrelated. Tells you how much systematic "
         "market risk you're running.",
         "CAPM, 1.0 = market"),
        ("Max DD", "max_drawdown",
         "Maximum Drawdown. Worst peak-to-trough loss over the period — the most painful scenario "
         "an investor lived through. -20% is typical for diversified equity; deeper drops signal "
         "concentration or high volatility.",
         "Retail drawdown tolerance"),
        ("Volatility", "volatility",
         "How bumpy the ride is. Annualized standard deviation of daily returns. A 15% vol means "
         "~±15% year-to-year noise around the average return. Equity indexes ~15–20%, bonds ~3–7%.",
         "Hist. equity vol ~15%"),
        ("Sharpe", "sharpe",
         "Return per unit of risk taken: (CAGR − risk-free rate) / Volatility. "
         "Above 1 is good, above 2 is excellent, negative means you were paid less than a safe "
         "bond for the risk you ran.",
         "Sharpe (1994)"),
        ("Sortino", "sortino",
         "Like Sharpe but only penalizes downside volatility (ignores upside swings). More honest "
         "when returns are asymmetric. Usually higher than Sharpe — the gap shows how much of "
         "your volatility is actually good volatility.",
         "Sortino & Price (1994)"),
        ("VaR 95%", "var_pct",
         "Value at Risk. The daily loss exceeded only 5% of the time (historical simulation). "
         "A VaR of -1.2% means on an average month you should expect about one day worse than "
         "-1.2%. Non-parametric: no normal-distribution assumption.",
         "Basel III (retail adj.)"),
        ("CVaR 95%", "cvar_pct",
         "Conditional VaR, a.k.a. Expected Shortfall. The average loss on the worst 5% of days. "
         "Always more negative than VaR. Captures tail risk VaR misses — how bad it really gets "
         "when it goes bad.",
         "Artzner et al. (1999)"),
    ]
    for ti, (metric_label, key, description, source) in enumerate(legend_rows):
        spec = ratings.get(key, {})
        thresholds = spec.get("thresholds", [None, None])
        labels = spec.get("labels", ["—", "—", "—"])
        invert = spec.get("invert", False)
        unit = spec.get("unit", "")
        good_t, warn_t = thresholds[0], thresholds[1]

        def fmt(v):
            return f"{v:.1f}{unit}" if v is not None else "—"

        if invert:
            strong = f"< {fmt(abs(good_t)) if good_t is not None else '—'}"
            fair = (
                f"{fmt(abs(warn_t))} – {fmt(abs(good_t))}"
                if good_t is not None and warn_t is not None else "—"
            )
            weak = f"> {fmt(abs(warn_t)) if warn_t is not None else '—'}"
        else:
            strong = f"> {fmt(good_t)}"
            fair = (
                f"{fmt(warn_t)} – {fmt(good_t)}"
                if good_t is not None and warn_t is not None else "—"
            )
            weak = f"< {fmt(warn_t)}"

        _write_data_cell(sheet, row, 1, metric_label, ti)
        # Description cell: wrap text for readability
        desc_cell = _write_data_cell(sheet, row, 2, description, ti)
        desc_cell.alignment = px_align(h='left', wrap=True)
        _write_data_cell(sheet, row, 3, strong, ti)
        _write_data_cell(sheet, row, 4, fair, ti)
        _write_data_cell(sheet, row, 5, weak, ti)
        _write_data_cell(sheet, row, 6, source, ti)
        sheet.row_dimensions[row].height = 48
        row += 1

    row += 2
    _write_footer(sheet, row, 1)


def _write_period_charts(workbook, sheet, metrics, row):
    """Write portfolio and holdings period charts."""
    port_hist = metrics.portfolio_history
    last_date = port_hist.index[-1]
    chart_periods = [
        ("1 Week", 7), ("1 Month", 30), ("3 Months", 90),
        ("YTD", None), ("1 Year", 365),
    ]

    portfolio_charts = []
    holdings_charts = []

    for period_name, period_days in chart_periods:
        cutoff = _get_cutoff(last_date, period_days)
        port_period = port_hist[port_hist.index >= cutoff]
        if len(port_period) < 2:
            continue

        port_norm = port_period / port_period.iloc[0] * 100
        step = max(1, len(port_norm) // 25)
        sampled = port_norm.iloc[::step]

        _apply_header(sheet, row, 1, "Date")
        _apply_header(sheet, row, 2, period_name)
        data_row = row + 1
        for dt_val, val in sampled.items():
            _write_data_cell(sheet, data_row, 1, dt_val.strftime("%m-%d"), data_row - row - 1)
            _write_data_cell(sheet, data_row, 2, float(val), data_row - row - 1, is_number=True)
            data_row += 1

        chart = LineChart()
        chart.title = f"Portfolio: {period_name} (Base 100)"
        chart.width = 15
        chart.height = 10
        chart.legend = None
        cats = Reference(sheet, min_col=1, min_row=row + 1, max_row=data_row - 1)
        vals = Reference(sheet, min_col=2, min_row=row + 1, max_row=data_row - 1)
        chart.add_data(vals, titles_from_data=False)
        chart.set_categories(cats)
        chart.series[0].tx = SeriesLabel(v="Portfolio")
        portfolio_charts.append(chart)
        row = data_row + 1

    # Position portfolio charts
    chart_anchor_row = row
    for i, chart in enumerate(portfolio_charts):
        col_letter = get_column_letter(1 + i * 6)
        # CHARTS DISABLED: sheet.add_chart(chart, f"{col_letter}{chart_anchor_row}")
    row = chart_anchor_row + 16

    # All holdings overlay charts
    holding_data = _get_holdings_with_history(metrics)
    if holding_data:
        holding_histories = {hd["name"][:20]: hd["history"] for hd in holding_data}
        for period_name, period_days in chart_periods:
            cutoff = _get_cutoff(last_date, period_days)
            series_data = {}
            for name, hist in holding_histories.items():
                period = hist[hist.index >= cutoff]
                if len(period) >= 2:
                    series_data[name] = period / period.iloc[0] * 100
            if not series_data:
                continue

            combined = pd.DataFrame(series_data).ffill().bfill()
            step = max(1, len(combined) // 25)
            sampled = combined.iloc[::step]

            _apply_header(sheet, row, 1, "Date")
            for sc, sname in enumerate(sampled.columns):
                _apply_header(sheet, row, 2 + sc, sname)
            data_row = row + 1
            for dt_val, vals in sampled.iterrows():
                _write_data_cell(sheet, data_row, 1, dt_val.strftime("%m-%d"), data_row - row - 1)
                for sc, sname in enumerate(sampled.columns):
                    v = vals[sname]
                    if pd.notna(v):
                        _write_data_cell(sheet, data_row, 2 + sc, float(v), data_row - row - 1, is_number=True)
                data_row += 1

            chart = LineChart()
            chart.title = f"All Holdings: {period_name} (Base 100)"
            chart.width = 28
            chart.height = 16
            chart.y_axis.scaling.min = cfg.holdings_chart_y_min()
            chart.y_axis.scaling.max = cfg.holdings_chart_y_max()
            chart.y_axis.title = "Value"
            cats = Reference(sheet, min_col=1, min_row=row + 1, max_row=data_row - 1)
            for sc, sname in enumerate(sampled.columns):
                vals_ref = Reference(sheet, min_col=2 + sc, min_row=row + 1, max_row=data_row - 1)
                chart.add_data(vals_ref, titles_from_data=False)
                chart.series[sc].tx = SeriesLabel(v=sname)
            chart.set_categories(cats)
            holdings_charts.append(chart)
            row = data_row + 1

        chart_anchor_row = row
        for i, chart in enumerate(holdings_charts):
            pass  # CHARTS DISABLED
        row = chart_anchor_row + len(holdings_charts) * 24

    return row


def _get_cutoff(last_date, period_days):
    """Compute the cutoff date for a chart period."""
    if period_days is not None:
        return last_date - pd.Timedelta(days=period_days)
    return pd.Timestamp(
        year=last_date.year, month=1, day=1,
        tz=last_date.tz if hasattr(last_date, "tz") else None,
    )


def _get_holdings_with_history(metrics: PortfolioMetrics) -> list[dict]:
    """Extract holdings that have price history for charting."""
    return [
        {"ticker": ticker, "name": data.get("name", ticker), "history": data["history"]}
        for ticker, data in metrics.holding_histories.items()
        if data.get("history") is not None and len(data["history"]) > 1
    ]



def _write_analysis(workbook, sheet, metrics: PortfolioMetrics):
    """Multi-Purpose Analysis: contribution bars, breakdowns, rebalancing actions."""
    _apply_title(sheet, 1, 1, "Multi-Purpose Analysis")
    row = 3

    df = metrics.holdings_df
    if df.empty:
        _write_data_cell(sheet, row, 1, "No data available", 0)
        return

    # Return contribution by holding
    _apply_subtitle(sheet, row, 1, "Return Contribution by Holding")
    row += 1
    for c, h in enumerate(["Name", "ISIN", "Ticker", "Weight %", "Gain %", "Contribution"]):
        _apply_header(sheet, row, c + 1, h)
    row += 1
    start_row = row
    contrib_rows = []
    for _, r in df.iterrows():
        contrib = r.get("weight_pct", 0) * r.get("gain_pct", 0) / 100
        contrib_rows.append((r, contrib))
    contrib_rows.sort(key=lambda x: -x[1])
    for ti, (r, contrib) in enumerate(contrib_rows):
        _write_data_cell(sheet, row, 1, r.get("name", ""), ti)
        _write_data_cell(sheet, row, 2, r.get("isin", ""), ti)
        _write_data_cell(sheet, row, 3, r.get("ticker", ""), ti)
        _write_data_cell(sheet, row, 4, r.get("weight_pct", 0), ti, is_number=True)
        _write_data_cell(sheet, row, 5, r.get("gain_pct", 0), ti, is_number=True)
        _write_data_cell(sheet, row, 6, contrib, ti, is_number=True)
        row += 1

    chart = _make_bar("Return Contribution by Holding", sheet,
                      1, [(6, "Return Contribution")], start_row, row - 1, width=20, height=12)
    # CHARTS DISABLED: sheet.add_chart(chart, "H3")

    row += 1

    # Breakdown by asset class
    _apply_subtitle(sheet, row, 1, "Breakdown by Asset Class")
    row += 1
    class_group = df.groupby("asset_class").agg(
        total_value=("current_value", "sum"),
        avg_gain=("gain_pct", "mean"),
        count=("ticker", "count"),
    ).reset_index().sort_values("avg_gain", ascending=False)
    for c, h in enumerate(["Asset Class", "Total Value", "Avg Gain %", "# Holdings"]):
        _apply_header(sheet, row, c + 1, h)
    row += 1
    for ti, (_, r) in enumerate(class_group.iterrows()):
        _write_data_cell(sheet, row, 1, r["asset_class"], ti, asset_class=r["asset_class"])
        _write_data_cell(sheet, row, 2, r["total_value"], ti, is_number=True, num_fmt='#,##0.00')
        _write_data_cell(sheet, row, 3, r["avg_gain"], ti, is_number=True)
        _write_data_cell(sheet, row, 4, r["count"], ti)
        row += 1

    row += 1

    # Breakdown by geography (equity only)
    _apply_subtitle(sheet, row, 1, "Breakdown by Geography (Equity Only)")
    row += 1
    if not metrics.allocation_by_geo.empty:
        geo_gains: dict[str, list[float]] = {}
        for _, r in df.iterrows():
            if r.get("asset_class") != "Equities":
                continue
            geo_str = r.get("geography", "")
            gain = r.get("gain_pct", 0)
            if "," in str(geo_str):
                for part in str(geo_str).split(","):
                    part = part.strip()
                    if ":" in part:
                        geo_name = part.split(":")[0].strip()
                        geo_gains.setdefault(geo_name, []).append(gain)
            else:
                geo_gains.setdefault(str(geo_str), []).append(gain)

        for c, h in enumerate(["Geography", "Weight % (within Equity)", "Avg Gain %"]):
            _apply_header(sheet, row, c + 1, h)
        row += 1
        for ti, (_, r) in enumerate(metrics.allocation_by_geo.iterrows()):
            cat = r["category"]
            gains = geo_gains.get(cat, [])
            avg_gain = sum(gains) / len(gains) if gains else 0
            _write_data_cell(sheet, row, 1, cat, ti, geography=cat)
            _write_data_cell(sheet, row, 2, r["weight_pct"], ti, is_number=True)
            _write_data_cell(sheet, row, 3, avg_gain, ti, is_number=True)
            row += 1
    else:
        _write_data_cell(sheet, row, 1, "No equity holdings", 0)
        row += 1



def _write_benchmark(workbook, sheet, metrics: PortfolioMetrics):
    """Benchmark: comparison table with Name, Ticker, Index, Description + metrics."""
    _apply_title(sheet, 1, 1, "Benchmark Comparison")
    row = 3

    perf = metrics.performance
    details = cfg.benchmark_details()

    headers = ["Index", "Ticker", "1 Day", "1 Week", "1 Month", "3 Months",
               "6 Months", "YTD", "1Y", "3Y", "5Y", "CAGR", "Volatility", "Sharpe",
               "Max DD", "Description"]
    for c, h in enumerate(headers):
        _apply_header(sheet, row, c + 1, h)
    row += 1

    # Portfolio row
    all_keys = ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y", "cagr"]
    port_vals = []
    for key in all_keys:
        port_vals.append(_format_number(perf.get(key), True))
    port_vals.append(_format_number(metrics.risk.get("volatility", 0), True))
    port_vals.append(_format_number(metrics.risk.get("sharpe", 0), True))
    port_vals.append(_format_number(metrics.risk.get("max_drawdown", 0), True))
    port_vals.append("")  # Description
    # Insert empty Ticker before metrics
    full_port_vals = [""] + port_vals
    _write_portfolio_row(sheet, row, 1, "** YOUR PORTFOLIO **", full_port_vals, 0)
    row += 1

    # Benchmark rows
    if not metrics.benchmark_comparison.empty:
        for ti, (_, r) in enumerate(metrics.benchmark_comparison.iterrows()):
            index_name = r.get("benchmark", "")
            detail = details.get(index_name, {})
            _write_data_cell(sheet, row, 1, index_name, ti)
            _write_data_cell(sheet, row, 2, detail.get("ticker", ""), ti)
            for c, key in enumerate(all_keys, 3):
                val = r.get(key)
                display = _format_number(val, True)
                _write_data_cell(sheet, row, c, display, ti, is_number=isinstance(display, (int, float)))
            _write_data_cell(sheet, row, 13, _format_number(r.get("volatility", 0), True), ti, is_number=True)
            _write_data_cell(sheet, row, 14, _format_number(r.get("sharpe", 0), True), ti, is_number=True)
            _write_data_cell(sheet, row, 15, _format_number(r.get("max_drawdown", 0), True), ti, is_number=True)
            _write_data_cell(sheet, row, 16, detail.get("description", ""), ti)
            row += 1

    row += 2
    _write_footer(sheet, row, 1)

