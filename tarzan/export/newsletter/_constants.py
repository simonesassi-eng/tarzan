"""Newsletter palette, colour maps, class-order constants + the render context.

Shared presentation leaf with explicitly prepared reference facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from typing import Optional

from tarzan.contracts.reference import ReportReference, reference_for
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.taxonomy import (
    ORDER_NEWSLETTER as _ORDER_NEWSLETTER,
    ORDER_PERF as _ORDER_PERF,
    class_key,
)
from tarzan.export._format import (
    ASSET_CLASS_COLORS,
    GEO_COLORS as _GEO_COLORS,
    css,
)

# The palette lives in a leaf module so tarzan.export._charts can read the
# same colours without closing an import cycle through this package.
from tarzan.export._palette import (
    text_width,  # noqa: E402 (re-exported)
    CLASS_COLOURS,
    FONT_STACK,
    MONO_STACK,
    PALETTE,
    TYPE,
    TYPE_PX,
    class_band,
)


# The class hues, from the one source every surface reads (``ASSET_CLASS_COLORS``
# in tarzan.export._format); ``CLASS_COLOURS`` is the same table as CSS colours.
ASSET_COLORS = {k: CLASS_COLOURS.get(k, css(v)) for k, v in ASSET_CLASS_COLORS.items()}

GEO_COLORS = {k: css(v) for k, v in _GEO_COLORS.items()}


# Display-only shortening for geography buckets. The long form is a
# CONFIGURATION KEY in constants.yaml, mapped to from dozens of countries, so it
# cannot be renamed -- but "Dev ex-USA ex-EMU ex-JP" wraps to four lines in a
# table cell and starves the columns beside it.
GEO_DISPLAY = {"Dev ex-USA ex-EMU ex-JP": "Dev ex-USA/EMU/JP"}


def geo_label(name) -> str:
    """The display form of a geography bucket; the key itself when unmapped."""
    return GEO_DISPLAY.get(str(name), str(name))

MARKET_REGION_COLORS = {
    "US": "#2563EB",           # blue
    "Europe": "#D97706",       # amber
    "Asia": "#DB2777",         # pink
    "Futures": "#0891B2",      # cyan
    "Bonds": "#4F46E5",        # indigo
    "Crypto": "#7C3AED",       # purple
    "Commodities": "#15803D",  # green
    "Currencies": "#64748B",   # slate
    "Indices": "#64748B",      # slate (offline fallback bucket)
}

# Country flag per markets-strip symbol (emoji: Gmail/Apple Mail draw them,
# desktop Outlook on Windows falls back to the two letters). The EU flag marks
# pan-European indices, a pair carries both currencies' flags, and
# commodities carry none: they belong to no country.
_US, _EU, _JP = "\U0001F1FA\U0001F1F8", "\U0001F1EA\U0001F1FA", "\U0001F1EF\U0001F1F5"
_GB, _DE, _IT = "\U0001F1EC\U0001F1E7", "\U0001F1E9\U0001F1EA", "\U0001F1EE\U0001F1F9"
_FR, _ES, _AU = "\U0001F1EB\U0001F1F7", "\U0001F1EA\U0001F1F8", "\U0001F1E6\U0001F1FA"
_CN = "\U0001F1E8\U0001F1F3"
MARKET_FLAGS = {
    "^AXJO": _AU, "^N225": _JP, "^KS11": "\U0001F1F0\U0001F1F7",
    "^HSI": "\U0001F1ED\U0001F1F0", "000001.SS": _CN,
    "^NSEI": "\U0001F1EE\U0001F1F3",
    "^STOXX50E": _EU, "^STOXX": _EU, "^N100": _EU,
    "FTSEMIB.MI": _IT, "^GDAXI": _DE, "^FCHI": _FR, "^FTSE": _GB,
    **{s: _US for s in ("^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX",
                        "ES=F", "NQ=F", "YM=F", "RTY=F",
                        "bond:US2Y", "bond:US10Y", "bond:US30Y")},
    "fut:SPI": _AU, "fut:NK": _JP, "fut:A50": _CN,
    "fut:KS200": "\U0001F1F0\U0001F1F7", "fut:HSI": "\U0001F1ED\U0001F1F0",
    "fut:NIFTY": "\U0001F1EE\U0001F1F3", "fut:FIB": _IT, "fut:MME": "",
    "fut:ES": _US, "fut:NQ": _US, "fut:YM": _US, "fut:RTY": _US,
    # Commodities belong to no country.
    **{k: "" for k in ("cmd:CL", "cmd:BRN", "cmd:GC", "cmd:SI", "cmd:HG",
                       "cmd:NG", "cmd:PL")},
    "fut:FESX": _EU, "fut:FXXP": _EU, "fut:FDAX": _DE, "fut:FCE": _FR, "fut:Z": _GB,
    "bond:DE2Y": _DE, "bond:DE10Y": _DE, "bond:IT2Y": _IT, "bond:IT10Y": _IT,
    "spread:IT-DE10Y": _IT + _DE, "bond:FR10Y": _FR,
    "bond:ES2Y": _ES, "bond:ES10Y": _ES, "bond:GB2Y": _GB, "bond:GB10Y": _GB,
    "bond:JP10Y": _JP,
    "EURUSD=X": _EU + _US, "JPY=X": _US + _JP, "GBP=X": _US + _GB,
}

_NEWSLETTER_CLASS_ORDER = list(_ORDER_NEWSLETTER)

_extra_classes = [c for c in ASSET_CLASS_COLORS if c not in _NEWSLETTER_CLASS_ORDER]

ASSET_CLASS_ORDER = _NEWSLETTER_CLASS_ORDER + sorted(_extra_classes)


_PERF_CLASS_ORDER = list(_ORDER_PERF)
_PERF_ROLE_ORDER = {
    "Equities": ["Equity Broad", "Equity Factor", "Equity Leveraged",
                 "Efficient Core", "Trend Stacked", "Gold Stacked",
                 "Carry Stacked", "Multi-Asset"],
    "Fixed Income": ["Govt Nominal", "Govt Linkers", "Aggregate/Credit",
                     "Long Duration"],
    "Commodities": ["Broad Basket", "Carry", "Market Neutral"],
    "Gold": ["Gold"],
    "Alternative": ["Managed Futures", "Currency", "Cat Bond", "Tail Hedge"],
    "Cash & Cash Equivalents": ["Cash / Money Market"],
}


_PF_INTRA_KEY = "__PORTFOLIO_INTRADAY__"


# ── Shared instrument-categorization engine ─────────────────────────────────
# One place that decides an instrument's (asset class, role) and how the
# class→role groups are ordered, so EVERY table (Holdings, Optimizer, Returns
# snapshot, Performance) splits and colours instruments identically. Before,
# each table re-derived this inline and drifted.

def role_for(isin, ticker, taxonomy) -> str:
    """The curated role (e.g. 'Equity Factor', 'Long Duration') for an
    instrument, from ``instrument_taxonomy.csv`` (ISIN first, then bare
    ticker). ``taxonomy`` is ``config.instrument_taxonomy()`` — passed in so
    this stays a pure function. Returns '—' when the role is unset."""
    from tarzan import config as cfg

    hit = cfg.catalog_lookup(taxonomy, isin, ticker)
    return hit[1] if hit and hit[1] else "—"


def _ordered(keys, preferred):
    """``preferred`` order first (those present), then any extras in their
    given order — so a new class/role is appended, never dropped."""
    return ([k for k in preferred if k in keys]
            + [k for k in keys if k not in preferred])


def group_by_class_role(items, *, asset_class, taxonomy=None,
                        isin=None, ticker=None, role=None):
    """Group an iterable of items into the canonical
    ``[(class, class_color, [(role, [item, ...]), ...]), ...]`` structure,
    ordered by _PERF_CLASS_ORDER then _PERF_ROLE_ORDER. The accessors are
    callables mapping an item to a field, so the SAME engine works for
    holdings-df rows, optimizer actions, and performance rows alike.

    Role is resolved one of two ways: pass ``role`` (a callable) when the item
    already carries its role, else pass ``isin``+``ticker``+``taxonomy`` to
    look it up via :func:`role_for`.

    Within a role the items are sorted alphabetically by ``ticker``, which is the
    first thing every row prints — so a reader finds an instrument in the same
    place in each table instead of a per-table order (value, trade size, feed
    order). Pass ``ticker`` even when ``role`` makes it redundant for lookup;
    without it the caller's own order is kept.

    Returns the ordered groups; the class colour is ASSET_COLORS[class] so the
    4px left bar / marker is consistent everywhere.
    """
    # No ticker accessor -> every key is "" and the stable sort is a no-op.
    def _tk(it) -> str:
        return str((ticker(it) if ticker is not None else "") or "").upper()

    grouped: dict = {}
    for it in items:
        ac = class_key(asset_class(it))
        if role is not None:
            r = str(role(it) or "") or "—"
        else:
            r = role_for(isin(it), ticker(it), taxonomy)
        grouped.setdefault(ac, {}).setdefault(r, []).append(it)
    groups = []
    for ac in _ordered(list(grouped.keys()), _PERF_CLASS_ORDER):
        col = ASSET_COLORS.get(ac, PALETTE["accent"])
        role_list = [(role, sorted(grouped[ac][role], key=_tk))
                     for role in _ordered(list(grouped[ac].keys()),
                                          _PERF_ROLE_ORDER.get(ac, []))]
        groups.append((ac, col, role_list))
    return groups


# ── The ONE canonical instrument-table renderer ─────────────────────────────
# Holdings, Returns snapshot, Performance, Historical risk profile and the
# Optimizer all render through render_unified_table, so they are visually
# identical bar their column set: same card shell, same light column header,
# same class header (colour square + 4px left bar), same role caption, same
# alternating row stripe. Only the columns + per-cell colours (which encode
# data) differ. Before, four separate renderers had drifted.

def uni_name(name, ticker="", *, tags=(), pill="", span="", line_height=None):
    """Canonical instrument label: an optional action pill (BUY/SELL), the
    ticker, then the name, then reference tags and a faint history-span chip.

    The ticker is accent-coloured monospace text, not a bordered chip. As a chip
    it carried a border, a background and 10px of horizontal padding -- about
    34px of furniture per row -- and that width came out of the name, which is
    why instrument names wrapped onto two and three lines in every table. As
    text it costs its own glyphs and nothing else, and it leads the row, so the
    column reads as a list of tickers with names attached rather than as a list
    of names with badges stuck on the end.
    """
    P = PALETTE
    tk = ticker_span(ticker)
    tag_html = "".join(
        f'<span style="display:inline-block;margin-left:4px;padding:1px 6px;'
        f'background:{t[2]};color:{t[1]};border-radius:4px;'
        f'font-size:{TYPE_PX["label"]}px;font-weight:700;'
        f'vertical-align:middle;">{escape(str(t[0]))}</span>'
        for t in (tags or ()))
    span_html = (f'<span style="margin-left:6px;'
                 f'font-size:{TYPE_PX["label"]}px;'
                 f'font-weight:600;color:{P["subtle"]};">{escape(str(span))}</span>'
                 if span else "")
    # Instrument names are provider/broker text, so this is the boundary where
    # untrusted text becomes markup and the only place it can be escaped once
    # for every table. Ten curated names carry an ampersand ("iShares Core S&P
    # 500", "L&G Multi-Strategy", the Return Stacked family), which reached the
    # document as a bare &. This builder owns the text-to-HTML boundary:
    # complete section HTML is intentionally passed through |safe later.
    name_html = (f'<span style="color:{P["muted"]};font-weight:400;">{escape(str(name))}</span>'
                 if name else "")
    inner = f'{pill}{tk}{name_html}{tag_html}{span_html}'
    # The row takes the DATA role's leading unless a caller sets its own. The
    # only reason to override it is a name allowed to wrap onto a second line,
    # where tight leading is what keeps the row the same height as its
    # neighbours -- not a preference about how the first line looks.
    lh = f"line-height:{line_height};" if line_height else "line-height:1.3;"
    return (f'<div style="font-size:{TYPE_PX["prose"]}px;{lh}color:{P["ink"]};'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{inner}</div>')


def ticker_span(ticker: str) -> str:
    """The ticker that leads an instrument row: accent-coloured text on the
    document type scale, plus the gap before the name.

    It was a bordered chip. The border, background and padding cost about 34px
    of every row, taken out of the name column, which is what made instrument
    names wrap onto two and three lines. The allocation tables carried their
    own copy of this span (``_div_pin``), whose docstring said it existed to
    match this one — so the two could and did drift apart.
    """
    if not ticker:
        return ""
    return (f'<span style="font-family:{FONT_STACK};font-size:{TYPE_PX["prose"]}px;'
            f'font-weight:700;color:{PALETTE["ink"]};">{escape(str(ticker))}</span>'
            f'<span style="padding-left:5px;"></span>')


def uni_cell(html, *, align="right", color=None, weight=600, sub=None,
             width=None, pad=None, valign="middle", bg=None):
    """One value cell for :func:`render_unified_table`. ``html`` is the cell's
    ready-to-render inner content; ``sub`` is an optional muted sub-line (e.g.
    the € under a %).

    ``bg`` overrides the row background for this cell alone, which is what lets
    a returns column carry a conditional-formatting tint. Without it the
    renderer could only paint whole rows.
    """
    return {"html": html, "align": align, "color": color or PALETTE["ink"],
            "weight": weight, "sub": sub, "width": width, "pad": pad,
            "valign": valign, "bg": bg}





# Barely-there vertical rule between value columns — lighter than the
# horizontal row rule, so it guides the eye down a period column without
# drawing a grid. Both rules and the row surface are palette roles now; they
# were inline literals, which is why these tables could not follow a palette.
_COL_RULE = PALETTE["col_rule"]


def _uni_td(cell, rbg, default_pad="8px 8px", sep=False):
    P = PALETTE
    w = f'width:{cell["width"]}px;' if cell.get("width") else ""
    pad = cell.get("pad") or default_pad
    lb = f'border-left:1px solid {_COL_RULE};' if (sep and not cell.get("no_sep")) else ""
    sub = (f'<div style="margin-top:2px;'
           f'font-size:{TYPE_PX["label"]}px;font-weight:600;'
           f'color:{P["subtle"]};font-variant-numeric:tabular-nums;">'
           f'{cell["sub"]}</div>' if cell.get("sub") else "")
    bg = cell.get("bg") or rbg
    return (f'<td align="{cell["align"]}" style="padding:{pad};background:{bg};'
            f'border-bottom:1px solid {P["row_rule"]};{lb}'
            f'font-variant-numeric:tabular-nums;'
            f'font-size:{TYPE_PX["data"]}px;'
            f'font-weight:{cell["weight"]};color:{cell["color"]};'
            f'vertical-align:{cell.get("valign", "middle")};{w}">{cell["html"]}{sub}</td>')


def _uni_header(first_col_label, columns, vpad, first_col_width, sep):
    P = PALETTE
    fw = f'width:{first_col_width}px;' if first_col_width else ""
    hc = (f'<td style="padding:7px 10px;background:{P["head_bg"]};'
          f'border-bottom:1px solid {P["border"]};{TYPE["label"]}'
          f'color:{P["muted"]};{fw}">'
          f'{first_col_label}</td>')
    for idx, col in enumerate(columns):
        label, align = col[0], col[1]
        width = col[2] if len(col) > 2 else None
        no_sep = col[3] if len(col) > 3 else False
        w = f'width:{width}px;' if width else ""
        # Separator on every value column except the first one after the name
        # (usually the 1D sparkline, which reads as part of the name block).
        lb = (f'border-left:1px solid {_COL_RULE};'
              if (sep and idx > 0 and not no_sep) else "")
        hc += (f'<td align="{align}" style="padding:{vpad};background:{P["head_bg"]};'
               f'border-bottom:1px solid {P["border"]};{lb}{TYPE["label"]}'
               f'color:{P["muted"]};{w}">'
               f'{label}</td>')
    return f"<tr>{hc}</tr>"


def render_unified_table(first_col_label, columns, groups, *,
                         portfolio_row=None, compact=False,
                         first_col_width=None, separators=False,
                         zebra=True, dense=False, radius=10,
                         fixed_layout=False):
    """Render the canonical instrument table shared by all five sections.

    Args:
        first_col_label: header label for the left (name) column — the only
            first-column text that varies (Holding / Instrument / Series).
        columns: ordered value columns after the name column, each
            ``(label, align)``, ``(label, align, width_px)`` or
            ``(label, align, width_px, no_sep)`` (``no_sep`` suppresses the
            column separator on that column).
        groups: ``[(class_name, class_color, [(role_name, [row, ...]), ...]),
            ...]`` — exactly what :func:`group_by_class_role` returns, with each
            ``row`` built by the caller. A falsy ``class_name`` renders that
            block flat, with no group header row: a table whose rows are already
            ordered by something else (the Optimizer sorts by trade size) gains
            nothing from headers that name a grouping it does not use.
        portfolio_row: optional highlighted top row rendered before the groups.
        compact: tighten the value-cell horizontal padding + font (used by the
            Risk profile with 10 columns and the returns tables with 8) so the
            name column keeps enough width to wrap cleanly.
        first_col_width: cap the name column at this pixel width, so a wide name
            column can't hoard space and leave a gap before the value columns.
        separators: draw a barely-there vertical rule between value columns so a
            reader can tell which period a % belongs to. The first value column
            (the 1D sparkline) is left un-ruled as it reads with the name block.
        fixed_layout: emit ``table-layout:fixed`` so every column lands at the
            width declared here regardless of its own cell contents. Two tables
            built with the SAME column widths then align to one grid — the
            Returns and Watchlist tables share a container width, so with fixed
            layout their sparklines and period columns fall at identical x
            positions. Under automatic layout each table sizes its columns from
            its own content (short tickers + small % in Returns, long names +
            large % in Watchlist), so the two grids drift apart.

    Each ``row`` (and ``portfolio_row``) is a dict:
        ``name_html`` — inner HTML for the name cell (use :func:`uni_name`).
        ``cells``     — list of :func:`uni_cell` dicts, one per ``columns`` entry.
        ``row_bg``    — optional background override (e.g. the Optimizer's
                        BUY/SELL tint); wins over the alternating stripe.
    """
    P = PALETTE
    ncols = len(columns) + 1
    # Value-cell padding: tight for dense tables (Risk = 10 columns, the
    # returns grids = 8) so the name column keeps enough width; normal
    # otherwise. The name column keeps its wider padding regardless.
    #
    # These flags used to trim the FIGURE as well (9.5 / 10 / 10.5px), which is
    # why the same instrument's 1M return was set three different sizes
    # depending on which section drew it. Padding is what a crowded table
    # actually has to give up: it is furniture, the figure is the content.
    vpad = "5px 4px" if dense else ("7px 3px" if compact else "7px 8px")
    # Per-value-column separator flags: on for every column except the first
    # (the 1D sparkline / first metric), and honouring an explicit no_sep.
    seps = [separators and i > 0 and not (len(c) > 3 and c[3])
            for i, c in enumerate(columns)]
    layout = "table-layout:fixed;" if fixed_layout else ""
    out = ['<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
           'border="0" style="margin-top:14px;border:1px solid '
           f'{P["border"]};border-radius:{radius}px;overflow:hidden;border-collapse:separate;'
           f'border-spacing:0;font-family:{FONT_STACK};'
           f'font-size:{TYPE_PX["data"]}px;{layout}">',
           _uni_header(first_col_label, columns, vpad, first_col_width, separators)]

    def _cells_html(cells, bg):
        return "".join(_uni_td(c, bg, vpad, sep=seps[i])
                       for i, c in enumerate(cells))

    if portfolio_row is not None:
        abg = P["accent_bg"]
        fw = f'width:{first_col_width}px;' if first_col_width else ""
        out.append(
            '<tr>'
            f'<td style="padding:8px 10px;background:{abg};color:{P["accent"]};'
            f'font-weight:700;font-size:{TYPE_PX["data"]}px;'
            f'vertical-align:middle;{fw}">'
            f'{portfolio_row["name_html"]}</td>'
            + _cells_html(portfolio_row["cells"], abg) + '</tr>')

    for cls, col, role_list in groups:
        # One header row per (class, role) block — "CLASS · Role" — carrying the
        # thin 4px colour bar + the class name in its colour. Putting the role
        # in the group header (instead of a caption stacked above each name)
        # frees a full line in the name cell, so long instrument names wrap
        # cleanly instead of truncating. The alternating row stripe continues
        # unbroken across the whole class (ri is not reset per role).
        ri = 0
        for role, rows in role_list:
            has_role = bool(role and role != "—")
            role_html = (f'&nbsp;&middot;&nbsp;<span style="color:{P["muted"]};'
                         f'font-weight:700;">{escape(str(role))}</span>'
                         if has_role else "")
            if not cls:
                # Flat block: no class name, so no header row to carry it.
                fw = f'width:{first_col_width}px;' if first_col_width else ""
                for row in rows:
                    rbg = row.get("row_bg") or (
                        P["card_alt"] if not zebra
                        else (P["card"] if ri % 2 == 0 else P["zebra"]))
                    ri += 1
                    out.append(
                        f'<tr><td style="padding:7px 10px;background:{rbg};'
                        f'border-bottom:1px solid {P["row_rule"]};'
                        f'font-size:{TYPE_PX["data"]}px;'
                        f'vertical-align:middle;{fw}">'
                        f'{row["name_html"]}</td>'
                        + _cells_html(row["cells"], rbg) + '</tr>')
                continue
            # The group header sits on its own surface rather than on the card,
            # where it was the same colour as the row under it. The class colour
            # is carried by the class name; the 4px bar down the left of every
            # row was a second, louder statement of the same fact and it made
            # each class read as a separate bordered block.
            out.append(
                # Mixed case, not the LABEL role's capitals: "Cash & Cash
                # Equivalents · Cash / Money Market" is forty-five characters,
                # a phrase rather than a label, and a phrase set in capitals is
                # read letter by letter.
                f'<tr><td colspan="{ncols}" style="padding:8px 10px;'
                f'background:{P["group_bg"]};'
                f'border-bottom:1px solid {P["border"]};'
                f'font-size:{TYPE_PX["label"]}px;font-weight:700;'
                f'letter-spacing:0.04em;">'
                f'<span style="color:{col};font-weight:700;">'
                f'{escape(str(cls))}</span>{role_html}</td></tr>')
            fw = f'width:{first_col_width}px;' if first_col_width else ""
            for row in rows:
                rbg = row.get("row_bg") or (
                    P["card_alt"] if not zebra
                    else (P["card"] if ri % 2 == 0 else P["zebra"]))
                ri += 1
                out.append(
                    f'<tr><td style="padding:7px 10px;background:{rbg};'
                    f'border-bottom:1px solid {P["row_rule"]};'
                    f'font-size:{TYPE_PX["data"]}px;'
                    f'vertical-align:middle;{fw}">'
                    f'{row["name_html"]}</td>'
                    + _cells_html(row["cells"], rbg) + '</tr>')
    out.append("</table>")
    return "".join(out)


def bench_short(name) -> str:
    """The benchmark's index word for a narrow header: "SPDR MSCI ACWI IMI Acc" ->
    "ACWI IMI", "iShares MSCI ACWI" -> "ACWI" (share-class words dropped; IMI
    kept with what it qualifies)."""
    drop = {"acc", "dist", "ucits", "etf", "eur", "usd", "hedged", "hdg"}
    parts = [w for w in str(name or "").replace("-", " ").split() if w.casefold() not in drop]
    if not parts:
        return "Benchmark"
    if parts[-1].upper() == "IMI" and len(parts) > 1:
        return f"{parts[-2]} {parts[-1]}".upper()
    return parts[-1].upper()


# ── The statement grid ───────────────────────────────────────────────────────
# Holdings, Target instruments and Watchlist at phone width (358px). One table
# per asset class · role block, carrying the class band and a 2px rail in the
# class hue; each instrument is two rows: the name row (ticker, name, and an
# optional right-hand block such as the book figures) and the window row (the
# 1D cell with its session sparkline, then one cell per window). Column widths
# are hard under ``table-layout:fixed`` so every grid in the issue lands its
# windows on the same x positions. The budget, measured in the system font:
# the rows of a group sit right of a 2px rail in the class hue and 6px of air,
# so they share 346px; the session column keeps 59px for a 24px sparkline, 4px
# of air and a figure, and the seven windows after it 41px each, which holds a
# six-glyph figure ("+16.46" is 34px at 11px regular) with a 1px seam on the
# left and 2px of air on the right, so two tinted neighbours never read as one
# band. Cells are the data size at REGULAR weight: bold figures in a tinted
# cell read as a second signal and ate the width the names needed.
GRID_W = 354
GRID_RAIL_W, GRID_RAIL_GAP = 2, 6
GRID_DAY_W = 59
GRID_CELL_PX = TYPE_PX["data"]


def grid_cell(html, *, bg=None, color=None, weight=400, align="right"):
    """One window cell of the statement grid."""
    return {"html": html, "bg": bg, "color": color or PALETTE["ink"],
            "weight": weight, "align": align}


def _grid_header_row(day_label, wins):
    P = PALETTE
    style = (f'padding:4px 3px 3px 0;{TYPE["label"]}font-size:9.5px;'
             f'color:{P["faint"]};')
    tds = f'<td align="right" style="{style}">{escape(str(day_label))}</td>'
    tds += "".join(f'<td align="right" style="{style}">{escape(str(w))}</td>' for w in wins)
    return f"<tr>{tds}</tr>"


def _grid_instrument_rows(row, ncol, *, last):
    P = PALETTE
    right = row.get("right_html") or ""
    # Fixed layout so the name truncates with an ellipsis instead of pushing the
    # figures off the column; the right column is sized from its own text
    # (11px tabular figures run ~6.2px per glyph).
    rtext = re.sub(r"<[^>]+>", "", right).replace("&nbsp;", " ")
    rw = int(text_width(rtext, TYPE_PX["data"]) + 8) if right else 0
    name_td = ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
               'border="0" style="table-layout:fixed;">'
               + (f'<colgroup><col><col style="width:{rw}px;"></colgroup>' if right else "")
               + '<tr>'
               f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{row["name_html"]}</td>'
               + (f'<td align="right" style="padding:0 0 0 6px;white-space:nowrap;'
                  f'vertical-align:baseline;">{right}</td>' if right else "")
               + '</tr></table>')
    out = f'<tr><td colspan="{ncol}" style="padding:5px 0 1px 0;">{name_td}</td></tr>'
    day_bg = f'background:{row["day_bg"]};' if row.get("day_bg") else ""
    tds = (f'<td align="right" style="padding:2px 3px 2px 0;{day_bg}white-space:nowrap;'
           f'font-variant-numeric:tabular-nums;">{row.get("day_html", "")}</td>')
    for c in row["cells"]:
        bg = f'background:{c["bg"]};' if c.get("bg") else ""
        tds += (f'<td align="{c["align"]}" style="padding:3px 2px 3px 0;{bg}'
                f'border-left:1px solid {P["page"]};'
                f'font-variant-numeric:tabular-nums;font-size:{GRID_CELL_PX}px;'
                f'font-weight:{c["weight"]};color:{c["color"]};white-space:nowrap;">'
                f'{c["html"]}</td>')
    rule = "" if last else f'border-bottom:1px solid {P["row_rule"]};'
    out += (f'<tr>{tds}</tr>'
            f'<tr><td colspan="{ncol}" style="padding:0;height:3px;font-size:0;'
            f'line-height:0;{rule}">&nbsp;</td></tr>')
    return out


def render_statement_grid(columns, groups, *, header_every=12, portfolio_row=None,
                          unit_label="%"):
    """Render grouped instrument rows as the statement grid.

    Args:
        columns: ordered window labels; the first is the day column's label
            ("1D" / "Intraday"), the rest the windows after it.
        groups: ``[(class_name, class_color, [(role_name, [row, ...]), ...]), ...]``
            where each ``row`` has ``name_html`` (from :func:`uni_name`),
            optional ``right_html`` (set beside the name, right-aligned),
            ``day_html`` (the 1D cell's inner HTML: sparkline + figure) with an
            optional ``day_bg``, and ``cells`` (one :func:`grid_cell` per window
            after the day column).
        header_every: repeat the window header at the next class header once
            this many instrument rows have passed since the last one.
        portfolio_row: an optional emphasised row rendered first, same shape.
        unit_label: the unit printed at the right of every class band.
    """
    P = PALETTE
    ncol = len(columns)
    day_label, wins = columns[0], columns[1:]

    def _grid_open(width: int) -> str:
        win_w = (width - GRID_DAY_W) // max(len(wins), 1)
        return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                'border="0" style="table-layout:fixed;border-collapse:collapse;'
                f'font-family:{FONT_STACK};"><colgroup><col style="width:{GRID_DAY_W}px;">'
                + "".join(f'<col style="width:{win_w}px;">' for _ in wins) + '</colgroup>')

    out = []
    since_header = header_every
    if portfolio_row is not None:
        out.append(f'<div style="margin-top:8px;">{_grid_open(GRID_W)}{_grid_header_row(day_label, wins)}'
                   f'{_grid_instrument_rows(portfolio_row, ncol, last=True)}</table></div>')
        since_header = 0
    inner_w = GRID_W - GRID_RAIL_W - GRID_RAIL_GAP
    for cls, col, role_list in groups:
        for role, rows in role_list:
            has_role = bool(role and role != "\u2014")
            band = class_band(cls) if cls else P["shade"]
            label = (f'<span style="display:inline-block;width:8px;height:8px;'
                     f'border-radius:1px;background:{col};margin-right:6px;'
                     f'vertical-align:1px;"></span>'
                     f'<span style="color:{P["ink"]};">{escape(str(cls or ""))}</span>'
                     + (f' <span style="color:{P["muted"]};font-weight:400;letter-spacing:0;'
                        f'text-transform:none;">\u00b7 {escape(str(role))}</span>'
                        if has_role else ""))
            # The group: its band across the column, then the rail (a real cell,
            # not a table border, which the collapsed-border model drops) beside
            # the rows.
            body = []
            if since_header >= header_every:
                body.append(_grid_header_row(day_label, wins))
                since_header = 0
            for i, row in enumerate(rows):
                since_header += 1
                body.append(_grid_instrument_rows(row, ncol, last=(i == len(rows) - 1)))
            out.append(
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                f'style="table-layout:fixed;border-collapse:collapse;margin-top:8px;font-family:{FONT_STACK};">'
                f'<colgroup><col style="width:{GRID_RAIL_W}px;"><col></colgroup>'
                f'<tr><td colspan="2" style="padding:5px 6px 4px 8px;background:{band};'
                f'border-radius:0 3px 3px 0;">'
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                f'<tr><td style="padding:0;{TYPE["label"]}">{label}</td>'
                f'<td align="right" style="padding:0;font-size:{TYPE_PX["label"]}px;'
                f'color:{P["muted"]};">{escape(str(unit_label))}</td></tr></table></td></tr>'
                f'<tr><td style="padding:0;background:{col};font-size:0;line-height:0;">&nbsp;</td>'
                f'<td style="padding:0 0 0 {GRID_RAIL_GAP}px;vertical-align:top;">'
                f'{_grid_open(inner_w)}{"".join(body)}</table></td></tr></table>')
    return "".join(out)


@dataclass
class _NewsletterContext:
    """Strongly-typed wrapper around the template context dict."""

    metrics: PortfolioMetrics
    config: InvestorConfig
    issue_number: int = 1
    # No literal index name as a default. Both are looked up in
    # ``metrics.benchmark_histories`` BY this name, so a default that does not
    # match the curated taxonomy row is not a harmless label — it silently
    # removes the benchmark from every chart that reads it. ``build_context``
    # always resolves them (see ``newsletter._benchmark_names``); a directly
    # constructed context draws no benchmark rather than the wrong one.
    benchmark_alpha_beta: Optional[str] = None
    benchmark_geo: Optional[str] = None
    # One run-scoped preprocessed intraday catalog is shared by every
    # performance table. The renderer performs no provider request or venue
    # resolution; the semantic gate checks this projection against metrics.
    performance_intraday_map: Optional[dict] = None
    semantic_audit: Optional[dict] = None
    reference: Optional[ReportReference] = None

    def __post_init__(self):
        if self.reference is None:
            self.reference = reference_for(self.metrics)

