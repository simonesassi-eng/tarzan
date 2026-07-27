"""Newsletter palette, colour maps, class-order constants + the render context.

Leaf module: depends only on external packages. Everything above sits on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tarzan.models.instrument_key import normalize_isin, normalize_ticker
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.taxonomy import (
    ORDER_NEWSLETTER as _ORDER_NEWSLETTER,
    ORDER_PERF as _ORDER_PERF,
)
from tarzan.export._format import (
    ASSET_CLASS_COLORS,
    GEO_COLORS as _GEO_COLORS,
    css,
)

# The palette lives in a leaf module so tarzan.export._charts can read the
# same colours without closing an import cycle through this package.
from tarzan.export._palette import PALETTE  # noqa: E402 (re-exported)


ASSET_COLORS = {k: css(v) for k, v in ASSET_CLASS_COLORS.items()}

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
    "Crypto": "#7C3AED",       # purple
    "Commodities": "#15803D",  # green
    "Currencies": "#64748B",   # slate
    "Indices": "#64748B",      # slate (offline fallback bucket)
}

_NEWSLETTER_CLASS_ORDER = list(_ORDER_NEWSLETTER)

_extra_classes = [c for c in ASSET_CLASS_COLORS if c not in _NEWSLETTER_CLASS_ORDER]

ASSET_CLASS_ORDER = _NEWSLETTER_CLASS_ORDER + sorted(_extra_classes)


_PERF_CLASS_ORDER = list(_ORDER_PERF)
_PERF_ROLE_ORDER = {
    "Equities": ["Equity Broad", "Equity Factor", "Equity Leveraged",
                 "Efficient Core", "Multi-Asset"],
    "Fixed Income": ["Govt Nominal", "Govt Linkers", "Aggregate/Credit",
                     "Long Duration"],
    "Commodities": ["Broad Basket", "Carry", "Market Neutral"],
    "Gold": ["Gold"],
    "Alternative": ["Managed Futures", "Cat Bond"],
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
    for k in (normalize_isin(isin), normalize_ticker(ticker)):
        if k and k in taxonomy and taxonomy[k][1]:
            return taxonomy[k][1]
    return "—"


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

    Returns the ordered groups; the class colour is ASSET_COLORS[class] so the
    4px left bar / marker is consistent everywhere.
    """
    grouped: dict = {}
    for it in items:
        ac = str(asset_class(it) or "") or "Other"
        if role is not None:
            r = str(role(it) or "") or "—"
        else:
            r = role_for(isin(it), ticker(it), taxonomy)
        grouped.setdefault(ac, {}).setdefault(r, []).append(it)
    groups = []
    for ac in _ordered(list(grouped.keys()), _PERF_CLASS_ORDER):
        col = ASSET_COLORS.get(ac, PALETTE["accent"])
        role_list = [(role, grouped[ac][role])
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

def uni_name(name, ticker="", *, tags=(), pill="", span=""):
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
    tk = (f'<span style="font-family:SFMono-Regular,Menlo,Consolas,monospace;'
          f'font-size:10px;font-weight:700;letter-spacing:0.02em;'
          f'color:{P["accent"]};">{ticker}</span>'
          f'<span style="padding-left:7px;"></span>') if ticker else ""
    tag_html = "".join(
        f'<span style="display:inline-block;margin-left:4px;padding:1px 6px;'
        f'background:{t[2]};color:{t[1]};border-radius:4px;font-size:9px;font-weight:700;'
        f'letter-spacing:0.04em;vertical-align:middle;">{t[0]}</span>' for t in (tags or ()))
    span_html = (f'<span style="margin-left:6px;font-size:9px;'
                 f'font-weight:600;color:{P["subtle"]};">{span}</span>'
                 if span else "")
    name_html = (f'<span style="color:{P["muted"]};">{name}</span>'
                 if name else "")
    inner = f'{pill}{tk}{name_html}{tag_html}{span_html}'
    return (f'<div style="font-size:10.5px;font-weight:600;line-height:1.35;'
            f'color:{P["ink"]};">{inner}</div>')


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


def _uni_td(cell, rbg, default_pad="8px 8px", fs="", sep=False):
    P = PALETTE
    w = f'width:{cell["width"]}px;' if cell.get("width") else ""
    pad = cell.get("pad") or default_pad
    lb = f'border-left:1px solid {_COL_RULE};' if (sep and not cell.get("no_sep")) else ""
    sub = (f'<div style="margin-top:2px;font-size:9.5px;font-weight:600;'
           f'color:{P["subtle"]};font-variant-numeric:tabular-nums;">'
           f'{cell["sub"]}</div>' if cell.get("sub") else "")
    bg = cell.get("bg") or rbg
    return (f'<td align="{cell["align"]}" style="padding:{pad};background:{bg};'
            f'border-bottom:1px solid {P["row_rule"]};{lb}'
            f'font-variant-numeric:tabular-nums;{fs}'
            f'font-weight:{cell["weight"]};color:{cell["color"]};'
            f'vertical-align:{cell.get("valign", "middle")};{w}">{cell["html"]}{sub}</td>')


def _uni_header(first_col_label, columns, vpad, first_col_width, sep):
    P = PALETTE
    fw = f'width:{first_col_width}px;' if first_col_width else ""
    hc = (f'<td style="padding:7px 10px;background:{P["head_bg"]};'
          f'border-bottom:1px solid {P["border"]};font-size:9.5px;font-weight:700;'
          f'letter-spacing:0.06em;color:{P["muted"]};text-transform:uppercase;{fw}">'
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
               f'border-bottom:1px solid {P["border"]};{lb}font-size:9.5px;font-weight:700;'
               f'letter-spacing:0.06em;color:{P["muted"]};text-transform:uppercase;{w}">'
               f'{label}</td>')
    return f"<tr>{hc}</tr>"


def render_unified_table(first_col_label, columns, groups, *,
                         portfolio_row=None, compact=False,
                         first_col_width=None, separators=False,
                         zebra=True, dense=False, radius=10):
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

    Each ``row`` (and ``portfolio_row``) is a dict:
        ``name_html`` — inner HTML for the name cell (use :func:`uni_name`).
        ``cells``     — list of :func:`uni_cell` dicts, one per ``columns`` entry.
        ``row_bg``    — optional background override (e.g. the Optimizer's
                        BUY/SELL tint); wins over the alternating stripe.
    """
    P = PALETTE
    ncols = len(columns) + 1
    # Value-cell padding + font: tight/smaller for dense tables (Risk = 10
    # columns) so the name column keeps enough width; normal otherwise. The
    # name column keeps its wider padding regardless.
    # ``dense`` is the returns-grid setting: the tint has to fill its column, so
    # the padding is trimmed rather than the figure.
    vpad = "5px 4px" if dense else ("7px 3px" if compact else "7px 8px")
    fs = ("font-size:9.5px;" if dense
          else ("font-size:10px;" if compact else "font-size:10.5px;"))
    # Per-value-column separator flags: on for every column except the first
    # (the 1D sparkline / first metric), and honouring an explicit no_sep.
    seps = [separators and i > 0 and not (len(c) > 3 and c[3])
            for i, c in enumerate(columns)]
    out = ['<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
           'border="0" style="margin-top:14px;border:1px solid '
           f'{P["border"]};border-radius:{radius}px;overflow:hidden;border-collapse:separate;'
           f'border-spacing:0;font-size:12px;">',
           _uni_header(first_col_label, columns, vpad, first_col_width, separators)]

    def _cells_html(cells, bg):
        return "".join(_uni_td(c, bg, vpad, fs, sep=seps[i])
                       for i, c in enumerate(cells))

    if portfolio_row is not None:
        abg = P["accent_bg"]
        fw = f'width:{first_col_width}px;' if first_col_width else ""
        out.append(
            '<tr>'
            f'<td style="padding:8px 10px;background:{abg};color:{P["accent"]};'
            f'font-weight:700;font-size:11px;vertical-align:middle;{fw}">'
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
                         f'font-weight:700;">{role}</span>' if has_role else "")
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
                        f'font-size:10.5px;vertical-align:middle;{fw}">'
                        f'{row["name_html"]}</td>'
                        + _cells_html(row["cells"], rbg) + '</tr>')
                continue
            # The group header sits on its own surface rather than on the card,
            # where it was the same colour as the row under it. The class colour
            # is carried by the class name; the 4px bar down the left of every
            # row was a second, louder statement of the same fact and it made
            # each class read as a separate bordered block.
            out.append(
                f'<tr><td colspan="{ncols}" style="padding:8px 10px;'
                f'background:{P["group_bg"]};'
                f'border-bottom:1px solid {P["border"]};'
                f'font-size:10px;letter-spacing:0.06em;text-transform:uppercase;">'
                f'<span style="color:{col};font-weight:700;">{cls}</span>{role_html}</td></tr>')
            fw = f'width:{first_col_width}px;' if first_col_width else ""
            for row in rows:
                rbg = row.get("row_bg") or (
                    P["card_alt"] if not zebra
                    else (P["card"] if ri % 2 == 0 else P["zebra"]))
                ri += 1
                out.append(
                    f'<tr><td style="padding:7px 10px;background:{rbg};'
                    f'border-bottom:1px solid {P["row_rule"]};'
                    f'font-size:10.5px;vertical-align:middle;{fw}">'
                    f'{row["name_html"]}</td>'
                    + _cells_html(row["cells"], rbg) + '</tr>')
    out.append("</table>")
    return "".join(out)


@dataclass
class _NewsletterContext:
    """Strongly-typed wrapper around the template context dict."""

    metrics: PortfolioMetrics
    config: InvestorConfig
    issue_number: int = 1
    benchmark_alpha_beta: str = "S&P 500"
    benchmark_geo: str = "MSCI ACWI"
    # One run-scoped preprocessed intraday catalog is shared by every
    # performance table. The renderer performs no provider request or venue
    # resolution; the semantic gate checks this projection against metrics.
    performance_intraday_map: Optional[dict] = None
    semantic_audit: Optional[dict] = None

