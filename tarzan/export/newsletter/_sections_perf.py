"""Performance / returns-table / risk / markets section builders."""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from tarzan.models.instrument_key import normalize_ticker
from tarzan.export._format import (
    display_instrument_name,
    eur_smart as _eur_smart,
    short_instrument_name,
)
from tarzan.export import _charts as _charts
from tarzan.export._perf_series import (
    _norm_series,
    benchmark_gap_pp,
    _perf_full_series,
    _perf_vol_series,
    _perf_window,
    _window_money_pnl,
    _window_twror,
    market_snapshot,
)
from tarzan.export import _heat
from tarzan.export.newsletter._constants import (
    ASSET_COLORS,
    MARKET_REGION_COLORS,
    PALETTE,
    _NewsletterContext,
    _PF_INTRA_KEY,
    group_by_class_role,
    render_unified_table,
    uni_cell,
)
from tarzan.export.newsletter._format import (
    _colorize_pct,
    _display_ticker,
    _pct,
    _pct_compact,
    is_missing,
)
from tarzan.export.newsletter._charts import (
    _day_spark,
    _flat_dashed_spark,
    _intraday_spark,
    _prev_session_label,
    day_column_label,
)

logger = logging.getLogger(__name__)

def _build_markets(ctx: _NewsletterContext) -> dict:
    """Markets strip (yfinance-style): roomy, neutral 5-column cards — level,
    a daily-change pill and a stretched mini sparkline — grouped by region.
    Each region is introduced by its own heading with a region-accent colour
    swatch, so colour lives in the section labels (not on every card) and no
    separate legend is needed. Live cached quotes, falling back to the
    benchmark-derived snapshot if the live fetch yields nothing."""
    m = ctx.metrics
    P = PALETTE
    COLS = 5

    def market_open_now(_ticker):  # safe default if the import below fails
        return None

    def is_continuous_market(_ticker):  # safe default if the import fails
        return False

    try:
        from tarzan.data.market_quotes import (fetch_market_quotes, CATEGORY_ORDER,
                                               market_open_now, is_continuous_market)
        snap = fetch_market_quotes()
    except Exception:  # noqa: BLE001
        snap, CATEGORY_ORDER = [], []
    if not snap:
        # Offline fallback: the indices we already hold histories for.
        snap = [dict(d, category="Indices") for d in market_snapshot(m)]
        CATEGORY_ORDER = ["Indices"]
    if not snap:
        return {"available": False, "html": ""}

    def _rc(cat) -> str:
        return MARKET_REGION_COLORS.get(cat, P["subtle"])

    def _spark_for(d: dict) -> str:
        """Session path for one index: the timestamped intraday series when
        there is one, else the daily fallback. Narrow, because this is context
        rather than subject matter."""
        sym = d.get("symbol", "")
        ss = d.get("spark_series")
        if ss is not None and len(ss) >= 2:
            # Continuous instruments (futures/FX/crypto) have no bounded cash
            # session, so they draw full width; exchange-listed ones grow
            # through their session while it is open.
            in_progress = (False if is_continuous_market(sym)
                           else market_open_now(sym))
            return _intraday_spark(ss, d.get("baseline", d["value"]),
                                   w=44, h=20, in_progress=in_progress)
        return _day_spark(d.get("spark", []), d.get("baseline", d["value"]),
                          w=44, h=20, stretch=False)

    def _row(d: dict) -> str:
        up = d["pct"] >= 0
        col = P["green"] if up else P["red"]
        name = d["name"]
        # Tag futures so a full-width sparkline reads as a continuously traded
        # contract (change vs previous settlement), not a finished session.
        if str(d.get("symbol", "")).upper().endswith("=F"):
            name = f"{name} (FUT)"
        level = (f'{d["value"]:,.0f}' if abs(d["value"]) >= 1000
                 else f'{d["value"]:,.2f}')
        td = (f'padding:4px 5px;border-bottom:1px solid {P["row_rule"]};'
              f'font-variant-numeric:tabular-nums;')
        return (
            f'<tr>'
            f'<td style="{td}font-size:10px;font-weight:600;color:{P["ink"]};'
            f'white-space:nowrap;">{name}</td>'
            f'<td align="right" style="{td}">{_spark_for(d)}</td>'
            f'<td align="right" style="{td}font-size:10px;color:{P["muted"]};'
            f'white-space:nowrap;">{level}</td>'
            f'<td align="right" style="{td}font-size:10px;font-weight:700;'
            f'color:{col};white-space:nowrap;">{d["pct"]:+.2f}%'
            f'<div style="font-size:8.5px;font-weight:600;color:{P["subtle"]};">'
            f'{d["change"]:+,.2f}</div></td>'
            f'</tr>')

    def _region_head(cat: str) -> str:
        return (f'<tr><td colspan="4" style="padding:6px 5px 4px;'
                f'border-bottom:1px solid {P["row_rule"]};font-size:9px;'
                f'font-weight:700;letter-spacing:0.06em;'
                f'text-transform:uppercase;color:{_rc(cat)};">{cat}</td></tr>')

    def _table(entries: list) -> str:
        """One column of the strip: region heads interleaved with their rows."""
        head = (f'<tr>' + "".join(
            f'<td align="{al}" style="padding:5px 5px;background:{P["card_alt"]};'
            f'border-bottom:1px solid {P["border"]};font-size:9px;'
            f'font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
            f'color:{P["muted"]};">{lbl}</td>'
            for lbl, al in (("Index", "left"), ("Session", "right"),
                            ("Level", "right"), ("Chg %", "right"))) + '</tr>')
        body, last = [], None
        for cat, d in entries:
            if cat != last:
                body.append(_region_head(cat))
                last = cat
            body.append(_row(d))
        return ('<table role="presentation" width="100%" cellpadding="0" '
                'cellspacing="0" border="0" style="border:1px solid '
                f'{P["border"]};border-radius:8px;overflow:hidden;'
                'border-collapse:separate;border-spacing:0;">'
                + head + "".join(body) + '</table>')

    # Region order: the configured order first, then any category the snapshot
    # carries that the configuration does not know about, so nothing is dropped.
    order = list(CATEGORY_ORDER)
    for d in snap:
        if d.get("category") not in order:
            order.append(d.get("category"))
    entries = [(cat, d) for cat in order
               for d in snap if d.get("category") == cat]
    if not entries:
        return {"available": False, "html": ""}

    # Two side-by-side columns instead of a five-wide card grid: the same
    # indices cost roughly half the height, and this section is the backdrop a
    # reader glances at, not the subject of the issue. The split lands on a
    # region boundary so neither column starts mid-region.
    half = len(entries) // 2
    cut = next((i for i in range(half, len(entries))
                if entries[i][0] != entries[i - 1][0]), half)
    left, right = entries[:cut], entries[cut:]
    if right:
        grid = ('<table role="presentation" width="100%" cellpadding="0" '
                'cellspacing="0" border="0" style="margin-top:10px;">'
                '<tr>'
                f'<td width="49%" valign="top">{_table(left)}</td>'
                '<td width="2%"></td>'
                f'<td width="49%" valign="top">{_table(right)}</td>'
                '</tr></table>')
    else:
        grid = f'<div style="margin-top:10px;">{_table(left)}</div>'
    # Section subtitle: which close the levels are, and how many indices. Both
    # change what the table means and neither was stated.
    close_label = _prev_session_label(m, "%d %b")
    n = sum(1 for d in snap if isinstance(d, dict))
    sub = (f'Session close \u00b7 {close_label} \u00b7 {n} indices'
           if close_label else f'{n} indices')
    return {"available": True, "html": grid, "sub": sub}

def _build_performance30(ctx: _NewsletterContext) -> dict:
    """Performance section: a 1D / 7D / 30D / since-inception returns matrix
    (Total P&L €+%, Unrealized P&L €+%, TWROR %) with an annualized footer,
    plus three 30-day trajectory charts (patrimony €, return vs benchmark,
    your-return-three-ways). All numbers come straight from the order-derived
    series — nothing the hero already shows is recomputed here. Returns
    ``{"available": False}`` on the holdings-only path."""
    m = ctx.metrics
    if (m.actual_value_series is None or m.pnl_series is None
            or m.portfolio_history is None or m.pnl_eur is None):
        return {"available": False, "html": ""}
    win = _perf_window(m, 30, ctx.benchmark_geo)
    if win is None:
        return {"available": False, "html": ""}

    P = PALETTE

    def _sgn(v: Optional[float]) -> str:
        return P["green"] if (v is not None and v >= 0) else P["red"]

    # ── Matrix values (windows reuse _window_money_pnl / _window_twror;
    #    "since inception" uses the authoritative lifetime fields). ──────
    tot = {d: _window_money_pnl(m.pnl_series, m.actual_value_series, d) for d in (1, 7, 30)}
    tot_since = (m.pnl_eur, m.pnl_pct)
    unr = {d: _window_money_pnl(m.unrealized_series, m.actual_value_series, d) for d in (1, 7, 30)}
    unr_since = (m.unrealized_pnl_eur, m.unrealized_pnl_pct)
    nav_norm = _norm_series(m.portfolio_history)
    tw = {d: _window_twror(nav_norm, d) for d in (1, 7, 30)}
    tw_since = m.twror_pct

    # Broker-style 1 Day: use the live "since previous close" portfolio move
    # (value-weighted from the intraday quotes, in performance_full["1d"]) so
    # the 1-day column updates during the session like the Returns tables.
    # The 7/30-day columns stay close-based. Falls back to the window figures
    # when no live quote is available.
    live_1d = (m.performance_full or {}).get("1d")
    if live_1d is not None and not (isinstance(live_1d, float) and math.isnan(live_1d)):
        tw[1] = float(live_1d)
        prev_val = m.total_value / (1.0 + float(live_1d) / 100.0) if (1.0 + float(live_1d) / 100.0) else None
        eur_1d = (m.total_value - prev_val) if prev_val is not None else None
        tot[1] = (eur_1d, float(live_1d))
        unr[1] = (eur_1d, float(live_1d))

    bt = f"1px solid {P['border']}"

    def _money_cell(pair) -> str:
        eur, pct = pair
        c = _sgn(eur)
        eur_s = _eur_smart(eur, signed=True) if eur is not None else "—"
        pct_s = _pct(pct, decimals=2, signed=True) if pct is not None else ""
        return (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};">'
                f'<div style="font-size:13px;font-weight:700;color:{c};font-variant-numeric:tabular-nums;">{eur_s}</div>'
                + (f'<div style="font-size:11px;color:{c};font-variant-numeric:tabular-nums;">{pct_s}</div>' if pct_s else "")
                + '</td>')

    def _pct_cell(v) -> str:
        return (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};font-size:13px;'
                f'font-weight:700;color:{_sgn(v)};font-variant-numeric:tabular-nums;">'
                f'{_pct(v, signed=True) if v is not None else "—"}</td>')

    # ── The matrix, windows as ROWS ──────────────────────────────────────
    # Transposed from measures-as-rows. A reader asks "how did the last week
    # go", which is one row here instead of one cell picked out of three rows,
    # and it puts the four measures in a fixed column order that matches the
    # returns grids below.
    _1d_live = bool((m.performance_full or {}).get("1d_live"))

    def _label(txt) -> str:
        return (f'<td style="padding:7px 0;border-top:{bt};font-size:12px;'
                f'font-weight:700;color:{P["ink"]};white-space:nowrap;">{txt}</td>')

    def _money_pair(pair, *, eur_only=False, pct_only=False) -> str:
        """One measure split across its own two columns, € then %."""
        eur, pct = pair
        c = _sgn(eur if not pct_only else pct)
        if pct_only:
            return (f'<td align="right" style="padding:7px 0 7px 10px;'
                    f'border-top:{bt};font-size:13px;font-weight:700;color:{c};'
                    f'font-variant-numeric:tabular-nums;">'
                    f'{_pct(pct, decimals=2, signed=True) if pct is not None else "\u2014"}</td>')
        return (f'<td align="right" style="padding:7px 0 7px 10px;'
                f'border-top:{bt};font-size:13px;font-weight:700;color:{c};'
                f'font-variant-numeric:tabular-nums;">'
                f'{_eur_smart(eur, signed=True) if eur is not None else "\u2014"}</td>')

    def _pct_cell(v) -> str:
        return (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};'
                f'font-size:13px;font-weight:700;color:{_sgn(v)};'
                f'font-variant-numeric:tabular-nums;">'
                f'{_pct(v, signed=True) if v is not None else "\u2014"}</td>')

    def _same(total, unreal, twror) -> bool:
        """True when the three measures are the same number for a window.

        With no external flow inside it, the P&L change, the unrealized change
        and the time-weighted return coincide. Printing the same figure three
        times invites the reader to look for a difference that is not there.
        """
        vals = [total[1], unreal[1], twror]
        if any(v is None for v in vals):
            return False
        return max(vals) - min(vals) < 0.005

    heads = ("Window", "P&amp;L \u20ac", "P&amp;L %", "Unrealized", "TWROR")
    head_html = '<tr>' + "".join(
        f'<td align="{"left" if i == 0 else "right"}" style="padding:0 0 5px'
        f'{"" if i == 0 else " 10px"};font-size:9px;font-weight:700;'
        f'color:{P["muted"]};letter-spacing:0.04em;text-transform:uppercase;">'
        f'{h}</td>' for i, h in enumerate(heads)) + '</tr>'

    windows = [
        ("1 day" + (" \u25CF LIVE" if _1d_live else ""), tot[1], unr[1], tw[1]),
        ("7 days", tot[7], unr[7], tw[7]),
        ("30 days", tot[30], unr[30], tw[30]),
        ("Since inception", tot_since, unr_since, tw_since),
    ]
    body = ""
    for label, total, unreal, twror in windows:
        coincide = _same(total, unreal, twror)
        eq = (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};'
              f'font-size:13px;font-weight:700;color:{P["subtle"]};">=</td>')
        body += ('<tr>' + _label(label)
                 + _money_pair(total)
                 + _money_pair(total, pct_only=True)
                 + (eq if coincide else _money_pair(unreal, pct_only=True))
                 + (eq if coincide else _pct_cell(twror))
                 + '</tr>')
    matrix = (f'<table role="presentation" width="100%" cellpadding="0" '
              f'cellspacing="0" border="0" style="border-collapse:collapse;">'
              f'{head_html}{body}</table>')
    # No footer under the matrix. It repeated the annualized TWROR and the XIRR,
    # which are the captions of the TWROR and MWR tiles in STATE, and pointed at
    # a tax note that has its own place in the appendix.
    footer = ""
    matrix_card = (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
                   f'style="margin-top:14px;background:{P["card_alt"]};border:1px solid {P["border"]};'
                   f'border-radius:12px;border-collapse:separate;border-spacing:0;">'
                   f'<tr><td style="padding:14px 16px;">{matrix}{footer}</td></tr></table>')

    # ── Charts: "You vs the market" — two compact side-by-side panels with
    #    the SAME lines (TWROR, Total P&L %, MSCI ACWI), differing only in
    #    window: last-30-days rebased vs since-inception cumulative. The
    #    portfolio value chart lives in the hero, so it is not repeated here.
    dates = win["dates"]
    GREEN, PNL, BENCH = _charts.GREEN, _charts.PNL, _charts.BENCH

    def _colcap(t: str) -> str:
        return f'<div style="font-size:11px;font-weight:700;color:{P["ink"]};margin-bottom:3px;">{t}</div>'

    # Last-30-day labels come from the exact arrays passed to the chart. The
    # shared-close endpoint is therefore the only number that can describe a
    # line; generic 1M return buckets are intentionally not consulted here.
    endpoints = dict(win.get("endpoints") or {})
    legend_values: dict[str, float] = {}
    legend_labels: dict[str, str] = {}

    def _window_label(key: str, prefix: str) -> str:
        value = float(endpoints[key])
        label = f'{prefix} {_pct(value, signed=True)}'
        legend_values[key] = value
        legend_labels[key] = label
        return label

    s30, l30 = [], []
    if win["twror"] is not None and endpoints.get("twror") is not None:
        s30.append({"values": win["twror"], "color": GREEN})
        l30.append((_window_label("twror", "TWROR"), GREEN, False))
    if win["pnl_pct"] is not None and endpoints.get("pnl_pct") is not None:
        s30.append({"values": win["pnl_pct"], "color": PNL})
        l30.append((_window_label("pnl_pct", "Total P&L %"), PNL, False))
    if win["acwi"] is not None and endpoints.get("acwi") is not None:
        s30.append({"values": win["acwi"], "color": BENCH})
        l30.append((_window_label("acwi", "MSCI ACWI"), BENCH, False))

    if ctx.semantic_audit is not None:
        ctx.semantic_audit["performance_30d"] = {
            "window_start": str(win.get("window_start") or ""),
            "window_end": str(win.get("window_end") or ""),
            "source_end_dates": {
                key: str(value or "")
                for key, value in (win.get("source_end_dates") or {}).items()
            },
            "endpoints": endpoints,
            "legend_values": legend_values,
            "legend_labels": legend_labels,
        }

    # Since inception (cumulative), over the WHOLE inception→today range — its
    # own x-axis, not the last-30-days window. Labels pinned to the lifetime
    # authoritative fields (m.twror_pct, m.pnl_pct).
    ssi, lsi = [], []
    full = _perf_full_series(m, ctx.benchmark_geo)
    si_dates = full["dates"] if full else dates
    if full is not None:
        if full["twror"] is not None:
            ssi.append({"values": full["twror"], "color": GREEN})
            lsi.append((f'TWROR {_pct(m.twror_pct, signed=True)}', GREEN, False))
        if full["pnl_pct"] is not None:
            ssi.append({"values": full["pnl_pct"], "color": PNL})
            lsi.append((f'Total P&L % {_pct(m.pnl_pct, signed=True)}', PNL, False))
        if full["acwi"] is not None:
            ssi.append({"values": full["acwi"], "color": BENCH})
            lsi.append((f'MSCI ACWI {_pct(full["acwi"][-1], signed=True)}', BENCH, False))

    # ── Volatility row (You vs the market, second row): rolling annualized
    #    volatility over the same two windows. Grey line = the benchmark, so the
    #    reader sees whether they run calmer or bumpier.
    VOL = "#B45309"  # amber-brown, distinct from the return lines
    vol_full = _perf_vol_series(m, ctx.benchmark_geo, n_days=None)
    vol_30 = _perf_vol_series(m, ctx.benchmark_geo, n_days=30)

    # Panel sizes. The card's content box is 544px wide; with the 8px gutter
    # between the two half cells each of them gets 264px. These are passed
    # explicitly because the SVG carries its own width — putting a chart in a
    # wider table cell does not make the chart wider.
    W_WIDE, H_WIDE = 544, 166
    W_HALF, H_HALF = 264, 138
    def _vol_panel(vs, dates_, *, month_ticks, min_day_ticks,
                   w=W_HALF, h=H_HALF) -> str:
        series, leg = [], []
        if vs and vs.get("port"):
            series.append({"values": vs["port"], "color": VOL})
            leg.append((f'You {_pct(vs["port"][-1], signed=False)}', VOL, False))
        if vs and vs.get("acwi"):
            series.append({"values": vs["acwi"], "color": BENCH})
            leg.append((f'MSCI ACWI {_pct(vs["acwi"][-1], signed=False)}', BENCH, False))
        if not series:
            return ""
        return (_charts.chart_pct_compact(series, dates_, include_zero=False,
                                          w=w, h=h, month_ticks=month_ticks,
                                          min_day_ticks=min_day_ticks)
                + _charts.legend(leg, 9))

    parts = []
    if s30 or ssi:
        # LEFT column = since inception (month grid); RIGHT column = last 30
        # days (>=12 day grid). Row 1 = cumulative return, row 2 = rolling
        # annualized volatility — each return chart sits above its vol twin.
        left_ret = (_colcap(f"Since inception <span style='font-weight:400;color:{P['subtle']};'>· cumulative</span>")
                    + _charts.chart_pct_compact(ssi, si_dates, include_zero=False,
                                                w=W_WIDE, h=H_WIDE, month_ticks=True)
                    + _charts.legend(lsi, 9)) if ssi else ""
        right_ret = (_colcap(f"Last 30 days <span style='font-weight:400;color:{P['subtle']};'>· rebased to 0</span>")
                     # Five date ticks, not twelve: at half width twelve
                     # rotated labels overlapped into a grey band, which is
                     # worse than no axis at all.
                     + _charts.chart_pct_compact(s30, dates, include_zero=True,
                                                 w=W_HALF, h=H_HALF,
                                                 min_day_ticks=5)
                     + _charts.legend(l30, 9)) if s30 else ""
        # One volatility panel, over the whole history. The 30-day rolling twin
        # was dropped: two volatility charts side by side invite a comparison
        # between two windows of the same measure, which is not the question
        # this section asks, and it cost a quarter of the section's height.
        vol_panel = _vol_panel(vol_full, vol_full["dates"] if vol_full else si_dates,
                               month_ticks=True, min_day_ticks=0)
        if vol_panel:
            vol_panel = _colcap(
                f"Volatility <span style='font-weight:400;color:{P['subtle']};'>"
                f"\u00b7 annualized, rolling 1-month</span>") + vol_panel

        def _row(l, r):
            return (f'<tr>'
                    f'<td width="50%" valign="top" style="padding:0 8px 0 0;">{l}</td>'
                    f'<td width="50%" valign="top" style="padding:0 0 0 8px;'
                    f'border-left:1px solid {P["border"]};">{r}</td>'
                    f'</tr>')

        def _wide(cell):
            return f'<tr><td colspan="2" valign="top">{cell}</td></tr>'

        # Since-inception runs the full width: it is the chart that answers the
        # section's question, and at half width its three series overlap into
        # noise. The shorter window and the volatility share the row below.
        rule = (f'<tr><td colspan="2" style="padding-top:12px;'
                f'border-top:1px solid {P["border"]};"></td></tr>')
        rows = _wide(left_ret) if left_ret else ""
        if right_ret or vol_panel:
            rows += (rule if rows else "") + _row(right_ret, vol_panel)
        charts_tbl = (
            f'<table role="presentation" width="100%" cellpadding="0" '
            f'cellspacing="0" border="0" style="margin-top:12px;">'
            + rows + '</table>'
        )
        # Quantitative "why are we diverging?" note, encapsulated in this card
        # right under the two charts. AI writes the prose when available; a
        # deterministic rule-based note (same figures) is used otherwise, so
        # the block is always present and never trivial.
        divergence_html = ""
        try:
            from tarzan.export.ai_summary import divergence_note
            note = divergence_note(m, ctx.config, ctx.benchmark_geo)
            if note:
                divergence_html = (
                    f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid {P["border"]};">'
                    f'<div style="font-size:11px;font-weight:700;letter-spacing:0.06em;color:{P["accent"]};'
                    f'text-transform:uppercase;">Why you’re diverging</div>'
                    f'<div style="margin-top:6px;font-size:13px;color:{P["ink"]};line-height:1.55;">'
                    f'{_colorize_pct(note)}</div>'
                    f'<div style="margin-top:8px;font-size:10px;color:{P["subtle"]};">'
                    f'✨ Quantitative attribution vs {ctx.benchmark_geo} · '
                    f'informational, not financial advice</div></div>'
                )
        except Exception as e:  # noqa: BLE001 — never break the newsletter
            logger.debug("Divergence note skipped: %s", e)

        # No card kicker or subtitle here: this is a top-level section now and
        # the template's heading carries both. Nested inside the performance
        # card it needed its own title; as a section it would print two
        # headings for one block.
        parts.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="background:{P["card_alt"]};border:1px solid {P["border"]};'
            f'border-radius:12px;border-collapse:separate;border-spacing:0;">'
            f'<tr><td style="padding:14px 16px;">{charts_tbl}{divergence_html}</td></tr></table>'
        )
    # Two blocks, two sections. The window matrix belongs with the portfolio's
    # own value chart; the benchmark comparison answers a different question and
    # gets its own heading, as in the concept. Returned separately so the
    # template can place each under its own ordinal rather than one section
    # carrying both.
    gap = benchmark_gap_pp(m, ctx.benchmark_geo)
    gap_sub = None
    if gap is not None:
        word = "behind" if gap < 0 else ("ahead of" if gap > 0 else "level with")
        sign = "+" if gap > 0 else ("\u2212" if gap < 0 else "")
        col = P["red"] if gap < 0 else (P["green"] if gap > 0 else P["muted"])
        gap_sub = (f'Now <strong style="color:{col};">{sign}{abs(gap):.2f}pp'
                   f'</strong> {word} {ctx.benchmark_geo}.')
    return {"available": True,
            "matrix_html": matrix_card,
            "vs_market_html": "".join(parts),
            "vs_market_sub": gap_sub}

def _intraday_quote_parts(quote) -> tuple[object, object]:
    """Return ``(series, baseline)`` from a preprocessed quote.

    A bare series is accepted for the synthetic portfolio path and lightweight
    render fixtures. Production rows always use the structured quote emitted
    by ``MetricsEngine._live_1d``.
    """
    if isinstance(quote, dict):
        return quote.get("intraday_series"), quote.get("intraday_baseline")
    return quote, None


def _shared_performance_intraday(ctx: _NewsletterContext) -> dict:
    """Share the run-scoped preprocessed quote catalog across all sections."""
    if ctx.performance_intraday_map is not None:
        return ctx.performance_intraday_map

    metrics = ctx.metrics
    requested = tuple(getattr(metrics, "intraday_requested_tickers", ()) or ())
    result = dict(getattr(metrics, "intraday_quotes", {}) or {})
    source_tickers = {
        canonical: str(
            quote.get("intraday_source_ticker")
            or quote.get("source_ticker")
            or canonical
        )
        for canonical, quote in result.items()
        if isinstance(quote, dict)
    }
    ctx.performance_intraday_map = result
    if ctx.semantic_audit is not None:
        ctx.semantic_audit["performance_intraday"] = {
            "origin": "metrics_preprocessing",
            "requested_tickers": requested,
            "returned_tickers": tuple(result.keys()),
            "source_tickers": source_tickers,
        }
    return result

def _perf_spark_cell(day_val, raw_ticker: str, intraday_map: dict, *,
                     bg: Optional[str] = None,
                     live: bool = False, prev_label: str = "") -> tuple:
    """Render the 1D cell: a sign-colored % pill (the change vs the previous
    close) above a Markets-style intraday sparkline (green above the previous
    close, red below).

    The sparkline is drawn ONLY when there is a real intraday series (>=2
    points). When the instrument has not traded intraday (illiquid, or the
    market is closed and the vendor exposes no session), there is no chart —
    just the % pill vs the previous close — because a synthetic line would be
    misleading. Returns ``(cell_html, inner_html)``; ``inner_html`` is the
    pill (+ optional LIVE tag) + sparkline without the surrounding ``<td>`` so
    callers that need a specific cell background can wrap it themselves."""
    P = PALETTE
    if is_missing(day_val):
        pill_txt, pill_col, pill_bg = "\u2014", P["muted"], P["page"]
        dv = None
    else:
        dv = float(day_val)
        pill_txt = _pct_compact(dv, signed=True)
        pill_col = P["green"] if dv >= 0 else P["red"]
        pill_bg = P["green_bg"] if dv >= 0 else P["red_bg"]
    pill = (f'<span style="font-size:10px;font-weight:700;color:{pill_col};'
            f'background:{pill_bg};padding:1px 6px;border-radius:999px;'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">{pill_txt}</span>')

    quote = intraday_map.get(raw_ticker) if raw_ticker else None
    intra, feed_baseline = _intraday_quote_parts(quote)
    if intra is not None and len(intra) >= 2:
        # Production uses the exact previous close retained by preprocessing
        # from the same feed as ``intra``. The derived fallback exists only for
        # synthetic portfolio paths and lightweight render fixtures.
        baseline = None
        try:
            candidate = float(feed_baseline)
            if math.isfinite(candidate) and candidate != 0:
                baseline = candidate
        except (TypeError, ValueError):
            pass
        if baseline is None:
            last = float(intra.iloc[-1])
            if dv is not None and (1.0 + dv / 100.0) != 0:
                baseline = last / (1.0 + dv / 100.0)
            else:
                baseline = float(intra.iloc[0])
        # Time-axis intraday: line fills only the elapsed part of the session.
        # ``live`` (from exchange hours) drives whether the session is still in
        # progress, so a closed same-day session renders full width.
        spark = _intraday_spark(intra, baseline, in_progress=live)
    else:
        # No intraday trades → a dashed placeholder line (prev-close
        # reference), so the cell keeps the same height and the pill stays
        # aligned with the intraday rows.
        spark = _flat_dashed_spark()

    bgc = f"background:{bg};" if bg else ""
    # Tag the 1D basis: green "● LIVE" for a live market-open quote, else a
    # muted "PREV. DAY" (last completed session / no intraday trades).
    if live:
        marker = (f'<span style="margin-left:4px;font-size:8px;font-weight:800;'
                  f'color:{P["green"]};letter-spacing:0.04em;vertical-align:middle;">'
                  f'&#9679;&nbsp;LIVE</span>')
    else:
        _pd = prev_label if prev_label else ''
        marker = (f'<span style="margin-left:4px;font-size:8px;font-weight:700;'
                  f'color:{P["subtle"]};letter-spacing:0.04em;vertical-align:middle;">'
                  f'{_pd}</span>') if _pd else ""
    inner = f'<div>{pill}{marker}</div><div style="margin-top:3px;">{spark}</div>'
    cell = f'<td width="96" align="center" style="padding:6px 8px;{bgc}">{inner}</td>'
    return cell, inner

def _portfolio_intraday_series(m, intraday_map: Optional[dict] = None,
                               raw1d: Optional[dict] = None):
    """Value-weighted intraday level path for the whole portfolio.

    Canonical holding tickers key the run-scoped quote catalog. Each quote may
    carry a different, guarded intraday source ticker, but this consumer never
    resolves or fetches it; it uses the exact series and baseline selected by
    preprocessing.
    """
    df = getattr(m, "holdings_df", None)
    if df is None or getattr(df, "empty", True):
        return None
    if not {"ticker", "weight_pct"}.issubset(set(getattr(df, "columns", []))):
        return None
    hp = getattr(m, "holding_performance", None)
    if raw1d is None:
        raw1d = {}
        if hp is not None and not getattr(hp, "empty", True) and "ticker" in hp.columns:
            for _, pr in hp.iterrows():
                raw1d[str(pr.get("ticker", ""))] = pr.get("1d")
    if intraday_map is None:
        intraday_map = dict(getattr(m, "intraday_quotes", {}) or {})
    if not intraday_map:
        return None
    paths = []  # (weight, pct-path indexed by timestamp)
    for ticker_value, weight in zip(df["ticker"], df["weight_pct"]):
        if weight is None:
            continue
        ticker = str(ticker_value)
        quote = intraday_map.get(ticker)
        intra, feed_baseline = _intraday_quote_parts(quote)
        if intra is None or len(intra) < 2:
            continue
        day_return = raw1d.get(ticker)
        try:
            base_i = None
            try:
                candidate = float(feed_baseline)
                if math.isfinite(candidate) and candidate != 0:
                    base_i = candidate
            except (TypeError, ValueError):
                pass
            if base_i is None:
                last = float(intra.iloc[-1])
                denominator = (
                    1.0 + float(day_return) / 100.0
                    if day_return is not None
                    else None
                )
                base_i = (
                    last / denominator
                    if denominator
                    else float(intra.iloc[0])
                )
            if not base_i:
                continue
            path = (intra.astype(float) / base_i - 1.0) * 100.0
            paths.append((float(weight), path))
        except Exception:  # noqa: BLE001
            continue
    if not paths:
        return None
    idx = None
    for _, p in paths:
        idx = p.index if idx is None else idx.union(p.index)
    agg = None
    wsum = 0.0
    for w, p in paths:
        pp = p.reindex(idx).ffill().bfill()
        agg = pp * w if agg is None else agg + pp * w
        wsum += w
    if agg is None or wsum <= 0:
        return None
    port_pct = agg / wsum
    return (1.0 + port_pct / 100.0) * 100.0

def _returns_table_html(period_cols, portfolio: dict, groups: list,
                        *, day_label: str = "1D") -> str:
    """Shared renderer for the grouped per-instrument returns tables.

    Used by both the Performance ("How markets moved") and the Returns
    snapshot sections so their font sizes, spacing, alternating-row
    backgrounds and asset-class · role grouping are identical by
    construction.

    Args:
        period_cols: ordered numeric column keys (e.g. 1w…5y); 1D is always
            the dedicated sparkline column before them.
        portfolio: ``{name, spark_inner, returns}`` where ``returns`` maps
            each period key to ``{value, color}``. ``spark_inner`` is the
            pill+sparkline inner HTML from ``_perf_spark_cell``.
        groups: ordered ``[(class_name, class_color, [(role_name, [row])])]``
            where each ``row`` is ``{name_html, spark_inner, returns}``.
    """
    P = PALETTE

    # Conditional formatting, scaled on each column's own extremes across the
    # rows THIS table renders (see tarzan.export._heat for why per column and
    # per table). Collected before any cell is built, since a cell cannot know
    # its column's range.
    every_row = ([portfolio] + [inst for _c, _col, role_list in groups
                                for _r, insts in role_list for inst in insts])
    scales = {}
    for p in period_cols:
        scales[p] = _heat.column_scale(
            row.get("returns", {}).get(p, {}).get("raw") for row in every_row)

    def _cells(returns_dict, spark_inner, *, weight):
        # 1D sparkline pulled hard left (left-aligned, minimal left padding) so
        # the gap after the name closes and the period columns get that width.
        cells = [uni_cell(spark_inner, align="left", width=84, valign="middle",
                          pad="6px 4px 6px 0")]
        for p in period_cols:
            r = returns_dict.get(p, {"value": "\u2014", "color": P["muted"]})
            neg, pos = scales[p]
            raw = r.get("raw")
            cells.append(uni_cell(
                r["value"],
                # On a saturated tint a green figure on green loses contrast, so
                # the ink takes over; below that the sign colour is kept.
                color=(_heat.heat_ink(raw, neg=neg, pos=pos) or r["color"]),
                weight=weight,
                bg=_heat.heat_bg(raw, neg=neg, pos=pos)))
        return cells

    # 1D column carries no_sep (True) \u2014 it reads with the name block, not the
    # ruled period grid.
    columns = ([(day_label, "left", 84, True)]
               + [(p.upper(), "right") for p in period_cols])
    portfolio_row = {
        "name_html": (f'<span style="color:{P["accent"]};font-weight:700;'
                      f'font-size:12px;">\u2605 {portfolio["name"]}</span>'),
        "cells": _cells(portfolio["returns"], portfolio.get("spark_inner", ""),
                        weight=700),
    }
    uni_groups = [
        (cls, col, [(role, [{"name_html": inst["name_html"],
                             "cells": _cells(inst["returns"],
                                             inst.get("spark_inner", ""), weight=600)}
                            for inst in insts])
                    for role, insts in role_list])
        for cls, col, role_list in groups]
    # Compact numeric cells + a capped name column (so it can't hoard width and
    # leave a gap before the sparkline) + faint vertical separators so a reader
    # can tell 1W from 1M from 1Y at a glance. Role lives in the group header
    # and the ticker trails the name, so names wrap cleanly.
    return render_unified_table("Instrument", columns, uni_groups,
                                portfolio_row=portfolio_row, compact=True,
                                first_col_width=150, separators=True)

def _perf_name_html(name: str, ticker: str, tags: list) -> str:
    """Instrument label used in the returns tables. Delegates to the shared
    :func:`uni_name` (ticker trails the name, role lives in the group header),
    so names wrap cleanly to ~2 lines without a hard clamp — matching the
    earlier layout the user preferred."""
    from tarzan.export.newsletter._constants import uni_name
    return uni_name(name, ticker or "", tags=tuple(tags or ()))

def _build_returns_snapshot(ctx: _NewsletterContext) -> dict:
    """Build the per-holding returns snapshot table.

    Mirrors the Excel ``Performance`` tab and uses the exact same eight
    time-return columns as the "Returns vs benchmarks" table below —
    1D / 1W / 1M / 3M / YTD / 1Y / 3Y / 5Y — so the two newsletter
    tables read as one consistent view. Risk metrics (Sharpe, Vol,
    alpha, beta) are not included here — the Excel report keeps the
    detailed risk-adjusted view for users who need it.

    The TOTAL PORTFOLIO row anchors the table at the top, then each
    holding sorted by asset class (matching the Holdings section
    above), then the benchmarks at the bottom.

    Periods longer than a holding's (or the portfolio's) available
    history render as "—", exactly like the Performance table.
    """
    m = ctx.metrics
    hp = m.holding_performance
    port_full = m.performance_full or {}
    from tarzan.export._format import short_instrument_name

    # Same history span label the Performance section shows in its
    # disclaimer (e.g. "2.0Y"): the consolidated portfolio history is
    # bounded by the youngest holding with >=1Y of data, so the
    # portfolio row's longer periods read "—". Surfacing it here keeps
    # the snapshot honest about why the Total Portfolio row can stop
    # short of the per-instrument columns.
    history_label = str(port_full.get("period_used") or "—")

    # 1D is now its own sparkline column (same concept as the "How markets
    # moved" table), so it is dropped from the numeric columns here. The rest
    # stay aligned with ``_build_performance``.
    period_keys = ["1w", "1m", "3m", "ytd", "1y", "3y", "5y"]

    # Intraday + live flags use the one render-wide batch of exact full
    # symbols already selected in preprocessing. Presentation never maps a
    # holding or benchmark to another venue.
    _snap_intraday = _shared_performance_intraday(ctx)
    _raw1d: dict = {}
    _live1d: dict = {}
    if hp is not None and not hp.empty:
        for _, _pr in hp.iterrows():
            _k = str(_pr.get("ticker", ""))
            _raw1d[_k] = _pr.get("1d")
            _live1d[_k] = bool(_pr.get("live_1d", False))


    # Locate the α/β benchmark's per-period returns so the Total
    # Portfolio row can be colored "did we beat the benchmark this
    # period?" — identical logic to the Performance table below, so the
    # two tables agree. Per-instrument rows stay colored by sign.
    ab_bench_returns: dict = {}
    ab_bench_name = ctx.benchmark_alpha_beta or "S&P 500"
    if hp is not None and not hp.empty and "type" in hp.columns and "name" in hp.columns:
        bench_match = hp[
            hp["type"].astype(str).str.contains("enchmark", case=False, na=False)
            & hp["name"].astype(str).str.contains(
                ab_bench_name, case=False, na=False, regex=False,
            )
        ]
        if not bench_match.empty:
            ab_row = bench_match.iloc[0]
            for key in period_keys:
                ab_bench_returns[key] = ab_row.get(key)

    def _vs_bench_color(value: float, bench_value) -> str:
        """Green if we beat the α/β benchmark by >0.25pp this period,
        amber within ±0.25pp (noise), red if we underperform. Falls back
        to sign-based coloring when the benchmark value is missing."""
        if is_missing(bench_value):
            return PALETTE["green"] if value >= 0 else PALETTE["red"]
        delta = value - float(bench_value)
        if abs(delta) <= 0.25:
            return PALETTE["amber"]
        return PALETTE["green"] if delta > 0 else PALETTE["red"]

    def _returns_dict(source: dict, *, is_portfolio: bool) -> dict:
        """Per-period ``{value, color, raw}`` map for the shared table renderer.

        ``raw`` is the signed percent the cell displays. The renderer needs it
        to scale the conditional-formatting ramp on the column's own extremes;
        formatted strings cannot be compared.
        """
        out: dict = {}
        for key in period_keys:
            val = source.get(key) if source else None
            if is_missing(val):
                out[key] = {"value": "\u2014", "color": PALETTE["subtle"],
                            "raw": None}
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                out[key] = {"value": "\u2014", "color": PALETTE["subtle"],
                            "raw": None}
                continue
            color = (_vs_bench_color(v, ab_bench_returns.get(key)) if is_portfolio
                     else (PALETTE["green"] if v >= 0 else PALETTE["red"]))
            out[key] = {"value": _pct_compact(v, signed=True), "color": color,
                        "raw": v}
        return out

    df = m.holdings_df
    if df is None or df.empty:
        return {"available": False, "table_html": "",
                "history_label": history_label, "benchmark_alpha_beta": ab_bench_name}

    # Per-holding period returns (join by ticker).
    perf_by_ticker: dict[str, dict] = {}
    if hp is not None and not hp.empty and "ticker" in hp.columns:
        type_col = hp["type"].astype(str).str.lower() if "type" in hp.columns else None
        is_holding = type_col.str.contains("portfolio") if type_col is not None else None
        holdings_perf = hp[is_holding] if is_holding is not None else hp
        for _, pr in holdings_perf.iterrows():
            perf_by_ticker[str(pr.get("ticker", ""))] = {k: pr.get(k) for k in period_keys}

    # Curated taxonomy (asset_class already on df) for the shared grouping
    # engine, so the snapshot groups exactly like every other instrument table.
    from tarzan import config as cfg
    _tax = cfg.instrument_taxonomy()

    # Portfolio (highlighted) row. The portfolio has no single ticker, but its
    # holdings trade intraday, so build a value-weighted synthetic intraday
    # path (reusing the already-fetched holdings intraday) for a real 1D
    # sparkline; fall back to the dashed placeholder when unavailable.
    _pf_series = _portfolio_intraday_series(
        m, intraday_map=_snap_intraday, raw1d=_raw1d
    )
    _prev_lbl = _prev_session_label(m)
    if _pf_series is not None and len(_pf_series) >= 2:
        _, port_inner = _perf_spark_cell(
            port_full.get("1d"), _PF_INTRA_KEY, {_PF_INTRA_KEY: _pf_series},
            live=bool(port_full.get("1d_live")), prev_label=_prev_lbl)
    else:
        _, port_inner = _perf_spark_cell(
            port_full.get("1d"), "", {}, live=bool(port_full.get("1d_live")),
            prev_label=_prev_lbl)
    portfolio = {"name": "Total Portfolio", "spark_inner": port_inner,
                 "returns": _returns_dict(port_full, is_portfolio=True)}

    # Build one row per holding, then group via the SHARED engine (class → role,
    # ordered) so this table splits/colours instruments identically to Holdings
    # and the Optimizer.
    row_items = []
    for _, h in df.iterrows():
        ticker = str(h.get("ticker", "") or "")
        isin = str(h.get("isin", "") or "")
        raw_name = str(h.get("name", "") or ticker)
        display_tk = _display_ticker(ticker) or ""
        _, inner = _perf_spark_cell(
            _raw1d.get(ticker), ticker, _snap_intraday,
            live=bool(_live1d.get(ticker, False)), prev_label=_prev_lbl)
        row_items.append({
            "_ac": str(h.get("asset_class", "") or "") or "Other",
            "_isin": isin, "_ticker": ticker,
            "name_html": _perf_name_html(
                display_instrument_name(isin, ticker, raw_name),
                display_tk, []),
            "spark_inner": inner,
            "returns": _returns_dict(perf_by_ticker.get(ticker, {}), is_portfolio=False),
        })
    groups = group_by_class_role(
        row_items, asset_class=lambda r: r["_ac"],
        isin=lambda r: r["_isin"], ticker=lambda r: r["_ticker"], taxonomy=_tax)

    return {
        "available": True,
        "table_html": _returns_table_html(
            period_keys, portfolio, groups,
            day_label=day_column_label(
                m, live=bool(port_full.get("1d_live"))
                or any(_live1d.values()))),
        "history_label": history_label,
        "benchmark_alpha_beta": ab_bench_name,
    }

def _build_movers(ctx: _NewsletterContext) -> dict:
    """Find best & worst performer over the last week."""
    m = ctx.metrics
    if m.holding_performance.empty:
        return {"available": False}

    hp = m.holding_performance
    # Filter to actual portfolio holdings (not benchmarks)
    if "type" in hp.columns:
        hp = hp[hp["type"].astype(str).str.lower().str.contains("portfolio") |
                ~hp["type"].astype(str).str.lower().str.contains("benchmark")]
    if hp.empty or "1w" not in hp.columns:
        return {"available": False}

    sorted_hp = hp.sort_values("1w", ascending=False, na_position="last")
    best = sorted_hp.iloc[0]
    worst = sorted_hp.iloc[-1]

    df = m.holdings_df

    def _enrich(row):
        ticker = row.get("ticker", "")
        match = df[df["ticker"] == ticker] if not df.empty else pd.DataFrame()
        klass = match["asset_class"].iloc[0] if not match.empty else "Equities"
        value = float(match["current_value"].iloc[0]) if not match.empty else 0.0
        pct = float(row.get("1w") or 0.0)
        eur = value * pct / 100
        return {
            "name": row.get("name", ticker),
            "ticker": ticker,
            "asset_class": klass,
            "asset_color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "pct": _pct(pct, signed=True),
            "is_positive": pct >= 0,
            "eur": _eur_smart(abs(eur)),
        }

    return {
        "available": True,
        "best": _enrich(best),
        "worst": _enrich(worst),
        "benchmark_name": ctx.benchmark_alpha_beta,
    }

def _build_performance(ctx: _NewsletterContext) -> dict:
    """Build the performance section: instruments sectioned by asset class &
    role, with a 1D intraday sparkline and multi-horizon % returns."""
    from tarzan import config as cfg
    m = ctx.metrics

    # Portfolio history span shown in the disclaimer. We use the
    # ``period_used`` label produced by _populate_perf_row on
    # performance_full, which reflects the same 5y-capped, holdings≥1Y
    # window used for all the metrics in this section. Falls back to
    # computing from portfolio_history_full when missing.
    pf = m.performance_full or {}
    history_label = str(pf.get("period_used") or "—")
    if history_label == "—":
        ph_full = m.portfolio_history
        if ph_full is not None and len(ph_full) >= 2:
            days = int((ph_full.index[-1] - ph_full.index[0]).days)
            yrs = days / 365.25
            if yrs >= 4.9:
                history_label = "5Y+"
            elif yrs >= 1.0:
                history_label = f"{yrs:.1f}Y"
            elif days >= 30:
                history_label = f"{int(round(days / 30))}M"
            elif days > 0:
                history_label = f"{days}D"

    # Period order shown in the Returns table (mirrors the Excel
    # Performance tab): 1D first, then progressively longer windows.
    periods = ("1d", "1w", "1m", "3m", "ytd", "1y", "3y", "5y")

    def _as_float(value):
        """The signed percent behind a cell, or None. The heat ramp needs the
        number; the cell only keeps a formatted string."""
        if is_missing(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _color_sign(value) -> str:
        """Sign-aware color for a period return cell — used on benchmarks."""
        if is_missing(value):
            return PALETTE["muted"]
        return PALETTE["green"] if float(value) >= 0 else PALETTE["red"]

    # Locate the α/β benchmark row (S&P 500 by default) so the portfolio
    # row can be colored "did we beat the benchmark on this period?"
    # rather than just "is it positive?". Sign-based coloring on the
    # portfolio row tends to look like a cheerleader — every positive
    # period is green even when we underperform.
    hp = m.holding_performance
    ab_bench_returns: dict = {}
    ab_bench_name = ctx.benchmark_alpha_beta or "S&P 500"
    if not hp.empty and "type" in hp.columns:
        bench_match = hp[
            hp["type"].astype(str).str.contains("enchmark", case=False, na=False)
            & hp["name"].astype(str).str.contains(
                ab_bench_name, case=False, na=False, regex=False,
            )
        ]
        if not bench_match.empty:
            ab_row = bench_match.iloc[0]
            for p in periods:
                ab_bench_returns[p] = ab_row.get(p)

    def _color_vs_bench(value, bench_value) -> str:
        """Color the portfolio cell by delta vs the α/β benchmark on the
        same period: green if we beat by >0.25pp, amber within ±0.25pp
        (statistical noise), red if we underperform. Falls back to
        sign-based when the benchmark value is unavailable."""
        if is_missing(value):
            return PALETTE["muted"]
        if (bench_value is None
                or (isinstance(bench_value, float) and pd.isna(bench_value))):
            return _color_sign(value)
        delta = float(value) - float(bench_value)
        if abs(delta) <= 0.25:
            return PALETTE["amber"]
        return PALETTE["green"] if delta > 0 else PALETTE["red"]

    def _build_portfolio_returns_dict(source: dict) -> dict:
        return {
            p: {
                "value": _pct_compact(source.get(p), signed=True),
                "color": _color_vs_bench(source.get(p), ab_bench_returns.get(p)),
                "raw": _as_float(source.get(p)),
            }
            for p in periods
        }

    def _build_bench_returns_dict(source: dict) -> dict:
        return {
            p: {
                "value": _pct_compact(source.get(p), signed=True),
                "color": _color_sign(source.get(p)),
                "raw": _as_float(source.get(p)),
            }
            for p in periods
        }

    # Portfolio row
    portfolio_row = {
        "name": "Your portfolio",
        "tag": None,
        "is_portfolio": True,
        "returns": _build_portfolio_returns_dict(pf),
    }

    # Benchmark rows (from holding_performance, type contains 'enchmark').
    # Each row is tagged with its curated (asset_class, role) from the
    # instrument taxonomy so the section can be grouped and ordered.
    taxonomy = cfg.instrument_taxonomy()
    benchmark_rows = []
    if not hp.empty and "type" in hp.columns:
        ab_name = (ctx.benchmark_alpha_beta or "").strip().lower()
        geo_name = (ctx.benchmark_geo or "").strip().lower()
        bench_df = hp[hp["type"].astype(str).str.contains("enchmark", case=False, na=False)]
        for _, r in bench_df.iterrows():
            name = str(r.get("name") or r.get("ticker", ""))
            name_norm = name.strip().lower()
            # Tag the configured benchmarks. The same index can be both
            # the α/β and the geo reference (e.g. MSCI ACWI), so we may
            # show both tags on one row.
            tags = []
            if ab_name and name_norm == ab_name:
                tags.append(("α/β", PALETTE["accent"], PALETTE["accent_bg"]))
            if geo_name and name_norm == geo_name:
                tags.append(("GEO", PALETTE["accent"], PALETTE["accent_bg"]))
            raw_ticker = str(r.get("ticker") or "").strip()
            bare = normalize_ticker(raw_ticker)
            asset_class, role = taxonomy.get(bare, (None, None))
            benchmark_rows.append({
                # Display name goes through the SAME shortener as the holding
                # rows so "iShares Nasdaq 100 UCITS ETF" reads like the rest of
                # the table (tag-matching above uses the raw name, not this).
                "name": display_instrument_name(r.get("isin"), raw_ticker, name),
                "ticker": _display_ticker(r.get("ticker")),
                "raw_ticker": raw_ticker,
                "asset_class": asset_class,
                "role": role,
                "d1": r.get("1d"),
                "live": bool(r.get("live_1d", False)),
                "tags": tags,
                # Back-compat single tag (first one) for any old template ref.
                "tag": tags[0] if tags else None,
                "is_portfolio": False,
                "returns": _build_bench_returns_dict(r.to_dict()),
            })

    # Risk metrics are now rendered in their own unified Risk Profile
    # section by ``_build_risk_profile``; we no longer return separate
    # chip data here.

    # Order-list returns (only present when an order list was supplied;
    # all None for a holdings-only run so the template renders nothing).
    m = ctx.metrics
    returns_block = None
    if m.xirr_pct is not None or m.twror_pct is not None:
        prov = m.returns_provenance or {}

        def _identifiers(values) -> set[str]:
            return {
                str(value).strip()
                for value in (values or [])
                if value is not None
                and not (isinstance(value, float) and pd.isna(value))
                and str(value).strip()
            }

        synthetic_ids = _identifiers(prov.get("synthetic"))
        carry_flat_ids = _identifiers(prov.get("carry_flat"))
        fallback_ids = synthetic_ids | carry_flat_ids
        excluded_ids = _identifiers(prov.get("excluded"))
        current_ids = (
            _identifiers(m.holdings_df["isin"].tolist())
            if not m.holdings_df.empty and "isin" in m.holdings_df.columns
            else set()
        )
        current_fallback_ids = fallback_ids & current_ids
        historical_closed_fallback_ids = fallback_ids - current_ids

        returns_block = {
            "xirr": _pct(m.xirr_pct, signed=True) if m.xirr_pct is not None else None,
            "twror": _pct(m.twror_pct, signed=True) if m.twror_pct is not None else None,
            "twror_annualized": (
                _pct(m.twror_annualized_pct, signed=True)
                if m.twror_annualized_pct is not None else None
            ),
            "coverage": (
                _pct(m.returns_coverage_pct, decimals=0)
                if m.returns_coverage_pct is not None else None
            ),
            # Back-compat: total distinct order-price fallback identifiers.
            "fallback_count": len(fallback_ids),
            "current_fallback_count": len(current_fallback_ids),
            "historical_closed_fallback_count": len(historical_closed_fallback_ids),
            "excluded_count": len(excluded_ids),
        }

    # One exact-symbol batch is shared with the holding snapshot above. Missing
    # series remain unavailable; no renderer-side listing fallback is allowed.
    P = PALETTE
    period_cols = ("1w", "1m", "3m", "ytd", "1y", "3y", "5y")
    intraday_map = _shared_performance_intraday(ctx)

    # Portfolio row (highlighted): a real 1D sparkline from a value-weighted
    # synthetic intraday path over the holdings (the portfolio has no single
    # ticker, but its holdings trade intraday); dashed placeholder when the
    # exact-symbol intraday data is unavailable.
    _pf_series = _portfolio_intraday_series(m, intraday_map=intraday_map)
    _prev_lbl = _prev_session_label(m)
    if _pf_series is not None and len(_pf_series) >= 2:
        _, port_inner = _perf_spark_cell(
            pf.get("1d"), _PF_INTRA_KEY, {_PF_INTRA_KEY: _pf_series},
            live=bool(pf.get("1d_live")), prev_label=_prev_lbl)
    else:
        _, port_inner = _perf_spark_cell(
            pf.get("1d"), "", {}, live=bool(pf.get("1d_live")), prev_label=_prev_lbl)
    portfolio = {"name": portfolio_row["name"], "spark_inner": port_inner,
                 "returns": portfolio_row["returns"]}

    # Group benchmark rows by asset class → role, in the configured order,
    # then hand off to the shared table renderer.
    raw_groups = group_by_class_role(
        benchmark_rows, asset_class=lambda r: r.get("asset_class") or "Other",
        role=lambda r: r.get("role"))
    groups = []
    for ac, col, role_list in raw_groups:
        rendered_roles = []
        for role, rows in role_list:
            insts = []
            for r in rows:
                _, inner = _perf_spark_cell(
                    r.get("d1"), r.get("raw_ticker"), intraday_map,
                    live=bool(r.get("live")), prev_label=_prev_lbl)
                insts.append({
                    "name_html": _perf_name_html(r["name"], r.get("ticker"),
                                                 r.get("tags")),
                    "spark_inner": inner,
                    "returns": r["returns"],
                })
            rendered_roles.append((role, insts))
        groups.append((ac, col, rendered_roles))

    table_html = _returns_table_html(
        period_cols, portfolio, groups,
        day_label=day_column_label(
            m, live=bool(pf.get("1d_live"))
            or any(bool(r.get("live")) for r in benchmark_rows)))

    subtitle_html = (
        f'Portfolio vs {ctx.benchmark_alpha_beta or "S&amp;P 500"}: '
        f'<span style="color:{P["green"]};font-weight:700;">&#9679;</span> beat &middot; '
        f'<span style="color:{P["amber"]};font-weight:700;">&#9679;</span> in line &middot; '
        f'<span style="color:{P["red"]};font-weight:700;">&#9679;</span> under'
    )

    return {
        "title": "How markets moved",
        # Two sections cannot share a kicker: this one is the per-instrument
        # grid, not the P&L matrix at the top of the body.
        "kicker": "Returns by asset class",
        "subtitle_html": subtitle_html,
        "table_html": table_html,
        "portfolio_row": portfolio_row,
        "benchmark_rows": benchmark_rows,  # show all configured benchmarks
        "periods": list(periods),
        "history_label": history_label,
        "benchmark_alpha_beta": ctx.benchmark_alpha_beta,
        "benchmark_geo": ctx.benchmark_geo,
        "returns": returns_block,
    }

def _build_risk_profile(ctx: _NewsletterContext) -> dict:
    """Build the "Historical risk profile" table (transposed layout).

    Data source is ``metrics.historical_risk`` (MetricsEngine._historical_risk):
    each instrument is measured over its OWN full available price history
    (uncapped, span shown per row), and the portfolio row is a current-weight
    static backtest over the common window of holdings with ≥1Y of history.
    This deliberately trades the old apples-to-apples single window for
    maximum history per series.

    Columns are the risk metrics (CAGR, Vol, Sharpe, Sortino, Max DD, Ulcer,
    VaR 95%, CVaR 95%, α, β). α and β are computed against the configured α/β
    benchmark (so that row reads β≈1.00 / α≈0), noted in a footnote.
    """
    m = ctx.metrics
    hr = m.historical_risk or {}
    if not hr.get("available"):
        return {"available": False, "rows": [], "columns": []}

    ab_bench_name = ctx.benchmark_alpha_beta or "S&P 500"
    ab_name = (ctx.benchmark_alpha_beta or "").strip().lower()
    geo_name = (ctx.benchmark_geo or "").strip().lower()

    def _fmt_pct(v) -> str:
        if is_missing(v):
            return "—"
        return _pct(float(v))

    def _fmt_num(v) -> str:
        if is_missing(v):
            return "—"
        return f"{float(v):.2f}"

    # Metric columns, in display order. Tuple: (label, key, is_pct, note).
    # α and β carry a "*" footnote marker because they are referenced to
    # a specific market index. Ulcer Index sits next to Max DD as a
    # duration-aware companion (RMS of drawdowns).
    # (label, metrics key, is_pct, note, ratings key). The last field says
    # which ``metric_ratings`` entry in constants.yaml describes the metric;
    # its ``invert`` flag is what drives the per-column tint direction. That
    # direction cannot be inferred from the sign -- a positive volatility is
    # not good news, and a -7% drawdown beats a -21% one -- and it is already
    # declared in configuration, so reading it here keeps one source of truth
    # instead of a second copy that can drift from the legend beside it.
    # Beta carries None: the configured bands rate it as market exposure, which
    # is a property to know rather than a score to win, so it stays uncoloured.
    metric_cols = [
        ("CAGR", "cagr", True, "", "cagr"),
        ("Vol", "volatility", True, "", "volatility"),
        ("Sharpe", "sharpe", False, "", "sharpe"),
        ("Sortino", "sortino", False, "", "sortino"),
        ("Max DD", "max_drawdown", True, "", "max_drawdown"),
        ("Ulcer", "ulcer_index", True, "", "ulcer_index"),
        # "95%" is spelled out in the legend below; drop it from the column
        # header so these two cells stay narrow in the 10-column table.
        ("VaR", "var_95", True, "", "var_pct"),
        ("CVaR", "cvar_95", True, "", "cvar_pct"),
        ("\u03b1", "alpha", True, "*", "alpha"),
        ("\u03b2", "beta", False, "*", None),
    ]

    # ``invert: true`` in constants.yaml means "a smaller magnitude is better",
    # and it is written against the metric's ABSOLUTE value: max_drawdown is
    # banded at [-15, -30] and VaR at [0.8, 1.5] while both carry the flag. So
    # an inverted metric is ranked on |value| with lower better, and every other
    # metric on its signed value with higher better. Ranking VaR on the signed
    # number instead greens the worst daily loss in the table, which is what a
    # first pass at reading the flag did.
    from tarzan import config as _rating_cfg
    _ratings = _rating_cfg.metric_ratings() or {}

    def _inverted(rating_key) -> Optional[bool]:
        """True when smaller-is-better, None when the metric is not rated."""
        if not rating_key or rating_key not in _ratings:
            return None
        return bool((_ratings.get(rating_key) or {}).get("invert", False))

    def _raw_from(metrics: dict) -> list:
        return [None if is_missing((metrics or {}).get(key))
                else float((metrics or {}).get(key))
                for _label, key, _p, _n, _rk in metric_cols]

    def _cells_from(metrics: dict) -> list[str]:
        out = []
        for _label, key, is_pct, _note, _rk in metric_cols:
            v = (metrics or {}).get(key)
            out.append(_fmt_pct(v) if is_pct else _fmt_num(v))
        return out

    def _tags_for(name: str) -> list:
        """Reuse the Performance table's α/β + GEO pins so the reference
        benchmarks are recognisable here too."""
        name_norm = (name or "").strip().lower()
        tags = []
        if ab_name and name_norm == ab_name:
            tags.append(("\u03b1/\u03b2", PALETTE["accent"], PALETTE["accent_bg"]))
        if geo_name and name_norm == geo_name:
            tags.append(("GEO", PALETTE["accent"], PALETTE["accent_bg"]))
        return tags

    from tarzan.export.newsletter._constants import (
        render_unified_table, uni_cell, uni_name)

    # ── Conditional formatting, one scale per column ────────────────────
    # Collected in a first pass over every series that will be drawn, so a
    # column's greenest cell is the best of the series actually shown. The
    # portfolio row takes part: where it sits among the alternatives is the
    # question the section exists to answer.
    _scale_rows = [_raw_from((hr.get("portfolio") or {}).get("metrics"))] if hr.get("portfolio") else []
    _scale_rows += [_raw_from(inst.get("metrics")) for inst in hr.get("instruments", [])]
    def _scale_value(j: int, v):
        """The number the column is ranked on: magnitude for an inverted
        metric, the signed value otherwise."""
        if v is None:
            return None
        return abs(v) if _inv[j] else v

    _inv = [_inverted(rk) for (_l, _k, _p, _n, rk) in metric_cols]
    _scales, _dirs = [], []
    for j in range(len(metric_cols)):
        # Beta is rated but excluded on purpose: the bands describe market
        # exposure, which is a property to know rather than a score to win.
        rated = _inv[j] is not None and metric_cols[j][1] != "beta"
        _dirs.append(None if not rated else (not _inv[j]))
        _scales.append(_heat.rank_scale([_scale_value(j, r[j]) for r in _scale_rows])
                       if rated else None)

    def _metric_cells(metrics, *, weight, color):
        raw = _raw_from(metrics)
        cells = []
        for j, txt in enumerate(_cells_from(metrics)):
            sc, hib = _scales[j], _dirs[j]
            bg = ink = None
            if sc is not None:
                lo, hi = sc
                v = _scale_value(j, raw[j])
                bg = _heat.rank_bg(v, lo=lo, hi=hi, higher_is_better=hib)
                ink = _heat.rank_ink(v, lo=lo, hi=hi, higher_is_better=hib)
            cells.append(uni_cell(txt, color=ink or color, weight=weight, bg=bg))
        return cells

    port = hr.get("portfolio")
    if not (port or hr.get("instruments")):
        return {"available": False, "rows": [], "columns": []}

    portfolio_row = None
    if port:
        span = port.get("span_label", "\u2014")
        span_html = (f'<span style="display:inline-block;margin-left:6px;font-size:9px;'
                     f'font-weight:700;color:{PALETTE["accent"]};opacity:0.75;'
                     f'vertical-align:middle;">{span}</span>' if span else "")
        portfolio_row = {
            "name_html": (f'<span style="color:{PALETTE["accent"]};font-weight:700;'
                          f'font-size:12px;">\u2605 {port.get("label", "Your portfolio")}'
                          f'</span>{span_html}'),
            "cells": _metric_cells(port.get("metrics"), weight=700,
                                   color=PALETTE["ink"]),
        }

    # Group instruments by asset_class -> role via the shared engine, then hand
    # off to the ONE unified renderer (same shell as every other table).
    uni_groups = []
    for ac, gc, role_list in group_by_class_role(
            hr.get("instruments", []),
            asset_class=lambda inst: inst.get("asset_class") or "Other",
            role=lambda inst: inst.get("role")):
        rendered = []
        for role, insts in role_list:
            block = [{
                "name_html": uni_name(
                    display_instrument_name(inst.get("isin"),
                                            inst.get("ticker"),
                                            inst.get("label", "")),
                    _display_ticker(inst.get("ticker")) or "",
                    tags=_tags_for(inst.get("label", "")),
                    span=inst.get("span_label", "\u2014")),
                "cells": _metric_cells(inst.get("metrics"), weight=600,
                                       color=PALETTE["muted"]),
            } for inst in insts]
            rendered.append((role, block))
        uni_groups.append((ac, gc, rendered))

    columns = [(f"{label}{note}", "right")
               for (label, _k, _p, note, _rk) in metric_cols]
    # Compact mode: 10 numeric columns, so tighten value-cell padding to keep
    # the instrument-name column wide enough. Faint vertical separators help
    # the reader track which metric a number belongs to across the wide row.
    table_html = render_unified_table("Series", columns, uni_groups,
                                      portfolio_row=portfolio_row, compact=True,
                                      first_col_width=132, separators=True)

    description = (
        "Each instrument is measured over its full available price history "
        "\u2014 the span is shown next to its name. Your portfolio is a "
        "backtest at today's weights held constant, over the longest window "
        "where every holding with at least 1 year of history overlaps. "
        "Colour is scaled per column over the series shown, green toward the "
        "better end of each metric, so the spans being unequal is visible in "
        "the span labels rather than hidden by the shading."
    )

    return {
        "available": True,
        "title": "Historical risk profile",
        # The section heading carries this now, so it names the one thing the
        # reader needs before reading a column: the window each series covers.
        "subtitle": (
            f'Portfolio over {(port or {}).get("span_label") or "its"} of '
            f'history; every instrument over its own full history, shown next '
            f'to its name.'
        ),
        "table_html": table_html,
        "description": description,
        # Backtest transparency note (holdings excluded / renormalized).
        "portfolio_note": (port or {}).get("note"),
        # Footnote: α and β are both referenced to the α/β benchmark.
        "alpha_beta_note": (
            f"\u03b1 and \u03b2 are computed against {ab_bench_name}."
        ),
        "legend": _build_risk_legend(),
        "benchmark_alpha_beta": ctx.benchmark_alpha_beta,
        "benchmark_geo": ctx.benchmark_geo,
    }

def _build_risk_legend() -> list[dict]:
    """Build the Risk Profile legend rows mirroring the Excel Performance
    tab Legend. Sources thresholds and units from
    ``constants.yaml::metric_ratings`` so the two stay in sync.

    Each entry: {label, strong, fair, weak, description}. The α and β
    rows here describe the metrics in general; the specific benchmark
    used for α/β is shown in the table label above (e.g. "α (vs S&P 500)")
    so the legend stays generic and reusable across configurations.
    """
    from tarzan import config as cfg
    ratings = cfg.metric_ratings() or {}

    # (label, ratings_key, description). Order matches the Risk Profile
    # table above. Descriptions are kept short to fit a compact layout
    # — the Excel Legend has the longer phrasing.
    legend_specs = [
        ("CAGR", "cagr",
         "Compound Annual Growth Rate. Yearly return that would grow your "
         "portfolio from start to end value, with compounding."),
        ("Volatility", "volatility",
         "Annualized standard deviation of daily returns, scaled from daily "
         "to yearly."),
        ("Sharpe", "sharpe",
         "(CAGR − risk-free rate) / Volatility. Return per unit of total "
         "risk."),
        ("Sortino", "sortino",
         "Like Sharpe but penalizes only downside volatility. Usually "
         "higher than Sharpe; the gap shows good (upside) volatility."),
        ("Max Drawdown", "max_drawdown",
         "Worst peak-to-trough loss over the period, measured from the "
         "running high to the low that followed it."),
        ("Ulcer Index", "ulcer_index",
         "Root-mean-square of drawdowns from the running peak; captures both "
         "depth and time spent underwater. Lower is smoother; penalizes long "
         "slumps more than a one-point Max DD."),
        ("VaR 95%", "var_pct",
         "Daily loss exceeded only 5% of the time (historical sim). "
         "Non-parametric: no normal-distribution assumption."),
        ("CVaR 95%", "cvar_pct",
         "Average loss on the worst 5% of days. More negative than VaR; "
         "captures tail risk."),
        (f"\u03b1", "alpha",
         "Extra annual return vs the benchmark, after adjusting for "
         "portfolio risk (CAPM). Positive = beat the market on risk-adjusted basis."),
        (f"\u03b2", "beta",
         "How much the portfolio moves when the benchmark moves 1%. "
         "β=1 in line, β=0.5 half as reactive, β≈0 uncorrelated."),
    ]

    def _fmt(value: Optional[float], unit: str) -> str:
        if value is None:
            return "\u2014"
        v = float(value)
        # Drop the ".0" on integer thresholds so the bands read tight
        # ("<3%" not "< 3.0%") — they sit inline next to the metric name.
        num = f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
        return f"{num}{unit}"

    legend_rows = []
    for label, key, description in legend_specs:
        spec = ratings.get(key, {}) or {}
        thresholds = spec.get("thresholds", [None, None])
        invert = bool(spec.get("invert", False))
        unit = spec.get("unit", "")
        good_t, warn_t = (thresholds + [None, None])[:2]

        if good_t is None or warn_t is None:
            strong = fair = weak = "\u2014"
        elif invert:
            # Lower-is-better metrics: better when below good threshold.
            strong = f"<{_fmt(abs(good_t), unit)}"
            fair = f"{_fmt(abs(warn_t), unit)}\u2013{_fmt(abs(good_t), unit)}"
            weak = f">{_fmt(abs(warn_t), unit)}"
        else:
            strong = f">{_fmt(good_t, unit)}"
            fair = f"{_fmt(warn_t, unit)}\u2013{_fmt(good_t, unit)}"
            weak = f"<{_fmt(warn_t, unit)}"

        legend_rows.append({
            "label": label,
            "strong": strong,
            "fair": fair,
            "weak": weak,
            "description": description,
        })
    return legend_rows

