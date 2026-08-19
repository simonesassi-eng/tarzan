"""Performance / returns-table / risk / markets section builders."""

from __future__ import annotations

import logging
import math
from html import escape as _esc
from typing import Optional

import pandas as pd

from tarzan.models.instrument_key import normalize_ticker
from tarzan.export._format import (
    display_instrument_name,
    greek_safe,
    eur_smart as _eur_smart,
)
from tarzan.export import _charts as _charts
from tarzan.export._perf_series import (
    _norm_series,
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
    TYPE,
    TYPE_PX,
    _NewsletterContext,
    _PF_INTRA_KEY,
    group_by_class_role,
    render_unified_table,
    uni_cell,
)
from tarzan.export.newsletter._format import (
    _display_ticker,
    _pct,
    _pct_compact,
    _signed,
    is_missing,
)
from tarzan.export.newsletter._charts import (
    _day_spark,
    _flat_dashed_spark,
    _intraday_spark,
    day_column_label,
    spark_note,
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

    def is_continuous_market(_ticker):  # safe default if the import below fails
        return False

    def session_caption(_ticker):  # safe default if the import fails
        return ""

    def market_status(_ticker, _now=None):  # safe default if the import fails
        return None, ""

    def session_span(_ticker, _now=None):  # safe default if the import fails
        return None

    try:
        from tarzan.data.market_quotes import (fetch_market_quotes, CATEGORY_ORDER,
                                               is_continuous_market, session_caption,
                                               market_status, session_span)
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
        up = str(sym).upper()
        # ONE answer to "is this venue trading right now", for every instrument
        # kind: market_status resolves the futures/FX weekly cycle and crypto
        # too, and returns None only for a venue with no modelled session.
        trading, _day = market_status(sym)
        ss = d.get("spark_series")
        if ss is not None and len(ss) >= 2:
            if up.endswith("=F") or up.endswith("=X"):
                # Futures/FX have no cash session to draw on, so they keep the
                # elapsed-fraction axis -- forcing in_progress=False stretched a
                # chart an hour into its ~23-24h window to fill the full cell,
                # making it look like a completed session.
                in_progress = bool(trading)
                sess_hours = 23.0 if up.endswith("=F") else 24.0
            else:
                # Crypto (-USD) never closes and has no session boundary to
                # grow from, so it keeps the full-width view. Exchange-listed
                # instruments are drawn on their real session window (span),
                # which makes the flag moot; it still covers venues whose
                # session is unmodelled.
                in_progress = (False if is_continuous_market(sym) else trading)
                sess_hours = None
            return _intraday_spark(ss, d.get("baseline", d["value"]),
                                   w=66, h=20, span=session_span(sym),
                                   in_progress=in_progress,
                                   session_hours=sess_hours)
        if d.get("stale_session") or trading:
            # Nothing to draw: either the venue has moved on to a session this
            # data does not cover yet (every run in the first minutes after an
            # open), or Yahoo has no intraday feed for the listing at all (Bund
            # 10Y). While the venue trades, BOTH used to draw ~40 DAILY closes
            # stretched to full width -- shaped exactly like a completed session
            # path, on a row whose own "Op." badge said it had only just started.
            # The dashed placeholder is itself the statement that there is no
            # session to draw; it is captioned only when the level and % belong
            # to a session that needs naming (the stale case).
            day = d.get("observed_day") if d.get("stale_session") else None
            return _flat_dashed_spark(
                w=66, h=20,
                note=(f'{day.strftime("%a")} close' if day is not None else ""))
        # Venue closed and no intraday feed for it -- a real, if uneven, gap in
        # Yahoo's coverage for some exchanges. The last ~40 daily closes are a
        # COMPLETE period, so filling the width claims nothing false. Label
        # matches _flat_dashed_spark's own wording for the same gap.
        chart = _day_spark(d.get("spark", []), d.get("baseline", d["value"]),
                           w=66, h=20, stretch=True)
        if not chart:
            return chart
        return chart + spark_note("no intraday")

    def _hours_line(d: dict) -> str:
        """Local trading hours (or \u224824h for a continuously traded
        instrument) plus an open/closed dot and the calendar day that
        status refers to \u2014 so "Cl." is never ambiguous about which
        session it means, and "Op." about which day is live. Abbreviated
        (not "Closed"/"Open") to leave room for the wider chart column."""
        sym = d.get("symbol", "")
        cap = session_caption(sym)
        if not cap:
            return ""
        is_open, day = market_status(sym)
        if is_open is None:
            return (f'<div style="font-size:{TYPE_PX["label"]}px;'
                    f'color:{P["subtle"]};margin-top:1px;">{cap}</div>')
        dot_col = P["green"] if is_open else P["subtle"]
        status = "Op." if is_open else "Cl."
        day_suffix = f" {day}" if day else ""
        return (f'<div style="font-size:{TYPE_PX["label"]}px;'
                f'color:{P["subtle"]};margin-top:1px;">{cap} &middot; '
                f'<span style="color:{dot_col};">&#9679;</span> '
                f'{status}{day_suffix}</div>')

    def _row(d: dict) -> str:
        up = d["pct"] >= 0
        col = P["green"] if up else P["red"]
        name = d["name"]
        # Tag futures so a full-width sparkline reads as a continuously traded
        # contract (change vs previous settlement), not a finished session.
        # Idempotent: a name that already carries the tag (set directly in
        # MARKETS, so it is unique from its cash-index counterpart) is left
        # alone rather than doubled.
        if (str(d.get("symbol", "")).upper().endswith("=F")
                and not name.endswith("(FUT)")):
            name = f"{name} (FUT)"
        level = (f'{d["value"]:,.0f}' if abs(d["value"]) >= 1000
                 else f'{d["value"]:,.2f}')
        td = (f'padding:4px 5px;border-bottom:1px solid {P["row_rule"]};'
              f'font-variant-numeric:tabular-nums;')
        return (
            f'<tr>'
            f'<td style="{td}{TYPE["data"]}color:{P["ink"]};'
            f'white-space:nowrap;">{name}{_hours_line(d)}</td>'
            f'<td align="right" style="{td}">{_spark_for(d)}</td>'
            f'<td align="right" style="{td}{TYPE["data"]}color:{P["muted"]};'
            f'white-space:nowrap;">{level}</td>'
            # The minus SIGN, as everywhere else in the issue: Python's "+"
            # format flag emits an ASCII hyphen, so this strip was the one table
            # drawing negatives with a different, shorter glyph than the
            # thirty-odd other tables around it.
            f'<td align="right" style="{td}{TYPE["data"]}'
            f'color:{col};white-space:nowrap;">{_signed(d["pct"], 2)}%</td>'
            f'</tr>')

    def _region_head(cat: str) -> str:
        return (f'<tr><td colspan="4" style="padding:6px 5px 4px;'
                f'border-bottom:1px solid {P["row_rule"]};{TYPE["label"]}'
                f'color:{_rc(cat)};">{cat}</td></tr>')

    def _table(entries: list) -> str:
        """One column of the strip: region heads interleaved with their rows."""
        head = (f'<tr>' + "".join(
            f'<td align="{al}" style="padding:5px 5px;background:{P["card_alt"]};'
            f'border-bottom:1px solid {P["border"]};{TYPE["label"]}'
            f'color:{P["muted"]};">{lbl}</td>'
            for lbl, al in (("Index", "left"), ("Chart", "right"),
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
    # Section subtitle: when this live snapshot was captured, in the reader's
    # timezone (Europe/Rome) with a DST-aware label (CEST/CET) -- the anchor
    # that explains why a level here can differ from a live quote viewed later
    # (this is a frozen email, not a live page). From the run-owned
    # captured_at, NOT datetime.now(): the scheduled build runs on a UTC CI
    # clock, so an explicit Europe/Rome conversion is required rather than the
    # system-local .astimezone() that runtime.now_stamp() would apply.
    sub = ""
    try:
        from zoneinfo import ZoneInfo
        from tarzan import runtime as _runtime
        _cap = _runtime.context().captured_at.astimezone(ZoneInfo("Europe/Rome"))
        sub = f"As of {_cap:%H:%M %Z}"
    except Exception:  # noqa: BLE001
        sub = ""
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
    tot = {b: _window_money_pnl(m.pnl_series, m.actual_value_series, b)
           for b in ("1d", "5d", "1m")}
    tot_since = (m.pnl_eur, m.pnl_pct)
    unr = {b: _window_money_pnl(m.unrealized_series, m.actual_value_series, b)
           for b in ("1d", "5d", "1m")}
    unr_since = (m.unrealized_pnl_eur, m.unrealized_pnl_pct)
    nav_norm = _norm_series(m.portfolio_history)
    tw = {b: _window_twror(nav_norm, b) for b in ("1d", "5d", "1m")}
    tw_since = m.twror_pct

    # No live-quote override here. The 1D row reads the same series as every
    # other row, and that series' current point IS the live valuation
    # (metrics._current_prices stamps it), so re-deriving the session move from
    # performance_full would be a second answer to a question already answered
    # — which is exactly how this row came to print -€2.8k beside a +€11
    # Session tile on 19 Aug 2026.

    bt = f"1px solid {P['border']}"

    def _money_cell(pair) -> str:
        eur, pct = pair
        c = _sgn(eur)
        eur_s = _eur_smart(eur, signed=True) if eur is not None else "—"
        pct_s = _pct(pct, decimals=2, signed=True) if pct is not None else ""
        return (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};">'
                f'<div style="{TYPE["figure"]}color:{c};'
                f'font-variant-numeric:tabular-nums;">{eur_s}</div>'
                + (f'<div style="{TYPE["data"]}color:{c};'
                   f'font-variant-numeric:tabular-nums;">{pct_s}</div>'
                   if pct_s else "")
                + '</td>')

    # ── The matrix, windows as ROWS ──────────────────────────────────────
    # Transposed from measures-as-rows. A reader asks "how did the last week
    # go", which is one row here instead of one cell picked out of three rows,
    # and it puts the four measures in a fixed column order that matches the
    # returns grids below.

    def _label(txt) -> str:
        return (f'<td style="padding:7px 0;border-top:{bt};{TYPE["figure"]}'
                f'color:{P["ink"]};white-space:nowrap;">{txt}</td>')

    def _money_pair(pair, *, eur_only=False, pct_only=False) -> str:
        """One measure split across its own two columns, € then %."""
        eur, pct = pair
        c = _sgn(eur if not pct_only else pct)
        if pct_only:
            return (f'<td align="right" style="padding:7px 0 7px 10px;'
                    f'border-top:{bt};{TYPE["figure"]}color:{c};'
                    f'font-variant-numeric:tabular-nums;">'
                    f'{_pct(pct, decimals=2, signed=True) if pct is not None else "\u2014"}</td>')
        return (f'<td align="right" style="padding:7px 0 7px 10px;'
                f'border-top:{bt};{TYPE["figure"]}color:{c};'
                f'font-variant-numeric:tabular-nums;">'
                f'{_eur_smart(eur, signed=True) if eur is not None else "\u2014"}</td>')

    def _pct_cell(v) -> str:
        return (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};'
                f'{TYPE["figure"]}color:{_sgn(v)};'
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
        f'{"" if i == 0 else " 10px"};{TYPE["label"]}'
        f'color:{P["muted"]};">'
        f'{h}</td>' for i, h in enumerate(heads)) + '</tr>'

    windows = [
        # No "\u25cf LIVE" on the row label: the masthead states whether a
        # session is open, and the 1 day row is live whenever it is.
        ("1D", tot["1d"], unr["1d"], tw["1d"]),
        ("5D", tot["5d"], unr["5d"], tw["5d"]),
        ("1M", tot["1m"], unr["1m"], tw["1m"]),
        ("Since inception", tot_since, unr_since, tw_since),
    ]
    body = ""
    for label, total, unreal, twror in windows:
        coincide = _same(total, unreal, twror)
        eq = (f'<td align="right" style="padding:7px 0 7px 10px;border-top:{bt};'
              f'{TYPE["figure"]}color:{P["subtle"]};">=</td>')
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
    UNREAL = _charts.UNREAL

    def _colcap(t: str) -> str:
        """Panel caption, in the concept's form: the label tier in subtle, not
        a title in ink. At ink weight it competed with the section heading
        above it for the same job."""
        return (f'<div style="{TYPE["label"]}color:{P["subtle"]};'
                f'margin-bottom:5px;">{t}</div>')

    def _mini_legend(items: list) -> str:
        """A one-line colour key under a chart: a small swatch and a name per
        line, so the lines carry only their end value and are named once here.

        Each chart draws only the end value on each line, to keep the plot
        wide, so the colours named nothing on their own. It matters most on the
        volatility panel, whose portfolio line is a colour (amber) that appears
        on no other chart, so there was no way to learn the mapping from
        anywhere on the page.
        """
        if not items:
            return ""
        parts = [
            (f'<span style="display:inline-block;width:7px;height:7px;'
             f'border-radius:2px;background:{color};vertical-align:baseline;'
             f'margin-right:4px;"></span>'
             f'<span style="color:{P["muted"]};">{label}</span>')
            for color, label in items
        ]
        return (f'<div style="{TYPE["data"]}margin:7px 0 0;">'
                + "&nbsp;&nbsp;&nbsp;".join(parts) + "</div>")

    # Last-30-day labels come from the exact arrays passed to the chart. The
    # shared-close endpoint is therefore the only number that can describe a
    # line; generic 1M return buckets are intentionally not consulted here.
    endpoints = dict(win.get("endpoints") or {})
    legend_values: dict[str, float] = {}
    legend_labels: dict[str, str] = {}

    def _window_label(key: str, _prefix: str) -> str:
        """The visible end label for a 30-day line: the signed percentage, and
        nothing else.

        It carried the series name too, but at half width the gutter that would
        hold "Total P&L % -0.97%" leaves the plot 148px wide. The three colours
        are named in full on the since-inception chart directly above, so the
        mapping is established once for the section. The audited string is this
        same string -- the gate checks that what is drawn agrees with the
        endpoint, and it still does.
        """
        value = float(endpoints[key])
        label = _pct(value, signed=True)
        legend_values[key] = value
        legend_labels[key] = label
        return label

    # The value is drawn at the end of each line; the line's NAME is in the
    # colour key built alongside here and rendered under the caption. The
    # audited string is the end-value %, unchanged, so the semantic gate still
    # finds it verbatim in the rendered HTML -- it is inside the SVG now.
    s30 = []
    ret_leg = []
    if win["twror"] is not None and endpoints.get("twror") is not None:
        s30.append({"values": win["twror"], "color": GREEN,
                    "end_label": _window_label("twror", "TWROR")})
        ret_leg.append((GREEN, "TWROR"))
    if win["pnl_pct"] is not None and endpoints.get("pnl_pct") is not None:
        s30.append({"values": win["pnl_pct"], "color": PNL,
                    "end_label": _window_label("pnl_pct", "Total P&L %")})
        ret_leg.append((PNL, "Total P&amp;L"))
    if win.get("unreal_pct") is not None and endpoints.get("unreal_pct") is not None:
        s30.append({"values": win["unreal_pct"], "color": UNREAL,
                    "end_label": _window_label("unreal_pct", "Unrealized P&L %")})
        ret_leg.append((UNREAL, "Unreal. P&amp;L"))
    if win["acwi"] is not None and endpoints.get("acwi") is not None:
        s30.append({"values": win["acwi"], "color": BENCH,
                    "end_label": _window_label("acwi", "MSCI ACWI")})
        ret_leg.append((BENCH, ctx.benchmark_geo))

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
    ssi = []
    si_leg = []
    full = _perf_full_series(m, ctx.benchmark_geo)
    si_dates = full["dates"] if full else dates
    if full is not None:
        # Bare value at each line end; the name is in the colour key below the
        # chart. The name sat beside the line too ("iShares MSCI ACWI +14.16%"),
        # which repeated the key for no gain and forced a 132px right gutter
        # that ate a fifth of the plot.
        if full["twror"] is not None:
            ssi.append({"values": full["twror"], "color": GREEN,
                        "end_label": _pct(m.twror_pct, signed=True)})
            si_leg.append((GREEN, "TWROR"))
        if full["pnl_pct"] is not None:
            ssi.append({"values": full["pnl_pct"], "color": PNL,
                        "end_label": _pct(m.pnl_pct, signed=True)})
            si_leg.append((PNL, "Total P&amp;L"))
        if full.get("unreal_pct") is not None and not is_missing(full["unreal_pct"][-1]):
            # Labelled from the line's own end point, not from a lifetime
            # metrics field: there is no ``m.unrealized_pct`` counterpart to
            # ``m.pnl_pct``, and inventing one from the final ratio would risk
            # printing a number the drawn line does not reach. A missing end
            # point drops the line rather than labelling it "—", which would
            # name a series the reader cannot read a value for.
            ssi.append({"values": full["unreal_pct"], "color": UNREAL,
                        "end_label": _pct(full["unreal_pct"][-1], signed=True)})
            si_leg.append((UNREAL, "Unreal. P&amp;L"))
        if full["acwi"] is not None:
            ssi.append({"values": full["acwi"], "color": BENCH,
                        "end_label": _pct(full["acwi"][-1], signed=True)})
            si_leg.append((BENCH, ctx.benchmark_geo))

    # ── Volatility row (You vs the market, second row): annualized volatility
    #    on a rolling 21-day window, plotted over the last 30 days -- the same
    #    window as the return panel beside it, so the two half panels sit on one
    #    timeline instead of pairing a 30-day return with a whole-history
    #    volatility. Grey line = the benchmark, so the reader sees whether they
    #    run calmer or bumpier.
    VOL = "#B45309"  # amber-brown, distinct from the return lines
    vol_30 = _perf_vol_series(m, ctx.benchmark_geo, n_days=30)

    # Panel sizes. The card's content box is 580px wide; with the 8px gutter
    # between the two half cells each of them gets 282px. These are passed
    # explicitly because the SVG carries its own width — putting a chart in a
    # wider table cell does not make the chart wider.
    W_WIDE, H_WIDE = 580, 166
    W_HALF, H_HALF = 282, 138
    # Room for the end labels: bare signed percentages on every chart now that
    # the names live in the colour keys, so ~54px is enough on all three (the
    # wide chart used to reserve 132px for "MSCI ACWI +14.16%").
    G_WIDE, G_HALF = 54, 52
    def _vol_panel(vs, dates_, *, month_ticks, min_day_ticks,
                   w=W_HALF, h=H_HALF) -> str:
        series = []
        if vs and vs.get("port"):
            series.append({"values": vs["port"], "color": VOL,
                           "end_label": _pct(vs["port"][-1], signed=False)})
        if vs and vs.get("acwi"):
            series.append({"values": vs["acwi"], "color": BENCH,
                           "end_label": _pct(vs["acwi"][-1], signed=False)})
        if not series:
            return ""
        return _charts.chart_pct_compact(series, dates_, include_zero=False,
                                         w=w, h=h, month_ticks=month_ticks,
                                         min_day_ticks=min_day_ticks,
                                         end_gutter=G_HALF)

    parts = []
    if s30 or ssi:
        # Every chart reads the same way: a caption naming measure and window on
        # top, the plot, then a colour key naming the lines at the bottom. The
        # wide chart is the same measure as the left half panel over a longer
        # window, so it is "Return · since inception" to the half panel's
        # "Return · last 30 days".
        left_ret = (_colcap("Return \u00b7 since inception")
                    + _charts.chart_pct_compact(
                        ssi, si_dates, include_zero=False, w=W_WIDE, h=H_WIDE,
                        month_ticks=True, end_gutter=G_WIDE)
                    + _mini_legend(si_leg)) if ssi else ""
        right_ret = (_colcap("Return \u00b7 last 30 days")
                     # Five date ticks, not twelve: at half width twelve
                     # rotated labels overlapped into a grey band, which is
                     # worse than no axis at all.
                     + _charts.chart_pct_compact(s30, dates, include_zero=True,
                                                 w=W_HALF, h=H_HALF,
                                                 min_day_ticks=5,
                                                 end_gutter=G_HALF)
                     + _mini_legend(ret_leg)) if s30 else ""
        # The volatility panel shares the return panel's 30-day window and its
        # five day-ticks, so the two half panels read on one x-axis. The caption
        # names the measure (annualized volatility, same units as the risk tile)
        # and the window, rather than the "rolling 1M" of the internal 21-day
        # estimator, which named the window the reader could not see.
        vol_panel = _vol_panel(vol_30, vol_30["dates"] if vol_30 else dates,
                               month_ticks=False, min_day_ticks=5)
        if vol_panel:
            vol_leg = []
            if vol_30 and vol_30.get("port"):
                vol_leg.append((VOL, "Portfolio"))
            if vol_30 and vol_30.get("acwi"):
                vol_leg.append((BENCH, ctx.benchmark_geo))
            vol_panel = (_colcap("Volatility \u00b7 last 30 days")
                         + vol_panel + _mini_legend(vol_leg))

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
        # No "why you're diverging" block. The concept does not carry one, and
        # it restated the section: the gap is in the heading's subtitle, the
        # three lines are on the chart with their end values, and the beta is a
        # column in RISK. What it added beyond that was an opinion about whether
        # to close the gap, which is advice this issue does not give.
        divergence_html = ""
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
    # No subtitle. The lead vs the benchmark is the distance between the TWROR
    # and benchmark lines on the since-inception chart directly below, both now
    # named in its colour key and labelled with their value, so a sentence
    # restating it in points was a third copy of the same fact.
    return {"available": True,
            "matrix_html": matrix_card,
            "vs_market_html": "".join(parts),
            "vs_market_sub": None}

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


def _bar_session_span(quote, raw_ticker: str):
    """Trading-session window of the venue whose bars these are, or ``None``.

    WHICH listing produced the series is the question, not which ticker the
    portfolio names: the quote resolver falls back to sibling venues (an ISIN
    Yahoo only quotes in Munich), and it is that venue's hours the chart's
    x-axis must span.
    """
    src = raw_ticker
    if isinstance(quote, dict):
        src = (quote.get("intraday_source_ticker")
               or quote.get("source_ticker") or raw_ticker)
    try:
        from tarzan.data.market_quotes import session_span
        return session_span(str(src or ""))
    except Exception:  # noqa: BLE001 — chart falls back to the elapsed-time axis
        return None


def _perf_spark_cell(day_val, raw_ticker: str, intraday_map: dict, *,
                     bg: Optional[str] = None,
                     live: bool = False) -> tuple:
    """Render the 1D cell: a sign-colored % pill (the change vs the previous
    close) above a Markets-style intraday sparkline (green above the previous
    close, red below).

    The sparkline is drawn ONLY when there is a real intraday series (>=2
    points). When the instrument has not traded intraday (illiquid, or the
    market is closed and the vendor exposes no session), there is no chart —
    just the % pill vs the previous close — because a synthetic line would be
    misleading. Returns ``(cell_html, inner_html)``; ``inner_html`` is the
    pill + sparkline without the surrounding ``<td>`` so
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
    pill = (f'<span style="{TYPE["data"]}color:{pill_col};'
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
        # Time-axis intraday, on the real session window of the venue that
        # PRODUCED these bars -- a sibling listing keeps its own hours (IS39 is
        # only quoted in Munich, 08:00-22:00, not on Milan's clock) -- so the
        # drawn extent is the traded extent: three prints from an illiquid ETF's
        # first hour cover a sliver, a completed session fills the width, and
        # neither needs a flag to say which. ``live`` only still decides for
        # venues with no modelled session.
        spark = _intraday_spark(intra, baseline, in_progress=live,
                                span=_bar_session_span(quote, raw_ticker))
    else:
        # No intraday trades → a dashed placeholder line (prev-close
        # reference), uncaptioned, so the cell keeps the same height as every
        # intraday row and the pill stays aligned with them.
        spark = _flat_dashed_spark()

    bgc = f"background:{bg};" if bg else ""
    # No per-row basis tag. Every row of the table shares one basis -- either the
    # session is open for all of them or it is closed for all of them -- so a
    # "\u25cf LIVE" or "PREV. DAY" badge on each of seventeen rows repeated one
    # fact seventeen times, in the width the figure needed. The column HEADER
    # says it once: "Intraday" while a session is open, "1D" once it has closed.
    inner = f'<div>{pill}</div><div style="margin-top:3px;">{spark}</div>'
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

def _returns_table_html(period_cols, portfolio: Optional[dict], groups: list,
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
    every_row = (([portfolio] if portfolio else [])
                 + [inst for _c, _col, role_list in groups
                    for _r, insts in role_list for inst in insts])
    scales = {}
    for p in period_cols:
        scales[p] = _heat.column_scale(
            row.get("returns", {}).get(p, {}).get("raw") for row in every_row)

    # The 1D column is a window like any other and is tinted like any other. It
    # was the one column left uncoloured, so it was the only cell in the row a
    # reader had to evaluate by reading the number. Its scale is damped, because
    # the cell also carries the session sparkline and a saturated background
    # competes with the line drawn on it.
    dneg, dpos = _heat.column_scale(row.get("day_raw") for row in every_row)
    # One width for every window column, so the tinted blocks form an even grid
    # rather than columns sized by whatever their widest figure happened to be.
    # Only the session column is wider, because it carries a chart. These are
    # HARD widths under fixed_layout (see the render call): name 148 + day 68 +
    # 7×52 = 580px, the exact content width of the 620px card less its 20px
    # gutters, so nothing is scaled and a five-figure return (+199.9%) still
    # fits its 52px column without spilling into the next.
    W_PERIOD, W_DAY = 52, 68

    def _cells(returns_dict, spark_inner, *, weight, day_raw=None):
        # 1D sparkline pulled hard left (left-aligned, minimal left padding) so
        # the gap after the name closes and the period columns get that width.
        cells = [uni_cell(spark_inner, align="right", width=W_DAY,
                          valign="middle", pad="5px 4px",
                          bg=_heat.heat_bg(day_raw, neg=dneg, pos=dpos,
                                           damp=_heat.DAY_DAMP))]
        for p in period_cols:
            r = returns_dict.get(p, {"value": "\u2014", "color": P["muted"]})
            neg, pos = scales[p]
            raw = r.get("raw")
            # Background AND figure colour from the one ramp. The figure used to
            # be sign-coloured, so the grid stated the sign twice and its text
            # colour changed from row to row for a reason unrelated to the tint.
            bg, fig = _heat.heat(raw, neg=neg, pos=pos)
            cells.append(uni_cell(r["value"], color=fig, weight=weight,
                                  bg=bg, width=W_PERIOD))
        return cells

    # 1D column carries no_sep (True) \u2014 it reads with the name block, not the
    # ruled period grid.
    columns = ([(day_label, "right", W_DAY, True)]
               + [(p.upper(), "right", W_PERIOD) for p in period_cols])
    portfolio_row = {
        "name_html": (f'<span style="color:{P["accent"]};font-weight:700;'
                      f'font-size:{TYPE_PX["data"]}px;">'
                      f'\u2605 {portfolio["name"]}</span>'),
        "cells": _cells(portfolio["returns"], portfolio.get("spark_inner", ""),
                        weight=700, day_raw=portfolio.get("day_raw")),
    } if portfolio else None
    uni_groups = [
        (cls, col, [(role, [{"name_html": inst["name_html"],
                             "cells": _cells(inst["returns"],
                                             inst.get("spark_inner", ""),
                                             weight=600,
                                             day_raw=inst.get("day_raw"))}
                            for inst in insts])
                    for role, insts in role_list])
        for cls, col, role_list in groups]
    # Compact numeric cells + a capped name column (so it can't hoard width and
    # leave a gap before the sparkline) + faint vertical separators so a reader
    # can tell 5D from 1M from 1Y at a glance. Role lives in the group header
    # and the ticker trails the name, so names wrap cleanly.
    # ``zebra=False``: the alternating row stripe fights the heat. Under a tinted
    # matrix an uncoloured cell has to be ONE surface, or the eye reads the
    # stripe as a signal too. ``dense``: 9.5px on 5px/4px padding, so the tint
    # fills its column instead of floating in it.
    # ``fixed_layout``: the Returns and Watchlist tables share a container
    # width, so pinning every column to the width declared here makes the two
    # grids align — same sparkline start, same 5D/1M/… x positions — instead of
    # each sizing its columns from its own content (short tickers/small % vs
    # long names/large %), which drifted them apart.
    return render_unified_table("Instrument", columns, uni_groups,
                                portfolio_row=portfolio_row, compact=True,
                                first_col_width=148, separators=False,
                                zebra=False, dense=True, radius=4,
                                fixed_layout=True)

def _intraday_column(flags) -> bool:
    """Whether the session column's header may say "Intraday" rather than "1D".

    The header states ONE basis for the whole column, so it may only claim
    intraday when every row in that column actually is. ``any`` was the rule,
    and at the 09:09 send a single instrument with bars — a German venue
    printing before Milan's open — put "INTRADAY" over thirty-nine
    close-to-close figures, which is precisely the "these look like yesterday's
    returns" report. Empty is not intraday."""
    flags = [bool(f) for f in flags]
    return bool(flags) and all(flags)


def _perf_name_html(name: str, ticker: str, tags: list, *,
                    name_chars: Optional[int] = None) -> str:
    """Instrument label used in the returns tables. Delegates to the shared
    :func:`uni_name`, so the ticker leads the name here exactly as it does in
    the book, the risk table and the optimizer, and the role stays in the group
    header rather than stacking a caption under every name."""
    from tarzan.export.newsletter._constants import uni_name
    if name_chars == 0:
        # Ticker only (unused by the returns tables now, kept for callers that
        # want the bare symbol).
        return uni_name("", ticker or "", tags=tuple(tags or ()),
                        line_height=1.05)
    if name_chars:
        name = name[:name_chars]
    # Tight line-height: the curated name (already abbreviated in _format so
    # it fits ~2 lines in the 148px column) wraps with an almost-zero gap
    # between line one and two, so a two-line name barely adds row height.
    return uni_name(name, ticker or "", tags=tuple(tags or ()),
                    line_height=1.05)

def _build_returns_snapshot(ctx: _NewsletterContext) -> dict:
    """Build the per-holding returns snapshot table.

    Mirrors the Excel ``Performance`` tab and uses the exact same eight
    time-return columns as the "Returns vs benchmarks" table below —
    1D / 5D / 1M / 3M / YTD / 1Y / 3Y / 5Y — so the two newsletter
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
    period_keys = ["5d", "1m", "3m", "ytd", "1y", "3y", "5y"]

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
    if _pf_series is not None and len(_pf_series) >= 2:
        _, port_inner = _perf_spark_cell(
            port_full.get("1d"), _PF_INTRA_KEY, {_PF_INTRA_KEY: _pf_series},
            live=bool(port_full.get("1d_live")))
    else:
        _, port_inner = _perf_spark_cell(
            port_full.get("1d"), "", {}, live=bool(port_full.get("1d_live")))
    portfolio = {"name": "Portfolio", "spark_inner": port_inner,
                 "day_raw": port_full.get("1d"),
                 "returns": _returns_dict(port_full, is_portfolio=True)}

    # Build one row per holding, then group via the SHARED engine (class → role,
    # ordered) so this table splits/colours instruments identically to Holdings
    # and the Optimizer.
    row_items = []
    # Per-row session basis, for the column header (see _intraday_column). Built
    # from the rows actually rendered, not from hp — hp carries the watchlist
    # benchmarks too, and they are a different table.
    row_live: list = []
    for _, h in df.iterrows():
        ticker = str(h.get("ticker", "") or "")
        isin = str(h.get("isin", "") or "")
        raw_name = str(h.get("name", "") or ticker)
        display_tk = _display_ticker(ticker) or ""
        row_live.append(bool(_live1d.get(ticker, False)))
        _, inner = _perf_spark_cell(
            _raw1d.get(ticker), ticker, _snap_intraday,
            live=bool(_live1d.get(ticker, False)))
        row_items.append({
            "_ac": str(h.get("asset_class", "") or "") or "Other",
            "_isin": isin, "_ticker": ticker,
            # Ticker + the curated (abbreviated) name, exactly like the
            # Watchlist rows, so the two grids read as one view. No hard cut:
            # the abbreviation in _format keeps it to ~2 tight lines. (The Book
            # below still carries the full untruncated name for every holding.)
            "name_html": _perf_name_html(
                display_instrument_name(isin, ticker, raw_name),
                display_tk, []),
            "spark_inner": inner,
            "day_raw": _raw1d.get(ticker),
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
                m, live=_intraday_column(
                    [bool(port_full.get("1d_live"))] + row_live))),
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
    if hp.empty or "5d" not in hp.columns:
        return {"available": False}

    sorted_hp = hp.sort_values("5d", ascending=False, na_position="last")
    best = sorted_hp.iloc[0]
    worst = sorted_hp.iloc[-1]

    df = m.holdings_df

    def _enrich(row):
        ticker = row.get("ticker", "")
        match = df[df["ticker"] == ticker] if not df.empty else pd.DataFrame()
        klass = match["asset_class"].iloc[0] if not match.empty else "Equities"
        value = float(match["current_value"].iloc[0]) if not match.empty else 0.0
        pct = float(row.get("5d") or 0.0)
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
    periods = ("1d", "5d", "1m", "3m", "ytd", "1y", "3y", "5y")

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

    # A taxonomy row can be BOTH held and flagged is_benchmark=true (a fund we
    # own that is also the reference for its role). Such an instrument already
    # has a row in the returns snapshot above, so listing it again here
    # duplicated it — and invited comparing the portfolio against something it
    # owns. The held row wins; the watchlist copy is dropped.
    #
    # Matched on the curated taxonomy name, which is the same identity the α/β
    # and GEO tags below match on: a benchmark row's ``name`` comes straight
    # from the taxonomy, and ``name_for`` resolves a holding to that row by
    # ISIN then bare ticker, so both sides meet on one key.
    held_bench_names = set()
    if m.holdings_df is not None and not m.holdings_df.empty:
        for _, h in m.holdings_df.iterrows():
            curated = cfg.name_for(h.get("isin"), h.get("ticker"))
            if curated:
                held_bench_names.add(curated.strip().lower())

    benchmark_rows = []
    if not hp.empty and "type" in hp.columns:
        ab_name = (ctx.benchmark_alpha_beta or "").strip().lower()
        geo_name = (ctx.benchmark_geo or "").strip().lower()
        bench_df = hp[hp["type"].astype(str).str.contains("enchmark", case=False, na=False)]
        for _, r in bench_df.iterrows():
            name = str(r.get("name") or r.get("ticker", ""))
            name_norm = name.strip().lower()
            if name_norm in held_bench_names:
                continue
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
    period_cols = ("5d", "1m", "3m", "ytd", "1y", "3y", "5y")
    intraday_map = _shared_performance_intraday(ctx)

    # Group benchmark rows by asset class → role, in the configured order,
    # then hand off to the shared table renderer.
    raw_groups = group_by_class_role(
        benchmark_rows, asset_class=lambda r: r.get("asset_class") or "Other",
        role=lambda r: r.get("role"), ticker=lambda r: r.get("ticker"))
    groups = []
    for ac, col, role_list in raw_groups:
        rendered_roles = []
        for role, rows in role_list:
            insts = []
            for r in rows:
                _, inner = _perf_spark_cell(
                    r.get("d1"), r.get("raw_ticker"), intraday_map,
                    live=bool(r.get("live")))
                insts.append({
                    # A short name here, not just the ticker: these instruments
                    # are tracked and NOT held, so no other section in the issue
                    # says what they are. No hard cut — the curated name is
                    # already abbreviated to ~2 tight lines in _format.
                    "name_html": _perf_name_html(r["name"], r.get("ticker"),
                                                 r.get("tags")),
                    "spark_inner": inner,
                    "day_raw": r.get("d1"),
                    "returns": r["returns"],
                })
            rendered_roles.append((role, insts))
        groups.append((ac, col, rendered_roles))

    # No portfolio row in the watchlist. The table lists instruments that are
    # NOT held; the portfolio's own returns are the whole of RETURNS one section
    # up, and repeating them here invited a comparison between the portfolio and
    # a list of things it does not own.
    table_html = _returns_table_html(
        period_cols, None, groups,
        day_label=day_column_label(
            m, live=_intraday_column(r.get("live") for r in benchmark_rows)))

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
    # which ``metric_ratings`` entry in constants.yaml describes the metric; its
    # thresholds and ``invert`` flag are what the tile's gauge is drawn from.
    # That direction cannot be inferred from the sign -- a positive volatility
    # is not good news, and a -7% drawdown beats a -21% one -- and it is already
    # declared in configuration, so reading it here keeps one source of truth
    # instead of a second copy that can drift from the legend beside it.
    # Beta carries None: the configured bands rate it as market exposure, which
    # is a property to know rather than a score to win, so it draws no gauge.
    # Full names, not abbreviations. "Vol", "VaR" and "CVaR" were squeezed for
    # an eleven-column table that no longer exists; a tile has room to say what
    # the metric is, and the confidence level on the two tail measures is part of
    # their definition rather than a footnote.
    metric_cols = [
        ("CAGR", "cagr", True, "", "cagr"),
        ("Volatility", "volatility", True, "", "volatility"),
        ("Sharpe", "sharpe", False, "", "sharpe"),
        ("Sortino", "sortino", False, "", "sortino"),
        ("Max DD", "max_drawdown", True, "", "max_drawdown"),
        ("Ulcer", "ulcer_index", True, "", "ulcer_index"),
        ("VaR 95%", "var_95", True, "", "var_pct"),
        ("CVaR 95%", "cvar_95", True, "", "cvar_pct"),
        # The asterisk points at the footnote naming the index these two are
        # measured against, which is their definition, not a comparison.
        ("\u03b1", "alpha", True, "*", "alpha"),
        ("\u03b2", "beta", False, "*", None),
    ]

    # ``invert: true`` in constants.yaml means "a smaller magnitude is better",
    # and it is written against the metric's ABSOLUTE value: max_drawdown is
    # banded at [-15, -30] and VaR at [0.8, 1.5] while both carry the flag --
    # which is why the gauge below is fed abs(value) against abs(threshold).
    from tarzan import config as _rating_cfg
    _ratings = _rating_cfg.metric_ratings() or {}

    port = hr.get("portfolio")
    if not (port or hr.get("instruments")):
        return {"available": False, "rows": [], "columns": []}

    # ── The portfolio's own ten metrics, as tiles ────────────────────────
    # This section used to be a 36-row table: the portfolio plus every reference
    # instrument, over eleven columns. The question it exists to answer is "what
    # shape was the ride" for THIS portfolio, and the answer was one row out of
    # thirty-six -- the other thirty-five set up a comparison nobody asked for
    # and cost a quarter of the issue's height.
    #
    # Each tile carries the figure and, where the configuration rates the
    # metric, a gauge placing it on its own weak/fair/strong scale. Those
    # thresholds are metric_ratings in constants.yaml, with its citations, so
    # the gauge draws a rating the project declares rather than one invented
    # here.
    port_metrics = (port or {}).get("metrics") or {}
    tiles = []
    for label, key, is_pct, note, rating_key in metric_cols:
        value = port_metrics.get(key)
        if is_missing(value):
            continue
        band = (_ratings.get(rating_key) or {}) if rating_key else {}
        thresholds = band.get("thresholds") or []
        gauge = ""
        if len(thresholds) >= 2:
            gauge = _charts.band_gauge(
                abs(float(value)), good=abs(float(thresholds[0])),
                warn=abs(float(thresholds[1])),
                invert=bool(band.get("invert", False)))
        tiles.append({
            # The label is uppercased by CSS, which folds a Greek alpha onto a
            # capital that is drawn like a Latin A. greek_safe scopes the
            # exception to the characters that break.
            "label": greek_safe(f"{label}{note}"),
            "value": _fmt_pct(value) if is_pct else _fmt_num(value),
            "gauge": gauge,
        })

    description = ""

    return {
        "available": True,
        "title": "Historical risk profile",
        # The section heading carries this now, so it names the one thing the
        # reader needs before reading a column: the window each series covers.
        "subtitle": (
            f'Portfolio over {(port or {}).get("span_label") or "its"} of '
            f'history: a backtest at today\u2019s weights held constant, over '
            f'the longest window where every holding with a year or more of '
            f'history overlaps.'
        ),
        "tiles": tuple(tiles),
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
    """Build the Risk Profile legend rows. Sources thresholds and units from
    ``constants.yaml::metric_ratings`` so the bands quoted here are the ones
    the ratings are actually computed from.

    Each entry: {label, strong, fair, weak, description}. One line per metric:
    the description says what the number IS and stops there. Calibration --
    whether a given value is good -- is the gauge drawn on the tile above and
    the three band chips beside the name, so the prose no longer repeats it
    ("equity indexes ~15-20%", ">1 is good, >2 excellent"). Ten boxed cards
    with three-line glosses were taller than the ten tiles they explained.

    The α and β rows describe the metrics in general; the benchmark they are
    measured against is named in the tile note above, so the legend stays
    reusable across configurations.
    """
    from tarzan import config as cfg
    ratings = cfg.metric_ratings() or {}

    # (label, ratings_key, description). Order matches the tiles above. One
    # clause each: what the number is, not how to feel about it.
    legend_specs = [
        ("CAGR", "cagr", "Yearly return, compounded, start to end value."),
        ("Volatility", "volatility",
         "Annualized standard deviation of daily returns."),
        ("Sharpe", "sharpe", "(CAGR \u2212 risk-free rate) / volatility."),
        ("Sortino", "sortino", "Sharpe counting downside volatility only."),
        ("Max Drawdown", "max_drawdown", "Worst peak-to-trough loss."),
        ("Ulcer Index", "ulcer_index",
         "RMS of drawdowns: depth and time underwater together."),
        ("VaR 95%", "var_pct", "Daily loss exceeded on 5% of days."),
        ("CVaR 95%", "cvar_pct", "Average loss on the worst 5% of days."),
        ("\u03b1", "alpha",
         "Return above the benchmark once risk is accounted for (CAPM)."),
        ("\u03b2", "beta", "Portfolio move per 1% benchmark move."),
    ]

    def _fmt(value: Optional[float], unit: str) -> str:
        if value is None:
            return "\u2014"
        v = float(value)
        # Drop the ".0" on integer thresholds so the bands read tight
        # ("<3%" not "< 3.0%") — they sit inline next to the metric name.
        # The minus SIGN, like every other negative figure in the issue: a band
        # chip reading "-15%" beside a tile reading "\u221215%" is two glyphs for
        # one idea.
        a = abs(v)
        num = f"{int(round(a))}" if abs(a - round(a)) < 1e-9 else f"{a:.1f}"
        sign = "\u2212" if v < 0 else ""
        return f"{sign}{num}{unit}"

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
            # Escaped here, not in the template: the digest template is
            # ``.html.j2``, whose extension select_autoescape() does not
            # recognise, so autoescape is off and "<10%" reached the document as
            # a bare angle bracket in a text node.
            "strong": _esc(strong),
            "fair": _esc(fair),
            "weak": _esc(weak),
            "description": description,
        })
    return legend_rows

