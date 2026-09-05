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
    _geo_benchmark_series,
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
            f'white-space:nowrap;">{_esc(name)}{_hours_line(d)}</td>'
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
    # — which is exactly how this row came to print a multi-thousand-euro loss
    # beside a +€11 Session tile on 19 Aug 2026.

    bt = f"1px solid {P['border']}"

    # ── The matrix, windows as ROWS ──────────────────────────────────────
    # Transposed from measures-as-rows. A reader asks "how did the last week
    # go", which is one row here instead of one cell picked out of three rows,
    # and it puts the four measures in a fixed column order that matches the
    # returns grids below.

    def _label(txt) -> str:
        return (f'<td style="padding:7px 0;border-top:{bt};{TYPE["figure"]}'
                f'color:{P["ink"]};white-space:nowrap;">{txt}</td>')

    def _money_pair(pair, *, pct_only=False) -> str:
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

    # Unrealized carries its own money column beside its percent, exactly like
    # Total P&L, and every cell prints its own figure. The three measures used to
    # collapse to a subtle "=" whenever they agreed to within half a basis point,
    # which on a window with no sale inside it is most windows: the P&L change,
    # the unrealized change and the time-weighted return are then the same
    # number. It read as "we could not compute this", and it hid the one case
    # that matters, a window containing a sale, where the unrealized change is
    # smaller than the P&L change by exactly the gain that got realized.
    heads = ("Window", "P&amp;L \u20ac", "P&amp;L %", "Unr. \u20ac", "Unr. %", "TWROR")
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
        body += ('<tr>' + _label(label)
                 + _money_pair(total)
                 + _money_pair(total, pct_only=True)
                 + _money_pair(unreal)
                 + _money_pair(unreal, pct_only=True)
                 + _pct_cell(twror)
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
    PORT, PNL, BENCH = _charts.PORT, _charts.PNL, _charts.BENCH
    UNREAL, TARGET = _charts.UNREAL, _charts.TARGET
    # Every legend label on this section is pre-escaped markup ("Total P&amp;L"),
    # and the benchmark's name is configured text, so it escapes once here rather
    # than at each of the three legends and the heading that reuse it.
    bench_label = _esc(str(ctx.benchmark_geo or ""))

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

    # ── The six standard windows, the vocabulary a broker's app prints ────────
    # One per panel, in this order. 1D is deliberately NOT a chart: from daily
    # closes a session is two points, i.e. a straight segment that looks like
    # data and carries none, so its cell renders the figures instead (the form
    # a single headline number asks for).
    WINDOWS = [("1d", "1D"), ("5d", "5D"), ("1m", "1M"),
               ("3m", "3M"), ("ytd", "YTD"), ("1y", "1Y")]

    #: Every line a window panel may draw, in draw order (references last, so the
    #: portfolio is never hidden under one). Total and Unrealized P&L are NOT
    #: here: five lines in a 182px cell is not a chart, and both keep their own
    #: € and % columns in the matrix above, where the reader can compare them.
    PANEL_LINES = (("twror", PORT, "TWROR", 2.2),
                   ("target", TARGET, "Target", 1.6),
                   ("acwi", BENCH, bench_label, 1.6))

    # Per-window audit, keyed by bucket. The gate recomputes each window from the
    # raw metrics and compares, so a panel whose label drifts from the line it
    # labels cannot ship — the same contract the single 30-day panel had, now
    # carried by every window.
    win_audit: dict[str, dict] = {}

    def _panel(win: Optional[dict], bucket: str) -> tuple[list, list]:
        """``(series, legend)`` for one window, recording the audit as it goes.

        The end label is the endpoint of the exact array handed to the chart, so what
        is drawn and what is written are one number by construction — EXCEPT where the
        window supplies its own ``labels``, which only 1D does. There the bars say
        when the day moved and the tape says by how much, because a blended session
        path is short by any sleeve the quote catalog did not return while that
        sleeve's own 1D is known and printed in its own row. Every other 1D cell in
        the newsletter has always taken its figure from the tape; this brings the grid
        into line rather than out of it.
        """
        if not win:
            return [], []
        endpoints = dict(win.get("endpoints") or {})
        authoritative = dict(win.get("labels") or {})
        series, legend = [], []
        values: dict[str, float] = {}
        labels: dict[str, str] = {}
        for key, colour, name, width in PANEL_LINES:
            line = win.get(key)
            if line is None or endpoints.get(key) is None:
                continue
            value = float(authoritative.get(key, endpoints[key]))
            label = _pct(value, signed=True)
            values[key] = value
            labels[key] = label
            series.append({"values": line, "color": colour, "width": width,
                           "end_label": label})
            legend.append((colour, name))
        if series:
            win_audit[bucket] = {
                "window_start": str(win.get("window_start") or ""),
                "window_end": str(win.get("window_end") or ""),
                "source_end_dates": {
                    k: str(v or "")
                    for k, v in (win.get("source_end_dates") or {}).items()},
                "endpoints": endpoints,
                "legend_values": values,
                "legend_labels": labels,
                "drawn": [k for k, _c, _n, _w in PANEL_LINES if k in values],
            }
        return series, legend

    # ``win`` (the 1M window, already built above for the matrix's anchor) is the
    # one this reuses rather than rebuilding, so the 1M panel and every 1M figure
    # on the page come from a single call.
    ret_windows: dict[str, Optional[dict]] = {"1m": win}
    for _b, _l in WINDOWS:
        if _b in ret_windows or _b == "1d":
            continue
        ret_windows[_b] = _perf_window(m, 30, ctx.benchmark_geo, bucket=_b)

    s30, ret_leg = _panel(win, "1m")

    # Build the remaining panels. 1D is excluded above and handled as figures.
    panels = {"1m": (s30, ret_leg)}
    for _b, _l in WINDOWS:
        if _b in panels or _b == "1d":
            continue
        panels[_b] = _panel(ret_windows.get(_b), _b)

    if ctx.semantic_audit is not None:
        # Keyed by bucket now. ``performance_30d`` stays as an alias for the 1M
        # window so the gate's existing entry point is unchanged; the per-window
        # map is what makes every OTHER panel verified rather than merely drawn.
        ctx.semantic_audit["performance_windows"] = win_audit
        if "1m" in win_audit:
            ctx.semantic_audit["performance_30d"] = win_audit["1m"]

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
            ssi.append({"values": full["twror"], "color": PORT, "width": 2.2,
                        "end_label": _pct(m.twror_pct, signed=True)})
            si_leg.append((PORT, "TWROR"))
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
        if full.get("target") is not None:
            ssi.append({"values": full["target"], "color": TARGET,
                        "end_label": _pct(full["target"][-1], signed=True)})
            si_leg.append((TARGET, "Target"))
        if full["acwi"] is not None:
            ssi.append({"values": full["acwi"], "color": BENCH,
                        "end_label": _pct(full["acwi"][-1], signed=True)})
            si_leg.append((BENCH, bench_label))

    # ── Volatility row (You vs the market, second row): annualized volatility
    #    on a rolling 21-day window, plotted over the last 30 days -- the same
    #    window as the return panel beside it, so the two half panels sit on one
    #    timeline instead of pairing a 30-day return with a whole-history
    #    volatility. Grey line = the benchmark, so the reader sees whether they
    #    run calmer or bumpier.
    VOL = "#B45309"  # amber-brown, distinct from the return lines
    # One volatility panel per return panel, on the SAME bucket, so the two grids
    # read on one vocabulary of windows instead of pairing a named return window
    # with an unnamed 30-calendar-day volatility.
    vol_windows = {b: _perf_vol_series(m, ctx.benchmark_geo, bucket=b)
                   for b, _l in WINDOWS if b != "1d"}
    # ...and the same measure over the whole life, under the return chart it
    # pairs with. A named window says whether that stretch was calm; it cannot
    # say whether the book has been calmer than its target and its benchmark all
    # along, which is the section's question.
    vol_si = _perf_vol_series(m, ctx.benchmark_geo, n_days=None)

    # Panel sizes. The card's content box is 580px wide. A 3-wide grid with 6px
    # of padding either side of each cell leaves 182px per plot; these are passed
    # explicitly because the SVG carries its own width — putting a chart in a
    # wider table cell does not make the chart wider.
    W_WIDE, H_WIDE = 580, 166
    W_CELL, H_CELL = 182, 116
    # The volatility pair sits two-up: 580px content box less the 8px gutter either
    # side of the divider leaves 282px each.
    W_HALF, H_HALF = 282, 138
    # Room for the end labels: bare signed percentages, three of them stacked at
    # the line ends, so ~46px at cell width and 54px on the wide chart.
    G_WIDE, G_CELL, G_HALF = 54, 46, 52
    def _last_estimate(values):
        """The last FINITE value of a line, or None when it has none.

        The dot and its label sit on the last point the estimator actually
        produced, so labelling ``values[-1]`` printed an em-dash beside a dot
        drawn mid-plot — a line whose stated end value belonged to a different
        place than its visible end.
        """
        return next((v for v in reversed(values or ()) if math.isfinite(v)), None)

    def _vol_panel(vs, dates_, *, month_ticks, min_day_ticks,
                   w=W_CELL, h=H_CELL, gutter=G_CELL) -> str:
        series = []
        if vs and vs.get("port"):
            series.append({"values": vs["port"], "color": VOL, "width": 2.2,
                           "end_label": _pct(_last_estimate(vs["port"]),
                                             signed=False)})
        if vs and vs.get("target"):
            series.append({"values": vs["target"], "color": TARGET,
                           "end_label": _pct(_last_estimate(vs["target"]),
                                             signed=False)})
        if vs and vs.get("acwi"):
            series.append({"values": vs["acwi"], "color": BENCH,
                           "end_label": _pct(_last_estimate(vs["acwi"]),
                                             signed=False)})
        if not series:
            return ""
        return _charts.chart_pct_compact(series, dates_, include_zero=False,
                                         w=w, h=h, month_ticks=month_ticks,
                                         min_day_ticks=min_day_ticks,
                                         end_gutter=gutter)

    def _vol_legend(vs, *, with_span=False) -> list:
        """The colour key, optionally carrying each line's σ over the book's own
        life.

        ``with_span`` states the like-for-like figure beside each name. Every
        series' OWN full history is not comparable — the benchmark holds two
        years where the book holds eight months, so its unclipped σ reads 14.76%
        against the book's 10.51% and looks like a risk gap when a third of it is
        just a longer, rougher period. Clipped to the shared span it is 11.79%,
        and the ranking (book 10.51 < target 11.11 < market 11.79) is then a
        statement about the same months.

        Not the same number as the RISK section's volatility (10.77% live on
        26 Aug 2026): that table measures a current-weight static backtest over a
        longer common window, by design. Two questions, two answers.
        """
        span = (vs or {}).get("span") or {}
        out = []
        for colour, name, key in ((VOL, "Portfolio", "port"),
                                  (TARGET, "Target", "target"),
                                  (BENCH, bench_label, "acwi")):
            if not (vs and vs.get(key)):
                continue
            label = name
            if with_span and span.get(key) is not None:
                label = f"{name} {_pct(span[key], signed=False)}"
            out.append((colour, label))
        return out

    # ── Grid plumbing ─────────────────────────────────────────────────────────
    def _cellcap(label: str, note: str = "") -> str:
        """A grid cell's own heading: the window in ink, its span beside it.

        The window name has to be the loudest thing in the cell -- with six plots
        on one grid, a reader locating "3M" does it from these labels and not from
        the axes. So this is the one caption in the section at ink weight;
        ``_colcap`` stays subtle for the grid titles above.
        """
        return (f'<div style="{TYPE["label"]}color:{P["ink"]};'
                f'font-weight:600;letter-spacing:.04em;margin-bottom:3px;">'
                f'{label}'
                + (f'<span style="color:{P["subtle"]};font-weight:400;'
                   f'letter-spacing:0;"> · {note}</span>' if note else "")
                + '</div>')

    def _empty_cell(label: str, why: str) -> str:
        """A window the book cannot show, said out loud.

        An omitted cell reads as "this window does not exist"; a reader who asked
        for six windows and got four cannot tell which two are missing, or why. A
        book younger than a year has no 1Y -- that is a fact about the book, and
        printing it beats a hole in the grid.
        """
        return (_cellcap(label)
                + f'<div style="{TYPE["data"]}color:{P["subtle"]};'
                  f'text-align:center;padding:{max(0, (H_CELL // 2) - 8)}px 0;">'
                  f'{why}</div>')

    def _grid(cells: list) -> str:
        """Six cells, three per row. Fixed columns, so a cell that cannot be drawn
        keeps its slot and the windows stay in their reading order."""
        out = []
        for i in range(0, len(cells), 3):
            tds = "".join(
                f'<td width="33.3%" valign="top" '
                f'style="padding:0 6px 12px 6px;">{c}</td>'
                for c in cells[i:i + 3])
            out.append(f'<tr>{tds}</tr>')
        return ('<table role="presentation" width="100%" cellpadding="0" '
                'cellspacing="0" border="0" style="table-layout:fixed;">'
                + "".join(out) + '</table>')

    # ── 1D: the intraday session, drawn ───────────────────────────────────────
    # The window itself is built by ``_perf_intraday_window`` at module level, in
    # the same shape ``_perf_window`` returns and with the reasons documented
    # there. It lives outside this function so the semantic gate can recompute it:
    # one authority, called twice, exactly as the other five windows work.
    def _day_cell() -> str:
        from tarzan.engine.stats import compute_period_return

        win1d = _perf_intraday_window(m, ctx.benchmark_geo)
        if win1d is not None:
            ser, leg = _panel(win1d, "1d")
            if ser:
                chart = _charts.chart_pct_compact(
                    ser, win1d["dates"], include_zero=True, w=W_CELL, h=H_CELL,
                    date_fmt="%H:%M", min_day_ticks=3, end_gutter=G_CELL,
                    x_span=win1d.get("session_span"))
                if chart:
                    return _cellcap("1D", "session") + chart

        # No bars at all -- a closed market whose vendor exposes no session, a
        # pinned/point-in-time run, or an offline one. The figures still exist and
        # the matrix already computed them, so the cell states them rather than
        # going blank: a one-session comparison as three numbers.
        rows = [(PORT, "TWROR", tw.get("1d"))]
        traw = getattr(m, "target_history", None)
        if traw is not None and len(traw) >= 2:
            rows.append((TARGET, "Target",
                         compute_period_return(_norm_series(traw).dropna(), "1d")))
        braw = (m.benchmark_histories or {}).get(ctx.benchmark_geo)
        if braw is not None and len(braw) >= 2:
            rows.append((BENCH, bench_label,
                         compute_period_return(braw.dropna(), "1d")))
        if all(v is None for _c, _n, v in rows):
            return _empty_cell("1D", "no completed session")
        body = "".join(
            f'<tr><td style="padding:2px 0;"><span style="display:inline-block;'
            f'width:7px;height:7px;border-radius:2px;background:{c};'
            f'margin-right:5px;"></span>'
            f'<span style="{TYPE["data"]}color:{P["muted"]};">{n}</span></td>'
            f'<td align="right" style="padding:2px 0;{TYPE["figure"]}'
            f'color:{_sgn(v)};font-variant-numeric:tabular-nums;">'
            f'{_pct(v, signed=True) if v is not None else "—"}</td></tr>'
            for c, n, v in rows)
        return (_cellcap("1D", "session · closes")
                + f'<table role="presentation" width="100%" cellpadding="0" '
                  f'cellspacing="0" border="0" style="margin-top:'
                  f'{max(0, (H_CELL // 2) - 26)}px;">{body}</table>')

    parts = []
    if any(panels.get(b, ([], []))[0] for b, _l in WINDOWS) or ssi:
        # Both grids read the same way: a subtle title naming the measure, then six
        # cells each headed by its own window. Every plot is hoisted before its
        # caption -- chart_pct_compact returns "" when no series holds a finite
        # point, and concatenating unconditionally would leave a caption
        # introducing a plot that is not there.
        ret_cells = []
        for _b, _label in WINDOWS:
            if _b == "1d":
                ret_cells.append(_day_cell())
                continue
            _w = ret_windows.get(_b)
            _ser = panels.get(_b, ([], []))[0]
            _chart = (_charts.chart_pct_compact(
                _ser, _w["dates"], include_zero=True, w=W_CELL, h=H_CELL,
                min_day_ticks=0, end_gutter=G_CELL) if (_ser and _w) else "")
            if _chart:
                # DAYS, not sessions. The plotted index comes from
                # ``actual_value_series``, which is calendar-daily with weekends
                # carried flat, so its length counts calendar days: a 1Y window
                # measures 364 of them and about 252 sessions. Calling them
                # sessions printed a figure 40% too high beside every window.
                _span_days = (pd.Timestamp(_w["window_end"])
                              - pd.Timestamp(_w["window_start"])).days
                ret_cells.append(_cellcap(_label, f'{_span_days} days') + _chart)
            else:
                ret_cells.append(_empty_cell(_label, "not enough history"))


        # One colour key per grid, beneath it. With six plots sharing three
        # colours, a key per cell would print the same three swatches six times.
        _ret_leg_all = next((lg for _b, _l in WINDOWS
                             for lg in [panels.get(_b, ([], []))[1]] if lg), [])
        ret_grid = (_colcap("Return · by window") + _grid(ret_cells)
                    + _mini_legend(_ret_leg_all))

        # The whole life, full width and on month ticks, under the grid. It is the
        # only place the lifetime trajectory lives, and the gap it draws is the
        # number the masthead and the state tile quote. None of the six windows
        # substitutes for it: 1Y is a window, inception is the book.
        _si_chart = _charts.chart_pct_compact(
            ssi, si_dates, include_zero=False, w=W_WIDE, h=H_WIDE,
            month_ticks=True, end_gutter=G_WIDE) if ssi else ""
        si_panel = (_colcap("Return · since inception")
                    + _si_chart + _mini_legend(si_leg)) if _si_chart else ""

        # ── Volatility: two panels, not six ───────────────────────────────────
        # The by-window grid is gone. Six volatility plots restated one fact the
        # RISK section already carries a tile for, and their per-window sigmas
        # ranged 8.74-10.99% on the reference book — a spread no reader acts on.
        # What survives is the pair that answers a question the tile cannot: was
        # the last quarter rougher than the book has been all along.
        vol_3m = _vol_panel(vol_windows.get("3m"),
                            (vol_windows.get("3m") or {}).get("dates") or si_dates,
                            month_ticks=True, min_day_ticks=0,
                            w=W_HALF, h=H_HALF, gutter=G_HALF)
        if vol_3m:
            _s3 = ((vol_windows.get("3m") or {}).get("window_sigma") or {}).get("port")
            vol_3m = (_colcap("Volatility · 3M"
                              + (f" · σ {_pct(_s3, signed=False)}" if _s3 is not None
                                 else ""))
                      + vol_3m + _mini_legend(_vol_legend(vol_windows.get("3m"))))
        vol_si_panel = _vol_panel(
            vol_si, vol_si["dates"] if vol_si else si_dates,
            month_ticks=True, min_day_ticks=0,
            w=W_HALF, h=H_HALF, gutter=G_HALF)
        if vol_si_panel:
            # The legend states each line's σ over the book's whole life, so the
            # three are compared over one period. The figure beside each name is
            # NOT the line's endpoint (that is one 21-session window, drawn at the
            # line's end) — the note under the row says which is which.
            vol_si_panel = (_colcap("Volatility · since inception")
                            + vol_si_panel
                            + _mini_legend(_vol_legend(vol_si, with_span=True)))
        vol_row = ""
        if vol_3m or vol_si_panel:
            # One note for the pair rather than a caption each: at half width the
            # estimator's description does not fit beside the window's name, and it
            # is the same sentence twice.
            vol_row = (
                f'<table role="presentation" width="100%" cellpadding="0" '
                f'cellspacing="0" border="0"><tr>'
                f'<td width="50%" valign="top" style="padding:0 8px 0 0;">'
                f'{vol_3m}</td>'
                f'<td width="50%" valign="top" style="padding:0 0 0 8px;'
                f'border-left:1px solid {P["border"]};">{vol_si_panel}</td>'
                f'</tr></table>'
                f'<div style="{TYPE["data"]}color:{P["subtle"]};margin-top:6px;">'
                f'Annualized. Line: rolling 21 sessions. '
                f'Figures in the keys: σ over the panel’s own span.</div>')

        _rule = (f'<div style="margin-top:12px;padding-top:12px;'
                 f'border-top:1px solid {P["border"]};"></div>')
        _blocks = [b for b in (ret_grid, si_panel, vol_row) if b]
        charts_tbl = f'<div style="margin-top:12px;">{_rule.join(_blocks)}</div>'
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
    # The heading names what the charts actually draw. Only claim "target" when
    # the target line is on them — on a run with no per-instrument targets (or a
    # sleeve without price history) it is withheld, and a heading promising a
    # comparison the reader cannot find is worse than the generic one.
    drawn = {name for _c, name in si_leg + ret_leg}
    title = (f"Vs target &amp; {bench_label}"
             if "Target" in drawn else "Vs the market")
    return {"available": True,
            "matrix_html": matrix_card,
            "vs_market_html": "".join(parts),
            "vs_market_title": title,
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


def _intraday_baseline_known(quote) -> bool:
    """Does this quote tell us what the instrument closed at yesterday?

    That is the whole question behind "has it moved today". With a baseline and no
    bars the answer is a definite ZERO -- the instrument is marked at that close --
    while without one there is no valuation at all. ``intraday_feeds`` emits the
    baseline on its no-bars branch for exactly this reason, so the distinction is
    already carried and only needed reading.
    """
    _intra, base = _intraday_quote_parts(quote)
    try:
        return float(base) > 0
    except (TypeError, ValueError):
        return False


def _intraday_pct_path(quote):
    """One instrument's session as % against its own previous close, or None.

    The baseline is the previous close preprocessing retained from the SAME feed
    as the bars. Falling back to the session's first print is for lightweight
    fixtures: it makes the line open at 0% instead of at the real gap.
    """
    intra, feed_baseline = _intraday_quote_parts(quote)
    if intra is None or len(intra) < 2:
        return None
    try:
        base = float(feed_baseline)
    except (TypeError, ValueError):
        base = float("nan")
    if not math.isfinite(base) or base == 0:
        base = float(intra.iloc[0])
    if not base:
        return None
    return (intra.astype(float) / base - 1.0) * 100.0


def _intraday_weighted_path(quotes: dict, weights: dict):
    """A weight-blended session, or None unless EVERY sleeve has one.

    Weights need not sum to 100 — a target may be stated over one sleeve of the
    book — so the blend divides by the weight it actually carries. What it may not
    do is SKIP a sleeve: a path weighted over the part of an allocation that
    happened to trade is a different portfolio under that allocation's name. Hence
    the all-or-nothing return, the same rule ``metrics._target_history`` applies to
    the daily line rather than renormalising what is left.
    """
    if not weights:
        return None
    paths, total = [], 0.0
    for key, weight in weights.items():
        try:
            weight = float(weight or 0.0)
        except (TypeError, ValueError):
            return None
        if weight <= 0:
            continue
        quote = quotes.get(str(key))
        path = _intraday_pct_path(quote)
        if path is not None:
            paths.append((weight, path))
            total += weight
            continue
        # No session data is not a reason to withhold the allocation. Whatever the
        # cause -- a thin fund that has not printed yet, or a symbol Yahoo did not
        # return at all -- the sleeve's mark today is its previous close, which is
        # exactly what the VALUATION does: ``current_session.pick_quote`` rejects an
        # unusable quote and the tape keeps the last close. So its contribution is
        # ZERO: the weight divides, and adds nothing to the numerator, which is a flat
        # 0% line without inventing timestamps for one.
        #
        # This is also the rule the PORTFOLIO line has always effectively applied,
        # and the asymmetry was mine: refusing here while the portfolio blended on
        # meant one panel drew its own line and withheld the other's over the same
        # missing symbol. Measured on the reference book: one sleeve of nine (5% of
        # the target) absent from the quote catalog withheld the whole target line
        # while all eight others had bars.
        total += weight
    if not paths or total <= 0:
        return None
    axis = None
    for _w, p in paths:
        axis = p.index if axis is None else axis.union(p.index)
    blend = None
    for weight, p in paths:
        aligned = p.reindex(axis).ffill().bfill() * weight
        blend = aligned if blend is None else blend + aligned
    return None if blend is None else blend / total


#: The lines a 1D panel may draw, keyed as every other window's are, so one gate
#: loop covers all six. Order is draw order: references after the portfolio.
_INTRADAY_LINE_KEYS = ("twror", "target", "acwi")


def _tape_one_day(m, geo_name: Optional[str] = None) -> dict:
    """The AUTHORITATIVE 1D per line, from the tape rather than from the bars.

    This is the convention the rest of the project already follows and the 1D grid
    cell was the only exception to. Every per-row 1D cell in RETURNS, the Watchlist
    and Target instruments takes its PILL from a tape figure and uses the bars only
    for the sparkline's shape (``_perf_spark_cell``'s ``day_val`` against its
    ``intraday_map``); the portfolio row takes its pill from the NAV. None of them
    reads the end of a drawn line.

    Reading the label off the bars is what let one sleeve spoil it: a symbol the quote
    catalog did not return contributes nothing to a blended path, so the drawn end was
    short by its weight times its move, and the label inherited that. The tape knows
    that instrument's day — it is the number its own row prints — so the label does
    not have to guess.

    Returns ``{key: pct}`` for the keys this panel draws, any of which may be absent:

    * ``twror``  the NAV's own 1D, the same figure the matrix row and the Session tile
      carry, so the panel cannot disagree with the two things beside it.
    * ``target`` the sleeves' 1D weighted by ``target_weights``. No single series
      exists for it, so it is composed here — and a sleeve whose own 1D is unknown is
      left out of both sides of the ratio, which is the one place renormalising is
      right: there is no third option for a figure nothing reports.
    * ``acwi``   the benchmark's 1D off its own daily history.
    """
    from tarzan.engine.stats import compute_period_return

    out: dict[str, float] = {}

    nav = getattr(m, "portfolio_history", None)
    if nav is not None and len(nav) >= 2:
        v = compute_period_return(_norm_series(nav).dropna(), "1d")
        if v is not None:
            out["twror"] = float(v)

    hp = getattr(m, "holding_performance", None)
    per_ticker: dict[str, float] = {}
    if hp is not None and not getattr(hp, "empty", True) and "ticker" in hp.columns:
        for _i, row in hp.iterrows():
            try:
                val = row.get("1d")
                if val is None or (isinstance(val, float) and val != val):
                    continue
                per_ticker[str(row.get("ticker", ""))] = float(val)
            except (TypeError, ValueError):
                continue

    weights = getattr(m, "target_weights", {}) or {}
    num = den = 0.0
    for key, weight in weights.items():
        try:
            weight = float(weight or 0.0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        move = per_ticker.get(str(key))
        if move is None:
            continue
        num += weight * move
        den += weight
    if den > 0:
        out["target"] = num / den

    bench = _geo_benchmark_series(m, geo_name)
    if bench is not None and len(bench) >= 2:
        v = compute_period_return(bench.dropna(), "1d")
        if v is not None:
            out["acwi"] = float(v)
    return out


def _perf_intraday_window(m, geo_name: Optional[str] = None) -> Optional[dict]:
    """The 1D panel's window: today's SESSION, shaped like ``_perf_window``.

    1D cannot come from daily bars. Such a window opens on the PREVIOUS session, so
    it spans at most two closes — and once truncated to the boundary the portfolio
    and the benchmark share it can span NONE, because their tapes routinely end a
    session apart (measured: a benchmark anchored 27 Aug against a NAV ending
    26 Aug, an empty window while both 1D returns existed and printed in the
    matrix). Two points is a straight segment that looks like a trajectory.

    So this reads the intraday bars instead — already fetched for the RETURNS
    table's sparklines, so it costs no call and cannot disagree with the 1D column.
    Each line is rebased on its own previous close and then sampled on ONE shared
    clock axis: the union of every feed's stamps, forward filled. Reindexing onto a
    single feed's stamps would drop the others' prints, and two venues do not tick
    together.

    Returns the same keys ``_perf_window`` does for the lines this panel draws, so
    the semantic gate verifies 1D through the identical code path as 5D or 1Y.
    ``None`` when no line can be built.

    One authority, called twice: once by the renderer to draw, once by the gate to
    check what was drawn.
    """
    quotes = dict(getattr(m, "intraday_quotes", {}) or {})
    if not quotes:
        return None

    raw: dict[str, object] = {}
    # Portfolio: the value-weighted path the RETURNS table's own portfolio row
    # draws, as % (that helper returns a level based at 100).
    pf = _portfolio_intraday_series(m)
    if pf is not None and len(pf) >= 2:
        raw["twror"] = pf.astype(float) - 100.0
    target = _intraday_weighted_path(quotes, getattr(m, "target_weights", {}) or {})
    if target is not None and len(target) >= 2:
        raw["target"] = target
    bench_ticker = str((getattr(m, "benchmark_tickers", {}) or {}).get(geo_name) or "")
    bench = _intraday_pct_path(quotes.get(bench_ticker))
    if bench is not None:
        raw["acwi"] = bench
    if not raw:
        return None

    axis = None
    for key in _INTRADAY_LINE_KEYS:
        s = raw.get(key)
        if s is None:
            continue
        axis = s.index if axis is None else axis.union(s.index)
    if axis is None or len(axis) < 2:
        return None

    out: dict[str, object] = {"dates": list(axis)}
    endpoints: dict[str, Optional[float]] = {}
    for key in _INTRADAY_LINE_KEYS:
        s = raw.get(key)
        vals = (list(s.reindex(axis).ffill().bfill().values.astype(float))
                if s is not None else [])
        # A line whose plotted end is not finite is NOT a line. Reporting it would
        # make the semantic gate demand a panel that ``chart_pct_compact`` declines
        # to draw (it returns "" when no series holds a finite point), so "resolved"
        # has to mean "drawable" or the gate accuses the renderer of dropping a line
        # it was never able to draw.
        if not vals or not all(math.isfinite(v) for v in vals[-1:]):
            out[key] = None
            endpoints[key] = None
            continue
        out[key] = vals
        # The endpoint is the LAST PLOTTED value, not ``s.iloc[-1]``. The two agree,
        # but only by an argument about forward-fill, and the rule this section is
        # built on is that a label is read off the exact array handed to the chart.
        endpoints[key] = float(vals[-1])
    if all(v is None for v in endpoints.values()):
        return None
    out["endpoints"] = endpoints
    # The figures the panel PRINTS, which are not the ends of the drawn lines: the
    # bars say WHEN the day moved, the tape says BY HOW MUCH. Every other 1D cell in
    # the newsletter already works this way — see ``_tape_one_day`` — and reading the
    # label off the bars was what let a single unreturned symbol shift it. Only for
    # keys the panel actually drew: labelling a line that is not there is worse than
    # the figure being absent.
    tape = _tape_one_day(m, geo_name)
    out["labels"] = {k: tape[k] for k in _INTRADAY_LINE_KEYS
                     if k in tape and endpoints.get(k) is not None}
    out["window_start"] = axis[0]
    out["window_end"] = axis[-1]
    out["source_end_dates"] = {"portfolio": axis[-1], "benchmark": axis[-1]}
    # The session the panel's x-axis must span, so a day that is half done draws
    # half a chart. Without it the plot spreads its points evenly over the full
    # width whatever the hour, and a running session looks finished. Taken from the
    # venue that produced the most heavily weighted path -- the holdings' own -- and
    # None when no cash session is modelled (a continuously traded book), which
    # leaves the caller on the even spread it has always used.
    from tarzan.data.market_quotes import session_span
    span = None
    try:
        first = next((str(t) for t in (m.holdings_df["ticker"]
                                       if getattr(m, "holdings_df", None) is not None
                                       and "ticker" in m.holdings_df else [])
                      if str(t) in quotes), "")
        if first:
            src = str((quotes.get(first) or {}).get("intraday_source_ticker") or first)
            span = session_span(src)
    except Exception:  # noqa: BLE001 — no venue is not an error, only no span
        span = None
    out["session_span"] = span
    return out


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
    flat_weight = 0.0   # held, valued, but has not traded today: contributes 0%
    for ticker_value, weight in zip(df["ticker"], df["weight_pct"]):
        if weight is None:
            continue
        ticker = str(ticker_value)
        quote = intraday_map.get(ticker)
        intra, feed_baseline = _intraday_quote_parts(quote)
        if intra is None or len(intra) < 2:
            # Skipping outright renormalised the blend over the holdings that HAD
            # traded, which overstates the day: a position marked at its previous
            # close moved 0%, it did not vanish from the book. Its weight belongs in
            # the denominator — the same rule the target blend applies, so the two
            # lines in one cell cannot disagree about a quiet holding.
            flat_weight += float(weight)
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
    # The non-traders' weight divides but does not add: that IS their flat 0%.
    wsum += flat_weight
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


# Short marks for the currencies that have one; the ISO code otherwise, which is
# clearer than a glyph nobody recognises (kr is three different currencies).
_CURRENCY_MARK = {"EUR": "\u20ac", "USD": "$", "GBP": "\u00a3", "JPY": "\u00a5"}


def _currency_mark(code) -> str:
    """`` [$]`` / `` [\u20ac]`` for the instrument's own LISTING currency.

    On every row whose currency is KNOWN, not only the non-EUR ones: "no mark means
    euro" is a convention the reader has to be told, and the mark is what says
    whether a figure was converted. It is the LISTING's currency, not the fund's
    base currency — EXUS.MI is named "Xtrackers MSCI World ex USA UCITS ETF 1C
    USD" and trades on Milan in EUR, so it reads [\u20ac] and its returns are
    unconverted.

    The one row with NO mark is the one whose currency the provider never reported
    (a listing whose ``.info`` call failed while its daily history succeeded). That
    used to be defaulted to EUR upstream, which printed [\u20ac] on figures that
    may well have been dollars. An absent mark states nothing; it does not state
    euro.
    """
    code = str(code or "").strip().upper()
    if not code:
        return ""
    return f" [{_CURRENCY_MARK.get(code, code)}]"


def _perf_name_html(name: str, ticker: str, tags: list,
                    currency: object = None) -> str:
    """Instrument label used in the returns tables. Delegates to the shared
    :func:`uni_name`, so the ticker leads the name here exactly as it does in
    the book, the risk table and the optimizer, and the role stays in the group
    header rather than stacking a caption under every name."""
    from tarzan.export.newsletter._constants import uni_name
    # Tight line-height: the curated name (already abbreviated in _format so
    # it fits ~2 lines in the 148px column) wraps with an almost-zero gap
    # between line one and two, so a two-line name barely adds row height.
    return uni_name(f"{name}{_currency_mark(currency)}", ticker or "",
                    tags=tuple(tags or ()), line_height=1.05)

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
            # ``period_keys`` alone dropped the listing currency, so every row in
            # the holdings table rendered without its marker while the watchlist
            # (which reads the frame row directly) carried all 55. Projecting a
            # fixed key list is what hid it: a new column is invisible here until
            # it is named.
            projected = {k: pr.get(k) for k in period_keys}
            projected["currency"] = pr.get("currency")
            perf_by_ticker[str(pr.get("ticker", ""))] = projected

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
            "_ac": h.get("asset_class"),
            "_isin": isin, "_ticker": ticker,
            # Ticker + the curated (abbreviated) name, exactly like the
            # Watchlist rows, so the two grids read as one view. No hard cut:
            # the abbreviation in _format keeps it to ~2 tight lines. (The Book
            # below still carries the full untruncated name for every holding.)
            "name_html": _perf_name_html(
                display_instrument_name(isin, ticker, raw_name),
                display_tk, [],
                perf_by_ticker.get(ticker, {}).get("currency")),
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

#: The windows the movers grid ranks over — the short end of PERIOD_WINDOWS. Beyond
#: three months a "mover" stops being news and becomes the lifetime return, which the
#: hero and the Returns table already carry.
_MOVER_WINDOWS = (("1d", "1D"), ("5d", "5D"), ("1m", "1M"), ("3m", "3M"))

#: How many each side of a window's ranking. Three is what fits a half-column beside
#: the target grid without truncating a ticker.
_MOVER_RANKS = 3


def _mover_pp(value: Optional[float]) -> str:
    """A contribution in points of the total, tapered by magnitude.

    A one-day contribution routinely runs under a tenth of a point, where two decimals
    round every line in the column to 0.00 and the ranking stops being legible; a
    three-month one runs to whole points, where three decimals are noise. Same taper
    as ``_pct_compact`` applies to percentages, for the same reason.
    """
    if value is None:
        return ""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(float(value)):.{3 if abs(value) < 0.1 else 2}f}"


def _mover_tint(fg: str, bg: str, alpha: float) -> str:
    """``fg`` over ``bg`` at ``alpha``, resolved to a flat hex.

    Email clients cannot be relied on for ``rgba()`` in an inline style, so the blend
    is computed here and shipped as an opaque colour.
    """
    a = max(0.0, min(1.0, alpha))
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{int(round(f[i] * a + b[i] * (1 - a))):02x}" for i in range(3))


def _mover_rows(universe: list[dict], key: str) -> tuple[list[dict], list[dict]]:
    """``(ahead, behind)`` for one window, ranked by CONTRIBUTION.

    Contribution, not return: a 4.5% sleeve down 6.6% and a 12.8% sleeve down 0.8%
    are not the same news, and the second moved the book more. Ranking on the return
    alone put the small one at the top of the list while it mattered a third as much.
    """
    have = [r for r in universe if r["contrib"].get(key) is not None]
    have.sort(key=lambda r: -r["contrib"][key])
    return have[:_MOVER_RANKS], list(reversed(have[-_MOVER_RANKS:]))


def _mover_universe(rows: list[dict]) -> list[dict]:
    """Attach each line's per-window contribution (weight x return / 100)."""
    out = []
    for r in rows:
        weight = float(r.get("weight") or 0.0)
        if weight <= 0:
            continue
        contrib = {}
        for key, _ in _MOVER_WINDOWS:
            value = r["rets"].get(key)
            contrib[key] = None if value is None else weight / 100.0 * value
        if all(v is None for v in contrib.values()):
            continue
        out.append(dict(r, weight=weight, contrib=contrib))
    return out


def _mover_grid_html(universe: list[dict], label: str, swatch: str,
                     total_by_window: dict) -> str:
    """One half of the movers section: a grid of ticker x window.

    Only the lines that reached a top-three or bottom-three in some window are shown.
    The alternative — every line, with the quiet ones tinted faintly — was rejected as
    too busy: it printed twenty-five rows to say something about eleven of them.

    The figure prints only where a line WAS an extreme. The tint carries the magnitude
    of everything else, which is what it is for; filling all fifty-six cells with
    numbers made them compete with each other and the tint stopped reading at all.
    """
    marks: dict[str, dict[str, int]] = {}
    scale: dict[str, float] = {}
    for key, _ in _MOVER_WINDOWS:
        ahead, behind = _mover_rows(universe, key)
        for r in ahead:
            marks.setdefault(r["ticker"], {})[key] = 1
        for r in behind:
            marks.setdefault(r["ticker"], {})[key] = -1
        magnitudes = [abs(r["contrib"][key]) for r in universe
                      if r["contrib"].get(key) is not None]
        scale[key] = max(magnitudes) if magnitudes else 1.0
    shown = sorted((r for r in universe if r["ticker"] in marks),
                   key=lambda r: -r["weight"])
    if not shown:
        return ""
    dropped = len(universe) - len(shown)
    heaviest = max(r["weight"] for r in universe)

    head = (f'<tr><td style="{TYPE["label"]}color:{PALETTE["subtle"]};'
            f'padding:0 0 4px 0;">Ticker</td>'
            f'<td style="{TYPE["label"]}color:{PALETTE["subtle"]};'
            f'padding:0 0 4px 6px;">Wt</td>')
    for _, column in _MOVER_WINDOWS:
        head += (f'<td align="center" style="{TYPE["label"]}'
                 f'color:{PALETTE["subtle"]};padding:0 0 4px 0;">{column}</td>')
    head += "</tr>"

    body = ""
    for r in shown:
        fill = max(2, int(round(r["weight"] / heaviest * 26)))
        body += (
            f'<tr><td style="{TYPE["data"]}color:{PALETTE["ink"]};padding:1px 0;'
            f'white-space:nowrap;">{_esc(r["bare"])}</td>'
            f'<td style="padding:1px 0 1px 6px;white-space:nowrap;">'
            f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
            f'style="width:26px;background:{PALETTE["group_bg"]};border-radius:2px;">'
            f'<tr><td style="width:{fill}px;height:5px;'
            f'background:{PALETTE["accent"]};border-radius:2px;font-size:0;'
            f'line-height:0;">&nbsp;</td>'
            f'<td style="font-size:0;line-height:0;">&nbsp;</td></tr></table></td>')
        for key, _ in _MOVER_WINDOWS:
            value = r["contrib"].get(key)
            if value is None:
                body += (f'<td align="center" style="{TYPE["prose"]}'
                         f'color:{PALETTE["subtle"]};padding:1px 2px;">&middot;</td>')
                continue
            colour = PALETTE["green"] if value >= 0 else PALETTE["red"]
            tint = _mover_tint(colour, PALETTE["card_alt"],
                               min(1.0, abs(value) / scale[key]) * 0.75)
            extreme = marks.get(r["ticker"], {}).get(key)
            body += (
                f'<td align="center" style="background:{tint};font-size:9px;'
                f'font-weight:{"700" if extreme else "400"};'
                f'color:{PALETTE["ink"] if extreme else PALETTE["subtle"]};'
                f'font-variant-numeric:tabular-nums;padding:1px 4px;'
                f'border:1px solid {PALETTE["card_alt"]};white-space:nowrap;">'
                f'{_mover_pp(value) if extreme else "&nbsp;"}</td>')
        body += "</tr>"

    foot = (f'<tr><td colspan="2" style="{TYPE["label"]}color:{PALETTE["subtle"]};'
            f'padding:4px 0 0 0;white-space:nowrap;">All</td>')
    for key, _ in _MOVER_WINDOWS:
        total = total_by_window.get(key)
        colour = (PALETTE["subtle"] if total is None
                  else PALETTE["green"] if total >= 0 else PALETTE["red"])
        foot += (f'<td align="center" style="{TYPE["data"]}color:{colour};'
                 f'font-variant-numeric:tabular-nums;padding:4px 2px 0 2px;'
                 f'white-space:nowrap;">'
                 f'{"—" if total is None else _pct(total, signed=True)}</td>')
    foot += "</tr>"

    note = f"{len(shown)} of {len(universe)}"
    if dropped:
        note += f" · {dropped} not top three"
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'border="0"><tr>'
        f'<td style="padding:0 0 5px 0;border-bottom:1px solid '
        f'{PALETTE["border"]};white-space:nowrap;">'
        f'<span style="display:inline-block;width:7px;height:7px;'
        f'background:{swatch};border-radius:2px;"></span>'
        f'<span style="{TYPE["label"]}color:{PALETTE["ink"]};padding-left:6px;">'
        f'{_esc(label)}</span>'
        f'<span style="{TYPE["prose"]}color:{PALETTE["subtle"]};padding-left:7px;">'
        f'{note}</span></td></tr></table>'
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'style="margin-top:8px;border-collapse:separate;border-spacing:0;">'
        f'{head}{body}{foot}</table>')


def _build_portfolio_movers(ctx: _NewsletterContext) -> dict:
    """Which lines moved the number, over each short window, held and planned.

    Replaces the Attribution section, which answered the same question over one window
    only — the lifetime — and answered it with a waterfall whose parts summed to the
    total. This ranks by the same measure (weight times return) over four windows, and
    covers the target as well as the book, so nothing Attribution said is lost.

    Weights come from where each universe's weights actually live: the book's from
    ``holdings_df`` (share of current value), the plan's from ``target_weights``. A
    line held AND planned appears in both, at its two different weights, which is the
    point — it is a different share of each.
    """
    m = ctx.metrics
    hp = getattr(m, "holding_performance", None)
    if hp is None or getattr(hp, "empty", True) or "ticker" not in hp.columns:
        return {"available": False}

    perf_by_ticker: dict[str, dict] = {}
    for _i, row in hp.iterrows():
        ticker = str(row.get("ticker") or "")
        if not ticker:
            continue
        rets = {}
        for key, _ in _MOVER_WINDOWS:
            value = row.get(key)
            if value is None or (isinstance(value, float) and value != value):
                rets[key] = None
            else:
                rets[key] = float(value)
        # A ticker can appear twice, held on one venue and tracked on another. The
        # held listing wins, because this section ranks positions.
        held = "portfolio" in str(row.get("type") or "").lower()
        if ticker not in perf_by_ticker or held:
            perf_by_ticker[ticker] = {
                "ticker": ticker,
                "bare": ticker.split(".")[0].upper(),
                "rets": rets,
            }

    df = getattr(m, "holdings_df", None)
    book_rows = []
    if df is not None and not df.empty and "ticker" in df.columns:
        for _i, row in df.iterrows():
            entry = perf_by_ticker.get(str(row.get("ticker") or ""))
            if entry is None:
                continue
            book_rows.append(dict(entry, weight=row.get("weight_pct")))

    plan_rows = []
    for ticker, weight in (getattr(m, "target_weights", {}) or {}).items():
        entry = perf_by_ticker.get(str(ticker))
        if entry is not None:
            plan_rows.append(dict(entry, weight=weight))

    book = _mover_universe(book_rows)
    plan = _mover_universe(plan_rows)
    if not book and not plan:
        return {"available": False}

    def _weighted(universe: list[dict], key: str) -> Optional[float]:
        """The universe's own return for the window — the number its parts add up
        towards, so a column of contributions can be read against something."""
        num = sum(r["contrib"][key] for r in universe
                  if r["contrib"].get(key) is not None)
        den = sum(r["weight"] / 100.0 for r in universe
                  if r["contrib"].get(key) is not None)
        return num / den if den else None

    # The book's own figure is the portfolio's measured return, not a reweighting of
    # its parts: the two differ by the cash and the lines with no market history, and
    # the measured one is what every other section prints.
    port = getattr(m, "performance", None) or {}
    book_totals = {}
    for key, _ in _MOVER_WINDOWS:
        value = port.get(key)
        if value is None or (isinstance(value, float) and value != value):
            value = _weighted(book, key)
        book_totals[key] = None if value is None else float(value)
    plan_totals = {key: _weighted(plan, key) for key, _ in _MOVER_WINDOWS}

    left = _mover_grid_html(book, "Portfolio", PALETTE["port"], book_totals) if book else ""
    right = _mover_grid_html(plan, "Target", PALETTE["target"], plan_totals) if plan else ""
    if not left and not right:
        return {"available": False}
    if left and right:
        html = ('<table role="presentation" width="100%" cellpadding="0" '
                'cellspacing="0" border="0"><tr>'
                f'<td width="50%" valign="top" style="padding:0 8px 0 0;">{left}</td>'
                f'<td width="50%" valign="top" style="padding:0 0 0 8px;">{right}</td>'
                '</tr></table>')
    else:
        html = left or right
    return {"available": True, "html": html}


def _build_movers(ctx: _NewsletterContext) -> dict:
    """Find best & worst performer over the last week."""
    m = ctx.metrics
    if m.holding_performance.empty:
        return {"available": False}

    hp = m.holding_performance
    # Held positions only, named positively. "portfolio OR not benchmark" let
    # every OTHER kind of row through on the second clause, so the day a third
    # kind arrived (the not-held target instruments) an instrument the book does
    # not own could be ranked its best or worst performer.
    if "type" in hp.columns:
        hp = hp[hp["type"].astype(str).str.contains("portfolio", case=False,
                                                    na=False)]
    if hp.empty or "5d" not in hp.columns:
        return {"available": False}

    # Drop rows with no 5D before ranking. With na_position="last" the LAST row
    # is the missing one whenever any holding lacks a 5D (a new position, a feed
    # with under two closes), so "worst" named an instrument with no return at
    # all: `float(nan or 0.0)` is NaN (NaN is truthy), which rendered the card as
    # "—" in red while the real worst performer was never shown.
    hp = hp.dropna(subset=["5d"])
    if hp.empty:
        return {"available": False}

    sorted_hp = hp.sort_values("5d", ascending=False)
    best = sorted_hp.iloc[0]
    worst = sorted_hp.iloc[-1]

    df = m.holdings_df

    def _enrich(row):
        ticker = row.get("ticker", "")
        match = df[df["ticker"] == ticker] if not df.empty else pd.DataFrame()
        value = float(match["current_value"].iloc[0]) if not match.empty else 0.0
        pct = float(row.get("5d") or 0.0)
        eur = value * pct / 100
        return {
            "name": row.get("name", ticker),
            "ticker": ticker,
            # No class/colour here either: both were unread, and both could only
            # be filled by defaulting an unmatched ticker to "Equities".
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

    # A taxonomy row can be BOTH held and flagged watchlist=true (a fund we own
    # and keep tracking). Such an instrument already has a row in the returns
    # snapshot above, so listing it again here duplicated it — and invited
    # comparing the portfolio against something it owns. The held row wins; the
    # watchlist copy is dropped.
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

    # Instruments the TARGET portfolio names but the book does not hold yet, from
    # the engine's own ``Target not held`` rows (one per seeded target, priced and
    # measured like a holding — see MetricsEngine._holding_performance). They are
    # not watched, they are the plan, so they get their own table above the
    # watchlist. A target already held needs no row here: it is in RETURNS, and
    # the held-name drop above has already taken it out of this table.
    target_rows: list[dict] = []
    if not hp.empty and "type" in hp.columns:
        target_df = hp[hp["type"].astype(str).str.contains(
            "target", case=False, na=False)]
        for _, r in target_df.iterrows():
            raw_ticker = str(r.get("ticker") or "").strip()
            bare = normalize_ticker(raw_ticker)
            asset_class, role = taxonomy.get(bare, (None, None))
            target_rows.append({
                "name": display_instrument_name(
                    r.get("isin"), raw_ticker,
                    str(r.get("name") or raw_ticker)),
                "ticker": _display_ticker(raw_ticker),
                "raw_ticker": raw_ticker,
                "asset_class": asset_class,
                "role": role,
                "currency": r.get("currency"),
                "d1": r.get("1d"),
                "live": bool(r.get("live_1d", False)),
                "tags": [],
                "tag": None,
                "is_portfolio": False,
                "returns": _build_bench_returns_dict(r.to_dict()),
            })

    # A target instrument that is ALSO a curated benchmark (most of them are)
    # has a row in the catalog too, so drop that copy the same way a held
    # benchmark's is dropped. Bare ticker is the key, not the name: the seed
    # carries the operational symbol the enricher resolved (AVWC.DE) and the
    # broker's own description, the catalog row carries the curated name and
    # whichever listing it priced (AVWC), so the two only meet on the stripped
    # symbol — which is what ``normalize_ticker`` produces on both sides.
    target_bares = {
        normalize_ticker(str(r.get("raw_ticker") or "")) for r in target_rows
    } - {""}

    # The catalog is the FETCH set, which is wider than the watchlist: a row
    # flagged only as the alpha/beta or geo reference has its series pulled for
    # the charts, the gap tile and the risk table, and printing it here would put
    # an instrument in the watchlist table that the taxonomy says is not on the
    # watchlist. ``watchlist=true`` is what this table lists.
    watchlisted = cfg.watchlist_names()

    benchmark_rows: list[dict] = []
    if not hp.empty and "type" in hp.columns:
        ab_name = (ctx.benchmark_alpha_beta or "").strip().lower()
        geo_name = (ctx.benchmark_geo or "").strip().lower()
        bench_df = hp[hp["type"].astype(str).str.contains("enchmark", case=False, na=False)]
        for _, r in bench_df.iterrows():
            name = str(r.get("name") or r.get("ticker", ""))
            name_norm = name.strip().lower()
            if name_norm in held_bench_names or name_norm not in watchlisted:
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
            if bare in target_bares:
                continue
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
                "currency": r.get("currency"),
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

    # Group rows by asset class → role, in the configured order, then hand off to
    # the shared table renderer. Two tables come out of the same rows here — the
    # target instruments and the rest of the watchlist — so each scales its own
    # conditional formatting on its own columns, which is the rule _heat states
    # for every returns grid in the issue.
    #
    # No portfolio row in either. Both list instruments that are NOT held; the
    # portfolio's own returns are the whole of RETURNS one section up, and
    # repeating them here invited a comparison between the portfolio and a list
    # of things it does not own.
    def _table_for(rows: list) -> str:
        raw_groups = group_by_class_role(
            rows, asset_class=lambda r: r.get("asset_class"),
            role=lambda r: r.get("role"), ticker=lambda r: r.get("ticker"))
        groups = []
        for ac, col, role_list in raw_groups:
            rendered_roles = []
            for role, grp_rows in role_list:
                insts = []
                for r in grp_rows:
                    _, inner = _perf_spark_cell(
                        r.get("d1"), r.get("raw_ticker"), intraday_map,
                        live=bool(r.get("live")))
                    insts.append({
                        # A short name here, not just the ticker: these
                        # instruments are tracked and NOT held, so no other
                        # section in the issue says what they are. No hard cut —
                        # the curated name is already abbreviated to ~2 tight
                        # lines in _format.
                        "name_html": _perf_name_html(r["name"], r.get("ticker"),
                                                     r.get("tags"),
                                                     r.get("currency")),
                        "spark_inner": inner,
                        "day_raw": r.get("d1"),
                        "returns": r["returns"],
                    })
                rendered_roles.append((role, insts))
            groups.append((ac, col, rendered_roles))
        return _returns_table_html(
            period_cols, None, groups,
            day_label=day_column_label(
                m, live=_intraday_column(r.get("live") for r in rows)))

    table_html = _table_for(benchmark_rows)
    target_table_html = _table_for(target_rows) if target_rows else ""

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
        # The target instruments get their own section above the watchlist; empty
        # when the target names nothing the book does not already hold.
        "target_table_html": target_table_html,
        "target_rows": target_rows,
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

