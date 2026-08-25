"""Inline-SVG chart/spark builders for the newsletter.

Holds the two SVG clipPath id counters (_day_spark_uid, _dual_uid) and the
reset_spark_uids() the orchestrator calls at the start of each render so ids
depend only on how many charts a render draws, not process history.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tarzan.export._format import eur_smart as _eur_smart
from tarzan.export.newsletter._constants import FONT_STACK, PALETTE, TYPE, TYPE_PX

_day_spark_uid = 0

_dual_uid = 0



def reset_spark_uids() -> None:
    """Reset the per-render SVG clipPath id counters. Called by build_context
    so a render's element ids depend only on how many charts it draws, not on
    how many newsletters the process rendered before it (deterministic ids)."""
    global _day_spark_uid, _dual_uid
    _day_spark_uid = 0
    _dual_uid = 0


def _spark(vals: list[float], target: Optional[float], color: str,
           w: int = 250, h: int = 40) -> str:
    """Tiny area+line sparkline with an optional dashed target line.

    Auto-scaled to the data range around the target so small slices stay
    legible (a fixed 0–100 axis would flatten a 6% sleeve). Inline SVG with
    ``preserveAspectRatio="none"`` so it stretches to the table cell width;
    legacy Outlook (Word engine) ignores SVG and simply shows nothing,
    which is acceptable — the target/gap numbers carry the same story.
    Returns an empty string when there are fewer than two points.
    """
    n = len(vals)
    if n < 2:
        return ""
    anchor = target if target is not None else vals[-1]
    lo = min(min(vals), anchor)
    hi = max(max(vals), anchor)
    span = (hi - lo) or 1.0
    lo -= span * 0.18
    hi += span * 0.18
    span = hi - lo

    def _x(i: int) -> float:
        return i / (n - 1) * w

    def _y(v: float) -> float:
        return h - (v - lo) / span * h

    pts = [(_x(i), _y(v)) for i, v in enumerate(vals)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h:.1f} " + line + f" {w},{h:.1f}"
    parts = [
        f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;">',
        f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>',
    ]
    if target is not None:
        ty = _y(float(target))
        parts.append(
            f'<line x1="0" y1="{ty:.1f}" x2="{w}" y2="{ty:.1f}" '
            f'stroke="{PALETTE["muted"]}" stroke-width="1" stroke-dasharray="3,3"/>'
        )
    parts.append(
        f'<polyline points="{line}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    lx, ly = pts[-1]
    parts.append(
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="{color}" '
        f'stroke="#fff" stroke-width="1.5"/>'
    )
    parts.append("</svg>")
    return "".join(parts)

def _day_spark(vals: list[float], baseline: float, w: int = 76, h: int = 22,
               stretch: bool = False) -> str:
    """Yahoo-style intraday sparkline: green area where the line is above the
    previous close (``baseline``), red where below, with a dashed baseline.

    Uses two clipped copies of the line-to-baseline area (above/below the
    baseline) so the fill is two-tone without per-segment crossing math.
    Email clients that render inline SVG (Gmail, Apple Mail) show it;
    legacy Outlook simply omits it, which is acceptable."""
    global _day_spark_uid
    n = len(vals)
    if n < 2:
        return ""
    lo = min(min(vals), baseline)
    hi = max(max(vals), baseline)
    span = (hi - lo) or 1.0
    pad = span * 0.14
    lo -= pad
    hi += pad
    span = hi - lo

    def _xx(i: int) -> float:
        return i / (n - 1) * w

    def _yy(v: float) -> float:
        return h - (v - lo) / span * h

    yb = _yy(baseline)
    line = " ".join(f"{_xx(i):.1f},{_yy(v):.1f}" for i, v in enumerate(vals))
    poly = f"{line} {_xx(n - 1):.1f},{yb:.1f} {_xx(0):.1f},{yb:.1f}"
    _day_spark_uid += 1
    gid, rid = f"sg{_day_spark_uid}", f"sr{_day_spark_uid}"
    yb_c = max(0.0, min(yb, h))
    # Endpoint dot, colored by sign vs the previous-close baseline.
    _dot_col = PALETTE["green"] if vals[-1] >= baseline else PALETTE["red"]
    _dot = f'<circle cx="{_xx(n - 1):.1f}" cy="{_yy(vals[-1]):.1f}" r="1.8" fill="{_dot_col}"/>'
    # ``stretch`` makes the sparkline fill its container width (width:100% +
    # preserveAspectRatio="none") so a card leaves no unused space on the
    # right; otherwise it renders at the fixed ``w`` px.
    _svg_w = "100%" if stretch else str(w)
    _par = 'preserveAspectRatio="none" ' if stretch else ""
    _style_w = "width:100%;" if stretch else ""
    return (
        f'<svg width="{_svg_w}" height="{h}" viewBox="0 0 {w} {h}" {_par}'
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;{_style_w}">'
        f'<defs>'
        f'<clipPath id="{gid}"><rect x="0" y="0" width="{w}" height="{yb_c:.1f}"/></clipPath>'
        f'<clipPath id="{rid}"><rect x="0" y="{yb_c:.1f}" width="{w}" height="{h - yb_c:.1f}"/></clipPath>'
        f'</defs>'
        # Two-tone fill: green above the previous-close baseline, red below.
        f'<polygon points="{poly}" fill="{PALETTE["green"]}" fill-opacity="0.22" clip-path="url(#{gid})"/>'
        f'<polygon points="{poly}" fill="{PALETTE["red"]}" fill-opacity="0.22" clip-path="url(#{rid})"/>'
        # Dashed baseline = previous close (the 0% reference, Yahoo-style).
        f'<line x1="0" y1="{yb:.1f}" x2="{w}" y2="{yb:.1f}" stroke="{PALETTE["subtle"]}" '
        f'stroke-width="0.75" stroke-dasharray="2,2"/>'
        # Two-tone line: same clips, so each segment is green above the
        # baseline and red below it, switching exactly at the dashed line.
        f'<polyline points="{line}" fill="none" stroke="{PALETTE["green"]}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{PALETTE["red"]}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#{rid})"/>'
        f'{_dot}'
        f'</svg>'
    )

def _hero_value_chart(values, pct, dates, flows, w: int = 580, h: int = 196,
                      total_pct=None) -> str:
    """Dual-axis hero chart: portfolio value in €, both P&L measures in %.

    The green/red split is on TOTAL P&L, with the same baseline semantics the
    value line used to carry: green above where the P&L stood when the window
    opened, red below it, about a dashed line at that level. The value line is
    neutral ink, which also frees the accent colour for the cash-flow triangles
    that sit on it.

    Unrealized P&L keeps the violet it carries on the return charts. Total P&L
    gives up its cyan HERE only, and the colour key says so with a two-tone
    swatch, so nothing else in the issue changes meaning.

    ``total_pct`` is optional: with no lifetime P&L (the holdings-only path) no
    series is split and the value simply draws neutral.
    """
    global _dual_uid
    _dual_uid += 1
    u = _dual_uid
    P = PALETTE
    n = len(values)
    if n < 2 or not pct or len(pct) != n:
        return ""
    if total_pct is not None and len(total_pct) != n:
        total_pct = None
    ML, MR, MT, MB = 52, 48, 12, 26
    PW, PH = w - ML - MR, h - MT - MB
    base = values[0]
    from tarzan.export import _charts as _ch
    # Seven target ticks, not four: on a 30-day window the value axis spans a
    # few thousand euros and four gridlines put the whole line between two of
    # them, so a reader could not tell €239k from €241k without the end label.
    _TICKS = 7
    vlo, vhi, vticks = _ch.nice_ticks(
        min(min(values), base), max(max(values), base), _TICKS
    )
    vstep = (vticks[1] - vticks[0]) if len(vticks) > 1 else None
    # The right axis must span BOTH P&L series, or the second line is drawn
    # against a scale that was fitted to the first and rides off the plot.
    pct_all = list(pct) + list(total_pct or ())
    plo, phi, pticks = _ch.nice_ticks(min(pct_all), max(pct_all), _TICKS)

    def X(i):
        return ML + (i / (n - 1) * PW if n > 1 else 0)

    def Yv(v):
        return MT + (1 - (v - vlo) / ((vhi - vlo) or 1)) * PH

    def Yp(p):
        return MT + (1 - (p - plo) / ((phi - plo) or 1)) * PH

    grid = ""
    for t in vticks:
        if t < vlo - 1e-9 or t > vhi + 1e-9:
            continue
        y = Yv(t)
        grid += (
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
            f'stroke="{P["border"]}" stroke-width="1"/>'
            f'<text x="{ML - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="{TYPE_PX["label"]}" fill="{P["subtle"]}">'
            f'{_ch.fmt_eur_tick(t, vstep)}</text>'
        )
    for t in pticks:
        if t < plo - 1e-9 or t > phi + 1e-9:
            continue
        grid += (
            f'<text x="{ML + PW + 6}" y="{Yp(t) + 3:.1f}" '
            f'text-anchor="start" font-size="{TYPE_PX["label"]}" fill="{P["muted"]}">'
            f'{_ch.fmt_pct_tick(t)}</text>'
        )
    xlab = ""
    for k in sorted({0, n // 3, 2 * n // 3, n - 1}):
        x = X(k)
        anc = "start" if k == 0 else "end" if k == n - 1 else "middle"
        xlab += (
            f'<text x="{x:.1f}" y="{h - 8}" text-anchor="{anc}" font-size="{TYPE_PX["label"]}" '
            f'fill="{P["subtle"]}">{pd.Timestamp(dates[k]).strftime("%b %d")}</text>'
        )

    vline = " ".join(f"{X(i):.1f},{Yv(v):.1f}" for i, v in enumerate(values))
    baseline_y = Yv(base)
    pline = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(pct))
    tline = (" ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(total_pct))
             if total_pct is not None else "")
    # The split boundary: where Total P&L stood when the window opened, on the
    # RIGHT axis — the same "versus the window open" reference the value line
    # used to be coloured against. Clamped into the plot band so the clip can
    # never be drawn off the canvas.
    split_y = max(MT, min(Yp(total_pct[0]), MT + PH)) if total_pct else baseline_y

    marks = ""
    if flows:
        xmap = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
        for d, v in flows:
            i = xmap.get(pd.Timestamp(d).normalize())
            if i is None:
                continue
            x, y = X(i), Yv(values[i])
            # Accent for money in, muted for money out. Green was wrong: it is
            # the value line's own colour, so a deposit marker read as part of
            # the line rather than as an event on it. The triangle also stands
            # 9px clear of the line and points away from it, which is what keeps
            # five of them legible on a 30-day window.
            col = P["accent"] if v >= 0 else P["muted"]
            if v >= 0:
                marks += (
                    f'<polygon points="{x:.1f},{y - 9:.1f} '
                    f'{x - 4.2:.1f},{y - 16:.1f} {x + 4.2:.1f},{y - 16:.1f}" '
                    f'fill="{col}"/>'
                )
            else:
                marks += (
                    f'<polygon points="{x:.1f},{y + 9:.1f} '
                    f'{x - 4.2:.1f},{y + 16:.1f} {x + 4.2:.1f},{y + 16:.1f}" '
                    f'fill="{col}"/>'
                )
    # A dot on the last value, so the series has a stated end rather than
    # running off the plot. Neutral: it sits on the value line, and the value no
    # longer carries a sign.
    endpoint = (
        f'<circle cx="{X(n - 1):.1f}" cy="{Yv(values[-1]):.1f}" r="3.4" '
        f'fill="{P["ink"]}" stroke="{P["card_alt"]}" stroke-width="1.8"/>'
    )
    # Total P&L, split green above its window-open level and red below it —
    # the same reference the value line used to be coloured against. Drawn as a
    # filled band rather than a hairline, because it is now the series carrying
    # the colour; the value line stays a plain line.
    if tline:
        tband = (
            f"{tline} {X(n - 1):.1f},{split_y:.1f} {X(0):.1f},{split_y:.1f}"
        )
        pnl_layer = (
            f'<polygon points="{tband}" fill="{P["green"]}" fill-opacity="0.16" '
            f'clip-path="url(#pg{u})"/>'
            f'<polygon points="{tband}" fill="{P["red"]}" fill-opacity="0.16" '
            f'clip-path="url(#pr{u})"/>'
            f'<polyline points="{tline}" fill="none" stroke="{P["green"]}" '
            f'stroke-width="2.4" stroke-linejoin="round" clip-path="url(#pg{u})"/>'
            f'<polyline points="{tline}" fill="none" stroke="{P["red"]}" '
            f'stroke-width="2.4" stroke-linejoin="round" clip-path="url(#pr{u})"/>'
        )
    else:
        pnl_layer = ""

    return (
        f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;'
        f'font-family:{FONT_STACK};">'
        f'<defs><clipPath id="pg{u}"><rect x="0" y="0" width="{w}" '
        f'height="{split_y:.1f}"/></clipPath>'
        f'<clipPath id="pr{u}"><rect x="0" y="{split_y:.1f}" width="{w}" '
        f'height="{h - split_y:.1f}"/></clipPath></defs>'
        + grid
        # One dashed reference, as before: the level the colours are measured
        # against. It used to be the value's window open and is now the P&L's,
        # because that is the series carrying the colour. Unlabelled — the
        # caption named a number the reader can read off the axis, and it sat on
        # top of the line often enough to be noise.
        + f'<line x1="{ML}" y1="{split_y:.1f}" x2="{ML + PW}" '
        f'y2="{split_y:.1f}" stroke="{P["subtle"]}" stroke-width="0.8" '
        f'stroke-dasharray="3,3"/>'
        + pnl_layer
        + f'<polyline points="{pline}" fill="none" stroke="{P["unreal"]}" '
        f'stroke-width="1.8" stroke-dasharray="4,3" stroke-linejoin="round"/>'
        + f'<polyline points="{vline}" fill="none" stroke="{P["ink"]}" '
        f'stroke-width="2.6" stroke-linejoin="round"/>'
        + marks + endpoint + xlab + "</svg>"
    )


def _hero_chart_legend(*, has_total: bool) -> str:
    """Colour key for the hero chart: the € line, then each P&L series drawn.

    The hero used inline axis labels instead of a key, which worked while the
    right axis held one series. With two it does not, and the two P&L colours
    are the same ones the return charts use, so naming them here also teaches
    the mapping the reader needs three sections later.
    """
    P = PALETTE

    def _swatch(color: str) -> str:
        return (f'<span style="display:inline-block;width:7px;height:7px;'
                f'border-radius:2px;background:{color};vertical-align:baseline;'
                f'margin-right:4px;"></span>')

    def _split_swatch() -> str:
        """Two half squares, green then red — the key for a series whose colour
        states its sign. A CSS gradient would say it in one square, but Outlook's
        Word engine drops gradients and the key would vanish."""
        return (f'<span style="display:inline-block;width:7px;height:7px;'
                f'border-radius:2px 0 0 2px;background:{P["green"]};'
                f'vertical-align:baseline;"></span>'
                f'<span style="display:inline-block;width:7px;height:7px;'
                f'border-radius:0 2px 2px 0;background:{P["red"]};'
                f'vertical-align:baseline;margin-right:4px;"></span>')

    items = [(_swatch(P["ink"]), "Value (€, left)")]
    if has_total:
        # Named for what the colour MEANS, since it is the one series here whose
        # colour carries information rather than identity: up or down over the
        # window. Deliberately does not repeat the "window open" caption this
        # change removed from the plot.
        items.append((_split_swatch(),
                      "Total P&amp;L (%, right): green up, red down"))
    items.append((_swatch(P["unreal"]), "Unreal. P&amp;L (%, right)"))
    parts = [f'{swatch}<span style="color:{P["muted"]};">{label}</span>'
             for swatch, label in items]
    return (f'<div style="{TYPE["data"]}margin:7px 0 0;">'
            + "&nbsp;&nbsp;&nbsp;".join(parts) + "</div>")


def _hero_flow_chips(flows) -> str:
    """The window's cash flows as filled chips: date, then amount.

    Date first because the chips read against the chart above them, where the
    triangles sit on dates. Filled tints rather than a white pill with a
    border: the old chips hardcoded ``background:#fff``, which on a dark
    palette printed white lozenges over the card.
    """
    if not flows:
        return ""
    P = PALETTE
    items = ""
    for d, v in sorted(flows, key=lambda t: t[0]):
        fg = P["accent"] if v >= 0 else P["muted"]
        bg = P["accent_bg"] if v >= 0 else P["card_alt"]
        items += (f'<span style="display:inline-block;margin:4px 6px 0 0;'
                  f'padding:2px 8px;border-radius:6px;background:{bg};'
                  f'{TYPE["data"]}color:{fg};'
                  f'white-space:nowrap;font-variant-numeric:tabular-nums;">'
                  f'{pd.Timestamp(d).strftime("%b %d")} '
                  f'{_eur_smart(v, signed=True)}</span>')
    return (f'<div style="margin-top:9px;{TYPE["label"]}'
            f'color:{P["muted"]};">Cash flows in the window</div>'
            f'<div style="margin-top:3px;">{items}</div>')

def _timeline_vals(series: Optional[list], key: str) -> Optional[list[float]]:
    """Extract the per-bucket weights for one category from an allocation
    timeline series. Returns None when the category never carried weight
    (so the caller can skip drawing an empty sparkline)."""
    if not series:
        return None
    vals = [float(pt.get(key, 0.0)) for pt in series]
    return vals if any(v > 0 for v in vals) else None

def _session_xs(ts: list, start, end, w: int) -> Optional[list]:
    """X positions of ``ts`` within the window ``[start, end]``, scaled to ``w``.

    Returns None when the window or the timestamps cannot be compared (a naive
    index against a tz-aware window, a zero-length window), so the caller can
    fall back rather than draw a line at x=0. Timestamps outside the window are
    clamped: a venue that quotes past its modelled close still ends at the right
    edge instead of running off the chart."""
    try:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        width = (end - start).total_seconds()
        if width <= 0:
            return None
        return [max(0.0, min(1.0, (pd.Timestamp(t) - start).total_seconds() / width)) * w
                for t in ts]
    except Exception:  # noqa: BLE001 — mismatched tz-awareness, unorderable index
        return None


def _intraday_spark(intra: "pd.Series", baseline: float,
                    w: int = 62, h: int = 22,
                    in_progress: Optional[bool] = None,
                    session_hours: Optional[float] = None,
                    span: Optional[tuple] = None) -> str:
    """Intraday sparkline on a full-session time axis.

    Unlike the stretched ``_day_spark`` (which spreads N points across the
    whole width regardless of how many there are), each bar is placed at its
    real position within the trading session [open → open+session_length], so
    early in the day the line only fills the left portion and grows rightward
    as the session progresses. A completed session fills the full width. This
    makes "how far into the day we are" visible and doubles as a live vs
    closed cue. Two-tone (green above the previous close, red below), with a
    dashed baseline; stretches to the cell width.

    ``in_progress`` says whether the market is trading *now*: when the caller
    knows this (from exchange hours) it should pass it explicitly so a closed
    same-day session renders full width even if its last bar is recent. When
    None, it is inferred from bar recency (a fallback for when the caller has
    no exchange-hours signal at all, not the normal path for futures/FX,
    which do have one via market_status()).

    ``session_hours`` overrides the 6.5h/8.5h cash-session heuristic below
    for a continuously traded instrument (futures ~23h, FX ~24h) -- without
    it, a bar an hour into a 23-hour session would be placed as if the
    session were 6.5-8.5 hours long, clamped to the right edge and making a
    just-opened market look like a nearly complete one.

    ``span`` is the venue's real ``(open, close)`` for this session
    (market_quotes.session_span) and supersedes both: with it the axis is the
    session itself rather than "the first bar plus a guess", so a market that
    printed three times in its first hour draws three points in the left tenth,
    and a session with a late first print does not start at x=0. Only when no
    span is known does ``in_progress`` still choose between the elapsed-time
    axis and evenly spreading the points.
    """
    global _day_spark_uid
    ts = list(intra.index)
    vals = [float(x) for x in intra.values]
    n = len(vals)
    if n < 2:
        return ""
    t0, t_last = ts[0], ts[-1]
    # Where each bar sits horizontally. Best: the venue's real session window,
    # so the drawn extent is the traded extent — a completed session fills the
    # width, a quiet morning does not, and neither needs a flag to say so.
    xs = _session_xs(ts, span[0], span[1], w) if span else None
    if xs is None:
        # No session window (continuous market, or an unmodelled venue). Fall
        # back to elapsed time from the first bar over an assumed session
        # length; only a session known to be OVER is spread evenly, since then
        # the bars cover it by definition. Prefer the caller's exchange-hours
        # signal; fall back to bar recency when it isn't provided.
        if in_progress is None:
            try:
                _lt = (t_last.tz_convert("UTC") if getattr(t_last, "tzinfo", None)
                       else t_last.tz_localize("UTC"))
                _age_min = (pd.Timestamp.now(tz="UTC") - _lt).total_seconds() / 60.0
                in_progress = _age_min <= 60
            except Exception:  # noqa: BLE001
                in_progress = False
        if in_progress:
            if session_hours is not None:
                sess = session_hours * 3600.0
            else:
                # Session length from the open's UTC hour (yfinance returns
                # intraday timestamps in UTC): US cash opens ~13:30 UTC, Europe
                # ~07:00 UTC.
                try:
                    oh = (t0.tz_convert("UTC").hour if getattr(t0, "tzinfo", None) else t0.hour)
                except Exception:  # noqa: BLE001
                    oh = 8
                sess = (6.5 if oh >= 12 else 8.5) * 3600.0
            xs = _session_xs(ts, t0, pd.Timestamp(t0) + pd.Timedelta(seconds=sess), w) or []
        if not xs:
            xs = [i / (n - 1) * w for i in range(n)]

    lo = min(min(vals), baseline)
    hi = max(max(vals), baseline)
    rng = (hi - lo) or 1.0
    pad = rng * 0.16
    lo -= pad
    hi += pad
    rng = hi - lo

    def _yy(v: float) -> float:
        return h - (v - lo) / rng * h

    yb = _yy(baseline)
    line = " ".join(f"{x:.1f},{_yy(v):.1f}" for x, v in zip(xs, vals))
    x_last = xs[-1]
    y_last = _yy(vals[-1])
    poly = f"{line} {x_last:.1f},{yb:.1f} {xs[0]:.1f},{yb:.1f}"
    _day_spark_uid += 1
    gid, rid = f"pg{_day_spark_uid}", f"pr{_day_spark_uid}"
    yb_c = max(0.0, min(yb, h))
    # Endpoint dot, colored by sign vs the previous close (baseline): anchors
    # the eye to the current level and its up/down direction for the day.
    dot_col = PALETTE["green"] if vals[-1] >= baseline else PALETTE["red"]
    dot = (f'<circle cx="{x_last:.1f}" cy="{y_last:.1f}" r="1.8" fill="{dot_col}"/>')
    return (
        f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;">'
        f'<defs>'
        f'<clipPath id="{gid}"><rect x="0" y="0" width="{w}" height="{yb_c:.1f}"/></clipPath>'
        f'<clipPath id="{rid}"><rect x="0" y="{yb_c:.1f}" width="{w}" height="{h - yb_c:.1f}"/></clipPath>'
        f'</defs>'
        f'<polygon points="{poly}" fill="{PALETTE["green"]}" fill-opacity="0.22" clip-path="url(#{gid})"/>'
        f'<polygon points="{poly}" fill="{PALETTE["red"]}" fill-opacity="0.22" clip-path="url(#{rid})"/>'
        f'<line x1="0" y1="{yb:.1f}" x2="{w}" y2="{yb:.1f}" stroke="{PALETTE["subtle"]}" '
        f'stroke-width="0.75" stroke-dasharray="2,2"/>'
        f'<polyline points="{line}" fill="none" stroke="{PALETTE["green"]}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{PALETTE["red"]}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#{rid})"/>'
        f'{dot}'
        f'</svg>'
    )

def _flat_dashed_spark(w: int = 62, h: int = 22, note: str = "") -> str:
    """Placeholder sparkline for instruments with no intraday trades: a single
    dashed horizontal line (the previous-close reference). Keeps the 1D cell
    the same height as the intraday rows so the pill stays aligned.

    No caption by default. An empty cell where every neighbour draws a path
    already says there is no intraday series, and a "no intraday" label under
    it was what made those rows a line taller than the rest of the table —
    seventeen of fifty-nine in the watchlist, so the column read as ragged.
    ``note`` is for the one caption the dash cannot imply: WHICH session the
    row's level and % belong to, when the venue has already moved on to a
    session this data does not cover ("Wed close").
    """
    y = h / 2.0
    return (
        f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;">'
        f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" stroke="{PALETTE["subtle"]}" '
        f'stroke-width="1" stroke-dasharray="3,2"/></svg>'
        + spark_note(note)
    )


def spark_note(text: str) -> str:
    """Caption under a 1D cell that carries no intraday series.

    The LABEL size, not the DATA size: the cell it sits in is 68px wide, and
    "no intraday" set at 10px needs 66px of it, so at the data size the caption
    wrapped onto a second line and took the row's height with it. ``nowrap``
    holds it on one line at any mono advance.
    """
    return (f'<div style="font-size:{TYPE_PX["label"]}px;font-weight:600;'
            f'line-height:1.25;white-space:nowrap;color:{PALETTE["subtle"]};'
            f'margin-top:1px;">{text}</div>' if text else "")

def _prev_session_label(m, fmt: str = "%d/%m") -> str:
    """Report-level 'previous session' date for the PREV. DAY tag and the 1D
    column header — the last completed trading day in the portfolio history
    (the close the non-live 1D moves are measured against). Empty when
    unavailable.

    ``fmt`` lets a caller ask for a longer form; both callers must read the
    same date, which is why this is a parameter rather than a second function.
    """
    ph = getattr(m, "portfolio_history", None)
    try:
        if ph is not None and len(ph) >= 1:
            # Run-owned clock: under --as_of "previous session" means the last
            # close before the effective date, not before the wall clock.
            from tarzan import runtime as _runtime

            today = pd.Timestamp(_runtime.today()).normalize()
            past = [d for d in ph.index if pd.Timestamp(d).normalize() < today]
            d = pd.Timestamp(past[-1]) if past else pd.Timestamp(ph.index[-1])
            return d.strftime(fmt)
    except Exception:  # noqa: BLE001
        pass
    return ""


def day_column_label(m, *, live: bool) -> str:
    """Header for the session column: ``Intraday`` or ``1D``.

    One word, because the header is the only place the basis needs saying. While
    a session is open the column carries the live move against the previous
    close; once every session has closed it carries the close-to-close return.
    Every row shares that basis, so it used to be stated on each of them with a
    "● LIVE" badge -- seventeen repetitions of one fact, in the width the figure
    needed.

    Which close the closed-market figure is measured against is in the masthead
    stamp ("Mon, 27 Jul 2026 · close 24 Jul · market CLOSED"), so it is not lost
    by keeping this to a word.
    """
    return "Intraday" if live else "1D"



def bullet(actual: float, target: float, *, tol: float, w: int = 96,
           h: int = 18, scale_max: Optional[float] = None,
           ref: Optional[float] = None) -> str:
    """Stephen-Few bullet: the bar is the actual weight, the rule is the
    target, the pale band is the tolerance corridor.

    Answers "am I inside the band?" at a glance, which a drift figure alone
    does not: -2.0pp reads the same whether the corridor is 1pp or 5pp wide.
    ``ref`` marks a second, fainter reference on the same axis (used for 100%
    of invested capital under a notional total).
    """
    P = PALETTE
    smax = scale_max or (max(actual, target) * 1.25) or 1.0

    def X(v: float) -> float:
        return max(0.0, min(float(w), v / smax * w))

    inside = abs(actual - target) <= tol
    bar = (P["green"] if inside
           else (P["amber"] if abs(actual - target) <= 2 * tol else P["red"]))
    bh, by = 8, (h - 8) / 2
    band_lo, band_hi = X(max(0.0, target - tol)), X(target + tol)
    ref_mark = ""
    if ref is not None:
        ref_mark = (f'<line x1="{X(ref):.1f}" y1="{by - 3:.1f}" '
                    f'x2="{X(ref):.1f}" y2="{by + bh + 3:.1f}" '
                    f'stroke="{P["muted"]}" stroke-width="1.2"/>')
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<rect x="0" y="{by:.1f}" width="{w}" height="{bh}" '
        f'fill="{P["row_rule"]}" rx="2"/>'
        f'<rect x="{band_lo:.1f}" y="{by - 2:.1f}" '
        f'width="{max(1.0, band_hi - band_lo):.1f}" height="{bh + 4}" '
        f'fill="{P["col_rule"]}"/>'
        f'<rect x="0" y="{by:.1f}" width="{X(actual):.1f}" height="{bh}" '
        f'fill="{bar}" rx="2"/>'
        f'{ref_mark}'
        f'<line x1="{X(target):.1f}" y1="{by - 4:.1f}" '
        f'x2="{X(target):.1f}" y2="{by + bh + 4:.1f}" '
        f'stroke="{P["ink"]}" stroke-width="2"/>'
        f'</svg>')
