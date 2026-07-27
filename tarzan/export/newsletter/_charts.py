"""Inline-SVG chart/spark builders for the newsletter.

Holds the two SVG clipPath id counters (_day_spark_uid, _dual_uid) and the
reset_spark_uids() the orchestrator calls at the start of each render so ids
depend only on how many charts a render draws, not process history.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tarzan.export._format import eur_smart as _eur_smart
from tarzan.export.newsletter._constants import PALETTE

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

def _hero_value_chart(values, pct, dates, flows, w: int = 544, h: int = 196) -> str:
    """Dual-axis hero chart with the same baseline semantics as Markets.

    Portfolio value is green above the window-start baseline and red below it;
    Unrealized P&L remains the neutral dashed secondary series. Cash-flow
    triangles stay attached to the value line.
    """
    global _dual_uid
    _dual_uid += 1
    u = _dual_uid
    P = PALETTE
    n = len(values)
    if n < 2 or not pct or len(pct) != n:
        return ""
    ML, MR, MT, MB = 52, 48, 12, 26
    PW, PH = w - ML - MR, h - MT - MB
    base = values[0]
    from tarzan.export import _charts as _ch
    vlo, vhi, vticks = _ch.nice_ticks(
        min(min(values), base), max(max(values), base), 4
    )
    plo, phi, pticks = _ch.nice_ticks(min(pct), max(pct), 4)

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
            f'font-size="9" fill="{P["subtle"]}">{_ch.fmt_eur_tick(t)}</text>'
        )
    for t in pticks:
        if t < plo - 1e-9 or t > phi + 1e-9:
            continue
        grid += (
            f'<text x="{ML + PW + 6}" y="{Yp(t) + 3:.1f}" '
            f'text-anchor="start" font-size="9" fill="{P["muted"]}">'
            f'{_ch.fmt_pct_tick(t)}</text>'
        )
    xlab = ""
    for k in sorted({0, n // 3, 2 * n // 3, n - 1}):
        x = X(k)
        anc = "start" if k == 0 else "end" if k == n - 1 else "middle"
        xlab += (
            f'<text x="{x:.1f}" y="{h - 8}" text-anchor="{anc}" font-size="9" '
            f'fill="{P["subtle"]}">{pd.Timestamp(dates[k]).strftime("%b %d")}</text>'
        )

    vline = " ".join(f"{X(i):.1f},{Yv(v):.1f}" for i, v in enumerate(values))
    baseline_y = Yv(base)
    value_poly = (
        f"{vline} {X(n - 1):.1f},{baseline_y:.1f} "
        f"{X(0):.1f},{baseline_y:.1f}"
    )
    pline = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(pct))
    baseline_clip_y = max(MT, min(baseline_y, MT + PH))

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
    # running off the plot.
    endpoint = (
        f'<circle cx="{X(n - 1):.1f}" cy="{Yv(values[-1]):.1f}" r="3.4" '
        f'fill="{P["green"] if values[-1] >= base else P["red"]}" '
        f'stroke="{P["card_alt"]}" stroke-width="1.8"/>'
    )
    # Two labels inside the plot instead of a legend underneath. The legend
    # spent a line naming the axes; naming the baseline where it is drawn and
    # the right axis where it is read says the same thing in the place the
    # reader is already looking.
    labels = (
        f'<text x="{ML + 5}" y="{max(9.0, baseline_y - 5):.1f}" '
        f'text-anchor="start" font-size="8.5" font-weight="700" '
        f'fill="{P["muted"]}">window open {_eur_smart(base)}</text>'
        f'<text x="{ML + PW + 6}" y="{h - 6}" text-anchor="start" '
        f'font-size="8.5" font-weight="700" fill="{P["muted"]}">unreal.</text>'
    )

    return (
        f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;'
        f'font-family:-apple-system,Helvetica,Arial,sans-serif;">'
        f'<defs><clipPath id="dg{u}"><rect x="0" y="0" width="{w}" '
        f'height="{baseline_clip_y:.1f}"/></clipPath>'
        f'<clipPath id="dr{u}"><rect x="0" y="{baseline_clip_y:.1f}" width="{w}" '
        f'height="{h - baseline_clip_y:.1f}"/></clipPath></defs>'
        + grid
        + f'<line x1="{ML}" y1="{baseline_y:.1f}" x2="{ML + PW}" '
        f'y2="{baseline_y:.1f}" stroke="{P["subtle"]}" stroke-width="0.8" '
        f'stroke-dasharray="3,3"/>'
        + f'<polygon points="{value_poly}" fill="{P["green"]}" fill-opacity="0.16" '
        f'clip-path="url(#dg{u})"/>'
        + f'<polygon points="{value_poly}" fill="{P["red"]}" fill-opacity="0.16" '
        f'clip-path="url(#dr{u})"/>'
        + f'<polyline points="{vline}" fill="none" stroke="{P["green"]}" '
        f'stroke-width="2.6" stroke-linejoin="round" clip-path="url(#dg{u})"/>'
        + f'<polyline points="{vline}" fill="none" stroke="{P["red"]}" '
        f'stroke-width="2.6" stroke-linejoin="round" clip-path="url(#dr{u})"/>'
        + f'<polyline points="{pline}" fill="none" stroke="{P["muted"]}" '
        f'stroke-width="1.8" stroke-dasharray="4,3" stroke-linejoin="round"/>'
        + marks + endpoint + labels + xlab + "</svg>"
    )


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
                  f'font-size:10px;font-weight:700;color:{fg};'
                  f'white-space:nowrap;font-variant-numeric:tabular-nums;">'
                  f'{pd.Timestamp(d).strftime("%b %d")} '
                  f'{_eur_smart(v, signed=True)}</span>')
    return (f'<div style="margin-top:9px;font-size:9px;font-weight:700;'
            f'letter-spacing:0.08em;color:{P["muted"]};'
            f'text-transform:uppercase;">Cash flows in the window</div>'
            f'<div style="margin-top:3px;">{items}</div>')

def _timeline_vals(series: Optional[list], key: str) -> Optional[list[float]]:
    """Extract the per-bucket weights for one category from an allocation
    timeline series. Returns None when the category never carried weight
    (so the caller can skip drawing an empty sparkline)."""
    if not series:
        return None
    vals = [float(pt.get(key, 0.0)) for pt in series]
    return vals if any(v > 0 for v in vals) else None

def _intraday_spark(intra: "pd.Series", baseline: float,
                    w: int = 62, h: int = 22,
                    in_progress: Optional[bool] = None) -> str:
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
    None, it is inferred from bar recency (FX/futures fallback)."""
    global _day_spark_uid
    ts = list(intra.index)
    vals = [float(x) for x in intra.values]
    n = len(vals)
    if n < 2:
        return ""
    t0, t_last = ts[0], ts[-1]
    # A completed session (e.g. a US index in the European morning, or any
    # market viewed after its close) fills the full width. Only the session
    # trading *now* grows from the left, so early in that market's day the
    # line covers just the elapsed portion. Prefer the caller's exchange-hours
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
        # Session length from the open's UTC hour (yfinance returns intraday
        # timestamps in UTC): US cash opens ~13:30 UTC, Europe ~07:00 UTC.
        try:
            oh = (t0.tz_convert("UTC").hour if getattr(t0, "tzinfo", None) else t0.hour)
        except Exception:  # noqa: BLE001
            oh = 8
        sess = (6.5 if oh >= 12 else 8.5) * 3600.0

        def _xpos(t) -> float:
            try:
                return max(0.0, min(1.0, (t - t0).total_seconds() / sess)) * w
            except Exception:  # noqa: BLE001
                return 0.0
        xs = [_xpos(t) for t in ts]
    else:
        xs = [i / (n - 1) * w for i in range(n)]

    lo = min(min(vals), baseline)
    hi = max(max(vals), baseline)
    span = (hi - lo) or 1.0
    pad = span * 0.16
    lo -= pad
    hi += pad
    span = hi - lo

    def _yy(v: float) -> float:
        return h - (v - lo) / span * h

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

def _flat_dashed_spark(w: int = 62, h: int = 22) -> str:
    """Placeholder sparkline for instruments with no intraday trades: a single
    dashed horizontal line (the previous-close reference). Keeps the 1D cell
    the same height as the intraday rows so the pill stays aligned, while
    signalling 'no intraday, this is the previous-day change'."""
    y = h / 2.0
    return (
        f'<svg width="100%" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;">'
        f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" stroke="{PALETTE["subtle"]}" '
        f'stroke-width="1" stroke-dasharray="3,2"/></svg>'
    )

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
            today = pd.Timestamp.now().normalize()
            past = [d for d in ph.index if pd.Timestamp(d).normalize() < today]
            d = pd.Timestamp(past[-1]) if past else pd.Timestamp(ph.index[-1])
            return d.strftime(fmt)
    except Exception:  # noqa: BLE001
        pass
    return ""


def day_column_label(m, *, live: bool) -> str:
    """Header for the 1D column, naming what the column is measured against.

    "1D / Intraday" read as two columns. There is only one: while a session is
    open it carries the live move against the previous close, and once every
    session has closed it carries the close-to-close return. The header says
    which of the two the reader is looking at, and against which close.
    """
    if live:
        return "1D \u00b7 intraday"
    when = _prev_session_label(m, "%d %b")
    return f"1D \u00b7 close {when}" if when else "1D"



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
