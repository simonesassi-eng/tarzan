"""Email-safe SVG chart builders for the newsletter Performance section.

These are pure functions ported from the approved prototype
(``prototypes/newsletter/render_perf_charts_v3.py``): time on the X axis,
magnitude on the Y axis, with labelled axes (value €/% ticks at round
numbers, dated X-axis) and optional deposit/withdrawal markers.

Approach: inline ``<svg>`` for the curve, gridlines and axis labels so the
labels align perfectly with the data. This renders in Gmail, Apple Mail and
most modern clients. NOTE: legacy Outlook desktop (Word engine) does not
render inline SVG — the surrounding HTML (the matrix numbers, the chip
callouts and the legends) carries the same story, and production should ship
a PNG fallback for Outlook before this becomes the primary surface.

All builders are deterministic and take plain Python lists / pandas
Timestamps, so they can be unit-tested without rendering the whole template.
"""

from __future__ import annotations

import math

import pandas as pd

# ── Palette (mirrors tarzan.export.newsletter.PALETTE; kept local to avoid a
#    circular import, since newsletter.py imports this module). ──────────────
INK = "#1E293B"
MUTED = "#64748B"
SUBTLE = "#94A3B8"
ACCENT = "#5B5BD6"
GREEN = "#15803D"
RED = "#DC2626"
AMBER = "#D97706"
BORDER = "#E5E7EF"
BENCH = "#94A3B8"   # benchmark grey
PNL = "#0EA5E9"     # P&L cyan
FLOW = "#7C3AED"    # deposit/withdrawal marker violet

# ── Plot geometry (one shared coordinate system so axes line up with curves) ─
W, H = 520, 210
ML, MR, MT, MB = 48, 12, 12, 26          # margins: left (Y labels), right, top, bottom (X labels)
PW, PH = W - ML - MR, H - MT - MB        # plot area


# ── axis helpers ─────────────────────────────────────────────────────────────

def nice_ticks(lo: float, hi: float, n: int = 4) -> tuple[float, float, list[float]]:
    """Round-number axis ticks (1/2/5 × 10ⁿ steps).

    Returns ``(nice_lo, nice_hi, [ticks])`` so the axis spans clean bounds and
    labels read 6% / €232k rather than 5.96% / €231.4k.
    """
    if hi <= lo:
        hi = lo + 1.0
    rng = hi - lo
    raw = rng / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    norm = raw / mag
    step = mag * (1 if norm < 1.5 else 2 if norm < 3 else 5 if norm < 7 else 10)
    lo_n = math.floor(lo / step) * step
    hi_n = math.ceil(hi / step) * step
    ticks = []
    v = lo_n
    while v <= hi_n + step * 1e-6:
        ticks.append(round(v, 10))
        v += step
    return lo_n, hi_n, ticks


def fmt_eur_tick(v: float) -> str:
    a = abs(v)
    s = "−" if v < 0 else ""
    return f"{s}€{a / 1000:.0f}k" if a >= 1000 else f"{s}€{a:.0f}"


def fmt_pct_tick(v: float) -> str:
    txt = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    return f"{txt}%"


def _x(i: int, n: int) -> float:
    return ML + (i / (n - 1) * PW if n > 1 else 0)


def _y(v: float, vmin: float, vmax: float) -> float:
    span = (vmax - vmin) or 1.0
    return MT + (1 - (v - vmin) / span) * PH


def _svg_head() -> str:
    return (f'<svg width="100%" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;max-width:100%;height:auto;'
            f'font-family:-apple-system,Helvetica,Arial,sans-serif;">')


def _grid_and_axes(vmin, vmax, ticks, dates, fmt, baseline=None) -> str:
    """Horizontal gridlines + Y-axis value labels (at the given nice ticks)
    + a 4-point dated X-axis."""
    parts = []
    for t in ticks:
        if t < vmin - 1e-9 or t > vmax + 1e-9:
            continue
        y = _y(t, vmin, vmax)
        parts.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        parts.append(f'<text x="{ML - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="10" fill="{SUBTLE}">{fmt(t)}</text>')
    if baseline is not None and vmin < baseline < vmax:
        yb = _y(baseline, vmin, vmax)
        parts.append(f'<line x1="{ML}" y1="{yb:.1f}" x2="{ML + PW}" y2="{yb:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2,3"/>')
    n = len(dates)
    xi = sorted({0, n // 3, 2 * n // 3, n - 1})
    for k in xi:
        x = _x(k, n)
        anchor = "start" if k == 0 else "end" if k == n - 1 else "middle"
        parts.append(f'<text x="{x:.1f}" y="{H - 8}" text-anchor="{anchor}" font-size="10" fill="{SUBTLE}">'
                     f'{pd.Timestamp(dates[k]).strftime("%b %d")}</text>')
    return "".join(parts)


def _marks(flows, dates) -> str:
    """Vertical dashed lines + triangles at deposit (▲) / withdrawal (▼) dates.
    ``flows`` is a list of ``(date, eur)`` with +ve = deposit, −ve = withdrawal."""
    if not flows:
        return ""
    n = len(dates)
    xmap = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    parts = []
    for d, v in flows:
        i = xmap.get(pd.Timestamp(d).normalize())
        if i is None:
            i = min(range(n), key=lambda k: abs((pd.Timestamp(dates[k]) - pd.Timestamp(d)).days))
        x = _x(i, n)
        parts.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT + PH}" stroke="{FLOW}" stroke-width="1" stroke-dasharray="2,3" opacity="0.7"/>')
        if v >= 0:
            parts.append(f'<polygon points="{x - 4:.1f},{MT + 8} {x + 4:.1f},{MT + 8} {x:.1f},{MT}" fill="{FLOW}"/>')
        else:
            parts.append(f'<polygon points="{x - 4:.1f},{MT} {x + 4:.1f},{MT} {x:.1f},{MT + 8}" fill="{FLOW}"/>')
    return "".join(parts)


# ── chart builders ─────────────────────────────────────────────────────────────

def chart_eur(values, dates, flows=None, color=ACCENT) -> str:
    """Area+line chart of an absolute € series with round € ticks and (optional)
    deposit/withdrawal markers."""
    vmin, vmax, ticks = nice_ticks(min(values), max(values), 4)
    n = len(values)
    pts = [(_x(i, n), _y(v, vmin, vmax)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{ML},{MT + PH:.1f} " + line + f" {ML + PW},{MT + PH:.1f}"
    lx, ly = pts[-1]
    return (_svg_head()
            + _grid_and_axes(vmin, vmax, ticks, dates, fmt_eur_tick)
            + (_marks(flows, dates) if flows else "")
            + f'<polygon points="{area}" fill="{color}" fill-opacity="0.13"/>'
            + f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
            + f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{color}" stroke="#fff" stroke-width="1.5"/>'
            + "</svg>")


def chart_pct(series, dates, flows=None, include_zero=True) -> str:
    """Multi-line % chart on a shared, round-tick axis.

    ``series``: list of ``{values, color, dash?, fill?, label?}``. When
    ``include_zero`` the 0% level is forced into the range and a dashed 0
    baseline is drawn (for rebased returns); otherwise the axis fits the data.
    """
    allv = [v for s in series for v in s["values"]]
    dlo, dhi = min(allv), max(allv)
    if include_zero:
        dlo = min(dlo, 0.0)
        dhi = max(dhi, 0.0)
    vmin, vmax, ticks = nice_ticks(dlo, dhi, 4)
    n = len(dates)
    parts = [_svg_head(), _grid_and_axes(vmin, vmax, ticks, dates, fmt_pct_tick,
                                         baseline=0.0 if include_zero else None)]
    if flows:
        parts.append(_marks(flows, dates))
    for s in series:
        pts = [(_x(i, n), _y(v, vmin, vmax)) for i, v in enumerate(s["values"])]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5,4"' if s.get("dash") else ""
        if s.get("fill"):
            parts.append(f'<polygon points="{ML},{MT + PH:.1f} {line} {ML + PW},{MT + PH:.1f}" fill="{s["color"]}" fill-opacity="0.08"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{s["color"]}" stroke-width="2.5"{dash} stroke-linejoin="round"/>')
        lx, ly = pts[-1]
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{s["color"]}" stroke="#fff" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_pct_compact(series, dates, include_zero=True, w=256, h=150, fs=9) -> str:
    """A smaller multi-line % chart for side-by-side use (same visual language
    as :func:`chart_pct`, tuned so the axis labels stay legible at ~half
    width). ``series``: list of ``{values, color, dash?}``."""
    ml, mr, mt, mb = 30, 8, 10, 20
    pw, ph = w - ml - mr, h - mt - mb
    allv = [v for s in series for v in s["values"]]
    dlo, dhi = min(allv), max(allv)
    if include_zero:
        dlo, dhi = min(dlo, 0.0), max(dhi, 0.0)
    vmin, vmax, ticks = nice_ticks(dlo, dhi, 4)
    n = len(dates)

    def X(i):
        return ml + (i / (n - 1) * pw if n > 1 else 0)

    def Y(v):
        return mt + (1 - (v - vmin) / ((vmax - vmin) or 1)) * ph

    out = [f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
           f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;'
           f'font-family:-apple-system,Helvetica,Arial,sans-serif;">']
    for t in ticks:
        if t < vmin - 1e-9 or t > vmax + 1e-9:
            continue
        y = Y(t)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        out.append(f'<text x="{ml - 5}" y="{y + 3:.1f}" text-anchor="end" font-size="{fs}" fill="{SUBTLE}">{fmt_pct_tick(t)}</text>')
    if include_zero and vmin < 0 < vmax:
        y0 = Y(0.0)
        out.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + pw}" y2="{y0:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2,3"/>')
    for k in sorted({0, n - 1}):
        x = X(k)
        anc = "start" if k == 0 else "end"
        out.append(f'<text x="{x:.1f}" y="{h - 7}" text-anchor="{anc}" font-size="{fs}" fill="{SUBTLE}">'
                   f'{pd.Timestamp(dates[k]).strftime("%b %d")}</text>')
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"])]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5,4"' if s.get("dash") else ""
        out.append(f'<polyline points="{line}" fill="none" stroke="{s["color"]}" stroke-width="2.2"{dash} stroke-linejoin="round"/>')
        lx, ly = pts[-1]
        out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="{s["color"]}" stroke="#fff" stroke-width="1.3"/>')
    out.append("</svg>")
    return "".join(out)


# ── presentational HTML atoms (legend + flow chips) ──────────────────────────

def legend(items, size: int = 11) -> str:
    """Inline legend. ``items``: list of ``(label, color, is_dashed)``."""
    out = []
    for label, color, dash in items:
        if dash:
            mark = (f'<span style="display:inline-block;width:24px;height:0;'
                    f'border-top:3px dashed {color};vertical-align:middle;margin-right:6px;"></span>')
        else:
            mark = (f'<span style="display:inline-block;width:24px;height:5px;'
                    f'background:{color};border-radius:3px;vertical-align:middle;margin-right:6px;"></span>')
        out.append(f'<span style="margin-right:14px;font-size:{size}px;font-weight:600;color:{INK};white-space:nowrap;">{mark}{label}</span>')
    return '<div style="margin-top:8px;line-height:1.9;">' + "".join(out) + "</div>"


def flow_chips(flows, eur_fmt) -> str:
    """Deposit/withdrawal callout chips below the value chart.

    ``flows``: list of ``(date, eur)`` (+ deposit / − withdrawal).
    ``eur_fmt``: a signed-euro formatter, e.g. ``lambda v: _eur_smart(v, signed=True)``.
    """
    if not flows:
        return ""
    chips = "".join(
        f'<span style="display:inline-block;margin:3px 6px 0 0;padding:3px 8px;border-radius:999px;background:#fff;'
        f'border:1px solid {BORDER};font-size:11px;white-space:nowrap;">'
        f'<span style="color:{FLOW};font-weight:700;">{"▲" if v >= 0 else "▼"}</span> '
        f'{pd.Timestamp(d).strftime("%b %d")} · '
        f'<span style="color:{GREEN if v >= 0 else RED};font-weight:700;">'
        f'{"Deposit" if v >= 0 else "Withdrawal"} {eur_fmt(v)}</span></span>'
        for d, v in flows)
    return (f'<div style="margin-top:8px;font-size:10px;font-weight:700;color:{MUTED};text-transform:uppercase;'
            f'letter-spacing:0.04em;">Cash flows '
            f'<span style="color:{FLOW};">▲ in</span> / <span style="color:{FLOW};">▼ out</span></div>'
            f'<div style="margin-top:4px;">{chips}</div>')
