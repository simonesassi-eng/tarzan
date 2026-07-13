"""Email-safe SVG chart primitives for the newsletter Performance section.

Pure functions (axis-tick math + a compact multi-line % chart + an inline
legend) that take plain Python lists / pandas Timestamps, so they can be
unit-tested without rendering the whole template. Inline ``<svg>`` renders in
Gmail, Apple Mail and most modern clients; legacy Outlook desktop (Word
engine) does not render inline SVG, so the surrounding HTML (matrix numbers,
chip callouts, legends) carries the same story.
"""

from __future__ import annotations

import math

import pandas as pd

# ── Palette (mirrors tarzan.export.newsletter.PALETTE; kept local to avoid a
#    circular import, since newsletter.py imports this module). ──────────────
INK = "#1E293B"
MUTED = "#64748B"
SUBTLE = "#94A3B8"
BORDER = "#E5E7EF"
GREEN = "#15803D"
BENCH = "#94A3B8"   # benchmark grey
PNL = "#0EA5E9"     # P&L cyan


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


def chart_pct_compact(series, dates, include_zero=True, w=256, h=150, fs=9,
                      date_fmt="%b %d", month_ticks=False, min_day_ticks=0) -> str:
    """A compact multi-line % chart for side-by-side use, tuned so the axis
    labels stay legible at ~half width. ``series``: list of
    ``{values, color, dash?}``. ``date_fmt`` sets the x-axis tick format.

    X-axis density:
      * ``month_ticks=True`` — label EVERY month boundary (year shown when it
        changes / on the first tick), each with a faint vertical gridline.
      * ``min_day_ticks=N`` — place at least N evenly-spaced day labels
        (``date_fmt``), each with a faint vertical gridline. Use for the
        short (30-day) window.
      * neither — just first/last labels (legacy).
    Vertical gridlines are drawn light (BORDER) so they read as a grid without
    competing with the data lines."""
    # Dense-label modes tilt the x-axis labels so many of them don't collide;
    # that needs more room below the plot, so grow the canvas + bottom margin
    # (keeps the plot area itself the same height as the non-rotated charts).
    _rotate = bool(month_ticks or min_day_ticks)
    if _rotate:
        h = h + 12
    ml, mr, mt, mb = 30, 8, 10, (32 if _rotate else 20)
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

    # ── X-axis ticks + light vertical gridlines ──
    def _vgrid(k: int) -> None:
        x = X(k)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" '
                   f'stroke="{BORDER}" stroke-width="1"/>')

    def _xlabel(k: int, text: str) -> None:
        x = X(k)
        if _rotate:
            # Tilt ~35° and right-anchor at the tick so dense labels never
            # overlap. Anchor the text end at (x, baseline) and rotate about it.
            # A slightly smaller font buys horizontal room at ~12 ticks.
            y = mt + ph + 10
            rfs = max(7, fs - 1)
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="end" '
                       f'font-size="{rfs}" fill="{SUBTLE}" '
                       f'transform="rotate(-35 {x:.1f} {y:.1f})">{text}</text>')
        else:
            anc = "start" if k == 0 else ("end" if k == n - 1 else "middle")
            out.append(f'<text x="{x:.1f}" y="{h - 7}" text-anchor="{anc}" font-size="{fs}" '
                       f'fill="{SUBTLE}">{text}</text>')

    if month_ticks and n > 1:
        ts = [pd.Timestamp(d) for d in dates]
        # One tick at each month boundary — ALL of them (labelled + gridline).
        month_idx = [0] + [i for i in range(1, n)
                           if (ts[i].year, ts[i].month) != (ts[i - 1].year, ts[i - 1].month)]
        # Drop the leading start-of-history tick when it sits right on top of
        # the first month boundary (e.g. history starts Dec 23 → "Dec 25" and
        # "Jan 26" would overlap). Keep the boundary; it carries the year.
        if len(month_idx) > 1 and (month_idx[1] - month_idx[0]) < max(6, n // 20):
            month_idx = month_idx[1:]
        # Do NOT append a separate end anchor: the last month boundary already
        # labels the current month, and adding n-1 would repeat it ("Jul Jul").
        # Every month gets exactly one tick at its first in-window day.
        prev_year = None
        for j, k in enumerate(month_idx):
            _vgrid(k)
            t = ts[k]
            label = t.strftime("%b %y") if (prev_year != t.year or j == 0) else t.strftime("%b")
            prev_year = t.year
            _xlabel(k, label)
    elif min_day_ticks and n > 1:
        # At least ``min_day_ticks`` evenly-spaced day ticks (labelled + grid).
        count = min(min_day_ticks, n)
        idxs = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
        for k in idxs:
            _vgrid(k)
            _xlabel(k, pd.Timestamp(dates[k]).strftime(date_fmt))
    else:
        for k in sorted({0, n - 1}):
            _xlabel(k, pd.Timestamp(dates[k]).strftime(date_fmt))
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
        out.append(f'<span style="display:inline-block;margin:0 14px 2px 0;font-size:{size}px;'
                   f'font-weight:600;color:{INK};white-space:nowrap;">{mark}{label}</span>')
    return '<div style="margin-top:6px;line-height:1.35;">' + "".join(out) + "</div>"
