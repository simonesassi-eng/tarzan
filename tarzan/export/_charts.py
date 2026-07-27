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

# ── Palette ──────────────────────────────────────────────────────────────────
# Read from the leaf palette module rather than duplicated here. The previous
# local copies existed to avoid an import cycle (the newsletter package imports
# this module), but they meant chart axes could drift from the tables beside
# them whenever the palette changed in only one place.
from tarzan.export._format import eur_smart as _eur_smart
from tarzan.export._palette import PALETTE as _P

INK = _P["ink"]
MUTED = _P["muted"]
SUBTLE = _P["subtle"]
BORDER = _P["border"]
GREEN = _P["green"]
BENCH = _P["bench"]
PNL = _P["pnl"]


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
    """Axis tick with the minus SIGN, not a hyphen.

    Every other negative figure in the issue is written with U+2212; an axis
    labelled "-5%" beside a table cell reading "\u22125%" is two glyphs for one
    idea, and the hyphen is visibly shorter at 9px.
    """
    a = abs(v)
    txt = f"{a:.0f}" if abs(a - round(a)) < 1e-9 else f"{a:.1f}"
    sign = "\u2212" if v < 0 else ""
    return f"{sign}{txt}%"


def chart_pct_compact(series, dates, include_zero=True, w=256, h=150, fs=9,
                      date_fmt="%b %d", month_ticks=False, min_day_ticks=0,
                      end_gutter: int = 0) -> str:
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
    # ``end_gutter`` reserves room to the right of the plot for a label at the
    # end of each line, which replaces a legend underneath: a legend makes the
    # reader match a colour swatch to a line, while a label at the line's own end
    # needs no matching -- and it costs no vertical space, which across a
    # three-panel section is a whole row of height.
    ml, mr, mt, mb = 30, (8 + max(0, int(end_gutter))), 10, (32 if _rotate else 20)
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
        # Thin the LABELS to what the width can hold. Every month still gets its
        # gridline, but a rotated label needs ~30px of horizontal room, and eight
        # months across a 264px panel left the first two printed on top of each
        # other. The last boundary is always kept, so the axis still ends on the
        # current month.
        _room = max(1, int((pw) // 30))
        _step = max(1, -(-len(month_idx) // _room))
        _candidates = list(dict.fromkeys(month_idx[::_step] + [month_idx[-1]]))
        # Thinning by count is not enough: a window that opens mid-month puts its
        # first two boundaries days apart, and "Dec 25" printed over "Jan 26"
        # whatever the count allowed. Keep a candidate only when it is 30px clear
        # of the last one kept, and always keep the final boundary so the axis
        # ends on the current month.
        _labelled: set[int] = set()
        _last_x = None
        for k in _candidates:
            if _last_x is not None and (X(k) - _last_x) < 30 and k != _candidates[-1]:
                continue
            _labelled.add(k)
            _last_x = X(k)
        prev_year = None
        for j, k in enumerate(month_idx):
            _vgrid(k)
            if k not in _labelled:
                continue
            t = ts[k]
            label = (t.strftime("%b %y")
                     if (prev_year != t.year or prev_year is None)
                     else t.strftime("%b"))
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
    _label_ys: list[float] = []
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"])]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        # Thin solid lines so small variations read clearly. The benchmark is
        # distinguished by its grey colour, not a dash (dashes hide the very
        # wobble the reader is trying to compare); ``dash`` is still honoured if
        # a caller explicitly asks for it.
        dash = ' stroke-dasharray="5,4"' if s.get("dash") else ""
        out.append(f'<polyline points="{line}" fill="none" stroke="{s["color"]}" stroke-width="1.5"{dash} stroke-linejoin="round"/>')
        lx, ly = pts[-1]
        out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.4" '
                   f'fill="{s["color"]}" stroke="{_P["card_alt"]}" '
                   f'stroke-width="1"/>')
        label = s.get("end_label")
        if label and end_gutter:
            # Nudged apart when two lines end within a label's height of each
            # other, so a tight finish does not print one label over another.
            ey = ly + 3.2
            for taken in _label_ys:
                if abs(ey - taken) < 11:
                    ey = taken + 11
            _label_ys.append(ey)
            ey = min(h - 4, max(10.0, ey))
            out.append(f'<text x="{lx + 6:.1f}" y="{ey:.1f}" '
                       f'text-anchor="start" font-size="9" font-weight="700" '
                       f'fill="{s["color"]}">{label}</text>')
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


def waterfall(items, *, w: int = 580, h: int = 205, total_label: str = "Net",
              footnote: str | None = None) -> str:
    """Contribution waterfall: every bar starts where the previous one ended.

    ``items`` is a list of ``(label, value_pct)``. The final bar is the running
    total, so the caller must pass steps that genuinely add up to the number it
    wants that bar to show — a waterfall whose parts do not reconcile with its
    total is worse than two ranked lists, because it looks like it reconciles.

    Two ranked lists let the reader compare within each list but not across
    them, and never show how the parts make the whole; here the connectors do.
    """
    steps, run = [], 0.0
    for label, val in items:
        v = float(val or 0.0)
        steps.append((str(label), v, run, run + v))
        run += v
    if not steps:
        return ""
    total = run

    ML, MR, MT, MB = 40, 10, 16, 46
    PW, PH = w - ML - MR, h - MT - MB
    lo = min([0.0, total] + [min(s[2], s[3]) for s in steps])
    hi = max([0.0, total] + [max(s[2], s[3]) for s in steps])
    span = (hi - lo) or 1.0
    lo, hi, ticks = nice_ticks(lo - span * 0.12, hi + span * 0.12, 4)
    nb = len(steps) + 1
    slot = PW / nb
    bw = min(46.0, slot * 0.62)

    def Y(v: float) -> float:
        return MT + (1 - (v - lo) / ((hi - lo) or 1.0)) * PH

    out = [f'<svg width="100%" viewBox="0 0 {w} {h}" '
           f'preserveAspectRatio="xMidYMid meet" '
           f'xmlns="http://www.w3.org/2000/svg" style="display:block;'
           f'width:100%;background:{_P["card_alt"]};">']
    for t in ticks:
        if t < lo - 1e-9 or t > hi + 1e-9:
            continue
        y, zero = Y(t), abs(t) < 1e-9
        out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
                   f'stroke="{SUBTLE if zero else BORDER}" '
                   f'stroke-width="{1.2 if zero else 1}"/>')
        sign = "+" if t > 0 else ("−" if t < 0 else "")
        out.append(f'<text x="{ML - 6}" y="{y + 3:.1f}" text-anchor="end" '
                   f'font-size="9" fill="{SUBTLE}">{sign}{abs(t):.1f}%</text>')

    prev_x = None
    for i, (label, val, y0, y1) in enumerate(steps):
        cx = ML + slot * (i + 0.5)
        x = cx - bw / 2
        top, bot = Y(max(y0, y1)), Y(min(y0, y1))
        col = GREEN if val >= 0 else _P["red"]
        sign = "+" if val >= 0 else "−"
        out.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                   f'height="{max(1.5, bot - top):.1f}" fill="{col}" rx="1.5"/>')
        out.append(f'<text x="{cx:.1f}" y="{(top - 4) if val >= 0 else (bot + 11):.1f}" '
                   f'text-anchor="middle" font-size="9.5" font-weight="700" '
                   f'fill="{col}">{sign}{abs(val):.2f}%</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - 24}" text-anchor="middle" '
                   f'font-size="8.5" font-weight="700" fill="{MUTED}">{label}</text>')
        # Connector from the previous bar's end to this bar's start: the line
        # that makes it a waterfall rather than a row of floating bars.
        if prev_x is not None:
            out.append(f'<line x1="{prev_x:.1f}" y1="{Y(y0):.1f}" x2="{x:.1f}" '
                       f'y2="{Y(y0):.1f}" stroke="{SUBTLE}" stroke-width="1" '
                       f'stroke-dasharray="2,2"/>')
        prev_x = x + bw

    cx = ML + slot * (len(steps) + 0.5)
    x = cx - bw / 2
    top, bot = Y(max(0.0, total)), Y(min(0.0, total))
    accent = _P["accent"]
    out.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw:.1f}" '
               f'height="{max(1.5, bot - top):.1f}" fill="{accent}" rx="1.5"/>')
    out.append(f'<text x="{cx:.1f}" y="{top - 4:.1f}" text-anchor="middle" '
               f'font-size="10" font-weight="700" fill="{accent}">'
               f'{"+" if total >= 0 else "−"}{abs(total):.2f}%</text>')
    out.append(f'<text x="{cx:.1f}" y="{h - 24}" text-anchor="middle" '
               f'font-size="8.5" font-weight="700" fill="{accent}">{total_label}</text>')
    if footnote:
        out.append(f'<text x="{ML}" y="{h - 8}" font-size="9" '
                   f'fill="{SUBTLE}">{footnote}</text>')
    out.append("</svg>")
    return "".join(out)


def funding_flow(steps, *, w: int = 580, h: int = 94,
                 footnote: str | None = None) -> str:
    """The serialized-action funding proof drawn as an identity check.

    ``steps`` is a list of ``(label, value, kind)`` where ``kind`` is one of
    ``"neutral"``, ``"in"``, ``"out"`` or ``"total"``. A ``"total"`` step takes
    the running sum instead of its own value, so the last bar is the arithmetic
    result of the ones before it: if the terms do not close, the picture cannot
    pretend they do.
    """
    if not steps:
        return ""
    COLOUR = {"neutral": MUTED, "in": _P["accent"], "out": _P["red"],
              "total": GREEN}
    ML, MR, MT = 8, 8, 20
    PW = w - ML - MR
    slot = PW / len(steps)
    run = 0.0
    vals = []
    for label, val, kind in steps:
        if kind == "total":
            vals.append((str(label), run, COLOUR["total"], True))
        else:
            run += float(val or 0.0)
            vals.append((str(label), float(val or 0.0),
                         COLOUR.get(kind, MUTED), False))
    peak = max(abs(v) for _l, v, _c, _t in vals) or 1.0
    bh = 30

    out = [f'<svg width="100%" viewBox="0 0 {w} {h}" '
           f'preserveAspectRatio="xMidYMid meet" '
           f'xmlns="http://www.w3.org/2000/svg" style="display:block;'
           f'width:100%;background:{_P["card_alt"]};">']
    for i, (label, val, col, is_total) in enumerate(vals):
        cx = ML + slot * (i + 0.5)
        bw = min(74.0, slot * 0.78)
        # Floor the width so a genuinely small term (fees) still reads as a
        # labelled step rather than a sliver its own number cannot sit in.
        sw = max(38.0, abs(val) / peak * bw)
        x = cx - sw / 2
        out.append(f'<rect x="{x:.1f}" y="{MT}" width="{sw:.1f}" height="{bh}" '
                   f'fill="{col}" fill-opacity="{1.0 if is_total else 0.20}" '
                   f'stroke="{col}" stroke-width="{2 if is_total else 1}" rx="3"/>')
        out.append(f'<text x="{cx:.1f}" y="{MT + bh / 2 + 3.8:.1f}" '
                   f'text-anchor="middle" font-size="10.5" font-weight="700" '
                   f'fill="{_P["card"] if is_total else INK}" '
                   f'style="font-variant-numeric:tabular-nums;">'
                   f'{_eur_smart(abs(val))}</text>')
        out.append(f'<text x="{cx:.1f}" y="{MT - 7}" text-anchor="middle" '
                   f'font-size="9" fill="{MUTED}">{label}</text>')
        if i < len(vals) - 1:
            out.append(f'<text x="{cx + slot / 2:.1f}" y="{MT + bh / 2 + 4:.1f}" '
                       f'text-anchor="middle" font-size="11" '
                       f'fill="{SUBTLE}">\u2192</text>')
    if footnote:
        out.append(f'<text x="{ML}" y="{h - 6}" font-size="9.5" '
                   f'fill="{MUTED}">{footnote}</text>')
    out.append("</svg>")
    return "".join(out)


def band_gauge(value: float, *, good: float, warn: float,
               invert: bool = False, w: int = 92, h: int = 26) -> str:
    """A metric on its own rating scale: weak / fair / strong zones as the
    track, the value as a needle.

    Replaces three chips of threshold text per metric. The thresholds are the
    configured ones (``metric_ratings`` in constants.yaml, with its citations),
    so this draws a rating the project already declares rather than inventing
    one.

    ``invert`` marks a smaller-is-better metric, in which case the strong zone
    is on the LEFT — and the end captions swap with it, or the words would
    contradict the colours.
    """
    hi = max(abs(value) * 1.35, abs(good) * 1.6, abs(warn) * 1.6) or 1.0

    def X(v: float) -> float:
        return max(0.0, min(float(w), abs(v) / hi * w))

    green, amber, red = _P["green"], _P["amber"], _P["red"]
    if invert:
        zones = [(0.0, X(good), green), (X(good), X(warn), amber),
                 (X(warn), float(w), red)]
    else:
        zones = [(0.0, X(warn), red), (X(warn), X(good), amber),
                 (X(good), float(w), green)]
    ty, th = 8, 8
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'xmlns="http://www.w3.org/2000/svg" style="display:block;">']
    for x0, x1, colour in zones:
        out.append(f'<rect x="{x0:.1f}" y="{ty}" width="{max(0.0, x1 - x0):.1f}" '
                   f'height="{th}" fill="{colour}" fill-opacity="0.22"/>')
    vx = X(value)
    inzone = next((c for x0, x1, c in zones if x0 <= vx <= x1), MUTED)
    out.append(f'<line x1="{vx:.1f}" y1="{ty - 4}" x2="{vx:.1f}" '
               f'y2="{ty + th + 4}" stroke="{inzone}" stroke-width="2.6"/>')
    left_cap, right_cap = ("strong", "weak") if invert else ("weak", "strong")
    out.append(f'<text x="0" y="{h - 1}" font-size="8" fill="{SUBTLE}">'
               f'{left_cap}</text>')
    out.append(f'<text x="{w}" y="{h - 1}" text-anchor="end" font-size="8" '
               f'fill="{SUBTLE}">{right_cap}</text>')
    out.append("</svg>")
    return "".join(out)
