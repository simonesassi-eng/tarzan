"""Generate the portfolio digest newsletter (HTML email).

This module renders an email-safe HTML newsletter from a PortfolioMetrics
object, using a Jinja2 template. The output mirrors the look-and-feel of
the Excel dashboard but is optimised for inbox consumption: 600px wide,
table-based layout, inline CSS, no JavaScript, no external resources.

The newsletter has the following structure:
    1. Header (brand + date + issue)
    2. Hero (total value + chip + KPI grid)
    3. Smart insights (3 takeaways: action, risk, win)
    4. Movers this week (best & worst by 1W return)
    5. Allocation by asset class (with stacked bar + per-class rows)
    6. Geographic exposure (equity, with target & ACWI ticks)
    7. Holdings (grouped by class, sorted as per Excel)
    8. Performance (returns by asset class & role, with 1D sparkline)
    9. Risk profile (chips + vs S&P 500 + vs MSCI ACWI)
   10. Suggested action (Optimizer)
   11. Return contribution (winners / laggards)
   12. CTA + footer

Public entry point: ``generate_newsletter(metrics, config, output_dir)``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tarzan import runtime as _runtime
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.taxonomy import (
    ORDER_NEWSLETTER as _ORDER_NEWSLETTER,
    ORDER_PERF as _ORDER_PERF,
)
from tarzan.export._format import (
    ASSET_CLASS_BG,
    ASSET_CLASS_COLORS,
    GEO_COLORS as _GEO_COLORS,
    css,
    eur_smart as _eur_smart,
    short_instrument_name,
)
from tarzan.export import _charts
# Pure performance/return series helpers extracted from this module. Re-imported
# under their original names so every call-site here (and the public
# ``market_snapshot`` symbol) is unchanged.
from tarzan.export._perf_series import (  # noqa: F401  (re-exported)
    _flow_list,
    _geo_benchmark_series,
    _norm_series,
    _perf_full_series,
    _perf_level_series,
    _perf_window,
    _window_money_pnl,
    _window_twror,
    market_snapshot,
)

logger = logging.getLogger(__name__)


# ── Palette (asset/geo colors bind to the shared tarzan.export._format source) ─

PALETTE = {
    "accent": "#5B5BD6",
    "ink": "#1E293B",
    "muted": "#64748B",
    "subtle": "#94A3B8",
    "page": "#F1F2F8",
    "card_alt": "#F8FAFF",
    "border": "#E5E7EF",
    "green": "#15803D",
    "amber": "#D97706",
    "red": "#DC2626",
    "green_bg": "#DCFCE7",
    "green_border": "#BBF7D0",
    "amber_bg": "#FFF7ED",
    "amber_border": "#FED7AA",
    "red_bg": "#FEE2E2",
    "red_border": "#FECACA",
    "accent_bg": "#EEF2FF",
    "gold_bg": "#FEF3C7",
    "fi_bg": "#FEF3C7",
}

ASSET_COLORS = {k: css(v) for k, v in ASSET_CLASS_COLORS.items()}
ASSET_BG = {k: css(v) for k, v in ASSET_CLASS_BG.items()}
GEO_COLORS = {k: css(v) for k, v in _GEO_COLORS.items()}

# Markets-strip region accent colours (left border + legend). Chosen from
# six clearly distinct hue families so no two regions look alike — in
# particular Asia (pink) and Commodities (green) no longer collide.
MARKET_REGION_COLORS = {
    "US": "#2563EB",           # blue
    "Europe": "#D97706",       # amber
    "Asia": "#DB2777",         # pink
    "Crypto": "#7C3AED",       # purple
    "Commodities": "#15803D",  # green
    "Currencies": "#64748B",   # slate
    "Indices": "#64748B",      # slate (offline fallback bucket)
}

# Asset class display order in the newsletter Holdings section.
# Cash is shown after Gold so the invested asset classes flow visually
# from highest-risk equity down to commodities/crypto/alternative; cash
# is reported as a separate accounting entity (no "% of portfolio" so
# it does not appear to compete with invested classes). Any asset class
# not listed here is appended (never silently dropped from the report).
_NEWSLETTER_CLASS_ORDER = list(_ORDER_NEWSLETTER)
_extra_classes = [c for c in ASSET_CLASS_COLORS if c not in _NEWSLETTER_CLASS_ORDER]
ASSET_CLASS_ORDER = _NEWSLETTER_CLASS_ORDER + sorted(_extra_classes)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eur(amount: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a number as a localised EUR amount: €1,234.56 / +€1,234.56."""
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "—"
    fmt = f",.{decimals}f"
    formatted = f"€{abs(amount):{fmt}}"
    if signed:
        sign = "+" if amount >= 0 else "−"
        return f"{sign}{formatted}"
    if amount < 0:
        return f"−{formatted}"
    return formatted


def _pct(value: Optional[float], decimals: int = 2, signed: bool = False) -> str:
    """Format a percentage. Already in pp (e.g. 8.59 means 8.59%)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if signed:
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value):.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def _pct_compact(value: Optional[float], signed: bool = True) -> str:
    """Percentage with width-aware precision for the dense returns grids.

    The 8-column returns tables (snapshot + performance) must fit eight
    values inside a 600px email. Two decimals are fine for normal
    returns, but three-digit values like ``+126.17%`` overflow the
    fixed cell width. So we taper precision by magnitude:

        |v| < 100   → 2 decimals   (+8.59%, −1.62%)
        |v| < 1000  → 1 decimal    (+126.2%)
        |v| >= 1000 → 0 decimals   (+1234%)

    This trims width exactly where it's needed without losing
    meaningful precision (a few basis points on a >100% multi-year
    return are noise).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    v = float(value)
    av = abs(v)
    decimals = 2 if av < 100 else (1 if av < 1000 else 0)
    if signed:
        sign = "+" if v >= 0 else "−"
        return f"{sign}{av:.{decimals}f}%"
    return f"{v:.{decimals}f}%"


def _pct_smart(value: Optional[float], max_decimals: int = 1, signed: bool = False) -> str:
    """Format a percentage with adaptive precision: drop the decimal
    digits when the value is already integer (saves horizontal space).

    Example with ``max_decimals=1``:
      70.0  → "70%"
      71.7  → "71.7%"
      −1.6  → "−1.6%" (or "+1.7%" with signed=True)
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    rounded = round(float(value), max_decimals)
    is_integer = abs(rounded - round(rounded)) < 10 ** (-(max_decimals + 1))
    decimals = 0 if is_integer else max_decimals
    if signed:
        sign = "+" if value >= 0 else "−"
        return f"{sign}{abs(value):.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def _signed_pp(value: Optional[float], decimals: int = 1) -> str:
    """Format a signed delta in percentage points (no % sign)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"


def _display_ticker(symbol: Optional[str]) -> Optional[str]:
    """Short, human-facing ticker for an inline pin: strip the exchange
    suffix (``XDEM.MI`` → ``XDEM``, the same ``split('.')[0]`` convention
    used by :func:`_clean_ticker`) and the index caret (``^GSPC`` →
    ``GSPC``). Returns None when there is nothing worth showing (empty, the
    synthetic PORTFOLIO ticker, or a blended-mix pseudo-ticker) so the
    caller can skip the pin entirely.

    Shared by the Holdings, Returns-vs-benchmarks and Historical-risk
    sections so the ticker is derived one way everywhere."""
    if not symbol:
        return None
    t = str(symbol).strip()
    if not t or t.upper() in ("PORTFOLIO", "—", "NAN"):
        return None
    if t.startswith("^"):
        t = t[1:]
    t = t.split(".")[0]  # strip exchange suffix (XDEM.MI → XDEM)
    # Blended-mix pseudo-tickers (e.g. "60/40 ACWI+Bond") carry no clean
    # symbol — skip the pin rather than show something confusing.
    if not t or "/" in t or " " in t:
        return None
    return t


def _semaphore(delta: Optional[float], tolerance: float) -> str:
    """Return 'green' / 'amber' / 'red' based on |delta| vs tolerance."""
    if delta is None or (isinstance(delta, float) and pd.isna(delta)):
        return "muted"
    abs_d = abs(delta)
    if abs_d <= tolerance:
        return "green"
    if abs_d <= 2 * tolerance:
        return "amber"
    return "red"


def _semaphore_color(sema: str) -> str:
    return {"green": PALETTE["green"], "amber": PALETTE["amber"],
            "red": PALETTE["red"], "muted": PALETTE["muted"]}.get(sema, PALETTE["ink"])


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


_day_spark_uid = 0


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


_dual_uid = 0


def _hero_value_chart(values, pct, dates, flows, w: int = 544, h: int = 180) -> str:
    """Dual-axis hero chart: portfolio value (€, left axis) as a two-tone
    green/red line (green above the window-start baseline, red below) and the
    Unrealized PnL % (right axis) as a grey dashed line, with cash-flow
    triangles on the value line (▲ deposit green / ▼ withdrawal grey)."""
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
    vlo, vhi, vticks = _ch.nice_ticks(min(min(values), base), max(max(values), base), 4)
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
        grid += (f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" stroke="{P["border"]}" stroke-width="1"/>'
                 f'<text x="{ML - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="{P["subtle"]}">{_ch.fmt_eur_tick(t)}</text>')
    for t in pticks:
        if t < plo - 1e-9 or t > phi + 1e-9:
            continue
        grid += f'<text x="{ML + PW + 6}" y="{Yp(t) + 3:.1f}" text-anchor="start" font-size="9" fill="{P["muted"]}">{_ch.fmt_pct_tick(t)}</text>'
    xlab = ""
    for k in sorted({0, n // 3, 2 * n // 3, n - 1}):
        x = X(k)
        anc = "start" if k == 0 else "end" if k == n - 1 else "middle"
        xlab += (f'<text x="{x:.1f}" y="{h - 8}" text-anchor="{anc}" font-size="9" '
                 f'fill="{P["subtle"]}">{pd.Timestamp(dates[k]).strftime("%b %d")}</text>')

    vline = " ".join(f"{X(i):.1f},{Yv(v):.1f}" for i, v in enumerate(values))
    yb = Yv(base)
    poly = f"{vline} {X(n - 1):.1f},{yb:.1f} {X(0):.1f},{yb:.1f}"
    pline = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(pct))
    yb_c = max(MT, min(yb, MT + PH))

    marks = ""
    if flows:
        xmap = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
        for d, v in flows:
            i = xmap.get(pd.Timestamp(d).normalize())
            if i is None:
                continue
            x, y = X(i), Yv(values[i])
            col = P["green"] if v >= 0 else P["muted"]
            if v >= 0:
                marks += f'<polygon points="{x:.1f},{y-5:.1f} {x-4:.1f},{y+3:.1f} {x+4:.1f},{y+3:.1f}" fill="{col}" stroke="#fff" stroke-width="0.8"/>'
            else:
                marks += f'<polygon points="{x:.1f},{y+5:.1f} {x-4:.1f},{y-3:.1f} {x+4:.1f},{y-3:.1f}" fill="{col}" stroke="#fff" stroke-width="0.8"/>'

    return (
        f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;font-family:-apple-system,Helvetica,Arial,sans-serif;">'
        f'<defs><clipPath id="dg{u}"><rect x="0" y="0" width="{w}" height="{yb_c:.1f}"/></clipPath>'
        f'<clipPath id="dr{u}"><rect x="0" y="{yb_c:.1f}" width="{w}" height="{h - yb_c:.1f}"/></clipPath></defs>'
        + grid
        + f'<polygon points="{poly}" fill="{P["green"]}" fill-opacity="0.16" clip-path="url(#dg{u})"/>'
        + f'<polygon points="{poly}" fill="{P["red"]}" fill-opacity="0.16" clip-path="url(#dr{u})"/>'
        + f'<polyline points="{vline}" fill="none" stroke="{P["green"]}" stroke-width="2.6" stroke-linejoin="round" clip-path="url(#dg{u})"/>'
        + f'<polyline points="{vline}" fill="none" stroke="{P["red"]}" stroke-width="2.6" stroke-linejoin="round" clip-path="url(#dr{u})"/>'
        + f'<polyline points="{pline}" fill="none" stroke="{P["muted"]}" stroke-width="1.8" stroke-dasharray="4,3" stroke-linejoin="round"/>'
        + marks + xlab + "</svg>"
        + f'<div style="margin-top:6px;font-size:11px;color:{P["muted"]};">'
        f'<span style="margin-right:16px;"><span style="display:inline-block;width:22px;height:5px;border-radius:3px;background:{P["green"]};vertical-align:middle;margin-right:5px;"></span>Value (\u20ac) \u00b7 left</span>'
        f'<span><span style="display:inline-block;width:22px;height:0;border-top:2px dashed {P["muted"]};vertical-align:middle;margin-right:5px;"></span>Unrealized PnL (%) \u00b7 right</span>'
        f'</div>'
    )


def _hero_flow_chips(flows) -> str:
    """Deposit/withdrawal chips with date + amount (green/grey), for the hero."""
    if not flows:
        return ""
    P = PALETTE
    items = ""
    for d, v in sorted(flows, key=lambda t: t[0]):
        col = P["green"] if v >= 0 else P["muted"]
        arrow = "\u25b2" if v >= 0 else "\u25bc"
        items += (f'<span style="display:inline-block;margin:4px 6px 0 0;padding:2px 9px;border-radius:999px;'
                  f'background:#fff;border:1px solid {P["border"]};font-size:11px;white-space:nowrap;">'
                  f'<span style="color:{col};font-weight:700;">{arrow} {_eur_smart(v, signed=True)}</span>'
                  f'<span style="color:{P["subtle"]};"> &middot; {pd.Timestamp(d).strftime("%b %d")}</span></span>')
    return (f'<div style="margin-top:8px;font-size:9px;font-weight:700;letter-spacing:0.04em;'
            f'color:{P["muted"]};text-transform:uppercase;">Cash flows</div>'
            f'<div style="margin-top:2px;">{items}</div>')


def _timeline_vals(series: Optional[list], key: str) -> Optional[list[float]]:
    """Extract the per-bucket weights for one category from an allocation
    timeline series. Returns None when the category never carried weight
    (so the caller can skip drawing an empty sparkline)."""
    if not series:
        return None
    vals = [float(pt.get(key, 0.0)) for pt in series]
    return vals if any(v > 0 for v in vals) else None


# ── Context builders ──────────────────────────────────────────────────────────

@dataclass
class _NewsletterContext:
    """Strongly-typed wrapper around the template context dict."""

    metrics: PortfolioMetrics
    config: InvestorConfig
    issue_number: int = 1
    benchmark_alpha_beta: str = "S&P 500"
    benchmark_geo: str = "MSCI ACWI"


def _build_headline(ctx: _NewsletterContext, hero: dict) -> dict:
    """Build the TL;DR headline shown above the Hero.

    Synthesizes the week into a single narrative sentence: ``how the
    portfolio moved + what to do next``. Designed to give the inbox
    reader the pugno-nello-stomaco answer in 5 seconds.
    """
    m = ctx.metrics
    perf_full = m.performance_full or {}
    week_return = perf_full.get("1w")

    parts: list[str] = []

    # Movement clause
    if week_return is None or (isinstance(week_return, float) and pd.isna(week_return)):
        parts.append("Your portfolio is steady this week")
    else:
        wk_eur = m.total_value * float(week_return) / 100
        if abs(float(week_return)) < 0.1:
            parts.append("Your portfolio is essentially flat this week")
        elif float(week_return) >= 0:
            parts.append(
                f"Your portfolio gained {_eur_smart(wk_eur)} "
                f"({_pct(float(week_return), signed=True)}) this week"
            )
        else:
            parts.append(
                f"Your portfolio lost {_eur_smart(abs(wk_eur))} "
                f"({_pct(float(week_return), signed=True)}) this week"
            )

    # Action clause
    suggestions = list(m.rebalancing_suggestions or [])
    if suggestions:
        parts.append("and the optimizer suggests rebalancing below")
    else:
        parts.append("and your allocation is on target")

    return {
        "text": ", ".join(parts) + ".",
        "is_positive": (
            week_return is None
            or (isinstance(week_return, float) and pd.isna(week_return))
            or float(week_return) >= 0
        ),
    }


def _build_header(ctx: _NewsletterContext) -> dict:
    """Build the header strip metadata.

    The portfolio inception date is taken automatically from the order
    list (``metrics.inception_date``, the first order).

    Issue number is computed dynamically: weeks since inception when
    available, otherwise the ISO week of the current year. The explicit
    ``ctx.issue_number`` value overrides this only when greater than 1,
    so callers wishing to pin a specific number still can.
    """
    now = datetime.now()
    inception_date = ctx.metrics.inception_date or ""
    issue_number = ctx.issue_number
    if issue_number <= 1 and inception_date:
        try:
            inception = pd.to_datetime(inception_date)
            weeks = max(1, int((now - inception.to_pydatetime()).days // 7) + 1)
            issue_number = weeks
        except Exception:
            issue_number = now.isocalendar().week
    elif issue_number <= 1:
        issue_number = now.isocalendar().week
    return {
        "date_short": now.strftime("%a, %d %b %Y"),
        "issue_number": issue_number,
        "inception_date": inception_date,
    }


def _build_hero(ctx: _NewsletterContext) -> dict:
    m = ctx.metrics
    cfg = ctx.config

    # Two distinct P&L views at portfolio level:
    #   * Unrealized PnL — the snapshot gain on the positions held *today*
    #     vs their cost basis (= the Excel "RTD"). Numerator/denominator
    #     are current-holdings only; realized gains and income are excluded.
    #   * Total PnL — the lifetime, all-in gain (realized + unrealized +
    #     coupons/dividends) from the order list, expressed over the *net*
    #     capital contributed (current_value − Total PnL). It answers "how
    #     much have I actually made on the money I put in".
    cost = float(m.holdings_df["cost_basis_eur"].sum()) if not m.holdings_df.empty else 0.0
    unrealized_eur = m.total_value - cost
    unrealized_pct = (unrealized_eur / cost * 100) if cost > 0 else 0.0

    has_total_pnl = m.pnl_eur is not None
    total_pnl_eur = m.pnl_eur if has_total_pnl else unrealized_eur
    total_pnl_pct = (
        m.pnl_pct if (has_total_pnl and m.pnl_pct is not None) else unrealized_pct
    )
    twror_pct = m.twror_pct

    # "Since inception" caption with the precise month/year of the first
    # order (derived automatically; falls back to empty when unknown).
    inception_label = ""
    if m.inception_date:
        try:
            inception_label = pd.to_datetime(m.inception_date).strftime("%b %Y")
        except Exception:
            inception_label = ""

    invested_pct = (m.invested_value / m.total_value * 100) if m.total_value > 0 else 0.0

    # Dual-axis hero chart: 30-day portfolio value (€, left) + Unrealized
    # PnL % (right, flow-adjusted via the daily cost-basis series), with
    # cash-flow triangles. Empty string when the order-derived series are
    # unavailable (holdings-only path).
    value_chart_html = ""
    hero_flow_chips = ""
    win = _perf_window(m, 30)
    if (win and win.get("value") and len(win["value"]) >= 2
            and m.unrealized_series is not None and m.actual_value_series is not None):
        dts = win["dates"]
        idx = pd.DatetimeIndex(dts)
        av = _norm_series(m.actual_value_series).reindex(idx, method="ffill").bfill()
        ur = _norm_series(m.unrealized_series).reindex(idx, method="ffill").bfill()
        unreal_series = list(((ur / (av - ur).replace(0, float("nan"))) * 100.0)
                             .bfill().values.astype(float))
        value_chart_html = _hero_value_chart(win["value"], unreal_series, dts, win["flows"])
        hero_flow_chips = _hero_flow_chips(win["flows"])

    # This-week figures, mirroring the since-inception group:
    #   * Total PnL — the real money gained over the last 7 days, net of any
    #     contributions in the week (delta of the cumulative P&L series);
    #   * TWROR — the 1-week time-weighted return (performance_full['1w']),
    #     the same series the Returns tables use.
    perf_full = m.performance_full or {}
    week_twror = perf_full.get("1w")
    try:
        week_twror = float(week_twror) if week_twror is not None else None
        if week_twror != week_twror:  # NaN
            week_twror = None
    except (TypeError, ValueError):
        week_twror = None
    week_pnl_eur, week_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, 7)
    # Last-30-days money P&L (net of contributions) for the scoreboard's
    # "Last 30 days" row, mirroring the chart window.
    month_pnl_eur, month_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, 30)
    month_twror = perf_full.get("1m")
    try:
        month_twror = float(month_twror) if month_twror is not None else None
        if month_twror != month_twror:  # NaN
            month_twror = None
    except (TypeError, ValueError):
        month_twror = None

    # Cash KPI: show only the amount (no "above/below/on target" message).
    cash_msg, cash_msg_color = "", PALETTE["muted"]

    # Rebalance status: traffic-light derived from the largest non-cash
    # drift in goal_deltas. Mirrors the banner shown in the Excel
    # Optimizer tab so the two outputs agree.
    tol = float(cfg.rebalancing_target_tolerance_pctg or 0.0)
    max_abs_delta = 0.0
    if m.goal_deltas is not None and not m.goal_deltas.empty:
        non_cash = m.goal_deltas[m.goal_deltas["type"] != "cash"]
        if not non_cash.empty:
            max_abs_delta = float(non_cash["delta_pct"].abs().max())
    n_actions = len(m.rebalancing_suggestions or [])

    # The engine flags every verification entry with no_solution=True
    # when the LP returned 0 actions because no plan was feasible at
    # the configured tolerance ceiling (distinct from "already
    # aligned"). It flags ``relaxed=True`` when it had to widen the
    # tolerance up to ``rebalancing_relax_cap_pctg`` to find a plan.
    rebal_infeasible = bool(
        m.rebalancing_verifications
        and any(v.get("no_solution") for v in m.rebalancing_verifications)
    )

    if rebal_infeasible:
        rebal_label = "Infeasible"
        rebal_sublabel = "no feasible plan"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif n_actions == 0:
        # Solved cleanly with no trades (inside tolerance, or pinned by
        # locked positions / auto-relax). Nothing for the user to do.
        rebal_label = "Aligned"
        rebal_sublabel = "no action needed"
        rebal_color = PALETTE["green"]
        rebal_bg = PALETTE["green_bg"]
    else:
        # Actions to take. Communicate only that action is needed — no
        # count, no sublabel. The Optimizer section below carries specifics.
        rebal_label = "Action"
        rebal_sublabel = ""
        # Amber for a moderate plan, red when drift is well beyond tol.
        if max_abs_delta > 2 * tol:
            rebal_color, rebal_bg = PALETTE["red"], PALETTE["red_bg"]
        else:
            rebal_color, rebal_bg = PALETTE["amber"], PALETTE["amber_bg"]

    return {
        # Hero big number, rounded to whole euros (€221,593): decimals add
        # visual noise to the largest figure in the Status section.
        "total_value": _eur(m.total_value, decimals=0),
        # Total PnL — lifetime, realized + unrealized (order path).
        "total_pnl_eur": _eur_smart(total_pnl_eur, signed=True),
        "total_pnl_pct": _pct(total_pnl_pct, signed=True),
        "total_pnl_is_positive": total_pnl_eur >= 0,
        # Unrealized PnL — snapshot gain on current holdings (= Excel RTD).
        "unrealized_eur": _eur_smart(unrealized_eur, signed=True),
        "unrealized_pct": _pct(unrealized_pct, signed=True),
        "unrealized_is_positive": unrealized_eur >= 0,
        # Whether the lifetime Total PnL is a real (order-derived) figure
        # distinct from Unrealized; False on the holdings-only path.
        "has_total_pnl": has_total_pnl,
        # "Since inception · Mon YYYY" caption (empty when inception unknown).
        "inception_label": inception_label,
        # Kept for the preheader/back-compat: the headline % is Total PnL.
        "gain_pct": _pct(total_pnl_pct, signed=True),
        # Cumulative time-weighted return since inception (order path only).
        "twror_pct": _pct(twror_pct, signed=True) if twror_pct is not None else None,
        "twror_is_positive": (twror_pct or 0.0) >= 0,
        # Dual-axis hero chart (value € + Unrealized PnL %) and cash-flow
        # chips, pre-rendered as safe HTML (empty on the holdings-only path).
        "value_chart": value_chart_html or None,
        "flow_chips_html": hero_flow_chips or None,
        "invested_value": _eur_smart(m.invested_value),
        "invested_pct": _pct(invested_pct, decimals=1),
        "cash_value": _eur_smart(m.cash_value),
        "cash_msg": cash_msg,
        "cash_msg_color": cash_msg_color,
        # This week: real money P&L (€ + %, net of contributions) and the
        # 1-week TWROR — both clearly labeled in the template.
        "week_pnl_eur": _eur_smart(week_pnl_eur, signed=True) if week_pnl_eur is not None else None,
        "week_pnl_pct": _pct(week_pnl_pct, signed=True) if week_pnl_pct is not None else None,
        "week_pnl_is_positive": (week_pnl_eur or 0.0) >= 0,
        "week_twror_pct": _pct(week_twror, signed=True) if week_twror is not None else None,
        "week_twror_is_positive": (week_twror or 0.0) >= 0,
        # Last 30 days (mirrors the chart window): money P&L + TWROR.
        "month_pnl_eur": _eur_smart(month_pnl_eur, signed=True) if month_pnl_eur is not None else None,
        "month_pnl_pct": _pct(month_pnl_pct, signed=True) if month_pnl_pct is not None else None,
        "month_pnl_is_positive": (month_pnl_eur or 0.0) >= 0,
        "month_twror_pct": _pct(month_twror, signed=True) if month_twror is not None else None,
        "month_twror_is_positive": (month_twror or 0.0) >= 0,
        # Annualized returns (card subtitles): money-weighted XIRR vs
        # time-weighted TWROR-annualized.
        "xirr_pct": _pct(m.xirr_pct, signed=True) if m.xirr_pct is not None else None,
        "twror_annualized_pct": _pct(m.twror_annualized_pct, signed=True) if m.twror_annualized_pct is not None else None,
        # Net-of-tax ESTIMATE (order path only). Shown as a small line under
        # the since-inception PnL and as a sub-line in the XIRR annualized
        # card; the gross figures above are never altered. `has_net_tax` is
        # False (all keys None) when no CGT was estimated.
        "has_net_tax": (m.estimated_cgt_eur is not None and m.estimated_cgt_eur > 0),
        "cgt_eur": _eur_smart(-m.estimated_cgt_eur, signed=True) if m.estimated_cgt_eur else None,
        "pnl_net_tax_eur": _eur_smart(m.pnl_eur_net_tax, signed=True) if m.pnl_eur_net_tax is not None else None,
        "pnl_net_tax_pct": _pct(m.pnl_pct_net_tax, signed=True) if m.pnl_pct_net_tax is not None else None,
        "pnl_net_tax_is_positive": (m.pnl_eur_net_tax or 0.0) >= 0,
        "xirr_net_tax_pct": _pct(m.xirr_net_tax_pct, signed=True) if m.xirr_net_tax_pct is not None else None,
        # Rebalance status KPI replaces the "This Week" KPI which was
        # already covered by the TL;DR headline above.
        "rebal_label": rebal_label,
        "rebal_sublabel": rebal_sublabel,
        "rebal_color": rebal_color,
        "rebal_bg": rebal_bg,
        "rebal_n_actions": n_actions,
    }


# ── Performance section (1D/7D/30D/since matrix + 30-day charts) ─────────────

def _build_tax_note(ctx: _NewsletterContext) -> dict:
    """Bottom-of-newsletter net-of-tax estimate: the net figures plus the
    methodology disclaimer, moved out of the Performance card so it does not
    crowd the headline numbers. Renders only when a CGT estimate exists."""
    m = ctx.metrics
    P = PALETTE
    if not (m.estimated_cgt_eur and m.estimated_cgt_eur > 0):
        return {"available": False, "html": ""}

    def _sgn(v):
        return P["green"] if (v is not None and v >= 0) else P["red"]

    figs = []
    if m.xirr_net_tax_pct is not None:
        figs.append(f'XIRR net of tax <strong style="color:{_sgn(m.xirr_net_tax_pct)};">'
                    f'{_pct(m.xirr_net_tax_pct, signed=True)}</strong>')
    if m.pnl_eur_net_tax is not None:
        figs.append(f'P&amp;L net of tax <strong style="color:{_sgn(m.pnl_eur_net_tax)};">'
                    f'{_eur_smart(m.pnl_eur_net_tax, signed=True)}</strong>')
    figs_html = (" &nbsp;·&nbsp; ".join(figs)) if figs else ""
    html = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:{P["card_alt"]};border:1px solid {P["border"]};border-radius:10px;'
        f'border-collapse:separate;border-spacing:0;"><tr><td style="padding:12px 14px;">'
        f'<div style="font-size:11px;color:{P["muted"]};line-height:1.5;">'
        f'<span style="font-weight:700;color:{P["ink"]};">Net-of-tax estimate</span>'
        + (f' &nbsp;{figs_html}' if figs_html else "")
        + f'<div style="margin-top:4px;font-size:10px;color:{P["subtle"]};">Estimate only: average-cost basis, '
        f'26% / 12.5% on government bonds, realized losses offset later gains where Italian rules allow '
        f'(ETF/fund gains are not offsettable). Excludes coupon/dividend withholding and the cost basis of '
        f'transferred-in positions. TWROR is gross of tax.</div>'
        f'</div></td></tr></table>'
    )
    return {"available": True, "html": html}


def _build_methodology(ctx: _NewsletterContext) -> dict:
    """Bottom-of-newsletter methodology note: the *actual* calendar spans each
    return bucket covers, computed live from the longest available price
    series so the reader sees exactly which dates a 1D / 1W / … return is
    measured between. Window lengths come from the shared ``stats.PERIOD_DAYS``
    map (same buckets the engine uses everywhere), so this note can never
    drift from the numbers in the tables."""
    from tarzan.engine.stats import PERIOD_DAYS
    P = PALETTE
    m = ctx.metrics

    # Longest daily series available (a benchmark spans ~5y, so every bucket
    # resolves); fall back to the portfolio's own history.
    series = None
    for s in (m.benchmark_histories or {}).values():
        if s is not None and len(s) >= 2 and (series is None or len(s) > len(series)):
            series = s
    if series is None:
        series = m.portfolio_history
    if series is None or len(series) < 2:
        return {"available": False, "html": ""}
    s = _norm_series(series).dropna()
    if len(s) < 2:
        return {"available": False, "html": ""}
    end = s.index[-1]

    def _fmt(a, b) -> str:
        cross = a.year != b.year
        def d(x):
            return f"{x.day} {x.strftime('%b')}" + (f" &rsquo;{x.strftime('%y')}" if cross else "")
        return f"{d(a)}\u2192{d(b)}"

    labels = {"1d": "1D", "1w": "1W", "1m": "1M", "3m": "3M", "6m": "6M",
              "1y": "1Y", "3y": "3Y", "5y": "5Y"}
    order = ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y"]
    spans: dict = {}
    for k, days in PERIOD_DAYS.items():
        if k == "1d":
            continue  # 1D is broker-style (live vs previous close), not a fixed span
        sub = s[s.index >= end - pd.Timedelta(days=days)]
        spans[k] = (sub.index[0], sub.index[-1]) if len(sub) >= 2 else None
    ytd = s[s.index.year == end.year]
    spans["ytd"] = (ytd.index[0], ytd.index[-1]) if len(ytd) >= 2 else None

    parts = [f'<strong style="color:{P["ink"]};">1D</strong> latest price vs previous close (live)']
    for k in order:
        if k == "1d":
            continue
        sp = spans.get(k)
        if sp:
            lbl = "YTD" if k == "ytd" else labels[k]
            parts.append(f'<strong style="color:{P["ink"]};">{lbl}</strong> {_fmt(sp[0], sp[1])}')
    windows = " &nbsp;&middot;&nbsp; ".join(parts)

    html = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:{P["card_alt"]};border:1px solid {P["border"]};border-radius:10px;'
        f'border-collapse:separate;border-spacing:0;"><tr><td style="padding:12px 14px;">'
        f'<div style="font-size:13px;font-weight:700;letter-spacing:0.08em;color:{P["accent"]};'
        f'text-transform:uppercase;">Methodology &middot; return windows</div>'
        f'<div style="margin-top:6px;font-size:11px;color:{P["muted"]};line-height:1.7;">{windows}</div>'
        f'<div style="margin-top:6px;font-size:10px;color:{P["subtle"]};line-height:1.5;">'
        f'Closing prices, ending at the last available close. If an instrument&rsquo;s history '
        f'is shorter than a window, its start is capped (or it shows &ldquo;\u2014&rdquo;).</div>'
        f'</td></tr></table>'
    )
    return {"available": True, "html": html}


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

    # Card width: equal share of the row minus the 1% spacers between cards.
    cw = f"{(100 - (COLS - 1)) // COLS}%"

    def _rc(cat) -> str:
        return MARKET_REGION_COLORS.get(cat, P["subtle"])

    def _card(d: dict) -> str:
        up = d["pct"] >= 0
        col = P["green"] if up else P["red"]
        bg = P["green_bg"] if up else P["red_bg"]
        val = f'{d["value"]:,.2f}'
        chg = f'{d["change"]:+,.2f} ({d["pct"]:+.2f}%)'
        sym = d.get("symbol", "")
        # Tag futures with "(FUT)" so a full-width sparkline reads as a
        # continuously-traded contract (change vs previous settlement), not a
        # finished equity session.
        name = d["name"]
        if sym.upper().endswith("=F"):
            name = f"{name} (FUT)"
        # Intraday time-axis sparkline (line grows through the session, with
        # an endpoint dot) when a timestamped series is available; otherwise
        # the stretched daily fallback.
        ss = d.get("spark_series")
        if ss is not None and len(ss) >= 2:
            # Continuous instruments (futures/FX/crypto) have no bounded cash
            # session → draw the sparkline full-width (in_progress=False).
            # Exchange-listed instruments grow through their session when open
            # (market_open_now True), else render full width (closed session).
            _ip = False if is_continuous_market(sym) else market_open_now(sym)
            spark = _intraday_spark(ss, d.get("baseline", d["value"]), w=90, h=26,
                                    in_progress=_ip)
        else:
            spark = _day_spark(d.get("spark", []), d.get("baseline", d["value"]),
                               w=90, h=26, stretch=True)
        return (
            f'<td width="{cw}" style="vertical-align:top;padding:6px 10px;'
            f'background:{P["card_alt"]};border:1px solid {P["border"]};border-radius:10px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.02em;'
            f'color:{P["muted"]};text-transform:uppercase;white-space:nowrap;overflow:hidden;">{name}</div>'
            f'<div style="margin-top:1px;font-size:11px;font-weight:700;color:{P["ink"]};'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">{val}</div>'
            f'<div style="margin-top:3px;"><span style="font-size:11px;font-weight:700;color:{col};'
            f'background:{bg};padding:2px 7px;border-radius:999px;font-variant-numeric:tabular-nums;'
            f'white-space:nowrap;">{chg}</span></div>'
            f'<div style="margin-top:5px;">{spark}</div></td>'
        )

    # Row-break groups: each region starts on a new row. Categories outside
    # CATEGORY_ORDER become their own group (never dropped). MERGE folds a
    # small category into another group's row (currently none).
    MERGE: dict[str, str] = {}
    group_order, group_members = [], {}
    for cat in CATEGORY_ORDER:
        if cat in MERGE:
            continue  # folded into its target group
        group_order.append(cat)
        group_members[cat] = [cat] + [c for c in CATEGORY_ORDER if MERGE.get(c) == cat]
    for d in snap:
        c = d.get("category")
        if c not in CATEGORY_ORDER and c not in group_order:
            group_order.append(c)
            group_members[c] = [c]

    # Each region: an underlined heading (region-accent colour) followed by its
    # own card grid. Colour lives in the heading, so the cards stay neutral and
    # no separate legend is needed.
    blocks = []
    for g in group_order:
        cards = []
        for cat in group_members[g]:
            cards += [_card(d) for d in snap if d.get("category") == cat]
        if not cards:
            continue
        while len(cards) % COLS != 0:
            cards.append(f'<td width="{cw}"></td>')
        rows = []
        for i in range(0, len(cards), COLS):
            rows.append("<tr>" + '<td width="1%"></td>'.join(cards[i:i + COLS]) + "</tr>")
        heading = (f'<div style="margin:14px 0 6px;font-size:11px;font-weight:700;'
                   f'color:{P["ink"]};letter-spacing:0.02em;">'
                   f'<span style="display:inline-block;width:10px;height:10px;background:{_rc(g)};'
                   f'border-radius:3px;vertical-align:middle;margin-right:6px;"></span>{g}</div>')
        grid = ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                'border="0" style="border-collapse:separate;">' + "".join(rows) + '</table>')
        blocks.append(heading + grid)

    header = (f'<div style="font-size:13px;font-weight:700;letter-spacing:0.08em;color:{P["accent"]};'
              f'text-transform:uppercase;">Markets</div>')
    return {"available": True, "html": header + "".join(blocks)}


def _benchmark_period_return(m, bench_name: Optional[str], period: str):
    """The AUTHORITATIVE period return (e.g. '1m') for a benchmark, read from
    the engine's ``holding_performance`` — the same source the Returns tables
    use. Returns None when unavailable. This is what chart legends must show so
    a benchmark's chart number equals its table number by construction."""
    if not bench_name:
        return None
    hp = getattr(m, "holding_performance", None)
    if hp is None or hp.empty or "name" not in hp.columns or period not in hp.columns:
        return None
    want = bench_name.strip().lower()
    match = hp[hp["name"].astype(str).str.strip().str.lower() == want]
    if match.empty:
        return None
    val = match.iloc[0].get(period)
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else float(val)


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
    cost = float(m.holdings_df["cost_basis_eur"].sum()) if not m.holdings_df.empty else 0.0
    unreal_now = m.total_value - cost
    unr_since = (unreal_now, (unreal_now / cost * 100.0) if cost > 0 else None)
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

    def _label(txt) -> str:
        return f'<td style="padding:7px 0;border-top:{bt};font-size:12px;color:{P["ink"]};">{txt}</td>'

    # Flag the 1-Day column as live when the portfolio 1D is a market-open
    # quote (set by the engine's _live_1d step).
    _1d_live = bool((m.performance_full or {}).get("1d_live"))
    head_1d = "1 Day \u25CF LIVE" if _1d_live else "1 Day"
    heads = (head_1d, "7 Days", "30 Days", "Since inception")
    head_html = '<tr><td></td>' + "".join(
        f'<td align="right" style="padding:0 0 5px 10px;font-size:9px;font-weight:700;color:{P["muted"]};'
        f'letter-spacing:0.04em;text-transform:uppercase;">{h}</td>' for h in heads) + '</tr>'
    row_t = '<tr>' + _label("Total P&amp;L") + _money_cell(tot[1]) + _money_cell(tot[7]) + _money_cell(tot[30]) + _money_cell(tot_since) + '</tr>'
    row_u = '<tr>' + _label("Unrealized P&amp;L") + _money_cell(unr[1]) + _money_cell(unr[7]) + _money_cell(unr[30]) + _money_cell(unr_since) + '</tr>'
    row_w = '<tr>' + _label("TWROR") + _pct_cell(tw[1]) + _pct_cell(tw[7]) + _pct_cell(tw[30]) + _pct_cell(tw_since) + '</tr>'
    matrix = (f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
              f'style="border-collapse:collapse;">{head_html}{row_t}{row_u}{row_w}</table>')

    footer = (f'<div style="margin-top:12px;padding-top:10px;border-top:2px solid {P["border"]};'
              f'font-size:12px;color:{P["muted"]};line-height:1.5;">Annualized — '
              f'TWROR <strong style="color:{_sgn(m.twror_annualized_pct)};">{_pct(m.twror_annualized_pct, signed=True)}</strong> · '
              f'XIRR <strong style="color:{_sgn(m.xirr_pct)};">{_pct(m.xirr_pct, signed=True)}</strong>'
              + (f' <span style="color:{P["subtle"]};">· net of tax: see note at the bottom &#8595;</span>'
                 if (m.xirr_net_tax_pct is not None or m.pnl_eur_net_tax is not None) else "")
              + '</div>')

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

    # Legend numbers are pinned to the engine's AUTHORITATIVE figures, never
    # the chart line's own endpoint: the line draws the trajectory, but the %
    # it is labelled with is the same number the Performance tables show, so a
    # chart and a table can never tell two stories. (The lines are anchored on
    # the same canonical window rule, so the endpoint already coincides — the
    # pin makes that a guarantee, not a coincidence.)
    perf_full = m.performance_full or {}
    acwi_1m = _benchmark_period_return(m, ctx.benchmark_geo, "1m")

    # Last 30 days (rebased to 0). Labels pinned to performance_full["1m"] and
    # the benchmark's authoritative 1m return.
    s30, l30 = [], []
    if win["twror"] is not None:
        s30.append({"values": win["twror"], "color": GREEN})
        l30.append((f'TWROR {_pct(perf_full.get("1m"), signed=True)}', GREEN, False))
    if win["pnl_pct"] is not None:
        s30.append({"values": win["pnl_pct"], "color": PNL})
        l30.append((f'Total P&L % {_pct(win["pnl_pct"][-1], signed=True)}', PNL, False))
    if win["acwi"] is not None:
        s30.append({"values": win["acwi"], "color": BENCH, "dash": True})
        l30.append((f'MSCI ACWI {_pct(acwi_1m, signed=True)}', BENCH, True))

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
            ssi.append({"values": full["acwi"], "color": BENCH, "dash": True})
            lsi.append((f'MSCI ACWI {_pct(full["acwi"][-1], signed=True)}', BENCH, True))

    parts = []
    if s30 or ssi:
        # LEFT = since inception (the fuller story); RIGHT = last 30 days. The
        # since-inception axis shows month ticks across the whole span; the
        # 30-day axis shows day-level ticks.
        left = (_colcap(f"Since inception <span style='font-weight:400;color:{P['subtle']};'>· cumulative</span>")
                + _charts.chart_pct_compact(ssi, si_dates, include_zero=False, month_ticks=True)
                + _charts.legend(lsi, 9)) if ssi else ""
        right = (_colcap(f"Last 30 days <span style='font-weight:400;color:{P['subtle']};'>· rebased to 0</span>")
                 + _charts.chart_pct_compact(s30, dates, include_zero=True) + _charts.legend(l30, 9)) if s30 else ""
        charts_tbl = (
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:12px;"><tr>'
            f'<td width="50%" valign="top" style="padding-right:8px;">{left}</td>'
            f'<td width="50%" valign="top" style="padding-left:8px;border-left:1px solid {P["border"]};">{right}</td>'
            f'</tr></table>'
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

        inner = (
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.06em;color:{P["accent"]};'
            f'text-transform:uppercase;">You vs the market</div>'
            f'<div style="margin-top:2px;font-size:12px;color:{P["muted"]};">Your return paths vs '
            f'<strong style="color:{P["ink"]};">MSCI ACWI</strong> — since inception (cumulative) and '
            f'the last 30 days (rebased).</div>{charts_tbl}{divergence_html}'
        )
        # Wrapped in the same card shell as the matrix above so the section
        # reads consistently with the rest of the newsletter.
        parts.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin-top:14px;background:{P["card_alt"]};border:1px solid {P["border"]};'
            f'border-radius:12px;border-collapse:separate;border-spacing:0;">'
            f'<tr><td style="padding:14px 16px;">{inner}</td></tr></table>'
        )

    header = (f'<div style="font-size:13px;font-weight:700;letter-spacing:0.08em;color:{P["accent"]};'
              f'text-transform:uppercase;">Performance</div>'
              f'<div style="margin-top:4px;font-size:18px;font-weight:700;color:{P["ink"]};">How your money moved</div>'
              f'<div style="margin-top:4px;font-size:12px;color:{P["muted"]};">Total &amp; unrealized P&amp;L and your '
              f'time-weighted return across 1 day, 7 days, 30 days and since inception — then your return vs the market.</div>')

    return {"available": True, "html": header + matrix_card + "".join(parts)}


def _build_allocation(ctx: _NewsletterContext) -> dict:
    """Build asset-class allocation rows (Excel Dashboard pattern)."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_target_tolerance_pctg

    targets = cfg.invested_allocation_targets_pctg or {}
    alloc_df = m.allocation_by_class

    timeline = m.allocation_timeline or {}
    asset_series = timeline.get("asset")

    # Physical capital per class (sum of the market value of holdings whose
    # PRIMARY class is that class) — the denominator for the per-class
    # leverage = notional exposure / physical capital. >1 means the class is
    # partly synthetic (e.g. an efficient-core bond overlay).
    inv_val = float(getattr(m, "invested_value", 0.0) or 0.0)
    phys_by_class: dict[str, float] = {}
    hdf = m.holdings_df
    if hdf is not None and not hdf.empty and "asset_class" in hdf.columns:
        for cls, grp in hdf.groupby("asset_class"):
            phys_by_class[str(cls)] = float(grp["current_value"].sum())

    rows = []
    for klass in ASSET_CLASS_ORDER:
        match = (alloc_df[alloc_df["category"] == klass]
                 if not alloc_df.empty else alloc_df)
        has_holding = match is not None and not match.empty
        target = targets.get(klass)
        # Show a class if it is held OR it carries a (non-zero) target, so a
        # targeted-but-not-yet-held class appears as Now 0% vs its target.
        if not has_holding and not (target and target > 0):
            continue
        actual = float(match["weight_pct"].iloc[0]) if has_holding else 0.0
        delta = actual - target if target is not None else None
        sema = _semaphore(delta, tol)
        color = ASSET_COLORS.get(klass, PALETTE["accent"])
        spark_vals = _timeline_vals(asset_series, klass)
        notional_eur = actual / 100.0 * inv_val
        phys = phys_by_class.get(klass, 0.0)
        leverage = (notional_eur / phys) if phys > 0 else None
        rows.append({
            "name": klass,
            "color": color,
            "actual_pct": _pct_smart(actual),
            "actual_pct_raw": actual,
            "target_pct": _pct_smart(target) if target is not None else None,
            "target_left": (
                min(max(float(target), 0), 100)
                if target is not None else None
            ),
            "delta": _signed_pp(delta) if delta is not None else None,
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max(actual, 1), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
            "leverage": leverage,
        })

    # Cash buffer (EUR-based, appended after invested classes).
    # The bar width is scaled as % of total portfolio so cash visually
    # matches the other rows (it would otherwise dominate the bar
    # because target_cash_buffer_eur is small relative to invested
    # value). Status color is still driven by the relative deviation
    # vs the cash target via _semaphore.
    if cfg.target_cash_buffer_eur > 0:
        cash_actual = m.cash_value
        cash_tgt = cfg.target_cash_buffer_eur
        rel_dev = (cash_actual - cash_tgt) / cash_tgt * 100 if cash_tgt > 0 else 0
        sema = _semaphore(rel_dev, tol)
        delta_eur = cash_actual - cash_tgt
        cash_pct_of_total = (cash_actual / m.total_value * 100) if m.total_value > 0 else 0
        rows.append({
            # Shorter label only inside the Diversification block where
            # horizontal space is critical; other sections (Holdings,
            # Optimizer, Insights) keep the full "Cash & Cash
            # Equivalents" string.
            "name": "Cash & Cash Eq.",
            "color": ASSET_COLORS["Cash & Cash Equivalents"],
            "actual_pct": _eur_smart(cash_actual),
            "actual_pct_raw": cash_pct_of_total,
            "target_pct": _eur_smart(cash_tgt),
            "delta": _eur_smart(delta_eur, signed=True),
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max(cash_pct_of_total, 1), 100),
            "is_eur": True,
            # Raw EUR figures so the diversification table can show cash as a
            # normal row without it participating in the invested base.
            "cash_actual_eur": cash_actual,
            "cash_target_eur": cash_tgt,
            "cash_delta_eur": delta_eur,
        })

    # Stacked bar segments (invested only)
    stacked = []
    for klass in ASSET_CLASS_ORDER:
        if alloc_df.empty:
            continue
        match = alloc_df[alloc_df["category"] == klass]
        if match.empty:
            continue
        w = float(match["weight_pct"].iloc[0])
        if w > 0:
            stacked.append({
                "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
                "width": w,
            })

    return {
        "rows": rows,
        "stacked": stacked,
        "tolerance": _pct(tol, decimals=1).rstrip("%") + "%",
        "has_timeline": any(r.get("spark") for r in rows),
    }


def _build_geography(ctx: _NewsletterContext) -> dict:
    """Build geographic equity rows with target & ACWI ticks."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_target_tolerance_pctg

    targets = cfg.equity_geo_targets_pctg or {}
    geo_df = m.allocation_by_geo
    acwi = m.acwi_geo or {}

    timeline = m.allocation_timeline or {}
    geo_series = timeline.get("geo")

    # Actual equity-geo weights, plus any region that only has a target (so a
    # targeted-but-absent region still shows as Now 0% vs its target).
    actual_by_region: dict[str, float] = {}
    if not geo_df.empty:
        for _, r in geo_df.iterrows():
            actual_by_region[str(r["category"])] = float(r["weight_pct"])
    regions = list(actual_by_region.keys())
    for region in targets:
        if region not in actual_by_region and (targets.get(region) or 0) > 0:
            regions.append(region)
    # Order by actual descending (target-only regions, actual 0, sort last).
    regions.sort(key=lambda rg: -actual_by_region.get(rg, 0.0))

    rows = []
    for region in regions:
        actual = actual_by_region.get(region, 0.0)
        target = targets.get(region)
        acwi_v = acwi.get(region)
        delta_target = actual - target if target is not None else None
        sema = _semaphore(delta_target, tol)
        color = GEO_COLORS.get(region, PALETTE["accent"])
        spark_vals = _timeline_vals(geo_series, region)
        rows.append({
            "name": region,
            "color": color,
            "actual_pct": _pct_smart(actual),
            "actual_pct_raw": actual,
            "target_pct": _pct_smart(target) if target is not None else "—",
            "acwi_pct": _pct_smart(acwi_v) if acwi_v is not None else "—",
            "delta": _signed_pp(delta_target) if delta_target is not None else "—",
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max(actual, 1), 100),
            "target_left": min(max(target or 0, 0), 100),
            "acwi_left": min(max(acwi_v or 0, 0), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
        })

    # Stacked equity bar
    stacked = [{"color": r["color"], "width": r["bar_width"]} for r in rows if r["bar_width"] > 0]

    return {
        "rows": rows,
        "stacked": stacked,
        "benchmark_name": ctx.benchmark_geo,
        "has_timeline": any(r.get("spark") for r in rows),
    }


# ── Diversification (unified tables: asset class + geography + by holding) ────
# Pre-rendered (trend sparklines can't be expressed in Jinja), injected as safe
# HTML — same pattern as the Markets / Performance blocks. Each group is one
# compact table: current weight, target, drift and a 1-month trend sparkline.


def _recent_timeline(series: Optional[list], dates: Optional[list],
                     days: int = 31) -> Optional[list]:
    """Slice a timeline series to its last ``days`` (the 1-month trend window).
    Falls back to the last 5 buckets so a sparkline always has ≥2 points."""
    if not series or not dates:
        return series
    cutoff = pd.Timestamp(dates[-1]) - pd.Timedelta(days=days)
    keep = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= cutoff]
    if len(keep) < 2:
        keep = list(range(max(0, len(dates) - 5), len(dates)))
    return [series[i] for i in keep]


def _div_pin(ticker: Optional[str]) -> str:
    """Small monospace ticker pin, identical to the Holdings/Risk/Returns
    sections, so the ticker looks the same everywhere."""
    if not ticker:
        return ""
    P = PALETTE
    return (f'<span style="display:inline-block;margin-right:5px;padding:1px 5px;'
            f'background:{P["page"]};color:{P["muted"]};border:1px solid {P["border"]};'
            f'border-radius:4px;font-size:9px;font-weight:700;'
            f'font-family:SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:0.02em;'
            f'vertical-align:middle;">{ticker}</span>')


def _div_label(name: str, color: str, ticker: Optional[str] = None) -> str:
    """Row label for the diversification tables: a small colour swatch, an
    optional ticker pin, then the name."""
    P = PALETTE
    sw = (f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
          f'background:{color};vertical-align:middle;margin-right:6px;"></span>')
    return f'{sw}{_div_pin(ticker)}<span style="color:{P["ink"]};">{name}</span>'


def _div_table(rows: list[dict], tol: float, base: Optional[float] = None,
               show_leverage: bool = False) -> str:
    """Unified diversification table (asset class / geography / by holding).

    One row per slice — current weight, target, drift and a 1-month trend
    sparkline with a ±pp badge — in a single table style shared by all three
    groups (no donuts). Each row dict carries ``label_html``, ``now``,
    ``target``, ``spark_vals`` and ``color``. When ``base`` (the EUR value of
    100%) is given, the Now/Target cells also show the compact absolute
    amount inline (e.g. "26.5% · €10.3k") — same row height, no extra
    columns, since the % is what drives width/alignment.

    ``show_leverage`` adds a "Lev" column = notional exposure / physical
    capital in that class (row dict ``leverage``); used only for the asset-
    class table, where >1.0 marks a partly-synthetic class (e.g. a bond
    overlay). Returns "" for an empty ``rows``.
    """
    if not rows:
        return ""
    P = PALETTE

    _bb = f'border-bottom:1px solid {P["border"]};'

    def _lev_cell(lev) -> str:
        if not show_leverage:
            return ""
        if lev is None:
            txt = "\u2014"
            col = P["subtle"]
        else:
            txt = f"{float(lev):.2f}\u00d7"
            # Emphasise real leverage (>1.05×); ~1.0× stays muted.
            col = P["ink"] if float(lev) > 1.05 else P["subtle"]
        return (f'<td align="right" style="padding:5px 6px;{_bb}font-size:11px;'
                f'font-weight:700;color:{col};white-space:nowrap;'
                f'font-variant-numeric:tabular-nums;width:46px;">{txt}</td>')

    def _num_cell(pct_val: float, color: str) -> str:
        """A Now/Target value cell: the % (bold, right-aligned) and, when a
        EUR base is known, the compact absolute in fixed sub-columns so the
        %, the '·' and the € line up vertically across every row."""
        pct = _pct_smart(pct_val)
        if base and base > 0:
            eur = _eur_smart(pct_val / 100.0 * base)
            return (
                f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
                f'<td align="right" style="font-size:12px;font-weight:700;color:{color};'
                f'font-variant-numeric:tabular-nums;white-space:nowrap;">{pct}</td>'
                f'<td align="center" width="10" style="font-size:11px;color:{P["subtle"]};">\u00b7</td>'
                f'<td align="right" width="50" style="font-size:11px;color:{P["subtle"]};'
                f'font-variant-numeric:tabular-nums;white-space:nowrap;">{eur}</td>'
                f'</tr></table>'
            )
        return (f'<span style="font-weight:700;color:{color};'
                f'font-variant-numeric:tabular-nums;">{pct}</span>')

    def _eur_cell(eur_val: float, color: str, weight: int = 700, signed: bool = False) -> str:
        """A value cell showing only a EUR amount (cash row) in the same
        right sub-column as the other rows' € so they stay aligned."""
        txt = _eur_smart(eur_val, signed=signed)
        return (
            f'<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
            f'<td align="right" style="font-size:11px;font-weight:{weight};color:{color};'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">{txt}</td>'
            f'</tr></table>'
        )

    body = []
    for r in rows:
        bb = f'border-bottom:1px solid {P["border"]};'
        # Total-portfolio summary row: highlighted, portfolio-level leverage
        # (total notional / capital), total notional Now vs total Target.
        if r.get("is_total"):
            abg = P["accent_bg"]
            now = float(r.get("now", 0.0) or 0.0)
            tgt = float(r.get("target", 0.0) or 0.0)
            drift = now - tgt
            vals = r.get("spark_vals")
            sp = _spark(vals, tgt, P["accent"], 70, 18) if vals else ""
            # Same trend cell as the other rows: sparkline + ±pp badge inline.
            if vals and len(vals) >= 2:
                _pp = vals[-1] - vals[0]
                _ar = "\u25b2" if _pp > 0.01 else ("\u25bc" if _pp < -0.01 else "\u2192")
                _bt = f"{_ar}{_pp:+.1f}".replace("+0.0", "0.0")
                _badge = (f'<span style="font-size:9px;font-weight:600;color:{P["accent"]};'
                          f'white-space:nowrap;">{_bt}</span>')
            else:
                _badge = ""
            sp = (f'<span style="display:inline-block;vertical-align:middle;">{sp}</span>'
                  f'{_badge}') if (sp or _badge) else ""
            lev = r.get("leverage")
            lev_txt = f"{float(lev):.2f}\u00d7" if lev is not None else "\u2014"
            lev_td = (f'<td align="right" style="padding:7px 6px;background:{abg};font-size:11px;'
                      f'font-weight:700;color:{P["accent"]};font-variant-numeric:tabular-nums;'
                      f'width:46px;">{lev_txt}</td>' if show_leverage else "")
            body.append(
                f'<tr>'
                f'<td style="padding:7px 8px;background:{abg};font-size:12px;font-weight:700;'
                f'color:{P["accent"]};">{r.get("label_html", "")}</td>'
                f'{lev_td}'
                f'<td align="right" style="padding:7px 8px;background:{abg};width:118px;">{_num_cell(now, P["accent"])}</td>'
                f'<td align="right" style="padding:7px 8px;background:{abg};width:118px;">{_num_cell(tgt, P["accent"])}</td>'
                f'<td align="right" style="padding:7px 8px;background:{abg};font-size:12px;'
                f'font-weight:700;color:{P["accent"]};white-space:nowrap;font-variant-numeric:tabular-nums;'
                f'width:74px;">{_signed_pp(drift)}</td>'
                f'<td align="right" valign="middle" style="padding:7px 4px;background:{abg};'
                f'width:100px;white-space:nowrap;">{sp}</td>'
                f'</tr>'
            )
            continue
        # Cash (EUR-native) row: not a share of the invested base, so show
        # plain EUR amounts and a EUR drift, no trend.
        if r.get("eur_row"):
            ddcol = r.get("delta_color", P["muted"])
            body.append(
                f'<tr>'
                f'<td style="padding:5px 8px;{bb}font-size:12px;color:{P["ink"]};">{r.get("label_html", "")}</td>'
                f'{_lev_cell(r.get("leverage"))}'
                f'<td align="right" style="padding:5px 8px;{bb}width:118px;">{_eur_cell(r.get("now_eur", 0.0), P["muted"])}</td>'
                f'<td align="right" style="padding:5px 8px;{bb}width:118px;">{_eur_cell(r.get("target_eur", 0.0), P["ink"])}</td>'
                f'<td align="right" style="padding:5px 8px;{bb}font-size:12px;font-weight:700;'
                f'color:{ddcol};white-space:nowrap;font-variant-numeric:tabular-nums;width:74px;">'
                f'{_eur_smart(r.get("delta_eur", 0.0), signed=True)}</td>'
                f'<td style="padding:5px 4px;{bb}width:100px;"></td>'
                f'</tr>'
            )
            continue
        now = float(r.get("now", 0.0) or 0.0)
        tgt = float(r.get("target", 0.0) or 0.0)
        drift = now - tgt
        dcol = _semaphore_color(_semaphore(drift, tol))
        vals = r.get("spark_vals")
        sp = _spark(vals, tgt, r.get("color", P["accent"]), 70, 18) if vals else ""
        # 1-month change badge — compact, tucked right of the sparkline on
        # the same line so the row height stays minimal.
        if vals and len(vals) >= 2:
            pp = vals[-1] - vals[0]
            arrow = "\u25b2" if pp > 0.01 else ("\u25bc" if pp < -0.01 else "\u2192")
            pp_txt = f"{arrow}{pp:+.1f}".replace("+0.0", "0.0")
            badge = (f'<span style="font-size:9px;font-weight:600;color:{P["muted"]};'
                     f'white-space:nowrap;">{pp_txt}</span>')
        else:
            badge = ""
        # Trend cell: sparkline + badge in one inline row (no extra vertical).
        trend_inner = (f'<span style="display:inline-block;vertical-align:middle;">{sp}</span>'
                       f'{badge}') if (sp or badge) else ""
        # Fixed column widths (Now/Target/Drift/Trend) so the three
        # Diversification sub-tables (asset class / geography / by holding)
        # line up on the same grid regardless of their content.
        body.append(
            f'<tr>'
            f'<td style="padding:5px 8px;{bb}font-size:12px;color:{P["ink"]};">{r.get("label_html", "")}</td>'
            f'{_lev_cell(r.get("leverage"))}'
            f'<td style="padding:5px 8px;{bb}width:118px;">{_num_cell(now, P["muted"])}</td>'
            f'<td style="padding:5px 8px;{bb}width:118px;">{_num_cell(tgt, P["ink"])}</td>'
            f'<td align="right" style="padding:5px 8px;{bb}font-size:12px;font-weight:700;'
            f'color:{dcol};white-space:nowrap;font-variant-numeric:tabular-nums;width:74px;">'
            f'\u25cf {_signed_pp(drift)}</td>'
            f'<td align="right" valign="middle" style="padding:5px 4px;{bb}width:100px;'
            f'white-space:nowrap;">{trend_inner}</td>'
            f'</tr>'
        )
    head = (
        f'<tr>'
        f'<td style="padding:4px 8px;font-size:10px;font-weight:700;letter-spacing:0.04em;'
        f'text-transform:uppercase;color:{P["subtle"]};">Name</td>'
        + (f'<td align="right" style="padding:4px 6px;font-size:10px;font-weight:700;'
           f'letter-spacing:0.04em;text-transform:uppercase;color:{P["subtle"]};width:46px;">Lev</td>'
           if show_leverage else "")
        + f'<td align="right" style="padding:4px 8px;font-size:10px;font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;color:{P["subtle"]};width:118px;">Now</td>'
        f'<td align="right" style="padding:4px 8px;font-size:10px;font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;color:{P["subtle"]};width:118px;">Target</td>'
        f'<td align="right" style="padding:4px 8px;font-size:10px;font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;color:{P["subtle"]};width:74px;">Drift</td>'
        f'<td align="right" style="padding:4px 8px;font-size:10px;font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;color:{P["subtle"]};width:100px;">Trend (1M)</td>'
        + '</tr>'
    )
    return (f'<table width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin-top:8px;background:{P["card_alt"]};border:1px solid {P["border"]};'
            f'border-radius:10px;border-collapse:separate;overflow:hidden;">{head}{"".join(body)}</table>')


def _ph_target_rows(ctx: _NewsletterContext, tol: float,
                    hold_inv_series: Optional[list]) -> tuple[list[dict], str]:
    """Rows (for :func:`_div_table`) for per-holding portfolio targets in
    ``target_use_per_holding_only`` mode: each targeted instrument's CURRENT
    weight (% of invested, straight from the rebalancer verification — so a
    not-yet-held target reads 0%), its target, and a 1-month weight trend on
    the SAME % of invested basis. Also returns a short "to exit" note for
    0%-target holdings still held. ``(rows, note)``.

    ``hold_inv_series`` is the timeline's ``holding_invested`` series (keyed by
    ISIN). Verification items carry a ticker/name, so we resolve each to its
    ISIN via the snapshot to look up the right trend line.
    """
    P = PALETTE
    m = ctx.metrics
    df = getattr(m, "holdings_df", None)

    # Resolver: ticker (bare + full) and name → ISIN, so a verification item
    # (which carries h.ticker like "NTSG.DE" and the name) can find its
    # ISIN-keyed trend line.
    invested = float(getattr(m, "invested_value", 0.0) or 0.0)
    if invested <= 0:
        invested = float(getattr(m, "total_value", 0.0) or 0.0)
    isin_of: dict[str, str] = {}
    cur_by_isin: dict[str, float] = {}  # CURRENT weight (% of invested) by ISIN
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            _isin = str(row.get("isin", "") or "").strip()
            if not _isin:
                continue
            for key in (row.get("isin"), row.get("ticker"), row.get("name")):
                if key and str(key).strip():
                    isin_of[str(key).strip().upper()] = _isin
            # Also index the exchange-stripped ticker (NTSG.DE → NTSG).
            tkr = str(row.get("ticker", "") or "").strip()
            if tkr:
                isin_of[tkr.upper().split(".")[0]] = _isin
            val = float(row.get("current_value", 0.0) or 0.0)
            cur_by_isin[_isin] = (val / invested * 100.0) if invested > 0 else 0.0

    def _isin_for(it) -> str:
        for key in (it.get("ticker"), it.get("category")):
            if not key:
                continue
            k = str(key).strip().upper()
            if k in isin_of:
                return isin_of[k]
            if k.split(".")[0] in isin_of:
                return isin_of[k.split(".")[0]]
        return ""

    def _trend_for(it) -> Optional[list]:
        if not hold_inv_series:
            return None
        isin = _isin_for(it)
        if not isin:
            return None
        xs = [float(pt.get(isin, 0.0)) for pt in hold_inv_series]
        if any(x > 0 for x in xs) and len(xs) >= 2:
            return xs
        return None

    items = []
    for v in (m.rebalancing_verifications or []):
        if v.get("kind") == "per_holding_portfolio":
            items = v.get("items", []) or []
            break

    targeted, exits = [], []
    for it in items:
        tgt = float(it.get("target_pct", 0.0) or 0.0)
        (targeted if tgt > 0 else exits).append(it)
    targeted.sort(key=lambda it: -float(it.get("target_pct", 0.0) or 0.0))

    rows = []
    for it in targeted:
        key = it.get("ticker") or it.get("category") or ""
        tk = _clean_ticker(key) or _display_ticker(key) or ""
        # "Now" = the CURRENT weight (% of invested) from the snapshot, so it
        # matches the By-asset-class table exactly. The verification's
        # actual_pct is POST-trade (it reflects the plan's buys), which would
        # not equal the current weight — a not-yet-held target reads 0%.
        now = cur_by_isin.get(_isin_for(it), 0.0)
        rows.append({
            "label_html": _div_label(
                short_instrument_name(it.get("category") or key, 42),
                P["accent"], ticker=tk),
            "now": now,
            "target": float(it.get("target_pct", 0.0) or 0.0),
            "spark_vals": _trend_for(it),
            "color": P["accent"],
        })

    note = ""
    held_exits = [it for it in exits if cur_by_isin.get(_isin_for(it), 0.0) > 0.05]
    if held_exits:
        names = ", ".join(
            (_clean_ticker(it.get("ticker") or "") or _display_ticker(it.get("ticker") or "")
             or short_instrument_name(it.get("category") or "", 16))
            for it in held_exits
        )
        note = (f'<div style="margin-top:6px;font-size:11px;color:{P["muted"]};">'
                f'To exit (0% target): <b>{names}</b></div>')
    return rows, note


def _holding_verif_rows(ctx: _NewsletterContext, tol: float,
                        hold_series: Optional[list], kind: str,
                        color: str) -> list[dict]:
    """Rows (for :func:`_div_table`) from an equity/FI per-holding verification
    (weight is % of the sleeve), used when NOT in per-holding-only mode."""
    rows = []
    for v in (ctx.metrics.rebalancing_verifications or []):
        if v.get("kind") != kind:
            continue
        # Sort by weight desc, with ticker as a STABLE tie-break: two holdings
        # at the same weight would otherwise keep their input order, which
        # traces back to a set() of open ISINs (hash-randomized per process) —
        # making the rendered order vary run-to-run and breaking reproducibility.
        items = sorted(v.get("items", []) or [],
                       key=lambda it: (-float(it.get("actual_pct", 0.0)),
                                       str(it.get("ticker", "")),
                                       str(it.get("category", ""))))
        for it in items:
            isin = it.get("ticker", "")
            tk = _clean_ticker(isin) or _display_ticker(isin) or ""
            vals = None
            if hold_series:
                xs = [float(pt.get(isin, 0.0)) for pt in hold_series]
                if any(x > 0 for x in xs) and len(xs) >= 2:
                    vals = xs
            rows.append({
                "label_html": _div_label(
                    short_instrument_name(it.get("category") or isin, 42),
                    color, ticker=tk),
                "now": float(it.get("actual_pct", 0.0)),
                "target": float(it.get("target_pct", 0.0)),
                "spark_vals": vals,
                "color": color,
            })
    return rows


def _build_diversification(ctx: _NewsletterContext) -> dict:
    """Pre-render the Diversification section (tile dashboard) as HTML.

    Reuses :func:`_build_allocation` / :func:`_build_geography` for the
    numbers and semaphore logic, the rebalancer's per-holding checks for the
    by-holding tiles, ``short_instrument_name`` for labels, the price-cache
    ISIN→symbol map for clean tickers, and :func:`_spark` for the trends — so
    this is a presentational layer, not a second source of truth.
    """
    P = PALETTE
    alloc = _build_allocation(ctx)
    geo = _build_geography(ctx)
    tol = ctx.config.rebalancing_target_tolerance_pctg

    # EUR bases (value of 100%) for the inline absolute amounts: invested
    # value for asset-class/per-holding-portfolio rows, and the equity/FI
    # sleeve totals for geography/per-holding-equity/per-holding-FI rows —
    # matching exactly what each row's % is already a share of.
    m = ctx.metrics
    invested_base = float(getattr(m, "invested_value", 0.0) or 0.0)
    if invested_base <= 0:
        invested_base = float(getattr(m, "total_value", 0.0) or 0.0)
    hdf = getattr(m, "holdings_df", None)
    equity_base = fi_base = 0.0
    if hdf is not None and not hdf.empty and "asset_class" in hdf.columns:
        equity_base = float(hdf.loc[hdf["asset_class"] == "Equities", "current_value"].sum())
        fi_base = float(hdf.loc[hdf["asset_class"] == "Fixed Income", "current_value"].sum())

    tl = ctx.metrics.allocation_timeline or {}
    dates = tl.get("dates") or []
    asset_series = _recent_timeline(tl.get("asset"), dates)
    geo_series = _recent_timeline(tl.get("geo"), dates)
    hold_series = _recent_timeline(tl.get("holding"), dates)
    hold_inv_series = _recent_timeline(tl.get("holding_invested"), dates)

    available = bool(alloc.get("rows") or geo.get("rows"))
    if not available:
        return {"available": False, "html": ""}

    def swatch(color, sz=10):
        return (f'<span style="display:inline-block;width:{sz}px;height:{sz}px;'
                f'border-radius:2px;background:{color};vertical-align:middle;"></span>')

    # ── Asset-class rows (cash folded in as a normal, EUR-native row that
    #    does NOT participate in the invested base) ──
    asset_rows = []
    for r in alloc["rows"]:
        if r.get("is_eur"):
            asset_rows.append({
                "label_html": _div_label(r["name"], r["color"]),
                "eur_row": True,
                "now_eur": r.get("cash_actual_eur", 0.0),
                "target_eur": r.get("cash_target_eur", 0.0),
                "delta_eur": r.get("cash_delta_eur", 0.0),
                "delta_color": r.get("delta_color", P["muted"]),
            })
            continue
        asset_rows.append({
            "label_html": _div_label(r["name"], r["color"]),
            "now": r.get("actual_pct_raw"),
            "target": r.get("target_left"),
            "spark_vals": _timeline_vals(asset_series, r["name"]),
            "color": r["color"],
            "leverage": r.get("leverage"),
        })

    # ── "Invested Portfolio" summary row (notional totals + leverage),
    # placed BELOW the invested classes and ABOVE cash — cash is a separate
    # accounting entity that does not participate in the invested base. ──
    _cls_rows = [r for r in asset_rows if not r.get("eur_row")]
    _cash_rows = [r for r in asset_rows if r.get("eur_row")]
    if _cls_rows:
        _tnow = sum(float(r.get("now") or 0.0) for r in _cls_rows)
        _ttgt = sum(float(r.get("target") or 0.0) for r in _cls_rows)
        _ttrend = None
        if asset_series and len(asset_series) >= 2:
            _ttrend = [sum(float(x) for x in b.values()) for b in asset_series]
        total_row = {
            "is_total": True,
            "label_html": "\u2605 Invested Portfolio",
            "now": _tnow,
            "target": _ttgt,
            "leverage": (_tnow / 100.0) if _tnow else None,
            "spark_vals": _ttrend,
            "color": P["accent"],
        }
        # Reorder: classes → Invested Portfolio total → cash.
        asset_rows = _cls_rows + [total_row] + _cash_rows

    # ── Geography rows ──
    geo_rows = [{
        "label_html": _div_label(r["name"], r["color"]),
        "now": r.get("actual_pct_raw"),
        "target": r.get("target_left"),
        "spark_vals": _timeline_vals(geo_series, r["name"]),
        "color": r["color"],
    } for r in geo["rows"]]

    # ── By-holding rows: per-holding-only → portfolio targets (current
    # weight); otherwise the equity/FI sleeve tables. ──
    per_holding_only = getattr(ctx.config, "target_use_per_holding_only", False)
    holding_rows, exits_note, eq_rows, fi_rows = [], "", [], []
    if per_holding_only:
        holding_rows, exits_note = _ph_target_rows(ctx, tol, hold_inv_series)
    else:
        eq_rows = _holding_verif_rows(ctx, tol, hold_series, "per_holding_equity",
                                      ASSET_COLORS.get("Equities", P["accent"]))
        fi_rows = _holding_verif_rows(ctx, tol, hold_series, "per_holding_fi",
                                      ASSET_COLORS.get("Fixed Income", P["accent"]))

    def sub(title):
        return (f'<div style="margin-top:20px;font-size:11px;font-weight:700;letter-spacing:0.06em;'
                f'color:{P["muted"]};text-transform:uppercase;">{title}</div>')

    html = [
        f'<div style="font-size:13px;font-weight:700;letter-spacing:0.08em;color:{P["accent"]};text-transform:uppercase;">Diversification</div>',
    ]
    if asset_rows:
        html.append(sub("By asset class"))
        html.append(_div_table(asset_rows, tol, base=invested_base, show_leverage=True))
        # Notional note: when capital-efficient/leveraged funds push total
        # exposure past 100% of capital, make the leverage explicit. Sum only
        # the per-class rows (exclude the Total and cash rows).
        _tot_notional = sum(float(r.get("now") or 0.0) for r in asset_rows
                            if not r.get("is_total") and not r.get("eur_row"))
        if _tot_notional > 100.6:
            html.append(
                f'<div style="margin-top:6px;font-size:11px;color:{P["muted"]};">'
                f'Total notional exposure <b>{_tot_notional:.0f}%</b> of invested '
                f'capital — above 100% via capital-efficient/leveraged funds '
                f'(e.g. efficient-core 90/60).</div>'
            )
    if geo_rows:
        html.append(sub("By geography"))
        html.append(_div_table(geo_rows, tol, base=equity_base))
    if holding_rows:
        html.append(sub("By holding"))
        html.append(_div_table(holding_rows, tol, base=invested_base))
        if exits_note:
            html.append(exits_note)
    if eq_rows:
        html.append(sub("By holding · Equities"))
        html.append(_div_table(eq_rows, tol, base=equity_base))
    if fi_rows:
        html.append(sub("By holding · Fixed Income"))
        html.append(_div_table(fi_rows, tol, base=fi_base))
    return {"available": True, "html": "".join(html)}


def _clean_ticker(isin: str) -> str:
    """Resolve an ISIN to its Yahoo symbol via the price cache and strip the
    exchange suffix (XDEM.MI → XDEM). Empty when unresolved (e.g. bonds with
    no listing), so callers can fall back to the name alone."""
    if not isin:
        return ""
    from tarzan.data import price_cache as _pc
    sym = _pc.load_resolution(isin) or ""
    # Reuse the single ticker-shortening helper so Holdings, Returns and
    # Historical-risk all strip the exchange suffix identically.
    return _display_ticker(sym) or ""


def _build_holdings(ctx: _NewsletterContext) -> dict:
    """Build holdings grouped by asset class (Excel sort order)."""
    m = ctx.metrics
    df = m.holdings_df
    if df.empty:
        return {"groups": [], "summary": []}

    # Class totals for header summary
    class_totals = df.groupby("asset_class")["current_value"].sum().to_dict()
    class_counts = df.groupby("asset_class").size().to_dict()

    summary = []
    for klass in ASSET_CLASS_ORDER:
        if klass not in class_counts:
            continue
        summary.append({
            "name": klass,
            "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "count": int(class_counts[klass]),
            "label": "positions" if class_counts[klass] != 1 else "position",
        })

    groups = []
    for klass in ASSET_CLASS_ORDER:
        sub = df[df["asset_class"] == klass]
        if sub.empty:
            continue
        # For invested classes the Weight column is reported as % of
        # *invested* value (cash sits outside the invested portfolio).
        # For cash the Weight column is shown as "—" because % of
        # invested is undefined for the cash bucket.
        is_cash_class = klass == "Cash & Cash Equivalents"
        invested_base = m.invested_value if m.invested_value > 0 else 0.0
        rows = []
        for i, (_, h) in enumerate(sub.iterrows()):
            value = float(h["current_value"])
            cls_total = class_totals.get(klass, 1) or 1
            pct_class = value / cls_total * 100
            quantity = float(h.get("quantity", 0) or 0)
            avg_price = float(h.get("avg_purchase_price", 0) or 0)
            gain_pct = h.get("gain_pct")
            gain_eur = h.get("gain_eur")
            if is_cash_class:
                weight_str = "—"
            elif invested_base > 0:
                weight_str = _pct(value / invested_base * 100, decimals=1)
            else:
                weight_str = "—"
            has_gain = gain_pct is not None and not pd.isna(gain_pct)
            rows.append({
                "name": short_instrument_name(h.get("name", ""), 34),
                "ticker": _clean_ticker(h.get("isin", "")),
                "isin": h.get("isin", ""),
                "quantity": quantity,
                "avg_price": _eur(avg_price, 2),
                "value": _eur(value, 2),
                "weight_pct": weight_str,
                "gain_pct": _pct(gain_pct, signed=True) if has_gain else "—",
                "gain_eur": (_eur_smart(gain_eur, signed=True)
                             if gain_eur is not None and not pd.isna(gain_eur) else "—"),
                "gain_color": (PALETTE["green"] if (gain_pct or 0) >= 0 else PALETTE["red"]) if has_gain else PALETTE["muted"],
                "pct_class": _pct(pct_class, decimals=1),
                "alt_bg": i % 2 == 1,
            })
        # Cash is reported as a separate entity, not part of the
        # "invested" portfolio. Skip the share stat for the cash group
        # so it does not appear to compete with invested classes; for
        # everything else, express the share as % of *invested* value
        # (consistent with the convention that cash sits outside the
        # invested allocation, exactly like the Diversification and
        # Optimizer sections).
        is_cash = klass == "Cash & Cash Equivalents"
        total_pct_str: Optional[str] = None
        if not is_cash:
            base = m.invested_value if m.invested_value > 0 else m.total_value
            pct = (class_totals.get(klass, 0) / base * 100) if base > 0 else 0
            total_pct_str = _pct(pct, decimals=1)
        groups.append({
            "name": klass,
            "name_short": "Cash & Cash Equivalents" if klass == "Cash & Cash Equivalents" else klass,
            "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "bg": ASSET_BG.get(klass, PALETTE["accent_bg"]),
            "count": int(class_counts.get(klass, 0)),
            "label": "positions" if class_counts.get(klass, 0) != 1 else "position",
            "total_value": _eur_smart(class_totals.get(klass, 0)),
            "total_pct": total_pct_str,
            "is_cash": is_cash,
            "rows": rows,
        })

    return {"groups": groups, "summary": summary, "total_count": int(len(df))}


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
    period_labels = ["1W", "1M", "3M", "YTD", "1Y", "3Y", "5Y"]

    # Intraday + live-flag inputs for the per-row 1D sparkline cell. Resolve
    # each holding_performance key (often an ISIN) to its Yahoo symbol so the
    # intraday lookup matches, and carry the raw 1D value + live flag.
    from tarzan.data import price_cache as _pc
    _hp_keys = ([str(t) for t in hp["ticker"].dropna().unique() if t]
                if (hp is not None and not hp.empty and "ticker" in hp.columns) else [])
    _resolve = {k: (_pc.load_resolution(k) or k) for k in _hp_keys}
    _snap_intraday = _perf_intraday_map(list({s for s in _resolve.values()}))
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
        if bench_value is None or (isinstance(bench_value, float) and pd.isna(bench_value)):
            return PALETTE["green"] if value >= 0 else PALETTE["red"]
        delta = value - float(bench_value)
        if abs(delta) <= 0.25:
            return PALETTE["amber"]
        return PALETTE["green"] if delta > 0 else PALETTE["red"]

    def _returns_dict(source: dict, *, is_portfolio: bool) -> dict:
        """Per-period ``{value, color}`` map for the shared table renderer."""
        out: dict = {}
        for key in period_keys:
            val = source.get(key) if source else None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                out[key] = {"value": "\u2014", "color": PALETTE["subtle"]}
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                out[key] = {"value": "\u2014", "color": PALETTE["subtle"]}
                continue
            color = (_vs_bench_color(v, ab_bench_returns.get(key)) if is_portfolio
                     else (PALETTE["green"] if v >= 0 else PALETTE["red"]))
            out[key] = {"value": _pct_compact(v, signed=True), "color": color}
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

    # Role per holding from the curated taxonomy (asset_class already on df),
    # so the snapshot groups exactly like the Performance table.
    from tarzan import config as cfg
    _tax = cfg.instrument_taxonomy()

    def _role_for(isin: str, ticker: str) -> str:
        for k in (str(isin or "").strip().upper(),
                  str(ticker or "").split(".")[0].upper()):
            if k and k in _tax and _tax[k][1]:
                return _tax[k][1]
        return "\u2014"

    # Portfolio (highlighted) row. The portfolio has no single ticker, but its
    # holdings trade intraday, so build a value-weighted synthetic intraday
    # path (reusing the already-fetched holdings intraday) for a real 1D
    # sparkline; fall back to the dashed placeholder when unavailable.
    _pf_series = _portfolio_intraday_series(m, resolve=_resolve,
                                            intraday_map=_snap_intraday, raw1d=_raw1d)
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

    # Group holdings by asset class → role.
    grouped: dict = {}
    for _, h in df.iterrows():
        ticker = str(h.get("ticker", "") or "")
        isin = str(h.get("isin", "") or "")
        ac = str(h.get("asset_class", "") or "") or "Other"
        role = _role_for(isin, ticker)
        raw_name = str(h.get("name", "") or ticker)
        display_tk = _clean_ticker(isin) or _display_ticker(ticker) or ""
        sym = _resolve.get(ticker, ticker)
        _, inner = _perf_spark_cell(
            _raw1d.get(ticker), sym, _snap_intraday,
            live=bool(_live1d.get(ticker, False)), prev_label=_prev_lbl)
        grouped.setdefault(ac, {}).setdefault(role, []).append({
            "name_html": _perf_name_html(short_instrument_name(raw_name),
                                         display_tk, []),
            "spark_inner": inner,
            "returns": _returns_dict(perf_by_ticker.get(ticker, {}), is_portfolio=False),
        })

    def _ordered(keys, preferred):
        return ([k for k in preferred if k in keys]
                + [k for k in keys if k not in preferred])

    groups = []
    for ac in _ordered(list(grouped.keys()), _PERF_CLASS_ORDER):
        col = ASSET_COLORS.get(ac, PALETTE["accent"])
        role_list = [(role, grouped[ac][role])
                     for role in _ordered(list(grouped[ac].keys()),
                                          _PERF_ROLE_ORDER.get(ac, []))]
        groups.append((ac, col, role_list))

    return {
        "available": True,
        "table_html": _returns_table_html(period_keys, portfolio, groups),
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


# Performance section grouping. The tracked instruments are sectioned by
# asset class and then by role (from instrument_taxonomy), in a fixed
# sequence that reads growth-engine → diversifiers → defensive. Classes or
# roles not listed here are appended (never dropped), so a new taxonomy
# value still shows up.
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


def _perf_intraday_map(tickers: list[str]) -> dict:
    """Batched intraday close series per ticker for the 1D sparkline.

    Reuses the Markets strip's single-request intraday download. Returns an
    empty dict on any failure so callers transparently fall back to the
    daily-history sparkline."""
    uniq = [t for t in {t for t in tickers if t}]
    if not uniq:
        return {}
    try:
        from tarzan.data.market_quotes import _fetch_intraday_with_fallback
        return _fetch_intraday_with_fallback(uniq)
    except Exception as e:  # noqa: BLE001
        logger.debug("performance intraday fetch failed: %s", e)
        return {}


def _intraday_spark(intra: "pd.Series", baseline: float,
                    w: int = 84, h: int = 18,
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


def _flat_dashed_spark(w: int = 84, h: int = 18) -> str:
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


def _prev_session_label(m) -> str:
    """Report-level 'previous session' date as ``dd/mm`` for the PREV. DAY tag
    — the last completed trading day in the portfolio history (the close the
    non-live 1D moves are measured against). Empty when unavailable."""
    ph = getattr(m, "portfolio_history", None)
    try:
        if ph is not None and len(ph) >= 1:
            today = pd.Timestamp.now().normalize()
            past = [d for d in ph.index if pd.Timestamp(d).normalize() < today]
            d = pd.Timestamp(past[-1]) if past else pd.Timestamp(ph.index[-1])
            return d.strftime("%d/%m")
    except Exception:  # noqa: BLE001
        pass
    return ""


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
    if day_val is None or (isinstance(day_val, float) and pd.isna(day_val)):
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

    intra = intraday_map.get(raw_ticker) if raw_ticker else None
    if intra is not None and len(intra) >= 2:
        # Baseline = previous close, derived from the known 1D return so no
        # extra fetch is needed: prev = last / (1 + day%).
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


_PF_INTRA_KEY = "__PORTFOLIO_INTRADAY__"


def _portfolio_intraday_series(m, resolve: Optional[dict] = None,
                               intraday_map: Optional[dict] = None,
                               raw1d: Optional[dict] = None):
    """Value-weighted intraday level path for the whole portfolio, rebased to
    100 at the previous close, so the Total Portfolio row can show a *real* 1D
    sparkline: its holdings trade intraday even though the portfolio has no
    single ticker.

    Each holding's intraday series is rebased to its own previous close
    (derived from its live 1D %, the same basis the per-row cells use), then
    the % paths are value-weighted by ``weight_pct`` on a common (union) time
    index. Callers that already fetched the holdings' intraday can pass
    ``resolve``/``intraday_map``/``raw1d`` to avoid a second download; missing
    inputs are computed from ``m``. Returns a pandas Series (level) or None
    when no holding has a usable intraday series."""
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
    keys = [str(t) for t in df["ticker"].dropna().unique() if t]
    if resolve is None:
        from tarzan.data import price_cache as _pc
        resolve = {k: (_pc.load_resolution(k) or k) for k in keys}
    if intraday_map is None:
        intraday_map = _perf_intraday_map(list({resolve.get(k, k) for k in keys}))
    if not intraday_map:
        return None
    paths = []  # (weight, pct-path indexed by timestamp)
    for tk, w in zip(df["ticker"], df["weight_pct"]):
        if w is None:
            continue
        sym = resolve.get(str(tk), str(tk))
        intra = intraday_map.get(sym)
        if intra is None or len(intra) < 2:
            continue
        dv = raw1d.get(str(tk))
        try:
            last = float(intra.iloc[-1])
            denom = (1.0 + float(dv) / 100.0) if dv is not None else None
            base_i = last / denom if denom else float(intra.iloc[0])
            if not base_i:
                continue
            p = (intra.astype(float) / base_i - 1.0) * 100.0
            paths.append((float(w), p))
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


def _returns_table_html(period_cols, portfolio: dict, groups: list) -> str:
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
    ncols = len(period_cols) + 2

    def _th(label, first=False, last=False, center=False):
        radius = ("border-top-left-radius:8px;" if first else
                  "border-top-right-radius:8px;" if last else "")
        align = "center" if center else ("left" if first else "right")
        return (f'<td align="{align}" style="padding:8px 6px;background:{P["ink"]};'
                f'color:#FFFFFF;font-size:10px;font-weight:700;letter-spacing:0.04em;'
                f'text-transform:uppercase;{radius}">{label}</td>')

    hcells = [_th("Instrument", first=True), _th("1D / Intraday", center=True)]
    for i, p in enumerate(period_cols):
        hcells.append(_th(p.upper(), last=(i == len(period_cols) - 1)))
    header = "<tr>" + "".join(hcells) + "</tr>"

    def _period_cells(returns_dict, *, weight, bg):
        bgc = f"background:{bg};" if bg else ""
        out = ""
        for p in period_cols:
            cell = returns_dict.get(p, {"value": "\u2014", "color": P["muted"]})
            out += (f'<td align="right" style="padding:8px 6px;{bgc}'
                    f'font-variant-numeric:tabular-nums;color:{cell["color"]};'
                    f'font-weight:{weight};">{cell["value"]}</td>')
        return out

    pbg = P["accent_bg"]
    rows_html = [
        "<tr>"
        f'<td style="padding:10px 8px;background:{pbg};color:{P["accent"]};'
        f'font-weight:700;font-size:12px;">\u2605 {portfolio["name"]}</td>'
        f'<td width="96" align="center" style="padding:6px 8px;background:{pbg};">'
        f'{portfolio.get("spark_inner", "")}</td>'
        + _period_cells(portfolio["returns"], weight=700, bg=pbg)
        + "</tr>"
    ]

    ridx = 0  # running instrument-row index for the alternating background
    for cls, col, role_list in groups:
        for role, insts in role_list:
            if not insts:
                continue
            rows_html.append(
                f'<tr><td colspan="{ncols}" style="padding:8px 8px 4px;'
                f'border-bottom:1px solid {P["border"]};font-size:10px;font-weight:700;'
                f'color:{P["muted"]};letter-spacing:0.02em;">'
                f'<span style="display:inline-block;width:8px;height:8px;background:{col};'
                f'border-radius:2px;vertical-align:middle;margin-right:6px;"></span>'
                f'<span style="color:{col};font-weight:800;text-transform:uppercase;">{cls}</span>'
                f'&nbsp;&middot;&nbsp;{role}</td></tr>'
            )
            for inst in insts:
                ridx += 1
                rowbg = "#FFFFFF" if (ridx % 2 == 1) else P["card_alt"]
                rows_html.append(
                    "<tr>"
                    f'<td style="padding:8px;background:{rowbg};border-bottom:1px solid #F1F2F8;'
                    f'color:{P["ink"]};font-size:12px;">{inst["name_html"]}</td>'
                    f'<td width="96" align="center" style="padding:6px 8px;background:{rowbg};'
                    f'border-bottom:1px solid #F1F2F8;">{inst.get("spark_inner", "")}</td>'
                    + _period_cells(inst["returns"], weight=600, bg=rowbg)
                    + "</tr>"
                )

    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'border="0" style="margin-top:14px;font-size:11px;border-collapse:separate;'
        f'border-spacing:0;">{header}{"".join(rows_html)}</table>'
    )


def _perf_name_html(name: str, ticker: str, tags: list) -> str:
    """Instrument label used in the returns tables: name + ticker chip +
    optional reference tags (α/β, GEO)."""
    P = PALETTE
    tk_chip = ""
    if ticker:
        tk_chip = (
            f'<span style="display:inline-block;margin-left:5px;padding:1px 5px;'
            f'background:{P["page"]};color:{P["muted"]};border:1px solid {P["border"]};'
            f'border-radius:4px;font-size:9px;font-weight:700;'
            f'font-family:SFMono-Regular,Menlo,Consolas,monospace;'
            f'letter-spacing:0.02em;vertical-align:middle;">{ticker}</span>')
    tag_chips = "".join(
        f'<span style="display:inline-block;margin-left:4px;padding:1px 6px;'
        f'background:{t[2]};color:{t[1]};border-radius:4px;font-size:9px;'
        f'font-weight:700;letter-spacing:0.04em;vertical-align:middle;">{t[0]}</span>'
        for t in (tags or []))
    return f"{name}{tk_chip}{tag_chips}"


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

    def _color_sign(value) -> str:
        """Sign-aware color for a period return cell — used on benchmarks."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
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
        if value is None or (isinstance(value, float) and pd.isna(value)):
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
            }
            for p in periods
        }

    def _build_bench_returns_dict(source: dict) -> dict:
        return {
            p: {
                "value": _pct_compact(source.get(p), signed=True),
                "color": _color_sign(source.get(p)),
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
            bare = raw_ticker.split(".")[0].upper() if raw_ticker else ""
            asset_class, role = taxonomy.get(bare, (None, None))
            benchmark_rows.append({
                # Display name goes through the SAME shortener as the holding
                # rows so "iShares Nasdaq 100 UCITS ETF" reads like the rest of
                # the table (tag-matching above uses the raw name, not this).
                "name": short_instrument_name(name),
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
        fallback = []
        prov = m.returns_provenance or {}
        for key in ("synthetic", "carry_flat", "excluded"):
            fallback.extend(prov.get(key, []))
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
            "fallback_count": len(set(fallback)),
        }

    # ── Pre-rendered table (grouped by asset class + role) ──────────────
    # 1D is a Markets-style intraday sparkline (with the % pill on top); the
    # remaining columns are % returns. One batched intraday fetch covers all
    # tracked tickers; instruments without a usable intraday series fall back
    # to a dashed daily sparkline.
    P = PALETTE
    period_cols = ("1w", "1m", "3m", "ytd", "1y", "3y", "5y")
    intraday_map = _perf_intraday_map([r.get("raw_ticker") for r in benchmark_rows])

    def _ordered(keys, preferred):
        seen = [k for k in preferred if k in keys]
        extra = [k for k in keys if k not in preferred]
        return seen + extra

    # Portfolio row (highlighted): a real 1D sparkline from a value-weighted
    # synthetic intraday path over the holdings (the portfolio has no single
    # ticker, but its holdings trade intraday); dashed placeholder when the
    # intraday isn't available.
    _pf_series = _portfolio_intraday_series(m)
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
    grouped: dict = {}
    for r in benchmark_rows:
        grouped.setdefault(r.get("asset_class") or "Other", {}) \
               .setdefault(r.get("role") or "\u2014", []).append(r)
    groups = []
    for ac in _ordered(list(grouped.keys()), _PERF_CLASS_ORDER):
        col = ASSET_COLORS.get(ac, P["accent"])
        role_list = []
        for role in _ordered(list(grouped[ac].keys()), _PERF_ROLE_ORDER.get(ac, [])):
            insts = []
            for r in grouped[ac][role]:
                _, inner = _perf_spark_cell(
                    r.get("d1"), r.get("raw_ticker"), intraday_map,
                    live=bool(r.get("live")), prev_label=_prev_lbl)
                insts.append({
                    "name_html": _perf_name_html(r["name"], r.get("ticker"),
                                                 r.get("tags")),
                    "spark_inner": inner,
                    "returns": r["returns"],
                })
            role_list.append((role, insts))
        groups.append((ac, col, role_list))

    table_html = _returns_table_html(period_cols, portfolio, groups)

    subtitle_html = (
        f'Portfolio vs {ctx.benchmark_alpha_beta or "S&amp;P 500"}: '
        f'<span style="color:{P["green"]};font-weight:700;">&#9679;</span> beat &middot; '
        f'<span style="color:{P["amber"]};font-weight:700;">&#9679;</span> in line &middot; '
        f'<span style="color:{P["red"]};font-weight:700;">&#9679;</span> under'
    )

    return {
        "title": "How markets moved",
        "kicker": "Performance",
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
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return _pct(float(v))

    def _fmt_num(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.2f}"

    # Metric columns, in display order. Tuple: (label, key, is_pct, note).
    # α and β carry a "*" footnote marker because they are referenced to
    # a specific market index. Ulcer Index sits next to Max DD as a
    # duration-aware companion (RMS of drawdowns).
    metric_cols = [
        ("CAGR", "cagr", True, ""),
        ("Vol", "volatility", True, ""),
        ("Sharpe", "sharpe", False, ""),
        ("Sortino", "sortino", False, ""),
        ("Max DD", "max_drawdown", True, ""),
        ("Ulcer", "ulcer_index", True, ""),
        ("VaR 95%", "var_95", True, ""),
        ("CVaR 95%", "cvar_95", True, ""),
        ("\u03b1", "alpha", True, "*"),
        ("\u03b2", "beta", False, "*"),
    ]

    def _cells_from(metrics: dict) -> list[str]:
        out = []
        for _label, key, is_pct, _note in metric_cols:
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

    rows = []
    port = hr.get("portfolio")
    if port:
        rows.append({
            "label": port.get("label", "Your portfolio"),
            "ticker": None,
            "span_label": port.get("span_label", "\u2014"),
            "tags": [],
            "is_portfolio": True,
            "cells": _cells_from(port.get("metrics")),
            "group_header": None,
            "group_color": None,
        })
    # Group instruments by asset_class → role, identical to Performance.
    from collections import OrderedDict
    _grouped: dict[str, dict[str, list]] = OrderedDict()
    for inst in hr.get("instruments", []):
        ac = inst.get("asset_class") or "Other"
        role = inst.get("role") or "\u2014"
        _grouped.setdefault(ac, OrderedDict()).setdefault(role, []).append(inst)
    # Same order as the Performance table (was a verbatim copy of
    # _PERF_CLASS_ORDER — now shares the one registry-backed list).
    _perf_cls_order = _PERF_CLASS_ORDER
    def _cls_sort(ac):
        return _perf_cls_order.index(ac) if ac in _perf_cls_order else 99
    for ac in sorted(_grouped.keys(), key=_cls_sort):
        gc = ASSET_COLORS.get(ac, PALETTE["accent"])
        for role, insts in _grouped[ac].items():
            # One group header per (class, role) block — same as Performance.
            first = True
            for inst in insts:
                rows.append({
                    # Shorten the display label like every other table (tags
                    # still match on the raw label below).
                    "label": short_instrument_name(inst.get("label", "")),
                    "ticker": _display_ticker(inst.get("ticker")),
                    "span_label": inst.get("span_label", "\u2014"),
                    "tags": _tags_for(inst.get("label", "")),
                    "is_portfolio": False,
                    "cells": _cells_from(inst.get("metrics")),
                    "group_header": (ac, role if role != "\u2014" else "") if first else None,
                    "group_color": gc if first else None,
                })
                first = False

    if not rows:
        return {"available": False, "rows": [], "columns": []}

    description = (
        "Each instrument is measured over its full available price history "
        "\u2014 the span is shown next to its name. Your portfolio is a "
        "backtest at today's weights held constant, over the longest window "
        "where every holding with at least 1 year of history overlaps."
    )

    return {
        "available": True,
        "title": "Historical risk profile",
        "subtitle": "Full available history, per instrument",
        "columns": [{"label": label, "note": note}
                    for (label, _k, _p, note) in metric_cols],
        "rows": rows,
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
         "Annualized standard deviation of daily returns. Equity indexes "
         "~15–20%, bonds ~3–7%."),
        ("Sharpe", "sharpe",
         "(CAGR − risk-free rate) / Volatility. Return per unit of total "
         "risk. >1 is good, >2 excellent."),
        ("Sortino", "sortino",
         "Like Sharpe but penalizes only downside volatility. Usually "
         "higher than Sharpe — gap shows good (upside) volatility."),
        ("Max Drawdown", "max_drawdown",
         "Worst peak-to-trough loss over the period. -20% is typical for "
         "diversified equity; deeper drops signal concentration risk."),
        ("Ulcer Index", "ulcer_index",
         "Root-mean-square of drawdowns from the running peak — captures both "
         "depth and time spent underwater. Lower is smoother; penalizes long "
         "slumps more than a one-point Max DD."),
        ("VaR 95%", "var_pct",
         "Daily loss exceeded only 5% of the time (historical sim). "
         "Non-parametric — no normal-distribution assumption."),
        ("CVaR 95%", "cvar_pct",
         "Average loss on the worst 5% of days. More negative than VaR — "
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


def _optimizer_plan_ctx(m: PortfolioMetrics, suggestions: list) -> dict:
    """Build one optimizer plan's render context (actions + totals) from a
    list of rebalancing suggestions."""
    df = m.holdings_df
    total_buy = sum(float(s["amount_eur"]) for s in suggestions
                    if s["direction"].lower() == "buy")
    total_sell = sum(float(s["amount_eur"]) for s in suggestions
                     if s["direction"].lower() == "sell")

    actions = []
    for s in sorted(suggestions, key=lambda s: -float(s["amount_eur"])):
        direction = s["direction"].upper()
        amount = float(s["amount_eur"])
        pct_of_port = (amount / m.total_value * 100) if m.total_value > 0 else 0.0
        ticker = s.get("ticker", "")
        klass = "Equities"
        if not df.empty:
            match = df[df["ticker"] == ticker]
            if not match.empty:
                klass = match["asset_class"].iloc[0]
        actions.append({
            "direction": direction,
            "direction_color": PALETTE["green"] if direction == "BUY" else PALETTE["red"],
            "direction_bg": PALETTE["green_bg"] if direction == "BUY" else PALETTE["red_bg"],
            "name": s.get("name", ""),
            # Clean pin ticker (no exchange suffix), same as Holdings/By holding:
            # resolve ISIN→symbol via price cache, else strip the suffix off
            # the raw ticker. Falls back to empty (no pin) for unresolved.
            "ticker": (_clean_ticker(s.get("isin", ""))
                       or _clean_ticker(ticker)
                       or _display_ticker(ticker)
                       or ""),
            "isin": s.get("isin", ""),
            "asset_class": klass,
            "asset_color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "amount": _eur(amount, decimals=2),
            "pct_of_portfolio": _pct(pct_of_port, decimals=1),
            "reason": s.get("reason", ""),
        })

    n_total = len(suggestions)
    n_buy = sum(1 for s in suggestions if s["direction"].lower() == "buy")
    return {
        "actions": actions,
        "n_total": n_total,
        "n_buy": n_buy,
        "n_sell": n_total - n_buy,
        "total_buy": _eur_smart(total_buy),
        "total_sell": _eur_smart(total_sell),
        "net": _eur_smart(total_buy - total_sell, signed=True),
        "net_color": (PALETTE["green"] if (total_buy - total_sell) >= 0
                      else PALETTE["red"]),
    }


def _build_optimizer(ctx: _NewsletterContext) -> dict:
    """Build the suggested-action card with BOTH rebalancing plans (buy-only
    and buy & sell), each ordered by absolute amount.

    Reads ``metrics.rebalancing_plans`` (always computed by the engine); falls
    back to the single ``rebalancing_suggestions`` set for back-compat.
    """
    m = ctx.metrics
    plans_src = getattr(m, "rebalancing_plans", None)

    if plans_src:
        plans = []
        for p in plans_src:
            pc = _optimizer_plan_ctx(m, list(p.get("suggestions") or []))
            pc["label"] = p.get("label", "")
            pc["no_sell"] = p.get("no_sell")
            plans.append(pc)
        if not any(pc["actions"] for pc in plans):
            return {"available": False}
        return {"available": True, "plans": plans}

    # Back-compat: single plan.
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions:
        return {"available": False}
    pc = _optimizer_plan_ctx(m, suggestions)
    pc["label"] = "Suggested actions"
    pc["no_sell"] = None
    return {"available": True, "plans": [pc]}



def _build_return_contrib(ctx: _NewsletterContext) -> dict:
    """Build winners / laggards by return contribution.

    Each item carries a ``bar_pct`` (0–100) scaled to the largest absolute
    contribution among the shown movers, so the template can draw
    magnitude bars that are comparable across winners and laggards.
    """
    m = ctx.metrics
    df = m.holdings_df
    if df.empty:
        return {"winners": [], "laggards": []}

    rows = []
    for _, r in df.iterrows():
        contrib = float(r.get("weight_pct", 0) or 0) * float(r.get("gain_pct", 0) or 0) / 100
        rows.append({"name": r.get("name", ""), "ticker": r.get("ticker", ""), "contrib": contrib})
    rows.sort(key=lambda x: -x["contrib"])

    top = rows[:3]
    bottom = list(reversed(rows[-3:]))  # worst first
    max_abs = max((abs(r["contrib"]) for r in (top + bottom)), default=0.0) or 1.0

    def _item(r: dict) -> dict:
        return {
            "name": r["name"],
            "value": _pct(r["contrib"], signed=True),
            "bar_pct": round(min(100.0, abs(r["contrib"]) / max_abs * 100.0), 1),
            "is_positive": r["contrib"] >= 0,
        }

    return {
        "winners": [_item(r) for r in top],
        "laggards": [_item(r) for r in bottom],
    }


def _build_preheader(ctx: _NewsletterContext, hero: dict) -> str:
    """Preview text shown in inbox preview."""
    m = ctx.metrics
    n_actions = len(m.rebalancing_suggestions or [])
    parts = [f"Portfolio at {hero['total_value']} ({hero['gain_pct']} since inception)"]
    if n_actions > 0:
        parts.append("rebalancing suggested")
    parts.append(f"{len(m.holdings_df)} holdings tracked")
    return " · ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def _colorize_pct(text: str) -> str:
    """HTML-escape ``text`` and wrap signed percentages AND percentage-point
    figures (e.g. +0.81%, -1.2%, +0.92pp, -4.53pp) in green/red spans, so both
    the market-context note (uses %) and the divergence note (uses pp) show
    moves in colour. Unsigned percentages (yield levels like 4.38%) are left
    neutral."""
    import html as _html
    import re as _re
    if not text:
        return ""
    esc = _html.escape(text)

    def _wrap(m):
        tok = m.group(0)
        neg = tok[0] in "-\u2212"
        col = PALETTE["red"] if neg else PALETTE["green"]
        return f'<span style="color:{col};font-weight:700;">{tok}</span>'

    # Signed number followed by %, pp, or "percentage point(s)" (bare beta like
    # "0.69" stays neutral). The model sometimes spells out "percentage points"
    # instead of "pp", so match both.
    return _re.sub(
        r"[+\-\u2212]\d+(?:[.,]\d+)?\s?(?:%|pp|percentage points?)", _wrap, esc)


def build_context(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: str = "S&P 500",
    benchmark_geo: str = "MSCI ACWI",
    ai_summary: Optional[str] = None,
) -> dict[str, Any]:
    """Build the full Jinja2 context dict for the newsletter template.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark (from constants.yaml).
        benchmark_geo: Display name of geographic allocation benchmark.

    Returns:
        A dict with all keys consumed by ``portfolio_digest.html.j2``.
    """
    # Reset the per-render SVG clipPath id counters so a render's element ids
    # depend only on how many charts it draws, not on how many newsletters the
    # process rendered before it. Without this, two renders in one process
    # emit different ids (dg1 vs dg2, ...), making the HTML non-reproducible —
    # which defeats deterministic mode. Ids are internal references (visually
    # invisible) and each render is a standalone document, so resetting is safe
    # and never collides.
    global _day_spark_uid, _dual_uid
    _day_spark_uid = 0
    _dual_uid = 0
    nctx = _NewsletterContext(
        metrics=metrics,
        config=config,
        issue_number=issue_number,
        benchmark_alpha_beta=benchmark_alpha_beta,
        benchmark_geo=benchmark_geo,
    )
    hero = _build_hero(nctx)
    return {
        "palette": PALETTE,
        "header": _build_header(nctx),
        "headline": _build_headline(nctx, hero),
        "hero": hero,
        "performance30": _build_performance30(nctx),
        "ai_summary": ai_summary,
        "ai_summary_html": _colorize_pct(ai_summary) if ai_summary else None,
        "movers": _build_movers(nctx),
        "diversification": _build_diversification(nctx),
        "holdings": _build_holdings(nctx),
        "returns_snapshot": _build_returns_snapshot(nctx),
        "performance": _build_performance(nctx),
        "markets": _build_markets(nctx),
        "risk_profile": _build_risk_profile(nctx),
        "optimizer": _build_optimizer(nctx),
        "return_contrib": _build_return_contrib(nctx),
        "tax_note": _build_tax_note(nctx),
        "methodology": _build_methodology(nctx),
        "preheader": _build_preheader(nctx, hero),
        "footer": {
            # Pinned stamp in deterministic mode so the header does not vary
            # run-to-run (live now() otherwise).
            "generated_at": _runtime.now_stamp("%d %b %Y, %H:%M"),
            "version": "v2.0",
        },
    }


def render_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: Optional[str] = None,
    benchmark_geo: Optional[str] = None,
    ai_summary: Optional[str] = None,
) -> str:
    """Render the newsletter HTML to a string.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark. When None it is
            resolved from configuration (instrument_taxonomy.csv
            ``is_benchmark_alpha_beta``) so the label/tag always match the
            benchmark the engine actually computed α/β against.
        benchmark_geo: Display name of geographic allocation benchmark. When
            None it is resolved from configuration
            (``is_benchmark_geo``).

    Returns:
        The full HTML newsletter as a single string.
    """
    # Resolve benchmark display names from config when not explicitly passed.
    # This closes a mismatch where the α/β footnote/tag could name a different
    # index than the β=1.00 row the engine produced.
    from tarzan import config as _cfg
    if benchmark_alpha_beta is None:
        try:
            benchmark_alpha_beta = _cfg.benchmark_beta_name()
        except Exception:
            benchmark_alpha_beta = "S&P 500"
    if benchmark_geo is None:
        try:
            benchmark_geo = _cfg.benchmark_geo_allocation()
        except Exception:
            benchmark_geo = "MSCI ACWI"

    # Resolve the optional AI market-context summary when not explicitly passed,
    # so every caller (CLI + email) renders the SAME newsletter. generate_summary
    # is fully self-guarding: it returns None when no GEMINI_API_KEY is set, in a
    # deterministic run, or on any error — it never raises and never blocks the
    # render. Pass ai_summary="" to force the block off even when a key exists.
    if ai_summary is None:
        from tarzan.export.ai_summary import generate_summary
        ai_summary = generate_summary(metrics, config)

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("portfolio_digest.html.j2")
    context = build_context(
        metrics, config, issue_number, benchmark_alpha_beta, benchmark_geo,
        ai_summary=ai_summary,
    )
    return template.render(**context)


def generate_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    output_dir: str,
    issue_number: int = 1,
    benchmark_alpha_beta: Optional[str] = None,
    benchmark_geo: Optional[str] = None,
) -> str:
    """Render the newsletter and write it to disk.

    Writes ``portfolio_digest_<YYYYMMDD_HHMM>.html`` into ``output_dir``. The
    rendering goes through :func:`render_newsletter`, so a CLI run and an
    emailed send produce the same HTML (benchmark names and the optional AI
    market-context summary are resolved there).

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        output_dir: Directory for the output file.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark. When None,
            resolved from configuration (so labels match the engine).
        benchmark_geo: Display name of geographic allocation benchmark.
            When None, resolved from configuration.

    Returns:
        Path to the generated HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filepath = os.path.join(output_dir, f"portfolio_digest_{date_str}.html")
    html = render_newsletter(
        metrics, config, issue_number, benchmark_alpha_beta, benchmark_geo,
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Newsletter written to %s", filepath)
    return filepath
