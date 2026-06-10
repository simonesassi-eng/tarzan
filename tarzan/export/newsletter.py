"""Generate the portfolio digest newsletter (HTML email).

This module renders an email-safe HTML newsletter from a PortfolioMetrics
object, using a Jinja2 template. The output mirrors the look-and-feel of
the Excel dashboard but is optimised for inbox consumption: 600px wide,
table-based layout, inline CSS, no JavaScript, no external resources.

The newsletter has the following structure:
    1. Header (brand + date + issue)
    2. Hero (total value + chip + KPI grid + 30-day sparkline)
    3. Smart insights (3 takeaways: action, risk, win)
    4. Movers this week (best & worst by 1W return)
    5. Allocation by asset class (with stacked bar + per-class rows)
    6. Geographic exposure (equity, with target & ACWI ticks)
    7. Holdings (grouped by class, sorted as per Excel)
    8. Performance (returns vs benchmarks)
    9. Risk profile (chips + vs S&P 500 + vs MSCI ACWI)
   10. Suggested action (Optimizer)
   11. Return contribution (winners / laggards)
   12. CTA + footer

Public entry point: ``generate_newsletter(metrics, config, output_dir)``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.export._format import (
    ASSET_CLASS_BG,
    ASSET_CLASS_COLORS,
    GEO_COLORS as _GEO_COLORS,
    css,
)

logger = logging.getLogger(__name__)


# ── Palette (mirrors tarzan.export.excel.C / ASSET_COLORS / GEO_COLORS) ───────

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

# Asset class display order in the newsletter Holdings section.
# Cash is shown after Gold so the invested asset classes flow visually
# from highest-risk equity down to commodities/crypto/alternative; cash
# is reported as a separate accounting entity (no "% of portfolio" so
# it does not appear to compete with invested classes). Any asset class
# not listed here is appended (never silently dropped from the report).
_NEWSLETTER_CLASS_ORDER = [
    "Equities", "Fixed Income", "Gold", "Cash & Cash Equivalents",
    "Commodities", "Crypto", "Alternative",
]
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


def _eur_smart(amount: Optional[float], signed: bool = False) -> str:
    """Compact EUR formatter — see :func:`tarzan.export._format.eur_smart`.

    Kept as a thin wrapper for backwards compatibility with the
    several call sites in this module.
    """
    from tarzan.export._format import eur_smart as _impl
    return _impl(amount, signed=signed)


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
        n = len(suggestions)
        parts.append(
            f"and the optimizer suggests {n} rebalancing action"
            f"{'s' if n != 1 else ''} below"
        )
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

    Issue number is computed dynamically: weeks since
    ``portfolio_inception_date`` when available, otherwise the ISO week
    of the current year. The explicit ``ctx.issue_number`` value
    overrides this only when greater than 1, so callers wishing to
    pin a specific number still can.
    """
    now = datetime.now()
    issue_number = ctx.issue_number
    if issue_number <= 1 and ctx.config.portfolio_inception_date:
        try:
            inception = pd.to_datetime(ctx.config.portfolio_inception_date)
            weeks = max(1, int((now - inception.to_pydatetime()).days // 7) + 1)
            issue_number = weeks
        except Exception:
            issue_number = now.isocalendar().week
    elif issue_number <= 1:
        issue_number = now.isocalendar().week
    return {
        "date_short": now.strftime("%a, %d %b %Y"),
        "issue_number": issue_number,
        "inception_date": ctx.config.portfolio_inception_date or "",
    }


def _build_hero(ctx: _NewsletterContext) -> dict:
    m = ctx.metrics
    cfg = ctx.config
    cost = float(m.holdings_df["cost_basis_eur"].sum()) if not m.holdings_df.empty else 0.0
    total_gain = m.total_value - cost
    gain_pct = (total_gain / cost * 100) if cost > 0 else 0.0
    invested_pct = (m.invested_value / m.total_value * 100) if m.total_value > 0 else 0.0
    # 1W return is sourced from performance_full (same series as the
    # Performance section and the Excel Performance tab) so the inbox
    # preview and Hero stay consistent with the metrics shown below.
    perf_full = m.performance_full or {}
    week_return = float(perf_full.get("1w") or 0.0)
    week_eur = m.total_value * week_return / 100 if week_return else 0.0

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

    n_act_str = f"{n_actions} action{'s' if n_actions != 1 else ''}"
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
        # Actions to take. Show only the action count — the technical
        # detail (drift pp, tolerance, "solved at ±X%") is intentionally
        # omitted so the KPI communicates just the action.
        rebal_label = "Action"
        rebal_sublabel = n_act_str
        # Amber for a moderate plan, red when drift is well beyond tol.
        if max_abs_delta > 2 * tol:
            rebal_color, rebal_bg = PALETTE["red"], PALETTE["red_bg"]
        else:
            rebal_color, rebal_bg = PALETTE["amber"], PALETTE["amber_bg"]

    return {
        # Hero big number keeps the full amount (€214,671.72): the
        # entire visual hierarchy of the Status section depends on it.
        "total_value": _eur(m.total_value),
        # Everything else uses the compact form (€9.6k / €215k / €1.2M)
        # so dense rows do not wrap.
        "total_gain": _eur_smart(total_gain, signed=True),
        "gain_pct": _pct(gain_pct, signed=True),
        "is_positive": total_gain >= 0,
        "invested_value": _eur_smart(m.invested_value),
        "invested_pct": _pct(invested_pct, decimals=1),
        "cash_value": _eur_smart(m.cash_value),
        "cash_msg": cash_msg,
        "cash_msg_color": cash_msg_color,
        "week_return_eur": _eur_smart(week_eur, signed=True),
        "week_return_pct": _pct(week_return, signed=True),
        "week_is_positive": week_eur >= 0,
        # Rebalance status KPI replaces the "This Week" KPI which was
        # already covered by the TL;DR headline above.
        "rebal_label": rebal_label,
        "rebal_sublabel": rebal_sublabel,
        "rebal_color": rebal_color,
        "rebal_bg": rebal_bg,
    }


def _build_sparkline(ctx: _NewsletterContext, n_days: int = 30) -> dict:
    """Build the 30-day sparkline data (start/end values + bar heights).

    Includes a "vs benchmark" pill computed from the α/β benchmark over
    the same 30-day window so the user can see at a glance whether the
    portfolio outperformed the market for the period.
    """
    m = ctx.metrics
    history = m.portfolio_history
    if history is None or len(history) < 2:
        # Synthesise a flat line at current value so the section still renders.
        return {
            "available": False,
            "start_label": "",
            "end_label": "",
            "start_value": _eur_smart(m.total_value),
            "end_value": _eur_smart(m.total_value),
            "change_pct": "0.00%",
            "is_positive": True,
            "bars": [{"height": 22} for _ in range(n_days)],
            "vs_bench": None,
        }

    last = history.tail(n_days)
    start_v, end_v = float(last.iloc[0]), float(last.iloc[-1])
    change_pct = (end_v - start_v) / start_v * 100 if start_v > 0 else 0.0
    vmin, vmax = float(last.min()), float(last.max())
    rng = max(vmax - vmin, 1.0)

    # Map values to bar heights between 12 and 44 px.
    bars = []
    for v in last.values:
        h = 12 + (float(v) - vmin) / rng * 32
        bars.append({"height": int(round(h))})

    # vs-benchmark pill: compare same-window % change of the portfolio
    # against the α/β benchmark. Uses benchmark_histories when present
    # (already aligned to portfolio start in initial-value units).
    vs_bench: Optional[dict] = None
    bench_name = ctx.benchmark_alpha_beta or "S&P 500"
    bench_hist = (m.benchmark_histories or {}).get(bench_name)
    if bench_hist is not None and len(bench_hist) >= 2:
        bench_window = bench_hist.tail(n_days)
        if len(bench_window) >= 2:
            bs, be = float(bench_window.iloc[0]), float(bench_window.iloc[-1])
            if bs > 0:
                bench_pct = (be - bs) / bs * 100
                delta = change_pct - bench_pct
                if abs(delta) <= 0.25:
                    color = PALETTE["amber"]
                    bg = PALETTE["amber_bg"]
                    icon = "●"
                elif delta > 0:
                    color = PALETTE["green"]
                    bg = PALETTE["green_bg"]
                    icon = "▲"
                else:
                    color = PALETTE["red"]
                    bg = PALETTE["red_bg"]
                    icon = "▼"
                vs_bench = {
                    "label": f"vs {bench_name}",
                    "delta": f"{icon} {_signed_pp(delta, decimals=2)} pp",
                    "color": color,
                    "bg": bg,
                }

    return {
        "available": True,
        "start_label": last.index[0].strftime("%b %d") if hasattr(last.index[0], "strftime") else "",
        "end_label": last.index[-1].strftime("%b %d") if hasattr(last.index[-1], "strftime") else "",
        "start_value": _eur_smart(start_v),
        "end_value": _eur_smart(end_v),
        "change_pct": _pct(change_pct, signed=True),
        "is_positive": change_pct >= 0,
        "bars": bars,
        "vs_bench": vs_bench,
    }


def _build_allocation(ctx: _NewsletterContext) -> dict:
    """Build asset-class allocation rows (Excel Dashboard pattern)."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_target_tolerance_pctg

    targets = cfg.invested_allocation_targets_pctg or {}
    alloc_df = m.allocation_by_class

    rows = []
    for klass in ASSET_CLASS_ORDER:
        if alloc_df.empty:
            continue
        match = alloc_df[alloc_df["category"] == klass]
        if match.empty:
            continue
        actual = float(match["weight_pct"].iloc[0])
        target = targets.get(klass)
        delta = actual - target if target is not None else None
        sema = _semaphore(delta, tol)
        rows.append({
            "name": klass,
            "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
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
    }


def _build_geography(ctx: _NewsletterContext) -> dict:
    """Build geographic equity rows with target & ACWI ticks."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_target_tolerance_pctg

    targets = cfg.equity_geo_targets_pctg or {}
    geo_df = m.allocation_by_geo
    acwi = m.acwi_geo or {}

    # Order regions consistently (descending by actual)
    rows = []
    if not geo_df.empty:
        sorted_geo = geo_df.sort_values("weight_pct", ascending=False)
        for _, r in sorted_geo.iterrows():
            region = r["category"]
            actual = float(r["weight_pct"])
            target = targets.get(region)
            acwi_v = acwi.get(region)
            delta_target = actual - target if target is not None else None
            sema = _semaphore(delta_target, tol)
            rows.append({
                "name": region,
                "color": GEO_COLORS.get(region, PALETTE["accent"]),
                "actual_pct": _pct_smart(actual),
                "target_pct": _pct_smart(target) if target is not None else "—",
                "acwi_pct": _pct_smart(acwi_v) if acwi_v is not None else "—",
                "delta": _signed_pp(delta_target) if delta_target is not None else "—",
                "delta_color": _semaphore_color(sema),
                "bar_width": min(max(actual, 1), 100),
                "target_left": min(max(target or 0, 0), 100),
                "acwi_left": min(max(acwi_v or 0, 0), 100),
            })

    # Stacked equity bar
    stacked = [{"color": r["color"], "width": r["bar_width"]} for r in rows if r["bar_width"] > 0]

    return {
        "rows": rows,
        "stacked": stacked,
        "benchmark_name": ctx.benchmark_geo,
    }


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
            if is_cash_class:
                weight_str = "—"
            elif invested_base > 0:
                weight_str = _pct(value / invested_base * 100, decimals=1)
            else:
                weight_str = "—"
            rows.append({
                "name": h.get("name", ""),
                "ticker": h.get("ticker", ""),
                "isin": h.get("isin", ""),
                "quantity": quantity,
                "avg_price": _eur(avg_price, 2),
                "value": _eur(value, 2),
                "weight_pct": weight_str,
                "gain_pct": _pct(gain_pct, signed=True) if gain_pct is not None and not pd.isna(gain_pct) else "—",
                "gain_color": (PALETTE["green"] if (gain_pct or 0) >= 0 else PALETTE["red"]) if gain_pct is not None and not pd.isna(gain_pct) else PALETTE["muted"],
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

    # Keep these aligned with ``_build_performance``'s ``periods`` tuple
    # so both tables always show the same columns in the same order.
    period_keys = ["1d", "1w", "1m", "3m", "ytd", "1y", "3y", "5y"]
    period_labels = ["1D", "1W", "1M", "3M", "YTD", "1Y", "3Y", "5Y"]

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

    def _row(name: str, ticker: str, asset_class: str, source: dict, *,
             is_portfolio: bool = False, is_benchmark: bool = False) -> dict:
        cells = []
        for key in period_keys:
            val = source.get(key) if source else None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append({"value": "—", "color": PALETTE["subtle"], "is_positive": True})
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                cells.append({"value": "—", "color": PALETTE["subtle"], "is_positive": True})
                continue
            if is_portfolio:
                color = _vs_bench_color(v, ab_bench_returns.get(key))
            else:
                color = PALETTE["green"] if v >= 0 else PALETTE["red"]
            cells.append({
                "value": _pct_compact(v, signed=True),
                "color": color,
                "is_positive": v >= 0,
            })
        return {
            "name": name,
            "ticker": ticker,
            "asset_class": asset_class,
            "asset_color": ASSET_COLORS.get(asset_class, PALETTE["accent"]),
            "is_portfolio": is_portfolio,
            "is_benchmark": is_benchmark,
            "cells": cells,
        }

    rows: list[dict] = []
    rows.append(_row("Total Portfolio", "", "", port_full, is_portfolio=True))

    # Use ``holdings_df`` as the source of truth for ordering and
    # membership, then enrich each row with the per-period returns
    # from ``holding_performance``. Going holdings-first guarantees
    # the snapshot lists every position the user sees in the
    # "All positions" table — even the ones (typically illiquid
    # single bonds) that Yahoo cannot price, which would otherwise
    # be silently dropped because ``holding_performance`` only
    # contains tickers with usable history.
    df = m.holdings_df
    if df is None or df.empty:
        return {
            "available": False,
            "period_labels": period_labels,
            "history_label": history_label,
            "benchmark_alpha_beta": ab_bench_name,
            "rows": rows,
        }

    perf_by_ticker: dict[str, dict] = {}
    if hp is not None and not hp.empty and "ticker" in hp.columns:
        type_col = hp["type"].astype(str).str.lower() if "type" in hp.columns else None
        is_holding = type_col.str.contains("portfolio") if type_col is not None else None
        holdings_perf = hp[is_holding] if is_holding is not None else hp
        for _, pr in holdings_perf.iterrows():
            perf_by_ticker[str(pr.get("ticker", ""))] = {
                k: pr.get(k) for k in period_keys
            }

    for _, h in df.iterrows():
        ticker = str(h.get("ticker", "") or "")
        raw_name = str(h.get("name", "") or ticker)
        rows.append(_row(
            short_instrument_name(raw_name),
            ticker,
            str(h.get("asset_class", "") or ""),
            perf_by_ticker.get(ticker, {}),
        ))

    return {
        "available": len(rows) > 1,  # at least one holding/benchmark beyond the portfolio row
        "period_labels": period_labels,
        "history_label": history_label,
        "benchmark_alpha_beta": ab_bench_name,
        "rows": rows,
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


def _build_smart_insights(ctx: _NewsletterContext) -> list[dict]:
    """Generate up to 3 conditional insights — only those genuinely
    worth surfacing are emitted, so the Signals section silently
    contracts when there is nothing to say.

    Slots, in priority order:
      1. Observation (allocation drift) — contextual, NOT prescriptive.
         Concrete trades live in the Optimizer section below; this one
         only explains the drift in plain language.
      2. Risk (rebalance infeasible / concentration top-2 / drawdown /
         cash buffer breach) — the strongest active risk signal that
         crosses a threshold.
      3. Performance (vs MSCI ACWI) — symmetric: Win when beating on a
         risk-adjusted basis, Underperform when below the noise floor,
         silent when in line.
    """
    m = ctx.metrics
    cfg = ctx.config
    tol = float(cfg.rebalancing_target_tolerance_pctg or 0.0)
    insights: list[dict] = []

    # 1. Observation: largest non-cash drift vs target. Phrased as a
    # contextual observation, not as a trade instruction.
    if m.goal_deltas is not None and not m.goal_deltas.empty:
        non_cash = m.goal_deltas[m.goal_deltas["type"] != "cash"]
        if not non_cash.empty:
            largest = non_cash.iloc[non_cash["delta_pct"].abs().argmax()]
            delta = float(largest["delta_pct"])
            cat = largest["category"]
            drift_type = str(largest["type"])
            if abs(delta) > tol:
                direction_word = "above" if delta > 0 else "below"
                # Light explanation to give context without prescribing
                # a trade. The Optimizer section already lists the
                # concrete actions.
                if drift_type == "asset_class":
                    scope = "asset-class allocation"
                    body_hint = (
                        "Typical for rising markets — riskier classes drift up."
                        if delta > 0
                        else "Allocation has eroded relative to target. Optimizer suggests how to rebalance."
                    )
                elif drift_type.startswith("geography"):
                    scope = "equity geography"
                    body_hint = (
                        "Currency, valuations and concentration in regional ETFs all contribute to drift."
                    )
                else:
                    scope = "per-holding allocation"
                    body_hint = (
                        "See the Optimizer section below for concrete trade recommendations."
                    )
                insights.append({
                    "kind": "observation",
                    "icon": "📊",
                    "color": PALETTE["accent"],
                    "bg": PALETTE["accent_bg"],
                    "category": "Observation · Drift",
                    "headline": (
                        f"{cat} is {abs(delta):.1f} pp {direction_word} target — "
                        f"largest drift in your {scope}."
                    ),
                    "body": body_hint,
                })

    # 2. Risk: pick the strongest active risk among (a) infeasible
    # rebalance, (b) concentration, (c) deep drawdown, (d) cash buffer
    # breach. Only the most actionable one is emitted to keep the
    # section tight.
    risk_pf = m.performance_full or {}
    risk_emitted = False

    # 2a. Rebalance infeasible — the LP could not find a plan within
    # the configured tolerance ceiling. Surfaces a strategic note
    # because this is upstream of every concrete trade decision.
    if not risk_emitted and m.rebalancing_verifications and any(
        v.get("no_solution") for v in m.rebalancing_verifications
    ):
        max_tol = float(cfg.rebalancing_target_tolerance_pctg or 0.0)
        insights.append({
            "kind": "risk",
            "icon": "⚠",
            "color": PALETTE["red"],
            "bg": PALETTE["red_bg"],
            "category": "Risk · Rebalance infeasible",
            "headline": (
                f"No feasible rebalance within \u00b1{max_tol:.1f}% tolerance."
            ),
            "body": (
                "At least one allocation drift is beyond the ceiling the "
                "optimizer is allowed to plan around. Either raise "
                "rebalancing_target_tolerance_pctg in your config to give "
                "the solver more room, or relax overly tight per-holding "
                "targets — see the Optimizer tab in the Excel report."
            ),
        })
        risk_emitted = True

    # 2a-bis. Rebalance solved at a relaxed tolerance — the LP found a
    # plan only by widening the tolerance beyond the user's configured
    # ceiling. Tell the user explicitly, not silently, so they can
    # decide whether the relaxed solution is acceptable.
    if not risk_emitted and m.rebalancing_verifications:
        relaxed_v = next(
            (v for v in m.rebalancing_verifications if v.get("relaxed")),
            None,
        )
        if relaxed_v is not None:
            used_tol = float(relaxed_v.get("tolerance") or 0.0)
            cfg_tol = float(relaxed_v.get("configured_max_tolerance") or 0.0)
            insights.append({
                "kind": "risk",
                "icon": "⚠",
                "color": PALETTE["amber"],
                "bg": PALETTE["amber_bg"],
                "category": "Risk · Tolerance relaxed",
                "headline": (
                    f"Optimizer relaxed tolerance to \u00b1{used_tol:.2f}% "
                    f"(configured \u00b1{cfg_tol:.1f}%) to find a plan."
                ),
                "body": (
                    "No feasible rebalance existed inside your configured "
                    "ceiling. The plan below uses a wider tolerance — "
                    "review per-holding targets if this is uncomfortable, "
                    "or raise rebalancing_target_tolerance_pctg to make this "
                    "the official ceiling."
                ),
            })
            risk_emitted = True

    # 2b. Concentration: top-2 weight ≥ 30% of portfolio.
    if not m.holdings_df.empty and len(m.holdings_df) >= 2 and not risk_emitted:
        top2 = m.holdings_df.nlargest(2, "weight_pct")
        top2_pct = float(top2["weight_pct"].sum())
        if top2_pct >= 30:
            names = " · ".join(
                f"{r['ticker']} ({_pct(float(r['weight_pct']), 1)})"
                for _, r in top2.iterrows()
            )
            insights.append({
                "kind": "risk",
                "icon": "◎",
                "color": PALETTE["amber"],
                "bg": PALETTE["amber_bg"],
                "category": "Risk · Concentration",
                "headline": f"Top 2 positions hold {top2_pct:.0f}% of the portfolio.",
                "body": (
                    f"{names}. Concentrated weight amplifies portfolio-level swings; "
                    "watch for overlap if both holdings are in the same asset class."
                ),
            })
            risk_emitted = True

    # 2b. Deep drawdown: max DD beyond -25%.
    max_dd = risk_pf.get("max_drawdown")
    if (max_dd is not None and not (isinstance(max_dd, float) and pd.isna(max_dd))
            and float(max_dd) <= -25.0 and not risk_emitted):
        insights.append({
            "kind": "risk",
            "icon": "▼",
            "color": PALETTE["amber"],
            "bg": PALETTE["amber_bg"],
            "category": "Risk · Drawdown",
            "headline": f"Max drawdown reached {float(max_dd):.1f}% on the 5Y window.",
            "body": (
                "Deeper than the typical -20% baseline for diversified equity. "
                "Consider whether your position sizing matches your stomach for that scenario."
            ),
        })
        risk_emitted = True

    # 2c. Cash buffer significantly off target (≥30% relative deviation).
    cash_target = float(cfg.target_cash_buffer_eur or 0.0)
    if cash_target > 0 and not risk_emitted:
        cash_delta_eur = m.cash_value - cash_target
        rel_dev = abs(cash_delta_eur) / cash_target * 100
        if rel_dev >= 30:
            sign_word = "above" if cash_delta_eur > 0 else "below"
            insights.append({
                "kind": "risk",
                "icon": "💧",
                "color": PALETTE["amber"],
                "bg": PALETTE["amber_bg"],
                "category": "Risk · Cash buffer",
                "headline": f"Cash is {_eur_smart(abs(cash_delta_eur))} {sign_word} target ({rel_dev:.0f}% deviation).",
                "body": (
                    "A larger-than-usual cash gap can drag returns "
                    "(if above target) or erode the safety buffer (if below). "
                    "Direct your next contribution accordingly."
                ),
            })
            risk_emitted = True

    # 3. Performance vs MSCI ACWI — symmetric (Win / Underperform / silent).
    sharpe = risk_pf.get("sharpe")
    cagr = risk_pf.get("cagr")
    vol = risk_pf.get("volatility")
    bench_cmp = m.benchmark_comparison
    if not bench_cmp.empty and sharpe is not None and cagr is not None and vol is not None:
        name_col = None
        for candidate in ("benchmark", "name"):
            if candidate in bench_cmp.columns:
                name_col = candidate
                break
        if name_col is not None:
            acwi_row = bench_cmp[
                bench_cmp[name_col].astype(str).str.contains(
                    "ACWI", case=False, na=False,
                )
            ]
            if not acwi_row.empty:
                acwi_cagr = acwi_row["cagr"].iloc[0] if "cagr" in acwi_row.columns else None
                acwi_vol = acwi_row["volatility"].iloc[0] if "volatility" in acwi_row.columns else None
                if acwi_cagr is not None and acwi_vol is not None:
                    cagr_delta = float(cagr) - float(acwi_cagr)
                    vol_delta = float(vol) - float(acwi_vol)
                    cagr_threshold = 1.0
                    vol_threshold = 0.5

                    # Win: beating CAGR by margin AND lower vol by margin.
                    if cagr_delta >= cagr_threshold and vol_delta <= -vol_threshold:
                        insights.append({
                            "kind": "win",
                            "icon": "✓",
                            "color": PALETTE["green"],
                            "bg": "#F0FDF4",
                            "category": "Performance · Risk-adjusted win",
                            "headline": "Beating MSCI ACWI on a risk-adjusted basis.",
                            "body": (
                                f"+{cagr_delta:.2f} pp CAGR with {vol_delta:.2f} pp lower volatility. "
                                f"Sharpe {float(sharpe):.2f} suggests real diversification benefit."
                            ),
                        })
                    # Underperform: lower CAGR by margin (regardless of vol).
                    elif cagr_delta <= -cagr_threshold:
                        insights.append({
                            "kind": "underperform",
                            "icon": "▼",
                            "color": PALETTE["red"],
                            "bg": PALETTE["red_bg"],
                            "category": "Performance · Below benchmark",
                            "headline": f"Trailing MSCI ACWI by {abs(cagr_delta):.2f} pp CAGR.",
                            "body": (
                                f"Volatility {float(vol):.2f}% vs ACWI {float(acwi_vol):.2f}%. "
                                "If the gap is consistent over multiple periods, review the "
                                "tilt that's driving it (asset-class mix or geography)."
                            ),
                        })
                    # Otherwise: silent — no Performance insight.

    return insights[:3]


def _build_performance(ctx: _NewsletterContext) -> dict:
    """Build returns table (portfolio + benchmarks) and risk metrics."""
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

    # Benchmark rows (from holding_performance, type contains 'enchmark')
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
            benchmark_rows.append({
                "name": name,
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

    return {
        "portfolio_row": portfolio_row,
        "benchmark_rows": benchmark_rows,  # show all configured benchmarks
        "periods": list(periods),
        "history_label": history_label,
        "benchmark_alpha_beta": ctx.benchmark_alpha_beta,
        "benchmark_geo": ctx.benchmark_geo,
        "returns": returns_block,
    }


def _build_risk_profile(ctx: _NewsletterContext) -> dict:
    """Build the unified Risk Profile table.

    A single table with rows per metric and columns: You / S&P 500 / MSCI
    ACWI. The first four metrics (CAGR, Volatility, Sharpe, Max Drawdown)
    are confronted with both benchmarks. The remaining four (Sortino,
    VaR 95%, CVaR 95%, β) are portfolio-only and show "—" in the
    benchmark columns to keep the layout consistent.

    All portfolio numbers are sourced from ``performance_full`` (5y cap,
    holdings <1Y excluded) so the rows are apples-to-apples with the
    benchmark series and consistent with the Returns table above. β
    naturally shows 1.00 under S&P 500 (the α/β benchmark) and "—" for
    MSCI ACWI to make the relationship explicit.
    """
    m = ctx.metrics
    perf_full = m.performance_full or {}
    bench_cmp = m.benchmark_comparison

    if (
        not perf_full
        or bench_cmp is None or bench_cmp.empty
        or "benchmark" not in bench_cmp.columns
    ):
        return {"available": False, "rows": [], "headers": []}

    def _bench_row(name_substr: str) -> Optional[dict]:
        match = bench_cmp[
            bench_cmp["benchmark"].astype(str).str.contains(
                name_substr, case=False, na=False, regex=False,
            )
        ]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    ab_bench_name = ctx.benchmark_alpha_beta or "S&P 500"
    geo_bench_name = ctx.benchmark_geo or "MSCI ACWI"
    sp500 = _bench_row(ab_bench_name) or {}
    acwi = _bench_row(geo_bench_name) or {}

    def _fmt_pct(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return _pct(float(v))

    def _fmt_num(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.2f}"

    def _verdict(port, bench, lower_is_better) -> tuple[str, str]:
        """Return (symbol, color) for inline verdict on the You cell.

        For "lower is better" metrics where the underlying values can be
        negative (Max Drawdown is reported as a negative percentage —
        −14.98 means a 14.98% peak-to-trough drop), comparing raw values
        is wrong: −14.98 > −20.22 numerically, but −14.98 is the
        shallower (better) drawdown. We compare absolute values so that
        a smaller magnitude wins.
        """
        if (port is None or bench is None
                or (isinstance(port, float) and pd.isna(port))
                or (isinstance(bench, float) and pd.isna(bench))):
            return ("", PALETTE["ink"])
        port_f = float(port)
        bench_f = float(bench)
        if lower_is_better:
            is_win = abs(port_f) < abs(bench_f)
        else:
            is_win = port_f > bench_f
        return ("▲" if is_win else "▼",
                PALETTE["green"] if is_win else PALETTE["amber"])

    # Confrontable metrics: present in both perf_full and bench_cmp.
    # Tuple: (label, perf_full key, bench_cmp key, is_pct, lower_is_better)
    # The α/β labels include the configured benchmark name dynamically
    # (default "S&P 500"). The benchmark name is also passed to the
    # legend so the two stay in sync.
    ab_name = ctx.benchmark_alpha_beta or "S&P 500"
    confrontable = [
        ("CAGR", "cagr", "cagr", True, False),
        ("Volatility", "volatility", "volatility", True, True),
        ("Sharpe", "sharpe", "sharpe", False, False),
        ("Sortino", "sortino", "sortino", False, False),
        ("Max Drawdown", "max_drawdown", "max_drawdown", True, True),
        ("VaR 95%", "var_95", "var_95", True, True),
        ("CVaR 95%", "cvar_95", "cvar_95", True, True),
        # α and β are vs the configured α/β benchmark. By definition the
        # α/β benchmark vs itself yields β=1.00 / α=0; α and β of any
        # other series are computed against the same reference, so the
        # columns are directly comparable.
        # α: higher is better (positive alpha = excess return).
        # β: lower is better in the Excel rating scale (Defensive at
        # < 0.7 is the "Strong" outcome).
        (f"\u03b1 (vs {ab_name})", "alpha", "alpha", True, False),
        (f"\u03b2 (vs {ab_name})", "beta", "beta", False, True),
    ]

    rows = []
    for label, port_key, bench_key, is_pct, lower_better in confrontable:
        port_v = perf_full.get(port_key)
        sp_v = sp500.get(bench_key)
        acwi_v = acwi.get(bench_key)
        sp_verdict_sym, sp_verdict_color = _verdict(port_v, sp_v, lower_better)
        rows.append({
            "label": label,
            "you": _fmt_pct(port_v) if is_pct else _fmt_num(port_v),
            "sp500": _fmt_pct(sp_v) if is_pct else _fmt_num(sp_v),
            "acwi": _fmt_pct(acwi_v) if is_pct else _fmt_num(acwi_v),
            "verdict_sp": sp_verdict_sym,
            "verdict_sp_color": sp_verdict_color,
        })

    return {
        "available": True,
        "headers": ["Metric", "You", ab_bench_name, geo_bench_name],
        "rows": rows,
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
            return "—"
        return f"{value:.1f}{unit}"

    legend_rows = []
    for label, key, description in legend_specs:
        spec = ratings.get(key, {}) or {}
        thresholds = spec.get("thresholds", [None, None])
        invert = bool(spec.get("invert", False))
        unit = spec.get("unit", "")
        good_t, warn_t = (thresholds + [None, None])[:2]

        if good_t is None or warn_t is None:
            strong = fair = weak = "—"
        elif invert:
            # Lower-is-better metrics: better when below good threshold.
            strong = f"< {_fmt(abs(good_t), unit)}"
            fair = f"{_fmt(abs(warn_t), unit)} – {_fmt(abs(good_t), unit)}"
            weak = f"> {_fmt(abs(warn_t), unit)}"
        else:
            strong = f"> {_fmt(good_t, unit)}"
            fair = f"{_fmt(warn_t, unit)} – {_fmt(good_t, unit)}"
            weak = f"< {_fmt(warn_t, unit)}"

        legend_rows.append({
            "label": label,
            "strong": strong,
            "fair": fair,
            "weak": weak,
            "description": description,
        })
    return legend_rows


def _build_sensitivity(ctx: _NewsletterContext) -> dict:
    """Build the drift-penalty sensitivity card.

    Mirrors the Excel "Drift-penalty sensitivity" table so the
    newsletter reader has the same view of the optimization regimes
    available with the current portfolio. The active regime is
    highlighted, and dynamic notes compare it to its immediate
    neighbours so the user can decide whether to tighten or loosen
    ``rebalancing_drift_penalty_weight`` in the next run.
    """
    m = ctx.metrics
    cfg = ctx.config
    sensitivity = getattr(m, "rebalancing_sensitivity", None)
    if not sensitivity:
        return {"available": False}

    # If every regime in the sweep produces zero trades, the
    # sensitivity card has nothing useful to say — typical when the
    # portfolio is already inside the tolerance band at every weight.
    # Hide the section in that case so the email isn't bloated by an
    # empty diagnostic.
    has_trades = any(
        (r.get("n_buy", 0) + r.get("n_sell", 0)) > 0
        for r in sensitivity
    )
    if not has_trades:
        return {"available": False}

    configured_w = float(getattr(cfg, "rebalancing_drift_penalty_weight", 0.0) or 0.0)

    def _net_by_class_str(d: dict, min_eur: float = 0.5) -> str:
        items = sorted((d or {}).items(), key=lambda x: -abs(x[1]))
        return "  \u00b7  ".join(
            f"{ac} {_eur_smart(v, signed=True)}"
            for ac, v in items
            if abs(v) > min_eur
        ) or "—"

    rows = []
    for r in sensitivity:
        wmin = float(r["weight_min"])
        wmax = float(r["weight_max"])
        is_active = (wmin <= configured_w <= wmax)
        range_str = f"{wmin:g}" if wmin == wmax else f"{wmin:g} – {wmax:g}"
        rows.append({
            "is_active": is_active,
            "weight_range": range_str,
            "actions": f"{r['n_buy']}B / {r['n_sell']}S",
            "buy_total": _eur_smart(float(r["total_buy"])),
            "sell_total": _eur_smart(float(r["total_sell"])),
            "friction": _eur_smart(float(r["total_tax"]) + float(r["total_fee"])),
            "max_drift": f"{float(r['max_drift_pp']):.2f}%",
            "net_by_class": _net_by_class_str(r.get("net_by_class") or {}),
        })

    # Reuse the Excel helper for the dynamic notes so phrasing stays
    # in lock-step between the two surfaces.
    from tarzan.export.excel import _build_sensitivity_notes
    notes = _build_sensitivity_notes(sensitivity, configured_w)

    return {
        "available": True,
        "active_weight": f"{configured_w:g}",
        "rows": rows,
        "notes": notes,
    }


def _build_optimizer(ctx: _NewsletterContext) -> dict:
    """Build the suggested-action card.

    Surfaces every rebalancing suggestion produced by the engine, ordered
    by absolute amount so the most impactful trades come first.
    """
    m = ctx.metrics
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions:
        return {"available": False}

    df = m.holdings_df

    # Plan totals across the entire suggestion set.
    total_buy = sum(float(s["amount_eur"]) for s in suggestions
                    if s["direction"].lower() == "buy")
    total_sell = sum(float(s["amount_eur"]) for s in suggestions
                     if s["direction"].lower() == "sell")

    # Display order: largest absolute amount first.
    selected = sorted(suggestions, key=lambda s: -float(s["amount_eur"]))

    actions = []
    for s in selected:
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
            "ticker": ticker,
            "isin": s.get("isin", ""),
            "asset_class": klass,
            "asset_color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            # Always show amounts as positive — the colored chip already
            # encodes the direction. Net is shown in the summary below.
            "amount": _eur(amount, decimals=2),
            "pct_of_portfolio": _pct(pct_of_port, decimals=1),
            "reason": s.get("reason", ""),
        })

    n_total = len(suggestions)
    n_buy = sum(1 for s in suggestions if s["direction"].lower() == "buy")
    n_sell = n_total - n_buy

    return {
        "available": True,
        "actions": actions,
        "n_total": n_total,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "total_buy": _eur_smart(total_buy),
        "total_sell": _eur_smart(total_sell),
        "net": _eur_smart(total_buy - total_sell, signed=True),
        "net_color": (PALETTE["green"] if (total_buy - total_sell) >= 0
                      else PALETTE["red"]),
    }



def _build_return_contrib(ctx: _NewsletterContext) -> dict:
    """Build winners / laggards by return contribution."""
    m = ctx.metrics
    df = m.holdings_df
    if df.empty:
        return {"winners": [], "laggards": []}

    rows = []
    for _, r in df.iterrows():
        contrib = float(r.get("weight_pct", 0) or 0) * float(r.get("gain_pct", 0) or 0) / 100
        rows.append({"name": r.get("name", ""), "ticker": r.get("ticker", ""), "contrib": contrib})
    rows.sort(key=lambda x: -x["contrib"])
    winners = [{"name": r["name"], "value": _pct(r["contrib"], signed=True)} for r in rows[:3]]
    laggards = [{"name": r["name"], "value": _pct(r["contrib"], signed=True)} for r in rows[-3:]]
    laggards.reverse()  # worst first
    return {"winners": winners, "laggards": laggards}


def _build_preheader(ctx: _NewsletterContext, hero: dict) -> str:
    """Preview text shown in inbox preview."""
    m = ctx.metrics
    n_actions = len(m.rebalancing_suggestions or [])
    parts = [f"Portfolio at {hero['total_value']} ({hero['gain_pct']} since inception)"]
    if n_actions > 0:
        parts.append(f"{n_actions} rebalancing action{'s' if n_actions != 1 else ''} suggested")
    parts.append(f"{len(m.holdings_df)} holdings tracked")
    return " · ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def build_context(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: str = "S&P 500",
    benchmark_geo: str = "MSCI ACWI",
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
        "sparkline": _build_sparkline(nctx),
        "smart_insights": _build_smart_insights(nctx),
        "movers": _build_movers(nctx),
        "allocation": _build_allocation(nctx),
        "geography": _build_geography(nctx),
        "holdings": _build_holdings(nctx),
        "returns_snapshot": _build_returns_snapshot(nctx),
        "performance": _build_performance(nctx),
        "risk_profile": _build_risk_profile(nctx),
        "optimizer": _build_optimizer(nctx),
        "sensitivity": _build_sensitivity(nctx),
        "return_contrib": _build_return_contrib(nctx),
        "preheader": _build_preheader(nctx, hero),
        "footer": {
            "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
            "version": "v2.0",
        },
    }


def render_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    issue_number: int = 1,
    benchmark_alpha_beta: str = "S&P 500",
    benchmark_geo: str = "MSCI ACWI",
) -> str:
    """Render the newsletter HTML to a string.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark.
        benchmark_geo: Display name of geographic allocation benchmark.

    Returns:
        The full HTML newsletter as a single string.
    """
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
    )
    return template.render(**context)


def generate_newsletter(
    metrics: PortfolioMetrics,
    config: InvestorConfig,
    output_dir: str,
    issue_number: int = 1,
    benchmark_alpha_beta: str = "S&P 500",
    benchmark_geo: str = "MSCI ACWI",
) -> str:
    """Render the newsletter and write it to disk.

    The output filename uses the same timestamp pattern as the Excel report:
    ``portfolio_digest_<YYYYMMDD_HHMM>.html``.

    Args:
        metrics: Computed portfolio metrics.
        config: Investor configuration.
        output_dir: Directory for the output file.
        issue_number: Sequential issue number for branding.
        benchmark_alpha_beta: Display name of α/β benchmark.
        benchmark_geo: Display name of geographic allocation benchmark.

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
