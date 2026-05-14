"""Generate the weekly portfolio digest newsletter (HTML email).

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

ASSET_COLORS = {
    "Equities": "#1D4ED8",
    "Fixed Income": "#A16207",
    "Cash & Cash Equivalents": "#15803D",
    "Gold": "#CA8A04",
    "Commodities": "#C2410C",
    "Crypto": "#7C3AED",
    "Alternative": "#7C3AED",
}

ASSET_BG = {
    "Equities": "#EEF2FF",
    "Fixed Income": "#FEF3C7",
    "Cash & Cash Equivalents": "#DCFCE7",
    "Gold": "#FEF3C7",
    "Commodities": "#FEF3C7",
    "Crypto": "#EEF2FF",
    "Alternative": "#EEF2FF",
}

GEO_COLORS = {
    "USA": "#1D4ED8",
    "Eurozone EMU": "#A16207",
    "Dev ex-USA ex-EMU ex-JP": "#15803D",
    "Emerging Markets": "#C2410C",
    "Japan": "#7C3AED",
}

# Asset class display order (matches tarzan/engine/metrics.py)
ASSET_CLASS_ORDER = [
    "Equities",
    "Fixed Income",
    "Cash & Cash Equivalents",
    "Gold",
    "Commodities",
    "Crypto",
    "Alternative",
]


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


def _build_header(ctx: _NewsletterContext) -> dict:
    now = datetime.now()
    return {
        "date_short": now.strftime("%a, %d %b %Y"),
        "issue_number": ctx.issue_number,
        "inception_date": ctx.config.portfolio_inception_date or "",
    }


def _build_hero(ctx: _NewsletterContext) -> dict:
    m = ctx.metrics
    cost = float(m.holdings_df["cost_basis_eur"].sum()) if not m.holdings_df.empty else 0.0
    total_gain = m.total_value - cost
    gain_pct = (total_gain / cost * 100) if cost > 0 else 0.0
    invested_pct = (m.invested_value / m.total_value * 100) if m.total_value > 0 else 0.0
    cash_delta = m.cash_value - m.cash_target_eur
    week_return = float(m.performance.get("1w") or m.performance.get("1W") or 0.0)
    week_eur = m.total_value * week_return / 100 if week_return else 0.0

    # Cash delta formatted message
    if abs(cash_delta) < 1.0:
        cash_msg, cash_msg_color = "on target", PALETTE["muted"]
    elif cash_delta > 0:
        cash_msg = f"{_eur(cash_delta, 0)} above target"
        cash_msg_color = PALETTE["amber"]
    else:
        cash_msg = f"{_eur(abs(cash_delta), 0)} below target"
        cash_msg_color = PALETTE["amber"]

    return {
        "total_value": _eur(m.total_value),
        "total_gain": _eur(total_gain, signed=True),
        "gain_pct": _pct(gain_pct, signed=True),
        "is_positive": total_gain >= 0,
        "invested_value": _eur(m.invested_value),
        "invested_pct": _pct(invested_pct, decimals=1),
        "cash_value": _eur(m.cash_value),
        "cash_msg": cash_msg,
        "cash_msg_color": cash_msg_color,
        "week_return_eur": _eur(week_eur, signed=True),
        "week_return_pct": _pct(week_return, signed=True),
        "week_is_positive": week_eur >= 0,
    }


def _build_sparkline(ctx: _NewsletterContext, n_days: int = 30) -> dict:
    """Build the 30-day sparkline data (start/end values + bar heights)."""
    m = ctx.metrics
    history = m.portfolio_history
    if history is None or len(history) < 2:
        # Synthesise a flat line at current value so the section still renders.
        return {
            "available": False,
            "start_label": "",
            "end_label": "",
            "start_value": _eur(m.total_value),
            "end_value": _eur(m.total_value),
            "change_pct": "0.00%",
            "is_positive": True,
            "bars": [{"height": 22} for _ in range(n_days)],
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

    return {
        "available": True,
        "start_label": last.index[0].strftime("%b %d") if hasattr(last.index[0], "strftime") else "",
        "end_label": last.index[-1].strftime("%b %d") if hasattr(last.index[-1], "strftime") else "",
        "start_value": _eur(start_v, 0),
        "end_value": _eur(end_v, 0),
        "change_pct": _pct(change_pct, signed=True),
        "is_positive": change_pct >= 0,
        "bars": bars,
    }


def _build_allocation(ctx: _NewsletterContext) -> dict:
    """Build asset-class allocation rows (Excel Dashboard pattern)."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_threshold_pctg

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
            "actual_pct": _pct(actual, decimals=1),
            "actual_pct_raw": actual,
            "target_pct": _pct(target, decimals=1) if target is not None else None,
            "delta": _signed_pp(delta) if delta is not None else None,
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max(actual, 1), 100),
        })

    # Cash buffer (EUR-based, appended after invested classes)
    if cfg.target_cash_buffer_eur > 0:
        cash_actual = m.cash_value
        cash_tgt = cfg.target_cash_buffer_eur
        rel_dev = (cash_actual - cash_tgt) / cash_tgt * 100 if cash_tgt > 0 else 0
        sema = _semaphore(rel_dev, tol)
        delta_eur = cash_actual - cash_tgt
        rows.append({
            "name": "Cash & Equivalents",
            "color": ASSET_COLORS["Cash & Cash Equivalents"],
            "actual_pct": _eur(cash_actual, 0),
            "actual_pct_raw": min((cash_actual / max(cash_tgt * 2, 1)) * 100, 100),
            "target_pct": f"target {_eur(cash_tgt, 0)}",
            "delta": ("+" if delta_eur >= 0 else "−") + _eur(abs(delta_eur), 0),
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max((cash_actual / max(cash_tgt * 2, 1)) * 100, 1), 100),
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
    tol = cfg.rebalancing_threshold_pctg

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
                "actual_pct": _pct(actual, decimals=1),
                "target_pct": _pct(target, decimals=1) if target is not None else "—",
                "acwi_pct": _pct(acwi_v, decimals=2) if acwi_v is not None else "—",
                "delta": _signed_pp(delta_target) if delta_target is not None else "—",
                "delta_color": _semaphore_color(sema),
                "bar_width": min(max(actual, 1), 100),
                "target_left": min(max(target or 0, 0), 100),
                "acwi_left": min(max(acwi_v or 0, 0), 100),
            })

    # Stacked equity bar
    stacked = [{"color": r["color"], "width": float(r["actual_pct"].rstrip("%"))} for r in rows]

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
        rows = []
        for i, (_, h) in enumerate(sub.iterrows()):
            value = float(h["current_value"])
            cls_total = class_totals.get(klass, 1) or 1
            pct_class = value / cls_total * 100
            quantity = float(h.get("quantity", 0) or 0)
            avg_price = float(h.get("avg_purchase_price", 0) or 0)
            gain_pct = h.get("gain_pct")
            rows.append({
                "name": h.get("name", ""),
                "ticker": h.get("ticker", ""),
                "isin": h.get("isin", ""),
                "quantity": quantity,
                "avg_price": _eur(avg_price, 2),
                "value": _eur(value, 2),
                "weight_pct": _pct(float(h.get("weight_pct", 0) or 0), decimals=1),
                "gain_pct": _pct(gain_pct, signed=True) if gain_pct is not None and not pd.isna(gain_pct) else "—",
                "gain_color": (PALETTE["green"] if (gain_pct or 0) >= 0 else PALETTE["red"]) if gain_pct is not None and not pd.isna(gain_pct) else PALETTE["muted"],
                "pct_class": _pct(pct_class, decimals=1),
                "alt_bg": i % 2 == 1,
            })
        groups.append({
            "name": klass,
            "name_short": "Cash & Cash Equivalents" if klass == "Cash & Cash Equivalents" else klass,
            "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "bg": ASSET_BG.get(klass, PALETTE["accent_bg"]),
            "count": int(class_counts.get(klass, 0)),
            "label": "positions" if class_counts.get(klass, 0) != 1 else "position",
            "total_value": _eur(class_totals.get(klass, 0), 0),
            "total_pct": _pct(
                class_totals.get(klass, 0) / m.total_value * 100 if m.total_value > 0 else 0,
                decimals=1,
            ),
            "rows": rows,
        })

    return {"groups": groups, "summary": summary, "total_count": int(len(df))}


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
            "eur": _eur(abs(eur), 0),
        }

    return {
        "available": True,
        "best": _enrich(best),
        "worst": _enrich(worst),
        "benchmark_name": ctx.benchmark_alpha_beta,
    }


def _build_smart_insights(ctx: _NewsletterContext) -> list[dict]:
    """Generate up to 3 rule-based insights from metrics."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_threshold_pctg
    insights: list[dict] = []

    # 1. Action — largest drift vs target
    if m.goal_deltas is not None and not m.goal_deltas.empty:
        non_cash = m.goal_deltas[m.goal_deltas["type"] != "cash"]
        if not non_cash.empty:
            largest = non_cash.iloc[non_cash["delta_pct"].abs().argmax()]
            delta = float(largest["delta_pct"])
            cat = largest["category"]
            sema = _semaphore(delta, tol)
            if sema != "green":
                # Find a suggested rebalance for this category
                action_text = ""
                if m.rebalancing_suggestions:
                    s = m.rebalancing_suggestions[0]
                    direction = s["direction"].upper()
                    action_text = (
                        f"Deploy {_eur(s['amount_eur'])} ({direction}) into "
                        f"{s.get('ticker', s.get('name', ''))} to bring the bucket back toward target."
                    )
                else:
                    action_text = (
                        f"Consider rebalancing your next contribution toward {cat}."
                    )
                insights.append({
                    "kind": "action",
                    "icon": "⚡",
                    "color": PALETTE["amber"],
                    "bg": PALETTE["amber_bg"],
                    "category": "Action · Rebalance",
                    "headline": f"{cat} is {abs(delta):.1f} pp {'above' if delta > 0 else 'below'} target — your largest drift.",
                    "body": action_text,
                })

    # 2. Risk — concentration in top 2 holdings
    if not m.holdings_df.empty and len(m.holdings_df) >= 2:
        top2 = m.holdings_df.nlargest(2, "weight_pct")
        top2_pct = float(top2["weight_pct"].sum())
        if top2_pct >= 30:  # threshold for raising concentration as insight
            names = " · ".join(
                f"{r['ticker']} ({_pct(float(r['weight_pct']), 1)})"
                for _, r in top2.iterrows()
            )
            insights.append({
                "kind": "risk",
                "icon": "◎",
                "color": PALETTE["accent"],
                "bg": PALETTE["accent_bg"],
                "category": "Risk · Concentration",
                "headline": f"Your top 2 positions hold {top2_pct:.0f}% of the portfolio.",
                "body": f"{names}. Watch for overlap — concentrated weight raises portfolio-level swings.",
            })

    # 3. Win — beating ACWI on risk-adjusted basis
    risk = m.risk or {}
    perf = m.performance_full or {}
    sharpe = risk.get("sharpe")
    cagr = perf.get("cagr")
    bench_cmp = m.benchmark_comparison
    if not bench_cmp.empty and sharpe is not None and cagr is not None:
        # The real pipeline uses 'benchmark' as the name column, while
        # the test fixture uses 'name'. Support both for robustness.
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
                vol = risk.get("volatility")
                if acwi_cagr is not None and acwi_vol is not None and vol is not None:
                    cagr_delta = cagr - float(acwi_cagr)
                    vol_delta = vol - float(acwi_vol)
                    if cagr_delta > 0 and vol_delta < 0:
                        insights.append({
                            "kind": "win",
                            "icon": "✓",
                            "color": PALETTE["green"],
                            "bg": "#F0FDF4",
                            "category": "Win · Risk-adjusted",
                            "headline": "You're beating MSCI ACWI on risk-adjusted basis.",
                            "body": (
                                f"+{cagr_delta:.2f} pp CAGR with {vol_delta:.2f} pp lower volatility. "
                                f"Sharpe {sharpe:.2f} vs ACWI implies real diversification benefit."
                            ),
                        })

    return insights[:3]


def _build_performance(ctx: _NewsletterContext) -> dict:
    """Build returns table (portfolio + benchmarks) and risk metrics."""
    m = ctx.metrics

    # Portfolio row
    pf = m.performance_full or {}
    portfolio_row = {
        "name": "Your portfolio",
        "tag": None,
        "is_portfolio": True,
        "returns": {p: _pct(pf.get(p), signed=True) for p in ("1w", "1m", "3m", "ytd", "1y", "3y", "5y")},
    }

    # Benchmark rows (from holding_performance, type contains 'enchmark')
    benchmark_rows = []
    hp = m.holding_performance
    if not hp.empty and "type" in hp.columns:
        bench_df = hp[hp["type"].astype(str).str.contains("enchmark", case=False, na=False)]
        for _, r in bench_df.iterrows():
            name = str(r.get("name") or r.get("ticker", ""))
            tag = None
            if "S&P 500" in name or "S&P500" in name:
                tag = ("α/β", PALETTE["accent"], PALETTE["accent_bg"])
            elif "ACWI" in name and "ACWI ex" not in name:
                tag = ("GEO", PALETTE["accent"], PALETTE["accent_bg"])
            benchmark_rows.append({
                "name": name,
                "tag": tag,
                "is_portfolio": False,
                "returns": {p: _pct(r.get(p), signed=True) for p in ("1w", "1m", "3m", "ytd", "1y", "3y", "5y")},
            })

    # Risk chips
    risk = m.risk or {}
    risk_chips = []
    chip_specs = [
        ("CAGR", "cagr", True, False),
        ("Volatility", "volatility", True, False),
        ("Sharpe", "sharpe", False, False),
        ("Sortino", "sortino", False, False),
        ("Max DD", "max_drawdown", True, True),
        ("VaR 95%", "var_95", True, True),
        ("CVaR 95%", "cvar_95", True, True),
        ("β", "beta", False, False),
    ]
    for label, key, is_pct, is_neg in chip_specs:
        v = pf.get(key) if key in pf else risk.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        formatted = _pct(v, decimals=2) if is_pct else f"{float(v):.2f}"
        color = PALETTE["red"] if is_neg else (PALETTE["green"] if (key == "cagr" and v > 0) else PALETTE["ink"])
        risk_chips.append({"label": label, "value": formatted, "color": color})

    return {
        "portfolio_row": portfolio_row,
        "benchmark_rows": benchmark_rows[:8],  # Cap for layout
        "risk_chips": risk_chips,
        "benchmark_alpha_beta": ctx.benchmark_alpha_beta,
        "benchmark_geo": ctx.benchmark_geo,
    }


def _build_risk_vs_benchmarks(ctx: _NewsletterContext) -> dict:
    """Build the side-by-side "Risk vs S&P 500" / "Risk vs MSCI ACWI" tables.

    For each benchmark we expose CAGR, Volatility, Sharpe, and Max Drawdown
    (the four metrics common to both PortfolioMetrics.risk/performance and
    PortfolioMetrics.benchmark_comparison). VaR is intentionally omitted
    because benchmark_comparison does not carry it.
    """
    m = ctx.metrics
    risk = m.risk or {}
    perf = m.performance or {}
    bench_cmp = m.benchmark_comparison

    portfolio_cagr = perf.get("cagr") or perf.get("CAGR")
    portfolio_vol = risk.get("volatility")
    portfolio_sharpe = risk.get("sharpe")
    portfolio_dd = risk.get("max_drawdown")

    if (
        bench_cmp is None or bench_cmp.empty
        or "benchmark" not in bench_cmp.columns
        or any(v is None for v in (portfolio_cagr, portfolio_vol,
                                    portfolio_sharpe, portfolio_dd))
    ):
        return {"available": False, "tables": []}

    def _row_for(name_substr: str) -> dict | None:
        match = bench_cmp[
            bench_cmp["benchmark"].astype(str).str.contains(
                name_substr, case=False, na=False, regex=False,
            )
        ]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    bench_specs = [
        ("S&P 500", "S&P 500", "α/β"),
        ("MSCI ACWI", "MSCI ACWI", "GEO"),
    ]
    tables = []
    for display_name, search_token, tag_label in bench_specs:
        bench = _row_for(search_token)
        if bench is None:
            continue
        # For each metric we compute the delta and decide if it's a
        # win for the portfolio. Volatility and MaxDD are "lower is
        # better"; CAGR and Sharpe are "higher is better".
        rows = []
        metric_specs = [
            ("CAGR", portfolio_cagr, bench.get("cagr"), True, False),
            ("Volatility", portfolio_vol, bench.get("volatility"), True, True),
            ("Sharpe", portfolio_sharpe, bench.get("sharpe"), False, False),
            ("Max Drawdown", portfolio_dd, bench.get("max_drawdown"), True, True),
        ]
        for label, port, bench_v, is_pct, lower_is_better in metric_specs:
            if bench_v is None or pd.isna(bench_v):
                continue
            port_f = float(port)
            bench_f = float(bench_v)
            delta = port_f - bench_f
            is_win = (delta < 0) if lower_is_better else (delta > 0)
            verdict_color = PALETTE["green"] if is_win else PALETTE["amber"]
            unit = "pp" if is_pct else ""
            rows.append({
                "label": label,
                "portfolio": _pct(port_f) if is_pct else f"{port_f:.2f}",
                "benchmark": _pct(bench_f) if is_pct else f"{bench_f:.2f}",
                "verdict": _signed_pp(delta, decimals=2)
                           + (f" {unit}" if unit else ""),
                "verdict_color": verdict_color,
                "is_loss_metric": lower_is_better,
            })
        if rows:
            tables.append({
                "name": display_name,
                "tag": tag_label,
                "rows": rows,
            })

    return {"available": bool(tables), "tables": tables}


def _build_optimizer(ctx: _NewsletterContext) -> dict:
    """Build the suggested-action card.

    Shows up to ``MAX_ACTIONS`` rebalancing suggestions, ensuring the
    user sees both BUY and SELL sides of the rebalance plan when both
    exist. Suggestions are surfaced ordered by absolute amount so the
    most impactful trades come first.
    """
    MAX_ACTIONS = 4
    m = ctx.metrics
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions:
        return {"available": False}

    df = m.holdings_df

    # Compute totals across the whole plan (not just the displayed subset)
    total_buy = sum(float(s["amount_eur"]) for s in suggestions
                    if s["direction"].lower() == "buy")
    total_sell = sum(float(s["amount_eur"]) for s in suggestions
                     if s["direction"].lower() == "sell")

    # Pick which actions to display: when the plan has both BUYs and
    # SELLs we want both surfaced. Strategy:
    #  - Sort all actions by absolute amount, descending.
    #  - Take the largest BUY and the largest SELL first.
    #  - Fill remaining slots with the next largest of either side.
    by_amount = sorted(suggestions, key=lambda s: -float(s["amount_eur"]))
    largest_buy = next((s for s in by_amount if s["direction"].lower() == "buy"), None)
    largest_sell = next((s for s in by_amount if s["direction"].lower() == "sell"), None)

    selected: list[dict] = []
    if largest_buy:
        selected.append(largest_buy)
    if largest_sell:
        selected.append(largest_sell)
    for s in by_amount:
        if s in selected:
            continue
        if len(selected) >= MAX_ACTIONS:
            break
        selected.append(s)
    # Re-order selected by amount desc for display
    selected.sort(key=lambda s: -float(s["amount_eur"]))

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
        "n_displayed": len(actions),
        "n_more": max(0, n_total - len(actions)),
        "total_buy": _eur(total_buy, decimals=0),
        "total_sell": _eur(total_sell, decimals=0),
        "net": _eur(total_buy - total_sell, signed=True),
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
        "hero": hero,
        "sparkline": _build_sparkline(nctx),
        "smart_insights": _build_smart_insights(nctx),
        "movers": _build_movers(nctx),
        "allocation": _build_allocation(nctx),
        "geography": _build_geography(nctx),
        "holdings": _build_holdings(nctx),
        "performance": _build_performance(nctx),
        "risk_vs_benchmarks": _build_risk_vs_benchmarks(nctx),
        "optimizer": _build_optimizer(nctx),
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
