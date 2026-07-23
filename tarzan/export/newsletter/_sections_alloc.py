"""Allocation / holdings / hero / optimizer section builders."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.export._format import (
    eur_smart as _eur_smart,
    short_instrument_name,
)
from tarzan.export._perf_series import (
    _norm_series,
    _perf_window,
    _window_money_pnl,
)
from tarzan.export.newsletter._constants import (
    ASSET_CLASS_ORDER,
    ASSET_COLORS,
    GEO_COLORS,
    PALETTE,
    _NewsletterContext,
    group_by_class_role,
    render_unified_table,
    role_for,
    uni_cell,
    uni_name,
)
from tarzan.export.newsletter._format import (
    _display_ticker,
    _eur,
    _pct,
    _pct_smart,
    _semaphore,
    _semaphore_color,
    _signed_pp,
    is_missing,
)
from tarzan.export.newsletter._charts import (
    _hero_flow_chips,
    _hero_value_chart,
    _spark,
    _timeline_vals,
)

def _funding_verification(verifications) -> Optional[dict[str, Any]]:
    """Return the final serialized-action funding proof, when available."""
    return next(
        (
            verification
            for verification in (verifications or [])
            if verification.get("kind") == "funding"
        ),
        None,
    )


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
    if is_missing(week_return):
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

    # Action clause. A failed funding proof takes precedence over the mere
    # presence of draft actions: those trades must never read as instructions.
    suggestions = list(m.rebalancing_suggestions or [])
    funding = _funding_verification(m.rebalancing_verifications)
    if funding and funding.get("status") == "NON_EXECUTABLE":
        parts.append("and the draft rebalance is not executable until cash funding is resolved")
    elif suggestions:
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
    unrealized_eur = m.unrealized_pnl_eur
    unrealized_pct = m.unrealized_pnl_pct or 0.0

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
    # aligned"). A serialized-action funding proof is a separate final
    # authority: draft actions are not actionable unless it is executable.
    rebal_infeasible = bool(
        m.rebalancing_verifications
        and any(v.get("no_solution") for v in m.rebalancing_verifications)
    )
    funding = _funding_verification(m.rebalancing_verifications)
    funding_non_executable = bool(
        funding and funding.get("status") == "NON_EXECUTABLE"
    )

    if rebal_infeasible:
        rebal_label = "Infeasible"
        rebal_sublabel = "no feasible plan"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif funding_non_executable:
        rebal_label = "Not executable"
        rebal_sublabel = "cash funding unresolved"
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
        "rebal_executable": not rebal_infeasible and not funding_non_executable,
    }

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
    # Rates from config, not hardcoded: the note only renders when a CGT
    # estimate exists (rates configured), so this states the rates actually
    # applied — no drift if the user runs a non-26%/12.5% jurisdiction.
    cfg = ctx.config
    std = float(cfg.rebalancing_capital_gains_tax_standard_pctg or 0.0)
    gov = float(cfg.rebalancing_capital_gains_tax_government_pctg or 0.0)
    rate_txt = f"{std:g}% / {gov:g}% on government bonds"
    html = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:{P["card_alt"]};border:1px solid {P["border"]};border-radius:10px;'
        f'border-collapse:separate;border-spacing:0;"><tr><td style="padding:12px 14px;">'
        f'<div style="font-size:11px;color:{P["muted"]};line-height:1.5;">'
        f'<span style="font-weight:700;color:{P["ink"]};">Net-of-tax estimate</span>'
        + (f' &nbsp;{figs_html}' if figs_html else "")
        + f'<div style="margin-top:4px;font-size:10px;color:{P["subtle"]};">Estimate only: average-cost basis, '
        f'{rate_txt}, realized losses offset later gains where Italian rules allow '
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
        tk = _display_ticker(key) or ""
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
            (_display_ticker(it.get("ticker") or "")
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
            ticker = it.get("ticker", "")
            tk = _display_ticker(ticker) or ""
            vals = None
            if hold_series:
                xs = [float(pt.get(ticker, 0.0)) for pt in hold_series]
                if any(x > 0 for x in xs) and len(xs) >= 2:
                    vals = xs
            rows.append({
                "label_html": _div_label(
                    short_instrument_name(it.get("category") or ticker, 42),
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
                f'capital: above 100% via capital-efficient/leveraged funds '
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

def _build_holdings(ctx: _NewsletterContext) -> dict:
    """Build holdings grouped by asset class + role, rendered through the
    shared unified-table renderer (identical shell to Returns / Performance /
    Risk / Optimizer)."""
    m = ctx.metrics
    df = m.holdings_df
    if df.empty:
        return {"summary": [], "table_html": "", "total_count": 0}

    # Curated taxonomy for the shared role categorizer (role sub-grouping).
    from tarzan import config as _cfg
    _tax = _cfg.instrument_taxonomy()

    # Class totals for header summary chips.
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

    # One row item per holding; the shared engine groups them by class → role.
    invested_base = m.invested_value if m.invested_value > 0 else 0.0
    row_items = []
    for _, h in df.iterrows():
        klass = str(h.get("asset_class", "") or "") or "Other"
        value = float(h["current_value"])
        cls_total = class_totals.get(klass, 1) or 1
        pct_class = value / cls_total * 100
        gain_pct = h.get("gain_pct")
        gain_eur = h.get("gain_eur")
        has_gain = gain_pct is not None and not pd.isna(gain_pct)
        # % of invested value; "—" for cash (undefined) or no invested base.
        if klass == "Cash & Cash Equivalents" or invested_base <= 0:
            weight_str = "—"
        else:
            weight_str = _pct(value / invested_base * 100, decimals=1)
        gain_color = (PALETTE["green"] if (gain_pct or 0) >= 0
                      else PALETTE["red"]) if has_gain else PALETTE["muted"]
        cls_color = ASSET_COLORS.get(klass, PALETTE["accent"])
        row_items.append({
            "_ac": klass,
            "_isin": h.get("isin", ""),
            "_ticker": h.get("ticker", ""),
            "name_html": uni_name(
                short_instrument_name(h.get("name", ""), 40),
                _display_ticker(h.get("ticker")) or "",
            ),
            "cells": [
                uni_cell(_eur(value, 2), width=72),
                uni_cell(weight_str, width=50),
                uni_cell(_pct(pct_class, decimals=1), color=cls_color, width=48),
                uni_cell(_pct(gain_pct, signed=True) if has_gain else "—",
                         color=gain_color, weight=700, width=50),
                uni_cell(_eur_smart(gain_eur, signed=True)
                         if (gain_eur is not None and not pd.isna(gain_eur)) else "—",
                         color=gain_color, weight=700, width=64),
            ],
        })
    groups = group_by_class_role(
        row_items, asset_class=lambda r: r["_ac"],
        isin=lambda r: r["_isin"], ticker=lambda r: r["_ticker"], taxonomy=_tax)

    table_html = render_unified_table(
        "Holding",
        [("Value €", "right", 72), ("% Inv.", "right", 50),
         ("% Class", "right", 48), ("Gain %", "right", 50), ("Gain €", "right", 64)],
        [(cls, col, [(role, [{"name_html": it["name_html"], "cells": it["cells"]}
                             for it in items])
                     for role, items in role_list])
         for cls, col, role_list in groups])

    return {"summary": summary, "table_html": table_html, "total_count": int(len(df))}

def _optimizer_plan_ctx(m: PortfolioMetrics, suggestions: list, taxonomy=None) -> dict:
    """Build one optimizer plan's render context (actions + totals) from a
    list of rebalancing suggestions, grouped by asset class → role like the
    other instrument tables."""
    df = m.holdings_df
    taxonomy = taxonomy or {}
    total_buy = sum(float(s["amount_eur"]) for s in suggestions
                    if s["direction"].lower() == "buy")
    total_sell = sum(float(s["amount_eur"]) for s in suggestions
                     if s["direction"].lower() == "sell")

    # One cell showing a share as "% (bold) over € (muted)", the compact
    # Diversification-cell style, so four value columns fit at 600px.
    def _pct_eur_cell(pct, eur, *, color=None, weight=700):
        """A "% (bold) over € (muted)" value cell for the unified renderer."""
        return uni_cell(_pct(pct, decimals=1) if pct is not None else "—",
                        color=color or PALETTE["ink"], weight=weight,
                        sub=(_eur_smart(eur) if eur is not None else ""))

    def _pill(direction):
        c = PALETTE["green"] if direction == "BUY" else PALETTE["red"]
        return (f'<span style="display:inline-block;padding:1px 6px;background:#FFFFFF;'
                f'color:{c};border:1px solid {c}33;border-radius:999px;font-weight:700;'
                f'font-size:9px;letter-spacing:0.04em;vertical-align:middle;'
                f'margin-right:5px;">{direction}</span>')

    actions = []
    for s in sorted(suggestions, key=lambda s: -float(s["amount_eur"])):
        direction = s["direction"].upper()
        amount = float(s["amount_eur"])
        ticker = s.get("ticker", "")
        isin = s.get("isin", "")
        klass = "Equities"
        if not df.empty:
            match = df[df["ticker"] == ticker]
            if not match.empty:
                klass = match["asset_class"].iloc[0]
        signed_amount = amount if direction == "BUY" else -amount
        # Suggestions already carry the full ticker selected during
        # preprocessing; presentation must not re-resolve it from ISIN/cache.
        tk = _display_ticker(ticker) or ""
        # Whole-row tint by action: light green for BUY, light red for SELL,
        # so the proposed action reads at a glance across all columns.
        row_bg = PALETTE["green_tint"] if direction == "BUY" else PALETTE["red_tint"]
        dir_color = PALETTE["green"] if direction == "BUY" else PALETTE["red"]
        actions.append({
            "direction": direction,
            "asset_class": klass,
            "role": role_for(isin, ticker, taxonomy),
            "_row_bg": row_bg,
            # Name cell: action pill + ticker pin + shortened name, via the
            # shared uni_name so it matches every other table.
            "name_html": uni_name(short_instrument_name(s.get("name", "")), tk,
                                   pill=_pill(direction)),
            # Now → Target → Trade → After, each abs + %.
            "cells": [
                _pct_eur_cell(s.get("current_pct"), s.get("current_eur")),
                _pct_eur_cell(s.get("target_pct"), s.get("target_eur"),
                              color=PALETTE["muted"]),
                uni_cell(_eur_smart(signed_amount, signed=True),
                         color=dir_color, weight=700),
                _pct_eur_cell(s.get("after_pct"), s.get("after_eur")),
            ],
        })

    # Group actions by class → role (shared engine); keep the flat sort (largest
    # trade first) WITHIN each role by leaving action order as-is.
    raw_groups = group_by_class_role(
        actions, asset_class=lambda a: a["asset_class"], role=lambda a: a["role"])
    table_html = render_unified_table(
        "Holding",
        [("Now", "right", 66), ("Target", "right", 66),
         ("Trade", "right", 64), ("After", "right", 66)],
        [(ac, col, [(role, [{"name_html": a["name_html"], "cells": a["cells"],
                             "row_bg": a["_row_bg"]} for a in items])
                    for role, items in role_list])
         for ac, col, role_list in raw_groups])

    n_total = len(suggestions)
    n_buy = sum(1 for s in suggestions if s["direction"].lower() == "buy")
    return {
        "actions": actions,
        "table_html": table_html,
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
    """Build both rebalancing plans with their final funding proof.

    Reads ``metrics.rebalancing_plans`` (always computed by the engine); falls
    back to the single ``rebalancing_suggestions`` set for back-compat. Draft
    actions remain visible for diagnosis, but are explicitly marked as such
    whenever the serialized-action proof says they are not executable.
    """
    m = ctx.metrics
    from tarzan import config as _cfg
    _tax = _cfg.instrument_taxonomy()
    plans_src = getattr(m, "rebalancing_plans", None)

    def _attach_cost(pc: dict, p: dict) -> None:
        # Estimated execution cost, shown atop the table: CGT on the sells and
        # fixed commission fees (from the engine's plan_cost, same tax/fee model
        # the optimizer solved for). Only render when non-zero.
        cgt = float(p.get("cgt_eur") or 0.0)
        fees = float(p.get("fees_eur") or 0.0)
        pc["cgt_eur"] = _eur_smart(cgt) if cgt else None
        pc["fees_eur"] = _eur_smart(fees) if fees else None
        pc["cost_total_eur"] = _eur_smart(cgt + fees) if (cgt or fees) else None

    def _attach_execution(pc: dict, verifications) -> None:
        funding = _funding_verification(verifications)
        if funding is None:
            pc["execution_status"] = None
            pc["executable"] = None
            return

        status = str(funding.get("status") or "UNKNOWN").upper()
        residual = float(funding.get("residual_eur") or 0.0)
        pc.update({
            "execution_status": status,
            "executable": status == "EXECUTABLE",
            "funding_initial_cash_eur": _eur_smart(
                float(funding.get("initial_cash_eur") or 0.0)),
            "funding_external_contribution_eur": _eur_smart(
                float(funding.get("external_contribution_eur") or 0.0)),
            "funding_ending_cash_eur": _eur_smart(
                float(funding.get("ending_cash_eur") or 0.0)),
            "funding_protected_cash_eur": _eur_smart(
                float(funding.get("protected_cash_eur") or 0.0)),
            "funding_residual_eur": _eur_smart(residual, signed=True),
            "funding_shortfall_eur": (
                _eur_smart(abs(residual)) if residual < -0.005 else None
            ),
        })

    if plans_src:
        plans = []
        for p in plans_src:
            pc = _optimizer_plan_ctx(m, list(p.get("suggestions") or []), _tax)
            pc["label"] = p.get("label", "")
            pc["no_sell"] = p.get("no_sell")
            _attach_cost(pc, p)
            _attach_execution(pc, p.get("verifications"))
            plans.append(pc)
        if not any(pc["actions"] for pc in plans):
            return {"available": False}
        return {"available": True, "plans": plans}

    # Back-compat: single plan.
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions:
        return {"available": False}
    pc = _optimizer_plan_ctx(m, suggestions, _tax)
    pc["label"] = "Suggested actions"
    pc["no_sell"] = None
    _attach_execution(pc, m.rebalancing_verifications)
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
    funding = _funding_verification(m.rebalancing_verifications)
    parts = [f"Portfolio at {hero['total_value']} ({hero['gain_pct']} since inception)"]
    if funding and funding.get("status") == "NON_EXECUTABLE":
        parts.append("rebalance draft not executable")
    elif n_actions > 0:
        parts.append("rebalancing suggested")
    parts.append(f"{len(m.holdings_df)} holdings tracked")
    return " · ".join(parts)

