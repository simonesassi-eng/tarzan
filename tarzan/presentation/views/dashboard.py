"""Dashboard — hero metrics, allocations, risk KPIs, sector breakdown."""

from __future__ import annotations

import streamlit as st

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.investor_config import InvestorConfig
from tarzan.presentation.formatters import fmt_eur, fmt_pct, gain_color
from tarzan.presentation.charts import donut, delta_bars, period_returns_bar, COLORS, GEO_COLORS


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    total_cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - total_cost
    rtd = (total_gain / total_cost * 100) if total_cost > 0 else 0.0

    inception_str = config.portfolio_inception_date if config and config.portfolio_inception_date else ""
    inception_label = f" · since {inception_str}" if inception_str else ""

    st.markdown(
        f"<div style='margin-bottom:16px'>"
        f"<p style='color:#8b949e; margin:0; font-size:0.85rem'>📊 PORTFOLIO STATUS{inception_label}</p>"
        f"<h1 style='margin:4px 0 0; font-size:2.4rem; font-weight:800; color:#f0f6fc;'>"
        f"{fmt_eur(metrics.total_value)}</h1>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Row 1: Total / Invested / Cash ────────────────────────────────
    col1, col2, col3 = st.columns(3)
    _hero_card(col1, "Total Value (EUR)", fmt_eur(metrics.total_value), None)
    _hero_card(col2, "Invested Value (EUR)", fmt_eur(metrics.invested_value), None)
    _hero_card(col3, "Cash (EUR)", fmt_eur(metrics.cash_value), None)

    # ── Row 2: Gain / RTD / Inception ─────────────────────────────────
    col4, col5, col6 = st.columns(3)
    _hero_card(col4, "Total Gain (EUR)", fmt_eur(total_gain), total_gain)
    _hero_card(col5, "RTD (%)", fmt_pct(rtd), total_gain)
    _hero_card(col6, "Inception", inception_str or "—", None)

    # ── Row 2: CAGR / YTD / Yield / TER ───────────────────────────────
    perf = metrics.performance or {}
    risk = metrics.risk or {}

    cagr_val = perf.get("cagr")
    ytd_val  = perf.get("ytd")
    cagr_str = fmt_pct(cagr_val) if cagr_val is not None else "—"
    ytd_str  = fmt_pct(ytd_val)  if ytd_val  is not None else "—"
    yield_str = f"{metrics.weighted_yield:.2f}%" if metrics.weighted_yield else "—"
    ter_str   = f"{metrics.avg_ter:.3f}%"        if metrics.avg_ter        else "—"

    st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _small_card(c1, "CAGR (inception)", cagr_str, cagr_val)
    _small_card(c2, "YTD Return", ytd_str, ytd_val)
    _small_card(c3, "Wtd. Yield", yield_str, None, neutral=True)
    _small_card(c4, "Avg TER", ter_str, None, neutral=True, invert=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Risk KPIs ───────────────────────────────────────────────
    vol   = risk.get("volatility")
    shrp  = risk.get("sharpe")
    mdd   = risk.get("max_drawdown")
    var95 = risk.get("var_95")

    if any(v is not None and str(v) != "nan" for v in [vol, shrp, mdd, var95]):
        st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)

        def _fmt_risk(v, pct=True):
            if v is None or str(v) == "nan":
                return "—"
            return f"{v:.2f}%" if pct else f"{v:.2f}"

        _small_card(r1, "Volatility (ann.)", _fmt_risk(vol), None, neutral=True)
        _small_card(r2, "Sharpe Ratio",      _fmt_risk(shrp, pct=False), shrp, invert=False)
        _small_card(r3, "Max Drawdown",      _fmt_risk(mdd), mdd, invert=True)
        _small_card(r4, "VaR 95% (daily)",   _fmt_risk(var95), var95, invert=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Period Returns bar chart ───────────────────────────────────────
    if perf:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📅 Period Returns")
        fig = period_returns_bar(perf, "")
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key="dash_period_returns")

    # ── Allocation ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Allocation")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Asset Class**")
        if not metrics.allocation_by_class.empty:
            fig = donut(
                metrics.allocation_by_class["category"].tolist(),
                metrics.allocation_by_class["weight_pct"].tolist(),
                COLORS, "",
            )
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key="dash_ac")

        if config and not metrics.allocation_by_class.empty:
            _delta_section(
                metrics.allocation_by_class,
                config.invested_allocation_targets_pctg,
                "dash_delta_class",
            )

    with col_b:
        st.markdown("**Geography (Equity only)**")
        if not metrics.allocation_by_geo.empty:
            fig = donut(
                metrics.allocation_by_geo["category"].tolist(),
                metrics.allocation_by_geo["weight_pct"].tolist(),
                GEO_COLORS, "",
            )
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key="dash_geo")

            # ACWI reference
            if metrics.acwi_geo:
                with st.expander("📐 vs MSCI ACWI reference", expanded=False):
                    _acwi_comparison_table(metrics.allocation_by_geo, metrics.acwi_geo)

        if config and not metrics.allocation_by_geo.empty:
            _delta_section(
                metrics.allocation_by_geo,
                config.equity_geo_targets_pctg,
                "dash_delta_geo",
            )

    # ── Sector Allocation ──────────────────────────────────────────────
    if not metrics.allocation_by_sector.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏭 Sector Allocation")
        sec_fig = donut(
            metrics.allocation_by_sector["category"].tolist(),
            metrics.allocation_by_sector["weight_pct"].tolist(),
            {}, "",  # no fixed color map for sectors
        )
        sec_fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(sec_fig, use_container_width=True, key="dash_sector")

    # ── Top 5 Holdings ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💼 Top 5 Holdings")

    if not metrics.holdings_df.empty:
        top5 = metrics.holdings_df.nlargest(5, "weight_pct")
        cards_html = (
            "<div style='display:grid; grid-template-columns:"
            " repeat(auto-fill, minmax(140px, 1fr)); gap:10px;'>"
        )
        for _, row in top5.iterrows():
            c = gain_color(row["gain_pct"])
            cards_html += (
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{row['ticker']}</div>"
                f"<div style='font-size:0.8rem; color:#f0f6fc; font-weight:600;"
                f" margin-top:4px; white-space:nowrap; overflow:hidden;"
                f" text-overflow:ellipsis;'>{row['name'][:20]}</div>"
                f"<div class='metric-value' style='font-size:1rem;'>"
                f"{fmt_eur(row['current_value'])}</div>"
                f"<div style='font-size:0.75rem;'>"
                f"<span style='color:#8b949e;'>{row['weight_pct']:.1f}%</span>"
                f" · <span style='color:{c};'>{fmt_pct(row['gain_pct'])}</span>"
                f"</div></div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Rebalancing alert ──────────────────────────────────────────────
    if metrics.rebalancing_suggestions:
        st.markdown("<br>", unsafe_allow_html=True)
        n = len(metrics.rebalancing_suggestions)
        st.warning(
            f"⚖️ **{n} rebalancing action{'s' if n > 1 else ''} suggested.** "
            "Go to the **Optimizer** tab for details.",
            icon="⚠️",
        )


# ── Helper renderers ───────────────────────────────────────────────────

def _hero_card(col, label: str, value: str, gain_for_color, hint: str | None = None):
    color = "#f0f6fc"
    if gain_for_color is not None and isinstance(gain_for_color, (int, float)):
        color = "#3fb950" if gain_for_color >= 0 else "#f85149"
    hint_html = (
        f"<div style='font-size:0.72rem; color:#8b949e; margin-top:4px;'>{hint}</div>"
        if hint else ""
    )
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color:{color};'>{value}</div>"
        f"{hint_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _small_card(col, label: str, value: str, raw_val, neutral=False, invert=False):
    """Compact card for secondary KPIs."""
    if neutral or raw_val is None or str(raw_val) == "nan":
        color = "#f0f6fc"
    elif invert:
        color = "#f85149" if raw_val > 0 else "#3fb950"
    else:
        color = "#3fb950" if raw_val >= 0 else "#f85149"
    col.markdown(
        f"<div class='metric-card' style='padding:12px;'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div style='font-size:1.1rem; font-weight:700; margin:4px 0 0; color:{color};'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _delta_section(df, targets: dict, chart_key: str):
    """Actual vs target: delta_bars chart + compact table."""
    actual_map = dict(zip(df["category"], df["weight_pct"]))
    cats = sorted(
        set(list(targets.keys()) + df["category"].tolist()),
        key=lambda c: -actual_map.get(c, 0),
    )
    actuals = [actual_map.get(c, 0.0) for c in cats]
    tgts    = [targets.get(c, 0.0) for c in cats]

    if any(t > 0 for t in tgts):
        fig = delta_bars(cats, actuals, tgts)
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    else:
        # No targets set — show simple table
        rows_html = ""
        for cat, actual in zip(cats, actuals):
            rows_html += (
                f"<tr>"
                f"<td style='padding:4px 8px; color:#c9d1d9;'>{cat}</td>"
                f"<td style='padding:4px 8px; text-align:right; color:#f0f6fc; font-weight:600;'>"
                f"{actual:.1f}%</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%; border-collapse:collapse; font-size:0.85rem;'>"
            f"<thead><tr>"
            f"<th style='text-align:left; padding:4px 8px; color:#8b949e; font-weight:500;"
            f" border-bottom:1px solid #21262d;'>Category</th>"
            f"<th style='text-align:right; padding:4px 8px; color:#8b949e; font-weight:500;"
            f" border-bottom:1px solid #21262d;'>Actual</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )


def _acwi_comparison_table(geo_df, acwi_geo: dict):
    """Compare portfolio geo vs MSCI ACWI reference weights."""
    actual_map = dict(zip(geo_df["category"], geo_df["weight_pct"]))
    all_cats = sorted(set(list(actual_map.keys()) + list(acwi_geo.keys())))
    rows_html = ""
    for cat in all_cats:
        actual = actual_map.get(cat, 0.0)
        ref    = acwi_geo.get(cat, 0.0)
        delta  = actual - ref
        delta_color = "#3fb950" if delta > 1 else "#f85149" if delta < -1 else "#8b949e"
        rows_html += (
            f"<tr>"
            f"<td style='padding:4px 8px; color:#c9d1d9;'>{cat}</td>"
            f"<td style='padding:4px 8px; text-align:right; color:#f0f6fc; font-weight:600;'>"
            f"{actual:.1f}%</td>"
            f"<td style='padding:4px 8px; text-align:right; color:#8b949e;'>{ref:.1f}%</td>"
            f"<td style='padding:4px 8px; text-align:right; color:{delta_color}; font-weight:600;'>"
            f"{delta:+.1f}%</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='width:100%; border-collapse:collapse; font-size:0.82rem;'>"
        f"<thead><tr>"
        f"<th style='text-align:left; padding:4px 8px; color:#8b949e; border-bottom:1px solid #21262d;'>Region</th>"
        f"<th style='text-align:right; padding:4px 8px; color:#8b949e; border-bottom:1px solid #21262d;'>Portfolio</th>"
        f"<th style='text-align:right; padding:4px 8px; color:#8b949e; border-bottom:1px solid #21262d;'>ACWI</th>"
        f"<th style='text-align:right; padding:4px 8px; color:#8b949e; border-bottom:1px solid #21262d;'>Delta</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
