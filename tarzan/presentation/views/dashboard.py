"""Dashboard — mirrors Excel Dashboard sheet.

Hero (Value/Gain/RTD) + Allocation (actual vs target) + Top 5 + Rebalancing alert.
"""

from __future__ import annotations

import streamlit as st

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.investor_config import InvestorConfig
from tarzan.presentation.formatters import fmt_eur, fmt_pct, gain_color
from tarzan.presentation.charts import donut, COLORS, GEO_COLORS


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    # =====================================================================
    # HERO — Portfolio Status
    # =====================================================================
    total_cost = float(metrics.holdings_df["cost_basis_eur"].sum()) if not metrics.holdings_df.empty else 0.0
    total_gain = metrics.total_value - total_cost
    rtd = (total_gain / total_cost * 100) if total_cost > 0 else 0.0

    inception_str = config.portfolio_inception if config and config.portfolio_inception else ""
    inception_label = f" · since {inception_str}" if inception_str else ""

    st.markdown(
        f"<div style='margin-bottom: 24px'>"
        f"<p style='color:#8b949e; margin:0; font-size: 0.85rem'>📊 PORTFOLIO STATUS{inception_label}</p>"
        f"<h1 style='margin:4px 0 0 0; font-size: 2.8rem; font-weight: 800; color: #f0f6fc;'>"
        f"{fmt_eur(metrics.total_value)}</h1>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    _hero_card(col1, "Total Value (AuM)", fmt_eur(metrics.total_value), None)
    _hero_card(col2, "Total Gain", fmt_eur(total_gain), total_gain)
    _hero_card(col3, "RTD (Return to Date)", fmt_pct(rtd), total_gain)

    # =====================================================================
    # ALLOCATION — Asset Class + Geography
    # =====================================================================
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
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key="dash_ac")

        if config and not metrics.allocation_by_class.empty:
            _allocation_table(metrics.allocation_by_class, config.allocation_targets)

    with col_b:
        st.markdown("**Geography (Equity only)**")
        if not metrics.allocation_by_geo.empty:
            fig = donut(
                metrics.allocation_by_geo["category"].tolist(),
                metrics.allocation_by_geo["weight_pct"].tolist(),
                GEO_COLORS, "",
            )
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, key="dash_geo")

        if config and not metrics.allocation_by_geo.empty:
            _allocation_table(metrics.allocation_by_geo, config.geo_allocation)

    # =====================================================================
    # TOP 5 HOLDINGS
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💼 Top 5 Holdings")

    if not metrics.holdings_df.empty:
        top5 = metrics.holdings_df.nlargest(5, "weight_pct")
        cols = st.columns(5)
        for col, (_, row) in zip(cols, top5.iterrows()):
            c = gain_color(row["gain_pct"])
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{row['ticker']}</div>"
                f"<div style='font-size: 0.85rem; color: #f0f6fc; font-weight: 600; margin-top: 4px; "
                f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{row['name'][:22]}</div>"
                f"<div class='metric-value' style='font-size: 1.1rem;'>{fmt_eur(row['current_value'])}</div>"
                f"<div style='font-size: 0.75rem;'>"
                f"<span style='color: #8b949e;'>{row['weight_pct']:.1f}%</span>"
                f" · <span style='color: {c};'>{fmt_pct(row['gain_pct'])}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # =====================================================================
    # REBALANCING ALERT
    # =====================================================================
    if metrics.rebalancing_suggestions:
        st.markdown("<br>", unsafe_allow_html=True)
        n = len(metrics.rebalancing_suggestions)
        s_str = "s" if n > 1 else ""
        st.warning(
            f"⚖️ **{n} rebalancing action{s_str} suggested.** "
            f"Go to the **Optimizer** tab for details.",
            icon="⚠️",
        )


def _hero_card(col, label: str, value: str, gain_for_color):
    """Render a hero card with optional gain/loss coloring."""
    color = "#f0f6fc"
    if gain_for_color is not None and isinstance(gain_for_color, (int, float)):
        color = "#3fb950" if gain_for_color >= 0 else "#f85149"
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value' style='color: {color};'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _allocation_table(df, targets: dict):
    """Render a compact actual/target table without delta column."""
    actual_map = dict(zip(df["category"], df["weight_pct"]))
    cats = sorted(set(list(targets.keys()) + df["category"].tolist()),
                  key=lambda c: -actual_map.get(c, 0))

    rows_html = ""
    for cat in cats:
        actual = actual_map.get(cat, 0.0)
        target = targets.get(cat, 0.0)
        rows_html += (
            f"<tr>"
            f"<td style='padding: 4px 8px; color: #c9d1d9;'>{cat}</td>"
            f"<td style='padding: 4px 8px; text-align: right; color: #f0f6fc; font-weight: 600;'>{actual:.1f}%</td>"
            f"<td style='padding: 4px 8px; text-align: right; color: #8b949e;'>{target:.1f}%</td>"
            f"</tr>"
        )

    st.markdown(
        f"<table style='width: 100%; border-collapse: collapse; font-size: 0.85rem;'>"
        f"<thead><tr>"
        f"<th style='text-align: left; padding: 4px 8px; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;'>Category</th>"
        f"<th style='text-align: right; padding: 4px 8px; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;'>Actual</th>"
        f"<th style='text-align: right; padding: 4px 8px; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;'>Target</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )