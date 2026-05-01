"""Allocation page — asset class, geography, sector with actual vs target."""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.models.investor_config import InvestorConfig
from portfolio_analyzer.presentation.charts import donut, delta_bars, COLORS, GEO_COLORS
from portfolio_analyzer.presentation.formatters import fmt_pct


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    st.markdown("## 🎯 Allocation")

    # Asset Class
    st.markdown("### Asset Class")
    col1, col2 = st.columns(2)
    with col1:
        if not metrics.allocation_by_class.empty:
            fig = donut(
                metrics.allocation_by_class["category"].tolist(),
                metrics.allocation_by_class["weight_pct"].tolist(),
                COLORS, "",
            )
            st.plotly_chart(fig, use_container_width=True, key="alloc_ac_donut")
    with col2:
        if config and not metrics.allocation_by_class.empty:
            cats = sorted(set(list(config.allocation_targets.keys()) +
                              metrics.allocation_by_class["category"].tolist()))
            actual_map = dict(zip(metrics.allocation_by_class["category"],
                                  metrics.allocation_by_class["weight_pct"]))
            actuals = [actual_map.get(c, 0) for c in cats]
            targets = [config.allocation_targets.get(c, 0) for c in cats]
            fig = delta_bars(cats, actuals, targets, "Actual vs Target")
            st.plotly_chart(fig, use_container_width=True, key="alloc_ac_delta")
            # Detail table
            for c, a, t in zip(cats, actuals, targets):
                delta = a - t
                color = "#3fb950" if abs(delta) < 3 else "#f0883e" if abs(delta) < 5 else "#f85149"
                st.markdown(f"<span style='color:{color}'>{c}: {a:.1f}% actual → {t:.1f}% target (Δ {delta:+.1f})</span>",
                            unsafe_allow_html=True)

    # Geography
    st.markdown("---")
    st.markdown("### Geographic Allocation (Equity Only)")
    col1, col2 = st.columns(2)
    with col1:
        if not metrics.allocation_by_geo.empty:
            fig = donut(
                metrics.allocation_by_geo["category"].tolist(),
                metrics.allocation_by_geo["weight_pct"].tolist(),
                GEO_COLORS, "",
            )
            st.plotly_chart(fig, use_container_width=True, key="alloc_geo_donut")
    with col2:
        if config and not metrics.allocation_by_geo.empty:
            cats = sorted(set(list(config.geo_allocation.keys()) +
                              metrics.allocation_by_geo["category"].tolist()))
            actual_map = dict(zip(metrics.allocation_by_geo["category"],
                                  metrics.allocation_by_geo["weight_pct"]))
            actuals = [actual_map.get(c, 0) for c in cats]
            targets = [config.geo_allocation.get(c, 0) for c in cats]
            fig = delta_bars(cats, actuals, targets, "Actual vs Target (Geo)")
            st.plotly_chart(fig, use_container_width=True, key="alloc_geo_delta")
            for c, a, t in zip(cats, actuals, targets):
                delta = a - t
                color = "#3fb950" if abs(delta) < 3 else "#f0883e" if abs(delta) < 5 else "#f85149"
                st.markdown(f"<span style='color:{color}'>{c}: {a:.1f}% actual → {t:.1f}% target (Δ {delta:+.1f})</span>",
                            unsafe_allow_html=True)

    # ACWI benchmark geo
    if metrics.acwi_geo:
        st.markdown("---")
        st.markdown("##### 🌍 MSCI ACWI Geographic Reference")
        for geo, pct in sorted(metrics.acwi_geo.items(), key=lambda x: -x[1]):
            st.markdown(f"- {geo}: {pct:.1f}%")
