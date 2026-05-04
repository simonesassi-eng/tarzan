"""Return Contribution — mirrors Excel Return Contribution sheet.

Shows how each holding contributed to overall portfolio return (weight × return).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from tarzan.models.portfolio import PortfolioMetrics


def render(metrics: PortfolioMetrics):
    st.markdown("## 🌊 Return Contribution")
    st.caption("How much each holding contributed to the total portfolio return.")

    if metrics.holdings_df.empty:
        st.info("No data available.")
        return

    df = metrics.holdings_df.copy()
    # Contribution = weight% × gain% / 100
    df["contribution_pct"] = (df["weight_pct"] / 100) * df["gain_pct"]
    df = df.sort_values("contribution_pct", ascending=False)

    # Waterfall-like chart
    fig = go.Figure(go.Bar(
        x=df["contribution_pct"],
        y=df["name"].str[:40],
        orientation="h",
        marker_color=["#3fb950" if v >= 0 else "#f85149" for v in df["contribution_pct"]],
        text=[f"{v:+.2f}%" for v in df["contribution_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), margin=dict(l=10, r=40, t=30, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Contribution to Portfolio Return (%)"),
        yaxis=dict(autorange="reversed"),
        height=max(400, len(df) * 30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("##### Summary")
    summary = df[["name", "ticker", "weight_pct", "gain_pct", "contribution_pct"]].copy()
    summary.columns = ["Name", "Ticker", "Weight %", "Gain %", "Contribution %"]
    st.dataframe(
        summary.style.format({
            "Weight %": "{:.1f}%",
            "Gain %": "{:+.1f}%",
            "Contribution %": "{:+.2f}%",
        }).applymap(
            lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
            else "color: #f85149" if isinstance(v, (int, float)) and v < 0
            else "", subset=["Gain %", "Contribution %"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    total_contrib = df["contribution_pct"].sum()
    st.caption(f"**Total portfolio return:** {total_contrib:+.2f}% (sum of all contributions)")
