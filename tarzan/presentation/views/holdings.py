"""Holdings page — full interactive holdings table."""

from __future__ import annotations

import streamlit as st

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.presentation.formatters import fmt_eur, fmt_pct


def render(metrics: PortfolioMetrics):
    st.markdown("## 💼 Holdings")

    if metrics.holdings_df.empty:
        st.info("No holdings data available.")
        return

    df = metrics.holdings_df.copy()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        classes = ["All"] + sorted(df["asset_class"].unique().tolist())
        selected_class = st.selectbox("Filter by Asset Class", classes)
    with col2:
        sort_by = st.selectbox("Sort by", ["weight_pct", "current_value", "gain_pct", "gain_eur", "name"], index=0)

    if selected_class != "All":
        df = df[df["asset_class"] == selected_class]

    df = df.sort_values(sort_by, ascending=(sort_by == "name")).reset_index(drop=True)

    # Main table columns (ISIN and Geography go to Technical Details)
    display_cols = ["name", "ticker", "asset_class", "security_type",
                    "currency", "quantity", "avg_purchase_price", "current_price",
                    "cost_basis_eur", "current_value",
                    "weight_pct", "weight_of_invested_pctg", "pct_of_class",
                    "gain_eur", "gain_pct"]
    # Only include columns that exist in the dataframe
    display_cols = [c for c in display_cols if c in df.columns]

    show_df = df[display_cols].copy()
    col_names = {
        "name": "Name", "ticker": "Ticker",
        "asset_class": "Class", "security_type": "Security Type",
        "currency": "Ccy", "quantity": "Qty",
        "avg_purchase_price": "Avg Price", "current_price": "Price",
        "cost_basis_eur": "Cost €", "current_value": "Value €",
        "weight_pct": "% of Portfolio",
        "weight_of_invested_pctg": "% of Invested",
        "pct_of_class": "% of Class",
        "gain_eur": "Gain €", "gain_pct": "Gain %",
    }
    show_df.columns = [col_names.get(c, c) for c in display_cols]

    # Formatting
    fmt = {}
    for col in show_df.columns:
        if col in ("Value €", "Cost €", "Gain €"):
            fmt[col] = "€{:,.2f}"
        elif col in ("Avg Price", "Price"):
            fmt[col] = "{:,.2f}"
        elif col == "Qty":
            fmt[col] = "{:,.2f}"
        elif col in ("% of Portfolio", "% of Invested", "% of Class"):
            fmt[col] = "{:.1f}%"
        elif col == "Gain %":
            fmt[col] = "{:+.1f}%"

    # Color gain columns + weight columns (text color, not background)
    gain_cols = [c for c in ["Gain %", "Gain €"] if c in show_df.columns]
    weight_cols = [
        c for c in ["% of Portfolio", "% of Invested", "% of Class"]
        if c in show_df.columns
    ]
    colored_cols = gain_cols + weight_cols

    styled = show_df.style.format(fmt, na_rep="—")
    styled = styled.applymap(
        lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
        else "color: #f85149" if isinstance(v, (int, float)) and v < 0
        else "", subset=gain_cols
    )
    # Weight columns: darker blue = higher weight
    if weight_cols:
        def _weight_color(v):
            if not isinstance(v, (int, float)) or v != v:
                return ""
            if v >= 20:
                return "color: #58a6ff; font-weight: 700"
            if v >= 10:
                return "color: #79b8ff"
            if v >= 5:
                return "color: #a5d6ff"
            return "color: #8b949e"
        styled = styled.applymap(_weight_color, subset=weight_cols)

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(len(show_df) * 38 + 40, 800),
    )

    # Summary
    total_cost = df["cost_basis_eur"].sum()
    total_val = df["current_value"].sum()
    total_gain = df["gain_eur"].sum()
    gain_pct = total_gain / total_cost * 100 if total_cost > 0 else 0
    st.markdown(f"**{len(df)} holdings** · Total: {fmt_eur(total_val)} · "
                f"Cost: {fmt_eur(total_cost)} · Gain: {fmt_eur(total_gain)} ({fmt_pct(gain_pct)})")

    # Technical details (expandable)
    tech_cols = [c for c in ["isin", "geography", "geo_source", "data_source", "fetch_timestamp"] if c in df.columns]
    if tech_cols:
        with st.expander("🔧 Technical Details"):
            tech_df = df[["name", "ticker"] + tech_cols].copy()
            tech_names = {"isin": "ISIN", "geography": "Geography",
                          "geo_source": "Geo Source", "data_source": "Data Source",
                          "fetch_timestamp": "Fetch Time"}
            tech_df.columns = ["Name", "Ticker"] + [tech_names.get(c, c) for c in tech_cols]
            st.dataframe(tech_df, use_container_width=True)