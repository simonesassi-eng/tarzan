"""Performance page — portfolio returns + per-holding performance table."""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.presentation.charts import period_returns_bar
from portfolio_analyzer.presentation.formatters import fmt_pct


def render(metrics: PortfolioMetrics):
    st.markdown("## 📈 Performance")

    perf = metrics.performance
    if not perf:
        st.info("No performance data available.")
        return

    # Portfolio period returns chart
    fig = period_returns_bar(perf, "Portfolio Period Returns")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**CAGR**: {fmt_pct(perf.get('cagr'))} · **YTD**: {fmt_pct(perf.get('ytd'))}")

    # Per-holding performance table
    hp = metrics.holding_performance
    if not hp.empty:
        st.markdown("---")
        st.markdown("##### Per-Holding Performance")

        display_cols = [c for c in ["name", "ticker", "1d", "1w", "1m", "3m", "6m",
                                     "ytd", "1y", "3y", "5y", "cagr",
                                     "volatility", "sharpe", "max_drawdown"]
                        if c in hp.columns]
        show = hp[display_cols].copy()

        col_map = {
            "name": "Name", "ticker": "Ticker",
            "1d": "1D", "1w": "1W", "1m": "1M", "3m": "3M", "6m": "6M",
            "ytd": "YTD", "1y": "1Y", "3y": "3Y", "5y": "5Y",
            "cagr": "CAGR", "volatility": "Vol %", "sharpe": "Sharpe",
            "max_drawdown": "Max DD %",
        }
        show.columns = [col_map.get(c, c) for c in display_cols]

        pct_cols = [c for c in ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y",
                                "CAGR", "Vol %", "Max DD %"] if c in show.columns]
        fmt = {c: "{:+.1f}%" for c in pct_cols}
        if "Sharpe" in show.columns:
            fmt["Sharpe"] = "{:.2f}"

        return_cols = [c for c in ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "CAGR"]
                       if c in show.columns]

        styled = show.style.format(fmt, na_rep="—")
        if return_cols:
            styled = styled.applymap(
                lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                else "color: #f85149" if isinstance(v, (int, float)) and v < 0
                else "", subset=return_cols
            )

        st.dataframe(styled, use_container_width=True,
                      height=min(len(show) * 38 + 40, 600))
