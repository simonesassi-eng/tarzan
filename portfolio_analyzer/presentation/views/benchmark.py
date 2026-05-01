"""Benchmark page — portfolio vs benchmarks, what-if scenarios."""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.presentation.formatters import fmt_pct, fmt_ratio
from portfolio_analyzer.presentation.charts import portfolio_line


def render(metrics: PortfolioMetrics):
    st.markdown("## 🏆 Benchmark Comparison")

    # Overlay chart
    if metrics.portfolio_history is not None and not metrics.portfolio_history.empty:
        fig = portfolio_line(metrics.portfolio_history, metrics.benchmark_histories,
                             "Portfolio vs Benchmarks")
        st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    if not metrics.benchmark_comparison.empty:
        st.markdown("---")
        st.markdown("##### Performance Comparison")
        df = metrics.benchmark_comparison.copy()

        # Add portfolio row at top
        perf = metrics.performance
        risk = metrics.risk
        port_row = {"benchmark": "📊 Your Portfolio", "cagr": perf.get("cagr", 0),
                    "1d": perf.get("1d"), "1w": perf.get("1w"), "1m": perf.get("1m"),
                    "3m": perf.get("3m"), "6m": perf.get("6m"), "ytd": perf.get("ytd"),
                    "1y": perf.get("1y"), "3y": perf.get("3y"), "5y": perf.get("5y"),
                    "volatility": risk.get("volatility", 0),
                    "sharpe": risk.get("sharpe", 0),
                    "max_drawdown": risk.get("max_drawdown", 0)}
        import pandas as pd
        df = pd.concat([pd.DataFrame([port_row]), df], ignore_index=True)

        display_cols = [c for c in ["benchmark", "cagr", "ytd", "1y", "3y", "5y",
                                     "volatility", "sharpe", "max_drawdown"] if c in df.columns]
        show = df[display_cols].copy()
        col_map = {"benchmark": "Benchmark", "cagr": "CAGR", "ytd": "YTD", "1y": "1Y",
                   "3y": "3Y", "5y": "5Y", "volatility": "Vol", "sharpe": "Sharpe",
                   "max_drawdown": "Max DD"}
        show.columns = [col_map.get(c, c) for c in display_cols]

        fmt_dict = {c: "{:+.1f}%" for c in ["CAGR", "YTD", "1Y", "3Y", "5Y", "Vol", "Max DD"] if c in show.columns}
        if "Sharpe" in show.columns:
            fmt_dict["Sharpe"] = "{:.2f}"

        st.dataframe(
            show.style.format(fmt_dict, na_rep="—").applymap(
                lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                else "color: #f85149" if isinstance(v, (int, float)) and v < 0
                else "", subset=[c for c in ["CAGR", "YTD", "1Y", "3Y", "5Y"] if c in show.columns]
            ),
            use_container_width=True,
            height=min(len(show) * 38 + 40, 800),
        )

        # Ranking
        if "CAGR" in show.columns and len(show) > 1:
            port_cagr = show.iloc[0]["CAGR"]
            if isinstance(port_cagr, (int, float)) and port_cagr == port_cagr:
                rank = (show["CAGR"].dropna() > port_cagr).sum() + 1
                total = len(show["CAGR"].dropna())
                st.markdown(f"Your portfolio ranks **#{rank}/{total}** by CAGR among these benchmarks.")
    else:
        st.info("No benchmark data available. Ensure yfinance can fetch benchmark tickers.")
