"""Performance — mirrors Excel Performance sheet.

Unified table: TOTAL PORTFOLIO + holdings + benchmarks, with Period Used, α/β dynamic,
+ Legend of rating thresholds.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.presentation.charts import portfolio_line
from tarzan.presentation.formatters import fmt_pct
from tarzan import config as cfg


def render(metrics: PortfolioMetrics):
    st.markdown("## 📈 Performance")

    bench_beta_name = cfg.benchmark_beta_name()
    alpha_col = f"α (vs {bench_beta_name})"
    beta_col = f"β (vs {bench_beta_name})"

    st.caption(
        f"Period returns (1D–5Y) and risk metrics calculated on available history "
        f"per instrument (max 5 years). α/β computed vs **{bench_beta_name}**."
    )

    # =====================================================================
    # Chart: portfolio vs benchmarks overlay
    # =====================================================================
    if metrics.portfolio_history is not None and not metrics.portfolio_history.empty:
        fig = portfolio_line(metrics.portfolio_history, metrics.benchmark_histories,
                             "Portfolio vs Benchmarks")
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # Unified table: TOTAL PORTFOLIO + holdings + benchmarks
    # =====================================================================
    st.markdown("##### Performance Table")

    rows = []
    # Row 0: TOTAL PORTFOLIO
    port_full = metrics.performance_full or {}
    if port_full:
        port_row = {"Name": "** TOTAL PORTFOLIO **", "Type": "Portfolio"}
        _add_metrics_to_row(port_row, port_full, alpha_col, beta_col)
        rows.append(port_row)

    # Rows 1..N: holdings + benchmarks from holding_performance
    if not metrics.holding_performance.empty:
        hp = metrics.holding_performance.sort_values(by="type", ascending=True, kind="stable")
        for _, hr in hp.iterrows():
            r = {"Name": hr.get("name", ""), "Type": hr.get("type", "")}
            _add_metrics_to_row(r, hr.to_dict(), alpha_col, beta_col)
            rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        _render_performance_table(df, alpha_col, beta_col)
    else:
        st.info("No performance data available.")

    # =====================================================================
    # Legend — Rating Thresholds
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Legend — Rating Thresholds", expanded=False):
        _render_legend(alpha_col, beta_col)


def _add_metrics_to_row(row: dict, data: dict, alpha_col: str, beta_col: str):
    """Copy metrics from a source dict into the display row with renamed columns."""
    mapping = {
        "1d": "1D", "1w": "1W", "1m": "1M", "3m": "3M", "6m": "6M",
        "ytd": "YTD", "1y": "1Y", "3y": "3Y", "5y": "5Y",
        "cagr": "CAGR", "volatility": "Volatility", "sharpe": "Sharpe",
        "sortino": "Sortino", "max_drawdown": "Max DD",
        "alpha": alpha_col, "beta": beta_col,
    }
    for src, dst in mapping.items():
        row[dst] = data.get(src)
    row["Period Used"] = data.get("period_used", "—")


def _render_performance_table(df: pd.DataFrame, alpha_col: str, beta_col: str):
    """Render the performance DataFrame with proper formatting and colors."""
    pct_cols = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y",
                "CAGR", "Volatility", "Max DD", alpha_col]
    ratio_cols = ["Sharpe", "Sortino", beta_col]
    return_cols = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "CAGR", alpha_col]

    fmt = {c: "{:+.2f}%" for c in pct_cols if c in df.columns}
    for c in ratio_cols:
        if c in df.columns:
            fmt[c] = "{:.2f}"

    styled = df.style.format(fmt, na_rep="—")

    # Color returns green/red
    existing_return_cols = [c for c in return_cols if c in df.columns]
    if existing_return_cols:
        styled = styled.applymap(
            lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
            else "color: #f85149" if isinstance(v, (int, float)) and v < 0
            else "", subset=existing_return_cols,
        )

    # Highlight portfolio row
    def _highlight_portfolio(row):
        if row.get("Name", "").startswith("** TOTAL PORTFOLIO"):
            return ["background-color: rgba(88,166,255,0.08); font-weight: 700;"] * len(row)
        return [""] * len(row)

    styled = styled.apply(_highlight_portfolio, axis=1)

    st.dataframe(styled, use_container_width=True,
                 height=min(len(df) * 38 + 60, 800))


def _render_legend(alpha_col: str, beta_col: str):
    """Render rating thresholds legend."""
    ratings = cfg.metric_ratings() or {}

    legend_rows = [
        ("CAGR", "cagr", "Equity risk premium"),
        (alpha_col, "alpha", "Jensen's Alpha (CAPM)"),
        (beta_col, "beta", "CAPM, 1.0 = market"),
        ("Max DD", "max_drawdown", "Retail drawdown tolerance"),
        ("Volatility", "volatility", "Historical equity vol ~15%"),
        ("Sharpe", "sharpe", "Sharpe (1994)"),
        ("Sortino", "sortino", "Sortino & Price (1994)"),
        ("VaR 95%", "var_pct", "Basel III retail adj."),
        ("CVaR 95%", "cvar_pct", "Artzner et al. (1999)"),
    ]

    legend_data = []
    for metric_label, key, source in legend_rows:
        spec = ratings.get(key, {})
        thresholds = spec.get("thresholds", [None, None])
        invert = spec.get("invert", False)
        good_t, warn_t = thresholds[0], thresholds[1]

        if invert:
            strong = f"< {abs(good_t):.1f}%" if good_t is not None else "—"
            fair = f"{abs(warn_t):.1f}% – {abs(good_t):.1f}%" if good_t is not None and warn_t is not None else "—"
            weak = f"> {abs(warn_t):.1f}%" if warn_t is not None else "—"
        else:
            strong = f"> {good_t}" if good_t is not None else "—"
            fair = f"{warn_t} – {good_t}" if good_t is not None and warn_t is not None else "—"
            weak = f"< {warn_t}" if warn_t is not None else "—"

        legend_data.append({
            "Metric": metric_label,
            "🟢 Strong": strong,
            "🟡 Fair": fair,
            "🔴 Weak": weak,
            "Source": source,
        })

    df = pd.DataFrame(legend_data)
    st.dataframe(df, use_container_width=True, hide_index=True)