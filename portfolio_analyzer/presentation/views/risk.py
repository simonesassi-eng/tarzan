"""Risk page — Sharpe, Sortino, VaR, CVaR, drawdown, rolling volatility."""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.presentation.formatters import fmt_pct, fmt_ratio, rate_metric
from portfolio_analyzer.presentation.charts import drawdown_chart


def render(metrics: PortfolioMetrics):
    st.markdown("## ⚡ Risk Analytics")

    risk = metrics.risk
    if not risk:
        st.info("No risk data available.")
        return

    # Risk scorecard — 4x2 grid
    _metric_cards(risk)

    # Drawdown chart
    st.markdown("---")
    st.markdown("##### Drawdown Over Time")
    fig = drawdown_chart(metrics.portfolio_history)
    st.plotly_chart(fig, use_container_width=True)

    # Metric explanations (tooltip-style)
    with st.expander("📖 Metric Definitions"):
        st.markdown("""
- **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility. Measures excess return per unit of total risk. >1.0 is good.
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility. Better for asymmetric returns. >1.5 is good.
- **VaR 95%**: Maximum expected daily loss at 95% confidence (historical simulation). Non-parametric, robust for fat tails.
- **CVaR 95%**: Average loss in the worst 5% of days. Coherent risk measure per Artzner et al. (1999).
- **Max Drawdown**: Largest peak-to-trough decline. Shows worst-case scenario experienced.
- **Volatility**: Annualized standard deviation of daily returns. sqrt(252) scaling.
- **Beta**: Portfolio sensitivity to benchmark (S&P 500). <1 = defensive, >1 = aggressive.
- **Alpha**: Excess return vs CAPM prediction. Positive = outperforming risk-adjusted benchmark.
        """)


def _metric_cards(risk: dict):
    """Render 2 rows of 4 risk metric cards."""
    specs = [
        ("Sharpe", "sharpe", fmt_ratio(risk.get("sharpe")), "sharpe"),
        ("Sortino", "sortino", fmt_ratio(risk.get("sortino")), "sortino"),
        ("Max Drawdown", "max_drawdown", fmt_pct(risk.get("max_drawdown")), "max_drawdown"),
        ("VaR 95%", "var_95", fmt_pct(risk.get("var_95")), "var_pct"),
        ("CVaR 95%", "cvar_95", fmt_pct(risk.get("cvar_95")), "cvar_pct"),
        ("Volatility", "volatility", fmt_pct(risk.get("volatility")), "volatility"),
        ("Beta", "beta", fmt_ratio(risk.get("beta")), "beta"),
        ("Alpha", "alpha", fmt_pct(risk.get("alpha")), "alpha"),
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    for i, (label, key, formatted, rating_key) in enumerate(specs):
        col = row1[i] if i < 4 else row2[i - 4]
        val = risk.get(key)
        r_label, r_color, r_emoji = rate_metric(rating_key, val)
        col.markdown(
            f"<div style='background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px;text-align:center'>"
            f"<div style='font-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px'>{label}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{r_color};margin:4px 0'>{formatted}</div>"
            f"<div style='font-size:0.65rem;color:{r_color}'>{r_emoji} {r_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
