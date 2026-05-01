"""Backtest page — full historical analysis with benchmarks."""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.presentation.charts import portfolio_line, drawdown_chart
from portfolio_analyzer.presentation.formatters import fmt_pct, fmt_ratio, rate_metric
from portfolio_analyzer.engine.metrics import (
    compute_cagr, compute_max_drawdown, compute_sharpe, compute_sortino,
    compute_var, compute_cvar, _scale_or_nan, TRADING_DAYS, RISK_FREE_RATE,
)


def render(metrics: PortfolioMetrics):
    st.markdown("## 📉 Backtest Analysis")

    ph = metrics.portfolio_history
    if ph is None or ph.empty:
        st.info("No portfolio history available for backtest.")
        return

    # --- KPI strip (full history) ---
    daily_returns = ph.pct_change().dropna()
    ann_return = compute_cagr(ph)
    ann_vol = float(daily_returns.std()) * np.sqrt(TRADING_DAYS) * 100 if not daily_returns.empty else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _kpi(k1, "CAGR", fmt_pct(ann_return), *rate_metric("cagr", ann_return))
    _kpi(k2, "Volatility", fmt_pct(ann_vol), *rate_metric("volatility", ann_vol))
    sharpe = compute_sharpe(ann_return, ann_vol)
    _kpi(k3, "Sharpe", fmt_ratio(sharpe), *rate_metric("sharpe", sharpe))
    mdd = compute_max_drawdown(ph) * 100
    _kpi(k4, "Max DD", fmt_pct(mdd), *rate_metric("max_drawdown", mdd))
    var95 = _scale_or_nan(compute_var(daily_returns, 0.95), 100) if not daily_returns.empty else float("nan")
    _kpi(k5, "VaR 95%", fmt_pct(var95), *rate_metric("var_pct", abs(var95) if var95 == var95 else float("nan")))
    cvar95 = _scale_or_nan(compute_cvar(daily_returns, 0.95), 100) if not daily_returns.empty else float("nan")
    _kpi(k6, "CVaR 95%", fmt_pct(cvar95), *rate_metric("cvar_pct", abs(cvar95) if cvar95 == cvar95 else float("nan")))

    # Additional metrics row
    risk = metrics.risk
    s1, s2, s3 = st.columns(3)
    sortino = compute_sortino(daily_returns, ann_return) if not daily_returns.empty else float("nan")
    _kpi(s1, "Sortino", fmt_ratio(sortino), *rate_metric("sortino", sortino))
    _kpi(s2, "Beta", fmt_ratio(risk.get("beta")), *rate_metric("beta", risk.get("beta")))
    _kpi(s3, "Alpha", fmt_pct(risk.get("alpha")), *rate_metric("alpha", risk.get("alpha")))

    # --- Portfolio vs Benchmarks chart ---
    st.markdown("---")
    fig = portfolio_line(ph, metrics.benchmark_histories, "Portfolio vs Benchmarks")
    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown ---
    st.markdown("---")
    fig = drawdown_chart(ph, "Drawdown from Peak")
    st.plotly_chart(fig, use_container_width=True)

    # --- Yearly stats ---
    st.markdown("---")
    st.markdown("##### 📅 Annual Statistics")
    yearly = _compute_yearly_stats(ph)
    if not yearly.empty:
        pct_cols = [c for c in ["Return %", "Volatility %", "Max DD %"] if c in yearly.columns]
        fmt = {c: "{:+.1f}%" for c in pct_cols}
        fmt["Sharpe"] = "{:.2f}"
        fmt["Start Value"] = "€{:,.0f}"
        fmt["End Value"] = "€{:,.0f}"

        styled = yearly.style.format(fmt, na_rep="—")
        if "Return %" in yearly.columns:
            styled = styled.applymap(
                lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                else "color: #f85149" if isinstance(v, (int, float)) and v < 0
                else "", subset=["Return %"]
            )
        st.dataframe(styled, use_container_width=True)

    # --- Rolling Sharpe ---
    st.markdown("---")
    st.markdown("##### Rolling Sharpe Ratio (252-day)")
    _rolling_sharpe_chart(ph)


# ======================================================================
# Helpers
# ======================================================================

def _compute_yearly_stats(ph: pd.Series) -> pd.DataFrame:
    rows = []
    for year in sorted(ph.index.year.unique()):
        yearly = ph[ph.index.year == year]
        if len(yearly) < 2:
            continue
        start_val = float(yearly.iloc[0])
        end_val = float(yearly.iloc[-1])
        ret = (end_val / start_val - 1) * 100
        daily_ret = yearly.pct_change().dropna()
        vol = float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100 if len(daily_ret) > 0 else 0
        sharpe = compute_sharpe(ret, vol) if vol > 0 else float("nan")
        mdd = compute_max_drawdown(yearly) * 100
        rows.append({"Year": year, "Return %": ret, "Volatility %": vol,
                      "Sharpe": sharpe, "Max DD %": mdd,
                      "Start Value": start_val, "End Value": end_val})
    return pd.DataFrame(rows)


def _rolling_sharpe_chart(ph: pd.Series):
    daily_ret = ph.pct_change().dropna()
    if len(daily_ret) < TRADING_DAYS:
        st.info("Not enough data for rolling Sharpe (need 252+ days).")
        return
    rolling_mean = daily_ret.rolling(TRADING_DAYS).mean() * TRADING_DAYS
    rolling_std = daily_ret.rolling(TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe = (rolling_mean * 100 - RISK_FREE_RATE) / (rolling_std * 100)
    rolling_sharpe = rolling_sharpe.dropna()

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        line=dict(color="#58a6ff", width=1.5), name="Rolling Sharpe",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#3fb950", opacity=0.5,
                  annotation_text="Good (1.0)")
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.3)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Sharpe"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _kpi(col, label: str, value: str, rating_label: str, rating_color: str, emoji: str):
    col.markdown(
        f"<div style='text-align:center'>"
        f"<div style='font-size:0.7rem;color:#8b949e;text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#f0f6fc'>{value}</div>"
        f"<div style='font-size:0.7rem;color:{rating_color}'>{emoji} {rating_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )