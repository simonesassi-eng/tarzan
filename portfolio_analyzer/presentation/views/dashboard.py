"""Dashboard page — portfolio snapshot with dynamic period selector."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.models.investor_config import InvestorConfig
from portfolio_analyzer.presentation.formatters import fmt_eur, fmt_pct, fmt_ratio, gain_color, rate_metric
from portfolio_analyzer.presentation.charts import donut, COLORS, GEO_COLORS
from portfolio_analyzer.engine.metrics import (
    compute_cagr, compute_max_drawdown, compute_sharpe, compute_sortino,
    compute_var, compute_cvar, _scale_or_nan, TRADING_DAYS, RISK_FREE_RATE,
)

# Period options: label → calendar days (None = ALL)
PERIODS = {
    "1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180,
    "YTD": -1, "1Y": 365, "ALL": None,
}


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    ph = metrics.portfolio_history

    # --- Hero ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<p style='color:#8b949e;margin:0'>Portfolio Value</p>"
            f"<h1 style='margin:0;font-size:2.5rem'>{fmt_eur(metrics.total_value)}</h1>",
            unsafe_allow_html=True,
        )
        gain_total = metrics.holdings_df["gain_eur"].sum() if not metrics.holdings_df.empty else 0
        cost_total = metrics.holdings_df["cost_basis_eur"].sum() if not metrics.holdings_df.empty else 0
        gain_pct = (gain_total / cost_total * 100) if cost_total > 0 else 0
        day_ret = metrics.performance.get("1d")
        c1 = gain_color(gain_total)
        c2 = gain_color(day_ret)
        st.markdown(
            f"<span style='background:rgba(63,185,80,0.12);color:{c1};padding:3px 10px;"
            f"border-radius:6px;font-size:0.85rem'>"
            f"{'▲' if gain_total >= 0 else '▼'} {fmt_eur(gain_total)} ({fmt_pct(gain_pct)}) all time</span>"
            f"&nbsp;&nbsp;"
            f"<span style='background:rgba(88,166,255,0.12);color:{c2};padding:3px 10px;"
            f"border-radius:6px;font-size:0.85rem'>"
            f"{fmt_pct(day_ret) if day_ret is not None else '—'} today</span>",
            unsafe_allow_html=True,
        )

    # --- Period selector ---
    st.markdown("---")
    period_labels = list(PERIODS.keys())
    selected = st.radio("Period", period_labels, index=6, horizontal=True, key="dash_period")

    # Slice portfolio history for selected period
    sub_ph = _slice_history(ph, selected)

    # --- Chart ---
    if sub_ph is not None and not sub_ph.empty:
        import plotly.graph_objects as go

        # Only plot the real (non-zero) portion for the line,
        # but keep the full x-axis range for context
        real_data = sub_ph[sub_ph > 0]
        fig = go.Figure()
        if not real_data.empty:
            fig.add_trace(go.Scatter(
                x=real_data.index, y=real_data.values,
                name="Portfolio", line=dict(color="#58a6ff", width=2.5),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
            ))

        # Scale Y axis: -5% of min, +5% of max
        if not real_data.empty:
            y_lo = float(real_data.min())
            y_hi = float(real_data.max())
            y_min = max(0, y_lo - abs(y_lo) * 0.05)
            y_max = y_hi + abs(y_hi) * 0.05
        else:
            y_min, y_max = 0, 100

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0", size=12),
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(showgrid=False, range=[sub_ph.index[0], sub_ph.index[-1]]),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       range=[y_min, y_max]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Metrics cards (grouped like Excel: Performance, Risk, Efficiency, Ex-Ante) ---
    period_metrics = _compute_period_metrics(sub_ph, metrics)

    col_perf, col_risk, col_eff, col_exante = st.columns(4)

    with col_perf:
        _card("📈 Performance", [
            ("Return", fmt_pct(period_metrics["return"]), *rate_metric("cagr", period_metrics["return"])),
            ("CAGR (ann.)", fmt_pct(period_metrics["cagr"]), *rate_metric("cagr", period_metrics["cagr"])),
            ("α Alpha", fmt_pct(period_metrics["alpha"]), *rate_metric("alpha", period_metrics["alpha"])),
            ("β Beta", fmt_ratio(period_metrics["beta"]), *rate_metric("beta", period_metrics["beta"])),
        ])

    with col_risk:
        _card("⚠️ Risk Profile", [
            ("Max Drawdown", fmt_pct(period_metrics["max_drawdown"]), *rate_metric("max_drawdown", period_metrics["max_drawdown"])),
            ("Volatility", fmt_pct(period_metrics["volatility"]), *rate_metric("volatility", period_metrics["volatility"])),
        ])

    with col_eff:
        _card("⚡ Efficiency", [
            ("Sharpe Ratio", fmt_ratio(period_metrics["sharpe"]), *rate_metric("sharpe", period_metrics["sharpe"])),
            ("Sortino Ratio", fmt_ratio(period_metrics["sortino"]), *rate_metric("sortino", period_metrics["sortino"])),
        ])

    with col_exante:
        v95 = period_metrics["var_95"]
        c95 = period_metrics["cvar_95"]
        _card("🛡️ Ex-Ante Risk", [
            ("VaR 95% Daily", fmt_pct(v95), *rate_metric("var_pct", abs(v95) if v95 == v95 else float("nan"))),
            ("CVaR 95% Daily", fmt_pct(c95), *rate_metric("cvar_pct", abs(c95) if c95 == c95 else float("nan"))),
        ])

    # --- Mini allocations ---
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if not metrics.allocation_by_class.empty:
            fig = donut(
                metrics.allocation_by_class["category"].tolist(),
                metrics.allocation_by_class["weight_pct"].tolist(),
                COLORS, "Asset Class",
            )
            st.plotly_chart(fig, use_container_width=True, key="dash_ac")
    with col_b:
        if not metrics.allocation_by_geo.empty:
            fig = donut(
                metrics.allocation_by_geo["category"].tolist(),
                metrics.allocation_by_geo["weight_pct"].tolist(),
                GEO_COLORS, "Geography (Equity)",
            )
            st.plotly_chart(fig, use_container_width=True, key="dash_geo")

    # --- Top 5 holdings ---
    st.markdown("---")
    st.markdown("##### 💼 Top Holdings")
    if not metrics.holdings_df.empty:
        top5 = metrics.holdings_df.nlargest(5, "weight_pct")
        for _, row in top5.iterrows():
            c = gain_color(row["gain_pct"])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid #21262d'>"
                f"<div><b>{row['name']}</b> <span style='color:#8b949e;font-size:0.8rem'>"
                f"{row['ticker']} · {row['asset_class']}</span></div>"
                f"<div style='text-align:right'><b>{fmt_eur(row['current_value'])}</b>"
                f" <span style='color:#8b949e'>{row['weight_pct']:.1f}%</span>"
                f" <span style='color:{c}'>{fmt_pct(row['gain_pct'])}</span></div></div>",
                unsafe_allow_html=True,
            )

    # --- Rebalancing alert ---
    if metrics.rebalancing_suggestions:
        st.markdown("---")
        n = len(metrics.rebalancing_suggestions)
        st.warning(f"⚖️ **{n} rebalancing action{'s' if n > 1 else ''}** suggested. "
                   f"Go to Rebalancing page for details.")


# ======================================================================
# Helpers
# ======================================================================

def _slice_history(ph, period_label: str) -> pd.Series:
    """Slice portfolio history to the selected period.
    
    If the selected period extends before the portfolio inception,
    pads with zeros so the X axis shows the full requested range.
    Metrics are computed only on the non-zero portion.
    """
    if ph is None or ph.empty:
        return pd.Series(dtype=float)
    if period_label == "ALL" or PERIODS.get(period_label) is None:
        return ph
    if period_label == "YTD":
        ytd = ph[ph.index.year == ph.index[-1].year]
        return ytd if not ytd.empty else ph

    days = PERIODS[period_label]
    cutoff = ph.index[-1] - pd.Timedelta(days=days)

    if cutoff < ph.index[0]:
        # Period extends before inception — pad with zeros
        pad_index = pd.date_range(start=cutoff, end=ph.index[0] - pd.Timedelta(days=1),
                                  freq="B", tz=ph.index.tz)
        pad_series = pd.Series(0.0, index=pad_index)
        return pd.concat([pad_series, ph])
    else:
        return ph[ph.index >= cutoff]


def _compute_period_metrics(ph: pd.Series, metrics: PortfolioMetrics) -> dict:
    """Compute risk/return metrics on a (possibly sliced) portfolio history.

    Only uses the non-zero portion (after inception) for calculations.
    Alpha/Beta come from the full metrics (need benchmark alignment).
    """
    result = {
        "return": 0.0, "cagr": 0.0, "volatility": 0.0,
        "sharpe": float("nan"), "sortino": float("nan"),
        "max_drawdown": 0.0,
        "var_95": float("nan"), "cvar_95": float("nan"),
        "alpha": float("nan"), "beta": float("nan"),
    }
    if ph is None or ph.empty or len(ph) < 2:
        return result

    # Filter out zero-padding (pre-inception)
    real_ph = ph[ph > 0]
    if real_ph.empty or len(real_ph) < 2:
        return result

    start_val = float(real_ph.iloc[0])
    end_val = float(real_ph.iloc[-1])
    total_return = (end_val / start_val - 1) * 100 if start_val > 0 else 0.0

    daily_returns = real_ph.pct_change().dropna()
    if daily_returns.empty:
        result["return"] = total_return
        return result

    ann_vol = float(daily_returns.std()) * np.sqrt(TRADING_DAYS) * 100
    ann_return = compute_cagr(real_ph)

    result["return"] = total_return
    result["cagr"] = ann_return
    result["volatility"] = ann_vol
    result["sharpe"] = compute_sharpe(ann_return, ann_vol)
    result["sortino"] = compute_sortino(daily_returns, ann_return)
    result["max_drawdown"] = compute_max_drawdown(real_ph) * 100
    result["var_95"] = _scale_or_nan(compute_var(daily_returns, 0.95), 100)
    result["cvar_95"] = _scale_or_nan(compute_cvar(daily_returns, 0.95), 100)
    result["alpha"] = metrics.risk.get("alpha", float("nan"))
    result["beta"] = metrics.risk.get("beta", float("nan"))
    return result


def _card(title: str, rows: list[tuple]):
    """Render a grouped metrics card with title and rows of (label, value, rating, color, emoji)."""
    html = (
        f"<div style='background:#161b22;border:1px solid #21262d;border-radius:10px;"
        f"padding:16px;height:100%'>"
        f"<div style='font-size:0.75rem;font-weight:700;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-bottom:12px;border-bottom:1px solid #21262d;padding-bottom:8px'>"
        f"{title}</div>"
    )
    for label, value, rating_label, rating_color, emoji in rows:
        html += (
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.03)'>"
            f"<span style='color:#c9d1d9;font-size:0.8rem'>{label}</span>"
            f"<div style='text-align:right'>"
            f"<span style='font-size:1.1rem;font-weight:700;color:#f0f6fc'>{value}</span>"
            f"<span style='font-size:0.65rem;color:{rating_color};margin-left:6px'>"
            f"{emoji} {rating_label}</span>"
            f"</div></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _kpi(col, label: str, value: str, rating_label: str, rating_color: str, emoji: str):
    col.markdown(
        f"<div style='text-align:center'>"
        f"<div style='font-size:0.7rem;color:#8b949e;text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#f0f6fc'>{value}</div>"
        f"<div style='font-size:0.7rem;color:{rating_color}'>{emoji} {rating_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )