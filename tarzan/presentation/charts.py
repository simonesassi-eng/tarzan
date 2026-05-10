"""Plotly chart factory for the Streamlit UI.

Single module generating all chart types with consistent dark theme styling.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

# Consistent color palette
COLORS = {
    "Equities": "#58a6ff", "Fixed Income": "#3fb950",
    "Cash & Cash Equivalents": "#d2a8ff", "Gold": "#f2c94c", "Commodities": "#f0883e",
    "Real Estate": "#8b4513", "Alternative": "#8b949e",
}
GEO_COLORS = {
    "USA": "#58a6ff", "Eurozone EMU": "#3fb950",
    "Dev ex-USA ex-EMU ex-JP": "#d2a8ff", "Emerging Markets": "#f0883e",
    "Japan": "#f85149", "Other": "#8b949e",
}
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0e0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


def portfolio_line(
    portfolio_history: pd.Series,
    benchmark_histories: Optional[dict] = None,
    title: str = "Portfolio Value",
) -> go.Figure:
    """Line chart: portfolio value over time with optional benchmark overlay."""
    fig = go.Figure()
    if portfolio_history is not None and not portfolio_history.empty:
        fig.add_trace(go.Scatter(
            x=portfolio_history.index, y=portfolio_history.values,
            name="Portfolio", line=dict(color="#58a6ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        ))
    if benchmark_histories:
        dash_styles = ["dash", "dot", "dashdot"]
        for i, (name, series) in enumerate(benchmark_histories.items()):
            if series is not None and not series.empty:
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, name=name,
                    line=dict(color="#8b949e", width=1.2, dash=dash_styles[i % len(dash_styles)]),
                    opacity=0.6,
                ))
    fig.update_layout(**_LAYOUT, title=title, xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"))
    return fig


def donut(labels: list[str], values: list[float], color_map: dict, title: str = "") -> go.Figure:
    """Donut/pie chart for allocation breakdowns."""
    colors = [color_map.get(l, "#8b949e") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f1117", width=2)),
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(**_LAYOUT, title=title, showlegend=False)
    return fig


def delta_bars(
    categories: list[str], actuals: list[float], targets: list[float], title: str = "",
) -> go.Figure:
    """Horizontal grouped bar chart showing actual vs target."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories, x=actuals, name="Actual", orientation="h",
        marker_color="#58a6ff",
        text=[f"{v:.1f}%" for v in actuals], textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        y=categories, x=targets, name="Target", orientation="h",
        marker_color="#f0883e", marker_opacity=0.6,
        text=[f"{v:.1f}%" for v in targets], textposition="outside",
        textfont=dict(size=11),
    ))
    max_val = max(max(actuals, default=0), max(targets, default=0))
    fig.update_layout(**_LAYOUT, title=title, barmode="group",
                      xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                 range=[0, max_val * 1.25]),
                      yaxis=dict(autorange="reversed"),
                      bargap=0.3, bargroupgap=0.1)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def drawdown_chart(portfolio_history: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Area chart showing drawdown from peak over time."""
    if portfolio_history is None or portfolio_history.empty:
        return go.Figure()
    cummax = portfolio_history.cummax()
    dd = (portfolio_history - cummax) / cummax * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy",
        fillcolor="rgba(248,81,73,0.15)", line=dict(color="#f85149", width=1.5),
        name="Drawdown",
    ))
    fig.update_layout(**_LAYOUT, title=title, xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="%"))
    return fig


def period_returns_bar(performance: dict, title: str = "Period Returns") -> go.Figure:
    """Bar chart of period returns (1d, 1w, ..., 5y)."""
    keys = ["1d", "1w", "1m", "3m", "6m", "ytd", "1y", "3y", "5y"]
    labels = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]
    vals = [performance.get(k) for k in keys]
    colors = ["#3fb950" if v is not None and v >= 0 else "#f85149" for v in vals]
    clean_vals = [v if v is not None else 0 for v in vals]
    fig = go.Figure(go.Bar(
        x=labels, y=clean_vals, marker_color=colors,
        text=[f"{v:+.1f}%" if v != 0 else "—" for v in clean_vals],
        textposition="outside",
    ))
    fig.update_layout(**_LAYOUT, title=title, xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="%"))
    return fig