"""Performance — portfolio chart, drawdown, risk KPIs, unified table."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.presentation.charts import portfolio_line, drawdown_chart
from tarzan.presentation.formatters import fmt_pct
from tarzan import config as cfg


def render(metrics: PortfolioMetrics):
    st.markdown("## 📈 Performance")

    bench_beta_name = cfg.benchmark_beta_name()
    alpha_col = f"α (vs {bench_beta_name})"
    beta_col  = f"β (vs {bench_beta_name})"

    st.caption(
        f"Period returns (1D–5Y) and risk metrics calculated on available history "
        f"per instrument (max 5 years). α/β computed vs **{bench_beta_name}**."
    )

    # ── Portfolio risk KPI bar ─────────────────────────────────────────
    risk = metrics.risk or {}
    _render_risk_kpis(risk, alpha_col, beta_col)

    # ── Portfolio vs Benchmarks chart ─────────────────────────────────
    if metrics.portfolio_history is not None and not metrics.portfolio_history.empty:
        st.markdown("#### 📊 Portfolio vs Benchmarks")
        fig = portfolio_line(
            metrics.portfolio_history, metrics.benchmark_histories, ""
        )
        st.plotly_chart(fig, use_container_width=True, key="perf_portfolio_line")

    # ── Drawdown chart ─────────────────────────────────────────────────
    if metrics.portfolio_history is not None and not metrics.portfolio_history.empty:
        st.markdown("#### 📉 Drawdown from Peak")
        fig_dd = drawdown_chart(metrics.portfolio_history, "")
        fig_dd.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_dd, use_container_width=True, key="perf_drawdown")

    # ── Benchmark comparison table ─────────────────────────────────────
    if not metrics.benchmark_comparison.empty:
        st.markdown("#### 🏁 Benchmark Comparison")
        _render_benchmark_table(metrics.benchmark_comparison)

    # ── Unified performance table ──────────────────────────────────────
    st.markdown("#### 📋 Full Performance Table")
    rows = []
    port_full = metrics.performance_full or {}
    if port_full:
        port_row = {"Name": "** TOTAL PORTFOLIO **", "Type": "Portfolio"}
        _add_metrics_to_row(port_row, port_full, alpha_col, beta_col)
        rows.append(port_row)

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

    # ── Legend ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 Legend — Rating Thresholds", expanded=False):
        _render_legend(alpha_col, beta_col)


# ── Risk KPI bar ───────────────────────────────────────────────────────

def _render_risk_kpis(risk: dict, alpha_col: str, beta_col: str):
    """Render a row of risk metric cards."""
    def _v(key, pct=True):
        val = risk.get(key)
        if val is None or str(val) == "nan":
            return "—", None
        return (f"{val:.2f}%" if pct else f"{val:.2f}"), val

    pairs = [
        ("Volatility", _v("volatility")),
        ("Sharpe",     _v("sharpe", pct=False)),
        ("Sortino",    _v("sortino", pct=False)),
        ("Max DD",     _v("max_drawdown")),
        ("VaR 95%",    _v("var_95")),
        ("CVaR 95%",   _v("cvar_95")),
        (alpha_col,    _v("alpha")),
        (beta_col,     _v("beta", pct=False)),
    ]

    # Show in 4-col rows
    visible = [(lbl, txt, raw) for lbl, (txt, raw) in pairs if txt != "—"]
    if not visible:
        return

    st.markdown("#### 🔬 Risk Metrics")
    for i in range(0, len(visible), 4):
        chunk = visible[i:i+4]
        cols = st.columns(len(chunk))
        for col, (lbl, txt, raw) in zip(cols, chunk):
            # Determine color
            if raw is None:
                color = "#f0f6fc"
            elif lbl in ("Max DD", "VaR 95%", "CVaR 95%", "Volatility"):
                color = "#f85149" if raw < 0 else "#f0f6fc"
            elif lbl in (alpha_col, "Sharpe", "Sortino"):
                color = "#3fb950" if raw > 0 else "#f85149"
            else:
                color = "#f0f6fc"
            col.markdown(
                f"<div class='metric-card' style='padding:12px;'>"
                f"<div class='metric-label'>{lbl}</div>"
                f"<div style='font-size:1.1rem; font-weight:700; margin:4px 0 0; color:{color};'>{txt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)


# ── Benchmark comparison ───────────────────────────────────────────────

def _render_benchmark_table(df: pd.DataFrame):
    display_cols = [c for c in ["benchmark", "cagr", "ytd", "1y", "3y", "5y",
                                 "volatility", "sharpe", "max_drawdown"] if c in df.columns]
    if not display_cols:
        return
    show = df[display_cols].copy()
    rename = {
        "benchmark": "Benchmark", "cagr": "CAGR", "ytd": "YTD",
        "1y": "1Y", "3y": "3Y", "5y": "5Y",
        "volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max DD",
    }
    show.columns = [rename.get(c, c) for c in display_cols]
    pct_cols = [c for c in ["CAGR", "YTD", "1Y", "3Y", "5Y", "Volatility", "Max DD"] if c in show.columns]
    fmt = {c: "{:+.2f}%" for c in pct_cols}
    if "Sharpe" in show.columns:
        fmt["Sharpe"] = "{:.2f}"
    ret_cols = [c for c in ["CAGR", "YTD", "1Y", "3Y", "5Y"] if c in show.columns]
    styled = show.style.format(fmt, na_rep="—")
    if ret_cols:
        styled = styled.applymap(
            lambda v: "color:#3fb950" if isinstance(v, (int, float)) and v > 0
            else "color:#f85149" if isinstance(v, (int, float)) and v < 0 else "",
            subset=ret_cols,
        )
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(len(show) * 38 + 40, 500))


# ── Helpers ────────────────────────────────────────────────────────────

def _add_metrics_to_row(row: dict, data: dict, alpha_col: str, beta_col: str):
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
    pct_cols   = ["1D","1W","1M","3M","6M","YTD","1Y","3Y","5Y","CAGR","Volatility","Max DD", alpha_col]
    ratio_cols = ["Sharpe","Sortino", beta_col]
    return_cols= ["1D","1W","1M","3M","6M","YTD","1Y","3Y","5Y","CAGR", alpha_col]

    fmt = {c: "{:+.2f}%" for c in pct_cols if c in df.columns}
    for c in ratio_cols:
        if c in df.columns:
            fmt[c] = "{:.2f}"

    styled = df.style.format(fmt, na_rep="—")
    existing_ret = [c for c in return_cols if c in df.columns]
    if existing_ret:
        styled = styled.applymap(
            lambda v: "color:#3fb950" if isinstance(v, (int, float)) and v > 0
            else "color:#f85149" if isinstance(v, (int, float)) and v < 0 else "",
            subset=existing_ret,
        )

    def _hl_portfolio(row):
        if row.get("Name", "").startswith("** TOTAL PORTFOLIO"):
            return ["background-color:rgba(88,166,255,0.08); font-weight:700;"] * len(row)
        return [""] * len(row)

    styled = styled.apply(_hl_portfolio, axis=1)
    st.dataframe(styled, use_container_width=True,
                 height=min(len(df) * 38 + 60, 800))


def _render_legend(alpha_col: str, beta_col: str):
    ratings = cfg.metric_ratings() or {}
    legend_rows = [
        ("CAGR",       "cagr",         "Equity risk premium"),
        (alpha_col,    "alpha",         "Jensen's Alpha (CAPM)"),
        (beta_col,     "beta",          "CAPM, 1.0 = market"),
        ("Max DD",     "max_drawdown",  "Retail drawdown tolerance"),
        ("Volatility", "volatility",    "Historical equity vol ~15%"),
        ("Sharpe",     "sharpe",        "Sharpe (1994)"),
        ("Sortino",    "sortino",       "Sortino & Price (1994)"),
        ("VaR 95%",    "var_pct",       "Basel III retail adj."),
        ("CVaR 95%",   "cvar_pct",      "Artzner et al. (1999)"),
    ]
    legend_data = []
    for metric_label, key, source in legend_rows:
        spec = ratings.get(key, {})
        thresholds = spec.get("thresholds", [None, None])
        invert = spec.get("invert", False)
        good_t, warn_t = thresholds[0], thresholds[1]
        if invert:
            strong = f"< {abs(good_t):.1f}%" if good_t is not None else "—"
            fair   = (f"{abs(warn_t):.1f}% – {abs(good_t):.1f}%"
                      if good_t is not None and warn_t is not None else "—")
            weak   = f"> {abs(warn_t):.1f}%" if warn_t is not None else "—"
        else:
            strong = f"> {good_t}" if good_t is not None else "—"
            fair   = (f"{warn_t} – {good_t}"
                      if good_t is not None and warn_t is not None else "—")
            weak   = f"< {warn_t}" if warn_t is not None else "—"
        legend_data.append({
            "Metric": metric_label, "🟢 Strong": strong,
            "🟡 Fair": fair, "🔴 Weak": weak, "Source": source,
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(legend_data), use_container_width=True, hide_index=True)
