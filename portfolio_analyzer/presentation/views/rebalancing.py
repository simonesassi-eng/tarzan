"""Rebalancing page — buy/sell actions, verification, lump sum toggle."""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.models.investor_config import InvestorConfig
from portfolio_analyzer.presentation.formatters import fmt_eur


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    st.markdown("## ⚖️ Rebalancing")

    actions = metrics.rebalancing_suggestions
    verifications = metrics.rebalancing_verifications

    if not actions:
        st.success("✅ Portfolio is within target thresholds. No rebalancing needed.")
        if verifications:
            _show_verifications(verifications)
        return

    # Summary
    total_buy = sum(a["amount_eur"] for a in actions if a["direction"] == "buy")
    total_sell = sum(a["amount_eur"] for a in actions if a["direction"] == "sell")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total BUY", fmt_eur(total_buy))
    col2.metric("Total SELL", fmt_eur(total_sell))
    col3.metric("Net", fmt_eur(total_buy - total_sell))

    if config:
        st.caption(f"Threshold: {config.rebalancing_threshold}% · "
                   f"Min transaction: {fmt_eur(config.rebalancing_min_transaction_eur)} · "
                   f"Precision: {config.rebalancing_precision}%")

    # Actions table
    st.markdown("---")
    st.markdown("##### Suggested Actions")
    for a in actions:
        if a["direction"] == "buy":
            badge_style = "background:rgba(63,185,80,0.15);color:#3fb950"
            badge_text = "BUY"
        else:
            badge_style = "background:rgba(248,81,73,0.15);color:#f85149"
            badge_text = "SELL"

        st.markdown(
            f"<div style='display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #21262d;gap:12px'>"
            f"<span style='{badge_style};padding:3px 10px;border-radius:5px;font-weight:700;font-size:0.8rem;min-width:45px;text-align:center'>{badge_text}</span>"
            f"<span style='flex:1'><b>{a['name']}</b> <span style='color:#8b949e'>({a['ticker']})</span></span>"
            f"<span style='font-weight:700;font-size:1.1rem'>{fmt_eur(a['amount_eur'])}</span>"
            f"<span style='color:#8b949e;font-size:0.8rem;max-width:250px'>{a.get('reason', '')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Verification
    if verifications:
        st.markdown("---")
        _show_verifications(verifications)


def _show_verifications(verifications: list[dict]):
    st.markdown("##### Post-Rebalancing Verification")
    for v in verifications:
        icon = "✅" if "OK" in v["status"] else "⚠️"
        tol = v.get("tolerance")
        tol_str = f" (precision: {tol}%)" if tol else ""
        st.markdown(f"{icon} **{v['check']}**: {v['status']}{tol_str}")
        if v.get("detail"):
            st.caption(v["detail"])
