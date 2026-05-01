"""Optimizer — mirrors Excel Optimizer sheet.

Only rebalancing actions + post-rebalancing verification (allocation is in Dashboard).
"""

from __future__ import annotations

import streamlit as st

from portfolio_analyzer.models.portfolio import PortfolioMetrics
from portfolio_analyzer.models.investor_config import InvestorConfig
from portfolio_analyzer.presentation.formatters import fmt_eur


def render(metrics: PortfolioMetrics, config: InvestorConfig | None = None):
    st.markdown("## ⚖️ Portfolio Optimizer")

    if config:
        header_parts = []
        if config.rebalancing_lump_sum_amount > 0:
            header_parts.append(f"Lump sum: **{fmt_eur(config.rebalancing_lump_sum_amount)}**")
        if config.rebalancing_no_sell:
            header_parts.append("**NO SELL**")
        if config.rebalancing_min_transaction_eur > 0:
            header_parts.append(f"Min transaction: **{fmt_eur(config.rebalancing_min_transaction_eur)}**")
        header_parts.append(f"Max tolerance: **{config.rebalancing_max_tolerance:.1f}%**")
        st.caption(" · ".join(header_parts))

    actions = metrics.rebalancing_suggestions or []
    verifications = metrics.rebalancing_verifications or []

    if not actions:
        st.success("✅ Portfolio is aligned with targets. No rebalancing actions needed.")
    else:
        # Summary KPIs
        total_buy = sum(a["amount_eur"] for a in actions if a["direction"] == "buy")
        total_sell = sum(a["amount_eur"] for a in actions if a["direction"] == "sell")
        net = total_buy - total_sell

        k1, k2, k3 = st.columns(3)
        k1.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Total BUY</div>"
            f"<div class='metric-value' style='color:#3fb950'>{fmt_eur(total_buy)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        k2.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Total SELL</div>"
            f"<div class='metric-value' style='color:#f85149'>{fmt_eur(total_sell)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        net_color = "#3fb950" if net >= 0 else "#f85149"
        k3.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Net</div>"
            f"<div class='metric-value' style='color:{net_color}'>{fmt_eur(net)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Actions list
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Suggested Actions")

        for a in actions:
            if a["direction"] == "buy":
                badge_bg = "rgba(63,185,80,0.15)"
                badge_fg = "#3fb950"
                badge_text = "BUY"
            else:
                badge_bg = "rgba(248,81,73,0.15)"
                badge_fg = "#f85149"
                badge_text = "SELL"

            st.markdown(
                f"<div style='display: flex; align-items: center; padding: 12px 16px; "
                f"background: #161b22; border: 1px solid #21262d; border-radius: 10px; margin-bottom: 8px; gap: 16px;'>"
                f"<span style='background: {badge_bg}; color: {badge_fg}; padding: 6px 14px; "
                f"border-radius: 6px; font-weight: 700; font-size: 0.8rem; min-width: 60px; text-align: center;'>"
                f"{badge_text}</span>"
                f"<div style='flex: 1;'>"
                f"<div style='color: #f0f6fc; font-weight: 600; font-size: 0.95rem;'>{a['name']}</div>"
                f"<div style='color: #8b949e; font-size: 0.75rem; margin-top: 2px;'>{a.get('ticker', '')}</div>"
                f"</div>"
                f"<div style='text-align: right; min-width: 120px;'>"
                f"<div style='color: #f0f6fc; font-weight: 700; font-size: 1.1rem;'>{fmt_eur(a['amount_eur'])}</div>"
                f"<div style='color: #8b949e; font-size: 0.7rem; max-width: 280px; margin-top: 2px;'>{a.get('reason', '')}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Verification
    if verifications:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Post-Rebalancing Verification", expanded=False):
            tol_used = verifications[0].get("tolerance")
            if tol_used is not None:
                st.caption(f"Solver tolerance: ±{tol_used:.1f}%")

            for v in verifications:
                is_ok = "OK" in v["status"]
                icon = "✅" if is_ok else "⚠️"
                st.markdown(f"**{icon} {v['check']}**")
                if v.get("detail"):
                    st.caption(v["detail"])
