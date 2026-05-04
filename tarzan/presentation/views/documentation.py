"""Documentation — methodological notes and metric definitions."""

from __future__ import annotations

import streamlit as st

from tarzan.models.portfolio import PortfolioMetrics
from tarzan import config as cfg


def render(metrics: PortfolioMetrics):
    st.markdown("## 📖 Documentation")

    st.markdown("### How this works")
    st.markdown(
        f"""
Tarzan analyzes your investment portfolio using live market data from yfinance
and a MILP-based rebalancing optimizer. All metrics are computed on the
available historical data per instrument, capped at **5 years**.

**Core assumption**: all backtest computations assume *current quantities* applied
to historical prices. This simulates "how would the current mix have performed",
not the actual historical portfolio (which would require transaction dates).
        """
    )

    st.markdown("### Key parameters")
    bench_beta_name = cfg.benchmark_beta_name()
    bench_geo_name = cfg.benchmark_geo_allocation()
    rfr = cfg.risk_free_rate() * 100
    st.markdown(
        f"""
- **Risk-free rate**: {rfr:.1f}% (configured in `constants.yaml`)
- **Alpha/Beta benchmark**: `{bench_beta_name}` (marked `is_benchmark_alfa_and_beta=true` in indexes.csv)
- **Geo allocation benchmark**: `{bench_geo_name}` (marked `is_benchmark_geo_allocation=true`)
- **Backtest period**: fixed 5 years
- **Trading days per year**: 252 (for volatility annualization)
        """
    )

    st.markdown("### Metric definitions")

    st.markdown(
        """
- **CAGR** — Compound Annual Growth Rate: (End / Start)^(1/years) - 1
- **Volatility** — Annualized standard deviation of daily returns × √252
- **Sharpe Ratio** — (Return - Risk-Free Rate) / Volatility. Higher = better risk-adjusted return.
- **Sortino Ratio** — Like Sharpe but only penalizes *downside* volatility.
  More appropriate for asymmetric returns.
- **Max Drawdown (MDD)** — Largest peak-to-trough decline. Shows worst-case historical loss.
- **VaR 95% (Value at Risk)** — 5th percentile of daily returns. Non-parametric historical simulation.
  "On 95% of days, losses don't exceed this."
- **CVaR 95% (Conditional VaR / Expected Shortfall)** — Average loss in the worst 5% of days.
  Coherent risk measure per Artzner et al. (1999).
- **Alpha** — Jensen's Alpha: return beyond what CAPM predicts given portfolio's Beta.
  Positive = outperforming risk-adjusted benchmark.
- **Beta** — CAPM sensitivity to benchmark. β=1.0 means moves 1:1 with benchmark.
  β<1 = defensive, β>1 = aggressive.
        """
    )

    st.markdown("### Rebalancing optimizer (MILP)")
    st.markdown(
        """
The rebalancer minimizes total transaction volume subject to constraints:

1. **Asset class targets** — actual allocation within tolerance of target
2. **Geography targets** (equity only) — multi-country ETF exposure split proportionally
3. **Per-holding targets** (optional via `target_equities`/`target_fixed_income`)
4. **Cash flow** — if lump sum > 0, total buy - total sell = lump sum
5. **Min transaction** — no action below `rebalancing_min_transaction_eur`
6. **No sell** (optional) — forbid all sell actions, only allow buys
7. **Frozen holdings** — `no_buy_no_sell=TRUE` excludes from rebalancing

The solver progressively relaxes tolerance from 0.1% up to `rebalancing_max_tolerance`.
If no feasible solution exists, returns 0 actions (rather than forcing a bad trade).
        """
    )

    st.markdown("### Data sources")
    st.markdown(
        """
- **yfinance** — Primary source for ETF and stock prices (market data + history)
- **OpenFIGI** — ISIN to ticker resolution for classification
- **Borsa Italiana** — Fallback for European bonds (BTP, EIB) not on yfinance
- **justETF** — ETF name resolution for geography matching
- **indexes.csv** — Reference data for benchmarks and geographic allocations
        """
    )

    st.markdown("### Limitations")
    st.markdown(
        """
- Historical prices depend on yfinance availability. Some instruments (certain bonds,
  delisted indices) may have limited or no data.
- Live prices may have a ~15 minute delay from the real market.
- Assumes current quantities are static throughout the backtest period.
- Currency conversion uses end-of-day EUR rates from yfinance.
        """
    )
