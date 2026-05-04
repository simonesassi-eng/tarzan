"""Portfolio-level metrics container.

PortfolioMetrics is the single output object produced by the Calculator layer.
It aggregates all computed data needed by the Reporting layer (Excel generator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio metrics produced by the Calculator.

    This is the canonical data transfer object between the Calculation
    and Reporting layers. All fields are populated by ``calculate_metrics()``.

    Attributes:
        total_value: Sum of all holding market values in EUR.
        holdings_df: Enriched holdings table with weights, gains, classifications.
        allocation_by_class: Allocation breakdown by asset class.
        allocation_by_geo: Geographic allocation (equity portion only).
        allocation_by_sector: Sector allocation breakdown.
        top_10: Top 10 holdings by portfolio weight.
        performance: Dict of period returns (1d, 1w, ..., cagr).
        risk: Dict of risk metrics (volatility, sharpe, sortino, var, etc.).
        weighted_yield: Portfolio-weighted average dividend/coupon yield.
        avg_ter: Portfolio-weighted average Total Expense Ratio.
        goal_deltas: Actual vs target allocation comparison.
        rebalancing_suggestions: Buy/sell suggestions exceeding threshold.
        benchmark_comparison: Portfolio vs benchmark metrics table.
        what_if: Hypothetical value if invested in each benchmark.
        portfolio_history: Daily portfolio value time series.
        benchmark_histories: Dict of benchmark name → daily price series.
        acwi_geo: MSCI ACWI geographic breakdown for benchmark reference.
        holding_performance: Per-holding period returns table.
        holding_histories: Dict of ticker → {name, history: pd.Series}.
    """

    total_value: float = 0.0
    holdings_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_geo: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_sector: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_10: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance: dict = field(default_factory=dict)
    performance_full: dict = field(default_factory=dict)  # Full 5y window (not inception-filtered)
    risk: dict = field(default_factory=dict)
    weighted_yield: float = 0.0
    avg_ter: float = 0.0
    goal_deltas: Optional[pd.DataFrame] = None
    rebalancing_suggestions: Optional[list] = None
    rebalancing_verifications: Optional[list] = None
    benchmark_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    what_if: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_history: Optional[pd.Series] = None
    benchmark_histories: dict = field(default_factory=dict)
    acwi_geo: dict = field(default_factory=dict)
    holding_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    holding_histories: dict = field(default_factory=dict)

    def to_summary_dict(self) -> dict:
        """Serialize key metrics to a JSON-compatible dictionary.

        Useful for API responses, logging, or downstream pipeline consumption.
        """
        return {
            "total_value_eur": round(self.total_value, 2),
            "performance": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.performance.items()
            },
            "risk": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.risk.items()
            },
            "weighted_yield": round(self.weighted_yield, 6),
            "avg_ter": round(self.avg_ter, 6),
            "num_holdings": len(self.holdings_df),
            "num_rebalancing_actions": (
                len(self.rebalancing_suggestions) if self.rebalancing_suggestions else 0
            ),
        }
