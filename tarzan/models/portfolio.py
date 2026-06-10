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
        invested_value: Total value excluding cash holdings. Denominator
            for the invested allocation percentages.
        cash_value: Sum of cash-equivalent holdings in EUR.
        cash_target_eur: Target cash buffer in EUR (from config).
        holdings_df: Enriched holdings table with weights, gains, classifications.
        allocation_by_class: Allocation breakdown by asset class (invested only,
            percentages relative to invested_value).
        allocation_by_geo: Geographic allocation (equity portion only).
        allocation_by_sector: Sector allocation breakdown.
        top_10: Top 10 holdings by portfolio weight.
        performance: Dict of period returns (1d, 1w, ..., cagr).
        risk: Dict of risk metrics (volatility, sharpe, sortino, var, etc.).
        weighted_yield: Portfolio-weighted average dividend/coupon yield.
        avg_ter: Portfolio-weighted average Total Expense Ratio.
        goal_deltas: Actual vs target allocation comparison. Rows carry a
            'type' column with values 'asset_class',
            'geography (equity only)' or 'cash' (cash row uses EUR deltas).
        rebalancing_suggestions: Buy/sell suggestions emitted by the
            optimizer (one per executed trade).
        benchmark_comparison: Portfolio vs benchmark metrics table.
        portfolio_history: Daily portfolio value time series.
        benchmark_histories: Dict of benchmark name → daily price series.
        acwi_geo: MSCI ACWI geographic breakdown for benchmark reference.
        holding_performance: Per-holding period returns table.
        holding_histories: Dict of ticker → {name, history: pd.Series}.
    """

    total_value: float = 0.0
    invested_value: float = 0.0
    cash_value: float = 0.0
    cash_target_eur: float = 0.0
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
    # Drift-penalty sensitivity sweep: list of regimes describing how
    # the optimization changes as ``rebalancing_drift_penalty_weight``
    # varies. Used by the Optimizer tab to surface tuning hints.
    rebalancing_sensitivity: Optional[list] = None
    benchmark_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_history: Optional[pd.Series] = None
    benchmark_histories: dict = field(default_factory=dict)
    acwi_geo: dict = field(default_factory=dict)
    holding_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    holding_histories: dict = field(default_factory=dict)
    # Holdings excluded from the TOTAL PORTFOLIO time series because their
    # price history span is below the minimum (used for the Performance tab
    # warning banner). Each entry: {"ticker", "name", "value_eur",
    # "weight_pct", "span_days"}.
    excluded_short_tenure: list = field(default_factory=list)
    # Order-list returns (populated only when an order list is supplied;
    # all None for a holdings-only run, preserving today's behavior).
    # xirr_pct: annualized money-weighted return. twror_pct/
    # twror_annualized_pct: cumulative/annualized time-weighted return.
    # returns_coverage_pct: % of value priced by real market data over the
    # window. returns_provenance: {source_tag: [isin, ...]}. The period
    # debug list carries per-period TWROR diagnostics.
    xirr_pct: Optional[float] = None
    twror_pct: Optional[float] = None
    twror_annualized_pct: Optional[float] = None
    returns_coverage_pct: Optional[float] = None
    returns_provenance: Optional[dict] = None
    returns_period_debug: Optional[list] = None
    # Lifetime P&L since inception (order path only). pnl_eur is the all-in
    # euro gain (realized + unrealized) = current value + distributions −
    # deposits; pnl_pct is that over the total capital deployed
    # (invested_capital_eur). actual_value_series is the dense daily real
    # euro worth of the patrimony (deposit/withdrawal jumps kept in) that
    # the newsletter mountain chart plots.
    pnl_eur: Optional[float] = None
    pnl_pct: Optional[float] = None
    invested_capital_eur: Optional[float] = None
    actual_value_series: Optional[pd.Series] = None
    # Portfolio inception date (ISO "YYYY-MM-DD"), derived automatically
    # from the first order when an order list is present. None on the
    # holdings-only path (the header then falls back to config).
    inception_date: Optional[str] = None

    def to_summary_dict(self) -> dict:
        """Serialize key metrics to a JSON-compatible dictionary.

        Useful for API responses, logging, or downstream pipeline consumption.
        """
        summary = {
            "total_value_eur": round(self.total_value, 2),
            "invested_value_eur": round(self.invested_value, 2),
            "cash_value_eur": round(self.cash_value, 2),
            "cash_target_eur": round(self.cash_target_eur, 2),
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
        # Order-list returns: include only when computed (an order list
        # was supplied), so a holdings-only summary is unchanged.
        if self.xirr_pct is not None:
            summary["xirr_pct"] = round(self.xirr_pct, 6)
        if self.twror_pct is not None:
            summary["twror_pct"] = round(self.twror_pct, 6)
            summary["twror_annualized_pct"] = (
                round(self.twror_annualized_pct, 6)
                if self.twror_annualized_pct is not None else None
            )
        if self.returns_coverage_pct is not None:
            summary["returns_coverage_pct"] = round(self.returns_coverage_pct, 6)
        return summary
