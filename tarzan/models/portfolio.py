"""Versioned portfolio metrics shared by analytics and presentation.

:class:`PortfolioMetrics` is built by the metrics engine and consumed by the
summary, artifact, newsletter, and local-export boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


def _round_or_none(value, ndigits: int):
    """Round a float for output, collapsing non-finite values to None first
    so ``round(nan, 6)`` (which is still NaN) never reaches the JSON layer."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, ndigits)
    return value


# Bump when the PortfolioMetrics field contract changes in a way consumers
# (newsletter, report, to_summary_dict) must be aware of — a renamed or
# removed field, or a changed field meaning. Consumers can assert this to fail
# loudly on a mismatch instead of silently reading a stale/renamed field.
PORTFOLIO_METRICS_SCHEMA_VERSION = 2

# The stable EXTERNAL output contract: the keys ``to_summary_dict`` always
# emits, regardless of run mode. This is the narrow, versioned surface any
# external consumer (API, mobile app, downstream pipeline) should depend on —
# NOT the ~50-field internal cube. Additional keys (xirr_pct, twror_pct,
# twror_annualized_pct, returns_coverage_pct) appear only on the order path and
# are documented as optional. A test pins this set so a field rename/removal is
# caught before it breaks a consumer.
SUMMARY_CONTRACT_KEYS = frozenset({
    "schema_version",
    "valuation_availability", "trustworthy_total_value_eur",
    "known_valuation_subtotal_eur", "history_availability",
    "history_unavailable_instruments",
    "total_value_eur", "invested_value_eur", "cash_value_eur", "cash_target_eur",
    "performance", "risk", "weighted_yield", "avg_ter",
    "num_holdings", "num_rebalancing_actions",
})
SUMMARY_CONTRACT_OPTIONAL_KEYS = frozenset({
    "xirr_pct", "twror_pct", "twror_annualized_pct", "returns_coverage_pct",
})


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio metrics produced by :class:`MetricsEngine`.

    This is the canonical transfer object between portfolio analytics and
    output projection. Fields are populated incrementally by the metrics engine.

    Deliberately NOT frozen: it is a wide report cube built incrementally
    (by MetricsEngine and, in tests, field-by-field), so immutability would
    cost ergonomics for no real gain — production never mutates it after
    construction. Contract stability is instead tracked via ``schema_version``.

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
        target_history: NAV (base 100) of the target allocation, or None.
    """

    # Field-contract version (see PORTFOLIO_METRICS_SCHEMA_VERSION). First
    # field so it is obvious in reprs / serialization.
    schema_version: int = PORTFOLIO_METRICS_SCHEMA_VERSION
    total_value: float = 0.0
    # Trust boundary for the displayed legacy subtotal. ``total_value`` remains
    # the compatibility field, while the fields below distinguish a complete
    # trustworthy total from a labeled partial known subtotal.
    valuation_availability: str = "AVAILABLE"
    trustworthy_total_value_eur: Optional[float] = None
    known_valuation_subtotal_eur: Optional[float] = None
    valuation_evidence: tuple = ()
    invested_value: float = 0.0
    cash_value: float = 0.0
    cash_target_eur: float = 0.0
    holdings_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_geo: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation_by_sector: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_10: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance: Optional[dict] = field(default_factory=dict)
    performance_full: Optional[dict] = field(default_factory=dict)  # Full 5y window (not inception-filtered)
    risk: Optional[dict] = field(default_factory=dict)
    weighted_yield: float = 0.0
    avg_ter: float = 0.0
    goal_deltas: Optional[pd.DataFrame] = None
    rebalancing_suggestions: Optional[list] = None
    rebalancing_verifications: Optional[list] = None
    # Both rebalancing variants (buy-only and buy & sell), always computed so
    # Excel and the newsletter can show them side by side. Each entry:
    # {"label", "no_sell", "suggestions", "verifications"}.
    rebalancing_plans: Optional[list] = None
    benchmark_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    history_availability: str = "AVAILABLE"
    history_unavailable_instruments: tuple[str, ...] = ()
    portfolio_history: Optional[pd.Series] = None
    benchmark_histories: dict = field(default_factory=dict)
    # Run-scoped benchmark identity contract.  Each curated benchmark name
    # maps to the one full provider ticker resolved during preprocessing; all
    # histories, metrics, tables and charts must project this same mapping.
    benchmark_tickers: dict[str, str] = field(default_factory=dict)
    benchmark_resolution_errors: tuple[str, ...] = ()
    acwi_geo: dict = field(default_factory=dict)
    holding_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Run-scoped intraday preprocessing contract. Canonical keys remain the
    # analytical/display identities; each quote records the selected feed,
    # exact series and same-feed previous-close baseline. Presentation only
    # consumes this catalog and never performs provider I/O.
    intraday_requested_tickers: tuple[str, ...] = ()
    intraday_quotes: dict[str, dict] = field(default_factory=dict)
    holding_histories: dict = field(default_factory=dict)
    # NAV (base 100) of the TARGET allocation — the per-instrument
    # ``target_portfolio`` weights held CONSTANT, over the window where every
    # sleeve has a price. None when any sleeve lacks history, since a
    # renormalized subset would be a different portfolio under the same name.
    target_history: Optional[pd.Series] = None
    # Canonical ticker decision audit, one record per ISIN (or exact ticker
    # when no ISIN exists). Rendered as the compact data-sources section at the
    # bottom of the newsletter and excluded from the narrow summary contract.
    ticker_resolutions: tuple[dict[str, object], ...] = ()
    # Historical risk profile (newsletter "Historical risk profile" section).
    # Uncapped, per-instrument full-history risk stats plus a current-weight
    # static backtest for the portfolio row. Populated by
    # MetricsEngine._historical_risk. Shape:
    #   {"available": bool,
    #    "portfolio": {"label", "ticker", "span_label", "note", "metrics": {..}},
    #    "instruments": [{"label", "ticker", "span_label", "metrics": {..}}, ..]}
    historical_risk: Optional[dict] = None
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
    # deposits; pnl_pct is that over the *net* capital contributed
    # (invested_capital_eur = current_value − pnl_eur = gross deposits −
    # everything withdrawn). actual_value_series is the dense daily real
    # euro worth of the patrimony (deposit/withdrawal jumps kept in) that
    # the newsletter mountain chart plots.
    pnl_eur: Optional[float] = None
    pnl_pct: Optional[float] = None
    invested_capital_eur: Optional[float] = None
    # Net-of-tax ESTIMATE (Italian CGT on realized gains; order path only).
    # estimated_cgt_eur is the estimated tax on realized capital gains;
    # pnl_eur_net_tax / pnl_pct_net_tax and xirr_net_tax_pct are the
    # money-weighted figures after subtracting it. The gross fields above
    # are never altered — these sit alongside them, clearly labeled as an
    # estimate (average cost, 26%/12.5% gov, losses offset where allowed).
    estimated_cgt_eur: Optional[float] = None
    pnl_eur_net_tax: Optional[float] = None
    pnl_pct_net_tax: Optional[float] = None
    xirr_net_tax_pct: Optional[float] = None
    actual_value_series: Optional[pd.Series] = None
    # Daily cumulative P&L (value − contributed capital); its delta over a
    # window is the real money gained in that window, net of contributions.
    pnl_series: Optional[pd.Series] = None
    # Daily unrealized P&L = market value of open positions − their cost
    # basis. Same definition as the hero's snapshot (total_value − cost),
    # but as a full daily series for charting. None on the holdings-only path.
    unrealized_series: Optional[pd.Series] = None
    # Net external capital flow per date: {date: eur} where deposits/buys are
    # positive and withdrawals/sells negative (distributions are negative —
    # cash leaving the securities portfolio). Drives the deposit/withdrawal
    # markers on the newsletter performance charts. None on the holdings-only
    # path. Same object the TWROR engine consumes (no recomputation).
    external_flows: Optional[dict] = None
    # Portfolio inception date (ISO "YYYY-MM-DD"), derived automatically
    # from the first order when an order list is present. None on the
    # holdings-only path (the header then falls back to config).
    inception_date: Optional[str] = None
    # Names of metric computers that raised during compute_all (empty when
    # every stage succeeded). A non-empty list means the corresponding
    # sections (risk, allocations, returns, ...) fell back to defaults rather
    # than real values, so a renderer can flag them as "unavailable" instead
    # of presenting silent zeros/blanks as authoritative figures.
    degraded_computers: list = field(default_factory=list)
    # Weekly allocation history over the last ~3 months (order path only).
    # Drives the per-category sparklines in the newsletter Diversification
    # section. Shape: {"dates": [date, ...],
    #                  "asset": [{class: pct_of_invested}, ...],
    #                  "geo":   [{region: pct_of_equity}, ...]}.
    # None on the holdings-only path (sparklines are then omitted).
    allocation_timeline: Optional[dict] = None

    @property
    def unrealized_pnl_eur(self) -> float:
        """Snapshot unrealized P&L in EUR: current market value − Σ cost basis
        of open holdings (the Hero's "Unrealized PnL"). Realized gains and
        income are excluded; a zero/empty cost basis yields ``total_value``.

        Single source for the ``total_value − cost`` figure the subject line,
        performance matrix, diversification hero and AI digest all show."""
        cost = float(self.holdings_df["cost_basis_eur"].sum()) if not self.holdings_df.empty else 0.0
        return self.total_value - cost

    @property
    def unrealized_pnl_pct(self) -> Optional[float]:
        """:attr:`unrealized_pnl_eur` over cost basis, as a percent. ``None``
        when there is no cost basis to divide by — a numeric ``0.0`` would
        falsely read as "break-even" instead of "not applicable" (the
        numeric-zero != Unavailable invariant)."""
        cost = float(self.holdings_df["cost_basis_eur"].sum()) if not self.holdings_df.empty else 0.0
        return (self.unrealized_pnl_eur / cost * 100.0) if cost > 0 else None

    def to_summary_dict(self) -> dict:
        """Serialize key metrics to a JSON-compatible dictionary — the stable,
        versioned EXTERNAL output contract.

        This is the narrow surface external consumers (API/app/pipeline) should
        depend on, distinct from the wide internal cube. It always emits
        :data:`SUMMARY_CONTRACT_KEYS`; the order-path figures in
        :data:`SUMMARY_CONTRACT_OPTIONAL_KEYS` appear only when computed.
        Every float is finite-or-null (valid strict JSON). ``schema_version``
        lets consumers detect a contract change.
        """
        # All floats are routed through _round_or_none so a non-finite metric
        # (e.g. Sharpe when volatility==0, or any risk metric on an empty
        # history) becomes JSON ``null`` instead of a bare NaN/Infinity token
        # that breaks strict parsers and downstream databases.
        valuation_available = self.valuation_availability != "UNAVAILABLE"
        trustworthy_total = self.trustworthy_total_value_eur
        if trustworthy_total is None and self.valuation_availability == "AVAILABLE":
            trustworthy_total = self.total_value
        known_subtotal = self.known_valuation_subtotal_eur
        if known_subtotal is None and self.valuation_availability == "AVAILABLE":
            known_subtotal = self.total_value
        history_available = self.history_availability != "UNAVAILABLE"
        performance = (
            {k: _round_or_none(v, 6) for k, v in (self.performance or {}).items()}
            if history_available and self.performance is not None
            else None
        )
        risk = (
            {k: _round_or_none(v, 6) for k, v in (self.risk or {}).items()}
            if history_available and self.risk is not None
            else None
        )
        summary = {
            "schema_version": self.schema_version,
            "valuation_availability": self.valuation_availability,
            "trustworthy_total_value_eur": _round_or_none(trustworthy_total, 2),
            "known_valuation_subtotal_eur": _round_or_none(known_subtotal, 2),
            "history_availability": self.history_availability,
            "history_unavailable_instruments": list(
                self.history_unavailable_instruments
            ),
            # Compatibility name now projects only the trustworthy total. A
            # partial known subtotal is exposed solely under its labeled key.
            "total_value_eur": _round_or_none(trustworthy_total, 2),
            "invested_value_eur": (
                _round_or_none(self.invested_value, 2)
                if valuation_available else None
            ),
            "cash_value_eur": (
                _round_or_none(self.cash_value, 2)
                if valuation_available else None
            ),
            "cash_target_eur": _round_or_none(self.cash_target_eur, 2),
            "performance": performance,
            "risk": risk,
            "weighted_yield": (
                _round_or_none(self.weighted_yield, 6)
                if valuation_available else None
            ),
            "avg_ter": (
                _round_or_none(self.avg_ter, 6)
                if valuation_available else None
            ),
            "num_holdings": len(self.holdings_df),
            "num_rebalancing_actions": (
                len(self.rebalancing_suggestions) if self.rebalancing_suggestions else 0
            ),
        }
        # Order-list returns: include only when computed (an order list
        # was supplied), so a holdings-only summary is unchanged.
        if self.xirr_pct is not None:
            summary["xirr_pct"] = _round_or_none(self.xirr_pct, 6)
        if self.twror_pct is not None:
            summary["twror_pct"] = _round_or_none(self.twror_pct, 6)
            summary["twror_annualized_pct"] = _round_or_none(
                self.twror_annualized_pct, 6
            )
        if self.returns_coverage_pct is not None:
            summary["returns_coverage_pct"] = _round_or_none(self.returns_coverage_pct, 6)
        return summary
