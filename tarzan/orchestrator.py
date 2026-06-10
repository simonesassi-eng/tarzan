"""Pipeline orchestrator: load → enrich → compute → PortfolioMetrics.

Single entry point used by the CLI (main.py).
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Union

from tarzan.data.loader import (
    load_holdings,
    load_config,
    load_orders,
    load_targets_per_holding,
)
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.engine.metrics import MetricsEngine

logger = logging.getLogger(__name__)


from tarzan.models.investor_config import InvestorConfig


def _apply_per_holding_targets(holdings, targets_by_isin: dict) -> None:
    """Attach per-holding rebalancing targets (by ISIN) in place.

    Used in order-only mode: the order list has no target columns, so the
    rebalancer's per-instrument targets are sourced from the side file
    loaded by ``load_targets_per_holding``. A holding with no matching
    ISIN is left untouched (no target).
    """
    if not targets_by_isin:
        return
    matched = 0
    for h in holdings:
        t = targets_by_isin.get(h.isin)
        if not t:
            continue
        if t.get("target_equities") is not None:
            h.target_equities = t["target_equities"]
        if t.get("target_fixed_income") is not None:
            h.target_fixed_income = t["target_fixed_income"]
        h.no_buy_no_sell = bool(t.get("no_buy_no_sell", False))
        matched += 1
    logger.info("Applied per-holding targets to %d/%d holdings", matched, len(holdings))


def run(
    holdings_source: Optional[Union[str, io.BytesIO]],
    config_source: Optional[Union[str, io.BytesIO]] = None,
    orders_source: Optional[Union[str, io.BytesIO]] = None,
    targets_per_holding_source: Optional[Union[str, io.BytesIO]] = None,
    holdings_filename: str = "",
    config_filename: str = "",
    orders_filename: str = "",
    targets_per_holding_filename: str = "",
) -> tuple[PortfolioMetrics, InvestorConfig]:
    """Execute the full analysis pipeline.

    Two snapshot sources are supported:

    * **Holdings + orders (hybrid, default):** the snapshot — valuation,
      allocations, targets, rebalancing — comes from ``holdings.csv``,
      while the order list owns the historical value series and XIRR/TWROR.

    * **Order-only:** when no holdings snapshot is available (the source is
      missing/empty) but an order list is, the snapshot is *derived* from
      the orders (net quantity, average-cost basis, market value via
      enrichment) and per-instrument rebalancing targets are joined by
      ISIN from ``targets_per_holding_source``. This lets the whole report
      run from the order list alone.

    Returns:
        Tuple of (PortfolioMetrics, InvestorConfig).
    """
    config = load_config(config_source)
    logger.info("Config loaded (target tolerance=±%.1f%%)", config.rebalancing_target_tolerance_pctg)

    # Optional order list — never fatal: log and continue.
    orders = None
    if orders_source is not None:
        try:
            orders = load_orders(orders_source, orders_filename) or None
            if orders:
                logger.info("Loaded %d orders (returns enabled)", len(orders))
        except Exception as e:
            logger.warning("Order list unreadable (%s); continuing.", e)
            orders = None

    # 1. Load the snapshot holdings, or derive them from the order list.
    holdings = _load_holdings_or_empty(holdings_source, holdings_filename)

    if not holdings and orders:
        logger.info("No holdings snapshot — deriving it from the order list (order-only mode).")
        from tarzan.engine.returns_builder import build_holdings_from_orders
        holdings = build_holdings_from_orders(orders)
        targets_by_isin = _load_targets_or_empty(
            targets_per_holding_source, targets_per_holding_filename
        )
        _apply_per_holding_targets(holdings, targets_by_isin)

    if not holdings:
        logger.error("No holdings available (no snapshot and no order list).")
        return PortfolioMetrics(), config

    logger.info("Snapshot has %d holdings", len(holdings))

    # Option Y scope: the order list owns the *historical value series*
    # and the history-dependent metrics. In the hybrid path the current
    # snapshot still comes from holdings.csv; in order-only mode the
    # snapshot was derived from the orders just above.

    # 2. Enrich
    from tarzan.data.enricher import enrich_holdings, set_portfolio_backtest_period
    set_portfolio_backtest_period(config.portfolio_backtest_period)
    logger.info("Enriching holdings (period=%s)...", config.portfolio_backtest_period)
    holdings = enrich_holdings(holdings)
    enriched = sum(1 for h in holdings if h.is_enriched())
    logger.info("Enriched %d/%d holdings", enriched, len(holdings))

    # 3. Compute
    logger.info("Computing metrics...")
    engine = MetricsEngine(holdings, config, orders=orders)
    metrics = engine.compute_all()
    logger.info("Total portfolio value: €%.2f", metrics.total_value)

    return metrics, config


def _load_holdings_or_empty(
    source: Optional[Union[str, io.BytesIO]], filename: str
) -> list:
    """Load the holdings snapshot, tolerating a missing/empty source.

    Returns ``[]`` (not an exception) when the source is absent or cannot
    be read, so the caller can fall back to deriving the snapshot from the
    order list.
    """
    if source is None:
        return []
    try:
        return load_holdings(source, filename) or []
    except FileNotFoundError:
        logger.info("Holdings snapshot not found; will try the order list.")
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("Holdings snapshot unreadable (%s); will try the order list.", e)
        return []


def _load_targets_or_empty(
    source: Optional[Union[str, io.BytesIO]], filename: str
) -> dict:
    """Load per-holding targets, tolerating a missing/unreadable source."""
    if source is None:
        return {}
    try:
        return load_targets_per_holding(source, filename)
    except Exception as e:  # noqa: BLE001
        logger.warning("Per-holding targets unreadable (%s); none applied.", e)
        return {}
