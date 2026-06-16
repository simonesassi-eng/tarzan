"""Pipeline orchestrator: load → enrich → compute → PortfolioMetrics.

Single entry point used by the CLI (main.py).
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Union

from tarzan.data.loader import (
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
    config_source: Optional[Union[str, io.BytesIO]] = None,
    orders_source: Optional[Union[str, io.BytesIO]] = None,
    targets_per_holding_source: Optional[Union[str, io.BytesIO]] = None,
    config_filename: str = "",
    orders_filename: str = "",
    targets_per_holding_filename: str = "",
) -> tuple[PortfolioMetrics, InvestorConfig]:
    """Execute the full analysis pipeline (order-only).

    The order list is the single source of truth: the snapshot — net
    quantity, average-cost basis, market value (via enrichment),
    allocations, targets, rebalancing — is *derived* from the orders, and
    the order list also owns the historical value series and XIRR/TWROR.
    Per-instrument rebalancing targets are joined by ISIN from
    ``targets_per_holding_source``.

    Returns:
        Tuple of (PortfolioMetrics, InvestorConfig).
    """
    # Re-read user inputs fresh on every run: drop the per-process caches of
    # indexes.csv / config and the geo resolver's copy, so an edit to a
    # user's Drive inputs is never shadowed by a previous run in the same
    # process. (Universal market data in price_cache is left cached.)
    from tarzan import config as _cfg
    from tarzan.data import geo_resolver as _geo
    _cfg.reset_input_caches()
    _geo.reset_caches()

    config = load_config(config_source)
    logger.info("Config loaded (target tolerance=±%.1f%%)", config.rebalancing_target_tolerance_pctg)

    # Load the order list — the single input that drives the whole report.
    orders = None
    if orders_source is not None:
        try:
            orders = load_orders(orders_source, orders_filename) or None
            if orders:
                logger.info("Loaded %d orders", len(orders))
        except Exception as e:
            logger.warning("Order list unreadable (%s).", e)
            orders = None

    if not orders:
        logger.error("No order list available — cannot run.")
        return PortfolioMetrics(), config

    # Derive the snapshot from the order list (net quantity, average-cost
    # basis, market value via enrichment) and attach per-instrument
    # rebalancing targets by ISIN.
    from tarzan.engine.returns_builder import build_holdings_from_orders
    holdings = build_holdings_from_orders(orders)
    targets_by_isin = _load_targets_or_empty(
        targets_per_holding_source, targets_per_holding_filename
    )
    _apply_per_holding_targets(holdings, targets_by_isin)

    if not holdings:
        logger.error("Order list produced no holdings.")
        return PortfolioMetrics(), config

    logger.info("Snapshot derived from orders: %d holdings", len(holdings))

    # Enrich
    from tarzan.data.enricher import enrich_holdings, set_portfolio_backtest_period
    set_portfolio_backtest_period(config.portfolio_backtest_period)
    logger.info("Enriching holdings (period=%s)...", config.portfolio_backtest_period)
    holdings = enrich_holdings(holdings)
    enriched = sum(1 for h in holdings if h.is_enriched())
    logger.info("Enriched %d/%d holdings", enriched, len(holdings))

    # Compute
    logger.info("Computing metrics...")
    engine = MetricsEngine(holdings, config, orders=orders)
    metrics = engine.compute_all()
    logger.info("Total portfolio value: €%.2f", metrics.total_value)

    return metrics, config


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
