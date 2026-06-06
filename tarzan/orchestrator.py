"""Pipeline orchestrator: load → enrich → compute → PortfolioMetrics.

Single entry point used by the CLI (main.py).
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Union

from tarzan.data.loader import load_holdings, load_config, load_orders
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.engine.metrics import MetricsEngine

logger = logging.getLogger(__name__)


from tarzan.models.investor_config import InvestorConfig


def run(
    holdings_source: Union[str, io.BytesIO],
    config_source: Optional[Union[str, io.BytesIO]] = None,
    orders_source: Optional[Union[str, io.BytesIO]] = None,
    holdings_filename: str = "",
    config_filename: str = "",
    orders_filename: str = "",
) -> tuple[PortfolioMetrics, InvestorConfig]:
    """Execute the full analysis pipeline.

    When ``orders_source`` is provided and loads successfully, the order
    list becomes the single source of the historical value series and
    XIRR/TWROR are computed (Option Y). When it is absent or unreadable,
    behavior is identical to a holdings-only run.

    Returns:
        Tuple of (PortfolioMetrics, InvestorConfig).
    """
    # 1. Load
    logger.info("Loading holdings...")
    holdings = load_holdings(holdings_source, holdings_filename)
    if not holdings:
        logger.error("No valid holdings loaded.")
        return PortfolioMetrics(), InvestorConfig()

    logger.info("Loaded %d holdings", len(holdings))

    config = load_config(config_source)
    logger.info("Config loaded (target tolerance=±%.1f%%)", config.rebalancing_target_tolerance_pctg)

    # Optional order list — never fatal: log and continue holdings-only.
    orders = None
    if orders_source is not None:
        try:
            orders = load_orders(orders_source, orders_filename) or None
            if orders:
                logger.info("Loaded %d orders (returns enabled)", len(orders))
        except Exception as e:
            logger.warning("Order list unreadable (%s); continuing holdings-only.", e)
            orders = None

    # Option Y scope: the order list owns the *historical value series*
    # and the history-dependent metrics (handled inside the engine via
    # _portfolio_history_from_orders). The current snapshot — valuation,
    # allocations, targets, rebalancing — still comes from holdings.csv,
    # which carries the targets and cost basis the order list does not.
    # So we do NOT replace `holdings` here.

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
