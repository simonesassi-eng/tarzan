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


def _apply_per_holding_targets(holdings, targets: dict) -> None:
    """Attach per-holding rebalancing targets in place.

    ``targets`` is keyed by ISIN (when present) or uppercased ticker, as
    produced by ``load_targets_per_holding``. Each holding is matched by its
    ISIN first, then by its (uppercased) ticker — so ticker-only target rows
    still attach. A holding with no matching target is left untouched.
    """
    if not targets:
        return
    matched = 0
    for h in holdings:
        t = targets.get(h.isin)
        if t is None and h.ticker:
            t = targets.get(h.ticker.strip().upper())
        if not t:
            continue
        if t.get("target_equities") is not None:
            h.target_equities = t["target_equities"]
        if t.get("target_fixed_income") is not None:
            h.target_fixed_income = t["target_fixed_income"]
        if t.get("target_portfolio") is not None:
            h.target_portfolio = t["target_portfolio"]
        h.no_buy_no_sell = bool(t.get("no_buy_no_sell", False))
        matched += 1
    logger.info("Applied per-holding targets to %d/%d holdings", matched, len(holdings))


def _seed_missing_targets(holdings, targets: dict) -> list:
    """Create zero-value holdings for target instruments not currently held.

    Used by the per-holding-only rebalancing mode so the optimizer can open
    a new position toward an instrument's ``target_portfolio`` weight. Matches
    existing holdings by ISIN or (uppercased) ticker; every remaining target
    row with a positive ``target_portfolio`` becomes a seeded Holding
    (quantity 0) to be enriched alongside the rest.
    """
    from tarzan.models.holding import Holding

    held_isins = {h.isin for h in holdings if h.isin}
    held_tickers = {(h.ticker or "").strip().upper() for h in holdings if h.ticker}
    seeded = []
    for row in targets.values():
        tpf = row.get("target_portfolio")
        if tpf is None or tpf <= 0:
            continue
        r_isin = (row.get("isin") or "").strip()
        r_ticker = (row.get("ticker") or "").strip()
        if not r_isin and not r_ticker:
            continue
        if (r_isin and r_isin in held_isins) or (r_ticker and r_ticker.upper() in held_tickers):
            continue  # already held
        seeded.append(Holding(
            isin=r_isin, ticker=r_ticker, quantity=0.0,
            cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR",
            name=row.get("name") or r_ticker or r_isin,
            target_portfolio=float(tpf), is_seeded_target=True,
        ))
    return seeded


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
    # instrument_taxonomy.csv / config and the geo resolver's copy, so an edit to a
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

    # Per-holding-only mode: seed zero-value positions for target instruments
    # not currently held, so the optimizer can buy them toward target. These
    # seeds are kept SEPARATE from the real snapshot — they must never appear
    # in Holdings / Returns / allocations, only feed the rebalancer.
    seeds = []
    if config.target_use_per_holding_only and targets_by_isin:
        seeds = _seed_missing_targets(holdings, targets_by_isin)
        if seeds:
            logger.info(
                "Seeding %d not-held target instrument(s) for buy-new: %s",
                len(seeds), ", ".join(h.ticker or h.isin for h in seeds),
            )

    logger.info("Snapshot derived from orders: %d holdings", len(holdings))

    # Enrich real holdings and seeds together, then split them back apart.
    from tarzan.data.enricher import enrich_holdings, set_portfolio_backtest_period
    set_portfolio_backtest_period(config.portfolio_backtest_period)
    logger.info("Enriching holdings (period=%s)...", config.portfolio_backtest_period)
    combined = enrich_holdings(holdings + seeds)
    holdings = [h for h in combined if not getattr(h, "is_seeded_target", False)]
    seeds = [h for h in combined if getattr(h, "is_seeded_target", False)]
    enriched = sum(1 for h in holdings if h.is_enriched())
    logger.info("Enriched %d/%d holdings (+%d rebalance seeds)",
                enriched, len(holdings), len(seeds))

    # Compute
    logger.info("Computing metrics...")
    engine = MetricsEngine(holdings, config, orders=orders, rebalance_seeds=seeds)
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
