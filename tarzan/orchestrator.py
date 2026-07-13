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
from tarzan.contracts.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


from tarzan.models.investor_config import InvestorConfig


def _apply_per_holding_targets(holdings, targets: dict) -> None:
    """Attach per-holding rebalancing targets in place.

    ``targets`` is keyed by ISIN (when present) or uppercased ticker, as
    produced by ``load_targets_per_holding``. Matching uses the canonical
    ``instrument_key`` (ISIN, else ``TICKER:<symbol>``) with a fallback to the
    legacy raw-ISIN / uppercased-ticker keys, so existing target files keep
    matching identically while new lookups go through one rule. A holding with
    no matching target is left untouched.
    """
    if not targets:
        return
    from tarzan.models.instrument_key import instrument_key, normalize_ticker
    matched = 0
    for h in holdings:
        # Canonical key first (built the same way for holding and target row),
        # then the historical raw-ISIN / uppercased-bare-ticker fallbacks.
        t = (targets.get(instrument_key(h.isin, h.ticker))
             or targets.get(h.isin)
             or (targets.get(normalize_ticker(h.ticker)) if h.ticker else None))
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
    from tarzan.models.instrument_key import instrument_key

    # The targets dict now stores each entry under BOTH a legacy and a
    # canonical key, so iterate UNIQUE entries (by identity) to avoid
    # double-seeding the same instrument.
    #
    # A target must be recognised as "already held" if it matches a holding by
    # ISIN, ticker, OR name — not only the canonical instrument_key. Otherwise a
    # fund whose order carries only an ISIN (holding key = ISIN) but whose target
    # row carries only a ticker (target key = TICKER:...) is seeded as a phantom
    # buy-new AND left target-less as a real holding, so the SAME fund shows up
    # twice: once "buy to N%" and once "exit to 0%". Matching on any identifier
    # collapses them back to one instrument. (See _apply_per_holding_targets,
    # which should have attached the target; this is the safety net.)
    def _norm(s) -> str:
        return str(s or "").strip().upper()

    def _bare(t) -> str:
        return _norm(t).split(".")[0]  # NTSG.DE → NTSG

    held_keys: set[str] = set()
    held_isins: set[str] = set()
    held_tickers: set[str] = set()
    held_names: set[str] = set()
    for h in holdings:
        held_keys.add(instrument_key(h.isin, h.ticker))
        if h.isin:
            held_isins.add(_norm(h.isin))
        if h.ticker:
            held_tickers.add(_norm(h.ticker))
            held_tickers.add(_bare(h.ticker))
        if h.name:
            held_names.add(_norm(h.name))
    held_keys.discard("")

    seen: set[int] = set()
    seeded = []
    for row in targets.values():
        if id(row) in seen:
            continue
        seen.add(id(row))
        tpf = row.get("target_portfolio")
        if tpf is None or tpf <= 0:
            continue
        r_isin = (row.get("isin") or "").strip()
        r_ticker = (row.get("ticker") or "").strip()
        r_name = (row.get("name") or "").strip()
        if not r_isin and not r_ticker:
            continue
        # Name match tolerates the common short-name convention: a target row's
        # name is often a prefix of the holding's full legal name ("WisdomTree
        # Global Efficient Core" vs "...UCITS ETF USD Acc"). Match when either
        # name starts with the other (min 8 chars, so short tokens don't collide).
        rn = _norm(r_name)
        name_hit = bool(rn) and len(rn) >= 8 and any(
            hn.startswith(rn) or rn.startswith(hn) for hn in held_names
        )
        # Held by canonical key, ISIN, (bare or full) ticker, or name → skip.
        already_held = (
            instrument_key(r_isin, r_ticker) in held_keys
            or (r_isin and _norm(r_isin) in held_isins)
            or (r_ticker and (_norm(r_ticker) in held_tickers or _bare(r_ticker) in held_tickers))
            or name_hit
        )
        if already_held:
            continue  # already held — don't seed a phantom duplicate
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
    deterministic: bool = False,
    as_of=None,
    strict: bool = False,
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
    from tarzan.runtime import data_quality as _dq
    from tarzan.runtime import audit as _audit
    from tarzan import runtime as _runtime
    # Configure the run clock/determinism FIRST so every downstream default
    # ``today`` and the live/AI gates see it. Default (both falsy) restores
    # the pre-existing live behavior exactly.
    if deterministic or as_of is not None:
        _runtime.configure(deterministic=deterministic, as_of=as_of)
    else:
        _runtime.reset()
    _cfg.reset_input_caches()
    _geo.reset_caches()
    # Start a fresh per-run data-quality report and rebalancing audit trail so
    # this run never shows a previous run's issues/plans. The CLI writes both
    # out after the run.
    _dq.reset()
    _audit.reset()

    config = load_config(config_source)
    logger.info("Config loaded (target tolerance=±%.1f%%)", config.rebalancing_target_tolerance_pctg)

    # Load the order list — the single input that drives the whole report.
    orders = None
    if orders_source is not None:
        try:
            orders = load_orders(orders_source, orders_filename, strict=strict) or None
            if orders:
                logger.info("Loaded %d orders", len(orders))
        except DataIngestionError:
            # In strict mode a schema violation must surface as an actionable
            # rejection, not silently degrade to "no orders" (empty report).
            if strict:
                raise
            logger.warning("Order list rejected by schema; treating as no orders.")
            orders = None
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

    # Referential-integrity check (order → taxonomy dimension): flag held
    # instruments that have no curated taxonomy row AND no explicit asset class
    # (so they fell back to a default). A missing reference is otherwise
    # silent — the instrument just shows up as "Alternative"/"Other" — which
    # can quietly distort allocations. Surface it in the data-quality report.
    _check_taxonomy_coverage(holdings)

    # Compute
    logger.info("Computing metrics...")
    engine = MetricsEngine(holdings, config, orders=orders, rebalance_seeds=seeds)
    metrics = engine.compute_all()
    logger.info("Total portfolio value: €%.2f", metrics.total_value)

    return metrics, config


def _check_taxonomy_coverage(holdings) -> None:
    """Referential-integrity check: held instruments with no taxonomy row.

    The curated ``instrument_taxonomy.csv`` is the dimension that supplies an
    instrument's asset class / role / geography / notional exposure. A held
    ISIN absent from it is a dangling reference — the instrument silently
    falls back to defaults (typically ``Alternative``), which can distort the
    allocation without any error. We surface each such instrument as a
    data-quality WARNING (best-effort; never fatal).
    """
    try:
        from tarzan import config as _cfg
        from tarzan.runtime import data_quality as _dq
        from tarzan.models.instrument_key import normalize_isin, normalize_ticker

        taxonomy = _cfg.instrument_taxonomy()  # keys: uppercased ISIN + bare ticker
        if not taxonomy:
            _dq.warning(
                "taxonomy",
                "instrument_taxonomy.csv is empty/unavailable — every held "
                "instrument used default classification (asset class / geography "
                "/ exposure); allocations may be approximate.",
                context="instrument_taxonomy.csv",
            )
            return
        for h in holdings:
            if getattr(h, "is_seeded_target", False):
                continue  # seeds aren't part of the real snapshot
            isin_k = normalize_isin(getattr(h, "isin", None))
            tick_k = normalize_ticker(getattr(h, "ticker", None))
            if (isin_k and isin_k in taxonomy) or (tick_k and tick_k in taxonomy):
                continue
            label = h.ticker or h.isin or "?"
            ac = h.asset_class.value if getattr(h, "asset_class", None) else "Alternative"
            _dq.warning(
                "taxonomy",
                f"{label} has no instrument_taxonomy.csv row — classified by "
                f"fallback (asset class '{ac}'); add a taxonomy row to control "
                "its asset class / geography / notional exposure.",
                context=(h.isin or h.ticker),
            )
    except Exception as e:  # noqa: BLE001 — a diagnostic must never break the run
        logger.debug("Taxonomy coverage check failed: %s", e)


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
