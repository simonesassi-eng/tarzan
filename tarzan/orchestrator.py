"""Pipeline orchestrator: load → enrich → compute → PortfolioMetrics.

Single entry point used by the CLI (main.py).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union

from tarzan.contracts.exceptions import DataIngestionError
from tarzan.contracts.targets import TargetError, TargetSetOutcome, TargetSetStatus
from tarzan.data.loader import (
    load_config,
    load_orders,
    load_targets_per_holding,
)
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.effective_orders import EffectiveOrderSnapshot

logger = logging.getLogger(__name__)


def _apply_per_holding_targets(holdings, targets: dict, *, catalog=None) -> None:
    """Attach per-holding rebalancing targets in place.

    Every direct, cross-reference, and taxonomy-resolved match contributes a
    candidate before selection. A target is applied only when all compatible
    evidence converges on one distinct source row; collisions are never
    resolved by lookup order, and ticker aliases never override a conflicting
    explicit ISIN.
    """
    if isinstance(targets, TargetSetOutcome):
        for holding in holdings:
            holding.target_plan = targets
    if not targets:
        return

    from tarzan.data.identity_catalog import TargetIdentityCatalog, identity_aliases
    from tarzan.models.instrument_key import normalize_isin

    catalog = catalog or TargetIdentityCatalog()
    _identity_aliases = identity_aliases

    def _usable_isin(value) -> str:
        normalized = normalize_isin(value)
        return (
            normalized
            if len(normalized) == 12 and normalized[:2].isalpha()
            else ""
        )

    # Both canonical and legacy mapping keys point to the same row object.
    # Resolve each source row once, then publish every safe identity alias and
    # retain its strongest ISIN evidence for compatibility filtering.
    resolved_aliases: dict[str, dict[int, dict]] = {}
    target_isins: dict[int, str] = {}
    seen_rows: set[int] = set()
    for row in targets.values():
        row_id = id(row)
        if row_id in seen_rows:
            continue
        seen_rows.add(row_id)
        resolved_isin, resolved_ticker = catalog.resolve_row(row)
        target_isins[row_id] = (
            _usable_isin(row.get("isin")) or _usable_isin(resolved_isin)
        )
        for alias in _identity_aliases(resolved_isin, resolved_ticker):
            resolved_aliases.setdefault(alias, {})[row_id] = row

    matched = 0
    for h in holdings:
        candidates: dict[int, dict] = {}
        identity = catalog.holding(h)
        holding_aliases = set(identity.aliases)
        holding_isin = _usable_isin(identity.isin)

        def _add_target_keys(keys: set[str]) -> None:
            for key in keys - {""}:
                row = targets.get(key)
                if row is not None:
                    candidates[id(row)] = row

        # Direct canonical and compatibility keys are evidence, not an early
        # winner. A second row discovered below must make the match ambiguous.
        _add_target_keys(holding_aliases)

        for alias in holding_aliases:
            candidates.update(resolved_aliases.get(alias, {}))

        if holding_isin:
            compatible = {
                row_id: row
                for row_id, row in candidates.items()
                if not target_isins.get(row_id)
                or target_isins[row_id] == holding_isin
            }
            if len(compatible) != len(candidates):
                logger.info(
                    "Ignored target alias with conflicting ISIN for holding %s",
                    h.isin or h.ticker,
                )
            candidates = compatible

        if len(candidates) != 1:
            if len(candidates) > 1:
                # In per-holding-only planning, a missing target otherwise
                # means 0% and could turn an ambiguity into a liquidation.
                # Freeze the affected holding until the input collision is
                # resolved; never execute either candidate implicitly.
                h.no_buy_no_sell = True
                logger.error(
                    "Multiple target rows resolve to holding %s; holding frozen "
                    "and no target applied",
                    h.isin or h.ticker,
                )
            continue

        target = next(iter(candidates.values()))
        requested_ticker = str(target.get("ticker") or "").strip()
        if requested_ticker:
            _, target_ticker = catalog.resolve_row(target)
            h.ticker_requested = requested_ticker
            if target_ticker:
                # A matched target row is explicit user ticker authority. Feed
                # its curated identity into the sole market resolver so one
                # canonical listing is selected for history/current data and
                # remains the first choice for intraday data.
                h.ticker = target_ticker
        if target.get("target_equities") is not None:
            h.target_equities = target["target_equities"]
        if target.get("target_fixed_income") is not None:
            h.target_fixed_income = target["target_fixed_income"]
        if target.get("target_portfolio") is not None:
            h.target_portfolio = target["target_portfolio"]
        h.no_buy_no_sell = bool(target.get("no_buy_no_sell", False))
        matched += 1
    logger.info("Applied per-holding targets to %d/%d holdings", matched, len(holdings))


def _seed_missing_targets(holdings, targets: dict, *, catalog=None) -> list:
    """Create enriched zero-value holdings for target instruments not held.

    Target rows may intentionally carry a bare ticker. Before constructing a
    seed, resolve that partial identity through curated taxonomy. Explicit ISIN
    evidence wins; a suffixed ticker requires an exact listing match; bare
    matching is reserved for tickers that remain genuinely unresolved.
    """
    from tarzan.data.identity_catalog import TargetIdentityCatalog
    from tarzan.models.holding import Holding
    from tarzan.models.instrument_key import instrument_key

    catalog = catalog or TargetIdentityCatalog()

    def _norm(value) -> str:
        return str(value or "").strip().upper()

    def _bare(ticker) -> str:
        return _norm(ticker).split(".")[0]  # NTSG.DE → NTSG

    def _resolved_identity(row: dict) -> tuple[str, str]:
        return catalog.resolve_row(row)

    # Detect many-to-one target rows before creating any synthetic holding.
    # Canonical/legacy aliases of the same row share object identity and count
    # once; distinct source rows resolving to one economic identity are all
    # suppressed rather than emitted as duplicate buy-new instructions.
    target_identity_rows: dict[str, set[int]] = {}
    scanned_rows: set[int] = set()
    for row in targets.values():
        row_id = id(row)
        if row_id in scanned_rows:
            continue
        scanned_rows.add(row_id)
        target_portfolio = row.get("target_portfolio")
        if target_portfolio is None or target_portfolio <= 0:
            continue
        raw_isin = (row.get("isin") or "").strip()
        raw_ticker = (row.get("ticker") or "").strip()
        if not raw_isin and not raw_ticker:
            continue
        resolved_isin, resolved_ticker = _resolved_identity(row)
        identity = instrument_key(resolved_isin, resolved_ticker)
        if identity:
            target_identity_rows.setdefault(identity, set()).add(row_id)

    collided_row_ids = {
        row_id
        for row_ids in target_identity_rows.values()
        if len(row_ids) > 1
        for row_id in row_ids
    }
    for identity, row_ids in target_identity_rows.items():
        if len(row_ids) > 1:
            logger.error(
                "Multiple positive target rows resolve to %s; no rebalance "
                "seed created for that identity",
                identity,
            )

    # Keep each holding's identity evidence together. This prevents a ticker or
    # name from overriding a different explicit ISIN merely because aggregate
    # sets happen to contain both values.
    held_identities: list[tuple[str, set[str], set[str], str]] = []
    for holding in holdings:
        identity = catalog.holding(holding)
        identity_isin, exact_tickers = identity.isin, set(identity.tickers)
        bare_tickers = {_bare(ticker) for ticker in exact_tickers if ticker}
        held_identities.append(
            (identity_isin, exact_tickers, bare_tickers, _norm(holding.name))
        )

    seen: set[int] = set()
    seeded = []
    for row in targets.values():
        if id(row) in seen:
            continue
        seen.add(id(row))
        if id(row) in collided_row_ids:
            continue
        target_portfolio = row.get("target_portfolio")
        if target_portfolio is None or target_portfolio <= 0:
            continue

        raw_isin = (row.get("isin") or "").strip()
        raw_ticker = (row.get("ticker") or "").strip()
        row_name = (row.get("name") or "").strip()
        if not raw_isin and not raw_ticker:
            continue
        resolved_isin, resolved_ticker = _resolved_identity(row)
        if (resolved_isin, resolved_ticker) != (raw_isin, raw_ticker):
            logger.info(
                "Resolved target identity %s → %s",
                raw_ticker or raw_isin,
                resolved_ticker or resolved_isin,
            )

        target_isin = _norm(resolved_isin)
        target_ticker = _norm(resolved_ticker)
        target_has_full_ticker = bool(target_ticker and "." in target_ticker)
        normalized_name = _norm(row_name)

        if target_isin:
            # A different known ISIN cannot be overridden by ticker or name.
            # Exact full ticker is accepted only for a holding with no ISIN.
            already_held = any(
                held_isin == target_isin
                or (
                    not held_isin
                    and target_has_full_ticker
                    and target_ticker in exact_tickers
                )
                for held_isin, exact_tickers, _, _ in held_identities
            )
        elif target_has_full_ticker:
            target_bare = _bare(target_ticker)
            already_held = any(
                target_ticker in exact_tickers
                or any(
                    "." not in held_ticker
                    and _bare(held_ticker) == target_bare
                    for held_ticker in exact_tickers
                )
                for _, exact_tickers, _, held_name in held_identities
                if held_name or exact_tickers
            )
        elif target_ticker:
            target_bare = _bare(target_ticker)
            already_held = any(
                target_bare in bare_tickers
                for _, _, bare_tickers, _ in held_identities
            )
        else:
            # Similar names are not an economic-identity assertion.
            already_held = False

        if already_held:
            continue

        seeded.append(Holding(
            isin=resolved_isin,
            ticker=resolved_ticker,
            quantity=0.0,
            cost_basis_eur=0.0,
            market_value_eur=0.0,
            currency="EUR",
            name=row_name or resolved_ticker or resolved_isin,
            ticker_requested=raw_ticker or None,
            target_portfolio=float(target_portfolio),
            is_seeded_target=True,
        ))
    return seeded


def _run_once(
    config_source: Optional[Union[str, io.BytesIO]] = None,
    orders_source: Optional[Union[str, io.BytesIO]] = None,
    targets_per_holding_source: Optional[Union[str, io.BytesIO]] = None,
    orders_filename: str = "",
    targets_per_holding_filename: str = "",
    strict: bool = False,
) -> tuple[PortfolioMetrics, InvestorConfig]:
    """Execute the full analysis pipeline (order-only).

    The order list is the single source of truth: the snapshot — net
    quantity, average-cost basis, market value (via enrichment),
    allocations, targets, rebalancing — is *derived* from the orders, and
    the order list also owns the historical value series and XIRR/TWR.
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
    from tarzan import runtime as _runtime
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.session import current_session

    # The public ``run`` boundary has already acquired the serial lease,
    # established the sole run clock, activated a RunSession, and reset the
    # context-local diagnostic projections. Input caches remain process-wide
    # compatibility state, but can only be reset by the active leased run.
    session = current_session()
    if session is None:
        raise RuntimeError("_run_once requires an active RunSession")
    _cfg.reset_input_caches()
    _geo.reset_caches()

    config = load_config(config_source)
    config_snapshot = asdict(config)
    session.bind_config(config_snapshot)
    from tarzan.runtime.transport import current_budget
    current_budget()
    session.ledger.append(LedgerEntryType.STAGE, {
        "stage": "configuration",
        "outcome": "SUCCEEDED",
        "availability": "AVAILABLE",
    })
    logger.info("Config loaded (target tolerance=±%.1f%%)", config.rebalancing_target_tolerance_pctg)

    # Load the order list — the single input that drives the whole report.
    orders = None
    if orders_source is not None:
        try:
            orders = load_orders(orders_source, orders_filename, strict=strict) or None
            if orders:
                logger.info("Loaded %d orders", len(orders))
        except DataIngestionError:
            # A rejected file never establishes a valid portfolio, regardless
            # of whether extra columns are tolerated by the row contract.
            raise
        except Exception as e:
            raise DataIngestionError(
                f"Order list could not be read ({type(e).__name__})."
            ) from e

    if not orders:
        raise DataIngestionError(
            "No accepted orders are available; an analysis cannot be initialized."
        )

    # Form the effective ledger exactly once. Every financial consumer below
    # receives this immutable on/before-boundary view; the original accepted
    # list remains input provenance only and cannot leak into pinned analysis.
    order_snapshot = EffectiveOrderSnapshot.build(orders, _runtime.as_of())
    orders = list(order_snapshot.orders)
    logger.info(
        "Effective order snapshot: accepted=%d excluded=%d boundary=%s digest=%s",
        order_snapshot.accepted_count,
        order_snapshot.excluded_count,
        order_snapshot.boundary.isoformat() if order_snapshot.boundary else "live",
        order_snapshot.digest,
    )
    session.memo["effective_orders"] = {
        "digest": order_snapshot.digest,
        "accepted_count": order_snapshot.accepted_count,
        "excluded_count": order_snapshot.excluded_count,
        "boundary": order_snapshot.boundary.isoformat() if order_snapshot.boundary else None,
    }
    session.ledger.append(LedgerEntryType.BOUNDARY, {
        "stage": "effective_orders",
        **session.memo["effective_orders"],
        "outcome": "SUCCEEDED",
    })
    if not orders:
        raise DataIngestionError(
            "No orders are effective on or before the analysis boundary."
        )

    _check_offline_identity_bootstrap(orders)

    # Resolve mechanics and identifier continuity only from the effective
    # ledger. Profile reads are as-of aware; only a LIVE run may refresh them
    # through OpenFIGI. Work on copies so accepted input provenance remains
    # unchanged while every downstream financial consumer receives the same
    # resolved evidence.
    from tarzan.data.enricher import (
        reset_run_caches,
        resolve_effective_order_instruments,
    )
    reset_run_caches()
    orders, instrument_resolution = resolve_effective_order_instruments(orders)
    session.memo["instrument_resolution"] = instrument_resolution
    session.ledger.append(LedgerEntryType.STAGE, {
        "stage": "instrument_resolution",
        "outcome": "SUCCEEDED",
        "profiles_requested": instrument_resolution["profiles_requested"],
        "profile_sources": instrument_resolution["profile_sources"],
        "profile_statuses": instrument_resolution["profile_statuses"],
        "resolved_kind_count": len(
            instrument_resolution["resolved_kind_isins"]
        ),
        "equivalence_groups": instrument_resolution["equivalence_groups"],
        "conflicts": instrument_resolution["conflicts"],
    })

    # Derive the snapshot from the effective order list (net quantity,
    # basis, market value via enrichment) and attach per-instrument
    # rebalancing targets by ISIN.
    from tarzan.engine.returns_builder import build_holdings_from_orders
    holdings = build_holdings_from_orders(orders)
    targets_by_isin = _load_targets_or_empty(
        targets_per_holding_source, targets_per_holding_filename
    )
    from tarzan.data.identity_catalog import TargetIdentityCatalog

    target_catalog = TargetIdentityCatalog()
    _apply_per_holding_targets(holdings, targets_by_isin, catalog=target_catalog)
    session.memo["target_plan"] = {
        "status": targets_by_isin.status.value,
        "planning_eligible": targets_by_isin.planning_eligible,
        "error_codes": [error.code for error in targets_by_isin.errors],
        "rows": dict(targets_by_isin),
    }
    session.ledger.append(LedgerEntryType.STAGE, {
        "stage": "targets", **session.memo["target_plan"],
    })

    # When the per-instrument plan carries weights, IT is the plan: the asset-class and
    # equity-geography targets are derived from it and the ones declared in targets.csv
    # are not used. Three hand-maintained lists had no way to stay in agreement and on
    # the reference book had stopped -- the plan implies 21% fixed income against a
    # declared 28%, and 15% alternative against 11% -- so every drift figure measured
    # the book against a portfolio its own plan does not produce.
    #
    # Placed here, before enrichment and before any consumer reads a target, so the
    # engine, the rebalancer and the newsletter all see one set of numbers.
    from tarzan.engine.target_derivation import apply_to_config as _derive_targets

    target_derivation = _derive_targets(config, targets_by_isin)
    if target_derivation and target_derivation.get("applied"):
        session.ledger.append(LedgerEntryType.STAGE, {
            "stage": "targets",
            "outcome": "SUCCEEDED",
            "availability": Availability.AVAILABLE.value,
            "source": "derived_from_per_holding_plan",
            "instruments": len(target_derivation["plan"]),
            "notional_pct": round(target_derivation["notional_pct"], 4),
        })
        for _note in target_derivation.get("notes") or []:
            from tarzan.runtime import data_quality as _dq_t
            _dq_t.warning("targets", _note, context="targets_per_holding.csv")

    if not holdings:
        # A fully liquidated book is a legitimate state, not bad input: every
        # effective order nets to zero, so €0 IS the true total and the realized
        # gain on the closed round trips is a real figure the reader is owed.
        # Returning an empty PortfolioMetrics here threw that away and called it an
        # error — the run exited 1, no issue was rendered, and summary.json still
        # said SEND_NORMAL, so three parts of one run disagreed about it.
        #
        # ZERO EFFECTIVE ORDERS is the input error, and both branches above already
        # catch it and return. Reaching here means the orders are real; only the
        # position is empty. So the pipeline continues: the returns stage derives
        # lifetime P&L from the order flows, which for a liquidated book IS the
        # realized gain, and TWR is the return over the periods the money was in.
        logger.warning(
            "All %d effective order(s) net to zero: the book is fully liquidated "
            "and holds nothing. Reporting realized P&L over the closed positions.",
            len(orders),
        )
        session.ledger.append(LedgerEntryType.STAGE, {
            "stage": "snapshot",
            "outcome": "SUCCEEDED",
            "availability": Availability.AVAILABLE.value,
            "holdings": 0,
            "fully_liquidated": True,
        })

    # Per-holding-only mode: seed zero-value positions for target instruments
    # not currently held, so the optimizer can buy them toward target. These
    # seeds are kept SEPARATE from the real snapshot — they must never appear
    # in Holdings / Returns / allocations, only feed the rebalancer.
    seeds = []
    if config.target_use_per_holding_only and targets_by_isin:
        seeds = _seed_missing_targets(holdings, targets_by_isin, catalog=target_catalog)
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

    # Write today's market point ONCE, before anything judges or consumes a
    # price. Every "today" figure reads a price series while the portfolio's own
    # valuation is selected separately, so "today" used to exist twice: the
    # stamp lived in MetricsEngine and therefore ran AFTER this policy and after
    # _valuation, which is why total_value and the series terminal could not
    # agree by construction. Measured live on 24 Aug 2026 they sat €115 apart on
    # a six-figure book, entirely from two holdings whose valuation had fallen back
    # (MONEY.MI to its 10.0920 order price against a market 10.1840).
    #
    # The seeds are stamped alongside the book. They are excluded from every
    # table, but their price series is read — the target-allocation line on the
    # "Vs target &" charts is built from held sleeves AND seeds, and the tracked
    # benchmarks (which include the same instruments) are stamped in the engine.
    # Leaving the seeds out ended that line 77% on today's live prices and 23% on
    # yesterday's closes: measured live on 26 Aug 2026 its since-inception figure
    # read +13.87% where the all-closes computation gives +13.64%. It also priced
    # a buy-new action at the previous close.
    from tarzan.data import current_session
    stamped_tickers = current_session.apply_to_holdings(holdings + seeds)
    if stamped_tickers:
        session.ledger.append(LedgerEntryType.STAGE, {
            "stage": "current_session_stamp",
            "stamped": list(stamped_tickers),
        })

    # Apply the explicit instrument/data-class valuation policy before any
    # optimizer call. A material or indeterminate gap keeps a labeled known
    # subtotal for evidence but suppresses planning and normal publication.
    from tarzan.runtime.provider import ValuationCompletenessEvaluator
    valuation = ValuationCompletenessEvaluator(
        config.provider_quality_policies
    ).evaluate(holdings, session.ledger)
    session.memo["valuation"] = {
        "availability": valuation.availability.value,
        "trustworthy_total_eur": valuation.trustworthy_total_eur,
        "known_subtotal_eur": valuation.known_subtotal_eur,
        "missing_materiality_pct": valuation.missing_materiality_pct,
        "planning_eligible": valuation.planning_eligible,
        "failure_refs": list(valuation.failure_refs),
        "coverage": [asdict(item) for item in valuation.coverage],
    }
    session.ledger.append(LedgerEntryType.STAGE, {
        "stage": "valuation_completeness",
        **session.memo["valuation"],
    })

    # Compute
    logger.info("Computing metrics...")
    from tarzan.contracts.reference import ReportReference
    report_reference = ReportReference.prepare()
    engine = MetricsEngine(
        holdings,
        config,
        orders=orders,
        rebalance_seeds=seeds,
    )
    # Preserve the established constructor contract for injected engines while
    # attaching the run-scoped valuation gate before computation.
    engine.planning_eligible = valuation.planning_eligible
    engine.target_plan = targets_by_isin
    engine.valuation_assessment = valuation
    metrics = engine.compute_all()
    metrics.report_reference = report_reference
    from tarzan.data.market_quotes import fetch_market_quotes
    metrics.market_quotes = tuple(fetch_market_quotes())
    session.memo["workload"] = {
        "orders": len(orders),
        "holdings": len(holdings),
        "rebalance_seeds": len(seeds),
        "diagnostics": len(session.diagnostics),
        "plan_records": len(session.audit),
    }
    if metrics.trustworthy_total_value_eur is None:
        logger.error(
            "Portfolio total is Unavailable; known partial subtotal: €%.2f",
            metrics.known_valuation_subtotal_eur or 0.0,
        )
    else:
        logger.info("Total portfolio value: €%.2f", metrics.trustworthy_total_value_eur)

    return metrics, config


def _check_offline_identity_bootstrap(orders) -> None:
    """Explain the one failure a first-ever pinned run cannot avoid.

    A pinned run (``--deterministic``) may not touch the network, so an ISIN is
    identifiable only from the curated taxonomy or from identity learned by an
    earlier live run. With neither, every holding resolves to an UNKNOWN kind,
    the valuation policy rejects it and the whole report is blocked — correctly,
    but the operator only saw an opaque failure reference and had no way to know
    the fix is a single online run. Emitted before enrichment so the guidance
    arrives immediately rather than after minutes of doomed resolution.
    """
    try:
        from tarzan import runtime as _runtime
        if _runtime.allows_live_transport():
            return

        from tarzan import config as _cfg
        from tarzan.data import price_cache
        from tarzan.models.instrument_key import normalize_isin

        isins = {
            normalize_isin(getattr(o, "isin", None))
            for o in orders
            if normalize_isin(getattr(o, "isin", None))
        }
        if not isins:
            return
        known = {
            isin for isin in isins
            if _cfg.resolve_taxonomy_identity(isin, "")[1]
            or price_cache.load_ticker_isin_reverse(isin)
            or price_cache.load_resolution(isin)
        }
        if known:
            return  # some identity exists; normal partial-coverage paths apply

        # Console, not the data-quality report: the run is already going to be
        # blocked by the valuation policy, and a second failure reference adds
        # noise instead of guidance. What was missing is an actionable line the
        # operator actually reads.
        logger.error(
            "None of the %d ISIN(s) in the order list can be identified "
            "offline: --deterministic forbids network access, and neither "
            "instrument_taxonomy.csv nor the local cache maps them to a market "
            "symbol yet. Run once WITHOUT --deterministic to learn the "
            "ISIN→symbol mapping (cached permanently), then pinned runs work "
            "offline.",
            len(isins),
        )
    except Exception as e:  # noqa: BLE001 — guidance must never break the run
        logger.debug("Offline identity bootstrap check failed: %s", e)


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
) -> TargetSetOutcome:
    """Retain the distinction between optional absence and invalid input."""
    if source is None:
        return TargetSetOutcome.absent()
    try:
        return load_targets_per_holding(source, filename)
    except Exception as e:  # noqa: BLE001
        from tarzan.runtime import data_quality as dq
        message = f"Per-holding target input is unreadable ({type(e).__name__}); planning is Unavailable."
        logger.error(message)
        dq.error("target_load", message, context=filename or "targets")
        return TargetSetOutcome(status=TargetSetStatus.INVALID, errors=(
            TargetError("UNREADABLE_TARGET_INPUT", message, "", ()),
        ))


# Supported production boundary: one active single-user analysis. Waiting
# attempts own only their bootstrap envelope and in-memory correlation ledger
# until the FIFO lease is acquired.
from tarzan.runtime.ledger import Availability, LedgerEntryType, RunLedger
from tarzan.runtime.session import (
    RunAttemptEnvelope,
    RunResult,
    RunSession,
    SerialExecutionGate,
    activate_session,
    canonical_analysis_id,
    record_last_run_result,
)

_SERIAL_EXECUTION_GATE = SerialExecutionGate()


def _analysis_evidence(session: RunSession, metrics: Any = None) -> dict[str, Any]:
    """Return canonical output-affecting evidence for the stable Analysis ID."""
    context = session.context
    from tarzan.contracts.material import MATERIAL_FACTS_VERSION, material_metrics, material_value

    evidence: dict[str, Any] = {
        "material_facts_version": MATERIAL_FACTS_VERSION,
        "mode": context.mode.value,
        "effective_date": (
            context.effective_date.isoformat() if context.effective_date else None
        ),
        "schema_versions": {
            key: value
            for key, value in context.schema_versions.items()
            if key != "telemetry"
        },
        "policy_versions": dict(context.policy_versions),
        "config": dict(session.config_snapshot),
        "effective_orders": (
            {key: value for key, value in session.memo["effective_orders"].items()
             if key != "excluded_count"}
            if session.memo.get("effective_orders") else None
        ),
        "valuation": session.memo.get("valuation"),
        "instrument_resolution": session.memo.get("instrument_resolution"),
        "target_plan": session.memo.get("target_plan"),
    }
    evidence = material_value(evidence)
    if metrics is not None:
        evidence["metrics"] = material_metrics(metrics)
    return evidence


def _record_terminal_result(
    session: RunSession,
    *,
    metrics: Any,
    config: Any,
    outcome: str,
    started_at: float,
) -> RunResult:
    from tarzan.contracts.report import ReportView

    report_view = None
    if metrics is not None:
        from tarzan.contracts.reference import reference_for

        reference = reference_for(metrics)
        report_view = ReportView.build(
            metrics, config, benchmark_alpha_beta=reference.benchmark_alpha_beta,
            benchmark_geo=reference.benchmark_geo,
        )
    evidence = _analysis_evidence(session, metrics)
    # Per-holding policy can derive effective class/geography targets after the
    # initial CSV was loaded. Both declared and actually used config matter.
    if config is not None and is_dataclass(config):
        from tarzan.contracts.material import material_value

        evidence["effective_config"] = material_value(config)
    analysis_id = canonical_analysis_id(evidence)
    session.analysis_id = analysis_id
    from tarzan.runtime.workload import build_workload_observation

    session.ledger.append(
        LedgerEntryType.TELEMETRY,
        build_workload_observation(
            session,
            metrics,
            outcome=outcome,
            duration_ms=(time.perf_counter() - started_at) * 1000.0,
        ),
    )
    session.ledger.append(LedgerEntryType.STAGE, {
        "stage": "run",
        "outcome": outcome,
        "analysis_id": analysis_id,
        "availability": (
            Availability.AVAILABLE.value
            if outcome == "SUCCEEDED"
            else Availability.UNAVAILABLE.value
        ),
    })
    result = RunResult(
        metrics=metrics,
        config=config,
        attempt_id=session.context.attempt_id,
        analysis_id=analysis_id,
        ledger=session.ledger,
        session=session,
        report_view=report_view,
    )
    record_last_run_result(result)
    return result


def run(
    config_source: Optional[Union[str, io.BytesIO]] = None,
    orders_source: Optional[Union[str, io.BytesIO]] = None,
    targets_per_holding_source: Optional[Union[str, io.BytesIO]] = None,
    orders_filename: str = "",
    targets_per_holding_filename: str = "",
    deterministic: bool = False,
    as_of=None,
    strict: bool = False,
) -> tuple[PortfolioMetrics, InvestorConfig]:
    """Run one serialized analysis while preserving the public tuple result."""
    from tarzan import runtime as _runtime
    from tarzan.runtime import audit as _audit
    from tarzan.runtime import data_quality as _dq

    envelope = RunAttemptEnvelope.create("orchestrator")
    bootstrap_ledger = RunLedger(envelope.attempt_id)
    with _SERIAL_EXECUTION_GATE.acquire(envelope):
        try:
            context = _runtime.configure(
                deterministic=deterministic,
                as_of=as_of,
                attempt_id=envelope.attempt_id,
                invocation_source=envelope.invocation_source,
            )
        except BaseException as error:
            bootstrap_ledger.open_failure(
                stage="run_context",
                stable_code="INVALID_RUN_CONTEXT",
                severity="CRITICAL",
                error=error,
                affected_outputs=["run", "portfolio", "planning", "publication"],
                analytical_impact="analysis did not start",
                publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
                context={"attempt_id": envelope.attempt_id},
            )
            # Configuration failed before a RunSession existed. Still publish
            # a fresh failed result; a caller must never see yesterday's result
            # as the outcome of this attempt.
            _runtime.reset(
                attempt_id=envelope.attempt_id,
                invocation_source=envelope.invocation_source,
            )
            analysis_id = canonical_analysis_id({
                "failure_code": "INVALID_RUN_CONTEXT",
                "deterministic": deterministic,
                "as_of": as_of,
            })
            bootstrap_ledger.append(LedgerEntryType.STAGE, {
                "stage": "run",
                "outcome": "FAILED",
                "analysis_id": analysis_id,
                "availability": Availability.UNAVAILABLE.value,
            })
            record_last_run_result(RunResult(
                metrics=None, config=None, attempt_id=envelope.attempt_id,
                analysis_id=analysis_id, ledger=bootstrap_ledger,
            ))
            raise

        session = RunSession(
            context=context,
            config_snapshot={},
            ledger=bootstrap_ledger,
            gate=_SERIAL_EXECUTION_GATE,
        )
        from tarzan.runtime.logging import bind_capture_to_session
        bind_capture_to_session(session)
        started_at = time.perf_counter()
        with activate_session(session):
            _dq.reset()
            _audit.reset()
            session.ledger.append(LedgerEntryType.BOUNDARY, {
                "stage": "run_session",
                "outcome": "LEASE_ACQUIRED",
                "mode": context.mode.value,
                "effective_date": (
                    context.effective_date.isoformat()
                    if context.effective_date else None
                ),
            })
            try:
                metrics, config = _run_once(
                    config_source=config_source,
                    orders_source=orders_source,
                    targets_per_holding_source=targets_per_holding_source,
                    orders_filename=orders_filename,
                    targets_per_holding_filename=targets_per_holding_filename,
                    strict=strict,
                )
                result = _record_terminal_result(
                    session,
                    metrics=metrics,
                    config=config,
                    outcome="SUCCEEDED",
                    started_at=started_at,
                )
                return result.compatibility_tuple()
            except BaseException as error:
                session.ledger.open_failure(
                    stage="orchestration",
                    stable_code="UNHANDLED_RUN_FAILURE",
                    severity="CRITICAL",
                    error=error,
                    affected_outputs=["portfolio", "planning", "publication"],
                    analytical_impact="analysis terminated before completion",
                    publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
                )
                _record_terminal_result(
                    session,
                    metrics=None,
                    config=(dict(session.config_snapshot) if session.config_snapshot else None),
                    outcome="FAILED",
                    started_at=started_at,
                )
                raise
            finally:
                if session.transport is not None:
                    try:
                        session.transport.close()
                    except Exception:
                        logger.warning("Provider client cleanup failed; analysis outcome retained.")
