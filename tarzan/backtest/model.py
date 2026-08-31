"""What-if / backtest domain model: a resolved instrument line and a portfolio.

Composition is taken from the CANONICAL source — ``holding.class_breakdown``
(the ``exp_*`` taxonomy columns, leverage-aware, may sum to >100%) — and the
funded-capital vector is derived from it. No name/description heuristics and no
hardcoded geography overrides live here anymore: both are data in
``instrument_taxonomy.csv`` flowing through the enricher.
"""

from __future__ import annotations

from typing import Optional

from tarzan.engine import allocations as alloc
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.taxonomy import GEO_ORDER as _GEO_ORDER, ORDER_WHATIF

# Asset-class labels (identical to AssetClass.value so they match the target
# keys parsed by InvestorConfig).
EQ = AssetClass.EQUITIES.value
FI = AssetClass.FIXED_INCOME.value
GOLD = AssetClass.GOLD.value
COMM = AssetClass.COMMODITIES.value
CRYPTO = AssetClass.CRYPTO.value
ALT = AssetClass.ALTERNATIVE.value
CASH = AssetClass.CASH_EQUIVALENTS.value

# Display orders — single source of truth in tarzan.models.taxonomy.
ASSET_ORDER = list(ORDER_WHATIF)
GEO_ORDER = list(_GEO_ORDER)


def _capital_from(notl: dict[str, float]) -> dict[str, float]:
    """Funded-capital vector from a notional one (levered → cash collateral).

    When notional sums to ≤100 the euros sit exactly where the exposure is
    (capital == notional). When it is levered (>100, e.g. an efficient-core
    90/60 or a 2x fund) the funded leg is capped at 100 and the remainder is
    collateral cash, since the leverage is financed rather than funded.
    """
    total = sum(notl.values())
    if total <= 100.0 + 1e-9:
        return dict(notl)
    funded = min(notl.get(EQ, 0.0), 100.0)
    return {EQ: funded, CASH: 100.0 - funded}


def composition_for(holding: Holding) -> tuple[dict[str, float], dict[str, float], str]:
    """Instrument composition as (notional, capital, explanation).

    Notional exposure comes from ``holding.class_breakdown`` (the taxonomy
    ``exp_*`` columns); when absent it falls back to a single 100% slice of the
    holding's primary asset class. Capital is derived via :func:`_capital_from`.
    """
    bd = holding.class_breakdown or {}
    notl = {(k.value if hasattr(k, "value") else str(k)): float(v)
            for k, v in bd.items() if v}
    if not notl:
        base = holding.asset_class.value if holding.asset_class else ALT
        return {base: 100.0}, {base: 100.0}, f"single class ({base.lower()})"
    gross = sum(notl.values())
    why = "taxonomy exp_* (levered)" if gross > 100.5 else "taxonomy exp_*"
    return notl, _capital_from(notl), why


class WhatIfItem:
    """One resolved + enriched line of a portfolio."""

    def __init__(self, bare, symbol, isin, weight, holding, comp_notional,
                 comp_capital, explanation, geo_taxonomy=None):
        self.bare = bare
        self.symbol = symbol
        self.isin = isin
        self.weight = weight
        self.holding = holding
        self.comp_notional = comp_notional
        self.comp_capital = comp_capital
        self.explanation = explanation
        # Authoritative equity geography from the taxonomy geo columns
        # (index_geo_allocation), resolved once by the loader via the canonical
        # geo_resolver lookup. Preferred over holding.geo_breakdown, which can
        # fall back to the LISTING country for EUR-listed swap ETFs (e.g. NTSG
        # / CL2 scrape as "100% Eurozone").
        self.geo_taxonomy = geo_taxonomy

    @property
    def gross(self) -> float:
        return sum(self.comp_notional.values())

    @property
    def geo_breakdown(self):
        """Equity geography: taxonomy geo columns first (authoritative for
        index/swap ETFs), else the enriched holding's breakdown."""
        return self.geo_taxonomy or self.holding.geo_breakdown


def compute_allocations(items: list[WhatIfItem]) -> dict:
    """Aggregate funded-capital and notional allocations (asset + equity geo).

    Uses the shared :mod:`tarzan.engine.allocations` primitive so a candidate
    portfolio is aggregated the SAME way the live portfolio is in
    ``MetricsEngine._allocations``.
    """
    cap = alloc.accumulate((it.weight, it.comp_capital) for it in items)
    notl = alloc.accumulate((it.weight, it.comp_notional) for it in items)

    # Equity sleeve distributed over each instrument's (cleaned) geography.
    def _geo_pairs(source: str):
        for it in items:
            eq = it.weight * (getattr(it, source).get(EQ, 0.0)) / 100.0
            if eq <= 0:
                continue
            gb = alloc.clean_geo(it.geo_breakdown)
            yield (eq, gb if gb else {"Other": 100.0})

    geo_cap = alloc.accumulate(_geo_pairs("comp_capital"))
    geo_notl = alloc.accumulate(_geo_pairs("comp_notional"))
    return {"cap": cap, "notl": notl, "geo_cap": geo_cap, "geo_notl": geo_notl}


class Portfolio:
    """A named portfolio with allocations (normalised views) and risk metrics."""

    def __init__(self, name, items, allocs, metrics, is_real=False):
        self.name = name
        self.items = items
        self.alloc = allocs
        self.metrics = metrics
        self.is_real = is_real
        self.cap = alloc.renorm(allocs["cap"])          # funded capital %, sums 100
        self.notl_gross = dict(allocs["notl"])          # notional %, sums to gross
        self.notl_mix = alloc.renorm(allocs["notl"])    # notional mix %, sums 100
        self.geo_cap = alloc.renorm(allocs["geo_cap"])  # equity geo (capital) %
        self.geo_notl = alloc.renorm(allocs["geo_notl"])  # equity geo (notional) %
        # Leverage applied per class = notional exposure − funded capital
        # (both as % of capital); sums to (gross − 100). Zero on unlevered legs.
        self.lev_by_class = {
            cls: self.notl_gross.get(cls, 0.0) - self.cap.get(cls, 0.0)
            for cls in set(self.notl_gross) | set(self.cap)
        }
        # Single aligned history (filled in later by the engine).
        self.synth_nav = None
        self.nav = None
        self.window = None
        self.metrics_aligned: dict = {}
        self.metrics_aligned_eur: dict = {}
        self.metrics_aligned_usd: dict = {}
        self.rob: dict = {}
        # Per-sleeve daily returns (columns = tickers) and their target weights,
        # in the reporting currency. The blended NAV above has the correlations
        # and the rebalancing baked in; these keep them separable.
        self.sleeve_returns = None
        self.sleeve_weights = None

    @property
    def gross(self) -> float:
        return sum(self.alloc["notl"].values())

    @property
    def leverage(self) -> float:
        return self.gross / 100.0

    def weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for it in self.items:
            out[it.bare] = out.get(it.bare, 0.0) + it.weight
        return out
