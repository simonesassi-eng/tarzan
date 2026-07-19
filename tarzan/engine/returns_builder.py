"""Build the order-derived historical value series for returns.

This module is the only place that knows about *orders + price history
together*. It turns a list of ``Order`` into:

  * a list of synthetic ``Holding`` objects (net quantity per ISIN, with
    cum/ex BTP netting) ready for the standard enrichment pipeline;
  * a dated portfolio-value series ``V(day)`` built from the real held
    quantity on each day, valued through the shared ``value_position``
    primitive so it agrees with the current-value path;
  * the external cash-flow series and the XIRR cash-flow list;
  * an explicit provenance record of which price source priced each
    instrument (yfinance / synthetic / carry_flat / excluded), so the
    coverage of the returns can be disclosed rather than hidden.

Design: Option Y — when an order list is present it is the single source
of truth for the historical series, and every history-dependent metric
is computed on it.

The causal fallback ladder for missing market history:

    1. yfinance     last market observation at or before date → "yfinance"
    2. synthetic    exact dated order-price observation       → "synthetic"
    3. carry_flat   latest prior order price carried forward  → "carry_flat"
    4. excluded     no causal price at or before date          → "excluded"

Market-complete history is AVAILABLE, causal order-price fallback is
DEGRADED, and any held-date exclusion makes history UNAVAILABLE.
"""

from __future__ import annotations

import bisect
import datetime
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd

from tarzan.data.bond_fetcher import value_position
from tarzan.engine import allocations as _alloc
from tarzan.instruments.registry import InstrumentKind, TypeEvidenceGateway
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType
from tarzan.runtime import data_quality as dq
from tarzan.runtime.ledger import Availability

logger = logging.getLogger(__name__)

_QTY_EPS = 0.01
_IdentityKey = tuple[str, str]


class InstrumentIdentityConflict(ValueError):
    """Raised when one ISIN asserts multiple explicit identity groups."""


def _normalized_isin(isin: str) -> str:
    """Return the complete normalized identifier used by identity keys."""
    return str(isin or "").strip().upper()


def _instrument_identity_by_isin(orders: list[Order]) -> dict[str, _IdentityKey]:
    """Resolve one explicit identity for every ISIN in ``orders``.

    A documented equivalence group is the only authority that may connect
    different ISINs. Otherwise the complete normalized ISIN is its own
    identity. Multiple explicit groups for one ISIN are contradictory input;
    fail closed instead of selecting one based on row order.
    """
    explicit_groups: dict[str, set[str]] = {}
    isins: set[str] = set()
    for order in orders:
        isin = _normalized_isin(order.isin)
        isins.add(isin)
        group = str(order.instrument_equivalence_group or "").strip()
        if group:
            explicit_groups.setdefault(isin, set()).add(group.casefold())

    conflicts = {
        isin: groups
        for isin, groups in explicit_groups.items()
        if len(groups) > 1
    }
    if conflicts:
        detail = "; ".join(
            f"{isin}: {', '.join(sorted(groups))}"
            for isin, groups in sorted(conflicts.items())
        )
        raise InstrumentIdentityConflict(
            "conflicting instrument_equivalence_group assertions for " + detail
        )

    return {
        isin: (
            ("equivalence", next(iter(explicit_groups[isin])))
            if isin in explicit_groups
            else ("isin", isin)
        )
        for isin in isins
    }


def _identity_key(
    isin: str,
    identity_by_isin: Optional[dict[str, _IdentityKey]] = None,
) -> _IdentityKey:
    """Look up an ISIN's resolved identity, defaulting to that full ISIN."""
    normalized = _normalized_isin(isin)
    return (identity_by_isin or {}).get(normalized, ("isin", normalized))


@dataclass
class OrderDerivedSeries:
    """Everything the returns computer needs from the order list.

    Attributes:
        valuations: ``(date, V_after)`` pairs in chronological order.
        external_flows: external inflow into the portfolio per date
            (deposits/buys positive, withdrawals/sells negative).
        xirr_cashflows: ``(date, amount)`` from the bank-account
            perspective (deposits negative, distributions positive),
            terminated by today's portfolio value.
        coverage_pct: % of latest portfolio value priced by real market
            data (rung 1) over the window.
        provenance: ``{source_tag: [isin, ...]}`` for disclosure.
        span_days: calendar days from the first flow to today.
        daily_series: dense daily-indexed portfolio value over the whole
            window, valued at market on every calendar day. This is the
            series risk metrics (volatility, Sharpe, VaR, beta) must use —
            the sparse ``valuations`` (trade dates only) would make
            ``pct_change`` span arbitrary multi-day gaps.
        actual_value_series: dense daily-indexed *raw* portfolio value
            (with the deposit/withdrawal jumps left in). Unlike
            ``daily_series`` (a flow-adjusted NAV index) this is the real
            euro worth of the whole patrimony over time, so it is what the
            newsletter's mountain chart plots.
    """

    valuations: list[tuple[datetime.date, float]]
    external_flows: dict[datetime.date, float]
    xirr_cashflows: list[tuple[datetime.date, float]]
    coverage_pct: float
    provenance: dict[str, list[str]]
    span_days: int
    history_availability: Availability = Availability.AVAILABLE
    unavailable_instruments: tuple[str, ...] = ()
    mechanics_unavailable_instruments: tuple[str, ...] = ()
    causal_price_unavailable_instruments: tuple[str, ...] = ()
    daily_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    actual_value_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    pnl_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Daily unrealized P&L = market value of open positions − their cost basis.
    # Consistent with the hero's snapshot (total_value − cost_basis), but as a
    # full daily series for charting.
    unrealized_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Holdings derivation (net quantity per ISIN + cum/ex netting)
# ---------------------------------------------------------------------------

def _net_qty_by_isin(orders: list[Order]) -> dict[str, float]:
    qty: dict[str, float] = {}
    for o in orders:
        if o.is_position_change():
            qty[o.isin] = qty.get(o.isin, 0.0) + o.quantity
    return qty


def _open_isins(
    qty_by_isin: dict[str, float],
    identity_by_isin: Optional[dict[str, _IdentityKey]] = None,
) -> set[str]:
    """Return open ISINs after explicit-equivalence closure.

    Different ISINs net together only when orders explicitly assign them the
    same equivalence group. Every ungrouped instrument retains its complete
    normalized ISIN as an isolated identity.
    """
    identity_totals: dict[_IdentityKey, float] = {}
    for isin, quantity in qty_by_isin.items():
        key = _identity_key(isin, identity_by_isin)
        identity_totals[key] = identity_totals.get(key, 0.0) + quantity

    open_isins: set[str] = set()
    for isin, quantity in qty_by_isin.items():
        key = _identity_key(isin, identity_by_isin)
        if abs(identity_totals[key]) < _QTY_EPS:
            continue  # explicitly equivalent variants net flat → closed
        if abs(quantity) < _QTY_EPS:
            continue  # individually closed
        open_isins.add(isin)
    return open_isins


def _order_instrument_kind_resolution(orders: list[Order], isin: str):
    """Resolve and retain every exact kind assertion for ``isin``."""
    assertions = tuple(
        order.instrument_kind.value
        for order in orders
        if order.isin == isin and order.instrument_kind is not None
    )
    return TypeEvidenceGateway().resolve(*assertions)


def _order_instrument_kind(
    orders: list[Order], isin: str
) -> Optional[InstrumentKind]:
    """Return one unambiguous exact kind declared by orders for ``isin``."""
    return _order_instrument_kind_resolution(orders, isin).kind


def _causal_enriched_by_isin(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
) -> dict[str, Holding]:
    """Project enriched holdings onto the effective order authority.

    ``Holding.instrument_kind_evidence`` is inherited from source orders. A
    holding built from a larger ledger can therefore retain declarations that
    were excluded by an as-of boundary. Replace those declarations before any
    mechanics consumer runs, and clear derived fields when excluded evidence
    may have selected them. Price history and independently fetched geography
    remain usable; historical category is re-derived from taxonomy or the
    effective intrinsic kind when inherited order evidence changed.
    """
    effective_evidence: dict[str, tuple[str, ...]] = {}
    for order in orders:
        if order.instrument_kind is not None:
            effective_evidence.setdefault(order.isin, ())
            effective_evidence[order.isin] += (order.instrument_kind.value,)

    gateway = TypeEvidenceGateway()
    projected: dict[str, Holding] = {}
    for isin, holding in enriched_by_isin.items():
        current_evidence = effective_evidence.get(isin, ())
        inherited_evidence = tuple(holding.instrument_kind_evidence or ())
        inherited_kinds = {value.strip().upper() for value in inherited_evidence}
        current_kinds = {value.strip().upper() for value in current_evidence}
        order_evidence_changed = bool(inherited_evidence) and (
            inherited_kinds != current_kinds
        )

        security_type = holding.security_type
        instrument_type = holding.instrument_type
        provider_resolution = gateway.resolve(instrument_type)
        inherited_resolved_kinds = {
            resolution.kind
            for value in inherited_evidence
            if (resolution := gateway.resolve(value)).kind is not None
        }
        provider_corroborates_causal_kind = False
        causal_kind: Optional[InstrumentKind] = None
        if current_evidence:
            resolution = gateway.resolve(*current_evidence)
            causal_kind = resolution.kind
            security_type = (
                resolution.kind.value if resolution.kind is not None else None
            )
            provider_corroborates_causal_kind = (
                resolution.kind is not None
                and provider_resolution.kind is resolution.kind
            )
            if (
                resolution.kind is None
                or provider_resolution.kind is not None
                and not provider_corroborates_causal_kind
            ):
                instrument_type = None
        elif inherited_evidence:
            # No order assertion survives the boundary. ``security_type`` may
            # have been derived from the excluded declaration, but a provider
            # ``instrument_type`` that resolves to a different kind is
            # independent contradictory evidence and remains causal.
            security_type = None
            provider_corroborates_causal_kind = (
                provider_resolution.kind is not None
                and provider_resolution.kind not in inherited_resolved_kinds
            )
            if provider_corroborates_causal_kind:
                causal_kind = provider_resolution.kind
            else:
                instrument_type = None

        category_corroborated = provider_corroborates_causal_kind
        intrinsic_category = {
            InstrumentKind.STOCK: AssetClass.EQUITIES,
            InstrumentKind.BOND: AssetClass.FIXED_INCOME,
            InstrumentKind.CASH: AssetClass.CASH_EQUIVALENTS,
        }.get(causal_kind)
        if category_corroborated and intrinsic_category is not None:
            if holding.asset_class not in (None, intrinsic_category):
                category_corroborated = False
            if holding.class_breakdown:
                normalized_breakdown = {
                    (
                        key if isinstance(key, AssetClass) else AssetClass(str(key))
                    ): float(value)
                    for key, value in holding.class_breakdown.items()
                }
                category_corroborated = (
                    category_corroborated
                    and set(normalized_breakdown) == {intrinsic_category}
                    and abs(normalized_breakdown[intrinsic_category] - 100.0)
                    < 1e-9
                )
        clear_historical_category = (
            order_evidence_changed and not category_corroborated
        )
        projected[isin] = replace(
            holding,
            security_type=security_type,
            instrument_type=instrument_type,
            instrument_kind_evidence=current_evidence,
            asset_class=(
                None if clear_historical_category else holding.asset_class
            ),
            class_breakdown=(
                None if clear_historical_category else holding.class_breakdown
            ),
        )
    return projected


def _seed_market_value(orders: list[Order], isin: str, qty: float) -> float:
    """Seed from the latest order price only when exact mechanics are known.

    The seed is replaced by an admissible provider quote during enrichment.
    Missing or conflicting order kinds produce no seed instead of treating an
    unknown instrument as a unit-priced stock/ETF.
    """
    priced = sorted(
        (
            o for o in orders
            if o.isin == isin and _usable_price(o.price_native) is not None
        ),
        key=lambda o: o.trade_date,
    )
    kind = _order_instrument_kind(orders, isin)
    if not priced or qty == 0 or kind is None:
        return 0.0
    last = priced[-1]
    price = _usable_price(last.price_native)
    if price is None:
        return 0.0
    fx = last.fx_rate or 1.0
    try:
        fx_numeric = float(fx)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(fx_numeric):
        return 0.0
    seeded = value_position(
        abs(qty),
        price,
        instrument_kind=kind,
    ) / (fx_numeric if fx_numeric > 0 else 1.0)
    return seeded if math.isfinite(seeded) else 0.0


def build_holdings_from_orders(orders: list[Order]) -> list[Holding]:
    """Aggregate orders into synthetic Holdings for the open positions.

    Net quantity per exact ISIN, explicit-equivalence closure, a seeded market
    value, and the average-cost basis of the units still held. The returned
    Holdings carry only what the enricher needs; enrichment fills in price
    history, asset class, etc.
    """
    qty_by_isin = _net_qty_by_isin(orders)
    identity_by_isin = _instrument_identity_by_isin(orders)
    open_isins = _open_isins(qty_by_isin, identity_by_isin)
    cost_by_isin = cost_basis_by_isin(orders)

    name_by_isin: dict[str, str] = {}
    ccy_by_isin: dict[str, str] = {}
    for o in orders:
        if o.name and o.isin not in name_by_isin:
            name_by_isin[o.isin] = o.name
        if o.currency and o.isin not in ccy_by_isin:
            ccy_by_isin[o.isin] = o.currency

    holdings: list[Holding] = []
    # Sort the open ISINs: ``_open_isins`` returns a set, whose iteration order
    # is hash-randomized per process. Leaving it unsorted makes the derived
    # holdings list — and everything downstream that preserves its order
    # (Excel rows, newsletter sleeve tables, tie-broken sorts) — vary run to
    # run, defeating reproducibility. A stable ISIN sort fixes it at the root.
    for isin in sorted(open_isins):
        qty = qty_by_isin[isin]
        kind_resolution = _order_instrument_kind_resolution(orders, isin)
        kind = kind_resolution.kind
        holdings.append(Holding(
            isin=isin,
            ticker=isin,
            quantity=qty,
            cost_basis_eur=cost_by_isin.get(isin, 0.0),
            market_value_eur=_seed_market_value(orders, isin, qty),
            currency=ccy_by_isin.get(isin, "EUR"),
            name=name_by_isin.get(isin, ""),
            security_type=kind.value if kind is not None else None,
            instrument_kind_evidence=kind_resolution.evidence,
        ))
    return holdings


def _apply_cost_order(qty: dict[str, float], cost: dict[str, float], o: Order) -> None:
    """Advance the running average-cost ``(qty, cost)`` books by one order.

      * a buy / transfer-in adds the EUR it committed — the net cash paid
        including fees (``net_eur``), or the gross transferred value when
        no cash moved (a ``transfer_in`` has ``net_eur == 0``);
      * a sell / transfer-out removes cost at the running *average* price,
        so realized gains/losses do not distort the basis of the units
        that remain;
      * coupons and dividends never touch cost basis (they are income, not
        a return of capital).
    """
    q = qty.get(o.isin, 0.0)
    c = cost.get(o.isin, 0.0)
    if o.quantity > 0:  # buy / transfer_in
        committed = abs(o.net_eur) if o.net_eur else abs(o.gross_eur or 0.0)
        qty[o.isin] = q + o.quantity
        cost[o.isin] = c + committed
    elif o.quantity < 0:  # sell / transfer_out
        sold = abs(o.quantity)
        if q > _QTY_EPS:
            avg = c / q
            cost[o.isin] = max(c - avg * min(sold, q), 0.0)
        qty[o.isin] = max(q - sold, 0.0)


def cost_basis_by_isin(orders: list[Order]) -> dict[str, float]:
    """Average-cost basis (EUR) of the *currently held* units per ISIN.

    Walks the position-changing orders in date order (see
    :func:`_apply_cost_order`). The result is the acquisition cost of the
    units still open today — the denominator the snapshot uses for
    per-holding unrealized P&L. Derived purely from the order list, so the
    holdings-only and order-only paths agree without needing a
    ``cost_basis_eur`` column in any CSV.
    """
    pos = sorted(
        (o for o in orders if o.is_position_change()),
        key=lambda o: o.trade_date,
    )
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    for o in pos:
        _apply_cost_order(qty, cost, o)
    return cost


# ---------------------------------------------------------------------------
# Quantity timeline (binary-searchable cumulative quantity per ISIN)
# ---------------------------------------------------------------------------

class QuantityTimeline:
    """Cumulative held quantity per ISIN as of end-of-day, with O(log n)
    lookup. Built once from the position-changing orders."""

    def __init__(self, orders: list[Order]):
        events: list[tuple[datetime.date, str, float]] = [
            (o.trade_date, o.isin, o.quantity)
            for o in orders if o.is_position_change()
        ]
        events.sort(key=lambda e: e[0])
        self._cum: dict[str, list[tuple[datetime.date, float]]] = {}
        running: dict[str, float] = {}
        for d, isin, delta in events:
            running[isin] = running.get(isin, 0.0) + delta
            self._cum.setdefault(isin, []).append((d, running[isin]))
        # Pre-extract the sorted event dates and running quantities per ISIN
        # once, so ``qty_at`` (called O(days × isins) in the series build) does
        # an O(log n) bisect instead of rebuilding the date list on every call.
        self._dates: dict[str, list[datetime.date]] = {}
        self._qtys: dict[str, list[float]] = {}
        for isin, series in self._cum.items():
            self._dates[isin] = [e[0] for e in series]
            self._qtys[isin] = [e[1] for e in series]

    def isins(self) -> list[str]:
        return list(self._cum.keys())

    def qty_at(self, isin: str, d: datetime.date) -> float:
        dates = self._dates.get(isin)
        if not dates or d < dates[0]:
            return 0.0
        # rightmost index whose date <= d
        i = bisect.bisect_right(dates, d) - 1
        return self._qtys[isin][i] if i >= 0 else 0.0


# ---------------------------------------------------------------------------
# Price lookup with explicit fallback ladder
# ---------------------------------------------------------------------------

def _usable_price(value: object) -> Optional[float]:
    """Return a finite positive price, otherwise ``None``.

    NaN and infinity are missing evidence, not market observations. Keeping
    this check at the resolver boundary ensures every source either supplies a
    usable price or falls through to the next causal rung.
    """
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _price_at(price_history: pd.Series, d: datetime.date) -> Optional[float]:
    """Last observed price at or before ``d`` in a tz-aware-safe way.

    Uses ``searchsorted`` (binary search, O(log n)) on the sorted price index
    rather than a full ``index <= threshold`` boolean scan (O(n)). This is the
    per-(ISIN, day) hot path in the daily-series build — at O(days × isins)
    calls, the linear scan dominated the run. The price history is always
    date-sorted (yfinance history is sorted by the enricher; the synthetic
    series is ``sort_index()``-ed), so the two are equivalent.
    """
    if price_history is None or len(price_history) == 0:
        return None
    idx = price_history.index
    idx_tz = getattr(idx, "tz", None)
    threshold = pd.Timestamp(d)
    if idx_tz is not None:
        threshold = threshold.tz_localize(idx_tz)
    # Rightmost usable observation whose index value is <= threshold. Invalid
    # rows from a provider do not become evidence and cannot poison valuation;
    # continue to the previous causal market observation before falling
    # through to the next resolver rung.
    pos = idx.searchsorted(threshold, side="right") - 1
    while pos >= 0:
        price = _usable_price(price_history.iloc[pos])
        if price is not None:
            return price
        pos -= 1
    return None


def _build_synthetic_history(orders: list[Order], isin: str) -> Optional[pd.Series]:
    """Daily-indexed series of order prices for an ISIN, converted to
    EUR and mean-aggregated per day. None if no observation.

    Order ``price_native`` is in the instrument's trade currency and
    ``fx_rate`` is Fineco's ``Cambio`` — units of native currency per
    EUR — so the EUR price is ``price_native / fx_rate``. Converting
    here means the synthetic/carry-flat rungs return EUR-per-unit
    prices, consistent with the yfinance rung (which the enricher has
    already converted to EUR). Without this a ZAR- or USD-denominated
    bond would be valued in its native currency and overstated by the
    FX rate.
    """
    obs = []
    for o in orders:
        if o.isin != isin:
            continue
        native_price = _usable_price(o.price_native)
        if native_price is None:
            continue
        fx = o.fx_rate or 1.0
        try:
            fx_numeric = float(fx)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(fx_numeric):
            continue
        eur_price = (
            native_price / fx_numeric
            if fx_numeric > 0.0
            else native_price
        )
        usable_eur_price = _usable_price(eur_price)
        if usable_eur_price is not None:
            obs.append((o.trade_date, usable_eur_price))
    if not obs:
        return None
    s = pd.Series(
        [p for _, p in obs],
        index=pd.to_datetime([d for d, _ in obs]),
    ).sort_index()
    return s.groupby(s.index).mean()


def _causal_order_price(
    series: pd.Series,
    d: datetime.date,
) -> tuple[Optional[float], str]:
    """Return the latest order-price observation at or before ``d``.

    An exact dated observation retains the ``synthetic`` provenance tag; a
    prior observation carried forward is ``carry_flat``. A later order may
    never value an earlier date, so dates before the first observation are
    explicitly excluded instead of backward-filled or interpolated.
    """
    if series is None or series.empty:
        return None, "excluded"
    ts = pd.Timestamp(d)
    position = series.index.searchsorted(ts, side="right") - 1
    while position >= 0:
        observation_date = series.index[position]
        price = _usable_price(series.iloc[position])
        if price is not None:
            source = "synthetic" if observation_date == ts else "carry_flat"
            return price, source
        position -= 1
    return None, "excluded"


class PriceResolver:
    """Resolve an ISIN's price on any date via the explicit fallback
    ladder, recording the source tag used per ISIN for provenance."""

    def __init__(
        self,
        orders: list[Order],
        enriched_by_isin: dict[str, Holding],
        today: Optional[datetime.date] = None,
    ):
        self._orders = [
            order for order in orders
            if today is None or order.trade_date <= today
        ]
        self._enriched = enriched_by_isin
        self._today = today
        self._synth: dict[str, Optional[pd.Series]] = {}
        self._instrument_kinds: dict[str, Optional[InstrumentKind]] = {}

    def _synthetic(self, isin: str) -> Optional[pd.Series]:
        if isin not in self._synth:
            self._synth[isin] = _build_synthetic_history(self._orders, isin)
        return self._synth[isin]

    def instrument_kind(self, isin: str) -> Optional[InstrumentKind]:
        """Resolve one exact, non-conflicting mechanics kind for an ISIN.

        Effective order declarations are the first authority. When they carry
        kind evidence, provider/enrichment evidence must not reintroduce a
        conflict derived from an order outside the as-of boundary. Provider
        evidence remains the fallback for ledgers that declare no kind.
        """
        if isin not in self._instrument_kinds:
            holding = self._enriched.get(isin)
            order_resolution = _order_instrument_kind_resolution(
                self._orders, isin
            )
            if order_resolution.evidence:
                resolution = order_resolution
            else:
                holding_evidence = tuple(
                    getattr(holding, "instrument_kind_evidence", ()) or ()
                ) if holding else ()
                resolution = TypeEvidenceGateway().resolve(
                    getattr(holding, "security_type", None) if holding else None,
                    getattr(holding, "instrument_type", None) if holding else None,
                    *holding_evidence,
                )
            self._instrument_kinds[isin] = resolution.kind
        return self._instrument_kinds[isin]

    def is_bond(self, isin: str) -> bool:
        """Compatibility predicate backed only by exact kind evidence."""
        return self.instrument_kind(isin) is InstrumentKind.BOND

    def _borsa_price(self, isin: str) -> Optional[float]:
        """Borsa Italiana today-price for a bond as EUR-per-unit, if the
        enricher set one, else None.

        yfinance does not quote BTPs / US Treasuries / foreign-currency
        notes, so for those the enricher's ``_try_terrapin_fallback``
        scrapes Borsa Italiana, converts the native clean price to EUR via
        the shared FX machinery, and stamps the holding with an
        EUR-per-unit ``current_price`` and a ``data_source`` like
        ``"borsa_italiana/mot/btp"``. We trust ``current_price`` as a
        market quote only when that source tag says so (a yfinance-derived
        price would already be reachable via ``price_history``; a stale CSV
        seed must not be mistaken for a live quote).
        """
        h = self._enriched.get(isin)
        if h is None:
            return None
        src = getattr(h, "data_source", None)
        if not src or not str(src).startswith("borsa_italiana"):
            return None
        price = getattr(h, "current_price", None)
        if not self.is_bond(isin):
            return None
        return _usable_price(price)

    def price_on(self, isin: str, d: datetime.date) -> tuple[Optional[float], str]:
        """Return (price_eur_per_unit, source) for an ISIN on a date.

        Note: prices from the enricher are already EUR-per-unit and, for
        bonds, already FX-converted and rescaled by /100 — so the caller
        must NOT apply the bond /100 again on the 'yfinance' or
        'borsa_italiana' rungs. Synthetic/carry_flat prices are raw order
        prices (already FX-converted to EUR) that still need the /100 via
        value_position. The returned source disambiguates which.
        """
        if self.instrument_kind(isin) is None:
            return None, "excluded"
        h = self._enriched.get(isin)
        ph = getattr(h, "price_history", None) if h else None
        if ph is not None and len(ph) > 0:
            price = _price_at(ph, d)
            if price is not None:
                return price, "yfinance"
        # Borsa Italiana single-point rung: only for the TODAY/terminal
        # valuation of a bond with no yfinance history. The scrape gives
        # only today's price (no series), so historical dates still fall
        # through to synthetic/carry_flat. The price is already EUR-per-unit
        # (FX-converted, post /100), so it is tagged like the yfinance rung
        # and the caller must NOT apply value_position.
        if self._today is not None and d >= self._today:
            borsa = self._borsa_price(isin)
            if borsa is not None:
                return borsa, "borsa_italiana"
        series = self._synthetic(isin)
        if series is not None and not series.empty:
            return _causal_order_price(series, d)
        return None, "excluded"


# ---------------------------------------------------------------------------
# Main: build the dated value series + cash flows + provenance
# ---------------------------------------------------------------------------

def _closed_identity_groups(
    timeline: "QuantityTimeline",
    d: datetime.date,
    identity_by_isin: Optional[dict[str, _IdentityKey]] = None,
) -> set[_IdentityKey]:
    """Explicit identity groups with approximately zero quantity on ``d``.

    This is the dated form of :func:`_open_isins`. A cum/ex reclassification
    is collapsed only when its orders carry the same explicit equivalence
    group; identifier shape never supplies that evidence.
    """
    identity_totals: dict[_IdentityKey, float] = {}
    for isin in timeline.isins():
        key = _identity_key(isin, identity_by_isin)
        identity_totals[key] = (
            identity_totals.get(key, 0.0) + timeline.qty_at(isin, d)
        )
    return {
        key for key, total in identity_totals.items()
        if abs(total) < _QTY_EPS
    }


def build_order_derived_series(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    today: Optional[datetime.date] = None,
) -> OrderDerivedSeries:
    """Build the order-derived valuation series, cash flows and
    provenance. ``enriched_by_isin`` maps ISIN → enriched Holding (from
    running the standard enrichment on ``build_holdings_from_orders``).

    All date-keyed logic (quantity timeline, cash flows, cost basis,
    synthetic prices, inception) keys on each order's ``trade_date`` —
    the date market exposure is taken on — not its settlement ``date``.
    This keeps the cash side and the asset side on the same clock, so a
    trade that settles after the run date (T+2) cannot make the cash
    flow land while the position it creates is still invisible.
    """
    # Defensive anchor for the live "value now" path (no explicit
    # ``today``): should a trade ever be dated after the run date, value
    # as of that date so the terminal valuation still covers every order
    # the cash flows count. With trade-date keying this is a safety net,
    # not the primary fix. An explicit ``today`` (historical/backtest
    # as-of valuation) is always respected verbatim.
    if today is None:
        from tarzan import runtime
        today = runtime.today()
        last_trade_date = max((o.trade_date for o in orders), default=today)
        if last_trade_date > today:
            today = last_trade_date
    # An explicit as-of boundary applies to the complete order authority, not
    # only price lookup. Future rows must not influence identity, mechanics,
    # quantities, cash flows, cost basis, or historical provenance.
    orders = [order for order in orders if order.trade_date <= today]
    enriched_by_isin = _causal_enriched_by_isin(orders, enriched_by_isin)
    identity_by_isin = _instrument_identity_by_isin(orders)
    timeline = QuantityTimeline(orders)
    resolver = PriceResolver(orders, enriched_by_isin, today=today)
    mechanics_unavailable = tuple(sorted(
        isin for isin in timeline.isins()
        if resolver.instrument_kind(isin) is None
    ))

    # Per-build memo for the closed explicit-identity set. It is a pure
    # function of (timeline, date, identity evidence), and the inputs are
    # immutable for this build, so compute it only once per calendar day.
    _closed_cache: dict[datetime.date, set[_IdentityKey]] = {}

    def closed_identity_groups(d: datetime.date) -> set[_IdentityKey]:
        cached = _closed_cache.get(d)
        if cached is None:
            cached = _closed_identity_groups(timeline, d, identity_by_isin)
            _closed_cache[d] = cached
        return cached

    # Track source use across every held date. Terminal provenance remains
    # separate because ``coverage_pct`` is explicitly a latest-value measure,
    # while history availability must account for the entire held interval.
    provenance: dict[str, list[str]] = {
        "yfinance": [], "borsa_italiana": [], "synthetic": [],
        "carry_flat": [], "excluded": [],
    }
    terminal_provenance: dict[str, list[str]] = {
        source: [] for source in provenance
    }

    def record_history_source(isin: str, source: str) -> None:
        provenance.setdefault(source, []).append(isin)

    def value_isin_on(isin: str, d: datetime.date) -> Optional[float]:
        """EUR value of one unit of ``isin`` on ``d`` at market price
        (None if unpriceable). Used to value quantity deltas for the
        TWROR external flow at the same price basis as the series.

        Flow-date evidence is part of whole-history completeness even when
        offsetting orders leave no end-of-day position to be valued.
        """
        price, source = resolver.price_on(isin, d)
        record_history_source(isin, source)
        if price is None:
            return None
        # 'yfinance' and 'borsa_italiana' prices are already EUR-per-unit
        # (bonds FX-converted and pre-/100 by the enricher); raw synthetic
        # prices are not.
        if source in ("yfinance", "borsa_italiana"):
            return price
        kind = resolver.instrument_kind(isin)
        if kind is None:
            return None
        return value_position(1.0, price, instrument_kind=kind)

    def value_on(
        d: datetime.date,
        *,
        record_terminal_source: bool = False,
    ) -> float:
        """Total EUR portfolio value on day ``d``.

        Values *every* ISIN that had a non-zero held quantity on ``d``,
        not only the ISINs still open today — otherwise a position opened
        and fully closed inside the window would contribute nothing to the
        historical series and its holding-period market move would be
        invisible to TWROR. The cum/ex ``open_isins`` gate is only used
        for the "what is open now" coverage snapshot, not for history.

        Explicitly equivalent variants that net to ~0 quantity as of ``d``
        are the one exception: they are one instrument reclassified across
        identifiers, so the group is treated as closed (contributes 0),
        consistent with the order-derived snapshot. Valuing each leg at its
        own carry-flat price would otherwise leave a spurious residual.
        """
        total = 0.0
        closed = closed_identity_groups(d)
        for isin in timeline.isins():
            qty = timeline.qty_at(isin, d)
            if abs(qty) < _QTY_EPS:
                continue
            if _identity_key(isin, identity_by_isin) in closed:
                continue  # explicitly equivalent group nets flat → closed
            price, source = resolver.price_on(isin, d)
            record_history_source(isin, source)
            if record_terminal_source:
                terminal_provenance.setdefault(source, []).append(isin)
            if price is None:
                continue
            # The enricher already applied /100 to bond price_history, so
            # the 'yfinance' rung is EUR-per-unit; the Borsa Italiana
            # today-price is likewise FX-converted and pre-/100. Raw
            # synthetic prices still need the /100 via value_position.
            if source in ("yfinance", "borsa_italiana"):
                total += qty * price
            else:
                kind = resolver.instrument_kind(isin)
                if kind is None:
                    continue
                total += value_position(qty, price, instrument_kind=kind)
        return total

    # TWROR external flow per date, valued at MARKET price (same basis as
    # the valuation series), not at execution price. For each
    # position-changing order we value its quantity delta at that day's
    # market price; this makes V_before(d) = V_after(d) - flow(d) use one
    # consistent price basis, so a trade does not inject a fictitious
    # jump from the gap between execution and market price (Option 2).
    # Round-trip positions (closed before today) are included: their
    # buys/sells are real external flows over their holding window.
    #
    # Income (coupons/dividends) is treated the GIPS total-return way.
    # Our series values only the securities, and the income cash is
    # credited to the bank account (net_eur > 0), so from the securities
    # portfolio's perspective the distribution is a *withdrawal*: a
    # negative external flow of -net_eur. Because TWROR computes
    # V_before(d) = V_after(d) - external_flow(d), that withdrawal is
    # added back into the pre-flow value, so the income is captured as
    # return rather than vanishing. (It is the mirror image of XIRR,
    # where the same distribution is a positive bank cash inflow.)
    external_flows: dict[datetime.date, float] = {}
    for o in orders:
        if o.is_position_change():
            unit = value_isin_on(o.isin, o.trade_date)
            if unit is None:
                continue
            external_flows[o.trade_date] = external_flows.get(o.trade_date, 0.0) + o.quantity * unit
        elif o.type in (OrderType.COUPON, OrderType.DIVIDEND) and o.net_eur:
            # Distribution paid out of the securities portfolio → withdrawal.
            external_flows[o.trade_date] = external_flows.get(o.trade_date, 0.0) - o.net_eur

    cf_dates = sorted(external_flows.keys())
    position_dates = sorted({
        order.trade_date for order in orders if order.is_position_change()
    })
    history_dates = sorted(set(cf_dates) | set(position_dates))
    valuations: list[tuple[datetime.date, float]] = [
        (d, value_on(d)) for d in cf_dates
    ]
    # Terminal valuation today, retaining a separate latest-value source set
    # for coverage while also contributing to whole-history provenance.
    current_value = value_on(today, record_terminal_source=True)
    valuations.append((today, current_value))

    # Dense daily value series for risk metrics (volatility, Sharpe, VaR,
    # beta). pct_change on the sparse trade-date `valuations` would treat
    # arbitrary multi-day gaps as single trading days and badly distort
    # annualized risk; the daily series fixes that at the root. The same
    # pass also yields the raw actual-value series (jumps kept in) for the
    # newsletter mountain chart.
    daily_series, actual_value_series = _build_daily_series(
        timeline, resolver, external_flows, history_dates, today,
        identity_by_isin=identity_by_isin,
        closed_identity_groups=closed_identity_groups,
        record_source=record_history_source,
    )

    # Trim the trailing carried-forward tail. The daily calendar runs to
    # ``today``, but a data vendor often has not yet served the most recent
    # close(s) — e.g. early next morning the previous European session is
    # still NaN at Yahoo while the US index already has it. On those days
    # the resolver carries the last real price forward, producing bit-
    # identical trailing NAV points. Measuring a "1 day" (or 7/30-day)
    # move against these stale duplicates would report a fake 0% change,
    # so we cut the series back to the last date that actually moved. The
    # terminal ``valuations``/``xirr_cashflows`` (which drive XIRR/TWROR)
    # are left untouched: their value equals the carried price anyway.
    daily_series = _trim_carried_tail(daily_series)
    actual_value_series = _trim_carried_tail(actual_value_series)

    # Whole-history evidence authority. A terminal quote cannot erase a gap
    # earlier in the held interval: missing causal evidence is unavailable,
    # while explicit order-price fallback remains usable but degraded.
    provenance = {key: sorted(set(values)) for key, values in provenance.items()}
    excluded = set(provenance.get("excluded", ()))
    causal_price_unavailable = tuple(sorted(
        excluded.difference(mechanics_unavailable)
    ))
    fallback_instruments = tuple(sorted(
        set(provenance.get("synthetic", ()))
        | set(provenance.get("carry_flat", ()))
    ))
    unavailable_instruments = tuple(sorted(
        set(mechanics_unavailable) | set(causal_price_unavailable)
    ))
    if unavailable_instruments:
        history_availability = Availability.UNAVAILABLE
    elif fallback_instruments:
        history_availability = Availability.DEGRADED
    else:
        history_availability = Availability.AVAILABLE

    if mechanics_unavailable:
        dq.error(
            "instrument_capability",
            "Order-derived return history is unavailable because exact "
            "instrument mechanics are missing or conflicting.",
            context=",".join(mechanics_unavailable),
        )
    if causal_price_unavailable:
        dq.error(
            "returns",
            "Order-derived return history is unavailable because at least "
            "one held date lacks causal price evidence.",
            context=",".join(causal_price_unavailable),
        )
    elif fallback_instruments:
        dq.record(
            dq.WARNING,
            "returns",
            "Order-derived return history is degraded because causal "
            "order-price fallback was required.",
            context=",".join(fallback_instruments),
        )

    # Coverage: share of today's value priced by real market data. Use the
    # SAME value_position basis as value_on so bonds (priced /100) are
    # Coverage: share of today's value priced by real market data. Borsa
    # Italiana is a real market quote (the best available for BTPs / US
    # Treasuries / foreign-currency notes that yfinance does not cover), so
    # it counts alongside yfinance. Both are EUR-per-unit (no value_position)
    # so the ratio cannot exceed 100%.
    real_value = 0.0
    real_isins = (
        set(terminal_provenance["yfinance"])
        | set(terminal_provenance["borsa_italiana"])
    )
    for isin in real_isins:
        qty = timeline.qty_at(isin, today)
        price, source = resolver.price_on(isin, today)
        if price is not None and source in ("yfinance", "borsa_italiana"):
            real_value += qty * price
    coverage_pct = (real_value / current_value * 100.0) if current_value > 0 else 0.0
    # Defensive clamp: coverage is a share of value priced by real market
    # data and is meaningless above 100%. Cum/ex netting keeps the
    # numerator and denominator consistent, but clamp anyway so a future
    # pricing edge case can never surface an impossible >100% figure.
    coverage_pct = max(0.0, min(coverage_pct, 100.0))

    # Disclosure: warn once per instrument that fell back.
    for source in ("synthetic", "carry_flat", "excluded"):
        for isin in sorted(set(provenance[source])):
            logger.warning(
                "TWROR/TWR: %s priced by %s (no full market history).",
                isin, source.upper(),
            )
            # 'excluded' means the instrument had NO price at all and dropped
            # out of the valuation on some dates — a heavier caveat than a
            # trend-only synthetic/carry-flat fill, so flag it as an ERROR.
            sev = dq.ERROR if source == "excluded" else dq.WARNING
            detail = {
                "synthetic": "priced by an exact ORDER observation (no real "
                             "market history) — its return contribution is approximate",
                "carry_flat": "priced CARRY-FLAT (single known price held flat, "
                              "zero volatility contribution) — its return contribution is approximate",
                "excluded": "had NO usable price and dropped out of the valuation "
                            "on some dates — its market move is invisible to TWROR",
            }[source]
            dq.record(sev, "returns", f"{isin}: {detail}", context=isin)
    if coverage_pct < 100.0:
        dq.info(
            "returns",
            f"historical value series is {coverage_pct:.1f}% priced by real "
            "market data; the remainder used the synthetic/carry-flat fallback ladder",
        )
    # XIRR cash flows (bank-account perspective): transfer_in is a
    # deposit at its market value; others use net_eur. Dated on the
    # trade date (when market exposure is taken on), consistent with
    # every other metric. Terminated by today's value.
    xirr_cashflows: list[tuple[datetime.date, float]] = []
    for o in orders:
        if o.type == OrderType.TRANSFER_IN:
            if (o.gross_eur or 0.0) > 0:
                xirr_cashflows.append((o.trade_date, -(o.gross_eur)))
        elif o.net_eur != 0.0:
            xirr_cashflows.append((o.trade_date, o.net_eur))
    xirr_cashflows.append((today, current_value))

    span_days = (today - history_dates[0]).days if history_dates else 0

    # Daily cumulative P&L (realized + unrealized), net of contributed
    # capital: actual value + cumulative bank cash flows (deposits negative,
    # distributions positive). At today it equals the lifetime pnl_eur. The
    # newsletter uses it to show the real money gained over a window, net of
    # the contributions made *inside* that window.
    pnl_series = _build_pnl_series(actual_value_series, xirr_cashflows)

    # Daily unrealized P&L = market value of open positions − their cost
    # basis on each day. Same average-cost logic as ``cost_basis_by_isin``
    # (so today's value reconciles with the hero's snapshot), expressed as a
    # full daily series for charting.
    cost_basis_series = _build_cost_basis_series(orders, actual_value_series.index)
    if actual_value_series is not None and not actual_value_series.empty:
        unrealized_series = actual_value_series - cost_basis_series
    else:
        unrealized_series = pd.Series(dtype=float)

    return OrderDerivedSeries(
        valuations=valuations,
        external_flows=external_flows,
        xirr_cashflows=xirr_cashflows,
        coverage_pct=coverage_pct,
        provenance=provenance,
        span_days=span_days,
        history_availability=history_availability,
        unavailable_instruments=unavailable_instruments,
        mechanics_unavailable_instruments=mechanics_unavailable,
        causal_price_unavailable_instruments=causal_price_unavailable,
        daily_series=daily_series,
        actual_value_series=actual_value_series,
        pnl_series=pnl_series,
        unrealized_series=unrealized_series,
    )


def _trim_carried_tail(s: pd.Series, max_trim: int = 10) -> pd.Series:
    """Drop a *short, recent* trailing run of bit-identical values.

    These appear when the price vendor has not yet published the latest
    close(s) and the resolver carries the last real price forward onto the
    calendar days up to ``today`` (weekends included). The carried points
    are exact duplicates of the last real value, so a strict-equality trim
    removes only stale filler — never a genuine move — leaving the series
    ending on the last date that actually changed.

    Guards keep this conservative: it only cuts a short tail (``max_trim``
    points, enough for a long weekend plus a stale session or two), always
    leaves at least two points, and never collapses a genuinely flat
    series (where the whole series is one constant value). Non-mutating;
    returns a (possibly shorter) view.
    """
    if s is None or len(s) < 3:
        return s
    last = s.iloc[-1]
    i = len(s) - 1
    while i > 0 and s.iloc[i - 1] == last:
        i -= 1
    trimmed = (len(s) - 1) - i
    if trimmed == 0 or trimmed > max_trim or i < 1:
        return s
    return s.iloc[: i + 1]


def _build_cost_basis_series(orders: list[Order], index: pd.Index) -> pd.Series:
    """Daily total cost basis of the OPEN positions, reindexed onto ``index``.

    Same average-cost walk as ``cost_basis_by_isin`` (see
    :func:`_apply_cost_order`), but it records the running total after each
    trade date so the result is a step function forward-filled across
    calendar days. Subtracting it from the daily market value yields the
    unrealized P&L series.
    """
    if index is None or len(index) == 0:
        return pd.Series(dtype=float)
    pos = sorted(
        (o for o in orders if o.is_position_change()),
        key=lambda o: o.trade_date,
    )
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    by_date: dict[pd.Timestamp, float] = {}
    for o in pos:
        _apply_cost_order(qty, cost, o)
        by_date[pd.Timestamp(o.trade_date)] = sum(cost.values())
    if not by_date:
        return pd.Series(0.0, index=index)
    s = pd.Series(by_date).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(index, method="ffill").fillna(0.0)


def _build_pnl_series(actual: pd.Series, xirr_cashflows: list) -> pd.Series:
    """Daily cumulative P&L = actual value + cumulative bank cash flows.

    The XIRR flows are deposits (negative) and distributions (positive);
    adding their running sum to the portfolio value cancels the invested
    capital and leaves the realized + unrealized gain on each day. At the
    final day it equals the lifetime ``pnl_eur``. The terminal valuation is
    excluded (it is not a real cash flow), so the series is purely
    value + contributions accounting.
    """
    if actual is None or actual.empty or not xirr_cashflows:
        return pd.Series(dtype=float)
    agg: dict = {}
    for d, amt in xirr_cashflows[:-1]:  # drop the terminal valuation
        ts = pd.Timestamp(d)
        agg[ts] = agg.get(ts, 0.0) + amt
    if not agg:
        return pd.Series(dtype=float)
    cum = pd.Series(agg).sort_index().cumsum()
    cum = cum.reindex(actual.index, method="ffill").fillna(0.0)
    return actual + cum


def _build_daily_series(
    timeline: "QuantityTimeline",
    resolver: "PriceResolver",
    external_flows: dict[datetime.date, float],
    history_dates: list[datetime.date],
    today: datetime.date,
    identity_by_isin: Optional[dict[str, _IdentityKey]] = None,
    closed_identity_groups=None,
    record_source=None,
) -> tuple[pd.Series, pd.Series]:
    """Dense daily NAV index + raw actual-value series, first trade → today.

    Returns ``(nav_index, actual_value_series)``:

      * ``nav_index`` is flow-adjusted. Each day's raw market value is
        computed through the resolver from the held quantity that day; the
        day-over-day return strips that day's external flow (deposits/
        withdrawals valued at market), so buying more units does not
        register as a market gain::

            r_t = (V_t - flow_t) / V_{t-1} - 1

        The returns are chained into an index anchored at the first day's
        real value. Risk metrics (volatility, Sharpe, Sortino, VaR, beta)
        must use THIS series.

      * ``actual_value_series`` is the same daily raw value with the
        deposit/withdrawal jumps left in — the real euro worth of the
        patrimony over time, for the mountain chart.

    Both are anchored on the first strictly-positive value so leading
    zero-value days (before the first priced position) are dropped.
    """
    if not history_dates:
        empty = pd.Series(dtype=float)
        return empty, empty

    days = pd.date_range(start=history_dates[0], end=today, freq="D")
    isins = timeline.isins()
    # Reuse the caller's date-memoized identity-closure cache when provided
    # (so the set is computed once per date across value_on and this loop);
    # fall back to direct explicit-identity resolution for standalone use.
    _closed = closed_identity_groups or (
        lambda d: _closed_identity_groups(timeline, d, identity_by_isin)
    )

    def raw_value(d: datetime.date) -> float:
        total = 0.0
        closed = _closed(d)
        for isin in isins:
            qty = timeline.qty_at(isin, d)
            if abs(qty) < _QTY_EPS:
                continue
            if _identity_key(isin, identity_by_isin) in closed:
                continue  # explicitly equivalent group nets flat → closed
            price, source = resolver.price_on(isin, d)
            if record_source is not None:
                record_source(isin, source)
            if price is None:
                continue
            if source in ("yfinance", "borsa_italiana"):
                total += qty * price
            else:
                kind = resolver.instrument_kind(isin)
                if kind is None:
                    continue
                total += value_position(qty, price, instrument_kind=kind)
        return total

    raw = [(ts.date(), raw_value(ts.date())) for ts in days]
    # Anchor on the first strictly-positive value.
    anchor_i = next((i for i, (_, v) in enumerate(raw) if v > 0), None)
    if anchor_i is None:
        empty = pd.Series(dtype=float)
        return empty, empty

    index_dates: list[pd.Timestamp] = []
    index_vals: list[float] = []
    actual_vals: list[float] = []
    nav = raw[anchor_i][1]            # start the index at the real value
    index_dates.append(pd.Timestamp(raw[anchor_i][0]))
    index_vals.append(nav)
    actual_vals.append(raw[anchor_i][1])

    prev_v = raw[anchor_i][1]
    for d, v in raw[anchor_i + 1:]:
        flow = external_flows.get(d, 0.0)
        # Only advance the flow-adjusted NAV on a day with a real positive
        # value. A day where every held ISIN prices to 0/None (a transient
        # pricing gap) or the book is briefly fully liquidated to cash would
        # otherwise give r=(0-0)/prev_v-1=-1, and since prev_v is only
        # refreshed when v>0, nav*=(1+r)=0 would pin the index at zero for the
        # ENTIRE rest of the window — fabricating a permanent -100% that
        # poisons volatility/Sharpe/VaR/drawdown/CAGR. Carry the index flat
        # across such days instead (they contribute a 0% return, not -100%).
        if prev_v > 0 and v > 0:
            r = (v - flow) / prev_v - 1.0
            nav *= (1.0 + r)
        index_dates.append(pd.Timestamp(d))
        index_vals.append(nav)
        actual_vals.append(v)
        if v > 0:
            prev_v = v

    idx = pd.DatetimeIndex(index_dates)
    return (
        pd.Series(index_vals, index=idx),
        pd.Series(actual_vals, index=idx),
    )


# ---------------------------------------------------------------------------
# Allocation timeline (per asset-class / per equity-geography weekly weights)
# ---------------------------------------------------------------------------

# Price-source preference when collapsing a cum/ex group to a single
# representative quote: real market quotes first, then the synthetic
# ladder. Mirrors the ranking used when valuing the portfolio history.
_TIMELINE_SOURCE_RANK = {
    "yfinance": 0, "borsa_italiana": 0, "synthetic": 1, "carry_flat": 2,
}


def build_allocation_timeline(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    *,
    months: int = 3,
    today: Optional[datetime.date] = None,
) -> Optional[dict]:
    """Reconstruct the historical allocation mix over a recent window.

    Returns weekly snapshots of the invested asset-class mix and the
    equity-geography mix over the last ``months`` months (clamped to the
    portfolio inception), so the newsletter can draw a per-category
    sparkline of how each weight drifted toward/away from its target.

    The reconstruction reuses the same primitives as the value series —
    ``QuantityTimeline`` for as-of held quantity, ``PriceResolver`` for the
    EUR price ladder, and explicit equivalence evidence so a documented BTP
    rotation nets to zero rather than lingering as a phantom leg. Asset class
    and equity geo come from the already-enriched holdings (constant per
    instrument), so this adds no network calls.

    Output (or ``None`` when there is no order history):
        ``{"dates": [date, ...],
            "asset":   [{class: pct_of_invested}, ...],
            "geo":     [{region: pct_of_equity}, ...],
            "holding": [{isin: pct_of_its_class}, ...]}``
    The lists are parallel to ``dates``; the caller typically anchors the
    final bucket to the authoritative live allocation.
    """
    if not orders:
        return None
    if today is None:
        from tarzan import runtime
        today = runtime.today()
    orders = [order for order in orders if order.trade_date <= today]
    enriched_by_isin = _causal_enriched_by_isin(orders, enriched_by_isin)

    pos_dates = [o.trade_date for o in orders if o.is_position_change()]
    if not pos_dates:
        return None
    inception = min(pos_dates)

    # Weekly (W-FRI) buckets across the window, clamped to inception for a
    # portfolio younger than the window, and always terminated by today.
    window_start = (pd.Timestamp(today) - pd.DateOffset(months=months)).date()
    start = max(window_start, inception)
    pts = list(pd.date_range(start=start, end=today, freq="W-FRI"))
    dates = [p.date() for p in pts]
    if not dates or dates[0] > start:
        dates.insert(0, start)
    if dates[-1] != today:
        dates.append(today)

    identity_by_isin = _instrument_identity_by_isin(orders)
    timeline = QuantityTimeline(orders)
    resolver = PriceResolver(orders, enriched_by_isin, today=today)
    unresolved_mechanics = sorted(
        isin for isin in timeline.isins()
        if resolver.instrument_kind(isin) is None
    )
    if unresolved_mechanics:
        dq.error(
            "instrument_capability",
            "Historical allocation is unavailable because exact instrument "
            "mechanics are missing or conflicting.",
            context=",".join(unresolved_mechanics),
        )
        return None
    cash_class = AssetClass.CASH_EQUIVALENTS.value

    # Asset-class resolution for every traded ISIN, including positions no
    # longer in the enriched current snapshot. Enriched/taxonomy category
    # declarations remain authoritative; intrinsic categories may be derived
    # only from an exact instrument kind. ETF category is never inferred from
    # names, prices, quantities, or bond-like wording.
    from tarzan import config as _cfg
    _taxonomy = _cfg.instrument_taxonomy()
    _exp_lut = _cfg.class_exposure_lookup()

    def _class_for(isin: str) -> Optional[str]:
        holding = enriched_by_isin.get(isin)
        if holding and holding.asset_class:
            return holding.asset_class.value
        hit = _taxonomy.get(isin.upper()) or _taxonomy.get(isin.upper().split(".")[0])
        if hit and hit[0]:
            return hit[0]
        intrinsic = {
            InstrumentKind.STOCK: AssetClass.EQUITIES.value,
            InstrumentKind.BOND: AssetClass.FIXED_INCOME.value,
            InstrumentKind.CASH: AssetClass.CASH_EQUIVALENTS.value,
        }
        return intrinsic.get(resolver.instrument_kind(isin))

    def _breakdown_for(isin: str) -> dict[str, float]:
        """Notional asset-class exposure {class: pct} for an ISIN — the same
        basis as the live snapshot's ``class_breakdown``. Explicit exp_*
        override (enriched holding, else taxonomy by ISIN/bare ticker), else
        100% of an exact intrinsic category. An ETF without explicit tracked
        category evidence remains unavailable instead of becoming equities,
        alternative, or an ``Other`` allocation."""
        h = enriched_by_isin.get(isin)
        if h is not None and getattr(h, "class_breakdown", None):
            return {(k.value if hasattr(k, "value") else str(k)): v
                    for k, v in h.class_breakdown.items()}
        ov = _exp_lut.get(isin.upper()) or _exp_lut.get(isin.upper().split(".")[0])
        if ov:
            return dict(ov)
        category = _class_for(isin)
        return {category: 100.0} if category is not None else {}

    # Identity groups are used only to (a) suppress a documented equivalent
    # rotation whose held quantity nets flat and (b) let a priced equivalent
    # identifier stand in for an unpriced sibling. Each exact ISIN is still
    # valued and attributed individually. Without an explicit group, the full
    # normalized ISIN forms a singleton, regardless of identifier shape.
    groups: dict[_IdentityKey, list[str]] = {}
    for isin in timeline.isins():
        groups.setdefault(_identity_key(isin, identity_by_isin), []).append(isin)

    def _price_for(isin: str, members: list[str],
                   d: datetime.date) -> tuple[Optional[float], str]:
        """(price, source) for an ISIN, borrowing the best-priced sibling in
        its explicit equivalence group when the ISIN has no quote on ``d``."""
        price, source = resolver.price_on(isin, d)
        if price is not None:
            return price, source
        best = None
        best_rank = 99
        for s in members:
            p, src = resolver.price_on(s, d)
            if p is None:
                continue
            rank = _TIMELINE_SOURCE_RANK.get(src, 3)
            if rank < best_rank:
                best_rank, best = rank, (p, src)
        return best if best is not None else (None, "excluded")

    unpriceable_instruments: set[str] = set()

    def _value_isin(
        isin: str,
        members: list[str],
        qty: float,
        d: datetime.date,
    ) -> Optional[float]:
        """EUR value of a single ISIN's held quantity on ``d``.

        Mirrors the source-tag handling of the prices: quotes from
        yfinance/Borsa are already EUR-per-unit (and bond-rescaled), while
        synthetic/carry-flat order prices still need ``value_position`` to
        apply the bond /100. ``None`` means the required snapshot cannot be
        valued and therefore the complete allocation timeline is unavailable.
        """
        price, source = _price_for(isin, members, d)
        if price is None:
            return None
        if source in ("yfinance", "borsa_italiana"):
            return qty * price
        kind = resolver.instrument_kind(isin)
        if kind is None:
            return None
        return value_position(qty, price, instrument_kind=kind)

    def _per_isin_values(d: datetime.date) -> dict[str, float]:
        """Return complete per-ISIN EUR values for one allocation snapshot.

        An explicit identity group whose held quantity nets to approximately
        zero contributes nothing; ungrouped ISINs remain singleton groups.
        Missing evidence is retained separately so callers cannot silently
        renormalize the priced subset to a valid-looking 100% allocation.
        """
        out: dict[str, float] = {}
        for members in groups.values():
            if abs(sum(timeline.qty_at(i, d) for i in members)) < _QTY_EPS:
                continue
            for isin in members:
                q = timeline.qty_at(isin, d)
                if abs(q) < _QTY_EPS:
                    continue
                v = _value_isin(isin, members, q, d)
                if v is None:
                    unpriceable_instruments.add(isin)
                elif v > 0:
                    out[isin] = out.get(isin, 0.0) + v
        return out

    asset_series: list[dict[str, float]] = []
    geo_series: list[dict[str, float]] = []
    classification_unavailable = False
    # Per-holding weight as % of its own asset class (parallel to dates),
    # keyed by ISIN and derived from the SAME per-ISIN valuation as the
    # aggregates, so the asset/geo trend lines and the by-holding trends all
    # agree (and distinct prefix-sharing ETFs each keep their own line).
    holding_series: list[dict[str, float]] = []
    # Parallel series expressing each holding as % of the INVESTED portfolio
    # (not of its class) — the basis the per-holding-portfolio target table
    # uses, so its "Now" column and this trend line share one scale.
    holding_invested_series: list[dict[str, float]] = []
    eq_class = AssetClass.EQUITIES.value
    for d in dates:
        iso_val = _per_isin_values(d)
        iso_ac: dict[str, str] = {}          # primary (dominant) class per ISIN
        iso_eq: dict[str, float] = {}        # equity notional € per ISIN
        class_val: dict[str, float] = {}     # NOTIONAL € per class
        geo_val: dict[str, float] = {}
        # Invested CAPITAL (denominator): the actual money in non-cash
        # positions — NOT the sum of notional class exposure (which can exceed
        # it), so leveraged/efficient-core funds make the class weights sum to
        # >100% rather than being normalised away.
        invested = 0.0
        class_pairs: list = []           # (value, class breakdown minus cash)
        geo_pairs: list = []             # (equity €, per-holding normalised geo)
        for isin, v in iso_val.items():
            bd = _breakdown_for(isin)
            if not bd:
                classification_unavailable = True
                continue
            prim = max(bd, key=bd.get)
            iso_ac[isin] = prim
            if prim == cash_class:
                continue
            invested += v
            class_pairs.append((v, {cls: pct for cls, pct in bd.items()
                                    if cls != cash_class}))
            eq_contrib = v * (float(bd.get(eq_class, 0.0)) / 100.0)
            iso_eq[isin] = eq_contrib
            if eq_contrib > 0:
                h = enriched_by_isin.get(isin)
                if h and h.geo_breakdown:
                    geo_pairs.append((eq_contrib, _alloc.renorm(
                        {(g.value if hasattr(g, "value") else str(g)): p
                         for g, p in h.geo_breakdown.items()})))
        # Shared aggregation primitive (same as the live snapshot & backtest).
        class_val = _alloc.accumulate(class_pairs)
        geo_val = _alloc.accumulate(geo_pairs)
        eq_val = sum(iso_eq.values())
        asset_series.append({
            k: (val / invested * 100.0)
            for k, val in class_val.items()
            if k != cash_class and invested > 0
        })
        geo_series.append({
            k: (val / eq_val * 100.0) for k, val in geo_val.items() if eq_val > 0
        })
        # Per-holding weight as % of its dominant class's notional (kept for
        # the equity/FI sleeve tables); and as % of invested capital.
        holding_series.append({
            isin: (v / class_val[iso_ac[isin]] * 100.0)
            for isin, v in iso_val.items()
            if isin in iso_ac and class_val.get(iso_ac[isin], 0.0) > 0
        })
        holding_invested_series.append({
            isin: (v / invested * 100.0)
            for isin, v in iso_val.items()
            if invested > 0 and isin in iso_ac and iso_ac[isin] != cash_class
        })

    if unpriceable_instruments:
        dq.error(
            "returns",
            "Historical allocation is unavailable because at least one held "
            "snapshot lacks causal price evidence.",
            context=",".join(sorted(unpriceable_instruments)),
        )
        return None

    if classification_unavailable:
        dq.error(
            "instrument_capability",
            "Historical allocation is unavailable because at least one valued "
            "instrument lacks an exact tracked category.",
            context="allocation_timeline",
        )
        return None

    return {"dates": dates, "asset": asset_series, "geo": geo_series,
            "holding": holding_series,
            "holding_invested": holding_invested_series}
