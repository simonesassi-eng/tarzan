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

The fallback ladder for a missing yfinance history (the user's choice
"(a) synthetic interpolation, made explicit when it happens"):

    1. yfinance     real daily series                       → "yfinance"
    2. synthetic    linear interpolation between order       → "synthetic"
                    price_native observations (≥2 points)
    3. carry_flat   hold the single known price flat          → "carry_flat"
    4. excluded     no price at all → drops out of valuation  → "excluded"
"""

from __future__ import annotations

import bisect
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from tarzan.data.bond_fetcher import (
    is_bond,
    looks_like_bond_from_orders,
    value_position,
)
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType

logger = logging.getLogger(__name__)

# Two ISINs whose first 9 characters match are treated as cum/ex variants
# of the same security (Italian retail BTPs). When their net quantities
# cancel, the pair is closed and contributes nothing.
_CUM_EX_PREFIX_LEN = 9
_QTY_EPS = 0.01


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
    """

    valuations: list[tuple[datetime.date, float]]
    external_flows: dict[datetime.date, float]
    xirr_cashflows: list[tuple[datetime.date, float]]
    coverage_pct: float
    provenance: dict[str, list[str]]
    span_days: int


# ---------------------------------------------------------------------------
# Holdings derivation (net quantity per ISIN + cum/ex netting)
# ---------------------------------------------------------------------------

def _net_qty_by_isin(orders: list[Order]) -> dict[str, float]:
    qty: dict[str, float] = {}
    for o in orders:
        if o.is_position_change():
            qty[o.isin] = qty.get(o.isin, 0.0) + o.quantity
    return qty


def _open_isins(qty_by_isin: dict[str, float]) -> set[str]:
    """Return the ISINs that represent open positions, applying cum/ex
    prefix netting: when all variants sharing a 9-char prefix net to
    zero, the whole group is closed (Property 5)."""
    prefix_totals: dict[str, float] = {}
    for isin, q in qty_by_isin.items():
        prefix_totals[isin[:_CUM_EX_PREFIX_LEN]] = (
            prefix_totals.get(isin[:_CUM_EX_PREFIX_LEN], 0.0) + q
        )
    open_isins: set[str] = set()
    for isin, q in qty_by_isin.items():
        prefix = isin[:_CUM_EX_PREFIX_LEN]
        if abs(prefix_totals[prefix]) < _QTY_EPS:
            continue  # cum/ex pair fully nets out → closed
        if abs(q) < _QTY_EPS:
            continue  # individually closed
        open_isins.add(isin)
    return open_isins


def _order_bond_flag(orders: list[Order], isin: str) -> bool:
    """Bond detection from order data alone (for ISINs that never reach
    yfinance), via the shared order heuristic."""
    sub = [o for o in orders if o.isin == isin and o.price_native is not None]
    if not sub:
        return False
    avg_price = sum(o.price_native for o in sub) / len(sub)
    avg_qty = sum(abs(o.quantity) for o in sub) / len(sub) if sub else 0.0
    return looks_like_bond_from_orders(avg_price, avg_qty)


def _seed_market_value(orders: list[Order], isin: str, qty: float) -> float:
    """Rough current-value seed from the latest order price, scaled by
    the bond convention via ``value_position``. Replaced by a real quote
    during enrichment whenever one is available; only a floor so an
    unpriceable holding does not collapse to zero (Req 8.3)."""
    priced = sorted(
        (o for o in orders if o.isin == isin and o.price_native is not None),
        key=lambda o: o.date,
    )
    if not priced or qty == 0:
        return 0.0
    last = priced[-1]
    fx = last.fx_rate or 1.0
    bond = _order_bond_flag(orders, isin)
    return value_position(abs(qty), last.price_native, bond=bond) / (fx if fx > 0 else 1.0)


def build_holdings_from_orders(orders: list[Order]) -> list[Holding]:
    """Aggregate orders into synthetic Holdings for the open positions.

    Net quantity per ISIN, cum/ex prefix netting, and a seeded
    market value. The returned Holdings carry only what the enricher
    needs; enrichment fills in price history, asset class, etc.
    """
    qty_by_isin = _net_qty_by_isin(orders)
    open_isins = _open_isins(qty_by_isin)

    name_by_isin: dict[str, str] = {}
    ccy_by_isin: dict[str, str] = {}
    for o in orders:
        if o.name and o.isin not in name_by_isin:
            name_by_isin[o.isin] = o.name
        if o.currency and o.isin not in ccy_by_isin:
            ccy_by_isin[o.isin] = o.currency

    holdings: list[Holding] = []
    for isin in open_isins:
        qty = qty_by_isin[isin]
        holdings.append(Holding(
            isin=isin,
            ticker=isin,
            quantity=qty,
            cost_basis_eur=0.0,
            market_value_eur=_seed_market_value(orders, isin, qty),
            currency=ccy_by_isin.get(isin, "EUR"),
            name=name_by_isin.get(isin, ""),
        ))
    return holdings


# ---------------------------------------------------------------------------
# Quantity timeline (binary-searchable cumulative quantity per ISIN)
# ---------------------------------------------------------------------------

class QuantityTimeline:
    """Cumulative held quantity per ISIN as of end-of-day, with O(log n)
    lookup. Built once from the position-changing orders."""

    def __init__(self, orders: list[Order]):
        events: list[tuple[datetime.date, str, float]] = [
            (o.date, o.isin, o.quantity)
            for o in orders if o.is_position_change()
        ]
        events.sort(key=lambda e: e[0])
        self._cum: dict[str, list[tuple[datetime.date, float]]] = {}
        running: dict[str, float] = {}
        for d, isin, delta in events:
            running[isin] = running.get(isin, 0.0) + delta
            self._cum.setdefault(isin, []).append((d, running[isin]))

    def isins(self) -> list[str]:
        return list(self._cum.keys())

    def qty_at(self, isin: str, d: datetime.date) -> float:
        series = self._cum.get(isin)
        if not series or d < series[0][0]:
            return 0.0
        dates = [e[0] for e in series]
        # rightmost index whose date <= d
        i = bisect.bisect_right(dates, d) - 1
        return series[i][1] if i >= 0 else 0.0


# ---------------------------------------------------------------------------
# Price lookup with explicit fallback ladder
# ---------------------------------------------------------------------------

def _price_at(price_history: pd.Series, d: datetime.date) -> Optional[float]:
    """Last observed price at or before ``d`` in a tz-aware-safe way."""
    if price_history is None or len(price_history) == 0:
        return None
    idx_tz = getattr(price_history.index, "tz", None)
    threshold = pd.Timestamp(d)
    if idx_tz is not None:
        threshold = threshold.tz_localize(idx_tz)
    avail = price_history.loc[price_history.index <= threshold]
    if avail.empty:
        return None
    return float(avail.iloc[-1])


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
        if o.isin == isin and o.price_native is not None:
            fx = o.fx_rate or 1.0
            eur_price = o.price_native / fx if fx > 0 else o.price_native
            obs.append((o.date, eur_price))
    if not obs:
        return None
    s = pd.Series(
        [p for _, p in obs],
        index=pd.to_datetime([d for d, _ in obs]),
    ).sort_index()
    return s.groupby(s.index).mean()


def _interp_synthetic(s: pd.Series, d: datetime.date) -> tuple[Optional[float], str]:
    """Linear interpolation between order price points, or carry-flat
    when only one point / outside the range. Returns (price, source)."""
    ts = pd.Timestamp(d)
    if len(s) == 1:
        return float(s.iloc[0]), "carry_flat"
    if ts <= s.index[0]:
        return float(s.iloc[0]), "carry_flat"
    if ts >= s.index[-1]:
        return float(s.iloc[-1]), "carry_flat"
    before = s.loc[s.index <= ts]
    after = s.loc[s.index >= ts]
    bd, bv = before.index[-1], float(before.iloc[-1])
    ad, av = after.index[0], float(after.iloc[0])
    if ad == bd:
        return bv, "synthetic"
    w = (ts - bd).days / (ad - bd).days
    return bv + w * (av - bv), "synthetic"


class PriceResolver:
    """Resolve an ISIN's price on any date via the explicit fallback
    ladder, recording the source tag used per ISIN for provenance."""

    def __init__(self, orders: list[Order], enriched_by_isin: dict[str, Holding]):
        self._orders = orders
        self._enriched = enriched_by_isin
        self._synth: dict[str, Optional[pd.Series]] = {}
        self._is_bond: dict[str, bool] = {}

    def _synthetic(self, isin: str) -> Optional[pd.Series]:
        if isin not in self._synth:
            self._synth[isin] = _build_synthetic_history(self._orders, isin)
        return self._synth[isin]

    def is_bond(self, isin: str) -> bool:
        if isin not in self._is_bond:
            h = self._enriched.get(isin)
            flag = is_bond(
                asset_class=getattr(h, "asset_class", None) if h else None,
                instrument_type=getattr(h, "instrument_type", None) if h else None,
            )
            if not flag:
                flag = _order_bond_flag(self._orders, isin)
            self._is_bond[isin] = flag
        return self._is_bond[isin]

    def price_on(self, isin: str, d: datetime.date) -> tuple[Optional[float], str]:
        """Return (price_eur_per_unit, source) for an ISIN on a date.

        Note: prices from the enricher are already EUR-per-unit and, for
        bonds, already rescaled by /100 — so the caller must NOT apply
        the bond /100 again on the 'yfinance' rung. Synthetic/carry_flat
        prices are raw order prices that still need the /100 via
        value_position. The returned source disambiguates which.
        """
        h = self._enriched.get(isin)
        ph = getattr(h, "price_history", None) if h else None
        if ph is not None and len(ph) > 0:
            price = _price_at(ph, d)
            if price is not None:
                return price, "yfinance"
        s = self._synthetic(isin)
        if s is not None and not s.empty:
            return _interp_synthetic(s, d)
        return None, "excluded"


# ---------------------------------------------------------------------------
# Main: build the dated value series + cash flows + provenance
# ---------------------------------------------------------------------------

def build_order_derived_series(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    today: Optional[datetime.date] = None,
) -> OrderDerivedSeries:
    """Build the order-derived valuation series, cash flows and
    provenance. ``enriched_by_isin`` maps ISIN → enriched Holding (from
    running the standard enrichment on ``build_holdings_from_orders``)."""
    today = today or datetime.datetime.now().date()
    timeline = QuantityTimeline(orders)
    resolver = PriceResolver(orders, enriched_by_isin)
    open_isins = _open_isins(_net_qty_by_isin(orders))

    # Track which source priced each open ISIN at its latest valuation,
    # for the coverage/provenance disclosure.
    provenance: dict[str, list[str]] = {
        "yfinance": [], "synthetic": [], "carry_flat": [], "excluded": [],
    }

    def value_isin_on(isin: str, d: datetime.date) -> Optional[float]:
        """EUR value of one unit of ``isin`` on ``d`` at market price
        (None if unpriceable). Used to value quantity deltas for the
        TWROR external flow at the same price basis as the series."""
        price, source = resolver.price_on(isin, d)
        if price is None:
            return None
        if source == "yfinance":
            return price
        return value_position(1.0, price, bond=resolver.is_bond(isin))

    def value_on(d: datetime.date, record_source: bool = False) -> float:
        total = 0.0
        for isin in timeline.isins():
            if isin not in open_isins:
                continue
            qty = timeline.qty_at(isin, d)
            if abs(qty) < _QTY_EPS:
                continue
            price, source = resolver.price_on(isin, d)
            if record_source:
                provenance[source].append(isin)
            if price is None:
                continue
            # The enricher already applied /100 to bond price_history, so
            # the 'yfinance' rung is EUR-per-unit; raw synthetic prices
            # still need it via value_position.
            if source == "yfinance":
                total += qty * price
            else:
                total += value_position(qty, price, bond=resolver.is_bond(isin))
        return total

    # TWROR external flow per date, valued at MARKET price (same basis as
    # the valuation series), not at execution price. For each
    # position-changing order we value its quantity delta at that day's
    # market price; this makes V_before(d) = V_after(d) - flow(d) use one
    # consistent price basis, so a trade does not inject a fictitious
    # jump from the gap between execution and market price (Option 2).
    # Coupons/dividends are not position changes and are NOT external
    # flows for TWROR — they are income earned by the held portfolio, so
    # they belong inside the market return, not subtracted from it.
    external_flows: dict[datetime.date, float] = {}
    for o in orders:
        if not o.is_position_change():
            continue
        if o.isin not in open_isins:
            continue
        unit = value_isin_on(o.isin, o.date)
        if unit is None:
            continue
        external_flows[o.date] = external_flows.get(o.date, 0.0) + o.quantity * unit

    cf_dates = sorted(external_flows.keys())
    valuations: list[tuple[datetime.date, float]] = [
        (d, value_on(d)) for d in cf_dates
    ]
    # Terminal valuation today, recording provenance for coverage.
    current_value = value_on(today, record_source=True)
    valuations.append((today, current_value))

    # Coverage: share of today's value priced by real market data.
    yf_isins = set(provenance["yfinance"])
    real_value = 0.0
    for isin in yf_isins:
        qty = timeline.qty_at(isin, today)
        price, source = resolver.price_on(isin, today)
        if price is not None and source == "yfinance":
            real_value += qty * price
    coverage_pct = (real_value / current_value * 100.0) if current_value > 0 else 0.0

    # Disclosure: warn once per instrument that fell back.
    for source in ("synthetic", "carry_flat", "excluded"):
        for isin in sorted(set(provenance[source])):
            logger.warning(
                "TWROR/TWR: %s priced by %s (no full market history).",
                isin, source.upper(),
            )
    # Deduplicate provenance lists.
    provenance = {k: sorted(set(v)) for k, v in provenance.items()}

    # XIRR cash flows (bank-account perspective): transfer_in is a
    # deposit at its market value; others use net_eur. Terminated by
    # today's value.
    xirr_cashflows: list[tuple[datetime.date, float]] = []
    for o in orders:
        if o.type == OrderType.TRANSFER_IN:
            if (o.gross_eur or 0.0) > 0:
                xirr_cashflows.append((o.date, -(o.gross_eur)))
        elif o.net_eur != 0.0:
            xirr_cashflows.append((o.date, o.net_eur))
    xirr_cashflows.append((today, current_value))

    span_days = (today - cf_dates[0]).days if cf_dates else 0

    return OrderDerivedSeries(
        valuations=valuations,
        external_flows=external_flows,
        xirr_cashflows=xirr_cashflows,
        coverage_pct=coverage_pct,
        provenance=provenance,
        span_days=span_days,
    )
