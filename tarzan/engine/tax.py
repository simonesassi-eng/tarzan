"""Estimate Italian capital-gains tax (CGT) on realized gains.

This estimate presents net-of-tax money-weighted figures alongside the gross
ones; it is not a tax return. Realizations use running average cost, sells are
taxable while transfers are not, and losses offset only eligible capital gains.
Exact instrument-kind evidence selects tax mechanics. Full ISIN is the default
cost-basis identity, and cross-ISIN pooling requires an explicit equivalence
group on the source orders.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Optional

from tarzan.instruments.registry import (
    InstrumentKind,
    TypeEvidenceGateway,
    TypeResolutionState,
)
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType

# Fineco short-name / description markers for state-issued bonds that get the
# reduced rate. Names may distinguish sovereign from corporate debt only after
# exact evidence has resolved the instrument mechanics to BOND.
_GOV_BOND_NAME = re.compile(
    r"\b(BTP|BOT|CCT|CTZ|BUND|OAT|BONOS?|GILT|TREASUR|T-?NOTE|T-?BOND)\b"
    r"|^USA-",
    re.IGNORECASE,
)


def is_government_bond(subtypes=(), names=()) -> bool:
    """Sovereign-debt test for the reduced CGT rate, from provider subtype
    markers OR a sovereign short-name/description match.

    The single home for "is this a government bond?" so the realized-CGT
    estimate (``tax``) and the rebalancer's plan-cost tax model apply the same
    reduced rate to the same instruments. The caller must only pass instruments
    already established as BOND mechanics: the name regex distinguishes
    sovereign from corporate debt, it does NOT prove an instrument is a bond
    (an equity named "Treasury Wine" must never reach here).
    """
    if any("govern" in str(s).casefold() or "govt" in str(s).casefold()
           for s in subtypes):
        return True
    return any(_GOV_BOND_NAME.search(str(n)) for n in names if n)

# How many years a realized loss stays usable (year of realization + 4).
_LOSS_CARRY_YEARS = 4


class TaxEvidenceUnavailable(ValueError):
    """Raised when exact evidence cannot support an authoritative tax estimate."""


@dataclass
class CgtEstimate:
    """Estimated CGT on realized gains and the cash flows it implies."""

    tax_flows: list[tuple[datetime.date, float]] = field(default_factory=list)
    total_tax_eur: float = 0.0
    total_realized_gain_eur: float = 0.0
    total_realized_loss_eur: float = 0.0
    taxable_base_eur: float = 0.0


@dataclass
class _Realization:
    date: datetime.date
    isin: str
    pnl: float
    rate: float
    is_capital_income: bool


_IdentityKey = tuple[str, str]


def _cost_basis_key(order: Order) -> _IdentityKey:
    """Return explicit equivalence identity or the complete normalized ISIN."""
    group = str(order.instrument_equivalence_group or "").strip()
    if group:
        return "equivalence", group.casefold()
    return "isin", str(order.isin or "").strip().upper()


def _identity_orders(orders: list[Order], key: _IdentityKey) -> list[Order]:
    return [order for order in orders if _cost_basis_key(order) == key]


def _classify(
    order: Order,
    enriched_by_isin: dict[str, Holding],
    orders: list[Order],
    identity_key: _IdentityKey,
) -> tuple[bool, bool]:
    """Return ``(is_government_bond, is_capital_income)`` from exact evidence.

    Order and holding declarations for the same explicit identity are resolved
    together. Unknown, ambiguous, or unsupported kinds make the estimate
    unavailable instead of selecting financial behavior from names or prices.
    Once BOND mechanics are established, provider subtype and issuer-name
    markers may distinguish sovereign debt for the reduced rate.
    """
    related_orders = _identity_orders(orders, identity_key)
    related_isins = {related.isin for related in related_orders}
    holdings = [
        enriched_by_isin[isin]
        for isin in related_isins
        if isin in enriched_by_isin
    ]

    assertions: list[str] = [
        related.instrument_kind.value
        for related in related_orders
        if related.instrument_kind is not None
    ]
    for holding in holdings:
        assertions.extend(
            value
            for value in (
                holding.security_type,
                holding.instrument_type,
                *tuple(holding.instrument_kind_evidence or ()),
            )
            if value
        )

    resolution = TypeEvidenceGateway().resolve(*assertions)
    if resolution.state is not TypeResolutionState.RESOLVED:
        raise TaxEvidenceUnavailable(
            f"tax classification unavailable for {order.isin}: "
            f"instrument kind is {resolution.state.value.lower()} "
            f"from evidence {resolution.evidence!r}"
        )

    if resolution.kind is InstrumentKind.ETF:
        return False, True
    if resolution.kind is InstrumentKind.STOCK:
        return False, False
    if resolution.kind is not InstrumentKind.BOND:
        raise TaxEvidenceUnavailable(
            f"tax classification unavailable for {order.isin}: "
            f"unsupported instrument kind {resolution.kind.value}"
        )

    provider_subtypes = [
        str(value)
        for holding in holdings
        for value in (holding.instrument_type, holding.security_type)
        if value
    ]
    names = [related.name for related in related_orders if related.name]
    return is_government_bond(provider_subtypes, names), False


def _realizations(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    std_rate: float,
    gov_rate: float,
) -> list[_Realization]:
    """Emit sell realizations using average cost per authoritative identity."""
    position_orders = sorted(
        (order for order in orders if order.is_position_change()),
        key=lambda order: order.trade_date,
    )
    quantity: dict[_IdentityKey, float] = {}
    cost: dict[_IdentityKey, float] = {}
    basis_complete: dict[_IdentityKey, bool] = {}
    realizations: list[_Realization] = []

    for order in position_orders:
        key = _cost_basis_key(order)
        held_quantity = quantity.get(key, 0.0)
        held_cost = cost.get(key, 0.0)
        if order.quantity > 0:
            # BUY net cash includes acquisition fees. TRANSFER_IN has no cash
            # purchase; only an explicit gross amount is transferred basis,
            # while a nonzero net amount may be a fee and cannot prove basis.
            committed = (
                abs(order.gross_eur or 0.0)
                if order.type is OrderType.TRANSFER_IN
                else abs(order.net_eur)
                if order.net_eur
                else abs(order.gross_eur or 0.0)
            )
            quantity[key] = held_quantity + order.quantity
            cost[key] = held_cost + committed
            basis_complete[key] = (
                basis_complete.get(key, True) and committed > 0
            )
            continue
        if order.quantity >= 0:
            continue

        requested_units = abs(order.quantity)
        tolerance = max(1e-9, requested_units * 1e-9)
        if order.type is OrderType.SELL and (
            held_quantity + tolerance < requested_units
            or held_cost <= 0
            or not basis_complete.get(key, True)
        ):
            raise TaxEvidenceUnavailable(
                f"tax cost basis unavailable for {order.isin}: sell requests "
                f"{requested_units:g} units but authoritative identity "
                f"{key!r} has {held_quantity:g} units and EUR {held_cost:.2f} cost"
            )

        units = min(requested_units, held_quantity) if held_quantity > 0 else 0.0
        average_cost = held_cost / held_quantity if held_quantity > 0 else 0.0
        cost_removed = average_cost * units
        if order.type is OrderType.SELL and units > 0:
            fraction = units / abs(order.quantity) if order.quantity else 1.0
            proceeds = (order.net_eur or 0.0) * fraction
            pnl = proceeds - cost_removed
            is_government, is_capital_income = _classify(
                order,
                enriched_by_isin,
                orders,
                key,
            )
            realizations.append(
                _Realization(
                    date=order.trade_date,
                    isin=order.isin,
                    pnl=pnl,
                    rate=gov_rate if is_government else std_rate,
                    is_capital_income=is_capital_income,
                )
            )
        remaining_quantity = max(held_quantity - units, 0.0)
        cost[key] = max(held_cost - cost_removed, 0.0)
        quantity[key] = remaining_quantity
        if remaining_quantity <= tolerance:
            basis_complete[key] = True

    realizations.sort(key=lambda realization: realization.date)
    return realizations


def estimate_realized_cgt(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    std_rate_pctg: float,
    gov_rate_pctg: float,
) -> CgtEstimate:
    """Estimate CGT on realized gains; return dated tax cash flows and summary.

    Rates are percentages such as 26 and 12.5. With both rates zero, tax is
    disabled and no classification evidence is required.
    """
    std_rate = max(0.0, float(std_rate_pctg or 0.0)) / 100.0
    gov_rate = max(0.0, float(gov_rate_pctg or 0.0)) / 100.0
    estimate = CgtEstimate()
    if not orders or (std_rate <= 0 and gov_rate <= 0):
        return estimate

    realizations = _realizations(orders, enriched_by_isin, std_rate, gov_rate)

    # Loss carryforward entries are [year, remaining_loss_eur]. ETF gains do
    # not consume them; eligible gains consume still-valid entries by year.
    carry: list[list[float | int]] = []
    tax_by_date: dict[datetime.date, float] = {}

    for realization in realizations:
        if realization.pnl < 0:
            estimate.total_realized_loss_eur += -realization.pnl
            carry.append([realization.date.year, -realization.pnl])
            continue
        if realization.pnl <= 0:
            continue
        estimate.total_realized_gain_eur += realization.pnl

        taxable = realization.pnl
        if not realization.is_capital_income:
            minimum_year = realization.date.year - _LOSS_CARRY_YEARS
            for entry in carry:
                if taxable <= 0:
                    break
                entry_year = int(entry[0])
                remaining_loss = float(entry[1])
                if remaining_loss <= 0 or entry_year < minimum_year:
                    continue
                used = min(taxable, remaining_loss)
                taxable -= used
                entry[1] = remaining_loss - used
        if taxable > 0:
            tax = taxable * realization.rate
            estimate.taxable_base_eur += taxable
            estimate.total_tax_eur += tax
            tax_by_date[realization.date] = (
                tax_by_date.get(realization.date, 0.0) + tax
            )

    estimate.tax_flows = [
        (date, -tax) for date, tax in sorted(tax_by_date.items())
    ]
    return estimate
