"""Estimate Italian capital-gains tax (CGT) on *realized* gains.

This is an **estimate** used to present net-of-tax money-weighted figures
(XIRR, lifetime PnL) alongside the gross ones. It is NOT a tax return and
deliberately keeps the headline gross metrics untouched.

Scope and method (all disclosed to the user in the report):

  * Only **realized capital gains** are taxed — sells, not transfers,
    coupons or dividends (income withholding is a separate matter).
  * **Average-cost basis**: the cost removed by a sell is the running
    weighted-average cost of the units held, matching how the rest of
    Tarzan computes cost basis. (Italian brokers may use LIFO, so the
    estimate can differ from the broker's exact figure.)
  * **Rates**: ``gov_rate`` for government bonds (BTP, US Treasury and
    similar state issuers — Italy taxes these at 12.5%), ``std_rate``
    (26%) for everything else.
  * **Loss offset ("zainetto fiscale")**: realized losses are carried
    forward and offset later realized gains, but only where the law
    allows it. Gains on harmonized ETFs/funds (OICR) are *redditi di
    capitale* and CANNOT be reduced by capital losses; gains on single
    bonds, stocks, ETCs and certificates are *redditi diversi* and CAN.
    Every realized loss (including ETF losses) feeds the carryforward,
    which expires four years after the year it arose.
  * Tax is attributed to the **trade date** of the sell (when the broker
    withholds it in regime amministrato), so it lands as a negative cash
    flow at that date for the net-of-tax XIRR.

The result is a list of ``(date, -tax_eur)`` cash flows plus a summary
breakdown for disclosure.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import Optional

from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType

# Two ISINs whose first 9 characters match are cum/ex variants of the same
# Italian retail BTP (sold under one ISIN, transferred in under a sibling).
# Cost basis must be tracked per prefix group, or the realized gain on the
# rotation is lost (cost on one leg, proceeds on the other).
_CUM_EX_PREFIX_LEN = 9

# Fineco short-name / description markers for state-issued bonds that get
# the reduced (12.5%) rate. Matched case-insensitively on the order name.
_GOV_BOND_NAME = re.compile(
    r"\b(BTP|BOT|CCT|CTZ|BUND|OAT|BONOS?|GILT|TREASUR|T-?NOTE|T-?BOND)\b"
    r"|^USA-",  # Fineco labels US Treasuries like "USA-31GE30 3,5%"
    re.IGNORECASE,
)

# How many years a realized loss stays usable (year of realization + 4).
_LOSS_CARRY_YEARS = 4


@dataclass
class CgtEstimate:
    """Estimated CGT on realized gains and the cash flows it implies."""

    tax_flows: list[tuple[datetime.date, float]] = field(default_factory=list)
    total_tax_eur: float = 0.0
    total_realized_gain_eur: float = 0.0
    total_realized_loss_eur: float = 0.0
    taxable_base_eur: float = 0.0  # gains actually taxed, after offsets/rules


@dataclass
class _Realization:
    date: datetime.date
    isin: str
    pnl: float          # realized result (proceeds − avg cost), signed
    rate: float         # tax rate that applies to a gain on this instrument
    is_capital_income: bool  # True = OICR/ETF gain → not offsettable by losses


def _classify(
    isin: str,
    name: str,
    enriched: Optional[Holding],
    orders: list[Order],
) -> tuple[bool, bool]:
    """Return ``(is_government_bond, is_capital_income)`` for an ISIN.

    * ``is_government_bond`` selects the reduced rate.
    * ``is_capital_income`` is True for harmonized ETF/fund (OICR) gains,
      which cannot be offset by carried-forward losses.

    Uses the enriched ``instrument_type`` when available, falling back to
    the Fineco order name and the shared bond heuristic for positions that
    are already closed (and therefore never enriched).
    """
    it = (getattr(enriched, "instrument_type", None) or "").lower()
    nm = (name or "").lower()

    is_gov = (
        "govern" in it or "govt" in it or bool(_GOV_BOND_NAME.search(name or ""))
    )
    if is_gov:
        # Single government bond → redditi diversi (offsettable), gov rate.
        return True, False

    is_etf = "etf" in it or "fund" in it
    if is_etf:
        return False, True  # OICR gain = reddito di capitale, no offset

    # Bond (corporate/other) or single equity → redditi diversi.
    is_bondish = "bond" in it or _order_bond_flag(orders, isin)
    if is_bondish:
        return False, False
    if enriched is not None:
        # Enriched, not a bond, not flagged ETF → treat single share as
        # redditi diversi.
        return False, False
    # Closed position with no enrichment and not a bond: the portfolio is
    # ETF-based, so default to capital income (the conservative choice —
    # it does not let losses wrongly shelter the gain).
    return False, True


def _order_bond_flag(orders: list[Order], isin: str) -> bool:
    """Bond detection from order prices alone (clean price ~50–150 on a
    large nominal), for ISINs that never reached enrichment."""
    from tarzan.data.bond_fetcher import looks_like_bond_from_orders
    sub = [o for o in orders if o.isin == isin and o.price_native is not None]
    if not sub:
        return False
    avg_price = sum(o.price_native for o in sub) / len(sub)
    avg_qty = sum(abs(o.quantity) for o in sub) / len(sub)
    return looks_like_bond_from_orders(avg_price, avg_qty)


def _realizations(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    std_rate: float,
    gov_rate: float,
) -> list[_Realization]:
    """Walk position-changing orders in trade-date order, tracking running
    average cost per ISIN, and emit one ``_Realization`` per *sell*.

    Transfers (in/out) move cost but are not taxable events; only ``SELL``
    realizes a gain/loss for tax.
    """
    pos = sorted(
        (o for o in orders if o.is_position_change()),
        key=lambda o: o.trade_date,
    )
    name_by_isin: dict[str, str] = {}
    for o in orders:
        if o.name and o.isin not in name_by_isin:
            name_by_isin[o.isin] = o.name

    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    out: list[_Realization] = []

    for o in pos:
        # Track cost basis per cum/ex prefix group so a BTP sold under its
        # "ex" ISIN draws down the cost transferred in under its "cum"
        # sibling. For ordinary instruments the 9-char prefix is unique to
        # the ISIN, so this is a no-op.
        key = o.isin[:_CUM_EX_PREFIX_LEN]
        q = qty.get(key, 0.0)
        c = cost.get(key, 0.0)
        if o.quantity > 0:  # buy / transfer_in
            committed = abs(o.net_eur) if o.net_eur else abs(o.gross_eur or 0.0)
            qty[key] = q + o.quantity
            cost[key] = c + committed
            continue
        if o.quantity >= 0:
            continue
        # sell / transfer_out
        units = min(abs(o.quantity), q) if q > 0 else 0.0
        avg = (c / q) if q > 0 else 0.0
        cost_removed = avg * units
        if o.type == OrderType.SELL and units > 0:
            # Proceeds for the units actually held (prorate if the order
            # nominally sells more than we tracked as held).
            frac = units / abs(o.quantity) if o.quantity else 1.0
            proceeds = (o.net_eur or 0.0) * frac
            pnl = proceeds - cost_removed
            is_gov, is_cap = _classify(
                o.isin, name_by_isin.get(o.isin, ""),
                enriched_by_isin.get(o.isin), orders,
            )
            out.append(_Realization(
                date=o.trade_date, isin=o.isin, pnl=pnl,
                rate=(gov_rate if is_gov else std_rate),
                is_capital_income=is_cap,
            ))
        cost[key] = max(c - cost_removed, 0.0)
        qty[key] = max(q - units, 0.0)

    out.sort(key=lambda r: r.date)
    return out


def estimate_realized_cgt(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    std_rate_pctg: float,
    gov_rate_pctg: float,
) -> CgtEstimate:
    """Estimate CGT on realized gains; return the tax cash flows + summary.

    ``std_rate_pctg`` / ``gov_rate_pctg`` are percentages (e.g. 26, 12.5).
    With both rates 0 the estimate is empty (net == gross).
    """
    std_rate = max(0.0, float(std_rate_pctg or 0.0)) / 100.0
    gov_rate = max(0.0, float(gov_rate_pctg or 0.0)) / 100.0
    est = CgtEstimate()
    if not orders or (std_rate <= 0 and gov_rate <= 0):
        return est

    realizations = _realizations(orders, enriched_by_isin, std_rate, gov_rate)

    # Loss carryforward entries: (year, remaining_loss_eur). Gains that are
    # redditi diversi consume them (most recent first is irrelevant; FIFO by
    # year keeps the 4-year expiry simple); ETF gains never do.
    carry: list[list] = []  # [year, remaining]
    tax_by_date: dict[datetime.date, float] = {}

    for r in realizations:
        if r.pnl < 0:
            est.total_realized_loss_eur += -r.pnl
            carry.append([r.date.year, -r.pnl])
            continue
        if r.pnl <= 0:
            continue
        est.total_realized_gain_eur += r.pnl

        taxable = r.pnl
        if not r.is_capital_income:
            # Offset against still-valid carried-forward losses (year of
            # the loss within the gain's year − 4 .. gain's year).
            min_year = r.date.year - _LOSS_CARRY_YEARS
            for entry in carry:
                if taxable <= 0:
                    break
                if entry[1] <= 0 or entry[0] < min_year:
                    continue
                used = min(taxable, entry[1])
                taxable -= used
                entry[1] -= used
        if taxable > 0:
            tax = taxable * r.rate
            est.taxable_base_eur += taxable
            est.total_tax_eur += tax
            tax_by_date[r.date] = tax_by_date.get(r.date, 0.0) + tax

    est.tax_flows = [(d, -tax) for d, tax in sorted(tax_by_date.items())]
    return est
