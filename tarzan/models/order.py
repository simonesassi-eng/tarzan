"""Domain model for a single order-list entry.

An ``Order`` is one dated movement in the portfolio's journal: a buy,
sell, coupon, dividend, or security transfer. Where ``Holding`` is a
*current snapshot*, ``Order`` is a *dated flow* — the data source for
money-weighted (XIRR) and time-weighted (TWROR) returns and, when
present, the single source of truth for the portfolio's historical
value series.

The field set mirrors the canonical ``input/order_list.csv`` schema
produced by ``scripts/preprocess_orders.py``.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tarzan.instruments.registry import InstrumentKind


class OrderType(str, Enum):
    """The kind of movement an order represents.

    Sign conventions (carried on the ``Order`` fields, not the enum):
      * BUY / TRANSFER_IN  → quantity > 0
      * SELL / TRANSFER_OUT → quantity < 0
      * COUPON / DIVIDEND  → quantity == 0 (a distribution, no position change)
    """

    BUY = "buy"
    SELL = "sell"
    COUPON = "coupon"
    DIVIDEND = "dividend"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"

    @classmethod
    def from_raw(cls, raw: str) -> Optional["OrderType"]:
        """Parse a raw string into an OrderType, or None if unknown.

        Case-insensitive and whitespace-tolerant so loader code can feed
        a CSV cell directly. ``None`` signals "skip this row" rather than
        raising, mirroring the loader's skip-with-warning behavior.
        """
        if raw is None:
            return None
        key = str(raw).strip().lower()
        try:
            return cls(key)
        except ValueError:
            return None


# Movement kinds that move cash on the bank account (XIRR cash flows).
_CASHFLOW_TYPES = frozenset(
    {OrderType.BUY, OrderType.SELL, OrderType.COUPON, OrderType.DIVIDEND}
)
# Movement kinds that change the held quantity of a security.
_POSITION_CHANGE_TYPES = frozenset(
    {OrderType.BUY, OrderType.SELL, OrderType.TRANSFER_IN, OrderType.TRANSFER_OUT}
)


@dataclass
class Order:
    """A single normalized order-list entry.

    Attributes mirror the order_list.csv columns. ``quantity`` is signed
    with the direction of the position change; ``net_eur`` is signed with
    the cash-flow direction on the bank account (positive = cash in).
    """

    date: datetime.date            # settlement / value date (used by XIRR)
    trade_date: datetime.date      # order date (audit; can equal `date`)
    type: OrderType
    isin: str
    name: str
    ticker: str
    quantity: float                # signed: + buy/transfer_in, − sell, 0 coupon/div
    currency: str
    price_native: Optional[float]
    fx_rate: Optional[float]
    gross_eur: float               # signed gross EUR before fees
    fees_eur: float
    net_eur: float                 # signed bank cash flow
    source: str = "fineco"
    # Stable primary key for this ledger row. Defaults to the natural-key
    # fingerprint (see ``natural_key``); the loader overrides it with an
    # ordinal-disambiguated id so two genuinely-identical movements on the same
    # day remain distinct rows. Never used to *merge* orders (the ledger is
    # append-only and identical events are legitimate) — only to reference,
    # dedup-detect, and audit a specific entry.
    order_id: str = ""
    # Exact instrument mechanics authority. Category, name, price, quantity,
    # and identifier shape may never substitute for this kind declaration.
    instrument_kind: Optional[InstrumentKind] = None

    def __post_init__(self) -> None:
        if self.instrument_kind is not None and not isinstance(
            self.instrument_kind, InstrumentKind
        ):
            try:
                self.instrument_kind = InstrumentKind(
                    str(self.instrument_kind).strip().upper()
                )
            except ValueError as error:
                raise ValueError(
                    f"unsupported instrument kind: {self.instrument_kind!r}"
                ) from error
        if not self.order_id:
            self.order_id = self.natural_key()

    def natural_key(self) -> str:
        """A short, deterministic fingerprint of the fields that define the
        economic event (date, instrument, type, quantity, cash). Two rows with
        the same natural key describe the same movement — the signal the loader
        uses to flag an accidentally duplicated input block."""
        raw = "|".join(str(x) for x in (
            self.trade_date, self.isin, self.type.value,
            f"{self.quantity:.6f}", f"{self.net_eur:.2f}",
            f"{self.gross_eur:.2f}", self.source,
        ))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def is_cashflow(self) -> bool:
        """True if this order moves cash on the bank account."""
        return self.type in _CASHFLOW_TYPES

    def is_position_change(self) -> bool:
        """True if this order changes the held quantity of a security."""
        return self.type in _POSITION_CHANGE_TYPES
