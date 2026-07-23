"""Immutable effective-order boundary for point-in-time analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from tarzan.models.order import Order
from tarzan.runtime.io_utils import canonical_json_bytes


@dataclass(frozen=True)
class EffectiveOrderSnapshot:
    """Stable, immutable order view shared by every financial consumer."""

    orders: tuple[Order, ...]
    boundary: Optional[date]
    accepted_count: int
    excluded_count: int
    digest: str

    @classmethod
    def build(
        cls,
        orders: Iterable[Order],
        boundary: Optional[date],
    ) -> "EffectiveOrderSnapshot":
        source = tuple(orders)

        def effective_trade_date(order: Order) -> date:
            return getattr(order, "trade_date", None) or order.date

        visible = tuple(
            sorted(
                (
                    order
                    for order in source
                    if boundary is None or effective_trade_date(order) <= boundary
                ),
                key=lambda order: (
                    effective_trade_date(order),
                    order.date,
                    order.order_id,
                ),
            )
        )
        canonical = [
            {
                "order_id": order.order_id,
                "effective_trade_date": effective_trade_date(order).isoformat(),
                "settlement_date": order.date.isoformat(),
                "type": order.type.value,
                "isin": order.isin,
                "quantity": format(float(order.quantity), ".12g"),
                "gross_eur": format(float(order.gross_eur), ".12g"),
                "net_eur": format(float(order.net_eur), ".12g"),
            }
            for order in visible
        ]
        digest = hashlib.sha256(
            canonical_json_bytes(canonical, ascii_only=True)
        ).hexdigest()
        return cls(
            orders=visible,
            boundary=boundary,
            accepted_count=len(visible),
            excluded_count=len(source) - len(visible),
            digest=digest,
        )
