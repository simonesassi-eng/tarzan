"""Persistent mechanics authority and explicit EUR-per-unit valuation facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tarzan.instruments.registry import TypeEvidenceGateway, TypeResolution


def holding_kind_resolution(holding: object, *assertions: str) -> TypeResolution:
    """A display/taxonomy rewrite must not erase a provider rejection."""
    retained = getattr(holding, "instrument_resolution", None)
    return TypeEvidenceGateway().resolve(
        *(retained.evidence if retained is not None else ()),
        getattr(holding, "security_type", None),
        getattr(holding, "instrument_type", None),
        *tuple(getattr(holding, "instrument_kind_evidence", ()) or ()),
        *assertions,
    )


class ValuationOrigin(str, Enum):
    MARKET = "MARKET"
    FALLBACK = "FALLBACK"
    ANCHOR = "ANCHOR"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ValuationResult:
    """Selected valuation, not display metadata or an unvalued raw quote."""

    resolution: TypeResolution
    origin: ValuationOrigin
    price_eur_per_unit: Optional[float]
    value_eur: Optional[float]
    source: Optional[str]

    def __post_init__(self) -> None:
        if self.value_eur is not None and (
            not math.isfinite(self.value_eur) or self.value_eur < 0
        ):
            raise ValueError("valuation must be finite and nonnegative")
        if self.origin in (ValuationOrigin.MARKET, ValuationOrigin.FALLBACK):
            if (
                self.resolution.kind is None or self.value_eur is None
                or self.price_eur_per_unit is None
                or not math.isfinite(self.price_eur_per_unit)
                or self.price_eur_per_unit <= 0
            ):
                raise ValueError("market valuation requires resolved mechanics and a positive price")
        elif self.price_eur_per_unit is not None:
            raise ValueError("an anchor/unavailable result cannot carry a selected market price")


def select_holding_valuation(holding: object) -> ValuationResult:
    """Freeze the selected origin after the one-time clean-price/FX conversion."""
    resolution = holding_kind_resolution(holding)
    source = getattr(holding, "data_source", None)

    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    price = finite(getattr(holding, "current_price", None))
    value = finite(getattr(holding, "current_value", None))
    quantity = finite(getattr(holding, "quantity", None))
    anchor_source = str(source or "").startswith(("input_csv", "last-known", "historical-only"))
    if (
        resolution.kind is not None and price is not None and price > 0
        and quantity is not None and quantity >= 0
        and value is not None and value >= 0 and not anchor_source
        and math.isclose(value, quantity * price, rel_tol=1e-12, abs_tol=1e-8)
    ):
        return ValuationResult(
            resolution,
            ValuationOrigin.FALLBACK if getattr(holding, "price_is_fallback", False)
            else ValuationOrigin.MARKET,
            price, value, source,
        )
    anchor = finite(getattr(holding, "market_value_eur", None))
    if anchor is not None and anchor >= 0:
        return ValuationResult(resolution, ValuationOrigin.ANCHOR, None, anchor, source)
    return ValuationResult(resolution, ValuationOrigin.UNAVAILABLE, None, None, source)
