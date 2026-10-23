"""Issuer-documented liquidation-price links, not instrument equivalence.

The MEF Information Memoranda, section 1.4, explicitly name both ISINs and
state that a sale converts the loyalty code to the freely negotiable code.
The latter's clean price therefore measures a sale's market value, not the
value of retaining the loyalty right. Never use these links to rewrite orders,
net positions, infer tax basis, or accrue a bonus/coupon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class LiquidationPriceLink:
    policy_id: str
    held_isin: str
    market_isin: str
    currency: str
    first_trading_day: date
    maturity: date
    verified_at: datetime
    issuer_document: str


_DOCUMENTS = (
    "https://www.dt.mef.gov.it/export/sites/sitodt/modules/documenti_it/"
    "debito_pubblico/comunicazioni_emissioni/btp_valore_comunicazione_emissione/"
)
_VERIFIED_AT = datetime(2026, 10, 23, 15, 27, 29, tzinfo=timezone.utc)
_LINKS = (
    LiquidationPriceLink(
        policy_id="MEF_BTP_VALORE_2028_CLEAN_LIQUIDATION_V1",
        held_isin="IT0005565392", market_isin="IT0005565400", currency="EUR",
        first_trading_day=date(2023, 10, 10), maturity=date(2028, 10, 10),
        verified_at=_VERIFIED_AT,
        issuer_document=_DOCUMENTS + "Information-Memorandum-BTP-Valore-10.10.2028-2.pdf",
    ),
    LiquidationPriceLink(
        policy_id="MEF_BTP_VALORE_2030_CLEAN_LIQUIDATION_V1",
        held_isin="IT0005583478", market_isin="IT0005583486", currency="EUR",
        first_trading_day=date(2024, 3, 5), maturity=date(2030, 3, 5),
        verified_at=_VERIFIED_AT,
        issuer_document=_DOCUMENTS + "Information-Memorandum-BTP-Valore-05.03.2030-3.pdf",
    ),
)


def liquidation_price_link(isin: str, *, currency: str, captured_at: datetime):
    """Exact, directional, dated issuer authority; no discovery or heuristics."""
    if not isinstance(captured_at, datetime) or captured_at.tzinfo is None:
        return None
    return next((
        link for link in _LINKS
        if link.held_isin == isin and link.currency == currency
        and captured_at >= link.verified_at
    ), None)
