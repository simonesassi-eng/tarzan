"""Borsa Italiana scraper for bond pricing data.

Fallback data source for bonds that yfinance cannot resolve (e.g. BTP, EIB,
US Treasury on MOT/EuroTLX). Provides current price by ISIN via web scraping.

No API key required. Covers all bonds listed on Borsa Italiana MOT and EuroTLX.
"""

from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.request import Request, urlopen

from tarzan.instruments.registry import InstrumentKind

logger = logging.getLogger(__name__)

# Borsa Italiana segments to search (in priority order)
_SEGMENTS = [
    "mot/btp",
    "mot/euro-obbligazioni",
    "eurotlx/obbligazioni-euro",
    "eurotlx/obbligazioni-dollaro",
    "mot/altri-titoli-di-stato",
    "mot/cct",
]


def fetch_bond_price(isin: str) -> Optional[dict]:
    """Fetch current bond price from Borsa Italiana by ISIN.

    Tries multiple market segments (MOT BTP, Euro-obbligazioni, EuroTLX).

    Returns:
        Dict with 'price' (clean price as % of par) and 'source', or None.
    """
    if not isin or len(isin.replace("-", "")) != 12:
        return None

    for segment in _SEGMENTS:
        result = _try_segment(isin, segment)
        if result:
            return result

    logger.debug("Bond %s not found on any Borsa Italiana segment", isin)
    return None


def _try_segment(isin: str, segment: str) -> Optional[dict]:
    """Try fetching bond price from a specific Borsa Italiana segment."""
    url = f"https://www.borsaitaliana.it/borsa/obbligazioni/{segment}/scheda/{isin}.html"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return None
            html = resp.read().decode("utf-8")
            if len(html) < 5000:
                return None

            price = _extract_price(html)
            if price is not None:
                logger.info("Borsa Italiana price for %s (%s): %.4f", isin, segment, price)
                return {"price": price, "source": f"borsa_italiana/{segment}"}
    except Exception:
        pass
    return None


def _extract_price(html: str) -> Optional[float]:
    """Extract the official/last price from a Borsa Italiana bond page.

    Tries multiple price fields in priority order:
    1. Prezzo ufficiale (official price)
    2. Ultimo prezzo (last price)
    3. Prezzo di riferimento (reference price)
    """
    patterns = [
        r"Prezzo ufficiale.*?(\d{1,3}(?:\.\d{3})*,\d{2,6})",
        r"Ultimo prezzo.*?(\d{1,3}(?:\.\d{3})*,\d{2,6})",
        r"Prezzo di riferimento.*?(\d{1,3}(?:\.\d{3})*,\d{2,6})",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            price_str = match.group(1)
            # Italian format: 1.234,56 → 1234.56
            price_str = price_str.replace(".", "").replace(",", ".")
            try:
                return float(price_str)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Shared position valuation
# ---------------------------------------------------------------------------


def is_bond(instrument_kind: Optional[InstrumentKind]) -> bool:
    """Return whether the exact registered mechanics kind is ``BOND``.

    Asset category, display strings, names, identifiers, prices, and quantities
    are intentionally not accepted. They describe exposure or appearance, not
    the valuation convention.
    """
    return instrument_kind is InstrumentKind.BOND


def value_position(
    quantity: float,
    price_eur_per_unit: float,
    *,
    instrument_kind: InstrumentKind,
) -> float:
    """EUR value under the convention declared by ``instrument_kind``.

    Bonds quote a clean price per 100 of face value and ``quantity`` is nominal,
    so their value is ``quantity * price / 100``. Stock, ETF, and cash prices
    are per unit. Unknown mechanics are rejected instead of being guessed as a
    non-bond.
    """
    if not isinstance(instrument_kind, InstrumentKind):
        raise ValueError("explicit InstrumentKind is required for valuation")
    if instrument_kind is InstrumentKind.BOND:
        return quantity * price_eur_per_unit / 100.0
    return quantity * price_eur_per_unit
