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


def bond_price_to_value(price: float, quantity: float) -> float:
    """Convert bond clean price (% of par) to EUR market value.

    For European government bonds: value = quantity * price / 100
    (quantity is the nominal amount, price is percentage of par).

    Thin alias of :func:`value_position` with ``bond=True`` — kept for
    readability at call sites that already know they hold a bond.
    """
    return value_position(quantity, price, bond=True)


# ---------------------------------------------------------------------------
# Shared position valuation + bond classification
# ---------------------------------------------------------------------------
# These three helpers are the single source of truth for the bond
# per-100-nominal convention. Both the current-value path (enricher) and
# the historical order-derived path use them, so a bond is valued the
# same way everywhere (design Property 2: valuation consistency).

# Keywords in instrument_type that indicate the per-100-nominal bond
# convention applies. Matched as whole words (case-insensitive) so a
# substring like "NOTE" inside an unrelated name does not false-positive.
_BOND_INSTRUMENT_KEYWORDS = ("BOND", "TREASURY", "GOVT", "GOVERNMENT",
                             "CORP", "CORPORATE", "NOTE", "BTP", "BUND", "GILT")

# OpenFIGI marketSector values that denote fixed income (authoritative).
_FIGI_BOND_MARKET_SECTORS = ("GOVT", "CORP", "MTGE", "MUNI")

# OpenFIGI securityType2 fragments that denote a bond/note.
_FIGI_BOND_SEC_TYPES = ("BOND", "NOTE", "BILL", "DEBENTURE", "SOVEREIGN",
                        "TREASURY", "FIXED")

# Order-list bond heuristic thresholds: a clean bond price is quoted per
# 100 of face value (typically 50–150), and retail bond face amounts are
# at least ~1,000, whereas ETF/equity unit prices in the 50–150 range
# come with much smaller share counts.
_ORDER_BOND_PRICE_MIN = 50.0
_ORDER_BOND_PRICE_MAX = 150.0
_ORDER_BOND_MIN_QTY = 1000.0


def value_position(
    quantity: float, price_eur_per_unit: float, bond: bool
) -> float:
    """EUR market value of a position, applying the bond convention once.

    Bonds quote a clean price per 100 of face value, and ``quantity`` is
    the nominal amount, so the value is ``quantity * price / 100``.
    Non-bonds use ``quantity * price`` directly.

    This is the ONLY place the ``/100`` is applied, so the current and
    historical valuation paths agree by construction.
    """
    if bond:
        return quantity * price_eur_per_unit / 100.0
    return quantity * price_eur_per_unit


def _has_word(text: str, words) -> bool:
    """True if any of ``words`` appears as a whole token in ``text``
    (case-insensitive). Avoids substring false-positives."""
    tokens = re.findall(r"[A-Z]+", str(text).upper())
    token_set = set(tokens)
    return any(w in token_set for w in words)


def is_bond(
    *,
    asset_class=None,
    instrument_type: Optional[str] = None,
    quote_type: Optional[str] = None,
    sec_type: Optional[str] = None,
    market_sector: Optional[str] = None,
    figi_sec_type: Optional[str] = None,
) -> bool:
    """Single source of truth for the bond classification that triggers
    the per-100-nominal convention.

    Accepts whatever evidence the caller has — the enricher passes
    yfinance ``quote_type``/``sec_type`` plus OpenFIGI's authoritative
    ``market_sector``/``figi_sec_type``; the orders path passes the
    holding's ``asset_class``/``instrument_type``. Any one positive
    signal is enough.

    OpenFIGI's ``marketSector`` ("Govt"/"Corp"/…) and ``securityType2``
    are the most reliable signals (they classify the instrument itself,
    not a display string), so they are checked first.
    """
    # 1. OpenFIGI authoritative classification.
    if market_sector and str(market_sector).strip().upper() in _FIGI_BOND_MARKET_SECTORS:
        return True
    if figi_sec_type and _has_word(figi_sec_type, _FIGI_BOND_SEC_TYPES):
        return True

    # 2. AssetClass enum or its string value "Fixed Income".
    if asset_class is not None:
        ac_value = getattr(asset_class, "value", asset_class)
        if str(ac_value).strip().lower() == "fixed income":
            return True

    # 3. yfinance / instrument-type keyword evidence (whole-word match).
    for field in (instrument_type, quote_type, sec_type):
        if field and _has_word(field, _BOND_INSTRUMENT_KEYWORDS):
            return True
    return False


def looks_like_bond_from_orders(avg_price: float, avg_qty: float) -> bool:
    """Heuristic bond detection from order-list data alone.

    Used only for ISINs that never reach yfinance (so we lack
    ``quote_type``). A clean bond price sits in ``[50, 150]`` and the
    nominal quantity is at least ``1000``; ETF/equity unit prices in
    that band come with far smaller share counts. Deliberately distinct
    from :func:`is_bond` — different evidence, same module.
    """
    return (
        _ORDER_BOND_PRICE_MIN <= avg_price <= _ORDER_BOND_PRICE_MAX
        and avg_qty >= _ORDER_BOND_MIN_QTY
    )