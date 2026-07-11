"""Single source of truth for asset-class / geography display order.

Historically each surface kept its own hardcoded ordered list of asset
classes, and they had drifted into FOUR different orders (the engine groups
Cash with the invested classes; the newsletter shows Gold before Cash; the
performance table shows Commodities before Gold; the what-if workbook puts
Cash last). Adding a new asset class meant editing ~7 files.

This module centralizes those orders as *named variants* so there is one
place to change. It deliberately **preserves** each surface's existing
sequence rather than unifying them — unifying would visibly reorder rows in
reports the user files, which is a separate, sign-off-gated decision. The
variants are asserted against their historical literals in the tests, so an
accidental reorder is caught.

Lives in the models layer (not export/_format) so the engine, config and
export can all import it without a layering cycle — the engine must not
import from export/.
"""

from __future__ import annotations

from typing import Optional

# Canonical membership (every asset class Tarzan knows). Mirrors AssetClass
# in models.holding; kept as strings here so config/export need not import the
# enum just to order things.
ASSET_CLASSES: tuple[str, ...] = (
    "Equities",
    "Fixed Income",
    "Cash & Cash Equivalents",
    "Gold",
    "Commodities",
    "Crypto",
    "Alternative",
)

# --- Named asset-class order variants (each preserved verbatim) -------------

# Dashboard / engine holdings table (metrics._valuation): Cash sits with the
# invested classes, right after Fixed Income.
ORDER_DASHBOARD: tuple[str, ...] = (
    "Equities", "Fixed Income", "Cash & Cash Equivalents",
    "Gold", "Commodities", "Crypto", "Alternative",
)

# Newsletter allocation/holdings order: Gold before Cash.
ORDER_NEWSLETTER: tuple[str, ...] = (
    "Equities", "Fixed Income", "Gold", "Cash & Cash Equivalents",
    "Commodities", "Crypto", "Alternative",
)

# Newsletter performance table order: Commodities before Gold.
# NOTE: this variant historically omits "Crypto"; a crypto holding is
# appended by the caller's extras logic. Preserved as-is (adding Crypto here
# would visibly reorder a filed report — deferred, needs sign-off).
ORDER_PERF: tuple[str, ...] = (
    "Equities", "Fixed Income", "Commodities", "Gold",
    "Alternative", "Cash & Cash Equivalents",
)

# What-if workbook order: Cash last.
ORDER_WHATIF: tuple[str, ...] = (
    "Equities", "Fixed Income", "Gold", "Commodities", "Crypto",
    "Alternative", "Cash & Cash Equivalents",
)

# Base order used by export._format.asset_class_order() (was
# _ASSET_CLASS_BASE_ORDER). Same sequence as the dashboard.
ORDER_BASE: tuple[str, ...] = ORDER_DASHBOARD

# --- Geography order --------------------------------------------------------

GEO_ORDER: tuple[str, ...] = (
    "USA", "Japan", "Eurozone EMU", "Dev ex-USA ex-EMU ex-JP",
    "Emerging Markets",
)


def order_with_extras(
    order: tuple[str, ...], present: Optional[list[str]] = None
) -> list[str]:
    """A display order that never silently drops an unlisted class.

    Returns ``order`` (optionally filtered to ``present``) with any class in
    ``present`` that is not in ``order`` appended alphabetically at the end —
    the shared "extras go last" convention every surface already used, so a
    newly added asset class still shows up in every report.
    """
    if not present:
        return list(order)
    extra = sorted(set(present) - set(order))
    return [c for c in order if c in present] + extra
