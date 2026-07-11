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

# --- Canonical asset-class display order ------------------------------------
# ONE order used by every surface (user-approved "Cash last" convention):
# invested classes first in a risk/theme progression, cash as the residual
# settlement line at the bottom. All 7 classes are present, so no surface is
# missing a class (the performance table previously omitted Crypto, sorting a
# crypto holding to the end via the extras logic — it now sits in its natural
# slot everywhere). Row order is identical across the Excel dashboard, the
# newsletter (allocation / holdings / performance / historical-risk) and the
# what-if workbook.
CANONICAL_ORDER: tuple[str, ...] = (
    "Equities", "Fixed Income", "Gold", "Commodities", "Crypto",
    "Alternative", "Cash & Cash Equivalents",
)

# All surfaces now share the one canonical order. The named aliases are kept
# so call-sites read intentfully and a future re-divergence (should a surface
# ever need its own order again) is a one-line change here, not a hunt across
# modules.
ORDER_DASHBOARD: tuple[str, ...] = CANONICAL_ORDER
ORDER_NEWSLETTER: tuple[str, ...] = CANONICAL_ORDER
ORDER_PERF: tuple[str, ...] = CANONICAL_ORDER
ORDER_WHATIF: tuple[str, ...] = CANONICAL_ORDER
ORDER_BASE: tuple[str, ...] = CANONICAL_ORDER

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
