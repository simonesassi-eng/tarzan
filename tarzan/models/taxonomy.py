"""Canonical asset-class and geography display ordering.

Every presentation surface uses :data:`CANONICAL_ORDER`. Named aliases keep
call sites explicit while preserving one membership and ordering authority in
the models layer, below both computation and presentation.
"""

from __future__ import annotations

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


