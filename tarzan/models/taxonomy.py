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


# --- The one asset-class key ------------------------------------------------

#: The class an item with no class of its own is grouped, totalled and counted
#: under. Not a member of :data:`CANONICAL_ORDER`: it is a residual bucket, so
#: the ordering helpers append it after the canonical classes rather than
#: reserving it a slot.
UNCLASSIFIED = "Other"


def class_key(value) -> str:
    """The asset class an item is grouped, totalled and counted under — the class
    itself, or :data:`UNCLASSIFIED` when it is unset.

    ONE normaliser, because a per-class aggregate and the row that divides by it
    have to be keyed on the same string. They were not: The book grouped its
    totals on the RAW ``asset_class`` column (and ``groupby`` drops None/NaN
    outright) while each row looked its total up under a normalised key, so an
    unclassified holding found no entry, took a ``.get(klass, 1)`` default of one
    EURO, and printed €3,013.09 as 301309.3%. The same split made the summary
    chips count only the classed holdings.

    Lives here, in the models layer, so the engine can key on it too — a
    normaliser in a presentation module is one the computation side cannot reach,
    which is how four copies of this expression came to exist.

    The NaN test is restricted to floats on purpose. A bare ``value != value``
    returns ``pd.NA`` for pandas' own missing sentinel, and ``if pd.NA`` raises
    ``TypeError: boolean value of NA is ambiguous`` — a normaliser must not be
    the thing that breaks. Enum members are unwrapped first, because
    ``str(AssetClass.EQUITIES)`` is ``"AssetClass.EQUITIES"``, not ``"Equities"``.
    """
    value = getattr(value, "value", value)
    if value is None or (isinstance(value, float) and value != value):
        return UNCLASSIFIED
    return str(value).strip() or UNCLASSIFIED


