"""Newsletter palette, colour maps, class-order constants + the render context.

Leaf module: depends only on external packages. Everything above sits on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.models.taxonomy import (
    ORDER_NEWSLETTER as _ORDER_NEWSLETTER,
    ORDER_PERF as _ORDER_PERF,
)
from tarzan.export._format import (
    ASSET_CLASS_BG,
    ASSET_CLASS_COLORS,
    GEO_COLORS as _GEO_COLORS,
    css,
)

PALETTE = {
    "accent": "#5B5BD6",
    "ink": "#1E293B",
    "muted": "#64748B",
    "subtle": "#94A3B8",
    "page": "#F1F2F8",
    "card_alt": "#F8FAFF",
    "border": "#E5E7EF",
    "green": "#15803D",
    "amber": "#D97706",
    "red": "#DC2626",
    "green_bg": "#DCFCE7",
    "green_border": "#BBF7D0",
    "amber_bg": "#FFF7ED",
    "amber_border": "#FED7AA",
    "red_bg": "#FEE2E2",
    "red_border": "#FECACA",
    "accent_bg": "#EEF2FF",
    "gold_bg": "#FEF3C7",
    "fi_bg": "#FEF3C7",
}

ASSET_COLORS = {k: css(v) for k, v in ASSET_CLASS_COLORS.items()}

ASSET_BG = {k: css(v) for k, v in ASSET_CLASS_BG.items()}

GEO_COLORS = {k: css(v) for k, v in _GEO_COLORS.items()}

MARKET_REGION_COLORS = {
    "US": "#2563EB",           # blue
    "Europe": "#D97706",       # amber
    "Asia": "#DB2777",         # pink
    "Crypto": "#7C3AED",       # purple
    "Commodities": "#15803D",  # green
    "Currencies": "#64748B",   # slate
    "Indices": "#64748B",      # slate (offline fallback bucket)
}

_NEWSLETTER_CLASS_ORDER = list(_ORDER_NEWSLETTER)

_extra_classes = [c for c in ASSET_CLASS_COLORS if c not in _NEWSLETTER_CLASS_ORDER]

ASSET_CLASS_ORDER = _NEWSLETTER_CLASS_ORDER + sorted(_extra_classes)


_PERF_CLASS_ORDER = list(_ORDER_PERF)
_PERF_ROLE_ORDER = {
    "Equities": ["Equity Broad", "Equity Factor", "Equity Leveraged",
                 "Efficient Core", "Multi-Asset"],
    "Fixed Income": ["Govt Nominal", "Govt Linkers", "Aggregate/Credit",
                     "Long Duration"],
    "Commodities": ["Broad Basket", "Carry", "Market Neutral"],
    "Gold": ["Gold"],
    "Alternative": ["Managed Futures", "Cat Bond"],
    "Cash & Cash Equivalents": ["Cash / Money Market"],
}


_PF_INTRA_KEY = "__PORTFOLIO_INTRADAY__"


# ── Shared instrument-categorization engine ─────────────────────────────────
# One place that decides an instrument's (asset class, role) and how the
# class→role groups are ordered, so EVERY table (Holdings, Optimizer, Returns
# snapshot, Performance) splits and colours instruments identically. Before,
# each table re-derived this inline and drifted.

def role_for(isin, ticker, taxonomy) -> str:
    """The curated role (e.g. 'Equity Factor', 'Long Duration') for an
    instrument, from ``instrument_taxonomy.csv`` (ISIN first, then bare
    ticker). ``taxonomy`` is ``config.instrument_taxonomy()`` — passed in so
    this stays a pure function. Returns '—' when the role is unset."""
    for k in (str(isin or "").strip().upper(),
              str(ticker or "").split(".")[0].upper()):
        if k and k in taxonomy and taxonomy[k][1]:
            return taxonomy[k][1]
    return "—"


def _ordered(keys, preferred):
    """``preferred`` order first (those present), then any extras in their
    given order — so a new class/role is appended, never dropped."""
    return ([k for k in preferred if k in keys]
            + [k for k in keys if k not in preferred])


def group_by_class_role(items, *, asset_class, taxonomy=None,
                        isin=None, ticker=None, role=None):
    """Group an iterable of items into the canonical
    ``[(class, class_color, [(role, [item, ...]), ...]), ...]`` structure,
    ordered by _PERF_CLASS_ORDER then _PERF_ROLE_ORDER. The accessors are
    callables mapping an item to a field, so the SAME engine works for
    holdings-df rows, optimizer actions, and performance rows alike.

    Role is resolved one of two ways: pass ``role`` (a callable) when the item
    already carries its role, else pass ``isin``+``ticker``+``taxonomy`` to
    look it up via :func:`role_for`.

    Returns the ordered groups; the class colour is ASSET_COLORS[class] so the
    4px left bar / marker is consistent everywhere.
    """
    grouped: dict = {}
    for it in items:
        ac = str(asset_class(it) or "") or "Other"
        if role is not None:
            r = str(role(it) or "") or "—"
        else:
            r = role_for(isin(it), ticker(it), taxonomy)
        grouped.setdefault(ac, {}).setdefault(r, []).append(it)
    groups = []
    for ac in _ordered(list(grouped.keys()), _PERF_CLASS_ORDER):
        col = ASSET_COLORS.get(ac, PALETTE["accent"])
        role_list = [(role, grouped[ac][role])
                     for role in _ordered(list(grouped[ac].keys()),
                                          _PERF_ROLE_ORDER.get(ac, []))]
        groups.append((ac, col, role_list))
    return groups


@dataclass
class _NewsletterContext:
    """Strongly-typed wrapper around the template context dict."""

    metrics: PortfolioMetrics
    config: InvestorConfig
    issue_number: int = 1
    benchmark_alpha_beta: str = "S&P 500"
    benchmark_geo: str = "MSCI ACWI"

