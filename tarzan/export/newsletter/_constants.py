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


@dataclass
class _NewsletterContext:
    """Strongly-typed wrapper around the template context dict."""

    metrics: PortfolioMetrics
    config: InvestorConfig
    issue_number: int = 1
    benchmark_alpha_beta: str = "S&P 500"
    benchmark_geo: str = "MSCI ACWI"

