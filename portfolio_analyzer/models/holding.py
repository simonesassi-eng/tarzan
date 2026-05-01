"""Domain models for individual holdings, asset classes, and geographies.

Defines the core value objects used throughout the pipeline:
- AssetClass: MSCI-aligned asset classification enum
- Geography: MSCI-based geographic bucket enum
- Holding: immutable-ish dataclass representing a single investment position
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class AssetClass(str, Enum):
    """Standardized asset class categories aligned with MSCI/Morningstar taxonomy."""

    EQUITIES = "Equities"
    FIXED_INCOME = "Fixed Income"
    CASH_EQUIVALENTS = "Cash & Cash Equivalents"
    COMMODITIES = "Commodities"
    REAL_ESTATE = "Real Estate"
    ALTERNATIVE = "Alternative"


class Geography(str, Enum):
    """MSCI-based geographic categories for equity allocation analysis."""

    USA = "USA"
    JAPAN = "Japan"
    EUROZONE_EMU = "Eurozone EMU"
    DEVELOPED_EX_USA_EMU_JP = "Dev ex-USA ex-EMU ex-JP"
    EMERGING_MARKETS = "Emerging Markets"
    OTHER = "Other"


@dataclass
class Holding:
    """A single investment position with required and enriched fields.

    Required fields come from the input file (CSV/XLSX).
    Enriched fields are populated by the DataFetcher during the enrichment phase.

    Attributes:
        isin: 12-character International Securities Identification Number.
        ticker: Yahoo Finance-compatible ticker symbol.
        quantity: Number of units held (must be > 0).
        cost_basis_eur: Total acquisition cost in EUR.
        market_value_eur: Current market value in EUR (from input or enriched).
        currency: Original instrument currency code (e.g. "USD", "EUR").
    """

    # --- Required fields (from input) ---
    isin: str
    ticker: str
    quantity: float
    cost_basis_eur: float
    market_value_eur: float
    currency: str

    # --- Enriched fields (populated by DataFetcher) ---
    name: Optional[str] = None
    instrument_type: Optional[str] = None
    security_type: Optional[str] = None
    asset_type: Optional[str] = None
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    weight_pct: Optional[float] = None
    gain_pct: Optional[float] = None
    asset_class: Optional[AssetClass] = None
    geography: Optional[Geography] = None
    geo_breakdown: Optional[dict[Geography, float]] = None
    geo_source: Optional[str] = None
    input_geo: Optional[dict[Geography, float]] = None
    input_geo_source: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    ter: Optional[float] = None
    yield_pct: Optional[float] = None
    duration: Optional[float] = None
    data_source: Optional[str] = None
    target_equities: Optional[float] = None  # target weight as % of equity portion
    target_fixed_income: Optional[float] = None  # target weight as % of fixed income portion
    no_buy_no_sell: bool = False  # if True, exclude from rebalancing actions
    fetch_timestamp: Optional[datetime] = None
    price_history: Optional[pd.Series] = field(default=None, repr=False)

    def is_enriched(self) -> bool:
        """Return True if this holding has been enriched with market data."""
        return self.current_price is not None

    def unrealized_gain_eur(self) -> float:
        """Compute unrealized P&L in EUR."""
        value = self.current_value if self.current_value is not None else self.market_value_eur
        return value - self.cost_basis_eur
