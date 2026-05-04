"""Shared test fixtures for all tests."""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.models.investor_config import InvestorConfig


@pytest.fixture
def sample_config() -> InvestorConfig:
    """Minimal investor config for testing."""
    config = InvestorConfig()
    config.allocation_targets = {
        "Equities": 60.0,
        "Fixed Income": 30.0,
        "Cash & Cash Equivalents": 10.0,
    }
    config.geo_allocation = {
        "USA": 50.0,
        "Eurozone EMU": 20.0,
        "Japan": 10.0,
        "Dev ex-USA ex-EMU ex-JP": 10.0,
        "Emerging Markets": 10.0,
    }
    config.rebalancing_min_transaction_eur = 0.0
    config.rebalancing_max_tolerance = 2.0
    config.rebalancing_no_sell = False
    config.rebalancing_lump_sum_amount = 0.0
    return config


@pytest.fixture
def sample_holdings() -> list[Holding]:
    """5 holdings: 3 equity (different geos), 1 bond, 1 cash."""
    return [
        Holding(
            isin="US0000000001", ticker="USA_ETF", quantity=100.0,
            cost_basis_eur=5000.0, market_value_eur=6000.0, currency="EUR",
            name="USA ETF", current_price=60.0, current_value=6000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.USA,
            geo_breakdown={Geography.USA: 100.0},
        ),
        Holding(
            isin="EU0000000001", ticker="EU_ETF", quantity=50.0,
            cost_basis_eur=2000.0, market_value_eur=2000.0, currency="EUR",
            name="Eurozone ETF", current_price=40.0, current_value=2000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.EUROZONE_EMU,
            geo_breakdown={Geography.EUROZONE_EMU: 100.0},
        ),
        Holding(
            isin="EM0000000001", ticker="EM_ETF", quantity=30.0,
            cost_basis_eur=1500.0, market_value_eur=2000.0, currency="EUR",
            name="Emerging ETF", current_price=66.66, current_value=2000.0,
            asset_class=AssetClass.EQUITIES,
            geography=Geography.EMERGING_MARKETS,
            geo_breakdown={Geography.EMERGING_MARKETS: 100.0},
        ),
        Holding(
            isin="BOND00000001", ticker="BOND_ETF", quantity=50.0,
            cost_basis_eur=5000.0, market_value_eur=5000.0, currency="EUR",
            name="Bond ETF", current_price=100.0, current_value=5000.0,
            asset_class=AssetClass.FIXED_INCOME,
        ),
        Holding(
            isin="CASH00000001", ticker="CASH_ETF", quantity=10.0,
            cost_basis_eur=1000.0, market_value_eur=1000.0, currency="EUR",
            name="Cash ETF", current_price=100.0, current_value=1000.0,
            asset_class=AssetClass.CASH_EQUIVALENTS,
        ),
    ]


@pytest.fixture
def sample_price_history() -> pd.Series:
    """5-year synthetic daily price series, linearly growing from 100 to 150."""
    dates = pd.date_range(end="2026-03-01", periods=1260, freq="B")
    import numpy as np
    prices = np.linspace(100, 150, len(dates))
    return pd.Series(prices, index=dates, name="TEST")
