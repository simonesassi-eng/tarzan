"""Tests for data/loader.py — CSV parsing and validation."""

from __future__ import annotations

import io

import pytest

from tarzan.data.loader import load_config, load_holdings
from tarzan.exceptions import DataIngestionError


def _csv_bytesio(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


class TestHoldingsLoader:
    def test_load_valid_csv(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency
US0000000001,AAA.MI,100,5000.00,6000.00,EUR
US0000000002,BBB.MI,50,2500.00,2600.00,EUR
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")

        assert len(holdings) == 2
        assert holdings[0].isin == "US0000000001"
        assert holdings[0].quantity == 100.0
        assert holdings[0].cost_basis_eur == 5000.00
        assert holdings[1].ticker == "BBB.MI"

    def test_missing_required_column_raises(self):
        csv = """isin,ticker,quantity,currency
US0000000001,AAA.MI,100,EUR
"""
        with pytest.raises(DataIngestionError, match="Missing required columns"):
            load_holdings(_csv_bytesio(csv), filename="test.csv")

    def test_numeric_parsing_with_comma_thousands(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency
US0000000001,AAA.MI,"1,000.0","5,742.07","6,099.00",EUR
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")

        assert holdings[0].quantity == 1000.0
        assert holdings[0].cost_basis_eur == 5742.07
        assert holdings[0].market_value_eur == 6099.00

    def test_skip_zero_quantity_rows(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency
US0000000001,AAA.MI,100,5000,6000,EUR
US0000000002,BBB.MI,0,0,0,EUR
US0000000003,CCC.MI,50,2500,2600,EUR
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")

        assert len(holdings) == 2
        tickers = [h.ticker for h in holdings]
        assert "BBB.MI" not in tickers

    def test_currency_uppercased(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency
US0000000001,AAA.MI,100,5000,6000,eur
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")
        assert holdings[0].currency == "EUR"

    def test_no_buy_no_sell_flag(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency,no_buy_no_sell
US0000000001,AAA.MI,100,5000,6000,EUR,TRUE
US0000000002,BBB.MI,50,2500,2600,EUR,FALSE
US0000000003,CCC.MI,50,2500,2600,EUR,
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")

        assert holdings[0].no_buy_no_sell is True
        assert holdings[1].no_buy_no_sell is False
        assert holdings[2].no_buy_no_sell is False

    def test_ticker_defaults_to_isin_when_empty(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency
US0000000001,,100,5000,6000,EUR
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")
        assert holdings[0].ticker == "US0000000001"

    def test_target_equities_parsed(self):
        csv = """isin,ticker,quantity,cost_basis_eur,market_value_eur,currency,target_equities
US0000000001,AAA.MI,100,5000,6000,EUR,25.5
"""
        holdings = load_holdings(_csv_bytesio(csv), filename="test.csv")
        assert holdings[0].target_equities == 25.5


class TestConfigLoader:
    def test_config_from_csv(self):
        csv = """key,value
rebalancing_lump_sum_amount_eur,1000
rebalancing_max_tolerance_pctg,3.5
target_invested_allocation_equities_pctg,60
target_invested_allocation_fixed_income_pctg,30
target_invested_allocation_gold_pctg,10
target_equity_geo_usa_pctg,50
target_equity_geo_japan_pctg,10
target_equity_geo_eurozone_emu_pctg,20
target_equity_geo_dev_ex_usa_ex_emu_ex_jp_pctg,10
target_equity_geo_emerging_markets_pctg,10
"""
        config = load_config(_csv_bytesio(csv))

        assert config.rebalancing_lump_sum_amount_eur == 1000.0
        assert config.rebalancing_max_tolerance_pctg == 3.5
        assert config.invested_allocation_targets_pctg["Equities"] == 60.0
        assert config.equity_geo_targets_pctg["USA"] == 50.0

    def test_config_normalizes_to_100_if_off(self):
        """If allocations don't sum to 100, they get normalized."""
        csv = """key,value
target_invested_allocation_equities_pctg,50
target_invested_allocation_fixed_income_pctg,25
target_invested_allocation_gold_pctg,25
target_equity_geo_usa_pctg,50
target_equity_geo_japan_pctg,10
target_equity_geo_eurozone_emu_pctg,20
target_equity_geo_dev_ex_usa_ex_emu_ex_jp_pctg,10
target_equity_geo_emerging_markets_pctg,10
"""
        config = load_config(_csv_bytesio(csv))

        total = sum(config.invested_allocation_targets_pctg.values())
        assert abs(total - 100.0) < 0.01

    def test_config_boolean_no_sell(self):
        csv = """key,value
rebalancing_no_sell,TRUE
"""
        config = load_config(_csv_bytesio(csv))
        assert config.rebalancing_no_sell is True

    def test_config_none_returns_defaults(self):
        config = load_config(None)
        assert config is not None
        # Should have sensible defaults
        assert config.rebalancing_lump_sum_amount_eur == 0.0
