"""Tests for data/loader.py — CSV parsing and validation."""

from __future__ import annotations

import io

import pytest

from tarzan.data.loader import load_config, load_orders


def _csv_bytesio(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


class TestOrderLoader:
    def test_explicit_instrument_equivalence_group_is_preserved(self):
        source = _csv_bytesio(
            "date,type,isin,quantity,gross_eur,net_eur,instrument_kind,"
            "instrument_equivalence_group\n"
            "2025-01-01,transfer_in,IT0005565392,20000,20000,0,BOND,"
            "BTP-VALORE-2028-CUM-EX\n"
        )

        orders = load_orders(source, filename="order_list.csv", strict=True)

        assert len(orders) == 1
        assert orders[0].instrument_equivalence_group == (
            "BTP-VALORE-2028-CUM-EX"
        )


class TestConfigLoader:
    def test_config_from_csv(self):
        csv = """key,value
rebalancing_lump_sum_amount_eur,1000
rebalancing_target_tolerance_pctg,3.5
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
        assert config.rebalancing_target_tolerance_pctg == 3.5
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


class TestParseNumber:
    """The shared numeric parser must accept both US and European
    notation without the old silent '1,5' → 15 corruption."""

    def _p(self, val):
        from tarzan.data.loader import _parse_number
        return _parse_number(val)

    def test_us_format(self):
        assert self._p("1,234.56") == pytest.approx(1234.56)

    def test_european_format(self):
        assert self._p("1.234,56") == pytest.approx(1234.56)

    def test_european_decimal_no_grouping(self):
        assert self._p("1234,56") == pytest.approx(1234.56)

    def test_european_small_decimal_not_inflated(self):
        # Regression: "1,5" must be 1.5, not 15.
        assert self._p("1,5") == pytest.approx(1.5)

    def test_single_comma_is_european_decimal(self):
        # A single comma is read as the European decimal mark (Fineco source).
        # Regression: "99,750" is a 99.75 bond clean price, NOT 99750; and
        # "1,234" is 1.234, NOT 1234. Real numeric cells never reach this
        # string path (pandas types them float64), so this only affects
        # hand-edited European values, which decimal-first parses correctly.
        assert self._p("99,750") == pytest.approx(99.75)
        assert self._p("1,234") == pytest.approx(1.234)

    def test_multi_comma_is_us_grouping(self):
        # Multiple commas are unambiguous US grouping (EU uses dots to group).
        assert self._p("1,234,567") == pytest.approx(1234567.0)

    def test_single_dot_stays_decimal(self):
        # A single dot is ALWAYS a decimal mark, never grouping: real ETF
        # prices are genuinely 3-decimal (31.975), so this must not become
        # 31975. Guards against a 1000x error in the opposite direction.
        assert self._p("31.975") == pytest.approx(31.975)
        assert self._p("1.234") == pytest.approx(1.234)

    def test_negative_european(self):
        assert self._p("-1.234,5") == pytest.approx(-1234.5)

    def test_plain_us_decimal_preserved(self):
        assert self._p("9591.50472") == pytest.approx(9591.50472)

    def test_multi_group_european(self):
        assert self._p("1.234.567,89") == pytest.approx(1234567.89)

    def test_passthrough_numeric(self):
        assert self._p(42) == 42.0
        assert self._p(3.14) == pytest.approx(3.14)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self._p("")

    def test_non_finite_raises(self):
        # A blank numeric cell reaches the parser as numpy/pandas NaN (a
        # float); it must be rejected, not silently returned as an
        # Order(quantity=nan) that poisons NAV/weights/XIRR to NaN.
        import numpy as np
        for bad in (np.float64("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                self._p(bad)

    def test_safe_wrappers_treat_nan_as_missing(self):
        # _parse_number_safe/_optional must map a NaN cell to None so the
        # loader skips the row (required fields) or defaults it (fees -> 0.0).
        import numpy as np
        from tarzan.data.loader import _parse_number_safe, _parse_number_optional
        assert _parse_number_safe(np.float64("nan"), "quantity", 0) is None
        assert _parse_number_optional(np.float64("nan")) is None
        assert (_parse_number_safe(np.float64("nan"), "fees_eur", 0) or 0.0) == 0.0
