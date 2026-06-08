"""Tests for the shared bond valuation primitives in bond_fetcher.

Covers the single source of truth for the per-100-nominal bond
convention used by both the current-value and historical paths
(design Property 2: valuation consistency).
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from tarzan.data.bond_fetcher import (
    bond_price_to_value,
    is_bond,
    looks_like_bond_from_orders,
    value_position,
)
from tarzan.models.holding import AssetClass


class TestValuePosition:
    def test_non_bond_is_quantity_times_price(self):
        assert value_position(10.0, 60.0, bond=False) == pytest.approx(600.0)

    def test_bond_applies_per_100_nominal(self):
        # 10,000 nominal at a clean price of 99.5 → 9,950 EUR.
        assert value_position(10_000.0, 99.5, bond=True) == pytest.approx(9_950.0)

    def test_bond_price_to_value_is_alias(self):
        assert bond_price_to_value(103.99, 4_000.0) == pytest.approx(
            value_position(4_000.0, 103.99, bond=True)
        )

    def test_zero_quantity_is_zero(self):
        assert value_position(0.0, 123.0, bond=True) == 0.0
        assert value_position(0.0, 123.0, bond=False) == 0.0


class TestIsBond:
    def test_asset_class_enum_fixed_income(self):
        assert is_bond(asset_class=AssetClass.FIXED_INCOME) is True

    def test_asset_class_string_fixed_income(self):
        assert is_bond(asset_class="Fixed Income") is True

    def test_asset_class_equities_is_not_bond(self):
        assert is_bond(asset_class=AssetClass.EQUITIES) is False

    @pytest.mark.parametrize("sig", ["BOND", "Government Bond", "US TREASURY", "Corp Note"])
    def test_instrument_type_keywords(self, sig):
        assert is_bond(instrument_type=sig) is True

    def test_quote_type_and_sec_type(self):
        assert is_bond(quote_type="BOND") is True
        assert is_bond(sec_type="Treasury") is True

    def test_equity_signals_are_not_bond(self):
        assert is_bond(quote_type="ETF", sec_type="Equity") is False

    def test_no_signal_defaults_false(self):
        assert is_bond() is False

    def test_openfigi_market_sector_govt(self):
        # OpenFIGI's authoritative marketSector is the strongest signal.
        assert is_bond(market_sector="Govt") is True
        assert is_bond(market_sector="Corp") is True

    def test_openfigi_market_sector_equity_not_bond(self):
        assert is_bond(market_sector="Equity") is False

    def test_openfigi_sec_type_bond(self):
        assert is_bond(figi_sec_type="Global Sovereign") is True
        assert is_bond(figi_sec_type="Corporate Bond") is True

    def test_whole_word_avoids_note_substring_false_positive(self):
        # "Notebook" contains "NOTE" as a substring but is not a bond; the
        # whole-word matcher must not flag it (the old substring match did).
        assert is_bond(instrument_type="Notebook Makers Inc") is False
        # But a standalone "Note" token still counts.
        assert is_bond(instrument_type="Medium Term Note") is True


class TestLooksLikeBondFromOrders:
    def test_typical_btp(self):
        # clean price ~100, large nominal → bond
        assert looks_like_bond_from_orders(avg_price=99.8, avg_qty=10_000.0) is True

    def test_etf_in_price_band_but_small_qty(self):
        # ETF unit price can be in 50-150 but with retail share counts
        assert looks_like_bond_from_orders(avg_price=85.0, avg_qty=120.0) is False

    def test_price_above_band(self):
        assert looks_like_bond_from_orders(avg_price=350.0, avg_qty=10_000.0) is False

    def test_boundaries_inclusive(self):
        assert looks_like_bond_from_orders(avg_price=50.0, avg_qty=1000.0) is True
        assert looks_like_bond_from_orders(avg_price=150.0, avg_qty=1000.0) is True


# ── Property-based (Property 2: valuation consistency) ──────────────────────

_finite_qty = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_finite_price = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


class TestValuePositionProperties:
    @given(q=_finite_qty, p=_finite_price, bond=st.booleans())
    def test_linear_in_quantity(self, q, p, bond):
        # value_position is linear in quantity: scaling qty by k scales value by k.
        base = value_position(q, p, bond=bond)
        scaled = value_position(2.0 * q, p, bond=bond)
        assert scaled == pytest.approx(2.0 * base, rel=1e-9, abs=1e-6)

    @given(q=_finite_qty, p=_finite_price)
    def test_bond_is_exactly_non_bond_over_100(self, q, p):
        # The /100 is applied iff bond, and nowhere else.
        assert value_position(q, p, bond=True) == pytest.approx(
            value_position(q, p, bond=False) / 100.0, rel=1e-9, abs=1e-9
        )
