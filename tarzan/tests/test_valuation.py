"""Tests for exact-kind position valuation primitives."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from tarzan.data.bond_fetcher import is_bond, value_position
from tarzan.instruments.registry import InstrumentKind


class TestValuePosition:
    def test_stock_is_quantity_times_price(self):
        assert value_position(
            10.0, 60.0, instrument_kind=InstrumentKind.STOCK
        ) == pytest.approx(600.0)

    def test_etf_is_quantity_times_price(self):
        assert value_position(
            1_000.0, 100.0, instrument_kind=InstrumentKind.ETF
        ) == pytest.approx(100_000.0)

    def test_bond_applies_per_100_nominal(self):
        assert value_position(
            10_000.0, 99.5, instrument_kind=InstrumentKind.BOND
        ) == pytest.approx(9_950.0)

    def test_zero_quantity_is_zero_for_every_exact_kind(self):
        for kind in InstrumentKind:
            assert value_position(0.0, 123.0, instrument_kind=kind) == 0.0

    def test_unknown_kind_is_rejected_not_treated_as_non_bond(self):
        with pytest.raises(ValueError, match="explicit InstrumentKind"):
            value_position(10.0, 100.0, instrument_kind=None)  # type: ignore[arg-type]


class TestIsBond:
    def test_only_exact_bond_kind_selects_bond_mechanics(self):
        assert is_bond(InstrumentKind.BOND) is True
        assert is_bond(InstrumentKind.STOCK) is False
        assert is_bond(InstrumentKind.ETF) is False
        assert is_bond(InstrumentKind.CASH) is False
        assert is_bond(None) is False


_finite_qty = st.floats(
    min_value=0.0,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)
_finite_price = st.floats(
    min_value=0.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_non_bond_kind = st.sampled_from([
    InstrumentKind.STOCK,
    InstrumentKind.ETF,
    InstrumentKind.CASH,
])


class TestValuePositionProperties:
    @given(q=_finite_qty, p=_finite_price, kind=st.sampled_from(list(InstrumentKind)))
    def test_linear_in_quantity(self, q, p, kind):
        base = value_position(q, p, instrument_kind=kind)
        scaled = value_position(2.0 * q, p, instrument_kind=kind)
        assert scaled == pytest.approx(2.0 * base, rel=1e-9, abs=1e-6)

    @given(q=_finite_qty, p=_finite_price, kind=_non_bond_kind)
    def test_bond_is_exactly_unit_priced_value_over_100(self, q, p, kind):
        assert value_position(
            q, p, instrument_kind=InstrumentKind.BOND
        ) == pytest.approx(
            value_position(q, p, instrument_kind=kind) / 100.0,
            rel=1e-9,
            abs=1e-9,
        )
