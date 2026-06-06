"""Tests for the Order domain model."""

from __future__ import annotations

import datetime

import pytest

from tarzan.models.order import Order, OrderType


def _order(otype: OrderType, quantity: float = 0.0, net_eur: float = 0.0) -> Order:
    return Order(
        date=datetime.date(2026, 1, 15),
        trade_date=datetime.date(2026, 1, 13),
        type=otype,
        isin="IT0005542359",
        name="BTP",
        ticker="",
        quantity=quantity,
        currency="EUR",
        price_native=99.5,
        fx_rate=1.0,
        gross_eur=9950.0,
        fees_eur=0.0,
        net_eur=net_eur,
        source="fineco",
    )


class TestOrderTypeFromRaw:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("buy", OrderType.BUY),
            ("SELL", OrderType.SELL),
            ("  Coupon ", OrderType.COUPON),
            ("transfer_in", OrderType.TRANSFER_IN),
            ("transfer_out", OrderType.TRANSFER_OUT),
            ("dividend", OrderType.DIVIDEND),
        ],
    )
    def test_known_types(self, raw, expected):
        assert OrderType.from_raw(raw) is expected

    @pytest.mark.parametrize("raw", ["", "split", "rights_issue", None])
    def test_unknown_types_return_none(self, raw):
        assert OrderType.from_raw(raw) is None


class TestOrderHelpers:
    @pytest.mark.parametrize(
        "otype,is_cf,is_pos",
        [
            (OrderType.BUY, True, True),
            (OrderType.SELL, True, True),
            (OrderType.COUPON, True, False),
            (OrderType.DIVIDEND, True, False),
            (OrderType.TRANSFER_IN, False, True),
            (OrderType.TRANSFER_OUT, False, True),
        ],
    )
    def test_classification(self, otype, is_cf, is_pos):
        o = _order(otype)
        assert o.is_cashflow() is is_cf
        assert o.is_position_change() is is_pos
