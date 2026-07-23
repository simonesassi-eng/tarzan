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
        "otype,is_pos",
        [
            (OrderType.BUY, True),
            (OrderType.SELL, True),
            (OrderType.COUPON, False),
            (OrderType.DIVIDEND, False),
            (OrderType.TRANSFER_IN, True),
            (OrderType.TRANSFER_OUT, True),
        ],
    )
    def test_classification(self, otype, is_pos):
        o = _order(otype)
        assert o.is_position_change() is is_pos


class TestOrderId:
    def test_default_order_id_is_natural_key(self):
        o = _order(OrderType.BUY, quantity=10, net_eur=-1000)
        assert o.order_id == o.natural_key()
        assert len(o.order_id) == 16  # sha1[:16]

    def test_same_economic_event_same_natural_key(self):
        a = _order(OrderType.BUY, quantity=10, net_eur=-1000)
        b = _order(OrderType.BUY, quantity=10, net_eur=-1000)
        assert a.natural_key() == b.natural_key()

    def test_different_event_different_natural_key(self):
        a = _order(OrderType.BUY, quantity=10, net_eur=-1000)
        b = _order(OrderType.BUY, quantity=11, net_eur=-1100)
        assert a.natural_key() != b.natural_key()


class TestLoaderOrderIds:
    def _csv(self, rows):
        import io
        head = "date,type,isin,quantity,gross_eur,net_eur,currency,price_native,fx_rate\n"
        return io.BytesIO((head + "\n".join(rows)).encode())

    def test_duplicate_rows_kept_but_ids_unique_and_flagged(self):
        import datetime as _dt
        from tarzan.runtime import data_quality as dq
        from tarzan.data.loader import load_orders
        dq.reset()
        rows = [
            "2025-01-01,buy,IE00B4L5Y983,10,1000,-1000,EUR,100,1.0",
            "2025-01-01,buy,IE00B4L5Y983,10,1000,-1000,EUR,100,1.0",  # exact dup
            "2025-02-01,sell,IE00B4L5Y983,-5,500,500,EUR,100,1.0",
        ]
        orders = load_orders(self._csv(rows), "dup.csv")
        assert len(orders) == 3                       # NOT dropped
        assert len({o.order_id for o in orders}) == 3  # ids unique
        # exactly one data-quality warning about the duplicate
        dups = [i for i in dq.issues() if "identical order rows" in i.message]
        assert len(dups) == 1
        assert dups[0].context == "IE00B4L5Y983"

    def test_no_duplicates_no_warning(self):
        from tarzan.runtime import data_quality as dq
        from tarzan.data.loader import load_orders
        dq.reset()
        rows = [
            "2025-01-01,buy,IE00B4L5Y983,10,1000,-1000,EUR,100,1.0",
            "2025-02-01,sell,IE00B4L5Y983,-5,500,500,EUR,100,1.0",
        ]
        orders = load_orders(self._csv(rows), "clean.csv")
        assert len(orders) == 2
        assert not [i for i in dq.issues() if "identical order rows" in i.message]
