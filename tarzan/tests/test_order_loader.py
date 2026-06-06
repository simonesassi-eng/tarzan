"""Tests for load_orders in the data loader."""

from __future__ import annotations

import io

import pytest

from tarzan.data.loader import load_orders
from tarzan.exceptions import DataIngestionError
from tarzan.models.order import OrderType

_HEADER = (
    "date,trade_date,type,isin,name,ticker,quantity,currency,"
    "price_native,fx_rate,gross_eur,fees_eur,net_eur,source\n"
)


def _csv(*rows: str) -> io.BytesIO:
    return io.BytesIO((_HEADER + "".join(r + "\n" for r in rows)).encode("utf-8"))


class TestLoadOrders:
    def test_valid_rows_parse_to_typed_orders(self):
        buf = _csv(
            "2026-01-15,2026-01-13,buy,IE0006WW1TQ4,XTRS,,100,EUR,35.03,1.0,3503,0,-3503,fineco",
            "2026-02-01,2026-02-01,coupon,IT0005542359,BTP,,0,EUR,,,40,0,40,fineco",
        )
        orders = load_orders(buf, filename="order_list.csv")
        assert len(orders) == 2
        assert orders[0].type is OrderType.BUY
        assert orders[0].isin == "IE0006WW1TQ4"
        assert orders[0].quantity == pytest.approx(100.0)
        assert orders[0].net_eur == pytest.approx(-3503.0)
        # coupon: optional price/fx blank → None
        assert orders[1].type is OrderType.COUPON
        assert orders[1].price_native is None
        assert orders[1].fx_rate is None

    def test_unknown_type_is_skipped(self):
        buf = _csv(
            "2026-01-15,2026-01-13,buy,IE0006WW1TQ4,XTRS,,100,EUR,35.03,1.0,3503,0,-3503,fineco",
            "2026-01-16,2026-01-16,split,IE0006WW1TQ4,XTRS,,0,EUR,,,0,0,0,fineco",
        )
        orders = load_orders(buf, filename="order_list.csv")
        assert len(orders) == 1
        assert orders[0].type is OrderType.BUY

    def test_malformed_row_is_skipped(self):
        # second row: non-numeric quantity → skipped, first row survives
        buf = _csv(
            "2026-01-15,2026-01-13,buy,IE0006WW1TQ4,XTRS,,100,EUR,35.03,1.0,3503,0,-3503,fineco",
            "2026-01-16,2026-01-16,buy,IE0006WW1TQ4,XTRS,,abc,EUR,35,1.0,3500,0,-3500,fineco",
        )
        orders = load_orders(buf, filename="order_list.csv")
        assert len(orders) == 1

    def test_missing_isin_is_skipped(self):
        buf = _csv(
            "2026-01-15,2026-01-13,buy,,XTRS,,100,EUR,35.03,1.0,3503,0,-3503,fineco",
        )
        orders = load_orders(buf, filename="order_list.csv")
        assert orders == []

    def test_missing_path_returns_empty_list(self):
        assert load_orders("/nonexistent/path/order_list.csv") == []

    def test_missing_required_column_raises(self):
        bad = io.BytesIO(b"date,type,isin\n2026-01-15,buy,IE0006WW1TQ4\n")
        with pytest.raises(DataIngestionError):
            load_orders(bad, filename="bad.csv")

    def test_trade_date_defaults_to_date_when_blank(self):
        buf = _csv(
            "2026-01-15,,buy,IE0006WW1TQ4,XTRS,,100,EUR,35.03,1.0,3503,0,-3503,fineco",
        )
        orders = load_orders(buf, filename="order_list.csv")
        assert len(orders) == 1
        assert orders[0].trade_date == orders[0].date
