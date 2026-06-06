"""Tests for the order-derived historical series builder."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from tarzan.engine.returns_builder import (
    QuantityTimeline,
    build_holdings_from_orders,
    build_order_derived_series,
    _open_isins,
    _net_qty_by_isin,
)
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType


def _o(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 1)):
    return Order(
        date=datetime.date(*d),
        trade_date=datetime.date(*d),
        type=otype,
        isin=isin,
        name="X",
        ticker="",
        quantity=qty,
        currency="EUR",
        price_native=price,
        fx_rate=1.0,
        gross_eur=gross,
        fees_eur=0.0,
        net_eur=net,
        source="fineco",
    )


class TestCumExNetting:
    def test_cum_ex_pair_nets_to_closed(self):
        # Same 9-char prefix, opposite quantities → both closed.
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005565392", qty=20000.0, gross=20000.0),
            _o(OrderType.SELL, "IT0005565400", qty=-20000.0, net=20100.0),
        ]
        assert _open_isins(_net_qty_by_isin(orders)) == set()

    def test_distinct_isins_both_open(self):
        orders = [
            _o(OrderType.BUY, "IE0006WW1TQ4", qty=100.0, net=-3500.0),
            _o(OrderType.BUY, "IT0005542359", qty=4000.0, net=-4000.0),
        ]
        assert _open_isins(_net_qty_by_isin(orders)) == {"IE0006WW1TQ4", "IT0005542359"}

    def test_partial_prefix_group_stays_open(self):
        # Three variants of one prefix that net to non-zero → the
        # non-zero ones stay open.
        orders = [
            _o(OrderType.BUY, "IE00BL25JL35", qty=157.0, net=-1000.0),
            _o(OrderType.BUY, "IE00BL25JM42", qty=165.0, net=-1000.0),
        ]
        assert _open_isins(_net_qty_by_isin(orders)) == {"IE00BL25JL35", "IE00BL25JM42"}


class TestBuildHoldings:
    def test_derives_open_holdings_with_seed(self):
        orders = [
            _o(OrderType.BUY, "IE0006WW1TQ4", qty=100.0, net=-3500.0, price=35.0),
        ]
        holdings = build_holdings_from_orders(orders)
        assert len(holdings) == 1
        h = holdings[0]
        assert h.isin == "IE0006WW1TQ4"
        assert h.quantity == pytest.approx(100.0)
        # non-bond seed = qty * price
        assert h.market_value_eur == pytest.approx(3500.0)

    def test_bond_seed_uses_per_100(self):
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005542359", qty=4000.0,
               gross=4000.0, price=100.0),
        ]
        holdings = build_holdings_from_orders(orders)
        # bond seed = qty * price / 100 = 4000 * 100 / 100 = 4000
        assert holdings[0].market_value_eur == pytest.approx(4000.0)


class TestQuantityTimeline:
    def test_qty_at_steps(self):
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "AAA", qty=50.0, d=(2025, 6, 1)),
            _o(OrderType.SELL, "AAA", qty=-30.0, d=(2025, 9, 1)),
        ]
        tl = QuantityTimeline(orders)
        assert tl.qty_at("AAA", datetime.date(2024, 12, 31)) == 0.0
        assert tl.qty_at("AAA", datetime.date(2025, 3, 1)) == 100.0
        assert tl.qty_at("AAA", datetime.date(2025, 7, 1)) == 150.0
        assert tl.qty_at("AAA", datetime.date(2025, 10, 1)) == 120.0


def _enriched_with_history(isin, prices, start=(2025, 1, 1)):
    idx = pd.date_range(start=datetime.date(*start), periods=len(prices), freq="D")
    h = Holding(isin=isin, ticker=isin, quantity=0.0, cost_basis_eur=0.0,
                market_value_eur=0.0, currency="EUR")
    h.price_history = pd.Series(prices, index=idx)
    return h


class TestFallbackLadder:
    def test_yfinance_rung_used_when_history_present(self):
        orders = [_o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0,
                     d=(2025, 1, 1))]
        enriched = {"AAA": _enriched_with_history("AAA", [100.0, 101.0, 102.0])}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 1, 3))
        assert "AAA" in res.provenance["yfinance"]
        assert res.coverage_pct == pytest.approx(100.0, abs=1e-6)

    def test_synthetic_rung_when_no_history(self):
        # No enriched history → must interpolate between two order prices.
        orders = [
            _o(OrderType.BUY, "BBB", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "BBB", qty=10.0, net=-1200.0, price=120.0, d=(2025, 3, 1)),
        ]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 2, 1))
        # Between Jan 1 (100) and Mar 1 (120), Feb 1 interpolates inside.
        assert "BBB" in res.provenance["synthetic"]
        assert res.coverage_pct == pytest.approx(0.0)  # no real market data

    def test_carry_flat_with_single_price(self):
        orders = [_o(OrderType.BUY, "CCC", qty=10.0, net=-1000.0, price=100.0,
                     d=(2025, 1, 1))]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 6, 1))
        assert "CCC" in res.provenance["carry_flat"]

    def test_excluded_when_no_price(self):
        orders = [_o(OrderType.BUY, "DDD", qty=10.0, net=-1000.0, price=None,
                     d=(2025, 1, 1))]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 6, 1))
        assert "DDD" in res.provenance["excluded"]


class TestCumExConservationProperty:
    @given(
        face=st.floats(min_value=1000.0, max_value=1e6),
        price=st.floats(min_value=80.0, max_value=120.0),
    )
    def test_cum_ex_contributes_zero_principal(self, face, price):
        # Property 5: a transfer-in "cum" later sold "ex" with equal face
        # nets to a closed position → not valued at all.
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005565392", qty=face,
               gross=face * price / 100.0, price=price, d=(2025, 1, 1)),
            _o(OrderType.SELL, "IT0005565400", qty=-face,
               net=face * price / 100.0, price=price, d=(2025, 6, 1)),
        ]
        assert _open_isins(_net_qty_by_isin(orders)) == set()
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 12, 1))
        # No open ISIN → terminal valuation is zero principal.
        assert res.valuations[-1][1] == pytest.approx(0.0)
