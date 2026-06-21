"""Engine-wiring tests for the optional order list (Properties 1 & 6).

These assert the structural contract without hitting the network:
- no orders → pipeline is exactly today's (identity).
- with orders → the history provider is swapped and _returns appended,
  and the order computers populate the single ctx["portfolio_history"]
  that _performance/_risk read.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType


def _order(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 2)):
    return Order(
        date=datetime.date(*d), trade_date=datetime.date(*d), type=otype,
        isin=isin, name="X", ticker="", quantity=qty, currency="EUR",
        price_native=price, fx_rate=1.0, gross_eur=gross, fees_eur=0.0,
        net_eur=net, source="fineco",
    )


def _enriched_holding(isin, qty, prices, start=(2025, 1, 1)):
    idx = pd.date_range(start=datetime.date(*start), periods=len(prices), freq="D")
    h = Holding(isin=isin, ticker=isin, quantity=qty, cost_basis_eur=0.0,
                market_value_eur=qty * prices[-1], currency="EUR",
                name=isin, current_price=prices[-1],
                current_value=qty * prices[-1], asset_class=AssetClass.EQUITIES)
    h.price_history = pd.Series(prices, index=idx)
    return h


class TestProperty1Identity:
    def test_no_orders_keeps_default_pipeline(self, sample_holdings, sample_config):
        engine = MetricsEngine(sample_holdings, sample_config)
        names = [c.__name__ for c in engine._computers]
        assert "_portfolio_history" in names
        assert "_portfolio_history_from_orders" not in names
        assert "_returns" not in names

    def test_orders_swap_provider_and_append_returns(self, sample_holdings, sample_config):
        orders = [_order(OrderType.BUY, "US0000000001", qty=100.0, net=-6000.0)]
        engine = MetricsEngine(sample_holdings, sample_config, orders=orders)
        names = [c.__name__ for c in engine._computers]
        assert "_portfolio_history" not in names
        assert "_portfolio_history_from_orders" in names
        assert "_returns" in names
        # _allocation_timeline is appended last (after _returns) on the
        # order path to feed the newsletter Diversification sparklines.
        assert names[-1] == "_allocation_timeline"
        # Same number of computers as default + 2 appended (the history
        # provider swap is in-place): _returns and _allocation_timeline.
        assert len(names) == 14


class TestProperty6SingleSeries:
    def test_order_computers_share_one_history_series(self, sample_config):
        # Two enriched holdings with real history; orders that open them.
        h1 = _enriched_holding("US0000000001", 100.0, [60.0, 61.0, 62.0])
        h2 = _enriched_holding("EU0000000001", 50.0, [40.0, 40.0, 41.0])
        orders = [
            _order(OrderType.BUY, "US0000000001", qty=100.0, net=-6000.0, price=60.0),
            _order(OrderType.BUY, "EU0000000001", qty=50.0, net=-2000.0, price=40.0),
        ]
        engine = MetricsEngine([h1, h2], sample_config, orders=orders)

        ctx: dict = {}
        engine._portfolio_history_from_orders(ctx)
        # The provider populated the single series the others consume.
        assert "portfolio_history" in ctx
        assert isinstance(ctx["portfolio_history"], pd.Series)
        assert not ctx["portfolio_history"].empty
        # _returns reads the stashed order series and fills metrics.
        engine._returns(ctx)
        assert "twror_pct" in ctx
        assert "xirr_pct" in ctx
        assert ctx["returns_coverage_pct"] == pytest.approx(100.0, abs=1e-6)

        # _performance reads the SAME ctx["portfolio_history"] object.
        series_obj = ctx["portfolio_history"]
        engine._performance(ctx)
        assert ctx["portfolio_history"] is series_obj
