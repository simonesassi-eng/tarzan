"""Tests for the deterministic / as-of run context (tarzan.runtime).

Scope of the guarantee (deliberately): deterministic mode makes the computed
NUMBERS reproducible (pinned clock, no live intraday, no AI) and leaves an
ordinary run byte-for-byte unchanged. It does NOT promise a byte-identical
newsletter — a residual row-order non-determinism remains (holding rows can
tie on weight and their order traces to a set() of open ISINs); that is a
known, documented limitation, not something these tests assert away.
"""

from __future__ import annotations

import datetime

from tarzan import runtime


class TestRuntimeContext:
    def teardown_method(self):
        runtime.reset()  # never leak a pinned context into other tests

    def test_default_is_live_and_not_deterministic(self):
        runtime.reset()
        assert runtime.is_deterministic() is False
        assert runtime.as_of() is None
        assert runtime.today() == datetime.date.today()

    def test_as_of_pins_today_without_deterministic(self):
        runtime.configure(as_of=datetime.date(2026, 6, 30))
        assert runtime.is_deterministic() is False
        assert runtime.today() == datetime.date(2026, 6, 30)

    def test_deterministic_flag_and_pinned_stamp(self):
        runtime.configure(deterministic=True, as_of=datetime.date(2026, 6, 30))
        assert runtime.is_deterministic() is True
        assert runtime.today() == datetime.date(2026, 6, 30)
        # Pinned stamp derives from as_of (midnight), so headers don't vary.
        assert runtime.now_stamp("%Y-%m-%d") == "2026-06-30"

    def test_explicit_stamp_wins(self):
        runtime.configure(deterministic=True, as_of=datetime.date(2026, 6, 30),
                          stamp="FIXED STAMP")
        assert runtime.now_stamp() == "FIXED STAMP"

    def test_reset_restores_live(self):
        runtime.configure(deterministic=True, as_of=datetime.date(2020, 1, 1))
        runtime.reset()
        assert runtime.is_deterministic() is False
        assert runtime.today() == datetime.date.today()


class TestReturnsHonorAsOf:
    """The order-derived series must value AS OF the pinned date, so a pinned
    run is reproducible and can reproduce a historically-reported figure."""

    def teardown_method(self):
        runtime.reset()

    def test_build_series_uses_runtime_today_when_no_explicit_today(self):
        import datetime as _dt
        from tarzan.engine.returns_builder import build_order_derived_series
        from tarzan.models.order import Order, OrderType

        def _o(d):
            return Order(date=d, trade_date=d, type=OrderType.BUY, isin="IE00TEST0001",
                         name="x", ticker="", quantity=10.0, currency="EUR",
                         price_native=100.0, fx_rate=1.0, gross_eur=1000.0,
                         fees_eur=0.0, net_eur=-1000.0, source="t")
        orders = [_o(_dt.date(2026, 1, 5))]

        runtime.configure(as_of=_dt.date(2026, 6, 30))
        s = build_order_derived_series(orders, {})  # no explicit today
        # Terminal valuation date is the pinned as_of, not the live date.
        assert s.valuations[-1][0] == _dt.date(2026, 6, 30)
        assert s.span_days == (_dt.date(2026, 6, 30) - _dt.date(2026, 1, 5)).days


class TestOpenIsinsDeterministicOrder:
    """Regression: build_holdings_from_orders must emit holdings in a stable
    (sorted-ISIN) order, since it derives them from a set() of open ISINs."""

    def test_holdings_sorted_by_isin(self):
        import datetime as _dt
        from tarzan.engine.returns_builder import build_holdings_from_orders
        from tarzan.models.order import Order, OrderType

        def _o(isin):
            return Order(date=_dt.date(2026, 1, 1), trade_date=_dt.date(2026, 1, 1),
                         type=OrderType.BUY, isin=isin, name="x", ticker="",
                         quantity=10.0, currency="EUR", price_native=100.0,
                         fx_rate=1.0, gross_eur=1000.0, fees_eur=0.0,
                         net_eur=-1000.0, source="t")
        # Deliberately unsorted input ISINs.
        orders = [_o("IE00ZZZ00001"), _o("IE00AAA00001"), _o("IE00MMM00001")]
        holdings = build_holdings_from_orders(orders)
        isins = [h.isin for h in holdings]
        assert isins == sorted(isins)  # stable, not set-hash order
