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
from tarzan.models.holding import Holding, AssetClass
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


def _enriched_borsa_bond(isin, current_price, qty=0.0):
    """An enriched bond Holding priced ONLY by Borsa Italiana: no yfinance
    price_history, but a current_price (already EUR-per-unit, post /100)
    and a borsa_italiana data_source, exactly as the enricher's
    _try_terrapin_fallback leaves it."""
    h = Holding(isin=isin, ticker=isin, quantity=qty, cost_basis_eur=0.0,
                market_value_eur=0.0, currency="EUR")
    h.price_history = None
    h.current_price = current_price
    h.current_value = qty * current_price
    h.data_source = "borsa_italiana/mot/btp"
    h.asset_class = AssetClass.FIXED_INCOME
    h.instrument_type = "Government Bond"
    return h


class TestBorsaItalianaRung:
    """A bond with no yfinance history but a Borsa Italiana today-price
    must be valued at that price on the terminal date (source
    'borsa_italiana'), counting as real market coverage, while historical
    dates still fall back to carry_flat/synthetic."""

    def test_terminal_value_uses_borsa_price_and_per_100(self):
        # qty=1000 nominal, Borsa clean price 103.84 → enricher stores it
        # as 1.0384 EUR-per-unit. Terminal value must be 1000 * 1.0384 =
        # 1038.40, NOT 1000 * 1.0384 / 100.
        isin = "IT0005542359"
        orders = [
            _o(OrderType.BUY, isin, qty=1000.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1)),
        ]
        enriched = {isin: _enriched_borsa_bond(isin, current_price=1.0384,
                                                qty=1000.0)}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 6, 1))
        # Terminal valuation priced by Borsa Italiana.
        assert isin in res.provenance["borsa_italiana"]
        assert res.valuations[-1][1] == pytest.approx(1038.40)
        # Real market coverage reaches 100%.
        assert res.coverage_pct == pytest.approx(100.0, abs=1e-6)

    def test_historical_dates_still_use_carry_flat(self):
        # The Borsa scrape is a single point (today only); historical
        # valuations for the bond must still come from carry_flat/synthetic.
        isin = "IT0005542359"
        orders = [
            _o(OrderType.BUY, isin, qty=1000.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1)),
        ]
        enriched = {isin: _enriched_borsa_bond(isin, current_price=1.0384,
                                               qty=1000.0)}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 6, 1))
        # The single order price (100, one point) carries flat on the
        # historical trade-date valuation → 1000 * 100 / 100 = 1000.
        jan_val = next(v for d, v in res.valuations if d == datetime.date(2025, 1, 1))
        assert jan_val == pytest.approx(1000.0)
        # Provenance for the terminal date is borsa_italiana, and the
        # historical fallback (carry_flat) is also recorded for the
        # synthetic price series used on trade dates.
        assert isin in res.provenance["borsa_italiana"]

    def test_borsa_price_ignored_without_borsa_source(self):
        # If current_price is set but data_source is NOT borsa_italiana
        # (e.g. a stale CSV seed), it must NOT be treated as a market
        # quote: the ISIN falls back to carry_flat, not borsa_italiana.
        isin = "IT0005542359"
        orders = [
            _o(OrderType.BUY, isin, qty=1000.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1)),
        ]
        h = _enriched_borsa_bond(isin, current_price=1.0384, qty=1000.0)
        h.data_source = "input_csv (no market data)"
        res = build_order_derived_series(
            orders, {isin: h}, today=datetime.date(2025, 6, 1))
        assert isin not in res.provenance["borsa_italiana"]
        assert isin in res.provenance["carry_flat"]
        # No real market data → 0% coverage.
        assert res.coverage_pct == pytest.approx(0.0)


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


class TestMarketPricedFlowsNoJump:
    """A trade valued at market price must not inject a fictitious TWROR
    jump (Option 2): buying more of a flat-priced holding leaves the
    chained period return at ~0."""

    def test_same_day_buy_on_flat_prices_is_neutral(self):
        from tarzan.engine.metrics import twror

        # AAA: flat at 100 the whole window. Buy 10 on day 1, buy 10 more
        # on day 15 (a mid-window trade), prices never move.
        enriched = {"AAA": _enriched_with_history(
            "AAA", [100.0] * 40, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 15)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 2, 1))
        res = twror(series.valuations, series.external_flows, series.span_days)
        # Flat prices → the buy must not create a positive/negative return.
        assert res.cumulative_pct == pytest.approx(0.0, abs=1e-6)

    def test_real_growth_is_captured(self):
        from tarzan.engine.metrics import twror

        # AAA rises 100 → 110 over the window, single initial buy.
        prices = [100.0 + i * (10.0 / 30.0) for i in range(31)]
        enriched = {"AAA": _enriched_with_history("AAA", prices, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 1, 31))
        res = twror(series.valuations, series.external_flows, series.span_days)
        assert res.cumulative_pct == pytest.approx(10.0, abs=0.5)


class TestRoundTripInclusion:
    """A position opened and fully closed inside the window must still
    contribute its holding-period market move to TWROR (Lotto 3 #2)."""

    def test_closed_position_contributes_to_history(self):
        # Buy AAA at 100 on Jan 1, sell all at 110 on Jan 31. Closed today,
        # but its +10% move over the window must be visible in valuations.
        prices = [100.0 + i * (10.0 / 30.0) for i in range(40)]
        enriched = {"AAA": _enriched_with_history("AAA", prices, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.SELL, "AAA", qty=-10.0, net=1100.0, price=110.0, d=(2025, 1, 31)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 2, 10))
        # The position is closed today…
        assert _open_isins(_net_qty_by_isin(orders)) == set()
        # …yet its buy and sell are recorded as external flows.
        assert datetime.date(2025, 1, 1) in series.external_flows
        assert datetime.date(2025, 1, 31) in series.external_flows
        # And its holding-period market move is visible in the daily
        # series (non-zero while held), instead of being dropped entirely
        # as it was before the open-today gate was removed.
        ds = series.daily_series
        held = ds.loc[pd.Timestamp(2025, 1, 2):pd.Timestamp(2025, 1, 30)]
        assert (held > 0).all()


class TestDailySeries:
    """The daily flow-adjusted NAV index used for risk metrics."""

    def test_is_dense_daily(self):
        prices = [100.0 + i * 0.1 for i in range(40)]
        enriched = {"AAA": _enriched_with_history("AAA", prices, start=(2025, 1, 1))}
        orders = [_o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0,
                     d=(2025, 1, 1))]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 1, 31))
        ds = series.daily_series
        assert not ds.empty
        # One point per calendar day from first trade to today.
        gaps = ds.index.to_series().diff().dropna().dt.days
        assert (gaps == 1).all()

    def test_deposit_is_not_a_market_gain(self):
        # Flat prices, but a second buy doubles the position mid-window.
        # A raw value series would jump +100% on the deposit day; the
        # flow-adjusted index must stay flat (no market move).
        prices = [100.0] * 40
        enriched = {"AAA": _enriched_with_history("AAA", prices, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 15)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 1, 31))
        ds = series.daily_series
        # NAV index is flat: first ≈ last despite the deposit.
        assert ds.iloc[-1] == pytest.approx(ds.iloc[0], rel=1e-6)
        # Daily returns are all ~0 → zero volatility, the correct answer.
        assert float(ds.pct_change().dropna().abs().max()) == pytest.approx(0.0, abs=1e-9)

    def test_market_move_shows_in_index(self):
        # 100 → 110 over 30 days, single buy: index should rise ~10%.
        prices = [100.0 + i * (10.0 / 30.0) for i in range(31)]
        enriched = {"AAA": _enriched_with_history("AAA", prices, start=(2025, 1, 1))}
        orders = [_o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0,
                     d=(2025, 1, 1))]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 1, 31))
        ds = series.daily_series
        ret = ds.iloc[-1] / ds.iloc[0] - 1.0
        assert ret == pytest.approx(0.10, abs=0.01)


class TestIncomeInTwror:
    """GIPS total-return convention: coupons/dividends are income earned
    by the held portfolio and must be captured in TWROR, not dropped."""

    def test_coupon_lifts_twror_on_flat_prices(self):
        from tarzan.engine.metrics import twror

        # Flat price (100 throughout): with no income the market return
        # is 0%. A coupon paid mid-window is income → must lift TWROR.
        prices = [100.0] * 60
        enriched = {"BTP": _enriched_with_history("BTP", prices, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "BTP", qty=1000.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.COUPON, "BTP", qty=0.0, net=20.0, d=(2025, 2, 1)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 2, 28))
        res = twror(series.valuations, series.external_flows, series.span_days)
        # Coupon is recorded as a negative external flow (withdrawal from
        # the securities portfolio) so it is added back into V_before.
        assert series.external_flows.get(datetime.date(2025, 2, 1)) == pytest.approx(-20.0)
        # Income makes the time-weighted return strictly positive.
        assert res.cumulative_pct > 0.0

    def test_no_income_stays_flat(self):
        from tarzan.engine.metrics import twror

        prices = [100.0] * 60
        enriched = {"BTP": _enriched_with_history("BTP", prices, start=(2025, 1, 1))}
        orders = [
            _o(OrderType.BUY, "BTP", qty=1000.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 2, 28))
        res = twror(series.valuations, series.external_flows, series.span_days)
        assert res.cumulative_pct == pytest.approx(0.0, abs=1e-6)
