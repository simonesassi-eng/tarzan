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


def _o(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 1),
       td=None):
    return Order(
        date=datetime.date(*d),
        trade_date=datetime.date(*td) if td is not None else datetime.date(*d),
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
        # cost basis derived from the buy's net cash paid (incl. fees)
        assert h.cost_basis_eur == pytest.approx(3500.0)

    def test_bond_seed_uses_per_100(self):
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005542359", qty=4000.0,
               gross=4000.0, price=100.0),
        ]
        holdings = build_holdings_from_orders(orders)
        # bond seed = qty * price / 100 = 4000 * 100 / 100 = 4000
        assert holdings[0].market_value_eur == pytest.approx(4000.0)


class TestCostBasis:
    """cost_basis_by_isin: average-cost basis of the units still held."""

    def test_single_buy_is_net_cash_paid(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        orders = [_o(OrderType.BUY, "AAA", qty=100.0, net=-1000.0)]
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(1000.0)

    def test_multiple_buys_accumulate(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1000.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1500.0, d=(2025, 2, 1)),
        ]
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(2500.0)

    def test_partial_sell_removes_at_average_cost(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        # Buy 200 @ avg 12.5 (1000 + 1500), then sell 100 → remove 1250.
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1000.0, d=(2025, 1, 1)),
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1500.0, d=(2025, 2, 1)),
            _o(OrderType.SELL, "AAA", qty=-100.0, net=2000.0, d=(2025, 3, 1)),
        ]
        # avg = 2500/200 = 12.5; remove 100*12.5 = 1250 → 1250 left.
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(1250.0)

    def test_sell_at_gain_does_not_inflate_remaining_basis(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        # Realized gain on the sell must not change the basis of the rest.
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1000.0, d=(2025, 1, 1)),
            _o(OrderType.SELL, "AAA", qty=-50.0, net=900.0, d=(2025, 3, 1)),
        ]
        # avg = 10; remove 50*10 = 500 → 500 left (not reduced by proceeds).
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(500.0)

    def test_coupon_does_not_reduce_cost(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, net=-1000.0, d=(2025, 1, 1)),
            _o(OrderType.COUPON, "AAA", qty=0.0, net=50.0, d=(2025, 6, 1)),
        ]
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(1000.0)

    def test_transfer_in_uses_gross_when_no_cash(self):
        from tarzan.engine.returns_builder import cost_basis_by_isin
        orders = [
            _o(OrderType.TRANSFER_IN, "AAA", qty=4000.0, gross=4000.0, net=0.0),
        ]
        assert cost_basis_by_isin(orders)["AAA"] == pytest.approx(4000.0)


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


def _enriched_borsa_bond(isin, current_price, qty=0.0, market_value=0.0):
    """An enriched bond Holding priced ONLY by Borsa Italiana: no yfinance
    price_history, but an EUR-per-unit current_price (FX-converted, post
    /100) and a borsa_italiana data_source, exactly as the enricher's
    _try_terrapin_fallback leaves it."""
    h = Holding(isin=isin, ticker=isin, quantity=qty, cost_basis_eur=0.0,
                market_value_eur=market_value, currency="EUR")
    h.price_history = None
    h.current_price = current_price
    h.current_value = qty * current_price
    h.data_source = "borsa_italiana/mot/btp"
    h.asset_class = AssetClass.FIXED_INCOME
    h.instrument_type = "Government Bond"
    return h


class TestBorsaItalianaRung:
    """A bond with no yfinance history but a Borsa Italiana today-price
    (already EUR-per-unit) must be valued at that price on the terminal
    date (source 'borsa_italiana'), counting as real market coverage,
    while historical dates still fall back to carry_flat/synthetic."""

    def test_terminal_value_uses_borsa_price_eur_per_unit(self):
        # EUR bond: qty 4000 nominal, Borsa clean 103.84 → enricher stores
        # 1.0384 EUR-per-unit → terminal value 4000 * 1.0384 = 4153.60.
        isin = "IT0005542359"
        orders = [_o(OrderType.TRANSFER_IN, isin, qty=4000.0, gross=4000.0,
                     price=100.0, d=(2025, 1, 1))]
        enriched = {isin: _enriched_borsa_bond(isin, current_price=1.0384, qty=4000.0)}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 6, 1))
        assert isin in res.provenance["borsa_italiana"]
        assert res.valuations[-1][1] == pytest.approx(4153.60)
        assert res.coverage_pct == pytest.approx(100.0, abs=1e-6)

    def test_foreign_currency_bond_not_inflated(self):
        # Regression: a ZAR EIB note, qty 110000 nominal. The enricher has
        # already FX-converted the Borsa ZAR clean price to EUR-per-unit
        # (≈ 0.99 ZAR/100 ÷ 19.2 ZAR/EUR ≈ 0.0516 EUR-per-unit). Terminal
        # value must be ≈ 110000 * 0.0516 ≈ 5676, NOT 110000 * 0.99 ≈
        # 108900 (the bug that came from skipping the FX conversion).
        isin = "XS2105803527"
        orders = [_o(OrderType.TRANSFER_IN, isin, qty=110000.0,
                     gross=5624.0, price=98.14, d=(2025, 1, 1))]
        # EUR-per-unit after the enricher's FX + /100 conversion.
        eur_per_unit = 0.05159
        enriched = {isin: _enriched_borsa_bond(isin, current_price=eur_per_unit,
                                               qty=110000.0)}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 6, 1))
        terminal = res.valuations[-1][1]
        assert terminal == pytest.approx(110000 * eur_per_unit, rel=1e-6)
        assert 4000 < terminal < 8000  # sane EUR value, not ~108k

    def test_historical_dates_still_use_carry_flat(self):
        isin = "IT0005542359"
        orders = [_o(OrderType.TRANSFER_IN, isin, qty=4000.0, gross=4000.0,
                     price=100.0, d=(2025, 1, 1))]
        enriched = {isin: _enriched_borsa_bond(isin, current_price=1.0384, qty=4000.0)}
        res = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 6, 1))
        jan_val = next(v for d, v in res.valuations if d == datetime.date(2025, 1, 1))
        # Historical date: single order price 100 carried flat → 4000*100/100.
        assert jan_val == pytest.approx(4000.0)
        assert isin in res.provenance["borsa_italiana"]

    def test_borsa_price_ignored_without_borsa_source(self):
        isin = "IT0005542359"
        orders = [_o(OrderType.TRANSFER_IN, isin, qty=4000.0, gross=4000.0,
                     price=100.0, d=(2025, 1, 1))]
        h = _enriched_borsa_bond(isin, current_price=1.0384, qty=4000.0)
        h.data_source = "input_csv (no market data)"
        res = build_order_derived_series(
            orders, {isin: h}, today=datetime.date(2025, 6, 1))
        assert isin not in res.provenance["borsa_italiana"]
        assert isin in res.provenance["carry_flat"]


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

    def test_cum_ex_with_different_leg_prices_nets_to_zero(self):
        # Regression: the cum leg arrives at 100 and the ex leg is sold at
        # 105 (the real BTP reclassification case). The group nets to zero
        # quantity, so the terminal valuation must be exactly 0 — pricing
        # each leg at its own carry-flat price would otherwise leave a
        # spurious residual (here -1,000) that desynced the returns-path
        # valuation from the snapshot and pushed coverage above 100%.
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005565392", qty=20000.0,
               gross=20000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.SELL, "IT0005565400", qty=-20000.0,
               net=21000.0, price=105.0, d=(2025, 6, 1)),
        ]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 12, 1))
        assert res.valuations[-1][1] == pytest.approx(0.0)
        # The closed legs must not be disclosed as fallback-priced, and
        # coverage must never exceed 100%.
        assert "IT0005565392" not in res.provenance["carry_flat"]
        assert "IT0005565400" not in res.provenance["carry_flat"]
        assert res.coverage_pct <= 100.0


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


class TestUnsettledFutureOrders:
    """Returns are keyed on ``trade_date`` (market-exposure date), not the
    settlement ``date``. A trade executed before the run date but settling
    after it (T+2) must be fully reflected in every metric — otherwise the
    cash flow lands while the position it creates is invisible and PnL
    drops by the net unsettled capital (regression: the +€4.5k buy that
    vanished from the terminal value)."""

    def test_trade_before_run_settles_after_is_valued(self):
        today_real = datetime.date.today()
        # Executed yesterday, settles tomorrow (T+2 straddling the run).
        trade = today_real - datetime.timedelta(days=1)
        settle = today_real + datetime.timedelta(days=1)
        start = today_real - datetime.timedelta(days=5)
        enriched = {
            "AAA": _enriched_with_history(
                "AAA", [100.0] * 11,
                start=(start.year, start.month, start.day),
            )
        }
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0,
               d=(settle.year, settle.month, settle.day),
               td=(trade.year, trade.month, trade.day)),
        ]
        series = build_order_derived_series(orders, enriched, today=None)

        # The 10 units (trade_date = yesterday) are held as of the run
        # date, so the terminal valuation is 10 * 100, not 0.
        terminal = series.valuations[-1][1]
        assert terminal == pytest.approx(1000.0)

        # The cash flow is dated on the trade date, not the settlement.
        assert any(d == trade for d, _ in series.xirr_cashflows)

        # PnL = current value + Σ cash flows = 1000 + (-1000) = 0, instead
        # of -1000 when the cash was counted but the asset was not.
        pnl = sum(amount for _, amount in series.xirr_cashflows)
        assert pnl == pytest.approx(0.0, abs=1e-6)

    def test_timeline_keys_on_trade_date(self):
        # Buy executed Jan 1, settling Jan 3: held from Jan 1, not Jan 3.
        orders = [
            _o(OrderType.BUY, "AAA", qty=10.0, net=-1000.0, price=100.0,
               d=(2025, 1, 3), td=(2025, 1, 1)),
        ]
        tl = QuantityTimeline(orders)
        assert tl.qty_at("AAA", datetime.date(2025, 1, 1)) == pytest.approx(10.0)
        assert tl.qty_at("AAA", datetime.date(2025, 1, 2)) == pytest.approx(10.0)


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
