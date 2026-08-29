"""Tests for the order-derived historical series builder."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from tarzan.engine.returns_builder import (
    InstrumentIdentityConflict,
    QuantityTimeline,
    build_holdings_from_orders,
    build_order_derived_series,
    build_allocation_timeline,
    _instrument_identity_by_isin,
    _open_isins,
    _net_qty_by_isin,
    _order_fx_rate,
    _seed_market_value,
    _build_synthetic_history,
)
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import Holding, AssetClass, Geography
from tarzan.models.order import Order, OrderType
from tarzan.runtime.ledger import Availability


def _o(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 1),
       td=None, kind=InstrumentKind.STOCK, equivalence_group=None,
       currency="EUR", fx=1.0):
    return Order(
        date=datetime.date(*d),
        trade_date=datetime.date(*td) if td is not None else datetime.date(*d),
        type=otype,
        isin=isin,
        name="X",
        ticker="",
        quantity=qty,
        currency=currency,
        price_native=price,
        fx_rate=fx,
        gross_eur=gross,
        fees_eur=0.0,
        net_eur=net,
        source="fineco",
        instrument_kind=kind,
        instrument_equivalence_group=equivalence_group,
    )


def _open_from_orders(orders):
    return _open_isins(
        _net_qty_by_isin(orders),
        _instrument_identity_by_isin(orders),
    )


class TestCumExNetting:
    def test_explicit_cum_ex_pair_nets_to_closed(self):
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "IT0005565392",
                qty=20000.0,
                gross=20000.0,
                equivalence_group="btp-italia-2028",
            ),
            _o(
                OrderType.SELL,
                "IT0005565400",
                qty=-20000.0,
                net=20100.0,
                equivalence_group="BTP-ITALIA-2028",
            ),
        ]
        assert _open_from_orders(orders) == set()

    def test_distinct_isins_both_open(self):
        orders = [
            _o(OrderType.BUY, "IE0006WW1TQ4", qty=100.0, net=-3500.0),
            _o(OrderType.BUY, "IT0005542359", qty=4000.0, net=-4000.0),
        ]
        assert _open_from_orders(orders) == {"IE0006WW1TQ4", "IT0005542359"}

    def test_ungrouped_same_prefix_positions_stay_open(self):
        orders = [
            _o(OrderType.BUY, "IE00BL25JL35", qty=157.0, net=-1000.0),
            _o(OrderType.BUY, "IE00BL25JM42", qty=165.0, net=-1000.0),
        ]
        assert _open_from_orders(orders) == {"IE00BL25JL35", "IE00BL25JM42"}

    def test_equal_opposite_ungrouped_same_prefix_stay_separate(self):
        orders = [
            _o(OrderType.BUY, "IE00BL25JL35", qty=100.0, net=-5000.0),
            _o(OrderType.SELL, "IE00BL25JM42", qty=-100.0, net=3000.0),
        ]
        assert _open_from_orders(orders) == {"IE00BL25JL35", "IE00BL25JM42"}

    def test_cum_ex_rollover_releases_cost_from_the_daily_series(self):
        """A position opened under a cum ISIN and sold under its ex ISIN must
        release its cost from the daily cost-basis series, not just from the
        snapshot.

        Regression: the daily builder walked raw ISINs, so the disposal found
        an empty book under the ex ISIN and never released the opening cost.
        The stranded cost surfaced as a phantom unrealized step (~the opening
        notional) that flattened every since-inception line above it.
        """
        from tarzan.engine.returns_builder import (
            _build_cost_basis_series,
        )

        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005565392", qty=20000.0,
               gross=20000.0, equivalence_group="btp-2028", d=(2026, 1, 8)),
            _o(OrderType.SELL, "IT0005565400", qty=-20000.0, net=21000.0,
               equivalence_group="BTP-2028", d=(2026, 1, 23)),
        ]
        identity = _instrument_identity_by_isin(orders)
        index = pd.date_range("2026-01-08", "2026-01-31", freq="D")

        # Identity-aware: cost is released on the ex-ISIN disposal.
        aware = _build_cost_basis_series(orders, index, identity)
        assert aware.iloc[-1] == pytest.approx(0.0), aware.iloc[-1]
        assert aware.max() == pytest.approx(20000.0)

        # Raw-ISIN walk (no identity): the €20k never leaves the books — the
        # exact defect this guards against.
        raw = _build_cost_basis_series(orders, index, None)
        assert raw.iloc[-1] == pytest.approx(20000.0)

    def test_conflicting_explicit_groups_fail_closed(self):
        orders = [
            _o(
                OrderType.BUY,
                "IT0005565392",
                qty=100.0,
                equivalence_group="document-a",
            ),
            _o(
                OrderType.SELL,
                "IT0005565392",
                qty=-100.0,
                equivalence_group="document-b",
            ),
        ]
        with pytest.raises(
            InstrumentIdentityConflict,
            match="IT0005565392: document-a, document-b",
        ):
            build_holdings_from_orders(orders)


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
               gross=4000.0, price=100.0, kind=InstrumentKind.BOND),
        ]
        holdings = build_holdings_from_orders(orders)
        # bond seed = qty * price / 100 = 4000 * 100 / 100 = 4000
        assert holdings[0].market_value_eur == pytest.approx(4000.0)
        assert holdings[0].security_type == "BOND"

    def test_unknown_order_kind_has_no_guessed_seed(self):
        orders = [
            _o(OrderType.BUY, "UNKNOWN", qty=1000.0, net=-1000.0,
               price=100.0, kind=None),
        ]
        holdings = build_holdings_from_orders(orders)
        assert holdings[0].market_value_eur == 0.0
        assert holdings[0].security_type is None


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

    def test_qty_at_exact_event_date_boundaries(self):
        # A query exactly on an event date must include that day's change.
        orders = [
            _o(OrderType.BUY, "AAA", qty=100.0, d=(2025, 1, 1)),
            _o(OrderType.SELL, "AAA", qty=-30.0, d=(2025, 9, 1)),
        ]
        tl = QuantityTimeline(orders)
        assert tl.qty_at("AAA", datetime.date(2025, 1, 1)) == 100.0
        assert tl.qty_at("AAA", datetime.date(2025, 9, 1)) == 70.0
        assert tl.qty_at("ZZZ", datetime.date(2025, 1, 1)) == 0.0  # unknown ISIN


class TestPriceLookupBinarySearch:
    """The searchsorted-based price lookups must match a naive linear scan
    (the pre-optimization behavior) exactly, on and off the sample points."""

    def _series(self):
        idx = pd.to_datetime(["2025-01-01", "2025-01-10", "2025-02-01", "2025-03-15"])
        return pd.Series([100.0, 110.0, 90.0, 130.0], index=idx)

    def test_price_at_matches_linear_scan(self):
        from tarzan.engine.returns_builder import _price_at
        s = self._series()

        def naive(series, d):
            avail = series.loc[series.index <= pd.Timestamp(d)]
            return None if avail.empty else float(avail.iloc[-1])

        for d in [datetime.date(2024, 12, 31),   # before first
                  datetime.date(2025, 1, 1),     # exact first
                  datetime.date(2025, 1, 5),     # between
                  datetime.date(2025, 2, 1),     # exact middle
                  datetime.date(2025, 4, 1)]:    # after last
            assert _price_at(s, d) == naive(s, d), d

    def test_causal_order_price_uses_only_exact_or_prior_observations(self):
        from tarzan.engine.returns_builder import _causal_order_price

        series = self._series()
        assert _causal_order_price(
            series, datetime.date(2024, 12, 31)
        ) == (None, "excluded")
        assert _causal_order_price(
            series, datetime.date(2025, 1, 1)
        ) == (100.0, "synthetic")
        assert _causal_order_price(
            series, datetime.date(2025, 1, 6)
        ) == (100.0, "carry_flat")
        assert _causal_order_price(
            series, datetime.date(2025, 2, 1)
        ) == (90.0, "synthetic")
        assert _causal_order_price(
            series, datetime.date(2026, 1, 1)
        ) == (130.0, "carry_flat")

    def test_causal_order_price_never_backfills_single_future_point(self):
        from tarzan.engine.returns_builder import _causal_order_price

        series = pd.Series([42.0], index=pd.to_datetime(["2025-06-01"]))
        assert _causal_order_price(
            series, datetime.date(2025, 5, 31)
        ) == (None, "excluded")
        assert _causal_order_price(
            series, datetime.date(2025, 6, 1)
        ) == (42.0, "synthetic")
        assert _causal_order_price(
            series, datetime.date(2025, 6, 2)
        ) == (42.0, "carry_flat")


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
        assert res.history_availability is Availability.AVAILABLE

    def test_future_order_price_is_not_interpolated_into_history(self):
        from tarzan.runtime.ledger import Availability

        orders = [
            _o(OrderType.BUY, "BBB", qty=10.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1)),
            _o(OrderType.BUY, "BBB", qty=10.0, net=-1200.0, price=120.0,
               d=(2025, 3, 1)),
        ]
        today = datetime.date(2025, 2, 1)
        result = build_order_derived_series(
            orders, enriched_by_isin={}, today=today
        )

        assert result.valuations[-1][1] == pytest.approx(1000.0)
        assert result.provenance["synthetic"] == ["BBB"]
        assert result.provenance["carry_flat"] == ["BBB"]
        assert result.history_availability is Availability.DEGRADED
        assert all(date <= today for date, _ in result.xirr_cashflows)

    def test_carry_flat_with_single_price(self):
        orders = [_o(OrderType.BUY, "CCC", qty=10.0, net=-1000.0, price=100.0,
                     d=(2025, 1, 1))]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 6, 1))
        assert "CCC" in res.provenance["carry_flat"]
        assert res.history_availability is Availability.DEGRADED

    def test_excluded_when_no_price(self):
        orders = [_o(OrderType.BUY, "DDD", qty=10.0, net=-1000.0, price=None,
                     d=(2025, 1, 1))]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 6, 1))
        assert "DDD" in res.provenance["excluded"]
        assert res.history_availability is Availability.UNAVAILABLE
        assert res.unavailable_instruments == ("DDD",)

    def test_same_day_round_trip_records_fallback_evidence(self):
        orders = [
            _o(
                OrderType.BUY,
                "SAME-DAY",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                d=(2025, 1, 1),
            ),
            _o(
                OrderType.SELL,
                "SAME-DAY",
                qty=-10.0,
                net=1100.0,
                price=110.0,
                d=(2025, 1, 1),
            ),
        ]

        result = build_order_derived_series(
            orders,
            enriched_by_isin={},
            today=datetime.date(2025, 1, 1),
        )

        assert result.provenance["synthetic"] == ["SAME-DAY"]
        assert result.history_availability is Availability.DEGRADED

    def test_same_day_unpriceable_round_trip_nulls_metrics(self, monkeypatch):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        analysis_date = datetime.date(2025, 1, 1)
        monkeypatch.setattr("tarzan.runtime.today", lambda: analysis_date)
        orders = [
            _o(
                OrderType.BUY,
                "SAME-DAY-MISSING",
                qty=10.0,
                net=-1000.0,
                price=None,
                d=(2025, 1, 1),
            ),
            _o(
                OrderType.SELL,
                "SAME-DAY-MISSING",
                qty=-10.0,
                net=1100.0,
                price=None,
                d=(2025, 1, 1),
            ),
        ]
        engine = MetricsEngine([], InvestorConfig(), orders=orders)
        context = {}

        engine._portfolio_history_from_orders(context)
        engine._returns(context)
        metrics = engine._build_result(context)

        assert context["_order_series"].provenance["excluded"] == [
            "SAME-DAY-MISSING"
        ]
        assert metrics.history_availability == "UNAVAILABLE"
        assert metrics.history_unavailable_instruments == (
            "SAME-DAY-MISSING",
        )
        assert metrics.portfolio_history is None
        assert metrics.xirr_pct is None
        assert metrics.twror_pct is None
        assert metrics.pnl_eur is None
        assert metrics.estimated_cgt_eur is None

    @pytest.mark.parametrize("invalid_price", [float("nan"), float("inf")])
    def test_non_finite_prices_fail_closed_every_history_surface(
        self,
        monkeypatch,
        invalid_price,
    ):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        analysis_date = datetime.date(2025, 1, 3)
        monkeypatch.setattr("tarzan.runtime.today", lambda: analysis_date)
        bad, good = "NON-FINITE", "GOOD-PRICE"
        orders = [
            _o(
                OrderType.BUY,
                bad,
                qty=1000.0,
                net=-1000.0,
                price=invalid_price,
                d=(2025, 1, 1),
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                good,
                qty=10.0,
                net=-500.0,
                price=50.0,
                d=(2025, 1, 1),
                kind=InstrumentKind.STOCK,
            ),
        ]
        invalid_holding = _enriched_with_history(
            bad,
            [invalid_price, invalid_price, invalid_price],
        )
        invalid_holding.asset_class = AssetClass.FIXED_INCOME
        invalid_holding.instrument_type = "Government Bond"
        invalid_holding.current_price = invalid_price
        invalid_holding.data_source = "borsa_italiana/mot/btp"
        valid_holding = _enriched_with_history(good, [50.0, 51.0, 52.0])
        valid_holding.asset_class = AssetClass.EQUITIES
        enriched = {bad: invalid_holding, good: valid_holding}

        series = build_order_derived_series(
            orders,
            enriched,
            today=analysis_date,
        )
        allocation = build_allocation_timeline(
            orders,
            enriched,
            months=1,
            today=analysis_date,
        )
        engine = MetricsEngine(
            [invalid_holding, valid_holding],
            InvestorConfig(),
            orders=orders,
        )
        context = {}
        engine._portfolio_history_from_orders(context)
        engine._returns(context)
        metrics = engine._build_result(context)

        assert series.history_availability is Availability.UNAVAILABLE
        assert series.unavailable_instruments == (bad,)
        assert series.provenance["excluded"] == [bad]
        assert bad not in series.provenance["yfinance"]
        assert bad not in series.provenance["borsa_italiana"]
        assert bad not in series.provenance["synthetic"]
        assert allocation is None
        assert metrics.history_availability == "UNAVAILABLE"
        assert metrics.portfolio_history is None
        assert metrics.returns_coverage_pct is None
        assert metrics.xirr_pct is None
        assert metrics.twror_pct is None
        assert metrics.pnl_eur is None

    def test_terminal_price_does_not_hide_earlier_missing_history(self):
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "LATE-PRICE",
                qty=10.0,
                gross=1000.0,
                price=None,
                d=(2025, 1, 1),
            ),
            _o(
                OrderType.BUY,
                "LATE-PRICE",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                d=(2025, 1, 10),
            ),
        ]

        result = build_order_derived_series(
            orders,
            enriched_by_isin={},
            today=datetime.date(2025, 1, 15),
        )

        assert result.valuations[-1][1] == pytest.approx(2000.0)
        assert "LATE-PRICE" in result.provenance["excluded"]
        assert "LATE-PRICE" in result.provenance["carry_flat"]
        assert result.history_availability is Availability.UNAVAILABLE
        assert result.unavailable_instruments == ("LATE-PRICE",)

    def test_fixed_income_etf_uses_unit_pricing_not_bond_per_100(self):
        orders = [
            _o(OrderType.BUY, "FI-ETF", qty=1000.0, net=-100000.0,
               price=100.0, d=(2025, 1, 1), kind=InstrumentKind.ETF),
        ]
        holding = Holding(
            isin="FI-ETF", ticker="FI-ETF", quantity=1000.0,
            cost_basis_eur=100000.0, market_value_eur=100000.0,
            currency="EUR", security_type="ETF",
            asset_class=AssetClass.FIXED_INCOME,
        )
        res = build_order_derived_series(
            orders, {"FI-ETF": holding}, today=datetime.date(2025, 6, 1)
        )
        assert res.valuations[-1][1] == pytest.approx(100000.0)
        assert "FI-ETF" in res.provenance["carry_flat"]

    def test_unknown_order_kind_is_excluded_instead_of_guessed(self):
        orders = [
            _o(OrderType.BUY, "UNKNOWN-KIND", qty=1000.0, net=-100000.0,
               price=100.0, d=(2025, 1, 1), kind=None),
        ]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 6, 1)
        )
        assert res.valuations[-1][1] == 0.0
        assert "UNKNOWN-KIND" in res.provenance["excluded"]
        assert res.history_availability is Availability.UNAVAILABLE


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
                     price=100.0, d=(2025, 1, 1), kind=InstrumentKind.BOND)]
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
                     gross=5624.0, price=98.14, d=(2025, 1, 1),
                     kind=InstrumentKind.BOND)]
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
                     price=100.0, d=(2025, 1, 1), kind=InstrumentKind.BOND)]
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
                     price=100.0, d=(2025, 1, 1), kind=InstrumentKind.BOND)]
        h = _enriched_borsa_bond(isin, current_price=1.0384, qty=4000.0)
        h.data_source = "input_csv (no market data)"
        res = build_order_derived_series(
            orders, {isin: h}, today=datetime.date(2025, 6, 1))
        assert isin not in res.provenance["borsa_italiana"]
        assert isin in res.provenance["carry_flat"]


class TestOrderFxRate:
    """A foreign order missing its FX rate is UNAVAILABLE, never booked 1:1
    (booking a ZAR/USD native price as EUR overstates value by the FX rate)."""

    def test_eur_needs_no_rate(self):
        assert _order_fx_rate(_o(OrderType.BUY, "X", currency="EUR", fx=None)) == 1.0
        assert _order_fx_rate(_o(OrderType.BUY, "X", currency="", fx=None)) == 1.0

    def test_foreign_with_rate_uses_it(self):
        assert _order_fx_rate(_o(OrderType.BUY, "X", currency="USD", fx=1.08)) == 1.08

    def test_foreign_missing_or_bad_rate_is_unavailable(self):
        assert _order_fx_rate(_o(OrderType.BUY, "X", currency="USD", fx=None)) is None
        assert _order_fx_rate(_o(OrderType.BUY, "X", currency="ZAR", fx=0.0)) is None
        assert _order_fx_rate(
            _o(OrderType.BUY, "X", currency="ZAR", fx=float("nan"))) is None

    def test_synthetic_history_drops_foreign_row_without_rate(self):
        # A ZAR buy at 98 native with no rate must NOT enter the series as 98 EUR.
        isin = "XS2105803527"
        orders = [_o(OrderType.BUY, isin, qty=100.0, price=98.0,
                     d=(2025, 1, 1), currency="ZAR", fx=None,
                     kind=InstrumentKind.BOND)]
        assert _build_synthetic_history(orders, isin) is None

    def test_seed_is_zero_for_foreign_without_rate(self):
        isin = "US91282CGJ45"
        orders = [_o(OrderType.BUY, isin, qty=10.0, price=98.0,
                     d=(2025, 1, 1), currency="USD", fx=None,
                     kind=InstrumentKind.BOND)]
        assert _seed_market_value(orders, isin, 10.0) == 0.0


class TestDistributionDoubleCountGuard:
    """A distribution booked as a cash flow is the GIPS total-return treatment,
    correct ONLY on a PRICE-ONLY series. The yfinance rung is total-return
    (auto_adjust=True), so a DIVIDEND order on such a holding would count the
    income twice — the builder must warn, not silently inflate the return. A
    bond coupon (clean price, Borsa rung) is price-only → no warning."""

    def test_dividend_on_yfinance_series_warns(self):
        from tarzan.runtime import data_quality as dq
        dq.reset()
        isin = "AAA"
        orders = [
            _o(OrderType.BUY, isin, qty=10.0, net=-1000.0, price=100.0, d=(2025, 1, 1)),
            _o(OrderType.DIVIDEND, isin, net=19.0, d=(2025, 1, 2)),
        ]
        enriched = {isin: _enriched_with_history(isin, [100.0, 101.0, 102.0])}
        build_order_derived_series(orders, enriched, today=datetime.date(2025, 1, 3))
        msgs = [i for i in dq.issues() if i.context == isin and "double-count" in i.message]
        assert msgs, "expected a double-count warning for a dividend on a total-return series"

    def test_bond_coupon_on_clean_price_does_not_warn(self):
        from tarzan.runtime import data_quality as dq
        dq.reset()
        isin = "IT0005542359"
        orders = [
            _o(OrderType.TRANSFER_IN, isin, qty=4000.0, gross=4000.0, price=100.0,
               d=(2025, 1, 1), kind=InstrumentKind.BOND),
            _o(OrderType.COUPON, isin, net=98.0, d=(2025, 1, 2)),
        ]
        enriched = {isin: _enriched_borsa_bond(isin, current_price=1.0384, qty=4000.0)}
        build_order_derived_series(orders, enriched, today=datetime.date(2025, 1, 3))
        assert not [i for i in dq.issues() if "double-count" in i.message]


class TestCumExConservationProperty:
    @given(
        face=st.floats(min_value=1000.0, max_value=1e6),
        price=st.floats(min_value=80.0, max_value=120.0),
    )
    def test_cum_ex_contributes_zero_principal(self, face, price):
        # Property 5: explicitly equivalent cum/ex legs with equal face net
        # to a closed position and are not valued.
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "IT0005565392",
                qty=face,
                gross=face * price / 100.0,
                price=price,
                d=(2025, 1, 1),
                equivalence_group="btp-italia-2028",
            ),
            _o(
                OrderType.SELL,
                "IT0005565400",
                qty=-face,
                net=face * price / 100.0,
                price=price,
                d=(2025, 6, 1),
                equivalence_group="BTP-ITALIA-2028",
            ),
        ]
        assert _open_from_orders(orders) == set()
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 12, 1))
        # No open ISIN → terminal valuation is zero principal.
        assert res.valuations[-1][1] == pytest.approx(0.0)

    def test_cum_ex_with_different_leg_prices_nets_to_zero(self):
        # Regression: explicit cum/ex equivalents net to zero quantity even
        # when their separate carry-flat prices differ. Without group closure,
        # separate valuation would leave a spurious residual (here -1,000).
        opened = datetime.date(2025, 1, 1)
        closed = datetime.date(2025, 6, 1)
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "IT0005565392",
                qty=20000.0,
                gross=20000.0,
                price=100.0,
                d=(2025, 1, 1),
                kind=InstrumentKind.BOND,
                equivalence_group="btp-italia-2028",
            ),
            _o(
                OrderType.SELL,
                "IT0005565400",
                qty=-20000.0,
                net=21000.0,
                price=105.0,
                d=(2025, 6, 1),
                kind=InstrumentKind.BOND,
                equivalence_group="btp-italia-2028",
            ),
        ]
        res = build_order_derived_series(
            orders, enriched_by_isin={}, today=datetime.date(2025, 12, 1))
        sparse = dict(res.valuations)

        assert sparse[opened] == pytest.approx(20000.0)
        assert sparse[closed] == pytest.approx(0.0)
        assert res.valuations[-1][1] == pytest.approx(0.0)
        assert res.actual_value_series.loc[pd.Timestamp(opened)] == pytest.approx(20000.0)
        assert res.actual_value_series.loc[pd.Timestamp(closed)] == pytest.approx(0.0)
        # Full liquidation carries the flow-adjusted NAV flat; it must not
        # leave a residual or fabricate a permanent -100% return.
        assert res.daily_series.loc[pd.Timestamp(closed)] == pytest.approx(
            res.daily_series.loc[pd.Timestamp(closed - datetime.timedelta(days=1))]
        )
        # Whole-history provenance retains the fallback used while the cum leg
        # was held and the exact observation used to value the closing ex-leg
        # external flow. The closed group still contributes no terminal value,
        # so terminal coverage remains bounded.
        assert "IT0005565392" in res.provenance["synthetic"]
        assert "IT0005565392" in res.provenance["carry_flat"]
        assert "IT0005565400" in res.provenance["synthetic"]
        assert "IT0005565400" not in res.provenance["carry_flat"]
        assert res.history_availability is Availability.DEGRADED
        assert res.coverage_pct <= 100.0

    def test_ungrouped_same_prefix_legs_remain_in_sparse_and_dense_history(self):
        long_isin, short_isin = "IE00BL25JL35", "IE00BL25JM42"
        opened = datetime.date(2025, 1, 1)
        shorted = datetime.date(2025, 1, 3)
        orders = [
            _o(
                OrderType.BUY,
                long_isin,
                qty=100.0,
                net=-105.0,
                price=105.0,
                d=(2025, 1, 1),
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.SELL,
                short_isin,
                qty=-100.0,
                net=100.0,
                price=100.0,
                d=(2025, 1, 3),
                kind=InstrumentKind.BOND,
            ),
        ]

        result = build_order_derived_series(
            orders,
            enriched_by_isin={},
            today=datetime.date(2025, 1, 4),
        )
        sparse = dict(result.valuations)

        assert sparse[opened] == pytest.approx(105.0)
        assert sparse[shorted] == pytest.approx(5.0)
        assert result.valuations[-1][1] == pytest.approx(5.0)
        assert result.actual_value_series.loc[pd.Timestamp(opened)] == pytest.approx(105.0)
        assert result.actual_value_series.loc[pd.Timestamp(shorted)] == pytest.approx(5.0)
        assert result.daily_series.loc[pd.Timestamp(shorted)] > 0.0
        assert result.provenance["carry_flat"] == [long_isin, short_isin]


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
        assert _open_from_orders(orders) == set()
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
    drops by the net unsettled capital (regression: a four-figure buy that
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

    def test_transient_unpriced_day_does_not_collapse_nav(self, monkeypatch):
        """A single interior day where the holding resolves to no price must
        carry the NAV index flat, not pin it at zero for the rest of the
        window (which would fabricate a -100% and poison every risk metric)."""
        import tarzan.engine.returns_builder as rb

        class _TL:
            def isins(self):
                return ["X"]
            def qty_at(self, isin, d):
                return 10.0

        class _Res:
            _px = {
                datetime.date(2024, 1, 1): 100.0,
                datetime.date(2024, 1, 2): 110.0,
                datetime.date(2024, 1, 3): None,   # transient all-unpriced day
                datetime.date(2024, 1, 4): 120.0,
                datetime.date(2024, 1, 5): 130.0,
            }
            def price_on(self, isin, d):
                return self._px.get(d, 100.0), "yfinance"
            def instrument_kind(self, isin):
                return InstrumentKind.STOCK

        monkeypatch.setattr(
            rb,
            "_closed_identity_groups",
            lambda timeline, day, identity_by_isin=None: set(),
        )
        nav, actual = rb._build_daily_series(
            _TL(), _Res(), {}, [datetime.date(2024, 1, 1)], datetime.date(2024, 1, 5))
        vals = [round(v, 2) for v in nav.values]
        # Flat on the unpriced day, then recovers — never a permanent zero.
        assert vals == [1000.0, 1100.0, 1100.0, 1200.0, 1300.0]
        assert all(v > 0 for v in nav.values)


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
            _o(OrderType.BUY, "BTP", qty=1000.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1), kind=InstrumentKind.BOND),
            _o(OrderType.COUPON, "BTP", qty=0.0, net=20.0, d=(2025, 2, 1),
               kind=InstrumentKind.BOND),
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
            _o(OrderType.BUY, "BTP", qty=1000.0, net=-1000.0, price=100.0,
               d=(2025, 1, 1), kind=InstrumentKind.BOND),
        ]
        series = build_order_derived_series(
            orders, enriched, today=datetime.date(2025, 2, 28))
        res = twror(series.valuations, series.external_flows, series.span_days)
        assert res.cumulative_pct == pytest.approx(0.0, abs=1e-6)


class TestAllocationTimelinePerHolding:
    """Per-holding trends retain exact-ISIN attribution.

    Explicit equivalence evidence may control closure and quote borrowing, but
    unrelated instruments remain isolated even when their identifiers happen
    to share leading characters.
    """

    def _enriched(self, isin):
        return Holding(isin=isin, ticker=isin, quantity=100.0,
                       cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR",
                       asset_class=AssetClass.EQUITIES)

    def test_holding_series_present_and_per_isin(self):
        # Two equity ETFs sharing the first 9 ISIN chars, different prices.
        a, b = "IE00BL25JP72", "IE00BL25JM42"
        orders = [
            _o(OrderType.BUY, a, qty=100.0, net=-5000.0, price=50.0, d=(2025, 4, 1)),
            _o(OrderType.BUY, b, qty=100.0, net=-3000.0, price=30.0, d=(2025, 4, 1)),
        ]
        enriched = {a: self._enriched(a), b: self._enriched(b)}
        tl = build_allocation_timeline(
            orders, enriched, months=1, today=datetime.date(2025, 6, 1)
        )
        assert tl is not None
        assert "holding" in tl
        assert len(tl["holding"]) == len(tl["dates"])
        last = tl["holding"][-1]
        # Both kept distinct (not merged by the shared prefix)...
        assert a in last and b in last
        # ...and normalized to % of the equity class (50k vs 30k → 62.5/37.5).
        assert last[a] == pytest.approx(62.5, abs=0.5)
        assert last[b] == pytest.approx(37.5, abs=0.5)
        assert last[a] + last[b] == pytest.approx(100.0, abs=0.5)

    def test_distinct_same_prefix_not_merged_in_aggregates(self):
        # Two DIFFERENT instruments sharing a 9-char prefix but in different
        # asset classes / geographies. The old group-and-sum valuation would
        # have collapsed them into one position (classified by a single leg);
        # per-ISIN valuation must keep both in the asset and geo aggregates.
        eq, fi = "IE00BL25JP72", "IE00BL25JM42"   # share "IE00BL25J"
        orders = [
            _o(OrderType.BUY, eq, qty=100.0, net=-5000.0, price=50.0, d=(2025, 4, 1)),
            _o(OrderType.BUY, fi, qty=100.0, net=-3000.0, price=30.0, d=(2025, 4, 1)),
        ]
        h_eq = Holding(isin=eq, ticker=eq, quantity=100.0, cost_basis_eur=0.0,
                       market_value_eur=0.0, currency="EUR",
                       asset_class=AssetClass.EQUITIES,
                       geo_breakdown={Geography.USA: 100.0})
        h_fi = Holding(isin=fi, ticker=fi, quantity=100.0, cost_basis_eur=0.0,
                       market_value_eur=0.0, currency="EUR",
                       asset_class=AssetClass.EQUITIES,  # keep equity to avoid bond /100
                       geo_breakdown={Geography.JAPAN: 100.0})
        tl = build_allocation_timeline(
            orders, {eq: h_eq, fi: h_fi}, months=1, today=datetime.date(2025, 6, 1)
        )
        # Geography aggregate keeps both regions, weighted by value (5k vs 3k).
        geo = tl["geo"][-1]
        assert geo.get(Geography.USA.value) == pytest.approx(62.5, abs=0.5)
        assert geo.get(Geography.JAPAN.value) == pytest.approx(37.5, abs=0.5)

    def test_explicit_equivalence_allows_quote_borrowing(self):
        unpriced, priced = "IT0005565392", "IT0005565400"
        group = "btp-italia-2028"
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                unpriced,
                qty=100.0,
                gross=100.0,
                price=None,
                d=(2025, 4, 1),
                equivalence_group=group,
            ),
            _o(
                OrderType.TRANSFER_IN,
                priced,
                qty=100.0,
                gross=100.0,
                price=30.0,
                d=(2025, 4, 1),
                equivalence_group=group.upper(),
            ),
        ]

        timeline = build_allocation_timeline(
            orders,
            {unpriced: self._enriched(unpriced), priced: self._enriched(priced)},
            months=1,
            today=datetime.date(2025, 6, 1),
        )

        assert timeline is not None
        assert timeline["holding"][-1][unpriced] == pytest.approx(50.0)
        assert timeline["holding"][-1][priced] == pytest.approx(50.0)

    def test_ungrouped_same_prefix_missing_quote_fails_closed(self):
        unpriced, priced = "IE00BL25JL35", "IE00BL25JM42"
        orders = [
            _o(
                OrderType.BUY,
                unpriced,
                qty=100.0,
                net=-5000.0,
                price=None,
                d=(2025, 4, 1),
            ),
            _o(
                OrderType.BUY,
                priced,
                qty=100.0,
                net=-3000.0,
                price=30.0,
                d=(2025, 4, 1),
            ),
        ]

        timeline = build_allocation_timeline(
            orders,
            {unpriced: self._enriched(unpriced), priced: self._enriched(priced)},
            months=1,
            today=datetime.date(2025, 6, 1),
        )

        assert timeline is None

    def test_later_closed_unpriceable_position_fails_closed(self):
        unpriced, priced = "CLOSED-WITHOUT-PRICE", "PRICED-OPEN"
        orders = [
            _o(
                OrderType.BUY,
                unpriced,
                qty=100.0,
                net=-5000.0,
                price=None,
                d=(2025, 4, 1),
            ),
            _o(
                OrderType.BUY,
                priced,
                qty=100.0,
                net=-3000.0,
                price=30.0,
                d=(2025, 4, 1),
            ),
            _o(
                OrderType.SELL,
                unpriced,
                qty=-100.0,
                net=5500.0,
                price=None,
                d=(2025, 5, 15),
            ),
        ]

        timeline = build_allocation_timeline(
            orders,
            {unpriced: self._enriched(unpriced), priced: self._enriched(priced)},
            months=3,
            today=datetime.date(2025, 6, 1),
        )

        assert timeline is None

    def test_explicit_equivalence_net_flat_group_contributes_nothing(self):
        cum, ex = "IT0005565392", "IT0005565400"
        group = "btp-italia-2028"
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                cum,
                qty=100.0,
                gross=100.0,
                price=100.0,
                d=(2025, 4, 1),
                equivalence_group=group,
            ),
            _o(
                OrderType.SELL,
                ex,
                qty=-100.0,
                net=105.0,
                price=105.0,
                d=(2025, 4, 15),
                equivalence_group=group.upper(),
            ),
        ]

        timeline = build_allocation_timeline(
            orders,
            {cum: self._enriched(cum), ex: self._enriched(ex)},
            months=1,
            today=datetime.date(2025, 6, 1),
        )

        assert timeline is not None
        assert all(not bucket for bucket in timeline["holding"])
        assert all(not bucket for bucket in timeline["holding_invested"])
        assert all(not bucket for bucket in timeline["asset"])

    def test_closed_cum_ex_pair_contributes_nothing(self):
        # A true cum/ex rotation that fully nets out (cum sold, ex never held
        # on the boundary) must not leave a residual in the series.
        cum, ex = "IT0005565392", "IT0005565400"   # share "IT0005565"
        orders = [
            _o(OrderType.TRANSFER_IN, cum, qty=20000.0, gross=20000.0, price=100.0,
               d=(2025, 4, 1)),
            _o(OrderType.SELL, cum, qty=-20000.0, net=20000.0, price=100.0,
               d=(2025, 4, 15)),
        ]
        h = Holding(isin=cum, ticker=cum, quantity=0.0, cost_basis_eur=0.0,
                    market_value_eur=0.0, currency="EUR",
                    asset_class=AssetClass.FIXED_INCOME)
        tl = build_allocation_timeline(
            orders, {cum: h}, months=1, today=datetime.date(2025, 6, 1)
        )
        # Position is closed → no holding weight and an empty asset bucket.
        assert all(not bucket for bucket in tl["holding"])
        assert all(not bucket for bucket in tl["asset"])


# ---------------------------------------------------------------------------
# Production-readiness bug exploration: C4 historical exposure consistency
# ---------------------------------------------------------------------------

from hypothesis import settings as _settings  # noqa: E402


# **Validates: Requirements 2.4**
@given(capital_eur=st.integers(min_value=100, max_value=100_000))
@_settings(max_examples=5, deadline=None, derandomize=True)
def test_c4_history_and_optimizer_share_90_60_notional_exposure(capital_eur):
    """Property 1 / C4 exploration for history versus planning.

    The historical allocation builder consumes ``class_breakdown`` and keeps
    the legitimate 150% total, while the planning objective sees only the
    holding's primary class.
    """
    import numpy as np

    from tarzan.engine.rebalancer import _ObjectiveModel
    from tarzan.models.investor_config import InvestorConfig

    capital = float(capital_eur)
    quantity = capital / 100.0
    order = _o(
        OrderType.BUY,
        "LEV",
        qty=quantity,
        net=-capital,
        gross=capital,
        price=100.0,
        d=(2025, 1, 1),
    )
    holding = _enriched_with_history("LEV", [100.0] * 90)
    holding.quantity = quantity
    holding.current_price = 100.0
    holding.current_value = capital
    holding.market_value_eur = capital
    holding.asset_class = AssetClass.EQUITIES
    holding.class_breakdown = {
        AssetClass.EQUITIES: 90.0,
        AssetClass.FIXED_INCOME: 60.0,
    }

    timeline = build_allocation_timeline(
        [order], {"LEV": holding}, months=3, today=datetime.date(2025, 3, 31)
    )
    historical = {
        str(key): float(value)
        for key, value in timeline["asset"][-1].items()
    }

    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {
        "Equities": 90.0,
        "Fixed Income": 60.0,
    }
    config.equity_geo_targets_pctg = {}
    config.target_cash_buffer_eur = 0.0
    model = _ObjectiveModel([holding], config, np.array([capital], dtype=float))
    gaps = model.gaps(np.array([capital], dtype=float))
    optimizer = {
        key: float(target + gap)
        for key, target, gap in zip(model.ac_keys, model.ac_targets, gaps)
    }

    expected = {"Equities": 90.0, "Fixed Income": 60.0}
    assert historical == pytest.approx(expected, abs=1e-6)
    assert sum(historical.values()) == pytest.approx(150.0, abs=1e-6), (
        "notional exposure above 100% is intentional and must be preserved"
    )
    assert optimizer == pytest.approx(historical, abs=1e-6), (
        "historical analysis and planning disagree for the same 90/60 holding: "
        f"history={historical}, optimizer={optimizer}"
    )


class TestConflictingOrderKindEvidence:
    def _conflicting_orders(self):
        return [
            _o(
                OrderType.BUY,
                "CONFLICTING-KIND",
                qty=5.0,
                net=-500.0,
                price=100.0,
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                "CONFLICTING-KIND",
                qty=5.0,
                net=-500.0,
                price=100.0,
                d=(2025, 1, 2),
                kind=InstrumentKind.ETF,
            ),
        ]

    def test_synthetic_holding_preserves_every_conflicting_assertion(self):
        holding = build_holdings_from_orders(self._conflicting_orders())[0]

        assert holding.security_type is None
        assert holding.instrument_kind_evidence == ("BOND", "ETF")
        assert holding.market_value_eur == 0.0

    def test_provider_kind_cannot_override_conflicting_order_evidence(self):
        orders = self._conflicting_orders()
        holding = build_holdings_from_orders(orders)[0]
        # Simulate a later provider declaring ETF. The original BOND/ETF
        # conflict remains authoritative and keeps mechanics unavailable.
        holding.security_type = "ETF"
        holding.instrument_type = "ETF"
        holding.price_history = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )

        result = build_order_derived_series(
            orders,
            {holding.isin: holding},
            today=datetime.date(2025, 1, 3),
        )

        assert result.valuations[-1][1] == 0.0
        assert holding.isin in result.provenance["excluded"]


class TestUnclassifiedEtfHistory:
    def test_allocation_timeline_is_unavailable_without_tracked_category(self):
        orders = [
            _o(
                OrderType.BUY,
                "UNCLASSIFIED-ETF",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.ETF,
            )
        ]
        holding = Holding(
            isin="UNCLASSIFIED-ETF",
            ticker="UNCLASSIFIED-ETF",
            quantity=10.0,
            cost_basis_eur=1000.0,
            market_value_eur=1000.0,
            currency="EUR",
            security_type="ETF",
            instrument_kind_evidence=("ETF",),
        )
        holding.price_history = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2025-01-01", "2025-06-01"]),
        )

        assert build_allocation_timeline(
            orders,
            {holding.isin: holding},
            months=3,
            today=datetime.date(2025, 6, 1),
        ) is None


class TestUnavailableOrderHistory:
    def test_mixed_valid_and_conflicting_history_is_typed_unavailable(self):
        from tarzan.runtime.ledger import Availability

        valid_orders = [
            _o(
                OrderType.BUY,
                "VALID-STOCK",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.STOCK,
            )
        ]
        conflicting_orders = [
            _o(
                OrderType.BUY,
                "CONFLICTING-HISTORY",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                "CONFLICTING-HISTORY",
                qty=1.0,
                net=-100.0,
                price=100.0,
                d=(2025, 1, 2),
                kind=InstrumentKind.ETF,
            ),
        ]
        orders = valid_orders + conflicting_orders
        valid = _enriched_with_history("VALID-STOCK", [100.0, 101.0])
        valid.security_type = "STOCK"
        conflict = build_holdings_from_orders(conflicting_orders)[0]
        conflict.security_type = "ETF"
        conflict.price_history = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )

        result = build_order_derived_series(
            orders,
            {"VALID-STOCK": valid, "CONFLICTING-HISTORY": conflict},
            today=datetime.date(2025, 1, 3),
        )

        assert result.history_availability is Availability.UNAVAILABLE
        assert result.unavailable_instruments == ("CONFLICTING-HISTORY",)

    def test_metrics_null_history_dependent_outputs_when_mechanics_unavailable(self):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        orders = [
            _o(
                OrderType.BUY,
                "CONFLICTING-METRICS",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                "CONFLICTING-METRICS",
                qty=1.0,
                net=-100.0,
                price=100.0,
                d=(2025, 1, 2),
                kind=InstrumentKind.ETF,
            ),
        ]
        holding = build_holdings_from_orders(orders)[0]
        holding.security_type = "ETF"
        holding.price_history = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        )
        engine = MetricsEngine([holding], InvestorConfig(), orders=orders)
        context: dict = {}

        engine._portfolio_history_from_orders(context)
        engine._performance(context)
        engine._risk(context)
        engine._returns(context)

        assert context["_order_history_unavailable"] == [
            "CONFLICTING-METRICS"
        ]
        assert context["portfolio_history"].empty
        assert context["performance"] is None
        assert context["risk"] is None
        assert context["xirr_pct"] is None
        assert context["twror_pct"] is None
        assert context["pnl_eur"] is None
        assert set(context["_degraded"]) == {
            "_portfolio_history_from_orders",
            "_returns",
        }

    def test_metrics_null_history_outputs_when_causal_price_is_missing(self):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        orders = [
            _o(
                OrderType.BUY,
                "NO-CAUSAL-PRICE",
                qty=10.0,
                net=-1000.0,
                price=None,
                kind=InstrumentKind.STOCK,
            )
        ]
        holding = build_holdings_from_orders(orders)[0]
        engine = MetricsEngine([holding], InvestorConfig(), orders=orders)
        context: dict = {}

        engine._portfolio_history_from_orders(context)
        engine._performance(context)
        engine._risk(context)
        engine._returns(context)

        assert context["history_availability"] == Availability.UNAVAILABLE.value
        assert context["_order_history_unavailable"] == ["NO-CAUSAL-PRICE"]
        assert context["portfolio_history"].empty
        assert context["performance"] is None
        assert context["risk"] is None
        assert context["xirr_pct"] is None
        assert context["twror_pct"] is None
        assert context["pnl_eur"] is None

    def test_price_only_history_gap_preserves_current_rebalancing(
        self,
        monkeypatch,
    ):
        from tarzan import runtime
        import tarzan.engine.metrics as metrics_module
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "LATE-MARKET-HISTORY",
                qty=10.0,
                gross=1000.0,
                price=None,
                d=(2025, 1, 1),
                kind=InstrumentKind.STOCK,
            )
        ]
        holding = build_holdings_from_orders(orders)[0]
        holding.name = "Late market history"
        holding.instrument_type = "Stock"
        holding.asset_class = AssetClass.EQUITIES
        holding.class_breakdown = {AssetClass.EQUITIES: 100.0}
        holding.current_price = 101.0
        holding.current_value = 1010.0
        holding.market_value_eur = 1010.0
        holding.price_history = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
        )

        plan_calls = []

        def _plan(_holdings, plan_config, _total_value, *, lump_sum=None):
            plan_calls.append((plan_config.rebalancing_no_sell, lump_sum))
            return [], []

        monkeypatch.setattr(metrics_module, "BENCHMARKS", {})
        monkeypatch.setattr(
            metrics_module,
            "_fetch_benchmark_history",
            lambda _ticker: pd.Series(dtype=float),
        )
        monkeypatch.setattr(
            "tarzan.data.geo_resolver.lookup_geo_by_index_name",
            lambda _name: None,
        )
        monkeypatch.setattr(
            "tarzan.engine.rebalancer.compute_unified_rebalancing",
            _plan,
        )
        monkeypatch.setattr(
            "tarzan.engine.rebalancer.plan_cost",
            lambda *_args, **_kwargs: {"cgt_eur": 0.0, "fees_eur": 0.0},
        )
        monkeypatch.setattr(
            "tarzan.runtime.audit.record_rebalancing_plan",
            lambda *_args, **_kwargs: None,
        )
        runtime.configure(
            deterministic=True,
            as_of=datetime.date(2025, 1, 3),
            attempt_id="price-only-history-gap",
            invocation_source="test",
        )
        try:
            result = MetricsEngine(
                [holding],
                InvestorConfig(),
                orders=orders,
            ).compute_all()
        finally:
            runtime.reset()

        assert result.history_availability == "UNAVAILABLE"
        assert result.history_unavailable_instruments == (
            "LATE-MARKET-HISTORY",
        )
        assert result.portfolio_history is None
        assert result.performance is None
        assert result.risk is None
        assert result.xirr_pct is None
        assert result.twror_pct is None
        assert result.allocation_timeline is None
        assert sorted(no_sell for no_sell, _ in plan_calls) == [False, True]
        assert result.rebalancing_suggestions == []
        assert result.rebalancing_verifications == []
        assert result.rebalancing_plans is not None
        assert len(result.rebalancing_plans) == 2

    def test_compute_all_preserves_valid_rows_without_recreating_portfolio_history(
        self, monkeypatch
    ):
        from tarzan import runtime
        import tarzan.engine.metrics as metrics_module
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        valid_orders = [
            _o(
                OrderType.BUY,
                "VALID-STOCK",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.STOCK,
            )
        ]
        conflicting_orders = [
            _o(
                OrderType.BUY,
                "CONFLICTING-HISTORY",
                qty=10.0,
                net=-1000.0,
                price=100.0,
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                "CONFLICTING-HISTORY",
                qty=1.0,
                net=-100.0,
                price=100.0,
                d=(2025, 1, 2),
                kind=InstrumentKind.ETF,
            ),
        ]
        valid = build_holdings_from_orders(valid_orders)[0]
        valid.name = "Valid stock"
        valid.instrument_type = "STOCK"
        valid.asset_class = AssetClass.EQUITIES
        valid.current_price = 100.0
        valid.current_value = 1000.0
        valid.market_value_eur = 1000.0
        valid.price_history = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )

        conflicting = build_holdings_from_orders(conflicting_orders)[0]
        # A provider may supply a plausible current classification and value,
        # but it cannot erase the two contradictory order-kind assertions.
        conflicting.name = "Conflicting history"
        conflicting.security_type = "ETF"
        conflicting.instrument_type = "ETF"
        conflicting.asset_class = AssetClass.FIXED_INCOME
        conflicting.current_price = 100.0
        conflicting.current_value = 1000.0
        conflicting.market_value_eur = 1000.0
        conflicting.price_history = pd.Series(
            [100.0, 50.0, 25.0],
            index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        )

        # Exercise the complete MetricsEngine pipeline without network or
        # optimizer side effects. Live 1D is deliberately simulated even in a
        # reproducible context to prove it cannot recreate a null portfolio
        # performance section after typed history unavailability is known.
        monkeypatch.setattr(metrics_module, "BENCHMARKS", {})
        monkeypatch.setattr(
            metrics_module,
            "_fetch_benchmark_history",
            lambda _ticker: pd.Series(dtype=float),
        )
        monkeypatch.setattr(
            "tarzan.data.geo_resolver.lookup_geo_by_index_name",
            lambda _name: None,
        )
        monkeypatch.setattr(
            "tarzan.engine.rebalancer.compute_unified_rebalancing",
            lambda *_args, **_kwargs: pytest.fail(
                "unavailable instrument mechanics reached executable planning"
            ),
        )
        monkeypatch.setattr(
            "tarzan.runtime.audit.record_rebalancing_plan",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "tarzan.data.price_cache.load_resolution",
            lambda key: key,
        )
        live_requests = []

        def _intraday_feeds(symbols, *, allow_sibling_fallback):
            assert allow_sibling_fallback is True
            live_requests.extend(symbols)
            return {
                symbol: {"pct": 3.25, "live": True}
                for symbol in symbols
            }

        monkeypatch.setattr("tarzan.data.market_quotes.intraday_feeds", _intraday_feeds)
        runtime.configure(
            deterministic=True,
            as_of=datetime.date(2025, 1, 3),
            attempt_id="mixed-unavailable-history",
            invocation_source="test",
        )
        monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
        try:
            result = MetricsEngine(
                [valid, conflicting],
                InvestorConfig(),
                orders=valid_orders + conflicting_orders,
            ).compute_all()
        finally:
            runtime.reset()

        assert result.history_availability == "UNAVAILABLE"
        assert result.history_unavailable_instruments == (
            "CONFLICTING-HISTORY",
        )
        assert result.portfolio_history is None
        assert result.performance is None
        assert result.performance_full is None
        assert result.risk is None
        assert result.xirr_pct is None
        assert result.twror_pct is None
        assert result.twror_annualized_pct is None
        assert result.returns_coverage_pct is None
        assert result.pnl_eur is None
        assert result.pnl_pct is None
        assert result.invested_capital_eur is None
        assert result.allocation_timeline is None
        assert result.rebalancing_suggestions is None
        assert result.rebalancing_verifications is None
        assert result.rebalancing_plans is None
        # Exact current valuation and tracked-category evidence are an
        # independent point-in-time capability; history unavailability must
        # not erase those valid rows or silently renormalize them away.
        current_allocation = {
            str(row.category): float(row.weight_pct)
            for row in result.allocation_by_class.itertuples()
        }
        assert current_allocation == pytest.approx(
            {"Equities": 50.0, "Fixed Income": 50.0}
        )

        performance_tickers = set(result.holding_performance["ticker"])
        assert "CONFLICTING-HISTORY" not in performance_tickers
        assert "VALID-STOCK" in performance_tickers
        assert set(result.holding_histories) == {"VALID-STOCK"}
        assert live_requests == ["VALID-STOCK"]
        valid_row = result.holding_performance.set_index("ticker").loc[
            "VALID-STOCK"
        ]
        # The Returns-table 1D now comes from the same price series as every
        # other window and as the historical-risk 1D below — 121/110-1 = 10%.
        # It used to be overwritten by the mocked broker quote (3.25), so one
        # reproducible run reported two different 1D figures for one holding;
        # unifying on the series removed the second source (and a live quote
        # can no longer leak into a deterministic run).
        assert valid_row["1d"] == pytest.approx(10.0)
        assert bool(valid_row["live_1d"]) is True
        # The valid holding remains usable in the independent static
        # backtest, while the conflicting -50% series is excluded. If it
        # leaked into the equal-value candidate set, this would be -20%.
        historical = result.historical_risk
        assert historical is not None
        assert historical["portfolio"] is not None
        assert historical["portfolio"]["metrics"]["1d"] == pytest.approx(10.0)
        assert "unavailable order history" in historical["portfolio"]["note"]
        assert set(result.degraded_computers) == {
            "_portfolio_history_from_orders",
            "_returns",
        }
