"""Tests for the estimated capital-gains-tax module (approach b)."""

from __future__ import annotations

import datetime

import pytest

from tarzan.engine.tax import estimate_realized_cgt
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType


def _o(otype, isin, qty=0.0, net=0.0, gross=0.0, name="X", d=(2025, 1, 1), td=None):
    return Order(
        date=datetime.date(*d),
        trade_date=datetime.date(*td) if td is not None else datetime.date(*d),
        type=otype, isin=isin, name=name, ticker="",
        quantity=qty, currency="EUR", price_native=None, fx_rate=1.0,
        gross_eur=gross, fees_eur=0.0, net_eur=net, source="fineco",
    )


def _etf(isin):
    return Holding(isin=isin, ticker=isin, quantity=0.0, cost_basis_eur=0.0,
                   market_value_eur=0.0, currency="EUR",
                   instrument_type="Equity ETF")


class TestBasics:
    def test_no_rates_means_no_tax(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=1200, d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {"IE00AAA": _etf("IE00AAA")}, 0, 0)
        assert est.total_tax_eur == 0.0
        assert est.tax_flows == []

    def test_etf_gain_taxed_at_standard_rate(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=1200, d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5)
        # gain 200 * 26% = 52
        assert est.total_realized_gain_eur == pytest.approx(200.0)
        assert est.total_tax_eur == pytest.approx(52.0)
        assert est.tax_flows == [(datetime.date(2025, 2, 1), -52.0)]

    def test_loss_is_not_taxed(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=800, d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5)
        assert est.total_tax_eur == 0.0
        assert est.total_realized_loss_eur == pytest.approx(200.0)


class TestGovernmentBond:
    def test_btp_uses_reduced_rate_by_name(self):
        # No enrichment → classification falls back to the order name.
        orders = [
            _o(OrderType.BUY, "IT0005AAA", qty=10000, net=-10000,
               name="BTP-1MZ35 3,35%"),
            _o(OrderType.SELL, "IT0005AAA", qty=-10000, net=11000,
               name="BTP-1MZ35 3,35%", d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {}, 26, 12.5)
        # gain 1000 * 12.5% = 125
        assert est.total_tax_eur == pytest.approx(125.0)


class TestCumExRotation:
    def test_gain_captured_across_prefix_siblings(self):
        # Cost transferred in under the "cum" ISIN, proceeds booked under
        # the "ex" sibling sharing the 9-char prefix → realized gain must
        # still be taxed (regression: it was lost when keyed per ISIN).
        orders = [
            _o(OrderType.TRANSFER_IN, "IT0005565392", qty=20000, gross=20000,
               net=0, name="BTP-10OT28 VALSU CUM"),
            _o(OrderType.SELL, "IT0005565400", qty=-20000, net=21025.88,
               name="BTP-10OT28 VALORE SU", d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {}, 26, 12.5)
        # realized gain 1025.88, gov rate 12.5%
        assert est.total_realized_gain_eur == pytest.approx(1025.88)
        assert est.total_tax_eur == pytest.approx(1025.88 * 0.125)


class TestLossOffset:
    def test_diversi_gain_offset_by_prior_loss(self):
        # A single-bond loss then a later single-bond gain: the gain is
        # redditi diversi, so the prior loss shelters it.
        orders = [
            _o(OrderType.BUY, "IT0005AAA", qty=10000, net=-10000,
               name="BTP A"),
            _o(OrderType.SELL, "IT0005AAA", qty=-10000, net=9000,
               name="BTP A", d=(2025, 2, 1)),                 # loss 1000
            _o(OrderType.BUY, "IT0009BBB", qty=10000, net=-10000,
               name="BTP B", d=(2025, 3, 1)),
            _o(OrderType.SELL, "IT0009BBB", qty=-10000, net=11000,
               name="BTP B", d=(2025, 4, 1)),                 # gain 1000
        ]
        est = estimate_realized_cgt(orders, {}, 26, 12.5)
        # 1000 gain fully offset by 1000 carried loss → no tax.
        assert est.total_tax_eur == pytest.approx(0.0)
        assert est.taxable_base_eur == pytest.approx(0.0)

    def test_etf_gain_not_offset_by_loss(self):
        # An ETF gain is reddito di capitale: a prior loss CANNOT shelter it.
        orders = [
            _o(OrderType.BUY, "IE00LOSS", qty=10, net=-1000, name="ETF L"),
            _o(OrderType.SELL, "IE00LOSS", qty=-10, net=800,
               name="ETF L", d=(2025, 2, 1)),                 # loss 200
            _o(OrderType.BUY, "IE00GAIN", qty=10, net=-1000, name="ETF G",
               d=(2025, 3, 1)),
            _o(OrderType.SELL, "IE00GAIN", qty=-10, net=1300,
               name="ETF G", d=(2025, 4, 1)),                 # gain 300
        ]
        enriched = {"IE00LOSS": _etf("IE00LOSS"), "IE00GAIN": _etf("IE00GAIN")}
        est = estimate_realized_cgt(orders, enriched, 26, 12.5)
        # ETF gain 300 taxed fully at 26% = 78 (loss does NOT offset it).
        assert est.total_tax_eur == pytest.approx(78.0)

    def test_transfer_out_is_not_a_taxable_event(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000, name="ETF"),
            _o(OrderType.TRANSFER_OUT, "IE00AAA", qty=-10, net=0,
               name="ETF", d=(2025, 2, 1)),
        ]
        est = estimate_realized_cgt(orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5)
        assert est.total_tax_eur == 0.0
        assert est.total_realized_gain_eur == 0.0
