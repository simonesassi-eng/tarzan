"""Tests for evidence-backed estimated capital-gains tax."""

from __future__ import annotations

import datetime

import pytest

from tarzan.engine.tax import TaxEvidenceUnavailable, estimate_realized_cgt
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType


def _o(
    order_type,
    isin,
    qty=0.0,
    net=0.0,
    gross=0.0,
    name="X",
    d=(2025, 1, 1),
    td=None,
    kind=None,
    equivalence_group=None,
):
    return Order(
        date=datetime.date(*d),
        trade_date=datetime.date(*td) if td is not None else datetime.date(*d),
        type=order_type,
        isin=isin,
        name=name,
        ticker="",
        quantity=qty,
        currency="EUR",
        price_native=None,
        fx_rate=1.0,
        gross_eur=gross,
        fees_eur=0.0,
        net_eur=net,
        source="fineco",
        instrument_kind=kind,
        instrument_equivalence_group=equivalence_group,
    )


def _etf(isin):
    return Holding(
        isin=isin,
        ticker=isin,
        quantity=0.0,
        cost_basis_eur=0.0,
        market_value_eur=0.0,
        currency="EUR",
        instrument_type="Equity ETF",
    )


class TestBasics:
    def test_no_rates_means_no_tax_and_requires_no_kind(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=1200, d=(2025, 2, 1)),
        ]
        estimate = estimate_realized_cgt(
            orders, {"IE00AAA": _etf("IE00AAA")}, 0, 0
        )
        assert estimate.total_tax_eur == 0.0
        assert estimate.tax_flows == []

    def test_etf_gain_taxed_at_standard_rate(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=1200, d=(2025, 2, 1)),
        ]
        estimate = estimate_realized_cgt(
            orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5
        )
        assert estimate.total_realized_gain_eur == pytest.approx(200.0)
        assert estimate.total_tax_eur == pytest.approx(52.0)
        assert estimate.tax_flows == [(datetime.date(2025, 2, 1), -52.0)]

    def test_loss_is_not_taxed(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000),
            _o(OrderType.SELL, "IE00AAA", qty=-10, net=800, d=(2025, 2, 1)),
        ]
        estimate = estimate_realized_cgt(
            orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5
        )
        assert estimate.total_tax_eur == 0.0
        assert estimate.total_realized_loss_eur == pytest.approx(200.0)

    def test_partial_position_sell_uses_average_cost(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=100, net=-1000, d=(2025, 1, 1)),
            _o(OrderType.BUY, "IE00AAA", qty=100, net=-1500, d=(2025, 1, 2)),
            _o(OrderType.SELL, "IE00AAA", qty=-100, net=2000, d=(2025, 2, 1)),
        ]
        estimate = estimate_realized_cgt(
            orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5
        )
        assert estimate.total_realized_gain_eur == pytest.approx(750.0)
        assert estimate.total_tax_eur == pytest.approx(195.0)

    @pytest.mark.parametrize("kind", [InstrumentKind.STOCK, InstrumentKind.BOND])
    def test_stock_and_corporate_bond_do_not_depend_on_private_helpers(self, kind):
        orders = [
            _o(OrderType.BUY, "XS0000000001", qty=10, net=-1000, kind=kind),
            _o(
                OrderType.SELL,
                "XS0000000001",
                qty=-10,
                net=1200,
                name="ACME 2030",
                d=(2025, 2, 1),
                kind=kind,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_tax_eur == pytest.approx(52.0)


class TestEvidenceAuthority:
    def test_unknown_kind_makes_estimate_unavailable(self):
        orders = [
            _o(OrderType.BUY, "XS0000000002", qty=10, net=-1000),
            _o(OrderType.SELL, "XS0000000002", qty=-10, net=1200),
        ]
        with pytest.raises(TaxEvidenceUnavailable, match="unknown"):
            estimate_realized_cgt(orders, {}, 26, 12.5)

    def test_conflicting_bond_and_etf_evidence_is_unavailable(self):
        isin = "XS0000000003"
        orders = [
            _o(
                OrderType.BUY,
                isin,
                qty=10,
                net=-1000,
                kind=InstrumentKind.BOND,
            ),
            _o(OrderType.SELL, isin, qty=-10, net=1200),
        ]
        holding = _etf(isin)
        with pytest.raises(TaxEvidenceUnavailable, match="ambiguous"):
            estimate_realized_cgt(orders, {isin: holding}, 26, 12.5)

    def test_government_name_cannot_override_non_bond_kind(self):
        orders = [
            _o(
                OrderType.BUY,
                "IT0000000001",
                qty=10,
                net=-1000,
                name="BTP LOOKALIKE",
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.SELL,
                "IT0000000001",
                qty=-10,
                net=1200,
                name="BTP LOOKALIKE",
                d=(2025, 2, 1),
                kind=InstrumentKind.STOCK,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_tax_eur == pytest.approx(52.0)


class TestGovernmentBond:
    def test_btp_uses_reduced_rate_after_exact_bond_resolution(self):
        orders = [
            _o(
                OrderType.BUY,
                "IT0005AAA",
                qty=10000,
                net=-10000,
                name="BTP-1MZ35 3,35%",
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.SELL,
                "IT0005AAA",
                qty=-10000,
                net=11000,
                name="BTP-1MZ35 3,35%",
                d=(2025, 2, 1),
                kind=InstrumentKind.BOND,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_tax_eur == pytest.approx(125.0)


class TestInstrumentIdentity:
    def test_same_prefix_unrelated_instruments_keep_separate_cost_basis(self):
        first = "IT0005565392"
        second = "IT0005565399"
        orders = [
            _o(
                OrderType.BUY,
                first,
                qty=10,
                net=-1000,
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.BUY,
                second,
                qty=10,
                net=-2000,
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.SELL,
                second,
                qty=-10,
                net=3000,
                d=(2025, 2, 1),
                kind=InstrumentKind.STOCK,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_realized_gain_eur == pytest.approx(1000.0)
        assert estimate.total_tax_eur == pytest.approx(260.0)

    def test_documented_cum_ex_variants_require_explicit_equivalence(self):
        group = "BTP-VALORE-2028-CUM-EX"
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "IT0005565392",
                qty=20000,
                gross=20000,
                name="BTP-10OT28 VALSU CUM",
                kind=InstrumentKind.BOND,
                equivalence_group=group,
            ),
            _o(
                OrderType.SELL,
                "IT0005565400",
                qty=-20000,
                net=21025.88,
                name="BTP-10OT28 VALORE SU",
                d=(2025, 2, 1),
                kind=InstrumentKind.BOND,
                equivalence_group=group,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_realized_gain_eur == pytest.approx(1025.88)
        assert estimate.total_tax_eur == pytest.approx(1025.88 * 0.125)

    def test_missing_equivalence_fails_closed_instead_of_dropping_sale(self):
        orders = [
            _o(
                OrderType.TRANSFER_IN,
                "IT0005565392",
                qty=20000,
                gross=20000,
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.SELL,
                "IT0005565400",
                qty=-20000,
                net=21025.88,
                kind=InstrumentKind.BOND,
            ),
        ]
        with pytest.raises(TaxEvidenceUnavailable, match="cost basis unavailable"):
            estimate_realized_cgt(orders, {}, 26, 12.5)

    def test_partial_cost_basis_fails_closed_instead_of_prorating_sale(self):
        orders = [
            _o(
                OrderType.BUY,
                "XS0000000004",
                qty=5,
                net=-500,
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.SELL,
                "XS0000000004",
                qty=-10,
                net=1200,
                kind=InstrumentKind.STOCK,
            ),
        ]
        with pytest.raises(TaxEvidenceUnavailable, match="cost basis unavailable"):
            estimate_realized_cgt(orders, {}, 26, 12.5)

    def test_mixed_known_and_fee_only_transfer_basis_fail_closed(self):
        isin = "XS0000000005"
        orders = [
            _o(
                OrderType.BUY,
                isin,
                qty=5,
                net=-500,
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.TRANSFER_IN,
                isin,
                qty=5,
                gross=0,
                net=-5,
                kind=InstrumentKind.STOCK,
            ),
            _o(
                OrderType.SELL,
                isin,
                qty=-10,
                net=1200,
                kind=InstrumentKind.STOCK,
            ),
        ]
        with pytest.raises(TaxEvidenceUnavailable, match="cost basis unavailable"):
            estimate_realized_cgt(orders, {}, 26, 12.5)


class TestLossOffset:
    def test_diversi_gain_offset_by_prior_loss(self):
        orders = [
            _o(
                OrderType.BUY,
                "IT0005AAA",
                qty=10000,
                net=-10000,
                name="BTP A",
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.SELL,
                "IT0005AAA",
                qty=-10000,
                net=9000,
                name="BTP A",
                d=(2025, 2, 1),
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.BUY,
                "IT0009BBB",
                qty=10000,
                net=-10000,
                name="BTP B",
                d=(2025, 3, 1),
                kind=InstrumentKind.BOND,
            ),
            _o(
                OrderType.SELL,
                "IT0009BBB",
                qty=-10000,
                net=11000,
                name="BTP B",
                d=(2025, 4, 1),
                kind=InstrumentKind.BOND,
            ),
        ]
        estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
        assert estimate.total_tax_eur == pytest.approx(0.0)
        assert estimate.taxable_base_eur == pytest.approx(0.0)

    def test_etf_gain_not_offset_by_loss(self):
        orders = [
            _o(OrderType.BUY, "IE00LOSS", qty=10, net=-1000, name="ETF L"),
            _o(
                OrderType.SELL,
                "IE00LOSS",
                qty=-10,
                net=800,
                name="ETF L",
                d=(2025, 2, 1),
            ),
            _o(
                OrderType.BUY,
                "IE00GAIN",
                qty=10,
                net=-1000,
                name="ETF G",
                d=(2025, 3, 1),
            ),
            _o(
                OrderType.SELL,
                "IE00GAIN",
                qty=-10,
                net=1300,
                name="ETF G",
                d=(2025, 4, 1),
            ),
        ]
        enriched = {
            "IE00LOSS": _etf("IE00LOSS"),
            "IE00GAIN": _etf("IE00GAIN"),
        }
        estimate = estimate_realized_cgt(orders, enriched, 26, 12.5)
        assert estimate.total_tax_eur == pytest.approx(78.0)

    def test_transfer_out_is_not_a_taxable_event(self):
        orders = [
            _o(OrderType.BUY, "IE00AAA", qty=10, net=-1000, name="ETF"),
            _o(
                OrderType.TRANSFER_OUT,
                "IE00AAA",
                qty=-10,
                net=0,
                name="ETF",
                d=(2025, 2, 1),
            ),
        ]
        estimate = estimate_realized_cgt(
            orders, {"IE00AAA": _etf("IE00AAA")}, 26, 12.5
        )
        assert estimate.total_tax_eur == 0.0
        assert estimate.total_realized_gain_eur == 0.0
