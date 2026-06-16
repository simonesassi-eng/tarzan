"""Tests for order-only mode: derive the snapshot from the order list.

Network-free: the enricher is monkeypatched to a no-op that marks each
holding enriched, so the orchestrator wiring (derive → enrich → compute)
is exercised without hitting yfinance.
"""

from __future__ import annotations

import datetime

import pytest

from tarzan import orchestrator
from tarzan.data.loader import load_targets_per_holding
from tarzan.engine.returns_builder import build_holdings_from_orders
from tarzan.models.order import Order, OrderType


TARGETS_CSV = (
    "name,isin,ticker,target_equities,target_fixed_income,no_buy_no_sell\n"
    'BTP,IT0005542359,JB67SW.MOT,,10,TRUE\n'
    "World Momentum,IE00BL25JP72,XDEM.MI,9,,\n"
)


def _o(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 1)):
    return Order(
        date=datetime.date(*d), trade_date=datetime.date(*d), type=otype,
        isin=isin, name="X", ticker="", quantity=qty, currency="EUR",
        price_native=price, fx_rate=1.0, gross_eur=gross, fees_eur=0.0,
        net_eur=net, source="fineco",
    )


# ── Per-holding targets loader ──────────────────────────────────────────────

def test_load_targets_per_holding(tmp_path):
    p = tmp_path / "targets_per_holding.csv"
    p.write_text(TARGETS_CSV)
    targets = load_targets_per_holding(str(p))
    assert set(targets) == {"IT0005542359", "IE00BL25JP72"}
    assert targets["IT0005542359"]["target_fixed_income"] == pytest.approx(10.0)
    assert targets["IT0005542359"]["target_equities"] is None
    assert targets["IT0005542359"]["no_buy_no_sell"] is True
    assert targets["IE00BL25JP72"]["target_equities"] == pytest.approx(9.0)
    assert targets["IE00BL25JP72"]["no_buy_no_sell"] is False


def test_load_targets_missing_file_is_empty():
    assert load_targets_per_holding("/nonexistent/targets.csv") == {}


def test_apply_per_holding_targets():
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BL25JP72", qty=100.0, net=-1000.0),
        _o(OrderType.TRANSFER_IN, "IT0005542359", qty=4000.0, gross=4000.0),
    ])
    targets = {
        "IE00BL25JP72": {"target_equities": 9.0, "target_fixed_income": None,
                         "no_buy_no_sell": False},
        "IT0005542359": {"target_equities": None, "target_fixed_income": 10.0,
                         "no_buy_no_sell": True},
    }
    orchestrator._apply_per_holding_targets(holdings, targets)
    by_isin = {h.isin: h for h in holdings}
    assert by_isin["IE00BL25JP72"].target_equities == pytest.approx(9.0)
    assert by_isin["IT0005542359"].target_fixed_income == pytest.approx(10.0)
    assert by_isin["IT0005542359"].no_buy_no_sell is True


# ── Orchestrator order-only mode ────────────────────────────────────────────

def _no_network_enrich(holdings):
    """Stand-in enricher: price each holding at its seeded value so it is
    'enriched' without any network call."""
    for h in holdings:
        h.current_price = 10.0
        h.current_value = h.market_value_eur or (h.quantity * 10.0)
    return holdings


def _stub_benchmarks(monkeypatch):
    """Neutralize benchmark fetches so the engine never touches the network
    (the order path produces a portfolio history, which would otherwise
    trigger benchmark downloads in _risk/_performance/_benchmarks)."""
    import pandas as pd
    empty = pd.Series(dtype=float)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: empty)
    monkeypatch.setattr("tarzan.engine.metrics._build_benchmark_series",
                        lambda *a, **k: empty)


def test_order_only_derives_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _no_network_enrich)
    _stub_benchmarks(monkeypatch)

    orders_csv = tmp_path / "order_list.csv"
    orders_csv.write_text(
        "date,type,isin,quantity,gross_eur,net_eur\n"
        "2025-01-02,buy,IE00BL25JP72,100,1000,-1000\n"
        "2025-02-02,buy,IE00BL25JP72,50,600,-600\n"
    )
    targets_csv = tmp_path / "targets_per_holding.csv"
    targets_csv.write_text(TARGETS_CSV)

    # The order list is the single source of truth.
    metrics, config = orchestrator.run(
        config_source=None,
        orders_source=str(orders_csv),
        targets_per_holding_source=str(targets_csv),
    )
    assert not metrics.holdings_df.empty
    assert metrics.total_value > 0
    # The single open ISIN was derived from the orders and carries its
    # joined per-holding target.
    row = metrics.holdings_df.iloc[0]
    assert row["isin"] == "IE00BL25JP72"
    # Derived cost basis = 1000 + 600 = 1600.
    assert row["cost_basis_eur"] == pytest.approx(1600.0)
    # Inception is taken automatically from the first order, not config.
    assert metrics.inception_date == "2025-01-02"


def test_run_without_per_holding_targets(tmp_path, monkeypatch):
    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _no_network_enrich)
    _stub_benchmarks(monkeypatch)
    orders_csv = tmp_path / "order_list.csv"
    orders_csv.write_text(
        "date,type,isin,quantity,gross_eur,net_eur\n"
        "2025-01-02,buy,IE00BL25JP72,100,1000,-1000\n"
    )
    # Order list alone (no per-holding targets) must still run end to end.
    metrics, _ = orchestrator.run(
        orders_source=str(orders_csv),
        targets_per_holding_source=None,
    )
    assert metrics.total_value > 0


def test_run_without_orders_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _no_network_enrich)
    _stub_benchmarks(monkeypatch)
    # No order list → nothing to derive; returns empty metrics, not a crash.
    metrics, _ = orchestrator.run(
        orders_source=str(tmp_path / "does_not_exist.csv"),
    )
    assert metrics.total_value == 0
