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
        "date,type,isin,quantity,gross_eur,net_eur,instrument_kind\n"
        "2025-01-02,buy,IE00BL25JP72,100,1000,-1000,ETF\n"
        "2025-02-02,buy,IE00BL25JP72,50,600,-600,ETF\n"
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
        "date,type,isin,quantity,gross_eur,net_eur,instrument_kind\n"
        "2025-01-02,buy,IE00BL25JP72,100,1000,-1000,ETF\n"
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


# ── Per-holding PORTFOLIO targets + buy-new seeding ──────────────────────────

def test_load_targets_portfolio_and_ticker_key(tmp_path):
    p = tmp_path / "tph.csv"
    p.write_text(
        "name,isin,ticker,target_equities,target_fixed_income,target_portfolio,no_buy_no_sell\n"
        "Held,IE00BDBRDM35,AGGH.MI,,,30,\n"
        "NewOne,,NTSG.DE,,,40,\n"
    )
    t = load_targets_per_holding(str(p))
    # ISIN-keyed row carries the portfolio target.
    assert t["IE00BDBRDM35"]["target_portfolio"] == pytest.approx(30.0)
    # Ticker-only row is kept, keyed by uppercased ticker.
    assert "NTSG.DE" in t
    assert t["NTSG.DE"]["target_portfolio"] == pytest.approx(40.0)
    assert t["NTSG.DE"]["isin"] == ""


def test_seed_missing_targets_only_creates_not_held():
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0),
    ])
    holdings[0].ticker = "AGGH.MI"  # held instrument, matched by ISIN
    targets = {
        "IE00BDBRDM35": {"isin": "IE00BDBRDM35", "ticker": "AGGH.MI",
                         "name": "Held", "target_portfolio": 30.0,
                         "no_buy_no_sell": False},
        "NTSG.DE": {"isin": "", "ticker": "NTSG.DE", "name": "New",
                    "target_portfolio": 40.0, "no_buy_no_sell": False},
    }
    seeded = orchestrator._seed_missing_targets(holdings, targets)
    assert len(seeded) == 1
    s = seeded[0]
    assert s.ticker == "NTSG.DE"
    assert s.quantity == 0.0
    assert s.is_seeded_target is True
    assert s.target_portfolio == pytest.approx(40.0)


def test_seed_skips_held_fund_matched_by_ticker_or_name():
    # Regression (the "NTSG" bug): the order carries only an ISIN (so the
    # holding's ticker defaults to the ISIN), while the target row carries only
    # a ticker for the SAME fund. The canonical instrument_key differs, but the
    # fund is clearly held — it must NOT be seeded as a phantom buy-new, or it
    # shows up twice (once "exit to 0%", once "buy to N%").
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00077IIPQ8", qty=341.0, net=-10000.0),
    ])
    # Order had no ticker → holding.ticker is the ISIN; give it a name too.
    holdings[0].ticker = "IE00077IIPQ8"
    holdings[0].name = "WisdomTree Global Efficient Core UCITS ETF USD Acc"
    targets = {
        # Ticker-keyed target for the same fund (no ISIN) — the mismatch case.
        "NTSG.DE": {"isin": "", "ticker": "NTSG.DE",
                    "name": "WisdomTree Global Efficient Core",
                    "target_portfolio": 35.0, "no_buy_no_sell": False},
    }
    seeded = orchestrator._seed_missing_targets(holdings, targets)
    assert seeded == [], "held fund must not be seeded as a phantom buy-new"


def test_seed_matches_held_by_bare_ticker():
    # A target ticker "NTSG.DE" must match a holding whose ticker is the bare
    # "NTSG" (exchange suffix stripped), and vice-versa — no phantom.
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00077IIPQ8", qty=10.0, net=-1000.0),
    ])
    holdings[0].ticker = "NTSG"  # bare, no exchange suffix
    targets = {
        "NTSG.DE": {"isin": "", "ticker": "NTSG.DE", "name": "WT Efficient Core",
                    "target_portfolio": 35.0, "no_buy_no_sell": False},
    }
    assert orchestrator._seed_missing_targets(holdings, targets) == []


def test_seed_skips_held_fund_with_empty_isin_in_taxonomy():
    # Test our new fallback behavior in resolve_taxonomy_identity:
    # A holding is built with ISIN "FR0010755611" (no ticker, so ticker defaults to ISIN)
    # and name "Amundi MSCI USA (2x) Leveraged".
    # The taxonomy has a row for ticker "CL2" with the same name, but its ISIN cell is empty.
    # The target row has ticker "CL2" and target_portfolio = 8.0.
    # Applying targets should match the holding to the CL2 target (8.0),
    # and seeding should NOT create a duplicate phantom buy-new seed for "CL2".
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "FR0010755611", qty=10.0, net=-1000.0),
    ])
    # Ticker defaults to ISIN pre-enrichment.
    holdings[0].ticker = "FR0010755611"
    holdings[0].name = "Amundi MSCI USA (2x) Leveraged"

    # Target row keyed exactly like the loader keys a ticker-only row.
    row = {"isin": "", "ticker": "CL2", "name": "Amundi MSCI USA (2x) Leveraged",
           "target_portfolio": 8.0, "no_buy_no_sell": False}
    targets = {"CL2": row, "TICKER:CL2": row}

    # 1. Apply target
    orchestrator._apply_per_holding_targets(holdings, targets)
    assert holdings[0].target_portfolio == pytest.approx(8.0)
    assert holdings[0].ticker == "CL2"

    # 2. Seed missing targets - should NOT create a duplicate "CL2" seed!
    seeded = orchestrator._seed_missing_targets(holdings, targets)
    assert seeded == [], "held CL2 must not be seeded as a duplicate phantom buy-new"


def test_apply_target_bridges_isin_to_ticker_via_xref(monkeypatch):
    # Regression (the "UEQC" optimizer bug): a Fineco order carries only an
    # ISIN (holding.ticker defaults to the ISIN, pre-enrichment), the target
    # row is keyed only by TICKER, and the NAMES do NOT prefix-match
    # ("UBS CMCI USD-A-AC" vs "UBS CMCI Commodity Carry"). Only the learned
    # ISIN↔ticker xref can bridge them; without it the target never attaches,
    # the position sells to 0%, and a phantom buy-new is seeded — the SAME fund
    # as both BUY and SELL.
    from tarzan.data import price_cache
    monkeypatch.setattr(price_cache, "load_ticker_isin_reverse",
                        lambda isin: "UEQC" if isin == "IE00BKFB6L02" else None)
    monkeypatch.setattr(price_cache, "load_ticker_isin", lambda t: None)
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BKFB6L02", qty=40.0, net=-5099.9),
    ])
    holdings[0].ticker = "IE00BKFB6L02"  # no ticker on the order → ISIN
    holdings[0].name = "UBS CMCI USD-A-AC"  # broker name, does NOT match target
    # Target keyed exactly like the real loader keys a ticker-only row.
    row = {"isin": "", "ticker": "UEQC.DE", "name": "UBS CMCI Commodity Carry",
           "target_portfolio": 5.0, "no_buy_no_sell": False}
    targets = {"UEQC.DE": row, "TICKER:UEQC": row}
    orchestrator._apply_per_holding_targets(holdings, targets)
    assert holdings[0].target_portfolio == pytest.approx(5.0)
    # And it must NOT be seeded as a phantom buy-new.
    assert orchestrator._seed_missing_targets(holdings, targets) == []


def test_apply_target_bridges_ticker_to_isin_via_xref(monkeypatch):
    # The mirror case: holding known by ticker, target keyed by ISIN, bridged
    # through the ticker→ISIN xref.
    from tarzan.data import price_cache
    monkeypatch.setattr(price_cache, "load_ticker_isin_reverse", lambda isin: None)
    monkeypatch.setattr(price_cache, "load_ticker_isin",
                        lambda t: "IE00BKFB6L02" if t.split(".")[0].upper() == "UEQC" else None)
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "PLACEHOLDER01", qty=40.0, net=-5099.9),
    ])
    # Holding known by ticker only (ISIN not carried on this leg).
    holdings[0].isin = ""
    holdings[0].ticker = "UEQC.DE"
    holdings[0].name = "UBS CMCI USD-A-AC"
    row = {"isin": "IE00BKFB6L02", "ticker": "", "name": "UBS CMCI Commodity Carry",
           "target_portfolio": 5.0, "no_buy_no_sell": False}
    targets = {"IE00BKFB6L02": row}
    orchestrator._apply_per_holding_targets(holdings, targets)
    assert holdings[0].target_portfolio == pytest.approx(5.0)


def test_per_holding_only_objective_uses_portfolio_targets():
    import numpy as np
    from tarzan.engine.rebalancer import _ObjectiveModel
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.holding import AssetClass

    cfg = InvestorConfig()
    cfg.target_use_per_holding_only = True
    cfg.target_cash_buffer_eur = 0.0

    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0),
        _o(OrderType.BUY, "IE00BL25JP72", qty=10.0, net=-1000.0),
    ])
    for h in holdings:
        h.asset_class = AssetClass.EQUITIES
    holdings[0].target_portfolio = 70.0
    holdings[1].target_portfolio = 30.0

    model = _ObjectiveModel(holdings, cfg, np.array([6000.0, 4000.0]))
    assert model.per_holding_only is True
    # Only per-holding-portfolio objectives (no asset/geo labels).
    assert all(kind == "ph_pf" for kind, _ in model.labels)
    assert list(model.targets) == [70.0, 30.0]


def test_zero_target_pins_full_sell_when_selling_allowed():
    import numpy as np
    from tarzan.engine.rebalancer import _bounds
    from tarzan.models.holding import AssetClass

    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0),
        _o(OrderType.BUY, "IE00BL25JP72", qty=10.0, net=-1000.0),
    ])
    for h in holdings:
        h.asset_class = AssetClass.EQUITIES
    holdings[0].target_portfolio = 0.0     # unlisted → exit fully
    holdings[1].target_portfolio = 100.0
    values = np.array([2000.0, 3000.0])

    # Selling allowed: the 0%-target position is pinned to a full sell,
    # regardless of tolerance; the positive-target one stays freely tradeable.
    lo, hi, tr = _bounds(holdings, values, no_sell=False, per_holding_only=True)
    assert lo[0] == hi[0] == -2000.0 and not tr[0]
    assert tr[1] and lo[1] == -3000.0

    # Buy-only: nothing is force-sold (can't sell at all).
    lo2, _, tr2 = _bounds(holdings, values, no_sell=True, per_holding_only=True)
    assert lo2[0] == 0.0 and tr2[0]

    # Classic mode (flag off): no pinning even with selling allowed.
    lo3, hi3, tr3 = _bounds(holdings, values, no_sell=False, per_holding_only=False)
    assert tr3[0] and hi3[0] == np.inf


def test_extract_actions_carries_current_target_after():
    # The optimizer newsletter table needs structured Now → Target → After per
    # trade (not just an amount). Verify _extract_actions emits them, and that
    # After = current + trade (in EUR), with % computed off the invested base.
    import numpy as np
    from tarzan.engine.rebalancer import _extract_actions, _ObjectiveModel
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.holding import AssetClass

    cfg = InvestorConfig()
    cfg.target_use_per_holding_only = True
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0),
        _o(OrderType.BUY, "IE00BK5BQT80", qty=10.0, net=-3000.0),
    ])
    holdings[0].asset_class = AssetClass.EQUITIES
    holdings[1].asset_class = AssetClass.EQUITIES
    holdings[0].target_portfolio = 60.0
    holdings[1].target_portfolio = 40.0
    values = np.array([1000.0, 3000.0])
    model = _ObjectiveModel(holdings, cfg, values)
    trade = np.array([+1400.0, -1400.0])  # move toward 60/40 of 4000
    actions = {a["ticker"]: a for a in _extract_actions(trade, holdings, model, values)}

    a0 = actions[holdings[0].ticker]
    assert a0["direction"] == "buy"
    assert a0["current_eur"] == 1000.0
    assert a0["target_pct"] == 60.0
    # After EUR = current + trade; % on the post-trade invested base (4000).
    assert a0["after_eur"] == pytest.approx(2400.0)
    assert a0["after_pct"] == pytest.approx(60.0, abs=0.5)
    # target_eur is target_pct × invested base.
    assert a0["target_eur"] == pytest.approx(0.60 * 4000.0, abs=1.0)


def test_plan_cost_cgt_and_fees():
    # plan_cost estimates CGT on the sells (per-unit tax model) + fixed fees,
    # reusing the same _tax_per_unit_sold the optimizer used — so the displayed
    # cost matches the cash-conservation the plan solved for.
    from tarzan.engine.rebalancer import plan_cost
    from tarzan.models.investor_config import InvestorConfig

    cfg = InvestorConfig()
    cfg.rebalancing_transaction_fee_buy_eur = 19.0
    cfg.rebalancing_transaction_fee_sell_eur = 19.0
    cfg.rebalancing_capital_gains_tax_standard_pctg = 26.0
    holdings = build_holdings_from_orders([
        _o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0),
        _o(OrderType.BUY, "IE00BK5BQT80", qty=10.0, net=-1000.0),
    ])
    holdings[0].gain_pct = 100.0   # +100% → taxable fraction 100/200 = 0.5
    holdings[1].gain_pct = 0.0     # flat → no CGT
    actions = [
        {"idx": 0, "direction": "sell", "amount_eur": 2000.0},  # appreciated
        {"idx": 1, "direction": "buy", "amount_eur": 500.0},
    ]
    cost = plan_cost(actions, holdings, cfg)
    # CGT = 2000 × 0.26 × (100/200) = 260 ; fees = 1 buy + 1 sell = 38
    assert cost["cgt_eur"] == pytest.approx(260.0, abs=0.5)
    assert cost["fees_eur"] == pytest.approx(38.0)


def test_plan_cost_buy_only_has_no_cgt():
    from tarzan.engine.rebalancer import plan_cost
    from tarzan.models.investor_config import InvestorConfig
    cfg = InvestorConfig()
    cfg.rebalancing_transaction_fee_buy_eur = 19.0
    cfg.rebalancing_capital_gains_tax_standard_pctg = 26.0
    holdings = build_holdings_from_orders([_o(OrderType.BUY, "IE00BDBRDM35", qty=10.0, net=-1000.0)])
    holdings[0].gain_pct = 50.0
    cost = plan_cost([{"idx": 0, "direction": "buy", "amount_eur": 500.0}], holdings, cfg)
    assert cost["cgt_eur"] == 0.0          # no sells → no CGT
    assert cost["fees_eur"] == pytest.approx(19.0)
