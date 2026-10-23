"""Account-cash and cash-equivalent securities are different mechanics.

The strategic allocation remains Cash & Cash Equivalents for both, but a
security can fund an acquisition only through a serialized sell with its
commission, capital-gains tax and EUR-per-unit quantity change.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tarzan.engine import rebalancer as rb
from tarzan.instruments.registry import InstrumentKind, TypeEvidenceGateway
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig


def _holding(
    ticker="RESERVE.MI", *, kind="ETF", value=1000.0, price=10.0,
    quantity=None, asset_class=AssetClass.CASH_EQUIVALENTS,
    gain=0.0, target=None, frozen=False,
):
    quantity = value / price if quantity is None and price else quantity or 0.0
    return Holding(
        isin=f"ID:{ticker}", ticker=ticker, quantity=quantity,
        cost_basis_eur=value / (1.0 + gain / 100.0),
        market_value_eur=value, currency="EUR", name="Cash",
        current_price=price, current_value=value, security_type=kind,
        asset_class=asset_class, gain_pct=gain, target_portfolio=target,
        no_buy_no_sell=frozen,
    )


def _config(*, buffer=900.0, no_sell=False, per_holding=False, tax=26.0, fee=1.0):
    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {"Equities": 100.0}
    config.equity_geo_targets_pctg = {}
    config.target_cash_buffer_eur = buffer
    config.rebalancing_no_sell = no_sell
    config.target_use_per_holding_only = per_holding
    config.rebalancing_transaction_fee_buy_eur = fee
    config.rebalancing_transaction_fee_sell_eur = fee
    config.rebalancing_capital_gains_tax_standard_pctg = tax
    config.rebalancing_capital_gains_tax_government_pctg = 12.5
    return config


def _proof(holdings, actions, *, config=None, contribution=0.0, buffer=900.0):
    config = config or _config(buffer=buffer)
    values = np.array([h.current_value for h in holdings], dtype=float)
    return rb._finalize_action_proof(
        holdings=holdings, values=values, actions=actions, config=config,
        external_contribution=contribution, protected_cash=buffer,
    )


def _actions(holdings, trades, config):
    values = np.array([h.current_value for h in holdings], dtype=float)
    model = rb._ObjectiveModel(holdings, config, values)
    return rb._extract_actions(np.array(trades, dtype=float), holdings, model, values)


def test_cash_equivalent_etf_is_not_already_settled_money():
    equity = _holding("EQUITY", asset_class=AssetClass.EQUITIES)
    reserve = _holding("CASH", kind="ETF")
    before = copy.deepcopy([equity, reserve])
    reconstructed, proof = _proof(
        [equity, reserve],
        [{"idx": 0, "direction": "buy", "amount_eur": 100.0}],
    )
    assert reconstructed[1] == 1000.0
    assert proof["initial_cash_eur"] == 0.0
    assert proof["ending_cash_eur"] == -101.0
    assert proof["status"] == "NON_EXECUTABLE"
    assert reserve.quantity == before[1].quantity
    assert reserve.cost_basis_eur == before[1].cost_basis_eur


def test_cash_equivalent_etf_can_fund_a_buy_only_via_an_explicit_taxed_sell():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("RESERVE", kind="ETF", gain=100.0),
    ]
    config = _config()
    actions = _actions(holdings, [85.0, -100.0], config)
    reconstructed, proof = _proof(holdings, actions, config=config)
    assert proof["status"] == "EXECUTABLE"
    assert proof["initial_cash_eur"] == 0.0
    assert proof["gross_sales_eur"] == 100.0
    assert proof["gross_purchases_eur"] == 85.0
    assert proof["estimated_tax_eur"] == 13.0
    assert proof["fees_eur"] == 2.0
    assert proof["ending_cash_eur"] == 0.0
    assert reconstructed.tolist() == [1085.0, 900.0]
    sale = next(action for action in actions if action["direction"] == "sell")
    assert sale["quantity_delta"] == -10.0
    assert sale["after_quantity"] == 90.0
    assert proof["units_invariants_satisfied"]
    assert proof["portfolio_wealth_after_eur"] == 1985.0


@pytest.mark.parametrize("kind", ["ETF", "BOND", "STOCK"])
def test_cash_equivalent_security_is_tradeable_not_a_virtual_account(kind):
    holding = _holding("CASH", kind=kind)
    values = np.array([1000.0])
    lo, hi, tradeable = rb._bounds([holding], values, False, True)
    assert lo.tolist() == [-1000.0]
    assert np.isposinf(hi[0])
    assert tradeable.tolist() == [True]


def test_cash_equivalent_explicit_zero_target_remains_forced_exit():
    holding = _holding("RESERVE", target=0.0)
    lo, hi, tradeable = rb._bounds([holding], np.array([1000.0]), False, True)
    assert lo.tolist() == hi.tolist() == [-1000.0]
    assert not tradeable[0]


@pytest.mark.parametrize("direction", ["buy", "sell"])
def test_true_account_cash_cannot_be_an_explicit_security_trade(direction):
    account = _holding("ETF", kind="CASH", price=1.0)
    _, proof = _proof(
        [account], [{"idx": 0, "direction": direction, "amount_eur": 100.0}],
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["frozen_positions_satisfied"]


def test_no_sell_is_verified_independently_from_serialized_security_actions():
    reserve = _holding(kind="ETF")
    _, proof = _proof(
        [reserve], [{"idx": 0, "direction": "sell", "amount_eur": 100.0}],
        config=_config(no_sell=True),
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["no_sell_satisfied"]


def test_residue_without_cash_holding_is_a_plan_ledger_not_new_etf_units():
    reserve = _holding(kind="ETF")
    reconstructed, proof = _proof([reserve], [], contribution=123.45)
    assert reconstructed.tolist() == [1000.0]
    assert reserve.quantity == 100.0
    assert proof["initial_cash_eur"] == 0.0
    assert proof["ending_cash_eur"] == 123.45
    assert proof["unallocated_settlement_cash_eur"] == 123.45
    assert proof["ending_cash_equivalents_eur"] == 1123.45
    assert proof["portfolio_wealth_after_eur"] == 1123.45


@pytest.mark.parametrize("kind", [None, "FUND", "ETF"])
def test_unknown_or_conflicting_cash_equivalents_fail_closed(kind):
    reserve = _holding(kind=kind)
    if kind == "ETF":
        reserve.instrument_resolution = TypeEvidenceGateway().resolve("ETF", "CASH")
    reconstructed, proof = _proof([reserve], [])
    assert reconstructed.tolist() == [1000.0]
    assert proof["initial_cash_eur"] == 0.0
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["cash_mechanics_satisfied"]


def test_bond_quantity_uses_enriched_eur_per_unit_without_second_divide_100():
    bond = _holding(kind="BOND", value=1060.0, price=1.06, quantity=1000.0)
    config = _config(buffer=954.0, tax=0.0, fee=0.0)
    actions = _actions([bond], [-106.0], config)
    reconstructed, proof = _proof(
        [bond], actions, config=config, buffer=954.0,
    )
    assert reconstructed.tolist() == [954.0]
    assert actions[0]["quantity_delta"] == -100.0
    assert actions[0]["after_quantity"] == 900.0
    assert proof["ending_cash_eur"] == 106.0
    assert proof["status"] == "EXECUTABLE"


def test_fixture_uses_explicit_mechanics_not_name_ticker_or_category():
    cash = _holding("MONEY.MI", kind="CASH", price=1.0)
    security = _holding("CASH", kind="ETF")
    values = np.array([1000.0, 1000.0])
    model = rb._ObjectiveModel([cash, security], _config(), values)
    assert model.settlement_cash_mask.tolist() == [1.0, 0.0]
    assert model.cash_security_mask.tolist() == [0.0, 1.0]
    assert model.cash_mask.tolist() == [1.0, 1.0]
    assert model.cash_mechanics.kinds == (InstrumentKind.CASH, InstrumentKind.ETF)


@pytest.mark.parametrize("bank", [0.0, 100.0, 700.0])
@pytest.mark.parametrize("security", [0.0, 700.0, 1000.0])
@pytest.mark.parametrize("contribution", [0.0, 100.0, 1000.0])
def test_account_and_security_reserve_are_proved_separately(bank, security, contribution):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("BANK", kind="CASH", value=bank, price=1.0),
        _holding("SEC", value=security, price=10.0),
    ]
    config = _config(buffer=900.0, tax=0.0, fee=0.0)
    spendable = max(bank + contribution - max(900.0 - security, 0.0), 0.0)
    actions = (
        [{"idx": 0, "direction": "buy", "amount_eur": spendable}]
        if spendable else []
    )
    reconstructed, proof = _proof(
        holdings, actions, config=config, contribution=contribution,
    )
    assert reconstructed[2] == security
    assert proof["initial_cash_eur"] == bank
    assert proof["ending_cash_eur"] == bank + contribution - spendable
    assert proof["ending_cash_equivalents_eur"] == bank + contribution - spendable + security
    assert proof["cash_equivalent_securities_after_eur"] == security
    assert proof["protected_cash_eur"] == max(900.0 - security, 0.0)
    assert proof["portfolio_wealth_after_eur"] == 1000.0 + bank + security + contribution
    expected = "EXECUTABLE" if bank + contribution + security >= 900.0 else "NON_EXECUTABLE"
    assert proof["status"] == expected


def test_only_real_account_rows_share_a_settlement_balance_change():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("BANK1", kind="CASH", value=100.0, price=1.0),
        _holding("BANK2", kind="CASH", value=200.0, price=1.0),
        _holding("ETF", value=1000.0),
    ]
    reconstructed, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 50.0}],
        config=_config(buffer=800.0, fee=0.0), buffer=800.0,
    )
    assert reconstructed[1] == pytest.approx(250.0 / 3.0)
    assert reconstructed[2] == pytest.approx(500.0 / 3.0)
    assert reconstructed[3] == 1000.0
    assert proof["initial_cash_eur"] == 300.0
    assert proof["ending_cash_eur"] == 250.0
    assert proof["wealth_residual_eur"] == 0.0
    assert [h.quantity for h in holdings[1:]] == [100.0, 200.0, 100.0]


def test_cash_without_initial_balance_receives_contribution_only_in_account_ledger():
    holdings = [
        _holding("BANK1", kind="CASH", value=0.0, price=1.0),
        _holding("BANK2", kind="CASH", value=0.0, price=1.0),
        _holding("ETF", value=1000.0),
    ]
    reconstructed, proof = _proof(holdings, [], contribution=123.45)
    assert reconstructed.tolist() == [123.45, 0.0, 1000.0]
    assert proof["unallocated_settlement_cash_eur"] == 0.0
    assert proof["portfolio_wealth_after_eur"] == 1123.45
    assert [h.quantity for h in holdings] == [0.0, 0.0, 100.0]


@pytest.mark.parametrize("no_sell", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("target", [None, 0.0, 10.0])
def test_cash_security_bounds_retain_no_sell_frozen_and_explicit_target_authority(
    no_sell, frozen, target,
):
    security = _holding(target=target, frozen=frozen)
    lo, hi, tradeable = rb._bounds([security], np.array([1000.0]), no_sell, True)
    if frozen:
        assert lo[0] == hi[0] == 0.0
        assert not tradeable[0]
    elif target == 0.0 and not no_sell:
        assert lo[0] == hi[0] == -1000.0
        assert not tradeable[0]
    else:
        assert lo[0] == (0.0 if no_sell else -1000.0)
        assert np.isposinf(hi[0]) and tradeable[0]


@pytest.mark.parametrize("evidence", [
    ("security_type", "ETF", "instrument_type", "CASH"),
    ("security_type", "ETF", "instrument_kind_evidence", ("CASH",)),
    ("security_type", "CASH", "instrument_kind_evidence", ("ETF",)),
    ("security_type", "FUND", "instrument_kind_evidence", ()),
])
def test_conflicting_or_missing_mechanics_are_never_inferred_from_market_value(evidence):
    security = _holding()
    for field, value in zip(evidence[::2], evidence[1::2]):
        setattr(security, field, value)
    lo, hi, tradeable = rb._bounds([security], np.array([1000.0]), False, True)
    assert lo[0] == hi[0] == 0.0 and not tradeable[0]
    _, proof = _proof([security], [], contribution=100.0)
    assert proof["initial_cash_eur"] == 0.0
    assert proof["ending_cash_eur"] == 100.0
    assert not proof["cash_mechanics_satisfied"]


@pytest.mark.parametrize("field", [
    "price_eur_per_unit", "current_quantity", "quantity_delta", "after_quantity",
])
@pytest.mark.parametrize("bad_value", [None, True, "100.0", float("nan"), float("inf"), -99999.0])
def test_unit_metadata_cannot_override_or_coerce_derived_trade_units(field, bad_value):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("SEC", gain=100.0),
    ]
    actions = _actions(holdings, [85.0, -100.0], _config())
    sale = next(action for action in actions if action["direction"] == "sell")
    sale[field] = bad_value
    _, proof = _proof(holdings, actions)
    assert not proof["units_invariants_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


@pytest.mark.parametrize("action", [
    {"idx": True, "direction": "sell", "amount_eur": 10.0},
    {"idx": -1, "direction": "sell", "amount_eur": 10.0},
    {"idx": 999, "direction": "sell", "amount_eur": 10.0},
    {"idx": 0, "direction": "wat", "amount_eur": 10.0},
    {"idx": 0, "direction": "buy", "amount_eur": -10.0},
    {"idx": 0, "direction": "buy", "amount_eur": 0.0},
    {"idx": 0, "direction": "buy", "amount_eur": "10.0"},
    {"idx": 0, "direction": "buy", "amount_eur": True},
    {"idx": 0, "direction": "buy", "amount_eur": float("nan")},
    {"idx": 0, "direction": "buy", "amount_eur": float("inf")},
    {"idx": 0, "direction": "buy", "amount_eur": 10.001},
])
def test_malformed_serialized_actions_never_prove_an_executable_source(action):
    _, proof = _proof([_holding()], [action], contribution=100.0)
    assert not proof["serialized_actions_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


def test_duplicate_actions_cannot_hide_a_gross_oversell_as_a_net_position():
    security = _holding()
    _, proof = _proof([security], [
        {"idx": 0, "direction": "sell", "amount_eur": 2000.0},
        {"idx": 0, "direction": "buy", "amount_eur": 2000.0},
    ])
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["serialized_actions_satisfied"]
    assert not proof["units_invariants_satisfied"]


@pytest.mark.parametrize("price", [None, 0.0, -1.0, "10.0", True, float("nan"), float("inf"), 11.0])
def test_unproved_eur_per_unit_price_cannot_authorize_cash_security_units(price):
    security = _holding()
    security.current_price = price
    _, proof = _proof([security], [
        {"idx": 0, "direction": "sell", "amount_eur": 100.0},
    ], config=_config(fee=0.0, tax=0.0))
    assert not proof["units_invariants_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


@pytest.mark.parametrize("fee,gain", [(1.0, 0.0), (0.0, 100.0), (1.0, 100.0)])
@pytest.mark.parametrize("other_frozen", [False, True])
def test_cash_security_shuffle_does_not_count_tax_or_fees_as_target_progress(
    fee, gain, other_frozen,
):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, frozen=other_frozen),
        _holding("SEC", gain=gain),
    ]
    config = _config(fee=fee)
    _, proof = _proof(
        holdings, [{"idx": 1, "direction": "sell", "amount_eur": 10.0}],
        config=config,
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["non_cost_only_progress_satisfied"]
    assert proof["ending_cash_equivalents_eur"] < 1000.0


def test_free_cash_security_shuffle_has_no_cost_only_wealth_burn():
    security = _holding()
    _, proof = _proof(
        [security], [{"idx": 0, "direction": "sell", "amount_eur": 10.0}],
        config=_config(tax=0.0, fee=0.0),
    )
    assert proof["status"] == "EXECUTABLE"
    assert proof["non_cost_only_progress_satisfied"]
    assert proof["ending_cash_equivalents_eur"] == 1000.0


@pytest.mark.parametrize("target,amount", [(0.0, 1000.0), (20.0, 100.0)])
def test_explicit_cash_security_target_is_not_replaced_by_cost_only_policy(target, amount):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, frozen=True, target=100.0),
        _holding("SEC", gain=100.0, target=target),
    ]
    config = _config(per_holding=True)
    _, proof = _proof(
        holdings, [{"idx": 1, "direction": "sell", "amount_eur": amount}],
        config=config, contribution=100.0,
    )
    assert proof["status"] == "EXECUTABLE"
    assert proof["non_cost_only_progress_satisfied"]


def test_an_explicit_zero_target_still_does_not_override_hard_reserve_funding():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, frozen=True, target=100.0),
        _holding("SEC", gain=100.0, target=0.0),
    ]
    _, proof = _proof(
        holdings, [{"idx": 1, "direction": "sell", "amount_eur": 1000.0}],
        config=_config(per_holding=True),
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert proof["non_cost_only_progress_satisfied"]
    assert not proof["protected_cash_satisfied"]


@pytest.mark.parametrize("frozen", [False, True])
def test_final_cost_only_shuffle_is_suppressed_only_if_empty_plan_is_feasible(frozen):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, frozen=frozen),
        _holding("SEC", gain=100.0),
    ]
    before = copy.deepcopy(holdings)
    actions, verifications = rb.optimize_local_search(
        holdings, _config(), 2000.0,
        params=rb.LSParams(n_restarts=0, max_iters_per_descent=40),
    )
    funding = verifications[-1]
    assert actions == []
    assert funding["status"] == "EXECUTABLE"
    if funding.get("unneeded_cash_security_trades_suppressed"):
        assert funding["suppressed_cost_eur"] > 0
        assert funding["suppression_reason"]
    assert funding["estimated_tax_eur"] == funding["fees_eur"] == 0.0
    assert funding["portfolio_wealth_after_eur"] == 2000.0
    assert holdings == before


@pytest.mark.parametrize("bank", [0.001, 0.004, 0.006, 0.009])
def test_subcent_bank_deficit_is_not_a_valid_half_cent_loan(bank):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("BANK", kind="CASH", value=bank, price=1.0),
        _holding("SEC"),
    ]
    _, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 0.01}],
        config=_config(fee=0.0),
    )
    assert proof["ending_cash_eur"] == round(bank - 0.01, 2)
    assert not proof["settlement_cash_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


def _fixed_plan(monkeypatch, holdings, config, trades, contribution=0.0, min_trade=1.0):
    candidate = np.array(trades, dtype=float)
    monkeypatch.setattr(
        rb, "_descent", lambda *_args: (candidate.copy(), 0.0),
    )
    monkeypatch.setattr(
        rb, "_project_to_budget", lambda t, *_args: t.copy(),
    )
    return rb.optimize_local_search(
        holdings, config, sum(h.current_value for h in holdings),
        lump_sum=contribution,
        params=rb.LSParams(n_restarts=0, min_trade_eur=min_trade),
    )


@pytest.mark.parametrize("bank", [0.001, 0.004, 0.006, 0.009])
def test_cent_reconciliation_repairs_bank_cash_without_shrinking_etf_units(monkeypatch, bank):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("BANK", kind="CASH", value=bank, price=1.0),
        _holding("SEC"),
    ]
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, _config(fee=0.0), [0.01, 0.0, 0.0],
        min_trade=0.01,
    )
    assert actions == []
    assert verifications[-1]["status"] == "EXECUTABLE"
    assert verifications[-1]["ending_settlement_cash_eur"] == bank
    assert verifications[-1]["ending_cash_equivalents_eur"] == round(1000.0 + bank, 2)
    assert holdings[2].quantity == 100.0


def test_a_clipped_cash_security_buy_does_not_magically_refill_strategic_shortfall(monkeypatch):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("SEC", value=800.0),
    ]
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, _config(fee=1.0), [0.0, 100.0],
        contribution=100.0,
    )
    # With fee 1 the purchase leaves only 899 combined cash equivalents;
    # canceling that fee (not virtually growing the ETF) repairs the floor.
    assert actions == []
    funding = verifications[-1]
    assert funding["status"] == "EXECUTABLE"
    assert funding["ending_cash_eur"] == 100.0
    assert funding["cash_equivalent_securities_after_eur"] == 800.0
    assert funding["ending_cash_equivalents_eur"] == 900.0
    assert holdings[1].quantity == 80.0


def test_canceled_trade_fee_is_recomputed_from_final_cent_actions(monkeypatch):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("SEC", value=900.0),
    ]
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, _config(fee=1.0), [1.0, 0.0],
        contribution=1.0,
    )
    assert actions == []
    assert verifications[-1]["fees_eur"] == 0.0
    assert verifications[-1]["ending_cash_eur"] == 1.0
    assert verifications[-1]["status"] == "EXECUTABLE"


def test_full_cash_security_sale_never_rounds_up_past_owned_units(monkeypatch):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, target=100.0),
        _holding("SEC", value=1000.006, target=0.0),
    ]
    config = _config(buffer=0.0, fee=0.0, tax=0.0, per_holding=True)
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, config, [1000.0, -1000.006],
    )
    sale = next(action for action in actions if action["direction"] == "sell")
    assert sale["amount_eur"] == 1000.0
    assert sale["after_quantity"] >= 0
    assert verifications[-1]["position_invariants_satisfied"]
    assert verifications[-1]["units_invariants_satisfied"]
    assert verifications[-1]["status"] == "EXECUTABLE"


def test_unallocated_account_residue_enters_cash_and_wealth_verification(monkeypatch):
    holdings = [_holding("EQ", asset_class=AssetClass.EQUITIES)]
    config = _config(buffer=100.0, no_sell=True, tax=0.0, fee=0.0)
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, config, [0.0], contribution=123.45,
    )
    assert actions == []
    cash = next(row for row in verifications if row["kind"] == "cash")
    assert cash["items"][0]["actual_eur"] == 123.45
    assert cash["items"][0]["target_eur"] == 100.0
    assert verifications[-1]["portfolio_wealth_after_eur"] == 1123.45


def test_buy_only_deployment_does_not_spend_cash_equivalent_security_surplus(monkeypatch):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("SEC", value=1000.0),
    ]
    config = _config(buffer=900.0, no_sell=True, tax=0.0, fee=0.0)
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, config, [2000.0, 0.0], contribution=2000.0,
    )
    assert len(actions) == 1 and actions[0]["amount_eur"] == 2000.0
    assert verifications[-1]["initial_cash_eur"] == 0.0
    assert verifications[-1]["cash_equivalent_securities_after_eur"] == 1000.0
    assert verifications[-1]["ending_cash_eur"] == 0.0
    assert verifications[-1]["status"] == "EXECUTABLE"


@pytest.mark.parametrize("bank", [0.0, 200.0])
@pytest.mark.parametrize("security", [500.0, 1000.0])
@pytest.mark.parametrize("target", [0.0, 900.0, 1000.0])
@pytest.mark.parametrize("contribution", [0.0, 1000.0])
def test_hybrid_search_cash_equations_match_an_independent_scalar_ledger(
    bank, security, target, contribution,
):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, gain=100.0),
        _holding("BANK", kind="CASH", value=bank, price=1.0),
        _holding("SEC", value=security, gain=100.0),
    ]
    config = _config(buffer=target)
    values = np.array([h.current_value for h in holdings])
    model = rb._ObjectiveModel(holdings, config, values)
    funding = rb._CashFunding(
        values, model.cash_security_mask, bank, contribution, target,
    )
    params = rb.LSParams()
    cost = rb._Cost(model, values, config, 0.0, params, cash_funding=funding)
    for trades in (
        np.array([100.0, 0.0, -100.0]),
        np.array([0.0, 0.0, 100.0]),
        np.array([300.0, 0.0, 0.0]),
        np.array([-100.0, 0.0, 100.0]),
        np.zeros(3),
    ):
        # This oracle intentionally does not call production cash-cost/budget
        # helpers. Tax is 13% of proceeds for these +100% ETF positions.
        purchases = sum(float(t) for t in trades if t > 0)
        sales = sum(-float(t) for t in trades if t < 0)
        taxed_gain = sum(-float(t) * 0.13 for t in trades if t < 0)
        commissions = sum(1.0 for t in trades if abs(t) >= 1.0)
        net_cost = purchases - (sales - taxed_gain) + commissions
        expected_bank = bank + contribution - net_cost
        expected_security = security + float(trades[2])
        expected_reserve = max(target - expected_security, 0.0)
        expected_budget = bank + contribution - expected_reserve
        assert funding.security_value(trades) == expected_security
        assert funding.protected_settlement_cash(trades) == expected_reserve
        assert funding.budget(trades) == expected_budget

        gaps = model.gaps(values + trades)
        if target > 0:
            gaps[-1] = (expected_bank + expected_security - target) / target * 100.0
        gaps = np.abs(gaps)
        expected = cost.gap_term(gaps)
        excess = np.clip(gaps - cost.allowed, 0.0, None)
        expected += params.guard_weight * float(np.sum(excess ** 2))
        expected += params.turnover_weight * float(np.abs(trades).sum()) / cost.tv
        expected += params.trade_count_weight * int((np.abs(trades) >= 1.0).sum())
        expected += params.residual_weight * (
            (net_cost - expected_budget) / cost.tv * 100.0
        ) ** 2
        expected += params.residual_weight * (
            max(-expected_bank, 0.0) / cost.tv * 100.0
        ) ** 2
        assert cost(trades) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_cash_security_government_bond_uses_same_tax_gate_as_other_bonds():
    bond = _holding("TBILL", kind="BOND", gain=100.0, price=1.06)
    bond.name = "BTP Italia 2030"
    config = _config(tax=26.0, fee=0.0)
    actions = [{"idx": 0, "direction": "sell", "amount_eur": 100.0}]
    assert rb.plan_cost(actions, [bond], config)["cgt_eur"] == 6.25
    bond.security_type = "ETF"
    assert rb.plan_cost(actions, [bond], config)["cgt_eur"] == 13.0


def test_duplicate_ticker_cash_securities_keep_distinct_units_and_values():
    holdings = [
        _holding("COLLISION", value=1000.0, price=10.0),
        _holding("COLLISION", value=2000.0, price=20.0, target=0.0),
        _holding("EQ", asset_class=AssetClass.EQUITIES, target=100.0),
    ]
    holdings[1].isin = "A-DIFFERENT-ISIN"
    config = _config(buffer=900.0, tax=0.0, fee=0.0, per_holding=True)
    actions = _actions(holdings, [0.0, -2000.0, 2000.0], config)
    reconstructed, proof = _proof(
        holdings, actions, config=config,
    )
    assert reconstructed.tolist() == [1000.0, 0.0, 3000.0]
    changes = proof["cash_equivalent_security_unit_changes"]
    assert len(changes) == 1
    assert changes[0]["idx"] == 1 and changes[0]["isin"] == "A-DIFFERENT-ISIN"
    assert changes[0]["quantity_delta"] == -100.0
    assert holdings[0].quantity == holdings[1].quantity == 100.0
    assert proof["status"] == "EXECUTABLE"


def test_explicit_cash_security_target_survives_per_holding_verification():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, target=100.0),
        _holding("SEC", target=0.0),
    ]
    values = np.array([1000.0, 1000.0])
    config = _config(per_holding=True, fee=0.0, tax=0.0)
    model = rb._ObjectiveModel(holdings, config, values)
    assert model.ph_pf_idx == [0, 1]
    verifications = rb._verify(
        np.array([1000.0, 0.0]), holdings, config, model.geo_frac, model.all_geos,
        unallocated_settlement_cash=1000.0,
        cash_mechanics=model.cash_mechanics,
    )
    cash = next(row for row in verifications if row["kind"] == "cash")
    assert cash["items"][0]["actual_eur"] == 1000.0
    per_holding = next(row for row in verifications if row["kind"] == "per_holding_portfolio")
    assert per_holding["status"] == "✓ OK"


@pytest.mark.parametrize("kind", ["ETF", "BOND", "STOCK"])
def test_ordinary_security_oversell_below_a_half_cent_is_not_executable(kind):
    holding = _holding(
        "ORDINARY", kind=kind, value=1000.006, price=10.0,
        asset_class=AssetClass.EQUITIES,
    )
    _, proof = _proof(
        [holding], [{"idx": 0, "direction": "sell", "amount_eur": 1000.01}],
        config=_config(buffer=0.0, tax=0.0, fee=0.0), buffer=0.0,
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["position_invariants_satisfied"]
    assert not proof["units_invariants_satisfied"]


@pytest.mark.parametrize("kind", ["ETF", "BOND", "STOCK"])
@pytest.mark.parametrize("field,forged", [
    ("price_eur_per_unit", 999.0),
    ("current_quantity", 999.0),
    ("quantity_delta", 999.0),
    ("after_quantity", 1099.0),
    ("quantity_delta", "10.0"),
    ("after_quantity", True),
])
def test_ordinary_security_unit_metadata_is_independently_verified(kind, field, forged):
    holding = _holding("ORDINARY", kind=kind, asset_class=AssetClass.EQUITIES)
    action = {"idx": 0, "direction": "buy", "amount_eur": 100.0, field: forged}
    _, proof = _proof(
        [holding], [action], contribution=100.0, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["units_invariants_satisfied"]


@pytest.mark.parametrize("field,mismatching", [
    ("isin", "A-DIFFERENT-ISIN"),
    ("isin", None),
    ("isin", ""),
    ("isin", True),
    ("ticker", "A-DIFFERENT-TICKER"),
    ("ticker", "ORDINARY.A-DIFFERENT-VENUE"),
    ("ticker", None),
    ("ticker", ""),
    ("ticker", True),
])
def test_present_action_identity_cannot_override_its_holding_index(field, mismatching):
    holding = _holding("ORDINARY", asset_class=AssetClass.EQUITIES)
    action = {"idx": 0, "direction": "buy", "amount_eur": 100.0, field: mismatching}
    reconstructed, proof = _proof(
        [holding], [action], contribution=100.0, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["serialized_actions_satisfied"]
    assert not proof["action_identity_satisfied"]
    assert reconstructed.tolist() == [1000.0]


@pytest.mark.parametrize("action", [None, True, 123, 10.0, "buy", [], ("buy", 100)])
def test_non_mapping_action_is_non_executable_without_an_attribute_error(action):
    reconstructed, proof = _proof(
        [_holding("ORDINARY", asset_class=AssetClass.EQUITIES)],
        [action], contribution=100.0, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert reconstructed.tolist() == [1000.0]
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["serialized_actions_satisfied"]


@pytest.mark.parametrize("include_identity", [False, True])
@pytest.mark.parametrize("include_units", [False, True])
def test_correct_or_absent_legacy_security_metadata_remains_compatible(
    include_identity, include_units,
):
    holding = _holding("ORDINARY", asset_class=AssetClass.EQUITIES)
    action = {"idx": 0, "direction": "buy", "amount_eur": 100.0}
    if include_identity:
        action.update({"isin": holding.isin, "ticker": holding.ticker})
    if include_units:
        action.update({
            "price_eur_per_unit": 10.0,
            "current_quantity": 100.0,
            "quantity_delta": 10.0,
            "after_quantity": 110.0,
        })
    reconstructed, proof = _proof(
        [holding], [action], contribution=100.0, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert reconstructed.tolist() == [1100.0]
    assert proof["status"] == "EXECUTABLE"
    assert proof["units_invariants_satisfied"]
    assert proof["action_identity_satisfied"]


@pytest.mark.parametrize("kind", ["ETF", "BOND", "STOCK"])
@pytest.mark.parametrize("asset_class", [AssetClass.EQUITIES, AssetClass.CASH_EQUIVALENTS])
@pytest.mark.parametrize("amount", [1.0, 100.0])
def test_finite_inputs_cannot_authorize_overflowing_derived_trade_units(
    kind, asset_class, amount,
):
    price = 1e-308
    quantity = 1e308
    holding = _holding(
        "EXTREME", kind=kind, price=price, quantity=quantity,
        value=price * quantity, asset_class=asset_class,
    )
    _, proof = _proof(
        [holding], [{"idx": 0, "direction": "buy", "amount_eur": amount}],
        contribution=amount, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["units_invariants_satisfied"]
    assert proof["security_unit_changes"] == []
    assert rb._unit_trade_facts(holding, amount, holding.current_value) is None


def test_extreme_but_finite_unit_derivation_remains_compatible():
    price, quantity = 1e-305, 1e306
    holding = _holding(
        "FINITE", price=price, quantity=quantity, value=price * quantity,
        asset_class=AssetClass.EQUITIES,
    )
    _, proof = _proof(
        [holding], [{"idx": 0, "direction": "buy", "amount_eur": 100.0}],
        contribution=100.0, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert proof["status"] == "EXECUTABLE"
    change = proof["security_unit_changes"][0]
    assert np.isfinite(change["quantity_delta"])
    assert np.isfinite(change["after_quantity"])
    assert change["after_quantity"] == pytest.approx(1.1e307)


def test_frozen_true_cash_cannot_finance_a_security_buy():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("RESERVE", value=1000.0, frozen=True),
    ]
    before = copy.deepcopy(holdings)
    reconstructed, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 20.0}],
        config=_config(fee=0.0),
    )
    assert reconstructed.tolist() == [1020.0, 100.0, 1000.0]
    assert proof["ending_settlement_cash_eur"] == 80.0
    assert proof["ending_free_settlement_cash_eur"] == -20.0
    assert proof["unallocated_settlement_cash_eur"] == -20.0
    assert proof["protected_cash_eur"] == 100.0
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["settlement_cash_satisfied"]
    assert not proof["protected_cash_satisfied"]
    assert proof["frozen_positions_satisfied"]
    assert proof["wealth_invariants_satisfied"]
    assert proof["authorized_existing_cash_release_eur"] == 0.0
    assert proof["portfolio_wealth_after_eur"] == 2100.0
    assert holdings == before


@pytest.mark.parametrize("locked", [0.0, 100.0])
@pytest.mark.parametrize("has_unlocked", [False, True])
def test_contribution_never_credits_a_frozen_cash_account(locked, has_unlocked):
    holdings = [
        _holding("LOCKED", kind="CASH", value=locked, price=1.0, frozen=True),
        _holding("RESERVE", value=1000.0, frozen=True),
    ]
    if has_unlocked:
        holdings.append(_holding("FREE", kind="CASH", value=0.0, price=1.0))
    reconstructed, proof = _proof(holdings, [], contribution=123.45)
    assert reconstructed[0] == locked
    assert reconstructed[1] == 1000.0
    assert proof["ending_free_settlement_cash_eur"] == pytest.approx(123.45)
    assert proof["locked_settlement_cash_eur"] == locked
    assert proof["locked_cash_balances_satisfied"]
    assert proof["status"] == "EXECUTABLE"
    assert proof["portfolio_wealth_after_eur"] == 1123.45 + locked
    if has_unlocked:
        assert reconstructed[2] == pytest.approx(123.45)
        assert proof["unallocated_settlement_cash_eur"] == 0.0
    else:
        assert proof["unallocated_settlement_cash_eur"] == pytest.approx(123.45)


def test_only_unlocked_accounts_share_a_spend_and_locked_accounts_remain_individual():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCK1", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("FREE1", kind="CASH", value=100.0, price=1.0),
        _holding("LOCK2", kind="CASH", value=200.0, price=1.0, frozen=True),
        _holding("FREE2", kind="CASH", value=200.0, price=1.0),
        _holding("RESERVE", value=1000.0, frozen=True),
    ]
    reconstructed, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 50.0}],
        config=_config(fee=0.0),
    )
    assert reconstructed[1] == 100.0
    assert reconstructed[3] == 200.0
    assert reconstructed[2] == pytest.approx(250.0 / 3.0)
    assert reconstructed[4] == pytest.approx(500.0 / 3.0)
    assert reconstructed[5] == 1000.0
    assert proof["ending_settlement_cash_eur"] == 550.0
    assert proof["ending_free_settlement_cash_eur"] == 250.0
    assert proof["protected_cash_eur"] == 300.0
    assert proof["authorized_existing_cash_release_eur"] == 50.0
    assert proof["status"] == "EXECUTABLE"
    assert proof["wealth_residual_eur"] == 0.0
    assert [h.quantity for h in holdings[1:5]] == [100.0, 100.0, 200.0, 200.0]


@pytest.mark.parametrize("locked", [0.0, 100.0, 1000.0])
@pytest.mark.parametrize("free", [0.0, 50.0, 250.0])
@pytest.mark.parametrize("security", [0.0, 800.0, 1000.0])
@pytest.mark.parametrize("contribution", [0.0, 125.0])
@pytest.mark.parametrize("buffer", [0.0, 900.0])
def test_locked_cash_funding_matches_an_independent_bank_and_reserve_ledger(
    locked, free, security, contribution, buffer,
):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=locked, price=1.0, frozen=True),
        _holding("FREE", kind="CASH", value=free, price=1.0),
        _holding("RESERVE", value=security, frozen=True),
    ]
    required_cash = max(buffer - security, locked, 0.0)
    capacity = locked + free + contribution - required_cash
    purchases = max(capacity, 0.0)
    actions = [{"idx": 0, "direction": "buy", "amount_eur": purchases}] if purchases else []
    reconstructed, proof = _proof(
        holdings, actions, contribution=contribution, buffer=buffer,
        config=_config(buffer=buffer, tax=0.0, fee=0.0),
    )
    expected_free = free + contribution - purchases
    assert reconstructed[1] == locked
    assert reconstructed[2] == pytest.approx(expected_free)
    assert reconstructed[3] == security
    assert proof["protected_cash_eur"] == required_cash
    assert proof["ending_settlement_cash_eur"] == locked + expected_free
    assert proof["ending_free_settlement_cash_eur"] == pytest.approx(expected_free)
    assert proof["unallocated_settlement_cash_eur"] == 0.0
    assert proof["portfolio_wealth_after_eur"] == 1000.0 + locked + free + security + contribution
    assert proof["frozen_positions_satisfied"]
    assert proof["no_sell_satisfied"]
    assert proof["wealth_invariants_satisfied"]
    expected_status = "EXECUTABLE" if capacity >= 0.0 else "NON_EXECUTABLE"
    assert proof["status"] == expected_status


@pytest.mark.parametrize("no_sell", [False, True])
def test_security_sales_and_their_costs_are_separate_from_frozen_bank_funding(no_sell):
    holdings = [
        _holding("SELL", asset_class=AssetClass.EQUITIES, gain=100.0),
        _holding("BUY", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("RESERVE", value=1000.0, frozen=True),
    ]
    reconstructed, proof = _proof(
        holdings, [
            {"idx": 0, "direction": "sell", "amount_eur": 100.0},
            {"idx": 1, "direction": "buy", "amount_eur": 85.0},
        ], config=_config(no_sell=no_sell),
    )
    assert reconstructed.tolist() == [900.0, 1085.0, 100.0, 1000.0]
    assert proof["estimated_tax_eur"] == 13.0
    assert proof["fees_eur"] == 2.0
    assert proof["ending_free_settlement_cash_eur"] == 0.0
    assert proof["unallocated_settlement_cash_eur"] == 0.0
    assert proof["portfolio_wealth_after_eur"] == 3085.0
    assert proof["protected_cash_satisfied"]
    assert proof["no_sell_satisfied"] is (not no_sell)
    assert proof["status"] == ("NON_EXECUTABLE" if no_sell else "EXECUTABLE")


def test_cash_security_sale_proceeds_do_not_credit_a_frozen_cash_account():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, target=100.0),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("RESERVE", value=1000.0, target=0.0),
    ]
    reconstructed, proof = _proof(
        holdings, [{"idx": 2, "direction": "sell", "amount_eur": 1000.0}],
        config=_config(per_holding=True, tax=0.0, fee=1.0),
    )
    assert reconstructed.tolist() == [1000.0, 100.0, 0.0]
    assert proof["ending_free_settlement_cash_eur"] == 999.0
    assert proof["unallocated_settlement_cash_eur"] == 999.0
    assert proof["ending_cash_equivalents_eur"] == 1099.0
    assert proof["portfolio_wealth_after_eur"] == 2099.0
    assert proof["status"] == "EXECUTABLE"


@pytest.mark.parametrize("free", [0.001, 0.004, 0.006, 0.009])
def test_subcent_free_cash_deficit_is_not_masked_by_a_frozen_balance(free):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("FREE", kind="CASH", value=free, price=1.0),
        _holding("RESERVE", frozen=True),
    ]
    reconstructed, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 0.01}],
        config=_config(fee=0.0),
    )
    assert reconstructed[1] == 100.0
    assert reconstructed[2] == pytest.approx(free - 0.01)
    assert proof["ending_settlement_cash_eur"] > 0.0
    assert not proof["settlement_cash_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


@pytest.mark.parametrize("deficit,executable", [(0.5e-9, True), (2e-9, False)])
def test_free_cash_uses_the_existing_strict_proof_epsilon(deficit, executable):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=0.0, price=1.0, frozen=True),
    ]
    _, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 0.01}],
        contribution=0.01 - deficit, buffer=0.0,
        config=_config(buffer=0.0, tax=0.0, fee=0.0),
    )
    assert proof["ending_free_settlement_cash_eur"] == pytest.approx(-deficit, abs=1e-15)
    assert proof["settlement_cash_satisfied"] is executable
    assert proof["status"] == ("EXECUTABLE" if executable else "NON_EXECUTABLE")


@pytest.mark.parametrize("has_free_account", [False, True])
def test_cent_reconciliation_never_spends_frozen_account_cash(monkeypatch, has_free_account):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
        _holding("RESERVE", frozen=True),
    ]
    if has_free_account:
        holdings.append(_holding("FREE", kind="CASH", value=0.004, price=1.0))
    trades = [0.01] + [0.0] * (len(holdings) - 1)
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, _config(fee=0.0), trades, min_trade=0.01,
    )
    funding = verifications[-1]
    assert actions == []
    assert funding["status"] == "EXECUTABLE"
    assert funding["locked_settlement_cash_eur"] == 100.0
    assert funding["ending_settlement_cash_eur"] == (100.004 if has_free_account else 100.0)
    assert funding["ending_free_settlement_cash_eur"] == pytest.approx(
        0.004 if has_free_account else 0.0,
    )
    assert funding["fees_eur"] == 0.0
    assert holdings[1].current_value == holdings[1].quantity == 100.0
    assert holdings[2].current_value == 1000.0


def test_free_plan_ledger_is_counted_in_verification_but_never_in_a_frozen_account(monkeypatch):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True),
    ]
    actions, verifications = _fixed_plan(
        monkeypatch, holdings, _config(buffer=100.0, fee=0.0),
        [0.0, 0.0], contribution=123.45,
    )
    assert actions == []
    cash = next(row for row in verifications if row["kind"] == "cash")
    assert cash["items"][0]["actual_eur"] == pytest.approx(223.45)
    assert verifications[-1]["unallocated_settlement_cash_eur"] == pytest.approx(123.45)
    assert verifications[-1]["portfolio_wealth_after_eur"] == 1223.45
    assert holdings[1].current_value == 100.0


@pytest.mark.parametrize("locked", [0.0, 100.0, 1000.0])
@pytest.mark.parametrize("security", [0.0, 800.0, 1000.0])
def test_hybrid_cash_capacity_reserves_locked_accounts_as_well_as_strategy(locked, security):
    values = np.array([1000.0, locked, 50.0, security])
    security_mask = np.array([0.0, 0.0, 0.0, 1.0])
    funding = rb._CashFunding(
        values, security_mask, locked + 50.0, 125.0, 900.0,
        locked_cash=locked,
    )
    for sale in (0.0, min(security, 100.0)):
        trades = np.array([0.0, 0.0, 0.0, -sale])
        required = max(900.0 - (security - sale), locked, 0.0)
        assert funding.protected_settlement_cash(trades) == required
        assert funding.budget(trades) == locked + 175.0 - required


@pytest.mark.parametrize("locked,free,contribution,buffer,expected", [
    (100.0, 0.0, 0.0, 0.0, 0.0),
    (100.0, 0.0, 25.0, 0.0, 25.0),
    (100.0, 50.0, 25.0, 0.0, 75.0),
    (100.0, 50.0, 25.0, 120.0, 55.0),
    (100.0, 50.0, 25.0, 200.0, 0.0),
])
def test_account_only_search_cannot_allocate_the_locked_bank_balance(
    monkeypatch, locked, free, contribution, buffer, expected,
):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=locked, price=1.0, frozen=True),
        _holding("FREE", kind="CASH", value=free, price=1.0),
    ]
    seen = []

    def descent(t, cost, *_args):
        seen.append(cost)
        return np.zeros_like(t), 0.0

    monkeypatch.setattr(rb, "_descent", descent)
    monkeypatch.setattr(rb, "_project_to_budget", lambda t, *_args: t.copy())
    rb.optimize_local_search(
        holdings, _config(buffer=buffer, fee=0.0),
        sum(h.current_value for h in holdings), lump_sum=contribution,
        params=rb.LSParams(n_restarts=0),
    )
    assert len(seen) == 1
    assert seen[0].cash_funding is None
    assert seen[0].lump_sum == expected


def test_default_zero_locked_funding_is_bit_identical_to_existing_hybrid_math():
    values = np.array([1000.0, 200.0, 800.0])
    mask = np.array([0.0, 0.0, 1.0])
    funding = rb._CashFunding(values, mask, 200.0, 100.0, 900.0)
    for trades in (
        np.zeros(3), np.array([100.0, 0.0, -100.0]),
        np.array([-100.0, 0.0, 100.0]),
    ):
        original_floor = max(900.0 - float(mask @ (values + trades)), 0.0)
        original_budget = 200.0 + 100.0 - original_floor
        assert funding.protected_settlement_cash(trades).hex() == original_floor.hex()
        assert funding.budget(trades).hex() == original_budget.hex()


@pytest.mark.parametrize("locked", [100.0, 1000.0])
@pytest.mark.parametrize("free", [0.0, 200.0])
@pytest.mark.parametrize("security", [500.0, 1000.0])
@pytest.mark.parametrize("contribution", [0.0, 1000.0])
def test_locked_hybrid_scalar_cost_matches_independent_funding_and_bank_penalties(
    locked, free, security, contribution,
):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES, gain=100.0),
        _holding("LOCKED", kind="CASH", value=locked, price=1.0, frozen=True),
        _holding("FREE", kind="CASH", value=free, price=1.0),
        _holding("SEC", value=security, gain=100.0),
    ]
    values = np.array([h.current_value for h in holdings])
    config = _config()
    model = rb._ObjectiveModel(holdings, config, values)
    funding = rb._CashFunding(
        values, model.cash_security_mask, locked + free, contribution, 900.0,
        locked_cash=locked,
    )
    params = rb.LSParams()
    cost = rb._Cost(model, values, config, 0.0, params, cash_funding=funding)
    for trades in (
        np.array([100.0, 0.0, 0.0, -100.0]),
        np.array([0.0, 0.0, 0.0, 100.0]),
        np.array([300.0, 0.0, 0.0, 0.0]),
        np.array([-100.0, 0.0, 0.0, 100.0]),
        np.zeros(4),
    ):
        buys = sum(float(t) for t in trades if t > 0)
        sells = sum(-float(t) for t in trades if t < 0)
        tax = sum(-float(t) * 0.13 for t in trades if t < 0)
        fees = sum(1.0 for t in trades if abs(t) >= 1.0)
        net_cost = buys + tax + fees - sells
        free_after = free + contribution - net_cost
        security_after = security + float(trades[3])
        bank_floor = max(900.0 - security_after, locked, 0.0)
        budget = locked + free + contribution - bank_floor
        gaps = model.gaps(values + trades)
        gaps[-1] = (locked + free_after + security_after - 900.0) / 900.0 * 100.0
        gaps = np.abs(gaps)
        expected = cost.gap_term(gaps)
        excess = np.clip(gaps - cost.allowed, 0.0, None)
        expected += params.guard_weight * float(np.sum(excess ** 2))
        expected += params.turnover_weight * float(np.abs(trades).sum()) / cost.tv
        expected += params.trade_count_weight * int((np.abs(trades) >= 1.0).sum())
        expected += params.residual_weight * ((net_cost - budget) / cost.tv * 100.0) ** 2
        expected += params.residual_weight * (max(-free_after, 0.0) / cost.tv * 100.0) ** 2
        assert cost(trades) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("locked", [100.0, 1e16])
@pytest.mark.parametrize("has_free_account", [False, True])
def test_large_locked_balances_do_not_cancel_raw_free_cash_deficits(locked, has_free_account):
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("LOCKED", kind="CASH", value=locked, price=1.0, frozen=True),
        _holding("RESERVE", frozen=True),
    ]
    free = 0.004 if has_free_account else 0.0
    if has_free_account:
        holdings.append(_holding("FREE", kind="CASH", value=free, price=1.0))
    reconstructed, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 0.01}],
        config=_config(fee=0.0),
    )
    assert reconstructed[1] == locked
    assert proof["ending_free_settlement_cash_eur"] == free - 0.01
    assert proof["status"] == "NON_EXECUTABLE"
    assert not proof["settlement_cash_satisfied"]
    if has_free_account:
        assert reconstructed[-1] == pytest.approx(free - 0.01)
        assert proof["unallocated_settlement_cash_eur"] == 0.0
    else:
        assert proof["unallocated_settlement_cash_eur"] == -0.01


@pytest.mark.parametrize("direction", ["buy", "sell"])
def test_frozen_account_action_is_rejected_without_creating_sale_proceeds(direction):
    account = _holding("LOCKED", kind="CASH", value=100.0, price=1.0, frozen=True)
    reconstructed, proof = _proof(
        [account], [{"idx": 0, "direction": direction, "amount_eur": 10.0}],
        buffer=0.0, config=_config(buffer=0.0, fee=0.0),
    )
    assert reconstructed.tolist() == [100.0]
    assert proof["gross_sales_eur"] == proof["gross_purchases_eur"] == 0.0
    assert proof["ending_free_settlement_cash_eur"] == 0.0
    assert proof["locked_cash_balances_satisfied"]
    assert not proof["frozen_positions_satisfied"]
    assert proof["status"] == "NON_EXECUTABLE"


def test_unlocked_account_funding_retains_the_existing_serialized_proof_contract():
    holdings = [
        _holding("EQ", asset_class=AssetClass.EQUITIES),
        _holding("FREE", kind="CASH", value=100.0, price=1.0),
        _holding("RESERVE", frozen=True),
    ]
    _, proof = _proof(
        holdings, [{"idx": 0, "direction": "buy", "amount_eur": 20.0}],
        config=_config(fee=0.0),
    )
    assert proof["status"] == "EXECUTABLE"
    assert proof["ending_settlement_cash_eur"] == 80.0
    assert proof["protected_cash_eur"] == 0.0
    assert proof["authorized_existing_cash_release_eur"] == 20.0
    assert not {
        "locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
        "ending_free_settlement_cash_eur", "locked_cash_balances_satisfied",
    }.intersection(proof)
