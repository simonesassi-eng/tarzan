"""Rollover pruning must preserve every original authority and ambiguity."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from itertools import combinations
import math

from hypothesis import given, settings, strategies as st
import pytest

from tarzan.data import enricher
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.order import Order, OrderType


def order(isin, kind, quantity, **changes):
    day = date(2026, 1, 5)
    fields = dict(
        date=day, trade_date=day, type=kind, isin=isin,
        name="Synthetic issuer ISSUE2030", ticker="", quantity=quantity,
        currency="EUR", price_native=100.0, fx_rate=1.0,
        gross_eur=quantity * 100.0, fees_eur=0.0, net_eur=0.0,
        source="synthetic",
    )
    fields.update(changes)
    return Order(**fields)


def pair(**changes):
    incoming = order("XS0000000001", OrderType.TRANSFER_IN, 1000.0)
    outgoing = order("XS0000000002", OrderType.SELL, -1000.0)
    for field, value in changes.items():
        setattr(outgoing, field, value)
    return [incoming, outgoing]


def exhaustive(orders, kinds, excluded):
    """Independent exhaustive reference, before per-instrument pruning.

    The summary boundary is unchanged; this oracle retains all original pair
    checks and constructs the complete ambiguity graph before selecting pairs.
    """
    summaries = enricher._order_instrument_summaries(orders)
    candidates = []
    for left_isin, right_isin in combinations(sorted(set(summaries) - excluded), 2):
        left, right = summaries[left_isin], summaries[right_isin]
        left_values, right_values = left["position_quantities"], right["position_quantities"]
        if not left_values or not right_values:
            continue
        left_direction = (
            1 if all(value > 0 for value in left_values)
            else -1 if all(value < 0 for value in left_values) else 0
        )
        right_direction = (
            1 if all(value > 0 for value in right_values)
            else -1 if all(value < 0 for value in right_values) else 0
        )
        if {left_direction, right_direction} != {-1, 1}:
            continue
        positive, negative = (left, right) if left_direction > 0 else (right, left)
        if positive["position_types"] != {OrderType.TRANSFER_IN}:
            continue
        if not negative["position_types"] or not negative["position_types"].issubset(
            {OrderType.SELL, OrderType.TRANSFER_OUT}
        ):
            continue
        if not math.isclose(
            left["net_quantity"], -right["net_quantity"], rel_tol=1e-10,
            abs_tol=enricher._EQUIVALENCE_QUANTITY_EPSILON,
        ):
            continue
        if (
            len(left["currencies"]) != 1 or left["currencies"] != right["currencies"]
            or len(right["currencies"]) != 1 or len(left["sources"]) != 1
            or left["sources"] != right["sources"] or len(right["sources"]) != 1
        ):
            continue
        if not positive["dates"] or not negative["dates"]:
            continue
        elapsed = (min(negative["dates"]) - max(positive["dates"])).days
        if not 0 <= elapsed <= enricher._TRANSACTIONAL_EQUIVALENCE_MAX_DAYS:
            continue
        left_issue, left_issuer = enricher._name_identity_tokens(left["names"])
        right_issue, right_issuer = enricher._name_identity_tokens(right["names"])
        if not left_issue & right_issue or not left_issuer & right_issuer:
            continue
        left_price, right_price = enricher._median(left["prices"]), enricher._median(right["prices"])
        if left_price is None or right_price is None:
            continue
        if min(left_price, right_price) / max(left_price, right_price) < 0.8:
            continue
        if {kinds.get(left_isin), kinds.get(right_isin)} != {None, InstrumentKind.BOND}:
            continue
        candidates.append((left_isin, right_isin))
    neighbors = {}
    for left, right in candidates:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    return [
        candidate for candidate in candidates
        if all(len(neighbors[isin]) == 1 for isin in candidate)
    ]


@pytest.mark.parametrize("known_on_incoming", [False, True])
@pytest.mark.parametrize("disposal", [OrderType.SELL, OrderType.TRANSFER_OUT])
def test_exact_unique_rollover_is_retained_in_either_authority_direction(
    known_on_incoming, disposal,
):
    orders = pair(type=disposal)
    known = orders[0 if known_on_incoming else 1].isin
    assert enricher._transactional_equivalence_pairs(
        orders, {known: InstrumentKind.BOND}, set(),
    ) == [(orders[0].isin, orders[1].isin)]


@pytest.mark.parametrize("kind", [
    InstrumentKind.STOCK, InstrumentKind.ETF, InstrumentKind.CASH,
])
@pytest.mark.parametrize("known_on_incoming", [False, True])
def test_non_bond_kind_never_acquires_rollover_authority(kind, known_on_incoming):
    orders = pair()
    known = orders[0 if known_on_incoming else 1].isin
    assert enricher._transactional_equivalence_pairs(orders, {known: kind}, set()) == []


@pytest.mark.parametrize("known_kind", [None, InstrumentKind.BOND])
def test_two_equal_kind_declarations_are_not_a_new_rollover_rule(known_kind):
    orders = pair()
    kinds = {row.isin: known_kind for row in orders}
    assert enricher._transactional_equivalence_pairs(orders, kinds, set()) == []


@pytest.mark.parametrize("change", [
    {"quantity": -999.0},
    {"currency": "USD"},
    {"source": "another-source"},
    {"name": "Another issuer ISSUE2031"},
    {"price_native": 79.0},
    {"price_native": None},
    {"price_native": float("inf")},
    {"type": OrderType.BUY},
    {"type": OrderType.DIVIDEND},
    {"quantity": 1000.0},
    {"trade_date": date(2026, 1, 4)},
    {"trade_date": date(2026, 1, 5) + timedelta(
        days=enricher._TRANSACTIONAL_EQUIVALENCE_MAX_DAYS + 1,
    )},
])
def test_every_original_pair_disqualifier_remains(change):
    orders = pair(**change)
    kinds = {orders[1].isin: InstrumentKind.BOND}
    assert exhaustive(orders, kinds, set()) == []
    assert enricher._transactional_equivalence_pairs(orders, kinds, set()) == []


def test_mixed_position_directions_are_not_flattened_into_a_rollover():
    orders = pair()
    orders.extend([
        order(orders[0].isin, OrderType.SELL, -1.0),
        order(orders[0].isin, OrderType.TRANSFER_IN, 1.0),
    ])
    kinds = {orders[1].isin: InstrumentKind.BOND}
    assert exhaustive(orders, kinds, set()) == []
    assert enricher._transactional_equivalence_pairs(orders, kinds, set()) == []


@pytest.mark.parametrize("excluded", ["XS0000000001", "XS0000000002"])
def test_reserved_or_conflicting_identity_is_excluded(excluded):
    orders = pair()
    assert enricher._transactional_equivalence_pairs(
        orders, {orders[1].isin: InstrumentKind.BOND}, {excluded},
    ) == []


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_complete_ambiguity_graph_cannot_greedily_accept_a_first_match(reverse_rows):
    orders = pair()
    orders.append(replace(orders[1], isin="XS0000000003"))
    kinds = {orders[1].isin: InstrumentKind.BOND, orders[2].isin: InstrumentKind.BOND}
    if reverse_rows:
        orders.reverse()
    assert exhaustive(orders, kinds, set()) == []
    assert enricher._transactional_equivalence_pairs(orders, kinds, set()) == []


def test_real_duplicate_tranches_and_distributions_are_preserved_without_mutation():
    orders = [
        order("XS0000000001", OrderType.TRANSFER_IN, 500.0),
        order("XS0000000001", OrderType.TRANSFER_IN, 500.0),
        order("XS0000000001", OrderType.COUPON, 0.0, gross_eur=11.0),
        order("XS0000000002", OrderType.SELL, -1000.0),
    ]
    original = deepcopy(orders)
    kinds, excluded = {orders[-1].isin: InstrumentKind.BOND}, set()
    assert enricher._transactional_equivalence_pairs(
        orders, kinds, excluded,
    ) == [("XS0000000001", "XS0000000002")]
    assert orders == original and len(orders) == 4
    assert kinds == {orders[-1].isin: InstrumentKind.BOND} and excluded == set()


def test_complete_pairs_keep_original_lexical_order_and_financial_tolerances():
    orders = [
        *pair(price_native=80.0),
        order("XS0000000003", OrderType.TRANSFER_IN, 500.0,
              name="Other issuer ISSUE2040"),
        order("XS0000000004", OrderType.TRANSFER_OUT, -500.0,
              name="Other issuer ISSUE2040"),
    ]
    kinds = {orders[1].isin: InstrumentKind.BOND, orders[3].isin: InstrumentKind.BOND}
    expected = [
        ("XS0000000001", "XS0000000002"), ("XS0000000003", "XS0000000004"),
    ]
    assert exhaustive(orders, kinds, set()) == expected
    assert enricher._transactional_equivalence_pairs(
        list(reversed(orders)), kinds, set(),
    ) == expected


def test_effective_order_copies_keep_basis_and_propagate_only_exact_unique_mechanics(
    monkeypatch,
):
    orders = pair()
    original = deepcopy(orders)
    monkeypatch.setattr(enricher, "_openfigi_raw_many", lambda isins: {})
    monkeypatch.setattr(
        enricher, "_load_or_fetch_instrument_profile",
        lambda isin: (
            ({"status": "VERIFIED", "kind": "BOND"}, "synthetic")
            if isin == orders[1].isin else (None, "unavailable")
        ),
    )
    resolved, evidence = enricher.resolve_effective_order_instruments(orders)
    assert orders == original
    assert all(row.instrument_kind is InstrumentKind.BOND for row in resolved)
    groups = {row.instrument_equivalence_group for row in resolved}
    assert len(groups) == 1 and next(iter(groups)).startswith("AUTO-LEDGER-")
    assert evidence["equivalence_groups"][0]["source"] == "unique_transactional_rollover"
    for before, after in zip(original, resolved):
        assert after is not before
        assert after.gross_eur == before.gross_eur
        assert after.net_eur == before.net_eur
        assert after.price_native == before.price_native
        assert after.quantity == before.quantity
        assert after.acquisition_cost_eur() == before.acquisition_cost_eur()


@settings(max_examples=180, deadline=None)
@given(st.lists(st.tuples(
    st.integers(0, 7), st.sampled_from(list(OrderType)),
    st.sampled_from([-1000.0, -500.0, -1.0, 0.0, 1.0, 500.0, 1000.0]),
    st.sampled_from(["EUR", "USD", "GBp", "GBP", "ZAR", ""]),
    st.sampled_from(["synthetic", "SYNTHETIC", "another", ""]),
    st.sampled_from([None, 79.0, 80.0, 100.0, 125.0]),
    st.integers(-1, enricher._TRANSACTIONAL_EQUIVALENCE_MAX_DAYS + 1),
    st.sampled_from([None, *InstrumentKind]),
    st.booleans(),
), min_size=0, max_size=30))
def test_generated_ledger_matches_exhaustive_original_pair_selection(rows):
    orders, kinds, excluded = [], {}, set()
    for number, kind, quantity, currency, source, price, offset, mechanics, reserve in rows:
        isin = f"XS{number:010d}"
        day = date(2026, 1, 5) + timedelta(days=offset)
        orders.append(order(
            isin, kind, quantity, currency=currency, source=source,
            price_native=price, date=day, trade_date=day,
        ))
        kinds[isin] = mechanics
        if reserve:
            excluded.add(isin)
    before = deepcopy(orders)
    assert enricher._transactional_equivalence_pairs(
        orders, kinds, excluded,
    ) == exhaustive(orders, kinds, excluded)
    assert orders == before
