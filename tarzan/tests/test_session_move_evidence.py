"""The euro caption shares the return's native window and capital authority.

Synthetic quotes only. Builder tests capture the original chaining loop; the
oracles use actual endpoint amounts/flows rather than the display's formula.
"""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.session_move import DailyReturnStep, paired_session_move
from tarzan.engine import returns_builder as rb
from tarzan.engine.metrics import MetricsEngine
from tarzan.engine.stats import compute_period_return, to_business_day_series
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType
from tarzan.models.portfolio import PortfolioMetrics


FRIDAY = date(2026, 10, 23)
THURSDAY = FRIDAY - timedelta(days=1)


def _evidence(
    prices, *, steps=(), basis="ORDER_FLOW_ADJUSTED_NAV", availability="AVAILABLE",
    daily_nav=None,
):
    return paired_session_move(
        prices, compute_period_return(prices, "1d"),
        source_basis=basis, steps=steps, history_availability=availability,
        daily_nav=daily_nav,
    )


def _loop(*, values, quantities=None, flows=None, start=THURSDAY):
    """Exercise the real dense loop with a synthetic, typed EUR/unit resolver."""
    dates = [start + timedelta(days=i) for i in range(len(values))]
    amounts = dict(zip(dates, values))
    quantities = dict(zip(dates, quantities or [1.0] * len(values)))

    class Timeline:
        def isins(self):
            return ["SYNTHETIC"]

        def qty_at(self, isin, day):
            return quantities[day]

    class Resolver:
        def price_on(self, isin, day):
            value = amounts[day]
            quantity = quantities[day]
            return (None if value is None else value / (quantity or 1.0)), "yfinance"

        def instrument_kind(self, isin):
            return InstrumentKind.STOCK

    steps = []
    arguments = (
        Timeline(), Resolver(), flows or {}, dates, dates[-1],
    )
    keywords = {"closed_identity_groups": lambda _: set()}
    plain = rb._build_daily_series(*arguments, **keywords)
    captured = rb._build_daily_series(*arguments, **keywords, step_evidence=steps)
    for actual, expected in zip(captured, plain):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    return captured[0], captured[1], tuple(steps)


@pytest.mark.parametrize("deposit", [0.0, 500.0, 5000.0, -200.0])
def test_original_loop_profit_is_not_decompounded_current_wealth(deposit):
    nav, actual, steps = _loop(values=[1000.0, 1100.0 + deposit], flows={FRIDAY: deposit})
    result = _evidence(nav, steps=steps)
    assert result.availability == "AVAILABLE"
    assert result.return_pct == pytest.approx(10.0)
    assert result.profit_eur == pytest.approx(100.0)
    assert result.external_flow_eur == deposit
    assert result.anchor_date == THURSDAY and result.end_date == FRIDAY
    assert actual.iloc[-1] == 1100.0 + deposit
    if deposit:
        wrong = actual.iloc[-1] * 0.1 / 1.1
        assert wrong != pytest.approx(result.profit_eur)


@pytest.mark.parametrize("income", [10.0, 30.0, 100.0])
def test_income_is_counted_once_with_the_original_external_flow_sign(income):
    nav, _, steps = _loop(
        values=[1000.0, 980.0], flows={FRIDAY: -income},
    )
    result = _evidence(nav, steps=steps)
    assert result.profit_eur == pytest.approx(-20.0 + income)
    assert result.external_flow_eur == -income
    assert result.return_pct == pytest.approx((-20.0 + income) / 1000.0 * 100.0)


@pytest.mark.parametrize("values", [[1000.0, 1000.0], [1000.0, 900.0], [1000.0, 1100.0]])
def test_numeric_zero_and_negative_profit_are_available_real_measurements(values):
    nav, _, steps = _loop(values=values)
    result = _evidence(nav, steps=steps)
    assert result.availability == "AVAILABLE"
    assert result.profit_eur == values[-1] - values[0]
    assert result.reason == ""


@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("availability", ["AVAILABLE", "DEGRADED"])
def test_holdings_delta_uses_its_own_basket_including_cash_equivalent_securities(
    timezone, availability,
):
    index = pd.DatetimeIndex([THURSDAY, FRIDAY], tz=timezone)
    prices = pd.Series([1200.0, 1302.0], index=index)
    original = prices.copy(deep=True)
    result = _evidence(
        prices, basis="HOLDINGS_PRICE_AGGREGATE", availability=availability,
    )
    assert result.profit_eur == 102.0
    assert result.return_pct == pytest.approx(8.5)
    assert result.external_flow_eur == 0.0
    assert result.availability == availability
    pd.testing.assert_series_equal(prices, original, check_exact=True)
    with pytest.raises(FrozenInstanceError):
        result.profit_eur = 500.0


@pytest.mark.parametrize("collapse", [False, True])
def test_weekend_flows_follow_the_original_native_endpoint_of_each_session_bin(collapse):
    monday = date(2026, 10, 26)
    nav, actual, steps = _loop(
        start=FRIDAY,
        values=[1000.0, 1500.0, 1520.0, 1590.0],
        flows={FRIDAY + timedelta(days=1): 500.0, monday: 50.0},
    )
    session_nav = to_business_day_series(nav) if collapse else nav
    result = _evidence(session_nav, steps=steps, daily_nav=nav if collapse else None)
    assert result.availability == "AVAILABLE"
    assert result.anchor_date == FRIDAY and result.end_date == monday
    # Original B.last assigns Sunday's observation to Friday's label. Do not
    # silently change that analytical policy or include income twice in EUR.
    assert result.source_anchor_date == (date(2026, 10, 25) if collapse else FRIDAY)
    assert result.source_end_date == monday
    assert result.profit_eur == (20.0 if collapse else 40.0)
    assert result.external_flow_eur == (50.0 if collapse else 550.0)
    assert len(result.steps) == (1 if collapse else 3)
    factor = (1540.0 / 1520.0)
    if not collapse:
        factor *= (1520.0 / 1500.0)
    assert result.return_pct == pytest.approx((factor - 1.0) * 100.0)
    assert actual.iloc[-1] == 1590.0


def test_a_priced_disposal_day_proves_its_gain_even_when_the_book_ends_flat():
    nav, actual, steps = _loop(
        values=[1000.0, 0.0], quantities=[1.0, 0.0], flows={FRIDAY: -1100.0},
    )
    result = _evidence(nav, steps=steps)
    assert actual.iloc[-1] == 0.0
    assert result.profit_eur == pytest.approx(100.0)
    assert result.return_pct == pytest.approx(10.0)


def test_unpriced_held_day_never_proves_zero_euro_return():
    nav, actual, steps = _loop(values=[1000.0, None])
    assert nav.iloc[-1] == 1000.0 and actual.iloc[-1] == 0.0
    result = _evidence(nav, steps=steps)
    assert result.availability == "UNAVAILABLE"
    assert result.profit_eur is None
    assert result.reason == "UNMEASURED_NAV_STEP"
    assert steps[-1].state == "UNPRICED"


@pytest.mark.parametrize("values,quantities,flows", [
    ([1000.0, 0.0], [1.0, 0.0], {}),
    ([1000.0, 0.0, 500.0], [1.0, 0.0, 1.0], {FRIDAY: -1000.0, FRIDAY + timedelta(days=1): 500.0}),
])
def test_suspended_or_reentry_chain_does_not_invent_a_capital_denominator(
    values, quantities, flows,
):
    nav, _, steps = _loop(values=values, quantities=quantities, flows=flows)
    assert any(step.state == "SUSPENDED" for step in steps)
    result = _evidence(nav, steps=steps)
    assert result.availability == "UNAVAILABLE"
    assert result.profit_eur is None


def test_fixed_basket_total_loss_has_a_valid_euro_delta_without_dividing_by_one_plus_return():
    prices = pd.Series([1000.0, 0.0], index=pd.DatetimeIndex([THURSDAY, FRIDAY]))
    result = _evidence(prices, basis="HOLDINGS_PRICE_AGGREGATE")
    assert result.availability == "AVAILABLE"
    assert result.return_pct == -100.0
    assert result.profit_eur == -1000.0


@pytest.mark.parametrize("availability", [None, "", True, "UNKNOWN", "UNAVAILABLE"])
def test_unknown_or_unavailable_history_cannot_publish_numeric_euro_evidence(availability):
    nav, _, steps = _loop(values=[1000.0, 1100.0])
    result = _evidence(nav, steps=steps, availability=availability)
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("basis", [None, "UNPREPARED", "ORDER_SPARSE_VALUATION", "ACCOUNTING_PNL", "CASH_WEIGHT"])
def test_unknown_or_sparse_source_cannot_be_converted_as_if_it_were_daily_nav(basis):
    nav, _, steps = _loop(values=[1000.0, 1100.0])
    result = _evidence(nav, steps=steps, basis=basis)
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("pct", [None, True, np.bool_(True), "10", float("nan"), float("inf"), -float("inf"), 0.0, 11.0])
def test_missing_nonfinite_or_unpaired_percentage_is_unavailable(pct):
    nav, _, steps = _loop(values=[1000.0, 1100.0])
    result = paired_session_move(
        nav, pct, source_basis="ORDER_FLOW_ADJUSTED_NAV", steps=steps,
    )
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("index", [
    pd.DatetimeIndex([FRIDAY, THURSDAY]),
    pd.DatetimeIndex([THURSDAY, THURSDAY]),
    pd.Index(["2026-10-22", "2026-10-23"]),
    pd.Index([0, 1]),
])
def test_unknown_unsorted_or_duplicate_window_index_fails_closed(index):
    prices = pd.Series([1000.0, 1100.0], index=index)
    result = paired_session_move(prices, 10.0, source_basis="HOLDINGS_PRICE_AGGREGATE")
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("values", [[0.0, 1.0], [-1.0, 1.0], [1000.0, -1.0], [1000.0, np.inf], [np.nan, 1000.0], []])
def test_invalid_or_insufficient_basket_endpoints_are_not_zero(values):
    prices = pd.Series(values, index=pd.date_range(THURSDAY, periods=len(values)), dtype=float)
    result = paired_session_move(prices, 10.0, source_basis="HOLDINGS_PRICE_AGGREGATE")
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("field,value", [
    ("start_date", FRIDAY),
    ("end_date", THURSDAY),
    ("state", "SUSPENDED"),
    ("state", "UNPRICED"),
    ("factor", None),
    ("factor", 0.0),
    ("factor", np.inf),
    ("factor", 1.2),
    ("profit_eur", None),
    ("profit_eur", np.nan),
    ("profit_eur", 999.0),
    ("profit_eur", True),
    ("starting_value_eur", 900.0),
    ("starting_value_eur", 0.0),
    ("starting_value_eur", np.inf),
    ("ending_value_eur", 1200.0),
    ("ending_value_eur", -100.0),
    ("ending_value_eur", np.nan),
    ("external_flow_eur", 500.0),
    ("external_flow_eur", np.inf),
    ("external_flow_eur", True),
])
def test_corrupt_native_step_facts_do_not_publish_a_coherent_looking_caption(field, value):
    nav, _, steps = _loop(values=[1000.0, 1100.0])
    changed = (replace(steps[0], **{field: value}),)
    result = _evidence(nav, steps=changed)
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


@pytest.mark.parametrize("steps", [(), (None,), (object(),), ({"factor": 1.1},)])
def test_missing_or_untyped_steps_are_unavailable(steps):
    nav, _, _ = _loop(values=[1000.0, 1100.0])
    result = _evidence(nav, steps=steps)
    assert result.availability == "UNAVAILABLE" and result.profit_eur is None


def test_step_outside_the_selected_window_is_not_adopted():
    nav, _, steps = _loop(values=[1000.0, 1100.0])
    stale = replace(steps[0], start_date=THURSDAY - timedelta(days=1), end_date=THURSDAY)
    future = replace(steps[0], start_date=FRIDAY, end_date=FRIDAY + timedelta(days=1))
    result = _evidence(nav, steps=(stale, *steps, future))
    assert result.steps == steps and result.profit_eur == 100.0


def test_disconnected_capital_bases_are_rejected_even_if_chained_percent_matches():
    friday = date(2026, 10, 23)
    monday = date(2026, 10, 26)
    index = pd.DatetimeIndex([friday, monday])
    nav = pd.Series([1000.0, 1210.0], index=index)
    steps = (
        DailyReturnStep(friday + timedelta(days=1), friday, 1000.0, 1100.0, 0.0, 1.1, 100.0, "MEASURED"),
        DailyReturnStep(monday, friday + timedelta(days=1), 2000.0, 2200.0, 0.0, 1.1, 200.0, "MEASURED"),
    )
    result = _evidence(nav, steps=steps)
    assert result.availability == "UNAVAILABLE"
    assert result.reason == "NAV_FACTS_MISMATCH"


def _stock_order(kind, day, qty, net, price):
    return Order(
        date=day, trade_date=day, type=kind, isin="XX0000000001",
        ticker="SYNTHETIC", name="Synthetic share", quantity=qty,
        currency="EUR", price_native=price, fx_rate=1.0,
        gross_eur=abs(net), fees_eur=0.0, net_eur=net,
        source="synthetic-fixture", instrument_kind=InstrumentKind.STOCK,
    )


def test_real_order_builder_pairs_trade_contribution_and_terminal_market_move():
    start = date(2026, 10, 20)
    history = pd.Series(
        [100.0, 100.0, 100.0, 110.0], index=pd.date_range(start, FRIDAY),
        name="SYNTHETIC",
    )
    holding = Holding(
        isin="XX0000000001", ticker="SYNTHETIC", quantity=15.0,
        cost_basis_eur=1500.0, market_value_eur=1650.0, current_value=1650.0,
        current_price=110.0, currency="EUR", security_type="STOCK",
        asset_class=AssetClass.EQUITIES, price_history=history,
        price_history_price_only=history.copy(),
    )
    orders = [
        _stock_order(OrderType.BUY, start, 10.0, -1000.0, 100.0),
        _stock_order(OrderType.BUY, FRIDAY, 5.0, -550.0, 110.0),
    ]
    series = rb.build_order_derived_series(
        orders, {holding.isin: holding}, today=FRIDAY,
    )
    assert isinstance(series.daily_return_steps, tuple)
    nav = to_business_day_series(series.daily_series)
    result = _evidence(
        nav, steps=series.daily_return_steps,
        availability=series.history_availability.value,
    )
    # The incoming five units are market-valued external capital, not gain.
    assert result.external_flow_eur == 550.0
    assert result.profit_eur == 100.0
    assert result.return_pct == pytest.approx(10.0)
    assert series.actual_value_series.iloc[-1] == 1650.0


def test_metrics_performance_adopts_only_declared_builder_evidence():
    nav, _, steps = _loop(values=[1000.0, 1600.0], flows={FRIDAY: 500.0})
    engine = MetricsEngine.__new__(MetricsEngine)
    engine._rf_daily = lambda _: None
    engine._alpha_beta_benchmark = lambda _: None

    class OrderSeries:
        from tarzan.runtime.ledger import Availability
        history_availability = Availability.AVAILABLE
        daily_return_steps = steps

    context = {
        "portfolio_history": nav, "portfolio_history_full": pd.Series(dtype=float),
        "_portfolio_history_basis": "ORDER_FLOW_ADJUSTED_NAV",
        "_order_series": OrderSeries(),
    }
    engine._performance(context)
    result = context["session_move_evidence"]
    assert result.return_pct == context["performance"]["1d"]
    assert result.profit_eur == 100.0
    untyped = {"portfolio_history": nav, "portfolio_history_full": pd.Series(dtype=float)}
    engine._performance(untyped)
    assert untyped["performance"] == context["performance"]
    assert untyped["session_move_evidence"].availability == "UNAVAILABLE"


def test_model_field_is_additive_internal_not_a_silent_external_schema_migration():
    assert PortfolioMetrics.__dataclass_fields__["session_move_evidence"].default is None
    assert "session_move_evidence" not in PortfolioMetrics().to_summary_dict()
