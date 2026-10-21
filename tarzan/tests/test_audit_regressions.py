"""Counterexamples found in the October 2026 correctness audit."""

from __future__ import annotations

import datetime as dt
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pandas as pd
import pytest

from tarzan.contracts.exceptions import DataIngestionError
from tarzan.delivery.claims import (
    DeliveryIntent,
    DeliveryPurpose,
    DeliveryState,
    LocalJsonDeliveryClaimStore,
    DeliveryTransitionConflict,
)
from tarzan.engine.returns_builder import (
    _seed_market_value,
    build_holdings_from_orders,
    build_order_derived_series,
    cost_basis_by_isin,
)
from tarzan.engine.stats import twr
from tarzan.engine.tax import estimate_realized_cgt
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType
from tarzan.orchestrator import run
from tarzan.runtime.effective_orders import EffectiveOrderSnapshot
from tarzan.runtime.publication import PublicationDecision, PublicationEvaluator
from tarzan.runtime.session import last_run_result
from tarzan.runtime.summary import SummaryProjector


def _order(
    day=1,
    kind=OrderType.BUY,
    quantity=10.0,
    net=-1000.0,
    gross=None,
    price=100.0,
    currency="EUR",
    fx=1.0,
    isin="US0000000001",
):
    return Order(
        date=dt.date(2025, 1, day),
        trade_date=dt.date(2025, 1, day),
        type=kind,
        isin=isin,
        name="Example",
        ticker="EXAMPLE",
        quantity=quantity,
        currency=currency,
        price_native=price,
        fx_rate=fx,
        gross_eur=net if gross is None else gross,
        fees_eur=0.0,
        net_eur=net,
        instrument_kind=InstrumentKind.STOCK,
    )


def _holding(prices, quantity=10.0):
    holding = Holding(
        isin="US0000000001",
        ticker="EXAMPLE",
        quantity=quantity,
        cost_basis_eur=1000.0,
        market_value_eur=quantity * prices[-1],
        currency="EUR",
        security_type="STOCK",
    )
    holding.price_history = pd.Series(
        prices, index=pd.date_range("2025-01-01", periods=len(prices))
    )
    return holding


@pytest.mark.parametrize(
    ("movement", "quantity", "net", "terminal_quantity"),
    [(OrderType.BUY, 10, -1100, 20), (OrderType.SELL, -5, 550, 5)],
)
def test_terminal_trade_is_counted_once_in_twr(
    movement, quantity, net, terminal_quantity,
):
    orders = [
        _order(),
        _order(3, movement, quantity, net, price=110),
    ]
    holding = _holding([100, 105, 110], terminal_quantity)
    series = build_order_derived_series(
        orders, {holding.isin: holding}, today=dt.date(2025, 1, 3)
    )
    result = twr(series.valuations, series.external_flows, series.span_days)
    assert result.cumulative_pct == pytest.approx(10.0)
    assert result.cumulative_pct == pytest.approx(
        (series.daily_series.iloc[-1] / series.daily_series.iloc[0] - 1) * 100
    )
    assert len({day for day, _ in series.valuations}) == len(series.valuations)


def test_effective_snapshot_preserves_same_day_trade_chronology():
    orders = [
        _order(),
        _order(2, quantity=10, net=-2000),
        _order(2, OrderType.SELL, -10, 2000),
    ]
    snapshot = EffectiveOrderSnapshot.build(orders, dt.date(2025, 1, 3))
    assert snapshot.orders == tuple(orders)
    assert cost_basis_by_isin(list(snapshot.orders))["US0000000001"] == 1500
    assert estimate_realized_cgt(
        list(snapshot.orders), {}, 26, 12.5
    ).total_tax_eur == 130


def test_settlement_date_does_not_reorder_same_day_exposure():
    buy = replace(_order(), date=dt.date(2025, 1, 3))
    sell = replace(
        _order(kind=OrderType.SELL, quantity=-5, net=600),
        date=dt.date(2025, 1, 2),
    )
    snapshot = EffectiveOrderSnapshot.build([buy, sell], dt.date(2025, 1, 1))
    assert snapshot.orders == (buy, sell)


@pytest.mark.parametrize("currency", ["GBp", "GBX", "ZAc", "ILa"])
def test_minor_currency_fallback_agrees_with_snapshot_and_cash(currency):
    order = _order(
        quantity=10, net=-625, price=5000, currency=currency, fx=0.8
    )
    series = build_order_derived_series(
        [order], {}, today=dt.date(2025, 1, 2)
    )
    assert series.valuations[-1][1] == pytest.approx(625)
    assert series.valuations[-1][1] == pytest.approx(
        _seed_market_value([order], order.isin, order.quantity)
    )
    assert sum(value for _, value in series.xirr_cashflows) == pytest.approx(0)


def test_fractional_position_is_not_discarded():
    order = _order(quantity=0.005, net=-500, price=100000)
    holdings = build_holdings_from_orders([order])
    assert len(holdings) == 1
    assert holdings[0].quantity == 0.005
    assert holdings[0].market_value_eur == 500
    series = build_order_derived_series(
        [order], {}, today=dt.date(2025, 1, 2)
    )
    assert series.valuations[-1][1] == 500
    sell = _order(2, OrderType.SELL, -0.0025, 250, price=100000)
    assert cost_basis_by_isin([order, sell])[order.isin] == 250


@pytest.mark.parametrize("source_kind", ["missing", "empty", "future", "malformed"])
def test_invalid_order_input_has_critical_failure_evidence(source_kind, tmp_path):
    header = "date,type,isin,quantity,gross_eur,net_eur\n"
    sources = {
        "missing": str(tmp_path / "absent.csv"),
        "empty": io.BytesIO(header.encode()),
        "future": io.BytesIO(
            (header + "2025-02-01,buy,US0000000001,10,-1000,-1000\n").encode()
        ),
        "malformed": io.BytesIO(b"date,type\n2025-01-01,buy\n"),
    }
    with pytest.raises(DataIngestionError):
        run(
            orders_source=sources[source_kind],
            orders_filename="order_list.csv",
            deterministic=True,
            as_of=dt.date(2025, 1, 31),
        )
    result = last_run_result()
    assert result is not None
    assert result.metrics is None
    publication = PublicationEvaluator.evaluate(result.ledger.failure_records())
    assert publication.decision is PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE
    summary = SummaryProjector.project(result)
    assert summary.sections["portfolio"]["availability"] == "UNAVAILABLE"
    assert summary.sections["planning"]["availability"] == "UNAVAILABLE"


def test_only_one_concurrent_sender_can_cross_the_smtp_boundary(tmp_path):
    store = LocalJsonDeliveryClaimStore(tmp_path / "claims.json")
    intent = DeliveryIntent(
        "same-event", DeliveryPurpose.NORMAL_NEWSLETTER, "digest", "v1"
    )
    barrier = Barrier(2)

    def authorize():
        claim = store.claim(intent)
        assert claim.state is DeliveryState.CLAIMED
        barrier.wait(timeout=5)
        try:
            store.transition(
                intent.logical_id,
                (DeliveryState.CLAIMED,),
                DeliveryState.SMTP_INVOCATION_STARTED,
            )
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(authorize) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results.count(True) == 1
    assert results.count(False) == 1


def _authorize_in_separate_process(path, barrier, results):
    store = LocalJsonDeliveryClaimStore(path)
    intent = DeliveryIntent(
        "same-process-event", DeliveryPurpose.NORMAL_NEWSLETTER, "digest", "v1"
    )
    store.claim(intent)
    barrier.wait(timeout=15)
    try:
        store.transition(
            intent.logical_id, (DeliveryState.CLAIMED,),
            DeliveryState.SMTP_INVOCATION_STARTED,
        )
    except DeliveryTransitionConflict:
        results.put(False)
    else:
        results.put(True)


def test_claim_compare_and_swap_is_also_exclusive_between_processes(tmp_path):
    pytest.importorskip("fcntl")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    workers = [
        context.Process(
            target=_authorize_in_separate_process,
            args=(tmp_path / "process-claims.json", barrier, results),
        ) for _ in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        outcomes = [results.get(timeout=20) for _ in range(2)]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sorted(outcomes) == [False, True]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()


def test_unavailable_process_lock_blocks_the_local_claim(monkeypatch, tmp_path):
    fcntl = pytest.importorskip("fcntl")
    store = LocalJsonDeliveryClaimStore(tmp_path / "unlocked-claims.json")

    def unavailable(*args):
        raise OSError("locking unavailable")

    monkeypatch.setattr(fcntl, "flock", unavailable)
    intent = DeliveryIntent("same-event", DeliveryPurpose.NORMAL_NEWSLETTER, "digest", "v1")
    with pytest.raises(RuntimeError, match="process-safe file locking"):
        store.claim(intent)
    assert not store.path.exists()


def test_transfer_commission_does_not_replace_transferred_cost_basis():
    transfer = _order(
        kind=OrderType.TRANSFER_IN, gross=1000, net=-5
    )
    sell = _order(2, OrderType.SELL, -5, 500)
    orders = [transfer, sell]
    assert cost_basis_by_isin(orders)[transfer.isin] == 500
    holdings = build_holdings_from_orders(orders)
    assert holdings[0].cost_basis_eur == 500
    assert estimate_realized_cgt(orders, {}, 26, 12.5).total_tax_eur == 0
    series = build_order_derived_series(
        orders, {}, today=dt.date(2025, 1, 3)
    )
    assert series.unrealized_series.iloc[-1] == pytest.approx(0)


def test_distribution_on_total_return_history_requires_price_only_evidence():
    orders = [
        _order(),
        _order(2, OrderType.DIVIDEND, 0, 20, price=None),
    ]
    holding = _holding([100, 102, 102])
    series = build_order_derived_series(
        orders, {holding.isin: holding}, today=dt.date(2025, 1, 3)
    )
    assert series.history_availability.value == "UNAVAILABLE"
    assert series.unavailable_instruments == (holding.isin,)


def test_tax_kind_resolution_is_once_per_economic_identity(monkeypatch):
    from tarzan.engine import tax

    orders = [_order(quantity=100, net=-10000)]
    orders.extend(
        _order(2, OrderType.SELL, -1, 110) for _ in range(50)
    )
    calls = []
    classify = tax._classify

    def counted(*args, **kwargs):
        calls.append(1)
        return classify(*args, **kwargs)

    monkeypatch.setattr(tax, "_classify", counted)
    estimate = estimate_realized_cgt(orders, {}, 26, 12.5)
    assert estimate.total_tax_eur == pytest.approx(130)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("movement", "net", "gross", "quantity", "expected_pnl"),
    [
        (OrderType.TRANSFER_OUT, 0, 0, -5, 0),
        (OrderType.TRANSFER_OUT, -5, 0, -5, -5),
        (OrderType.TRANSFER_IN, -5, 1000, 10, -5),
    ],
)
def test_security_transfers_are_external_capital_not_investment_loss(
    movement, net, gross, quantity, expected_pnl,
):
    orders = (
        [_order(), _order(2, movement, quantity, net, gross=gross)]
        if movement is OrderType.TRANSFER_OUT
        else [_order(kind=movement, quantity=quantity, net=net, gross=gross)]
    )
    series = build_order_derived_series(orders, {}, today=dt.date(2025, 1, 3))
    assert sum(value for _, value in series.xirr_cashflows) == expected_pnl
    assert series.pnl_series.iloc[-1] == expected_pnl


def test_invalid_run_context_cannot_leave_an_earlier_result_as_current():
    with pytest.raises(DataIngestionError):
        run(
            orders_source=io.BytesIO(b"date,type,isin,quantity,gross_eur,net_eur\n"),
            orders_filename="order_list.csv",
            deterministic=True, as_of=dt.date(2025, 1, 31),
        )
    previous = last_run_result()
    with pytest.raises(ValueError):
        run(deterministic=True, as_of=None)
    failed = last_run_result()
    assert failed is not previous
    assert failed.metrics is None
    assert failed.attempt_id != previous.attempt_id
    assert any(record.stable_code == "INVALID_RUN_CONTEXT" for record in failed.ledger.failure_records())
