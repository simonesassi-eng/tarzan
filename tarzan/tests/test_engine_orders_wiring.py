"""Engine-wiring tests for the optional order list (Properties 1 & 6).

These assert the structural contract without hitting the network:
- no orders → pipeline is exactly today's (identity).
- with orders → the history provider is swapped and _returns appended,
  and the order computers populate the single ctx["portfolio_history"]
  that _performance/_risk read.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType
from tarzan.instruments.registry import InstrumentKind


def _order(otype, isin, qty=0.0, net=0.0, gross=0.0, price=None, d=(2025, 1, 2)):
    return Order(
        date=datetime.date(*d), trade_date=datetime.date(*d), type=otype,
        isin=isin, name="X", ticker="", quantity=qty, currency="EUR",
        price_native=price, fx_rate=1.0, gross_eur=gross, fees_eur=0.0,
        net_eur=net, source="fineco", instrument_kind=InstrumentKind.STOCK,
    )


def _enriched_holding(isin, qty, prices, start=(2025, 1, 1)):
    idx = pd.date_range(start=datetime.date(*start), periods=len(prices), freq="D")
    h = Holding(isin=isin, ticker=isin, quantity=qty, cost_basis_eur=0.0,
                market_value_eur=qty * prices[-1], currency="EUR",
                name=isin, current_price=prices[-1],
                current_value=qty * prices[-1], asset_class=AssetClass.EQUITIES)
    h.price_history = pd.Series(prices, index=idx)
    return h


class TestProperty1Identity:
    def test_no_orders_keeps_default_pipeline(self, sample_holdings, sample_config):
        engine = MetricsEngine(sample_holdings, sample_config)
        names = [c.__name__ for c in engine._computers]
        assert "_portfolio_history" in names
        assert "_portfolio_history_from_orders" not in names
        assert "_returns" not in names

    def test_orders_swap_provider_and_append_returns(self, sample_holdings, sample_config):
        orders = [_order(OrderType.BUY, "US0000000001", qty=100.0, net=-6000.0)]
        engine = MetricsEngine(sample_holdings, sample_config, orders=orders)
        names = [c.__name__ for c in engine._computers]
        assert "_portfolio_history" not in names
        assert "_portfolio_history_from_orders" in names
        assert "_returns" in names
        # _allocation_timeline is appended last (after _returns) on the
        # order path to feed the newsletter Diversification sparklines.
        assert names[-1] == "_allocation_timeline"
        # Benchmark preprocessing is the first base computer. The order path
        # keeps that 16-computer base, swaps the history provider in place,
        # then appends _returns and _allocation_timeline.
        assert len(names) == 18
        # One current point is stamped onto every price series before anything
        # reads a price, so "today" has a single source.
        assert "_current_prices" in names
        # Broker-style live 1D runs after per-instrument performance.
        assert "_live_1d" in names
        # The historical-risk computer is part of the base pipeline.
        assert "_historical_risk" in names


class TestProperty6SingleSeries:
    def test_order_computers_share_one_history_series(self, sample_config):
        # Two enriched holdings with real history; orders that open them.
        h1 = _enriched_holding("US0000000001", 100.0, [60.0, 61.0, 62.0])
        h2 = _enriched_holding("EU0000000001", 50.0, [40.0, 40.0, 41.0])
        orders = [
            _order(OrderType.BUY, "US0000000001", qty=100.0, net=-6000.0, price=60.0),
            _order(OrderType.BUY, "EU0000000001", qty=50.0, net=-2000.0, price=40.0),
        ]
        engine = MetricsEngine([h1, h2], sample_config, orders=orders)

        ctx: dict = {}
        engine._portfolio_history_from_orders(ctx)
        # The provider populated the single series the others consume.
        assert "portfolio_history" in ctx
        assert isinstance(ctx["portfolio_history"], pd.Series)
        assert not ctx["portfolio_history"].empty
        # _returns reads the stashed order series and fills metrics.
        engine._returns(ctx)
        assert "twror_pct" in ctx
        assert "xirr_pct" in ctx
        assert ctx["returns_coverage_pct"] == pytest.approx(100.0, abs=1e-6)

        # _performance reads the SAME ctx["portfolio_history"] object.
        series_obj = ctx["portfolio_history"]
        engine._performance(ctx)
        assert ctx["portfolio_history"] is series_obj


# ---------------------------------------------------------------------------
# Production-readiness bug exploration: C1 effective-order boundary
# ---------------------------------------------------------------------------

from hypothesis import given as _given, settings as _settings, strategies as _st  # noqa: E402


# **Validates: Requirements 2.1**
@_given(
    days_after=_st.integers(min_value=1, max_value=30),
    future_quantity=_st.integers(min_value=1, max_value=25),
)
@_settings(max_examples=5, deadline=None, derandomize=True)
def test_c1_orchestrator_never_wires_post_asof_orders_to_financial_consumers(
    days_after, future_quantity,
):
    """Property 1 / C1 exploration.

    A pinned run must form the effective order view before both snapshot
    derivation and MetricsEngine wiring.  The unfixed orchestrator passes the
    accepted list through unchanged, so this captures the shared causal seam
    for holdings/cost, returns/tax/timeline, targets, and planning.
    """
    import io

    from tarzan import orchestrator, runtime
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.portfolio import PortfolioMetrics

    cutoff = datetime.date(2025, 6, 29)
    future_day = cutoff + datetime.timedelta(days=days_after)
    orders = [
        _order(OrderType.BUY, "US0000000001", qty=10.0, net=-1000.0,
               price=100.0, d=(2025, 1, 2)),
        _order(OrderType.BUY, "US0000000001", qty=float(future_quantity),
               net=-100.0 * future_quantity, price=100.0,
               d=(future_day.year, future_day.month, future_day.day)),
    ]
    config = InvestorConfig()
    captured: dict[str, list[Order]] = {}

    snapshot_holding = Holding(
        isin="US0000000001", ticker="TEST", quantity=10.0,
        cost_basis_eur=1000.0, market_value_eur=1000.0, currency="EUR",
        name="Test holding", current_price=100.0, current_value=1000.0,
        asset_class=AssetClass.EQUITIES,
    )

    def fake_build_holdings(received):
        captured["snapshot"] = list(received)
        return [snapshot_holding]

    class CapturingMetricsEngine:
        def __init__(self, holdings, cfg, orders=None, rebalance_seeds=None):
            captured["engine"] = list(orders or [])

        def compute_all(self):
            return PortfolioMetrics(total_value=1000.0)

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(orchestrator, "load_config", lambda *a, **k: config)
            mp.setattr(orchestrator, "load_orders", lambda *a, **k: list(orders))
            mp.setattr(orchestrator, "_load_targets_or_empty", lambda *a, **k: {})
            mp.setattr(orchestrator, "_check_taxonomy_coverage", lambda *a, **k: None)
            mp.setattr(orchestrator, "MetricsEngine", CapturingMetricsEngine)
            mp.setattr(
                "tarzan.engine.returns_builder.build_holdings_from_orders",
                fake_build_holdings,
            )
            mp.setattr("tarzan.data.enricher.enrich_holdings", lambda hs: hs)
            mp.setattr(
                "tarzan.data.enricher.set_portfolio_backtest_period",
                lambda *a, **k: None,
            )
            orchestrator.run(
                orders_source=io.BytesIO(b"intercepted"),
                deterministic=True,
                as_of=cutoff,
            )
    finally:
        runtime.reset()

    leaked = {
        consumer: [o.trade_date.isoformat() for o in received
                   if o.trade_date > cutoff]
        for consumer, received in captured.items()
        if any(o.trade_date > cutoff for o in received)
    }
    assert leaked == {}, (
        "post-as_of orders crossed the effective-input boundary; all financial "
        f"consumers share this unfiltered list: cutoff={cutoff}, leaked={leaked}"
    )
