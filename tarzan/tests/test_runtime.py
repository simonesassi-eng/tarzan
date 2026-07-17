"""Tests for the deterministic / as-of run context (tarzan.runtime).

Scope of the guarantee (deliberately): deterministic mode makes the computed
NUMBERS reproducible (pinned clock, no live intraday, no AI) and leaves an
ordinary run byte-for-byte unchanged. It does NOT promise a byte-identical
newsletter — a residual row-order non-determinism remains (holding rows can
tie on weight and their order traces to a set() of open ISINs); that is a
known, documented limitation, not something these tests assert away.
"""

from __future__ import annotations

import datetime

from tarzan import runtime


class TestRuntimeContext:
    def teardown_method(self):
        runtime.reset()  # never leak a pinned context into other tests

    def test_default_is_live_and_not_deterministic(self):
        runtime.reset()
        assert runtime.is_deterministic() is False
        assert runtime.as_of() is None
        assert runtime.today() == datetime.date.today()

    def test_as_of_pins_today_without_deterministic(self):
        runtime.configure(as_of=datetime.date(2026, 6, 30))
        assert runtime.is_deterministic() is False
        assert runtime.today() == datetime.date(2026, 6, 30)

    def test_deterministic_flag_and_pinned_stamp(self):
        runtime.configure(deterministic=True, as_of=datetime.date(2026, 6, 30))
        assert runtime.is_deterministic() is True
        assert runtime.today() == datetime.date(2026, 6, 30)
        # Pinned stamp derives from as_of (midnight), so headers don't vary.
        assert runtime.now_stamp("%Y-%m-%d") == "2026-06-30"

    def test_reset_restores_live(self):
        runtime.configure(deterministic=True, as_of=datetime.date(2020, 1, 1))
        runtime.reset()
        assert runtime.is_deterministic() is False
        assert runtime.today() == datetime.date.today()


class TestReturnsHonorAsOf:
    """The order-derived series must value AS OF the pinned date, so a pinned
    run is reproducible and can reproduce a historically-reported figure."""

    def teardown_method(self):
        runtime.reset()

    def test_build_series_uses_runtime_today_when_no_explicit_today(self):
        import datetime as _dt
        from tarzan.engine.returns_builder import build_order_derived_series
        from tarzan.models.order import Order, OrderType

        def _o(d):
            return Order(date=d, trade_date=d, type=OrderType.BUY, isin="IE00TEST0001",
                         name="x", ticker="", quantity=10.0, currency="EUR",
                         price_native=100.0, fx_rate=1.0, gross_eur=1000.0,
                         fees_eur=0.0, net_eur=-1000.0, source="t")
        orders = [_o(_dt.date(2026, 1, 5))]

        runtime.configure(as_of=_dt.date(2026, 6, 30))
        s = build_order_derived_series(orders, {})  # no explicit today
        # Terminal valuation date is the pinned as_of, not the live date.
        assert s.valuations[-1][0] == _dt.date(2026, 6, 30)
        assert s.span_days == (_dt.date(2026, 6, 30) - _dt.date(2026, 1, 5)).days


class TestOpenIsinsDeterministicOrder:
    """Regression: build_holdings_from_orders must emit holdings in a stable
    (sorted-ISIN) order, since it derives them from a set() of open ISINs."""

    def test_holdings_sorted_by_isin(self):
        import datetime as _dt
        from tarzan.engine.returns_builder import build_holdings_from_orders
        from tarzan.models.order import Order, OrderType

        def _o(isin):
            return Order(date=_dt.date(2026, 1, 1), trade_date=_dt.date(2026, 1, 1),
                         type=OrderType.BUY, isin=isin, name="x", ticker="",
                         quantity=10.0, currency="EUR", price_native=100.0,
                         fx_rate=1.0, gross_eur=1000.0, fees_eur=0.0,
                         net_eur=-1000.0, source="t")
        # Deliberately unsorted input ISINs.
        orders = [_o("IE00ZZZ00001"), _o("IE00AAA00001"), _o("IE00MMM00001")]
        holdings = build_holdings_from_orders(orders)
        isins = [h.isin for h in holdings]
        assert isins == sorted(isins)  # stable, not set-hash order


# ---------------------------------------------------------------------------
# Production-readiness bug exploration: C2 coherent run modes
# ---------------------------------------------------------------------------

from hypothesis import given as _given, settings as _settings, strategies as _st  # noqa: E402


# **Validates: Requirements 2.2**
@_given(
    effective_date=_st.dates(
        min_value=datetime.date(2018, 1, 1),
        max_value=datetime.date(2020, 12, 31),
    )
)
@_settings(max_examples=5, deadline=None, derandomize=True)
def test_c2_asof_only_mode_uses_one_clock_and_blocks_live_transports(effective_date):
    """Property 1 / C2 exploration for the accepted ``as_of``-only mode.

    CLI messaging calls this deterministic, but runtime gates only on the
    separate deterministic flag.  Capture the report/artifact stamp plus live
    market and Gemini transports in one diagnostic rather than stopping at the
    first inconsistent surface.
    """
    import pandas as pd
    import pytest

    from tarzan.data import market_quotes, price_cache
    from tarzan.engine.metrics import MetricsEngine
    from tarzan.export import ai_summary
    from tarzan.models.investor_config import InvestorConfig

    calls: list[object] = []
    violations: dict[str, object] = {}

    def fake_broker_1d(tickers):
        calls.append(("market", tuple(tickers)))
        return {str(t): {"pct": 1.25, "live": True} for t in tickers}

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("TARZAN_DISABLE_AI", raising=False)
            mp.setenv("GEMINI_API_KEY", "test-key-not-sent")
            mp.setattr(ai_summary, "build_digest", lambda *a, **k: {})
            mp.setattr(
                ai_summary,
                "_call_gemini",
                lambda *a, **k: calls.append("gemini") or "Market context complete.",
            )
            mp.setattr(price_cache, "load_resolution", lambda ticker: ticker)
            mp.setattr(market_quotes, "broker_1d", fake_broker_1d)

            runtime.configure(deterministic=False, as_of=effective_date)
            artifact_stamp = runtime.now_stamp("%Y-%m-%d %H:%M")
            ai_summary.generate_summary(object(), object())

            engine = MetricsEngine([], InvestorConfig())
            engine._live_1d({
                "holding_performance": pd.DataFrame({"ticker": ["TEST"], "1d": [0.0]}),
                "holdings_df": pd.DataFrame(),
                "performance": {"1d": 0.0},
                "performance_full": {"1d": 0.0},
            })

            expected_stamp = f"{effective_date.isoformat()} 00:00"
            if artifact_stamp != expected_stamp:
                violations["artifact_timestamp"] = {
                    "expected": expected_stamp,
                    "actual": artifact_stamp,
                }
            live_calls = [call for call in calls if call == "gemini" or
                          (isinstance(call, tuple) and call[0] == "market")]
            if live_calls:
                violations["live_transports"] = live_calls
    finally:
        runtime.reset()

    assert violations == {}, (
        "as_of-only execution advertises a point-in-time run but retains "
        f"split clocks/live surfaces: {violations}"
    )


# **Validates: Requirements 2.2**
def test_c2_reproducible_flag_without_effective_date_is_rejected_actionably():
    """A reproducible request without its effective boundary is unsupported."""
    import pytest

    try:
        with pytest.raises(ValueError, match="as.of|effective date"):
            runtime.configure(deterministic=True, as_of=None)
    finally:
        runtime.reset()
