"""Tests for the pure XIRR/TWROR return functions in metrics.py."""

from __future__ import annotations

import datetime
import math

import pytest
from hypothesis import given, settings, strategies as st

from tarzan.engine.metrics import TwrorResult, twror, xirr, xnpv


def _d(y, m, day) -> datetime.date:
    return datetime.date(y, m, day)


class TestXirr:
    def test_single_deposit_then_growth(self):
        # Invest 1000, worth 1100 exactly one year later → ~10% XIRR.
        cf = [(_d(2025, 1, 1), -1000.0), (_d(2026, 1, 1), 1100.0)]
        assert xirr(cf) == pytest.approx(0.10, abs=1e-3)

    def test_doubling_in_one_year(self):
        cf = [(_d(2025, 1, 1), -1000.0), (_d(2026, 1, 1), 2000.0)]
        assert xirr(cf) == pytest.approx(1.0, abs=1e-3)

    def test_multi_deposit(self):
        # Two deposits, ends flat in nominal terms but later money had
        # less time invested → small positive rate. Just assert the NPV
        # root holds (precise value checked by the property test).
        cf = [
            (_d(2025, 1, 1), -1000.0),
            (_d(2025, 7, 1), -1000.0),
            (_d(2026, 1, 1), 2100.0),
        ]
        r = xirr(cf)
        assert not math.isnan(r)
        assert abs(xnpv(r, cf)) < 1e-4

    def test_with_withdrawal(self):
        cf = [
            (_d(2025, 1, 1), -1000.0),
            (_d(2025, 6, 1), 200.0),     # partial withdrawal
            (_d(2026, 1, 1), 950.0),
        ]
        r = xirr(cf)
        assert not math.isnan(r)
        assert abs(xnpv(r, cf)) < 1e-4

    def test_all_same_sign_returns_nan(self):
        cf = [(_d(2025, 1, 1), -1000.0), (_d(2026, 1, 1), -500.0)]
        assert math.isnan(xirr(cf))

    def test_too_few_flows_returns_nan(self):
        assert math.isnan(xirr([(_d(2025, 1, 1), -1000.0)]))


class TestTwror:
    def test_flat_price_with_deposit_is_zero(self):
        # Day 1: deposit 1000, value 1000. Day 2: deposit another 1000,
        # value 2000, but prices were flat → 0% market return.
        valuations = [(_d(2025, 1, 1), 1000.0), (_d(2025, 2, 1), 2000.0)]
        flows = {_d(2025, 1, 1): 1000.0, _d(2025, 2, 1): 1000.0}
        res = twror(valuations, flows, span_days=31)
        assert isinstance(res, TwrorResult)
        assert res.cumulative_pct == pytest.approx(0.0, abs=1e-9)

    def test_pure_growth_no_flows(self):
        # 1000 → 1100 with no external flow after inception → +10%.
        valuations = [(_d(2025, 1, 1), 1000.0), (_d(2026, 1, 1), 1100.0)]
        flows = {_d(2025, 1, 1): 1000.0}  # only the initial deposit
        res = twror(valuations, flows, span_days=365)
        assert res.cumulative_pct == pytest.approx(10.0, abs=1e-6)
        assert res.annualized_pct == pytest.approx(10.0, abs=1e-2)

    def test_deposit_does_not_distort_return(self):
        # Grow 1000→1100 (+10%), then deposit 5000 (value 6100), then
        # flat → still +10% cumulative, deposit excluded.
        valuations = [
            (_d(2025, 1, 1), 1000.0),
            (_d(2025, 6, 1), 1100.0),
            (_d(2025, 6, 2), 6100.0),   # +5000 deposit, no market move
        ]
        flows = {_d(2025, 1, 1): 1000.0, _d(2025, 6, 2): 5000.0}
        res = twror(valuations, flows, span_days=152)
        assert res.cumulative_pct == pytest.approx(10.0, abs=1e-6)

    def test_coverage_passthrough(self):
        res = twror([(_d(2025, 1, 1), 1000.0)], {}, span_days=0, coverage_pct=82.5)
        assert res.coverage_pct == 82.5


# ── Property-based ──────────────────────────────────────────────────────────

@st.composite
def _feasible_cashflows(draw):
    """An initial outflow, optional interior flows, and a positive
    terminal inflow large enough to keep the implied rate inside the
    solver bracket (i.e. not a near-total wipeout, where XIRR is
    ill-conditioned and legitimately returns NaN)."""
    n = draw(st.integers(min_value=0, max_value=4))
    start = _d(2024, 1, 1)
    initial = draw(st.floats(min_value=100, max_value=1e6))
    flows = [(start, -initial)]
    total_out = initial
    day = 30
    for _ in range(n):
        day += draw(st.integers(min_value=1, max_value=400))
        amt = draw(st.floats(min_value=-1e5, max_value=1e5))
        if amt < 0:
            total_out += -amt
        flows.append((start + datetime.timedelta(days=day), amt))
    day += draw(st.integers(min_value=1, max_value=800))
    # Terminal inflow at least 10% of total invested, so the implied
    # rate stays well inside (-1, 10) and the NPV root is well-conditioned.
    terminal = draw(st.floats(min_value=0.1 * total_out, max_value=3.0 * total_out))
    flows.append((start + datetime.timedelta(days=day), terminal))
    return flows


class TestReturnProperties:
    @settings(max_examples=150)
    @given(cf=_feasible_cashflows())
    def test_xirr_is_npv_root_or_nan(self, cf):
        # Property 4: returned rate zeroes the NPV, or is NaN. The
        # residual is checked relative to the cash-flow scale — an
        # absolute threshold is not scale-invariant, and near the
        # bracket edge (deep losses, rate → -1) brentq's tolerance on
        # the rate does not pin the NPV to a fixed absolute size.
        r = xirr(cf)
        if not math.isnan(r):
            scale = max(abs(a) for _, a in cf)
            assert abs(xnpv(r, cf)) < 1e-4 * scale

    @settings(max_examples=150)
    @given(
        v0=st.floats(min_value=100, max_value=1e6),
        deposit=st.floats(min_value=0, max_value=1e6),
    )
    def test_deposit_on_flat_prices_is_zero_return(self, v0, deposit):
        # Property 3: a pure deposit on flat prices yields ~0% return.
        valuations = [(_d(2025, 1, 1), v0), (_d(2025, 2, 1), v0 + deposit)]
        flows = {_d(2025, 1, 1): v0, _d(2025, 2, 1): deposit}
        res = twror(valuations, flows, span_days=31)
        assert res.cumulative_pct == pytest.approx(0.0, abs=1e-6)
