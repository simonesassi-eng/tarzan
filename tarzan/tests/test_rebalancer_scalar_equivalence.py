"""Exact-reference gates for the scalar-safe solver optimizations.

The reference expressions/descent are intentionally the pre-optimization
algorithm at 4da5acf, not a second call to the optimized implementation.
Compare float bits, vectors and complete executable plans, including fees,
tax, funding and verification; similar objective values are not sufficient.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from tarzan.engine import rebalancer as rb


def _reference_cost(cost, t):
    p = cost.params
    nv = cost.values + t
    g = np.abs(cost.model.gaps(nv))
    j = cost.gap_term(g)
    allowed = np.maximum(cost.gap0, cost.model.tol)
    excess = np.clip(g - allowed, 0.0, None)
    j += p.guard_weight * float(np.sum(excess ** 2))
    turnover = float(np.abs(t).sum())
    n_trades = int((np.abs(t) >= p.min_trade_eur).sum())
    residual = rb._net_cash_cost(
        t, cost.tax, cost.fee_buy, cost.fee_sell, p.min_trade_eur
    ) - cost.lump_sum
    j += p.turnover_weight * (turnover / max(cost.tv, 1e-9))
    j += p.trade_count_weight * n_trades
    j += p.residual_weight * (residual / max(cost.tv, 1e-9) * 100.0) ** 2
    return j


class _ReferenceCost(rb._Cost):
    def __call__(self, t):
        return _reference_cost(self, t)


def _reference_descent(t, cost, lo, hi, tradeable, steps, params):
    t = t.copy()
    cur = cost(t)
    tabu = {}
    for it in range(params.max_iters_per_descent):
        best_gain = 1e-12
        best_move = None
        for i in np.where(tradeable)[0]:
            for s in steps:
                for sign in (+1.0, -1.0):
                    if tabu.get((int(i), int(sign)), -1) > it:
                        continue
                    new_ti = min(max(t[i] + sign * s, lo[i]), hi[i])
                    if new_ti == t[i]:
                        continue
                    old = t[i]
                    t[i] = new_ti
                    c = cost(t)
                    t[i] = old
                    gain = cur - c
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (int(i), new_ti, int(sign))
        if best_move is None:
            break
        i, new_ti, sign = best_move
        t[i] = new_ti
        cur = cost(t)
        tabu[(i, -sign)] = it + params.tabu_tenure
    return t, cur


def _assert_exact_float(actual, expected):
    assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


def _assert_exact_descent(t, cost, lo, hi, tradeable, steps, params):
    inputs = [a.copy() for a in (t, lo, hi, tradeable)]
    expected_t, expected_cost = _reference_descent(
        t, cost, lo, hi, tradeable, steps, params
    )
    actual_t, actual_cost = rb._descent(
        t, cost, lo, hi, tradeable, steps, params
    )
    assert actual_t.dtype == expected_t.dtype
    if actual_t.dtype.itemsize > np.dtype(np.float64).itemsize:
        # x86 extended precision uses 80 meaningful bits in a 16-byte slot;
        # its padding bytes need not agree even for identical scalar values.
        np.testing.assert_array_equal(actual_t, expected_t)
        np.testing.assert_array_equal(
            np.signbit(actual_t), np.signbit(expected_t)
        )
    else:
        assert actual_t.tobytes() == expected_t.tobytes()
    _assert_exact_float(actual_cost, expected_cost)
    for actual, before in zip((t, lo, hi, tradeable), inputs):
        assert actual.tobytes() == before.tobytes()
    return actual_t


@pytest.mark.parametrize("objective", [
    "ambit_maximin", "ambit_sq", "sum_sq", "minimax",
])
@pytest.mark.parametrize("gap_norm", ["initial", "tolerance", "absolute"])
@pytest.mark.parametrize("per_holding_only", [False, True])
@pytest.mark.parametrize("min_trade", [0.0, 1.0, 10.0])
def test_cost_hoisting_is_bitwise_equivalent(
    sample_holdings, sample_config, objective, gap_norm,
    per_holding_only, min_trade,
):
    sample_config.target_use_per_holding_only = per_holding_only
    sample_config.rebalancing_transaction_fee_buy_eur = 1.75
    sample_config.rebalancing_transaction_fee_sell_eur = 2.25
    for i, h in enumerate(sample_holdings):
        h.target_portfolio = (45.0, 20.0, 10.0, 25.0, 0.0)[i]
        h.target_equities = (60.0, 25.0, 15.0, None, None)[i]
        h.target_fixed_income = 100.0 if i == 3 else None
        h.gain_pct = (100.0, -35.0, 0.0, 12.5, 0.0)[i]
    values = np.array([h.current_value for h in sample_holdings])
    model = rb._ObjectiveModel(sample_holdings, sample_config, values)
    params = rb.LSParams(
        objective=objective, gap_norm=gap_norm, min_trade_eur=min_trade
    )
    cost = rb._Cost(model, values, sample_config, 237.42, params)
    rng = np.random.default_rng(48271)
    vectors = [np.zeros(len(values)), -values.copy()]
    vectors.extend(rng.uniform(-1000.0, 2000.0, size=(12, len(values))))
    for boundary in (min_trade, -min_trade, 0.0):
        for value in (
            np.nextafter(boundary, -np.inf), boundary,
            np.nextafter(boundary, np.inf),
        ):
            vector = np.zeros(len(values))
            vector[0] = value
            vectors.append(vector)
    for vector in vectors:
        _assert_exact_float(cost(vector), _reference_cost(cost, vector))


@pytest.mark.parametrize("seed", range(48))
def test_adversarial_descent_prefixes_match_scalar_reference(seed):
    """Each iteration prefix probes the accepted path, not just its basin."""
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    lo = -rng.uniform(0.0, 20.0, n)
    hi = rng.uniform(0.0, 20.0, n)
    t = rng.uniform(lo, hi)
    tradeable = rng.random(n) > 0.25
    if seed % 3 == 0:
        hi[0] = lo[0]
        t[0] = lo[0]
        tradeable[0] = False
    if seed % 7 == 0:
        # The helper's existing behavior outside bounds must stay unchanged.
        t[-1] = hi[-1] + 2.0
    target = rng.uniform(lo, hi)
    matrix = rng.normal(size=(n, n))
    min_trade = (0.0, 1.0, 3.0)[seed % 3]
    fee = (0.0, 0.04, 4.0)[seed % 3]

    def cost(vector):
        gap = matrix @ (vector - target)
        result = float(np.sum(gap ** 2))
        result += fee * int((np.abs(vector) >= min_trade).sum())
        result += 0.5 * float(vector.sum()) ** 2
        return result

    steps = [
        100.0, 20.0, 3.0, 1.0, 1.0, 0.0, -1.0, -3.0,
        np.nextafter(1.0, -np.inf), np.nextafter(1.0, np.inf),
    ]
    params = rb.LSParams(tabu_tenure=seed % 6)
    for iterations in (0, 1, 2, 3, 5, 8):
        _assert_exact_descent(
            t, cost, lo, hi, tradeable, steps,
            replace(params, max_iters_per_descent=iterations),
        )


@pytest.mark.parametrize("gain", [
    np.nextafter(1e-12, 0.0), 1e-12, np.nextafter(1e-12, np.inf),
])
def test_strict_gain_threshold_does_not_merge_nearby_costs(gain):
    def cost(t):
        return -gain if t[0] == 1.0 else (0.0 if t[0] == 0.0 else 2.0)

    result = _assert_exact_descent(
        np.array([0.0]), cost, np.array([-1.0]), np.array([1.0]),
        np.array([True]), [1.0, 1.0],
        rb.LSParams(max_iters_per_descent=1),
    )
    assert result[0] == (1.0 if gain > 1e-12 else 0.0)


def test_close_but_distinct_candidates_are_not_deduplicated():
    above_one = np.nextafter(1.0, np.inf)

    def cost(t):
        if t[0] == above_one:
            return -2.0
        if t[0] == 1.0:
            return -1.0
        return 0.0

    result = _assert_exact_descent(
        np.array([0.0]), cost, np.array([0.0]), np.array([2.0]),
        np.array([True]), [1.0, above_one],
        rb.LSParams(max_iters_per_descent=1),
    )
    _assert_exact_float(result[0], above_one)


def test_equal_cost_moves_keep_first_coordinate():
    def cost(t):
        return 0.0 if np.any(t != 0.0) else 1.0

    result = _assert_exact_descent(
        np.zeros(3), cost, np.full(3, -1.0), np.ones(3),
        np.ones(3, dtype=bool), [1.0, 2.0],
        rb.LSParams(max_iters_per_descent=3),
    )
    assert result.tolist() == [1.0, 0.0, 0.0]


def test_first_clamped_occurrence_keeps_its_sign_and_reverse_tabu():
    def cost(t):
        return float(t[0])

    result = _assert_exact_descent(
        np.array([2.0]), cost, np.array([0.0]), np.array([1.0]),
        np.array([True]), [1.0],
        rb.LSParams(max_iters_per_descent=3, tabu_tenure=5),
    )
    # Both signs first clamp to +1; the first (+) occurrence wins, so
    # its reverse (-) remains tabu, despite +0 having a smaller cost.
    assert result.tolist() == [1.0]


def test_signed_zero_candidates_remain_distinct():
    def cost(t):
        if t[0] == 0.0:
            return 0.0 if np.signbit(t[0]) else 1.0
        return 2.0

    result = _assert_exact_descent(
        np.array([-1.0]), cost, np.array([-0.0]), np.array([0.0]),
        np.array([True]), [1.0],
        rb.LSParams(max_iters_per_descent=1),
    )
    assert np.signbit(result[0])


@pytest.mark.parametrize("dtype", [np.int64, object])
def test_wide_integers_do_not_alias_through_python_float(dtype):
    lower = 2 ** 53
    upper = lower + 1

    def cost(t):
        return float(upper - int(t[0]))

    args = (
        np.array([lower - 2], dtype=dtype), cost,
        np.array([lower], dtype=dtype), np.array([upper], dtype=dtype),
        np.array([True]), [1.0, 100.0],
        rb.LSParams(max_iters_per_descent=1),
    )
    expected = _reference_descent(*args)
    actual = rb._descent(*args)
    np.testing.assert_array_equal(actual[0], expected[0])
    _assert_exact_float(actual[1], expected[1])
    assert int(actual[0][0]) == upper


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
    reason="longdouble has no precision beyond float64 on this platform",
)
def test_extended_precision_candidates_remain_distinct():
    lower = np.longdouble(1.0)
    upper = np.nextafter(lower, np.longdouble(np.inf))
    assert float(lower) == float(upper)  # the formerly lossy-key case

    def cost(t):
        if t[0] == upper:
            return -2.0
        if t[0] == lower:
            return -1.0
        return 0.0

    result = _assert_exact_descent(
        np.array([0.0], dtype=np.longdouble), cost,
        np.array([0.0], dtype=np.longdouble),
        np.array([2.0], dtype=np.longdouble), np.array([True]),
        [lower, upper], rb.LSParams(max_iters_per_descent=1),
    )
    assert result[0] == upper


def test_duplicate_candidates_reduce_only_redundant_scalar_evaluations():
    calls = []

    def cost(t):
        calls.append(t.copy())
        return float(((t - 1.0) ** 2).sum())

    args = (
        np.zeros(2), cost, np.zeros(2), np.ones(2),
        np.ones(2, dtype=bool), [1.0, 2.0, 3.0, 4.0],
        rb.LSParams(max_iters_per_descent=1),
    )
    expected = _reference_descent(*args)
    reference_calls = len(calls)
    calls.clear()
    actual = rb._descent(*args)
    assert actual[0].tobytes() == expected[0].tobytes()
    _assert_exact_float(actual[1], expected[1])
    assert reference_calls == 10
    assert len(calls) == 4  # initial + one unique candidate/coordinate + move


def test_seen_is_reset_when_another_coordinate_has_changed():
    result = _assert_exact_descent(
        np.zeros(3), lambda t: float(((t - 1.0) ** 2).sum()),
        np.zeros(3), np.ones(3), np.ones(3, dtype=bool),
        [1.0, 2.0, 3.0], rb.LSParams(max_iters_per_descent=5),
    )
    assert result.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("step", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidates_keep_scalar_path(step):
    calls = []

    def cost(t):
        calls.append(t.copy())
        return 0.0 if np.all(t == 0.0) else 1.0

    args = (
        np.array([0.0]), cost, np.array([-np.inf]), np.array([np.inf]),
        np.array([True]), [step, step],
        rb.LSParams(max_iters_per_descent=1),
    )
    with np.errstate(invalid="ignore"):
        expected = _reference_descent(*args)
        reference_calls = len(calls)
        calls.clear()
        actual = rb._descent(*args)
    assert actual[0].tobytes() == expected[0].tobytes()
    _assert_exact_float(actual[1], expected[1])
    assert len(calls) == reference_calls


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("no_sell", [False, True])
@pytest.mark.parametrize("per_holding_only", [False, True])
def test_complete_plans_match_original_scalar_search(
    sample_holdings, sample_config, monkeypatch,
    seed, no_sell, per_holding_only,
):
    """Restarts/RNG, taxation, fees, forced exits and final proof stay exact."""
    sample_config.target_use_per_holding_only = per_holding_only
    sample_config.rebalancing_no_sell = no_sell
    sample_config.rebalancing_transaction_fee_buy_eur = 1.75
    sample_config.rebalancing_transaction_fee_sell_eur = 2.25
    sample_holdings[1].ticker = sample_holdings[0].ticker
    for i, h in enumerate(sample_holdings):
        h.target_portfolio = (55.0, 0.0, 20.0, 25.0, 0.0)[i]
        h.gain_pct = (125.0, -35.0, 0.0, 17.0, 0.0)[i]
    sample_holdings[2].no_buy_no_sell = True
    params = rb.LSParams(
        seed=seed, n_restarts=5, max_iters_per_descent=25,
        step_fractions=(0.05, 0.02, 0.01, 0.005, 0.001, 0.01),
    )
    originals = copy.deepcopy(sample_holdings)
    total_value = sum(h.current_value for h in sample_holdings)
    expected = None
    with monkeypatch.context() as mp:
        mp.setattr(rb, "_Cost", _ReferenceCost)
        mp.setattr(rb, "_descent", _reference_descent)
        expected = rb.optimize_local_search(
            sample_holdings, sample_config, total_value,
            lump_sum=250.37, params=params,
        )
    actual = rb.optimize_local_search(
        sample_holdings, sample_config, total_value,
        lump_sum=250.37, params=params,
    )
    assert actual == expected
    assert sample_holdings == originals
    for action in actual[0]:
        assert sample_holdings[action["idx"]].ticker == action["ticker"]
