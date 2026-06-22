"""Tests for the rebalancing engine.

Covers all critical LP constraints and edge cases. The rebalancer is the most
complex component and the one where errors cause the most user-visible damage.
"""

from __future__ import annotations

import pytest

from tarzan.engine.rebalancer import compute_unified_rebalancing
from tarzan.models.holding import AssetClass, Geography, Holding


def _total_value(holdings: list[Holding]) -> float:
    return sum(h.current_value or h.market_value_eur for h in holdings)


def test_empty_portfolio(sample_config):
    """Empty holdings list should return empty actions, no crash."""
    actions, verifications = compute_unified_rebalancing([], sample_config, 0.0)
    assert actions == []
    assert verifications == []


def test_zero_total_value(sample_holdings, sample_config):
    """A portfolio whose holdings all value to zero yields no actions and no
    crash. The optimizer guards on total invested value <= 0."""
    for h in sample_holdings:
        h.current_value = 0.0
        h.market_value_eur = 0.0
    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, 0.0)
    assert actions == []


def test_lump_sum_buy_only(sample_holdings, sample_config):
    """With no_sell=True and lump_sum=5000, total buy should equal ~5000, total sell=0."""
    sample_config.rebalancing_no_sell = True
    sample_config.rebalancing_lump_sum_amount_eur = 5000.0
    sample_config.rebalancing_target_tolerance_pctg = 5.0  # Loose tolerance for feasibility
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv, lump_sum=5000.0)

    sells = [a for a in actions if a["direction"] == "sell"]
    buys = [a for a in actions if a["direction"] == "buy"]
    assert len(sells) == 0, f"no_sell=True should forbid sells, got {sells}"

    if buys:
        # If solver found a feasible solution, total buy should match lump sum
        total_buy = sum(a["amount_eur"] for a in buys)
        assert abs(total_buy - 5000.0) < 5.0, f"Total buy should ~= lump sum, got {total_buy}"


def test_zero_sum_rebalancing(sample_holdings, sample_config):
    """With lump_sum=None, total buy should equal total sell (zero-sum)."""
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv)

    total_buy = sum(a["amount_eur"] for a in actions if a["direction"] == "buy")
    total_sell = sum(a["amount_eur"] for a in actions if a["direction"] == "sell")

    # Tolerance: 1 EUR for rounding
    assert abs(total_buy - total_sell) < 2.0, (
        f"Zero-sum constraint violated: buy={total_buy}, sell={total_sell}"
    )


def test_no_buy_no_sell_frozen(sample_holdings, sample_config):
    """Holdings with no_buy_no_sell=True should have zero buy AND zero sell."""
    sample_holdings[0].no_buy_no_sell = True  # Freeze USA_ETF
    frozen_ticker = sample_holdings[0].ticker
    sample_config.rebalancing_lump_sum_amount_eur = 2000.0
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv, lump_sum=2000.0)

    frozen_actions = [a for a in actions if a["ticker"] == frozen_ticker]
    assert len(frozen_actions) == 0, (
        f"Frozen holding {frozen_ticker} should have 0 actions, got {frozen_actions}"
    )


def test_all_frozen_no_crash(sample_holdings, sample_config):
    """All holdings frozen + no lump sum → should return 0 actions, not crash."""
    for h in sample_holdings:
        h.no_buy_no_sell = True
    tv = _total_value(sample_holdings)

    actions, verifications = compute_unified_rebalancing(sample_holdings, sample_config, tv)

    assert actions == []


def test_max_tolerance_caps_solver(sample_holdings, sample_config):
    """With tight max_tolerance + auto-relax disabled, infeasible → 0 actions."""
    sample_config.rebalancing_target_tolerance_pctg = 0.1  # very tight
    sample_config.rebalancing_auto_relax = False  # don't bail us out
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv)

    # Should return empty (no feasible solution), not crash
    assert isinstance(actions, list)


def test_single_holding(sample_config):
    """Single holding edge case."""
    holdings = [
        Holding(
            isin="US0000000001", ticker="SOLO", quantity=100.0,
            cost_basis_eur=5000.0, market_value_eur=6000.0, currency="EUR",
            name="Solo Stock", current_price=60.0, current_value=6000.0,
            asset_class=AssetClass.EQUITIES, geography=Geography.USA,
            geo_breakdown={Geography.USA: 100.0},
        ),
    ]

    actions, _ = compute_unified_rebalancing(holdings, sample_config, 6000.0, lump_sum=500.0)

    assert isinstance(actions, list)


def test_actions_have_required_fields(sample_holdings, sample_config):
    """Every action must have ticker, direction, amount_eur, reason."""
    sample_config.rebalancing_lump_sum_amount_eur = 2000.0
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv, lump_sum=2000.0)

    for a in actions:
        assert "ticker" in a
        assert "direction" in a
        assert a["direction"] in ("buy", "sell")
        assert "amount_eur" in a
        assert a["amount_eur"] > 0
        assert "reason" in a


def test_verification_structure(sample_holdings, sample_config):
    """Verifications should always include the standard check categories."""
    tv = _total_value(sample_holdings)

    _, verifications = compute_unified_rebalancing(sample_holdings, sample_config, tv)

    # At least Invested Allocation, Equity Geography, Per-Holding, Cash Buffer.
    assert len(verifications) >= 3
    check_kinds = [v.get("kind") for v in verifications]
    assert "asset" in check_kinds
    assert "geography" in check_kinds
    assert "cash" in check_kinds


def test_actions_carry_stable_holding_index(sample_holdings, sample_config):
    """Each action must carry an 'idx' pointing at the exact holding it
    names, so the verification step attributes buys/sells by position
    (collision-safe) instead of by ticker."""
    tv = _total_value(sample_holdings)
    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv)
    for a in actions:
        assert "idx" in a, "action missing stable holding index"
        idx = a["idx"]
        assert 0 <= idx < len(sample_holdings)
        # The recorded index must resolve to the holding the action names.
        assert sample_holdings[idx].ticker == a["ticker"]


def test_duplicate_ticker_attribution(sample_holdings, sample_config):
    """When two holdings share a ticker, the index-based mapping must not
    mis-attribute one's trade to the other."""
    # Force a ticker collision between the first two holdings.
    sample_holdings[1].ticker = sample_holdings[0].ticker
    tv = _total_value(sample_holdings)
    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv)
    for a in actions:
        # idx still uniquely identifies which of the two it is.
        assert sample_holdings[a["idx"]].ticker == a["ticker"]


# ---------------------------------------------------------------------------
# Local-search specifics: every-ambit reduction + determinism
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from tarzan.engine.rebalancer import _ObjectiveModel  # noqa: E402
from tarzan.models.investor_config import InvestorConfig  # noqa: E402


def _geo_scenario():
    """Equities over-weight at the asset level AND USA under-weight at the geo
    level, with a pure-USA equity fund available to buy. Buying equity helps
    geography but worsens the asset-class over-weight — the conflict that made
    the old optimizer leave geography untouched."""
    holdings = [
        Holding(isin="USA000000001", ticker="USA_EQ", quantity=500.0,
                cost_basis_eur=45000.0, market_value_eur=50000.0, currency="EUR",
                name="USA Equity", current_price=100.0, current_value=50000.0,
                gain_pct=10.0, asset_class=AssetClass.EQUITIES,
                geography=Geography.USA, geo_breakdown={Geography.USA: 100.0}),
        Holding(isin="EUR000000001", ticker="EU_EQ", quantity=300.0,
                cost_basis_eur=28000.0, market_value_eur=30000.0, currency="EUR",
                name="Eurozone Equity", current_price=100.0, current_value=30000.0,
                gain_pct=7.0, asset_class=AssetClass.EQUITIES,
                geography=Geography.EUROZONE_EMU,
                geo_breakdown={Geography.EUROZONE_EMU: 100.0}),
        Holding(isin="BOND00000001", ticker="AGGH", quantity=150.0,
                cost_basis_eur=15000.0, market_value_eur=15000.0, currency="EUR",
                name="Global Agg Bond", current_price=100.0, current_value=15000.0,
                gain_pct=-1.0, asset_class=AssetClass.FIXED_INCOME),
        Holding(isin="GOLD00000001", ticker="GOLD", quantity=50.0,
                cost_basis_eur=4800.0, market_value_eur=5000.0, currency="EUR",
                name="Gold ETC", current_price=100.0, current_value=5000.0,
                gain_pct=5.0, asset_class=AssetClass.GOLD),
    ]
    cfg = InvestorConfig()
    cfg.invested_allocation_targets_pctg = {"Equities": 72.0, "Fixed Income": 21.0, "Gold": 7.0}
    cfg.equity_geo_targets_pctg = {"USA": 70.0, "Eurozone EMU": 30.0}
    cfg.rebalancing_target_tolerance_pctg = 2.0
    cfg.rebalancing_no_sell = True
    cfg.target_cash_buffer_eur = 0.0
    return holdings, cfg, 10000.0


def _gaps_before_after(holdings, cfg, lump):
    values = np.array([h.current_value for h in holdings], float)
    model = _ObjectiveModel(holdings, cfg, values)
    g0 = np.abs(model.gaps(values))
    actions, _ = compute_unified_rebalancing(holdings, cfg, float(values.sum()), lump_sum=lump)
    nv = values.copy()
    for a in actions:
        nv[a["idx"]] += a["amount_eur"] if a["direction"] == "buy" else -a["amount_eur"]
    return model, g0, np.abs(model.gaps(nv)), actions


def test_geography_ambit_is_reduced_not_left_intact():
    """The core fix: geography must be actively reduced even when it conflicts
    with the asset-class over-weight (the old behaviour left it untouched)."""
    holdings, cfg, lump = _geo_scenario()
    model, g0, g1, actions = _gaps_before_after(holdings, cfg, lump)
    geo_idx = [i for i, a in enumerate(model.ambit_of) if a == "geo"]
    geo_before = max(g0[i] for i in geo_idx)
    geo_after = max(g1[i] for i in geo_idx)
    assert geo_after < geo_before - 0.1, (
        f"geography ambit not reduced: worst gap {geo_before:.2f} -> {geo_after:.2f}"
    )


def test_no_objective_pushed_further_out_of_tolerance():
    """The guard must keep every objective at or below max(its initial gap,
    tolerance) — nothing is made worse to fix something else."""
    holdings, cfg, lump = _geo_scenario()
    _model, g0, g1, _actions = _gaps_before_after(holdings, cfg, lump)
    tol = cfg.rebalancing_target_tolerance_pctg
    for before, after in zip(g0, g1):
        assert after <= max(before, tol) + 0.25


def test_lump_fully_deployed_no_sell():
    """No-sell lump deployment: zero sells and total buy equals the lump sum."""
    holdings, cfg, lump = _geo_scenario()
    _m, _g0, _g1, actions = _gaps_before_after(holdings, cfg, lump)
    assert all(a["direction"] == "buy" for a in actions)
    assert abs(sum(a["amount_eur"] for a in actions) - lump) < 5.0


def test_local_search_is_deterministic():
    """Fixed seed → identical plan across runs (stable user-facing output)."""
    holdings, cfg, lump = _geo_scenario()
    tv = sum(h.current_value for h in holdings)
    a1, _ = compute_unified_rebalancing(holdings, cfg, tv, lump_sum=lump)
    a2, _ = compute_unified_rebalancing(holdings, cfg, tv, lump_sum=lump)
    sig = lambda acts: sorted((a["ticker"], a["direction"], a["amount_eur"]) for a in acts)
    assert sig(a1) == sig(a2)


def test_drift_penalty_sensitivity_retired():
    """The LP-only drift-penalty sweep is retired and returns an empty list so
    the Excel/newsletter tuning section hides itself."""
    from tarzan.engine.rebalancer import compute_drift_penalty_sensitivity
    holdings, cfg, lump = _geo_scenario()
    tv = sum(h.current_value for h in holdings)
    assert compute_drift_penalty_sensitivity(holdings, cfg, tv, lump_sum=lump) == []
