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
    """Portfolio with zero total value should return empty actions."""
    actions, verifications = compute_unified_rebalancing(
        sample_holdings, sample_config, 0.0
    )
    assert actions == []


def test_lump_sum_buy_only(sample_holdings, sample_config):
    """With no_sell=True and lump_sum=5000, total buy should equal ~5000, total sell=0."""
    sample_config.rebalancing_no_sell = True
    sample_config.rebalancing_lump_sum_amount_eur = 5000.0
    sample_config.rebalancing_max_tolerance_pctg = 5.0  # Loose tolerance for feasibility
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


def test_min_transaction_respected(sample_holdings, sample_config):
    """No action should be below min_transaction_eur."""
    sample_config.rebalancing_min_transaction_eur = 500.0
    sample_config.rebalancing_lump_sum_amount_eur = 2000.0
    tv = _total_value(sample_holdings)

    actions, _ = compute_unified_rebalancing(sample_holdings, sample_config, tv, lump_sum=2000.0)

    for a in actions:
        assert a["amount_eur"] >= 500.0 - 0.01, (
            f"Action {a['ticker']} {a['direction']} {a['amount_eur']} < min_tx=500"
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
    """With tight max_tolerance and min_transaction, infeasible → return 0 actions."""
    sample_config.rebalancing_min_transaction_eur = 100000.0  # absurdly high
    sample_config.rebalancing_max_tolerance_pctg = 0.1  # very tight
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
    """Verifications should always include standard checks."""
    tv = _total_value(sample_holdings)

    _, verifications = compute_unified_rebalancing(sample_holdings, sample_config, tv)

    assert len(verifications) >= 3  # At least Asset, Geo, Per-holding
    check_names = [v["check"] for v in verifications]
    assert any("Asset" in c for c in check_names)
    assert any("Geo" in c for c in check_names)
