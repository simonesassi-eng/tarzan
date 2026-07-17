"""Observation-first preservation harness for production-readiness changes.

The helper deliberately strips only newly versioned operational metadata. Tests
compare established financial values, ordering, public fields, and side-effect
intent using deterministic local fixtures.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy

import numpy as np

from tarzan import runtime
from tarzan.engine.metrics import MetricsEngine
from tarzan.engine.rebalancer import _ObjectiveModel
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


_OPERATIONAL_METADATA = {
    "analysis_id",
    "attempt_id",
    "availability",
    "capability",
    "delivery",
    "ephemerality",
    "failures",
    "provenance",
    "publication",
    "telemetry",
}


def canonical_domain_value(value):
    """Return a deterministic financial/consumer comparison representation."""
    if isinstance(value, dict):
        return {
            key: canonical_domain_value(item)
            for key, item in sorted(value.items())
            if key not in _OPERATIONAL_METADATA
        }
    if isinstance(value, (list, tuple)):
        return tuple(canonical_domain_value(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 8)
    return value


# **Validates: Requirements 3.2, 3.6, 3.9**
def test_live_mode_and_versioned_summary_public_fields_remain_stable():
    runtime.reset()
    assert runtime.is_deterministic() is False
    assert re.fullmatch(r"\d{8}_\d{4}", runtime.now_stamp("%Y%m%d_%H%M"))

    metrics = PortfolioMetrics(total_value=123.456, invested_value=100.0, cash_value=23.456)
    first = canonical_domain_value(metrics.to_summary_dict())
    second = canonical_domain_value(deepcopy(metrics).to_summary_dict())
    assert first == second
    assert {"schema_version", "total_value_eur", "invested_value_eur", "cash_value_eur"} <= set(first)


# **Validates: Requirements 3.4, 3.12, 3.15**
def test_notional_exposure_above_one_hundred_is_preserved_without_normalization():
    holding = Holding(
        isin="LEV", ticker="LEV", quantity=1.0, cost_basis_eur=100.0,
        market_value_eur=100.0, current_price=100.0, current_value=100.0,
        currency="EUR", asset_class=AssetClass.EQUITIES,
        class_breakdown={AssetClass.EQUITIES: 90.0, AssetClass.FIXED_INCOME: 60.0},
    )
    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {"Equities": 90.0, "Fixed Income": 60.0}
    config.equity_geo_targets_pctg = {}

    engine = MetricsEngine([holding], config)
    ctx: dict = {}
    engine._valuation(ctx)
    engine._allocations(ctx)
    observed = {
        str(row.category): float(row.weight_pct)
        for row in ctx["allocation_by_class"].itertuples()
    }
    assert observed == {"Equities": 90.0, "Fixed Income": 60.0}
    assert sum(observed.values()) == 150.0


# **Validates: Requirements 3.3**
def test_deterministic_objective_search_inputs_remain_repeatable():
    holding = Holding(
        isin="EQ", ticker="EQ", quantity=10.0, cost_basis_eur=1000.0,
        market_value_eur=1000.0, current_price=100.0, current_value=1000.0,
        currency="EUR", asset_class=AssetClass.EQUITIES,
    )
    config = InvestorConfig()
    config.invested_allocation_targets_pctg = {"Equities": 100.0}
    config.equity_geo_targets_pctg = {}
    values = np.array([1000.0])
    first = _ObjectiveModel([holding], config, values).gaps(values)
    second = _ObjectiveModel([holding], config, values).gaps(values)
    assert np.array_equal(first, second)


# **Validates: Requirements 3.7, 3.11, 3.13, 3.17**
def test_successful_zero_remains_a_real_zero_in_strict_machine_output():
    summary = PortfolioMetrics(total_value=0.0, invested_value=0.0, cash_value=0.0).to_summary_dict()
    assert summary["total_value_eur"] == 0.0
    assert summary["num_holdings"] == 0
    assert summary["num_rebalancing_actions"] == 0
