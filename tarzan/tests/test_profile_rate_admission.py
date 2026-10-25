"""Optional income/cost rates must not turn bad metadata into financial facts."""

from decimal import Decimal
from fractions import Fraction
import math

import numpy as np
import pandas as pd
import pytest

from tarzan.data import enricher
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import Holding


@pytest.mark.parametrize("value", [
    True, np.bool_(True), float("inf"), float("-inf"), np.float64("inf"),
    float("nan"), pd.NA, "not a rate", 10 ** 400, [2.4], {"raw": 2.4},
    pd.Series([2.4]), complex(.024, .01), np.complex128(.024 + .01j),
])
def test_invalid_profile_scalar_is_unavailable_not_an_income_rate(value):
    assert enricher._as_fraction(value) is None


@pytest.mark.parametrize("value,expected", [
    (.021, .021), ("2.4", .024), (np.float64(.0007), .0007),
    (Decimal(".024"), .024), (Fraction(3, 100), .03),
])
def test_valid_profile_rates_keep_existing_fraction_or_percent_rule(value, expected):
    assert enricher._as_fraction(value) == pytest.approx(expected)


@pytest.fixture
def enrich_profile(monkeypatch):
    history = pd.DataFrame(
        {"Close": [100., 101.], "Price Close": [100., 101.]},
        index=pd.to_datetime(["2026-10-22", "2026-10-23"]),
    )
    monkeypatch.setattr(enricher.cfg, "ter_for", lambda *args: None)

    def run(extra):
        info = {
            "quoteType": "ETF", "currency": "EUR", "shortName": "Synthetic",
            "regularMarketPrice": 101., "previousClose": 100., **extra,
        }
        monkeypatch.setattr(enricher, "_fetch_ticker_data", lambda *a, **kw: {
            "info": info, "history": history,
        })
        holding = Holding(
            isin="SYNTHETIC", ticker="PROFILE.MI", quantity=1.,
            cost_basis_eur=100., market_value_eur=100., currency="EUR",
        )
        return enricher._enrich_single(holding)[0]

    return run


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), True, pd.NA, "invalid"])
def test_bad_high_priority_metadata_does_not_hide_valid_lower_priority_rate(
    enrich_profile, invalid,
):
    holding = enrich_profile({
        "annualReportExpenseRatio": invalid, "expenseRatio": .003,
        "yield": invalid, "dividendYield": 2.4,
    })
    assert holding.ter == pytest.approx(.003)
    assert holding.yield_pct == pytest.approx(.024)
    assert holding.current_value == pytest.approx(101.)


def test_valid_metadata_priority_and_curated_ter_override_are_unchanged(
    monkeypatch, enrich_profile,
):
    monkeypatch.setattr(enricher.cfg, "ter_for", lambda *args: .001)
    holding = enrich_profile({
        "annualReportExpenseRatio": .002, "expenseRatio": .003,
        "yield": .03, "dividendYield": 4.,
    })
    assert holding.ter == pytest.approx(.001)
    assert holding.yield_pct == pytest.approx(.03)


def test_invalid_metadata_does_not_contaminate_weighted_income_or_cost(enrich_profile):
    holding = enrich_profile({
        "annualReportExpenseRatio": float("inf"), "yield": float("inf"),
    })
    context = {"holdings_df": pd.DataFrame({
        "weight_pct": [100.], "yield_pct": [holding.yield_pct], "ter": [holding.ter],
    })}
    MetricsEngine._income_costs(None, context)
    assert holding.ter is None and holding.yield_pct is None
    assert math.isfinite(context["weighted_yield"]) and context["weighted_yield"] == 0.
    assert math.isfinite(context["avg_ter"]) and context["avg_ter"] == 0.


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), True, np.bool_(True)])
def test_native_income_consumer_rejects_bad_direct_fractional_fields(invalid):
    context = {"holdings_df": pd.DataFrame({
        "weight_pct": [50., 50.], "yield_pct": [invalid, .024], "ter": [invalid, .002],
    })}
    MetricsEngine._income_costs(None, context)
    assert context["weighted_yield"] == pytest.approx(1.2)
    assert context["avg_ter"] == pytest.approx(.1)


def test_fractional_income_consumer_does_not_normalize_percentages_a_second_time():
    context = {"holdings_df": pd.DataFrame({
        "weight_pct": [100.], "yield_pct": [1.2], "ter": [.002],
    })}
    MetricsEngine._income_costs(None, context)
    assert context["weighted_yield"] == pytest.approx(120.)
    assert context["avg_ter"] == pytest.approx(.2)


@pytest.mark.parametrize("rates,expected", [
    (pd.Series([10 ** 400, .024], dtype=object), 1.2),
    (pd.Series([1 + 2j, .024], dtype=object), 1.2),
    (pd.Series([np.complex128(.03 + .01j), .024], dtype=object), 1.2),
    (pd.Series([True, pd.NA], dtype="boolean"), 0.),
    (pd.Series([float("inf"), pd.NA], dtype="Float64"), 0.),
    (pd.Series([float("inf"), .024], dtype="category"), 1.2),
    (pd.Series([np.array([.03]), .024], dtype=object), 1.2),
    (pd.Series([pd.Series([.03]), .024], dtype=object), 1.2),
    (pd.Series([pd.Timestamp("2026-10-23"), .024], dtype=object), 1.2),
    (pd.Series([".03", ".024"], dtype="string"), 2.7),
    (pd.Series([Decimal(".03"), Fraction(3, 125)], dtype=object), 2.7),
    (pd.Series([-.03, .024], dtype=float), -.3),
])
def test_income_rate_admission_is_scalar_safe_across_pandas_dtypes(rates, expected):
    context = {"holdings_df": pd.DataFrame({
        "weight_pct": [50., 50.], "yield_pct": rates, "ter": rates,
    })}
    MetricsEngine._income_costs(None, context)
    assert context["weighted_yield"] == pytest.approx(expected)
    assert context["avg_ter"] == pytest.approx(expected)
