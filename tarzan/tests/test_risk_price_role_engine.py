"""The portfolio risk block declares price levels, even for a cheap NAV."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import metrics, stats


@pytest.mark.parametrize("level", [0.00025, 0.25, 100.0, 10000.0])
@pytest.mark.parametrize("timezone", [None, "Europe/Rome", "America/New_York"])
def test_engine_risk_declares_native_price_role_without_scale_heuristics(
    monkeypatch, level, timezone,
):
    rng = np.random.default_rng(112023)
    index = pd.bdate_range("2024-01-05", periods=140, tz=timezone)
    benchmark = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, len(index))), index=index)
    portfolio = benchmark / benchmark.iloc[0] * level
    observed = []
    original = stats._compute_beta_alpha

    def inspect(*args, **kwargs):
        observed.append(kwargs.get("input_kind"))
        return original(*args, **kwargs)

    monkeypatch.setattr(metrics, "_compute_beta_alpha", inspect)
    engine = metrics.MetricsEngine.__new__(metrics.MetricsEngine)
    engine._rf_daily = lambda _: None
    engine._alpha_beta_benchmark = lambda _: SimpleNamespace(history=benchmark)
    context = {"portfolio_history": portfolio}
    engine._risk(context)
    assert observed == ["prices"]
    assert context["risk"]["beta"] == pytest.approx(1.0, abs=1e-12)
    assert context["risk"]["alpha"] == pytest.approx(0.0, abs=1e-10)
