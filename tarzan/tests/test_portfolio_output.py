"""Output-contract tests for PortfolioMetrics.

Covers the Phase-1 hardening of the Calculator→Reporting boundary:
non-finite floats must not reach the JSON layer, and a crashed metric
computer must be surfaced rather than silently zeroed.
"""

from __future__ import annotations

import json

from tarzan.models.portfolio import PortfolioMetrics


class TestSummaryDictSanitization:
    def test_nan_and_inf_become_none_and_json_is_strict_valid(self):
        m = PortfolioMetrics()
        m.total_value = 100.0
        m.risk = {"sharpe": float("nan"), "volatility": 12.3, "beta": float("inf")}
        m.performance = {"cagr": float("nan"), "1y": 5.5}
        m.xirr_pct = float("nan")
        m.twror_pct = 8.0
        m.twror_annualized_pct = float("nan")

        s = m.to_summary_dict()

        assert s["risk"]["sharpe"] is None
        assert s["risk"]["beta"] is None
        assert s["risk"]["volatility"] == 12.3
        assert s["performance"]["cagr"] is None
        assert s["performance"]["1y"] == 5.5
        assert s["xirr_pct"] is None
        assert s["twror_annualized_pct"] is None

        # The whole payload must survive a STRICT (allow_nan=False) dump —
        # the exact failure a downstream API/DB consumer would hit.
        json.dumps(s, allow_nan=False)

    def test_finite_values_preserved(self):
        m = PortfolioMetrics()
        m.total_value = 1234.567
        m.risk = {"sharpe": 1.23456789}
        s = m.to_summary_dict()
        assert s["total_value_eur"] == 1234.57
        assert s["risk"]["sharpe"] == round(1.23456789, 6)


class TestDegradedComputersSurfaced:
    def test_failed_computer_is_recorded(self):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        eng = MetricsEngine([], InvestorConfig())

        def boom(ctx):
            raise RuntimeError("kaboom")
        boom.__name__ = "_boom"
        eng._computers = [boom]

        m = eng.compute_all()
        assert "_boom" in m.degraded_computers

    def test_clean_run_has_no_degraded_computers(self):
        from tarzan.engine.metrics import MetricsEngine
        from tarzan.models.investor_config import InvestorConfig

        eng = MetricsEngine([], InvestorConfig())
        eng._computers = [lambda ctx: ctx.update({"ok": True})]
        m = eng.compute_all()
        assert m.degraded_computers == []
