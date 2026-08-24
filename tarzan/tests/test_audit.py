"""Tests for the rebalancing audit trail and the extracted _perf_series module."""

from __future__ import annotations

import datetime

import pandas as pd

from tarzan.runtime import audit
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig


def _h(isin, ac, value):
    return Holding(isin=isin, ticker=isin, quantity=1.0, cost_basis_eur=value,
                   market_value_eur=value, currency="EUR", asset_class=ac,
                   current_value=value)


class TestRebalancingAudit:
    def test_reset_clears_records(self):
        audit.reset()
        audit.record_rebalancing_plan(
            "p", no_sell=True, total_value=100.0, lump_sum=None,
            config=InvestorConfig(), holdings=[], suggestions=[], verifications=[])
        assert len(audit.records()) == 1
        audit.reset()
        assert audit.records() == []

    def test_record_captures_inputs_and_outputs(self):
        audit.reset()
        cfg = InvestorConfig()
        cfg.rebalancing_target_tolerance_pctg = 2.5
        holdings = [_h("IE00AAA", AssetClass.EQUITIES, 6000.0),
                    _h("IE00BBB", AssetClass.FIXED_INCOME, 4000.0)]
        actions = [{"ticker": "IE00AAA", "direction": "buy", "amount_eur": 500.0}]
        audit.record_rebalancing_plan(
            "Buy only", no_sell=True, total_value=10000.0, lump_sum=500.0,
            config=cfg, holdings=holdings, suggestions=actions,
            verifications=[{"check": "Invested Allocation", "status": "OK"}])
        rec = audit.records()[0]
        assert rec["plan"] == "Buy only"
        assert rec["total_value_eur"] == 10000.0
        assert rec["config"]["no_sell"] is True
        assert rec["config"]["lump_sum_eur"] == 500.0
        assert rec["config"]["target_tolerance_pctg"] == 2.5
        assert len(rec["holdings"]) == 2
        assert rec["holdings"][0]["isin"] == "IE00AAA"
        assert rec["holdings"][0]["value_eur"] == 6000.0
        assert rec["holdings"][0]["asset_class"] == "Equities"
        assert rec["actions"] == actions

    def test_record_never_raises_on_bad_input(self):
        audit.reset()
        # A holding-like object missing attributes must not crash the recorder.
        class Bad:
            pass
        audit.record_rebalancing_plan(
            "p", no_sell=True, total_value=float("nan"), lump_sum=None,
            config=None, holdings=[Bad()], suggestions=None, verifications=None)
        # Best-effort: it either records a degraded entry or skips, but never raises.
        assert isinstance(audit.records(), list)


class TestPerfSeriesExtraction:
    """The math moved out of newsletter.py must be importable from BOTH the
    new module and (re-exported) from newsletter, and behave identically."""

    def test_importable_from_both_modules(self):
        from tarzan.export import _perf_series
        from tarzan.export import newsletter
        for name in ("_window_money_pnl", "_norm_series", "market_snapshot",
                     "_flow_list", "_window_twror", "_geo_benchmark_series",
                     "_perf_window", "_perf_level_series"):
            assert getattr(_perf_series, name) is getattr(newsletter, name), name

    def test_norm_series_dedups_and_naive(self):
        from tarzan.export._perf_series import _norm_series
        idx = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"], utc=True)
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        out = _norm_series(s)
        assert out.index.tz is None
        assert len(out) == 2                 # duplicate collapsed (keep last)
        assert out.iloc[0] == 2.0

    def test_window_money_pnl_basic(self, monkeypatch):
        from tarzan.export._perf_series import _window_money_pnl
        idx = pd.date_range("2025-01-01", periods=40, freq="D")
        pnl = pd.Series(range(40), index=idx, dtype=float)     # +1/day
        actual = pd.Series([1000.0] * 40, index=idx)
        # Windows end on the last SESSION the series observed. This fixture is
        # calendar-daily and ends Sunday 9 Feb, so the window ends on Friday
        # 7 Feb — the same roll a real portfolio NAV gets, since it also carries
        # weekends flat.
        monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2025, 2, 9))
        gain, pct = _window_money_pnl(pnl, actual, "1m")
        # 1M is a calendar month: 7 Feb back to 7 Jan (index 6), so 39 - 6.
        assert gain == 33.0
        assert round(pct, 6) == 3.3           # 33 / 1000 * 100

    def test_window_twror_none_on_short_series(self):
        from tarzan.export._perf_series import _window_twror
        assert _window_twror(None, "1m") is None
        assert _window_twror(pd.Series([100.0]), "1m") is None
