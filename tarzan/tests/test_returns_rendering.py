"""Render tests: XIRR/TWROR appear in Excel + newsletter only when set."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from tarzan.export.excel import generate_excel
from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.models.portfolio import PortfolioMetrics


def _minimal_metrics(with_returns: bool) -> PortfolioMetrics:
    df = pd.DataFrame([{
        "isin": "US0000000001", "ticker": "AAA", "name": "Alpha ETF",
        "asset_class": "Equities", "current_value": 6000.0,
        "cost_basis_eur": 5000.0, "weight_pct": 100.0, "gain_pct": 20.0,
        "quantity": 100.0, "avg_purchase_price": 50.0, "pct_of_class": 100.0,
        "currency": "EUR",
    }])
    m = PortfolioMetrics(
        total_value=6000.0, invested_value=6000.0, cash_value=0.0,
        holdings_df=df,
        allocation_by_class=pd.DataFrame([{"category": "Equities", "weight_pct": 100.0}]),
        performance_full={"1d": 0.1, "1w": 0.5, "ytd": 8.0, "period_used": "1.0Y"},
    )
    if with_returns:
        m.xirr_pct = 11.09
        m.twror_pct = 14.49
        m.twror_annualized_pct = 36.43
        m.returns_coverage_pct = 93.9
        m.returns_provenance = {
            "yfinance": ["US0000000001"], "synthetic": [],
            "carry_flat": ["IT0005542359"], "excluded": [],
        }
    return m


class TestNewsletterReturns:
    def test_returns_block_absent_when_none(self):
        ctx = build_context(_minimal_metrics(with_returns=False), _config())
        assert ctx["performance"]["returns"] is None

    def test_returns_block_present_when_set(self):
        ctx = build_context(_minimal_metrics(with_returns=True), _config())
        rb = ctx["performance"]["returns"]
        assert rb is not None
        assert "11.09%" in rb["xirr"]
        assert "14.49%" in rb["twror"]
        assert rb["fallback_count"] == 1

    def test_html_shows_chips_only_when_set(self):
        html_off = render_newsletter(_minimal_metrics(False), _config())
        html_on = render_newsletter(_minimal_metrics(True), _config())
        assert "money-weighted" not in html_off
        assert "money-weighted" in html_on
        assert "TWROR" in html_on


class TestExcelReturns:
    def test_excel_renders_both_ways(self):
        for with_returns in (False, True):
            m = _minimal_metrics(with_returns)
            with tempfile.TemporaryDirectory() as d:
                path = generate_excel(m, [], _config(), d)
                assert os.path.exists(path)


def _config():
    from tarzan.models.investor_config import InvestorConfig
    c = InvestorConfig()
    c.invested_allocation_targets_pctg = {"Equities": 100.0}
    return c
