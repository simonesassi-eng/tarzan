"""Render tests: XIRR/TWROR appear in the newsletter only when set."""

from __future__ import annotations

import pandas as pd

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
        # The order path always ships the daily series; the Performance
        # matrix + charts render off these (and the lifetime P&L).
        idx = pd.date_range("2025-12-29", periods=6, freq="W")
        m.pnl_eur = 1000.0
        m.pnl_pct = 20.0
        m.actual_value_series = pd.Series(
            [4800.0, 5000.0, 5200.0, 5100.0, 5600.0, 6000.0], index=idx)
        m.pnl_series = pd.Series([0.0, 60.0, 120.0, 90.0, 250.0, 350.0], index=idx)
        m.unrealized_series = pd.Series([0.0, 50.0, 100.0, 80.0, 200.0, 300.0], index=idx)
        m.portfolio_history = pd.Series(
            [100.0, 100.8, 101.5, 101.2, 102.8, 103.5], index=idx)
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

    def test_html_shows_perf_section_only_when_set(self):
        html_off = render_newsletter(_minimal_metrics(False), _config())
        html_on = render_newsletter(_minimal_metrics(True), _config())
        # The window matrix renders only when the order-derived returns and daily
        # series exist. Its first column header is the anchor: the section
        # heading above it is always present, because the PORTFOLIO section also
        # carries the value chart.
        assert ">Window<" not in html_off
        assert ">Window<" in html_on
        assert "TWROR" in html_on

    def test_no_decorative_em_dash_in_prose(self):
        # "—" is allowed ONLY as the standalone missing-data placeholder in its
        # own table cell. A connective em-dash inside prose (subtitles, captions,
        # footers, the AI note) is the taste-skill anti-slop tell and must not
        # come back (e.g. "Annualized — TWROR"). We scan the RENDERED HTML but
        # require the dash's neighbours to be in the SAME text run (no HTML tag
        # between them) — so a "—" placeholder cell sitting next to a table
        # label (always separated by </td><td>) is correctly ignored.
        import re
        import html as _html
        # Unescape first so entity names (&ldquo;/&rdquo; around a quoted "—"
        # placeholder) become punctuation, not letters, and don't look like a
        # word next to the dash.
        html_out = _html.unescape(render_newsletter(_minimal_metrics(True), _config()))
        # A connective em-dash in prose has a WORD immediately before and after
        # it within the SAME text run (no HTML tag between) — e.g.
        # "capital — above". The standalone "—" placeholder cell is separated
        # from any label by </td><td> tags, so ``[^<>]`` never spans it, and a
        # quoted "—" has curly quotes (non-letters) on both sides.
        offenders = re.findall(r"[A-Za-z)][^<>]{0,2}—[^<>]{0,2}[A-Za-z(]", html_out)
        assert not offenders, f"connective em-dashes in prose: {offenders[:8]}"


def _config():
    from tarzan.models.investor_config import InvestorConfig
    c = InvestorConfig()
    c.invested_allocation_targets_pctg = {"Equities": 100.0}
    return c
