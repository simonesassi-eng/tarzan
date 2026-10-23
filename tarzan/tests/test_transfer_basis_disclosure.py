"""Rendering must distinguish transfer contributions from recorded cost basis."""

from html import unescape

import pytest

from tarzan.export.newsletter import build_context, render_newsletter
from tarzan.tests.test_returns_rendering import _config, _minimal_metrics
from tarzan.verification.portfolio import portfolio_kpis


@pytest.mark.parametrize("pnl", [0., -100., 1000.])
def test_tracked_returns_explain_transfer_market_value_without_changing_kpis(pnl):
    metrics = _minimal_metrics(True)
    metrics.pnl_eur = pnl
    before = portfolio_kpis(metrics)
    context = build_context(metrics, _config())
    note = unescape(str(context["hero"]["return_basis_note"]))
    assert "Total P&L, TWR and MWR cover the tracked period" in note
    assert "market value on their operation date" in note
    assert "order-price fallback" in note
    assert "Carrying prices remain the cost-basis reference" in note
    assert portfolio_kpis(metrics) == before


def test_holdings_only_report_does_not_claim_a_tracked_transfer_return():
    context = build_context(_minimal_metrics(False), _config())
    assert context["hero"]["return_basis_note"] == ""


def test_holdings_only_template_does_not_print_a_transfer_contribution_claim():
    html = render_newsletter(_minimal_metrics(False), _config(), ai_summary="")
    assert "Securities transfers are contributions" not in html


def test_real_template_prints_the_disclosure_once_and_escapes_the_ampersand():
    html = render_newsletter(_minimal_metrics(True), _config(), ai_summary="")
    assert html.count("Total P&amp;L, TWR and MWR cover the tracked period") == 1
    assert "Total P&amp;amp;L, TWR" not in html
    assert "Carrying prices remain the cost-basis reference" in html
