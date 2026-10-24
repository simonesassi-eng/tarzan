"""Session euros use the percentage's source, never today's wealth subtotals."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest

from tarzan.contracts.session_move import DailyReturnStep, paired_session_move
from tarzan.engine.stats import compute_period_return
from tarzan.export.newsletter import render_newsletter
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_hero
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


THURSDAY, FRIDAY = date(2026, 10, 22), date(2026, 10, 23)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    import urllib.request
    import requests
    from curl_cffi.requests import Session

    def blocked(*a, **kw):
        pytest.fail("a session display test attempted network transport")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(Session, "request", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("tarzan.runtime.today", lambda: FRIDAY)


def basket(start, end, *, total=50000.0, invested=0.0, availability="AVAILABLE"):
    m = PortfolioMetrics(
        total_value=total,
        invested_value=invested,
        cash_value=total - invested,
    )
    m.portfolio_history = pd.Series([start, end], index=pd.DatetimeIndex([THURSDAY, FRIDAY]))
    m.performance = {"1d": compute_period_return(m.portfolio_history, "1d")}
    m.history_availability = availability
    m.history_source_basis = "HOLDINGS_PRICE_AGGREGATE"
    m.session_move_evidence = paired_session_move(
        m.portfolio_history, m.performance["1d"], source_basis="HOLDINGS_PRICE_AGGREGATE",
        history_availability=availability,
    )
    return m


def order_nav(start=1000.0, end=1600.0, flow=500.0):
    factor = (end - flow) / start
    m = basket(start, start * factor)
    m.actual_value_series = pd.Series([start, end], index=m.portfolio_history.index)
    m.return_nav_series = m.portfolio_history.copy(deep=True)
    m.external_flows = {FRIDAY: flow}
    m.history_source_basis = "ORDER_FLOW_ADJUSTED_NAV"
    step = DailyReturnStep(FRIDAY, THURSDAY, start, end, flow, factor, end - flow - start, "MEASURED")
    m.session_move_evidence = paired_session_move(
        m.portfolio_history, m.performance["1d"], source_basis="ORDER_FLOW_ADJUSTED_NAV",
        steps=(step,), daily_nav=m.portfolio_history,
    )
    return m


def tile(m):
    hero = _build_hero(_NewsletterContext(
        metrics=m, config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI",
    ))
    return next(t for t in hero["tiles"] if t["label"] == "Session")


def caption(m):
    return re.sub(r"<[^>]+>", "", tile(m)["caption"])


def render(m):
    return render_newsletter(m, InvestorConfig(), ai_summary="")


@pytest.mark.parametrize("total, invested", [(50000.0, 0.0), (100000.0, 96000.0), (1.0, 1.0)])
def test_basket_uses_own_start_and_end_including_priced_cash_equivalents(total, invested):
    m = basket(10000.0, 11000.0, total=total, invested=invested)
    assert caption(m).startswith("+€1,000.00")  # synthetic exact endpoint delta
    assert "historical holdings basket" in caption(m)
    assert tile(m)["session_proof"]["profit"] == "1000.0"
    assert validate_newsletter_semantics(m, {}, render(m)) == ()


@pytest.mark.parametrize("end, expected", [
    (1100.0, "+€100.00"), (900.0, "−€100.00"), (1000.0, "+€0.00"),
    (0.0, "−€1,000.00"),  # synthetic complete-loss endpoint
])
def test_own_euro_delta_handles_gain_loss_zero_and_complete_loss(end, expected):
    m = basket(1000.0, end)
    assert caption(m).startswith(expected)
    assert validate_newsletter_semantics(m, {}, render(m)) == ()


@pytest.mark.parametrize("flow, end, expected", [
    (500.0, 1600.0, "+€100.00"), (5000.0, 6100.0, "+€100.00"),
    (-200.0, 900.0, "+€100.00"), (0.0, 900.0, "−€100.00"),
    (-1100.0, 0.0, "+€100.00"), (500.0, 1500.0, "+€0.00"),
])
def test_order_nav_uses_original_signed_flows_not_today_value_times_return(flow, end, expected):
    m = order_nav(end=end, flow=flow)
    assert caption(m).startswith(expected)
    assert "flow-adjusted NAV" in caption(m)
    assert validate_newsletter_semantics(m, {}, render(m)) == ()
