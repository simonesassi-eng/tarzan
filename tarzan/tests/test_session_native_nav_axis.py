"""The return's original NAV axis is distinct from the raw accounting calendar."""

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.material import material_metrics
from tarzan.contracts.report import ReportView
from tarzan.engine.metrics import MetricsEngine
from tarzan.engine.returns_builder import _trim_carried_tail
from tarzan.engine.stats import to_business_day_series
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.runtime.ledger import Availability
from tarzan.tests.test_hero_session_euro import caption, render
from tarzan.tests.test_session_move_evidence import _loop


THURSDAY = date(2026, 10, 22)
FRIDAY = date(2026, 10, 23)
SATURDAY = date(2026, 10, 24)
SUNDAY = date(2026, 10, 25)
MONDAY = date(2026, 10, 26)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import smtplib
    import socket
    import urllib.request

    import requests
    from curl_cffi.requests import Session

    def forbidden(*args, **kwargs):
        pytest.fail("native NAV axis test attempted transport or delivery")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(Session, "request", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(smtplib, "SMTP", forbidden)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbidden)


def metrics_for(values, *, flows=None, start=THURSDAY, trim=True, availability="AVAILABLE"):
    """Exercise the unchanged dense loop, original trim, and performance producer."""
    nav, actual, steps = _loop(values=values, flows=flows, start=start)
    source_nav = _trim_carried_tail(nav) if trim else nav
    source = SimpleNamespace(
        daily_series=source_nav,
        daily_return_steps=steps,
        history_availability=Availability(availability),
    )
    context = {
        "portfolio_history": to_business_day_series(source_nav),
        "portfolio_history_full": pd.Series(dtype=float),
        "_portfolio_history_basis": "ORDER_FLOW_ADJUSTED_NAV",
        "_order_series": source,
    }
    engine = MetricsEngine.__new__(MetricsEngine)
    engine._performance(context)
    metrics = PortfolioMetrics(
        portfolio_history=context["portfolio_history"],
        performance=context["performance"],
        return_nav_series=context["return_nav_series"],
        session_move_evidence=context["session_move_evidence"],
        actual_value_series=actual,
        external_flows=dict(flows or {}),
        history_source_basis="ORDER_FLOW_ADJUSTED_NAV",
        history_availability=availability,
    )
    assert metrics.return_nav_series is source.daily_series
    return metrics


@pytest.mark.parametrize("values,flows", [
    ([1000.0, 1100.0, 1100.0], {}),
    ([1000.0, 1100.0, 1100.0, 1100.0], {}),
    ([1000.0, 1100.0, 1100.0, 1100.0, 1100.0], {}),
    ([1000.0, 1100.0, 1300.0], {SATURDAY: 200.0}),
    ([1000.0, 1100.0, 3100.0], {SATURDAY: 2000.0}),
    ([1000.0, 1100.0, 900.0], {SATURDAY: -200.0}),
    ([1000.0, 1100.0, 1090.0], {SATURDAY: -10.0}),
    ([1000.0, 1100.0, 1300.0, 1300.0], {SATURDAY: 200.0}),
    ([1000.0, 1100.0, 1100.0, 1300.0], {SUNDAY: 200.0}),
    ([1000.0, 1100.0, 1100.0, 1100.0, 1300.0], {MONDAY: 200.0}),
])
@pytest.mark.parametrize("availability", ["AVAILABLE", "DEGRADED"])
def test_accounting_tail_does_not_change_the_return_native_window(
    values, flows, availability, monkeypatch,
):
    monkeypatch.setattr("tarzan.runtime.today", lambda: THURSDAY + timedelta(days=len(values) - 1))
    metrics = metrics_for(values, flows=flows, availability=availability)
    nav_before = metrics.return_nav_series.copy(deep=True)
    actual_before = metrics.actual_value_series.copy(deep=True)
    flows_before = dict(metrics.external_flows)
    assert metrics.return_nav_series.index[-1].date() == FRIDAY
    assert metrics.actual_value_series.index[-1].date() > FRIDAY
    proof = metrics.session_move_evidence
    assert proof.source_end_date == FRIDAY and proof.profit_eur == 100.0
    assert proof.external_flow_eur == 0.0
    assert caption(metrics).startswith("+€100.00")
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()
    pd.testing.assert_series_equal(metrics.return_nav_series, nav_before, check_exact=True)
    pd.testing.assert_series_equal(metrics.actual_value_series, actual_before, check_exact=True)
    assert metrics.external_flows == flows_before


@pytest.mark.parametrize("values,native_end,profit", [
    ([1000.0, 1100.0, 1122.0], SATURDAY, 122.0),
    ([1000.0, 1100.0, 1122.0, 1144.0], SUNDAY, 144.0),
])
def test_genuine_weekend_nav_is_not_clipped_to_a_business_day_label(
    values, native_end, profit,
):
    metrics = metrics_for(values, trim=False)
    proof = metrics.session_move_evidence
    assert proof.end_date == FRIDAY and proof.source_end_date == native_end
    assert proof.profit_eur == pytest.approx(profit)
    assert f"source {THURSDAY} to {native_end}" in caption(metrics)
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()


def test_monday_uses_the_original_sunday_anchor_and_preserves_weekend_flows():
    metrics = metrics_for(
        [1000.0, 1500.0, 1520.0, 1590.0], start=FRIDAY,
        flows={SATURDAY: 500.0, MONDAY: 50.0}, trim=False,
    )
    proof = metrics.session_move_evidence
    assert proof.anchor_date == FRIDAY and proof.source_anchor_date == SUNDAY
    assert proof.source_end_date == MONDAY
    assert proof.profit_eur == 20.0 and proof.external_flow_eur == 50.0
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()


def test_long_flat_nav_preserves_the_original_trim_limit_instead_of_forcing_friday():
    metrics = metrics_for([1000.0, 1100.0] + [1100.0] * 12)
    assert metrics.return_nav_series.index[-1] == metrics.actual_value_series.index[-1]
    assert metrics.return_nav_series.index[-1].date() > MONDAY
    assert metrics.session_move_evidence.profit_eur == 0.0
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()


@pytest.mark.parametrize("timezone", ["Europe/Rome", "America/New_York", "Asia/Tokyo", "UTC"])
def test_nav_observation_dates_remain_venue_local_not_shifted_to_utc(timezone):
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    metrics.return_nav_series.index = metrics.return_nav_series.index.tz_localize(timezone)
    metrics.actual_value_series.index = metrics.actual_value_series.index.tz_localize(timezone)
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()


def test_nan_nav_tail_is_not_reinterpreted_as_a_native_observation():
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    metrics.return_nav_series = pd.concat([
        metrics.return_nav_series,
        pd.Series([np.nan], index=pd.DatetimeIndex([SATURDAY])),
    ])
    assert validate_newsletter_semantics(metrics, {}, render(metrics)) == ()


@pytest.mark.parametrize("bad_source", [
    None, {}, [1000.0, 1100.0],
    pd.Series([1000.0, 1100.0], index=["2026-10-22", "2026-10-23"]),
    pd.Series([1100.0, 1000.0], index=pd.DatetimeIndex([FRIDAY, THURSDAY])),
    pd.Series([1000.0, 1100.0], index=pd.DatetimeIndex([THURSDAY, THURSDAY])),
    pd.Series([1000.0, 1100.0], index=pd.to_datetime(["2026-10-23T09:00", "2026-10-23T10:00"])),
    pd.Series([1000.0, np.nan], index=pd.DatetimeIndex([THURSDAY, FRIDAY])),
    pd.Series([1000.0, np.inf], index=pd.DatetimeIndex([THURSDAY, FRIDAY])),
    pd.Series([True, False], index=pd.DatetimeIndex([THURSDAY, FRIDAY])),
    pd.Series([1000.0, 1111.0], index=pd.DatetimeIndex([THURSDAY, FRIDAY])),
    pd.Series([1000.0], index=pd.DatetimeIndex([THURSDAY])),
    pd.Series([1000.0, 1100.0], index=pd.DatetimeIndex([THURSDAY, SATURDAY])),
    pd.Series([1000.0, 1100.0, 1200.0], index=pd.DatetimeIndex([THURSDAY, FRIDAY, MONDAY])),
])
def test_missing_corrupt_or_self_relabelled_nav_authority_blocks_delivery(bad_source):
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    metrics.return_nav_series = bad_source
    assert validate_newsletter_semantics(metrics, {}, render(metrics))


@pytest.mark.parametrize("field,false_date", [
    ("source_end_date", SATURDAY),
    ("source_anchor_date", FRIDAY),
    ("end_date", SATURDAY),
    ("anchor_date", THURSDAY - timedelta(days=1)),
])
def test_display_packet_cannot_choose_or_clip_its_own_native_axis(field, false_date):
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    metrics.session_move_evidence = replace(
        metrics.session_move_evidence, **{field: false_date},
    )
    assert validate_newsletter_semantics(metrics, {}, render(metrics))


def test_completed_report_defensively_preserves_the_original_nav_source():
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    view = ReportView.build(metrics, InvestorConfig())
    metrics.return_nav_series.iloc[-1] = 9999.0
    projected, _ = view.project()
    assert projected.return_nav_series.iloc[-1] == 1100.0
    assert validate_newsletter_semantics(projected, {}, render(projected)) == ()
    projected.return_nav_series.iloc[-1] = 8888.0
    second, _ = view.project()
    assert second.return_nav_series.iloc[-1] == 1100.0


def test_nav_authority_is_material_but_does_not_expand_the_external_summary_schema():
    metrics = metrics_for([1000.0, 1100.0, 1100.0])
    source = material_metrics(metrics)["return_nav_series"]
    assert source["index"] == ["2026-10-22T00:00:00", "2026-10-23T00:00:00"]
    assert source["values"] == [1000.0, 1100.0]
    assert "return_nav_series" not in metrics.to_summary_dict()
    assert tuple(PortfolioMetrics.__dataclass_fields__)[-1] == "return_nav_series"
