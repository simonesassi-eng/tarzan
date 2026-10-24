"""Independent delivery claims, with synthetic values and no live adapters."""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.session_move import DailyReturnStep, SessionMoveEvidence, paired_session_move
from tarzan.engine.stats import compute_period_return, to_business_day_series
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_hero
from tarzan.export.newsletter._semantic import validate_newsletter_semantics
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan.tests.test_hero_session_euro import (  # noqa: F401
    THURSDAY, FRIDAY, basket, caption, no_network, order_nav, render, tile,
)


def weekend_nav():
    dates = pd.date_range("2026-10-16", periods=4)
    values = [1000.0, 1500.0, 1520.0, 1590.0]
    flows = {date(2026, 10, 17): 500.0, date(2026, 10, 19): 50.0}
    nav = pd.Series([1000.0, 1000.0, 1000.0 * 1520 / 1500, 1000.0 * 1540 / 1500], index=dates)
    m = PortfolioMetrics()
    m.portfolio_history = to_business_day_series(nav)
    m.return_nav_series = nav.copy(deep=True)
    m.actual_value_series = pd.Series(values, index=dates)
    m.external_flows = flows
    m.history_source_basis = "ORDER_FLOW_ADJUSTED_NAV"
    m.performance = {"1d": compute_period_return(m.portfolio_history, "1d")}
    steps = (
        DailyReturnStep(date(2026, 10, 19), date(2026, 10, 18), 1520.0, 1590.0,
                        50.0, 1540 / 1520, 20.0, "MEASURED"),
    )
    m.session_move_evidence = paired_session_move(
        m.portfolio_history, m.performance["1d"], source_basis="ORDER_FLOW_ADJUSTED_NAV",
        steps=steps, daily_nav=nav,
    )
    return m


def test_native_weekend_observations_are_not_relabelled_as_friday_values():
    m = weekend_nav()
    assert caption(m).startswith("+€20.00")
    assert "source 2026-10-18 to 2026-10-19" in caption(m)
    assert validate_newsletter_semantics(m, {}, render(m)) == ()


def test_degraded_source_is_measured_but_visibly_qualified():
    m = basket(1000.0, 1100.0, availability="DEGRADED")
    assert caption(m).startswith("+€100.00")
    assert "estimated historical data" in caption(m)
    assert tile(m)["session_proof"]["availability"] == "DEGRADED"
    html = render(m)
    assert validate_newsletter_semantics(m, {}, html) == ()
    assert validate_newsletter_semantics(m, {}, html.replace("estimated historical data", "measured history"))


@pytest.mark.parametrize("proof", [None, SessionMoveEvidence(), {}, "untrusted"])
def test_legacy_missing_or_unavailable_proof_never_invents_zero_or_a_subtotal_base(proof):
    m = basket(10000.0, 11000.0)
    m.session_move_evidence = proof
    m.pnl_series = pd.Series([0.0, 99999.0], index=m.portfolio_history.index)
    m.actual_value_series = pd.Series([100000.0, 110000.0], index=m.portfolio_history.index)
    assert caption(m).startswith("EUR attribution unavailable")
    assert tile(m)["value"] == "+10.00%"
    assert validate_newsletter_semantics(m, {}, render(m)) == ()


@pytest.mark.parametrize("percent", [None, float("nan"), float("inf"), float("-inf"), True, np.bool_(False), 10 ** 1000])
def test_unavailable_percent_does_not_emit_a_signed_numeric_session_claim(percent):
    m = basket(1000.0, 1100.0)
    m.performance["1d"] = percent
    hero = _build_hero(_NewsletterContext(
        metrics=m, config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI",
    ))
    assert all(t["label"] != "Session" for t in hero["tiles"])
    assert validate_newsletter_semantics(m, {}, "") == ()


@pytest.mark.parametrize("change", [
    {"return_pct": 12.0}, {"profit_eur": float("nan")}, {"profit_eur": True},
    {"external_flow_eur": float("inf")}, {"profit_eur": 10 ** 1000},
    {"anchor_date": None}, {"source_end_date": THURSDAY}, {"source_basis": "ORDER_SPARSE_VALUATION"},
])
def test_invalid_paired_source_never_reconstructs_an_alternative_euro_estimate(change):
    m = basket(1000.0, 1100.0)
    m.session_move_evidence = replace(m.session_move_evidence, **change)
    assert caption(m).startswith("EUR attribution unavailable")
    assert validate_newsletter_semantics(m, {}, render(m))


@pytest.mark.parametrize("change", [
    {"profit_eur": 101.0}, {"external_flow_eur": 501.0}, {"factor": 1.2},
    {"starting_value_eur": 999.0}, {"ending_value_eur": 1599.0},
    {"state": "SUSPENDED"}, {"start_date": date(2026, 10, 21)},
])
def test_independent_guard_recomputes_source_facts_not_renderer_rounding(change):
    m = order_nav()
    m.session_move_evidence = replace(
        m.session_move_evidence,
        steps=(replace(m.session_move_evidence.steps[0], **change),),
    )
    assert validate_newsletter_semantics(m, {}, render(m))


@pytest.mark.parametrize("tamper", [
    ('data-tarzan-session-profit="100.0"', 'data-tarzan-session-profit="101.0"'),
    ('data-tarzan-session-return="10.000000000000009"', 'data-tarzan-session-return="11.0"'),
    ("+€100.00", "+€999.00"), ("+10.00%", "+11.00%"),
    ("flow-adjusted NAV", "bank cash return"),
    ('data-tarzan-session-source-anchor="2026-10-22"', 'data-tarzan-session-source-anchor="2026-10-21"'),
    ('data-tarzan-session-availability="AVAILABLE"', 'data-tarzan-session-availability="UNAVAILABLE"'),
    ('data-tarzan-session-caption="true"', 'data-foreign-session-caption="true"'),
])
def test_guard_reads_the_delivered_markup_and_rejects_amount_interval_or_label_tampering(tamper):
    m = order_nav()
    original = render(m)
    assert validate_newsletter_semantics(m, {}, original) == ()
    assert tamper[0] in original
    assert validate_newsletter_semantics(m, {}, original.replace(*tamper))


def test_unavailable_attribution_cannot_be_relabelled_as_an_observed_zero():
    m = basket(1000.0, 1100.0)
    m.session_move_evidence = None
    html = render(m)
    assert validate_newsletter_semantics(m, {}, html.replace("EUR attribution unavailable", "+€0.00"))


def test_original_values_and_flow_tape_are_independent_of_the_step_packet():
    m = order_nav()
    original = render(m)
    m.actual_value_series.iloc[0] = 999.0
    assert validate_newsletter_semantics(m, {}, original)
    m = order_nav()
    m.external_flows[FRIDAY] = 499.0
    assert validate_newsletter_semantics(m, {}, render(m))


@pytest.mark.parametrize("false_note", [
    "source 2026-10-16 to 2026-10-19", "source 2026-10-18 to 2026-10-20", "",
])
def test_visible_native_interval_is_independent_of_hidden_attributes(false_note):
    m = weekend_nav()
    html = render(m)
    assert validate_newsletter_semantics(m, {}, html) == ()
    tampered = html.replace("source 2026-10-18 to 2026-10-19", false_note)
    assert tampered != html
    assert validate_newsletter_semantics(m, {}, tampered)


def test_visible_percent_cannot_contain_a_second_contradictory_token():
    m = order_nav()
    html = render(m)
    tampered = html.replace("+10.00%", "−99.00% +10.00%")
    assert validate_newsletter_semantics(m, {}, tampered)


def test_packet_cannot_upgrade_the_original_degraded_history():
    m = basket(1000.0, 1100.0, availability="DEGRADED")
    m.session_move_evidence = replace(m.session_move_evidence, availability="AVAILABLE")
    assert validate_newsletter_semantics(m, {}, render(m))


def test_nav_index_units_cannot_be_relabelled_as_euro_basket_values():
    m = order_nav()
    m.portfolio_history *= 10.0
    m.return_nav_series *= 10.0
    m.session_move_evidence = paired_session_move(
        m.portfolio_history, m.performance["1d"], source_basis="ORDER_FLOW_ADJUSTED_NAV",
        steps=m.session_move_evidence.steps, daily_nav=m.portfolio_history,
    )
    assert m.session_move_evidence.profit_eur == 100.0
    assert validate_newsletter_semantics(m, {}, render(m)) == ()
    m.session_move_evidence = paired_session_move(
        m.portfolio_history, m.performance["1d"], source_basis="HOLDINGS_PRICE_AGGREGATE",
    )
    assert m.session_move_evidence.profit_eur == 1000.0
    assert validate_newsletter_semantics(m, {}, render(m))


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("invented", [
    " · +€999.00",
    " · EUR 999.00",
    " · EUR −999.00",
    " · 999.00 EUR",
    " · +999.00 EUR",
    " · −999.00 EUR",
    " · 999,00 EUR",
])
def test_session_caption_cannot_append_a_second_or_unavailable_money_claim(unavailable, invented):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    html = render(m)
    original = caption(m)
    assert original in html
    assert validate_newsletter_semantics(m, {}, html) == ()
    assert validate_newsletter_semantics(m, {}, html.replace(original, original + invented))


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("invented", [
    "01 Jan session, close to close",
    "22 Oct session, close to close",
    "24 Oct session, close to close",
    "market open",
    "market closed",
    "close-to-close",
    "23 Oct session, close to close · 100% of the book priced today",
    "23 Oct session, close to close · 50% of the book priced today",
    "23 Oct session, close to close · market open",
    "23 Oct session, close to close · 23 Oct session, close to close",
    "23 Oct session, close to close · source 2026-10-01 to 2026-10-23",
])
def test_closed_session_caption_cannot_invent_dates_market_state_or_coverage(unavailable, invented):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    html = render(m)
    original = caption(m)
    tail = "23 Oct session, close to close"
    assert original.endswith(tail)
    assert validate_newsletter_semantics(m, {}, html) == ()
    tampered = html.replace(original, original.removesuffix(tail) + invented)
    assert tampered != html
    assert validate_newsletter_semantics(m, {}, tampered)


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("market_open", [None, False, True])
@pytest.mark.parametrize("coverage", [None, 0.0, 1.4, 49.5, 97.5, 99.49, 99.5, 100.0])
def test_intraday_caption_uses_original_flags_and_existing_coverage_rounding(
    unavailable, market_open, coverage,
):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    m.performance.update(
        {"1d_intraday": True, "market_open": market_open, "1d_coverage_pct": coverage},
    )
    html = render(m)
    original = caption(m)
    state = "market closed" if market_open is False else "market open"
    tail = state
    if coverage is not None and coverage < 99.5:
        tail += f" · {coverage:.0f}% of the book priced today"
    assert original.endswith(tail)
    assert validate_newsletter_semantics(m, {}, html) == ()

    opposite = "market open" if state == "market closed" else "market closed"
    tampered = html.replace(original, original.removesuffix(tail) + opposite)
    assert tampered != html
    assert validate_newsletter_semantics(m, {}, tampered)


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("invented", [
    "market open",
    "market open · 99% of the book priced today",
    "market open · 100% of the book priced today",
    "market open · 50% of the book priced today · 75% of the book priced today",
    "market open · 50% of the book priced today · 50% of the book priced today",
    "23 Oct session, close to close · 50% of the book priced today",
])
def test_intraday_caption_cannot_hide_or_duplicate_partial_coverage(unavailable, invented):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    m.performance.update(
        {"1d_intraday": True, "market_open": True, "1d_coverage_pct": 50.0},
    )
    html = render(m)
    original = caption(m)
    tail = "market open · 50% of the book priced today"
    assert original.endswith(tail)
    assert validate_newsletter_semantics(m, {}, html) == ()
    tampered = html.replace(original, original.removesuffix(tail) + invented)
    assert tampered != html
    assert validate_newsletter_semantics(m, {}, tampered)


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("coverage", [
    True, False, np.bool_(True), float("nan"), float("inf"), float("-inf"),
    -0.1, 100.1, 10 ** 1000,
])
def test_intraday_unknown_or_invalid_coverage_is_not_a_numeric_source_fact(unavailable, coverage):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    m.performance.update(
        {"1d_intraday": True, "market_open": True, "1d_coverage_pct": coverage},
    )
    assert validate_newsletter_semantics(m, {}, render(m))


@pytest.mark.parametrize("unavailable", [False, True])
def test_delivered_caption_is_rechecked_against_original_market_facts(unavailable):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    m.performance.update(
        {"1d_intraday": True, "market_open": True, "1d_coverage_pct": 50.0},
    )
    html = render(m)
    assert validate_newsletter_semantics(m, {}, html) == ()
    m.performance["market_open"] = False
    assert validate_newsletter_semantics(m, {}, html)
    m.performance["market_open"] = True
    m.performance["1d_coverage_pct"] = 75.0
    assert validate_newsletter_semantics(m, {}, html)
    m.performance["1d_coverage_pct"] = 50.0
    m.performance["1d_intraday"] = False
    assert validate_newsletter_semantics(m, {}, html)


@pytest.mark.parametrize("unavailable", [False, True])
def test_caption_date_comes_from_the_original_history_not_the_wall_clock(unavailable, monkeypatch):
    m = order_nav()
    if unavailable:
        m.session_move_evidence = None
    monkeypatch.setattr("tarzan.runtime.today", lambda: date(2026, 10, 26))
    html = render(m)
    assert caption(m).endswith("23 Oct session, close to close")
    assert validate_newsletter_semantics(m, {}, html) == ()


def test_unavailable_legacy_history_does_not_invent_a_session_date():
    m = order_nav()
    m.session_move_evidence = None
    m.portfolio_history = pd.Series(dtype=float)
    html = render(m)
    original = caption(m)
    assert original == "EUR attribution unavailable · close-to-close"
    assert validate_newsletter_semantics(m, {}, html) == ()
    assert validate_newsletter_semantics(
        m, {}, html.replace(original, original + " · 23 Oct session, close to close"),
    )


def test_caption_text_can_be_styled_without_changing_the_native_claim():
    m = order_nav()
    html = render(m)
    original = caption(m)
    styled = original.replace("23 Oct", "<strong>23 Oct</strong>")
    assert styled != original
    assert validate_newsletter_semantics(m, {}, html.replace(original, styled)) == ()
