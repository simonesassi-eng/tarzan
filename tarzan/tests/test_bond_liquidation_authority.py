"""A clean liquidation ruler must not become an identity or loyalty ruler."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from tarzan import runtime
from tarzan.data import bond_history as bonds, bond_pricing_authority as authority
from tarzan.runtime.session import RunContext, RunMode
from tarzan.tests.test_official_bond_history import carrier, packet


CAPTURED = datetime(2026, 10, 23, 15, 30, tzinfo=timezone.utc)
START, END = date(2026, 1, 5), date(2026, 1, 9)


def entry(link, *, currency="EUR"):
    native = packet(currency)
    native["payload"]["transco"]["code"] = link.market_isin
    return bonds.parse_packet(native, link.market_isin)


@pytest.fixture
def live(monkeypatch):
    context = RunContext("liquidation-pricing", RunMode.LIVE, None, CAPTURED)
    monkeypatch.setattr(runtime, "context", lambda: context)
    return context


@pytest.mark.parametrize("link", authority._LINKS)
def test_exact_documented_pair_is_directional_currency_specific_and_dated(link):
    assert authority.liquidation_price_link(
        link.held_isin, currency="EUR", captured_at=CAPTURED,
    ) is link
    for isin, currency, captured in (
        (link.market_isin, "EUR", CAPTURED),
        (link.held_isin[:-1] + "0", "EUR", CAPTURED),
        (link.held_isin, "USD", CAPTURED),
        (link.held_isin, "EUR", link.verified_at - timedelta(microseconds=1)),
        (link.held_isin, "EUR", CAPTURED.replace(tzinfo=None)),
        (link.held_isin, "EUR", None),
    ):
        assert authority.liquidation_price_link(isin, currency=currency, captured_at=captured) is None


@pytest.mark.parametrize("link", authority._LINKS)
def test_derived_ruler_keeps_native_isin_and_policy_separate_without_rescaling(link):
    raw = entry(link)
    original = deepcopy(raw)
    prices = bonds._liquidation_prices(raw, link, as_of=END, captured_at=CAPTURED)
    assert raw == original
    assert raw["isin"] == link.market_isin
    assert prices.attrs[bonds.ISIN_ATTR] == link.held_isin
    assert prices.attrs[bonds.SOURCE_ISIN_ATTR] == link.market_isin
    assert prices.attrs[bonds.PRICING_LINK_ATTR] == link.policy_id
    assert bonds.price_on(prices, link.held_isin, END) == pytest.approx(1.02)
    assert bonds.price_on(prices, link.market_isin, END) is None
    assert bonds.price_on(prices, link.held_isin, date(2026, 1, 4)) is None
    assert bonds.price_on(prices, link.held_isin, date(2026, 1, 17)) is None


@pytest.mark.parametrize("link", authority._LINKS)
def test_a_window_before_all_native_prices_remains_unavailable_without_raising(link):
    raw = entry(link)
    original = deepcopy(raw)
    prices = bonds._liquidation_prices(
        raw, link, as_of=START - timedelta(days=1), captured_at=CAPTURED,
    )
    assert prices.empty
    assert bonds.price_on(prices, link.held_isin, START) is None
    assert raw == original


@pytest.mark.parametrize("change", [
    {bonds.PRICING_LINK_ATTR: "undeclared-pair"},
    {bonds.SOURCE_ISIN_ATTR: "IT0005542359"},
    {bonds.AUTHORITY_TIME_ATTR: "2026-10-22T12:00:00+00:00"},
    {bonds.AUTHORITY_TIME_ATTR: "not-a-date"},
    {bonds.AUTHORITY_TIME_ATTR: None},
    {bonds.BASIS_ATTR: "NATIVE_CLEAN_PER_100"},
])
def test_wrong_or_future_price_link_authority_fails_closed(change):
    link = authority._LINKS[0]
    prices = bonds._liquidation_prices(entry(link), link, as_of=END, captured_at=CAPTURED)
    prices.attrs.update(change)
    assert bonds.price_on(prices, link.held_isin, END) is None


def test_stripping_the_link_cannot_turn_native_other_isin_into_exact_market_evidence():
    link = authority._LINKS[0]
    prices = bonds._liquidation_prices(entry(link), link, as_of=END, captured_at=CAPTURED)
    del prices.attrs[bonds.PRICING_LINK_ATTR]
    assert bonds.price_on(prices, link.held_isin, END) is None


def test_a_series_authorized_in_a_later_run_is_not_evidence_in_an_older_run():
    from tarzan.runtime.session import RunSession, activate_session

    link = authority._LINKS[0]
    prices = bonds._liquidation_prices(entry(link), link, as_of=END, captured_at=CAPTURED)
    context = RunContext("earlier-capture", RunMode.LIVE, None, link.verified_at)
    with activate_session(RunSession(context, {}, ledger=None)):
        assert bonds.price_on(prices, link.held_isin, END) is None


@pytest.mark.parametrize("link", authority._LINKS)
def test_pre_issue_and_post_maturity_dates_are_not_liquidation_quotes(link):
    prices = bonds._liquidation_prices(entry(link), link, as_of=END, captured_at=CAPTURED)
    assert bonds.price_on(prices, link.held_isin, link.first_trading_day - timedelta(days=1)) is None
    assert bonds.price_on(prices, link.held_isin, link.maturity) is None


@pytest.mark.parametrize("link", authority._LINKS)
def test_hydration_uses_exact_quotable_isin_and_never_changes_orders_or_current_facts(
    monkeypatch, live, link,
):
    calls = []
    raw = entry(link)
    original = deepcopy(raw)
    holding = carrier(isin=link.held_isin)
    frozen = deepcopy(holding)
    monkeypatch.setattr(bonds, "raw_history", lambda isin, *window: calls.append((isin, window)) or raw)
    assert bonds.hydrate_historical_bonds(
        [holding], {link.held_isin: (START, END)}, as_of=END,
    ) == (link.held_isin,)
    assert calls == [(link.market_isin, (START, END))]
    assert holding.isin == frozen.isin
    assert holding.ticker == frozen.ticker
    assert holding.current_ticker == frozen.current_ticker
    assert holding.current_price == frozen.current_price
    assert holding.current_value == frozen.current_value
    assert holding.cost_basis_eur == frozen.cost_basis_eur
    assert holding.price_history is None
    assert raw == original
    assert bonds.price_on(holding.price_history_price_only, link.held_isin, END) == pytest.approx(1.02)


def test_wrong_native_isin_or_currency_cannot_be_declared_equivalent(monkeypatch, live):
    link = authority._LINKS[0]
    for raw in (entry(authority._LINKS[1]), entry(link, currency="USD")):
        holding = carrier(isin=link.held_isin)
        monkeypatch.setattr(bonds, "raw_history", lambda *args: raw)
        assert bonds.hydrate_historical_bonds(
            [holding], {link.held_isin: (START, END)}, as_of=END,
        ) == ()
        assert holding.price_history_price_only is None


def test_non_bond_current_or_wrong_currency_carriers_do_not_use_a_loyalty_link(monkeypatch, live):
    link = authority._LINKS[0]
    holdings = [
        carrier(isin=link.held_isin, instrument_type="ETF"),
        carrier(isin=link.held_isin, is_historical_only=False),
    ]
    monkeypatch.setattr(bonds, "raw_history", lambda *args: pytest.fail("out-of-scope price link"))
    assert bonds.hydrate_historical_bonds(holdings, {link.held_isin: (START, END)}, as_of=END) == ()


def test_declared_order_equivalence_cannot_authorize_an_undeclared_price_link():
    from tarzan.engine.returns_builder import _instrument_identity_by_isin
    from tarzan.tests.test_official_bond_history import order

    orders = [
        order(isin="IT0005358806", instrument_equivalence_group="issuer-proof-missing"),
        order(isin="IT0005542359", instrument_equivalence_group="issuer-proof-missing"),
    ]
    assert len(set(_instrument_identity_by_isin(orders).values())) == 1
    assert authority.liquidation_price_link(
        "IT0005358806", currency="EUR", captured_at=CAPTURED,
    ) is None


def test_actual_coupon_cashflow_and_order_tax_basis_are_not_rewritten_by_the_ruler(monkeypatch, live):
    from tarzan.engine.returns_builder import build_order_derived_series
    from tarzan.models.order import OrderType
    from tarzan.tests.test_official_bond_history import order

    link = authority._LINKS[0]
    holding = carrier(isin=link.held_isin, price_history_price_only=bonds._liquidation_prices(
        entry(link), link, as_of=END, captured_at=CAPTURED,
    ))
    orders = [
        order(isin=link.held_isin),
        order(date(2026, 1, 7), isin=link.held_isin, kind=OrderType.COUPON,
              quantity=0, price_native=None, gross_eur=10, net_eur=10),
    ]
    original = deepcopy(orders)
    result = build_order_derived_series(orders, {link.held_isin: holding}, today=END)
    assert result.valuations[-1][1] == pytest.approx(1020.)
    assert result.external_flows[date(2026, 1, 7)] == -10.
    assert result.pnl_series.iloc[-1] == pytest.approx(50.)
    assert set(result.provenance["borsa_italiana"]) == {link.held_isin}
    assert orders == original
