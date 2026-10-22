"""A usable newer session must supersede a preferred but older venue.

These quotes are synthetic observations. No external data or email is used.
"""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from tarzan.data import current_session
from tarzan.models.holding import Holding


@pytest.fixture(autouse=True)
def live_thursday(monkeypatch):
    monkeypatch.setattr("tarzan.runtime.today", lambda: date(2026, 10, 22))
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.data.justetf.prefetch", lambda isins: None)
    monkeypatch.setattr("tarzan.data.justetf.quote", lambda isin: None)


def quote(symbol, day, price=100.5, previous=100.25, hour=10):
    return {
        "symbol": symbol,
        "price": price,
        "prev_close": previous,
        "currency": "EUR",
        "time": int(datetime(2026, 10, day, hour, tzinfo=timezone.utc).timestamp()),
    }


def test_new_session_supersedes_preferred_older_venue():
    older = quote("THIN.MI", 21)
    current = quote("THIN.DE", 22, price=101, previous=100.5)

    selected = current_session.pick_quote(
        ["THIN.MI", "THIN.DE"],
        {"THIN.MI": older, "THIN.DE": current},
        reference_price=100.5,
    )

    assert selected == current


def test_dated_current_venue_supersedes_undated_primary():
    undated = {"symbol": "THIN.MI", "price": 100.5, "prev_close": 100.25}
    current = quote("THIN.DE", 22, price=101, previous=100.5)

    selected = current_session.pick_quote(
        ["THIN.MI", "THIN.DE"],
        {"THIN.MI": undated, "THIN.DE": current},
        reference_price=100.5,
    )

    assert selected == current


def test_same_session_priority_stays_intact_inside_consensus_window():
    primary = quote("THIN.MI", 22, price=101, previous=100.5)
    sibling = quote("THIN.DE", 22, price=101.1, previous=100.5)

    assert current_session.pick_quote(
        ["THIN.MI", "THIN.DE"],
        {"THIN.MI": primary, "THIN.DE": sibling},
        reference_price=100.5,
    ) == primary


def test_latest_completed_session_remains_usable_without_current_quotes():
    primary = quote("THIN.MI", 21)
    sibling = quote("THIN.DE", 21, price=100.6)

    assert current_session.pick_quote(
        ["THIN.MI", "THIN.DE"],
        {"THIN.MI": primary, "THIN.DE": sibling},
        reference_price=100.5,
    ) == primary


def test_newer_incoherent_quote_cannot_displace_valid_older_evidence():
    primary = quote("THIN.MI", 21)
    corrupt = quote("THIN.DE", 22, price=150, previous=100.5)

    assert current_session.pick_quote(
        ["THIN.MI", "THIN.DE"],
        {"THIN.MI": primary, "THIN.DE": corrupt},
        reference_price=100.5,
    ) == primary


def test_undated_compatibility_quote_remains_usable_when_alone():
    undated = {"symbol": "THIN.MI", "price": 100.5, "prev_close": 100.25}

    assert current_session.pick_quote(
        ["THIN.MI"], {"THIN.MI": undated}, reference_price=100.5,
    ) == undated


def test_application_updates_all_price_rulers_from_new_session(monkeypatch):
    history = pd.Series(
        [100, 100.25, 100.5],
        index=pd.bdate_range("2026-10-19", periods=3),
    )
    holding = Holding(
        isin="XX0000000001", ticker="THIN.MI", quantity=10,
        cost_basis_eur=1000, market_value_eur=1005, currency="EUR",
        instrument_type="ETF", price_currency="EUR",
        price_history=history.copy(), price_history_native=history.copy(),
        price_history_price_only=history.copy(),
        current_price=100.5, current_value=1005,
    )
    quotes = {
        "THIN.MI": quote("THIN.MI", 21),
        "THIN.DE": quote("THIN.DE", 22, price=101, previous=100.5),
    }
    monkeypatch.setattr(
        "tarzan.data.market_quotes.official_quotes",
        lambda symbols: {symbol: quotes[symbol] for symbol in symbols if symbol in quotes},
    )

    assert current_session.apply_to_holdings([holding]) == ("THIN.MI",)
    for series in (
        holding.price_history, holding.price_history_native,
        holding.price_history_price_only,
    ):
        assert series.index[-1] == pd.Timestamp("2026-10-22")
        assert series.iloc[-1] == 101
        assert series.loc["2026-10-21"] == 100.5
    assert holding.current_value == 1010
    assert holding.price_observation_timestamp == datetime(
        2026, 10, 22, 10, tzinfo=timezone.utc,
    )
