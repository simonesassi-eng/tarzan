"""Different provider roots need explicit, causal, same-ISIN authority."""

from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import current_session, market_quotes, price_cache
from tarzan.models.holding import Holding
from tarzan.runtime.session import RunContext, RunMode


ISIN = "LU0290357507"
CAPTURED = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)


def identity_profile():
    return {
        "isin": ISIN, "kind": "ETF", "status": "VERIFIED",
        "source": "OpenFIGI", "confidence": "HIGH",
        "observed_at": "2026-10-21T13:28:22+00:00",
        "provenance": {"provider": "OpenFIGI", "idType": "ID_ISIN"},
        "tickers": ["X15E", "DBXF"],
        "identifiers": {"shareClassFIGI": ["BBG001SR7X43"]},
    }


@pytest.fixture
def verified_identity(monkeypatch):
    context = RunContext(
        attempt_id="verified-venue-test", mode=RunMode.LIVE,
        effective_date=None, captured_at=CAPTURED,
    )
    profile = identity_profile()
    reads = []

    def load(isin, *, as_of=None, allow_stale=False):
        reads.append((isin, as_of, allow_stale))
        return deepcopy(profile)

    monkeypatch.setattr(runtime, "context", lambda: context)
    monkeypatch.setattr(runtime, "today", lambda: context.analysis_date)
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: context.allows_live_transport)
    monkeypatch.setattr(price_cache, "load_instrument_profile", load)
    monkeypatch.setattr("tarzan.data.enricher._openfigi_raw",
                        lambda *args: pytest.fail("alias selection must not open transport"))
    return profile, reads


def test_different_verified_root_precedes_guessed_sibling_names(verified_identity):
    siblings = market_quotes._sibling_symbols("X15E.MI")

    assert siblings[0] == "DBXF.DE"
    assert len(siblings) <= 5
    assert len(set(siblings)) == len(siblings)
    assert "X15E.MI" not in siblings


def test_curated_alias_is_bidirectional(verified_identity):
    assert market_quotes._sibling_symbols("DBXF.DE")[0] == "X15E.MI"


def test_profile_is_read_on_the_owning_capture_boundary(verified_identity):
    _, reads = verified_identity
    market_quotes._sibling_symbols("X15E.MI")

    assert reads == [(ISIN, CAPTURED, True)]


@pytest.mark.parametrize("changes", [
    {"isin": "LU0000000001"},
    {"kind": "STOCK"},
    {"status": "CONFLICTING"},
    {"source": "manual"},
    {"confidence": "LOW"},
    {"tickers": ["X15E"]},
    {"tickers": None},
    {"identifiers": {"shareClassFIGI": ["BBG000WRONG1"]}},
    {"identifiers": {"shareClassFIGI": ["BBG001SR7X43", "BBG000WRONG1"]}},
    {"identifiers": None},
    {"observed_at": "2026-10-23T00:00:00+00:00"},
    {"observed_at": "2026-10-21T00:00:00"},
    {"observed_at": "not-an-observation"},
    {"provenance": {"provider": "OpenFIGI", "idType": "TICKER"}},
    {"provenance": None},
])
def test_alias_cannot_override_conflicting_or_noncausal_identity(
    verified_identity, changes,
):
    profile, _ = verified_identity
    profile.update(changes)

    assert "DBXF.DE" not in market_quotes._sibling_symbols("X15E.MI")


def test_alias_is_not_visible_before_its_own_verification(monkeypatch, verified_identity):
    previous = RunContext(
        attempt_id="prior-capture", mode=RunMode.LIVE, effective_date=None,
        captured_at=datetime(2026, 10, 21, 15, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(runtime, "context", lambda: previous)

    assert "DBXF.DE" not in market_quotes._sibling_symbols("X15E.MI")


def test_pinned_analysis_does_not_import_a_new_live_listing(monkeypatch, verified_identity):
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)

    assert "DBXF.DE" not in market_quotes._sibling_symbols("X15E.MI")


def test_unknown_root_keeps_existing_bounded_candidates(verified_identity):
    _, reads = verified_identity
    assert market_quotes._sibling_symbols("UNLISTED.MI") == [
        "UNLISTED.DE", "UNLISTED.PA", "UNLISTED.AS", "UNLISTED.F",
    ]
    assert reads == []


def test_missing_profile_keeps_existing_candidates(monkeypatch, verified_identity):
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *args, **kwargs: None)

    assert market_quotes._sibling_symbols("X15E.MI") == [
        "X15E.DE", "X15E.PA", "X15E.AS", "X15E.F",
    ]


def test_optional_profile_read_failure_cannot_break_quote_candidates(
    monkeypatch, verified_identity,
):
    def unavailable(*args, **kwargs):
        raise OSError("unavailable local profile")

    monkeypatch.setattr(price_cache, "load_instrument_profile", unavailable)

    assert "DBXF.DE" not in market_quotes._sibling_symbols("X15E.MI")


def test_identity_decision_is_memoized_only_inside_its_own_run(verified_identity):
    from tarzan.runtime.session import RunSession, activate_session

    profile, reads = verified_identity
    context = runtime.context()
    with activate_session(RunSession(context, {}, ledger=None)):
        assert market_quotes._sibling_symbols("X15E.MI")[0] == "DBXF.DE"
        assert market_quotes._sibling_symbols("DBXF.DE")[0] == "X15E.MI"
    assert len(reads) == 1

    profile["status"] = "CONFLICTING"
    with activate_session(RunSession(context, {}, ledger=None)):
        assert "DBXF.DE" not in market_quotes._sibling_symbols("X15E.MI")
    assert len(reads) == 2


def synthetic_holding(isin=ISIN):
    history = pd.Series(
        [250, 250.25, 250.5], index=pd.bdate_range("2026-10-19", periods=3),
    )
    return Holding(
        isin=isin, ticker="X15E.MI", quantity=10,
        cost_basis_eur=2500, market_value_eur=2505, currency="EUR",
        instrument_type="ETF", price_currency="EUR",
        price_history=history.copy(), price_history_native=history.copy(),
        price_history_price_only=history.copy(),
        current_price=250.5, current_value=2505,
        current_ticker="X15E.MI",
    )


def quotes():
    return {
        "X15E.MI": {
            "price": 250.5, "prev_close": 250.25, "currency": "EUR",
            "time": int(datetime(2026, 10, 21, 14, tzinfo=timezone.utc).timestamp()),
        },
        "DBXF.DE": {
            "price": 251, "prev_close": 250.5, "currency": "EUR",
            "time": int(datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp()),
        },
    }


def mock_quotes(monkeypatch, mapping):
    requested = []

    def fetch(symbols):
        requested.extend(symbols)
        return {symbol: deepcopy(mapping[symbol]) for symbol in symbols if symbol in mapping}

    monkeypatch.setattr(market_quotes, "official_quotes", fetch)
    monkeypatch.setattr("tarzan.data.justetf.prefetch", lambda *args: None)
    monkeypatch.setattr("tarzan.data.justetf.quote", lambda *args: None)
    return requested


def test_quote_stamp_requests_verified_alias_and_retains_actual_source(
    monkeypatch, verified_identity,
):
    holding = synthetic_holding()
    mapping = quotes()
    original = deepcopy(mapping)
    requested = mock_quotes(monkeypatch, mapping)

    current_session.apply_to_holdings([holding])

    assert "DBXF.DE" in requested
    assert holding.price_history.index[-1] == pd.Timestamp("2026-10-22")
    assert holding.current_price == 251
    assert holding.ticker == holding.current_ticker == "X15E.MI"
    assert holding.data_source == "yfinance:DBXF.DE:current_session"
    assert mapping == original


def test_holding_with_another_isin_cannot_borrow_curated_alias(
    monkeypatch, verified_identity,
):
    holding = synthetic_holding(isin="LU0000000001")
    requested = mock_quotes(monkeypatch, quotes())

    current_session.apply_to_holdings([holding])

    assert "DBXF.DE" not in requested
    assert holding.current_price == 250.5
    assert holding.price_history.index[-1] == pd.Timestamp("2026-10-21")


def test_conflicting_isins_for_one_ticker_do_not_request_curated_alias(
    monkeypatch, verified_identity,
):
    holdings = [synthetic_holding(), synthetic_holding(isin="LU0000000001")]
    requested = mock_quotes(monkeypatch, quotes())

    current_session.apply_to_holdings(holdings)

    assert "DBXF.DE" not in requested
    assert all(holding.current_price == 250.5 for holding in holdings)


def test_missing_holding_isin_cannot_borrow_curated_alias(monkeypatch, verified_identity):
    requested = mock_quotes(monkeypatch, quotes())
    holding = synthetic_holding(isin="")

    current_session.apply_to_holdings([holding])

    assert "DBXF.DE" not in requested
    assert holding.current_price == 250.5


def test_alias_with_wrong_quote_currency_cannot_be_stamped(
    monkeypatch, verified_identity,
):
    holding = synthetic_holding()
    mapping = quotes()
    mapping["DBXF.DE"]["currency"] = "USD"
    mock_quotes(monkeypatch, mapping)

    current_session.apply_to_holdings([holding])

    assert holding.current_price == 250.5
    assert holding.price_history.index[-1] == pd.Timestamp("2026-10-21")


def test_untrusted_provider_symbol_does_not_replace_actual_selected_key(monkeypatch):
    monkeypatch.setattr(runtime, "today", lambda: CAPTURED.date())
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.data.justetf.prefetch", lambda *args: None)
    monkeypatch.setattr("tarzan.data.justetf.quote", lambda *args: None)
    holding = synthetic_holding(isin="XX0000000001")
    holding.ticker = "THIN.MI"
    holding.current_ticker = "THIN.MI"
    mapping = {"THIN.DE": dict(quotes()["DBXF.DE"], symbol="WRONG.MI")}
    mock_quotes(monkeypatch, mapping)

    current_session.apply_to_holdings([holding])

    assert holding.ticker == holding.current_ticker == "THIN.MI"
    assert holding.data_source == "yfinance:THIN.DE:current_session"
