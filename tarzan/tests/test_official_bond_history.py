"""Exact-ISIN, unadjusted history, one unit conversion, and causal evidence."""

from copy import deepcopy
from datetime import date, datetime, timezone
from html import escape
import json
from io import BytesIO

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import bond_history as bonds, price_cache
from tarzan.instruments.registry import InstrumentKind
from tarzan.models.holding import Holding
from tarzan.models.order import Order, OrderType
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session
from tarzan.runtime import transport
from tarzan.verification.tape import UnrecordedInput


ISIN = "IT0005358806"
OTHER = "IT0005542359"
CAPTURED = datetime(2026, 10, 22, 13, tzinfo=timezone.utc)
START, END = date(2026, 1, 5), date(2026, 1, 9)


def packet(currency="EUR"):
    return {
        "exchange": "XMIL", "fetched_at": "2026-10-22T12:40:00+00:00",
        "payload": {
            "status": 0, "entityID": "euronext", "period": "5Y",
            "adjustement": False, "addDayLastPrice": False,
            "transco": {"code": ISIN, "codification": "ISIN", "exchCode": "XMIL"},
            "history": {
                "currency": currency, "instrType": "BOND", "accuracy": 2,
                "historyDt": [
                    {"dt": f"2026010{day}", "closePx": 93 + day, "lastPx": 999}
                    for day in range(5, 10)
                ],
            },
        },
    }


def entry(currency="EUR"):
    result = bonds.parse_packet(packet(currency), ISIN)
    assert result is not None
    return result


def negative():
    raw = packet()
    raw["payload"].update(status=2304, transco=None, history={"historyDt": []})
    return bonds.parse_packet(raw, ISIN)


def carrier(**changes):
    values = {
        "isin": ISIN, "ticker": "SYNTHETIC", "quantity": 0,
        "cost_basis_eur": 0, "market_value_eur": 0, "currency": "EUR",
        "instrument_type": "BOND", "is_historical_only": True,
        "current_price": 1.23, "current_value": 0, "current_ticker": "SYNTHETIC",
    }
    values.update(changes)
    return Holding(**values)


@pytest.fixture
def live(monkeypatch):
    context = RunContext(
        attempt_id="official-bond-test", mode=RunMode.LIVE,
        effective_date=None, captured_at=CAPTURED,
    )
    monkeypatch.setattr(runtime, "context", lambda: context)
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: True)
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *args, **kwargs: {
        "isin": ISIN, "kind": "BOND", "status": "VERIFIED", "confidence": "HIGH",
        "source": "OpenFIGI", "observed_at": "2026-10-21T12:00:00+00:00",
        "provenance": {"provider": "OpenFIGI", "idType": "ID_ISIN"},
    })
    budget = transport.ProviderBudget(transport.ProviderTransportSettings.configured())
    monkeypatch.setattr(transport, "current_budget", lambda: budget)
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: None)
    writes = []
    monkeypatch.setattr(price_cache, "store_timed", lambda *args: writes.append(deepcopy(args)))
    return context, budget, writes


def test_native_header_and_daily_close_are_preserved_without_mutating_input():
    raw = packet()
    original = deepcopy(raw)
    result = bonds.parse_packet(raw, ISIN)

    assert result["prices"][0] == {"date": "2026-01-05", "close": 98.0}
    assert result["basis"] == "NATIVE_CLEAN_PER_100"
    assert bonds.cache_entry_valid(result, ISIN)
    assert raw == original


@pytest.mark.parametrize("path,value", [
    (("exchange",), "OTHER"),
    (("fetched_at",), "2026-10-22T12:40:00"),
    (("payload", "status"), True),
    (("payload", "entityID"), "unknown"),
    (("payload", "period"), "1Y"),
    (("payload", "adjustement"), True),
    (("payload", "addDayLastPrice"), True),
    (("payload", "transco", "code"), OTHER),
    (("payload", "transco", "codification"), "TICKER"),
    (("payload", "transco", "exchCode"), "XETR"),
    (("payload", "history", "instrType"), "ETF"),
    (("payload", "history", "currency"), ""),
    (("payload", "history", "currency"), "eur"),
])
def test_wrong_native_identity_or_return_basis_cannot_become_prices(path, value):
    raw = packet()
    target = raw
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    assert bonds.parse_packet(raw, ISIN) is None


@pytest.mark.parametrize("invalid", [
    None, False, 0, -1, "100", float("nan"), float("inf"), 10 ** 400,
])
def test_invalid_price_rows_are_not_observations(invalid):
    raw = packet()
    raw["payload"]["history"]["historyDt"][0]["closePx"] = invalid
    result = bonds.parse_packet(raw, ISIN)
    assert result["prices"][0]["date"] == "2026-01-06"


def test_conflicting_duplicate_day_is_rejected_instead_of_arbitrarily_selected():
    raw = packet()
    points = raw["payload"]["history"]["historyDt"]
    points.append({**points[0], "closePx": 123.0})
    assert bonds.parse_packet(raw, ISIN) is None


def test_same_day_and_future_points_are_not_completed_daily_closes():
    raw = packet()
    raw["payload"]["history"]["historyDt"].extend([
        {"dt": "20261022", "closePx": 103}, {"dt": "20261023", "closePx": 104},
        {"dt": "invalid-date", "closePx": 105},
    ])
    result = bonds.parse_packet(raw, ISIN)
    assert len(result["prices"]) == 5


@pytest.mark.parametrize("dates", [
    ["20260105", "20260205", "20260305"],
    ["20260105", "20260112", "20260119"],
])
def test_coarse_bucket_closes_cannot_be_imported_as_daily_history(dates):
    raw = packet()
    raw["payload"]["history"]["historyDt"] = [
        {"dt": day, "closePx": 100} for day in dates
    ]
    assert bonds.parse_packet(raw, ISIN) is None


def test_real_negative_receipt_is_distinct_from_transport_failure():
    result = negative()
    assert result["state"] == "UNAVAILABLE"
    assert bonds.cache_entry_valid(result, ISIN)
    for status in (500, 401, None, "2304"):
        raw = packet()
        raw["payload"]["status"] = status
        assert bonds.parse_packet(raw, ISIN) is None
    raw = packet()
    raw["payload"].update(status=2304, entityID="unknown")
    assert bonds.parse_packet(raw, ISIN) is None


@pytest.mark.parametrize("change", [
    {"Authorization": "private"},
    {"isin": OTHER},
    {"basis": "EUR_PER_NOMINAL"},
    {"currency": "zar"},
    {"fetched_at": "not-observed"},
    {"prices": [{"date": "2026-10-22", "close": 100}]},
])
def test_cached_schema_rejects_extra_credentials_wrong_units_and_bad_dates(change):
    value = entry()
    value.update(change)
    assert not bonds.cache_entry_valid(value, ISIN)


def test_eur_prices_are_divided_by_100_exactly_once():
    prices = bonds.eur_per_nominal(entry(), as_of=END)
    assert prices.iloc[-1] == pytest.approx(1.02)
    assert bonds.price_on(prices, ISIN, END) == pytest.approx(1.02)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 4)) is None
    assert bonds.price_on(prices, OTHER, END) is None
    assert bonds.price_on(prices, ISIN, date(2026, 1, 16)) == pytest.approx(1.02)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 17)) is None


@pytest.mark.parametrize("currency,rate", [("ZAR", 0.05), ("USD", 0.9)])
def test_foreign_daily_fx_labels_and_per_100_units_are_causal(currency, rate):
    fx = pd.Series(
        [rate], index=pd.DatetimeIndex(["2026-01-06"], tz="Europe/Rome"),
    )
    prices = bonds.eur_per_nominal(entry(currency), as_of=END, fx=fx)
    assert prices.index[0] == pd.Timestamp("2026-01-06")
    assert prices.iloc[-1] == pytest.approx(102 * rate / 100)
    assert bonds.price_on(prices, ISIN, START) is None  # no FX backfill
    assert bonds.price_on(prices, ISIN, date(2026, 1, 13)) is not None
    assert bonds.price_on(prices, ISIN, date(2026, 1, 14)) is None  # FX itself is stale


@pytest.mark.parametrize("currency,rate", [("ZAR", 0.05), ("USD", 0.9)])
def test_new_fx_revalues_last_native_close_even_without_a_security_trade(currency, rate):
    raw = entry(currency)
    raw["prices"] = raw["prices"][::2]  # Jan 5, 7, 9: no bond print on Jan 6/8
    fx = pd.Series(
        [rate, rate * 1.01, rate * 1.02, rate * 1.03, rate * 1.04],
        index=pd.date_range("2026-01-05", periods=5, tz="Europe/London"),
    )
    prices = bonds.eur_per_nominal(raw, as_of=END, fx=fx)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 6)) == pytest.approx(98 * rate * 1.01 / 100)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 8)) == pytest.approx(100 * rate * 1.03 / 100)
    assert prices.attrs[bonds.OBSERVATIONS_ATTR]["2026-01-06"] == "2026-01-05"
    assert prices.attrs[bonds.OBSERVATIONS_ATTR]["2026-01-08"] == "2026-01-07"


def test_fresh_fx_cannot_hide_an_overage_native_bond_price():
    fx = pd.Series([0.05] * 16, index=pd.date_range("2026-01-05", periods=16))
    prices = bonds.eur_per_nominal(entry("ZAR"), as_of=date(2026, 1, 20), fx=fx)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 16)) == pytest.approx(102 * 0.05 / 100)
    assert bonds.price_on(prices, ISIN, date(2026, 1, 17)) is None


def test_fx_dates_before_first_bond_print_and_after_analysis_are_not_backward_evidence():
    raw = entry("ZAR")
    raw["prices"] = raw["prices"][2:]  # first native close Jan 7
    fx = pd.Series([0.05] * 5 + [0.99], index=pd.date_range("2026-01-05", periods=6))
    prices = bonds.eur_per_nominal(raw, as_of=date(2026, 1, 8), fx=fx)
    assert prices.index[0] == pd.Timestamp("2026-01-07")
    assert prices.index[-1] == pd.Timestamp("2026-01-08")
    assert bonds.price_on(prices, ISIN, date(2026, 1, 6)) is None
    assert bonds.price_on(prices, ISIN, date(2026, 1, 8)) == pytest.approx(101 * 0.05 / 100)


def test_analysis_before_all_native_prices_is_empty_even_when_fx_exists():
    fx = pd.Series([0.05], index=pd.to_datetime(["2026-01-02"]))
    assert bonds.eur_per_nominal(entry("ZAR"), as_of=date(2026, 1, 4), fx=fx).empty


def test_missing_stale_invalid_or_future_fx_is_never_a_one_to_one_conversion():
    for fx in (
        pd.Series(dtype=float),
        pd.Series([0, -1, float("nan"), float("inf")],
                  index=pd.date_range("2026-01-05", periods=4)),
        pd.Series([0.9], index=pd.to_datetime(["2025-12-01"])),
        pd.Series([0.9], index=pd.to_datetime(["2026-01-12"])),
    ):
        assert bonds.eur_per_nominal(entry("USD"), as_of=END, fx=fx).empty


def test_missing_wrong_or_future_observation_metadata_fails_closed():
    for change in (
        {bonds.BASIS_ATTR: "NATIVE_CLEAN_PER_100"},
        {bonds.SOURCE_ATTR: "unknown"},
        {bonds.OBSERVATIONS_ATTR: {}},
        {bonds.OBSERVATIONS_ATTR: {"2026-01-09": "2026-01-10"}},
    ):
        prices = bonds.eur_per_nominal(entry(), as_of=END)
        prices.attrs.update(change)
        assert bonds.price_on(prices, ISIN, date(2026, 1, 12)) is None


def test_complete_cached_closed_history_needs_no_provider_or_identity_io(monkeypatch, live):
    value = entry()
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: deepcopy(value))
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: pytest.fail("cached native transport"))
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *args, **kwargs: pytest.fail("identity IO"))

    assert bonds.raw_history(ISIN, START, END) == value
    assert live[2] == []


def test_future_cache_receipt_is_not_visible_at_capture_boundary(monkeypatch, live):
    value = entry()
    value["fetched_at"] = "2026-10-23T00:00:00+00:00"
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: value)
    monkeypatch.setattr(runtime, "allows_live_transport", lambda: False)
    assert bonds.raw_history(ISIN, START, END) is None


def test_recent_negative_receipt_prevents_repeated_native_discovery(monkeypatch, live):
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: negative())
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: pytest.fail("recent negative transport"))
    assert bonds.raw_history(ISIN, START, END) is None
    assert live[2] == []


def test_expired_negative_can_be_refreshed_without_caching_transport_errors(monkeypatch, live):
    old = negative()
    old["fetched_at"] = "2026-10-20T00:00:00+00:00"
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: old)
    calls = []

    def failed(isin):
        calls.append(isin)
        raise TimeoutError("private native request")

    monkeypatch.setattr(bonds, "_request_raw", failed)
    with activate_session(RunSession(live[0], {}, ledger=None)):
        assert bonds.raw_history(ISIN, START, END) is None
    assert calls == [ISIN]
    assert live[2] == []


def test_partial_cache_merges_older_head_without_relabelling_observation_dates(monkeypatch, live):
    old = entry()
    old["prices"] = old["prices"][:3]
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: deepcopy(old))
    new = packet()
    new["payload"]["history"]["historyDt"] = new["payload"]["history"]["historyDt"][2:]
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: deepcopy(new))
    with activate_session(RunSession(live[0], {}, ledger=None)):
        result = bonds.raw_history(ISIN, START, END)
    assert result["prices"] == entry()["prices"]
    assert len(live[2]) == 1
    assert result["prices"][0]["date"] == "2026-01-05"


def test_currency_contradiction_cannot_merge_histories_or_erase_existing_cache(monkeypatch, live):
    old = entry()
    monkeypatch.setattr(price_cache, "load_timed", lambda *args, **kwargs: deepcopy(old))
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: packet("USD"))
    with activate_session(RunSession(live[0], {}, ledger=None)):
        assert bonds.raw_history(ISIN, START, date(2026, 1, 12)) == old
    assert live[2] == []


@pytest.mark.parametrize("change", [
    {"isin": OTHER}, {"kind": "ETF"}, {"status": "AMBIGUOUS"},
    {"confidence": "LOW"}, {"source": "manual"},
    {"observed_at": "2026-10-23T00:00:00+00:00"},
    {"provenance": {"provider": "OpenFIGI", "idType": "TICKER"}},
])
def test_unverified_or_future_native_identity_does_not_open_transport(monkeypatch, live, change):
    profile = price_cache.load_instrument_profile(ISIN)
    profile.update(change)
    monkeypatch.setattr(price_cache, "load_instrument_profile", lambda *args, **kwargs: profile)
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: pytest.fail("unauthorized transport"))
    with activate_session(RunSession(live[0], {}, ledger=None)):
        assert bonds.raw_history(ISIN, START, END) is None


def test_raw_decisions_are_shared_inside_one_run_only(monkeypatch, live):
    calls = []
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or packet())
    for _ in range(2):
        with activate_session(RunSession(live[0], {}, ledger=None)):
            assert bonds.raw_history(ISIN, START, END) is not None
            assert bonds.raw_history(ISIN, START, END) is not None
    assert calls == [ISIN, ISIN]


def test_hydration_adds_only_exact_historical_bond_price_only_series(monkeypatch, live):
    original = carrier()
    others = [
        carrier(is_historical_only=False), carrier(instrument_type="ETF"),
        carrier(isin=OTHER), carrier(instrument_type=None),
    ]
    monkeypatch.setattr(bonds, "raw_history", lambda *args: entry())
    before = (original.ticker, original.current_ticker, original.current_price, original.current_value)
    assert bonds.hydrate_historical_bonds([original, *others], {ISIN: (START, END)}, as_of=END) == (ISIN,)
    assert (original.ticker, original.current_ticker, original.current_price, original.current_value) == before
    assert original.price_history is None
    assert original.price_history_price_only.iloc[-1] == pytest.approx(1.02)
    assert all(holding.price_history_price_only is None for holding in others)


def test_existing_market_history_is_not_replaced_even_when_partial(monkeypatch, live):
    history = pd.Series([1.1], index=pd.to_datetime(["2026-01-06"]))
    holding = carrier(price_history_price_only=history)
    monkeypatch.setattr(bonds, "raw_history", lambda *args: pytest.fail("replacement transport"))
    assert bonds.hydrate_historical_bonds([holding], {ISIN: (START, END)}, as_of=END) == ()
    assert holding.price_history_price_only is history


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0])
def test_wholly_unusable_existing_tape_does_not_block_verified_market_history(
    monkeypatch, live, invalid,
):
    old = pd.Series([invalid], index=pd.to_datetime(["2026-01-06"]))
    holding = carrier(price_history_price_only=old)
    monkeypatch.setattr(bonds, "raw_history", lambda *args: entry())
    assert bonds.hydrate_historical_bonds([holding], {ISIN: (START, END)}, as_of=END) == (ISIN,)
    assert holding.price_history_price_only is not old
    assert holding.price_history_price_only.iloc[-1] == pytest.approx(1.02)


def test_currency_mismatch_and_optional_failure_leave_fallback_untouched(monkeypatch, live):
    holding = carrier(currency="USD")
    monkeypatch.setattr(bonds, "raw_history", lambda *args: entry())
    assert bonds.hydrate_historical_bonds([holding], {ISIN: (START, END)}, as_of=END) == ()
    assert holding.price_history_price_only is None

    def unavailable(*args):
        raise RuntimeError("private provider details")

    monkeypatch.setattr(bonds, "raw_history", unavailable)
    assert bonds.hydrate_historical_bonds([carrier()], {ISIN: (START, END)}, as_of=END) == ()


def test_unrecorded_frozen_input_cannot_be_hidden_by_optional_fallback(monkeypatch, live):
    def missing(*args):
        raise UnrecordedInput("missing native receipt")

    monkeypatch.setattr(bonds, "raw_history", missing)
    with pytest.raises(UnrecordedInput):
        bonds.hydrate_historical_bonds([carrier()], {ISIN: (START, END)}, as_of=END)


def test_public_access_descriptor_is_exact_origin_and_never_in_repr_or_market_payload():
    html = (
        f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
        'url="https://grafici.borsaitaliana.it/api/" token="anonymous-public-view">'
    )
    parser = bonds._ViewParser(ISIN)
    parser.feed(html)
    assert len(parser.views) == 1
    assert "anonymous-public-view" not in repr(parser.views)
    raw = packet()["payload"]
    raw.update(token="anonymous-public-view", Authorization="private", error={"cookie": "private"})
    raw["history"]["historyDt"][0]["cookie"] = "private"
    clean = bonds._public_market_payload(raw)
    assert "anonymous-public-view" not in json.dumps(clean)
    assert "private" not in json.dumps(clean)


@pytest.mark.parametrize("replace", [
    ("XMIL", "XETR"),
    (ISIN, OTHER),
    ("https://grafici.borsaitaliana.it/api/", "https://example.invalid/api/"),
    ("anonymous-public-view", "header&#13;&#10;injection"),
])
def test_public_view_rejects_wrong_identity_origin_and_header_injection(replace):
    html = (
        f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
        'url="https://grafici.borsaitaliana.it/api/" token="anonymous-public-view">'
    )
    parser = bonds._ViewParser(ISIN)
    parser.feed(html.replace(*replace))
    assert parser.views == []


def test_native_request_follows_publisher_descriptor_only_and_returns_no_token(monkeypatch):
    calls = []

    def read(opener, url, *, headers=None, limit=None):
        calls.append((url, headers, limit))
        if "interactive-chart/" in url:
            return (
                f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
                'url="https://grafici.borsaitaliana.it/api/" token="anonymous-public-view">'
            ).encode()
        return json.dumps(packet()["payload"]).encode()

    monkeypatch.setattr(bonds, "_read_body", read)
    result = bonds._request_raw(ISIN)
    assert len(calls) == 2
    assert "adjustment=false" in calls[1][0] and "add-last-price=false" in calls[1][0]
    assert calls[1][1]["Authorization"] == "Bearer anonymous-public-view"
    assert "anonymous-public-view" not in json.dumps(result)
    assert bonds._NoRedirect().redirect_request(None, None, 302, None, None, "https://example.invalid") is None


@pytest.mark.parametrize("location,token", [
    ("header", "anonymous-public-view"),
    ("nested", "anonymous-public-view"),
    ("header", 'anonymous"public-view'),
])
def test_provider_echo_of_public_access_descriptor_cannot_enter_raw_market_tape(
    monkeypatch, location, token,
):
    def read(opener, url, **kwargs):
        if "interactive-chart/" in url:
            return (
                f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
                f'url="https://grafici.borsaitaliana.it/api/" token="{escape(token, quote=True)}">'
            ).encode()
        raw = packet()["payload"]
        if location == "header":
            raw["transco"]["code"] = token
        else:
            raw["history"]["historyDt"][0]["closePx"] = {token: "unexpected source field"}
        return json.dumps(raw).encode()

    monkeypatch.setattr(bonds, "_read_body", read)
    assert bonds._request_raw(ISIN) is None


def test_ambiguous_public_views_do_not_open_price_endpoint(monkeypatch):
    html = (
        f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
        'url="https://grafici.borsaitaliana.it/api/" token="anonymous">'
    ).encode() * 2
    calls = []
    monkeypatch.setattr(bonds, "_read_body", lambda *args, **kwargs: calls.append(args[1]) or html)
    assert bonds._request_raw(ISIN) is None
    assert len(calls) == 1


def test_duplicate_attributes_and_non_ascii_header_controls_are_not_access_authority():
    for extra in (
        f' code="{OTHER}"',
        ' token="another-token"',
        ' token="control&#0;character"',
    ):
        html = (
            f'<chart-allinone code="{ISIN}" exchcode="XMIL" '
            f'url="https://grafici.borsaitaliana.it/api/" token="anonymous"{extra}>'
        )
        parser = bonds._ViewParser(ISIN)
        parser.feed(html)
        assert parser.views == []


def test_native_response_limit_closes_the_response_and_passes_a_bounded_timeout(monkeypatch):
    response = BytesIO(b"oversized")
    calls = []

    class Opener:
        def open(self, request, *, timeout):
            calls.append(timeout)
            return response

    monkeypatch.setattr(transport, "provider_call",
                        lambda provider, producer, *, timeout: producer(timeout))
    with pytest.raises(ValueError, match="byte limit"):
        bonds._read_body(Opener(), "https://example.invalid", limit=3)
    assert calls == [8]
    assert response.closed


def test_exhausted_optional_budget_cannot_start_a_new_history_request(monkeypatch, live):
    elapsed = [0.0]
    budget = transport.ProviderBudget(
        transport.ProviderTransportSettings.configured(), clock=lambda: elapsed[0],
    )
    elapsed[0] = budget.settings.run_budget_seconds - budget.settings.publication_reserve_seconds
    monkeypatch.setattr(transport, "current_budget", lambda: budget)
    monkeypatch.setattr(bonds, "_request_raw", lambda *args: pytest.fail("publication reserve was consumed"))
    with activate_session(RunSession(live[0], {}, ledger=None)), budget.optional_work("test"):
        assert bonds.raw_history(ISIN, START, END) is None


def test_unsorted_or_duplicate_market_index_cannot_be_binary_searched_as_causal_evidence():
    prices = bonds.eur_per_nominal(entry(), as_of=END)
    assert bonds.price_on(prices.iloc[::-1], ISIN, END) is None
    duplicated = pd.concat([prices, prices])
    duplicated.attrs.update(prices.attrs)
    assert bonds.price_on(duplicated, ISIN, END) is None


def test_source_adapter_is_not_visible_before_its_own_verification(monkeypatch, live):
    context = RunContext(
        attempt_id="before-verified-source", mode=RunMode.LIVE, effective_date=None,
        captured_at=datetime(2026, 10, 21, 12, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(runtime, "context", lambda: context)
    monkeypatch.setattr(bonds, "raw_history", lambda *args: pytest.fail("future source was imported"))
    assert bonds.hydrate_historical_bonds([carrier()], {ISIN: (START, END)}, as_of=END) == ()


def order(day=START, *, isin=ISIN, kind=OrderType.TRANSFER_IN, quantity=1000, **changes):
    values = {
        "date": day, "trade_date": day, "type": kind, "isin": isin,
        "name": "Synthetic bond", "ticker": "SYNTHETIC", "quantity": quantity,
        "currency": "EUR", "price_native": 98.0, "fx_rate": 1.0,
        "gross_eur": 980.0, "fees_eur": 0.0, "net_eur": 0.0,
        "instrument_kind": InstrumentKind.BOND,
    }
    values.update(changes)
    return Order(**values)


def test_resolver_records_official_source_and_does_not_rescale_a_second_time():
    from tarzan.engine.returns_builder import PriceResolver, build_order_derived_series

    holding = carrier(price_history_price_only=bonds.eur_per_nominal(entry(), as_of=END))
    orders = [order()]
    resolver = PriceResolver(orders, {ISIN: holding}, today=END)
    assert resolver.price_on(ISIN, END) == (1.02, "borsa_italiana")
    result = build_order_derived_series(orders, {ISIN: holding}, today=END)
    assert result.valuations[0][1] == pytest.approx(980.0)
    assert result.valuations[-1][1] == pytest.approx(1020.0)
    assert set(result.provenance["borsa_italiana"]) == {ISIN}
    assert result.provenance["yfinance"] == []


def test_official_price_only_history_keeps_coupons_separate_from_price_return():
    from tarzan.engine.returns_builder import build_order_derived_series

    holding = carrier(price_history_price_only=bonds.eur_per_nominal(entry(), as_of=END))
    orders = [
        order(),
        order(date(2026, 1, 7), kind=OrderType.COUPON, quantity=0,
              price_native=None, gross_eur=10, net_eur=10),
    ]
    result = build_order_derived_series(orders, {ISIN: holding}, today=END)
    assert result.valuations[-1][1] == pytest.approx(1020.0)  # securities, without paid-out cash
    assert result.external_flows[date(2026, 1, 7)] == -10.0
    assert (date(2026, 1, 7), 10.0) in result.xirr_cashflows
    assert result.pnl_series.iloc[-1] == pytest.approx(50.0)  # 40 price move + 10 coupon


@pytest.mark.parametrize("change", [
    {bonds.BASIS_ATTR: "NATIVE_CLEAN_PER_100"},
    {bonds.ISIN_ATTR: OTHER},
    {bonds.SOURCE_ATTR: "unknown"},
])
def test_invalid_official_metadata_cannot_fall_through_as_yahoo_prices(change):
    from tarzan.engine.returns_builder import PriceResolver

    prices = bonds.eur_per_nominal(entry(), as_of=END)
    prices.attrs.update(change)
    resolver = PriceResolver([order()], {ISIN: carrier(price_history_price_only=prices)}, today=END)
    assert resolver.price_on(ISIN, END) == (98.0, "carry_flat")


def test_stale_official_history_returns_to_explicit_order_fallback():
    from tarzan.engine.returns_builder import PriceResolver

    prices = bonds.eur_per_nominal(entry(), as_of=END)
    resolver = PriceResolver([order()], {ISIN: carrier(price_history_price_only=prices)},
                             today=date(2026, 1, 20))
    assert resolver.price_on(ISIN, date(2026, 1, 20)) == (98.0, "carry_flat")


def test_required_windows_follow_explicit_economic_identity_but_do_not_borrow_prices():
    from tarzan.engine.returns_builder import required_historical_price_windows

    first = order(instrument_equivalence_group="documented-pair")
    sale = order(END, isin=OTHER, kind=OrderType.SELL, quantity=-1000,
                 instrument_equivalence_group="documented-pair")
    windows = required_historical_price_windows([first, sale], as_of=date(2026, 10, 22))
    assert windows == {ISIN: (START, END), OTHER: (START, END)}
    # Each source still authorizes only its own exact ISIN.
    assert bonds.parse_packet(packet(), OTHER) is None


def test_ungrouped_identifiers_and_future_orders_cannot_extend_closed_history():
    from tarzan.engine.returns_builder import required_historical_price_windows

    sale = order(END, kind=OrderType.SELL, quantity=-1000)
    unrelated = order(date(2026, 1, 7), isin=OTHER)
    future = order(date(2026, 10, 23), instrument_equivalence_group="future-authority")
    as_of = date(2026, 10, 22)
    windows = required_historical_price_windows([order(), sale, unrelated, future], as_of=as_of)
    assert windows[ISIN] == (START, END)
    assert windows[OTHER] == (date(2026, 1, 7), as_of)


def test_frozen_port_records_raw_market_fields_and_replays_without_native_access(monkeypatch):
    from tarzan.verification.tape import ProviderTape

    calls = []
    monkeypatch.setattr(bonds, "_request_raw", lambda isin: calls.append(isin) or packet())
    with ProviderTape() as capture:
        original = bonds._request_raw(ISIN)
    assert calls == [ISIN]
    assert len(capture.records) == 1
    record = next(iter(capture.records.values()))
    assert record["operation"] == "borsa.bond_history"
    assert "Authorization" not in json.dumps(record)
    with ProviderTape(records=capture.records, replay=True) as replay:
        assert bonds._request_raw(ISIN) == original
        with pytest.raises(UnrecordedInput):
            bonds._request_raw(OTHER)
    assert calls == [ISIN]
    assert replay.network_attempts == []


def test_encrypted_cache_roundtrip_preserves_public_prices_and_native_receipt_time(
    monkeypatch, tmp_path,
):
    from tarzan.data import protected_cache

    key = b"synthetic-key-is-not-a-secret-00"
    private, staging = tmp_path / "private", tmp_path / "sealed"
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(private))
    price_cache.store_timed(bonds.NAMESPACE, ISIN, entry())
    original = (private / "timed" / f"{bonds.NAMESPACE}.json").read_bytes()
    assert protected_cache.export_snapshot(staging, key) == 1
    payload = (staging / protected_cache.FILENAME).read_bytes()
    assert ISIN.encode() not in payload and b"prices" not in payload
    restored = tmp_path / "restored"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(restored))
    assert protected_cache.import_snapshot(staging, key) == 1
    restored_file = restored / "timed" / f"{bonds.NAMESPACE}.json"
    assert json.loads(restored_file.read_bytes()) == json.loads(original)
    assert restored_file.stat().st_mode & 0o777 == 0o600
    assert price_cache.load_timed(bonds.NAMESPACE, ISIN, ttl_days=36525) == entry()


@pytest.mark.parametrize("case", [
    "authorization", "manual-unit", "extra-input", "bad-timestamp", "overflow-timestamp",
])
def test_new_cache_namespace_cannot_export_credentials_manual_data_or_customer_columns(
    monkeypatch, tmp_path, case,
):
    from tarzan.data import protected_cache

    private, staging = tmp_path / "private", tmp_path / "sealed"
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(private))
    value = entry()
    if case == "authorization":
        value["Authorization"] = "private"
    elif case == "manual-unit":
        value["basis"] = "manual"
    elif case == "extra-input":
        value["prices"][0]["quantity"] = 12345
    price_cache.store_timed(bonds.NAMESPACE, ISIN, value)
    if case in {"bad-timestamp", "overflow-timestamp"}:
        price_cache._atomic_write_json(
            private / "timed" / f"{bonds.NAMESPACE}.json", bonds.NAMESPACE,
            {ISIN: {"ts": True if case == "bad-timestamp" else 10 ** 400, "value": value}},
        )
    price_cache.store_resolution(ISIN, "SYNTHETIC.MI")
    assert protected_cache.export_snapshot(staging, b"synthetic-key-is-not-a-secret-00") == 1


def test_real_negative_receipts_are_eligible_for_encrypted_cache_not_fake_price_series(
    monkeypatch, tmp_path,
):
    from tarzan.data import protected_cache

    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "private"))
    price_cache.store_timed(bonds.NAMESPACE, ISIN, negative())
    assert protected_cache.export_snapshot(
        tmp_path / "sealed", b"synthetic-key-is-not-a-secret-00",
    ) == 1
    assert bonds.eur_per_nominal(negative(), as_of=END).empty
