"""Synthetic listing identity and native-number guards; no market transport."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.data import current_session, market_quotes
from tarzan.data import listing_identity as li
from tarzan.engine import cross_check as xc
from tarzan.log_redaction import RestrictedConsoleFormatter
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.session import RunContext, RunMode, RunSession, activate_session


_NATIVE_QUOTE_DECODER = market_quotes._fetch_official_quotes
DAY = date(2026, 10, 22)
ISIN = "IE00SYNT0011"
OTHER_ISIN = "IE00OTHR0011"


def _session(name="synthetic"):
    context = RunContext(name, RunMode.LIVE, None, datetime(2026, 10, 23, tzinfo=timezone.utc))
    return RunSession(context, {}, RunLedger(name))


def _identity(listing="SYNTH.MI", **changes):
    return replace(li.QuoteIdentity(
        listing, currency="EUR", full_name="Synthetic Global UCITS ETF EUR Acc",
        quote_type="ETF",
    ), **changes)


def _tape():
    return pd.Series([99.0, 100.0, 110.0], index=pd.date_range("2026-10-20", periods=3))


def _qualified(**changes):
    arguments = dict(
        ticker="SYNTH.MI", symbol="SYNTH.DE", isin=ISIN, at=DAY,
        primary_identity=_identity(), sibling_identity=_identity("SYNTH.DE"),
    )
    arguments.update(changes)
    return xc.qualify_sibling(_tape(), _tape(), **arguments)


def _quote(**changes):
    value = {
        "symbol": "SYNTH.MI",
        "currency": "EUR",
        "longName": "Synthetic Global UCITS ETF EUR Acc",
        "quoteType": "ETF",
        "regularMarketPrice": 101.0,
        "regularMarketPreviousClose": 100.0,
        "regularMarketVolume": 0,
    }
    value.update(changes)
    return value


def _decode(monkeypatch, records):
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"quoteResponse": {"result": records}},
    )
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr(
        "tarzan.data._yf_net.yahoo_data",
        lambda: SimpleNamespace(get=lambda *_args, **_kwargs: response),
    )
    monkeypatch.setattr(
        "tarzan.data._yf_net.fetch_yf", lambda action, **_kwargs: action(),
    )
    return _NATIVE_QUOTE_DECODER(["SYNTH.MI"])


@pytest.mark.parametrize("field", (
    "regularMarketPrice", "regularMarketPreviousClose", "regularMarketVolume", "regularMarketTime",
))
@pytest.mark.parametrize("value", (
    True, np.bool_(True), 101 + 1j, np.complex128(101 + 1j),
    np.array([101.0]), ["101"],
))
def test_native_quote_numbers_cannot_be_boolean_complex_or_non_scalar(
        monkeypatch, field, value):
    quotes = _decode(monkeypatch, [_quote(**{field: value})])
    if field == "regularMarketTime":
        assert "SYNTH.MI" not in quotes
        return
    quote = quotes["SYNTH.MI"]
    decoded = {
        "regularMarketPrice": "price",
        "regularMarketPreviousClose": "prev_close",
        "regularMarketVolume": "volume",
        "regularMarketTime": "time",
    }[field]
    assert decoded not in quote


def test_native_quote_zero_volume_is_valid_and_prices_keep_their_contract(monkeypatch):
    assert _decode(monkeypatch, [_quote()]) == {
        "SYNTH.MI": {
            "price": 101.0, "prev_close": 100.0, "currency": "EUR", "volume": 0.0,
        },
    }


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, "bad"))
def test_a_malformed_present_clock_is_not_promoted_to_an_undated_price(monkeypatch, value):
    assert _decode(monkeypatch, [_quote(regularMarketTime=value)]) == {}


@pytest.mark.parametrize("value", (0, -1, -86400))
def test_finite_past_epoch_clocks_keep_their_date_and_stale_verdict(monkeypatch, value):
    quote = _decode(monkeypatch, [_quote(regularMarketTime=value)])["SYNTH.MI"]
    assert quote["time"] == value
    assert current_session.quote_is_stale(quote)


def test_a_missing_clock_keeps_the_explicit_legacy_undated_contract(monkeypatch):
    quote = _decode(monkeypatch, [_quote(regularMarketTime=None)])["SYNTH.MI"]
    assert quote["price"] == 101.0
    assert "time" not in quote


def test_one_malformed_quote_record_does_not_discard_its_valid_neighbour(monkeypatch):
    assert _decode(monkeypatch, [None, True, "broken", _quote()])["SYNTH.MI"]["price"] == 101.0


def test_native_batch_retains_an_immutable_listing_signature_without_changing_prices(monkeypatch):
    with activate_session(_session()):
        prices = _decode(monkeypatch, [_quote()])
        identity = li.quote_identities(["SYNTH.MI"])["SYNTH.MI"]
        assert prices["SYNTH.MI"]["price"] == 101.0
        assert "longName" not in prices["SYNTH.MI"]
        assert identity.full_name == "synthetic global ucits etf eur acc"
        assert identity.currency == "EUR"
        with pytest.raises(FrozenInstanceError):
            identity.currency = "USD"


def test_listing_signatures_do_not_cross_owners_or_acquire_legacy_global_authority():
    li.remember_quote_identities([_quote()], ["SYNTH.MI"])
    assert li.quote_identities(["SYNTH.MI"]) == {}
    with activate_session(_session("first")):
        li.remember_quote_identities([_quote()], ["SYNTH.MI"])
        assert li.quote_identities(["SYNTH.MI"])
        market_quotes.reset_quote_memo()
        assert li.quote_identities(["SYNTH.MI"]) == {}
        li.remember_quote_identities([_quote()], ["SYNTH.MI"])
    with activate_session(_session("second")):
        assert li.quote_identities(["SYNTH.MI"]) == {}
    assert li.quote_identities(["SYNTH.MI"]) == {}


@pytest.mark.parametrize("symbol", ("SYNTH", "SYNTH.DE", "", ".MI", "SYNTH..MI", True))
def test_signature_is_not_indexed_by_a_bare_root_or_an_unrequested_listing(symbol):
    with activate_session(_session()):
        li.remember_quote_identities([_quote(symbol=symbol)], ["SYNTH.MI"])
        assert li.quote_identities(["SYNTH", "SYNTH.MI", "SYNTH.DE"]) == {}


@pytest.mark.parametrize("field", ("currency", "longName", "quoteType", "isin"))
@pytest.mark.parametrize("value", (True, 1, ["EUR"], {"currency": "EUR"}))
def test_malformed_native_identity_declarations_cannot_prove_a_listing(field, value):
    with activate_session(_session()):
        li.remember_quote_identities([_quote(**{field: value})], ["SYNTH.MI"])
        identity = li.quote_identities(["SYNTH.MI"])["SYNTH.MI"]
        assert identity.malformed
        assert li.relative_signature(identity) is None


@pytest.mark.parametrize("change", (
    {"currency": "USD"}, {"longName": "Synthetic Global UCITS ETF EUR Dist"},
    {"quoteType": "EQUITY"}, {"isin": OTHER_ISIN},
))
def test_a_conflicting_batch_does_not_overwrite_an_authoritative_looking_signature(change):
    with activate_session(_session()):
        li.remember_quote_identities([_quote(isin=ISIN)], ["SYNTH.MI"])
        li.remember_quote_identities([_quote(isin=ISIN, **change)] if "isin" not in change
                                     else [_quote(**change)], ["SYNTH.MI"])
        assert li.quote_identities(["SYNTH.MI"])["SYNTH.MI"].conflicting
        li.remember_quote_identities([_quote(isin=ISIN)], ["SYNTH.MI"])
        assert li.relative_signature(li.quote_identities(["SYNTH.MI"])["SYNTH.MI"]) is None


def test_a_partial_later_packet_keeps_existing_positive_facts_in_the_same_owner():
    with activate_session(_session()):
        li.remember_quote_identities([_quote()], ["SYNTH.MI"])
        li.remember_quote_identities([{"symbol": "SYNTH.MI", "currency": "EUR"}], ["SYNTH.MI"])
        assert li.relative_signature(li.quote_identities(["SYNTH.MI"])["SYNTH.MI"])


def test_parallel_metadata_capture_is_owned_and_does_not_mutate_caller_records():
    record = _quote()
    with activate_session(_session()):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(copy_context().run, li.remember_quote_identities,
                            [record], ["SYNTH.MI"])
                for _ in range(12)
            ]
            for future in futures:
                future.result(timeout=2)
        identity = li.quote_identities(["SYNTH.MI"])["SYNTH.MI"]
        assert not identity.conflicting
        assert record["longName"] == "Synthetic Global UCITS ETF EUR Acc"
        observed = li.quote_identities(["SYNTH.MI"])
        observed.clear()
        assert li.quote_identities(["SYNTH.MI"])


def test_a_relative_signature_keeps_prices_and_labels_its_limited_authority():
    vote = _qualified()
    assert vote is not None
    pd.testing.assert_series_equal(vote, _tape())
    assert vote.attrs["third_source_listing"]["identity_basis"] == "relative_quote_signature"
    assert not {"verified_isin", "adjustment_basis"}.intersection(vote.attrs)


@pytest.mark.parametrize("change", (
    {"full_name": "Synthetic Global UCITS ETF EUR Dist"},
    {"full_name": "Synthetic Global UCITS ETF USD Acc"},
    {"full_name": "Synthetic Global UCITS ETF EUR Hedged Acc"},
    {"full_name": "Synthetic Global UCITS ETF EUR Class B Acc"},
    {"full_name": None}, {"full_name": "ETF"}, {"full_name": "UCITS ETF"},
    {"full_name": "Global Equity UCITS ETF Acc"},
    {"quote_type": "EQUITY"}, {"quote_type": "UNKNOWN"}, {"quote_type": None},
    {"currency": "USD"}, {"currency": "GBp"}, {"currency": None},
    {"isin": OTHER_ISIN}, {"symbol": "SYNTH.F"},
    {"malformed": True}, {"conflicting": True},
))
def test_a_similar_name_or_an_uncertain_currency_identity_or_type_abstains(change):
    assert _qualified(sibling_identity=_identity("SYNTH.DE", **change)) is None


def test_short_name_cannot_substitute_for_the_exact_full_name():
    with activate_session(_session()):
        quote = _quote(shortName=_quote()["longName"])
        quote.pop("longName")
        li.remember_quote_identities([quote], ["SYNTH.MI"])
        assert li.relative_signature(li.quote_identities(["SYNTH.MI"])["SYNTH.MI"]) is None


def test_full_name_equality_preserves_all_tokens_but_ignores_case_and_whitespace():
    assert _qualified(sibling_identity=_identity(
        "SYNTH.DE", full_name="  SYNTHETIC   Global UCITS ETF EUR Acc  ",
    )) is not None


def test_an_identical_generic_full_name_is_still_not_share_class_evidence():
    identity = _identity(full_name="Global Equity UCITS ETF EUR Acc")
    assert _qualified(primary_identity=identity,
                      sibling_identity=replace(identity, symbol="SYNTH.DE")) is None


def test_an_identical_numeric_placeholder_name_does_not_prove_a_fund():
    identity = _identity(full_name="0001 0002 0003")
    assert _qualified(primary_identity=identity,
                      sibling_identity=replace(identity, symbol="SYNTH.DE")) is None


def test_a_name_shared_by_known_different_isins_is_ambiguous_but_two_venues_are_not():
    identities = {"SYNTH.MI": _identity(), "SYNTH.DE": _identity("SYNTH.DE")}
    ambiguous = li.ambiguous_relative_signatures(
        [("SYNTH.MI", ISIN), ("SYNTH.DE", OTHER_ISIN)], identities,
    )
    assert _qualified(ambiguous_signatures=ambiguous) is None
    assert li.ambiguous_relative_signatures(
        [("SYNTH.MI", ISIN), ("SYNTH.DE", ISIN)], identities,
    ) == set()


def test_a_different_root_needs_causal_alias_authority_even_with_matching_claims():
    assert _qualified(
        symbol="OTHER.DE", sibling_identity=_identity("OTHER.DE", isin=ISIN),
    ) is None


def test_a_causal_verified_alias_can_have_a_different_full_name():
    vote = _qualified(
        symbol="OTHER.DE",
        sibling_identity=_identity("OTHER.DE", full_name="A Different Verified Name"),
        verified_alias=True, verified_currency="EUR",
    )
    assert vote is not None
    assert vote.attrs["third_source_listing"]["identity_basis"] == "causal_verified_alias"


@pytest.mark.parametrize("identity", (
    _identity("OTHER.DE", currency="USD"), _identity("OTHER.DE", isin=OTHER_ISIN),
    _identity("OTHER.DE", quote_type="BOND"), _identity("OTHER.DE", conflicting=True),
))
def test_alias_authority_cannot_override_an_explicit_native_contradiction(identity):
    assert _qualified(
        symbol="OTHER.DE", sibling_identity=identity,
        verified_alias=True, verified_currency="EUR",
    ) is None


@pytest.mark.parametrize("currency", (None, "USD", "GBp"))
def test_verified_aliases_still_need_currency_authority(currency):
    assert _qualified(
        symbol="OTHER.DE", sibling_identity=None,
        verified_alias=True, verified_currency=currency,
    ) is None


@pytest.mark.parametrize("factor", (0.01, 100.0, 1.11, float("inf")))
def test_same_names_do_not_override_incompatible_price_units_or_common_close_levels(factor):
    assert xc.qualify_sibling(
        _tape(), _tape() * factor, ticker="SYNTH.MI", symbol="SYNTH.DE", isin=ISIN,
        primary_identity=_identity(), sibling_identity=_identity("SYNTH.DE"), at=DAY,
    ) is None


@pytest.mark.parametrize("tolerance", (True, -1, np.inf, np.nan, "bad"))
def test_malformed_level_tolerances_cannot_license_an_uncertain_vote(tolerance):
    assert _qualified(level_tolerance=tolerance) is None


def test_a_date_missing_from_either_native_tape_is_not_a_common_close():
    earlier = _tape().iloc[:-1]
    assert xc.qualify_sibling(
        _tape(), earlier, ticker="SYNTH.MI", symbol="SYNTH.DE", isin=ISIN, at=DAY,
        primary_identity=_identity(), sibling_identity=_identity("SYNTH.DE"),
    ) is None


@pytest.mark.parametrize("missing", ("2026-10-21", "2026-10-22"))
def test_a_nearby_close_cannot_vote_over_a_different_interval(missing):
    third = _tape().drop(pd.Timestamp(missing))
    verdict = {
        "status": xc.DIVERGED, "ours": 10.0, "alt": 3.0,
        "anchor": pd.Timestamp("2026-10-21"), "alt_anchor": pd.Timestamp("2026-10-21"),
        "at": DAY,
    }
    result = xc.settle(verdict, _tape(), [third])
    assert result["status"] == xc.DIVERGED
    assert result["third"] == []


def test_a_second_source_using_an_earlier_anchor_cannot_adjudicate_our_return():
    verdict = {
        "status": xc.DIVERGED, "ours": 10.0, "alt": 3.0,
        "anchor": pd.Timestamp("2026-10-21"), "alt_anchor": pd.Timestamp("2026-10-20"),
        "at": DAY,
    }
    assert xc.settle(verdict, _tape(), [_tape()])["status"] == xc.DIVERGED


def test_a_borrowed_baseline_cannot_masquerade_as_a_native_common_close():
    tape = _tape()
    tape.attrs["current_session_baseline"] = {
        "at": "2026-10-23", "anchor": DAY.isoformat(), "observation_date": "2026-10-21",
    }
    assert xc.qualify_sibling(
        tape, _tape(), ticker="SYNTH.MI", symbol="SYNTH.DE", isin=ISIN, at=DAY,
        primary_identity=_identity(), sibling_identity=_identity("SYNTH.DE"),
    ) is None


@pytest.mark.parametrize("with_alias", (False, True))
@pytest.mark.parametrize("failure", ("none", "history", "discovery"))
@pytest.mark.parametrize("restricted_console", (False, True))
def test_production_voter_searches_every_venue_after_a_vote_or_provider_error(
        monkeypatch, caplog, with_alias, failure, restricted_console):
    from tarzan.data import justetf, listing_aliases
    from tarzan.engine.metrics import MetricsEngine
    from tarzan.models.holding import AssetClass, Holding
    from tarzan.models.investor_config import InvestorConfig

    aliases = ("OTHER.DE",) if with_alias else ()
    monkeypatch.setattr(
        listing_aliases, "verified_siblings",
        lambda ticker, *, isin=None: aliases if ticker == "SYNTH.MI" and isin == ISIN else (),
    )
    monkeypatch.setattr(
        listing_aliases, "alias_currency",
        lambda ticker, source, *, isin: (
            "EUR" if ticker == "SYNTH.MI" and source in aliases and isin == ISIN else None
        ),
    )
    expected = [*aliases, "SYNTH.DE", "SYNTH.PA", "SYNTH.AS", "SYNTH.F"]
    histories, admitted = [], []
    caplog.set_level("INFO", logger="tarzan.engine.metrics")

    def close_series(symbol):
        histories.append(symbol)
        if failure == "history" and symbol == "SYNTH.DE":
            raise ValueError("https://example.invalid?crumb=synthetic-secret")
        return _tape()

    monkeypatch.setattr("tarzan.data.enricher._sibling_close_series", close_series)
    if failure == "discovery":
        def failed_discovery(*_args, **_kwargs):
            raise ValueError("https://example.invalid?crumb=synthetic-secret")
        monkeypatch.setattr(market_quotes, "_sibling_symbols", failed_discovery)
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.runtime.today", lambda: DAY)
    monkeypatch.setattr("tarzan.config._load_indexes_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(justetf, "prefetch", lambda _isins: None)
    monkeypatch.setattr(justetf, "series", lambda _isin: None)
    monkeypatch.setattr(justetf, "quote", lambda _isin: None)
    monkeypatch.setattr("tarzan.data._yf_net.yahoo_data", lambda: pytest.fail("new HTTP"))
    monkeypatch.setattr("tarzan.data.enricher._fetch_ticker_info",
                        lambda *_args: pytest.fail("new info request"))
    monkeypatch.setattr("tarzan.data.enricher._openfigi_raw",
                        lambda *_args: pytest.fail("new identity discovery"))

    def verdicts(*_args, **_kwargs):
        out = {window: {"status": xc.UNAVAILABLE} for window in xc.LONG_WINDOWS}
        out["1m"] = {
            "status": xc.DIVERGED, "ours": 10.0, "alt": 3.0,
            "anchor": pd.Timestamp("2026-10-21"), "alt_anchor": pd.Timestamp("2026-10-21"),
            "at": DAY,
        }
        return out

    settle = xc.settle

    def collect(verdict, tape, sources):
        admitted.extend(source.attrs["third_source_listing"]["symbol"] for source in sources)
        return settle(verdict, tape, sources)

    monkeypatch.setattr(xc, "check_windows", verdicts)
    monkeypatch.setattr(xc, "settle", collect)
    holding = Holding(
        ISIN, "SYNTH.MI", 1, 100, 110, "EUR",
        instrument_type="ETF", security_type="ETF", asset_type="ETF",
        instrument_provider_evidence=("ETF",), asset_class=AssetClass.EQUITIES,
        price_history=_tape(), price_history_native=_tape(), price_currency="EUR",
    )
    engine = MetricsEngine([holding], InvestorConfig())
    ctx = {"holding_performance": pd.DataFrame([
        {"ticker": "SYNTH.MI", "type": "In portfolio", "1d": 10.0, "1m": 10.0},
    ])}
    records = [
        _quote(),
        _quote(symbol="SYNTH.DE"),
        _quote(symbol="SYNTH.PA", longName="Synthetic Global UCITS ETF EUR Dist"),
        _quote(symbol="SYNTH.F"),
    ]
    if with_alias:
        records.append(_quote(symbol="OTHER.DE", longName="A Different Verified Name"))
    with activate_session(_session()):
        caplog.handler.setFormatter(
            RestrictedConsoleFormatter() if restricted_console
            else logging.Formatter("%(levelname)s %(message)s")
        )
        li.remember_quote_identities(records, ["SYNTH.MI", *expected])
        engine._second_source(ctx)
    assert histories == ([] if failure == "discovery" else expected)
    assert admitted == ([] if failure == "discovery" else [
        *aliases, *([] if failure == "history" else ["SYNTH.DE"]), "SYNTH.F",
    ])
    assert ctx["holding_performance"].iloc[0]["xc_1m"] == (
        xc.DIVERGED if failure == "discovery" else xc.CONFIRMED
    )
    assert ctx["holding_performance"].iloc[0]["1m"] == 10.0
    if failure != "none":
        # CI intentionally formats the public console as fixed status codes.
        # Inspect original private records without disabling that protection.
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "(ValueError)" in messages
        for output in (messages, caplog.text):
            assert "synthetic-secret" not in output
            assert "example.invalid" not in output
        if restricted_console:
            assert "(ValueError)" not in caplog.text
            assert "PRIVATE_DIAGNOSTIC" in caplog.text
