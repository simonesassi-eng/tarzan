"""A rejected raw quote must never become a primary valuation by relabeling."""

from dataclasses import FrozenInstanceError, replace
import datetime as dt

import pandas as pd
import pytest

from tarzan import runtime
from tarzan.data import current_session, enricher
from tarzan.instruments.registry import InstrumentKind, TypeEvidenceGateway
from tarzan.instruments.valuation import (
    ValuationOrigin, holding_kind_resolution, select_holding_valuation,
)
from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.ledger import RunLedger
from tarzan.runtime.provider import ValuationCompletenessEvaluator, ValuationEvidenceState


def holding(**changes):
    values = dict(
        isin="SYNTHETIC", ticker="SYNTH.MI", quantity=10.0,
        cost_basis_eur=73.0, market_value_eur=73.0, currency="EUR",
        instrument_type="ETF", security_type="ETF",
        current_price=10.5, current_value=105.0,
        data_source="fixture-market", fetch_timestamp=runtime.context().captured_at,
    )
    values.update(changes)
    return Holding(**values)


def evaluate(h, *, fallback=False):
    policies = {
        key: replace(policy, allow_fallback=fallback)
        for key, policy in InvestorConfig().provider_quality_policies.items()
    }
    return ValuationCompletenessEvaluator(policies).evaluate([h], RunLedger("authority"))


@pytest.mark.parametrize("provider_kind", ["STOCK", "BOND"])
def test_provider_conflict_survives_display_rewrite(provider_kind):
    h = holding(instrument_resolution=TypeEvidenceGateway().resolve("ETF", provider_kind))
    h.instrument_type = h.security_type = "ETF"
    h.valuation_result = select_holding_valuation(h)
    verdict = evaluate(h)
    assert holding_kind_resolution(h).kind is None
    assert verdict.evidence[0].state is ValuationEvidenceState.UNSUPPORTED
    assert verdict.trustworthy_total_eur is None
    assert not verdict.planning_eligible


def test_an_anchor_is_not_primary_even_when_raw_price_matches_its_amount():
    h = holding(current_value=73.0, current_price=7.3, data_source="input_csv (no market data)")
    h.valuation_result = select_holding_valuation(h)
    assert h.valuation_result.origin is ValuationOrigin.ANCHOR
    assert h.valuation_result.price_eur_per_unit is None
    assert evaluate(h).evidence[0].accepted_by_policy is False
    assert evaluate(h, fallback=True).evidence[0].state is ValuationEvidenceState.FALLBACK


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_market_price_never_licenses_primary(price):
    h = holding(current_price=price)
    assert evaluate(h).evidence[0].state is not ValuationEvidenceState.PRIMARY
    assert not evaluate(h).planning_eligible


def test_mutating_legacy_fields_cannot_reuse_a_selected_valuation():
    h = holding()
    h.valuation_result = select_holding_valuation(h)
    assert evaluate(h).evidence[0].state is ValuationEvidenceState.PRIMARY
    with pytest.raises(FrozenInstanceError):
        h.valuation_result.value_eur = 999.0
    h.current_value = 999.0
    assert not evaluate(h).planning_eligible


@pytest.mark.parametrize("dated", [False, True])
def test_a_new_verified_quote_can_replace_an_anchor_with_its_own_provenance(monkeypatch, dated):
    h = holding(current_value=73.0, data_source="input_csv (no market data)")
    h.valuation_result = select_holding_valuation(h)
    h.price_history = pd.Series([10.0, 10.5], index=pd.bdate_range("2026-10-19", periods=2))
    quote = {"price": 10.6, "prev_close": 10.5}
    if dated:
        quote["time"] = int(dt.datetime(2026, 10, 21, 15, tzinfo=dt.timezone.utc).timestamp())
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.runtime.today", lambda: dt.date(2026, 10, 21))
    monkeypatch.setattr(
        "tarzan.data.market_quotes.official_quotes", lambda symbols: {"SYNTH.MI": quote},
    )
    assert current_session.apply_to_holdings([h]) == ("SYNTH.MI",)
    assert h.current_value == pytest.approx(106.0)
    assert not h.data_source.startswith("input_csv")
    assert h.valuation_result.origin is (
        ValuationOrigin.MARKET if dated else ValuationOrigin.FALLBACK
    )
    assert h.price_is_fallback is not dated


@pytest.mark.parametrize("assertions", [(), ("ETF", "STOCK"), ("ETF", "BOND")])
def test_live_stamp_cannot_erase_unknown_or_conflicting_mechanics(monkeypatch, assertions):
    h = holding(instrument_type=None, security_type=None)
    h.instrument_resolution = TypeEvidenceGateway().resolve(*assertions)
    # A later display/taxonomy rewrite cannot cure a retained conflict.
    if assertions:
        h.instrument_type = "ETF"
    h.price_history = pd.Series([10.0, 10.5], index=pd.bdate_range("2026-10-19", periods=2))
    original = h.price_history.copy()
    monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
    monkeypatch.setattr("tarzan.runtime.today", lambda: dt.date(2026, 10, 21))
    monkeypatch.setattr(
        "tarzan.data.market_quotes.official_quotes",
        lambda symbols: {"SYNTH.MI": {"price": 10.6, "prev_close": 10.5}},
    )
    assert current_session.apply_to_holdings([h]) == ()
    pd.testing.assert_series_equal(h.price_history, original)
    assert h.current_value == 105.0


@pytest.mark.parametrize("kind,expected_price,expected_value", [
    ("ETF", 99.0, 990.0), ("STOCK", 99.0, 990.0), ("BOND", 0.99, 9.9),
])
def test_enrichment_applies_declared_units_once(monkeypatch, kind, expected_price, expected_value):
    h = holding(instrument_type=kind, security_type=kind)
    frame = pd.DataFrame({"Close": [98.0, 99.0]}, index=pd.bdate_range(end=runtime.today(), periods=2))
    monkeypatch.setattr(enricher, "_fetch_ticker_data", lambda *a, **k: {
        "info": {"quoteType": kind, "currency": "EUR"}, "history": frame,
    })
    monkeypatch.setattr(enricher, "_apply_taxonomy_override", lambda h: False)
    h, _ = enricher._enrich_single(h)
    assert h.current_price == pytest.approx(expected_price)
    assert h.current_value == pytest.approx(expected_value)
    assert h.valuation_result.origin is ValuationOrigin.MARKET
    assert h.valuation_result.price_eur_per_unit == pytest.approx(expected_price)
