"""A selected market price must retain finite-real scalar type evidence."""

from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tarzan.data import enricher
from tarzan.instruments.registry import TypeEvidenceGateway
from tarzan.instruments.valuation import (
    select_holding_valuation, ValuationOrigin, ValuationResult,
)
from tarzan.models.holding import Holding


INVALID = [
    True, np.bool_(True), 1 + 2j, np.complex128(1 + 2j), np.array([1.]),
    np.array(True), [1.], {"raw": 1.}, pd.Series([1.]), 10 ** 400,
    np.inf, np.nan, pd.NA, "not a price",
]


def holding():
    return Holding(
        isin="SYNTHETIC", ticker="SYNTH.MI", quantity=2.,
        cost_basis_eur=180., market_value_eur=190., currency="EUR",
    )


def history(values):
    return pd.DataFrame({
        "Close": pd.Series(
            values, dtype=object,
            index=pd.date_range("2026-10-21", periods=len(values)),
        ),
    })


@pytest.mark.parametrize("value", INVALID)
def test_invalid_market_quote_scalar_never_corroborates_a_listing(value):
    assert enricher._positive_market_quote(value) is False


@pytest.mark.parametrize("value", INVALID)
def test_invalid_history_rows_never_prove_market_evidence_or_abort_valid_rows(value):
    assert enricher._history_has_market_evidence(history([value])) is False
    assert enricher._history_has_market_evidence(history([value, 101.])) is True


@pytest.mark.parametrize("field", [
    "current_price", "current_value", "quantity", "market_value_eur",
])
@pytest.mark.parametrize("invalid", [True, np.complex128(1 + 2j), np.array([1.])])
def test_valuation_selection_rejects_type_erasure_at_each_financial_input(field, invalid):
    fields = {
        "security_type": "STOCK", "instrument_type": "Stock",
        "current_price": 1., "current_value": 1., "quantity": 1.,
        "market_value_eur": 190., "data_source": "synthetic-provider",
    }
    if field == "market_value_eur":
        fields.update(current_price=None, current_value=None)
    fields[field] = invalid
    selected = select_holding_valuation(SimpleNamespace(**fields))
    if field == "market_value_eur":
        assert selected.origin is ValuationOrigin.UNAVAILABLE
        assert selected.value_eur is None
    else:
        assert selected.origin is ValuationOrigin.ANCHOR
        assert selected.price_eur_per_unit is None
        assert selected.value_eur == 190.


@pytest.mark.parametrize("invalid", [True, np.bool_(True), np.complex128(1 + 2j)])
def test_an_invalid_later_history_row_cannot_replace_the_last_real_price(invalid):
    enriched = holding()
    frame = history([101., invalid])
    frame.attrs["_synthetic_source"] = "retained"
    before = frame.copy(deep=True)
    enricher._set_price_data(enriched, frame, {}, "EUR")
    assert enriched.current_price == 101.
    assert enriched.price_history.tolist() == [101.]
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert frame.attrs == before.attrs


@pytest.mark.parametrize("invalid", [True, np.complex128(1 + 2j), 10 ** 400])
def test_invalid_price_only_rows_do_not_override_valid_total_return_history(invalid):
    enriched = holding()
    frame = history([100., 101.])
    frame["Price Close"] = pd.Series(
        [invalid, invalid], index=frame.index, dtype=object,
    )
    enricher._set_price_data(enriched, frame, {}, "EUR")
    assert enriched.current_price == 101.
    assert enriched.price_history_price_only is None


@pytest.mark.parametrize("invalid", [True, np.complex128(1 + 2j), np.array([1.])])
def test_invalid_previous_close_is_unavailable_not_a_one_euro_price(invalid):
    enriched = holding()
    enricher._set_price_data(enriched, pd.DataFrame(), {"previousClose": invalid}, "EUR")
    assert enriched.current_price is None


@pytest.mark.parametrize("value", [
    101., np.float64(101.), "101", Decimal("101"), Fraction(101, 1),
])
def test_valid_scalar_prices_keep_the_existing_conversion_and_value(value):
    assert enricher._positive_market_quote(value) is True
    enriched = holding()
    enricher._set_price_data(enriched, pd.DataFrame(), {"previousClose": value}, "EUR")
    assert np.float64(enriched.current_price).tobytes() == np.float64(101.).tobytes()
    selected = select_holding_valuation(SimpleNamespace(
        security_type="STOCK", instrument_type="Stock",
        current_price=value, current_value=202., quantity=2.,
        market_value_eur=190., data_source="synthetic-provider",
    ))
    assert selected.origin is ValuationOrigin.MARKET
    assert selected.value_eur == 202.


@pytest.mark.parametrize("field", ["price_eur_per_unit", "value_eur"])
@pytest.mark.parametrize("invalid", [True, np.complex128(1 + 2j), np.array([1.])])
def test_direct_valuation_facts_cannot_bypass_scalar_price_admission(field, invalid):
    fields = {
        "resolution": TypeEvidenceGateway().resolve("STOCK"),
        "origin": ValuationOrigin.MARKET,
        "price_eur_per_unit": 1., "value_eur": 1.,
        "source": "synthetic-provider",
    }
    fields[field] = invalid
    with pytest.raises(ValueError):
        ValuationResult(**fields)
