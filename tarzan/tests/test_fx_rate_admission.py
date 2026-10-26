"""Malformed FX cannot become a 1:1 conversion, even after inversion/merging."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tarzan.data import enricher, price_cache


INDEX = pd.to_datetime(["2026-10-22", "2026-10-23"])


@pytest.mark.parametrize("invalid", [
    True, np.bool_(True), complex(0.9, 0), complex(0.9, 0.1),
    np.complex128(0.9 + 0.1j), 10 ** 400, pd.NA,
    [0.9], {"raw": 0.9}, pd.Series([0.9]), pd.Timestamp("2026-10-23"),
    float("nan"), float("inf"), float("-inf"), 0.0, -0.9,
])
def test_invalid_fx_row_does_not_abort_or_contaminate_other_rates(invalid):
    source = pd.Series([invalid, 0.9], index=INDEX, dtype=object, name="FX")
    source.attrs = {"synthetic": {"observed": "unchanged"}}
    original = deepcopy(source)
    result = enricher._usable_fx_series(source)
    assert list(result.index) == [INDEX[1]]
    assert list(result) == [0.9]
    assert result.name == source.name
    assert result.attrs == source.attrs
    pd.testing.assert_series_equal(source, original, check_exact=True)


@pytest.mark.parametrize("source", [
    pd.Series([True, False], index=INDEX, dtype=bool),
    pd.Series([True, pd.NA], index=INDEX, dtype="boolean"),
    pd.Series([0.9 + 0j, 0.8 + 0j], index=INDEX, dtype="complex128"),
])
def test_boolean_and_complex_dtypes_are_not_real_fx_prices(source):
    assert enricher._usable_fx_series(source).empty


@pytest.mark.parametrize("dtype", ["float64", "float32", "Float64", "int64", "object", "string"])
def test_valid_fx_admission_matches_the_previous_numeric_path_exactly(dtype):
    source = pd.Series([1, 2], index=INDEX, dtype=dtype, name="valid")
    source.attrs = {"synthetic": "metadata"}
    expected = pd.to_numeric(source, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan"),
    ).dropna()
    expected = expected[expected > 0]
    expected.attrs.update(source.attrs)
    pd.testing.assert_series_equal(
        enricher._usable_fx_series(source), expected, check_exact=True,
    )


@pytest.fixture
def fx_source(monkeypatch):
    frames, calls, stored = {}, [], []
    state = {"cached": None}
    monkeypatch.setattr(price_cache, "load_history", lambda key: state["cached"])
    monkeypatch.setattr(
        price_cache, "store_history",
        lambda key, result: stored.append((key, result.copy(deep=True))),
    )
    monkeypatch.setattr(enricher, "_space_yf_call", lambda: None)
    monkeypatch.setattr(enricher, "_retry", lambda fn, **kwargs: fn())

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **arguments):
            assert arguments["interval"] == "1d"
            assert arguments["auto_adjust"] is True
            calls.append(self.symbol)
            return frames.get(self.symbol, pd.DataFrame()).copy(deep=True)

    monkeypatch.setattr(enricher._yf_net, "ticker", Ticker)
    return frames, calls, stored, state


def frame(values, index=INDEX):
    return pd.DataFrame(
        {"Close": pd.Series(values, index=index, dtype=object)},
    )


@pytest.mark.parametrize("invalid", [True, np.bool_(True), complex(0.9, 0.1), 10 ** 400])
@pytest.mark.parametrize("inverse", [False, True])
def test_raw_fx_is_validated_before_inverse_arithmetic(fx_source, invalid, inverse):
    frames, calls, stored, _ = fx_source
    pair = "EURUSD=X" if inverse else "USDEUR=X"
    frames[pair] = frame([invalid, 2.0 if inverse else 0.5])
    result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result.index) == [INDEX[1]]
    assert list(result) == [0.5]
    assert calls == (["USDEUR=X", "EURUSD=X"] if inverse else ["USDEUR=X"])
    assert len(stored) == 1 and stored[0][0] == "FX_USD"
    assert list(stored[0][1].index) == [INDEX[1]]
    assert list(stored[0][1]) == [0.5]


def test_invalid_direct_pair_keeps_the_valid_inverse_fallback(fx_source):
    frames, calls, stored, _ = fx_source
    frames["USDEUR=X"] = frame([True, True])
    frames["EURUSD=X"] = frame([2.0, 4.0])
    result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result) == [0.5, 0.25]
    assert calls == ["USDEUR=X", "EURUSD=X"]
    assert len(stored) == 1


def test_cached_boolean_is_rejected_before_a_fresh_merge_can_cast_it_to_float(fx_source):
    frames, calls, stored, state = fx_source
    cached = pd.Series([True], index=INDEX[:1], dtype=object)
    original = deepcopy(cached)
    state["cached"] = cached
    frames["USDEUR=X"] = frame([0.9], INDEX[1:])
    result = enricher._fetch_fx_pair_uncached("USD")
    assert list(result.index) == [INDEX[1]]
    assert list(result) == [0.9]
    assert calls == ["USDEUR=X"]
    assert len(stored) == 1
    pd.testing.assert_series_equal(cached, original, check_exact=True)


def test_missing_real_fx_cannot_publish_a_one_to_one_security_conversion(monkeypatch):
    monkeypatch.setattr(
        enricher, "_get_fx_series",
        lambda currency: pd.Series([True, True], index=INDEX),
    )
    prices = pd.Series([100.0, 101.0], index=INDEX)
    original = prices.copy(deep=True)
    assert enricher.convert_to_eur(prices, "USD").empty
    pd.testing.assert_series_equal(prices, original, check_exact=True)
