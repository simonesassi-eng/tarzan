"""Unit tests for enricher FX helpers (no network)."""

from __future__ import annotations

import pandas as pd

from tarzan.data.enricher import _normalize_minor_currency


class TestNormalizeMinorCurrency:
    def test_gbp_pence_rescaled_to_gbp(self):
        # 28450 GBp ≡ 284.50 GBP
        prices = pd.Series([28450.0, 28580.0, 28700.0])
        rescaled, currency = _normalize_minor_currency(prices, "GBp")
        assert currency == "GBP"
        assert rescaled.iloc[0] == 284.50
        assert rescaled.iloc[1] == 285.80
        assert rescaled.iloc[2] == 287.00

    def test_gbx_alternate_code_rescaled(self):
        prices = pd.Series([100.0])
        rescaled, currency = _normalize_minor_currency(prices, "GBX")
        assert currency == "GBP"
        assert rescaled.iloc[0] == 1.0

    def test_zac_rescaled_to_zar(self):
        prices = pd.Series([5000.0])
        rescaled, currency = _normalize_minor_currency(prices, "ZAc")
        assert currency == "ZAR"
        assert rescaled.iloc[0] == 50.0

    def test_ila_rescaled_to_ils(self):
        prices = pd.Series([200.0])
        rescaled, currency = _normalize_minor_currency(prices, "ILa")
        assert currency == "ILS"
        assert rescaled.iloc[0] == 2.0

    def test_major_currencies_unchanged(self):
        prices = pd.Series([100.0, 101.0])
        for cur in ("USD", "EUR", "GBP", "JPY", "CHF"):
            rescaled, currency = _normalize_minor_currency(prices, cur)
            assert currency == cur
            assert rescaled.iloc[0] == 100.0
            assert rescaled.iloc[1] == 101.0

    def test_unknown_currency_passthrough(self):
        prices = pd.Series([42.0])
        rescaled, currency = _normalize_minor_currency(prices, "XYZ")
        assert currency == "XYZ"
        assert rescaled.iloc[0] == 42.0

    def test_idempotent_on_major_after_rescale(self):
        prices = pd.Series([28450.0])
        # First call: GBp → GBP, divide by 100
        p1, c1 = _normalize_minor_currency(prices, "GBp")
        # Second call on already-major currency: no change
        p2, c2 = _normalize_minor_currency(p1, c1)
        assert c2 == "GBP"
        assert p2.iloc[0] == 284.50
