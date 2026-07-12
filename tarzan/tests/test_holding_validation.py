"""Track A / A4-safe — Holding required-field integrity guard.

Holding.__post_init__ enforces that a holding is identifiable (ISIN or ticker)
and its required numeric fields are finite. Zeros are allowed (rebalancer
seeds are legitimately 0.0); only NaN/Inf and a wholly-unidentifiable row are
rejected, so a malformed Holding can never enter the pipeline silently.
"""

from __future__ import annotations

import pytest

from tarzan.models.holding import Holding


def _h(**kw):
    base = dict(isin="IE00X", ticker="", quantity=1.0, cost_basis_eur=100.0,
                market_value_eur=120.0, currency="EUR")
    base.update(kw)
    return Holding(**base)


class TestHoldingValidation:
    def test_valid_holding(self):
        assert _h().quantity == 1.0

    def test_zero_quantity_seed_is_valid(self):
        # A not-yet-held rebalancer seed: zeros + ticker-only are legitimate.
        h = _h(isin="", ticker="ZZZQ", quantity=0.0, cost_basis_eur=0.0,
               market_value_eur=0.0)
        assert h.ticker == "ZZZQ"

    def test_isin_only_and_ticker_only_valid(self):
        assert _h(isin="IE00X", ticker="").isin == "IE00X"
        assert _h(isin="", ticker="VWCE").ticker == "VWCE"

    def test_no_identifier_rejected(self):
        with pytest.raises(ValueError):
            _h(isin="", ticker="")

    @pytest.mark.parametrize("field", ["quantity", "cost_basis_eur", "market_value_eur"])
    def test_non_finite_required_field_rejected(self, field):
        with pytest.raises(ValueError):
            _h(**{field: float("nan")})
        with pytest.raises(ValueError):
            _h(**{field: float("inf")})
