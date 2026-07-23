"""TER gap-fill resolver — shared by the live enricher and the backtest.

Guards the money-path contract: curated/yfinance TER wins (applied upstream on
holding.ter), then justETF by ISIN, then a per-asset-class default; an unknown
fee stays Unavailable (None), never a silent 0%.
"""

from __future__ import annotations

import datetime

import pytest

from tarzan.data import geo_resolver
from tarzan.data.geo_resolver import resolve_ter, _TER_CLASS_DEFAULT


@pytest.fixture(autouse=True)
def _pinned_run():
    """Pin the run so justETF is cache-only (no network); resolve_ter then
    exercises the class-default branch deterministically."""
    from tarzan import runtime
    runtime.configure(deterministic=True, as_of=datetime.date(2025, 6, 29))
    yield


def test_unknown_class_and_no_isin_stays_unavailable():
    # numeric-zero≠unavailable: a fee we cannot estimate must be None, not 0.0.
    assert resolve_ter("", None) is None


def test_class_default_when_justetf_absent():
    assert resolve_ter("", "Equities") == pytest.approx(_TER_CLASS_DEFAULT["Equities"])
    assert resolve_ter("", "Fixed Income") == pytest.approx(_TER_CLASS_DEFAULT["Fixed Income"])
    assert resolve_ter("", "Alternative") == pytest.approx(0.0090)


def test_justetf_takes_precedence_over_class_default(monkeypatch):
    # A real justETF fee (fraction) must win over the class default.
    monkeypatch.setattr(geo_resolver, "justetf_ter", lambda isin: 0.0007)
    assert resolve_ter("IE00EXAMPLE00", "Equities") == pytest.approx(0.0007)


def test_junk_justetf_falls_through_to_default(monkeypatch):
    # Out-of-band justETF value (>=5%) is rejected → class default used.
    monkeypatch.setattr(geo_resolver, "justetf_ter", lambda isin: 0.9)
    assert resolve_ter("IE00EXAMPLE00", "Gold") == pytest.approx(_TER_CLASS_DEFAULT["Gold"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
