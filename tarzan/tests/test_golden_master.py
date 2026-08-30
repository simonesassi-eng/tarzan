"""Golden-master regression gate (Track A).

Runs the full order → enrich → compute pipeline on a fixed, network-free
synthetic portfolio (deterministic price history via a stub enricher, pinned
``as_of``) and asserts every headline reported number against a committed
golden. Any change — refactor or otherwise — that moves a reported number
fails HERE, loudly, instead of silently shipping a wrong figure.

This is the safety net for all Track-A data-structure work and any future
structural change: if a value legitimately changes, update GOLDEN in the same
commit (a visible, reviewed diff).
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan import orchestrator


# --- The committed golden (network-free, deterministic) --------------------
GOLDEN = {
    "total_value": 19650.0,
    "invested_value": 19650.0,
    "cash_value": 0.0,
    "xirr_pct": 32.769384,
    "twror_pct": 15.375076,
    "twror_annualized_pct": 33.887401,
    "n_holdings": 2,
    "holdings_isins": ["IE00B4L5Y983", "IE00B4WXJJ64"],
    "alloc_by_class": {"Equities": 73.2824, "Fixed Income": 26.7176},
}

# Deterministic synthetic instruments: a linear price ramp per ISIN, so the
# derived value series / XIRR / TWROR / allocations are fully reproducible.
_META = {
    "IE00B4L5Y983": ("Equities", "USA", 100.0, 1.20),        # +20% over the window
    "IE00B4WXJJ64": ("Fixed Income", None, 100.0, 1.05),     # +5%
}
_AS_OF = datetime.date(2025, 6, 29)
_ORDERS_CSV = (
    "date,type,isin,quantity,gross_eur,net_eur,currency,price_native,fx_rate,instrument_kind\n"
    "2025-01-01,buy,IE00B4L5Y983,100,10000,-10000,EUR,100,1.0,ETF\n"
    "2025-01-01,buy,IE00B4WXJJ64,50,5000,-5000,EUR,100,1.0,ETF\n"
    "2025-03-01,buy,IE00B4L5Y983,20,2200,-2200,EUR,110,1.0,ETF\n"
)


def _stub_enrich(holdings):
    from tarzan.models.holding import AssetClass, Geography
    idx = pd.date_range("2025-01-01", periods=180, freq="D")
    ac_map = {a.value: a for a in AssetClass}
    geo_map = {g.value: g for g in Geography}
    for h in holdings:
        ac_s, geo_s, p0, mult = _META.get(
            h.isin, ("Equities", "USA", 100.0, 1.10))
        series = pd.Series([p0 * (1 + (mult - 1) * i / 179) for i in range(180)], index=idx)
        h.price_history = series
        # The enricher never writes a tape without its listing currency — they are
        # assigned in one block — so a fixture that omitted it was leaning on a
        # production EUR fallback for the golden's own currency marks. State it.
        h.price_currency = "EUR"
        h.current_price = float(series.iloc[-1])
        h.current_value = h.quantity * h.current_price
        h.asset_class = ac_map[ac_s]
        if geo_s:
            h.geography = geo_map[geo_s]
            h.geo_breakdown = {geo_map[geo_s]: 100.0}
        h.class_breakdown = {ac_map[ac_s]: 100.0}
    return holdings


@pytest.fixture
def _golden_run(tmp_path, monkeypatch):
    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
    empty = pd.Series(dtype=float)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: empty)
    orders = tmp_path / "order_list.csv"
    orders.write_text(_ORDERS_CSV)
    metrics, _ = orchestrator.run(
        config_source=None, orders_source=str(orders),
        targets_per_holding_source=None,
        deterministic=True, as_of=_AS_OF,
    )
    return metrics


class TestGoldenMaster:
    def test_scalar_metrics_match_golden(self, _golden_run):
        m = _golden_run
        assert round(m.total_value, 2) == GOLDEN["total_value"]
        assert round(m.invested_value, 2) == GOLDEN["invested_value"]
        assert round(m.cash_value, 2) == GOLDEN["cash_value"]
        assert round(m.xirr_pct, 6) == GOLDEN["xirr_pct"]
        assert round(m.twror_pct, 6) == GOLDEN["twror_pct"]
        assert round(m.twror_annualized_pct, 6) == GOLDEN["twror_annualized_pct"]

    def test_holdings_match_golden(self, _golden_run):
        m = _golden_run
        assert len(m.holdings_df) == GOLDEN["n_holdings"]
        assert list(m.holdings_df["isin"]) == GOLDEN["holdings_isins"]

    def test_allocation_matches_golden(self, _golden_run):
        m = _golden_run
        alloc = {r["category"]: round(r["weight_pct"], 4)
                 for _, r in m.allocation_by_class.iterrows()}
        assert alloc == GOLDEN["alloc_by_class"]
