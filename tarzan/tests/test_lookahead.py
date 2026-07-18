"""Point-in-time (lookahead) guard for ``--as_of`` valuation.

Determinism claim under test: with a pinned ``as_of=T``, every reported metric
must depend ONLY on data at or before T. Post-T prices must never leak into the
XIRR, TWROR, total value or daily value series — otherwise an as-of report of a
past date would silently change as new data arrives, and backtests would peek.

Method: run the full pipeline twice with the SAME ``as_of``, changing ONLY the
prices strictly AFTER ``as_of`` between the two runs (and extending the series
further into the future). If any metric reads past-T data, the two runs diverge.
Network-free: enrichment + benchmarks are stubbed (same approach as the golden
master), so this is fully deterministic.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan import orchestrator

_AS_OF = datetime.date(2025, 6, 29)
_ORDERS_CSV = (
    "date,type,isin,quantity,gross_eur,net_eur,currency,price_native,fx_rate,instrument_kind\n"
    "2025-01-01,buy,IE00B4L5Y983,100,10000,-10000,EUR,100,1.0,ETF\n"
    "2025-01-01,buy,IE00B4WXJJ64,50,5000,-5000,EUR,100,1.0,ETF\n"
    "2025-03-01,buy,IE00B4L5Y983,20,2200,-2200,EUR,110,1.0,ETF\n"
)


def _make_enrich(post_asof_multiplier: float, extra_future_days: int):
    """Build a stub enricher whose per-ISIN price series is IDENTICAL up to and
    including ``_AS_OF`` but DIFFERENT after it: prices strictly after ``_AS_OF``
    are scaled by ``post_asof_multiplier`` and the series is extended
    ``extra_future_days`` further. A lookahead-clean pipeline must ignore both
    changes when valuing as of ``_AS_OF``."""
    from tarzan.models.holding import AssetClass, Geography

    meta = {
        "IE00B4L5Y983": ("Equities", "USA", 100.0, 1.20),
        "IE00B4WXJJ64": ("Fixed Income", None, 100.0, 1.05),
    }
    # Common daily grid that spans well before and after as_of.
    start = datetime.date(2025, 1, 1)
    base_days = 180 + extra_future_days
    idx = pd.date_range(start, periods=base_days, freq="D")
    asof_ts = pd.Timestamp(_AS_OF)

    def _enrich(holdings):
        # Mirror the REAL enricher's point-in-time clip so this test exercises
        # the actual production guard (_clip_to_as_of), not a bespoke one: build
        # the full (future-inclusive) series, clip it, then derive current_price/
        # current_value from the clipped tail exactly as _set_price_data does.
        from tarzan.data.enricher import _clip_to_as_of
        ac_map = {a.value: a for a in AssetClass}
        geo_map = {g.value: g for g in Geography}
        for h in holdings:
            ac_s, geo_s, p0, mult = meta.get(h.isin, ("Equities", "USA", 100.0, 1.10))
            vals = []
            for i, ts in enumerate(idx):
                p = p0 * (1 + (mult - 1) * i / 179)
                if ts > asof_ts:
                    p *= post_asof_multiplier  # perturb ONLY the future
                vals.append(p)
            series = _clip_to_as_of(pd.Series(vals, index=idx))
            h.price_history = series
            h.current_price = float(series.iloc[-1])
            h.current_value = h.quantity * h.current_price
            h.asset_class = ac_map[ac_s]
            if geo_s:
                h.geography = geo_map[geo_s]
                h.geo_breakdown = {geo_map[geo_s]: 100.0}
            h.class_breakdown = {ac_map[ac_s]: 100.0}
        return holdings

    return _enrich


def _run(monkeypatch, tmp_path, *, post_asof_multiplier, extra_future_days):
    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings",
                        _make_enrich(post_asof_multiplier, extra_future_days))
    empty = pd.Series(dtype=float)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: empty)
    monkeypatch.setattr("tarzan.engine.metrics._build_benchmark_series",
                        lambda *a, **k: empty)
    orders = tmp_path / "order_list.csv"
    orders.write_text(_ORDERS_CSV)
    metrics, _ = orchestrator.run(
        config_source=None, orders_source=str(orders),
        targets_per_holding_source=None,
        deterministic=True, as_of=_AS_OF,
    )
    return metrics


def _fingerprint(m) -> dict:
    """The as-of-sensitive scalars + the value-series tail, rounded so we
    compare real numbers not float noise."""
    def _r(x, n=6):
        return None if x is None else round(float(x), n)

    avs = m.actual_value_series
    series_tail = None
    if avs is not None and len(avs):
        # Everything up to as_of must be identical; sample the last in-window point.
        in_win = avs[avs.index <= pd.Timestamp(_AS_OF)]
        if len(in_win):
            series_tail = (str(in_win.index[-1].date()), round(float(in_win.iloc[-1]), 4))
    return {
        "total_value": _r(m.total_value, 4),
        "xirr_pct": _r(m.xirr_pct),
        "twror_pct": _r(m.twror_pct),
        "twror_annualized_pct": _r(m.twror_annualized_pct),
        "pnl_eur": _r(m.pnl_eur, 4),
        "series_tail": series_tail,
    }


def test_asof_metrics_ignore_future_prices(monkeypatch, tmp_path):
    # Baseline vs a run where every post-as_of price is +50% and the series runs
    # 90 days longer. A point-in-time-correct pipeline yields identical metrics.
    base = _fingerprint(_run(monkeypatch, tmp_path,
                             post_asof_multiplier=1.0, extra_future_days=0))
    perturbed = _fingerprint(_run(monkeypatch, tmp_path,
                                  post_asof_multiplier=1.5, extra_future_days=90))
    assert base == perturbed, (
        "as_of metrics changed when only POST-as_of prices changed — "
        f"lookahead leak.\n base={base}\n pert={perturbed}"
    )


def test_asof_valuation_uses_asof_not_last_price(monkeypatch, tmp_path):
    # Sanity: the guard above would also pass if as_of were silently ignored and
    # BOTH runs used the same (wrong) price. Prove the valuation actually tracks
    # as_of by checking total_value matches the as_of-date price, not the last.
    m = _run(monkeypatch, tmp_path, post_asof_multiplier=3.0, extra_future_days=120)
    # Equity ramps 100→120 over 180 days from 2025-01-01; as_of is day 179.
    # If future (×3) prices leaked, total_value would balloon far above the
    # ~120-priced holding value. Assert it stays in the as-of regime.
    assert m.total_value < 30000, (
        f"total_value {m.total_value} implies future ×3 prices leaked into the "
        "as-of valuation"
    )


# ---------------------------------------------------------------------------
# Production-readiness bug exploration: C1 output invariance
# ---------------------------------------------------------------------------

_C1_TAX_CONFIG = (
    "key,value\n"
    "rebalancing_capital_gains_tax_standard_pctg,26\n"
    "rebalancing_capital_gains_tax_government_pctg,12.5\n"
    "rebalancing_lump_sum_amount_eur,5000\n"
    "target_invested_allocation_equities_pctg,65\n"
    "target_invested_allocation_fixed_income_pctg,35\n"
)


def _run_c1_orders(monkeypatch, tmp_path, orders_csv: str):
    """Run the same pinned, network-free pipeline with a supplied order ledger."""
    monkeypatch.setattr(
        "tarzan.data.enricher.enrich_holdings", _make_enrich(1.0, 4)
    )
    empty = pd.Series(dtype=float)
    monkeypatch.setattr(
        "tarzan.engine.metrics._fetch_benchmark_history", lambda *a, **k: empty
    )
    monkeypatch.setattr(
        "tarzan.engine.metrics._build_benchmark_series", lambda *a, **k: empty
    )
    orders = tmp_path / "c1_order_list.csv"
    config = tmp_path / "c1_targets.csv"
    orders.write_text(orders_csv)
    config.write_text(_C1_TAX_CONFIG)
    metrics, _ = orchestrator.run(
        config_source=str(config),
        orders_source=str(orders),
        targets_per_holding_source=None,
        deterministic=True,
        as_of=_AS_OF,
    )
    return metrics


def _c1_financial_fingerprint(metrics) -> dict:
    hdf = metrics.holdings_df.sort_values("isin")
    holdings = tuple(
        (
            row.isin,
            round(float(row.quantity), 6),
            round(float(row.cost_basis_eur), 2),
            round(float(row.current_value), 2),
        )
        for row in hdf.itertuples()
    )
    timeline = metrics.allocation_timeline or {}
    timeline_tail = tuple(sorted(
        (str(k), round(float(v), 6))
        for k, v in ((timeline.get("asset") or [{}])[-1]).items()
    ))
    actions = tuple(sorted(
        (
            action.get("ticker"),
            action.get("direction"),
            round(float(action.get("amount_eur", 0.0)), 2),
        )
        for action in (metrics.rebalancing_suggestions or [])
    ))
    return {
        "holdings_and_cost": holdings,
        "total_value": round(float(metrics.total_value), 2),
        "returns": (
            round(float(metrics.xirr_pct), 6) if metrics.xirr_pct is not None else None,
            round(float(metrics.twror_pct), 6) if metrics.twror_pct is not None else None,
            round(float(metrics.pnl_eur), 2) if metrics.pnl_eur is not None else None,
        ),
        "estimated_cgt_eur": round(float(metrics.estimated_cgt_eur or 0.0), 2),
        "allocation_timeline_tail": timeline_tail,
        "displayed_plan": actions,
    }


# **Validates: Requirements 2.1**
def test_c1_post_asof_sell_cannot_change_any_pinned_financial_surface(
    monkeypatch, tmp_path,
):
    """A future profitable sale must be invisible to the entire pinned run.

    The fixed counterexample changes quantity/cost, returns, estimated tax,
    the timeline endpoint, and planning through the one unfiltered order list.
    """
    future_sell = (
        "2025-07-01,sell,IE00B4L5Y983,-5,1000,1000,EUR,200,1.0\n"
    )
    baseline = _c1_financial_fingerprint(
        _run_c1_orders(monkeypatch, tmp_path, _ORDERS_CSV)
    )
    with_future_order = _c1_financial_fingerprint(
        _run_c1_orders(monkeypatch, tmp_path, _ORDERS_CSV + future_sell)
    )
    assert with_future_order == baseline, (
        "post-as_of sale changed pinned financial output; "
        f"baseline={baseline}; with_future_order={with_future_order}"
    )
