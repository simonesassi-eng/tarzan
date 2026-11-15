"""One window, one return — wherever the issue prints it.

The target's 3M once read three different figures in one issue: the KPI matrix,
the Vs-target chart and the Portfolio movers total. This renders a synthetic
book with a per-holding target and holds the three surfaces to one number per
window, for the portfolio and for the target.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarzan.tests.test_golden_master import _AS_OF, _ORDERS_CSV, _stub_enrich


@pytest.fixture
def analysed(tmp_path, monkeypatch):
    from tarzan import orchestrator

    monkeypatch.setattr("tarzan.data.enricher.enrich_holdings", _stub_enrich)
    monkeypatch.setattr("tarzan.engine.metrics._fetch_benchmark_history",
                        lambda *a, **k: pd.Series(dtype=float))
    orders = tmp_path / "order_list.csv"
    orders.write_text(_ORDERS_CSV)
    targets = tmp_path / "targets_per_holding.csv"   # synthetic weights
    targets.write_text(
        "name,isin,ticker,target_equities,target_fixed_income,target_portfolio,no_buy_no_sell\n"
        "A,IE00B4L5Y983,,,,60,\n"
        "B,IE00B4WXJJ64,,,,40,\n")
    metrics, config = orchestrator.run(
        config_source=None, orders_source=str(orders),
        targets_per_holding_source=str(targets),
        deterministic=True, as_of=_AS_OF,
    )
    return metrics, config


def test_every_section_prints_one_return_per_window(analysed):
    from tarzan.export.newsletter._constants import _NewsletterContext
    from tarzan.export.newsletter._sections_perf import (
        _MOVER_WINDOWS, _build_portfolio_movers, _window_kpis,
    )
    from tarzan.export._perf_series import _perf_window

    m, config = analysed
    assert m.target_history is not None and len(m.target_history) > 2
    windows = [k for k, _ in _MOVER_WINDOWS if k != "1d"]
    kpis = _window_kpis(m, windows, None)
    movers = _build_portfolio_movers(_NewsletterContext(metrics=m, config=config))
    checked = 0
    for b in windows:
        chart = _perf_window(m, 30, None, bucket=b)
        if not chart:
            continue
        for who, key in (("portfolio", "twr"), ("target", "target")):
            kpi = kpis.get(who, {}).get(b, {}).get("return")
            drawn = (chart.get("endpoints") or {}).get(key)
            listed = (movers.get("totals") or {}).get(who, {}).get(b)
            if kpi is None:
                continue
            assert drawn == pytest.approx(kpi, abs=1e-9), (who, b, "chart", drawn, kpi)
            assert listed == pytest.approx(kpi, abs=1e-9), (who, b, "movers", listed, kpi)
            checked += 1
    assert checked >= 4, checked


def test_the_volatility_chart_states_the_kpi_matrix_volatility(analysed):
    """The volatility panel's key computes its σ itself; it must print the KPI
    matrix's since-inception figure (the risk engine's), not a lookalike."""
    from tarzan.export.newsletter._sections_perf import _window_kpis
    from tarzan.export._perf_series import _perf_vol_series

    m, _config = analysed
    kpis = _window_kpis(m, ["si"], None)
    vs = _perf_vol_series(m, None)
    assert vs is not None
    for who, key in (("portfolio", "port"), ("target", "target")):
        kpi = kpis.get(who, {}).get("si", {}).get("volatility")
        if kpi is not None and vs["span"].get(key) is not None:
            assert vs["span"][key] == pytest.approx(kpi, rel=1e-9), (who, vs["span"][key], kpi)
