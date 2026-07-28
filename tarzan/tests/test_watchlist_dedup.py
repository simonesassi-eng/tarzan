"""A held instrument flagged ``is_benchmark=true`` appears once, in the
holdings returns table — never a second time in the watchlist."""

import pandas as pd

from tarzan.export.newsletter import _sections_perf


def test_held_benchmark_dropped_from_watchlist(monkeypatch):
    """NTSG is both held and a curated benchmark; UEQC is only a benchmark."""
    from tarzan import config as cfg

    monkeypatch.setattr(
        cfg, "name_for",
        lambda isin, ticker: {
            "IE00077IIPQ8": "WisdomTree Global Efficient Core",
        }.get(str(isin or "").upper()),
    )
    monkeypatch.setattr(cfg, "instrument_taxonomy", lambda: {})

    holdings_df = pd.DataFrame([{
        "isin": "IE00077IIPQ8", "ticker": "NTSG.MI", "name": "WS GL EFF C",
        "asset_class": "Equities", "current_value": 30000.0, "weight_pct": 100.0,
    }])
    holding_performance = pd.DataFrame([
        {"ticker": "NTSG.MI", "name": "WisdomTree Global Efficient Core",
         "type": "In portfolio", "1d": 0.5},
        {"ticker": "NTSG", "name": "WisdomTree Global Efficient Core",
         "type": "Benchmark index", "1d": 0.5},
        {"ticker": "UEQC", "name": "UBS CMCI Commodity Carry",
         "type": "Benchmark index", "1d": -0.2},
    ])

    class _M:
        holdings_df = None
        holding_performance = None
        performance_full = {"period_used": "2.0Y"}
        portfolio_history = None
        xirr_pct = None
        twror_pct = None
        returns_provenance = None
        intraday_quotes = {}
        total_value = 30000.0

    m = _M()
    m.holdings_df = holdings_df
    m.holding_performance = holding_performance

    class _Ctx:
        metrics = None
        benchmark_alpha_beta = "iShares MSCI ACWI"
        benchmark_geo = "iShares MSCI ACWI"
        performance_intraday_map: dict = {}

    ctx = _Ctx()
    ctx.metrics = m

    out = _sections_perf._build_performance(ctx)
    names = [r["name"] for r in out["benchmark_rows"]]

    assert not any("Efficient Core" in n or "WS GL" in n for n in names), (
        f"held benchmark leaked into the watchlist: {names}"
    )
    assert any("CMCI" in n or "UEQC" in n for n in names), (
        f"unheld benchmark was wrongly dropped: {names}"
    )
