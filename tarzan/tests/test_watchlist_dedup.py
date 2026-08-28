"""Each tracked instrument appears in exactly ONE of the three returns tables.

A held instrument flagged ``watchlist=true`` belongs to the holdings returns
table and never to the watchlist; a per-holding TARGET the book does not hold yet
belongs to the target-instruments table and never to the watchlist either.

The target rows are the engine's own ``Target not held`` rows, so the section
depends on neither the taxonomy's ``watchlist`` flag nor the rebalancer having
produced a plan. Both ends of that contract are pinned here: the engine emitting
the row, and the renderer routing it to the right table.
"""

import pandas as pd

from tarzan.export.newsletter import _sections_perf


def _fixture(monkeypatch, *, target_rows=()):
    """NTSG is held AND tracked; UEQC is tracked only; AVWC is tracked AND a
    positive per-holding target that is not held. Returns the built section."""
    from tarzan import config as cfg

    monkeypatch.setattr(
        cfg, "name_for",
        lambda isin, ticker: {
            "IE00077IIPQ8": "WisdomTree Global Efficient Core",
        }.get(str(isin or "").upper()),
    )
    monkeypatch.setattr(cfg, "instrument_taxonomy", lambda: {})
    # Hermetic: the watchlist membership this fixture asserts on comes from here,
    # not from the repository's own instrument_taxonomy.csv.
    monkeypatch.setattr(cfg, "watchlist_names", lambda: frozenset({
        "wisdomtree global efficient core",
        "ubs cmci commodity carry",
        "avantis global equity",
    }))

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
        {"ticker": "AVWC", "name": "Avantis Global Equity",
         "type": "Benchmark index", "1d": 0.1},
        *target_rows,
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
    return _sections_perf._build_performance(ctx)


def test_held_benchmark_dropped_from_watchlist(monkeypatch):
    out = _fixture(monkeypatch)
    names = [r["name"] for r in out["benchmark_rows"]]

    assert not any("Efficient Core" in n or "WS GL" in n for n in names), (
        f"held benchmark leaked into the watchlist: {names}"
    )
    assert any("CMCI" in n or "UEQC" in n for n in names), (
        f"unheld benchmark was wrongly dropped: {names}"
    )
    # No seeded target, no target table — and then nothing may be withheld from
    # the watchlist on the strength of a target set that does not exist.
    assert not out["target_rows"]
    assert not out["target_table_html"]
    assert any("Avant" in n for n in names), names


def test_unheld_target_moves_from_watchlist_to_its_own_table(monkeypatch):
    # The engine's target row carries the operational symbol the enricher
    # resolved (AVWC.DE) and the broker's own description, while the catalog copy
    # carries the curated name on another listing (AVWC): the bare ticker is the
    # only key the two share.
    out = _fixture(monkeypatch, target_rows=[
        {"ticker": "AVWC.DE", "name": "Avantis Global Equity UCITS ETF USD Acc",
         "type": "Target not held", "1d": 0.1},
    ])

    watch = [r["name"] for r in out["benchmark_rows"]]
    target = [r["name"] for r in out["target_rows"]]

    assert any("Avant" in n for n in target), target
    assert not any("Avant" in n for n in watch), watch
    assert out["target_table_html"], "target rows built no table"
    # Every other watchlist row is untouched, and the held target appears in
    # neither table (Returns owns it).
    assert any("CMCI" in n or "UEQC" in n for n in watch), watch
    assert not any("Efficient Core" in n for n in watch + target), watch + target


def test_a_target_is_not_ranked_among_the_holdings_movers(monkeypatch):
    """Best/worst performer ranks the BOOK. A target is not in the book.

    The filter read "portfolio OR not benchmark", so the second clause admitted
    every other kind of row — and a not-held target would be ranked the week's
    best performer on a week it beat everything owned.
    """
    from tarzan.export.newsletter._constants import _NewsletterContext
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.portfolio import PortfolioMetrics

    metrics = PortfolioMetrics(
        total_value=100.0, invested_value=100.0, cash_value=0.0,
        holdings_df=pd.DataFrame([
            {"ticker": "NTSG.MI", "asset_class": "Equities",
             "current_value": 100.0}]),
    )
    metrics.holding_performance = pd.DataFrame([
        {"ticker": "NTSG.MI", "name": "Held", "type": "In portfolio", "5d": 1.0},
        {"ticker": "AVWC.DE", "name": "Target", "type": "Target not held",
         "5d": 9.0},
        {"ticker": "AVWS.DE", "name": "Target2", "type": "Target not held",
         "5d": -9.0},
    ])

    out = _sections_perf._build_movers(_NewsletterContext(
        metrics=metrics, config=InvestorConfig()))

    assert out["available"] is True
    assert out["best"]["ticker"] == "NTSG.MI", out["best"]
    assert out["worst"]["ticker"] == "NTSG.MI", out["worst"]


def test_a_seeded_target_gets_its_returns_row_from_the_engine():
    """The engine measures every seeded target, whatever else it is or is not.

    Two things the row must not depend on, because both were how it was first
    reconstructed downstream: the taxonomy's ``watchlist`` flag (nothing here
    is a benchmark, and there is no benchmark catalog in this ctx) and the
    rebalancer (no plan runs, no verification exists). The target set is
    configuration; a suppressed plan must not empty it.
    """
    import numpy as np
    from tarzan.engine.metrics import MetricsEngine
    from tarzan.models.holding import Holding
    from tarzan.models.investor_config import InvestorConfig

    idx = pd.date_range("2025-06-01", "2026-08-20", freq="B")

    def _h(ticker, ret, *, seeded):
        h = Holding(isin=f"IE{ticker:0<10}", ticker=ticker, quantity=1.0,
                    cost_basis_eur=1000.0, market_value_eur=1000.0,
                    currency="EUR")
        h.price_history = pd.Series(
            np.linspace(100.0, 100.0 * (1.0 + ret), len(idx)), index=idx)
        h.current_value = 1000.0
        h.is_seeded_target = seeded
        h.target_portfolio = 50.0
        return h

    engine = MetricsEngine([_h("HELD.MI", 0.20, seeded=False)],
                           InvestorConfig(),
                           rebalance_seeds=[_h("TGT.DE", 0.60, seeded=True)])
    ctx: dict = {}
    engine._holding_performance(ctx)
    hp = ctx["holding_performance"]

    by_ticker = {str(r["ticker"]): r for _, r in hp.iterrows()}
    assert set(by_ticker) == {"HELD.MI", "TGT.DE"}, list(by_ticker)
    assert by_ticker["HELD.MI"]["type"] == "In portfolio"
    tgt = by_ticker["TGT.DE"]
    assert tgt["type"] == "Target not held"
    # A real measured return, not a placeholder.
    assert tgt["1m"] is not None and tgt["1m"] == tgt["1m"]
    # The type must match NEITHER selector every other consumer of this frame
    # uses, or the target leaks into the book's tables or the benchmark
    # projections.
    low = tgt["type"].lower()
    assert "portfolio" not in low and "benchmark" not in low


def test_leaving_the_target_returns_a_tracked_instrument_to_the_watchlist(monkeypatch):
    """``watchlist=true`` is what Tarzan remembers; the target only re-routes it.

    Same taxonomy row, same tracked flag, two runs: with the instrument in the
    target it prints under Target instruments, and the moment it leaves the
    target it prints under Watchlist — never in both, never in neither. So
    dropping a name from ``targets_per_holding.csv`` is enough to stop planning
    for it while still following it.
    """
    targeted = _fixture(monkeypatch, target_rows=[
        {"ticker": "AVWC.DE", "name": "Avantis Global Equity UCITS ETF USD Acc",
         "type": "Target not held", "1d": 0.1},
    ])
    dropped = _fixture(monkeypatch)  # no seed: the target row is gone

    def _where(out):
        return ("target" if any("Avant" in r["name"] for r in out["target_rows"])
                else "watchlist" if any("Avant" in r["name"]
                                        for r in out["benchmark_rows"])
                else "nowhere")

    assert _where(targeted) == "target"
    assert _where(dropped) == "watchlist"
    # Exactly one table each time.
    for out in (targeted, dropped):
        hits = sum(1 for r in out["target_rows"] + out["benchmark_rows"]
                   if "Avant" in r["name"])
        assert hits == 1, f"tracked instrument printed {hits} times"


def test_a_reference_role_is_fetched_even_when_it_is_not_watchlisted(monkeypatch):
    """``is_benchmark_alpha_beta`` / ``is_benchmark_geo`` earn a price series on
    their own, and stay out of the watchlist table on their own.

    The series is what alpha/beta, the benchmark-gap tile, the vs-market charts
    and the risk table all read, so gating the fetch on ``watchlist`` made two
    independent flags interdependent: turning the row off the watchlist took the
    reference line off three charts. The two questions are now separate — fetched
    (``benchmarks``) and printed in the watchlist (``watchlist_names``).
    """
    from tarzan import config as cfg

    monkeypatch.setattr(cfg, "_load_indexes_csv", lambda: pd.DataFrame([
        {"name": "Tracked Fund", "ticker": "TRK.MI", "isin": "IE0000000001",
         cfg.WATCHLIST_FLAG: "True", "is_benchmark_alpha_beta": "",
         "is_benchmark_geo": ""},
        {"name": "Reference Only", "ticker": "REF.MI", "isin": "IE0000000002",
         cfg.WATCHLIST_FLAG: "", "is_benchmark_alpha_beta": "True",
         "is_benchmark_geo": "True"},
    ]))

    # Fetched: both, or the reference has no history to be a reference with.
    assert cfg.benchmarks() == {"Tracked Fund": "TRK.MI",
                                "Reference Only": "REF.MI"}
    # The feed audit lists whatever was fetched, ISIN included.
    assert {n for n, _t, _i in cfg.benchmark_identities()} == {
        "Tracked Fund", "Reference Only"}
    # Printed in the watchlist: only what the investor flagged as watched.
    assert cfg.watchlist_names() == frozenset({"tracked fund"})
    # And the roles still resolve to it.
    assert cfg.benchmark_beta_name() == "Reference Only"
    assert cfg.benchmark_geo_allocation() == "Reference Only"


def test_a_reference_only_row_is_not_printed_in_the_watchlist(monkeypatch):
    """The renderer end of the same contract."""
    from tarzan import config as cfg

    monkeypatch.setattr(cfg, "name_for", lambda isin, ticker: None)
    monkeypatch.setattr(cfg, "instrument_taxonomy", lambda: {})
    monkeypatch.setattr(cfg, "watchlist_names",
                        lambda: frozenset({"tracked fund"}))

    class _M:
        holdings_df = pd.DataFrame()
        holding_performance = pd.DataFrame([
            {"ticker": "TRK.MI", "name": "Tracked Fund",
             "type": "Benchmark index", "1d": 0.2},
            {"ticker": "REF.MI", "name": "Reference Only",
             "type": "Benchmark index", "1d": 0.3},
        ])
        performance_full = {"period_used": "2.0Y"}
        portfolio_history = None
        xirr_pct = None
        twror_pct = None
        returns_provenance = None
        intraday_quotes: dict = {}
        total_value = 0.0

    class _Ctx:
        metrics = _M()
        benchmark_alpha_beta = "Reference Only"
        benchmark_geo = "Reference Only"
        performance_intraday_map: dict = {}

    out = _sections_perf._build_performance(_Ctx())
    names = [r["name"] for r in out["benchmark_rows"]]
    assert names == ["Tracked Fund"], names
