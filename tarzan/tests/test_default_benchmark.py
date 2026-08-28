"""An uncurated taxonomy still yields one benchmark, so a first run computes
alpha/beta instead of reporting None under a footnote naming an unused index."""

import pandas as pd

from tarzan import config as cfg


def _empty_taxonomy(monkeypatch):
    """Simulate a new user's taxonomy: correct columns, zero rows."""
    columns = ["name", "ticker", "isin", cfg.WATCHLIST_FLAG,
               "is_benchmark_alpha_beta", "is_benchmark_geo"]
    monkeypatch.setattr(
        cfg, "_load_indexes_csv", lambda: pd.DataFrame(columns=columns)
    )


def test_empty_taxonomy_falls_back_to_default_benchmark(monkeypatch):
    _empty_taxonomy(monkeypatch)

    universe = cfg.benchmarks()
    assert universe, "an uncurated taxonomy left the benchmark universe empty"

    # Every accessor must name the SAME instrument as the universe, or the
    # semantic gate rejects the render for a catalog/config mismatch.
    assert cfg.benchmark_beta_name() in universe
    assert cfg.benchmark_geo_allocation() in universe
    assert cfg.watchlist_names() == {n.lower() for n in universe}

    # Bare tickers need a taxonomy name to promote against, which is exactly
    # what an empty taxonomy cannot supply — so the default must be qualified.
    for ticker in universe.values():
        assert "." in ticker, f"default benchmark {ticker} is not a full listing"


def test_curated_taxonomy_wins_over_the_default(monkeypatch):
    monkeypatch.setattr(cfg, "_load_indexes_csv", lambda: pd.DataFrame([{
        "name": "My Index", "ticker": "MINE.MI", "isin": "",
        cfg.WATCHLIST_FLAG: "True", "is_benchmark_alpha_beta": "True",
        "is_benchmark_geo": "True",
    }]))

    assert cfg.benchmarks() == {"My Index": "MINE.MI"}
    assert cfg.benchmark_beta_name() == "My Index"
    assert cfg.benchmark_geo_allocation() == "My Index"
