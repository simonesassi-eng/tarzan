"""What-if / long-history backtest package.

Thin orchestration over Tarzan's shared engine: enrich candidate portfolios,
build per-instrument spliced synthetic histories, align to one common window
and compare. All metrics/statistics are reused from ``tarzan.engine`` (stats,
robustness, synthetic, allocations) — nothing is re-implemented here.

Public API::

    from tarzan.backtest import run_backtest
    portfolios = run_backtest("input/portfolio_test.csv", currency="eur")
"""

from __future__ import annotations

from tarzan.backtest.model import (  # noqa: F401
    ASSET_ORDER, GEO_ORDER, Portfolio, WhatIfItem, compute_allocations,
)
from tarzan.backtest.engine import (  # noqa: F401
    compute_robustness, instrument_exposures,
    newsletter_portfolios, portfolio_long_returns, run_backtest, simulation_rows,
)
from tarzan.backtest.loader import (  # noqa: F401
    build_symbol_map, enrich_universe, load_portfolios, portfolio_items,
)
from tarzan.backtest.ter import instrument_ter  # noqa: F401
from tarzan.backtest.testfol import (  # noqa: F401
    testfol_instrument_map, testfol_lines,
)

__all__ = [
    "run_backtest", "newsletter_portfolios", "Portfolio", "WhatIfItem", "compute_allocations",
    "simulation_rows", "instrument_exposures", "portfolio_long_returns",
    "compute_robustness", "instrument_ter",
    "testfol_lines", "testfol_instrument_map", "load_portfolios",
    "build_symbol_map", "enrich_universe", "portfolio_items",
    "ASSET_ORDER", "GEO_ORDER",
]
