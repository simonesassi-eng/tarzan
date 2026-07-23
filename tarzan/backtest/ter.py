"""Per-instrument TER (total expense ratio) estimation for the backtest.

Single source used by BOTH the TER drag on the modeled base and the testfol
export. Most-precise source first:

  1. ``holding.ter`` — the curated ``ter`` column of instrument_taxonomy.csv
     (the enricher sets it there, overriding yfinance), else the yfinance TER
     for funds that carry one;
  2. justETF profile TER by ISIN (auto for EU UCITS not curated yet);
  3. a per-asset-class default as the last resort.

There is NO hardcoded fee table here anymore: curated fees live in the taxonomy
CSV (``ter`` column), the single editable source of truth.
"""

from __future__ import annotations

from tarzan.data import geo_resolver

from tarzan.backtest.model import EQ, WhatIfItem


def instrument_ter(it: "WhatIfItem") -> float:
    """Best TER estimate (%) for an instrument, most-precise source first.

    holding.ter (curated taxonomy / yfinance, then the enricher's justETF +
    class-default gap-fill) is a FRACTION and is authoritative — the live and
    backtest paths share that single resolution. Only if it is somehow still
    absent do we resolve here against the dominant notional class, so a bare
    WhatIfItem never crashes the drag model.
    """
    ter = getattr(it.holding, "ter", None)
    if ter is not None and ter == ter and 0 < ter < 0.05:
        return ter * 100.0
    isin = getattr(it, "isin", "") or getattr(it.holding, "isin", "")
    dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
    resolved = geo_resolver.resolve_ter(isin, dom)
    return (resolved if resolved is not None else 0.0020) * 100.0
