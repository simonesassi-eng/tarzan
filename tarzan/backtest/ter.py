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

from tarzan.backtest.model import (
    ALT, COMM, CRYPTO, EQ, FI, GOLD, WhatIfItem,
)

# Per-asset-class default TER (%) — last-resort only.
_TER_FALLBACK = {EQ: 0.20, FI: 0.15, GOLD: 0.15, COMM: 0.40,
                 ALT: 0.90, CRYPTO: 0.50}


def instrument_ter(it: "WhatIfItem") -> float:
    """Best TER estimate (%) for an instrument, most-precise source first."""
    # holding.ter is a FRACTION (0.0035 == 0.35%): taxonomy ter (curated) or
    # yfinance. Bounded to reject junk values.
    ter = getattr(it.holding, "ter", None)
    if ter is not None and ter == ter and 0 < ter < 0.05:
        return ter * 100.0
    isin = getattr(it, "isin", "") or getattr(it.holding, "isin", "")
    if isin:
        jt = geo_resolver.justetf_ter(isin)
        if jt is not None and 0 < jt < 0.05:
            return jt * 100.0
    dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
    return _TER_FALLBACK.get(dom, 0.20)
