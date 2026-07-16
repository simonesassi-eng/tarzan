"""Shared allocation-aggregation primitives.

Pure functions (no I/O) for turning per-instrument breakdowns into portfolio
allocations. Centralised here so BOTH the live-portfolio metrics engine
(:class:`tarzan.engine.metrics.MetricsEngine`) and the what-if / backtest
engine aggregate class and geography exposure the SAME way — a holding's value
(or a candidate weight) is distributed across its class/geo breakdown and
summed. This is the "compute_allocations" logic, kept in one place so a change
to how notional exposure is aggregated applies everywhere it is shown.

The methodology (notional exposure via ``class_breakdown``, leverage-aware, may
sum to >100%) is unchanged from what each caller did inline before.
"""

from __future__ import annotations

from typing import Iterable, Optional

from tarzan.models.holding import Geography


def accumulate(pairs: Iterable[tuple[float, dict]]) -> dict:
    """Sum weighted breakdowns.

    ``pairs`` is an iterable of ``(amount, {key: pct})`` where ``amount`` is a
    holding value or a candidate weight and ``pct`` a percentage of that amount
    attributed to ``key`` (a class label or region). Returns ``{key: Σ amount *
    pct / 100}``. Keys may be strings or enums (used as-is). Zero/None amounts
    and empty breakdowns contribute nothing.
    """
    out: dict = {}
    for amount, bd in pairs:
        if not amount or not bd:
            continue
        for key, pct in bd.items():
            if not pct:
                continue
            out[key] = out.get(key, 0.0) + amount * float(pct) / 100.0
    return out


def renorm(d: dict) -> dict:
    """Scale values so they sum to 100 (a no-op mix when the total is ≤0)."""
    total = sum(d.values())
    if total <= 0:
        return dict(d)
    return {k: v * 100.0 / total for k, v in d.items()}


def clean_geo(gb: Optional[dict]) -> Optional[dict]:
    """Normalise a ``{Geography: pct}`` breakdown to real MSCI regions.

    Drops the scraped ``Geography.OTHER`` bucket (noise from yfinance
    top-holdings) and renormalises the remaining known regions to 100%.
    Returns a ``{region_value: pct}`` dict keyed by the region's string value,
    or None when no known region remains.
    """
    if not gb:
        return None
    known = {
        (g.value if hasattr(g, "value") else str(g)): v
        for g, v in gb.items()
        if g != Geography.OTHER and v and v > 0
    }
    total = sum(known.values())
    if total <= 0:
        return None
    return {k: v * 100.0 / total for k, v in known.items()}
