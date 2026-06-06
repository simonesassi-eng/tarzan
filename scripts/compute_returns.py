"""Print XIRR and TWROR from ``input/order_list.csv``.

Thin wrapper around the package: all the analytics (cum/ex netting,
valuation, the price fallback ladder, XIRR, TWROR) live in
``tarzan.engine`` and are reused here. This script only loads the order
list, runs the shared returns builder, and prints the result.

  * **XIRR** (money-weighted): the constant annualised rate that makes
    the NPV of every external cash flow plus today's value zero.
  * **TWROR** (time-weighted): chained period returns, neutral to
    deposit timing — the market behaviour of the held portfolio.

Coverage note: when an instrument has no usable market history the value
is filled by the explicit fallback ladder (synthetic interpolation →
carry-flat → excluded). The coverage % and the per-source instrument
lists are printed so the figures are transparent about how much rests on
real market data.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.data.enricher import enrich_holdings  # noqa: E402
from tarzan.data.loader import load_orders  # noqa: E402
from tarzan.engine.metrics import twror, xirr  # noqa: E402
from tarzan.engine.returns_builder import (  # noqa: E402
    build_holdings_from_orders,
    build_order_derived_series,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("compute_returns")


def main() -> int:
    orders_path = ROOT / "input/order_list.csv"
    if not orders_path.exists():
        logger.error("Missing %s — run scripts/preprocess_orders.py first.", orders_path)
        return 1

    orders = load_orders(str(orders_path))
    if not orders:
        logger.error("No valid orders loaded from %s.", orders_path)
        return 1

    # Derive open holdings from the orders and enrich them through the
    # standard pipeline (price histories, asset classes, bond fallback).
    holdings = build_holdings_from_orders(orders)
    logger.info("Derived %d open holding(s) from %d order(s).", len(holdings), len(orders))
    logger.info("Enriching holdings (price histories, asset classes)…")
    holdings = enrich_holdings(holdings)
    enriched_by_isin = {h.isin: h for h in holdings if h.isin}

    series = build_order_derived_series(orders, enriched_by_isin)
    current_value = series.valuations[-1][1] if series.valuations else 0.0

    # XIRR
    rate = xirr(series.xirr_cashflows)
    deposits = -sum(cf for _, cf in series.xirr_cashflows if cf < 0)
    distributions = sum(cf for _, cf in series.xirr_cashflows if cf > 0) - current_value
    start = min(d for d, _ in series.xirr_cashflows)

    print()
    print("=" * 72)
    print("XIRR — Money-weighted return (sensitive to deposit timing)")
    print("=" * 72)
    print(f"  Period:                {start} → today ({series.span_days} days)")
    print(f"  Total deposits:        {deposits:>16,.2f} EUR")
    print(f"  Total distributions:   {distributions:>16,.2f} EUR  (sells + coupons)")
    print(f"  Current value:         {current_value:>16,.2f} EUR")
    print(f"  Net P&L:               {distributions + current_value - deposits:>16,.2f} EUR")
    rate_str = f"{rate * 100:>15.2f}%" if rate == rate else "            n/a"  # NaN check
    print(f"  XIRR (annualised):     {rate_str}")

    # TWROR
    res = twror(series.valuations, series.external_flows, series.span_days,
                coverage_pct=series.coverage_pct)
    print()
    print("=" * 72)
    print("TWROR — Time-weighted return (neutral to deposit timing)")
    print("=" * 72)
    print(f"  Cumulative TWROR:      {res.cumulative_pct:>15.2f}%")
    print(f"  Annualised TWROR:      {res.annualized_pct:>15.2f}%")
    print(f"  Market-data coverage:  {series.coverage_pct:>15.2f}%")

    # Provenance disclosure: which instruments fell back from yfinance.
    print()
    print("  Price provenance:")
    for source in ("yfinance", "synthetic", "carry_flat", "excluded"):
        isins = series.provenance.get(source, [])
        if isins:
            print(f"    {source:<11} {len(isins):>2}: {', '.join(isins)}")

    # Per-period anomalies (sanity aid).
    anomalies = [p for p in res.periods if abs(p["r"]) > 0.15]
    if anomalies:
        print()
        print("  Per-period anomalies (>±15%):")
        for p in anomalies:
            print(f"    {p['date']}  v_prev={p['v_after_prev']:>14,.2f}  "
                  f"v_before={p['v_before']:>14,.2f}  r={p['r']*100:>+8.2f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
