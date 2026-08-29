"""Audit the geo/TER completeness of instrument_taxonomy.csv.

The digest attributes equity geography from five taxonomy columns that must add
up to 100 for every instrument carrying equity. A blank or a 97.3 there is not a
rounding curiosity: it silently misstates the USA weight the portfolio is
steered by. This prints every offender instead of failing, so it can be run
while the gaps are still being filled.

    python3 scripts/check_taxonomy_geo.py            # portfolio + watchlist
    python3 scripts/check_taxonomy_geo.py --all      # every row
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAX = ROOT / "input" / "instrument_taxonomy.csv"
PORTFOLIO_FILES = [
    ("input/order_list.csv", "ticker"),
    ("input/portfolio_test.csv", "ticker"),
    ("input/targets_per_holding.csv", "ticker"),
]
GEO = ["usa", "emerging_markets", "eurozone_emu", "japan",
       "dev_ex_usa_ex_emu_ex_jp"]
TOLERANCE = 0.6


def _rows():
    with TAX.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _scope(rows):
    """Tickers held or targeted, plus everything flagged watchlist."""
    by_isin = {r["isin"]: r["ticker"] for r in rows if r["isin"]}
    held = set()
    for rel, col in PORTFOLIO_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get(col) or "").strip()
                if key:
                    # some files carry the ISIN in the ticker column
                    held.add(by_isin.get(key, key))
    watched = {r["ticker"] for r in rows if r["watchlist"] == "True"}
    return held | watched


def _carries_equity(row):
    return row["exp_equities"] not in ("", "0.0") or row["asset_class"] == "Equities"


def geo_problem(row):
    """None when the five geo columns are a usable allocation."""
    values = [row[g] for g in GEO]
    if all(v == "" for v in values):
        return "geo empty"
    if any(v == "" for v in values):
        filled = [g for g in GEO if row[g] != ""]
        return f"geo partial (only {', '.join(filled)})"
    total = sum(float(v) for v in values)
    if abs(total - 100.0) > TOLERANCE:
        return f"geo sums to {total:.1f}, not 100"
    return None


def main():
    rows = _rows()
    scope = _scope(rows) if "--all" not in sys.argv else {r["ticker"] for r in rows}
    problems = []
    for row in sorted(rows, key=lambda r: r["ticker"]):
        if row["ticker"] not in scope:
            continue
        if not row["ter"]:
            problems.append((row["ticker"], "TER missing"))
        if _carries_equity(row) and (issue := geo_problem(row)):
            problems.append((row["ticker"], issue))

    in_scope = sum(1 for r in rows if r["ticker"] in scope)
    print(f"{in_scope} instruments in scope, {len(problems)} problems")
    for ticker, issue in problems:
        print(f"  {ticker:8} {issue}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
