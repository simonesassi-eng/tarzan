#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the instrument taxonomy coverage across every Tarzan instrument input.

The curated catalog ``input/instrument_taxonomy.csv`` is the source of truth
for asset_class + role (see tarzan/data/enricher.py::_apply_taxonomy_override).
Any instrument that appears in an input file but is NOT in the catalog falls
back to auto-classification (yfinance/keyword/OpenFIGI) with no curated role.

This script scans all instrument inputs and prints the ones missing from the
catalog — a TODO list to keep the taxonomy complete.

Usage:
    python3 scripts/audit_taxonomy.py            # report, exit 0
    python3 scripts/audit_taxonomy.py --strict   # exit 1 if any are missing
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
CATALOG = INPUT / "instrument_taxonomy.csv"

# Instrument inputs to audit: (filename, isin_col, ticker_col, name_col)
INPUTS = [
    ("order_list.csv", "isin", "ticker", "name"),
    ("portfolio_test.csv", "isin", "ticker", "portfolio_name"),
    ("targets_per_holding.csv", "isin", "ticker", "name"),
]


def _norm(s: str) -> str:
    return (s or "").strip().upper()


def _bare(ticker: str) -> str:
    return _norm(ticker).split(".")[0]


def _load_catalog_keys() -> set[str]:
    """ISIN + bare-ticker keys of rows that carry a curated asset_class."""
    keys: set[str] = set()
    if not CATALOG.exists():
        return keys
    with open(CATALOG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("asset_class") or "").strip():
                continue
            if _norm(r.get("isin")):
                keys.add(_norm(r.get("isin")))
            if _bare(r.get("ticker")):
                keys.add(_bare(r.get("ticker")))
    return keys


def _rows(path: Path, isin_col: str, ticker_col: str, name_col: str):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = {c.lower(): c for c in (rd.fieldnames or [])}
        for r in rd:
            def g(col):
                real = cols.get(col)
                return r.get(real, "") if real else ""
            isin, tk, nm = g(isin_col), g(ticker_col), g(name_col)
            if _norm(isin) or _bare(tk):
                yield _norm(isin), _bare(tk), (nm or "").strip()


def _held_rows():
    """order_list.csv carries the full order history; audit only the instruments
    actually held today (net quantity > 0). Returns (isin, bare_ticker, name)
    tuples, or None if the orders builder is unavailable (falls back to raw rows).
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from tarzan.data.loader import load_orders
        from tarzan.engine.returns_builder import build_holdings_from_orders
        holdings = build_holdings_from_orders(load_orders(str(INPUT / "order_list.csv")))
        return [(_norm(h.isin), _bare(h.ticker), (h.name or "").strip()) for h in holdings]
    except Exception:
        return None


def main() -> int:
    strict = "--strict" in sys.argv
    keys = _load_catalog_keys()
    print(f"Catalog: {CATALOG.name} — {len(keys)} curated keys (ISIN + ticker)\n")

    total_missing = 0
    for fname, ic, tc, nc in INPUTS:
        path = INPUT / fname
        if not path.exists():
            print(f"  [skip] {fname} (not found)")
            continue
        seen, missing = set(), []
        held = _held_rows() if fname == "order_list.csv" else None
        rows_iter = held if held is not None else _rows(path, ic, tc, nc)
        label = f"{fname} (held today)" if held is not None else fname
        for isin, bare, nm in rows_iter:
            ident = isin or bare
            if ident in seen:
                continue
            seen.add(ident)
            if not ((isin and isin in keys) or (bare and bare in keys)):
                missing.append((isin, bare, nm))
        status = "OK" if not missing else f"{len(missing)} MISSING"
        print(f"== {label}: {len(seen)} instruments, {status} ==")
        for isin, bare, nm in missing:
            print(f"   - {isin or '-':14s} {bare or '-':10s} {nm[:48]}")
        total_missing += len(missing)
        print()

    if total_missing:
        print(f"TOTAL missing from catalog: {total_missing}. "
              f"Add rows to {CATALOG.name} (isin, ticker, asset_class, role).")
    else:
        print("All instrument inputs are fully covered by the catalog.")
    return 1 if (strict and total_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
