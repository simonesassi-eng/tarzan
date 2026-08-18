"""Load candidate portfolios and resolve/enrich their instruments.

Ticker↔ISIN are interchangeable inputs. Resolution reuses Tarzan's canonical
infrastructure rather than a private copy:

  * ISIN validation → :func:`tarzan.contracts.validation.isin_format_error`;
  * exchange-suffix probing → :func:`tarzan.config.isin_exchange_suffixes`;
  * enrichment (price, class/geo breakdown, TER) → :func:`enrich_holdings`;
  * ticker→ISIN learning cache → :mod:`tarzan.data.price_cache`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tarzan.contracts.validation import isin_format_error
from tarzan.data import geo_resolver, price_cache
from tarzan.data.enricher import enrich_holdings
from tarzan.models.holding import Holding

from tarzan.backtest.model import WhatIfItem, composition_for

logger = logging.getLogger("backtest.loader")

ROOT = Path(__file__).resolve().parents[2]

# Curated Fama-French (Developed) factor loadings for SHORT-HISTORY factor ETFs.
# Part of the script's own reference data (NOT a user input): used only as the
# fallback tilt when a fund has too little real history (~24 months) to regress
# its own, so the pre-inception past is simulated as
#   base(geo/market) + smb·SMB + hml·HML + rmw·RMW  on the real Ken French
# Developed factor legs (see ``backtest.engine.portfolio_long_returns``). Funds
# with enough real history keep their REGRESSED loadings; this table is ignored.
#
# Values are ESTIMATES of each strategy's TARGET exposure (Avantis/DFA
# methodology + published RR-community regressions of the US-listed siblings
# AVUV/AVDV/DFSVX), not a promise. MOM is omitted (≈0): Avantis screens to stay
# momentum-neutral, unlike passive ZPRV/ZPRX which run negative. Emerging-market
# funds (AVEM) are intentionally absent — Developed factors are the wrong
# regressors for EM.
CURATED_FACTOR_LOADINGS: dict[str, dict[str, float]] = {
    "AVWS": {"SMB": 0.80, "HML": 0.35, "RMW": 0.25},   # global dev small-cap value
    "AVWC": {"SMB": 0.15, "HML": 0.25, "RMW": 0.25},   # global dev all-cap value
    "AVUS": {"SMB": 0.15, "HML": 0.25, "RMW": 0.25},   # US all-cap value
    "AVEU": {"SMB": 0.15, "HML": 0.25, "RMW": 0.25},   # Europe all-cap value
    "AVPE": {"SMB": 0.20, "HML": 0.25, "RMW": 0.25},   # Pacific all-cap value
}


def curated_factor_loadings() -> dict[str, dict]:
    """Curated FF-Developed factor loadings keyed by bare ticker (see
    :data:`CURATED_FACTOR_LOADINGS`)."""
    return CURATED_FACTOR_LOADINGS


def is_isin(s: str) -> bool:
    """True if ``s`` is a structurally valid ISIN (delegates to contracts)."""
    return bool(s) and isin_format_error(s.strip().upper()) is None


def load_portfolios(path: Path) -> list[tuple[str, list[tuple[str, float, str]]]]:
    """Load N portfolios from the weights CSV (long or wide layout).

    * **Long (tidy)** — ``portfolio_name, ticker, target_portfolio``: one row
      per holding; portfolios are the distinct ``portfolio_name`` values.
    * **Wide** — ``ticker`` (+ optional ``isin``) + one weight column per
      portfolio.

    Weights are percentages; blank/≤0 means "not held in that portfolio". Each
    row is returned as ``(ticker, weight, isin)`` (isin may be "").
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    tcol = lower.get("ticker")
    if tcol is None:
        raise ValueError("weights CSV needs a 'ticker' column")
    icol = lower.get("isin")

    pcol = lower.get("portfolio_name") or lower.get("portfolio")
    if pcol is not None:
        wcol = (lower.get("target_portfolio") or lower.get("weight")
                or lower.get("target"))
        if wcol is None:
            others = [c for c in df.columns if c not in (tcol, pcol, icol)]
            if not others:
                raise ValueError("long-format CSV needs a weight column")
            wcol = others[0]
        order: list[str] = []
        groups: dict[str, list[tuple[str, float, str]]] = {}
        for _, r in df.iterrows():
            name = str(r[pcol]).strip()
            tkr = str(r[tcol]).strip()
            isin = str(r[icol]).strip() if icol else ""
            if isin.lower() == "nan":
                isin = ""
            if not isin and is_isin(tkr):
                isin = tkr
            try:
                w = float(r[wcol])
            except (TypeError, ValueError):
                continue
            if not name or not tkr or w <= 0:
                continue
            if name not in groups:
                groups[name] = []
                order.append(name)
            groups[name].append((tkr, w, isin))
        return [(n, groups[n]) for n in order]

    weight_cols = [c for c in df.columns if c not in (tcol, icol)]
    if not weight_cols:
        raise ValueError("weights CSV needs at least one portfolio (weight) column")
    portfolios: list[tuple[str, list[tuple[str, float, str]]]] = []
    for wc in weight_cols:
        rows: list[tuple[str, float, str]] = []
        for _, r in df.iterrows():
            tkr = str(r[tcol]).strip()
            isin = str(r[icol]).strip() if icol else ""
            if isin.lower() == "nan":
                isin = ""
            if not isin and is_isin(tkr):
                isin = tkr
            try:
                w = float(r[wc])
            except (TypeError, ValueError):
                continue
            if tkr and w > 0:
                rows.append((tkr, w, isin))
        if rows:
            name = "What-if" if wc.lower() == "target_portfolio" else wc
            portfolios.append((name, rows))
    return portfolios


def _clean(v) -> str:
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def build_symbol_map() -> dict[str, tuple[Optional[str], str]]:
    """bare ticker → (isin, full yfinance symbol), from the reference CSVs.

    Sourced from ``targets_per_holding.csv`` and ``instrument_taxonomy.csv``
    (both carry the suffixed ticker AND, for most rows, the ISIN), keyed by the
    bare ticker. Providing the ISIN is what lets the backtest enrich through the
    SAME robust, ISIN-driven resolution the live portfolio uses
    (``enricher._resolve_isin``: suffix sweep + ranking + history validation),
    instead of a private, weaker ticker probe.
    """
    sym_map: dict[str, tuple[Optional[str], str]] = {}
    for name in ("targets_per_holding.csv", "instrument_taxonomy.csv"):
        p = ROOT / "input" / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, r in df.iterrows():
            full = _clean(r.get("ticker", ""))
            if not full:
                continue
            isin = _clean(r.get("isin", "")) or None
            bare = full.split(".")[0].upper()
            prev = sym_map.get(bare)
            if prev is None:
                sym_map[bare] = (isin, full)
                continue
            # Merge: keep any known ISIN, and prefer a SUFFIXED symbol (e.g.
            # CL2.MI) over a bare one (CL2), which resolves far better for a
            # non-US listing.
            merged_isin = prev[0] or isin
            merged_sym = prev[1]
            if "." not in merged_sym and "." in full:
                merged_sym = full
            sym_map[bare] = (merged_isin, merged_sym)
    return sym_map


_GEO_BY_KEY: Optional[dict] = None


def _taxonomy_geo(bare: str, isin: str):
    """Authoritative equity geography from the taxonomy geo columns, keyed by
    BARE ticker or ISIN. Reuses the canonical ``geo_resolver._parse_geo_row``
    parser (same ``exp_*``/geo columns the enricher uses) so index/swap ETFs
    get their real regional split instead of the listing country. Cached.
    """
    global _GEO_BY_KEY
    if _GEO_BY_KEY is None:
        _GEO_BY_KEY = {}
        idx = ROOT / "input" / "instrument_taxonomy.csv"
        if idx.exists():
            df = pd.read_csv(idx)
            df.columns = [c.strip().lower() for c in df.columns]
            for _, r in df.iterrows():
                geo = geo_resolver._parse_geo_row(r)
                if not geo:
                    continue
                full = str(r.get("ticker", "")).strip()
                rin = str(r.get("isin", "")).strip()
                if full:
                    _GEO_BY_KEY[full.split(".")[0].upper()] = geo
                if rin and rin.lower() != "nan":
                    _GEO_BY_KEY[rin.upper()] = geo
    key = (bare or "").split(".")[0].upper()
    if key in _GEO_BY_KEY:
        return _GEO_BY_KEY[key]
    if isin and isin.lower() != "nan" and isin.upper() in _GEO_BY_KEY:
        return _GEO_BY_KEY[isin.upper()]
    return None


def resolve_symbol(bare: str, sym_map: dict) -> tuple[Optional[str], str]:
    """Resolve a bare ticker (or ISIN) to (isin, yfinance symbol) WITHOUT any
    network probe — the enricher does the actual resolution.

    An ISIN identifies the instrument directly. A ticker known to the reference
    maps yields its (isin, suffixed symbol) so enrichment goes through the
    robust ISIN path. An unknown ticker is passed through as-is for the
    enricher's own ticker fallback (no private suffix guessing here).
    """
    if is_isin(bare):
        return bare.strip().upper(), bare.strip().upper()
    key = bare.split(".")[0].upper()
    if key in sym_map:
        return sym_map[key]
    return None, bare


def _symbol_from_holding(h: Holding) -> str:
    """Best yfinance symbol for an enriched holding (from its data_source)."""
    ds = h.data_source or ""
    if ds.startswith("yfinance:"):
        return ds.split(":", 1)[1]
    return h.ticker


def enrich_universe(portfolios_raw, sym_map) -> dict[str, Optional[Holding]]:
    """Resolve + enrich every unique instrument across all portfolios once.

    A ticker-only candidate is resolved to its ISIN via the reference maps so
    enrichment uses the SAME robust ISIN path as the live portfolio. Enriched
    holdings are read back by OBJECT identity (``enrich_holdings`` mutates each
    Holding in place), so a resolved-symbol rewrite can't de-align the mapping.
    """
    resolved: dict[str, tuple[Optional[str], str]] = {}
    for _, rows in portfolios_raw:
        for tkr, _w, isin in rows:
            if isin:
                rkey = isin.upper()
                resolved.setdefault(rkey, (isin.upper(), isin.upper()))
            else:
                rkey = tkr.split(".")[0].upper()
                resolved.setdefault(rkey, resolve_symbol(tkr, sym_map))

    built: dict[str, Holding] = {
        rkey: Holding(isin=isin or "", ticker=symbol, quantity=1.0,
                      cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR")
        for rkey, (isin, symbol) in resolved.items()
    }
    logger.info("Enriching %d unique instrument(s) across %d portfolio(s)...",
                len(built), len(portfolios_raw))
    enrich_holdings(list(built.values()))  # mutates each Holding in place
    return built


def portfolio_items(rows, universe) -> list[WhatIfItem]:
    """Build items for one portfolio from the shared enriched universe."""
    items: list[WhatIfItem] = []
    for tkr, w, isin in rows:
        rkey = isin.upper() if isin else tkr.split(".")[0].upper()
        h = universe.get(rkey)
        if h is None:
            logger.warning("No enriched holding for %s; skipping", tkr)
            continue
        sym = h.ticker
        disp = tkr.split(".")[0]
        if is_isin(disp):
            s2 = _symbol_from_holding(h)
            if s2 and not is_isin(s2.split(".")[0]):
                disp = s2.split(".")[0]
        comp_notl, comp_cap, why = composition_for(h)
        # Ticker ⇄ ISIN interchangeability: learn any ISIN we have so a later
        # ticker-only run resolves for free; else try to resolve from ticker.
        final_isin = (isin or h.isin or "").strip()
        if final_isin:
            price_cache.store_ticker_isin(disp, final_isin)
        else:
            final_isin = (geo_resolver.resolve_isin(sym)
                          or geo_resolver.resolve_isin(disp) or "")
        # Authoritative equity geography from the taxonomy geo columns, keyed
        # by the BARE ticker / original ISIN (robust to enrichment mangling the
        # symbol/ISIN when offline).
        geo_tax = _taxonomy_geo(disp, isin or final_isin)
        items.append(WhatIfItem(disp, sym, final_isin, w, h,
                                comp_notl, comp_cap, why, geo_taxonomy=geo_tax))
    return items
