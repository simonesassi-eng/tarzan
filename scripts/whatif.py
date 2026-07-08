"""What-if analysis comparing one or more portfolios, side by side.

Ad-hoc companion to the main pipeline. It evaluates every portfolio in a
single view — **portfolios are columns**, metrics are rows — reusing
Tarzan's existing engines without touching the core:

  * enrichment (``enrich_holdings``)  → live price, price history, asset
    class, geography breakdown, TER/yield;
  * metrics (``robustness.full_metrics`` on one aligned synthetic NAV)
    → returns + risk factors (CAGR, volatility, Sharpe, Sortino,
    VaR/CVaR, Beta/Alpha) plus rolling / stress / Monte-Carlo robustness;
  * config (``InvestorConfig``)        → target asset & equity-geo mix.

Every portfolio comes from the weights CSV — nothing is added implicitly.
The order list (if present) is used only to derive the EUR anchor for the
notional figures; pass ``--notional`` to set it explicitly instead.
The weights CSV can be in either layout:

  * long (tidy): ``portfolio_name, ticker, target_portfolio`` — one row
    per holding, portfolios are the distinct ``portfolio_name`` values;
  * wide: ``ticker`` + one weight column per portfolio.

More portfolios are added just by adding rows (long) or columns (wide):

    portfolio_name,ticker,target_portfolio
    CL2+NTSG,NTSG,40
    CL2+NTSG,CL2,10
    ...

Two things this script adds on top, both required by leveraged / mixed
instruments the single-asset-class model cannot express:

  1. **Automatic look-through composition** (no extra input file): each
     instrument is split across asset classes by inferring its make-up
     from name + ``instrument_taxonomy.csv`` description + classification. It yields
     two vectors — notional (economic exposure, leverage-aware) and
     funded capital (where the euros sit, synthetic legs → collateral
     cash).
  2. **Three allocation views**: funded capital, notional exposure, and
     equity geography, each compared to the target.

EUR figures are anchored to the real portfolio's invested value (ex-cash).

Usage:
    python -m scripts.whatif --weights input/portfolio_test.csv \
        --config input/targets.csv --orders input/order_list.csv --excel
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.data.enricher import enrich_holdings, _fetch_ticker_info  # noqa: E402
from tarzan.data.loader import load_config, load_orders  # noqa: E402
from tarzan.data import proxy_data  # noqa: E402
from tarzan.engine.returns_builder import build_holdings_from_orders  # noqa: E402
from tarzan.engine import robustness as rob  # noqa: E402
from tarzan.engine import synthetic as syn  # noqa: E402
from tarzan.export._format import short_instrument_name  # noqa: E402
from tarzan.models.holding import AssetClass, Geography, Holding  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("whatif")

# Asset-class labels (identical to AssetClass.value so they match the
# target keys parsed by InvestorConfig).
EQ = AssetClass.EQUITIES.value
FI = AssetClass.FIXED_INCOME.value
GOLD = AssetClass.GOLD.value
COMM = AssetClass.COMMODITIES.value
CRYPTO = AssetClass.CRYPTO.value
ALT = AssetClass.ALTERNATIVE.value
CASH = AssetClass.CASH_EQUIVALENTS.value

ASSET_ORDER = [EQ, FI, GOLD, COMM, CRYPTO, ALT, CASH]
GEO_ORDER = [g.value for g in Geography]

# Rows of the returns & risk block: (label, metric key, source, unit).
RISK_ROWS = [
    ("CAGR", "cagr", "perf", "%"),
    ("Return 1Y", "1y", "perf", "%"),
    ("Return 3Y", "3y", "perf", "%"),
    ("Volatility (ann.)", "volatility", "risk", "%"),
    ("Sharpe", "sharpe", "risk", ""),
    ("Sortino", "sortino", "risk", ""),
    ("Max Drawdown", "max_drawdown", "risk", "%"),
    ("VaR 95% (daily)", "var_95", "risk", "%"),
    ("CVaR 95% (daily)", "cvar_95", "risk", "%"),
    ("Beta vs S&P 500", "beta", "risk", ""),
    ("Alpha (ann.)", "alpha", "risk", "%"),
]

# Candidate exchange suffixes probed when an ISIN is unknown.
_SUFFIX_PROBES = (".MI", ".DE", ".AS", ".L", ".PA", "")


# ---------------------------------------------------------------------------
# Inputs & resolution
# ---------------------------------------------------------------------------

def _load_portfolios(path: Path) -> list[tuple[str, list[tuple[str, float]]]]:
    """Load N portfolios from the weights CSV. Two layouts are accepted:

    * **Long (tidy)** — ``portfolio_name, ticker, target_portfolio``: one
      row per holding; portfolios are the distinct ``portfolio_name``
      values (kept in first-seen order). Preferred when you have many
      portfolios.
    * **Wide** — ``ticker`` + one weight column *per portfolio*: the column
      header is the portfolio name (``target_portfolio`` → "What-if").

    Weights are percentages; blank/≤0 means "not held in that portfolio".
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
        # Long format: group rows by portfolio name.
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
            # A ticker cell that is itself an ISIN doubles as the isin.
            if not isin and _looks_like_isin(tkr):
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

    # Wide format: every non-ticker/isin column is a portfolio.
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
            if not isin and _looks_like_isin(tkr):
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


def _build_reference_maps() -> tuple[dict, dict]:
    """ticker → (isin, full symbol) and symbol/base → description maps.

    Sourced from ``targets_per_holding.csv`` (ISIN + suffixed ticker) and
    ``instrument_taxonomy.csv`` (suffixed ticker + a description the composition
    inference parses). Keyed by the bare ticker so a plain-ticker weights
    file matches a suffixed listing.
    """
    sym_map: dict[str, tuple[Optional[str], str]] = {}
    desc_map: dict[str, str] = {}

    tph = ROOT / "input" / "targets_per_holding.csv"
    if tph.exists():
        df = pd.read_csv(tph)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, r in df.iterrows():
            full = str(r.get("ticker", "")).strip()
            isin = str(r.get("isin", "")).strip() or None
            if full:
                sym_map.setdefault(full.split(".")[0].upper(), (isin, full))

    idx = ROOT / "input" / "instrument_taxonomy.csv"
    if idx.exists():
        df = pd.read_csv(idx)
        df.columns = [c.strip().lower() for c in df.columns]
        for _, r in df.iterrows():
            full = str(r.get("ticker", "")).strip()
            desc = str(r.get("description", "")).strip()
            if not full:
                continue
            base = full.split(".")[0].upper()
            sym_map.setdefault(base, (None, full))
            if desc and desc.lower() != "nan":
                desc_map[base] = desc
                desc_map[full.upper()] = desc
    return sym_map, desc_map


def _looks_like_isin(s: str) -> bool:
    """True if ``s`` looks like a 12-char ISIN (2 letters + 9 alnum + digit)."""
    s = s.strip().upper()
    return bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", s))


def _resolve_symbol(bare: str, sym_map: dict) -> tuple[Optional[str], str]:
    """Resolve a bare ticker (or ISIN) to (isin, yfinance symbol)."""
    # An ISIN identifies the instrument directly: let the enricher resolve it
    # (same deterministic path the real portfolio uses), so current-portfolio
    # bonds without a clean ticker still enrich.
    if _looks_like_isin(bare):
        return bare.strip().upper(), bare.strip().upper()
    key = bare.split(".")[0].upper()
    if key in sym_map:
        return sym_map[key]
    for suffix in _SUFFIX_PROBES:
        cand = f"{bare}{suffix}" if suffix else bare
        info = _fetch_ticker_info(cand)
        if info.get("regularMarketPrice") or info.get("previousClose"):
            logger.info("Resolved %s → %s (suffix probe)", bare, cand)
            return None, cand
    logger.warning("Could not resolve a priced symbol for %s; using %s.MI", bare, bare)
    return None, f"{bare}.MI"


def _real_portfolio(orders_path: Path) -> tuple[Optional[float], list[Holding]]:
    """Real portfolio: (invested value ex-cash, enriched holdings)."""
    if not orders_path.exists():
        return None, []
    orders = load_orders(str(orders_path))
    if not orders:
        return None, []
    holdings = enrich_holdings(build_holdings_from_orders(orders))
    invested = sum(
        float(h.current_value or 0.0)
        for h in holdings
        if h.asset_class != AssetClass.CASH_EQUIVALENTS
    )
    return (invested or None), holdings


def _symbol_from_holding(h: Holding) -> str:
    """Best yfinance symbol for an enriched holding (from its data_source)."""
    ds = h.data_source or ""
    if ds.startswith("yfinance:"):
        return ds.split(":", 1)[1]
    return h.ticker


# ---------------------------------------------------------------------------
# Automatic look-through composition (multi-asset split + leverage)
# ---------------------------------------------------------------------------

def infer_composition(
    holding: Holding, desc: str = ""
) -> tuple[dict[str, float], dict[str, float], str]:
    """Infer an instrument's composition as (notional, capital, explanation).

      * **notional** — economic exposure, leverage-aware (may sum > 100),
        e.g. Efficient Core 90/60 → ``{Equities: 90, Fixed Income: 60}``.
      * **capital** — where the euros sit, always sums to 100; a synthetic
        overlay leg consumes no capital, so the remainder is collateral
        **cash** (90/60 → ``{Equities: 90, Cash: 10}``); a 2x ETF is funded
        1x in the underlying class (``{Equities: 100}``).

    Heuristic; the explanation is surfaced so splits can be verified.
    """
    text = " ".join(t for t in (holding.name, desc, holding.instrument_type) if t).lower()
    base = holding.asset_class.value if holding.asset_class else ALT

    # Priority — curated role from instrument_taxonomy.csv (set on the holding
    # by the enricher's taxonomy override). Deterministic; replaces the old
    # hardcoded ISIN list for price-less bonds and the fragile text heuristics.
    role = holding.role
    if role:
        if role == "Efficient Core":
            mm = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
            eq, fi = (float(mm.group(1)), float(mm.group(2))) if mm else (90.0, 60.0)
            return {EQ: eq, FI: fi}, {EQ: eq, CASH: max(0.0, 100.0 - eq)}, f"role: efficient core {eq:g}/{fi:g}"
        if role == "Multi-Asset":
            m = re.search(r"(\d{2})", text)
            eq = float(m.group(1)) if m else 60.0
            notl = {EQ: eq, FI: 100.0 - eq}
            return notl, dict(notl), f"role: multi-asset {eq:g}/{100.0 - eq:g}"
        if role == "Equity Leveraged":
            lm = re.search(r"(\d(?:\.\d)?)\s*x\b", text)
            lev = float(lm.group(1)) if lm else 2.0
            return {EQ: lev * 100.0}, {EQ: 100.0}, f"role: {lev:g}x leveraged equity"
        # All other roles map 1:1 to their canonical asset class (Carry/Market
        # Neutral → Commodities; Managed Futures/Tail/Cat Bond → Alternative;
        # Govt/Linkers/Aggregate/Long Duration → Fixed Income; Gold; Cash; etc.)
        return {base: 100.0}, {base: 100.0}, f"role: {role.lower()} → {base.lower()}"

    m = re.search(r"(\d{1,3})\s*%?\s*equit\w*.*?(\d{1,3})\s*%?\s*bond", text)
    if m:
        notl = {EQ: float(m.group(1)), FI: float(m.group(2))}
        return notl, _capital_from(notl), "name/desc equity+bond split"

    if "efficient core" in text or ("overlay" in text and "bond" in text):
        mm = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
        eq, fi = (float(mm.group(1)), float(mm.group(2))) if mm else (90.0, 60.0)
        why = "efficient-core ratio (desc)" if mm else "efficient-core default 90/60"
        return {EQ: eq, FI: fi}, {EQ: eq, CASH: max(0.0, 100.0 - eq)}, why

    m = re.search(r"lifestrategy\s*(\d{2})", text)
    if m:
        eq = float(m.group(1))
        notl = {EQ: eq, FI: 100.0 - eq}
        return notl, dict(notl), "lifestrategy ratio"

    # Alternative strategies (managed futures / CTA / long-short / carry /
    # market-neutral / trend). Their internal "Nx" is a long/short SPREAD, not
    # directional market leverage → 1x alternative. Checked BEFORE the leverage
    # rule so UEQC's "2.5x long/short carry" is not read as 2.5x commodity beta.
    if any(k in text for k in ("managed futures", "cta", "long/short", "long-short",
                               "carry", "market neutral", "market-neutral",
                               "relative value", "trend")):
        return {ALT: 100.0}, {ALT: 100.0}, "alternative strategy (1x)"

    # Daily-leveraged single-asset (e.g. "2x", "3x").
    if "leverag" in text or re.search(r"\b\d(?:\.\d)?\s*x\b", text):
        lm = re.search(r"(\d(?:\.\d)?)\s*x\b", text)
        lev = float(lm.group(1)) if lm else 2.0
        if lev > 1:
            cls = base if base in (EQ, FI, COMM, GOLD) else EQ
            return {cls: lev * 100.0}, {cls: 100.0}, f"{lev:g}x leveraged {cls.lower()}"

    # Bond-like fallback: an unpriced/unclassified instrument whose name or
    # type looks like a bond → fixed income (maps to the bond proxy, not cash).
    # Keywords include full issuer names, since price-less bonds carry a verbose
    # name rather than a ticker (e.g. "BUONI POLIENNALI DEL TESORO" = BTP,
    # "EUROPEAN INVESTMENT BANK" = EIB) — matching only "btp"/"eib" missed them.
    if base == ALT and any(k in text for k in (
            "bond", "treasury", "govt", "government", "gilt", "bund", "btp",
            "oat", "obligation", "obbligazion", "fixed income", "aggregate", "eib",
            "buoni poliennali", "poliennali", "tesoro", "investment bank",
            "schatz", "bono", "note", "notes")):
        return {FI: 100.0}, {FI: 100.0}, "bond-like → fixed income"

    return {base: 100.0}, {base: 100.0}, f"single class ({base.lower()})"


def _capital_from(notl: dict[str, float]) -> dict[str, float]:
    """Funded-capital vector from a notional one (levered → cash collateral)."""
    total = sum(notl.values())
    if total <= 100.0 + 1e-9:
        return dict(notl)
    funded = min(notl.get(EQ, 0.0), 100.0)
    return {EQ: funded, CASH: 100.0 - funded}


class WhatIfItem:
    """One resolved + enriched line of a portfolio."""

    def __init__(self, bare, symbol, isin, weight, holding, comp_notional,
                 comp_capital, explanation):
        self.bare = bare
        self.symbol = symbol
        self.isin = isin
        self.weight = weight
        self.holding = holding
        self.comp_notional = comp_notional
        self.comp_capital = comp_capital
        self.explanation = explanation

    @property
    def gross(self) -> float:
        return sum(self.comp_notional.values())

    @property
    def geo_breakdown(self):
        return self.holding.geo_breakdown


def compute_allocations(items: list[WhatIfItem]) -> dict:
    """Aggregate funded-capital and notional allocations (asset + equity geo)."""
    cap: dict[str, float] = {}
    notl: dict[str, float] = {}
    geo_cap: dict[str, float] = {}
    geo_notl: dict[str, float] = {}
    for it in items:
        for cls, v in it.comp_capital.items():
            cap[cls] = cap.get(cls, 0.0) + it.weight * v / 100.0
        for cls, v in it.comp_notional.items():
            notl[cls] = notl.get(cls, 0.0) + it.weight * v / 100.0
        eq_cap = it.weight * it.comp_capital.get(EQ, 0.0) / 100.0
        eq_notl = it.weight * it.comp_notional.get(EQ, 0.0) / 100.0
        gb = _clean_geo(it.geo_breakdown)
        for amount, dest in ((eq_cap, geo_cap), (eq_notl, geo_notl)):
            if amount <= 0:
                continue
            if gb:
                for g, gv in gb.items():
                    dest[g] = dest.get(g, 0.0) + amount * gv / 100.0
            else:
                dest["Other"] = dest.get("Other", 0.0) + amount
    return {"cap": cap, "notl": notl, "geo_cap": geo_cap, "geo_notl": geo_notl}


def _clean_geo(gb) -> Optional[dict]:
    """Drop the scraped 'Other' geography bucket (noise from yfinance
    top-holdings) and renormalise the real regions to 100%. Returns a
    ``{region_value: pct}`` dict, or None if no known region remains."""
    if not gb:
        return None
    known = {g.value: v for g, v in gb.items() if g != Geography.OTHER and v > 0}
    total = sum(known.values())
    if total <= 0:
        return None
    return {k: v * 100.0 / total for k, v in known.items()}


def _renorm(d: dict[str, float]) -> dict[str, float]:
    """Scale percent-of-portfolio values to sum to 100."""
    total = sum(d.values())
    if total <= 0:
        return dict(d)
    return {k: v * 100.0 / total for k, v in d.items()}


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

class Portfolio:
    """A named portfolio with allocations (normalised views) and risk metrics."""

    def __init__(self, name, items, alloc, metrics, is_real=False):
        self.name = name
        self.items = items
        self.alloc = alloc
        self.metrics = metrics
        self.is_real = is_real
        self.cap = _renorm(alloc["cap"])          # funded capital %, sums 100
        self.notl_gross = dict(alloc["notl"])     # notional %, sums to gross
        self.notl_mix = _renorm(alloc["notl"])    # notional mix %, sums 100
        self.geo_cap = _renorm(alloc["geo_cap"])  # equity geo (capital) %
        self.geo_notl = _renorm(alloc["geo_notl"])  # equity geo (notional) %
        # Leverage applied per class = notional exposure − funded capital
        # (both as % of capital). Isolates where/how much leverage sits;
        # sums to (gross − 100) = the leverage amount. Zero on unlevered legs.
        self.lev_by_class = {
            cls: self.notl_gross.get(cls, 0.0) - self.cap.get(cls, 0.0)
            for cls in set(self.notl_gross) | set(self.cap)
        }
        # Single aligned history (filled in later): synthetic per-instrument
        # spliced NAV clipped to the window shared by all portfolios, with
        # the merged return+risk metrics and the rolling/stress/MC robustness.
        self.synth_nav = None      # full-length synthetic NAV (before common clip)
        self.nav = None            # NAV clipped to the common aligned window
        self.window = None         # (start, end) Timestamps of the aligned window
        self.metrics_aligned: dict = {}
        self.rob: dict = {}

    @property
    def gross(self) -> float:
        return sum(self.alloc["notl"].values())

    @property
    def leverage(self) -> float:
        return self.gross / 100.0

    def weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for it in self.items:
            out[it.bare] = out.get(it.bare, 0.0) + it.weight
        return out


def _enrich_universe(portfolios_raw, sym_map) -> dict[str, Optional[Holding]]:
    """Resolve + enrich every unique instrument across all portfolios once,
    returning ``{resolution key: enriched Holding}``. When a row carries an
    ISIN it is authoritative (deterministic enrichment + reliable geo);
    otherwise the bare ticker is resolved."""
    resolved: dict[str, tuple[Optional[str], str]] = {}
    for _, rows in portfolios_raw:
        for tkr, _w, isin in rows:
            if isin:
                rkey = isin.upper()
                resolved.setdefault(rkey, (isin.upper(), isin.upper()))
            else:
                rkey = tkr.split(".")[0].upper()
                resolved.setdefault(rkey, _resolve_symbol(tkr, sym_map))

    holdings = [
        Holding(isin=isin or "", ticker=symbol, quantity=1.0,
                cost_basis_eur=0.0, market_value_eur=0.0, currency="EUR")
        for (isin, symbol) in resolved.values()
    ]
    logger.info("Enriching %d unique instrument(s) across %d portfolio(s)...",
                len(holdings), len(portfolios_raw))
    holdings = enrich_holdings(holdings)
    by_ticker = {h.ticker: h for h in holdings}
    return {rkey: by_ticker.get(sym) for rkey, (isin, sym) in resolved.items()}


def _portfolio_items(rows, universe, desc_map) -> list[WhatIfItem]:
    """Build items for one portfolio from the shared enriched universe."""
    items: list[WhatIfItem] = []
    for tkr, w, isin in rows:
        rkey = isin.upper() if isin else tkr.split(".")[0].upper()
        h = universe.get(rkey)
        if h is None:
            logger.warning("No enriched holding for %s; skipping", tkr)
            continue
        sym = h.ticker
        # Clean display ticker (no exchange suffix); if the row was keyed by
        # ISIN, prefer the resolved yfinance symbol base for display.
        disp = tkr.split(".")[0]
        if _looks_like_isin(disp):
            s2 = _symbol_from_holding(h)
            if s2 and not _looks_like_isin(s2.split(".")[0]):
                disp = s2.split(".")[0]
        desc = (desc_map.get(disp.upper(), "") or desc_map.get(sym.upper(), "")
                or desc_map.get((isin or "").upper(), ""))
        comp_notl, comp_cap, why = infer_composition(h, desc)
        items.append(WhatIfItem(disp, sym, isin or h.isin, w, h,
                                comp_notl, comp_cap, why))
    return items


def _current_rows(holdings, invested) -> list[tuple[str, float, str]]:
    """(ticker, weight%, isin) rows for the real portfolio, ex-cash.

    The ISIN is always emitted so geography/enrichment resolve reliably;
    the ticker is a clean display symbol (bond tickers fall back to ISIN).
    """
    rows: list[tuple[str, float, str]] = []
    for h in holdings:
        if h.asset_class == AssetClass.CASH_EQUIVALENTS:
            continue
        cv = float(h.current_value or 0.0)
        if cv <= 0 or invested <= 0:
            continue
        sym = _symbol_from_holding(h)
        base = sym.split(".")[0] if sym else ""
        if not base or _looks_like_isin(base):
            base = h.isin or base
        rows.append((base, round(cv / invested * 100.0, 2), h.isin or ""))
    return rows


def _refresh_current_in_csv(path: Path, rows: list[tuple[str, float, str]]) -> None:
    """Rewrite the CURRENT rows in the weights CSV (long format, with an
    ``isin`` column), preserving every other portfolio. Reads utf-8-sig so a
    BOM never hides existing rows."""
    import csv
    existing: list[tuple[str, str, str, str]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            low = {c.lower(): c for c in (r.fieldnames or [])}
            pcol = low.get("portfolio_name") or low.get("portfolio")
            tcol = low.get("ticker")
            wcol = low.get("target_portfolio") or low.get("weight")
            icol = low.get("isin")
            if pcol and tcol and wcol:
                for row in r:
                    if str(row[pcol]).strip().upper() != "CURRENT":
                        isin = str(row[icol]).strip() if icol else ""
                        existing.append((row[pcol].strip(), row[tcol].strip(),
                                         str(row[wcol]).strip(),
                                         "" if isin.lower() == "nan" else isin))
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["portfolio_name", "ticker", "target_portfolio", "isin"])
        for p, t, wt, isin in existing:
            wr.writerow([p, t, wt, isin])
        for token, wt, isin in rows:
            wr.writerow(["CURRENT", token, wt, isin])


def _scaled_holdings(items, anchor) -> list[Holding]:
    """Per-portfolio holding copies scaled to the weights, for the risk engine.

    Copies (not the shared universe holdings) so scaling one portfolio never
    corrupts another that references the same instrument.
    """
    out: list[Holding] = []
    for it in items:
        h = copy.copy(it.holding)
        cp = h.current_price
        if cp and cp > 0:
            h.quantity = anchor * it.weight / 100.0 / cp
            h.current_value = anchor * it.weight / 100.0
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# Robustness (rolling / stress / bootstrap) on actual + synthetic long history
# ---------------------------------------------------------------------------

def _instrument_exposures(item: WhatIfItem) -> dict:
    """Per-instrument exposure (fraction of the instrument's own value).

    Equity is split by the instrument's own geo breakdown; other classes
    from its notional composition. Gross may exceed 1 (a leveraged fund),
    which the replicator finances. Drives the instrument's synthetic base.
    """
    exp: dict[str, float] = {}
    eqf = item.comp_notional.get(EQ, 0.0) / 100.0
    gb = item.geo_breakdown
    if eqf > 0:
        # Keep only geo buckets ≥5% and renormalise: a noisy sub-5% sliver
        # (e.g. a scraped 4% "Other") would otherwise pull in a short-history
        # proxy (ACWI, 2008) and truncate the whole synthetic window.
        filt = {g: pct for g, pct in (gb or {}).items() if pct >= 5.0}
        tot = sum(filt.values())
        if tot > 0:
            for g, pct in filt.items():
                exp[g.value] = exp.get(g.value, 0.0) + eqf * pct / tot
        else:
            exp["Other"] = exp.get("Other", 0.0) + eqf
    for cls in (FI, GOLD, COMM, CRYPTO, ALT, CASH):
        v = item.comp_notional.get(cls, 0.0) / 100.0
        if v > 0:
            exp[cls] = exp.get(cls, 0.0) + v
    return exp


def _real_daily_returns(holding) -> Optional[pd.Series]:
    """Normalised daily returns from a holding's real price history."""
    ph = getattr(holding, "price_history", None)
    if ph is None or len(ph) < 2:
        return None
    r = ph.pct_change().dropna()
    idx = r.index
    r.index = (idx.tz_convert("UTC").tz_localize(None).normalize()
               if getattr(idx, "tz", None) is not None else idx.normalize())
    return r[~r.index.duplicated(keep="last")]


def _portfolio_long_returns(p: "Portfolio", proxies: dict, fin) -> pd.Series:
    """Portfolio long-history daily returns from PER-INSTRUMENT spliced series.

    Each instrument gets a long base (its composition replicated on index
    proxies, with its own leverage financed) spliced with its REAL returns
    where available, so recent history is the true fund (factor tilt and
    all) and the pre-inception tail is the modeled proxy. Instruments are
    then combined by capital weight (daily rebalanced).
    """
    cols: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for i, it in enumerate(p.items):
        base = syn.replicate_portfolio_returns(
            _instrument_exposures(it), proxies, financing_daily=fin, spread_annual=0.005)
        spliced = syn.splice_returns(base, _real_daily_returns(it.holding))
        if spliced is None or spliced.empty:
            continue
        key = f"i{i}"
        cols[key] = spliced
        weights[key] = it.weight / 100.0
    if not cols:
        return pd.Series(dtype=float)
    df = pd.DataFrame(cols).dropna(how="any")
    if df.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights)
    return (df * w).sum(axis=1)


def _compute_robustness(portfolios: list["Portfolio"]) -> None:
    """Build ONE aligned history for every portfolio and compute the merged
    return+risk metrics and rolling/stress/Monte-Carlo robustness on it.

    Each portfolio's synthetic per-instrument spliced NAV is clipped to the
    window shared by ALL portfolios (like testfol.io's single overlapping
    window), so Sharpe/CAGR/etc. are strictly comparable.
    """
    needed: set[str] = {"USA"}  # ^GSPC is also the beta/alpha benchmark
    for p in portfolios:
        for it in p.items:
            needed |= set(_instrument_exposures(it))
    logger.info("Fetching %d long-history proxy series for the aligned backtest...", len(needed))
    proxies, fin = proxy_data.proxy_returns_for(needed)

    # Per-instrument spliced synthetic NAV (real where available).
    for p in portfolios:
        p.synth_nav = syn.returns_to_price(_portfolio_long_returns(p, proxies, fin))

    navs = [p.synth_nav for p in portfolios
            if p.synth_nav is not None and not p.synth_nav.empty]
    if not navs:
        return
    common_start = max(n.index.min() for n in navs)
    common_end = min(n.index.max() for n in navs)

    bench_ret = proxies.get("USA")
    bench_px = syn.returns_to_price(bench_ret) if bench_ret is not None else None
    bench_w = (bench_px.loc[common_start:common_end]
               if bench_px is not None and not bench_px.empty else None)

    for p in portfolios:
        if p.synth_nav is None or p.synth_nav.empty:
            continue
        nav = p.synth_nav.loc[common_start:common_end]
        p.nav = nav
        p.window = (common_start, common_end)
        p.metrics_aligned = rob.full_metrics(nav, bench_w)
        p.rob = {
            "rolling1y": rob.rolling_return_distribution(nav, 252),
            "rolling3y": rob.rolling_return_distribution(nav, 252 * 3),
            "sharpe": rob.rolling_sharpe_range(nav, 252),
            "stress": rob.stress_scenarios(nav),
            "bootstrap": rob.block_bootstrap(nav),
        }


# ---------------------------------------------------------------------------
# Console reporting (portfolios as columns)
# ---------------------------------------------------------------------------

_COLW = 13
_LABW = 26


def _risk_str(metrics, key, src, unit) -> str:
    d = (metrics.performance if src == "perf" else metrics.risk) or {}
    v = d.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.2f}{unit}"


def _present(order, portfolios, attr) -> list[str]:
    """Labels from ``order`` that are non-zero in at least one portfolio."""
    labels = []
    for c in order:
        if any(getattr(p, attr).get(c, 0.0) for p in portfolios):
            labels.append(c)
    return labels


def _header(names, with_target=False) -> str:
    line = f"  {'':<{_LABW}}" + "".join(f"{n[:_COLW - 1]:>{_COLW}}" for n in names)
    if with_target:
        line += f"{'Target':>{_COLW}}"
    return line


def _matrix(title, labels, portfolios, attr, target=None) -> None:
    print("\n" + "-" * (2 + _LABW + _COLW * (len(portfolios) + (1 if target else 0))))
    print(title)
    names = [p.name for p in portfolios]
    print(_header(names, with_target=target is not None))
    for lbl in labels:
        row = f"  {lbl:<{_LABW}}"
        for p in portfolios:
            row += f"{getattr(p, attr).get(lbl, 0.0):>{_COLW - 1}.1f}%"
        if target is not None:
            row += f"{target.get(lbl, 0.0):>{_COLW - 1}.1f}%"
        print(row)


def _print_report(portfolios, asset_target, geo_target, anchor) -> None:
    names = [p.name for p in portfolios]
    width = 2 + _LABW + _COLW * len(names)

    print("\n" + "=" * width)
    print("WHAT-IF PORTFOLIO COMPARISON")
    print("=" * width)
    print(f"  Anchor (real invested, ex-cash): €{anchor:,.0f}")

    # --- Specs ---
    print("\nPORTFOLIO SPECS")
    print(_header(names))
    specs = [
        ("Instruments", lambda p: f"{len(p.items)}"),
        ("Gross exposure", lambda p: f"{p.gross:.0f}%"),
        ("Leverage", lambda p: f"{p.leverage:.2f}x"),
        ("Notional EUR", lambda p: f"{anchor * p.gross / 100.0:,.0f}"),
    ]
    for label, fn in specs:
        row = f"  {label:<{_LABW}}" + "".join(f"{fn(p):>{_COLW}}" for p in portfolios)
        print(row)

    # --- Instrument weights matrix (Ticker + Description + per-portfolio %) ---
    print("\nINSTRUMENT WEIGHTS (%)")
    tickers = sorted({it.bare for p in portfolios for it in p.items})
    name_by: dict[str, str] = {}
    for p in portfolios:
        for it in p.items:
            name_by.setdefault(it.bare, short_instrument_name(it.holding.name or it.symbol, 44))
    _TW, _DW = 9, 46
    print(f"  {'Ticker':<{_TW}}{'Description':<{_DW}}"
          + "".join(f"{n[:_COLW - 1]:>{_COLW}}" for n in names))
    wmaps = [p.weights() for p in portfolios]
    for tk in tickers:
        row = f"  {tk:<{_TW}}{name_by.get(tk, ''):<{_DW}}"
        for wm in wmaps:
            row += (f"{wm[tk]:>{_COLW - 1}.1f}%" if tk in wm else f"{'·':>{_COLW}}")
        print(row)

    _print_simulation_map(portfolios, names)

    # --- Allocation matrices ---
    _matrix("ASSET ALLOCATION — FUNDED CAPITAL (incl. cash)",
            _present(ASSET_ORDER, portfolios, "cap"), portfolios, "cap", asset_target)
    _matrix("ASSET ALLOCATION — NOTIONAL EXPOSURE (mix, normalised)",
            _present(ASSET_ORDER, portfolios, "notl_mix"), portfolios, "notl_mix", asset_target)
    # gross exposure per class (leveraged, non-normalised)
    _matrix("ASSET ALLOCATION — NOTIONAL EXPOSURE (gross % of capital, leveraged)",
            _present(ASSET_ORDER, portfolios, "notl_gross"), portfolios, "notl_gross")
    _matrix("LEVERAGE BY CLASS (notional − funded, pp of capital)",
            _present(ASSET_ORDER, portfolios, "lev_by_class"), portfolios, "lev_by_class")
    _matrix("EQUITY GEOGRAPHY — NOTIONAL (% of equity sleeve)",
            _present(GEO_ORDER, portfolios, "geo_notl"), portfolios, "geo_notl", geo_target)
    _matrix("EQUITY GEOGRAPHY — CAPITAL (% of equity sleeve)",
            _present(GEO_ORDER, portfolios, "geo_cap"), portfolios, "geo_cap", geo_target)

    _print_metrics(portfolios)
    _print_robustness(portfolios)


# ---------------------------------------------------------------------------
# testfol.io proxy expansion (from KB section 10C) — lines to paste on the site
# ---------------------------------------------------------------------------
# ("one", TICKER): 1:1 proxy. ("lev2", EXPR): native daily-leverage ticker.
# ("ec_world"/"ec_eur",): efficient-core 90/60 expanded into equity+bond−CASHX.
_TESTFOL_PROXY = {
    # Leveraged UCITS: params follow the Rational Reminder deep-dive. USA 2x
    # (CL2) tracks close to TER (~0.5); World 2x (LVWC/Amumbo) has a much higher
    # real cost (~1.6%) once gross-to-net div drag is doubled by leverage.
    # SW=1.1/SP=0.4 are testfol's default financing spread (made explicit).
    "CL2": ("lev2", "SPYSIM?L=2&E=0.5&SW=1.1&SP=0.4"),   # Amundi MSCI USA 2x
    "LVWC": ("lev2", "VTSIM?L=2&E=1.60&SW=1.1&SP=0.4"),  # Amundi MSCI World 2x (RR cost)
    "NTSG": ("ec_world",),                          # WT Global Efficient Core 90/60
    "NTSZ": ("ec_eur",),                            # WT Eurozone Efficient Core 90/60
    "NTSX": ("ec_world",),
    "EXUS": ("one", "EFASIM"),                      # dev ex-US
    "SWDA": ("one", "VTSIM"), "VWCE": ("one", "VTSIM"), "FWRA": ("one", "VTSIM"),
    "XDEM": ("one", "MTUM"), "IWMO": ("one", "MTUM"),     # momentum (US proxy)
    "XDEQ": ("one", "QUAL"), "IWQU": ("one", "QUAL"),     # quality (US proxy)
    "XDEV": ("one", "VLUE"), "IWVL": ("one", "VLUE"),     # value (US proxy)
    "ZPRV": ("one", "AVUV"), "ZPRX": ("one", "AVDV"),     # small value US / dev
    "SGLD": ("one", "GLDSIM"), "WGLD": ("one", "GLDSIM"),
    # Managed futures (trend): blend DBMFSIM + KMLMSIM 50/50. The community
    # (RR / TBTF) notes large dispersion between the two trend SIMs, so half-
    # and-half reduces single-model risk (both are long-history, ~1992+).
    "MFEH": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    "DBMFE": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    "DBMF": ("blend", [("DBMFSIM", 0.5), ("KMLMSIM", 0.5)]),
    # Commodity carry: there is NO ready testfol ticker (carry ≠ trend, using a
    # trend SIM "ti porta fuori strada" per ILS). Correct method is a custom CSV
    # (see tab note). DBMFSIM is kept only as a rough shareable stand-in.
    "UEQC": ("carry", "DBMFSIM"), "CRRY": ("carry", "DBMFSIM"), "CRRE": ("carry", "DBMFSIM"),
    "XMME": ("one", "EEM"), "AVEM": ("one", "EEM"), "EM710": ("one", "EEM"),
    "XMJP": ("one", "EWJ"),
    "XESC": ("one", "EZU"),
    "XSX6": ("one", "IEUR"),
    "AGGH": ("one", "BNDW"),
    "XGIN": ("one", "TIP"),
    "XS5E": ("one", "SPYSIM"),
    "LMTH": ("one", "TLT"), "X710": ("one", "IEFSIM"), "X15E": ("one", "IEFSIM"),
    "CATB": ("one", "CASHX"),
}

# Fallback testfol proxy by dominant asset class when a ticker is unmapped.
_TESTFOL_FALLBACK = {EQ: "VTSIM", FI: "TLT", GOLD: "GLDSIM", COMM: "DBC",
                     ALT: "KMLMSIM", CRYPTO: "BTC-USD", CASH: "CASHX"}

# Approximate annual TER (%) per instrument, applied as testfol `E=` drag.
# CL2/LVWC already carry E= in their leveraged expression, so are omitted here.
_TESTFOL_TER = {
    "NTSG": 0.35, "NTSZ": 0.35, "NTSX": 0.20,
    "XDEM": 0.25, "XDEQ": 0.25, "XDEV": 0.25, "IWMO": 0.30, "IWQU": 0.30, "IWVL": 0.30,
    "EXUS": 0.15, "SWDA": 0.12, "VWCE": 0.22, "FWRA": 0.15,
    "XMME": 0.18, "AVEM": 0.33, "EM710": 0.20, "XMJP": 0.12, "XESC": 0.09, "XSX6": 0.12,
    "ZPRV": 0.30, "ZPRX": 0.30,
    "AGGH": 0.10, "XGIN": 0.25, "X15E": 0.15, "X710": 0.15, "LMTH": 0.14,
    "SGLD": 0.12, "WGLD": 0.15,
    "MFEH": 0.75, "DBMFE": 0.75, "DBMF": 0.85,   # MFEH/DBMFE = iMGP DBi UCITS 0.75; DBMF = US ETF 0.85
    "UEQC": 0.34, "CRRY": 0.45, "CRRE": 0.45,
}
_TESTFOL_TER_FALLBACK = {EQ: 0.20, FI: 0.15, GOLD: 0.15, COMM: 0.40,
                         ALT: 0.90, CRYPTO: 0.50, CASH: 0.0}


def _instrument_ter(it: "WhatIfItem") -> float:
    """Best TER estimate (%) for an instrument: map → real holding.ter → class."""
    t = _TESTFOL_TER.get(it.bare.upper())
    if t is not None:
        return t
    ter = getattr(it.holding, "ter", None)
    if ter is not None and ter == ter and 0 < ter < 0.05:  # yfinance gives a fraction
        return ter * 100.0
    dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
    return _TESTFOL_TER_FALLBACK.get(dom, 0.20)


def _testfol_lines(p: "Portfolio") -> list[tuple[str, float]]:
    """testfol.io (ticker, weight%) lines for a portfolio, using the KB proxies.

    Efficient-core funds expand into equity + bond − CASHX (leverage financed);
    2x funds use a native ``?L=2`` ticker. Each proxy carries the fund TER as an
    ``E=`` drag (exposure-weighted average when several funds share a proxy).
    Unmapped instruments fall back to a proxy chosen by their dominant class.
    """
    agg_w: dict[str, float] = {}       # proxy/expr → weight
    agg_wter: dict[str, float] = {}    # proxy → Σ weight×ter (for averaging)
    baked: set[str] = set()            # exprs that already carry E= / take no E

    def add(t, w, ter=None):
        agg_w[t] = agg_w.get(t, 0.0) + w
        if ter is None:
            baked.add(t)
        else:
            agg_wter[t] = agg_wter.get(t, 0.0) + w * ter

    for it in p.items:
        w = it.weight
        ter = _instrument_ter(it)
        rule = _TESTFOL_PROXY.get(it.bare.upper())
        if rule is None:
            dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
            add(_TESTFOL_FALLBACK.get(dom, it.bare), w, ter)
        elif rule[0] == "lev2":
            add(rule[1], w)                       # E= already in the expression
        elif rule[0] == "one" or rule[0] == "carry":
            add(rule[1], w, ter)                  # carry: rough DBMFSIM stand-in
        elif rule[0] == "blend":
            for t, f in rule[1]:
                add(t, w * f, ter)
        elif rule[0] == "ec_world":
            for t, f in (("SPYSIM", 0.63), ("EFASIM", 0.27), ("IEFSIM", 0.60)):
                add(t, w * f, ter)
            add("CASHX", w * -0.50)
        elif rule[0] == "ec_eur":
            for t, f in (("EZU", 0.90), ("IEFSIM", 0.60)):
                add(t, w * f, ter)
            add("CASHX", w * -0.50)

    out = []
    for t, w in agg_w.items():
        if abs(w) <= 0.05:
            continue
        if t in baked or t == "CASHX" or t not in agg_wter or w <= 0:
            expr = t
        else:
            e = round(agg_wter[t] / w, 2)
            expr = f"{t}?E={e}" if "?" not in t else f"{t}&E={e}"
        out.append((expr, round(w, 1)))
    return sorted(out, key=lambda kv: -kv[1])


def _testfol_expand_item(it: "WhatIfItem") -> list[tuple[str, float]]:
    """testfol code line(s) for ONE instrument (may be several, e.g. an
    efficient-core fund → equity + bond − CASHX). TER baked as ?E=."""
    w = it.weight
    ter = _instrument_ter(it)

    def e(t):
        return f"{t}?E={ter}"

    rule = _TESTFOL_PROXY.get(it.bare.upper())
    if rule is None:
        dom = max(it.comp_notional, key=it.comp_notional.get) if it.comp_notional else EQ
        base = _TESTFOL_FALLBACK.get(dom, it.bare)
        lines = [(e(base) if base != "CASHX" else base, w)]
    elif rule[0] == "lev2":
        lines = [(rule[1], w)]                         # E= already in expr
    elif rule[0] == "one":
        lines = [(e(rule[1]), w)]
    elif rule[0] == "carry":
        # No testfol ticker for commodity carry → surface the correct custom-CSV
        # recipe here; DBMFSIM is only a rough shareable stand-in for the paste.
        lines = [(f"{e(rule[1])}  ⟨rough: carry≠trend; custom CSV → "
                  f"BNPIF73P + CASHX − 0.66%⟩", w)]
    elif rule[0] == "blend":
        lines = [(e(t), f * w) for t, f in rule[1]]
    elif rule[0] == "ec_world":
        lines = [(e("SPYSIM"), 0.63 * w), (e("EFASIM"), 0.27 * w),
                 (e("IEFSIM"), 0.60 * w), ("CASHX", -0.50 * w)]
    elif rule[0] == "ec_eur":
        lines = [(e("EZU"), 0.90 * w), (e("IEFSIM"), 0.60 * w), ("CASHX", -0.50 * w)]
    else:
        lines = [(e(it.bare), w)]
    return [(t, round(x, 1)) for t, x in lines]


def _testfol_instrument_map(p: "Portfolio") -> list[dict]:
    """Per-instrument testfol breakdown: {ticker, name, weight, codes:[(expr,w)]}."""
    rows = []
    for it in p.items:
        rows.append({
            "ticker": it.bare,
            "name": short_instrument_name(it.holding.name or it.symbol, 40),
            "weight": it.weight,
            "codes": _testfol_expand_item(it),
        })
    return rows


def _simulation_rows(portfolios) -> list[dict]:
    """Per-instrument simulation description: (ticker, real_from, base_from, base)."""
    seen: dict[str, WhatIfItem] = {}
    for p in portfolios:
        for it in p.items:
            seen.setdefault(it.bare, it)
    rows: list[dict] = []
    for bare, it in sorted(seen.items()):
        exp = _instrument_exposures(it)
        starts = [proxy_data.USED_PROXY[b][1] for b in exp if b in proxy_data.USED_PROXY]
        base_from = max(starts).strftime("%Y-%m") if starts else "—"
        ph = it.holding.price_history
        real_from = ph.index.min().strftime("%Y-%m") if ph is not None and len(ph) else "—"
        parts = [
            f"{proxy_data.USED_PROXY.get(b, ('?', None))[0]} {frac * 100:.0f}%"
            for b, frac in sorted(exp.items(), key=lambda kv: -kv[1])
        ]
        lev = it.gross / 100.0
        base = ", ".join(parts) + (f"  [lev {lev:.2f}x, fin ^IRX+0.5%]" if lev > 1.01 else "")
        rows.append({"ticker": bare, "real_from": real_from,
                     "base_from": base_from, "base": base})
    return rows


def _print_simulation_map(portfolios, names) -> None:
    """Per-instrument: which proxy 'simulation' backs its pre-inception history."""
    print("\nSIMULATION MAP (per instrument — proxy used before the fund's real history)")
    print(f"  {'Ticker':<9}{'Real from':>10}{'Base from':>11}  Simulation base (proxy × exposure)")
    print("  " + "-" * 88)
    for r in _simulation_rows(portfolios):
        print(f"  {r['ticker']:<9}{r['real_from']:>10}{r['base_from']:>11}  {r['base']}")
    print("  (recent period = REAL fund returns from 'Real from'; earlier = the proxy base above)")


def _print_metrics(portfolios) -> None:
    """Merged return + risk metrics on the single aligned history."""
    names = [p.name for p in portfolios]
    sep = "-" * (2 + _LABW + _COLW * len(names))
    w = next((p.window for p in portfolios if p.window), None)
    win = f"{w[0]:%Y-%m} → {w[1]:%Y-%m}" if w else "—"

    print("\n" + sep)
    print(f"PORTFOLIO METRICS — single aligned history ({win})")
    print(_header(names))
    rows = [
        ("CAGR", "cagr", "%"), ("Volatility (ann.)", "volatility", "%"),
        ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("VaR 95% (daily)", "var_95", "%"), ("CVaR 95% (daily)", "cvar_95", "%"),
        ("Beta vs S&P 500", "beta", ""), ("Alpha (ann.)", "alpha", "%"),
    ]
    for label, key, unit in rows:
        row = f"  {label:<{_LABW}}"
        for p in portfolios:
            v = (p.metrics_aligned or {}).get(key)
            s = "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}{unit}"
            row += f"{s:>{_COLW}}"
        print(row)


def _print_robustness(portfolios) -> None:
    """Rolling / stress / Monte-Carlo, all on the same single aligned history."""
    names = [p.name for p in portfolios]
    sep = "-" * (2 + _LABW + _COLW * len(names))

    def line(label, fn):
        print(f"  {label:<{_LABW}}" + "".join(f"{fn(p):>{_COLW}}" for p in portfolios))

    def roll(p, sub, key):
        v = p.rob.get(sub, {}).get(key)
        return "—" if v is None else f"{v * 100:.1f}%"

    def stress(p, scen, field):
        d = p.rob.get("stress", {}).get(scen, {})
        return "—" if not d.get("covered") else f"{d[field]:.1f}%"

    print("\n" + sep)
    print("ROBUSTNESS (same aligned history)")
    print(_header(names))
    line("Roll 1Y ret p05", lambda p: roll(p, "rolling1y", "p05"))
    line("Roll 1Y ret median", lambda p: roll(p, "rolling1y", "median"))
    line("Roll 1Y ret p95", lambda p: roll(p, "rolling1y", "p95"))
    line("1Y windows positive",
         lambda p: (lambda d: "—" if not d else f"{d['pct_positive']:.0f}%")(p.rob.get("rolling1y", {})))
    line("Roll 3Y ret p05", lambda p: roll(p, "rolling3y", "p05"))
    line("Roll 3Y ret median", lambda p: roll(p, "rolling3y", "median"))
    line("Roll 1Y Sharpe min–max",
         lambda p: (lambda d: "—" if not d else f"{d['min']:.2f}–{d['max']:.2f}")(p.rob.get("sharpe", {})))
    line("MC CAGR 1Y [p05/p95]",
         lambda p: (lambda d: "—" if not d else f"{d['p05']:.0f}/{d['p95']:.0f}%")(p.rob.get("bootstrap", {}).get("cagr", {})))
    line("MC MaxDD p05 (worst)",
         lambda p: (lambda d: "—" if not d else f"{d['p05']:.1f}%")(p.rob.get("bootstrap", {}).get("max_drawdown", {})))
    line("Dot-com maxDD", lambda p: stress(p, "Dot-com 2000-02", "max_drawdown"))
    line("GFC'08 maxDD", lambda p: stress(p, "GFC 2008", "max_drawdown"))
    line("COVID'20 maxDD", lambda p: stress(p, "COVID 2020", "max_drawdown"))
    line("2022 maxDD", lambda p: stress(p, "2022 rate shock", "max_drawdown"))
    print("  (single history = per-instrument splice: REAL fund returns where available,")
    print("   proxy-reconstructed (geo + leverage financing) before inception; modeled, USD-based)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="What-if comparison of one or more portfolios")
    parser.add_argument("--weights", default="input/portfolio_test.csv")
    parser.add_argument("--config", default="input/targets.csv")
    parser.add_argument("--orders", default="input/order_list.csv")
    parser.add_argument("--notional", type=float, default=100_000.0,
                        help="Fallback anchor (EUR) if the real portfolio is unavailable")
    parser.add_argument("--excel", nargs="?", const="output/whatif.xlsx", default=None,
                        help="Write an Excel report (default path: output/whatif.xlsx)")
    parser.add_argument("--refresh-current", action="store_true",
                        help="Rewrite the CURRENT portfolio rows in the weights CSV "
                             "from the live order list, then run the analysis")
    args = parser.parse_args(argv)

    weights_path = ROOT / args.weights
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        return 1
    portfolios_raw = _load_portfolios(weights_path)
    if not portfolios_raw:
        logger.error("No portfolios found in %s", weights_path)
        return 1

    config = load_config(str(ROOT / args.config))
    asset_target = config.invested_allocation_targets_pctg
    geo_target = config.equity_geo_targets_pctg

    logger.info("Deriving the EUR anchor from %s...", args.orders)
    anchor, _real_holdings = _real_portfolio(ROOT / args.orders)
    if anchor is None:
        anchor = args.notional
        logger.warning("Real portfolio unavailable; using fallback anchor €%.0f", anchor)
    else:
        logger.info("Anchor (real invested, ex-cash): €%.2f", anchor)

    # Optionally refresh the CURRENT snapshot in the weights CSV, then reload.
    if args.refresh_current:
        if _real_holdings:
            rows = _current_rows(_real_holdings, anchor)
            _refresh_current_in_csv(weights_path, rows)
            logger.info("Refreshed CURRENT (%d holdings) in %s", len(rows), weights_path)
            portfolios_raw = _load_portfolios(weights_path)
        else:
            logger.warning("No real portfolio available; CURRENT not refreshed.")

    sym_map, desc_map = _build_reference_maps()
    universe = _enrich_universe(portfolios_raw, sym_map)

    portfolios: list[Portfolio] = []

    # Every portfolio comes from the weights CSV; the order list is only the
    # EUR anchor, never an implicit extra portfolio.
    for name, rows in portfolios_raw:
        items = _portfolio_items(rows, universe, desc_map)
        if not items:
            logger.warning("Portfolio '%s' has no resolvable instruments; skipping", name)
            continue
        portfolios.append(Portfolio(name, items, compute_allocations(items), None))

    if not portfolios:
        logger.error("No portfolios could be built.")
        return 1

    _compute_robustness(portfolios)
    _print_report(portfolios, asset_target, geo_target, anchor)

    if args.excel:
        from tarzan.export.whatif_excel import export_whatif_excel
        out = ROOT / args.excel
        out.parent.mkdir(parents=True, exist_ok=True)
        export_whatif_excel(str(out), portfolios, asset_target, geo_target, anchor,
                            tolerance=config.rebalancing_target_tolerance_pctg,
                            sim_rows=_simulation_rows(portfolios),
                            testfol={p.name: _testfol_lines(p) for p in portfolios},
                            testfol_byinst={p.name: _testfol_instrument_map(p)
                                            for p in portfolios})
        print(f"\n  Excel report: {out}")

    print("\n" + "=" * (2 + _LABW + _COLW * len(portfolios)))
    print("Done. Composition splits are auto-inferred — verify with the specs.")
    print("=" * (2 + _LABW + _COLW * len(portfolios)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
