"""Backtest compute core — the genuinely-new what-if orchestration.

Everything metric/statistical is delegated to the shared engine:
  * NAV construction / rebalancing → :func:`tarzan.engine.synthetic.combine_returns`;
  * splicing / replication / factor backfill → :mod:`tarzan.engine.synthetic`;
  * return+risk metrics + rolling/stress/MC → :mod:`tarzan.engine.robustness`
    / :mod:`tarzan.engine.stats` (with the fine, currency-matched, time-varying
    risk-free from :mod:`tarzan.data.proxy_data`).

This module only assembles per-instrument spliced histories into portfolios,
aligns them to one common window, and drives the dual-currency comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tarzan.data import proxy_data
from tarzan.data.loader import load_config
from tarzan.engine import robustness as rob
from tarzan.engine import synthetic as syn
from tarzan.engine.stats import TRADING_DAYS

from tarzan.backtest.loader import (
    build_symbol_map, enrich_universe, load_portfolios, portfolio_items,
)
from tarzan.backtest.model import (
    CASH, COMM, CRYPTO, EQ, FI, GOLD, ALT, Portfolio, WhatIfItem,
    compute_allocations,
)
from tarzan.backtest.ter import instrument_ter

logger = logging.getLogger("backtest.engine")

ROOT = Path(__file__).resolve().parents[2]

# Commodity-carry funds → dedicated "Carry" backtest proxy (BNP carry index).
_CARRY_TICKERS = {"UEQC", "CRRY", "CRRE"}

# Factor tilt discovered per instrument by the factor-aware backfill, keyed by
# bare ticker → {factor: loading}. Populated during the aligned backtest so the
# simulation map can describe the SAME reconstruction the metrics use.
_FACTOR_TILT: dict[str, dict] = {}

# Name/description keywords that identify a FACTOR equity ETF. Only these get
# the factor-aware backfill; plain market / leveraged-market equity funds keep
# the naive splice, so tracking noise is never mis-fit as a factor tilt.
_FACTOR_KEYWORDS = (
    "momentum", "value", "quality", "growth", "dividend", "buyback",
    "minimum volatility", "min vol", "min. vol", "low volatility", "low vol",
    "small cap", "small-cap", "smallcap", "size factor", "size tilt",
    "multifactor", "multi-factor", "multi factor", "equal weight",
    "equal-weight", "enhanced value", "prime value", "high beta", "low beta",
    "profitability", "single factor", "multi-strategy factor",
)


def is_factor_fund(it: "WhatIfItem") -> bool:
    """True if the instrument is a FACTOR equity ETF (by name/role/ticker)."""
    text = " ".join(t for t in (getattr(it.holding, "name", ""),
                                 getattr(it.holding, "role", ""),
                                 getattr(it, "explanation", ""), it.bare) if t).lower()
    return any(k in text for k in _FACTOR_KEYWORDS)


def instrument_exposures(item: WhatIfItem) -> dict:
    """Per-instrument exposure (fraction of the instrument's own value).

    Equity is split by the instrument's own geo breakdown; other classes from
    its notional composition. Gross may exceed 1 (a leveraged fund), which the
    replicator finances. Drives the instrument's synthetic base.
    """
    if item.bare.upper() in _CARRY_TICKERS:
        gross = sum(item.comp_notional.values()) / 100.0
        return {"Carry": gross if gross > 0 else 1.0}

    exp: dict[str, float] = {}
    eqf = item.comp_notional.get(EQ, 0.0) / 100.0
    gb = item.geo_breakdown
    if eqf > 0:
        # Keep only geo buckets ≥5% and renormalise: a noisy sub-5% sliver would
        # otherwise pull in a short-history proxy and truncate the window.
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
    """Normalised daily returns from a holding's real price history.

    A daily bar's MARKET date is its wall-clock date, so we drop any timezone
    keeping the local date (``tz_localize(None)``). We must NOT ``tz_convert``
    to UTC first: a European bar stamped midnight local would roll back to the
    previous day and de-align the fund's real returns from the proxy/factor
    series across the whole backtest.
    """
    ph = getattr(holding, "price_history", None)
    if ph is None or len(ph) < 2:
        return None
    r = ph.pct_change().dropna()
    idx = r.index
    r.index = (idx.tz_localize(None).normalize()
               if getattr(idx, "tz", None) is not None else idx.normalize())
    return r[~r.index.duplicated(keep="last")]


def portfolio_long_returns(p: "Portfolio", proxies: dict, fin,
                           backfill: str = "naive",
                           rebalance: str = "quarterly",
                           factors=None) -> pd.Series:
    """Portfolio long-history daily returns from PER-INSTRUMENT spliced series.

    Each instrument gets a long base (its composition replicated on index
    proxies, its own leverage financed) spliced with its REAL returns where
    available. Each instrument's TER is charged as a daily drag on its MODELED
    base only (the spliced real fund returns are already net of fees).
    Instruments are combined under the chosen ``rebalance`` policy via the
    shared :func:`tarzan.engine.synthetic.combine_returns`.
    """
    cols: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for i, it in enumerate(p.items):
        base = syn.replicate_portfolio_returns(
            instrument_exposures(it), proxies, financing_daily=fin, spread_annual=0.005)
        if base is not None and not base.empty:
            base = base - (instrument_ter(it) / 100.0) / TRADING_DAYS
        real = _real_daily_returns(it.holding)
        eq_dominant = (it.comp_notional.get(EQ, 0.0) >= 50.0)
        if backfill == "factor" and eq_dominant and is_factor_fund(it):
            loads = syn.factor_loadings(base, real, factors)
            if loads:
                _FACTOR_TILT[it.bare] = loads
            spliced = syn.factor_splice(base, real, factors)
        elif backfill == "calibrated":
            spliced = syn.calibrated_splice(base, real)
        else:
            spliced = syn.splice_returns(base, real)
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
    return syn.combine_returns(df, pd.Series(weights), rebalance)


def compute_robustness(portfolios: list["Portfolio"], backfill: str = "naive",
                       rebalance: str = "quarterly") -> None:
    """Build ONE aligned history per portfolio and compute the merged
    return+risk metrics and rolling/stress/Monte-Carlo robustness on it, in
    BOTH numeraires (EUR + USD)."""
    needed: set[str] = {"USA"}  # ^GSPC is also the beta/alpha benchmark
    for p in portfolios:
        for it in p.items:
            needed |= set(instrument_exposures(it))

    factors = proxy_data.factor_daily() if backfill == "factor" else None

    default_ccy = proxy_data._TARGET_CCY or "EUR"
    currencies = [c for c in ("EUR", "USD") if c != default_ccy] + [default_ccy]
    for ccy in currencies:
        proxy_data.set_target_currency(ccy)
        is_default = (ccy == default_ccy)
        logger.info("Fetching %d proxy series for the aligned backtest (%s)...",
                    len(needed), ccy)
        proxies, fin = proxy_data.proxy_returns_for(needed)

        synth = {p.name: syn.returns_to_price(
                    portfolio_long_returns(p, proxies, fin, backfill, rebalance, factors))
                 for p in portfolios}

        navs = [n for n in synth.values() if n is not None and not n.empty]
        if not navs:
            continue
        common_start = max(n.index.min() for n in navs)
        common_end = min(n.index.max() for n in navs)

        bench_ret = proxies.get("USA")
        bench_px = syn.returns_to_price(bench_ret) if bench_ret is not None else None
        bench_w = (bench_px.loc[common_start:common_end]
                   if bench_px is not None and not bench_px.empty else None)

        rf = proxy_data.risk_free_annual(common_start, common_end)
        rf_daily = proxy_data.risk_free_daily(common_start, common_end)

        for p in portfolios:
            nav_full = synth.get(p.name)
            if nav_full is None or nav_full.empty:
                continue
            nav = nav_full.loc[common_start:common_end]
            metrics = rob.full_metrics(nav, bench_w, risk_free=rf, rf_daily=rf_daily)
            setattr(p, f"metrics_aligned_{ccy.lower()}", metrics)
            if is_default:
                p.synth_nav = nav_full
                p.nav = nav
                p.window = (common_start, common_end)
                p.metrics_aligned = metrics
                p.rob = {
                    "rolling1y": rob.rolling_return_distribution(nav, 252),
                    "rolling3y": rob.rolling_return_distribution(nav, 252 * 3),
                    "sharpe": rob.rolling_sharpe_range(nav, 252),
                    "stress": rob.stress_scenarios(nav),
                    "bootstrap": rob.block_bootstrap(nav),
                }

    proxy_data.set_target_currency(default_ccy)


def simulation_rows(portfolios) -> list[dict]:
    """Per-instrument simulation description: (ticker, real_from, base_from, base)."""
    seen: dict[str, WhatIfItem] = {}
    for p in portfolios:
        for it in p.items:
            seen.setdefault(it.bare, it)
    rows: list[dict] = []
    for bare, it in sorted(seen.items()):
        exp = instrument_exposures(it)
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
        tilt = _FACTOR_TILT.get(bare)
        if tilt:
            shown = {k: v for k, v in tilt.items() if abs(v) >= 0.05}
            if shown:
                legs = ", ".join(f"{k}{v:+.2f}"
                                 for k, v in sorted(shown.items(), key=lambda kv: -abs(kv[1])))
                base += f"  + factor tilt ⟨{legs}⟩"
        rows.append({"ticker": bare, "real_from": real_from,
                     "base_from": base_from, "base": base})
    return rows


def backfill_label(backfill: str) -> str:
    return {"calibrated": "regression-calibrated",
            "factor": "factor-aware (Developed FF SMB/HML/RMW/MOM on factor ETFs)"}.get(
        backfill, "1:1 proxy")


_DEFAULT_WEIGHTS = ROOT / "input" / "portfolio_test.csv"


def newsletter_portfolios(*, deterministic: bool = False):
    """Guarded backtest for the newsletter/Excel: run the candidate backtest
    ONCE with the production defaults, or return None when it should be skipped.

    Shared by the CLI (``tarzan.main``) and the email path (``tarzan.delivery``)
    so both wire the backtest identically. Skipped in deterministic/as-of runs
    (network-bound, not reproducible) and when the weights file is absent; any
    failure is swallowed so it can never break the primary newsletter/send.
    """
    if deterministic:
        logger.info("Backtest skipped (deterministic/as-of run).")
        return None
    if not _DEFAULT_WEIGHTS.exists():
        logger.info("Backtest skipped (%s not found).", _DEFAULT_WEIGHTS)
        return None
    try:
        logger.info("Running candidate-portfolio backtest (%s)...", _DEFAULT_WEIGHTS)
        portfolios = run_backtest(_DEFAULT_WEIGHTS, currency="eur",
                                  backfill="factor", rebalance="quarterly")
        logger.info("Backtest built %d candidate portfolio(s).", len(portfolios or []))
        return portfolios or None
    except Exception as e:  # noqa: BLE001
        logger.warning("Backtest skipped (%s): %s", type(e).__name__, e)
        return None


def run_backtest(weights_path: str | Path, *, currency: str = "eur",
                 backfill: str = "factor", rebalance: str = "quarterly",
                 ) -> list["Portfolio"]:
    """Load candidates, enrich, build portfolios and run the aligned backtest.

    Returns the list of :class:`Portfolio` with ``metrics_aligned_*``, ``nav``,
    ``window`` and ``rob`` populated. The single public entry point reused by
    the CLI, the Excel export and the newsletter section.
    """
    proxy_data.set_target_currency(currency)
    raw = load_portfolios(Path(weights_path))
    if not raw:
        return []
    sym_map = build_symbol_map()
    universe = enrich_universe(raw, sym_map)
    portfolios: list[Portfolio] = []
    for name, rows in raw:
        items = portfolio_items(rows, universe)
        if not items:
            logger.warning("Portfolio '%s' has no resolvable instruments; skipping", name)
            continue
        portfolios.append(Portfolio(name, items, compute_allocations(items), None))
    if not portfolios:
        return []
    compute_robustness(portfolios, backfill=backfill, rebalance=rebalance)
    return portfolios
