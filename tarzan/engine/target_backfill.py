"""Reach back a target sleeve whose own listing is younger than the charts.

The target line holds the plan's weights constant over the common window of its
sleeves, so one young listing (the UCITS MATE, listed Nov 2026) used to cut the
line down to a few sessions and every chart but 1D dropped it. Here a sleeve
that starts after ``start`` gets a MODELED history in front of its real one:

1. its twin — another catalog row for the same strategy (shared symbol root or
   one curated name containing the other) that the run already priced as a
   benchmark with enough history; EUR-hedged when the sleeve is, by adding the
   EUR-USD short-rate differential to the twin's native USD returns;
2. otherwise the backtest's reconstruction from the sleeve's own composition
   (index proxies, financed leverage, currency hedge), calibrated on the
   overlap with no extrapolated alpha unless it is significant.

The real tape is never altered: modeled returns only precede it. Every modeled
sleeve is reported, so the chart can draw that stretch dashed and say what it is.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: A sleeve whose real tape starts within this slack of ``start`` needs no model.
_SLACK = pd.Timedelta(days=7)


def _is_eur_hedged(row) -> bool:
    name = str((row or {}).get("name") or "").casefold().replace("(", "").replace(")", "")
    return "eur hedged" in name


_SHARE_CLASS_WORDS = frozenset({
    "ucits", "etf", "acc", "dist", "r", "i", "h", "eur", "usd", "gbp", "chf",
    "hedged", "hdg", "class", "share", "units",
})


def _strategy_name(name: str) -> str:
    """A curated name without its share-class words: "iMGP DBi Managed Futures
    R EUR" and "iMGP DBi Managed Futures (EUR Hedged)" are one strategy."""
    words = re.sub(r"[()\[\],]", " ", str(name or "").casefold()).split()
    return " ".join(w for w in words if w not in _SHARE_CLASS_WORDS)


def _priced(benchmark_catalog, holdings):
    """``(isin, currency, eur tape, native tape)`` for everything the run priced."""
    for r in (benchmark_catalog or {}).values():
        yield (getattr(r, "isin", ""), str(getattr(r, "currency", "") or "").upper(),
               getattr(r, "history", None), getattr(r, "history_native", None))
    for h in holdings or ():
        yield (str(getattr(h, "isin", "") or ""),
               str(getattr(h, "price_currency", "") or "").upper(),
               getattr(h, "price_history", None), getattr(h, "price_history_native", None))


def _twin_returns(holding, row, benchmark_catalog, start, holdings=()) -> tuple[Optional[pd.Series], str]:
    """Daily returns of the sleeve's twin — another share class of the same
    strategy that the run priced — in the sleeve's own currency treatment."""
    from tarzan import config as cfg
    from tarzan.data import proxy_data
    from tarzan.engine.stats import normalize_index

    index = cfg.catalog()
    own_isin = index.isin_of(row)
    own_strategy = _strategy_name(row.get("name"))
    if len(own_strategy) < 12:
        return None, ""
    twins = [other for other in index.rows
             if other is not row and index.isin_of(other)
             and index.isin_of(other) != own_isin
             and _strategy_name(other.get("name")) == own_strategy]
    priced = list(_priced(benchmark_catalog, holdings))
    for twin in twins:
        found = next((p for p in priced if p[0] == index.isin_of(twin)), None)
        if found is None:
            continue
        _isin, currency, eur_tape, native_tape = found
        sleeve_hedged, twin_hedged = _is_eur_hedged(row), _is_eur_hedged(twin)
        label = f"its twin fund {twin.get('ticker')} ({currency or 'native'}"
        if sleeve_hedged and not twin_hedged:
            fx, carry = proxy_data.currency_hedge_legs()
            if carry is None or fx is None:
                continue  # a hedged class cannot borrow an unhedged path
            if currency == "USD" and native_tape is not None and len(native_tape) > 1:
                r = normalize_index(native_tape, drop_duplicates=True).pct_change().dropna()
            elif eur_tape is not None and len(eur_tape) > 1:
                # A EUR class of a dollar strategy carries EUR/USD: take it out.
                r = normalize_index(eur_tape, drop_duplicates=True).pct_change().dropna()
                fx_int = proxy_data._interval_fx_returns((1.0 + fx.fillna(0.0)).cumprod(), r.index)
                r = (1.0 + r) / (1.0 + fx_int) - 1.0
                label += ", currency move removed"
            else:
                continue
            r = (1.0 + r) * (1.0 + carry.reindex(r.index).ffill().fillna(0.0)) - 1.0
            label += ", EUR-hedged"
        else:
            if eur_tape is None or len(eur_tape) < 2:
                continue
            r = normalize_index(eur_tape, drop_duplicates=True).pct_change().dropna()
        return r.dropna(), label + ")"
    return None, ""


def _reconstructed_returns(holding) -> tuple[Optional[pd.Series], str]:
    """The backtest's reconstruction of the sleeve from its composition."""
    from tarzan.backtest.engine import instrument_exposures, is_currency_hedged
    from tarzan.backtest.model import WhatIfItem, composition_for
    from tarzan.backtest.ter import instrument_ter
    from tarzan.data import proxy_data
    from tarzan.engine import synthetic as syn
    from tarzan.engine.stats import TRADING_DAYS
    from tarzan.models.instrument_key import normalize_ticker

    notl, cap, why = composition_for(holding)
    item = WhatIfItem(normalize_ticker(holding.ticker), holding.ticker, holding.isin or "",
                      float(getattr(holding, "target_portfolio", 0.0) or 0.0), holding,
                      notl, cap, why)
    exposures = instrument_exposures(item)
    if not exposures:
        return None, ""
    proxies, fin = proxy_data.proxy_returns_for(set(exposures))
    hedge = proxy_data.currency_hedge_legs() if is_currency_hedged(item) else (None, None)
    base = syn.replicate_portfolio_returns(
        exposures, proxies, financing_daily=fin, spread_annual=0.005,
        hedge_fx=hedge[0], hedge_carry=hedge[1])
    if base is None or base.empty:
        return None, ""
    base = base - (instrument_ter(item) / 100.0) / TRADING_DAYS
    names = {"Alternative": "trend index", "Fixed Income": "bonds",
             "Commodities": "commodities", "Gold": "gold"}
    grouped: dict[str, float] = {}
    for k, v in exposures.items():   # geography legs read as one equity leg
        label = names.get(k, "bonds" if "Fixed Income" in k else "equity")
        grouped[label] = grouped.get(label, 0.0) + v
    legs = ", ".join(f"{k} {v * 100:.0f}%" for k, v in
                     sorted(grouped.items(), key=lambda kv: -kv[1]))
    return base, f"its composition ({legs}{', EUR-hedged' if hedge[0] is not None else ''})"


def extend(holding, start, benchmark_catalog, holdings=()) -> tuple[pd.Series, Optional[dict]]:
    """``(price level, modeled)`` for one target sleeve reaching back to ``start``.

    ``modeled`` is None when the real tape already reaches ``start``; otherwise
    ``{"ticker", "real_from", "source"}``. When no model is available the real
    tape is returned unchanged (the common window then stays short, as before).
    """
    from tarzan import config as cfg
    from tarzan.backtest.engine import _check_backfill_realism
    from tarzan.engine import synthetic as syn
    from tarzan.engine.stats import normalize_index

    px = normalize_index(holding.price_history, drop_duplicates=True).dropna()
    start = pd.Timestamp(start)
    if len(px) < 2 or px.index[0] <= start + _SLACK:
        return px, None
    real = px.pct_change().dropna()
    row = cfg.catalog().row(holding.isin or None, holding.ticker or None)
    twin, twin_label = (None, "")
    if row is not None:
        twin, twin_label = _twin_returns(holding, row, benchmark_catalog, start, holdings)
    spliced, source = None, ""
    if twin is not None:
        # Same strategy: the twin's returns stand in 1:1 where it has them.
        spliced, source = syn.splice_returns(twin, real), f"from {twin_label}"
    if spliced is None or spliced.empty or spliced.index[0] > start + _SLACK:
        try:
            recon, recon_label = _reconstructed_returns(holding)
        except Exception as exc:  # noqa: BLE001 — a model must never sink the real line
            logger.info("Target sleeve %s: no reconstruction (%s)", holding.ticker, exc)
            recon = None
        if recon is not None:
            if spliced is not None and not spliced.empty:
                # The twin where it exists, the composition only before it.
                twin_from = spliced.index[0]
                spliced = syn.splice_returns(recon, spliced)
                source += f", and {recon_label} before {twin_from:%b %Y}"
            else:
                spliced = syn.calibrated_splice(recon, real)
                source = f"from {recon_label}"
    if spliced is None or spliced.empty:
        return px, None
    spliced = spliced.loc[spliced.index > start - pd.Timedelta(days=1)]
    if spliced.empty or spliced.index[0] > start + _SLACK:
        return px, None
    _check_backfill_realism(str(holding.ticker), spliced, real)
    # Levels anchored so the modeled stretch ends exactly on the first real
    # close: the real tape is reproduced, only preceded.
    level = (1.0 + spliced).cumprod()
    first_real = px.index[0]
    upto = level.loc[level.index <= first_real]
    if len(upto) < 2:
        return px, None
    # The level ON the first real close (or the last modeled day before it)
    # is pinned to that close; only the stretch before it is modeled.
    scale = float(px.iloc[0]) / float(upto.iloc[-1])
    modeled_part = (upto * scale).loc[lambda s: s.index < first_real]
    out = pd.concat([modeled_part, px]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out, {"ticker": str(holding.ticker), "real_from": first_real, "source": source}
