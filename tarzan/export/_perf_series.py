"""Pure performance/return series helpers for the newsletter.

Window P&L, TWROR, normalisation, benchmark alignment and the "Markets"
strip — all pure transforms of a ``PortfolioMetrics`` (or its series), with
no HTML, no template, no context object. Extracted from ``newsletter.py`` so
the financial math is importable and testable on its own, away from the
3,800-line HTML generator. ``newsletter`` re-imports these under their
original names, so its call-sites and the public ``market_snapshot`` symbol
are unchanged.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics


def _window_money_pnl(
    pnl_series, actual_series, days: int
) -> tuple[Optional[float], Optional[float]]:
    """Real money P&L over the last ``days`` calendar days, net of any
    contributions in that window.

    Returns ``(gain_eur, gain_pct)`` where the € gain is the delta of the
    cumulative P&L series across the window and the % expresses it over the
    portfolio value at the window start. Either may be None when the order
    path produced no series (holdings-only run) or the data is too short.
    """
    if pnl_series is None:
        return None, None
    pnl = pnl_series.dropna()
    if len(pnl) < 2:
        return None, None
    last_date = pnl.index[-1]
    try:
        cutoff = last_date - pd.Timedelta(days=days)
    except (TypeError, ValueError):
        return None, None
    prior = pnl[pnl.index <= cutoff]
    start_pnl = float(prior.iloc[-1]) if len(prior) else float(pnl.iloc[0])
    gain = float(pnl.iloc[-1]) - start_pnl
    if gain != gain:  # NaN guard
        return None, None

    base: Optional[float] = None
    if actual_series is not None:
        av = actual_series.dropna()
        if len(av):
            prior_av = av[av.index <= cutoff]
            base = float(prior_av.iloc[-1]) if len(prior_av) else float(av.iloc[0])
    pct = (gain / base * 100) if base and base > 0 else None
    return gain, pct


def _norm_series(s: pd.Series) -> pd.Series:
    """tz-naive, calendar-day-normalised, de-duplicated copy of a series.

    The export layer's spelling of ``stats.normalize_index(..., drop_duplicates
    =True)`` — a thin alias so all the newsletter chart/series builders keep the
    same short name while the tz-collapse logic lives in exactly one place.
    """
    from tarzan.engine.stats import normalize_index
    return normalize_index(s, drop_duplicates=True)


# Curated set of major indices for the "Markets" strip, in display order.
# Only those present in the already-fetched benchmark histories are shown,
# so the strip needs no extra network calls.
_MARKETS_ORDER = [
    "S&P 500", "Nasdaq 100", "MSCI World", "MSCI Emerging Markets",
    "EURO STOXX 50", "STOXX Europe 600",
]


def market_snapshot(m: PortfolioMetrics, spark_points: int = 30) -> list[dict]:
    """Latest level, daily change and a short spark series for the major
    indices, derived from ``m.benchmark_histories`` (no extra network).

    Each entry: ``{name, value, change, pct, spark}``. Indices with fewer
    than two observations are skipped. Used by both the newsletter Markets
    strip and the AI digest (so the narrative can cite real figures)."""
    out: list[dict] = []
    bh = m.benchmark_histories or {}
    for name in _MARKETS_ORDER:
        s = bh.get(name)
        if s is None or len(s) < 2:
            continue
        s = _norm_series(s).dropna()
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        if prev == 0:
            continue
        out.append({
            "name": name,
            "value": last,
            "change": last - prev,
            "pct": (last - prev) / prev * 100.0,
            "spark": list(s.iloc[-spark_points:].values.astype(float)),
        })
    return out


def _flow_list(external_flows, start, end, threshold: float = 500.0):
    """``{date: eur}`` → sorted ``[(Timestamp, eur)]`` inside ``[start, end]``,
    dropping flows below ``threshold`` (nets out small rebalancing trades so
    only real deposits/withdrawals mark the chart)."""
    if not external_flows:
        return []
    out = []
    for d, v in external_flows.items():
        ts = pd.Timestamp(d)
        if ts < start or ts > end or abs(float(v)) < threshold:
            continue
        out.append((ts, float(v)))
    return sorted(out, key=lambda t: t[0])


def _window_twror(nav: Optional[pd.Series], days: int) -> Optional[float]:
    """Window TWROR (%) from the flow-adjusted NAV index over the last
    ``days`` calendar days.

    Delegates to the engine's ``compute_period_return`` so this matches the
    authoritative ``performance_full`` figures exactly — the matrix cell and
    the chart line then tell one story. (Previously this anchored on the last
    point *before* the window while ``compute_period_return`` anchors on the
    first point *inside* it, producing the 30-day TWROR split.)"""
    if nav is None:
        return None
    from tarzan.engine.stats import compute_period_return
    s = nav.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(s) < 2:
        return None
    return compute_period_return(s, days)


def _rebase_to_window(araw: "pd.Series", idx: "pd.DatetimeIndex") -> "list | None":
    """Rebase a level series over exactly the dates spanned by ``idx``.

    The denominator is the first real source observation inside the window;
    alignment uses the union of source and target dates before forward-fill so
    observations on a source-only trading day are never silently discarded.
    No observation after ``idx[-1]`` can affect the line.  Returns a percentage
    list aligned to ``idx``, or ``None`` when fewer than two real in-window
    observations exist.
    """
    a = _norm_series(araw)
    if a is None:
        return None
    a = a.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(a) < 2 or len(idx) < 2:
        return None
    in_win = a[(a.index >= idx[0]) & (a.index <= idx[-1])]
    if len(in_win) < 2 or not float(in_win.iloc[0]):
        return None
    anchor = float(in_win.iloc[0])
    # Reindexing directly to ``idx`` would throw away valid source observations
    # that fall on dates absent from the target calendar.  Fill on the union,
    # then sample the exact target window.
    aligned = (
        a.reindex(a.index.union(idx)).sort_index().ffill().reindex(idx)
    )
    # Leading target dates before the first in-window observation start at the
    # anchor (0%) rather than carrying a stale pre-window source value.
    lead = aligned.index < in_win.index[0]
    aligned = aligned.mask(lead, anchor).bfill()
    return list(((aligned / anchor - 1.0) * 100.0).values.astype(float))


# Back-compat alias: benchmark rebasing is just the general case.
_rebase_benchmark = _rebase_to_window


def _geo_benchmark_series(m: PortfolioMetrics, geo_name: Optional[str]) -> "pd.Series | None":
    """Price history of the geographic benchmark (the taxonomy row flagged
    ``is_benchmark_geo=TRUE``, e.g. iShares MSCI ACWI) from
    ``benchmark_histories``. Looked up by that configured name — no fuzzy
    matching — since ``benchmark_histories`` is keyed by the same curated
    instrument names as the taxonomy."""
    if not geo_name:
        return None
    return (m.benchmark_histories or {}).get(geo_name)


def _perf_window(m: PortfolioMetrics, n_days: int = 30,
                 geo_name: Optional[str] = None) -> Optional[dict]:
    """Comparable trailing performance window on one shared close boundary.

    Portfolio value, TWROR, P&L and the geographic benchmark are all truncated
    to the latest date observed by both portfolio and benchmark before the
    ``n_days`` cutoff is calculated.  This prevents a live/partial benchmark
    candle from supplying a legend value while the chart stops at the prior
    portfolio close (or vice versa).  ``endpoints`` is the sole source for
    chart labels and is derived from the exact plotted arrays.
    """
    val_raw = m.actual_value_series
    if val_raw is None or len(val_raw) < 2:
        return None
    val_all = _norm_series(val_raw).replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()
    if len(val_all) < 2:
        return None

    acwi_all = None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    if acwi_raw is not None:
        candidate = _norm_series(acwi_raw).replace(
            [float("inf"), float("-inf")], float("nan")
        ).dropna()
        if len(candidate) >= 2:
            acwi_all = candidate

    common_end = val_all.index[-1]
    if acwi_all is not None:
        common_dates = val_all.index.intersection(acwi_all.index)
        if len(common_dates) >= 2:
            common_end = common_dates[-1]
        else:
            # A benchmark without two common closes cannot support a
            # comparable line; retain the portfolio-only window instead.
            acwi_all = None

    cutoff = common_end - pd.Timedelta(days=n_days)
    val = val_all[(val_all.index >= cutoff) & (val_all.index <= common_end)]
    if len(val) < 2:
        return None
    idx = val.index

    twror = (
        _rebase_to_window(m.portfolio_history, idx)
        if m.portfolio_history is not None else None
    )
    acwi = (
        _rebase_benchmark(acwi_all[acwi_all.index <= common_end], idx)
        if acwi_all is not None else None
    )

    pnl_pct = None
    if m.pnl_series is not None:
        pnl = _norm_series(m.pnl_series).reindex(idx, method="ffill").bfill()
        base = pnl - pnl.iloc[0]
        v0 = float(val.iloc[0]) or 1.0
        pnl_pct = [float(p) / v0 * 100.0 for p in base.values]

    dates = list(idx)
    endpoints = {
        "twror": float(twror[-1]) if twror else None,
        "acwi": float(acwi[-1]) if acwi else None,
        "pnl_pct": float(pnl_pct[-1]) if pnl_pct else None,
    }
    return {
        "dates": dates,
        "value": list(val.values.astype(float)),
        "twror": twror,
        "acwi": acwi,
        "pnl_pct": pnl_pct,
        "flows": _flow_list(m.external_flows, dates[0], dates[-1]),
        "window_start": dates[0],
        "window_end": dates[-1],
        "source_end_dates": {
            "portfolio": val_all.index[-1],
            "benchmark": acwi_all.index[-1] if acwi_all is not None else None,
        },
        "endpoints": endpoints,
    }


def _perf_level_series(m: PortfolioMetrics, dates, geo_name: Optional[str] = None):
    """The indicators as % over the window, each matching the hero's
    definition, from existing series (no recomputation):
      * TWROR since inception — NAV index anchored at inception.
      * Total P&L %           — P&L ÷ net invested capital (value − P&L).
      * Unrealized P&L %       — unrealized ÷ cost basis (value − unrealized).
      * MSCI ACWI              — benchmark cumulative return anchored at the
        portfolio's inception, for a like-for-like since-inception compare.
    Returns ``(twror_si, total_pct, unreal_pct, acwi_si)`` (any may be None)."""
    if m.portfolio_history is None or m.actual_value_series is None:
        return None
    idx = pd.DatetimeIndex(dates)
    av = _norm_series(m.actual_value_series).reindex(idx, method="ffill").bfill()
    # TWROR since inception: anchor the NAV index at the FULL series' first
    # point (inception), THEN sample the window — so the line shows the
    # cumulative since-inception trajectory over the last 30 days (ending at
    # twror_pct), not a window-rebased 0%. (Reindexing before dividing would
    # rebase to the window start — the bug this avoids.)
    nav_full = _norm_series(m.portfolio_history)
    twror_si = None
    if len(nav_full) and float(nav_full.iloc[0]):
        twror_full = (nav_full / float(nav_full.iloc[0]) - 1.0) * 100.0
        twror_si = list(twror_full.reindex(idx, method="ffill").bfill().values.astype(float))
    total_pct = unreal_pct = None
    if m.pnl_series is not None:
        pnl = _norm_series(m.pnl_series).reindex(idx, method="ffill").bfill()
        total_pct = list((pnl / (av - pnl).replace(0, float("nan")) * 100.0).bfill().values.astype(float))
    if m.unrealized_series is not None:
        ur = _norm_series(m.unrealized_series).reindex(idx, method="ffill").bfill()
        unreal_pct = list((ur / (av - ur).replace(0, float("nan")) * 100.0).bfill().values.astype(float))
    # MSCI ACWI since inception: anchor on the benchmark's first observation
    # at/after inception (its own real price, not one stale-filled from before
    # the portfolio's start), then sample the window — same anchoring rule as
    # the 30-day chart and the Performance section.
    acwi_si = None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    if acwi_raw is not None and len(nav_full):
        acwi_si = _rebase_benchmark(acwi_raw, idx)
    return twror_si, total_pct, unreal_pct, acwi_si


def _perf_full_series(m: PortfolioMetrics, geo_name: Optional[str] = None,
                      max_points: int = 180) -> Optional[dict]:
    """The since-inception trajectory over the WHOLE date range (not the last
    30 days): cumulative TWROR (%), Total P&L (%) and MSCI ACWI (%) from
    inception to today, on a common daily index that is evenly downsampled to
    ``max_points`` so a multi-year series stays a light SVG. Reuses
    ``_perf_level_series`` for the cumulative math. None when unavailable.

    Keys mirror ``_perf_window`` so the chart builder is symmetric:
    ``{dates, twror, pnl_pct, acwi}`` (any line may be None)."""
    if m.portfolio_history is None or m.actual_value_series is None:
        return None
    nav_full = _norm_series(m.portfolio_history)
    if len(nav_full) < 2:
        return None
    idx = nav_full.index
    # Evenly downsample to <= max_points, always keeping the first and last
    # point (inception and today) so the endpoints are exact.
    if len(idx) > max_points:
        pos = sorted(set(
            [int(round(i * (len(idx) - 1) / (max_points - 1))) for i in range(max_points)]
        ))
        idx = idx[pos]
    lvl = _perf_level_series(m, list(idx), geo_name)
    if lvl is None:
        return None
    twror_si, total_pct, _unreal_pct, acwi_si = lvl
    return {
        "dates": list(idx),
        "twror": twror_si,
        "pnl_pct": total_pct,
        "acwi": acwi_si,
    }


def _rolling_ann_vol(level: "pd.Series", window: int) -> "pd.Series":
    """Rolling annualized volatility (%) of a price/level series: stdev of
    daily returns over ``window`` trading days × √(trading days/year) × 100.
    NaN for the leading days that lack a full window."""
    from tarzan.engine.stats import TRADING_DAYS
    ret = level.pct_change()
    return ret.rolling(window, min_periods=max(2, window // 2)).std() * (TRADING_DAYS ** 0.5) * 100.0


def _perf_vol_series(m: PortfolioMetrics, geo_name: Optional[str] = None,
                     n_days: Optional[int] = None, vol_window: int = 21,
                     max_points: int = 180) -> Optional[dict]:
    """Rolling annualized volatility of the portfolio NAV vs the geo benchmark,
    on a common daily index. The twin of ``_perf_window`` / ``_perf_full_series``
    for the "You vs the market" box's second (risk) row.

    ``n_days=None`` → the whole inception→today span (downsampled to
    ``max_points``); ``n_days=30`` → the trailing window (with ``vol_window``
    days of lead-in so the first plotted point already has a full window).
    Returns ``{dates, port, acwi}`` (% lists; either line may be None), or None
    when unavailable."""
    nav_full = _norm_series(m.portfolio_history) if m.portfolio_history is not None else None
    if nav_full is None or len(nav_full) < vol_window + 1:
        return None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    acwi_full = _norm_series(acwi_raw) if acwi_raw is not None else None

    port_vol_full = _rolling_ann_vol(nav_full, vol_window)
    # Choose the display index (same convention as the return charts).
    if n_days is None:
        idx = nav_full.index
        if len(idx) > max_points:
            pos = sorted({int(round(i * (len(idx) - 1) / (max_points - 1)))
                          for i in range(max_points)})
            idx = idx[pos]
    else:
        cutoff = nav_full.index[-1] - pd.Timedelta(days=n_days)
        idx = nav_full.index[nav_full.index >= cutoff]
        if len(idx) < 2:
            return None

    port = list(port_vol_full.reindex(idx, method="ffill").bfill().values.astype(float))
    acwi = None
    if acwi_full is not None and len(acwi_full) >= vol_window + 1:
        acwi_vol_full = _rolling_ann_vol(acwi_full, vol_window)
        acwi = list(acwi_vol_full.reindex(idx, method="ffill").bfill().values.astype(float))
    return {"dates": list(idx), "port": port, "acwi": acwi}
