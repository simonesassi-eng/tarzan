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
    """tz-naive, calendar-day-normalised, de-duplicated copy of a series."""
    s = s.copy()
    ix = s.index
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    s.index = pd.DatetimeIndex(ix).normalize()
    return s[~s.index.duplicated(keep="last")]


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
    ``days`` calendar days (mirrors :func:`_window_money_pnl`'s as-of logic)."""
    if nav is None:
        return None
    s = nav.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(s) < 2:
        return None
    cutoff = s.index[-1] - pd.Timedelta(days=days)
    prior = s[s.index <= cutoff]
    base = float(prior.iloc[-1]) if len(prior) else float(s.iloc[0])
    return (float(s.iloc[-1]) / base - 1.0) * 100.0 if base > 0 else None


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
    """Last ``n_days`` of value (€), TWROR (%), MSCI ACWI (%) and P&L (€→%),
    on a common daily index, plus deposit/withdrawal markers. Reuses the
    order-derived series (no recomputation). None when unavailable."""
    val_raw = m.actual_value_series
    if val_raw is None or len(val_raw) < 2:
        return None
    val = _norm_series(val_raw).replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(val) < 2:
        return None
    cutoff = val.index[-1] - pd.Timedelta(days=n_days)
    val = val[val.index >= cutoff]
    if len(val) < 2:
        return None
    idx = val.index

    nav = (_norm_series(m.portfolio_history).reindex(idx, method="ffill").bfill()
           if m.portfolio_history is not None else None)
    twror = (list(((nav / nav.iloc[0] - 1.0) * 100.0).values.astype(float))
             if nav is not None and float(nav.iloc[0]) else None)

    acwi = None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    if acwi_raw is not None and len(acwi_raw) >= 2:
        a = _norm_series(acwi_raw).reindex(idx, method="ffill").bfill()
        if float(a.iloc[0]):
            acwi = list(((a / a.iloc[0] - 1.0) * 100.0).values.astype(float))

    pnl_pct = None
    if m.pnl_series is not None:
        pnl = _norm_series(m.pnl_series).reindex(idx, method="ffill").bfill()
        base = (pnl - pnl.iloc[0])
        v0 = float(val.iloc[0]) or 1.0
        pnl_pct = [float(p) / v0 * 100.0 for p in base.values]

    dates = list(idx)
    return {
        "dates": dates,
        "value": list(val.values.astype(float)),
        "twror": twror,
        "acwi": acwi,
        "pnl_pct": pnl_pct,
        "flows": _flow_list(m.external_flows, dates[0], dates[-1]),
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
    # MSCI ACWI since inception: anchor the benchmark at the portfolio's
    # inception value (same basis as twror_si) and sample the window.
    acwi_si = None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    if acwi_raw is not None and len(acwi_raw) >= 2 and len(nav_full):
        a_full = _norm_series(acwi_raw)
        a_at_inception = a_full.reindex(nav_full.index, method="ffill").bfill()
        if len(a_at_inception) and float(a_at_inception.iloc[0]):
            acwi_full = (a_full / float(a_at_inception.iloc[0]) - 1.0) * 100.0
            acwi_si = list(acwi_full.reindex(idx, method="ffill").bfill().values.astype(float))
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
