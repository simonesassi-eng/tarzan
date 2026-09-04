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

import math
from typing import Optional

import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics


def _window_money_pnl(
    pnl_series, actual_series, bucket: str
) -> tuple[Optional[float], Optional[float]]:
    """Real money P&L over a ``PERIOD_WINDOWS`` bucket, net of any
    contributions in that window.

    Reads the same ``window_anchor`` as the percentages beside it: the matrix's
    "5D" row used to walk seven CALENDAR days for the euros while its TWROR
    measured five sessions, so one row described two different spans.

    Returns ``(gain_eur, gain_pct)`` where the € gain is the delta of the
    cumulative P&L series across the window and the % expresses it over the
    portfolio value at the window start. Either may be None when the order
    path produced no series (holdings-only run) or the data is too short.
    """
    from tarzan.engine.stats import window_anchor

    if pnl_series is None:
        return None, None
    pnl = pnl_series.dropna()
    if len(pnl) < 2:
        return None, None
    cutoff = window_anchor(pnl, bucket)
    if cutoff is None:
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


def _window_twror(nav: Optional[pd.Series], bucket: str) -> Optional[float]:
    """Window TWROR (%) from the flow-adjusted NAV index over a
    ``PERIOD_WINDOWS`` bucket ("1d", "5d", "1m", …).

    Delegates to the engine's ``compute_period_return`` so the matrix cell and
    the chart line tell one story."""
    if nav is None:
        return None
    from tarzan.engine.stats import compute_period_return
    s = nav.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(s) < 2:
        return None
    return compute_period_return(s, bucket)


def _rebase_to_window(araw: "pd.Series", idx: "pd.DatetimeIndex",
                      *, blank_lead_in: bool = False) -> "list | None":
    """Rebase a level series over exactly the dates spanned by ``idx``.

    The denominator is the first real source observation inside the window;
    alignment uses the union of source and target dates before forward-fill so
    observations on a source-only trading day are never silently discarded.
    No observation after ``idx[-1]`` can affect the line.  Returns a percentage
    list aligned to ``idx``, or ``None`` when fewer than two real in-window
    observations exist.

    The window is opened by the caller on the ``window_anchor`` date (the last
    close on-or-before the trailing cutoff), so ``idx[0]`` is already the same
    reference every return table column uses — no per-series anchor override is
    needed here.

    ``blank_lead_in`` leaves the stretch BEFORE the series' first in-window
    observation as NaN instead of flattening it onto the anchor. Opt-in, because
    every line on every panel comes through here and the flattening is right for
    almost all of them: a source that merely skipped a session should not tear its
    own line in half. It is wrong for a series that genuinely does not exist yet —
    a target holding a fund four months old, drawn on a YTD window — where a flat
    run at 0% reads as "it went nowhere" across eight months it never saw.
    ``chart_pct_compact`` already draws one polyline per contiguous FINITE run, so
    a NaN lead-in becomes a gap rather than a claim.
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
    # anchor (0%) rather than carrying a stale pre-window source value — unless the
    # caller asked for the gap, in which case they stay NaN and the chart breaks the
    # line there.
    lead = aligned.index < in_win.index[0]
    aligned = (aligned.mask(lead, float("nan")) if blank_lead_in
               else aligned.mask(lead, anchor).bfill())
    return list(((aligned / anchor - 1.0) * 100.0).values.astype(float))


def _geo_benchmark_series(m: PortfolioMetrics, geo_name: Optional[str]) -> "pd.Series | None":
    """Price history of the geographic benchmark (the taxonomy row flagged
    ``is_benchmark_geo=TRUE``, e.g. iShares MSCI ACWI) from
    ``benchmark_histories``. Looked up by that configured name — no fuzzy
    matching — since ``benchmark_histories`` is keyed by the same curated
    instrument names as the taxonomy."""
    if not geo_name:
        return None
    return (m.benchmark_histories or {}).get(geo_name)


def _target_line(m: PortfolioMetrics, idx) -> "list | None":
    """The target allocation's cumulative return over ``idx``, or None.

    Drawn from wherever the target's own series starts, with the stretch before that
    left BLANK. It used to be withheld entirely when the target opened after the
    chart, because ``_rebase_to_window`` flattened the lead-in onto the anchor and a
    run at 0% reads as "the target went nowhere" across months it never saw. Blanking
    that stretch says the true thing instead — the target exists from here — and it is
    the honest answer for a target that names a young fund: measured on the reference
    book, one sleeve four months old truncated the common window to 83 days, which
    silently removed the target from YTD, 1Y, since-inception and both volatility
    panels at once.

    ``target_history`` is the INTERSECTION of its sleeves' histories, so the youngest
    sleeve sets that start date for the whole allocation. Nothing here can widen it;
    what this decides is only whether the part that does exist gets drawn.
    """
    raw = getattr(m, "target_history", None)
    if raw is None or len(raw) < 2 or len(idx) < 2:
        return None
    s = _norm_series(raw).replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()
    if len(s) < 2:
        return None
    return _rebase_to_window(s, pd.DatetimeIndex(idx), blank_lead_in=True)


def _perf_window(m: PortfolioMetrics, n_days: int = 30,
                 geo_name: Optional[str] = None,
                 bucket: str = "1m") -> Optional[dict]:
    """Comparable trailing performance window on one shared close boundary.

    ``bucket`` names the window in the shared vocabulary every return table uses
    ("1d", "5d", "1m", "3m", "ytd", "1y", …), so a panel and the column above it
    open on the same observation. It defaults to "1m" — the window this built
    before it could build any other — and ``n_days`` remains the fallback span
    for when no anchor resolves.

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

    # Open the window on the SAME observation the 1M return-table column anchors
    # on — window_anchor, which already rolls a weekend "today" back to the last
    # session — so every plotted line (portfolio + benchmark) shares the tables'
    # anchor and the chart's endpoints equal the table returns. A raw
    # n-calendar-day slice instead opened on whatever close fell n days back,
    # which on 21 Aug 2026 was the 22 Jul MSCI ACWI dip (104.7) and read +1.7%
    # while the 1M column, anchored a day earlier, read +0.57%. The benchmark is
    # the reference the digest is scrutinised against (vs Yahoo), so its anchor
    # governs the shared window; the portfolio falls back when none resolves.
    # ``n_days`` still bounds the span for the label/axis (~one month).
    from tarzan.engine.stats import window_anchor

    # Resolve the anchor on the RAW (tz-aware) benchmark — the exact series the
    # return tables call ``compute_period_return`` on — so the chart's ACWI
    # endpoint equals the 1M column to the basis point, then take its SESSION
    # DATE for slicing the chart's tz-naive calendar. Local wall time, not UTC:
    # the anchor bar is stamped at its venue's midnight, so a UTC trip moved a
    # European anchor to the previous calendar day and opened the window one
    # session early (see ``stats.normalize_index``).
    start = window_anchor(acwi_raw if acwi_all is not None else val_raw, bucket)
    if start is None:
        # The ``n_days`` fallback belongs to the DEFAULT window only. For a named
        # bucket a missing anchor means the book does not reach back that far, and
        # substituting a 30-day slice would draw one span under another's name — a
        # 1Y panel showing a month, on an eight-month book, labelled "1Y". No
        # window is the honest answer; the caller omits the panel.
        if bucket != "1m":
            return None
        start = val_all.index[-1] - pd.Timedelta(days=n_days)
    else:
        start = pd.Timestamp(start)
        if start.tz is not None:
            start = start.tz_localize(None)
        start = start.normalize()
    val = val_all[(val_all.index >= start) & (val_all.index <= common_end)]
    if len(val) < 2:
        return None
    idx = val.index

    twror = (
        _rebase_to_window(m.portfolio_history, idx)
        if m.portfolio_history is not None else None
    )
    acwi = (
        _rebase_to_window(acwi_all[acwi_all.index <= common_end], idx)
        if acwi_all is not None else None
    )

    # Both P&L lines are rebased on the window's own opening value, so each
    # reads as "what this measure added over the window" on the same axis as
    # TWROR — not as its since-inception level, which would sit far off the
    # window's scale and flatten the lines that belong to it.
    v0 = float(val.iloc[0]) or 1.0

    def _window_pct(source):
        # An empty series is not None (``unrealized_series`` defaults to an
        # empty Series, and the metrics shift can poison a whole series when
        # its last point is NaN), and reindexing one onto the window yields
        # all-NaN. A NaN line is not a line: it renders as "—" while the
        # semantic gate compares it numerically, so NaN != NaN blocks delivery
        # instead of simply omitting the series. Absent data must read as None.
        if source is None or len(source) == 0:
            return None
        s = _norm_series(source).reindex(idx, method="ffill").bfill()
        if not s.notna().any():
            return None
        return [float(p) / v0 * 100.0 for p in (s - s.iloc[0]).values]

    pnl_pct = _window_pct(m.pnl_series)
    unreal_pct = _window_pct(m.unrealized_series)
    target = _target_line(m, idx)

    dates = list(idx)
    endpoints = {
        "twror": float(twror[-1]) if twror else None,
        "acwi": float(acwi[-1]) if acwi else None,
        "target": float(target[-1]) if target else None,
        "pnl_pct": float(pnl_pct[-1]) if pnl_pct else None,
        "unreal_pct": float(unreal_pct[-1]) if unreal_pct else None,
    }
    return {
        "dates": dates,
        "value": list(val.values.astype(float)),
        "twror": twror,
        "acwi": acwi,
        "target": target,
        "pnl_pct": pnl_pct,
        "unreal_pct": unreal_pct,
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
    # Same absent-data rule as ``_perf_window._window_pct``: an empty or
    # all-NaN series must read as None (no line) rather than as a NaN line.
    if m.pnl_series is not None and len(m.pnl_series):
        pnl = _norm_series(m.pnl_series).reindex(idx, method="ffill").bfill()
        if pnl.notna().any():
            total_pct = list((pnl / (av - pnl).replace(0, float("nan")) * 100.0).bfill().values.astype(float))
    if m.unrealized_series is not None and len(m.unrealized_series):
        ur = _norm_series(m.unrealized_series).reindex(idx, method="ffill").bfill()
        if ur.notna().any():
            unreal_pct = list((ur / (av - ur).replace(0, float("nan")) * 100.0).bfill().values.astype(float))
    # MSCI ACWI since inception: anchor on the benchmark's first observation
    # at/after inception (its own real price, not one stale-filled from before
    # the portfolio's start), then sample the window — same anchoring rule as
    # the 30-day chart and the Performance section.
    acwi_si = None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    if acwi_raw is not None and len(nav_full):
        acwi_si = _rebase_to_window(acwi_raw, idx)
    return twror_si, total_pct, unreal_pct, acwi_si


def _perf_full_series(m: PortfolioMetrics, geo_name: Optional[str] = None,
                      max_points: int = 180) -> Optional[dict]:
    """The since-inception trajectory over the WHOLE date range (not the last
    30 days): cumulative TWROR (%), Total P&L (%) and MSCI ACWI (%) from
    inception to today, on a common daily index that is evenly downsampled to
    ``max_points`` so a multi-year series stays a light SVG. Reuses
    ``_perf_level_series`` for the cumulative math. None when unavailable.

    Keys mirror ``_perf_window`` so the chart builder is symmetric:
    ``{dates, twror, pnl_pct, unreal_pct, acwi}`` (any line may be None)."""
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
    twror_si, total_pct, unreal_pct, acwi_si = lvl
    return {
        "dates": list(idx),
        "twror": twror_si,
        "pnl_pct": total_pct,
        "unreal_pct": unreal_pct,
        "acwi": acwi_si,
        # Read through the same helper as the 30-day window rather than through
        # _perf_level_series' fixed 4-tuple, whose shape other callers depend on.
        "target": _target_line(m, list(idx)),
    }


def _rolling_ann_vol(level: "pd.Series", window: int) -> "pd.Series":
    """Rolling annualized volatility (%) of a price/level series.

    Sample standard deviation (ddof=1, mean removed) of the daily simple returns
    over ``window`` observations, scaled by √``TRADING_DAYS``. The same estimator
    ``stats.risk_metric_row`` reports in the RISK section, so the chart and the
    tile are the same number measured over different spans rather than two
    definitions of "volatility".

    NaN until a FULL window exists. It used to emit an estimate from half a
    window (``min_periods=window // 2``), which on a 21-day window is a σ off ten
    observations — a ±24% standard error — labelled as a 21-day figure. Invisible
    on the trailing panel, whose lead-in comes from earlier history; on a
    since-inception panel those are the opening points of the line.
    """
    from tarzan.engine.stats import TRADING_DAYS
    ret = level.pct_change()
    return ret.rolling(window).std() * (TRADING_DAYS ** 0.5) * 100.0


def _perf_vol_series(m: PortfolioMetrics, geo_name: Optional[str] = None,
                     n_days: Optional[int] = None, vol_window: int = 21,
                     max_points: int = 180,
                     bucket: Optional[str] = None) -> Optional[dict]:
    """Rolling annualized volatility of the portfolio NAV vs the geo benchmark,
    on a common daily index. The twin of ``_perf_window`` / ``_perf_full_series``
    for the "You vs the market" box's second (risk) row.

    ``n_days=None`` → the whole span from the first full-window estimate to today
    (downsampled to ``max_points``); ``n_days=30`` → the trailing window, whose
    lead-in already has a full window from earlier history.

    ``bucket`` overrides both with a named window ("5d", "1m", "ytd", "1y", …),
    resolved through the SAME ``window_anchor`` call ``_perf_window`` makes — on
    the benchmark's own raw series when there is one, the NAV otherwise — so a
    volatility panel and the return panel above it open on one observation. A
    bucket the book does not reach back far enough for yields None (no panel)
    rather than a shorter span wearing the bucket's name.

    Every series here is sampled on TRADING days (the order-derived NAV, the
    benchmark's own closes and the target's common window all are), so
    ``vol_window`` rows mean the same 21 sessions on each line and the √252
    scaling matches the sampling. A calendar-daily input would put ~29 calendar
    days in one line's window against 21 sessions in another's.

    Returns ``{dates, port, acwi, target, span, window_sigma}``. The three lines
    are % lists and ANY of them may be None — a window that holds no estimate has
    no line, the portfolio's included. ``span`` is one σ per series over the
    portfolio's whole life; ``window_sigma`` is one σ per series over THIS window,
    which is what a per-window caption states (the lines' end values are all
    today's rolling σ, identical in every window). None when unavailable."""
    from tarzan.engine.stats import TRADING_DAYS

    nav_full = _norm_series(m.portfolio_history) if m.portfolio_history is not None else None
    if nav_full is None or len(nav_full) < vol_window + 1:
        return None
    acwi_raw = _geo_benchmark_series(m, geo_name)
    acwi_full = _norm_series(acwi_raw) if acwi_raw is not None else None
    target_raw = getattr(m, "target_history", None)
    target_full = _norm_series(target_raw) if target_raw is not None else None

    port_vol_full = _rolling_ann_vol(nav_full, vol_window)
    # The line cannot open before its first real estimate. Back-filling the
    # leading NaNs (what this did) paints the first ``vol_window`` points with a
    # figure measured later — a flat opening run the data never supported, which
    # on the since-inception panel is the part of the line a reader reads first.
    first = port_vol_full.first_valid_index()
    if first is None:
        return None
    # Choose the display index (same convention as the return charts).
    if bucket is not None:
        from tarzan.engine.stats import window_anchor
        start = window_anchor(acwi_raw if acwi_raw is not None else m.portfolio_history,
                              bucket)
        if start is None:
            return None
        start = pd.Timestamp(start)
        if start.tz is not None:
            start = start.tz_localize(None)
        start = start.normalize()
        idx = nav_full.index[(nav_full.index >= start) & (nav_full.index >= first)]
    elif n_days is None:
        idx = nav_full.index[nav_full.index >= first]
        if len(idx) > max_points:
            pos = sorted({int(round(i * (len(idx) - 1) / (max_points - 1)))
                          for i in range(max_points)})
            idx = idx[pos]
    else:
        cutoff = nav_full.index[-1] - pd.Timedelta(days=n_days)
        idx = nav_full.index[(nav_full.index >= cutoff) & (nav_full.index >= first)]
    if len(idx) < 2:
        return None

    def _line(v: "pd.Series"):
        """A rolling-vol line sampled on ``idx``, or None when the window holds
        no estimate at all.

        ``first_valid_index`` above only proves an estimate exists SOMEWHERE in
        the history. A NAV sitting at zero across the window — a fully liquidated
        stretch, or one pinned there by a re-entry flow — has no computable
        returns, so the reindexed line is all-NaN. Same rule as
        ``_perf_window._window_pct``: a NaN line is not a line. It has no axis
        bounds (``_charts.nice_ticks`` cannot floor a NaN, which used to abort the
        WHOLE issue), and the legend beside it would name a series nothing drew.

        ``math.isfinite`` rather than ``notna``, because ``pct_change`` produces
        ±inf on a zero-to-positive step and an infinite bound is no more
        plottable than a NaN one.
        """
        vals = list(v.reindex(idx, method="ffill").values.astype(float))
        return vals if any(math.isfinite(x) for x in vals) else None

    port = _line(port_vol_full)

    # One σ per line over the PORTFOLIO's own life, so the three are comparable.
    # Each series' own full history is not: the benchmark holds two years where
    # the book holds eight months, and reporting 14.76% beside the book's 10.51%
    # compares two different periods and reads as a risk gap. Clipped to the
    # shared span it is 11.79%.
    #
    # The book's own figure is its whole life already, so it equals
    # ``stats.risk_metric_row`` on the flow-adjusted NAV — which is NOT the figure
    # the RISK section prints. That table renders ``historical_risk``, whose
    # portfolio row is a current-weight static backtest over the common window of
    # holdings with ≥1Y of history (10.77% live on 26 Aug 2026 against this
    # 10.51%): a different construction over a different span, by explicit design
    # there. Two honest answers to two questions, not one number twice.
    def _clipped_vol(full, lo, hi, floor: int):
        """σ of ``full`` over ``[lo, hi]``, annualized, or None when the clip holds
        too few returns to estimate one."""
        if full is None or len(full) < 3:
            return None
        w = full[(full.index >= lo) & (full.index <= hi)]
        rr = w.pct_change().dropna()
        if len(rr) < floor:
            return None
        return float(rr.std(ddof=1) * (TRADING_DAYS ** 0.5) * 100.0)

    def _span_vol(full):
        return _clipped_vol(full, nav_full.index[0], nav_full.index[-1],
                            vol_window)

    span = {"port": _span_vol(nav_full), "acwi": _span_vol(acwi_full),
            "target": _span_vol(target_full),
            "from": nav_full.index[0], "to": nav_full.index[-1]}

    # σ over THIS window, which is a different question from the line's end value.
    # The line is a ROLLING 21-session estimate, so its last point is today's σ —
    # the same number in every window, because every window ends today. Six panels
    # whose end labels all read 6.47% say nothing about the windows. This is the
    # per-window figure: one σ measured across the window's own returns, which
    # differs panel to panel and answers "how rough was this stretch".
    #
    # The floor is 3 returns, not ``vol_window``: a 5-session window has five, and
    # refusing to estimate there would blank the caption on the very panels a
    # reader checks after a bad week. A σ off five returns is noisy and the caption
    # names the window, so the span it covers is never in doubt.
    _lo, _hi = idx[0], idx[-1]
    window_sigma = {
        "port": _clipped_vol(nav_full, _lo, _hi, 3),
        "acwi": _clipped_vol(acwi_full, _lo, _hi, 3),
        "target": _clipped_vol(target_full, _lo, _hi, 3),
    }

    def _bench_vol(full):
        """A reference line's rolling vol over ``idx``, or None when it holds no
        estimate at all inside the window.

        It used to demand a full-window estimate at the chart's LEFT EDGE, on the
        stated grounds that a reference opening inside the window "would otherwise be
        back-filled with its first computable figure across a stretch it says nothing
        about". Measured: it would not. ``_line`` reindexes with ``method="ffill"``,
        which fills FORWARD only, so dates before the first estimate stay NaN — 114 of
        176 on a reference opening mid-window — and ``chart_pct_compact`` breaks its
        polyline there. The back-fill that justified the demand is not in this code.

        So the demand is gone, and what it was costing is the whole point: a target
        naming a fund four months old lost its volatility line on every panel wider
        than its own life, when the part that exists was drawable all along. The
        remaining guard is ``_line``'s — a series with nothing finite anywhere is not
        a line — which is the real failure this protects against.
        """
        if full is None or len(full) < vol_window + 1:
            return None
        return _line(_rolling_ann_vol(full, vol_window))

    return {"dates": list(idx), "port": port, "acwi": _bench_vol(acwi_full),
            "target": _bench_vol(target_full), "span": span,
            "window_sigma": window_sigma}


def benchmark_gap_pp(m: PortfolioMetrics,
                     geo_name: Optional[str] = None) -> Optional[float]:
    """Lifetime TWROR minus the geography benchmark's lifetime cumulative, in
    percentage points. None when either side is unavailable.

    Both terms are already computed and already printed side by side in the
    since-inception chart's legend ("TWROR +11.28%" next to "MSCI ACWI
    +14.16%"), so the difference is a subtraction rather than a new estimate.
    One helper so the masthead, the state tile and the section subtitle cannot
    disagree about the same number.
    """
    if m.twror_pct is None:
        return None
    full = _perf_full_series(m, geo_name)
    if not full or not full.get("acwi"):
        return None
    try:
        return float(m.twror_pct) - float(full["acwi"][-1])
    except (TypeError, ValueError, IndexError):
        return None


def benchmark_gap_history(m: PortfolioMetrics,
                          geo_name: Optional[str] = None) -> Optional[dict]:
    """How the gap against the geography benchmark got to where it is.

    Returns ``{now_pp, peak_pp, peak_when, turn_when}`` where ``peak_when`` is
    the month the lead was widest and ``turn_when`` the month the gap first went
    negative after that peak (``None`` while the portfolio is still ahead). All
    of it is read off the same two cumulative series the since-inception chart
    already draws, so nothing here is estimated -- the figures are the distance
    between two lines the reader can see.
    """
    full = _perf_full_series(m, geo_name)
    if not full or not full.get("acwi") or not full.get("twror"):
        return None
    dates, port, bench = full["dates"], full["twror"], full["acwi"]
    n = min(len(dates), len(port), len(bench))
    if n < 2:
        return None
    gaps = [float(port[i]) - float(bench[i]) for i in range(n)]
    peak_i = max(range(n), key=lambda i: gaps[i])
    turn_i = next((i for i in range(peak_i, n) if gaps[i] < 0), None)

    def _month(i):
        return pd.Timestamp(dates[i]).strftime("%b")

    return {
        "now_pp": gaps[-1],
        "peak_pp": gaps[peak_i],
        "peak_when": _month(peak_i),
        "turn_when": _month(turn_i) if turn_i is not None else None,
    }
