"""Benchmark series construction and benchmark-relative metrics.

This is the network-touching half of what used to live in ``metrics.py``:
it fetches benchmark price histories (via the enricher's memoized layer),
builds blended series (e.g. 60/40), and computes the standard metric set
for a benchmark or a performance row. The pure math it relies on lives in
``tarzan.engine.stats``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import dataclass
import logging
import operator
from threading import RLock
from typing import Optional

import pandas as pd

from tarzan import config as cfg
from tarzan.engine.stats import (
    DAYS_PER_YEAR,
    PERIOD_WINDOWS,
    PreparedStatistics,
    PreparedWeeklyBenchmark,
    compute_period_return,
    compute_ytd_return,
    _compute_beta_alpha,
    normalize_index,
    rf_annual_pct,
    risk_metric_row,  # noqa: F401 -- preserve the historical module re-export
    statistical_series,
)

logger = logging.getLogger(__name__)

BENCHMARKS = cfg.benchmarks()


def _clip_to_window(series: pd.Series, start, end) -> pd.Series:
    """Return the slice of ``series`` within ``[start, end]`` (inclusive).

    Indices are normalized to tz-naive dates so a benchmark series and the
    portfolio window compare cleanly regardless of timezone. This is what
    makes the risk comparison apples-to-apples: every benchmark's risk
    metrics are computed over the *same* span as the portfolio's own
    (short) track record, instead of the benchmark's full multi-year
    history. Empty/None input passes through as an empty series.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    def _naive(ts):
        # Session date, not instant: see stats.normalize_session_index.
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return ts.normalize()

    s = normalize_index(series)
    lo, hi = _naive(start), _naive(end)
    return s[(s.index >= lo) & (s.index <= hi)]


class BenchmarkPreparation:
    """Bounded, phase-owned weekly facts for one exact stable reference.

    No ticker/ISIN inference or cross-run cache. Construct after quote stamping
    and keep the source stable for the phase. Each cached window is clipped
    BEFORE weekly resampling, using the same venue-local dates as the original
    helper. The original source and its evidence remain independently owned.
    """

    def __init__(self, reference: pd.Series, *, max_windows: int = 128):
        max_windows = operator.index(max_windows)
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        self._source = reference
        self.full = PreparedWeeklyBenchmark(reference)
        self._max_windows = max_windows
        self._windows: OrderedDict[
            tuple[pd.Timestamp, pd.Timestamp], PreparedWeeklyBenchmark
        ] = OrderedDict()
        self._lock = RLock()

    def require_source(self, reference: pd.Series) -> None:
        if reference is not self._source:
            raise ValueError("benchmark preparation belongs to a different source")

    def window(self, start, end) -> PreparedWeeklyBenchmark:
        def _session_date(timestamp):
            timestamp = pd.Timestamp(timestamp)
            if timestamp.tz is not None:
                timestamp = timestamp.tz_localize(None)
            return timestamp.normalize()

        key = (_session_date(start), _session_date(end))
        with self._lock:
            prepared = self._windows.get(key)
            if prepared is not None:
                self._windows.move_to_end(key)
                return prepared
            # Use the established helper, including its duplicate-label
            # policy; never clip a pre-resampled whole-history tape instead.
            history = _clip_to_window(self.full.history, *key)
            prepared = PreparedWeeklyBenchmark(history)
            self._windows[key] = prepared
            if len(self._windows) > self._max_windows:
                self._windows.popitem(last=False)
            return prepared


@dataclass(frozen=True)
class ResolvedBenchmark:
    """One benchmark after the run-scoped ticker preprocessing boundary.

    ``requested_ticker`` is input provenance only.  Every analytical and
    presentation consumer must use ``ticker`` and ``history``; both identify
    the same exact provider listing.
    """

    name: str
    requested_ticker: str
    ticker: str
    history: pd.Series
    # The same series before the FX step, and the LISTING's currency. Every
    # portfolio-level consumer reads ``history`` (EUR); the per-instrument return
    # columns read ``history_native``, because a return is a ratio and FX does not
    # divide out of one. 24 of the 62 tracked instruments trade in USD.
    # Preprocessing moves the legacy native-series attr into this field; the
    # canonical EUR series must not carry a second, independently stale copy.
    history_native: Optional[pd.Series] = None
    currency: Optional[str] = None


_NATIVE_ATTR = "tarzan_native_series"
_CURRENCY_ATTR = "tarzan_listing_currency"


def _drop_weekends(series: pd.Series) -> pd.Series:
    """Drop weekend-dated rows from a benchmark price series.

    Every Tarzan benchmark is a weekday-traded ETF/index/fund (the crypto
    target is 0; bitcoin appears only *inside* weekday-listed ETFs like RSSX),
    so a Saturday/Sunday close is never real — it is an FX-conversion artifact
    (``convert_to_eur`` aligns against a ~24/5 FX series) or a weekend "today"
    stamp. Left in, such a point sits between real closes and steals the window
    anchor: ``window_anchor`` takes the last row on-or-before the cutoff, so a
    weekend row nearer the cutoff than the real close is picked instead. That
    drove MSCI ACWI (ISAC.MI) to a reported +1.7% against a real +0.08% on
    22 Aug 2026.

    The weekday is the VENUE's, read off the tz-aware index directly. Collapsing
    to UTC first — which this used to do — moves a venue east of UTC back a day,
    because a daily bar is stamped at local midnight: Milan's Monday
    ``00:00+02:00`` becomes Sunday ``22:00`` in UTC. That silently deleted EVERY
    MONDAY from every Milan/Xetra benchmark series, 118 of 576 rows on ISAC.MI —
    a fifth of the history — and with the last row a Friday, the Monday 1D
    collapsed onto the series end and reported as unavailable. A genuine
    FX-artifact weekend row is still dropped: its local weekday is the weekend
    too, which is what this exists to remove.
    """
    if series is None or series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return series
    return series[series.index.weekday < 5]


def _fetch_benchmark_history(ticker: str) -> pd.Series:
    """Resolve and fetch one benchmark for preprocessing only.

    This is the sole network/resolution boundary for benchmark identities.
    Downstream code receives :class:`ResolvedBenchmark` objects and must never
    call this function or reconstruct a symbol from the taxonomy ticker.
    """
    from tarzan.data import enricher as _enr

    from tarzan.runtime.memo import memoized

    return memoized(
        "enricher.benchmark", ticker, lambda: _fetch_benchmark_uncached(ticker),
        legacy=_enr._benchmark_memo,
    )


def _fetch_benchmark_uncached(ticker: str) -> pd.Series:
    from tarzan.data import enricher as _enr

    expected_name = cfg.name_for(None, ticker) or ""
    data = _enr._fetch_ticker_data(ticker, expected_name=expected_name)
    selected_ticker = str(
        data.get(_enr._TICKER_SYMBOL_KEY) or ticker
    ).strip()
    history = data.get("history", pd.DataFrame())
    if history.empty:
        series = pd.Series(dtype=float)
        # ``native`` and ``currency`` are only bound in the else branch, and both
        # are read below — an empty-history benchmark raised UnboundLocalError,
        # which ``preprocess_benchmarks`` then reported as a resolution failure.
        native = pd.Series(dtype=float)
        currency = ""
    else:
        prices = history["Close"]
        # A EUR venue is EUR by definition, so a flaky info.currency cannot
        # trigger a spurious FX conversion of an already-EUR benchmark (this
        # corrupted MSCI ACWI / ISAC.MI). A missing currency is likewise not
        # evidence of USD; only a positively non-EUR currency converts.
        currency = _enr.venue_currency(selected_ticker) or data.get("info", {}).get("currency")
        native = _drop_weekends(prices)
        series = (
            _drop_weekends(_enr.convert_to_eur(prices, currency))
            if currency and currency != "EUR"
            else native
        )

    # The exact provider symbol is part of the series contract.  ``name`` is
    # intentionally the same full symbol so serialized/debug series cannot
    # silently regress to the venue-neutral input ticker.
    series = series.copy()
    series.name = selected_ticker
    series.attrs.update({
        "resolved_ticker": selected_ticker,
        "requested_ticker": ticker,
        # Carried on attrs rather than through a second return value, the way the
        # FX evidence already travels: this function is injected in tests and
        # called in several places, and a widened signature would touch them all.
        _NATIVE_ATTR: native.reindex(series.index).dropna() if len(series) else native,
        # Unknown stays unknown: the returns table omits the mark rather than claim
        # euro for a listing whose currency the provider never reported. Empty
        # string, not None, because this travels into a pandas column where a NaN
        # renders as "[NAN]".
        _CURRENCY_ATTR: currency or "",
    })
    if not currency and len(series):
        logger.warning(
            "%s reported no listing currency; its returns are quoted unconverted "
            "and carry no currency mark", selected_ticker)

    return series


def preprocess_benchmarks(
    definitions: Mapping[str, str] | None = None,
    *,
    fetch_history: Callable[[str], pd.Series] | None = None,
    required_names: tuple[str, ...] = (),
    workers: int = 1,
    before_optional: Callable[[Mapping[str, ResolvedBenchmark]], None] | None = None,
) -> tuple[dict[str, ResolvedBenchmark], tuple[str, ...]]:
    """Resolve the complete benchmark universe exactly once for one run.

    The returned catalog is keyed by the curated display name, but every
    record exposes only one operational ticker: the full provider symbol that
    supplied its historical series.  Resolution failures remain explicit and
    are returned for the pre-delivery semantic gate.

    Required analytical references are resolved before the optional catalog.
    An owned caller may prefetch primary evidence through ``before_optional``;
    this callback receives only positively resolved required identities and
    runs before the optional deadline is entered.
    Optional work shares a bounded deadline across workers, but the returned
    catalog and errors retain input order regardless of scheduling.
    """
    requested = dict(BENCHMARKS if definitions is None else definitions)
    fetch = fetch_history or _fetch_benchmark_history
    if type(workers) is not int or workers < 1:
        raise ValueError("benchmark workers must be a positive integer")
    from tarzan.runtime.transport import current_budget

    budget = current_budget()

    def resolve(name, input_ticker):
        try:
            history = fetch(input_ticker)
        except Exception as error:  # noqa: BLE001
            return None, f"{name}: resolution failed ({type(error).__name__})"
        if history is None or history.empty or len(history) < 2:
            return None, f"{name}: no usable history for {input_ticker}"
        resolved_ticker = str(
            history.attrs.get("resolved_ticker")
            or history.attrs.get("provider_ticker")
            or input_ticker
        ).strip()
        if not resolved_ticker:
            return None, f"{name}: resolved ticker is empty"
        canonical_history = history.copy()
        canonical_history.name = resolved_ticker
        canonical_history.attrs.update({
            "resolved_ticker": resolved_ticker,
            "requested_ticker": input_ticker,
        })
        # Pandas' copy above already deep-copied attrs, including the native
        # tape. Give that independent tape one owner instead of retaining an
        # unused duplicate on every downstream EUR-series transformation.
        native = canonical_history.attrs.pop(_NATIVE_ATTR, None)
        if native is not None and len(native):
            native.name = resolved_ticker
        return ResolvedBenchmark(
            name=name,
            requested_ticker=input_ticker,
            ticker=resolved_ticker,
            history=canonical_history,
            history_native=native,
            # Same rule as the attr above: re-fabricating EUR here would undo it,
            # and this branch also covers an injected fetch stub that sets no attr.
            currency=canonical_history.attrs.get(_CURRENCY_ATTR) or "",
        ), None

    required = set(required_names)
    outcomes = {}
    for name, input_ticker in requested.items():
        if name in required:
            outcomes[name] = resolve(name, input_ticker)

    if before_optional is not None:
        before_optional({
            name: outcome[0] for name, outcome in outcomes.items()
            if outcome[0] is not None
        })

    def optional(name, input_ticker):
        if budget is not None and not budget.optional_request_allowed("yahoo"):
            return None, f"{name}: optional provider budget exhausted"
        return resolve(name, input_ticker)

    extras = [(name, ticker) for name, ticker in requested.items() if name not in required]
    scope = budget.optional_work("optional_benchmarks") if budget is not None else nullcontext()
    with scope:
        if workers == 1:
            outcomes.update({name: optional(name, ticker) for name, ticker in extras})
        elif extras:
            # Native requests in each worker inherit both the run and this
            # stage's deadline; no global SDK batch context is involved.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tarzan-benchmark") as pool:
                pending = {
                    name: pool.submit(copy_context().run, optional, name, ticker)
                    for name, ticker in extras
                }
                outcomes.update({name: future.result() for name, future in pending.items()})

    catalog = {
        name: outcomes[name][0] for name in requested if outcomes[name][0] is not None
    }
    errors = tuple(outcomes[name][1] for name in requested if outcomes[name][1] is not None)

    logger.info(
        "Benchmark preprocessing resolved %d/%d full provider tickers",
        len(catalog), len(requested),
    )
    return catalog, errors


def _compute_single_benchmark_metrics(
    bench: pd.Series,
    ab_benchmark: "pd.Series | None" = None,
    rf_daily=None,
    *,
    benchmark_preparation: BenchmarkPreparation | None = None,
) -> dict:
    """Compute the standard set of metrics for a benchmark series.

    Args:
        bench: The benchmark price series (in EUR).
        ab_benchmark: Optional reference series used to compute α and β
            for ``bench``. When provided, α/β are computed via the same
            CAPM logic used for the portfolio (OLS on aligned weekly
            returns). Pass the same series as ``bench`` to get the trivial
            β=1.00 / α=0 (vs itself).
        rf_daily: Optional time-varying daily risk-free path for Sharpe/Sortino
            and for alpha's native-return-date mean inside the shared OLS
            window. ``None`` falls back to the scalar ``RISK_FREE_RATE``;
            a scalar preserves the explicit annual-percent alpha contract.
    """
    # Numerical consumers do not need the full provider-evidence payload.
    # Keep it on the source catalog; retain the listing/calendar on each copy.
    prepared = PreparedStatistics(bench, rf_daily)
    bench = prepared.series
    weekly_reference = None
    if benchmark_preparation is not None:
        benchmark_preparation.require_source(ab_benchmark)
        weekly_reference = benchmark_preparation.full
        ab_benchmark = weekly_reference.history
    elif ab_benchmark is not None:
        ab_benchmark = statistical_series(ab_benchmark)
    rf_daily = prepared.rf_daily
    metrics = {
        **prepared.risk_metrics(),
        **{k: compute_period_return(bench, k) for k in PERIOD_WINDOWS},
        "ytd": compute_ytd_return(bench),
        "alpha": float("nan"),
        "beta": float("nan"),
    }
    if ab_benchmark is not None and not ab_benchmark.empty and len(ab_benchmark) > 1:
        beta, alpha = _compute_beta_alpha(
            bench, ab_benchmark, metrics["cagr"],
            risk_free=(
                None if isinstance(rf_daily, pd.Series) else rf_annual_pct(rf_daily)
            ),
            input_kind="prices",
            rf_daily=rf_daily,
            prepared_benchmark=weekly_reference,
        )
        metrics["alpha"] = alpha
        metrics["beta"] = beta
    return metrics


def _add_mix_to_histories(
    key_histories: dict,
    initial_value: float,
    catalog: Mapping[str, ResolvedBenchmark],
) -> None:
    """Build the optional 60/40 line from preprocessed histories only."""
    mix_cfg = cfg.mix_60_40()
    equity = next(
        (record for record in catalog.values()
         if record.requested_ticker == mix_cfg.get("equity_ticker")),
        None,
    )
    bond = next(
        (record for record in catalog.values()
         if record.requested_ticker == mix_cfg.get("bond_ticker")),
        None,
    )
    if equity is None or bond is None or equity.name not in key_histories:
        return
    combined = pd.DataFrame({
        "eq": key_histories[equity.name],
        "bd": bond.history,
    }).dropna()
    if combined.empty:
        return
    eq_n = combined["eq"] / combined["eq"].iloc[0]
    bd_n = combined["bd"] / combined["bd"].iloc[0]
    eq_w = mix_cfg.get("equity_weight", 0.6)
    bd_w = mix_cfg.get("bond_weight", 0.4)
    key_histories["60/40 ACWI+Bond"] = (
        eq_n * eq_w + bd_n * bd_w
    ) * initial_value


def _populate_perf_row(row: dict, s: pd.Series, bench_history: pd.Series,
                       rf_daily=None, *,
                       benchmark_preparation: BenchmarkPreparation | None = None,
                       ) -> None:
    """Populate a performance row dict with period returns + risk metrics + alpha/beta.

    All risk metrics (CAGR, Vol, Sharpe, Sortino, Max DD, Alpha, Beta) use the full series `s`
    (already capped to max 5 years). Period Used reflects the actual window covered.

    ``rf_daily`` is the optional daily risk-free path used for Sharpe/Sortino
    and alpha's native-return-date mean in the overlapping weekly OLS window;
    ``None`` falls back to the scalar rate, and scalar annual overrides retain
    their original authority.
    """
    prepared = PreparedStatistics(s, rf_daily)
    s = prepared.series
    if benchmark_preparation is not None:
        benchmark_preparation.require_source(bench_history)
    elif bench_history is not None:
        bench_history = statistical_series(bench_history)
    rf_daily = prepared.rf_daily
    # Period returns — single shared bucket→window mapping (stats.PERIOD_WINDOWS)
    # so every table in Tarzan measures the same windows.
    for key in PERIOD_WINDOWS:
        row[key] = compute_period_return(s, key)
    row["ytd"] = compute_ytd_return(s)

    # Risk metrics on full series (same shared block as the benchmark rows).
    row.update(prepared.risk_metrics())
    daily_ret = prepared.daily_returns
    cagr_val = row["cagr"] if isinstance(row["cagr"], (int, float)) else 0.0
    # Alpha/Beta vs the reference benchmark, on the *overlapping* window so
    # the figures are apples-to-apples with `s`: a 6-month track record is
    # measured against the benchmark over those same 6 months, not the
    # benchmark's full multi-year history (which made α incoherent before).
    row["alpha"] = float("nan")
    row["beta"] = float("nan")
    if (bench_history is not None and len(bench_history) > 1
            and len(s) > 1 and not daily_ret.empty):
        weekly_reference = None
        if benchmark_preparation is None:
            bench_win = _clip_to_window(
                bench_history, s.index.min(), s.index.max()
            )
        else:
            weekly_reference = benchmark_preparation.window(
                s.index.min(), s.index.max()
            )
            bench_win = weekly_reference.history
        if len(bench_win) > 1:
            beta, alpha = _compute_beta_alpha(
                s, bench_win, cagr_val,
                risk_free=(
                    None if isinstance(rf_daily, pd.Series) else rf_annual_pct(rf_daily)
                ),
                input_kind="prices",
                rf_daily=rf_daily,
                prepared_benchmark=weekly_reference,
            )
            row["beta"] = beta
            row["alpha"] = alpha

    # Period Used: "5Y", "3.2Y", etc.
    if len(s) >= 2:
        days_covered = (s.index[-1] - s.index[0]).days
        years_covered = days_covered / DAYS_PER_YEAR
        if years_covered >= 4.9:
            row["period_used"] = "5Y"
        elif years_covered >= 1.0:
            row["period_used"] = f"{years_covered:.1f}Y"
        else:
            months = int(years_covered * 12)
            row["period_used"] = f"{months}M"
    else:
        row["period_used"] = "—"
