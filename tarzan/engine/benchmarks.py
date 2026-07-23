"""Benchmark series construction and benchmark-relative metrics.

This is the network-touching half of what used to live in ``metrics.py``:
it fetches benchmark price histories (via the enricher's memoized layer),
builds blended series (e.g. 60/40), and computes the standard metric set
for a benchmark or a performance row. The pure math it relies on lives in
``tarzan.engine.stats``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging

import pandas as pd

from tarzan import config as cfg
from tarzan.engine.stats import (
    DAYS_PER_YEAR,
    PERIOD_DAYS,
    compute_period_return,
    compute_ytd_return,
    _compute_beta_alpha,
    normalize_index,
    risk_metric_row,
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
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.normalize()

    s = normalize_index(series)
    lo, hi = _naive(start), _naive(end)
    return s[(s.index >= lo) & (s.index <= hi)]


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


def _fetch_benchmark_history(ticker: str) -> pd.Series:
    """Resolve and fetch one benchmark for preprocessing only.

    This is the sole network/resolution boundary for benchmark identities.
    Downstream code receives :class:`ResolvedBenchmark` objects and must never
    call this function or reconstruct a symbol from the taxonomy ticker.
    """
    from tarzan.data import enricher as _enr

    with _enr._net_lock:
        if ticker in _enr._benchmark_memo:
            return _enr._benchmark_memo[ticker]

    expected_name = cfg.name_for(None, ticker) or ""
    data = _enr._fetch_ticker_data(ticker, expected_name=expected_name)
    selected_ticker = str(
        data.get(_enr._TICKER_SYMBOL_KEY) or ticker
    ).strip()
    history = data.get("history", pd.DataFrame())
    if history.empty:
        series = pd.Series(dtype=float)
    else:
        prices = history["Close"]
        # A missing currency is not evidence of USD; leave an already-EUR
        # series untouched rather than applying a speculative FX conversion.
        currency = data.get("info", {}).get("currency")
        series = (
            _enr.convert_to_eur(prices, currency)
            if currency and currency != "EUR"
            else prices
        )

    # The exact provider symbol is part of the series contract.  ``name`` is
    # intentionally the same full symbol so serialized/debug series cannot
    # silently regress to the venue-neutral input ticker.
    series = series.copy()
    series.name = selected_ticker
    series.attrs.update({
        "resolved_ticker": selected_ticker,
        "requested_ticker": ticker,
    })

    with _enr._net_lock:
        _enr._benchmark_memo[ticker] = series
    return series


def preprocess_benchmarks(
    definitions: Mapping[str, str] | None = None,
    *,
    fetch_history: Callable[[str], pd.Series] | None = None,
) -> tuple[dict[str, ResolvedBenchmark], tuple[str, ...]]:
    """Resolve the complete benchmark universe exactly once for one run.

    The returned catalog is keyed by the curated display name, but every
    record exposes only one operational ticker: the full provider symbol that
    supplied its historical series.  Resolution failures remain explicit and
    are returned for the pre-delivery semantic gate.
    """
    requested = BENCHMARKS if definitions is None else definitions
    fetch = fetch_history or _fetch_benchmark_history
    catalog: dict[str, ResolvedBenchmark] = {}
    errors: list[str] = []
    for name, input_ticker in requested.items():
        try:
            history = fetch(input_ticker)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{name}: resolution failed ({type(error).__name__})")
            continue
        if history is None or history.empty or len(history) < 2:
            errors.append(f"{name}: no usable history for {input_ticker}")
            continue
        resolved_ticker = str(
            history.attrs.get("resolved_ticker")
            or history.attrs.get("provider_ticker")
            or input_ticker
        ).strip()
        if not resolved_ticker:
            errors.append(f"{name}: resolved ticker is empty")
            continue
        canonical_history = history.copy()
        canonical_history.name = resolved_ticker
        canonical_history.attrs.update({
            "resolved_ticker": resolved_ticker,
            "requested_ticker": input_ticker,
        })
        catalog[name] = ResolvedBenchmark(
            name=name,
            requested_ticker=input_ticker,
            ticker=resolved_ticker,
            history=canonical_history,
        )

    logger.info(
        "Benchmark preprocessing resolved %d/%d full provider tickers",
        len(catalog), len(requested),
    )
    return catalog, tuple(errors)


def _build_benchmark_series(
    name: str,
    ticker: str,
    initial_value: float,
    catalog: Mapping[str, ResolvedBenchmark] | None = None,
) -> pd.Series:
    """Compatibility projection from the already-preprocessed catalog.

    No resolution or provider access is permitted here.  ``ticker`` must equal
    the catalog's full ticker; otherwise an empty series makes the contract
    violation visible instead of silently selecting another listing.
    """
    record = (catalog or {}).get(name)
    if record is None or record.ticker != ticker:
        return pd.Series(dtype=float)
    return record.history


def _compute_single_benchmark_metrics(
    bench: pd.Series,
    ab_benchmark: "pd.Series | None" = None,
) -> dict:
    """Compute the standard set of metrics for a benchmark series.

    Args:
        bench: The benchmark price series (in EUR).
        ab_benchmark: Optional reference series used to compute α and β
            for ``bench``. When provided, α/β are computed via the same
            CAPM logic used for the portfolio (regression of daily
            returns on overlap window; α annualized using benchmark
            CAGR). Pass the same series as ``bench`` to get the trivial
            β=1.00 / α=0 (vs itself).
    """
    metrics = {
        **risk_metric_row(bench),
        **{k: compute_period_return(bench, d) for k, d in PERIOD_DAYS.items()},
        "ytd": compute_ytd_return(bench),
        "alpha": float("nan"),
        "beta": float("nan"),
    }
    if ab_benchmark is not None and not ab_benchmark.empty and len(ab_benchmark) > 1:
        beta, alpha = _compute_beta_alpha(bench, ab_benchmark, metrics["cagr"])
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


def _populate_perf_row(row: dict, s: pd.Series, bench_history: pd.Series) -> None:
    """Populate a performance row dict with period returns + risk metrics + alpha/beta.

    All risk metrics (CAGR, Vol, Sharpe, Sortino, Max DD, Alpha, Beta) use the full series `s`
    (already capped to max 5 years). Period Used reflects the actual window covered.
    """
    # Period returns — single shared bucket→days mapping (stats.PERIOD_DAYS)
    # so every table in Tarzan measures the same windows.
    for key, days in PERIOD_DAYS.items():
        row[key] = compute_period_return(s, days)
    row["ytd"] = compute_ytd_return(s)

    # Risk metrics on full series (same shared block as the benchmark rows).
    row.update(risk_metric_row(s))
    daily_ret = s.pct_change().dropna()
    cagr_val = row["cagr"] if isinstance(row["cagr"], (int, float)) else 0.0
    # Alpha/Beta vs the reference benchmark, on the *overlapping* window so
    # the figures are apples-to-apples with `s`: a 6-month track record is
    # measured against the benchmark over those same 6 months, not the
    # benchmark's full multi-year history (which made α incoherent before).
    row["alpha"] = float("nan")
    row["beta"] = float("nan")
    if (bench_history is not None and len(bench_history) > 1
            and len(s) > 1 and not daily_ret.empty):
        bench_win = _clip_to_window(bench_history, s.index.min(), s.index.max())
        if len(bench_win) > 1:
            beta, alpha = _compute_beta_alpha(s, bench_win, cagr_val)
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
