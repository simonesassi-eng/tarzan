"""Benchmark series construction and benchmark-relative metrics.

This is the network-touching half of what used to live in ``metrics.py``:
it fetches benchmark price histories (via the enricher's memoized layer),
builds blended series (e.g. 60/40), and computes the standard metric set
for a benchmark or a performance row. The pure math it relies on lives in
``tarzan.engine.stats``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tarzan import config as cfg
from tarzan.engine.stats import (
    TRADING_DAYS,
    DAYS_PER_YEAR,
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_period_return,
    compute_sharpe,
    compute_sortino,
    compute_var,
    compute_ytd_return,
    _compute_beta_alpha,
    _scale_or_nan,
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

    idx = series.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s = series.copy()
    s.index = idx.normalize()
    lo, hi = _naive(start), _naive(end)
    return s[(s.index >= lo) & (s.index <= hi)]


def _fetch_benchmark_history(ticker: str) -> pd.Series:
    """EUR-converted close-price history for a benchmark ticker.

    Memoized per run (via the enricher's benchmark store) so the same
    benchmark fetched by _performance/_risk/_benchmarks/_holding_performance
    in one compute_all triggers a single network fetch + conversion.
    """
    from tarzan.data import enricher as _enr

    with _enr._net_lock:
        if ticker in _enr._benchmark_memo:
            return _enr._benchmark_memo[ticker]

    data = _enr._fetch_ticker_data(ticker)
    history = data.get("history", pd.DataFrame())
    if history.empty:
        series = pd.Series(dtype=float)
    else:
        prices = history["Close"]
        currency = data.get("info", {}).get("currency", "USD")
        series = _enr.convert_to_eur(prices, currency) if currency != "EUR" else prices

    with _enr._net_lock:
        _enr._benchmark_memo[ticker] = series
    return series


def _build_benchmark_series(name: str, ticker: str, initial_value: float) -> pd.Series:
    if ticker:
        return _fetch_benchmark_history(ticker)
    mix_c = cfg.mix_60_40()
    eq = _fetch_benchmark_history(mix_c["equity_ticker"])
    bd = _fetch_benchmark_history(mix_c["bond_ticker"])
    if eq.empty or bd.empty:
        return pd.Series(dtype=float)
    combined = pd.DataFrame({"eq": eq, "bd": bd}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    eq_norm = combined["eq"] / combined["eq"].iloc[0]
    bd_norm = combined["bd"] / combined["bd"].iloc[0]
    bench = eq_norm * mix_c.get("equity_weight", 0.6) + bd_norm * mix_c.get("bond_weight", 0.4)
    return bench * initial_value


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
    cagr = compute_cagr(bench)
    daily_ret = bench.pct_change().dropna()
    vol = float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100 if len(daily_ret) > 0 else 0.0
    metrics = {
        "cagr": cagr,
        "1d": compute_period_return(bench, 1), "1w": compute_period_return(bench, 7),
        "1m": compute_period_return(bench, 30), "3m": compute_period_return(bench, 90),
        "6m": compute_period_return(bench, 180), "ytd": compute_ytd_return(bench),
        "1y": compute_period_return(bench, 365), "3y": compute_period_return(bench, 1095),
        "5y": compute_period_return(bench, 1825),
        "volatility": vol, "sharpe": compute_sharpe(cagr, vol),
        "sortino": compute_sortino(daily_ret, cagr) if len(daily_ret) > 0 else float("nan"),
        "max_drawdown": compute_max_drawdown(bench) * 100,
        "var_95": _scale_or_nan(compute_var(daily_ret, 0.95), 100),
        "cvar_95": _scale_or_nan(compute_cvar(daily_ret, 0.95), 100),
        "alpha": float("nan"),
        "beta": float("nan"),
    }
    if ab_benchmark is not None and not ab_benchmark.empty and len(ab_benchmark) > 1:
        beta, alpha = _compute_beta_alpha(daily_ret, ab_benchmark, cagr)
        metrics["alpha"] = alpha
        metrics["beta"] = beta
    return metrics


def _add_mix_to_histories(key_histories: dict, initial_value: float) -> None:
    mix_cfg = cfg.mix_60_40()
    eq_ticker_name = None
    for bname, bticker in BENCHMARKS.items():
        if bticker == mix_cfg.get("equity_ticker"):
            eq_ticker_name = bname
            break
    if eq_ticker_name and eq_ticker_name in key_histories:
        try:
            bond_hist = _fetch_benchmark_history(mix_cfg["bond_ticker"])
            if not bond_hist.empty:
                eq_w = mix_cfg.get("equity_weight", 0.6)
                bd_w = mix_cfg.get("bond_weight", 0.4)
                combined = pd.DataFrame({"eq": key_histories[eq_ticker_name], "bd": bond_hist}).dropna()
                if not combined.empty:
                    eq_n = combined["eq"] / combined["eq"].iloc[0]
                    bd_n = combined["bd"] / combined["bd"].iloc[0]
                    key_histories["60/40 ACWI+Bond"] = (eq_n * eq_w + bd_n * bd_w) * initial_value
        except Exception as e:
            logger.warning("Failed to build 60/40 mix: %s", e)


def _populate_perf_row(row: dict, s: pd.Series, bench_history: pd.Series) -> None:
    """Populate a performance row dict with period returns + risk metrics + alpha/beta.

    All risk metrics (CAGR, Vol, Sharpe, Sortino, Max DD, Alpha, Beta) use the full series `s`
    (already capped to max 5 years). Period Used reflects the actual window covered.
    """
    periods = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180,
               "1y": 365, "3y": 1095, "5y": 1825}

    # Period returns
    for key, days in periods.items():
        row[key] = compute_period_return(s, days)
    row["ytd"] = compute_ytd_return(s)

    # Risk metrics on full series
    row["cagr"] = compute_cagr(s)
    daily_ret = s.pct_change().dropna()
    vol = float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100 if len(daily_ret) > 0 else 0.0
    cagr_val = row["cagr"] if isinstance(row["cagr"], (int, float)) else 0.0
    row["volatility"] = vol
    row["sharpe"] = compute_sharpe(cagr_val, vol)
    row["sortino"] = compute_sortino(daily_ret, cagr_val) if len(daily_ret) > 0 else float("nan")
    row["max_drawdown"] = compute_max_drawdown(s) * 100
    # Tail risk (historical simulation, 95% confidence)
    row["var_95"] = _scale_or_nan(compute_var(daily_ret, 0.95), 100)
    row["cvar_95"] = _scale_or_nan(compute_cvar(daily_ret, 0.95), 100)
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
            beta, alpha = _compute_beta_alpha(daily_ret, bench_win, cagr_val)
            row["beta"] = beta
            row["alpha"] = alpha

    # Alpha/Beta vs configured benchmark
    if not bench_history.empty and len(bench_history) > 1:
        beta, alpha = _compute_beta_alpha(daily_ret, bench_history, cagr_val)
        row["beta"] = beta
        row["alpha"] = alpha
    else:
        row["beta"] = float("nan")
        row["alpha"] = float("nan")

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
