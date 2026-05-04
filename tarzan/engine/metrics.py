"""MetricsEngine: single class computing all portfolio metrics.

Extensible via register() — append a callable and it runs in the pipeline.
Each computer receives a context dict and populates it with results.
The final context is used to build a PortfolioMetrics DTO.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan import config as cfg

logger = logging.getLogger(__name__)

RISK_FREE_RATE = cfg.risk_free_rate() * 100  # e.g. 4.0 = 4%
TRADING_DAYS = cfg.trading_days()
BENCHMARKS = cfg.benchmarks()


class MetricsEngine:
    """Computes all portfolio metrics. Extensible via register()."""

    def __init__(self, holdings: list[Holding], config: InvestorConfig):
        self.holdings = holdings
        self.config = config
        self._computers: list[Callable] = [
            self._valuation,
            self._portfolio_history,
            self._performance,
            self._risk,
            self._allocations,
            self._income_costs,
            self._goals,
            self._rebalancing,
            self._benchmarks,
            self._holding_performance,
            self._geo_benchmark,
            self._holding_histories,
        ]

    def register(self, fn: Callable) -> None:
        """Append a custom metric computer to the pipeline."""
        self._computers.append(fn)

    def compute_all(self) -> PortfolioMetrics:
        """Run all computers and return a populated PortfolioMetrics."""
        ctx: dict = {}
        for computer in self._computers:
            try:
                computer(ctx)
            except Exception as e:
                name = getattr(computer, "__name__", str(computer))
                logger.error("Metric computer '%s' failed: %s", name, e)
        return self._build_result(ctx)

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def _valuation(self, ctx: dict) -> None:
        rows = []
        for h in self.holdings:
            value = h.current_value if h.current_value is not None else h.market_value_eur
            cost = h.cost_basis_eur
            gain = _safe_pct_change(cost, value)
            avg_price = cost / h.quantity if h.quantity > 0 else 0.0
            geo_str = _format_geo_breakdown(h)
            rows.append({
                "isin": h.isin, "ticker": h.ticker,
                "name": h.name or h.ticker,
                "instrument_type": h.instrument_type or "Unknown",
                "security_type": h.security_type or h.instrument_type or "Unknown",
                "quantity": h.quantity, "avg_purchase_price": avg_price,
                "current_price": h.current_price or (value / h.quantity if h.quantity > 0 else 0),
                "current_value": value, "cost_basis_eur": cost,
                "gain_pct": gain, "gain_eur": value - cost,
                "asset_class": h.asset_class.value if h.asset_class else AssetClass.ALTERNATIVE.value,
                "geography": geo_str, "currency": h.currency,
                "ter": h.ter, "yield_pct": h.yield_pct,
                "data_source": h.data_source or "",
                "geo_source": h.geo_source or "",
                "fetch_timestamp": h.fetch_timestamp.strftime("%Y-%m-%d %H:%M") if h.fetch_timestamp else "",
            })
        df = pd.DataFrame(rows)
        total = df["current_value"].sum()
        df["weight_pct"] = (df["current_value"] / total * 100) if total > 0 else 0.0
        class_totals = df.groupby("asset_class")["current_value"].transform("sum")
        df["pct_of_class"] = (df["current_value"] / class_totals * 100).fillna(0.0)
        class_order = {v: i for i, v in enumerate([
            "Equities", "Fixed Income", "Cash & Cash Equivalents",
            "Commodities", "Real Estate", "Alternative",
        ])}
        df["_sort"] = df["asset_class"].map(class_order).fillna(99)
        df = df.sort_values(["_sort", "current_value"], ascending=[True, False]).drop(columns=["_sort"]).reset_index(drop=True)
        ctx["holdings_df"] = df
        ctx["total_value"] = float(total)

    # ------------------------------------------------------------------
    # Portfolio history
    # ------------------------------------------------------------------
    def _portfolio_history(self, ctx: dict) -> None:
        series_list = []
        for h in self.holdings:
            if h.price_history is not None and len(h.price_history) > 0:
                s = h.price_history * h.quantity
                s.name = h.ticker
                series_list.append(s)
        if not series_list:
            ctx["portfolio_history"] = pd.Series(dtype=float)
            ctx["portfolio_history_full"] = pd.Series(dtype=float)
            return
        combined = pd.concat(series_list, axis=1).ffill()
        # Normalize the index to naive calendar days so that series coming from
        # different exchanges (with different timezones) align cleanly, and
        # drop any duplicate days created by timezone offsets.
        combined.index = (
            combined.index.tz_convert("UTC").tz_localize(None).normalize()
            if combined.index.tz is not None
            else combined.index.normalize()
        )
        combined = combined[~combined.index.duplicated(keep="last")]
        # Drop dates where any holding is missing (pre-existence periods)
        combined = combined.dropna(how="any")
        if combined.empty:
            ctx["portfolio_history"] = pd.Series(dtype=float)
            ctx["portfolio_history_full"] = pd.Series(dtype=float)
            return
        ph_full = combined.sum(axis=1)
        # Cap to 5 years max
        ph_full = _cap_to_years(ph_full, 5)
        ctx["portfolio_history_full"] = ph_full

        # Filter by inception date if configured
        ph = ph_full.copy()
        inception = self.config.portfolio_inception if self.config else ""
        if inception:
            try:
                inception_dt = pd.to_datetime(inception)
                if inception_dt.tzinfo is None and ph.index.tz is not None:
                    inception_dt = inception_dt.tz_localize(ph.index.tz)
                ph = ph[ph.index >= inception_dt]
            except Exception as e:
                logger.warning("Failed to parse inception date '%s': %s", inception, e)

        ctx["portfolio_history"] = ph

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    def _performance(self, ctx: dict) -> None:
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        if ph.empty:
            ctx["performance"] = {"cagr": 0.0, "ytd": None, "1d": None, "1w": None,
                                  "1m": None, "3m": None, "6m": None, "1y": None,
                                  "3y": None, "5y": None, "irr": None}
        else:
            periods = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095, "5y": 1825}
            result = {"cagr": compute_cagr(ph), "ytd": compute_ytd_return(ph), "irr": None}
            for key, days in periods.items():
                result[key] = compute_period_return(ph, days)
            ctx["performance"] = result

        # Also compute full (non-inception) performance for Performance tab
        ph_full = ctx.get("portfolio_history_full", pd.Series(dtype=float))
        if ph_full.empty:
            ctx["performance_full"] = {}
        else:
            bench = pd.Series(dtype=float)
            try:
                bench = _fetch_benchmark_history(cfg.benchmark_beta())
            except Exception:
                pass
            full_row = {"ticker": "PORTFOLIO", "name": "** TOTAL PORTFOLIO **", "type": "Portfolio"}
            _populate_perf_row(full_row, ph_full, bench)
            ctx["performance_full"] = full_row

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------
    def _risk(self, ctx: dict) -> None:
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        result = {"volatility": 0.0, "sharpe": float("nan"), "max_drawdown": 0.0,
                  "sortino": float("nan"), "var_95": float("nan"), "cvar_95": float("nan"),
                  "beta": float("nan"), "alpha": float("nan")}
        if ph.empty or len(ph) < 2:
            ctx["risk"] = result
            return
        daily_returns = ph.pct_change().dropna()
        if daily_returns.empty:
            ctx["risk"] = result
            return
        annual_vol = float(daily_returns.std()) * np.sqrt(TRADING_DAYS)
        annual_return = compute_cagr(ph)
        result["volatility"] = annual_vol * 100
        result["sharpe"] = compute_sharpe(annual_return, annual_vol * 100)
        result["max_drawdown"] = compute_max_drawdown(ph) * 100
        result["sortino"] = compute_sortino(daily_returns, annual_return)
        result["var_95"] = _scale_or_nan(compute_var(daily_returns, 0.95), 100)
        result["cvar_95"] = _scale_or_nan(compute_cvar(daily_returns, 0.95), 100)
        # Beta/Alpha
        bench_history = pd.Series(dtype=float)
        try:
            bench_history = _fetch_benchmark_history(cfg.benchmark_beta())
        except Exception as e:
            logger.warning("Benchmark fetch for risk failed: %s", e)
        if not bench_history.empty and len(bench_history) > 1:
            beta, alpha = _compute_beta_alpha(daily_returns, bench_history, annual_return)
            result["beta"] = beta
            result["alpha"] = alpha
        ctx["risk"] = result

    # ------------------------------------------------------------------
    # Allocations
    # ------------------------------------------------------------------
    def _allocations(self, ctx: dict) -> None:
        df = ctx["holdings_df"]
        by_class = df.groupby("asset_class")["weight_pct"].sum().reset_index()
        by_class.columns = ["category", "weight_pct"]
        by_geo = _compute_geo_allocation(df, self.holdings)
        by_sector = pd.DataFrame(columns=["category", "weight_pct"])
        if "sector" in df.columns:
            by_sector = df.groupby("sector")["weight_pct"].sum().reset_index()
            by_sector.columns = ["category", "weight_pct"]
        top_10 = df.nlargest(10, "weight_pct")[
            ["ticker", "name", "isin", "current_value", "weight_pct", "gain_pct"]
        ].copy()
        ctx["allocation_by_class"] = by_class
        ctx["allocation_by_geo"] = by_geo
        ctx["allocation_by_sector"] = by_sector
        ctx["top_10"] = top_10

    # ------------------------------------------------------------------
    # Income & costs
    # ------------------------------------------------------------------
    def _income_costs(self, ctx: dict) -> None:
        df = ctx["holdings_df"]
        total_weight = df["weight_pct"].sum()
        if total_weight <= 0:
            ctx["weighted_yield"] = 0.0
            ctx["avg_ter"] = 0.0
            return
        ctx["weighted_yield"] = float((df["yield_pct"].fillna(0.0) * df["weight_pct"]).sum() / total_weight) * 100
        ctx["avg_ter"] = float((df["ter"].fillna(0.0) * df["weight_pct"]).sum() / total_weight) * 100

    # ------------------------------------------------------------------
    # Goal deltas
    # ------------------------------------------------------------------
    def _goals(self, ctx: dict) -> None:
        if self.config is None:
            ctx["goal_deltas"] = None
            return
        by_class = ctx["allocation_by_class"]
        by_geo = ctx["allocation_by_geo"]
        rows = []
        actual_class = dict(zip(by_class["category"], by_class["weight_pct"]))
        for cat in sorted(set(self.config.allocation_targets) | set(actual_class)):
            actual = actual_class.get(cat, 0.0)
            target = self.config.allocation_targets.get(cat, 0.0)
            rows.append({"category": cat, "type": "asset_class",
                         "actual_pct": actual, "target_pct": target, "delta_pct": actual - target})
        actual_geo = dict(zip(by_geo["category"], by_geo["weight_pct"]))
        for cat in sorted(set(self.config.geo_allocation) | set(actual_geo)):
            actual = actual_geo.get(cat, 0.0)
            target = self.config.geo_allocation.get(cat, 0.0)
            rows.append({"category": cat, "type": "geography (equity only)",
                         "actual_pct": actual, "target_pct": target, "delta_pct": actual - target})
        ctx["goal_deltas"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------
    def _rebalancing(self, ctx: dict) -> None:
        if self.config is None:
            ctx["rebalancing_suggestions"] = None
            ctx["rebalancing_verifications"] = None
            return
        from tarzan.engine.rebalancer import compute_unified_rebalancing
        lump = self.config.rebalancing_lump_sum_amount if self.config.rebalancing_lump_sum_amount > 0 else None
        suggestions, verifications = compute_unified_rebalancing(
            self.holdings, self.config, ctx["total_value"], lump_sum=lump)
        ctx["rebalancing_suggestions"] = suggestions
        ctx["rebalancing_verifications"] = verifications

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------
    def _benchmarks(self, ctx: dict) -> None:
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        if ph.empty:
            ctx["benchmark_comparison"] = pd.DataFrame()
            ctx["what_if"] = pd.DataFrame()
            ctx["benchmark_histories"] = {}
            return
        initial_value = float(ph.iloc[0])
        comp_rows = []
        key_histories: dict[str, pd.Series] = {}
        chart_benchmarks = set(cfg.chart_benchmarks())
        for name, ticker in BENCHMARKS.items():
            try:
                bench = _build_benchmark_series(name, ticker, initial_value)
                if bench.empty or len(bench) < 2:
                    continue
                if name in chart_benchmarks:
                    key_histories[name] = bench
                metrics = _compute_single_benchmark_metrics(bench)
                metrics["benchmark"] = name
                comp_rows.append(metrics)
            except Exception as e:
                logger.warning("Benchmark %s failed: %s", name, e)
        _add_mix_to_histories(key_histories, initial_value)
        ctx["benchmark_comparison"] = pd.DataFrame(comp_rows)
        ctx["what_if"] = pd.DataFrame()
        ctx["benchmark_histories"] = key_histories

    # ------------------------------------------------------------------
    # Per-holding performance
    # ------------------------------------------------------------------
    def _holding_performance(self, ctx: dict) -> None:
        rows = []
        periods = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180,
                   "1y": 365, "3y": 1095, "5y": 1825}

        # Fetch Alpha/Beta benchmark once for all holdings
        bench_history = pd.Series(dtype=float)
        try:
            bench_history = _fetch_benchmark_history(cfg.benchmark_beta())
        except Exception as e:
            logger.warning("Alpha/Beta benchmark fetch failed: %s", e)

        for h in self.holdings:
            if h.price_history is None or len(h.price_history) < 2:
                continue
            # Cap all metrics at max 5 years
            s = _cap_to_years(h.price_history, 5)
            row: dict = {"ticker": h.ticker, "name": h.name or h.ticker, "type": "In portfolio"}
            _populate_perf_row(row, s, bench_history)
            rows.append(row)

        # Add benchmark rows (from indexes.csv is_benchmark=true)
        for bench_name, bench_ticker in BENCHMARKS.items():
            try:
                bs = _fetch_benchmark_history(bench_ticker)
                if bs.empty or len(bs) < 2:
                    continue
                bs = _cap_to_years(bs, 5)
                row = {"ticker": bench_ticker, "name": bench_name, "type": "Benchmark index"}
                _populate_perf_row(row, bs, bench_history)
                rows.append(row)
            except Exception as e:
                logger.warning("Benchmark %s performance computation failed: %s", bench_name, e)

        ctx["holding_performance"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Geo benchmark (MSCI ACWI reference)
    # ------------------------------------------------------------------
    def _geo_benchmark(self, ctx: dict) -> None:
        from tarzan.data.geo_resolver import lookup_geo_by_index_name
        geo_bench_name = cfg.benchmark_geo_allocation()
        breakdown = lookup_geo_by_index_name(geo_bench_name)
        if breakdown:
            ctx["acwi_geo"] = {
                g.value if hasattr(g, "value") else str(g): v
                for g, v in breakdown.items()
            }
        else:
            ctx["acwi_geo"] = {}

    # ------------------------------------------------------------------
    # Holding histories for charting
    # ------------------------------------------------------------------
    def _holding_histories(self, ctx: dict) -> None:
        hh = {}
        for h in self.holdings:
            if h.price_history is not None and len(h.price_history) > 1:
                hh[h.ticker] = {"name": h.name or h.ticker, "history": h.price_history}
        ctx["holding_histories"] = hh

    # ------------------------------------------------------------------
    # Build final PortfolioMetrics
    # ------------------------------------------------------------------
    def _build_result(self, ctx: dict) -> PortfolioMetrics:
        return PortfolioMetrics(
            total_value=ctx.get("total_value", 0.0),
            holdings_df=ctx.get("holdings_df", pd.DataFrame()),
            allocation_by_class=ctx.get("allocation_by_class", pd.DataFrame()),
            allocation_by_geo=ctx.get("allocation_by_geo", pd.DataFrame()),
            allocation_by_sector=ctx.get("allocation_by_sector", pd.DataFrame()),
            top_10=ctx.get("top_10", pd.DataFrame()),
            performance=ctx.get("performance", {}),
            performance_full=ctx.get("performance_full", {}),
            risk=ctx.get("risk", {}),
            weighted_yield=ctx.get("weighted_yield", 0.0),
            avg_ter=ctx.get("avg_ter", 0.0),
            goal_deltas=ctx.get("goal_deltas"),
            rebalancing_suggestions=ctx.get("rebalancing_suggestions"),
            rebalancing_verifications=ctx.get("rebalancing_verifications"),
            benchmark_comparison=ctx.get("benchmark_comparison", pd.DataFrame()),
            what_if=ctx.get("what_if", pd.DataFrame()),
            portfolio_history=ctx.get("portfolio_history"),
            benchmark_histories=ctx.get("benchmark_histories", {}),
            holding_performance=ctx.get("holding_performance", pd.DataFrame()),
            holding_histories=ctx.get("holding_histories", {}),
            acwi_geo=ctx.get("acwi_geo", {}),
        )


# ======================================================================
# Pure helper functions (no state, no I/O)
# ======================================================================

def _safe_pct_change(old: float, new: float) -> float:
    if old <= 0 or new <= 0:
        return 0.0
    return (new - old) / old * 100


def _format_geo_breakdown(h: Holding) -> str:
    if not h.geo_breakdown:
        return "Not Available"
    if len(h.geo_breakdown) == 1:
        g = next(iter(h.geo_breakdown))
        return g.value if hasattr(g, "value") else str(g)
    return ", ".join(
        f"{(g.value if hasattr(g, 'value') else str(g))}: {int(p)}"
        for g, p in sorted(h.geo_breakdown.items(), key=lambda x: -x[1])
    )


def compute_cagr(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0:
        return 0.0
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return 0.0
    return ((end / start) ** (1 / (days / 365.25)) - 1) * 100


def compute_period_return(series: pd.Series, days: int) -> Optional[float]:
    if series.empty or len(series) < 2:
        return None
    if days <= 1:
        start = float(series.iloc[-2])
        return (((float(series.iloc[-1]) / start) - 1) * 100) if start > 0 else None
    cutoff = series.index[-1] - pd.Timedelta(days=days)
    subset = series[series.index >= cutoff]
    if subset.empty or len(subset) < 2:
        return None
    start = float(subset.iloc[0])
    return (((float(subset.iloc[-1]) / start) - 1) * 100) if start > 0 else None


def compute_ytd_return(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    ytd = series[series.index.year == series.index[-1].year]
    if ytd.empty or len(ytd) < 2:
        return None
    start = float(ytd.iloc[0])
    return (((float(ytd.iloc[-1]) / start) - 1) * 100) if start > 0 else None


def compute_sharpe(annual_return: float, annual_volatility: float) -> float:
    if annual_volatility <= 0:
        return float("nan")
    return (annual_return - RISK_FREE_RATE) / annual_volatility


def compute_sortino(daily_returns: pd.Series, annual_return: float) -> float:
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return float("nan")
    downside_std = float(downside.std()) * np.sqrt(TRADING_DAYS) * 100
    if downside_std <= 0:
        return float("nan")
    return (annual_return - RISK_FREE_RATE) / downside_std


def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return 0.0
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min())


def compute_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    return float(daily_returns.quantile(1 - confidence))


def compute_cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty or len(daily_returns) < 5:
        return float("nan")
    var = compute_var(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if not tail.empty else var


def _scale_or_nan(val: float, factor: float) -> float:
    if val != val:
        return val
    return val * factor


def _normalize_index(series: pd.Series) -> pd.Series:
    s = series.copy()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None).normalize()
    else:
        s.index = s.index.normalize()
    return s


def _compute_beta_alpha(
    daily_returns: pd.Series, benchmark_history: pd.Series, annual_return: float,
) -> tuple[float, float]:
    bench_returns = benchmark_history.pct_change().dropna()
    dr = _normalize_index(daily_returns)
    br = _normalize_index(bench_returns)
    aligned = pd.DataFrame({"port": dr, "bench": br}).dropna()
    if len(aligned) <= 1:
        return float("nan"), float("nan")
    cov = aligned["port"].cov(aligned["bench"])
    var_bench = aligned["bench"].var()
    if var_bench <= 0:
        return float("nan"), float("nan")
    beta = cov / var_bench
    bench_annual = compute_cagr(benchmark_history)
    alpha = annual_return - (RISK_FREE_RATE + beta * (bench_annual - RISK_FREE_RATE))
    return float(beta), float(alpha)


# ======================================================================
# Allocation helpers
# ======================================================================

def _compute_geo_allocation(df: pd.DataFrame, holdings: Optional[list[Holding]] = None) -> pd.DataFrame:
    equity_df = df[df["asset_class"] == "Equities"]
    if equity_df.empty:
        return pd.DataFrame(columns=["category", "weight_pct"])
    equity_total = equity_df["weight_pct"].sum()
    geo_lookup: dict[str, dict] = {}
    if holdings:
        for h in holdings:
            if h.geo_breakdown and h.ticker:
                geo_lookup[h.ticker] = h.geo_breakdown
    geo_weights: dict[str, float] = {}
    for _, row in equity_df.iterrows():
        ticker = row.get("ticker", "")
        weight = row["weight_pct"]
        breakdown = geo_lookup.get(ticker)
        if breakdown:
            total_bd = sum(breakdown.values())
            if total_bd > 0:
                for geo, pct in breakdown.items():
                    geo_name = geo.value if hasattr(geo, "value") else str(geo)
                    geo_weights[geo_name] = geo_weights.get(geo_name, 0) + weight * (pct / total_bd)
        else:
            geo_name = row.get("geography", "USA")
            geo_weights[geo_name] = geo_weights.get(geo_name, 0) + weight
    by_geo = pd.DataFrame([{"category": k, "weight_pct": v} for k, v in geo_weights.items()])
    if equity_total > 0 and not by_geo.empty:
        by_geo["weight_pct"] = by_geo["weight_pct"] / equity_total * 100
    return by_geo


# ======================================================================
# Benchmark helpers
# ======================================================================

def _fetch_benchmark_history(ticker: str) -> pd.Series:
    from tarzan.data.enricher import _fetch_ticker_data, convert_to_eur
    data = _fetch_ticker_data(ticker)
    history = data.get("history", pd.DataFrame())
    if history.empty:
        return pd.Series(dtype=float)
    prices = history["Close"]
    currency = data.get("info", {}).get("currency", "USD")
    if currency != "EUR":
        prices = convert_to_eur(prices, currency)
    return prices


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


def _compute_single_benchmark_metrics(bench: pd.Series) -> dict:
    cagr = compute_cagr(bench)
    daily_ret = bench.pct_change().dropna()
    vol = float(daily_ret.std()) * np.sqrt(TRADING_DAYS) * 100 if len(daily_ret) > 0 else 0.0
    return {
        "cagr": cagr,
        "1d": compute_period_return(bench, 1), "1w": compute_period_return(bench, 7),
        "1m": compute_period_return(bench, 30), "3m": compute_period_return(bench, 90),
        "6m": compute_period_return(bench, 180), "ytd": compute_ytd_return(bench),
        "1y": compute_period_return(bench, 365), "3y": compute_period_return(bench, 1095),
        "5y": compute_period_return(bench, 1825),
        "volatility": vol, "sharpe": compute_sharpe(cagr, vol),
        "max_drawdown": compute_max_drawdown(bench) * 100,
    }


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



def _cap_to_years(series: pd.Series, years: float) -> pd.Series:
    """Cap a price series to the last N years."""
    if series is None or series.empty:
        return series
    cutoff = series.index[-1] - pd.Timedelta(days=int(years * 365.25))
    return series[series.index >= cutoff]


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
        years_covered = days_covered / 365.25
        if years_covered >= 4.9:
            row["period_used"] = "5Y"
        elif years_covered >= 1.0:
            row["period_used"] = f"{years_covered:.1f}Y"
        else:
            months = int(years_covered * 12)
            row["period_used"] = f"{months}M"
    else:
        row["period_used"] = "—"
