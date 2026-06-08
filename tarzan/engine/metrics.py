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

# Pure return/risk math lives in stats.py; benchmark fetch/metrics in
# benchmarks.py. They are imported here and re-exported so the historical
# ``tarzan.engine.metrics`` public API (xirr, twror, compute_*, …) is
# preserved for callers, tests and scripts.
from tarzan.engine.stats import (  # noqa: F401  (re-exported)
    RISK_FREE_RATE,
    TRADING_DAYS,
    DAYS_PER_YEAR,
    TwrorResult,
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_period_return,
    compute_sharpe,
    compute_sortino,
    compute_var,
    compute_ytd_return,
    twror,
    xirr,
    xnpv,
    _compute_beta_alpha,
    _safe_pct_change,
    _scale_or_nan,
    _is_nan,
    _cap_to_years,
)
from tarzan.engine.benchmarks import (  # noqa: F401  (re-exported)
    BENCHMARKS,
    _add_mix_to_histories,
    _build_benchmark_series,
    _compute_single_benchmark_metrics,
    _fetch_benchmark_history,
    _populate_perf_row,
)

logger = logging.getLogger(__name__)


class MetricsEngine:
    """Computes all portfolio metrics. Extensible via register()."""

    def __init__(self, holdings: list[Holding], config: InvestorConfig,
                 orders: Optional[list] = None):
        self.holdings = holdings
        self.config = config
        self.orders = orders
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
        # Option Y: when an order list is supplied it becomes the single
        # source of truth for the historical value series. Swap the
        # provider so _performance/_risk read the same order-derived
        # series, and append the _returns computer for XIRR/TWROR.
        if orders:
            idx = self._computers.index(self._portfolio_history)
            self._computers[idx] = self._portfolio_history_from_orders
            self._computers.append(self._returns)

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
        cash_class = AssetClass.CASH_EQUIVALENTS.value
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
        total = float(df["current_value"].sum()) if not df.empty else 0.0

        # Split total into cash vs invested
        if df.empty:
            cash_value = 0.0
        else:
            cash_mask = df["asset_class"] == cash_class
            cash_value = float(df.loc[cash_mask, "current_value"].sum())
        invested_value = total - cash_value

        # Two weight columns: percentage of total portfolio (includes cash)
        # and percentage of invested portfolio (excludes cash). Cash rows
        # have a NaN weight_of_invested_pctg by design.
        df["weight_pct"] = (
            (df["current_value"] / total * 100) if total > 0 else 0.0
        )
        if not df.empty:
            cash_mask = df["asset_class"] == cash_class
            if invested_value > 0:
                df["weight_of_invested_pctg"] = (
                    df["current_value"] / invested_value * 100
                )
                df.loc[cash_mask, "weight_of_invested_pctg"] = float("nan")
            else:
                df["weight_of_invested_pctg"] = float("nan")
        class_totals = df.groupby("asset_class")["current_value"].transform("sum") if not df.empty else None
        if class_totals is not None:
            df["pct_of_class"] = (df["current_value"] / class_totals * 100).fillna(0.0)

        class_order = {v: i for i, v in enumerate([
            "Equities", "Fixed Income", "Cash & Cash Equivalents",
            "Gold", "Commodities", "Crypto", "Alternative",
        ])}
        if not df.empty:
            df["_sort"] = df["asset_class"].map(class_order).fillna(99)
            df = df.sort_values(
                ["_sort", "current_value"], ascending=[True, False]
            ).drop(columns=["_sort"]).reset_index(drop=True)

        ctx["holdings_df"] = df
        ctx["total_value"] = float(total)
        ctx["invested_value"] = float(invested_value)
        ctx["cash_value"] = float(cash_value)

    # ------------------------------------------------------------------
    # Portfolio history
    # ------------------------------------------------------------------
    def _portfolio_history(self, ctx: dict) -> None:
        # Holdings whose price history span is below this threshold are excluded
        # from the TOTAL PORTFOLIO time series, otherwise they would force the
        # whole portfolio history to be capped to their (short) window via the
        # dropna(how="any") step below. They still appear in their own per-row
        # metrics in the Performance tab.
        min_history_days = 365

        series_list: list = []
        # Track per-ticker metadata so we can report which holdings were excluded.
        # Tuple: (ticker, name, current_value_eur, span_days)
        meta: list[tuple[str, str, float, int]] = []
        for h in self.holdings:
            if h.price_history is None or len(h.price_history) == 0:
                continue
            ph = h.price_history
            span_days = int((ph.index[-1] - ph.index[0]).days)
            value = float(h.current_value if h.current_value is not None else h.market_value_eur)
            s = ph * h.quantity
            s.name = h.ticker
            series_list.append(s)
            meta.append((h.ticker, h.name or h.ticker, value, span_days))

        if not series_list:
            ctx["portfolio_history"] = pd.Series(dtype=float)
            ctx["portfolio_history_full"] = pd.Series(dtype=float)
            ctx["excluded_short_tenure"] = []
            return

        # Identify holdings with insufficient history. If filtering them would
        # leave us with nothing (e.g. brand-new portfolio), fall back to the
        # full set so we still produce a series.
        eligible_tickers = [
            ticker for (ticker, _, _, span) in meta if span >= min_history_days
        ]
        excluded: list[dict] = []
        kept_series = series_list
        if eligible_tickers and len(eligible_tickers) < len(series_list):
            kept_series = [s for s in series_list if s.name in eligible_tickers]
            total_value = sum(v for (_, _, v, _) in meta) or 1.0
            for (ticker, name, value, span) in meta:
                if ticker in eligible_tickers:
                    continue
                excluded.append({
                    "ticker": ticker,
                    "name": name,
                    "value_eur": value,
                    "weight_pct": value / total_value * 100.0,
                    "span_days": span,
                })
        ctx["excluded_short_tenure"] = excluded
        if excluded:
            names = ", ".join(item["name"] for item in excluded)
            logger.info(
                "TOTAL PORTFOLIO time series excludes %d holding(s) with <1Y "
                "of price history (%.1f%% of AuM): %s",
                len(excluded),
                sum(item["weight_pct"] for item in excluded),
                names,
            )

        combined = pd.concat(kept_series, axis=1).ffill()
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
        inception = self.config.portfolio_inception_date if self.config else ""
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
    # Portfolio history from orders (Option Y)
    # ------------------------------------------------------------------
    def _portfolio_history_from_orders(self, ctx: dict) -> None:
        """Build the historical value series from the order list.

        Replaces ``_portfolio_history`` when orders are supplied so that
        every downstream history-dependent computer reads the same
        order-derived series. Also stashes the raw valuation/flow series
        and provenance in ``ctx`` for the ``_returns`` computer.
        """
        from tarzan.engine.returns_builder import (
            build_holdings_from_orders,
            build_order_derived_series,
        )

        # Enriched holdings keyed by ISIN. Start from holdings.csv (which
        # carries cost basis/targets), then fill in any ISIN that appears
        # only in the order list by enriching the order-derived holdings.
        # Without this, order-only ISINs would have no yfinance history in
        # the live run and silently fall to the synthetic/carry_flat rung,
        # making TWROR coverage differ from the standalone returns script.
        enriched_by_isin = {h.isin: h for h in self.holdings if h.isin}
        missing = [
            h for h in build_holdings_from_orders(self.orders)
            if h.isin and h.isin not in enriched_by_isin
        ]
        if missing:
            from tarzan.data.enricher import enrich_holdings
            logger.info("Enriching %d order-only ISIN(s) for returns", len(missing))
            for h in enrich_holdings(missing):
                if h.isin:
                    enriched_by_isin[h.isin] = h

        series = build_order_derived_series(self.orders, enriched_by_isin)

        # Risk and period-return metrics must read the dense, daily,
        # flow-adjusted NAV index — not the sparse trade-date valuations.
        # The sparse series' pct_change would treat multi-month gaps as
        # one trading day (distorting volatility/Sharpe/VaR/beta), and a
        # raw value series would book deposits as market gains. The daily
        # series strips both problems, mirroring the holdings path (which
        # values a fixed basket of today's quantities over history).
        ph = series.daily_series
        if ph is None or ph.empty:
            # Fallback: the sparse valuations, so a portfolio with too few
            # observations still yields a (degraded) series rather than none.
            if series.valuations:
                idx = pd.to_datetime([d for d, _ in series.valuations])
                vals = [v for _, v in series.valuations]
                ph = pd.Series(vals, index=idx).sort_index()
                ph = ph[~ph.index.duplicated(keep="last")]
            else:
                ph = pd.Series(dtype=float)

        ctx["portfolio_history"] = ph
        ctx["portfolio_history_full"] = ph
        ctx["excluded_short_tenure"] = [
            {"ticker": isin, "name": isin, "value_eur": 0.0,
             "weight_pct": 0.0, "span_days": 0}
            for isin in series.provenance.get("excluded", [])
        ]
        # Stash for _returns.
        ctx["_order_series"] = series

    # ------------------------------------------------------------------
    # Returns: XIRR + TWROR (only registered when orders are present)
    # ------------------------------------------------------------------
    def _returns(self, ctx: dict) -> None:
        series = ctx.get("_order_series")
        if series is None:
            return
        rate = xirr(series.xirr_cashflows)
        ctx["xirr_pct"] = rate * 100.0 if not _is_nan(rate) else None

        res = twror(
            series.valuations, series.external_flows, series.span_days,
            coverage_pct=series.coverage_pct,
        )
        ctx["twror_pct"] = res.cumulative_pct
        ctx["twror_annualized_pct"] = res.annualized_pct
        ctx["returns_coverage_pct"] = res.coverage_pct
        ctx["returns_provenance"] = series.provenance
        ctx["returns_period_debug"] = res.periods

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    def _performance(self, ctx: dict) -> None:
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        if ph.empty:
            ctx["performance"] = {"cagr": 0.0, "ytd": None, "1d": None, "1w": None,
                                  "1m": None, "3m": None, "6m": None, "1y": None,
                                  "3y": None, "5y": None}
        else:
            periods = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095, "5y": 1825}
            result = {"cagr": compute_cagr(ph), "ytd": compute_ytd_return(ph)}
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
        invested_value = ctx.get("invested_value", 0.0)
        cash_class = AssetClass.CASH_EQUIVALENTS.value

        # Invested allocation: exclude cash, percentages relative to
        # invested_value (not total_value).
        invested_df = df[df["asset_class"] != cash_class] if not df.empty else df
        if not invested_df.empty and invested_value > 0:
            by_class = (
                invested_df.groupby("asset_class")["current_value"].sum().reset_index()
            )
            by_class["weight_pct"] = by_class["current_value"] / invested_value * 100
            by_class = by_class[["asset_class", "weight_pct"]]
            by_class.columns = ["category", "weight_pct"]
        else:
            by_class = pd.DataFrame(columns=["category", "weight_pct"])

        by_geo = _compute_geo_allocation(df, self.holdings)
        by_sector = pd.DataFrame(columns=["category", "weight_pct"])
        if not df.empty and "sector" in df.columns:
            by_sector = df.groupby("sector")["weight_pct"].sum().reset_index()
            by_sector.columns = ["category", "weight_pct"]
        top_10 = (
            df.nlargest(10, "weight_pct")[
                ["ticker", "name", "isin", "current_value", "weight_pct", "gain_pct"]
            ].copy()
            if not df.empty
            else pd.DataFrame()
        )
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

        # Invested asset-class rows: % of invested portfolio. Cash is
        # never in invested_allocation_targets_pctg, so it is correctly
        # skipped here.
        actual_class = dict(zip(by_class["category"], by_class["weight_pct"]))
        for cat in sorted(set(self.config.invested_allocation_targets_pctg) | set(actual_class)):
            actual = actual_class.get(cat, 0.0)
            target = self.config.invested_allocation_targets_pctg.get(cat, 0.0)
            rows.append({
                "category": cat, "type": "asset_class",
                "actual_pct": actual, "target_pct": target,
                "delta_pct": actual - target,
                "actual_eur": None, "target_eur": None, "delta_eur": None,
            })

        # Equity geography rows: % of equity portion.
        actual_geo = dict(zip(by_geo["category"], by_geo["weight_pct"]))
        for cat in sorted(set(self.config.equity_geo_targets_pctg) | set(actual_geo)):
            actual = actual_geo.get(cat, 0.0)
            target = self.config.equity_geo_targets_pctg.get(cat, 0.0)
            rows.append({
                "category": cat, "type": "geography (equity only)",
                "actual_pct": actual, "target_pct": target,
                "delta_pct": actual - target,
                "actual_eur": None, "target_eur": None, "delta_eur": None,
            })

        # Cash buffer row (only when a target is configured): absolute EUR,
        # no percentages. Pctg fields carry the relative deviation vs the
        # target buffer so the traffic-light helper can reuse
        # rebalancing_target_tolerance_pctg.
        cash_value = ctx.get("cash_value", 0.0)
        cash_target = float(self.config.target_cash_buffer_eur or 0.0)
        if cash_target > 0:
            delta_eur = cash_value - cash_target
            delta_pct = delta_eur / cash_target * 100.0
            rows.append({
                "category": "Cash & Cash Equivalents", "type": "cash",
                "actual_pct": None, "target_pct": None,
                "delta_pct": delta_pct,
                "actual_eur": cash_value,
                "target_eur": cash_target,
                "delta_eur": delta_eur,
            })

        ctx["goal_deltas"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------
    def _rebalancing(self, ctx: dict) -> None:
        if self.config is None:
            ctx["rebalancing_suggestions"] = None
            ctx["rebalancing_verifications"] = None
            ctx["rebalancing_sensitivity"] = None
            return
        from tarzan.engine.rebalancer import (
            compute_unified_rebalancing,
            compute_drift_penalty_sensitivity,
        )
        lump = self.config.rebalancing_lump_sum_amount_eur if self.config.rebalancing_lump_sum_amount_eur > 0 else None
        suggestions, verifications = compute_unified_rebalancing(
            self.holdings, self.config, ctx["total_value"], lump_sum=lump)
        ctx["rebalancing_suggestions"] = suggestions
        ctx["rebalancing_verifications"] = verifications

        # Drift-penalty sensitivity sweep — surfaces the optimization
        # turning points so the user can pick the weight that matches
        # their preferences (more trades / less drift vs. fewer
        # trades / more leftover drift). Cheap enough to always run.
        try:
            ctx["rebalancing_sensitivity"] = compute_drift_penalty_sensitivity(
                self.holdings, self.config, ctx["total_value"], lump_sum=lump,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Drift-penalty sensitivity sweep failed: %s", exc)
            ctx["rebalancing_sensitivity"] = None

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------
    def _benchmarks(self, ctx: dict) -> None:
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        if ph.empty:
            ctx["benchmark_comparison"] = pd.DataFrame()
            ctx["benchmark_histories"] = {}
            return
        initial_value = float(ph.iloc[0])
        comp_rows = []
        key_histories: dict[str, pd.Series] = {}
        chart_benchmarks = set(cfg.chart_benchmarks())
        # Fetch the α/β reference series once: every other benchmark row
        # gets α/β computed against this same series, so the columns are
        # comparable. The α/β benchmark vs itself yields β=1, α=0.
        ab_benchmark = pd.Series(dtype=float)
        try:
            ab_benchmark = _fetch_benchmark_history(cfg.benchmark_beta())
        except Exception as e:
            logger.warning("α/β benchmark fetch failed: %s", e)
        for name, ticker in BENCHMARKS.items():
            try:
                bench = _build_benchmark_series(name, ticker, initial_value)
                if bench.empty or len(bench) < 2:
                    continue
                if name in chart_benchmarks:
                    key_histories[name] = bench
                metrics = _compute_single_benchmark_metrics(bench, ab_benchmark)
                metrics["benchmark"] = name
                comp_rows.append(metrics)
            except Exception as e:
                logger.warning("Benchmark %s failed: %s", name, e)
        _add_mix_to_histories(key_histories, initial_value)
        ctx["benchmark_comparison"] = pd.DataFrame(comp_rows)
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
        cash_target = float(self.config.target_cash_buffer_eur) if self.config else 0.0
        return PortfolioMetrics(
            total_value=ctx.get("total_value", 0.0),
            invested_value=ctx.get("invested_value", 0.0),
            cash_value=ctx.get("cash_value", 0.0),
            cash_target_eur=cash_target,
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
            rebalancing_sensitivity=ctx.get("rebalancing_sensitivity"),
            benchmark_comparison=ctx.get("benchmark_comparison", pd.DataFrame()),
            portfolio_history=ctx.get("portfolio_history"),
            benchmark_histories=ctx.get("benchmark_histories", {}),
            holding_performance=ctx.get("holding_performance", pd.DataFrame()),
            holding_histories=ctx.get("holding_histories", {}),
            acwi_geo=ctx.get("acwi_geo", {}),
            excluded_short_tenure=ctx.get("excluded_short_tenure", []),
            xirr_pct=ctx.get("xirr_pct"),
            twror_pct=ctx.get("twror_pct"),
            twror_annualized_pct=ctx.get("twror_annualized_pct"),
            returns_coverage_pct=ctx.get("returns_coverage_pct"),
            returns_provenance=ctx.get("returns_provenance"),
            returns_period_debug=ctx.get("returns_period_debug"),
        )


# ======================================================================
# Pure helper functions (no state, no I/O)
# ======================================================================

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
