"""MetricsEngine: single class computing all portfolio metrics.

Each computer in the pipeline receives a context dict and populates it with
results. The final context is used to build a PortfolioMetrics DTO.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd

from tarzan.models.holding import AssetClass, Holding
from tarzan.models.instrument_key import normalize_ticker
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics
from tarzan import config as cfg

# Pure return/risk math lives in stats.py; benchmark fetch/metrics in
# benchmarks.py. They are imported here and re-exported so the historical
# ``tarzan.engine.metrics`` public API (xirr, twror, compute_*, …) is
# preserved for callers, tests and scripts.
from tarzan.engine.stats import (  # noqa: F401  (re-exported)
    RISK_FREE_RATE,
    DAYS_PER_YEAR,
    TwrorResult,
    PERIOD_WINDOWS,
    compute_cagr,
    compute_cvar,
    compute_max_drawdown,
    compute_period_return,
    compute_sharpe,
    compute_sortino,
    compute_ulcer_index,
    compute_var,
    compute_ytd_return,
    normalize_index,
    rf_annual_pct,
    risk_metric_row,
    twror,
    xirr,
    xnpv,
    _compute_beta_alpha,
    _safe_pct_change,
    _is_nan,
    _cap_to_years,
)
from tarzan.engine.benchmarks import (  # noqa: F401  (re-exported)
    BENCHMARKS,
    ResolvedBenchmark,
    _add_mix_to_histories,
    _clip_to_window,
    _compute_single_benchmark_metrics,
    _fetch_benchmark_history,
    _populate_perf_row,
    preprocess_benchmarks as _preprocess_benchmark_catalog,
)

logger = logging.getLogger(__name__)


class MetricsEngine:
    """Computes all portfolio metrics."""

    def __init__(self, holdings: list[Holding], config: InvestorConfig,
                 orders: Optional[list] = None,
                 rebalance_seeds: Optional[list] = None,
                 *,
                 planning_eligible: bool = True,
                 valuation_assessment=None):
        self.holdings = holdings
        self.config = config
        self.orders = orders
        self.planning_eligible = planning_eligible
        self.valuation_assessment = valuation_assessment
        # Not-held target instruments (quantity 0) used ONLY by the rebalancer
        # so it can open new positions. They are deliberately excluded from
        # the portfolio snapshot (holdings, returns, allocations).
        self.rebalance_seeds = rebalance_seeds or []
        self._computers: list[Callable] = [
            # Resolve every venue-neutral benchmark identity once, before any
            # metric, table or chart can consume it.  All later stages receive
            # only the resulting full provider ticker and its matching series.
            self._preprocess_benchmarks,
            self._valuation,
            # One current point per price series, before anything reads one.
            self._current_prices,
            self._portfolio_history,
            self._performance,
            self._risk,
            self._allocations,
            self._income_costs,
            self._goals,
            self._rebalancing,
            self._benchmarks,
            self._holding_performance,
            self._live_1d,
            self._session_coverage,
            self._geo_benchmark,
            self._holding_histories,
            self._target_history,
            self._historical_risk,
        ]
        # Option Y: when an order list is supplied it becomes the single
        # source of truth for the historical value series. Swap the
        # provider so _performance/_risk read the same order-derived
        # series, and append the _returns computer for XIRR/TWROR.
        if orders:
            idx = self._computers.index(self._portfolio_history)
            self._computers[idx] = self._portfolio_history_from_orders
            self._computers.append(self._returns)
            self._computers.append(self._allocation_timeline)

    def _current_valuation_holdings(self) -> list[Holding]:
        """Project only policy-accepted holdings at their selected EUR value.

        The valuation assessment is the authority for every current snapshot,
        allocation, and planning consumer. Rejected evidence is omitted rather
        than silently restored from ``market_value_eur``; accepted fallback or
        stale evidence uses the exact value selected by policy. Historical
        computations retain the original holdings and their provenance.
        """
        assessment = self.valuation_assessment
        if assessment is None:
            return list(self.holdings)

        from dataclasses import replace

        from tarzan.models.instrument_key import instrument_key

        selected_by_key = {
            evidence.instrument_key: float(evidence.value_eur)
            for evidence in assessment.evidence
            if evidence.accepted_by_policy and evidence.value_eur is not None
        }
        projected: list[Holding] = []
        for holding in self.holdings:
            key = instrument_key(holding.isin, holding.ticker) or "UNKNOWN"
            if key not in selected_by_key:
                continue
            selected = selected_by_key[key]
            quantity = float(holding.quantity or 0.0)
            selected_price = selected / quantity if quantity != 0.0 else 0.0
            projected.append(replace(
                holding,
                market_value_eur=selected,
                current_value=selected,
                current_price=selected_price,
            ))
        return projected

    def compute_all(self) -> PortfolioMetrics:
        """Run all computers and return a populated PortfolioMetrics.

        Each computer is isolated so one failure cannot abort the rest, but
        the names of any that raise are recorded in ``ctx["_degraded"]`` and
        surfaced on the result (``PortfolioMetrics.degraded_computers``). That
        turns a silent zero/blank section (a crashed computer leaves its ctx
        keys unset, and ``_build_result`` substitutes 0.0/{}/empty) into an
        explicit signal a renderer can flag as "unavailable" rather than
        presenting it as a real low-risk / zero result.
        """
        from tarzan.runtime import data_quality as dq
        ctx: dict = {}
        degraded: list[str] = []
        for computer in self._computers:
            name = getattr(computer, "__name__", str(computer))
            try:
                computer(ctx)
            except Exception as e:
                logger.error("Metric computer '%s' failed: %s", name, e)
                degraded.append(name)
                dq.error(
                    "metrics",
                    f"computer '{name}' failed ({e}); its section fell back to "
                    "defaults (blank/zero) and should NOT be read as a real result",
                    context=name,
                )
        degraded = list(dict.fromkeys(
            degraded + list(ctx.get("_degraded", []))
        ))
        if degraded:
            logger.warning(
                "Report is DEGRADED — %d computer(s) failed: %s. "
                "Affected sections fell back to defaults.",
                len(degraded), ", ".join(degraded),
            )
        ctx["_degraded"] = degraded
        return self._build_result(ctx)

    # ------------------------------------------------------------------
    # Run-scoped instrument preprocessing
    # ------------------------------------------------------------------
    def _preprocess_benchmarks(self, ctx: dict) -> None:
        """Resolve all benchmark tickers once before analytical consumers."""
        catalog, errors = _preprocess_benchmark_catalog(
            BENCHMARKS,
            fetch_history=_fetch_benchmark_history,
        )
        ctx["_benchmark_catalog"] = catalog
        ctx["benchmark_tickers"] = {
            name: record.ticker for name, record in catalog.items()
        }
        ctx["benchmark_resolution_errors"] = errors
        if errors:
            logger.warning(
                "Benchmark preprocessing left %d unresolved instrument(s): %s",
                len(errors), "; ".join(errors),
            )

    @staticmethod
    def _benchmark_record(
        ctx: dict,
        name: str,
    ) -> Optional[ResolvedBenchmark]:
        return (ctx.get("_benchmark_catalog") or {}).get(name)

    def _alpha_beta_benchmark(self, ctx: dict) -> Optional[ResolvedBenchmark]:
        return self._benchmark_record(ctx, cfg.benchmark_beta_name())

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def _valuation(self, ctx: dict) -> None:
        rows = []
        cash_class = AssetClass.CASH_EQUIVALENTS.value
        for h in self._current_valuation_holdings():
            # ``_current_valuation_holdings`` has already applied the exact
            # policy-selected value and excluded every rejected evidence row.
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
                "asset_class": h.asset_class.value if h.asset_class else None,
                "geography": geo_str, "sector": h.sector or "Unknown", "currency": h.currency,
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

        from tarzan.models.taxonomy import ORDER_DASHBOARD
        class_order = {v: i for i, v in enumerate(ORDER_DASHBOARD)}
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
        # Naive calendar-day index so cross-exchange (different tz) series align,
        # dropping the duplicate days the tz collapse can create.
        combined = normalize_index(combined, drop_duplicates=True)
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

        # Inception is the first dated point of the order-derived history;
        # there is no separate config override, so the full series is the
        # since-inception series.
        ctx["portfolio_history"] = ph_full.copy()

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
            _causal_enriched_by_isin,
            build_holdings_from_orders,
            build_order_derived_series,
        )
        from tarzan import runtime

        # Establish the as-of order authority before any full-ledger evidence
        # can enter holding construction, enrichment, mechanics, tax, or
        # inception. The effective list is retained for every later
        # order-dependent computer in this run.
        analysis_date = runtime.today()
        effective_orders = [
            order for order in (self.orders or [])
            if order.trade_date <= analysis_date
        ]
        ctx["_effective_orders"] = effective_orders
        ctx["_order_analysis_date"] = analysis_date

        # Project supplied enrichment onto the same effective order authority
        # used by the direct builders. This strips excluded order mechanics and
        # any category fields derived from them before history, tax, or
        # historical allocation consumes the map.
        supplied_by_isin = {
            holding.isin: holding for holding in self.holdings if holding.isin
        }
        enriched_by_isin = _causal_enriched_by_isin(
            effective_orders,
            supplied_by_isin,
        )
        ctx["_historical_classification_projected"] = any(
            supplied_by_isin[isin].asset_class != holding.asset_class
            or supplied_by_isin[isin].class_breakdown != holding.class_breakdown
            for isin, holding in enriched_by_isin.items()
        )
        # Re-derive the complete historical instrument universe and enrich any
        # exact ISIN not already present. Closed positions are represented by
        # historical-only carriers, so they can contribute real provider
        # history without entering current valuation or allocation.
        missing = [
            holding
            for holding in build_holdings_from_orders(
                effective_orders,
                include_closed=True,
            )
            if holding.isin and holding.isin not in enriched_by_isin
        ]
        if missing:
            from tarzan.data.enricher import enrich_holdings
            logger.info(
                "Enriching %d historical order ISIN(s) for returns",
                len(missing),
            )
            for holding in enrich_holdings(missing):
                if holding.isin:
                    enriched_by_isin[holding.isin] = holding

        series = build_order_derived_series(
            effective_orders,
            enriched_by_isin,
            today=analysis_date,
        )
        ctx["_order_series"] = series
        ctx["_enriched_by_isin"] = enriched_by_isin
        ctx["history_availability"] = series.history_availability.value
        ctx["history_unavailable_instruments"] = tuple(
            series.unavailable_instruments
        )
        ctx["_order_mechanics_unavailable"] = list(
            series.mechanics_unavailable_instruments
        )
        ctx["_order_price_history_unavailable"] = list(
            series.causal_price_unavailable_instruments
        )

        from tarzan.runtime import data_quality as dq
        from tarzan.runtime.ledger import Availability
        if series.history_availability is Availability.UNAVAILABLE:
            unavailable = list(series.unavailable_instruments)
            empty = pd.Series(dtype=float)
            ctx["_order_history_unavailable"] = unavailable
            ctx["portfolio_history"] = empty
            ctx["portfolio_history_full"] = empty
            ctx["excluded_short_tenure"] = [
                {"ticker": isin, "name": isin, "value_eur": None,
                 "weight_pct": None, "span_days": 0}
                for isin in unavailable
            ]
            ctx.setdefault("_degraded", []).append(
                "_portfolio_history_from_orders"
            )
            dq.error(
                "returns",
                "Portfolio return history is Unavailable because exact "
                "instrument mechanics or causal historical price evidence "
                "are missing or conflicting.",
                context=",".join(unavailable),
            )
            return

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

        # The daily series is a DENSE calendar-day NAV (freq="D", weekends
        # carried flat). Risk metrics annualize with sqrt(252) (trading days),
        # so collapse to business days here — otherwise weekend zero-returns
        # understate volatility ~17% and pollute VaR/CVaR. This is the single
        # series _risk / _performance_full read; XIRR/TWROR and the mountain
        # chart read series.* directly and are unaffected.
        from tarzan.engine.stats import to_business_day_series
        ph = to_business_day_series(ph)

        ctx["portfolio_history"] = ph
        ctx["portfolio_history_full"] = ph
        ctx["excluded_short_tenure"] = [
            {"ticker": isin, "name": isin, "value_eur": 0.0,
             "weight_pct": 0.0, "span_days": 0}
            for isin in series.provenance.get("excluded", [])
        ]
        # Stash for _returns.
        ctx["_order_series"] = series
        ctx["_enriched_by_isin"] = enriched_by_isin

    # ------------------------------------------------------------------
    # Returns: XIRR + TWROR (only registered when orders are present)
    # ------------------------------------------------------------------
    def _returns(self, ctx: dict) -> None:
        series = ctx.get("_order_series")
        if series is None:
            return
        effective_orders = ctx.get("_effective_orders")
        if effective_orders is None:
            effective_orders = list(self.orders or [])
        from tarzan.runtime.ledger import Availability
        if series.history_availability is Availability.UNAVAILABLE:
            for field in (
                "xirr_pct",
                "twror_pct",
                "twror_annualized_pct",
                "returns_coverage_pct",
                "returns_period_debug",
                "pnl_eur",
                "pnl_pct",
                "invested_capital_eur",
                "estimated_cgt_eur",
                "pnl_eur_net_tax",
                "pnl_pct_net_tax",
                "xirr_net_tax_pct",
                "actual_value_series",
                "pnl_series",
                "unrealized_series",
                "external_flows",
            ):
                ctx[field] = None
            ctx["returns_provenance"] = series.provenance
            ctx.setdefault("_degraded", []).append("_returns")
            order_dates = [
                o.trade_date for o in effective_orders
                if o.is_position_change()
            ]
            if order_dates:
                ctx["inception_date"] = min(order_dates).strftime("%Y-%m-%d")
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

        # Lifetime P&L (realized + unrealized), straight from the XIRR cash
        # flows. Those are the bank-account view — deposits negative, every
        # distribution (sells, coupons, dividends) positive, terminated by
        # today's portfolio value — so their algebraic sum is exactly
        #     current_value + Σ distributions − Σ deposits
        # i.e. the all-in euro gain since inception. PnL% expresses it over
        # the *net* capital contributed (gross deposits − everything taken
        # back out), which equals current_value − pnl_eur: "how much did I
        # gain on the money I actually left in". Using net (not gross)
        # deposits keeps cum/ex bond rotations and transfer-then-sell
        # round-trips from inflating the denominator.
        flows = series.xirr_cashflows or []
        pnl_eur = sum(amount for _, amount in flows)
        current_value = flows[-1][1] if flows else 0.0
        net_deposits = current_value - pnl_eur
        ctx["pnl_eur"] = pnl_eur
        ctx["invested_capital_eur"] = net_deposits
        ctx["pnl_pct"] = (pnl_eur / net_deposits * 100.0) if net_deposits > 0 else None
        ctx["actual_value_series"] = series.actual_value_series
        ctx["pnl_series"] = series.pnl_series
        # External capital flows (deposits/buys +, withdrawals/sells/distros −)
        # per date — the same dict TWROR consumes. Drives the deposit/withdrawal
        # markers on the newsletter performance charts (no recomputation).
        ctx["external_flows"] = series.external_flows
        # Unrealized P&L series: the order-derived reconstruction differs from
        # the snapshot in level (bonds priced carry-flat; cum/ex-netted legs
        # carry residual cost the open-positions snapshot drops). We keep its
        # daily SHAPE but anchor the END to the authoritative snapshot the
        # hero shows — unrealized = total_value − Σ cost_basis(open holdings) —
        # via a constant shift. Consistent with the hero, no double source.
        ur = series.unrealized_series
        if ur is not None and not ur.empty:
            hdf = ctx.get("holdings_df")
            cost_now = (float(hdf["cost_basis_eur"].sum())
                        if hdf is not None and not hdf.empty else 0.0)
            hero_unreal = float(ctx.get("total_value", 0.0)) - cost_now
            # Anchor on the last OBSERVED point, not on ``iloc[-1]``: a NaN
            # tail (an instrument with no price for today) would make the
            # shift NaN and poison every day in the series, not just the tail.
            observed = ur.dropna()
            if observed.empty:
                ctx["unrealized_series"] = None
            else:
                ctx["unrealized_series"] = ur + (hero_unreal - float(observed.iloc[-1]))
        else:
            ctx["unrealized_series"] = ur

        # Net-of-tax estimate (Italian CGT on realized gains). This is an
        # ESTIMATE shown alongside — never replacing — the gross figures:
        # the tax is a real cash outflow, so it lowers the money-weighted
        # views (lifetime PnL and XIRR). TWROR is left gross by convention.
        from tarzan.engine.tax import estimate_realized_cgt
        enriched_by_isin = ctx.get("_enriched_by_isin", {})
        cgt = estimate_realized_cgt(
            effective_orders, enriched_by_isin,
            std_rate_pctg=self.config.rebalancing_capital_gains_tax_standard_pctg,
            gov_rate_pctg=self.config.rebalancing_capital_gains_tax_government_pctg,
        )
        ctx["estimated_cgt_eur"] = cgt.total_tax_eur
        if cgt.total_tax_eur > 0:
            pnl_net = pnl_eur - cgt.total_tax_eur
            ctx["pnl_eur_net_tax"] = pnl_net
            ctx["pnl_pct_net_tax"] = (
                pnl_net / net_deposits * 100.0 if net_deposits > 0 else None
            )
            # Net XIRR: the tax outflows are dated at each sell's trade date
            # (regime amministrato withholding) and discounted like any
            # other cash flow.
            net_flows = list(flows) + list(cgt.tax_flows)
            r_net = xirr(net_flows)
            ctx["xirr_net_tax_pct"] = r_net * 100.0 if not _is_nan(r_net) else None
        else:
            ctx["pnl_eur_net_tax"] = pnl_eur
            ctx["pnl_pct_net_tax"] = ctx["pnl_pct"]
            ctx["xirr_net_tax_pct"] = ctx["xirr_pct"]

        # Inception is automatic from the order list: the first dated
        # position change. This is the authoritative start of the track
        # record, independent of any config value. Keyed on trade_date
        # (market-exposure date), consistent with the returns engine.
        order_dates = [
            o.trade_date for o in effective_orders if o.is_position_change()
        ]
        if order_dates:
            ctx["inception_date"] = min(order_dates).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Allocation timeline (weekly mix over the last ~3 months)
    # ------------------------------------------------------------------
    def _allocation_timeline(self, ctx: dict) -> None:
        """Reconstruct the weekly asset-class and equity-geo mix over the
        last 3 months for the newsletter sparklines.

        Reuses the enriched holdings stashed by
        ``_portfolio_history_from_orders`` (so it adds no network calls)
        and, when the current snapshot shares the same causal classification
        projection, anchors the final weekly bucket to the authoritative live
        allocation (``allocation_by_class`` / ``allocation_by_geo``) so the
        sparkline's endpoint matches the donut and metrics table exactly.
        """
        enriched = ctx.get("_enriched_by_isin")
        effective_orders = ctx.get("_effective_orders")
        if effective_orders is None:
            effective_orders = list(self.orders or [])
        if not effective_orders or not enriched:
            return
        # The point-in-time allocation can remain independently available from
        # exact current valuation/category evidence, but no historical
        # allocation may be reconstructed once any order mechanics are
        # ambiguous. Guard here as well as in the builder so a future partial
        # implementation cannot silently emit a valid-subset portfolio trend.
        if ctx.get("_order_history_unavailable"):
            ctx["allocation_timeline"] = None
            return
        if not ctx.get("classification_available", True):
            ctx["allocation_timeline"] = None
            return
        from tarzan.engine.returns_builder import build_allocation_timeline

        tl = build_allocation_timeline(
            effective_orders,
            enriched,
            months=3,
            today=ctx.get("_order_analysis_date"),
        )
        if not tl or len(tl.get("dates", [])) < 2:
            return

        # Anchor to the current allocation only when it is based on the same
        # causal classification projection. If excluded order evidence forced
        # historical category fields to be replaced, the original snapshot is
        # not a valid terminal authority for this as-of timeline.
        if not ctx.get("_historical_classification_projected"):
            cash_class = AssetClass.CASH_EQUIVALENTS.value
            by_class = ctx.get("allocation_by_class")
            if by_class is not None and not by_class.empty and tl["asset"]:
                tl["asset"][-1] = {
                    r["category"]: float(r["weight_pct"])
                    for _, r in by_class.iterrows()
                    if r["category"] != cash_class
                }
            by_geo = ctx.get("allocation_by_geo")
            if by_geo is not None and not by_geo.empty and tl["geo"]:
                tl["geo"][-1] = {
                    r["category"]: float(r["weight_pct"])
                    for _, r in by_geo.iterrows()
                }
        ctx["allocation_timeline"] = tl

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    def _performance(self, ctx: dict) -> None:
        if ctx.get("_order_history_unavailable"):
            ctx["performance"] = None
            ctx["performance_full"] = None
            return
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        if ph.empty:
            ctx["performance"] = {"cagr": 0.0, "ytd": None, "1d": None, "5d": None,
                                  "1m": None, "3m": None, "6m": None, "1y": None,
                                  "3y": None, "5y": None}
        else:
            result = {"cagr": compute_cagr(ph), "ytd": compute_ytd_return(ph)}
            for key in PERIOD_WINDOWS:
                result[key] = compute_period_return(ph, key)
            ctx["performance"] = result

        # Also compute full (non-inception) performance for Performance tab
        ph_full = ctx.get("portfolio_history_full", pd.Series(dtype=float))
        if ph_full.empty:
            ctx["performance_full"] = {}
        else:
            record = self._alpha_beta_benchmark(ctx)
            bench = (
                record.history if record is not None
                else pd.Series(dtype=float)
            )
            full_row = {"ticker": "PORTFOLIO", "name": "** TOTAL PORTFOLIO **", "type": "Portfolio"}
            _populate_perf_row(full_row, ph_full, bench, self._rf_daily(ctx))
            ctx["performance_full"] = full_row

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------
    def _rf_daily(self, ctx: dict):
        """The time-varying daily risk-free path for Sharpe/Sortino, memoized
        per run into ``ctx``.

        Fetched from ``proxy_data.risk_free_daily`` (real ECB SR_3M+EONIA / ^IRX
        historical series) so every day is charged its own prevailing short rate
        instead of a single hardcoded 4%. The proxy fetch layer already gates on
        ``runtime.allows_live_transport()`` and clips to as_of, so a pinned run
        returns only cache rows visible at as_of (never live transport, never
        future data); when nothing is cached it returns ``None`` and the risk
        block falls back to the scalar ``RISK_FREE_RATE``.
        """
        if "_rf_daily" not in ctx:
            from tarzan.data import proxy_data
            try:
                ctx["_rf_daily"] = proxy_data.risk_free_daily()
            except Exception:  # noqa: BLE001 — any fetch failure → scalar fallback
                ctx["_rf_daily"] = None
        return ctx["_rf_daily"]

    def _risk(self, ctx: dict) -> None:
        if ctx.get("_order_history_unavailable"):
            ctx["risk"] = None
            return
        ph = ctx.get("portfolio_history", pd.Series(dtype=float))
        # Unavailable, not zero: with <2 points there is no risk to measure, so
        # every metric is nan (rendered "—"). A 0.0 volatility/drawdown would
        # read as "measured, no risk" — the numeric-zero≠unavailable invariant.
        result = {"volatility": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"),
                  "sortino": float("nan"), "var_95": float("nan"), "cvar_95": float("nan"),
                  "beta": float("nan"), "alpha": float("nan")}
        if ph.empty or len(ph) < 2:
            ctx["risk"] = result
            return
        daily_returns = ph.pct_change().dropna()
        if daily_returns.empty:
            ctx["risk"] = result
            return
        # Single source of truth for the risk block (dedup with the benchmark /
        # per-holding rows): compute cagr/vol/sharpe/sortino/maxdd/var/cvar once
        # via risk_metric_row, with the time-varying risk-free path so Sharpe and
        # Sortino use per-day rates. Keep only the keys this dashboard block emits.
        block = risk_metric_row(ph, self._rf_daily(ctx))
        result["volatility"] = block["volatility"]
        result["sharpe"] = block["sharpe"]
        result["max_drawdown"] = block["max_drawdown"]
        result["sortino"] = block["sortino"]
        result["var_95"] = block["var_95"]
        result["cvar_95"] = block["cvar_95"]
        # Beta/Alpha use the same preprocessed full benchmark ticker/history
        # consumed by every other analytical and presentation stage.
        record = self._alpha_beta_benchmark(ctx)
        bench_history = (
            record.history if record is not None
            else pd.Series(dtype=float)
        )
        if not bench_history.empty and len(bench_history) > 1:
            beta, alpha = _compute_beta_alpha(
                ph, bench_history, block["cagr"],
                risk_free=rf_annual_pct(self._rf_daily(ctx)),
            )
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

        # Invested allocation as NOTIONAL exposure: each non-cash holding's
        # value is distributed across its ``class_breakdown`` (which may sum
        # to >100% for capital-efficient / leveraged funds — e.g. NTSG 90
        # equity + 60 fixed income), so the class weights reflect the true
        # economic exposure and can sum to more than 100% of invested capital.
        from tarzan.engine import allocations as alloc

        current_holdings = self._current_valuation_holdings()
        bd_by_isin: dict[str, dict] = {
            h.isin: alloc.holding_class_breakdown(h)
            for h in current_holdings
            if h.isin
        }
        unresolved_categories = sorted(
            h.isin or h.ticker
            for h in current_holdings
            if float(h.current_value or h.market_value_eur or 0.0) > 0
            and not alloc.holding_class_breakdown(h)
        )
        classification_available = not unresolved_categories
        ctx["classification_available"] = classification_available
        ctx["classification_unavailable_instruments"] = unresolved_categories
        if not classification_available:
            from tarzan.runtime import data_quality as dq
            dq.error(
                "instrument_capability",
                "Allocation, goals, and rebalancing are unavailable because "
                "one or more valued instruments lack an exact tracked category.",
                context=",".join(unresolved_categories),
            )
        notional: dict[str, float] = {}
        if classification_available and not df.empty and invested_value > 0:
            # Distribute each non-cash holding's value across its class
            # breakdown and sum — the shared notional-aggregation primitive,
            # so the live portfolio and the backtest aggregate identically.
            pairs = []
            for _, row in df.iterrows():
                if row.get("asset_class") == cash_class:
                    continue
                val = float(row.get("current_value", 0.0) or 0.0)
                bd = bd_by_isin.get(row.get("isin")) or {row.get("asset_class"): 100.0}
                pairs.append((val, {c: p for c, p in bd.items()
                                    if c and c != cash_class}))
            notional = alloc.accumulate(pairs)
            by_class = pd.DataFrame(
                [{"category": k, "weight_pct": v / invested_value * 100.0}
                 for k, v in notional.items()]
            ).sort_values("weight_pct", ascending=False).reset_index(drop=True)
        else:
            by_class = pd.DataFrame(columns=["category", "weight_pct"])

        by_geo = (
            _compute_geo_allocation(df, current_holdings)
            if classification_available
            else pd.DataFrame(columns=["category", "weight_pct"])
        )
        by_sector = pd.DataFrame(columns=["category", "weight_pct"])
        if (classification_available and not df.empty and invested_value > 0
                and "sector" in df.columns):
            eligible = df[df["asset_class"] != cash_class].copy()
            if not eligible.empty:
                eligible["sector"] = eligible["sector"].fillna("Unknown").replace("", "Unknown")
                by_sector = (
                    eligible.groupby("sector", dropna=False)["weight_of_invested_pctg"]
                    .sum()
                    .reset_index()
                )
                by_sector.columns = ["category", "weight_pct"]
                by_sector = by_sector.sort_values(
                    ["weight_pct", "category"], ascending=[False, True]
                ).reset_index(drop=True)
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
        ctx["weighted_yield"] = float((pd.to_numeric(df["yield_pct"], errors="coerce").fillna(0.0) * df["weight_pct"]).sum() / total_weight) * 100
        ctx["avg_ter"] = float((pd.to_numeric(df["ter"], errors="coerce").fillna(0.0) * df["weight_pct"]).sum() / total_weight) * 100

    # ------------------------------------------------------------------
    # Goal deltas
    # ------------------------------------------------------------------
    def _goals(self, ctx: dict) -> None:
        if self.config is None:
            ctx["goal_deltas"] = None
            return
        if not ctx.get("classification_available", True):
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

        # Equity geography rows: % of equity portion. The producer guarantees
        # this schema even when empty; retain a defensive boundary for injected
        # or legacy metric frames so goals cannot fail the whole report.
        actual_geo = (
            dict(zip(by_geo["category"], by_geo["weight_pct"]))
            if {"category", "weight_pct"}.issubset(by_geo.columns)
            else {}
        )
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
            return
        mechanics_unavailable = ctx.get("_order_mechanics_unavailable")
        if mechanics_unavailable:
            # Current allocation analytics may use independently established
            # valuation/category evidence, but executable planning requires an
            # exact instrument-kind adapter. Historical price completeness is
            # deliberately not a planning input and must not disable a plan
            # based on trustworthy current valuation and classification.
            ctx["rebalancing_suggestions"] = None
            ctx["rebalancing_verifications"] = None
            ctx["rebalancing_plans"] = None
            from tarzan.runtime import data_quality as dq
            dq.error(
                "rebalancing",
                "planning is Unavailable because exact instrument mechanics "
                "are missing or conflicting",
                context=",".join(mechanics_unavailable),
            )
            return
        if not self.planning_eligible:
            ctx["rebalancing_suggestions"] = None
            ctx["rebalancing_verifications"] = None
            ctx["rebalancing_plans"] = None
            from tarzan.runtime import data_quality as dq
            dq.error(
                "rebalancing",
                "planning is Unavailable because valuation completeness policy "
                "did not establish a trustworthy portfolio total",
                context="valuation_completeness",
            )
            return
        if not ctx.get("classification_available", True):
            ctx["rebalancing_suggestions"] = None
            ctx["rebalancing_verifications"] = None
            ctx["rebalancing_plans"] = None
            from tarzan.runtime import data_quality as dq
            dq.error(
                "rebalancing",
                "planning is Unavailable because tracked-category evidence "
                "is missing for a valued instrument",
                context="instrument_capability",
            )
            return
        from tarzan.engine.rebalancer import compute_unified_rebalancing
        lump = self.config.rebalancing_lump_sum_amount_eur if self.config.rebalancing_lump_sum_amount_eur > 0 else None

        # Always compute BOTH rebalancing variants so Excel and the
        # newsletter can present them side by side: a buy-only
        # (accumulation) plan and a full buy & sell plan. They differ only
        # in the ``rebalancing_no_sell`` flag; everything else (targets,
        # tolerance, lump sum) is identical.
        import dataclasses

        # The rebalancer sees only policy-accepted current holdings, projected
        # at their selected values, plus any seeded (not-held) targets. Rejected
        # valuation evidence cannot re-enter planning through the raw models.
        rebal_holdings = self._current_valuation_holdings() + list(self.rebalance_seeds)

        def _plan(no_sell: bool):
            cfg2 = dataclasses.replace(self.config, rebalancing_no_sell=no_sell)
            return compute_unified_rebalancing(
                rebal_holdings, cfg2, ctx["total_value"], lump_sum=lump)

        s_true, v_true = _plan(True)
        s_false, v_false = _plan(False)
        # Estimated cash cost of executing each plan (CGT on the sells + fixed
        # commission fees), from the same tax/fee model the optimizer solved
        # for. Computed here where ``rebal_holdings`` (real + seeds) is in scope
        # so the action ``idx`` lookups are correct.
        from tarzan.engine.rebalancer import plan_cost as _plan_cost
        cost_true = _plan_cost(s_true, rebal_holdings, self.config)
        cost_false = _plan_cost(s_false, rebal_holdings, self.config)
        ctx["rebalancing_plans"] = [
            {"label": "Buy only (accumulate)", "no_sell": True,
             "suggestions": s_true, "verifications": v_true,
             "cgt_eur": cost_true["cgt_eur"], "fees_eur": cost_true["fees_eur"]},
            {"label": "Buy & sell (full rebalance)", "no_sell": False,
             "suggestions": s_false, "verifications": v_false,
             "cgt_eur": cost_false["cgt_eur"], "fees_eur": cost_false["fees_eur"]},
        ]

        # Append both plans to the per-run rebalancing audit trail (inputs +
        # outputs), so the WHY behind each suggested trade is durably
        # recorded. Best-effort; never affects the plans or any number.
        from tarzan.runtime import audit as _audit
        for _label, _ns, _s, _v in (
            ("Buy only (accumulate)", True, s_true, v_true),
            ("Buy & sell (full rebalance)", False, s_false, v_false),
        ):
            _audit.record_rebalancing_plan(
                _label, no_sell=_ns, total_value=ctx["total_value"],
                lump_sum=lump, config=self.config, holdings=rebal_holdings,
                suggestions=_s, verifications=_v,
            )
        # Primary plan = the one matching the user's configured no_sell, kept
        # for back-compat (hero status banner, dashboard alert, etc.).
        if self.config.rebalancing_no_sell:
            ctx["rebalancing_suggestions"], ctx["rebalancing_verifications"] = s_true, v_true
        else:
            ctx["rebalancing_suggestions"], ctx["rebalancing_verifications"] = s_false, v_false

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
        # Publish a history for every TRACKED instrument, which is the same set
        # the catalog fetched — including a row flagged only as the alpha/beta or
        # geo reference. Reading a narrower set here withheld the geo benchmark's
        # own series from the charts that exist to draw it.
        chart_benchmarks = set(cfg.benchmarks())
        catalog: dict[str, ResolvedBenchmark] = ctx.get("_benchmark_catalog", {})

        # Every metric below consumes the exact history attached to the one
        # full ticker selected by preprocessing.  No fetch or symbol resolver
        # is reachable from this stage.
        risk_window = ctx.get("portfolio_history_full", pd.Series(dtype=float))
        if risk_window is None or risk_window.empty:
            risk_window = ph
        win_start, win_end = risk_window.index.min(), risk_window.index.max()
        ab_record = self._alpha_beta_benchmark(ctx)
        ab_benchmark = (
            ab_record.history if ab_record is not None
            else pd.Series(dtype=float)
        )
        ab_benchmark_win = _clip_to_window(ab_benchmark, win_start, win_end)

        for name, record in catalog.items():
            bench = record.history
            if bench.empty or len(bench) < 2:
                continue
            if name in chart_benchmarks:
                key_histories[name] = bench
            bench_win = _clip_to_window(bench, win_start, win_end)
            if len(bench_win) < 2:
                bench_win = bench
            metrics = _compute_single_benchmark_metrics(
                bench_win, ab_benchmark_win, self._rf_daily(ctx)
            )
            metrics["benchmark"] = name
            metrics["ticker"] = record.ticker
            comp_rows.append(metrics)

        _add_mix_to_histories(key_histories, initial_value, catalog)
        ctx["benchmark_comparison"] = pd.DataFrame(comp_rows)
        ctx["benchmark_histories"] = key_histories

    # ------------------------------------------------------------------
    # Per-holding performance
    # ------------------------------------------------------------------
    @staticmethod
    def _own_tape(series, native, currency: Optional[str]) -> tuple:
        """``(series, currency)`` for a per-instrument row: the instrument's OWN
        tape, not the book's currency.

        A return is a ratio and FX does not divide out of one. RSSY's five
        sessions to 28 Aug 2026 read −1.28% on its own Nasdaq tape and −2.18%
        with each end converted at its own day's rate; the second is a true
        statement about a euro investor's outcome, the first is what the
        instrument did, and a table that compares instruments to each other (and
        to the figure the reader sees on the issuer's or Yahoo's page) needs the
        first. Everything that values the PORTFOLIO keeps reading the EUR series.

        Falls back to the EUR series when no native one was kept, so a row is
        never dropped for want of a tape — 57 of the 81 rows list in EUR anyway,
        where the two series are the same object.

        An unknown currency stays UNKNOWN. ``_currency_mark`` already omits the
        mark for an empty code, so defaulting to "EUR" here printed ``[€]`` on a
        row whose listing currency nobody ever reported — and the mark exists
        precisely so the reader knows which currency a figure is in. Empty string
        rather than None: this value lands in a pandas column, and a NaN renders
        as ``[NAN]``.
        """
        if native is not None and len(native) >= 2:
            return native, (currency or "")
        return series, (currency or "")

    def _holding_performance(self, ctx: dict) -> None:
        rows = []
        ab_record = self._alpha_beta_benchmark(ctx)
        bench_history = (
            ab_record.history if ab_record is not None
            else pd.Series(dtype=float)
        )

        unavailable = set(ctx.get("_order_history_unavailable", []))
        for h in self.holdings:
            if h.isin in unavailable:
                continue
            if h.price_history is None or len(h.price_history) < 2:
                continue
            tape, ccy = self._own_tape(h.price_history, h.price_history_native,
                                       h.price_currency)
            s = _cap_to_years(tape, 5)
            # Holdings have already crossed the enrichment preprocessing
            # boundary: ``h.ticker`` is the sole full operational ticker.
            row: dict = {
                "ticker": h.ticker,
                "name": h.name or h.ticker,
                "type": "In portfolio",
                "currency": ccy,
            }
            _populate_perf_row(row, s, bench_history, self._rf_daily(ctx))
            rows.append(row)

        # Target instruments the book does not hold yet. The orchestrator seeds
        # one zero-value position per positive per-holding target and enriches
        # them in the SAME ``enrich_holdings`` call as the real holdings, so their
        # price history is a holding's history and their period returns come off
        # the same ``_populate_perf_row`` as every other row in this table.
        #
        # Emitted here rather than being recovered downstream, because the two
        # things a reader's target list must not depend on are exactly what a
        # downstream reconstruction would depend on: whether the instrument also
        # carries ``watchlist=true`` in the taxonomy (the tracked catalog is the
        # only other place a not-held instrument gets returns), and whether the
        # rebalancer produced a plan this run (the target set is configuration; a
        # suppressed plan must not empty it).
        for h in self.rebalance_seeds:
            if h.price_history is None or len(h.price_history) < 2:
                continue
            row = {
                "ticker": h.ticker,
                "name": h.name or h.ticker,
                # Neither "portfolio" nor "benchmark" as a substring: every
                # consumer of this frame selects on one of those two words, so a
                # third kind of row must match neither to stay out of the book's
                # tables and out of the benchmark projections.
                "type": "Target not held",
            }
            tape, row["currency"] = self._own_tape(
                h.price_history, h.price_history_native, h.price_currency)
            _populate_perf_row(row, _cap_to_years(tape, 5),
                               bench_history, self._rf_daily(ctx))
            rows.append(row)

        # Benchmark rows receive the exact same full ticker whose attached
        # history is used for every metric and chart.
        catalog: dict[str, ResolvedBenchmark] = ctx.get("_benchmark_catalog", {})
        for record in catalog.values():
            tape, ccy = self._own_tape(record.history, record.history_native,
                                       record.currency)
            bs = _cap_to_years(tape, 5)
            row = {
                "ticker": record.ticker,
                "name": record.name,
                "type": "Benchmark index",
                "currency": ccy,
            }
            _populate_perf_row(row, bs, bench_history, self._rf_daily(ctx))
            rows.append(row)

        ctx["holding_performance"] = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Broker-style (live) 1D
    # ------------------------------------------------------------------
    def _current_prices(self, ctx: dict) -> None:
        """Stamp the current session onto every TRACKED BENCHMARK's series.

        Holdings are stamped in the data layer, before the valuation policy —
        see :mod:`tarzan.data.current_session`, which owns the one rule and
        explains why the ordering matters. Benchmarks are not holdings: they
        carry no valuation and feed only windows and charts, so their stamp
        belongs to the run that builds their catalog, which is this engine.
        """
        from tarzan.data import current_session

        allowed, today = current_session.stamping_allowed()
        if not allowed:
            return

        from dataclasses import replace

        from tarzan.data.market_quotes import official_quotes

        catalog = ctx.get("_benchmark_catalog") or {}
        candidates = current_session._candidates(
            {r.ticker for r in catalog.values() if r.ticker}
        )
        if not candidates:
            return
        quotes = official_quotes(
            sorted({s for group in candidates.values() for s in group})
        )

        stamped: list[str] = []
        for name, record in list(catalog.items()):
            series = record.history
            usable = None if series is None else series.dropna()
            if usable is None or len(usable) == 0:
                continue

            def _stamp(tape):
                """Stamp one tape against ITS OWN last close, or None.

                Each tape resolves its own quote, because the level gate compares
                a price against the reference it is handed: an EUR-per-unit close
                against a native quote fails by the whole FX rate. That is why the
                EUR tape of a USD listing is never stamped (0 of 24 on 29 Aug
                2026) while its native tape is — and the two must be attempted
                INDEPENDENTLY. Stamping them in sequence, with the EUR failure
                short-circuiting the loop, left every native USD tape a session
                behind: RSSY read +0.74% (Wed→Thu) where its own Friday session was
                −0.37%, on the very rows whose reason for being native is that the
                reader checks them against Yahoo.
                """
                clean = None if tape is None else tape.dropna()
                if clean is None or len(clean) == 0:
                    return None
                q = current_session.pick_quote(
                    candidates.get(record.ticker, []), quotes,
                    float(clean.iloc[-1]))
                p = q.get("price")
                if not p:
                    return None
                return current_session.stamp_today(
                    tape, today, float(p), q, ticker=record.ticker)

            history = _stamp(series)
            native_stamped = _stamp(record.history_native)
            if history is None and native_stamped is None:
                continue
            ctx["_benchmark_catalog"][name] = replace(
                record,
                history=history if history is not None else record.history,
                history_native=(native_stamped if native_stamped is not None
                                else record.history_native))
            stamped.append(record.ticker)

        if stamped:
            logger.info(
                "Stamped a current point onto %d benchmark series", len(stamped))

    def _live_1d(self, ctx: dict) -> None:
        """Resolve each run's intraday feed once and project broker-style 1D.

        The canonical full ticker remains the instrument identity used by
        history, metrics and display. When that listing has no bars, the
        provider may select a price-coherent EUR sibling for the intraday data
        class only. The selected series, baseline and provenance are retained
        in ``ctx`` so presentation performs no market lookup or second fetch.
        """
        hp = ctx.get("holding_performance")
        if hp is None or getattr(hp, "empty", True) or "ticker" not in hp.columns:
            return

        keys = [str(t) for t in hp["ticker"].dropna().unique() if t]
        ctx["intraday_requested_tickers"] = tuple(keys)
        ctx["intraday_quotes"] = {}

        from tarzan import runtime
        if not runtime.allows_live_transport():
            return

        # Whether any venue the portfolio holds is TRADING right now, from
        # exchange hours alone. Deliberately separate from ``1d_live`` below,
        # and computed before any fetch so a provider failure cannot silence
        # it: ``1d_live`` says "the 1D figures are intraday", this says "the
        # market is open". They disagree for the ~half hour after an open,
        # when the venue trades but no bar exists yet — and reporting "market
        # closed" then (the 09:09 send) is wrong, while flipping ``1d_live``
        # would put an "Intraday" heading over close-to-close returns.
        # It is the venues carrying the VALUE that answer it, not whichever
        # tracked listing happens to quote earliest: at 08:58 CEST on 18 Aug
        # 2026 one Munich-only watchlist row (08:00–22:00) reported the
        # portfolio's market open while Milan, Xetra, London and New York — all
        # of its actual holdings — were shut. Same rule the Intraday header
        # already follows: one row does not speak for the whole column.
        # Instruments with no modelled cash session (cash, FX, futures) state
        # nothing either way and are counted on neither side.
        try:
            from tarzan.data.market_quotes import market_open_now
            states = {t: market_open_now(t) for t in keys}
            value_by_ticker: dict[str, float] = {}
            for holding in self.holdings:
                ticker = str(getattr(holding, "ticker", "") or "")
                if states.get(ticker) is None:
                    continue
                value_by_ticker[ticker] = value_by_ticker.get(ticker, 0.0) + float(
                    getattr(holding, "current_value", None)
                    or getattr(holding, "market_value_eur", None)
                    or 0.0
                )
            judged = sum(value_by_ticker.values())
            if judged > 0.0:
                open_value = sum(v for t, v in value_by_ticker.items() if states[t])
                is_open = open_value > judged / 2.0
            else:
                # Nothing valued to weigh (watchlist-only projection): fall back
                # to the presence of any open venue.
                is_open = any(bool(state) for state in states.values())
            ctx["market_open"] = is_open
            for key in ("performance", "performance_full"):
                projection = ctx.get(key)
                if isinstance(projection, dict):
                    projection["market_open"] = is_open
        except Exception as e:  # noqa: BLE001
            logger.debug("market-open check failed: %s", e)

        try:
            from tarzan.data.market_quotes import intraday_feeds, _sibling_symbols

            res = intraday_feeds(keys, allow_sibling_fallback=True)
            identity_errors: list[str] = []
            resolved: dict[str, dict] = {}
            intraday_quotes: dict[str, dict] = {}

            for ticker in keys:
                selected = res.get(ticker)
                if selected is None:
                    continue
                return_source = str(
                    selected.get("source_ticker") or ticker
                ).strip()
                intraday_source = str(
                    selected.get("intraday_source_ticker")
                    or return_source
                    or ticker
                ).strip()
                allowed_sources = {ticker, *_sibling_symbols(ticker)}
                invalid_sources = {
                    source
                    for source in (return_source, intraday_source)
                    if source not in allowed_sources
                }
                if invalid_sources:
                    identity_errors.append(
                        f"{ticker} returned invalid source(s) "
                        f"{sorted(invalid_sources)!r}"
                    )
                    continue

                normalized = dict(selected)
                normalized["source_ticker"] = return_source
                normalized["intraday_source_ticker"] = intraday_source
                resolved[ticker] = normalized
                series = normalized.get("intraday_series")
                if series is not None and len(series) >= 2:
                    intraday_quotes[ticker] = normalized

            if identity_errors:
                logger.error(
                    "Intraday source validation failed: %s",
                    "; ".join(identity_errors),
                )
                ctx.setdefault("_degraded", []).append("_live_1d")

            ctx["intraday_quotes"] = intraday_quotes
            live_flag = {
                ticker: bool(selected.get("live"))
                for ticker, selected in resolved.items()
            }

            for holding in self.holdings:
                canonical = str(holding.ticker or "")
                if not canonical:
                    continue
                selected = resolved.get(canonical)
                if selected is None:
                    holding.intraday_ticker = None
                    holding.intraday_observation_timestamp = None
                    holding.intraday_ticker_reason = (
                        f"The canonical ticker {canonical} was requested first, "
                        "but neither it nor a guarded price-coherent EUR sibling "
                        "supplied a usable intraday series."
                    )
                    continue

                effective = str(
                    selected.get("intraday_source_ticker") or canonical
                )
                holding.intraday_ticker = effective
                holding.intraday_observation_timestamp = selected.get(
                    "intraday_observation_timestamp"
                )
                holding.intraday_ticker_reason = (
                    "The canonical ticker supplied the intraday series."
                    if effective == canonical
                    else (
                        f"The canonical ticker {canonical} was requested first; "
                        f"{effective} supplied the price-coherent EUR venue "
                        "sibling fallback after the canonical-close collision "
                        "guard passed."
                    )
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("live 1D preprocessing failed: %s", e)
            return

        # No percentage is projected onto hp["1d"] or performance["1d"]: those
        # come from each instrument's price series, whose current point and
        # previous-session close are both written from this same published quote
        # pair by ``current_session``. The pair is therefore still the 1D
        # authority — it just reaches the reader once, through the series,
        # instead of being recomputed here into a second answer that drifted by
        # whatever the two feeds disagreed on (0.18pp on 19 Aug 2026).
        #
        # What this computer owns is the evidence the daily series cannot carry:
        # the intraday path for the sparklines, and whether the figures are
        # intraday at all.
        hp["live_1d"] = [
            bool(live_flag.get(str(ticker), False)) for ticker in hp["ticker"]
        ]
        ctx["holding_performance"] = hp
        if not resolved:
            # Nothing resolved, so nothing can be said about the session basis.
            # This used to test a dict of percentages whose values were then
            # discarded — the presence of a pct was standing in for "some feed
            # answered", which is what this says directly.
            return
        any_live = any(bool(flag) for flag in live_flag.values())
        for key in ("performance", "performance_full"):
            projection = ctx.get(key)
            if isinstance(projection, dict):
                projection["1d_live"] = any_live

    def _session_coverage(self, ctx: dict) -> None:
        """Publish how much of the book the 1D figure covers.

        A stage of its own rather than a line inside ``_live_1d``: that computer
        returns early on a provider exception and when no intraday feed resolves,
        so the coverage would go missing exactly when the run is degraded — which
        is when a reader most needs to know the figure is partial.
        """
        coverage = self._today_priced_share()
        for key in ("performance", "performance_full"):
            projection = ctx.get(key)
            if isinstance(projection, dict):
                projection["1d_coverage_pct"] = coverage

    def _today_priced_share(self) -> Optional[float]:
        """Share of the book BY VALUE whose price series carries a today point.

        The 1D is the NAV's own move, and the series resolver carries a holding
        the vendor has not priced today forward at its previous close — so that
        holding contributes exactly ZERO to the numerator while its whole value
        stays in the denominator. Minutes after an open that pulls the figure
        toward zero, and nothing in the digest said so.

        Measured on 27 Aug 2026: at 09:12 only CL2 (7.7% of the book) and UEQC
        (4.6%) had a 27 Aug tick. Their real +0.74% and +0.15% moves rendered as a
        portfolio "1D +0.06%" — reconstructed to +0.0637%, i.e. the whole figure
        was two instruments spread over sixteen. By 10:11, with 91% priced, the
        same book read +0.18%. The market had not tripled; the book had opened.

        The distortion is not even neutral: CL2 is a 2x leveraged ETF and one of
        the first to trade, so the early figure is weighted toward the most
        volatile sleeve.

        Returns None when there is nothing to measure, so a caller can tell "not
        computed" from "nothing priced".
        """
        from tarzan import runtime

        try:
            today = runtime.today()
        except Exception:  # noqa: BLE001 — a caption must never break a render
            return None
        priced = total = 0.0
        for holding in self._current_valuation_holdings():
            value = float(
                holding.current_value
                if holding.current_value is not None
                else (holding.market_value_eur or 0.0)
            )
            if value <= 0:
                continue
            total += value
            series = holding.price_history
            if series is None or not len(series):
                continue
            observed = series.dropna()
            if not len(observed):
                continue
            # The last stamp's own date. On a tz-aware series that is the VENUE's
            # calendar day, which is the day the reader means by "today".
            if pd.Timestamp(observed.index[-1]).date() == today:
                priced += value
        return (priced / total * 100.0) if total > 0 else None

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
        unavailable = set(ctx.get("_order_history_unavailable", []))
        for h in self.holdings:
            if h.isin in unavailable:
                continue
            if h.price_history is not None and len(h.price_history) > 1:
                hh[h.ticker] = {"name": h.name or h.ticker, "history": h.price_history}
        ctx["holding_histories"] = hh

    # ------------------------------------------------------------------
    # Target-portfolio NAV (the allocation the book is steering toward)
    # ------------------------------------------------------------------
    def _target_history(self, ctx: dict) -> None:
        """NAV of the TARGET portfolio: the per-instrument ``target_portfolio``
        weights held over their whole common window, rebalanced quarterly.

        Built here because this is the only scope holding BOTH the real book and
        the rebalance seeds. Nearly a quarter of the current target sits in
        instruments not owned yet (AVWC/AVWS/AVEM on 25 Aug 2026), so the held
        book alone would draw a different portfolio and label it "target".

        The weights are held CONSTANT, every day, on every window. That is what a
        target allocation is: 32% NTSG is 32% at every point on the line, not a
        weight that drifts with performance and snaps back four times a year. A
        periodic policy would make the line depend on where the rebalance dates
        happen to fall relative to the window the reader is looking at.

        Prices are aligned BEFORE differencing — taking each sleeve's returns
        first and inner-joining after would silently delete the return of any day
        a single venue was shut.
        """
        from tarzan.engine.synthetic import combine_returns

        unavailable = set(ctx.get("_order_history_unavailable", []))
        weights, prices, missing = {}, {}, []
        # The POLICY, before any availability filtering: every instrument the
        # target names and its weight, keyed the way the intraday quote catalog is
        # (canonical ticker, ISIN when there is no ticker). Exported because this
        # is the only scope holding both the book and the rebalance seeds, and the
        # newsletter's 1D panel weights intraday bars by exactly these numbers.
        # Kept separate from ``weights`` below on purpose: whether the target has a
        # daily HISTORY and whether it has a SESSION are different questions, and a
        # consumer of one must not inherit the other's verdict.
        policy: dict[str, float] = {}
        for h in list(self.holdings) + list(self.rebalance_seeds):
            weight = float(getattr(h, "target_portfolio", 0.0) or 0.0)
            if weight <= 0:
                continue
            key = h.ticker or h.isin
            policy[str(key)] = weight
            ph = h.price_history
            if h.isin in unavailable or ph is None or len(ph) < 2:
                missing.append(str(key))
                continue
            weights[key] = weight
            prices[key] = normalize_index(ph, drop_duplicates=True)
        ctx["target_weights"] = policy
        if not weights:
            return
        if missing:
            # Dropping a sleeve renormalizes the rest, which draws a DIFFERENT
            # portfolio under the target's name. Withhold the line instead.
            logger.warning(
                "Target-portfolio series withheld: no usable price history for %s",
                ", ".join(sorted(missing)),
            )
            return
        px = pd.concat(prices, axis=1).sort_index().ffill().dropna(how="any")
        rets = px.pct_change().dropna(how="any")
        if len(rets) < 2:
            return
        w = pd.Series(weights).reindex(px.columns).astype(float)
        nav = (1.0 + combine_returns(rets, w, "daily")).cumprod() * 100.0
        # Prepend the window-start baseline so the NAV opens at 100 on the last
        # day before the first measured return, exactly like _historical_risk.
        ctx["target_history"] = pd.concat(
            [pd.Series([100.0], index=[rets.index[0] - pd.Timedelta(days=1)]), nav]
        )

    # ------------------------------------------------------------------
    # Historical risk profile (uncapped, per-instrument full history)
    # ------------------------------------------------------------------
    def _historical_risk(self, ctx: dict) -> None:
        """Build the newsletter "Historical risk profile" section.

        Unlike the apples-to-apples Risk table (portfolio window, 5Y cap),
        this section maximises history:

          * Portfolio row — a **current-weight static backtest** (Approach
            A): daily returns are the weighted sum of each held
            instrument's daily return, weights = today's market value
            renormalized over the included set, over the COMMON window
            where every included holding has data. Holdings with under
            1 year of price history are excluded so a freshly-bought
            position cannot shrink the common window.
          * Instrument rows — each configured benchmark over its OWN full
            available history (uncapped), labelled with its span.

        Every row carries the full metric set incl. the Ulcer Index.
        """
        MIN_DAYS = 365  # exclude holdings younger than 1Y from the backtest

        def _norm(s: pd.Series) -> pd.Series:
            return normalize_index(s, drop_duplicates=True)

        def _span_label(s: pd.Series) -> str:
            if s is None or len(s) < 2:
                return "—"
            days = int((s.index[-1] - s.index[0]).days)
            yrs = days / DAYS_PER_YEAR
            if yrs >= 1.0:
                return f"{yrs:.1f}Y"
            if days >= 30:
                return f"{int(round(days / 30))}M"
            return f"{days}D"

        # α/β reference is the exact preprocessed benchmark series.
        ab_record = self._alpha_beta_benchmark(ctx)
        ab_bench = (
            ab_record.history if ab_record is not None
            else pd.Series(dtype=float)
        )

        # ---- Portfolio: current-weight static backtest (common window) ----
        unavailable = set(ctx.get("_order_history_unavailable", []))
        candidates = []  # (ticker, weight_value, daily_returns Series, span_days)
        for h in self.holdings:
            # A holding whose exact order mechanics are missing or conflicting
            # cannot participate in any history-derived portfolio number. Keep
            # independently valid holdings available, but never let the
            # unavailable instrument's price series leak back into this
            # separate historical-risk reconstruction.
            if h.isin in unavailable:
                continue
            ph = h.price_history
            if ph is None or len(ph) < 2:
                continue
            value = float(h.current_value if h.current_value is not None
                          else (h.market_value_eur or 0.0))
            if value <= 0:
                continue
            s = _norm(ph)
            span_days = int((s.index[-1] - s.index[0]).days)
            candidates.append((h.ticker, value, s, span_days))

        portfolio_block = None
        if candidates:
            eligible = [c for c in candidates if c[3] >= MIN_DAYS]
            used = eligible if eligible else candidates
            n_excluded = len(candidates) - len(used)
            total_w = sum(v for (_, v, _, _) in used) or 1.0
            # Align on the common window (inner join → every day has all series).
            ret_df = pd.concat(
                {tk: s.pct_change() for (tk, _v, s, _sp) in used}, axis=1
            ).dropna(how="any")
            if len(ret_df) >= 2:
                weights = pd.Series(
                    {tk: v / total_w for (tk, v, _s, _sp) in used}
                ).reindex(ret_df.columns).fillna(0.0)
                port_ret = ret_df.mul(weights, axis=1).sum(axis=1)
                nav = (1.0 + port_ret).cumprod() * 100.0
                # Prepend the window-start baseline so the NAV starts at 100.
                first_day = ret_df.index[0] - pd.Timedelta(days=1)
                nav = pd.concat([pd.Series([100.0], index=[first_day]), nav])
                metrics = _compute_single_benchmark_metrics(nav, ab_bench, self._rf_daily(ctx))
                notes = []
                if unavailable:
                    notes.append(
                        f"{len(unavailable)} holding(s) with unavailable order "
                        "history excluded from the backtest."
                    )
                if n_excluded > 0:
                    notes.append(
                        f"{n_excluded} holding(s) with under 1Y of history "
                        "excluded from the backtest; weights renormalized "
                        "over the rest. Before the common start date not all "
                        "instruments existed."
                    )
                note = " ".join(notes) or None
                portfolio_block = {
                    "label": "Your portfolio",
                    "ticker": None,
                    "span_label": _span_label(nav),
                    "note": note,
                    "metrics": metrics,
                    "is_portfolio": True,
                }

        # ---- Instruments: each benchmark over its OWN full history ----
        instrument_rows = []
        # Carry asset_class and role from the taxonomy so the newsletter can
        # group the risk profile the same way as the Performance table.
        from tarzan import config as _cfg_mod
        _taxonomy = _cfg_mod.instrument_taxonomy()
        catalog: dict[str, ResolvedBenchmark] = ctx.get("_benchmark_catalog", {})
        for record in catalog.values():
            bench = record.history
            if bench.empty or len(bench) < 2:
                continue
            bench = _norm(bench)
            metrics = _compute_single_benchmark_metrics(bench, ab_bench, self._rf_daily(ctx))
            # Taxonomy lookup is classification-only; the operational ticker
            # retained in the row is always the full preprocessed symbol.
            _meta = _taxonomy.get(normalize_ticker(record.ticker))
            if not _meta:
                _meta = _taxonomy.get(record.ticker.upper())
            if isinstance(_meta, tuple) and len(_meta) >= 2:
                _ac, _role = _meta[0], _meta[1]
            else:
                _ac, _role = "", ""
            instrument_rows.append({
                "label": record.name,
                "ticker": record.ticker,
                "span_label": _span_label(bench),
                "note": None,
                "metrics": metrics,
                "is_portfolio": False,
                "asset_class": _ac,
                "role": _role,
            })

        ctx["historical_risk"] = {
            "available": bool(portfolio_block or instrument_rows),
            "portfolio": portfolio_block,
            "instruments": instrument_rows,
        }

    # ------------------------------------------------------------------
    # Build final PortfolioMetrics
    # ------------------------------------------------------------------
    def _build_result(self, ctx: dict) -> PortfolioMetrics:
        cash_target = float(self.config.target_cash_buffer_eur) if self.config else 0.0
        assessment = self.valuation_assessment
        computed_total = ctx.get("total_value", 0.0)
        from tarzan.models.ticker_resolution import build_ticker_resolution_records

        enriched_by_isin = ctx.get("_enriched_by_isin", {}) or {}
        historical_only = [
            holding
            for holding in enriched_by_isin.values()
            if holding.is_historical_only
        ]
        effective_orders = ctx.get("_effective_orders", ()) or ()
        historical_isins = {
            order.isin
            for order in effective_orders
            if order.isin and order.is_position_change()
        }
        ticker_resolutions = build_ticker_resolution_records(
            [
                *self.holdings,
                *self.rebalance_seeds,
                *historical_only,
            ],
            historical_isins=historical_isins,
        )
        return PortfolioMetrics(
            total_value=computed_total,
            valuation_availability=(
                assessment.availability.value if assessment is not None else "AVAILABLE"
            ),
            trustworthy_total_value_eur=(
                assessment.trustworthy_total_eur
                if assessment is not None else computed_total
            ),
            known_valuation_subtotal_eur=(
                assessment.known_subtotal_eur
                if assessment is not None else computed_total
            ),
            valuation_evidence=(
                assessment.evidence if assessment is not None else ()
            ),
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
            rebalancing_plans=ctx.get("rebalancing_plans"),
            benchmark_comparison=ctx.get("benchmark_comparison", pd.DataFrame()),
            history_availability=ctx.get("history_availability", "AVAILABLE"),
            history_unavailable_instruments=tuple(
                ctx.get("history_unavailable_instruments", ())
            ),
            portfolio_history=(
                None
                if ctx.get("history_availability") == "UNAVAILABLE"
                else ctx.get("portfolio_history")
            ),
            benchmark_histories=ctx.get("benchmark_histories", {}),
            benchmark_tickers=ctx.get("benchmark_tickers", {}),
            benchmark_resolution_errors=tuple(
                ctx.get("benchmark_resolution_errors", ())
            ),
            holding_performance=ctx.get("holding_performance", pd.DataFrame()),
            intraday_requested_tickers=tuple(
                ctx.get("intraday_requested_tickers", ())
            ),
            intraday_quotes=ctx.get("intraday_quotes", {}),
            holding_histories=ctx.get("holding_histories", {}),
            target_history=ctx.get("target_history"),
            target_weights=ctx.get("target_weights") or {},
            ticker_resolutions=ticker_resolutions,
            historical_risk=ctx.get("historical_risk"),
            acwi_geo=ctx.get("acwi_geo", {}),
            excluded_short_tenure=ctx.get("excluded_short_tenure", []),
            xirr_pct=ctx.get("xirr_pct"),
            twror_pct=ctx.get("twror_pct"),
            twror_annualized_pct=ctx.get("twror_annualized_pct"),
            returns_coverage_pct=ctx.get("returns_coverage_pct"),
            returns_provenance=ctx.get("returns_provenance"),
            returns_period_debug=ctx.get("returns_period_debug"),
            pnl_eur=ctx.get("pnl_eur"),
            pnl_pct=ctx.get("pnl_pct"),
            invested_capital_eur=ctx.get("invested_capital_eur"),
            estimated_cgt_eur=ctx.get("estimated_cgt_eur"),
            pnl_eur_net_tax=ctx.get("pnl_eur_net_tax"),
            pnl_pct_net_tax=ctx.get("pnl_pct_net_tax"),
            xirr_net_tax_pct=ctx.get("xirr_net_tax_pct"),
            actual_value_series=ctx.get("actual_value_series"),
            pnl_series=ctx.get("pnl_series"),
            unrealized_series=ctx.get("unrealized_series"),
            external_flows=ctx.get("external_flows"),
            inception_date=ctx.get("inception_date"),
            allocation_timeline=ctx.get("allocation_timeline"),
            degraded_computers=ctx.get("_degraded", []),
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
    """Geographic distribution of the equity sleeve.

    Uses each holding's NOTIONAL equity exposure (``class_breakdown`` Equities
    %), so a capital-efficient fund like NTSG (90% equity) contributes 90% of
    its value — and a 60/40 multi-asset fund contributes its 60% equity leg —
    consistent with the notional asset-class table and the timeline. Falls
    back to 100% for a plain equity holding with no breakdown."""
    if df.empty:
        return pd.DataFrame(columns=["category", "weight_pct"])
    geo_lookup: dict[str, dict] = {}
    eq_frac: dict[str, float] = {}
    if holdings:
        for h in holdings:
            if not h.ticker:
                continue
            if h.geo_breakdown:
                geo_lookup[h.ticker] = h.geo_breakdown
            bd = h.class_breakdown or ({h.asset_class: 100.0} if h.asset_class else {})
            for k, v in bd.items():
                kv = k.value if hasattr(k, "value") else str(k)
                if kv == "Equities":
                    eq_frac[h.ticker] = float(v) / 100.0
    # Distribute each holding's equity notional across its (per-holding
    # normalised) geo breakdown and sum — the shared aggregation primitive.
    # Policy here (distinct from the backtest's): keep the scraped buckets as-is
    # and fall back to the row's single ``geography`` when no breakdown exists.
    from tarzan.engine import allocations as alloc
    pairs = []
    for _, row in df.iterrows():
        ticker = row.get("ticker", "")
        weight = float(row.get("weight_pct", 0.0) or 0.0)
        frac = eq_frac.get(ticker)
        if frac is None:
            frac = 1.0 if row.get("asset_class") == "Equities" else 0.0
        eq_weight = weight * frac
        if eq_weight <= 0:
            continue
        breakdown = geo_lookup.get(ticker)
        if breakdown:
            norm = alloc.renorm({(g.value if hasattr(g, "value") else str(g)): p
                                 for g, p in breakdown.items()})
        else:
            norm = {row.get("geography", "USA"): 100.0}
        pairs.append((eq_weight, norm))
    geo_weights = alloc.renorm(alloc.accumulate(pairs))
    return pd.DataFrame(
        [{"category": k, "weight_pct": v} for k, v in geo_weights.items()],
        columns=["category", "weight_pct"],
    )
