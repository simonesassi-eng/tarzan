"""Build the order-derived historical value series for returns.

This module is the only place that knows about *orders + price history
together*. It turns a list of ``Order`` into:

  * a list of synthetic ``Holding`` objects (net quantity per ISIN, with
    cum/ex BTP netting) ready for the standard enrichment pipeline;
  * a dated portfolio-value series ``V(day)`` built from the real held
    quantity on each day, valued through the shared ``value_position``
    primitive so it agrees with the current-value path;
  * the external cash-flow series and the XIRR cash-flow list;
  * an explicit provenance record of which price source priced each
    instrument (yfinance / synthetic / carry_flat / excluded), so the
    coverage of the returns can be disclosed rather than hidden.

Design: Option Y — when an order list is present it is the single source
of truth for the historical series, and every history-dependent metric
is computed on it.

The fallback ladder for a missing yfinance history (the user's choice
"(a) synthetic interpolation, made explicit when it happens"):

    1. yfinance     real daily series                       → "yfinance"
    2. synthetic    linear interpolation between order       → "synthetic"
                    price_native observations (≥2 points)
    3. carry_flat   hold the single known price flat          → "carry_flat"
    4. excluded     no price at all → drops out of valuation  → "excluded"
"""

from __future__ import annotations

import bisect
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from tarzan.data.bond_fetcher import (
    is_bond,
    looks_like_bond_from_orders,
    value_position,
)
from tarzan.models.holding import AssetClass, Holding
from tarzan.models.order import Order, OrderType

logger = logging.getLogger(__name__)

# Two ISINs whose first 9 characters match are treated as cum/ex variants
# of the same security (Italian retail BTPs). When their net quantities
# cancel, the pair is closed and contributes nothing.
_CUM_EX_PREFIX_LEN = 9
_QTY_EPS = 0.01


@dataclass
class OrderDerivedSeries:
    """Everything the returns computer needs from the order list.

    Attributes:
        valuations: ``(date, V_after)`` pairs in chronological order.
        external_flows: external inflow into the portfolio per date
            (deposits/buys positive, withdrawals/sells negative).
        xirr_cashflows: ``(date, amount)`` from the bank-account
            perspective (deposits negative, distributions positive),
            terminated by today's portfolio value.
        coverage_pct: % of latest portfolio value priced by real market
            data (rung 1) over the window.
        provenance: ``{source_tag: [isin, ...]}`` for disclosure.
        span_days: calendar days from the first flow to today.
        daily_series: dense daily-indexed portfolio value over the whole
            window, valued at market on every calendar day. This is the
            series risk metrics (volatility, Sharpe, VaR, beta) must use —
            the sparse ``valuations`` (trade dates only) would make
            ``pct_change`` span arbitrary multi-day gaps.
        actual_value_series: dense daily-indexed *raw* portfolio value
            (with the deposit/withdrawal jumps left in). Unlike
            ``daily_series`` (a flow-adjusted NAV index) this is the real
            euro worth of the whole patrimony over time, so it is what the
            newsletter's mountain chart plots.
    """

    valuations: list[tuple[datetime.date, float]]
    external_flows: dict[datetime.date, float]
    xirr_cashflows: list[tuple[datetime.date, float]]
    coverage_pct: float
    provenance: dict[str, list[str]]
    span_days: int
    daily_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    actual_value_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    pnl_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Daily unrealized P&L = market value of open positions − their cost basis.
    # Consistent with the hero's snapshot (total_value − cost_basis), but as a
    # full daily series for charting.
    unrealized_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ---------------------------------------------------------------------------
# Holdings derivation (net quantity per ISIN + cum/ex netting)
# ---------------------------------------------------------------------------

def _net_qty_by_isin(orders: list[Order]) -> dict[str, float]:
    qty: dict[str, float] = {}
    for o in orders:
        if o.is_position_change():
            qty[o.isin] = qty.get(o.isin, 0.0) + o.quantity
    return qty


def _open_isins(qty_by_isin: dict[str, float]) -> set[str]:
    """Return the ISINs that represent open positions, applying cum/ex
    prefix netting: when all variants sharing a 9-char prefix net to
    zero, the whole group is closed (Property 5)."""
    prefix_totals: dict[str, float] = {}
    for isin, q in qty_by_isin.items():
        prefix_totals[isin[:_CUM_EX_PREFIX_LEN]] = (
            prefix_totals.get(isin[:_CUM_EX_PREFIX_LEN], 0.0) + q
        )
    open_isins: set[str] = set()
    for isin, q in qty_by_isin.items():
        prefix = isin[:_CUM_EX_PREFIX_LEN]
        if abs(prefix_totals[prefix]) < _QTY_EPS:
            continue  # cum/ex pair fully nets out → closed
        if abs(q) < _QTY_EPS:
            continue  # individually closed
        open_isins.add(isin)
    return open_isins


def _order_bond_flag(orders: list[Order], isin: str) -> bool:
    """Bond detection from order data alone (for ISINs that never reach
    yfinance), via the shared order heuristic."""
    sub = [o for o in orders if o.isin == isin and o.price_native is not None]
    if not sub:
        return False
    avg_price = sum(o.price_native for o in sub) / len(sub)
    avg_qty = sum(abs(o.quantity) for o in sub) / len(sub) if sub else 0.0
    return looks_like_bond_from_orders(avg_price, avg_qty)


def _seed_market_value(orders: list[Order], isin: str, qty: float) -> float:
    """Rough current-value seed from the latest order price, scaled by
    the bond convention via ``value_position``. Replaced by a real quote
    during enrichment whenever one is available; only a floor so an
    unpriceable holding does not collapse to zero (Req 8.3)."""
    priced = sorted(
        (o for o in orders if o.isin == isin and o.price_native is not None),
        key=lambda o: o.trade_date,
    )
    if not priced or qty == 0:
        return 0.0
    last = priced[-1]
    fx = last.fx_rate or 1.0
    bond = _order_bond_flag(orders, isin)
    return value_position(abs(qty), last.price_native, bond=bond) / (fx if fx > 0 else 1.0)


def build_holdings_from_orders(orders: list[Order]) -> list[Holding]:
    """Aggregate orders into synthetic Holdings for the open positions.

    Net quantity per ISIN, cum/ex prefix netting, a seeded market value,
    and the average-cost basis of the units still held. The returned
    Holdings carry only what the enricher needs; enrichment fills in price
    history, asset class, etc.
    """
    qty_by_isin = _net_qty_by_isin(orders)
    open_isins = _open_isins(qty_by_isin)
    cost_by_isin = cost_basis_by_isin(orders)

    name_by_isin: dict[str, str] = {}
    ccy_by_isin: dict[str, str] = {}
    for o in orders:
        if o.name and o.isin not in name_by_isin:
            name_by_isin[o.isin] = o.name
        if o.currency and o.isin not in ccy_by_isin:
            ccy_by_isin[o.isin] = o.currency

    holdings: list[Holding] = []
    for isin in open_isins:
        qty = qty_by_isin[isin]
        holdings.append(Holding(
            isin=isin,
            ticker=isin,
            quantity=qty,
            cost_basis_eur=cost_by_isin.get(isin, 0.0),
            market_value_eur=_seed_market_value(orders, isin, qty),
            currency=ccy_by_isin.get(isin, "EUR"),
            name=name_by_isin.get(isin, ""),
        ))
    return holdings


def cost_basis_by_isin(orders: list[Order]) -> dict[str, float]:
    """Average-cost basis (EUR) of the *currently held* units per ISIN.

    Walks the position-changing orders in date order keeping a running
    ``(quantity, cost)`` pair per ISIN:

      * a buy / transfer-in adds the EUR it committed — the net cash paid
        including fees (``net_eur``), or the gross transferred value when
        no cash moved (a ``transfer_in`` has ``net_eur == 0``);
      * a sell / transfer-out removes cost at the running *average* price,
        so realized gains/losses do not distort the basis of the units
        that remain;
      * coupons and dividends never touch cost basis (they are income, not
        a return of capital).

    The result is the acquisition cost of the units still open today — the
    denominator the snapshot uses for per-holding unrealized P&L. Derived
    purely from the order list, so the holdings-only and order-only paths
    agree without needing a ``cost_basis_eur`` column in any CSV.
    """
    pos = sorted(
        (o for o in orders if o.is_position_change()),
        key=lambda o: o.trade_date,
    )
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    for o in pos:
        q = qty.get(o.isin, 0.0)
        c = cost.get(o.isin, 0.0)
        if o.quantity > 0:  # buy / transfer_in
            committed = abs(o.net_eur) if o.net_eur else abs(o.gross_eur or 0.0)
            qty[o.isin] = q + o.quantity
            cost[o.isin] = c + committed
        elif o.quantity < 0:  # sell / transfer_out
            sold = abs(o.quantity)
            if q > _QTY_EPS:
                avg = c / q
                cost[o.isin] = max(c - avg * min(sold, q), 0.0)
            qty[o.isin] = max(q - sold, 0.0)
    return cost


# ---------------------------------------------------------------------------
# Quantity timeline (binary-searchable cumulative quantity per ISIN)
# ---------------------------------------------------------------------------

class QuantityTimeline:
    """Cumulative held quantity per ISIN as of end-of-day, with O(log n)
    lookup. Built once from the position-changing orders."""

    def __init__(self, orders: list[Order]):
        events: list[tuple[datetime.date, str, float]] = [
            (o.trade_date, o.isin, o.quantity)
            for o in orders if o.is_position_change()
        ]
        events.sort(key=lambda e: e[0])
        self._cum: dict[str, list[tuple[datetime.date, float]]] = {}
        running: dict[str, float] = {}
        for d, isin, delta in events:
            running[isin] = running.get(isin, 0.0) + delta
            self._cum.setdefault(isin, []).append((d, running[isin]))

    def isins(self) -> list[str]:
        return list(self._cum.keys())

    def qty_at(self, isin: str, d: datetime.date) -> float:
        series = self._cum.get(isin)
        if not series or d < series[0][0]:
            return 0.0
        dates = [e[0] for e in series]
        # rightmost index whose date <= d
        i = bisect.bisect_right(dates, d) - 1
        return series[i][1] if i >= 0 else 0.0


# ---------------------------------------------------------------------------
# Price lookup with explicit fallback ladder
# ---------------------------------------------------------------------------

def _price_at(price_history: pd.Series, d: datetime.date) -> Optional[float]:
    """Last observed price at or before ``d`` in a tz-aware-safe way."""
    if price_history is None or len(price_history) == 0:
        return None
    idx_tz = getattr(price_history.index, "tz", None)
    threshold = pd.Timestamp(d)
    if idx_tz is not None:
        threshold = threshold.tz_localize(idx_tz)
    avail = price_history.loc[price_history.index <= threshold]
    if avail.empty:
        return None
    return float(avail.iloc[-1])


def _build_synthetic_history(orders: list[Order], isin: str) -> Optional[pd.Series]:
    """Daily-indexed series of order prices for an ISIN, converted to
    EUR and mean-aggregated per day. None if no observation.

    Order ``price_native`` is in the instrument's trade currency and
    ``fx_rate`` is Fineco's ``Cambio`` — units of native currency per
    EUR — so the EUR price is ``price_native / fx_rate``. Converting
    here means the synthetic/carry-flat rungs return EUR-per-unit
    prices, consistent with the yfinance rung (which the enricher has
    already converted to EUR). Without this a ZAR- or USD-denominated
    bond would be valued in its native currency and overstated by the
    FX rate.
    """
    obs = []
    for o in orders:
        if o.isin == isin and o.price_native is not None:
            fx = o.fx_rate or 1.0
            eur_price = o.price_native / fx if fx > 0 else o.price_native
            obs.append((o.trade_date, eur_price))
    if not obs:
        return None
    s = pd.Series(
        [p for _, p in obs],
        index=pd.to_datetime([d for d, _ in obs]),
    ).sort_index()
    return s.groupby(s.index).mean()


def _interp_synthetic(s: pd.Series, d: datetime.date) -> tuple[Optional[float], str]:
    """Linear interpolation between order price points, or carry-flat
    when only one point / outside the range. Returns (price, source)."""
    ts = pd.Timestamp(d)
    if len(s) == 1:
        return float(s.iloc[0]), "carry_flat"
    if ts <= s.index[0]:
        return float(s.iloc[0]), "carry_flat"
    if ts >= s.index[-1]:
        return float(s.iloc[-1]), "carry_flat"
    before = s.loc[s.index <= ts]
    after = s.loc[s.index >= ts]
    bd, bv = before.index[-1], float(before.iloc[-1])
    ad, av = after.index[0], float(after.iloc[0])
    if ad == bd:
        return bv, "synthetic"
    w = (ts - bd).days / (ad - bd).days
    return bv + w * (av - bv), "synthetic"


class PriceResolver:
    """Resolve an ISIN's price on any date via the explicit fallback
    ladder, recording the source tag used per ISIN for provenance."""

    def __init__(
        self,
        orders: list[Order],
        enriched_by_isin: dict[str, Holding],
        today: Optional[datetime.date] = None,
    ):
        self._orders = orders
        self._enriched = enriched_by_isin
        self._today = today
        self._synth: dict[str, Optional[pd.Series]] = {}
        self._is_bond: dict[str, bool] = {}

    def _synthetic(self, isin: str) -> Optional[pd.Series]:
        if isin not in self._synth:
            self._synth[isin] = _build_synthetic_history(self._orders, isin)
        return self._synth[isin]

    def is_bond(self, isin: str) -> bool:
        if isin not in self._is_bond:
            h = self._enriched.get(isin)
            flag = is_bond(
                asset_class=getattr(h, "asset_class", None) if h else None,
                instrument_type=getattr(h, "instrument_type", None) if h else None,
            )
            if not flag:
                flag = _order_bond_flag(self._orders, isin)
            self._is_bond[isin] = flag
        return self._is_bond[isin]

    def _borsa_price(self, isin: str) -> Optional[float]:
        """Borsa Italiana today-price for a bond as EUR-per-unit, if the
        enricher set one, else None.

        yfinance does not quote BTPs / US Treasuries / foreign-currency
        notes, so for those the enricher's ``_try_terrapin_fallback``
        scrapes Borsa Italiana, converts the native clean price to EUR via
        the shared FX machinery, and stamps the holding with an
        EUR-per-unit ``current_price`` and a ``data_source`` like
        ``"borsa_italiana/mot/btp"``. We trust ``current_price`` as a
        market quote only when that source tag says so (a yfinance-derived
        price would already be reachable via ``price_history``; a stale CSV
        seed must not be mistaken for a live quote).
        """
        h = self._enriched.get(isin)
        if h is None:
            return None
        src = getattr(h, "data_source", None)
        if not src or not str(src).startswith("borsa_italiana"):
            return None
        price = getattr(h, "current_price", None)
        if price is None or not self.is_bond(isin):
            return None
        return float(price)

    def price_on(self, isin: str, d: datetime.date) -> tuple[Optional[float], str]:
        """Return (price_eur_per_unit, source) for an ISIN on a date.

        Note: prices from the enricher are already EUR-per-unit and, for
        bonds, already FX-converted and rescaled by /100 — so the caller
        must NOT apply the bond /100 again on the 'yfinance' or
        'borsa_italiana' rungs. Synthetic/carry_flat prices are raw order
        prices (already FX-converted to EUR) that still need the /100 via
        value_position. The returned source disambiguates which.
        """
        h = self._enriched.get(isin)
        ph = getattr(h, "price_history", None) if h else None
        if ph is not None and len(ph) > 0:
            price = _price_at(ph, d)
            if price is not None:
                return price, "yfinance"
        # Borsa Italiana single-point rung: only for the TODAY/terminal
        # valuation of a bond with no yfinance history. The scrape gives
        # only today's price (no series), so historical dates still fall
        # through to synthetic/carry_flat. The price is already EUR-per-unit
        # (FX-converted, post /100), so it is tagged like the yfinance rung
        # and the caller must NOT apply value_position.
        if self._today is not None and d >= self._today:
            borsa = self._borsa_price(isin)
            if borsa is not None:
                return borsa, "borsa_italiana"
        s = self._synthetic(isin)
        if s is not None and not s.empty:
            return _interp_synthetic(s, d)
        return None, "excluded"


# ---------------------------------------------------------------------------
# Main: build the dated value series + cash flows + provenance
# ---------------------------------------------------------------------------

def _closed_cum_ex_prefixes(
    timeline: "QuantityTimeline", d: datetime.date
) -> set[str]:
    """Prefixes whose cum/ex group nets to ~0 held quantity as of ``d``.

    Mirrors ``_open_isins`` but as-of a date: Italian retail BTPs are
    reclassified cum→ex coupon, which appears in the order list as a sell
    of one ISIN and a transfer-in of a sibling sharing the 9-char prefix.
    When the group's net quantity is ~0 on ``d`` the position is closed,
    so it must contribute nothing to that day's valuation — even though
    each leg individually still shows a non-zero quantity. Pricing the
    legs separately (often by different carry-flat prices) would otherwise
    leave a spurious residual and desync the historical valuation from the
    order-derived snapshot.
    """
    prefix_totals: dict[str, float] = {}
    for isin in timeline.isins():
        prefix = isin[:_CUM_EX_PREFIX_LEN]
        prefix_totals[prefix] = prefix_totals.get(prefix, 0.0) + timeline.qty_at(isin, d)
    return {p for p, t in prefix_totals.items() if abs(t) < _QTY_EPS}


def build_order_derived_series(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    today: Optional[datetime.date] = None,
) -> OrderDerivedSeries:
    """Build the order-derived valuation series, cash flows and
    provenance. ``enriched_by_isin`` maps ISIN → enriched Holding (from
    running the standard enrichment on ``build_holdings_from_orders``).

    All date-keyed logic (quantity timeline, cash flows, cost basis,
    synthetic prices, inception) keys on each order's ``trade_date`` —
    the date market exposure is taken on — not its settlement ``date``.
    This keeps the cash side and the asset side on the same clock, so a
    trade that settles after the run date (T+2) cannot make the cash
    flow land while the position it creates is still invisible.
    """
    # Defensive anchor for the live "value now" path (no explicit
    # ``today``): should a trade ever be dated after the run date, value
    # as of that date so the terminal valuation still covers every order
    # the cash flows count. With trade-date keying this is a safety net,
    # not the primary fix. An explicit ``today`` (historical/backtest
    # as-of valuation) is always respected verbatim.
    if today is None:
        today = datetime.datetime.now().date()
        last_trade_date = max((o.trade_date for o in orders), default=today)
        if last_trade_date > today:
            today = last_trade_date
    timeline = QuantityTimeline(orders)
    resolver = PriceResolver(orders, enriched_by_isin, today=today)

    # Track which source priced each open ISIN at its latest valuation,
    # for the coverage/provenance disclosure.
    provenance: dict[str, list[str]] = {
        "yfinance": [], "borsa_italiana": [], "synthetic": [],
        "carry_flat": [], "excluded": [],
    }

    def value_isin_on(isin: str, d: datetime.date) -> Optional[float]:
        """EUR value of one unit of ``isin`` on ``d`` at market price
        (None if unpriceable). Used to value quantity deltas for the
        TWROR external flow at the same price basis as the series."""
        price, source = resolver.price_on(isin, d)
        if price is None:
            return None
        # 'yfinance' and 'borsa_italiana' prices are already EUR-per-unit
        # (bonds FX-converted and pre-/100 by the enricher); raw synthetic
        # prices are not.
        if source in ("yfinance", "borsa_italiana"):
            return price
        return value_position(1.0, price, bond=resolver.is_bond(isin))

    def value_on(d: datetime.date, record_source: bool = False) -> float:
        """Total EUR portfolio value on day ``d``.

        Values *every* ISIN that had a non-zero held quantity on ``d``,
        not only the ISINs still open today — otherwise a position opened
        and fully closed inside the window would contribute nothing to the
        historical series and its holding-period market move would be
        invisible to TWROR. The cum/ex ``open_isins`` gate is only used
        for the "what is open now" coverage snapshot, not for history.

        Cum/ex pairs that net to ~0 quantity as of ``d`` are the one
        exception: they are a single bond reclassified across coupon
        events, so the group is treated as closed (contributes 0),
        consistent with the order-derived snapshot. Valuing each leg at
        its own carry-flat price would otherwise leave a spurious residual.
        """
        total = 0.0
        closed = _closed_cum_ex_prefixes(timeline, d)
        for isin in timeline.isins():
            qty = timeline.qty_at(isin, d)
            if abs(qty) < _QTY_EPS:
                continue
            if isin[:_CUM_EX_PREFIX_LEN] in closed:
                continue  # cum/ex group nets flat → closed, contributes 0
            price, source = resolver.price_on(isin, d)
            if record_source:
                provenance[source].append(isin)
            if price is None:
                continue
            # The enricher already applied /100 to bond price_history, so
            # the 'yfinance' rung is EUR-per-unit; the Borsa Italiana
            # today-price is likewise FX-converted and pre-/100. Raw
            # synthetic prices still need the /100 via value_position.
            if source in ("yfinance", "borsa_italiana"):
                total += qty * price
            else:
                total += value_position(qty, price, bond=resolver.is_bond(isin))
        return total

    # TWROR external flow per date, valued at MARKET price (same basis as
    # the valuation series), not at execution price. For each
    # position-changing order we value its quantity delta at that day's
    # market price; this makes V_before(d) = V_after(d) - flow(d) use one
    # consistent price basis, so a trade does not inject a fictitious
    # jump from the gap between execution and market price (Option 2).
    # Round-trip positions (closed before today) are included: their
    # buys/sells are real external flows over their holding window.
    #
    # Income (coupons/dividends) is treated the GIPS total-return way.
    # Our series values only the securities, and the income cash is
    # credited to the bank account (net_eur > 0), so from the securities
    # portfolio's perspective the distribution is a *withdrawal*: a
    # negative external flow of -net_eur. Because TWROR computes
    # V_before(d) = V_after(d) - external_flow(d), that withdrawal is
    # added back into the pre-flow value, so the income is captured as
    # return rather than vanishing. (It is the mirror image of XIRR,
    # where the same distribution is a positive bank cash inflow.)
    external_flows: dict[datetime.date, float] = {}
    for o in orders:
        if o.is_position_change():
            unit = value_isin_on(o.isin, o.trade_date)
            if unit is None:
                continue
            external_flows[o.trade_date] = external_flows.get(o.trade_date, 0.0) + o.quantity * unit
        elif o.type in (OrderType.COUPON, OrderType.DIVIDEND) and o.net_eur:
            # Distribution paid out of the securities portfolio → withdrawal.
            external_flows[o.trade_date] = external_flows.get(o.trade_date, 0.0) - o.net_eur

    cf_dates = sorted(external_flows.keys())
    valuations: list[tuple[datetime.date, float]] = [
        (d, value_on(d)) for d in cf_dates
    ]
    # Terminal valuation today, recording provenance for coverage.
    current_value = value_on(today, record_source=True)
    valuations.append((today, current_value))

    # Dense daily value series for risk metrics (volatility, Sharpe, VaR,
    # beta). pct_change on the sparse trade-date `valuations` would treat
    # arbitrary multi-day gaps as single trading days and badly distort
    # annualized risk; the daily series fixes that at the root. The same
    # pass also yields the raw actual-value series (jumps kept in) for the
    # newsletter mountain chart.
    daily_series, actual_value_series = _build_daily_series(
        timeline, resolver, external_flows, cf_dates, today
    )

    # Coverage: share of today's value priced by real market data. Use the
    # SAME value_position basis as value_on so bonds (priced /100) are
    # Coverage: share of today's value priced by real market data. Borsa
    # Italiana is a real market quote (the best available for BTPs / US
    # Treasuries / foreign-currency notes that yfinance does not cover), so
    # it counts alongside yfinance. Both are EUR-per-unit (no value_position)
    # so the ratio cannot exceed 100%.
    real_value = 0.0
    real_isins = set(provenance["yfinance"]) | set(provenance["borsa_italiana"])
    for isin in real_isins:
        qty = timeline.qty_at(isin, today)
        price, source = resolver.price_on(isin, today)
        if price is not None and source in ("yfinance", "borsa_italiana"):
            real_value += qty * price
    coverage_pct = (real_value / current_value * 100.0) if current_value > 0 else 0.0
    # Defensive clamp: coverage is a share of value priced by real market
    # data and is meaningless above 100%. Cum/ex netting keeps the
    # numerator and denominator consistent, but clamp anyway so a future
    # pricing edge case can never surface an impossible >100% figure.
    coverage_pct = max(0.0, min(coverage_pct, 100.0))

    # Disclosure: warn once per instrument that fell back.
    for source in ("synthetic", "carry_flat", "excluded"):
        for isin in sorted(set(provenance[source])):
            logger.warning(
                "TWROR/TWR: %s priced by %s (no full market history).",
                isin, source.upper(),
            )
    # Deduplicate provenance lists.
    provenance = {k: sorted(set(v)) for k, v in provenance.items()}

    # XIRR cash flows (bank-account perspective): transfer_in is a
    # deposit at its market value; others use net_eur. Dated on the
    # trade date (when market exposure is taken on), consistent with
    # every other metric. Terminated by today's value.
    xirr_cashflows: list[tuple[datetime.date, float]] = []
    for o in orders:
        if o.type == OrderType.TRANSFER_IN:
            if (o.gross_eur or 0.0) > 0:
                xirr_cashflows.append((o.trade_date, -(o.gross_eur)))
        elif o.net_eur != 0.0:
            xirr_cashflows.append((o.trade_date, o.net_eur))
    xirr_cashflows.append((today, current_value))

    span_days = (today - cf_dates[0]).days if cf_dates else 0

    # Daily cumulative P&L (realized + unrealized), net of contributed
    # capital: actual value + cumulative bank cash flows (deposits negative,
    # distributions positive). At today it equals the lifetime pnl_eur. The
    # newsletter uses it to show the real money gained over a window, net of
    # the contributions made *inside* that window.
    pnl_series = _build_pnl_series(actual_value_series, xirr_cashflows)

    # Daily unrealized P&L = market value of open positions − their cost
    # basis on each day. Same average-cost logic as ``cost_basis_by_isin``
    # (so today's value reconciles with the hero's snapshot), expressed as a
    # full daily series for charting.
    cost_basis_series = _build_cost_basis_series(orders, actual_value_series.index)
    if actual_value_series is not None and not actual_value_series.empty:
        unrealized_series = actual_value_series - cost_basis_series
    else:
        unrealized_series = pd.Series(dtype=float)

    return OrderDerivedSeries(
        valuations=valuations,
        external_flows=external_flows,
        xirr_cashflows=xirr_cashflows,
        coverage_pct=coverage_pct,
        provenance=provenance,
        span_days=span_days,
        daily_series=daily_series,
        actual_value_series=actual_value_series,
        pnl_series=pnl_series,
        unrealized_series=unrealized_series,
    )


def _build_cost_basis_series(orders: list[Order], index: pd.Index) -> pd.Series:
    """Daily total cost basis of the OPEN positions, reindexed onto ``index``.

    Average-cost walk identical to ``cost_basis_by_isin`` (buys/transfers add
    the committed EUR; sells remove cost at the running average), but it
    records the running total after each trade date so the result is a step
    function forward-filled across calendar days. Subtracting it from the
    daily market value yields the unrealized P&L series.
    """
    if index is None or len(index) == 0:
        return pd.Series(dtype=float)
    pos = sorted(
        (o for o in orders if o.is_position_change()),
        key=lambda o: o.trade_date,
    )
    qty: dict[str, float] = {}
    cost: dict[str, float] = {}
    by_date: dict[pd.Timestamp, float] = {}
    for o in pos:
        q = qty.get(o.isin, 0.0)
        c = cost.get(o.isin, 0.0)
        if o.quantity > 0:  # buy / transfer_in
            committed = abs(o.net_eur) if o.net_eur else abs(o.gross_eur or 0.0)
            qty[o.isin] = q + o.quantity
            cost[o.isin] = c + committed
        elif o.quantity < 0:  # sell / transfer_out
            sold = abs(o.quantity)
            if q > _QTY_EPS:
                avg = c / q
                cost[o.isin] = max(c - avg * min(sold, q), 0.0)
            qty[o.isin] = max(q - sold, 0.0)
        by_date[pd.Timestamp(o.trade_date)] = sum(cost.values())
    if not by_date:
        return pd.Series(0.0, index=index)
    s = pd.Series(by_date).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(index, method="ffill").fillna(0.0)


def _build_pnl_series(actual: pd.Series, xirr_cashflows: list) -> pd.Series:
    """Daily cumulative P&L = actual value + cumulative bank cash flows.

    The XIRR flows are deposits (negative) and distributions (positive);
    adding their running sum to the portfolio value cancels the invested
    capital and leaves the realized + unrealized gain on each day. At the
    final day it equals the lifetime ``pnl_eur``. The terminal valuation is
    excluded (it is not a real cash flow), so the series is purely
    value + contributions accounting.
    """
    if actual is None or actual.empty or not xirr_cashflows:
        return pd.Series(dtype=float)
    agg: dict = {}
    for d, amt in xirr_cashflows[:-1]:  # drop the terminal valuation
        ts = pd.Timestamp(d)
        agg[ts] = agg.get(ts, 0.0) + amt
    if not agg:
        return pd.Series(dtype=float)
    cum = pd.Series(agg).sort_index().cumsum()
    cum = cum.reindex(actual.index, method="ffill").fillna(0.0)
    return actual + cum


def _build_daily_series(
    timeline: "QuantityTimeline",
    resolver: "PriceResolver",
    external_flows: dict[datetime.date, float],
    cf_dates: list[datetime.date],
    today: datetime.date,
) -> tuple[pd.Series, pd.Series]:
    """Dense daily NAV index + raw actual-value series, first trade → today.

    Returns ``(nav_index, actual_value_series)``:

      * ``nav_index`` is flow-adjusted. Each day's raw market value is
        computed through the resolver from the held quantity that day; the
        day-over-day return strips that day's external flow (deposits/
        withdrawals valued at market), so buying more units does not
        register as a market gain::

            r_t = (V_t - flow_t) / V_{t-1} - 1

        The returns are chained into an index anchored at the first day's
        real value. Risk metrics (volatility, Sharpe, Sortino, VaR, beta)
        must use THIS series.

      * ``actual_value_series`` is the same daily raw value with the
        deposit/withdrawal jumps left in — the real euro worth of the
        patrimony over time, for the mountain chart.

    Both are anchored on the first strictly-positive value so leading
    zero-value days (before the first priced position) are dropped.
    """
    if not cf_dates:
        empty = pd.Series(dtype=float)
        return empty, empty

    days = pd.date_range(start=cf_dates[0], end=today, freq="D")
    isins = timeline.isins()

    def raw_value(d: datetime.date) -> float:
        total = 0.0
        closed = _closed_cum_ex_prefixes(timeline, d)
        for isin in isins:
            qty = timeline.qty_at(isin, d)
            if abs(qty) < _QTY_EPS:
                continue
            if isin[:_CUM_EX_PREFIX_LEN] in closed:
                continue  # cum/ex group nets flat → closed, contributes 0
            price, source = resolver.price_on(isin, d)
            if price is None:
                continue
            if source in ("yfinance", "borsa_italiana"):
                total += qty * price
            else:
                total += value_position(qty, price, bond=resolver.is_bond(isin))
        return total

    raw = [(ts.date(), raw_value(ts.date())) for ts in days]
    # Anchor on the first strictly-positive value.
    anchor_i = next((i for i, (_, v) in enumerate(raw) if v > 0), None)
    if anchor_i is None:
        empty = pd.Series(dtype=float)
        return empty, empty

    index_dates: list[pd.Timestamp] = []
    index_vals: list[float] = []
    actual_vals: list[float] = []
    nav = raw[anchor_i][1]            # start the index at the real value
    index_dates.append(pd.Timestamp(raw[anchor_i][0]))
    index_vals.append(nav)
    actual_vals.append(raw[anchor_i][1])

    prev_v = raw[anchor_i][1]
    for d, v in raw[anchor_i + 1:]:
        flow = external_flows.get(d, 0.0)
        if prev_v > 0:
            r = (v - flow) / prev_v - 1.0
            nav *= (1.0 + r)
        index_dates.append(pd.Timestamp(d))
        index_vals.append(nav)
        actual_vals.append(v)
        if v > 0:
            prev_v = v

    idx = pd.DatetimeIndex(index_dates)
    return (
        pd.Series(index_vals, index=idx),
        pd.Series(actual_vals, index=idx),
    )


# ---------------------------------------------------------------------------
# Allocation timeline (per asset-class / per equity-geography weekly weights)
# ---------------------------------------------------------------------------

# Price-source preference when collapsing a cum/ex group to a single
# representative quote: real market quotes first, then the synthetic
# ladder. Mirrors the ranking used when valuing the portfolio history.
_TIMELINE_SOURCE_RANK = {
    "yfinance": 0, "borsa_italiana": 0, "synthetic": 1, "carry_flat": 2,
}


def build_allocation_timeline(
    orders: list[Order],
    enriched_by_isin: dict[str, Holding],
    *,
    months: int = 3,
    today: Optional[datetime.date] = None,
) -> Optional[dict]:
    """Reconstruct the historical allocation mix over a recent window.

    Returns weekly snapshots of the invested asset-class mix and the
    equity-geography mix over the last ``months`` months (clamped to the
    portfolio inception), so the newsletter can draw a per-category
    sparkline of how each weight drifted toward/away from its target.

    The reconstruction reuses the same primitives as the value series —
    ``QuantityTimeline`` for as-of held quantity, ``PriceResolver`` for the
    EUR price ladder, and 9-char cum/ex prefix netting so a BTP rotated
    across coupon events nets to zero rather than lingering as a phantom
    leg. Asset class and equity geo come from the already-enriched
    holdings (constant per instrument), so this adds no network calls.

    Output (or ``None`` when there is no order history):
        ``{"dates": [date, ...],
            "asset":   [{class: pct_of_invested}, ...],
            "geo":     [{region: pct_of_equity}, ...],
            "holding": [{isin: pct_of_its_class}, ...]}``
    The lists are parallel to ``dates``; the caller typically anchors the
    final bucket to the authoritative live allocation.
    """
    if not orders:
        return None
    if today is None:
        today = datetime.datetime.now().date()

    pos_dates = [o.trade_date for o in orders if o.is_position_change()]
    if not pos_dates:
        return None
    inception = min(pos_dates)

    # Weekly (W-FRI) buckets across the window, clamped to inception for a
    # portfolio younger than the window, and always terminated by today.
    window_start = (pd.Timestamp(today) - pd.DateOffset(months=months)).date()
    start = max(window_start, inception)
    pts = list(pd.date_range(start=start, end=today, freq="W-FRI"))
    dates = [p.date() for p in pts]
    if not dates or dates[0] > start:
        dates.insert(0, start)
    if dates[-1] != today:
        dates.append(today)

    timeline = QuantityTimeline(orders)
    resolver = PriceResolver(orders, enriched_by_isin, today=today)
    cash_class = AssetClass.CASH_EQUIVALENTS.value

    # Group cum/ex ISIN variants (shared 9-char prefix) so they value as a
    # single net position, exactly like ``build_holdings_from_orders``.
    groups: dict[str, list[str]] = {}
    for isin in timeline.isins():
        groups.setdefault(isin[:_CUM_EX_PREFIX_LEN], []).append(isin)

    def _value_group(group: list[str], d: datetime.date) -> tuple[float, Optional[str]]:
        """EUR value of a cum/ex group on ``d`` plus the representative
        ISIN (best-priced leg) used to classify it."""
        net = sum(timeline.qty_at(i, d) for i in group)
        if abs(net) < _QTY_EPS:
            return 0.0, None
        best = None
        best_rank = 99
        for i in group:
            price, source = resolver.price_on(i, d)
            if price is None:
                continue
            rank = _TIMELINE_SOURCE_RANK.get(source, 3)
            if rank < best_rank:
                best_rank, best = rank, (i, price, source)
        if best is None:
            return 0.0, None
        isin, price, source = best
        if source in ("yfinance", "borsa_italiana"):
            return net * price, isin
        return value_position(net, price, bond=resolver.is_bond(isin)), isin

    asset_series: list[dict[str, float]] = []
    geo_series: list[dict[str, float]] = []
    # Per-holding weight as % of its own asset class (parallel to dates).
    # Keyed by the representative (best-priced) ISIN of each net position so
    # the newsletter can draw a per-instrument trend vs its target — the
    # target itself is % of class, so this normalization matches it.
    holding_series: list[dict[str, float]] = []
    for d in dates:
        class_val: dict[str, float] = {}
        geo_val: dict[str, float] = {}
        eq_val = 0.0
        for group in groups.values():
            v, rep = _value_group(group, d)
            if v <= 0 or rep is None:
                continue
            h = enriched_by_isin.get(rep)
            ac = h.asset_class.value if (h and h.asset_class) else AssetClass.ALTERNATIVE.value
            class_val[ac] = class_val.get(ac, 0.0) + v
            if ac == "Equities":
                eq_val += v
                if h and h.geo_breakdown:
                    tot = sum(h.geo_breakdown.values()) or 1.0
                    for g, p in h.geo_breakdown.items():
                        gn = g.value if hasattr(g, "value") else str(g)
                        geo_val[gn] = geo_val.get(gn, 0.0) + v * (p / tot)
        invested = sum(val for k, val in class_val.items() if k != cash_class)
        asset_series.append({
            k: (val / invested * 100.0)
            for k, val in class_val.items()
            if k != cash_class and invested > 0
        })
        geo_series.append({
            k: (val / eq_val * 100.0) for k, val in geo_val.items() if eq_val > 0
        })
        # Per-holding attribution is computed per individual ISIN (NOT via the
        # cum/ex group) because live holdings are per-ISIN — two distinct ETFs
        # can share a 9-char ISIN prefix, so grouping would merge them. Each
        # ISIN is normalized against the per-ISIN total of its own class so the
        # weights match the per-holding targets (which are % of class).
        iso_val: dict[str, float] = {}
        iso_ac: dict[str, str] = {}
        for isin in timeline.isins():
            vv, _rep = _value_group([isin], d)
            if vv <= 0:
                continue
            hh = enriched_by_isin.get(isin)
            iso_val[isin] = vv
            iso_ac[isin] = hh.asset_class.value if (hh and hh.asset_class) else AssetClass.ALTERNATIVE.value
        ac_tot: dict[str, float] = {}
        for isin, vv in iso_val.items():
            ac_tot[iso_ac[isin]] = ac_tot.get(iso_ac[isin], 0.0) + vv
        holding_series.append({
            isin: (vv / ac_tot[iso_ac[isin]] * 100.0)
            for isin, vv in iso_val.items()
            if ac_tot.get(iso_ac[isin], 0.0) > 0
        })

    return {"dates": dates, "asset": asset_series, "geo": geo_series,
            "holding": holding_series}
