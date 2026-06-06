"""Compute XIRR and TWROR from ``input/order_list.csv``.

This is an analytical one-off script — not part of the main pipeline.
It does not write any files; it only prints the resulting metrics.

Definitions:

  * **XIRR** (money-weighted return) — the constant annualised rate
    that makes the NPV of every external cash flow plus today's
    portfolio value sum to zero. Sensitive to *when* you deposit and
    withdraw, so it captures both market behaviour and your timing.

  * **TWROR** (time-weighted return) — chained period returns,
    neutral to deposit timing. Captures only the market behaviour of
    the portfolio you held over time, which is what you want for
    apples-to-apples vs benchmarks.

Cash-flow conventions:

  * ``buy``        → bank account loses (gross + fees) → portfolio
    receives the same value (fees become a one-shot drag).
  * ``sell``       → bank account gains (gross − fees).
  * ``coupon``     → bank account gains the coupon amount; the
    security pays it out of its market value.
  * ``transfer_in``→ no bank-account movement; the portfolio
    receives the security at its market value (Fineco's transfer
    price), treated as a deposit by the investor for return
    purposes — that valuation comes pre-computed in the order
    list's ``gross_eur`` column.

Reliability notes:

  * **XIRR** is robust: it only needs the cash flows and today's
    portfolio value. Both come from data we have under control
    (``order_list.csv`` for cash flows, the standard Tarzan
    enrichment for the current value).
  * **TWROR** is best-effort and may show large per-period anomalies
    when the price history of a holding is missing for a stretch of
    its life. We fall back to a synthetic linear interpolation
    between trade prices for ISINs with no yfinance coverage, but
    that approach can't capture market moves between trade dates.
    Treat the TWROR figure as indicative until we have full price
    history coverage for every ISIN.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.data.enricher import enrich_holdings  # noqa: E402
from tarzan.models.holding import AssetClass  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("compute_returns")


# ---------------------------------------------------------------------------
# XIRR
# ---------------------------------------------------------------------------

def _xnpv(rate: float, cashflows: list[tuple[date, float]]) -> float:
    """Net present value of ``cashflows`` at constant rate ``rate``."""
    if not cashflows:
        return 0.0
    t0 = min(d for d, _ in cashflows)
    return sum(
        amount / (1.0 + rate) ** ((d - t0).days / 365.0)
        for d, amount in cashflows
    )


def xirr(cashflows: list[tuple[date, float]]) -> float:
    """Annualised money-weighted return, by bisection.

    Returns NaN when XIRR cannot be bracketed (typically because all
    cash flows have the same sign, which means the investor never
    realised a return at all).
    """
    try:
        return brentq(
            lambda r: _xnpv(r, cashflows),
            -0.999, 10.0, xtol=1e-7,
        )
    except ValueError as exc:
        logger.error("xirr could not converge: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_at(price_history, d: date) -> float | None:
    """Return the last observed price at or before ``d`` from a price
    history series. Returns ``None`` when no observation is available
    (e.g. yfinance had no data for that ISIN, or all observations are
    after ``d``).
    """
    if price_history is None or len(price_history) == 0:
        return None
    # The price history index may be tz-aware (yfinance returns
    # market-local timestamps). Match the comparison's tz to whatever
    # the index uses so we never raise on naive-vs-aware mixes.
    idx_tz = getattr(price_history.index, "tz", None)
    threshold = pd.Timestamp(d)
    if idx_tz is not None:
        threshold = threshold.tz_localize(idx_tz)
    avail = price_history.loc[price_history.index <= threshold]
    if avail.empty:
        return None
    return float(avail.iloc[-1])


def _is_bond(h) -> bool:
    """Detect whether the bond face-value convention applies."""
    if h is None:
        return False
    if h.asset_class == AssetClass.FIXED_INCOME:
        return True
    instr = (h.instrument_type or "").upper()
    return any(k in instr for k in ("BOND", "TREASURY", "GOV", "CORP"))


def _is_bond_via_orders(orders, isin: str) -> bool:
    """Decide whether ``isin`` quotes per-100-face value, using only
    the order-list data. The same heuristic the preprocessor applies:
    average ``price_native`` between 50 and 150 with average quantity
    ≥ 1,000 means the holding is a bond and its prices are clean
    quotations per 100 EUR of face value.
    """
    sub = orders[orders["isin"] == isin]
    if sub.empty:
        return False
    prices = pd.to_numeric(sub["price_native"], errors="coerce").dropna()
    if prices.empty:
        return False
    avg_price = float(prices.mean())
    avg_qty = float(sub["quantity"].abs().mean())
    return 50.0 <= avg_price <= 150.0 and avg_qty >= 1000.0


def _transfer_value(row) -> float:
    """Return the EUR market value of a transfer_in row.

    The preprocessor already wrote ``gross_eur`` for transfer rows,
    using Fineco's ``Prezzo`` × quantity ÷ ``Cambio`` and the bond
    /100 convention where applicable. We just trust that here.
    """
    return float(row.get("gross_eur") or 0.0)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> int:
    orders_path = ROOT / "input/order_list.csv"
    if not orders_path.exists():
        logger.error("Missing %s — run scripts/preprocess_orders.py first.", orders_path)
        return 1

    orders = pd.read_csv(orders_path)
    orders["date"] = pd.to_datetime(orders["date"]).dt.date

    # Build the universe of holdings directly from the order list,
    # then run the full Tarzan enrichment pipeline on it. We use a
    # temporary CSV so the orchestrator can load it through its
    # standard ``load_holdings`` path, which knows how to populate
    # the ``market_value_eur`` seed (used as a sanity check during
    # the bond fallback) and applies all the asset-class
    # classification logic. This avoids reinventing any of that
    # logic here and stays aligned with whatever Tarzan's regular
    # pipeline does.
    from tarzan.data.loader import load_holdings
    import csv as _csv
    import io
    import tempfile

    qty_by_isin = orders.groupby("isin")["quantity"].sum().to_dict()
    name_by_isin = (
        orders.dropna(subset=["name"]).groupby("isin")["name"].first().to_dict()
    )
    currency_by_isin = (
        orders.dropna(subset=["currency"]).groupby("isin")["currency"].first().to_dict()
    )

    # ``market_value_eur`` seed: latest observed price_native × qty,
    # scaled by /100 for bond-shaped rows (price 50–150, qty ≥ 1000)
    # and divided by FX. This is intentionally rough — the enricher
    # will replace it with a real quote whenever yfinance or the
    # bond fetcher returns one. Its only purpose is to give the
    # bond fallback's sanity check a reasonable reference value to
    # compare against, so the LP cum/ex BTPs do not collapse to 0.
    def _mv_seed(isin: str, qty: float) -> float:
        sub = orders[
            (orders["isin"] == isin)
            & (orders["price_native"].notna())
            & (orders["price_native"] != "")
        ].sort_values("date")
        if sub.empty or qty == 0:
            return 0.0
        last = sub.iloc[-1]
        price = float(last["price_native"])
        fx = float(last.get("fx_rate") or 1.0) or 1.0
        avg_price = float(
            pd.to_numeric(sub["price_native"], errors="coerce").mean()
        )
        avg_qty = float(orders[orders["isin"] == isin]["quantity"].abs().mean())
        is_bond = 50.0 <= avg_price <= 150.0 and avg_qty >= 1000.0
        denom = 100.0 if is_bond else 1.0
        return abs(qty) * price / denom / fx

    rows = []
    for isin, qty in qty_by_isin.items():
        # We keep ISINs whose net quantity is non-zero. The bond
        # cum/ex variants used by Italian retail BTPs share the
        # first 9 ISIN characters but differ on the suffix; their
        # quantities cancel each other when aggregated by prefix
        # (a transfer_in CUM is later sold EX with the same face),
        # so we group them so the closed pair contributes zero
        # rather than double-counting the principal.
        prefix = isin[:9]
        prefix_total = sum(
            q for i, q in qty_by_isin.items() if i[:9] == prefix
        )
        if abs(prefix_total) < 0.01:
            # The cum/ex pair fully nets out — treat each variant as
            # closed even if individually it has a non-zero balance.
            continue
        if abs(qty) < 0.01:
            continue
        rows.append({
            "isin": isin,
            "ticker": isin,
            "name": name_by_isin.get(isin, ""),
            "currency": currency_by_isin.get(isin, "EUR"),
            "quantity": qty,
            "cost_basis_eur": 0.0,
            "market_value_eur": _mv_seed(isin, qty),
            "target_equities": "",
            "target_fixed_income": "",
            "no_buy_no_sell": "",
        })

    # Write to a temp CSV and feed it to the standard loader. We
    # keep the file on disk only for the duration of this run so
    # nothing personal leaks.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".csv", newline=""
    ) as tmp:
        writer = _csv.DictWriter(tmp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as fh:
            holdings = load_holdings(fh, filename="orders_derived_holdings.csv")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(
        "Derived %d holding(s) from %d order(s) covering %d ISIN(s) "
        "(%d open, %d closed).",
        len(holdings), len(orders), len(qty_by_isin),
        sum(1 for v in qty_by_isin.values() if abs(v) >= 0.01),
        sum(1 for v in qty_by_isin.values() if abs(v) < 0.01),
    )
    logger.info("Enriching holdings (price histories, asset classes)…")
    holdings = enrich_holdings(holdings)
    isin_to_holding = {h.isin: h for h in holdings if h.isin}

    today = datetime.now().date()
    # Today's portfolio value: simply sum each holding's
    # ``current_value``. The enricher (with the bond /100 fix in
    # place) has already produced the right EUR figure for both
    # equities/ETFs and bonds, including those resolved via Borsa
    # Italiana fallback. Closed positions (``quantity == 0``)
    # contribute zero, so we can iterate over the full list.
    current_value = sum(
        float(h.current_value or 0.0) for h in holdings
    )

    # ------------------------------------------------------------------
    # XIRR
    # ------------------------------------------------------------------
    cashflows: list[tuple[date, float]] = []
    for _, row in orders.iterrows():
        if row["type"] == "transfer_in":
            v = _transfer_value(row)
            if v > 0:
                cashflows.append((row["date"], -v))
        else:
            cf = float(row["net_eur"])
            if cf != 0.0:
                cashflows.append((row["date"], cf))

    cashflows.append((today, current_value))

    deposits = -sum(cf for _, cf in cashflows if cf < 0)
    distributions = sum(cf for _, cf in cashflows if cf > 0) - current_value
    span_days = (today - min(d for d, _ in cashflows)).days
    rate = xirr(cashflows)

    print()
    print("=" * 72)
    print("XIRR — Money-weighted return (sensitive to deposit timing)")
    print("=" * 72)
    print(f"  Period:                {min(d for d, _ in cashflows)} → {today} ({span_days} days)")
    print(f"  Total deposits:        {deposits:>16,.2f} EUR")
    print(f"  Total distributions:   {distributions:>16,.2f} EUR  (sells + coupons)")
    print(f"  Current value:         {current_value:>16,.2f} EUR")
    print(f"  Net P&L:               {distributions + current_value - deposits:>16,.2f} EUR")
    print(f"  XIRR (annualised):     {rate * 100:>15.2f}%")

    # ------------------------------------------------------------------
    # TWROR — chained period returns
    # ------------------------------------------------------------------
    # Build a per-ISIN quantity timeline (cumulative as of end-of-day).
    # We materialise it once into a dict {isin: sorted [(date, cum_qty)]}
    # so ``qty_at`` is a binary search instead of a pandas filter; that
    # turns the whole TWROR loop from O(events × dates × isins) into
    # something close to O(events log events + dates × isins log events).
    qty_events = []
    for _, row in orders.iterrows():
        if row["type"] in ("buy", "sell", "transfer_in"):
            qty_events.append(
                (row["date"], row["isin"], float(row["quantity"]))
            )

    qty_events.sort(key=lambda r: r[0])
    cum_by_isin: dict[str, list[tuple[date, float]]] = {}
    running: dict[str, float] = {}
    for d, isin, delta in qty_events:
        running[isin] = running.get(isin, 0.0) + delta
        cum_by_isin.setdefault(isin, []).append((d, running[isin]))

    def qty_at(isin: str, d: date) -> float:
        series = cum_by_isin.get(isin)
        if not series:
            return 0.0
        # Last entry whose date <= d.
        lo, hi = 0, len(series) - 1
        if d < series[0][0]:
            return 0.0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if series[mid][0] <= d:
                lo = mid
            else:
                hi = mid - 1
        return series[lo][1]

    # Cache price lookups per (isin, date) so cf_dates that share
    # holdings do not re-walk the same price history.
    price_cache: dict[tuple[str, date], float | None] = {}

    # Per-ISIN classification via the same orders heuristic the
    # preprocessor uses. Only used for the ISINs that need a
    # synthetic price history fallback — those don't pass through
    # the enricher's bond-aware /100 scaling, so we apply it here.
    isin_is_bond: dict[str, bool] = {
        isin: _is_bond_via_orders(orders, isin) for isin in cum_by_isin.keys()
    }

    # Synthetic price history for ISINs whose yfinance lookup either
    # failed or returned an empty series. We fall back to linear
    # interpolation between the ``price_native`` observations from
    # the order list itself (every buy/sell/transfer carries a
    # broker-confirmed price). Bond prices come in per-100-face
    # form; we leave them as-is here and let the bond /100 scaling
    # in ``value_at`` handle them, mirroring the convention the
    # enricher applied to native yfinance series.
    synth_history: dict[str, pd.Series] = {}
    for isin in cum_by_isin.keys():
        h = isin_to_holding.get(isin)
        if h is not None and h.price_history is not None and len(h.price_history) > 0:
            continue
        obs = orders[
            (orders["isin"] == isin)
            & (orders["price_native"].notna())
            & (orders["price_native"] != "")
        ][["date", "price_native"]]
        if obs.empty:
            continue
        s = pd.Series(
            obs["price_native"].astype(float).values,
            index=pd.to_datetime(obs["date"]),
        ).sort_index()
        # Drop duplicate-date observations (same-day multiple trades
        # at slightly different prices), keep the mean.
        s = s.groupby(s.index).mean()
        synth_history[isin] = s
        logger.info(
            "Synthetic price history for %s from %d observation(s).",
            isin, len(s),
        )

    def _price_lookup(isin: str, h, d: date) -> tuple[float | None, bool]:
        """Return ``(price, needs_bond_scaling)`` for an ISIN at date.

        ``needs_bond_scaling`` is True only for the synthetic-history
        path: those raw prices come from order rows and still carry
        the per-100-face convention. Prices coming from the enricher
        are already in EUR-per-unit form (because the enricher
        rescaled bond histories by /100 in ``_enrich_single``).
        """
        if h is not None and h.price_history is not None and len(h.price_history) > 0:
            return _price_at(h.price_history, d), False
        s = synth_history.get(isin)
        if s is None or s.empty:
            return None, False
        ts = pd.Timestamp(d)
        if ts <= s.index[0]:
            return float(s.iloc[0]), True
        if ts >= s.index[-1]:
            return float(s.iloc[-1]), True
        before = s.loc[s.index <= ts].iloc[-1]
        after = s.loc[s.index >= ts].iloc[0]
        before_d = s.loc[s.index <= ts].index[-1]
        after_d = s.loc[s.index >= ts].index[0]
        if after_d == before_d:
            return float(before), True
        weight = (ts - before_d).days / (after_d - before_d).days
        return float(before + weight * (after - before)), True

    def value_at(d: date) -> float:
        total = 0.0
        for isin in cum_by_isin.keys():
            qty = qty_at(isin, d)
            if abs(qty) < 0.01:
                continue
            h = isin_to_holding.get(isin)
            key = (isin, d)
            if key not in price_cache:
                price, needs_scale = _price_lookup(isin, h, d)
                if price is not None and needs_scale and isin_is_bond.get(isin, False):
                    price = price / 100.0
                price_cache[key] = price
            price = price_cache[key]
            if price is None:
                continue
            total += qty * price
        return total

    # Compute external inflow per date in *portfolio terms*. For our
    # purposes the portfolio is the basket of securities; cash flows
    # *into* it are buys and transfer_ins, *out* are sells, coupons,
    # and transfer_outs. We negate ``net_eur`` because the file
    # records cash flows on the bank account, not on the portfolio.
    external_per_date: dict[date, float] = {}
    for _, row in orders.iterrows():
        d = row["date"]
        if row["type"] == "transfer_in":
            v = _transfer_value(row)
            external_per_date[d] = external_per_date.get(d, 0.0) + v
        else:
            external_per_date[d] = external_per_date.get(d, 0.0) - float(row["net_eur"])

    cf_dates = sorted(external_per_date.keys())

    # TWROR via chained period returns. Between two consecutive CF
    # dates d_{i-1} and d_i the portfolio earns a market return:
    #
    #     r_i = V_before(d_i) / V_after(d_{i-1}) - 1
    #
    # where ``V_after(d)`` is the portfolio value at the close of d
    # *with* its cash flow already applied — which is exactly what
    # ``value_at(d)`` computes (it uses the cumulative quantity as of
    # end-of-day, i.e. after the day's trades). ``V_before(d)`` is
    # the same value minus the day's external inflow:
    #
    #     V_before(d) = value_at(d) - external_per_date[d]
    #
    # That subtraction works because all positions are valued at the
    # day's close: the only thing that changes the basket value
    # between "before the day's trades" and "after" is the external
    # cash flow itself. Using V_before keeps the cash flow out of the
    # market return — that's the whole point of TWROR.
    twror = 1.0
    prev_v_after = 0.0
    twror_debug = []
    for d in cf_dates:
        if prev_v_after > 0:
            v_before = value_at(d) - external_per_date[d]
            if v_before > 0:
                r = v_before / prev_v_after - 1.0
                twror *= 1.0 + r
                twror_debug.append((d, prev_v_after, v_before, r))
        prev_v_after = value_at(d)

    if prev_v_after > 0:
        r = current_value / prev_v_after - 1.0
        twror *= 1.0 + r
        twror_debug.append((today, prev_v_after, current_value, r))

    twror_pct = (twror - 1.0) * 100
    annualised_twror = (twror ** (365 / span_days) - 1.0) * 100 if span_days > 0 else 0.0

    print()
    print("=" * 72)
    print("TWROR — Time-weighted return (neutral to deposit timing)")
    print("=" * 72)
    print(f"  Period:                {cf_dates[0]} → {today} ({span_days} days)")
    print(f"  Cumulative TWROR:      {twror_pct:>15.2f}%")
    print(f"  Annualised TWROR:      {annualised_twror:>15.2f}%")
    print()
    print("  Per-period debug (anomalies > ±10%):")
    for d, va, vb, r in twror_debug:
        if abs(r) > 0.10:
            print(f"    {d}  v_after_prev={va:>14,.2f}  v_before={vb:>14,.2f}  r={r*100:>+8.2f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
