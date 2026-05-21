"""Unified rebalancing engine using Linear Programming (scipy).

Minimizes total transaction volume while satisfying all 3 target levels:
1. Asset class allocation targets
2. Geographic allocation targets (within equity)
3. Per-holding targets (where specified)

Supports two modes:
- Rebalancing (default): zero-sum (total BUY = total SELL).
- Lump sum allocation: net cash inflow = lump_sum (BUY - SELL = amount).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig

logger = logging.getLogger(__name__)


def compute_unified_rebalancing(
    holdings: list[Holding],
    config: InvestorConfig,
    total_value: float,
    lump_sum: Optional[float] = None,
) -> tuple[list[dict], list[dict]]:
    """Compute optimal rebalancing actions via MILP.

    The invested allocation targets (``invested_allocation_targets_pctg``)
    apply to the *invested* portfolio (total minus cash). The cash position
    has a separate absolute target ``target_cash_buffer_eur``. The solver
    treats excess cash above the target as an implicit lump sum available
    for investment and, in no-sell mode, relaxes the cash target when cash
    is already below it.

    Returns:
        (actions, verifications)
    """
    is_explicit_lump_sum = lump_sum is not None and lump_sum > 0
    n = len(holdings)
    if n == 0:
        return [], []

    def _val(h: Holding) -> float:
        return h.current_value if h.current_value else h.market_value_eur

    values = np.array([_val(h) for h in holdings])
    if np.any(np.isnan(values)):
        valid = ~np.isnan(values)
        holdings = [h for h, v in zip(holdings, valid) if v]
        values = values[valid]
        logger.warning("Filtered %d holdings with NaN values", int((~valid).sum()))
        n = len(holdings)
    tv = values.sum()
    if tv <= 0:
        return [], []

    # --- Split total into cash vs invested ---
    cash_mask = np.array([
        1.0 if h.asset_class == AssetClass.CASH_EQUIVALENTS else 0.0
        for h in holdings
    ])
    cash_value = float((cash_mask * values).sum())
    invested_value = tv - cash_value
    cash_target = float(config.target_cash_buffer_eur or 0.0)

    # --- Build geo exposure matrix ---
    all_geos = sorted(config.equity_geo_targets_pctg.keys())
    geo_frac = np.zeros((n, len(all_geos)))
    for i, h in enumerate(holdings):
        if h.asset_class != AssetClass.EQUITIES:
            continue
        if h.geo_breakdown:
            total_bd = sum(h.geo_breakdown.values())
            if total_bd > 0:
                for geo, pct in h.geo_breakdown.items():
                    gn = geo.value if hasattr(geo, "value") else str(geo)
                    if gn in all_geos:
                        geo_frac[i][all_geos.index(gn)] = pct / total_bd
        elif h.geography:
            gn = h.geography.value if hasattr(h.geography, "value") else str(h.geography)
            if gn in all_geos:
                geo_frac[i][all_geos.index(gn)] = 1.0

    is_equity = np.array([1.0 if h.asset_class == AssetClass.EQUITIES else 0.0 for h in holdings])
    eq_value = (values * is_equity).sum()
    is_fi = np.array([1.0 if h.asset_class == AssetClass.FIXED_INCOME else 0.0 for h in holdings])
    fi_value = (values * is_fi).sum()

    # --- Effective net inflow: explicit lump sum + excess cash above target ---
    # In no-sell mode with cash already below target, we cannot shrink cash
    # further; treat cash as 'locked at current level' and skip the cash
    # constraint (the solver will still respect the explicit lump_sum).
    explicit_lump = float(lump_sum or 0.0)
    cash_excess = cash_value - cash_target
    enforce_cash_target = True
    if config.rebalancing_no_sell and cash_excess < 0:
        # Cash is below target and solver cannot sell to generate more.
        enforce_cash_target = False
        cash_excess = 0.0
        logger.info(
            "Cash (%.2f EUR) is below target (%.2f EUR) and no-sell mode is on; "
            "cash buffer target skipped for this run.",
            cash_value, cash_target,
        )
    effective_net_inflow = explicit_lump + cash_excess
    is_lump_sum = effective_net_inflow > 0

    # --- Solve MILP with progressive tolerance ---
    # Try a graded ladder of tolerances and stop at the first feasible one.
    # The user-configured max_tolerance caps the ladder; we always include
    # max_tol itself as the last step so the ladder reaches the configured
    # ceiling exactly (otherwise an LP that is only feasible at e.g. 2.5%
    # would be missed when the ladder stops at 2.0%).
    max_tol = config.rebalancing_max_tolerance_pctg
    base_steps = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    tolerances = sorted({t for t in base_steps if t < max_tol} | {max_tol})
    actions = []
    used_tolerance = None
    for tol in tolerances:
        result = _solve_lp(n, values, holdings, config, geo_frac, all_geos,
                           eq_value, fi_value, tv, tol,
                           cash_value=cash_value,
                           invested_value=invested_value,
                           net_inflow=effective_net_inflow,
                           lock_cash_holdings=config.rebalancing_no_sell or not enforce_cash_target)
        if result is not None and result.success:
            actions = _extract_actions(result, n, holdings, config, values, geo_frac, all_geos, eq_value)
            used_tolerance = tol
            mode = "lump sum" if is_lump_sum else "rebalancing"
            logger.info("LP %s solved with precision=%.1f%%", mode, tol)
            break
    else:
        logger.info("No rebalancing solution within %.1f%% max tolerance — portfolio is close enough "
                     "given min_transaction constraint", max_tol)

    # Verification
    new_values = values.copy()
    for a in actions:
        idx = next(j for j, h in enumerate(holdings) if h.ticker == a["ticker"])
        new_values[idx] += a["amount_eur"] if a["direction"] == "buy" else -a["amount_eur"]
    verifications = _verify(new_values, holdings, config, geo_frac, all_geos, fi_value)
    if used_tolerance is not None:
        for v in verifications:
            v["tolerance"] = used_tolerance

    return actions, verifications


def _extract_actions(result, n, holdings, config, values, geo_frac, all_geos, eq_value):
    actions = []
    # Loop-invariant aggregates computed once for _build_reason.
    invested_tv = float(sum(
        values[j] for j, hh in enumerate(holdings)
        if hh.asset_class != AssetClass.CASH_EQUIVALENTS
    ))
    fi_val = float(sum(
        values[j] for j, hh in enumerate(holdings)
        if hh.asset_class == AssetClass.FIXED_INCOME
    ))
    # Per-asset-class totals (invariant across the action loop).
    class_sums: dict[str, float] = {}
    for j, hh in enumerate(holdings):
        key = hh.asset_class.value if hh.asset_class else "Alternative"
        class_sums[key] = class_sums.get(key, 0.0) + float(values[j])
    # Per-geography totals within equities (same for every action).
    geo_totals = [
        float(sum(geo_frac[j][g_idx] * values[j] for j, hh in enumerate(holdings)
                  if hh.asset_class == AssetClass.EQUITIES))
        for g_idx in range(len(all_geos))
    ]
    reason_ctx = {
        "invested_tv": invested_tv,
        "fi_val": fi_val,
        "class_sums": class_sums,
        "geo_totals": geo_totals,
    }
    for i, h in enumerate(holdings):
        buy_i = result.x[i]
        sell_i = result.x[n + i]
        # Enforce binary z: if zb < 0.5, zero out buy; if zs < 0.5, zero out sell
        if len(result.x) > 2 * n:
            zb_i = result.x[2 * n + i] if 2*n+i < len(result.x) else 0
            zs_i = result.x[3 * n + i] if 3*n+i < len(result.x) else 0
            if zb_i < 0.5:
                buy_i = 0.0
            if zs_i < 0.5:
                sell_i = 0.0
        net = buy_i - sell_i
        if abs(net) < 1.0:
            continue
        direction = "buy" if net > 0 else "sell"
        reason = _build_reason(
            i, h, config, values, geo_frac, all_geos, eq_value, reason_ctx,
        )
        actions.append({"name": h.name or h.ticker, "ticker": h.ticker,
                        "direction": direction, "amount_eur": round(abs(net), 2), "reason": reason})
    return actions


def _solve_lp(n, values, holdings, config, geo_frac, all_geos, eq_value, fi_value, tv,
              opt_tolerance, cash_value=0.0, invested_value=0.0,
              net_inflow=0.0, lock_cash_holdings=False):
    """Build and solve the MILP for a given precision.

    The invested allocation percentages apply to ``invested_value_after``,
    i.e. (total_value + net_inflow - cash_after), where ``cash_after`` is
    either the cash target (if the cash constraint is enforced) or the
    current cash value (if cash is locked).
    """
    tol_frac = opt_tolerance / 100.0
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    new_tv = tv + net_inflow

    # Cash position after rebalancing. When the cash constraint is enforced,
    # the solver effectively moves (cash_value - cash_target) from cash into
    # the rest of the portfolio, so cash_after == cash_target. When cash is
    # locked (no-sell + cash below target), cash stays at cash_value.
    cash_after = cash_target if not lock_cash_holdings else cash_value
    invested_value_after = max(new_tv - cash_after, 0.0)

    A_eq_rows, b_eq_vals = [], []
    A_ub_rows, b_ub_vals = [], []

    # Cash flow: sum(buy) - sum(sell) = net_inflow  (net_inflow already includes
    # the explicit lump sum + any excess cash being moved into invested assets).
    row = np.zeros(2 * n)
    row[:n] = 1.0
    row[n:] = -1.0
    A_eq_rows.append(row)
    b_eq_vals.append(net_inflow)

    # Invested asset-class constraints: applied on invested_value_after.
    for ac_name, target_pct in config.invested_allocation_targets_pctg.items():
        mask = np.array([
            1.0 if (h.asset_class.value if h.asset_class else "Alternative") == ac_name
            else 0.0 for h in holdings
        ])
        current_sum = (mask * values).sum()
        target_val = target_pct / 100.0 * invested_value_after
        tol_val = tol_frac * invested_value_after
        row_upper = np.zeros(2 * n)
        row_upper[:n] = mask
        row_upper[n:] = -mask
        A_ub_rows.append(row_upper)
        b_ub_vals.append(target_val + tol_val - current_sum)
        A_ub_rows.append(-row_upper)
        b_ub_vals.append(current_sum - (target_val - tol_val))

    # Geo constraints
    new_eq_value = eq_value + net_inflow * (eq_value / tv) if tv > 0 else eq_value
    geo_ref = new_eq_value if new_eq_value > 0 else eq_value
    if geo_ref > 0:
        for g_idx, geo_name in enumerate(all_geos):
            target_pct = config.equity_geo_targets_pctg.get(geo_name, 0)
            target_val = target_pct / 100.0 * geo_ref
            tol_val = tol_frac * geo_ref
            gf = geo_frac[:, g_idx]
            current_sum = (gf * values).sum()
            row_upper = np.zeros(2 * n)
            row_upper[:n] = gf
            row_upper[n:] = -gf
            A_ub_rows.append(row_upper)
            b_ub_vals.append(target_val + tol_val - current_sum)
            A_ub_rows.append(-row_upper)
            b_ub_vals.append(current_sum - (target_val - tol_val))

    # Per-holding equity constraints
    if geo_ref > 0:
        for i, h in enumerate(holdings):
            if h.target_equities is None or h.asset_class != AssetClass.EQUITIES:
                continue
            target_val = h.target_equities / 100.0 * geo_ref
            tol_val = tol_frac * geo_ref
            row_upper = np.zeros(2 * n)
            row_upper[i] = 1.0
            row_upper[n + i] = -1.0
            A_ub_rows.append(row_upper)
            b_ub_vals.append(target_val + tol_val - values[i])
            A_ub_rows.append(-row_upper)
            b_ub_vals.append(values[i] - (target_val - tol_val))

    # Per-holding FI constraints
    new_fi_value = fi_value + net_inflow * (fi_value / tv) if tv > 0 else fi_value
    fi_ref = new_fi_value if new_fi_value > 0 else fi_value
    if fi_ref > 0:
        for i, h in enumerate(holdings):
            if h.target_fixed_income is None or h.asset_class != AssetClass.FIXED_INCOME:
                continue
            target_val = h.target_fixed_income / 100.0 * fi_ref
            tol_val = tol_frac * fi_ref
            row_upper = np.zeros(2 * n)
            row_upper[i] = 1.0
            row_upper[n + i] = -1.0
            A_ub_rows.append(row_upper)
            b_ub_vals.append(target_val + tol_val - values[i])
            A_ub_rows.append(-row_upper)
            b_ub_vals.append(values[i] - (target_val - tol_val))

    # Cash holdings cannot be traded by the solver — the cash buffer is
    # handled via net_inflow (as an external cash flow). Forbid buy/sell
    # on cash instruments explicitly to avoid double-counting.
    for i, h in enumerate(holdings):
        if h.asset_class == AssetClass.CASH_EQUIVALENTS:
            row = np.zeros(2 * n); row[i] = 1.0
            A_eq_rows.append(row); b_eq_vals.append(0.0)
            row = np.zeros(2 * n); row[n + i] = 1.0
            A_eq_rows.append(row); b_eq_vals.append(0.0)

    # MILP: min_tx binary linking for BOTH buy and sell
    # Variables: [buy_0..n-1, sell_0..n-1, zb_0..n-1, zs_0..n-1]  (4n total)
    # zb[i]=1 if buying i, zs[i]=1 if selling i. Both binary.
    # Constraints: buy[i] >= min_tx * zb[i], buy[i] <= M * zb[i]
    #              sell[i] >= min_tx * zs[i], sell[i] <= value[i] * zs[i]
    #              zb[i] + zs[i] <= 1  (can't buy and sell same holding)
    min_tx = config.rebalancing_min_transaction_eur
    M = new_tv
    for i in range(n):
        # Buy linking: buy[i] <= M * zb[i]
        row = np.zeros(4 * n); row[i] = 1.0; row[2*n+i] = -M
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Buy min: buy[i] >= min_tx * zb[i]  →  -buy[i] + min_tx*zb[i] <= 0
        row = np.zeros(4 * n); row[i] = -1.0; row[2*n+i] = min_tx
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Sell linking: sell[i] <= value[i] * zs[i]
        row = np.zeros(4 * n); row[n+i] = 1.0; row[3*n+i] = -float(values[i])
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Sell min: sell[i] >= min_tx * zs[i]  →  -sell[i] + min_tx*zs[i] <= 0
        row = np.zeros(4 * n); row[n+i] = -1.0; row[3*n+i] = min_tx
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Mutual exclusion: zb[i] + zs[i] <= 1
        row = np.zeros(4 * n); row[2*n+i] = 1.0; row[3*n+i] = 1.0
        A_ub_rows.append(row); b_ub_vals.append(1.0)

    # Pad existing constraint rows from 2n to 4n columns
    padded_eq = [np.concatenate([r, np.zeros(2*n)]) for r in A_eq_rows]
    padded_ub = []
    for r in A_ub_rows:
        if len(r) == 2 * n:
            padded_ub.append(np.concatenate([r, np.zeros(2*n)]))
        elif len(r) == 4 * n:
            padded_ub.append(r)
        else:
            padded_ub.append(np.concatenate([r, np.zeros(4*n - len(r))]))

    # Objective: minimize buy + sell (z variables have 0 cost)
    c = np.zeros(4 * n); c[:2*n] = 1.0

    # Bounds
    lb = np.zeros(4 * n)
    ub = np.full(4 * n, np.inf)
    ub[n:2*n] = values      # sell <= current value
    ub[2*n:3*n] = 1.0       # zb <= 1
    ub[3*n:4*n] = 1.0       # zs <= 1

    # Integer constraints: zb and zs are binary
    integrality = np.zeros(4 * n)
    integrality[2*n:] = 1   # all z variables are integers

    # Freeze holdings with no_buy_no_sell=True
    for i, h in enumerate(holdings):
        if h.no_buy_no_sell:
            ub[i] = 0.0        # buy[i] = 0
            ub[n + i] = 0.0    # sell[i] = 0
            ub[2*n + i] = 0.0  # zb[i] = 0
            ub[3*n + i] = 0.0  # zs[i] = 0

    # Global no_sell flag: forbid all sell actions
    if config.rebalancing_no_sell:
        for i in range(n):
            ub[n + i] = 0.0      # sell[i] = 0
            ub[3*n + i] = 0.0    # zs[i] = 0

    constraints = [LinearConstraint(np.array(padded_eq), np.array(b_eq_vals), np.array(b_eq_vals))]
    if padded_ub:
        constraints.append(LinearConstraint(np.array(padded_ub), -np.inf, np.array(b_ub_vals)))

    return milp(c, constraints=constraints, integrality=integrality,
                bounds=Bounds(lb, ub), options={"time_limit": 30, "mip_rel_gap": 1e-6})


def _build_reason(idx, h, config, values, geo_frac, all_geos, eq_value, ctx):
    """Format a short 'why' string for a suggested action.

    ``ctx`` carries loop-invariant aggregates precomputed by ``_extract_actions``:
    ``invested_tv`` (total minus cash), ``fi_val``, ``class_sums`` per class,
    and ``geo_totals`` per equity-geography bucket.
    """
    reasons = []
    invested_tv = ctx["invested_tv"]
    fi_val = ctx["fi_val"]
    class_sums = ctx["class_sums"]
    geo_totals = ctx["geo_totals"]

    ac = h.asset_class.value if h.asset_class else "Alternative"
    ac_sum = class_sums.get(ac, 0.0)
    ac_actual = ac_sum / invested_tv * 100 if invested_tv > 0 else 0.0
    ac_target = config.invested_allocation_targets_pctg.get(ac, 0)
    if h.asset_class != AssetClass.CASH_EQUIVALENTS and abs(ac_actual - ac_target) > 0.5:
        reasons.append(f"{ac} at {ac_actual:.1f}% vs target {ac_target:.0f}%")

    if h.asset_class == AssetClass.EQUITIES and eq_value > 0:
        for g_idx, gn in enumerate(all_geos):
            frac = geo_frac[idx][g_idx]
            if frac > 0.1:
                geo_actual = geo_totals[g_idx] / eq_value * 100
                geo_target = config.equity_geo_targets_pctg.get(gn, 0)
                if abs(geo_actual - geo_target) > 0.5:
                    reasons.append(f"{gn} at {geo_actual:.1f}% vs target {geo_target:.0f}%")

    if h.target_equities is not None and eq_value > 0:
        ph_actual = values[idx] / eq_value * 100
        if abs(ph_actual - h.target_equities) > 0.5:
            reasons.append(
                f"Holding at {ph_actual:.1f}% vs target {h.target_equities:.0f}% of Equities"
            )

    if h.target_fixed_income is not None and fi_val > 0:
        ph_actual = values[idx] / fi_val * 100
        if abs(ph_actual - h.target_fixed_income) > 0.5:
            reasons.append(
                f"Holding at {ph_actual:.1f}% vs target {h.target_fixed_income:.0f}% of FI"
            )
    return "; ".join(reasons[:3]) if reasons else "Optimization"


def _verify(new_values, holdings, config, geo_frac, all_geos, fi_value=0.0):
    verifications = []
    tv = new_values.sum()
    # Split cash from invested for the percentage-based checks.
    cash_mask = np.array([
        1.0 if h.asset_class == AssetClass.CASH_EQUIVALENTS else 0.0
        for h in holdings
    ])
    cash_new = float((cash_mask * new_values).sum())
    invested_new = max(float(tv - cash_new), 0.0)
    # Use the same threshold as the Optimizer traffic-light so the verification
    # "OK / PARTIAL" status does not contradict the user-facing colors.
    tol = float(config.rebalancing_threshold_pctg)

    # Invested asset allocation: percentages relative to invested_new
    # (excludes cash). Cash is never in invested_allocation_targets_pctg.
    class_pcts = {}
    if invested_new > 0:
        for i, h in enumerate(holdings):
            if h.asset_class == AssetClass.CASH_EQUIVALENTS:
                continue
            ac = h.asset_class.value if h.asset_class else "Alternative"
            class_pcts[ac] = class_pcts.get(ac, 0) + new_values[i] / invested_new * 100
    ac_details, ac_items, max_ac = [], [], 0.0
    for ac, target in config.invested_allocation_targets_pctg.items():
        actual = class_pcts.get(ac, 0)
        d = abs(actual - target); max_ac = max(max_ac, d)
        ac_details.append(f"{ac} {actual:.1f}% (tgt. {target:.1f}%)")
        ac_items.append({"category": ac, "actual_pct": actual, "target_pct": target})
    verifications.append({"check": "Invested Allocation", "kind": "asset",
                          "status": "✓ OK" if max_ac <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(ac_details),
                          "items": ac_items})

    # Geo allocation (equity portion, no cash involved)
    eq_mask = np.array([1.0 if h.asset_class == AssetClass.EQUITIES else 0.0 for h in holdings])
    eq_total = (new_values * eq_mask).sum()
    geo_details, geo_items, max_geo = [], [], 0.0
    if eq_total > 0:
        for g_idx, gn in enumerate(all_geos):
            actual = sum(geo_frac[i][g_idx] * new_values[i] for i, h in enumerate(holdings)
                         if h.asset_class == AssetClass.EQUITIES) / eq_total * 100
            target = config.equity_geo_targets_pctg.get(gn, 0)
            d = abs(actual - target); max_geo = max(max_geo, d)
            geo_details.append(f"{gn} {actual:.1f}% (tgt. {target:.1f}%)")
            geo_items.append({"category": gn, "actual_pct": actual, "target_pct": target})
    verifications.append({"check": "Equity Geography", "kind": "geography",
                          "status": "✓ OK" if max_geo <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(geo_details),
                          "items": geo_items})

    # Per-holding equity
    ph_details, ph_items, max_ph = [], [], 0.0
    for i, h in enumerate(holdings):
        if h.target_equities is None or h.asset_class != AssetClass.EQUITIES:
            continue
        actual = new_values[i] / eq_total * 100 if eq_total > 0 else 0
        d = abs(actual - h.target_equities); max_ph = max(max_ph, d)
        ph_details.append(f"{h.ticker} {actual:.1f}% (tgt. {h.target_equities:.0f}%)")
        ph_items.append({"category": h.name or h.ticker, "actual_pct": actual,
                         "target_pct": float(h.target_equities)})
    verifications.append({"check": "Per-Holding Equity Targets", "kind": "per_holding_equity",
                          "status": "✓ OK" if max_ph <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(ph_details) or "No targets set",
                          "items": ph_items})

    # Per-holding FI
    fi_details, fi_items, max_fi = [], [], 0.0
    fi_mask = np.array([1.0 if h.asset_class == AssetClass.FIXED_INCOME else 0.0 for h in holdings])
    fi_total = (new_values * fi_mask).sum()
    for i, h in enumerate(holdings):
        if h.target_fixed_income is None or h.asset_class != AssetClass.FIXED_INCOME:
            continue
        actual = new_values[i] / fi_total * 100 if fi_total > 0 else 0
        d = abs(actual - h.target_fixed_income); max_fi = max(max_fi, d)
        fi_details.append(f"{h.ticker} {actual:.1f}% (tgt. {h.target_fixed_income:.0f}%)")
        fi_items.append({"category": h.name or h.ticker, "actual_pct": actual,
                         "target_pct": float(h.target_fixed_income)})
    verifications.append({"check": "Per-Holding FI Targets", "kind": "per_holding_fi",
                          "status": "✓ OK" if max_fi <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(fi_details) or "No targets set",
                          "items": fi_items})

    # Cash buffer (absolute EUR)
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    cash_delta_eur = cash_new - cash_target
    cash_ok = abs(cash_delta_eur) <= max(cash_target * 0.01, 1.0)  # 1% of target or 1 EUR
    verifications.append({
        "check": "Cash & Cash Equivalents", "kind": "cash",
        "status": "✓ OK" if cash_ok else "⚠ PARTIAL",
        "detail": (
            f"Cash {cash_new:.2f} EUR (tgt. {cash_target:.2f} EUR, "
            f"delta {cash_delta_eur:+.2f} EUR)"
        ),
        "items": [{
            "category": "Cash & Cash Equivalents",
            "actual_eur": cash_new,
            "target_eur": cash_target,
            "delta_eur": cash_delta_eur,
        }],
    })

    return verifications