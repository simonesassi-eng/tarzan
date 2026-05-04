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

    Returns:
        (actions, verifications)
    """
    is_lump_sum = lump_sum is not None and lump_sum > 0
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

    # --- Build geo exposure matrix ---
    all_geos = sorted(config.geo_allocation.keys())
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

    # --- Phase 1: Check trigger (skip for lump sum) ---
    # Always run the solver — min_transaction_eur handles filtering

    # --- Phase 2: Solve MILP with progressive tolerance ---
    # Cap at user-configured max_tolerance to avoid solutions that worsen allocations.
    max_tol = config.rebalancing_max_tolerance
    tolerances = [t for t in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0] if t <= max_tol]
    if not tolerances:
        tolerances = [max_tol]
    actions = []
    used_tolerance = None
    for tol in tolerances:
        result = _solve_lp(n, values, holdings, config, geo_frac, all_geos,
                           eq_value, fi_value, tv, tol,
                           lump_sum=lump_sum if is_lump_sum else None)
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
        reason = _build_reason(i, h, holdings, config, values, geo_frac, all_geos, eq_value)
        actions.append({"name": h.name or h.ticker, "ticker": h.ticker,
                        "direction": direction, "amount_eur": round(abs(net), 2), "reason": reason})
    return actions


def _check_trigger(values, holdings, config, geo_frac, all_geos, eq_value, fi_value, tv, threshold):
    for ac_name, target_pct in config.allocation_targets.items():
        ac_sum = sum(values[i] for i, h in enumerate(holdings)
                     if (h.asset_class.value if h.asset_class else "Alternative") == ac_name)
        if abs(ac_sum / tv * 100 - target_pct) > threshold:
            return True
    if eq_value > 0:
        for g_idx, gn in enumerate(all_geos):
            actual = sum(geo_frac[i][g_idx] * values[i] for i, h in enumerate(holdings)
                         if h.asset_class == AssetClass.EQUITIES) / eq_value * 100
            if abs(actual - config.geo_allocation.get(gn, 0)) > threshold:
                return True
    if eq_value > 0:
        for i, h in enumerate(holdings):
            if h.target_equities is not None and h.asset_class == AssetClass.EQUITIES:
                if abs(values[i] / eq_value * 100 - h.target_equities) > threshold:
                    return True
    if fi_value > 0:
        for i, h in enumerate(holdings):
            if h.target_fixed_income is not None and h.asset_class == AssetClass.FIXED_INCOME:
                if abs(values[i] / fi_value * 100 - h.target_fixed_income) > threshold:
                    return True
    return False


def _solve_lp(n, values, holdings, config, geo_frac, all_geos, eq_value, fi_value, tv, opt_tolerance, lump_sum=None):
    tol_frac = opt_tolerance / 100
    net_inflow = lump_sum if lump_sum else 0.0
    new_tv = tv + net_inflow
    new_eq_value = eq_value + net_inflow * (eq_value / tv) if tv > 0 else eq_value

    A_eq_rows, b_eq_vals = [], []
    A_ub_rows, b_ub_vals = [], []

    # Cash flow: sum(buy) - sum(sell) = net_inflow
    row = np.zeros(2 * n)
    row[:n] = 1.0
    row[n:] = -1.0
    A_eq_rows.append(row)
    b_eq_vals.append(net_inflow)

    # Asset class constraints
    for ac_name, target_pct in config.allocation_targets.items():
        mask = np.array([1.0 if (h.asset_class.value if h.asset_class else "Alternative") == ac_name else 0.0 for h in holdings])
        current_sum = (mask * values).sum()
        target_val = target_pct / 100 * new_tv
        tol_val = tol_frac * new_tv
        row_upper = np.zeros(2 * n)
        row_upper[:n] = mask
        row_upper[n:] = -mask
        A_ub_rows.append(row_upper)
        b_ub_vals.append(target_val + tol_val - current_sum)
        A_ub_rows.append(-row_upper)
        b_ub_vals.append(current_sum - (target_val - tol_val))

    # Geo constraints
    geo_ref = new_eq_value if new_eq_value > 0 else eq_value
    if geo_ref > 0:
        for g_idx, geo_name in enumerate(all_geos):
            target_pct = config.geo_allocation.get(geo_name, 0)
            target_val = target_pct / 100 * geo_ref
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
            target_val = h.target_equities / 100 * geo_ref
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
            target_val = h.target_fixed_income / 100 * fi_ref
            tol_val = tol_frac * fi_ref
            row_upper = np.zeros(2 * n)
            row_upper[i] = 1.0
            row_upper[n + i] = -1.0
            A_ub_rows.append(row_upper)
            b_ub_vals.append(target_val + tol_val - values[i])
            A_ub_rows.append(-row_upper)
            b_ub_vals.append(values[i] - (target_val - tol_val))

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


def _build_reason(idx, h, holdings, config, values, geo_frac, all_geos, eq_value):
    reasons = []
    tv = values.sum()
    ac = h.asset_class.value if h.asset_class else "Alternative"
    ac_sum = sum(values[j] for j, hh in enumerate(holdings)
                 if (hh.asset_class.value if hh.asset_class else "Alternative") == ac)
    ac_actual = ac_sum / tv * 100
    ac_target = config.allocation_targets.get(ac, 0)
    if abs(ac_actual - ac_target) > 0.5:
        reasons.append(f"{ac} {ac_actual:.1f}% → {ac_target:.0f}%")
    if h.asset_class == AssetClass.EQUITIES and eq_value > 0:
        for g_idx, gn in enumerate(all_geos):
            frac = geo_frac[idx][g_idx]
            if frac > 0.1:
                geo_actual = sum(geo_frac[j][g_idx] * values[j] for j, hh in enumerate(holdings)
                                 if hh.asset_class == AssetClass.EQUITIES) / eq_value * 100
                geo_target = config.geo_allocation.get(gn, 0)
                if abs(geo_actual - geo_target) > 0.5:
                    reasons.append(f"{gn} {geo_actual:.1f}% → {geo_target:.0f}%")
    if h.target_equities is not None and eq_value > 0:
        ph_actual = values[idx] / eq_value * 100
        if abs(ph_actual - h.target_equities) > 0.5:
            reasons.append(f"Holding {ph_actual:.1f}% → {h.target_equities:.0f}% of Eq")
    fi_val = sum(values[j] for j, hh in enumerate(holdings) if hh.asset_class == AssetClass.FIXED_INCOME)
    if h.target_fixed_income is not None and fi_val > 0:
        ph_actual = values[idx] / fi_val * 100
        if abs(ph_actual - h.target_fixed_income) > 0.5:
            reasons.append(f"Holding {ph_actual:.1f}% → {h.target_fixed_income:.0f}% of FI")
    return "; ".join(reasons[:3]) if reasons else "Optimization"


def _verify(new_values, holdings, config, geo_frac, all_geos, fi_value=0.0):
    verifications = []
    tv = new_values.sum()
    tol = 1.0  # 1% tolerance for verification display

    # Asset allocation
    class_pcts = {}
    for i, h in enumerate(holdings):
        ac = h.asset_class.value if h.asset_class else "Alternative"
        class_pcts[ac] = class_pcts.get(ac, 0) + new_values[i] / tv * 100
    ac_details, max_ac = [], 0.0
    for ac, target in config.allocation_targets.items():
        actual = class_pcts.get(ac, 0)
        d = abs(actual - target); max_ac = max(max_ac, d)
        ac_details.append(f"{ac} {actual:.1f}% (tgt. {target:.1f}%)")
    verifications.append({"check": "Asset Allocation",
                          "status": "✓ OK" if max_ac <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(ac_details)})

    # Geo allocation
    eq_mask = np.array([1.0 if h.asset_class == AssetClass.EQUITIES else 0.0 for h in holdings])
    eq_total = (new_values * eq_mask).sum()
    geo_details, max_geo = [], 0.0
    if eq_total > 0:
        for g_idx, gn in enumerate(all_geos):
            actual = sum(geo_frac[i][g_idx] * new_values[i] for i, h in enumerate(holdings)
                         if h.asset_class == AssetClass.EQUITIES) / eq_total * 100
            target = config.geo_allocation.get(gn, 0)
            d = abs(actual - target); max_geo = max(max_geo, d)
            geo_details.append(f"{gn} {actual:.1f}% (tgt. {target:.1f}%)")
    verifications.append({"check": "Geo Allocation",
                          "status": "✓ OK" if max_geo <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(geo_details)})

    # Per-holding equity
    ph_details, max_ph = [], 0.0
    for i, h in enumerate(holdings):
        if h.target_equities is None or h.asset_class != AssetClass.EQUITIES:
            continue
        actual = new_values[i] / eq_total * 100 if eq_total > 0 else 0
        d = abs(actual - h.target_equities); max_ph = max(max_ph, d)
        ph_details.append(f"{h.ticker} {actual:.1f}% (tgt. {h.target_equities:.0f}%)")
    verifications.append({"check": "Per-Holding Equity Targets",
                          "status": "✓ OK" if max_ph <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(ph_details) or "No targets set"})

    # Per-holding FI
    fi_details, max_fi = [], 0.0
    fi_mask = np.array([1.0 if h.asset_class == AssetClass.FIXED_INCOME else 0.0 for h in holdings])
    fi_total = (new_values * fi_mask).sum()
    for i, h in enumerate(holdings):
        if h.target_fixed_income is None or h.asset_class != AssetClass.FIXED_INCOME:
            continue
        actual = new_values[i] / fi_total * 100 if fi_total > 0 else 0
        d = abs(actual - h.target_fixed_income); max_fi = max(max_fi, d)
        fi_details.append(f"{h.ticker} {actual:.1f}% (tgt. {h.target_fixed_income:.0f}%)")
    verifications.append({"check": "Per-Holding FI Targets",
                          "status": "✓ OK" if max_fi <= tol else "⚠ PARTIAL",
                          "detail": ", ".join(fi_details) or "No targets set"})

    return verifications