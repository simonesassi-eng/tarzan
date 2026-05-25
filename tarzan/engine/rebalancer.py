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

import copy
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
    relaxed = False
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

    # --- Auto-relax phase ---
    # If the configured ceiling cannot host any feasible plan, optionally
    # search for the *smallest* tolerance >max_tol that does, capped at
    # ``rebalancing_relax_cap_pctg`` (10% by default). The intent is to
    # surface SOMETHING actionable rather than a silent "no actions",
    # while making it absolutely clear in the output that the configured
    # ceiling was crossed.
    if used_tolerance is None and config.rebalancing_auto_relax:
        relax_cap = max(float(config.rebalancing_relax_cap_pctg or 0.0), max_tol)
        if relax_cap > max_tol:
            logger.info(
                "LP infeasible at ±%.1f%% — auto-relax search up to ±%.1f%%.",
                max_tol, relax_cap,
            )
            tol_relaxed = _find_min_feasible_tolerance(
                lo=max_tol, hi=relax_cap, precision=0.5,
                solve=lambda t: _solve_lp(
                    n, values, holdings, config, geo_frac, all_geos,
                    eq_value, fi_value, tv, t,
                    cash_value=cash_value,
                    invested_value=invested_value,
                    net_inflow=effective_net_inflow,
                    lock_cash_holdings=(
                        config.rebalancing_no_sell or not enforce_cash_target
                    ),
                ),
            )
            if tol_relaxed is not None:
                # Re-solve at the found tolerance to get the actual
                # solution (binary search may have stopped at a slightly
                # higher value due to precision rounding — re-solving at
                # the rounded value gives us a clean optimum).
                final = _solve_lp(
                    n, values, holdings, config, geo_frac, all_geos,
                    eq_value, fi_value, tv, tol_relaxed,
                    cash_value=cash_value,
                    invested_value=invested_value,
                    net_inflow=effective_net_inflow,
                    lock_cash_holdings=(
                        config.rebalancing_no_sell or not enforce_cash_target
                    ),
                )
                if final is not None and final.success:
                    actions = _extract_actions(
                        final, n, holdings, config, values, geo_frac,
                        all_geos, eq_value,
                    )
                    used_tolerance = tol_relaxed
                    relaxed = True
                    logger.warning(
                        "LP solved at relaxed tolerance ±%.1f%% (configured "
                        "ceiling ±%.1f%%). Review your per-holding targets "
                        "or raise rebalancing_max_tolerance_pctg.",
                        tol_relaxed, max_tol,
                    )

    # Final fallback log when nothing worked, in either auto-relax or
    # auto-relax-disabled mode.
    if used_tolerance is None:
        if _max_drift_exceeds_tolerance(
            values, holdings, config,
            invested_value=invested_value,
            eq_value=eq_value, fi_value=fi_value,
            cash_value=cash_value,
        ):
            relax_cap = max(float(config.rebalancing_relax_cap_pctg or 0.0), max_tol)
            if config.rebalancing_auto_relax and relax_cap > max_tol:
                logger.warning(
                    "No feasible rebalancing solution even at the auto-relax "
                    "cap ±%.1f%%. Targets are too aggressive — review them.",
                    relax_cap,
                )
            else:
                logger.warning(
                    "No feasible rebalancing solution within max tolerance "
                    "±%.1f%%. At least one allocation drift exceeds the "
                    "ceiling — raise rebalancing_max_tolerance_pctg, enable "
                    "rebalancing_auto_relax, or relax per-holding targets.",
                    max_tol,
                )
        else:
            logger.info(
                "Portfolio already aligned within ±%.1f%% — no rebalancing needed.",
                max_tol,
            )

    # Distinguish two cases for the UI:
    #   1. "Already aligned" — every category is within ±tolerance of
    #      its target, so the solver returned 0 actions because none
    #      were needed. ``infeasible`` is False.
    #   2. "Infeasible" — at least one category is beyond the
    #      tolerance ceiling, so the targets cannot be reached within
    #      ``rebalancing_max_tolerance_pctg``. The solver returned 0
    #      actions because no feasible plan exists. The Excel/newsletter
    #      should NOT render post-rebalancing values in this case
    #      (they would look identical to current and falsely suggest
    #      a rebalance happened).
    infeasible = (used_tolerance is None) and bool(_max_drift_exceeds_tolerance(
        values, holdings, config,
        invested_value=invested_value,
        eq_value=eq_value, fi_value=fi_value,
        cash_value=cash_value,
    ))

    # Verification
    new_values = values.copy()
    for a in actions:
        idx = next(j for j, h in enumerate(holdings) if h.ticker == a["ticker"])
        new_values[idx] += a["amount_eur"] if a["direction"] == "buy" else -a["amount_eur"]
    verifications = _verify(new_values, holdings, config, geo_frac, all_geos, fi_value)
    if used_tolerance is not None:
        for v in verifications:
            v["tolerance"] = used_tolerance
            if relaxed:
                v["relaxed"] = True
                v["configured_max_tolerance"] = max_tol
    if infeasible:
        # Mark every verification entry so downstream renderers know
        # post-rebalancing values are not actionable.
        for v in verifications:
            v["no_solution"] = True

    return actions, verifications


def _find_min_feasible_tolerance(lo: float, hi: float, precision: float,
                                 solve) -> Optional[float]:
    """Binary-search the smallest feasible tolerance for the LP.

    Args:
        lo: Lower bound (already known to be infeasible).
        hi: Upper bound — the largest tolerance we are willing to accept.
        precision: Stop when the bracket is smaller than this (in pp).
        solve: Callable ``f(tolerance)`` returning the scipy.optimize
            result (or None). The result is feasible iff
            ``result is not None and result.success``.

    Returns:
        The smallest feasible tolerance rounded up to ``precision``,
        or ``None`` if even ``hi`` is infeasible.
    """
    # Quick check: does the cap itself work?
    res_hi = solve(hi)
    if not (res_hi is not None and res_hi.success):
        return None

    # Binary search between lo (infeasible) and hi (feasible).
    feasible_hi = hi
    while (feasible_hi - lo) > precision:
        mid = (lo + feasible_hi) / 2.0
        res_mid = solve(mid)
        if res_mid is not None and res_mid.success:
            feasible_hi = mid
        else:
            lo = mid

    # Round up to the requested precision so the caller can re-solve at
    # a clean tolerance value.
    return round(feasible_hi + precision / 2.0 - 1e-9, 1)


def _max_drift_exceeds_tolerance(values, holdings, config, *,
                                 invested_value: float, eq_value: float,
                                 fi_value: float, cash_value: float) -> bool:
    """Return True if at least one allocation category has a drift
    larger than ``rebalancing_max_tolerance_pctg`` (or, for cash, a
    relative deviation greater than the same percentage).

    Used to distinguish "no actions because the portfolio is already
    aligned" from "no actions because the LP is infeasible at the
    configured tolerance ceiling". The first case should still report
    post-rebalancing values (they happen to equal current values); the
    second case should mark them as not actionable so the UI does not
    look like a rebalance was applied.
    """
    max_tol = float(config.rebalancing_max_tolerance_pctg or 0.0)
    if max_tol <= 0:
        return False
    inv = float(invested_value or 0.0)
    eq = float(eq_value or 0.0)
    fi = float(fi_value or 0.0)
    if inv <= 0:
        return False

    # Asset-class drift on invested portion.
    class_actual: dict[str, float] = {}
    for i, h in enumerate(holdings):
        if h.asset_class == AssetClass.CASH_EQUIVALENTS:
            continue
        ac = h.asset_class.value if h.asset_class else "Alternative"
        class_actual[ac] = class_actual.get(ac, 0.0) + float(values[i]) / inv * 100.0
    for ac, target in (config.invested_allocation_targets_pctg or {}).items():
        if abs(class_actual.get(ac, 0.0) - float(target)) > max_tol:
            return True

    # Equity geography drift.
    if eq > 0:
        # Already imported numpy as np at top of module.
        all_geos = sorted((config.equity_geo_targets_pctg or {}).keys())
        for g_idx, gn in enumerate(all_geos):
            target = float(config.equity_geo_targets_pctg.get(gn, 0.0))
            actual = sum(
                _geo_share(holdings[i], gn) * float(values[i])
                for i in range(len(holdings))
                if holdings[i].asset_class == AssetClass.EQUITIES
            ) / eq * 100.0
            if abs(actual - target) > max_tol:
                return True

    # Per-holding equity targets.
    if eq > 0:
        for i, h in enumerate(holdings):
            if h.target_equities is None or h.asset_class != AssetClass.EQUITIES:
                continue
            actual = float(values[i]) / eq * 100.0
            if abs(actual - float(h.target_equities)) > max_tol:
                return True

    # Per-holding fixed-income targets.
    if fi > 0:
        for i, h in enumerate(holdings):
            if h.target_fixed_income is None or h.asset_class != AssetClass.FIXED_INCOME:
                continue
            actual = float(values[i]) / fi * 100.0
            if abs(actual - float(h.target_fixed_income)) > max_tol:
                return True

    # Cash buffer relative deviation (skip when no target is set).
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    if cash_target > 0:
        rel_dev = abs(cash_value - cash_target) / cash_target * 100.0
        if rel_dev > max_tol:
            return True

    return False


def _geo_share(h, geo_name: str) -> float:
    """Return the share (0..1) of holding ``h``'s value attributable to
    ``geo_name``. Falls back to 1.0 when the holding's geography label
    matches and no breakdown is available.
    """
    if h.geo_breakdown:
        total = sum(h.geo_breakdown.values()) or 0
        if total > 0:
            for geo, pct in h.geo_breakdown.items():
                gn = geo.value if hasattr(geo, "value") else str(geo)
                if gn == geo_name:
                    return float(pct) / float(total)
        return 0.0
    if h.geography:
        gn = h.geography.value if hasattr(h.geography, "value") else str(h.geography)
        return 1.0 if gn == geo_name else 0.0
    return 0.0


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

    # Drift-tracking specs collected while constraints are built. Each
    # entry is a tuple (coeff_2n, target_val) describing a linear
    # combination of buy/sell variables whose post-rebalancing value
    # we want close to ``target_val``. After the LP is fully built we
    # expand the variable vector with one positive- and one negative-
    # slack per drift, and add a `drift_penalty_weight * (gp + gn)`
    # term to the objective. With weight 0 the slacks are unconstrained
    # at zero cost so the LP behaves exactly as before.
    drift_w = float(getattr(config, "rebalancing_drift_penalty_weight", 0.0) or 0.0)
    drift_specs: list[tuple[np.ndarray, float]] = []

    # ------------------------------------------------------------------
    # Per-holding trade frictions: capital gains tax and fixed fees.
    # Computed once and used in two places — the cash-flow constraint
    # (so the proceeds from a sell are NET of taxes and fees) and the
    # objective (so the solver pays a small extra weight per executed
    # transaction). Holdings currently in loss have ``tax_i = 0``
    # because realising the loss does not generate a tax bill.
    # ------------------------------------------------------------------
    fee_buy = float(config.rebalancing_transaction_fee_buy_eur or 0.0)
    fee_sell = float(config.rebalancing_transaction_fee_sell_eur or 0.0)
    cg_std = float(config.rebalancing_capital_gains_tax_standard_pctg or 0.0) / 100.0
    cg_gov = float(config.rebalancing_capital_gains_tax_government_pctg or 0.0) / 100.0

    tax_per_unit_sold = np.zeros(n)
    for i, h in enumerate(holdings):
        # Pro-rata realised gain per euro sold. If gain_pct is None or
        # ≤ 0 the position is not in profit and no tax is due.
        gp = float(h.gain_pct or 0.0)
        if gp <= 0:
            continue
        gain_frac = gp / 100.0
        instr = (h.instrument_type or "").lower()
        rate = cg_gov if "government bond" in instr else cg_std
        tax_per_unit_sold[i] = rate * gain_frac

    # ------------------------------------------------------------------
    # Cash flow constraint. Selling EUR `sell[i]` yields proceeds of
    # (1 - tax_i) * sell[i] (tax is withheld). Each executed buy/sell
    # pays a fixed fee that reduces available cash by `fee_buy * zb[i]`
    # or `fee_sell * zs[i]`. The cash equation is therefore:
    #
    #     Σ (1 - tax_i) * sell[i]  +  net_inflow
    #         = Σ buy[i]  +  fee_buy * Σ zb[i]  +  fee_sell * Σ zs[i]
    #
    # Re-arranged so all variables are on the LHS (4n columns):
    #
    #     Σ buy[i] - Σ (1 - tax_i) * sell[i]
    #         + fee_buy * Σ zb[i] + fee_sell * Σ zs[i] = net_inflow
    # ------------------------------------------------------------------
    cash_row = np.zeros(4 * n)
    cash_row[:n] = 1.0
    cash_row[n:2 * n] = -(1.0 - tax_per_unit_sold)
    cash_row[2 * n:3 * n] = fee_buy
    cash_row[3 * n:4 * n] = fee_sell
    A_eq_rows.append(cash_row)
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
        # Track residual drift relative to ``target_val``. The 2n
        # coefficient row times [buy, sell] equals the *change* in the
        # asset-class sum due to trades; its post-rebal value is
        # ``current_sum + change``. We want ``current_sum + change``
        # close to ``target_val``, i.e. drift = (current_sum + change)
        # - target_val. Stored as (coeff_2n, target - current).
        if drift_w > 0:
            drift_specs.append((row_upper.copy(), target_val - current_sum))

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
            if drift_w > 0:
                drift_specs.append((row_upper.copy(), target_val - current_sum))

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
            if drift_w > 0:
                drift_specs.append((row_upper.copy(), target_val - values[i]))

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
            if drift_w > 0:
                drift_specs.append((row_upper.copy(), target_val - values[i]))

    # Cash holdings cannot be traded by the solver — the cash buffer is
    # handled via net_inflow (as an external cash flow). Forbid buy/sell
    # on cash instruments explicitly to avoid double-counting.
    for i, h in enumerate(holdings):
        if h.asset_class == AssetClass.CASH_EQUIVALENTS:
            row = np.zeros(2 * n); row[i] = 1.0
            A_eq_rows.append(row); b_eq_vals.append(0.0)
            row = np.zeros(2 * n); row[n + i] = 1.0
            A_eq_rows.append(row); b_eq_vals.append(0.0)

    # MILP binary linking. zb[i]=1 if buying i, zs[i]=1 if selling i.
    # The binaries are needed so the cash-flow constraint can charge a
    # fixed buy/sell fee per executed trade (fee enters as
    # ``fee_buy * zb[i] + fee_sell * zs[i]``). They are also used to
    # enforce mutual exclusion (no holding can be simultaneously
    # bought and sold in the same plan).
    #
    # Variables: [buy_0..n-1, sell_0..n-1, zb_0..n-1, zs_0..n-1]  (4n).
    M = new_tv
    for i in range(n):
        # Buy linking: buy[i] <= M * zb[i]  →  buy[i] - M*zb[i] <= 0
        row = np.zeros(4 * n); row[i] = 1.0; row[2 * n + i] = -M
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Sell linking: sell[i] <= value[i] * zs[i]
        row = np.zeros(4 * n); row[n + i] = 1.0; row[3 * n + i] = -float(values[i])
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Mutual exclusion: zb[i] + zs[i] <= 1
        row = np.zeros(4 * n); row[2 * n + i] = 1.0; row[3 * n + i] = 1.0
        A_ub_rows.append(row); b_ub_vals.append(1.0)

    # Pad existing constraint rows up to the full variable width.
    # Layout: [buy(n), sell(n), zb(n), zs(n), gp(K), gn(K)] where K is
    # the number of drift specs collected. With ``drift_w == 0`` we
    # skip the slack expansion entirely (K = 0) so the LP is identical
    # to the previous formulation.
    K = len(drift_specs)
    total_vars = 4 * n + 2 * K
    padded_eq = []
    for r in A_eq_rows:
        if len(r) == total_vars:
            padded_eq.append(r)
        elif len(r) == 2 * n:
            padded_eq.append(np.concatenate([r, np.zeros(total_vars - 2 * n)]))
        else:
            padded_eq.append(np.concatenate([r, np.zeros(total_vars - len(r))]))
    padded_ub = []
    for r in A_ub_rows:
        if len(r) == total_vars:
            padded_ub.append(r)
        elif len(r) == 2 * n:
            padded_ub.append(np.concatenate([r, np.zeros(total_vars - 2 * n)]))
        elif len(r) == 4 * n:
            padded_ub.append(np.concatenate([r, np.zeros(total_vars - 4 * n)]))
        else:
            padded_ub.append(np.concatenate([r, np.zeros(total_vars - len(r))]))

    # Drift-tracking equalities. For each spec (coeff_2n, residual)
    # add the constraint
    #     coeff_2n · [buy, sell] + gp_k - gn_k = residual
    # so that gp_k - gn_k captures the "target - post-rebal" delta and
    # the |drift| can be charged in the objective via gp_k + gn_k.
    for k, (coeff_2n, residual) in enumerate(drift_specs):
        row = np.zeros(total_vars)
        row[:2 * n] = coeff_2n
        row[4 * n + k] = 1.0          # gp_k
        row[4 * n + K + k] = -1.0     # gn_k
        padded_eq.append(row)
        b_eq_vals.append(residual)

    # Objective: minimise total trade volume, optionally penalising
    # the absolute drift left at each tracked target. Friction costs
    # (fees and capital-gains taxes) are folded into the cash-flow
    # constraint rather than the objective so we don't double-count
    # their effect — a sell of a profitable position has to be larger
    # to fund the same buy because part of the proceeds goes to the
    # tax authority, and each transaction reduces the available cash
    # by the fixed fee.
    #
    # The drift penalty makes the LP prefer plans that pull every
    # tracked target *closer* to its desired value rather than only
    # ensuring the tolerance band is respected. With weight 0 the
    # solver behaves like the legacy "minimum-trading" optimiser; with
    # weight 1 a EUR of leftover drift costs the same as a EUR of
    # trade volume; higher values aggressively close drift even at
    # the price of more trades.
    c = np.zeros(total_vars)
    c[:2 * n] = 1.0  # buy and sell penalised at unit weight (volume)
    if K > 0:
        c[4 * n:4 * n + K] = drift_w        # gp_k
        c[4 * n + K:4 * n + 2 * K] = drift_w  # gn_k

    # Bounds
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)
    ub[n:2*n] = values      # sell <= current value
    ub[2*n:3*n] = 1.0       # zb <= 1
    ub[3*n:4*n] = 1.0       # zs <= 1
    # gp_k and gn_k are non-negative slacks with no upper bound.

    # Integer constraints: only zb and zs are binary; the slack
    # variables remain continuous.
    integrality = np.zeros(total_vars)
    integrality[2*n:4*n] = 1   # zb and zs are integers

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
    tol = float(config.alert_threshold_pctg)

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
        # ``ticker`` is included so renderers can correlate this entry
        # with the pre-rebalancing snapshot in ``holdings_df``. Two
        # holdings can share a name (e.g. two BTPs labelled "BUONI
        # POLIENNALI DEL TES") so name alone is not a stable key.
        ph_items.append({"category": h.name or h.ticker,
                         "ticker": h.ticker,
                         "actual_pct": actual,
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
        # See the per-holding equity comment above for the ``ticker``
        # field rationale.
        fi_items.append({"category": h.name or h.ticker,
                         "ticker": h.ticker,
                         "actual_pct": actual,
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



def compute_drift_penalty_sensitivity(
    holdings: list[Holding],
    config: InvestorConfig,
    total_value: float,
    lump_sum: Optional[float] = None,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """Run the optimizer at a series of drift-penalty weights.

    Returns a list of *regimes* — distinct optimization outcomes
    aggregated by the range of weights that produce them. Two
    consecutive sweep points produce the same regime when their
    actions are identical (within €1 tolerance) for every holding.

    The sweep covers a coarse-but-meaningful set of weights so the
    user can spot turning points without an explosion of rows. With
    LPs being piecewise-constant in this parameter, increasing
    granularity beyond this set rarely uncovers new behaviour.

    Each returned dict carries:
        - ``weight_min``, ``weight_max``: the inclusive range of
          weights that selects this regime.
        - ``actions``: trade list (same shape as the main optimizer).
        - ``n_buy``, ``n_sell``: count of executed buys/sells.
        - ``total_buy``, ``total_sell``: gross EUR volumes.
        - ``total_tax``, ``total_fee``: friction breakdown computed
          using the same formulas as the main solver.
        - ``max_drift_pp``: largest absolute drift the plan leaves on
          any tracked target (asset class, geography, per-holding).
        - ``buy_by_class``: total BUY volume grouped by asset class.

    The ``weights`` argument lets callers customize the scan.
    """
    if not holdings:
        return []

    if weights is None:
        # Default sweep: dense around the typical "balanced" regime
        # (0–2) and sparse beyond. Adjust if you need finer detail.
        weights = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    weights = sorted(set(round(float(w), 4) for w in weights))

    fee_buy = float(config.rebalancing_transaction_fee_buy_eur or 0.0)
    fee_sell = float(config.rebalancing_transaction_fee_sell_eur or 0.0)
    cg_std = float(config.rebalancing_capital_gains_tax_standard_pctg or 0.0) / 100.0
    cg_gov = float(config.rebalancing_capital_gains_tax_government_pctg or 0.0) / 100.0

    def _summarise(actions: list[dict], verifications: list[dict]) -> dict:
        n_buy = sum(1 for a in actions if a["direction"] == "buy")
        n_sell = sum(1 for a in actions if a["direction"] == "sell")
        total_buy = sum(a["amount_eur"] for a in actions if a["direction"] == "buy")
        total_sell = sum(a["amount_eur"] for a in actions if a["direction"] == "sell")
        total_fee = n_buy * fee_buy + n_sell * fee_sell
        total_tax = 0.0
        buy_by_class: dict[str, float] = {}
        sell_by_class: dict[str, float] = {}
        ticker_to_holding = {h.ticker: h for h in holdings}
        for a in actions:
            h = ticker_to_holding.get(a["ticker"])
            ac = h.asset_class.value if h and h.asset_class else "—"
            if a["direction"] == "buy":
                buy_by_class[ac] = buy_by_class.get(ac, 0.0) + a["amount_eur"]
            else:
                sell_by_class[ac] = sell_by_class.get(ac, 0.0) + a["amount_eur"]
                if not h:
                    continue
                gp = float(h.gain_pct or 0.0)
                if gp <= 0:
                    continue
                instr = (h.instrument_type or "").lower()
                rate = cg_gov if "government bond" in instr else cg_std
                total_tax += a["amount_eur"] * (gp / 100.0) * rate

        max_drift_pp = 0.0
        for v in verifications:
            if v.get("kind") not in ("asset", "geography",
                                     "per_holding_equity", "per_holding_fi"):
                continue
            for it in v.get("items", []) or []:
                d = abs(float(it["actual_pct"]) - float(it["target_pct"]))
                if d > max_drift_pp:
                    max_drift_pp = d
        return {
            "actions": actions,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "total_buy": total_buy,
            "total_sell": total_sell,
            "total_tax": total_tax,
            "total_fee": total_fee,
            "max_drift_pp": max_drift_pp,
            "buy_by_class": buy_by_class,
            "sell_by_class": sell_by_class,
            # Net change per asset class (BUY − SELL). Positive values
            # mean the class grows after the rebalance, negative means
            # it shrinks. Cleaner to read than two separate columns
            # when the user only cares about the resulting drift.
            "net_by_class": {
                ac: buy_by_class.get(ac, 0.0) - sell_by_class.get(ac, 0.0)
                for ac in set(buy_by_class) | set(sell_by_class)
            },
        }

    def _action_signature(actions: list[dict]) -> tuple:
        # Round amounts to nearest EUR so micro-numerical wiggles do
        # not split otherwise-identical regimes.
        return tuple(sorted(
            (a["ticker"], a["direction"], round(a["amount_eur"]))
            for a in actions
        ))

    samples = []
    for w in weights:
        sweep_cfg = copy.copy(config)
        sweep_cfg.rebalancing_drift_penalty_weight = float(w)
        actions, verifs = compute_unified_rebalancing(
            holdings, sweep_cfg, total_value, lump_sum=lump_sum,
        )
        summary = _summarise(actions, verifs)
        summary["weight"] = float(w)
        summary["signature"] = _action_signature(actions)
        samples.append(summary)

    # Group consecutive samples that share the same action signature.
    regimes: list[dict] = []
    for s in samples:
        if regimes and regimes[-1]["signature"] == s["signature"]:
            regimes[-1]["weight_max"] = s["weight"]
        else:
            entry = {k: v for k, v in s.items() if k != "weight"}
            entry["weight_min"] = s["weight"]
            entry["weight_max"] = s["weight"]
            regimes.append(entry)

    # Drop the signature key — it's an internal grouping device.
    for r in regimes:
        r.pop("signature", None)
    return regimes
