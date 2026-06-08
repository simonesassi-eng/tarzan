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
import math
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

    # Lump sum is the only external cash flow we treat as "must
    # deploy in full". The cash holdings (e.g. money-market sweep)
    # are now ordinary tradeable instruments controlled by the
    # ``target_cash_buffer_eur`` constraint, so the solver can
    # explicitly buy or sell them like any other holding. The
    # legacy ``cash_excess`` shortcut disappears: if cash is above
    # the buffer target, the solver will propose a SELL on the
    # cash holding (and a matching BUY elsewhere); if it is below,
    # a BUY on the cash holding.
    explicit_lump = float(lump_sum or 0.0)
    is_lump_sum = explicit_lump > 0

    # --- Solve MILP with progressive tolerance ---
    # Try a graded ladder of tolerances and stop at the first feasible one.
    # The user-configured max_tolerance caps the ladder; we always include
    # max_tol itself as the last step so the ladder reaches the configured
    # ceiling exactly (otherwise an LP that is only feasible at e.g. 2.5%
    # would be missed when the ladder stops at 2.0%).
    max_tol = config.rebalancing_target_tolerance_pctg
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
                           lump_sum=explicit_lump,
                           no_sell_mode=config.rebalancing_no_sell)
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
                "LP infeasible at ±%.2f%% — auto-relax search up to ±%.2f%%.",
                max_tol, relax_cap,
            )
            tol_relaxed = _find_min_feasible_tolerance(
                lo=max_tol, hi=relax_cap, precision=0.025,
                solve=lambda t: _solve_lp(
                    n, values, holdings, config, geo_frac, all_geos,
                    eq_value, fi_value, tv, t,
                    cash_value=cash_value,
                    invested_value=invested_value,
                    lump_sum=explicit_lump,
                    no_sell_mode=config.rebalancing_no_sell,
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
                    lump_sum=explicit_lump,
                    no_sell_mode=config.rebalancing_no_sell,
                )
                if final is not None and final.success:
                    actions = _extract_actions(
                        final, n, holdings, config, values, geo_frac,
                        all_geos, eq_value,
                    )
                    used_tolerance = tol_relaxed
                    relaxed = True
                    logger.warning(
                        "LP solved at relaxed tolerance ±%.2f%% (configured "
                        "ceiling ±%.2f%%). Review your per-holding targets "
                        "or raise rebalancing_target_tolerance_pctg.",
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
                    "cap ±%.2f%%. Targets are too aggressive — review them.",
                    relax_cap,
                )
            else:
                logger.warning(
                    "No feasible rebalancing solution within target tolerance "
                    "±%.2f%%. At least one allocation drift exceeds the "
                    "ceiling — raise rebalancing_target_tolerance_pctg, enable "
                    "rebalancing_auto_relax, or relax per-holding targets.",
                    max_tol,
                )
        else:
            logger.info(
                "Portfolio already aligned within ±%.2f%% — no rebalancing needed.",
                max_tol,
            )

    # Distinguish two cases for the UI:
    #   1. "Already aligned" — every category is within ±tolerance of
    #      its target, so the solver returned 0 actions because none
    #      were needed. ``infeasible`` is False.
    #   2. "Infeasible" — at least one category is beyond the
    #      tolerance ceiling, so the targets cannot be reached within
    #      ``rebalancing_target_tolerance_pctg``. The solver returned 0
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

    # Verification. With the new LP formulation, every holding —
    # including the cash buffer holdings — is tradeable, so
    # ``new_values`` already reflects the post-rebal position once we
    # apply each action's amount. No special cash bookkeeping is
    # required here; the LP guarantees that the lump sum exactly
    # matches the net trade flow via the cash-flow equality.
    new_values = values.copy()
    for a in actions:
        # Use the stable holding index recorded when the action was built.
        # Keying on ticker here was O(n²) and mis-attributed buys/sells
        # when two holdings shared a ticker (e.g. two BTPs).
        idx = a["idx"]
        if a["direction"] == "buy":
            new_values[idx] += float(a["amount_eur"])
        else:
            new_values[idx] -= float(a["amount_eur"])
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
            Also defines the grid the returned value is snapped onto.
        solve: Callable ``f(tolerance)`` returning the scipy.optimize
            result (or None). The result is feasible iff
            ``result is not None and result.success``.

    Returns:
        The smallest feasible tolerance snapped UP to the next
        multiple of ``precision`` (so the caller can re-solve at a
        clean tolerance value that is guaranteed to be ≥ the binary
        search's feasible upper bound), or ``None`` if even ``hi`` is
        infeasible.
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

    # Snap UP to the next multiple of ``precision`` so the caller can
    # re-solve at a clean grid value while staying ≥ feasible_hi (which
    # is known feasible). Subtracting a tiny epsilon before the ceil
    # avoids a spurious extra step when feasible_hi already sits
    # exactly on the grid.
    snapped = precision * math.ceil(feasible_hi / precision - 1e-9)
    return round(snapped, 6)


def _max_drift_exceeds_tolerance(values, holdings, config, *,
                                 invested_value: float, eq_value: float,
                                 fi_value: float, cash_value: float) -> bool:
    """Return True if at least one allocation category has a drift
    larger than ``rebalancing_target_tolerance_pctg`` (or, for cash, a
    relative deviation greater than the same percentage).

    Used to distinguish "no actions because the portfolio is already
    aligned" from "no actions because the LP is infeasible at the
    configured tolerance ceiling". The first case should still report
    post-rebalancing values (they happen to equal current values); the
    second case should mark them as not actionable so the UI does not
    look like a rebalance was applied.
    """
    max_tol = float(config.rebalancing_target_tolerance_pctg or 0.0)
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
        actions.append({"idx": i, "name": h.name or h.ticker, "ticker": h.ticker,
                        "direction": direction, "amount_eur": round(abs(net), 2), "reason": reason})
    return actions


def _solve_lp(n, values, holdings, config, geo_frac, all_geos, eq_value, fi_value, tv,
              opt_tolerance, cash_value=0.0, invested_value=0.0,
              lump_sum=0.0, no_sell_mode=False):
    """Build and solve the MILP for a given precision.

    The optimizer treats every holding (including cash) uniformly: each
    holding has its own buy/sell variables, the cash holdings get a
    target via ``target_cash_buffer_eur`` just like any asset class
    target, and the cash flow constraint enforces strict conservation
    of the explicit ``lump_sum``. Setting cash aside for the buffer
    is therefore an ordinary BUY action on the cash instrument(s),
    visible to the user in the actions table.
    """
    tol_frac = opt_tolerance / 100.0
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    new_tv = tv + lump_sum

    # The invested portfolio (= total portfolio − cash buffer)
    # determines the EUR scale of the asset-class targets. We pin
    # ``cash_after`` to the user's buffer target since the LP will
    # actively steer the cash position there via its constraint.
    cash_after = cash_target
    invested_value_after = max(new_tv - cash_after, 0.0)

    A_eq_rows, b_eq_vals = [], []
    A_ub_rows, b_ub_vals = [], []

    # Drift-tracking specs collected while constraints are built. Each
    # entry is a tuple (coeff_2n, residual, band_eur) describing a
    # linear combination of buy/sell variables whose post-rebalance
    # value we want close to ``residual`` (in the sense
    # ``coeff_2n · vars - residual = drift_value``). The drift penalty
    # only applies *outside* a band of ``band_eur`` EUR — drifts that
    # leave the position inside the configured tolerance are free.
    # With ``drift_w == 0`` we skip the slack expansion entirely so
    # the LP behaves exactly as before.
    drift_w = float(getattr(config, "rebalancing_drift_penalty_weight", 0.0) or 0.0)
    drift_specs: list[tuple[np.ndarray, float, float]] = []

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
    # Cash flow conservation. The lump_sum is the only external cash
    # injection: every euro of it must end up *somewhere* in the
    # portfolio (invested holdings or the cash buffer). Internally the
    # net flow between holdings nets to zero. The constraint is
    # therefore a strict equality:
    #
    #     Σ buy[i] − Σ (1 - tax_i) · sell[i]
    #         + fee_buy · Σ zb[i] + fee_sell · Σ zs[i] = lump_sum
    #
    # No band, no slack. If the user gives EUR 1,000 of fresh cash, the
    # LP must allocate exactly EUR 1,000 of net trade flow.
    # ------------------------------------------------------------------
    cash_row = np.zeros(4 * n)
    cash_row[:n] = 1.0
    cash_row[n:2 * n] = -(1.0 - tax_per_unit_sold)
    cash_row[2 * n:3 * n] = fee_buy
    cash_row[3 * n:4 * n] = fee_sell
    A_eq_rows.append(cash_row)
    b_eq_vals.append(float(lump_sum))

    # ------------------------------------------------------------------
    # Cash buffer target. The cash holdings (typically a money-market
    # ETF like MONEY.MI) are treated as an ordinary asset class with
    # a symmetric tolerance band around ``target_cash_buffer_eur``.
    # No privileged "fallback" behaviour for the cash position: if
    # the LP cannot fit the lump sum into invested holdings within
    # tolerance, the auto-relax phase widens the band for every
    # target uniformly until a feasible plan exists (or the relax
    # cap is hit and the run is reported as infeasible).
    # ------------------------------------------------------------------
    cash_mask = np.array([
        1.0 if h.asset_class == AssetClass.CASH_EQUIVALENTS else 0.0
        for h in holdings
    ])
    cash_pre_total = float((cash_mask * values).sum())
    cash_band = abs(tol_frac * cash_target)
    if cash_target > 0:
        # Upper: cash_finale ≤ target + band
        row = np.zeros(2 * n)
        row[:n] = cash_mask
        row[n:] = -cash_mask
        A_ub_rows.append(row)
        b_ub_vals.append(cash_target + cash_band - cash_pre_total)
        # Lower: cash_finale ≥ target − band
        A_ub_rows.append(-row)
        b_ub_vals.append(cash_pre_total - (cash_target - cash_band))
        if drift_w > 0:
            drift_specs.append((row.copy(), cash_target - cash_pre_total, cash_band))

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
        # - target_val. Stored as (coeff_2n, residual, band_eur).
        # ``tol_val`` (= tol_frac × invested_value_after) is the EUR
        # band inside which the drift carries no penalty.
        if drift_w > 0:
            drift_specs.append((row_upper.copy(), target_val - current_sum, tol_val))

    # Geo constraints — geographic allocation within the equity sleeve.
    # The denominator (equity portfolio value after rebalance) depends
    # on the LP decisions: every euro the solver buys or sells of an
    # equity holding moves both the numerator and the denominator. We
    # therefore express each constraint as a *dynamic ratio*:
    #
    #     geo_finale ≤ (target_frac + tol_frac) × eq_finale
    #
    # which expands to a linear inequality once we substitute
    # ``geo_finale = geo_pre + Σ gf[i] × (buy[i] - sell[i])`` and
    # ``eq_finale = eq_pre + Σ is_eq[i] × (buy[i] - sell[i])``. The
    # cash flow injected by the user (``lump_sum``) lands wherever
    # the LP wants — including the cash buffer holding when nothing
    # better is available — so the equity sleeve only changes via
    # actual trades on equity holdings.
    is_eq_arr = np.array([
        1.0 if h.asset_class == AssetClass.EQUITIES else 0.0 for h in holdings
    ])
    eq_pre = float((is_eq_arr * values).sum())
    if eq_pre + lump_sum > 0:  # there's an equity sleeve to constrain
        for g_idx, geo_name in enumerate(all_geos):
            target_pct = config.equity_geo_targets_pctg.get(geo_name, 0)
            tf = target_pct / 100.0
            tol = tol_frac
            gf = geo_frac[:, g_idx]
            current_geo = float((gf * values).sum())

            # Upper bound: geo_finale ≤ (tf + tol) × eq_finale
            #   → Σ gf·(buy-sell) − (tf+tol) × Σ is_eq·(buy-sell)
            #     ≤ (tf+tol) × eq_pre − current_geo
            row_upper = np.zeros(2 * n)
            row_upper[:n] = gf - (tf + tol) * is_eq_arr
            row_upper[n:] = -(gf - (tf + tol) * is_eq_arr)
            A_ub_rows.append(row_upper)
            b_ub_vals.append((tf + tol) * eq_pre - current_geo)

            # Lower bound: geo_finale ≥ (tf − tol) × eq_finale
            #   → −Σ gf·(buy-sell) + (tf−tol) × Σ is_eq·(buy-sell)
            #     ≤ current_geo − (tf−tol) × eq_pre
            row_lower = np.zeros(2 * n)
            row_lower[:n] = -gf + (tf - tol) * is_eq_arr
            row_lower[n:] = -(-gf + (tf - tol) * is_eq_arr)
            A_ub_rows.append(row_lower)
            b_ub_vals.append(current_geo - (tf - tol) * eq_pre)

            if drift_w > 0:
                # Track signed drift: gp − gn = target_frac × eq_finale
                # − geo_finale (sign flipped from "actual − target", but
                # gp + gn captures absolute value either way).
                # Band: ``tol_frac × eq_pre`` — approximation that
                # ignores the small change in the equity sleeve due to
                # trades inside it, but produces a stable reference so
                # the LP knows when to stop penalising drift.
                coeff = np.zeros(2 * n)
                coeff[:n] = gf - tf * is_eq_arr
                coeff[n:] = -(gf - tf * is_eq_arr)
                drift_specs.append((coeff, tf * eq_pre - current_geo, tol_frac * eq_pre))

    # Per-holding equity constraints — same dynamic-ratio formulation.
    # ``target_equities`` is a percentage of the equity sleeve, so the
    # denominator is again ``eq_finale``.
    if eq_pre + lump_sum > 0:
        for i, h in enumerate(holdings):
            if h.target_equities is None or h.asset_class != AssetClass.EQUITIES:
                continue
            tf = float(h.target_equities) / 100.0
            tol = tol_frac
            holding_indicator = np.zeros(n)
            holding_indicator[i] = 1.0

            # Upper bound: h_finale ≤ (tf + tol) × eq_finale
            row_upper = np.zeros(2 * n)
            row_upper[:n] = holding_indicator - (tf + tol) * is_eq_arr
            row_upper[n:] = -(holding_indicator - (tf + tol) * is_eq_arr)
            A_ub_rows.append(row_upper)
            b_ub_vals.append((tf + tol) * eq_pre - values[i])

            # Lower bound: h_finale ≥ (tf − tol) × eq_finale
            row_lower = np.zeros(2 * n)
            row_lower[:n] = -holding_indicator + (tf - tol) * is_eq_arr
            row_lower[n:] = -(-holding_indicator + (tf - tol) * is_eq_arr)
            A_ub_rows.append(row_lower)
            b_ub_vals.append(values[i] - (tf - tol) * eq_pre)

            if drift_w > 0:
                coeff = np.zeros(2 * n)
                coeff[:n] = holding_indicator - tf * is_eq_arr
                coeff[n:] = -(holding_indicator - tf * is_eq_arr)
                drift_specs.append((coeff, tf * eq_pre - values[i], tol_frac * eq_pre))

    # Per-holding FI constraints — same dynamic-ratio formulation
    # against the fixed-income sleeve.
    is_fi_arr = np.array([
        1.0 if h.asset_class == AssetClass.FIXED_INCOME else 0.0 for h in holdings
    ])
    fi_pre = float((is_fi_arr * values).sum())
    if fi_pre + lump_sum > 0:
        for i, h in enumerate(holdings):
            if h.target_fixed_income is None or h.asset_class != AssetClass.FIXED_INCOME:
                continue
            tf = float(h.target_fixed_income) / 100.0
            tol = tol_frac
            holding_indicator = np.zeros(n)
            holding_indicator[i] = 1.0

            # Upper bound: h_finale ≤ (tf + tol) × fi_finale
            row_upper = np.zeros(2 * n)
            row_upper[:n] = holding_indicator - (tf + tol) * is_fi_arr
            row_upper[n:] = -(holding_indicator - (tf + tol) * is_fi_arr)
            A_ub_rows.append(row_upper)
            b_ub_vals.append((tf + tol) * fi_pre - values[i])

            # Lower bound: h_finale ≥ (tf − tol) × fi_finale
            row_lower = np.zeros(2 * n)
            row_lower[:n] = -holding_indicator + (tf - tol) * is_fi_arr
            row_lower[n:] = -(-holding_indicator + (tf - tol) * is_fi_arr)
            A_ub_rows.append(row_lower)
            b_ub_vals.append(values[i] - (tf - tol) * fi_pre)

            if drift_w > 0:
                coeff = np.zeros(2 * n)
                coeff[:n] = holding_indicator - tf * is_fi_arr
                coeff[n:] = -(holding_indicator - tf * is_fi_arr)
                drift_specs.append((coeff, tf * fi_pre - values[i], tol_frac * fi_pre))

    # MILP binary linking. zb[i]=1 if buying i, zs[i]=1 if selling i.
    # The binaries are needed so the cash-flow constraint can charge a
    # fixed buy/sell fee per executed trade (fee enters as
    # ``fee_buy * zb[i] + fee_sell * zs[i]``). They are also used to
    # enforce mutual exclusion (no holding can be simultaneously
    # bought and sold in the same plan).
    #
    # We link the binaries to the trade variables in *both*
    # directions:
    #
    #   * upper:  buy[i] ≤ M · zb[i]   — zb must be 1 when we trade
    #   * lower:  buy[i] ≥ ε · zb[i]   — zb cannot be 1 with no trade
    #
    # The lower bound stops the solver from "padding" the cash-flow
    # constraint by raising idle binaries: every euro of fee in the
    # cash equation has to be balanced by at least ``ε`` euros of
    # actual trade volume, so spurious zb=1 / zs=1 with zero trade
    # are no longer optimal.
    #
    # Variables: [buy_0..n-1, sell_0..n-1, zb_0..n-1, zs_0..n-1]  (4n).
    M = new_tv
    eps = 1.0  # EUR — minimum trade for a binary to flip on
    for i in range(n):
        # Buy upper link: buy[i] - M*zb[i] ≤ 0
        row = np.zeros(4 * n); row[i] = 1.0; row[2 * n + i] = -M
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Buy lower link: ε*zb[i] - buy[i] ≤ 0
        row = np.zeros(4 * n); row[i] = -1.0; row[2 * n + i] = eps
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Sell upper link: sell[i] - value[i]*zs[i] ≤ 0
        row = np.zeros(4 * n); row[n + i] = 1.0; row[3 * n + i] = -float(values[i])
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Sell lower link: ε*zs[i] - sell[i] ≤ 0
        row = np.zeros(4 * n); row[n + i] = -1.0; row[3 * n + i] = eps
        A_ub_rows.append(row); b_ub_vals.append(0.0)
        # Mutual exclusion: zb[i] + zs[i] <= 1
        row = np.zeros(4 * n); row[2 * n + i] = 1.0; row[3 * n + i] = 1.0
        A_ub_rows.append(row); b_ub_vals.append(1.0)

    # Pad existing constraint rows up to the full variable width.
    # Layout:
    #   [buy(n), sell(n), zb(n), zs(n),
    #    over_pos(K), over_neg(K),    ← out-of-band overshoot slacks
    #    abs_pos(K),  abs_neg(K),     ← absolute-drift slacks (= |drift|)
    #    u(K)]                        ← quadratic envelope u_k ≈ |drift_k|²
    # where K is the number of drift specs collected. With
    # ``drift_w == 0`` we skip the slack expansion entirely (K = 0)
    # so the LP is identical to the previous formulation.
    K = len(drift_specs)
    total_vars = 4 * n + 5 * K
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

    # Drift-tracking inequalities. For each spec
    # ``(coeff_2n, residual, band_eur)`` we want to penalise
    # ``max(0, |drift_value| - band_eur)`` where
    # ``drift_value = coeff_2n · vars - residual``. We linearise this
    # by introducing two non-negative slacks per spec and adding two
    # one-sided inequalities so each slack captures the respective
    # overshoot:
    #
    #     coeff_2n · vars - over_pos_k ≤ residual + band_eur     (positive)
    #     -coeff_2n · vars - over_neg_k ≤ -residual + band_eur   (negative)
    #
    # When the LP minimises ``λ · (over_pos_k + over_neg_k)`` the
    # slacks are 0 whenever the drift is inside ``± band_eur``, and
    # otherwise carry the EUR amount the position is sticking out of
    # the band. With ``drift_w == 0`` the slacks are unconstrained at
    # zero cost so the LP behaves like the legacy minimum-trading
    # solver.
    #
    # We also linearise ``|drift_value|`` itself (without any band)
    # via two extra non-negative slacks ``abs_pos_k`` / ``abs_neg_k``
    # which act as a tie-breaker: when several plans would all keep
    # every position inside the tolerance band, the LP prefers the
    # one that minimises *total* drift, not just out-of-band drift.
    # The in-band penalty weight is set to a small fraction of
    # ``drift_w`` so out-of-band drift still dominates.
    for k, (coeff_2n, residual, band_eur) in enumerate(drift_specs):
        # Positive overshoot: drift > band  →  drift - over_pos ≤ band
        row = np.zeros(total_vars)
        row[:2 * n] = coeff_2n
        row[4 * n + k] = -1.0
        padded_ub.append(row)
        b_ub_vals.append(residual + band_eur)
        # Negative overshoot: drift < -band  →  -drift - over_neg ≤ band
        row = np.zeros(total_vars)
        row[:2 * n] = -coeff_2n
        row[4 * n + K + k] = -1.0
        padded_ub.append(row)
        b_ub_vals.append(-residual + band_eur)
        # Absolute drift (no band): drift ≤ abs_pos
        row = np.zeros(total_vars)
        row[:2 * n] = coeff_2n
        row[4 * n + 2 * K + k] = -1.0
        padded_ub.append(row)
        b_ub_vals.append(residual)
        # Absolute drift (no band): -drift ≤ abs_neg
        row = np.zeros(total_vars)
        row[:2 * n] = -coeff_2n
        row[4 * n + 3 * K + k] = -1.0
        padded_ub.append(row)
        b_ub_vals.append(-residual)
        # Quadratic envelope on |drift| via tangent lines at
        # breakpoints b_m: we want u_k ≈ |drift_k|², achieved by
        # constraining u_k from below with the tangent line of x²
        # at each breakpoint. For x ≥ 0:
        #
        #     u_k ≥ 2·b_m · x - b_m²    for every breakpoint b_m
        #     where x = abs_pos_k + abs_neg_k = |drift_k|
        #
        # The LP minimises u_k subject to these, so u_k tracks the
        # upper envelope of the tangents — a piecewise-linear convex
        # under-approximation of x² that gets tighter with more
        # breakpoints. The breakpoints are anchored around the
        # tolerance band so the approximation is precise where it
        # matters (on the spectrum from "in-band" to "out-of-band").
        # When ``band_eur`` is zero (e.g. cash target = 0) we skip
        # the envelope; the `abs_pos`/`abs_neg` slacks are still
        # constrained but contribute nothing to the objective.
        if band_eur > 0:
            for fraction in (0.5, 1.0, 2.0, 4.0):
                b_m = band_eur * fraction
                row = np.zeros(total_vars)
                row[4 * n + 2 * K + k] = 2.0 * b_m
                row[4 * n + 3 * K + k] = 2.0 * b_m
                row[4 * n + 4 * K + k] = -1.0
                padded_ub.append(row)
                b_ub_vals.append(b_m * b_m)

    # Objective: minimise total trade volume, optionally penalising
    # the EUR drift that leaves the tolerance band of each tracked
    # target. Friction costs (fees and capital-gains taxes) are
    # folded into the cash-flow constraint rather than the objective
    # so we don't double-count their effect — a sell of a profitable
    # position has to be larger to fund the same buy because part of
    # the proceeds goes to the tax authority, and each transaction
    # reduces the available cash by the fixed fee.
    #
    # The drift penalty makes the LP prefer plans that stay inside
    # every category's tolerance band rather than pushing them to the
    # band edge. Drift inside the band is free; drift outside the
    # band costs ``drift_penalty_weight`` per leftover EUR. With
    # weight 0 the solver behaves like a pure minimum-trading
    # optimiser; with weight 1 a EUR of out-of-band drift costs the
    # same as a EUR of trade volume; higher values close out-of-band
    # drift more aggressively at the price of more trades.
    #
    # An additional in-band drift penalty acts as a tie-breaker. When
    # the LP has multiple plans that all stay inside every tolerance
    # band, it picks the one that *also* minimises total drift across
    # all categories. The penalty is QUADRATIC in the residual drift,
    # implemented via the ``u_k`` slacks bounded below by the upper
    # envelope of tangent lines to ``x²``. Quadratic shape gives a
    # gradient of ``2·|drift|`` so the LP prefers closing 1 EUR off a
    # large drift over 1 EUR off a small one — exactly what a human
    # rebalancer would do. Without this term the cash-flow equality
    # (which forces ``Σ buy = lump_sum``) leaves several optima of
    # equal value, and the solver may emit fragmented plans like
    # "BUY XS5E 999 + BUY AGGH 1" or pick an asset that fails to
    # close the dominant drift. The in-band weight is a fraction
    # of ``drift_w`` so out-of-band drift always dominates the
    # decision; in-band drift only breaks ties.
    c = np.zeros(total_vars)
    c[:2 * n] = 1.0  # buy and sell penalised at unit weight (volume)
    # Per-trade fixed cost (epsilon × binary). Acts as a second
    # tie-breaker against fragmented plans: any extra holding the LP
    # adds to the plan pays a tiny "activation" cost regardless of
    # its trade size, so a 1 EUR top-up that does not improve the
    # main objective is no longer optimal. The cost is small enough
    # that genuinely useful trades (≥ a few EUR of drift improvement)
    # are still preferred over the no-op plan.
    binary_activation_cost = 0.01
    c[2 * n:4 * n] = binary_activation_cost
    # Note: we deliberately do NOT charge a per-trade *fee* in the
    # objective. The fee is already a real cash outflow captured in
    # the cash-flow equality (every executed trade subtracts
    # ``fee_buy`` or ``fee_sell`` from the cash buffer), and adding
    # the same fee a second time in the objective would over-
    # penalise legitimate sells — making the LP avoid trades that
    # are economically necessary to close drifts. The reverse
    # binary linking (``buy[i] >= eps · zb[i]``) prevents the LP
    # from flipping ``zb_i`` / ``zs_i`` to 1 with zero trade.
    if K > 0:
        # Out-of-band overshoot: full drift_w weight, linear in EUR.
        c[4 * n:4 * n + K] = drift_w        # over_pos_k
        c[4 * n + K:4 * n + 2 * K] = drift_w  # over_neg_k
        # In-band quadratic envelope: small tie-breaker weight.
        # ``u_k ≈ |drift_k|²`` so this contributes ``inband_w · drift²``
        # to the objective. The unit of u_k is EUR² which makes the
        # raw weight tiny in absolute terms, so we scale it back by a
        # representative band size to keep the magnitude comparable
        # to the linear penalties — a 1 EUR drift inside a 100 EUR
        # band contributes roughly the same as the linear penalty
        # would have.
        # Use the smallest non-zero band as the reference scale; for
        # typical portfolios all bands are within an order of
        # magnitude so the choice is not critical.
        bands = [b for (_, _, b) in drift_specs if b > 0]
        ref_band = min(bands) if bands else 1.0
        inband_w = 0.1 * drift_w / max(ref_band, 1.0)
        c[4 * n + 4 * K:4 * n + 5 * K] = inband_w  # u_k

    # Bounds
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, np.inf)
    ub[n:2*n] = values      # sell <= current value
    ub[2*n:3*n] = 1.0       # zb <= 1
    ub[3*n:4*n] = 1.0       # zs <= 1
    # over_pos_k and over_neg_k are non-negative slacks with no
    # upper bound; the inequalities above keep them as small as
    # the LP can manage.

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
    if no_sell_mode:
        for i in range(n):
            ub[n + i] = 0.0      # sell[i] = 0
            ub[3*n + i] = 0.0    # zs[i] = 0

    constraints = []
    if padded_eq:
        constraints.append(
            LinearConstraint(np.array(padded_eq), np.array(b_eq_vals), np.array(b_eq_vals))
        )
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
    tol = float(config.rebalancing_target_tolerance_pctg)

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

    # Cash buffer (absolute EUR). The cash position is allowed the
    # same tolerance everyone else gets, so the verification only
    # flags ``PARTIAL`` when the deviation is beyond
    # ``rebalancing_target_tolerance_pctg`` of the cash target.
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    cash_delta_eur = cash_new - cash_target
    cash_band_eur = abs(tol / 100.0 * cash_target)
    cash_ok = abs(cash_delta_eur) <= max(cash_band_eur, 1.0)
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
