"""Unified rebalancing engine (local search).

Suggests buy/sell actions that reduce the distance to target in EVERY ambit —
asset-class allocation, equity geography, per-holding targets and the cash
buffer — *proportionally*, instead of fully closing some targets while leaving
others far out of tolerance.

Why local search (this replaced the former scipy MILP): the goal is a
non-linear, multi-objective fairness criterion — maximise the *smallest*
per-ambit fractional gap reduction — which a linear/MILP objective cannot
express. The MILP minimised trade volume subject to hard tolerance bands and,
when infeasible, relaxed every band uniformly; inside the widened band it had
no incentive to shrink the smaller gaps, so whole ambits (e.g. geography) were
left untouched. The local search optimises the fairness objective directly.

Algorithm: Iterated Local Search.
    * Local move = adjust ONE holding's signed trade by ±δ over a geometric
      ladder of step sizes (best-improvement coordinate descent), with a short
      anti-reversal tabu list.
    * ILS wraps the descent with random multi-coordinate kicks and keeps the
      best basin found (deterministic via a fixed seed).
    * The budget is enforced softly by a residual penalty during search, then
      projected to an exact balance.

Objective shapes (LSParams.objective): "ambit_maximin" (default) maximises the
smallest per-ambit fractional reduction so every ambit is addressed;
"ambit_sq" spreads effort convexly; "sum_sq"/"minimax" operate per objective.
A guard term forbids pushing any objective past max(its initial gap, tolerance)
— so an in-tolerance ambit (e.g. the cash buffer) is never used as a dumping
ground and out-of-tolerance ones never get worse.

Constraints respected (identical to the former optimizer):
    * ``no_buy_no_sell`` holdings are frozen (trade fixed at 0).
    * ``rebalancing_no_sell`` forbids every sell.
    * a holding cannot be sold for more than its current value.
    * cash-flow conservation: net trade flow (net of CGT and fees) equals the
      lump sum (0 for a pure rebalance).
    * capital-gains tax (standard / government rate) and fixed buy/sell fees
      reduce the cash a sell frees up.

Public API (unchanged): ``compute_unified_rebalancing(holdings, config,
total_value, lump_sum) -> (actions, verifications)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_unified_rebalancing(
    holdings: list[Holding],
    config: InvestorConfig,
    total_value: float,
    lump_sum: Optional[float] = None,
) -> tuple[list[dict], list[dict]]:
    """Compute rebalancing actions via local search.

    The invested allocation targets apply to the invested portfolio (total
    minus cash); the cash position has its own absolute target
    ``target_cash_buffer_eur``. Returns ``(actions, verifications)`` where each
    action is ``{idx, name, ticker, direction, amount_eur, reason}`` and the
    verifications mirror the post-rebalance allocation per ambit.
    """
    return optimize_local_search(holdings, config, total_value, lump_sum=lump_sum)


def compute_drift_penalty_sensitivity(
    holdings: list[Holding],
    config: InvestorConfig,
    total_value: float,
    lump_sum: Optional[float] = None,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """Retired. The drift-penalty sweep was an artifact of the MILP objective
    (it varied ``rebalancing_drift_penalty_weight``); the local search has no
    such knob, so there is nothing meaningful to sweep. Returns an empty list,
    which makes the Excel/newsletter "tuning" section hide itself.
    """
    return []


# ---------------------------------------------------------------------------
# Verification of a rebalancing plan (shared output contract)
# ---------------------------------------------------------------------------

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
                          "status": "\u2713 OK" if max_ac <= tol else "\u26a0 PARTIAL",
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
                          "status": "\u2713 OK" if max_geo <= tol else "\u26a0 PARTIAL",
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
        # ``ticker`` is included so renderers can correlate this entry with the
        # pre-rebalancing snapshot in ``holdings_df`` (two holdings can share a
        # name, so name alone is not a stable key).
        ph_items.append({"category": h.name or h.ticker,
                         "ticker": h.ticker,
                         "actual_pct": actual,
                         "target_pct": float(h.target_equities)})
    verifications.append({"check": "Per-Holding Equity Targets", "kind": "per_holding_equity",
                          "status": "\u2713 OK" if max_ph <= tol else "\u26a0 PARTIAL",
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
        fi_items.append({"category": h.name or h.ticker,
                         "ticker": h.ticker,
                         "actual_pct": actual,
                         "target_pct": float(h.target_fixed_income)})
    verifications.append({"check": "Per-Holding FI Targets", "kind": "per_holding_fi",
                          "status": "\u2713 OK" if max_fi <= tol else "\u26a0 PARTIAL",
                          "detail": ", ".join(fi_details) or "No targets set",
                          "items": fi_items})

    # Cash buffer (absolute EUR). The cash position is allowed the same
    # tolerance everyone else gets.
    cash_target = float(config.target_cash_buffer_eur or 0.0)
    cash_delta_eur = cash_new - cash_target
    cash_band_eur = abs(tol / 100.0 * cash_target)
    cash_ok = abs(cash_delta_eur) <= max(cash_band_eur, 1.0)
    verifications.append({
        "check": "Cash & Cash Equivalents", "kind": "cash",
        "status": "\u2713 OK" if cash_ok else "\u26a0 PARTIAL",
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


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class LSParams:
    """Tunables for the local search. Defaults favour proportional, every-ambit
    gap reduction with light churn control."""

    # Objective shape:
    #   "ambit_maximin" — maximise the smallest per-ambit fractional gap
    #       reduction (asset / geo / per-holding treated as equal ambits).
    #       Guarantees EVERY ambit gets attention. DEFAULT.
    #   "ambit_sq"      — Σ_ambit (remaining_gap_fraction)², convex → spreads.
    #   "sum_sq"        — Σ_objective (gap/scale)² (can ignore a whole ambit).
    #   "minimax"       — worst single normalised gap.
    objective: str = "ambit_maximin"
    # Per-objective gap scale: "initial" (proportional), "tolerance", "absolute".
    gap_norm: str = "initial"
    # Kept for the object-level "minimax" mode only.
    minimax: bool = False
    # Floor for the "initial" scale so a near-on-target objective (tiny gap0)
    # does not dominate by division. In pp.
    initial_floor_pp: float = 1.0
    # Light regularisers (kept small so gap reduction dominates).
    turnover_weight: float = 0.02      # cost per unit of (turnover / total_value)
    trade_count_weight: float = 0.04   # cost per executed trade
    residual_weight: float = 25.0      # penalty on (unbalanced cash / tv * 100)²
    guard_weight: float = 2.0          # penalty on pushing any objective past max(initial gap, tol)
    # Search control.
    n_restarts: int = 60
    perturbation_strength: int = 3     # number of random kicks per restart
    tabu_tenure: int = 5
    min_trade_eur: float = 1.0         # trades below this are dropped
    max_iters_per_descent: int = 400
    seed: int = 0
    # Step-size ladder as fractions of total value (coarse → fine).
    step_fractions: tuple = (0.05, 0.02, 0.01, 0.005, 0.0025, 0.001)


# ---------------------------------------------------------------------------
# Objective model (vectorised gap computation)
# ---------------------------------------------------------------------------

class _ObjectiveModel:
    """Pre-computes the static structure needed to evaluate, for any trade
    vector ``t``, the per-objective deviation from target (in pp), mirroring
    the ``_verify`` math."""

    def __init__(self, holdings: list[Holding], config: InvestorConfig,
                 values: np.ndarray):
        self.holdings = holdings
        self.config = config
        self.values = values
        n = len(holdings)
        self.n = n
        self.tol = float(config.rebalancing_target_tolerance_pctg or 2.0)

        cls = [(h.asset_class.value if h.asset_class else "Alternative") for h in holdings]
        self.noncash_mask = np.array(
            [0.0 if c == AssetClass.CASH_EQUIVALENTS.value else 1.0 for c in cls]
        )
        self.eq_mask = np.array(
            [1.0 if c == AssetClass.EQUITIES.value else 0.0 for c in cls]
        )
        self.fi_mask = np.array(
            [1.0 if c == AssetClass.FIXED_INCOME.value else 0.0 for c in cls]
        )
        self.cash_mask = np.array(
            [1.0 if c == AssetClass.CASH_EQUIVALENTS.value else 0.0 for c in cls]
        )

        # Asset-class objectives (% of invested portfolio).
        self.ac_masks, self.ac_targets, self.ac_keys = [], [], []
        for ac, target in (config.invested_allocation_targets_pctg or {}).items():
            self.ac_masks.append(np.array([1.0 if c == ac else 0.0 for c in cls]))
            self.ac_targets.append(float(target))
            self.ac_keys.append(ac)
        self.ac_masks = np.array(self.ac_masks) if self.ac_masks else np.zeros((0, n))
        self.ac_targets = np.array(self.ac_targets)

        # Equity geography objectives (% of equity sleeve).
        self.all_geos = sorted((config.equity_geo_targets_pctg or {}).keys())
        self.geo_frac = np.zeros((n, len(self.all_geos)))
        for i, h in enumerate(holdings):
            if h.asset_class != AssetClass.EQUITIES:
                continue
            if h.geo_breakdown:
                tot = sum(h.geo_breakdown.values())
                if tot > 0:
                    for g, p in h.geo_breakdown.items():
                        gn = g.value if hasattr(g, "value") else str(g)
                        if gn in self.all_geos:
                            self.geo_frac[i][self.all_geos.index(gn)] = p / tot
            elif h.geography:
                gn = h.geography.value if hasattr(h.geography, "value") else str(h.geography)
                if gn in self.all_geos:
                    self.geo_frac[i][self.all_geos.index(gn)] = 1.0
        self.geo_targets = np.array(
            [float(config.equity_geo_targets_pctg.get(g, 0.0)) for g in self.all_geos]
        )

        # Per-holding equity / FI objectives.
        self.ph_eq_idx = [i for i, h in enumerate(holdings)
                          if h.target_equities is not None and h.asset_class == AssetClass.EQUITIES]
        self.ph_eq_tgt = np.array([float(holdings[i].target_equities) for i in self.ph_eq_idx])
        self.ph_fi_idx = [i for i, h in enumerate(holdings)
                          if h.target_fixed_income is not None and h.asset_class == AssetClass.FIXED_INCOME]
        self.ph_fi_tgt = np.array([float(holdings[i].target_fixed_income) for i in self.ph_fi_idx])

        self.cash_target = float(config.target_cash_buffer_eur or 0.0)

        # Stable label list parallel to the gap vector (for reporting / ambits).
        self.labels = (
            [("asset", k) for k in self.ac_keys]
            + [("geo", g) for g in self.all_geos]
            + [("ph_eq", holdings[i].ticker) for i in self.ph_eq_idx]
            + [("ph_fi", holdings[i].ticker) for i in self.ph_fi_idx]
            + ([("cash", "Cash & Cash Equivalents")] if self.cash_target > 0 else [])
        )
        # Ambit grouping: the macro-areas the user cares about. Per-holding
        # equity and FI are one "per_holding" ambit so a single unfixable
        # holding does not sink the whole ambit.
        self.ambit_of = []
        for kind, _ in self.labels:
            self.ambit_of.append("per_holding" if kind in ("ph_eq", "ph_fi") else kind)
        self.targets = np.concatenate([
            self.ac_targets, self.geo_targets, self.ph_eq_tgt, self.ph_fi_tgt,
            (np.array([self.cash_target]) if self.cash_target > 0 else np.zeros(0)),
        ])

    def gaps(self, new_values: np.ndarray) -> np.ndarray:
        """Signed deviations (pp) for every objective, in ``labels`` order.
        The cash entry is a relative deviation (% of the cash target)."""
        inv = max(float(self.noncash_mask @ new_values), 1e-9)
        eq = max(float(self.eq_mask @ new_values), 1e-9)
        fi = max(float(self.fi_mask @ new_values), 1e-9)

        parts = []
        if self.ac_masks.shape[0]:
            parts.append((self.ac_masks @ new_values) / inv * 100.0 - self.ac_targets)
        if len(self.all_geos):
            geo_actual = (self.geo_frac.T @ new_values) / eq * 100.0
            parts.append(geo_actual - self.geo_targets)
        if self.ph_eq_idx:
            parts.append(new_values[self.ph_eq_idx] / eq * 100.0 - self.ph_eq_tgt)
        if self.ph_fi_idx:
            parts.append(new_values[self.ph_fi_idx] / fi * 100.0 - self.ph_fi_tgt)
        if self.cash_target > 0:
            cash_actual = float(self.cash_mask @ new_values)
            parts.append(np.array([(cash_actual - self.cash_target) / self.cash_target * 100.0]))
        return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------------
# Cash / friction accounting
# ---------------------------------------------------------------------------

def _tax_per_unit_sold(holdings: list[Holding], config: InvestorConfig) -> np.ndarray:
    cg_std = float(config.rebalancing_capital_gains_tax_standard_pctg or 0.0) / 100.0
    cg_gov = float(config.rebalancing_capital_gains_tax_government_pctg or 0.0) / 100.0
    tax = np.zeros(len(holdings))
    for i, h in enumerate(holdings):
        gp = float(h.gain_pct or 0.0)
        if gp <= 0:
            continue
        instr = (h.instrument_type or "").lower()
        rate = cg_gov if "government bond" in instr else cg_std
        tax[i] = rate * (gp / 100.0)
    return tax


def _net_cash_cost(t: np.ndarray, tax: np.ndarray, fee_buy: float, fee_sell: float,
                   min_trade: float) -> float:
    """Net external cash the plan consumes: buys + fees − (sell proceeds net of
    CGT). Must equal the lump sum (0 for a pure rebalance)."""
    buys = np.clip(t, 0, None)
    sells = np.clip(-t, 0, None)
    traded_buy = t >= min_trade
    traded_sell = t <= -min_trade
    proceeds = float((sells * (1.0 - tax)).sum())
    fees = fee_buy * int(traded_buy.sum()) + fee_sell * int(traded_sell.sum())
    return float(buys.sum()) - proceeds + fees


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

class _Cost:
    def __init__(self, model: _ObjectiveModel, values: np.ndarray, config: InvestorConfig,
                 lump_sum: float, params: LSParams):
        self.model = model
        self.values = values
        self.tv = float(values.sum())
        self.params = params
        self.lump_sum = float(lump_sum or 0.0)
        self.tax = _tax_per_unit_sold(model.holdings, config)
        self.fee_buy = float(config.rebalancing_transaction_fee_buy_eur or 0.0)
        self.fee_sell = float(config.rebalancing_transaction_fee_sell_eur or 0.0)

        # Per-objective scale s_k.
        gap0 = np.abs(model.gaps(values))
        if params.gap_norm == "initial":
            self.scale = np.maximum(gap0, params.initial_floor_pp)
        elif params.gap_norm == "tolerance":
            self.scale = np.full_like(gap0, max(model.tol, 1e-6))
        else:  # absolute
            self.scale = np.ones_like(gap0)
        self.gap0 = gap0

        # Ambit groups: indices per ambit, restricted to ambits that have at
        # least one objective initially OUT of tolerance (so we only push
        # ambits that actually need work). base_a = Σ gap0² over the ambit.
        self.ambits: list[tuple[str, np.ndarray, float]] = []
        ambit_names = []
        for a in model.ambit_of:
            if a not in ambit_names:
                ambit_names.append(a)
        for a in ambit_names:
            idx = np.array([i for i, am in enumerate(model.ambit_of) if am == a])
            if not len(idx):
                continue
            if gap0[idx].max() <= model.tol + 1e-9:
                continue  # ambit already within tolerance → don't force it
            base = float(np.sum(gap0[idx] ** 2)) or 1.0
            self.ambits.append((a, idx, base))

    def gap_term(self, g: np.ndarray) -> float:
        mode = self.params.objective
        if mode in ("ambit_maximin", "ambit_sq") and self.ambits:
            fracs = np.array([float(np.sum(g[idx] ** 2)) / base
                              for _a, idx, base in self.ambits])  # cur_a / base_a
            if mode == "ambit_sq":
                # Convex per-ambit → diminishing returns spread effort across
                # ambits (every ambit gets touched).
                return float(np.sum(fracs ** 2))
            # maximin on reduction = minimax on remaining fraction. Scaled so
            # it dominates the light regularisers; mean term breaks ties and
            # gives the non-binding ambits a gradient.
            return float(10.0 * (fracs.max() + 0.05 * fracs.mean()))
        if mode == "minimax" or self.params.minimax:
            ng = g / self.scale
            return float(ng.max() + 1e-3 * (ng ** 2).sum())
        ng = g / self.scale  # sum_sq
        return float((ng ** 2).sum())

    def __call__(self, t: np.ndarray) -> float:
        p = self.params
        nv = self.values + t
        g = np.abs(self.model.gaps(nv))
        j = self.gap_term(g)
        # Guard: never push ANY objective past max(its initial gap, tolerance).
        # Protects in-tolerance objectives (e.g. the cash buffer) from being
        # used as a dumping ground, and stops out-of-tol objectives from
        # getting worse.
        allowed = np.maximum(self.gap0, self.model.tol)
        excess = np.clip(g - allowed, 0.0, None)
        j += p.guard_weight * float(np.sum(excess ** 2))
        turnover = float(np.abs(t).sum())
        n_trades = int((np.abs(t) >= p.min_trade_eur).sum())
        residual = _net_cash_cost(t, self.tax, self.fee_buy, self.fee_sell, p.min_trade_eur) - self.lump_sum
        j += p.turnover_weight * (turnover / max(self.tv, 1e-9))
        j += p.trade_count_weight * n_trades
        j += p.residual_weight * (residual / max(self.tv, 1e-9) * 100.0) ** 2
        return j


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _bounds(holdings: list[Holding], values: np.ndarray, no_sell: bool):
    """Return (lo, hi, tradeable) arrays for the signed trade vector t."""
    n = len(holdings)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    tradeable = np.ones(n, dtype=bool)
    for i, h in enumerate(holdings):
        if h.no_buy_no_sell:
            lo[i] = hi[i] = 0.0
            tradeable[i] = False
            continue
        lo[i] = 0.0 if no_sell else -float(values[i])  # cannot sell more than held
    return lo, hi, tradeable


def _descent(t: np.ndarray, cost: _Cost, lo, hi, tradeable, steps, params: LSParams):
    """Best-improvement coordinate descent with a short anti-reversal tabu."""
    t = t.copy()
    cur = cost(t)
    tabu: dict[tuple[int, int], int] = {}  # (idx, sign) -> expiry iteration
    for it in range(params.max_iters_per_descent):
        best_gain = 1e-12
        best_move = None
        for i in np.where(tradeable)[0]:
            for s in steps:
                for sign in (+1.0, -1.0):
                    if tabu.get((int(i), int(sign)), -1) > it:
                        continue
                    new_ti = min(max(t[i] + sign * s, lo[i]), hi[i])
                    if new_ti == t[i]:
                        continue
                    old = t[i]
                    t[i] = new_ti
                    c = cost(t)
                    t[i] = old
                    gain = cur - c
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (int(i), new_ti, int(sign))
        if best_move is None:
            break
        i, new_ti, sign = best_move
        t[i] = new_ti
        cur = cost(t)
        # Tabu the reverse direction on this holding for a few iterations.
        tabu[(i, -sign)] = it + params.tabu_tenure
    return t, cur


def _perturb(t: np.ndarray, lo, hi, tradeable, tv: float, rng, strength: int):
    """Random multi-coordinate kick to escape the current basin."""
    t = t.copy()
    idxs = np.where(tradeable)[0]
    if len(idxs) == 0:
        return t
    for _ in range(strength):
        i = int(rng.choice(idxs))
        amt = tv * float(rng.choice([0.08, 0.05, 0.03, 0.02])) * float(rng.choice([1.0, -1.0]))
        t[i] = min(max(t[i] + amt, lo[i]), hi[i])
    return t


def _project_to_budget(t: np.ndarray, cost: _Cost, model: _ObjectiveModel,
                       lo, hi, params: LSParams) -> np.ndarray:
    """Make the net cash flow equal the lump sum exactly. Balance on NON-cash
    trades first (growing/trimming the largest buys) so the cash buffer is not
    used as a dumping ground; fall back to the cash holding only if nothing
    else can absorb the residual."""
    t = t.copy()
    cash_idx = set(int(i) for i in np.where(model.cash_mask > 0)[0])

    def resid() -> float:
        return _net_cash_cost(t, cost.tax, cost.fee_buy, cost.fee_sell,
                              params.min_trade_eur) - cost.lump_sum

    def absorb(order_idx, sign):
        # sign=+1: reduce buys to remove positive residual; sign=-1: grow buys
        # to remove negative residual.
        nonlocal t
        guard = 0
        r = resid()
        while ((sign > 0 and r > 0.5) or (sign < 0 and r < -0.5)) and guard < 4000:
            moved = False
            for i in order_idx:
                if sign > 0 and t[i] > 0:           # trim a buy
                    cut = min(t[i], r)
                    t[i] -= cut
                    moved = True
                elif sign < 0:                       # grow a buy
                    room = hi[i] - t[i]
                    if room > 1e-6:
                        add = min(room, -r)
                        t[i] += add
                        moved = True
                r = resid()
                if (sign > 0 and r <= 0.5) or (sign < 0 and r >= -0.5):
                    break
            guard += 1
            if not moved:
                break

    r = resid()
    if abs(r) < 0.5:
        return t
    noncash_buys = [i for i in np.argsort(-t) if i not in cash_idx and t[i] > 0]
    noncash_any = [int(i) for i in np.argsort(-(hi - t)) if i not in cash_idx]
    if r > 0:
        absorb(noncash_buys + sorted(cash_idx), +1)
    else:
        absorb(noncash_any + sorted(cash_idx), -1)
    return t


def optimize_local_search(
    holdings: list[Holding],
    config: InvestorConfig,
    total_value: float,
    lump_sum: Optional[float] = None,
    params: Optional[LSParams] = None,
) -> tuple[list[dict], list[dict]]:
    """Local-search rebalancer. Returns ``(actions, verifications)``."""
    params = params or LSParams()
    n = len(holdings)
    if n == 0:
        return [], []

    values = np.array([
        (h.current_value if h.current_value else h.market_value_eur) or 0.0
        for h in holdings
    ], dtype=float)
    if values.sum() <= 0:
        return [], []

    model = _ObjectiveModel(holdings, config, values)
    cost = _Cost(model, values, config, float(lump_sum or 0.0), params)
    lo, hi, tradeable = _bounds(holdings, values, bool(config.rebalancing_no_sell))
    tv = float(values.sum())
    steps = [tv * f for f in params.step_fractions]
    rng = np.random.default_rng(params.seed)

    # Initial descent from "no trades".
    t = np.zeros(n)
    best, best_cost = _descent(t, cost, lo, hi, tradeable, steps, params)

    # Iterated Local Search.
    for _ in range(params.n_restarts):
        kicked = _perturb(best, lo, hi, tradeable, tv, rng, params.perturbation_strength)
        cand, cand_cost = _descent(kicked, cost, lo, hi, tradeable, steps, params)
        if cand_cost < best_cost - 1e-9:
            best, best_cost = cand, cand_cost

    best = _project_to_budget(best, cost, model, lo, hi, params)

    # Drop sub-threshold trades.
    best = np.where(np.abs(best) >= params.min_trade_eur, best, 0.0)

    actions = _extract_actions(best, holdings, model, values)
    new_values = values + best
    verifications = _verify(new_values, holdings, config, model.geo_frac,
                            model.all_geos, float((model.fi_mask * values).sum()))
    return actions, verifications


def _extract_actions(t: np.ndarray, holdings: list[Holding], model: _ObjectiveModel,
                     values: np.ndarray) -> list[dict]:
    actions = []
    for i, h in enumerate(holdings):
        amt = float(t[i])
        if abs(amt) < 1.0:
            continue
        actions.append({
            "idx": i,
            "name": h.name or h.ticker,
            "ticker": h.ticker,
            "direction": "buy" if amt > 0 else "sell",
            "amount_eur": round(abs(amt), 2),
            "reason": _reason(i, h, model, values + t),
        })
    # Largest trades first (matches the display ordering).
    actions.sort(key=lambda a: -a["amount_eur"])
    return actions


def _reason(i: int, h: Holding, model: _ObjectiveModel, new_values: np.ndarray) -> str:
    """Short 'why' string: which objective(s) this holding's trade serves."""
    bits = []
    ac = h.asset_class.value if h.asset_class else "Alternative"
    if ac in model.ac_keys:
        idx = model.ac_keys.index(ac)
        inv = max(float(model.noncash_mask @ new_values), 1e-9)
        actual = float(model.ac_masks[idx] @ new_values) / inv * 100.0
        bits.append(f"{ac} {actual:.1f}% vs {model.ac_targets[idx]:.0f}%")
    if h.asset_class == AssetClass.EQUITIES:
        eq = max(float(model.eq_mask @ new_values), 1e-9)
        for g_idx, gn in enumerate(model.all_geos):
            if model.geo_frac[i][g_idx] > 0.1:
                actual = float(model.geo_frac[:, g_idx] @ new_values) / eq * 100.0
                tgt = model.geo_targets[g_idx]
                if abs(actual - tgt) > 0.5:
                    bits.append(f"{gn} {actual:.1f}% vs {tgt:.0f}%")
    return "; ".join(bits[:3]) or "Proportional gap reduction"
