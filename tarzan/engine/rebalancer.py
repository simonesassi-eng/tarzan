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
      For proved account CASH this also freezes its individual balance:
      neither purchases nor contributions/proceeds debit or credit that row.
      Its existing value still counts toward wealth and the strategic reserve;
      any new free settlement balance uses unlocked CASH rows or a plan ledger.
    * ``rebalancing_no_sell`` forbids every sell.
    * a holding cannot be sold for more than its current value.
    * cash-flow conservation: purchases are funded by proved account cash,
      the contribution and explicit sells (net of CGT and fees).
    * a cash-equivalent security is not an account balance; spending its value
      requires a real security sell, while the combined reserve stays protected.
    * capital-gains tax (standard / government rate) and fixed buy/sell fees
      reduce the cash a sell frees up.

Public API (unchanged): ``compute_unified_rebalancing(holdings, config,
total_value, lump_sum) -> (actions, verifications)``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Optional

import numpy as np

from tarzan.models.holding import AssetClass, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.instruments.registry import InstrumentKind, TypeResolutionState
from tarzan.instruments.valuation import holding_kind_resolution

logger = logging.getLogger(__name__)
_CASH_PROOF_EPSILON_EUR = 1e-9


@dataclass(frozen=True)
class _CashMechanics:
    """Phase-owned mechanics, independent from strategic allocation categories.

    CASH is an account balance. An ETF/BOND/STOCK in the Cash & Cash
    Equivalents category is a security: its value is not settlement money
    until a sell is explicitly serialized. Unknown/conflicting declarations
    never authorize spending a category's market value.
    """

    kinds: tuple[Optional[InstrumentKind], ...]
    settlement_mask: np.ndarray
    locked_settlement_mask: np.ndarray
    security_mask: np.ndarray
    unresolved_indices: tuple[int, ...]

    @classmethod
    def prepare(cls, holdings: list[Holding]) -> _CashMechanics:
        resolutions = [holding_kind_resolution(h) for h in holdings]
        kinds = tuple(
            r.kind if r.state is TypeResolutionState.RESOLVED else None
            for r in resolutions
        )
        settlement_mask = np.array(
            [kind is InstrumentKind.CASH for kind in kinds], dtype=float,
        )
        locked_settlement_mask = np.array([
            kind is InstrumentKind.CASH and h.no_buy_no_sell
            for h, kind in zip(holdings, kinds)
        ], dtype=float)
        security_mask = np.array([
            h.asset_class == AssetClass.CASH_EQUIVALENTS
            and kind in (InstrumentKind.ETF, InstrumentKind.BOND, InstrumentKind.STOCK)
            for h, kind in zip(holdings, kinds)
        ], dtype=float)
        unresolved_indices = tuple(
            i for i, (holding, kind) in enumerate(zip(holdings, kinds))
            if (
                holding.asset_class == AssetClass.CASH_EQUIVALENTS and kind is None
            ) or (
                kind is InstrumentKind.CASH
                and holding.asset_class != AssetClass.CASH_EQUIVALENTS
            )
        )
        return cls(
            kinds, settlement_mask, locked_settlement_mask, security_mask,
            unresolved_indices,
        )


@dataclass(frozen=True)
class _CashFunding:
    """Account funding and the strategic reserve after explicit security trades."""

    values: np.ndarray
    security_mask: np.ndarray
    initial_cash: float
    external_contribution: float
    strategic_target: float
    locked_cash: float = 0.0

    @property
    def available_cash(self) -> float:
        return self.initial_cash + self.external_contribution

    def security_value(self, trades: np.ndarray) -> float:
        return float(self.security_mask @ (self.values + trades))

    def protected_settlement_cash(self, trades: np.ndarray) -> float:
        return max(
            self.strategic_target - self.security_value(trades),
            self.locked_cash,
            0.0,
        )

    def budget(self, trades: np.ndarray) -> float:
        # A negative budget requires explicit net sales to fill a reserve
        # deficit; it does not create money or implicitly liquidate a fund.
        return self.available_cash - self.protected_settlement_cash(trades)


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
    minus strategic cash equivalents); the cash category has its absolute target
    ``target_cash_buffer_eur``. Returns ``(actions, verifications)`` where each
    action is ``{idx, name, ticker, direction, amount_eur, reason}`` and the
    verifications mirror the post-rebalance allocation per ambit. Quantities
    are indicative at the selected EUR-per-unit price, not broker orders:
    lot sizes, fractional-order availability, ticks and future execution prices
    are not established by the funding-valid ``EXECUTABLE`` status.
    """
    return optimize_local_search(holdings, config, total_value, lump_sum=lump_sum)


# ---------------------------------------------------------------------------
# Verification of a rebalancing plan (shared output contract)
# ---------------------------------------------------------------------------

def _verify(new_values, holdings, config, geo_frac, all_geos, fi_value=0.0, *,
            unallocated_settlement_cash=0.0, cash_mechanics=None):
    verifications = []
    tv = new_values.sum() + unallocated_settlement_cash
    # Split cash from invested for the percentage-based checks.
    cash_mask = np.array([
        1.0 if h.asset_class == AssetClass.CASH_EQUIVALENTS else 0.0
        for h in holdings
    ])
    cash_new = float((cash_mask * new_values).sum()) + unallocated_settlement_cash
    invested_new = max(float(tv - cash_new), 0.0)
    # Use the same threshold as the Optimizer traffic-light so the verification
    # "OK / PARTIAL" status does not contradict the user-facing colors.
    tol = float(config.rebalancing_target_tolerance_pctg)

    # Per-holding-only mode: verify non-cash instruments (unlisted → 0), plus
    # explicit cash-security targets and the reserve. Asset/geo checks are
    # intentionally skipped; an untargeted reserve security is not a 0% exit.
    if getattr(config, "target_use_per_holding_only", False):
        cash_mechanics = cash_mechanics or _CashMechanics.prepare(holdings)
        details, items, max_d = [], [], 0.0
        for i, h in enumerate(holdings):
            if (
                h.asset_class == AssetClass.CASH_EQUIVALENTS
                and not (cash_mechanics.security_mask[i] > 0 and h.target_portfolio is not None)
            ):
                continue
            tgt = float(h.target_portfolio or 0.0)
            actual = new_values[i] / invested_new * 100 if invested_new > 0 else 0.0
            d = abs(actual - tgt)
            max_d = max(max_d, d)
            if tgt > 0 or actual > 0.05:
                details.append(f"{h.ticker} {actual:.1f}% (tgt. {tgt:.0f}%)")
                items.append({"category": h.name or h.ticker, "ticker": h.ticker,
                              "actual_pct": actual, "target_pct": tgt})
        verifications.append({"check": "Per-Holding Portfolio Targets",
                              "kind": "per_holding_portfolio",
                              "status": "\u2713 OK" if max_d <= tol else "\u26a0 PARTIAL",
                              "detail": ", ".join(details) or "No targets set",
                              "items": items})
        cash_target = float(config.target_cash_buffer_eur or 0.0)
        cash_delta_eur = cash_new - cash_target
        cash_band_eur = abs(tol / 100.0 * cash_target)
        cash_ok = abs(cash_delta_eur) <= max(cash_band_eur, 1.0)
        verifications.append({
            "check": "Cash & Cash Equivalents", "kind": "cash",
            "status": "\u2713 OK" if cash_ok else "\u26a0 PARTIAL",
            "detail": (f"Cash {cash_new:.2f} EUR (tgt. {cash_target:.2f} EUR, "
                       f"delta {cash_delta_eur:+.2f} EUR)"),
            "items": [{"category": "Cash & Cash Equivalents", "actual_eur": cash_new,
                       "target_eur": cash_target, "delta_eur": cash_delta_eur}],
        })
        return verifications

    # Invested asset allocation: leverage-aware notional percentages relative
    # to invested capital, shared with reporting/history/optimizer.
    from tarzan.engine import allocations as alloc

    class_pcts = {}
    if invested_new > 0:
        for i, h in enumerate(holdings):
            if h.asset_class == AssetClass.CASH_EQUIVALENTS:
                continue
            for asset_class, pct in alloc.holding_class_breakdown(h).items():
                if asset_class == AssetClass.CASH_EQUIVALENTS.value:
                    continue
                class_pcts[asset_class] = (
                    class_pcts.get(asset_class, 0.0)
                    + new_values[i] * pct / invested_new
                )
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

    # Geo allocation uses the canonical equity-notional denominator.
    eq_mask = np.array([
        alloc.exposure_fraction(h, AssetClass.EQUITIES.value) for h in holdings
    ])
    eq_total = (new_values * eq_mask).sum()
    geo_details, geo_items, max_geo = [], [], 0.0
    if eq_total > 0:
        for g_idx, gn in enumerate(all_geos):
            actual = sum(
                geo_frac[i][g_idx] * new_values[i]
                for i, _holding in enumerate(holdings)
            ) / eq_total * 100
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
        if h.target_equities is None or eq_mask[i] <= 0:
            continue
        actual = new_values[i] * eq_mask[i] / eq_total * 100 if eq_total > 0 else 0
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
    fi_mask = np.array([
        alloc.exposure_fraction(h, AssetClass.FIXED_INCOME.value) for h in holdings
    ])
    fi_total = (new_values * fi_mask).sum()
    for i, h in enumerate(holdings):
        if h.target_fixed_income is None or fi_mask[i] <= 0:
            continue
        actual = new_values[i] * fi_mask[i] / fi_total * 100 if fi_total > 0 else 0
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
        # Per-holding-only mode: drive each instrument to its portfolio-level
        # target (% of invested), ignoring asset-class and geo objectives.
        self.per_holding_only = bool(getattr(config, "target_use_per_holding_only", False))

        from tarzan.engine import allocations as alloc

        cls = [(h.asset_class.value if h.asset_class else "Alternative") for h in holdings]
        cash_name = AssetClass.CASH_EQUIVALENTS.value
        equity_name = AssetClass.EQUITIES.value
        fixed_income_name = AssetClass.FIXED_INCOME.value
        self.noncash_mask = np.array(
            [0.0 if c == cash_name else 1.0 for c in cls], dtype=float
        )
        self.eq_mask = np.array(
            [alloc.exposure_fraction(h, equity_name) for h in holdings], dtype=float
        )
        self.fi_mask = np.array(
            [alloc.exposure_fraction(h, fixed_income_name) for h in holdings], dtype=float
        )
        self.cash_mask = np.array(
            [1.0 if c == cash_name else 0.0 for c in cls], dtype=float
        )
        self.cash_mechanics = _CashMechanics.prepare(holdings)
        self.settlement_cash_mask = self.cash_mechanics.settlement_mask
        self.cash_security_mask = self.cash_mechanics.security_mask

        # Asset-class objectives (% of invested capital) use the same
        # leverage-aware notional exposure matrix as reporting and history.
        self.ac_targets = np.array(
            [float(target) for target in (config.invested_allocation_targets_pctg or {}).values()]
        )
        self.ac_keys = list((config.invested_allocation_targets_pctg or {}).keys())
        self.ac_masks = alloc.class_exposure_matrix(holdings, self.ac_keys)
        if not self.ac_keys:
            self.ac_masks = np.zeros((0, n))

        # Equity geography objectives (% of equity sleeve).
        self.all_geos = sorted((config.equity_geo_targets_pctg or {}).keys())
        self.geo_frac = np.zeros((n, len(self.all_geos)))
        for i, h in enumerate(holdings):
            equity_fraction = self.eq_mask[i]
            if equity_fraction <= 0:
                continue
            if h.geo_breakdown:
                tot = sum(h.geo_breakdown.values())
                if tot > 0:
                    for g, p in h.geo_breakdown.items():
                        gn = g.value if hasattr(g, "value") else str(g)
                        if gn in self.all_geos:
                            self.geo_frac[i][self.all_geos.index(gn)] = (p / tot) * equity_fraction
            elif h.geography:
                gn = h.geography.value if hasattr(h.geography, "value") else str(h.geography)
                if gn in self.all_geos:
                    self.geo_frac[i][self.all_geos.index(gn)] = equity_fraction
        self.geo_targets = np.array(
            [float(config.equity_geo_targets_pctg.get(g, 0.0)) for g in self.all_geos]
        )

        # Per-holding equity / FI objectives.
        self.ph_eq_idx = [i for i, h in enumerate(holdings)
                          if h.target_equities is not None and self.eq_mask[i] > 0]
        self.ph_eq_tgt = np.array([float(holdings[i].target_equities) for i in self.ph_eq_idx])
        self.ph_fi_idx = [i for i, h in enumerate(holdings)
                          if h.target_fixed_income is not None and self.fi_mask[i] > 0]
        self.ph_fi_tgt = np.array([float(holdings[i].target_fixed_income) for i in self.ph_fi_idx])

        # Per-holding PORTFOLIO objectives (% of invested portfolio). Every
        # non-cash holding gets a target (unlisted → 0, i.e. sell out). A
        # cash-equivalent security gets its explicitly declared target only.
        self.ph_pf_idx: list[int] = []
        self.ph_pf_tgt = np.zeros(0)
        if self.per_holding_only:
            self.ph_pf_idx = [i for i, h in enumerate(holdings)
                              if h.asset_class != AssetClass.CASH_EQUIVALENTS
                              or (self.cash_security_mask[i] > 0 and h.target_portfolio is not None)]
            self.ph_pf_tgt = np.array(
                [float(holdings[i].target_portfolio or 0.0) for i in self.ph_pf_idx]
            )

        self.cash_target = float(config.target_cash_buffer_eur or 0.0)

        cash_lbl = [("cash", "Cash & Cash Equivalents")] if self.cash_target > 0 else []
        cash_tgt = np.array([self.cash_target]) if self.cash_target > 0 else np.zeros(0)

        if self.per_holding_only:
            # Only per-instrument portfolio targets (+ cash). Asset-class and
            # geo objectives are intentionally excluded.
            self.labels = [("ph_pf", holdings[i].ticker) for i in self.ph_pf_idx] + cash_lbl
            self.ambit_of = ["per_holding" for _ in self.ph_pf_idx] + (["cash"] if cash_lbl else [])
            self.targets = np.concatenate([self.ph_pf_tgt, cash_tgt])
        else:
            # Stable label list parallel to the gap vector (for reporting / ambits).
            self.labels = (
                [("asset", k) for k in self.ac_keys]
                + [("geo", g) for g in self.all_geos]
                + [("ph_eq", holdings[i].ticker) for i in self.ph_eq_idx]
                + [("ph_fi", holdings[i].ticker) for i in self.ph_fi_idx]
                + cash_lbl
            )
            # Ambit grouping: the macro-areas the user cares about. Per-holding
            # equity and FI are one "per_holding" ambit so a single unfixable
            # holding does not sink the whole ambit.
            self.ambit_of = []
            for kind, _ in self.labels:
                self.ambit_of.append("per_holding" if kind in ("ph_eq", "ph_fi") else kind)
            self.targets = np.concatenate([
                self.ac_targets, self.geo_targets, self.ph_eq_tgt, self.ph_fi_tgt, cash_tgt,
            ])

    def gaps(self, new_values: np.ndarray, *,
             settlement_cash_delta: float = 0.0) -> np.ndarray:
        """Signed deviations (pp) for every objective, in ``labels`` order.
        The cash entry is a relative deviation (% of the cash target)."""
        inv = max(float(self.noncash_mask @ new_values), 1e-9)
        eq = max(float(self.eq_mask @ new_values), 1e-9)
        fi = max(float(self.fi_mask @ new_values), 1e-9)

        if self.per_holding_only:
            parts = []
            if self.ph_pf_idx:
                parts.append(new_values[self.ph_pf_idx] / inv * 100.0 - self.ph_pf_tgt)
            if self.cash_target > 0:
                cash_actual = float(self.cash_mask @ new_values) + settlement_cash_delta
                parts.append(np.array([(cash_actual - self.cash_target) / self.cash_target * 100.0]))
            return np.concatenate(parts) if parts else np.zeros(0)

        parts = []
        if self.ac_masks.shape[0]:
            parts.append((self.ac_masks @ new_values) / inv * 100.0 - self.ac_targets)
        if len(self.all_geos):
            geo_actual = (self.geo_frac.T @ new_values) / eq * 100.0
            parts.append(geo_actual - self.geo_targets)
        if self.ph_eq_idx:
            parts.append(
                (new_values[self.ph_eq_idx] * self.eq_mask[self.ph_eq_idx])
                / eq * 100.0 - self.ph_eq_tgt
            )
        if self.ph_fi_idx:
            parts.append(
                (new_values[self.ph_fi_idx] * self.fi_mask[self.ph_fi_idx])
                / fi * 100.0 - self.ph_fi_tgt
            )
        if self.cash_target > 0:
            cash_actual = float(self.cash_mask @ new_values) + settlement_cash_delta
            parts.append(np.array([(cash_actual - self.cash_target) / self.cash_target * 100.0]))
        return np.concatenate(parts) if parts else np.zeros(0)


# ---------------------------------------------------------------------------
# Cash / friction accounting
# ---------------------------------------------------------------------------

def _tax_per_unit_sold(holdings: list[Holding], config: InvestorConfig) -> np.ndarray:
    from tarzan.engine.tax import is_government_bond
    from tarzan.instruments.registry import InstrumentKind
    from tarzan.instruments.valuation import holding_kind_resolution

    cg_std = float(config.rebalancing_capital_gains_tax_standard_pctg or 0.0) / 100.0
    cg_gov = float(config.rebalancing_capital_gains_tax_government_pctg or 0.0) / 100.0
    tax = np.zeros(len(holdings))
    for i, h in enumerate(holdings):
        gp = float(h.gain_pct or 0.0)
        if gp <= 0:
            continue
        # Same gate as the realized-CGT estimate (tax._classify): prove BOND
        # mechanics BEFORE consulting the sovereign name/subtype markers. A bond
        # ETF (asset_class=Fixed Income, KIND=ETF) is capital income at the
        # standard rate, so an issuer token in its NAME (BTP/BUND/…) must not
        # buy it the reduced rate here while the estimate charges 26%.
        kind = holding_kind_resolution(h).kind
        subtypes = (h.instrument_type, h.security_type)
        rate = (
            cg_gov
            if kind is InstrumentKind.BOND and is_government_bond(subtypes, (h.name,))
            else cg_std
        )
        # Tax withheld per euro of SALE PROCEEDS (not per euro of cost). gp is
        # the gain relative to cost, so a position up gp% has proceeds
        # (100+gp) for every 100 of cost, of which gp is taxable gain. The
        # taxable-gain fraction of proceeds is therefore gp/(100+gp), NOT
        # gp/100. Using gp/100 overstates the tax on appreciated sells (e.g. a
        # +100% holding: true 0.13 vs 0.26 at 26%), making the optimizer
        # believe it must sell ~18% more to fund the same buys.
        tax[i] = rate * (gp / (100.0 + gp))
    return tax


def plan_cost(actions: list[dict], holdings: list[Holding],
              config: InvestorConfig) -> dict:
    """Estimated cash cost of executing a plan: capital-gains tax on the sells
    and fixed per-trade commission fees. Reuses the SAME per-unit tax model the
    optimizer used to build the plan (``_tax_per_unit_sold``) and the same fee
    params, so the displayed CGT/fees match the cash-conservation the optimizer
    solved for — no second formula to drift.

    ``actions`` are the dicts from ``_extract_actions`` (each carries ``idx``,
    ``direction``, ``amount_eur``). Returns ``{"cgt_eur", "fees_eur"}``."""
    tax = _tax_per_unit_sold(holdings, config)
    fee_buy = float(config.rebalancing_transaction_fee_buy_eur or 0.0)
    fee_sell = float(config.rebalancing_transaction_fee_sell_eur or 0.0)
    cgt = 0.0
    n_buy = n_sell = 0
    for a in actions:
        amt = float(a.get("amount_eur", 0.0) or 0.0)
        if a.get("direction") == "sell":
            n_sell += 1
            i = a.get("idx")
            if isinstance(i, int) and 0 <= i < len(tax):
                cgt += amt * float(tax[i])  # tax is per-euro-of-proceeds
        elif a.get("direction") == "buy":
            n_buy += 1
    fees = fee_buy * n_buy + fee_sell * n_sell
    return {"cgt_eur": round(cgt, 2), "fees_eur": round(fees, 2)}


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
    """Scalar evaluator for one search with fixed values, targets and tolerance."""

    def __init__(self, model: _ObjectiveModel, values: np.ndarray, config: InvestorConfig,
                 lump_sum: float, params: LSParams, *,
                 cash_funding: Optional[_CashFunding] = None):
        self.model = model
        self.values = values
        self.tv = float(values.sum())
        self.params = params
        self.lump_sum = float(lump_sum or 0.0)
        self.cash_funding = cash_funding
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
        # Model targets/tolerance are fixed for the lifetime of this search.
        # Keep the guard's exact array operation, but do it once rather than
        # allocating the same vector for every candidate.
        self.allowed = np.maximum(gap0, model.tol)

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
        if self.cash_funding is not None:
            net_cost = _net_cash_cost(
                t, self.tax, self.fee_buy, self.fee_sell, p.min_trade_eur,
            )
            # Account balances are not security coordinates. Include their
            # actual change in the strategic cash gap, also with no CASH row.
            settlement_delta = self.cash_funding.external_contribution - net_cost
            g = np.abs(self.model.gaps(nv, settlement_cash_delta=settlement_delta))
            residual = net_cost - self.cash_funding.budget(t)
        else:
            g = np.abs(self.model.gaps(nv))
            net_cost = None
            residual = None
        j = self.gap_term(g)
        # Guard: never push ANY objective past max(its initial gap, tolerance).
        # Protects in-tolerance objectives (e.g. the cash buffer) from being
        # used as a dumping ground, and stops out-of-tol objectives from
        # getting worse.
        excess = np.clip(g - self.allowed, 0.0, None)
        j += p.guard_weight * float(np.sum(excess ** 2))
        abs_t = np.abs(t)
        turnover = float(abs_t.sum())
        n_trades = int((abs_t >= p.min_trade_eur).sum())
        if residual is None:
            residual = _net_cash_cost(t, self.tax, self.fee_buy, self.fee_sell, p.min_trade_eur) - self.lump_sum
        j += p.turnover_weight * (turnover / max(self.tv, 1e-9))
        j += p.trade_count_weight * n_trades
        j += p.residual_weight * (residual / max(self.tv, 1e-9) * 100.0) ** 2
        if self.cash_funding is not None:
            # Below the strategic target, converting cash to a cash-equivalent
            # security leaves the combined bucket unchanged. The independent
            # bank constraint still forbids paying for it with negative cash.
            bank_deficit = max(
                net_cost - (self.cash_funding.available_cash - self.cash_funding.locked_cash),
                0.0,
            )
            j += p.residual_weight * (bank_deficit / max(self.tv, 1e-9) * 100.0) ** 2
        return j


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _bounds(holdings: list[Holding], values: np.ndarray, no_sell: bool,
            per_holding_only: bool = False, *, cash_mechanics=None):
    """Return (lo, hi, tradeable) arrays for the signed trade vector t.

    In per-holding-only mode a 0% target means "fully exit" — when selling is
    allowed such a held position is pinned to a full sell (lo=hi=-value), so
    the tolerance band can never leave a residual behind. Callers seed the
    initial trade from ``lo`` where ``lo == hi`` so these pins take effect.
    """
    n = len(holdings)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    tradeable = np.ones(n, dtype=bool)
    cash_mechanics = cash_mechanics or _CashMechanics.prepare(holdings)
    unresolved = set(cash_mechanics.unresolved_indices)
    for i, h in enumerate(holdings):
        # Only proved CASH mechanics denotes an account. A money-market ETF
        # remains a tradable security even in the same strategic category.
        if cash_mechanics.settlement_mask[i] > 0 or i in unresolved or h.no_buy_no_sell:
            lo[i] = hi[i] = 0.0
            tradeable[i] = False
            continue
        lo[i] = 0.0 if no_sell else -float(values[i])  # cannot sell more than held
        if (per_holding_only and not no_sell and values[i] > 0
                and float(getattr(h, "target_portfolio", 0.0) or 0.0) == 0.0
                and (h.asset_class != AssetClass.CASH_EQUIVALENTS
                     or h.target_portfolio is not None)):
            # Target 0% ⇒ liquidate fully, ignoring the tolerance band.
            lo[i] = hi[i] = -float(values[i])
            tradeable[i] = False
    return lo, hi, tradeable


def _descent(t: np.ndarray, cost: _Cost, lo, hi, tradeable, steps, params: LSParams):
    """Best-improvement coordinate descent with a short anti-reversal tabu.

    ``cost`` is deterministic for a fixed vector during the descent. Bounds
    can make several steps produce exactly the same candidate: evaluate only
    its first eligible occurrence, retaining that occurrence's sign and the
    original strict tie-breaking. No approximate/rounded candidates are merged.
    """
    t = t.copy()
    cur = cost(t)
    tabu: dict[tuple[int, int], int] = {}  # (idx, sign) -> expiry iteration
    tradeable_indices = np.where(tradeable)[0]
    # The public optimizer constructs float64 trades. Conversion to Python
    # float is also exact for smaller floating dtypes, but not for wide ints,
    # object arrays or extended-precision floats passed to this private helper.
    deduplicate = (
        np.issubdtype(t.dtype, np.floating)
        and t.dtype.itemsize <= np.dtype(np.float64).itemsize
    )
    for it in range(params.max_iters_per_descent):
        best_gain = 1e-12
        best_move = None
        for i in tradeable_indices:
            evaluated: set[str] = set()
            for s in steps:
                for sign in (+1.0, -1.0):
                    if tabu.get((int(i), int(sign)), -1) > it:
                        continue
                    new_ti = min(max(t[i] + sign * s, lo[i]), hi[i])
                    if new_ti == t[i]:
                        continue
                    old = t[i]
                    t[i] = new_ti
                    # Key the value actually assigned to this coordinate, not
                    # a ticker or a rounded trade. float.hex preserves signed
                    # zero and every finite float64 bit; nonfinite candidates
                    # retain the original scalar path. Mark only after tabu
                    # and no-op guards, and never share across coordinates or
                    # iterations (the rest of the vector may have changed).
                    if deduplicate:
                        candidate = float(t[i])
                        if math.isfinite(candidate):
                            key = candidate.hex()
                            if key in evaluated:
                                t[i] = old
                                continue
                            evaluated.add(key)
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
    """Project toward the funding budget, with independent final-action proof.

    Balance ordinary trades first so the reserve is not a dumping ground;
    cash-equivalent securities may absorb a residual via explicit trades,
    whereas account-CASH coordinates remain frozen.
    """
    t = t.copy()
    cash_idx = set(int(i) for i in np.where(model.cash_mask > 0)[0])

    def resid() -> float:
        net_cost = _net_cash_cost(t, cost.tax, cost.fee_buy, cost.fee_sell,
                                 params.min_trade_eur)
        budget = (
            cost.cash_funding.budget(t) if cost.cash_funding is not None
            else cost.lump_sum
        )
        return net_cost - budget

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
    initial_cash = float(model.settlement_cash_mask @ values)
    locked_cash = float(model.cash_mechanics.locked_settlement_mask @ values)
    protected_cash = max(float(config.target_cash_buffer_eur or 0.0), 0.0)
    external_contribution = max(float(lump_sum or 0.0), 0.0)
    # Reserve the protected buffer against all cash sources before making any
    # amount available to purchases. When current cash starts below target,
    # the external contribution first closes that deficit.
    funding_budget = max(
        initial_cash + external_contribution - max(protected_cash, locked_cash),
        0.0,
    )
    # Preserve the scalar account-only search. The additional mechanics is
    # needed when a strategic cash category contains actual tradable securities.
    cash_funding = (
        _CashFunding(
            values, model.cash_security_mask, initial_cash,
            external_contribution, protected_cash, locked_cash=locked_cash,
        )
        if np.any(model.cash_security_mask > 0) else None
    )
    cost = _Cost(
        model, values, config, funding_budget, params, cash_funding=cash_funding,
    )
    lo, hi, tradeable = _bounds(
        holdings, values, bool(config.rebalancing_no_sell),
        getattr(config, "target_use_per_holding_only", False),
        cash_mechanics=model.cash_mechanics)
    tv = float(values.sum())
    steps = [tv * f for f in params.step_fractions]
    rng = np.random.default_rng(params.seed)

    # Initial trade: pinned positions (lo == hi) start at that value — this
    # seeds the forced full-sell of 0%-target holdings; everything else at 0.
    t = np.where(lo == hi, lo, 0.0)
    best, best_cost = _descent(t, cost, lo, hi, tradeable, steps, params)

    # Iterated Local Search.
    for _ in range(params.n_restarts):
        kicked = _perturb(best, lo, hi, tradeable, tv, rng, params.perturbation_strength)
        cand, cand_cost = _descent(kicked, cost, lo, hi, tradeable, steps, params)
        if cand_cost < best_cost - 1e-9:
            best, best_cost = cand, cand_cost

    best = _project_to_budget(best, cost, model, lo, hi, params)

    # Apply the display threshold, quantize EUR proposals to cents, then
    # reconstruct positions and funding exclusively from that final vector.
    best = np.where(np.abs(best) >= params.min_trade_eur, best, 0.0)
    best = np.round(best, 2)
    # A full liquidation with a sub-cent market value must not round UP into
    # more units than held. Leave the genuine sub-cent residual instead.
    maximum_sales = np.floor(values * 100.0 + 1e-9) / 100.0
    best = np.maximum(best, -maximum_sales)

    actions = _extract_actions(
        best, holdings, model, values, min_trade=params.min_trade_eur,
    )
    new_values, funding = _finalize_action_proof(
        actions=actions,
        holdings=holdings,
        values=values,
        config=config,
        external_contribution=external_contribution,
        protected_cash=protected_cash,
        cash_mechanics=model.cash_mechanics,
    )

    # Independent cent rounding can make otherwise balanced buys exceed the
    # raw budget by fractions of a cent per action. Keep the proof strict and
    # reconcile that serialization residue by trimming buys in whole cents.
    # If no buys can absorb it, the unchanged proof remains NON_EXECUTABLE.
    if (
        not funding["protected_cash_satisfied"]
        and funding["position_invariants_satisfied"]
        and funding["frozen_positions_satisfied"]
    ):
        minimum_trade_cents = int(
            np.ceil(max(params.min_trade_eur, 0.0) * 100.0)
        )
        # Trimming a cash-equivalent purchase converts its value back into
        # settlement cash; it does NOT replenish the combined strategic
        # bucket, except when a canceled trade releases its commission.
        purchase_order = sorted(
            (int(i) for i in np.argsort(-best) if best[i] > 0),
            key=lambda i: model.cash_security_mask[i] > 0,
        )
        for i in purchase_order:
            if funding["protected_cash_satisfied"]:
                break
            if model.settlement_cash_mask[i] > 0:
                continue
            bank_shortfall = max(-float(funding.get(
                "ending_free_settlement_cash_eur",
                funding["ending_settlement_cash_eur"],
            )), 0.0)
            strategic_shortfall = max(
                protected_cash - float(funding["strategic_cash_equivalents_eur"]),
                0.0,
            )
            remaining_cents = int(np.ceil(
                max(round(max(bank_shortfall, strategic_shortfall), 9), 0.0) * 100.0,
            ))
            available_cents = int(round(float(best[i]) * 100.0))
            trim_cents = min(available_cents, remaining_cents)
            if model.cash_security_mask[i] > 0 and bank_shortfall <= _CASH_PROOF_EPSILON_EUR:
                # Only removal of this buy's fee can improve a reserve
                # shortfall. Never pretend that a smaller ETF purchase
                # replenishes cash while retaining the same fee.
                if float(config.rebalancing_transaction_fee_buy_eur or 0.0) <= 0:
                    continue
                trim_cents = available_cents
            updated_cents = available_cents - trim_cents
            if 0 < updated_cents < minimum_trade_cents:
                updated_cents = 0
                trim_cents = available_cents
            best[i] = updated_cents / 100.0
            actions = _extract_actions(
                best, holdings, model, values, min_trade=params.min_trade_eur,
            )
            new_values, funding = _finalize_action_proof(
                actions=actions,
                holdings=holdings,
                values=values,
                config=config,
                external_contribution=external_contribution,
                protected_cash=protected_cash,
                cash_mechanics=model.cash_mechanics,
            )

    if not funding["non_cost_only_progress_satisfied"]:
        # A cash-security shuffle with no configured target movement is not a
        # rebalance merely because tax/fees destroy part of an excess reserve.
        # Keep valid search alternatives available; only suppress this final
        # useless shuffle when the no-trade alternative is itself executable.
        empty_values, empty_funding = _finalize_action_proof(
            actions=[], holdings=holdings, values=values, config=config,
            external_contribution=external_contribution,
            protected_cash=protected_cash, cash_mechanics=model.cash_mechanics,
        )
        if empty_funding["status"] == "EXECUTABLE":
            empty_funding.update({
                "unneeded_cash_security_trades_suppressed": True,
                "suppressed_cash_security_actions": actions,
                "suppressed_cost_eur": round(
                    float(funding["estimated_tax_eur"]) + float(funding["fees_eur"]),
                    2,
                ),
                "suppression_reason": (
                    "Cash-security-only actions have no gross movement toward "
                    "a configured target; tax/fees are not target progress."
                ),
            })
            logger.warning(
                "Unneeded cost-only cash-security shuffle suppressed.",
                extra={"public_code": "COST_ONLY_CASH_SECURITY_SHUFFLE_SUPPRESSED"},
            )
            actions = []
            new_values, funding = empty_values, empty_funding

    verifications = _verify(new_values, holdings, config, model.geo_frac,
                            model.all_geos, float((model.fi_mask * values).sum()),
                            unallocated_settlement_cash=funding["unallocated_settlement_cash_eur"],
                            cash_mechanics=model.cash_mechanics)
    verifications.append(funding)
    return actions, verifications


def _unit_trade_facts(holding: Holding, signed_amount: float, value: float):
    """Trade units on the selected EUR-per-unit ruler, never raw bond per-100."""
    if (
        not isinstance(holding.current_price, Real)
        or isinstance(holding.current_price, bool)
        or not isinstance(holding.quantity, Real)
        or isinstance(holding.quantity, bool)
    ):
        return None
    try:
        price = float(holding.current_price)
        quantity = float(holding.quantity)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(price) or price <= 0
        or not math.isfinite(quantity) or quantity < 0
        or not math.isclose(quantity * price, value, rel_tol=1e-12, abs_tol=1e-8)
    ):
        return None
    delta = signed_amount / price
    after_quantity = quantity + delta
    if not math.isfinite(delta) or not math.isfinite(after_quantity):
        return None
    return {
        "price_eur_per_unit": price,
        "current_quantity": quantity,
        "quantity_delta": delta,
        "after_quantity": after_quantity,
    }


def _extract_actions(t: np.ndarray, holdings: list[Holding], model: _ObjectiveModel,
                     values: np.ndarray, *, min_trade: float = 1.0) -> list[dict]:
    new_values = values + t
    # Invested (non-cash) base BEFORE and AFTER the plan, so current/target and
    # post-trade weights each use the base they belong to (both are ~equal since
    # trades net-fund from cash, but computing each honestly avoids rounding
    # surprises). Guard against a zero base.
    inv_now = max(float(model.noncash_mask @ values), 1e-9)
    inv_after = max(float(model.noncash_mask @ new_values), 1e-9)
    actions = []
    for i, h in enumerate(holdings):
        amt = float(t[i])
        if amt == 0.0 or abs(amt) < min_trade:
            continue
        cur_eur = float(values[i])
        post_eur = float(new_values[i])
        # Per-instrument target only exists in per-holding mode; in asset-class
        # mode a holding has no individual target (only its class does), so we
        # leave it None and the newsletter shows "—".
        is_cash_security = model.cash_security_mask[i] > 0
        has_explicit_cash_target = is_cash_security and h.target_portfolio is not None
        tgt_pct = (
            float(h.target_portfolio or 0.0)
            if model.per_holding_only and (
                h.asset_class != AssetClass.CASH_EQUIVALENTS or has_explicit_cash_target
            ) else None
        )
        tgt_eur = (tgt_pct / 100.0 * inv_after) if tgt_pct is not None else None
        actions.append({
            "idx": i,
            "isin": h.isin,
            "name": h.name or h.ticker,
            "ticker": h.ticker,
            "direction": "buy" if amt > 0 else "sell",
            "amount_eur": round(abs(amt), 2),
            # Structured current → target → after, so the newsletter can render a
            # Diversification-style table (not just a bare amount + reason line).
            "current_eur": round(cur_eur, 2),
            "current_pct": (None if is_cash_security and not has_explicit_cash_target
                            else round(cur_eur / inv_now * 100.0, 2)),
            "target_eur": (round(tgt_eur, 2) if tgt_eur is not None else None),
            "target_pct": (round(tgt_pct, 2) if tgt_pct is not None else None),
            "after_eur": round(post_eur, 2),
            "after_pct": (None if is_cash_security and not has_explicit_cash_target
                          else round(post_eur / inv_after * 100.0, 2)),
            "reason": _reason(i, h, model, new_values),
        })
        if model.cash_mechanics.kinds[i] in (
            InstrumentKind.ETF, InstrumentKind.BOND, InstrumentKind.STOCK,
        ):
            serialized_amount = actions[-1]["amount_eur"] * (1.0 if amt > 0 else -1.0)
            unit_facts = _unit_trade_facts(h, serialized_amount, cur_eur)
            if unit_facts is not None:
                actions[-1].update(unit_facts)
    # Largest trades first (matches the display ordering).
    actions.sort(key=lambda a: -a["amount_eur"])
    return actions


def _finalize_action_proof(
    *,
    actions: list[dict],
    holdings: list[Holding],
    values: np.ndarray,
    config: InvestorConfig,
    external_contribution: float,
    protected_cash: float,
    cash_mechanics: Optional[_CashMechanics] = None,
) -> tuple[np.ndarray, dict]:
    """Independent sources/uses and units proof over the final serialized list.

    Security positions change ONLY through their own actions. The resulting
    settlement account may be represented by real CASH rows or by a separate
    plan ledger, but is never written into a money-market ETF's units/value.
    ``no_buy_no_sell`` on proved account CASH excludes its individual balance
    from all debit/credit redistribution. Only unlocked account balances fund
    buys or receive a balance change; absent such rows, free cash (including
    contributions and explicit sales) remains in the separate plan ledger.
    Frozen account value still belongs to wealth and the strategic reserve.
    ``EXECUTABLE`` is the legacy name for funding/position/policy validity;
    it does not prove broker lot/tick/fractional-order or execution-price rules.
    """
    cash_mechanics = cash_mechanics or _CashMechanics.prepare(holdings)
    trades = np.zeros(len(holdings), dtype=float)
    gross_purchases = 0.0
    gross_sales = 0.0
    frozen_ok = True
    no_sell_ok = True
    action_ok = True
    identity_ok = True
    units_ok = True
    valid_actions = []
    seen_indices = set()
    security_unit_changes = []
    cash_security_unit_changes = []
    for action in actions:
        if not isinstance(action, Mapping):
            action_ok = False
            continue
        idx = action.get("idx")
        raw_amount = action.get("amount_eur")
        if not isinstance(raw_amount, Real) or isinstance(raw_amount, bool):
            action_ok = False
            continue
        try:
            amount = float(raw_amount)
        except (TypeError, ValueError, OverflowError):
            action_ok = False
            continue
        direction = action.get("direction")
        if (
            not isinstance(idx, int) or isinstance(idx, bool)
            or not 0 <= idx < len(holdings) or idx in seen_indices
            or direction not in ("buy", "sell")
            or not math.isfinite(amount) or amount <= 0
            or not math.isclose(amount, round(amount, 2), rel_tol=0, abs_tol=1e-9)
        ):
            action_ok = False
            continue
        if any(
            field in action and (
                not isinstance(action[field], str)
                or action[field] != getattr(holdings[idx], field)
            )
            for field in ("isin", "ticker")
        ):
            # The stable row index is authoritative, but a contradictory
            # declared economic identity cannot be displayed as its trade.
            # Absent fields remain compatible with legacy index-only actions.
            identity_ok = False
            action_ok = False
            continue
        seen_indices.add(idx)
        if cash_mechanics.settlement_mask[idx] > 0:
            # A "cash sell" would double count an account balance as proceeds.
            frozen_ok = False
            continue
        sign = 1.0 if direction == "buy" else -1.0
        trades[idx] += sign * amount
        valid_actions.append(action)
        if sign > 0:
            gross_purchases += amount
        else:
            gross_sales += amount
            if config.rebalancing_no_sell:
                no_sell_ok = False
        if holdings[idx].no_buy_no_sell:
            frozen_ok = False
        if cash_mechanics.kinds[idx] in (
            InstrumentKind.ETF, InstrumentKind.BOND, InstrumentKind.STOCK,
        ):
            facts = _unit_trade_facts(holdings[idx], sign * amount, float(values[idx]))
            if facts is None:
                units_ok = False
            else:
                units_ok = units_ok and (
                    facts["after_quantity"] >= -max(
                        1e-12, abs(facts["current_quantity"]) * 1e-12,
                    )
                )
                # Metadata is evidence, not authority: independently derive
                # units from the final amount and selected price, then check
                # any supplied quantities against that reconstruction.
                for field, expected in facts.items():
                    if field in action:
                        if (
                            not isinstance(action[field], Real)
                            or isinstance(action[field], bool)
                        ):
                            units_ok = False
                            continue
                        try:
                            supplied = float(action[field])
                        except (TypeError, ValueError, OverflowError):
                            units_ok = False
                        else:
                            units_ok = units_ok and math.isclose(
                                supplied, expected, rel_tol=1e-12, abs_tol=1e-10,
                            )
                unit_change = {
                    "idx": idx, "isin": holdings[idx].isin,
                    "instrument_kind": cash_mechanics.kinds[idx].value, **facts,
                }
                security_unit_changes.append(unit_change)
                if cash_mechanics.security_mask[idx] > 0:
                    cash_security_unit_changes.append(unit_change.copy())

    costs = plan_cost(valid_actions, holdings, config)
    taxes = float(costs["cgt_eur"])
    fees = float(costs["fees_eur"])
    cash_indices = [
        i for i, holding in enumerate(holdings)
        if cash_mechanics.settlement_mask[i] > 0
    ]
    locked_cash_indices = [
        i for i in cash_indices if cash_mechanics.locked_settlement_mask[i] > 0
    ]
    unlocked_cash_indices = [
        i for i in cash_indices if cash_mechanics.locked_settlement_mask[i] == 0
    ]
    initial_cash = float(values[cash_indices].sum()) if cash_indices else 0.0
    locked_cash = float(values[locked_cash_indices].sum()) if locked_cash_indices else 0.0
    initial_unlocked_cash = (
        float(values[unlocked_cash_indices].sum()) if unlocked_cash_indices else 0.0
    )
    ending_cash = (
        initial_cash
        + external_contribution
        + gross_sales
        - taxes
        - fees
        - gross_purchases
    )
    # Reconstruct free funding from its own sources, not by subtracting a
    # possibly much larger locked balance from the total. This also prevents
    # cancellation from hiding a raw sub-cent overdraft. With no locked rows
    # the original arithmetic and serialized proof remain unchanged.
    ending_free_cash = (
        initial_unlocked_cash
        + external_contribution
        + gross_sales
        - taxes
        - fees
        - gross_purchases
    ) if locked_cash_indices else ending_cash

    reconstructed = values + trades
    if unlocked_cash_indices:
        if initial_unlocked_cash > 0:
            for idx in unlocked_cash_indices:
                reconstructed[idx] = (
                    ending_free_cash * float(values[idx]) / initial_unlocked_cash
                )
        else:
            reconstructed[unlocked_cash_indices[0]] = ending_free_cash
            for idx in unlocked_cash_indices[1:]:
                reconstructed[idx] = 0.0
    locked_cash_ok = all(
        math.isfinite(float(reconstructed[idx]))
        and reconstructed[idx] >= -_CASH_PROOF_EPSILON_EUR
        and reconstructed[idx] == values[idx]
        for idx in locked_cash_indices
    )
    frozen_ok = frozen_ok and locked_cash_ok

    position_ok = all(
        math.isfinite(float(reconstructed[i]))
        and reconstructed[i] >= -_CASH_PROOF_EPSILON_EUR
        for i in range(len(holdings))
        if cash_mechanics.settlement_mask[i] == 0
    )
    unallocated_cash = ending_free_cash if not unlocked_cash_indices else 0.0
    security_value_before = float(cash_mechanics.security_mask @ values)
    security_value_after = float(cash_mechanics.security_mask @ reconstructed)
    cash_category_mask = np.array([
        holding.asset_class == AssetClass.CASH_EQUIVALENTS for holding in holdings
    ], dtype=float)
    strategic_cash_before = float(cash_category_mask @ values)
    strategic_cash_after = float(cash_category_mask @ reconstructed) + unallocated_cash
    protected_settlement_cash = max(
        protected_cash - security_value_after, locked_cash, 0.0,
    )
    settlement_ok = (
        math.isfinite(ending_cash) and ending_cash >= -_CASH_PROOF_EPSILON_EUR
        and math.isfinite(ending_free_cash)
        and ending_free_cash >= -_CASH_PROOF_EPSILON_EUR
    )
    protected_ok = (
        settlement_ok
        and ending_cash >= protected_settlement_cash - _CASH_PROOF_EPSILON_EUR
        and strategic_cash_after >= protected_cash - _CASH_PROOF_EPSILON_EUR
    )
    residual = ending_cash - protected_settlement_cash
    sources = initial_cash + external_contribution + gross_sales
    uses = gross_purchases + taxes + fees + ending_cash
    equation_residual = sources - uses
    wealth_before = float(values.sum())
    wealth_after = float(reconstructed.sum()) + unallocated_cash
    wealth_residual = wealth_after - (
        wealth_before + external_contribution - taxes - fees
    )
    wealth_ok = math.isfinite(wealth_residual) and abs(wealth_residual) <= 0.005
    mechanics_ok = not cash_mechanics.unresolved_indices
    non_cost_only_progress_ok = True
    if (
        valid_actions and taxes + fees > 0
        and all(cash_mechanics.security_mask[action["idx"]] > 0 for action in valid_actions)
    ):
        # Compare the SAME contribution in both alternatives. For actions
        # exclusively within the cash category, the gross strategic cash
        # bucket is invariant: only tax/fees could reduce its target gap.
        # Explicit security/sleeve/geography targets remain authoritative.
        proof_model = _ObjectiveModel(holdings, config, values)
        baseline_gaps = np.abs(proof_model.gaps(
            values, settlement_cash_delta=external_contribution,
        ))
        gross_gaps = np.abs(proof_model.gaps(
            values + trades,
            settlement_cash_delta=(
                external_contribution + gross_sales - gross_purchases
            ),
        ))
        non_cash_objectives = [
            i for i, (kind, _label) in enumerate(proof_model.labels) if kind != "cash"
        ]
        non_cost_only_progress_ok = any(
            gross_gaps[i] < baseline_gaps[i] for i in non_cash_objectives
        )
    status = "EXECUTABLE" if (
        position_ok
        and protected_ok
        and frozen_ok
        and no_sell_ok
        and action_ok
        and identity_ok
        and units_ok
        and wealth_ok
        and mechanics_ok
        and non_cost_only_progress_ok
        and abs(equation_residual) <= 0.005
    ) else "NON_EXECUTABLE"

    authorized_release = min(
        max(initial_cash - protected_settlement_cash, 0.0),
        max(gross_purchases + taxes + fees - gross_sales - external_contribution, 0.0),
    )
    if locked_cash_indices:
        authorized_release = min(authorized_release, max(initial_unlocked_cash, 0.0))
    funding = {
        "check": "Final Serialized-Action Funding",
        "kind": "funding",
        "status": status,
        "initial_cash_eur": round(initial_cash, 2),
        "protected_cash_eur": round(protected_settlement_cash, 2),
        # Full-precision proof facts; the legacy display fields below round
        # to cents. Reconciliation must never infer a zero deficit from "-0".
        "initial_settlement_cash_eur": initial_cash,
        "ending_settlement_cash_eur": ending_cash,
        "strategic_cash_equivalents_eur": strategic_cash_after,
        "unallocated_settlement_cash_eur": unallocated_cash,
        "initial_cash_equivalents_eur": round(strategic_cash_before, 2),
        "ending_cash_equivalents_eur": round(strategic_cash_after, 2),
        "cash_equivalents_target_eur": round(protected_cash, 2),
        "cash_equivalent_securities_before_eur": round(security_value_before, 2),
        "cash_equivalent_securities_after_eur": round(security_value_after, 2),
        "security_unit_changes": security_unit_changes,
        "cash_equivalent_security_unit_changes": cash_security_unit_changes,
        "authorized_existing_cash_release_eur": round(authorized_release, 2),
        "external_contribution_eur": round(external_contribution, 2),
        "gross_sales_eur": round(gross_sales, 2),
        "estimated_tax_eur": round(taxes, 2),
        "fees_eur": round(fees, 2),
        "gross_purchases_eur": round(gross_purchases, 2),
        "ending_cash_eur": round(ending_cash, 2),
        "residual_eur": round(residual, 2),
        "equation_residual_eur": round(equation_residual, 2),
        "portfolio_wealth_before_eur": round(wealth_before, 2),
        "portfolio_wealth_after_eur": round(wealth_after, 2),
        "wealth_residual_eur": round(wealth_residual, 2),
        "position_invariants_satisfied": position_ok,
        "protected_cash_satisfied": protected_ok,
        "settlement_cash_satisfied": settlement_ok,
        "frozen_positions_satisfied": frozen_ok,
        "no_sell_satisfied": no_sell_ok,
        "serialized_actions_satisfied": action_ok,
        "action_identity_satisfied": identity_ok,
        "units_invariants_satisfied": units_ok,
        "wealth_invariants_satisfied": wealth_ok,
        "cash_mechanics_satisfied": mechanics_ok,
        "non_cost_only_progress_satisfied": non_cost_only_progress_ok,
        "items": [],
    }
    if locked_cash_indices:
        funding.update({
            "locked_settlement_cash_eur": locked_cash,
            "initial_free_settlement_cash_eur": initial_unlocked_cash,
            "ending_free_settlement_cash_eur": ending_free_cash,
            "locked_cash_balances_satisfied": locked_cash_ok,
        })
    return reconstructed, funding


def _reason(i: int, h: Holding, model: _ObjectiveModel, new_values: np.ndarray) -> str:
    """Short 'why' string: which objective(s) this holding's trade serves."""
    if model.cash_security_mask[i] > 0 and (
        not model.per_holding_only or h.target_portfolio is None
    ):
        return (
            f"cash-equivalent security \u2192 \u20ac{float(new_values[i]):,.2f}; "
            f"strategic reserve target \u20ac{model.cash_target:,.2f}"
        )
    if model.per_holding_only:
        inv = max(float(model.noncash_mask @ new_values), 1e-9)
        actual = float(new_values[i]) / inv * 100.0
        tgt = float(h.target_portfolio or 0.0)
        # ``actual`` is the POST-trade weight, so make that explicit — a
        # not-yet-held instrument being bought shows where it lands, not a
        # current position. The target is shown both as a % of the invested
        # portfolio and as its absolute euro value on the same invested base.
        tgt_eur = tgt / 100.0 * inv
        return (f"post-trade \u2192 {actual:.1f}% of portfolio "
                f"(target {tgt:.0f}% \u00b7 \u20ac{tgt_eur:,.0f})")
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
