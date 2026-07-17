"""Investor configuration model with serialization and normalization.

Loads investor preferences from a key-value CSV (no JSON, no legacy aliases).

Naming convention for CSV keys:
- `_eur` suffix → absolute EUR value
- `_pctg` suffix → percentage
- `_date` suffix → date (free-form string)
- no suffix → boolean flags

Asset-class targets are expressed as `target_invested_allocation_<class>_pctg`
and describe the allocation within the *invested* portion of the portfolio
(total minus cash). Cash is tracked separately via `target_cash_buffer_eur`.

Equity geography targets use `target_equity_geo_<region>_pctg` and describe
the allocation within the equity portion only.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field

from tarzan import config as cfg
from tarzan.runtime.provider import ProviderQualityPolicy

logger = logging.getLogger(__name__)


def normalize_percentages(d: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of percentages so they sum to 100.

    Guards two degenerate inputs that would otherwise produce absurd weights:
      * any NEGATIVE percentage (a typo like Japan=-59) — a share of a sleeve
        cannot be negative;
      * a total at/near zero, where dividing by it explodes the weights
        (e.g. {USA: 60, Japan: -59} sums to 1 → USA would become 6000%).
    In either case we return the input unchanged and let the caller flag it,
    rather than emit nonsense targets that drive wrong rebalancing trades.
    """
    if any(v < 0 for v in d.values()):
        return d
    total = sum(d.values())
    if total <= 1e-9:
        return d
    return {k: v * 100.0 / total for k, v in d.items()}


def _configured_provider_quality_policies() -> dict[str, ProviderQualityPolicy]:
    """Load and validate explicit instrument/data-class policy declarations."""
    raw = cfg.get("provider_quality_policies")
    if not isinstance(raw, dict):
        raise ValueError("provider_quality_policies must be declared in constants.yaml")
    required_kinds = {"STOCK", "ETF", "BOND", "CASH", "UNKNOWN"}
    if set(raw) != required_kinds:
        missing = sorted(required_kinds - set(raw))
        extra = sorted(set(raw) - required_kinds)
        raise ValueError(
            f"provider_quality_policies kind mismatch; missing={missing}, extra={extra}"
        )
    policies: dict[str, ProviderQualityPolicy] = {}
    for kind in sorted(raw):
        data_classes = raw[kind]
        if not isinstance(data_classes, dict) or set(data_classes) != {"current_valuation"}:
            raise ValueError(f"{kind} must declare exactly current_valuation policy")
        values = data_classes["current_valuation"]
        if not isinstance(values, dict):
            raise ValueError(f"{kind}:current_valuation policy must be a mapping")
        try:
            policies[f"{kind}:current_valuation"] = ProviderQualityPolicy(**values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid {kind}:current_valuation policy: {error}") from error
    return policies


@dataclass
class InvestorConfig:
    """Investor profile with allocation targets and rebalancing preferences.

    Allocation semantics:
    - `invested_allocation_targets_pctg` applies to the *invested* portfolio
      (= total portfolio value minus cash holdings). Must sum to 100.
    - `equity_geo_targets_pctg` applies to the equity portion only. Must sum to 100.
    - `target_cash_buffer_eur` is an absolute EUR amount, not a percentage.
    """

    # Rebalancing parameters
    rebalancing_lump_sum_amount_eur: float = 0.0
    # Tolerance band around every allocation target. Used in two
    # places: (1) as the LP solver's hard ceiling — the optimizer
    # tries progressively tighter tolerances and stops at this
    # value, (2) as the dashboard traffic-light threshold (green if
    # |drift| ≤ tolerance, amber up to 2×, red beyond). Keeping the
    # same value drives both the math and the visuals from a single
    # knob, so what the user sees is what the solver enforces.
    rebalancing_target_tolerance_pctg: float = 2.0
    rebalancing_no_sell: bool = False
    # When the LP is infeasible at ``rebalancing_target_tolerance_pctg``,
    # auto-relax the tolerance up to ``rebalancing_relax_cap_pctg`` to
    # surface the smallest feasible plan rather than emitting nothing.
    rebalancing_auto_relax: bool = True
    rebalancing_relax_cap_pctg: float = 10.0

    # Trade frictions injected into the LP objective. The fees are
    # fixed amounts paid per executed buy/sell (regardless of size) and
    # are modelled via the existing zb/zs binary variables. The capital
    # gains rates are applied proportionally to the EUR amount sold of
    # any holding currently in profit; the standard rate covers ETFs
    # and most equities while the government rate (typically lower in
    # several jurisdictions) is applied when ``instrument_type`` is a
    # government bond. All four parameters default to 0 so the
    # behavior is unchanged unless the user opts in.
    rebalancing_transaction_fee_buy_eur: float = 0.0
    rebalancing_transaction_fee_sell_eur: float = 0.0
    rebalancing_capital_gains_tax_standard_pctg: float = 0.0
    rebalancing_capital_gains_tax_government_pctg: float = 0.0

    # Soft penalty on the residual distance from each allocation
    # target. With ``drift_penalty_weight = 0`` the solver only
    # minimises trade volume and is happy as long as every category
    # stays inside its tolerance band, even if some are at the band
    # edge. With positive weights the solver also pays for any EUR
    # of drift left between the post-rebalancing position and the
    # target, so the lump sum tends to be distributed across all
    # under-target buckets rather than concentrated on the few that
    # would otherwise breach the band. The default of 1.0 means
    # "1 EUR of residual drift costs the same as 1 EUR of trade
    # volume" — a balanced trade-off between rebalancing strictness
    # and growth-oriented deployment of fresh capital.
    rebalancing_drift_penalty_weight: float = 1.0

    # When TRUE the optimizer ignores the asset-class and equity-geo targets
    # and rebalances purely toward the per-instrument portfolio-level targets
    # (``target_portfolio`` in targets_per_holding.csv): every held instrument
    # not listed there is driven to 0% (sold), and listed instruments not yet
    # held are seeded so they can be bought. FALSE = classic behavior.
    target_use_per_holding_only: bool = False

    # Cash buffer (absolute EUR amount)
    target_cash_buffer_eur: float = 0.0

    # Invested allocation (% of invested portfolio, excluding cash)
    invested_allocation_targets_pctg: dict[str, float] = field(
        default_factory=lambda: dict(cfg.default_invested_allocation_targets_pctg())
    )

    # Equity geography (% of equity portion)
    equity_geo_targets_pctg: dict[str, float] = field(
        default_factory=lambda: dict(cfg.default_equity_geo_targets_pctg())
    )

    # Explicit provider/valuation policy keyed by instrument kind and data
    # class. Loaded from constants.yaml and validated into immutable policy
    # values; callers cannot silently fall through to a universal threshold.
    provider_quality_policies: dict[str, ProviderQualityPolicy] = field(
        default_factory=_configured_provider_quality_policies
    )

    # Metadata
    portfolio_backtest_period: str = ""

    def __post_init__(self):
        """Fill metadata from constants.yaml if not set explicitly."""
        if not self.portfolio_backtest_period:
            self.portfolio_backtest_period = cfg.portfolio_backtest_period()

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str) -> "InvestorConfig":
        """Load investor config from a CSV key-value file.

        The CSV must have ``key`` and ``value`` columns. Any other
        columns (e.g. ``description``) are ignored, so users can keep
        a self-documented config file without the parser caring.
        """
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {
                row["key"].strip(): row["value"].strip()
                for row in reader
                if row.get("key") and "value" in row
            }
        return cls.from_dict(rows)

    @classmethod
    def from_dict(cls, rows: dict[str, str]) -> "InvestorConfig":
        """Load investor config from a pre-parsed key-value dict."""
        config = cls()

        # Scalar fields
        _set_float(config, rows, "rebalancing_lump_sum_amount_eur")
        _set_float(config, rows, "rebalancing_target_tolerance_pctg")
        _set_float(config, rows, "rebalancing_relax_cap_pctg")
        _set_float(config, rows, "rebalancing_transaction_fee_buy_eur")
        _set_float(config, rows, "rebalancing_transaction_fee_sell_eur")
        _set_float(config, rows, "rebalancing_capital_gains_tax_standard_pctg")
        _set_float(config, rows, "rebalancing_capital_gains_tax_government_pctg")
        _set_float(config, rows, "rebalancing_drift_penalty_weight")
        _set_float(config, rows, "target_cash_buffer_eur")

        # Boolean flags
        if "rebalancing_no_sell" in rows:
            config.rebalancing_no_sell = _parse_bool(rows["rebalancing_no_sell"])
        if "rebalancing_auto_relax" in rows:
            config.rebalancing_auto_relax = _parse_bool(rows["rebalancing_auto_relax"])
        if "target_use_per_holding_only" in rows:
            config.target_use_per_holding_only = _parse_bool(rows["target_use_per_holding_only"])

        # Dict fields
        _parse_invested_allocation(config, rows)
        _parse_equity_geo(config, rows)

        # Warn on unknown keys
        _warn_unknown_keys(rows)

        # Validate sums. The invested asset-class targets are NOT normalized:
        # with notional exposure (capital-efficient / leveraged funds) a target
        # allocation can legitimately sum to more than 100% (e.g. a 90/60
        # efficient-core sleeve), so we keep the raw values and only note the
        # total. Equity-geography targets ARE a within-sleeve distribution, so
        # they must still sum to 100% and are normalized.
        _note_target_total(
            config.invested_allocation_targets_pctg,
            "invested_allocation_targets_pctg",
        )
        _validate_sum_to_100(
            config.equity_geo_targets_pctg,
            "equity_geo_targets_pctg",
            normalize=True,
        )

        return config


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _set_float(config: InvestorConfig, rows: dict, key: str) -> None:
    if key in rows and rows[key] != "":
        try:
            setattr(config, key, float(rows[key]))
        except (ValueError, TypeError):
            logger.warning("Failed to parse %s='%s', using default", key, rows[key])


def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes", "y", "t")


# Canonical asset-class names for the invested allocation section
_INVESTED_ALLOC_KEYS: dict[str, str] = {
    "target_invested_allocation_equities_pctg": "Equities",
    "target_invested_allocation_fixed_income_pctg": "Fixed Income",
    "target_invested_allocation_gold_pctg": "Gold",
    "target_invested_allocation_commodities_pctg": "Commodities",
    "target_invested_allocation_crypto_pctg": "Crypto",
    "target_invested_allocation_alternative_pctg": "Alternative",
}


# Canonical region names for equity geography
_EQUITY_GEO_KEYS: dict[str, str] = {
    "target_equity_geo_usa_pctg": "USA",
    "target_equity_geo_japan_pctg": "Japan",
    "target_equity_geo_eurozone_emu_pctg": "Eurozone EMU",
    "target_equity_geo_dev_ex_usa_ex_emu_ex_jp_pctg": "Dev ex-USA ex-EMU ex-JP",
    "target_equity_geo_emerging_markets_pctg": "Emerging Markets",
}


def _parse_invested_allocation(config: InvestorConfig, rows: dict) -> None:
    found: dict[str, float] = {}
    for csv_key, canonical in _INVESTED_ALLOC_KEYS.items():
        if csv_key in rows and rows[csv_key] != "":
            try:
                found[canonical] = float(rows[csv_key])
            except (ValueError, TypeError):
                logger.warning("Failed to parse %s='%s'", csv_key, rows[csv_key])
    if found:
        config.invested_allocation_targets_pctg = found


def _parse_equity_geo(config: InvestorConfig, rows: dict) -> None:
    found: dict[str, float] = {}
    for csv_key, canonical in _EQUITY_GEO_KEYS.items():
        if csv_key in rows and rows[csv_key] != "":
            try:
                found[canonical] = float(rows[csv_key])
            except (ValueError, TypeError):
                logger.warning("Failed to parse %s='%s'", csv_key, rows[csv_key])
    if found:
        config.equity_geo_targets_pctg = found


_KNOWN_SCALAR_KEYS = frozenset({
    "rebalancing_lump_sum_amount_eur",
    "rebalancing_target_tolerance_pctg",
    "rebalancing_relax_cap_pctg",
    "rebalancing_no_sell",
    "rebalancing_auto_relax",
    "rebalancing_transaction_fee_buy_eur",
    "rebalancing_transaction_fee_sell_eur",
    "rebalancing_capital_gains_tax_standard_pctg",
    "rebalancing_capital_gains_tax_government_pctg",
    "rebalancing_drift_penalty_weight",
    "target_use_per_holding_only",
    "target_cash_buffer_eur",
})


def _known_keys() -> frozenset[str]:
    return _KNOWN_SCALAR_KEYS | frozenset(_INVESTED_ALLOC_KEYS) | frozenset(_EQUITY_GEO_KEYS)


def _warn_unknown_keys(rows: dict) -> None:
    known = _known_keys()
    for key in rows:
        if key and key not in known:
            logger.warning("Unknown target key '%s' — ignored", key)


def _validate_sum_to_100(
    d: dict[str, float], name: str, normalize: bool = False,
) -> None:
    if not d:
        return
    negatives = {k: v for k, v in d.items() if v < 0}
    total = sum(d.values())
    if negatives:
        # A within-sleeve share cannot be negative; do NOT normalize (that
        # would explode the weights). Keep the raw values and flag loudly.
        msg = (f"{name} contains negative target(s) {negatives} — a share of a "
               "sleeve cannot be negative; targets kept un-normalized, please fix")
        logger.warning(msg)
        _dq_config_warning(msg, name)
        return
    if abs(total - 100.0) > 0.01:
        logger.warning("%s sums to %.2f%%, expected 100%%", name, total)
        _dq_config_warning(
            f"{name} sums to {total:.2f}% (expected 100%)"
            + (" — normalized to 100%" if normalize else ""),
            name,
        )
        if normalize:
            normalized = normalize_percentages(d)
            d.clear()
            d.update(normalized)


def _dq_config_warning(message: str, context: str) -> None:
    """Route a config-validation warning into the data-quality report."""
    try:
        from tarzan.runtime import data_quality as dq
        dq.warning("config", message, context=context)
    except Exception:  # noqa: BLE001
        pass


def _note_target_total(d: dict[str, float], name: str) -> None:
    """Log the total of a target set WITHOUT normalizing it.

    Used for the invested asset-class targets, which may intentionally sum to
    more than 100% under notional exposure (capital-efficient / leveraged
    funds). The raw values are kept so "75% equities" stays 75%."""
    if not d:
        return
    total = sum(d.values())
    if abs(total - 100.0) > 0.01:
        logger.info("%s sums to %.1f%% (kept as-is — notional target allowed "
                    "to exceed 100%%)", name, total)
