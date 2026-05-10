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

logger = logging.getLogger(__name__)


def normalize_percentages(d: dict[str, float]) -> dict[str, float]:
    """Normalize a dict of percentages so they sum to 100."""
    total = sum(d.values())
    if total <= 0:
        return d
    return {k: v * 100.0 / total for k, v in d.items()}


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
    rebalancing_min_transaction_eur: float = 500.0
    rebalancing_max_tolerance_pctg: float = 2.0
    rebalancing_threshold_pctg: float = 5.0
    rebalancing_precision_pctg: float = 0.5
    rebalancing_no_sell: bool = False

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

    # Metadata
    portfolio_inception_date: str = ""
    portfolio_backtest_period: str = ""

    def __post_init__(self):
        """Fill metadata from constants.yaml if not set explicitly."""
        if not self.portfolio_backtest_period:
            self.portfolio_backtest_period = cfg.portfolio_backtest_period()
        if not self.portfolio_inception_date:
            self.portfolio_inception_date = cfg.portfolio_inception_date()

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str) -> "InvestorConfig":
        """Load investor config from a CSV key-value file."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {
                row["key"].strip(): row["value"].strip()
                for row in reader
                if "key" in row and "value" in row
            }
        return cls.from_dict(rows)

    @classmethod
    def from_dict(cls, rows: dict[str, str]) -> "InvestorConfig":
        """Load investor config from a pre-parsed key-value dict."""
        config = cls()

        # Scalar fields
        _set_float(config, rows, "rebalancing_lump_sum_amount_eur")
        _set_float(config, rows, "rebalancing_min_transaction_eur")
        _set_float(config, rows, "rebalancing_max_tolerance_pctg")
        _set_float(config, rows, "rebalancing_threshold_pctg")
        _set_float(config, rows, "rebalancing_precision_pctg")
        _set_float(config, rows, "target_cash_buffer_eur")

        # Boolean flags
        if "rebalancing_no_sell" in rows:
            config.rebalancing_no_sell = _parse_bool(rows["rebalancing_no_sell"])

        # Date / string fields
        if rows.get("portfolio_inception_date"):
            config.portfolio_inception_date = rows["portfolio_inception_date"]

        # Dict fields
        _parse_invested_allocation(config, rows)
        _parse_equity_geo(config, rows)

        # Warn on unknown keys
        _warn_unknown_keys(rows)

        # Validate sums
        _validate_sum_to_100(
            config.invested_allocation_targets_pctg,
            "invested_allocation_targets_pctg",
            normalize=True,
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
    "target_invested_allocation_alternative_pctg": "Alternative",
    "target_invested_allocation_real_estate_pctg": "Real Estate",
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
    "rebalancing_min_transaction_eur",
    "rebalancing_max_tolerance_pctg",
    "rebalancing_threshold_pctg",
    "rebalancing_precision_pctg",
    "rebalancing_no_sell",
    "target_cash_buffer_eur",
    "portfolio_inception_date",
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
    total = sum(d.values())
    if abs(total - 100.0) > 0.01:
        logger.warning("%s sums to %.2f%%, expected 100%%", name, total)
        if normalize:
            normalized = normalize_percentages(d)
            d.clear()
            d.update(normalized)
