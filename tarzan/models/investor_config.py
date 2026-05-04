"""Investor configuration model with serialization and normalization.

Handles loading investor preferences from CSV key-value files and provides
sensible defaults from the YAML configuration layer.
"""

from __future__ import annotations

import csv
import json
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
    """Investor profile with allocation targets and rebalancing preferences."""

    rebalancing_lump_sum_amount: float = 0.0
    geo_allocation: dict[str, float] = field(
        default_factory=lambda: dict(cfg.default_geo_allocation())
    )
    allocation_targets: dict[str, float] = field(
        default_factory=lambda: dict(cfg.default_allocation_targets())
    )
    rebalancing_threshold: float = 5.0
    rebalancing_precision: float = 0.5
    rebalancing_min_transaction_eur: float = 500.0
    rebalancing_max_tolerance: float = 2.0
    rebalancing_no_sell: bool = False
    portfolio_backtest_period: str = ""
    portfolio_inception: str = ""

    def __post_init__(self):
        """Fill portfolio_backtest_period and portfolio_inception from configs.csv if not set."""
        if not self.portfolio_backtest_period:
            self.portfolio_backtest_period = cfg.portfolio_backtest_period()
        if not self.portfolio_inception:
            self.portfolio_inception = cfg.portfolio_inception()

    @classmethod
    def from_csv(cls, path: str) -> "InvestorConfig":
        """Load investor config from a CSV key-value file."""
        config = cls()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = {
                row["key"].strip(): row["value"].strip()
                for row in reader
                if "key" in row and "value" in row
            }

        # Simple float fields
        _set_float(config, rows, "rebalancing_lump_sum_amount")
        _set_float(config, rows, "rebalancing_threshold")
        _set_float(config, rows, "rebalancing_precision")
        _set_float(config, rows, "rebalancing_min_transaction_eur")
        _set_float(config, rows, "rebalancing_max_tolerance")

        # Boolean flags
        if "rebalancing_no_sell" in rows:
            config.rebalancing_no_sell = str(rows["rebalancing_no_sell"]).strip().lower() in ("true", "1", "yes")

        # String fields
        if "portfolio_inception" in rows and rows["portfolio_inception"]:
            config.portfolio_inception = rows["portfolio_inception"]

        # Legacy JSON format (backward compatible)
        _set_json_dict(config, rows, "geo_allocation", cfg.default_geo_allocation())
        _set_json_dict(config, rows, "allocation_targets", cfg.default_allocation_targets())

        # Flat key format: target_geo_allocation_* and target_asset_allocation_*
        _parse_flat_geo(config, rows)
        _parse_flat_allocation(config, rows)

        # Normalize to 100%
        total_alloc = sum(config.allocation_targets.values())
        if abs(total_alloc - 100.0) > 0.01:
            logger.warning("allocation_targets sum=%.2f%%, normalizing to 100%%", total_alloc)
            config.allocation_targets = normalize_percentages(config.allocation_targets)

        total_geo = sum(config.geo_allocation.values())
        if abs(total_geo - 100.0) > 0.01:
            logger.warning("geo_allocation sum=%.2f%%, normalizing to 100%%", total_geo)
            config.geo_allocation = normalize_percentages(config.geo_allocation)

        return config

    @classmethod
    def from_dict(cls, rows: dict[str, str]) -> "InvestorConfig":
        """Load investor config from a pre-parsed key-value dict."""
        config = cls()
        _set_float(config, rows, "rebalancing_lump_sum_amount")
        _set_float(config, rows, "rebalancing_threshold")
        _set_float(config, rows, "rebalancing_precision")
        _set_float(config, rows, "rebalancing_min_transaction_eur")
        _set_float(config, rows, "rebalancing_max_tolerance")

        # Boolean flags
        if "rebalancing_no_sell" in rows:
            config.rebalancing_no_sell = str(rows["rebalancing_no_sell"]).strip().lower() in ("true", "1", "yes")

        # String fields
        if "portfolio_inception" in rows and rows["portfolio_inception"]:
            config.portfolio_inception = rows["portfolio_inception"]

        _set_json_dict(config, rows, "geo_allocation", cfg.default_geo_allocation())
        _set_json_dict(config, rows, "allocation_targets", cfg.default_allocation_targets())
        _parse_flat_geo(config, rows)
        _parse_flat_allocation(config, rows)

        total_alloc = sum(config.allocation_targets.values())
        if abs(total_alloc - 100.0) > 0.01:
            config.allocation_targets = normalize_percentages(config.allocation_targets)
        total_geo = sum(config.geo_allocation.values())
        if abs(total_geo - 100.0) > 0.01:
            config.geo_allocation = normalize_percentages(config.geo_allocation)
        return config


def _set_float(config: InvestorConfig, rows: dict, key: str) -> None:
    if key in rows:
        try:
            setattr(config, key, float(rows[key]))
        except (ValueError, TypeError):
            logger.warning("Failed to parse %s='%s', using default", key, rows[key])


def _set_str(config: InvestorConfig, rows: dict, key: str) -> None:
    if key in rows and rows[key]:
        setattr(config, key, rows[key])


def _set_json_dict(config: InvestorConfig, rows: dict, key: str, default: dict) -> None:
    if key in rows:
        try:
            parsed = json.loads(rows[key])
            if isinstance(parsed, dict):
                setattr(config, key, {k: float(v) for k, v in parsed.items()})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass


# Flat key → canonical name mapping (supports both old and new prefixes)
_GEO_KEY_MAP = {
    "target_geo_allocation_usa": "USA",
    "target_geo_allocation_japan": "Japan",
    "target_geo_allocation_eurozone_emu": "Eurozone EMU",
    "target_geo_allocation_dev_ex_usa_ex_emu_ex_jp": "Dev ex-USA ex-EMU ex-JP",
    "target_geo_allocation_emerging_markets": "Emerging Markets",
    # Legacy keys (backward compatible)
    "geo_allocation_usa": "USA",
    "geo_allocation_japan": "Japan",
    "geo_allocation_eurozone_emu": "Eurozone EMU",
    "geo_allocation_dev_ex_usa_ex_emu_ex_jp": "Dev ex-USA ex-EMU ex-JP",
    "geo_allocation_emerging_markets": "Emerging Markets",
}

_ALLOC_KEY_MAP = {
    "target_asset_allocation_equities": "Equities",
    "target_asset_allocation_fixed_income": "Fixed Income",
    "target_asset_allocation_cash": "Cash & Cash Equivalents",
    "target_asset_allocation_commodities": "Commodities",
    "target_asset_allocation_alternative": "Alternative",
    "target_asset_allocation_real_estate": "Real Estate",
    # Legacy keys (backward compatible)
    "allocation_target_equities": "Equities",
    "allocation_target_fixed_income": "Fixed Income",
    "allocation_target_cash": "Cash & Cash Equivalents",
    "allocation_target_commodities": "Commodities",
    "allocation_target_alternative": "Alternative",
    "allocation_target_real_estate": "Real Estate",
}


def _parse_flat_geo(config: InvestorConfig, rows: dict) -> None:
    found = {}
    for csv_key, canonical in _GEO_KEY_MAP.items():
        if csv_key in rows:
            try:
                found[canonical] = float(rows[csv_key])
            except (ValueError, TypeError):
                logger.warning("Failed to parse %s='%s'", csv_key, rows[csv_key])
    if found:
        config.geo_allocation = found


def _parse_flat_allocation(config: InvestorConfig, rows: dict) -> None:
    found = {}
    for csv_key, canonical in _ALLOC_KEY_MAP.items():
        if csv_key in rows:
            try:
                found[canonical] = float(rows[csv_key])
            except (ValueError, TypeError):
                logger.warning("Failed to parse %s='%s'", csv_key, rows[csv_key])
    if found:
        config.allocation_targets = found
