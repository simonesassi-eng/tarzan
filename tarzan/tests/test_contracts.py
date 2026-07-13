"""Contracts: explicit input schema, boundary validation gate, output DTO.

Covers the launch-hardening of the data contracts:
  * the declarative input schema is the single source of truth for the format;
  * column validation rejects missing-required (always) and unknown columns
    (strict only), tolerating unknowns by default so existing files/the
    automated newsletter are unaffected;
  * to_summary_dict is a stable, versioned external contract whose key set is
    pinned so a rename/removal is caught.
"""

from __future__ import annotations

import io
import json

import pytest

from tarzan.contracts import schema as sch
from tarzan.data.loader import load_orders, ORDER_REQUIRED_COLUMNS
from tarzan.contracts.exceptions import DataIngestionError
from tarzan.models.portfolio import (
    PortfolioMetrics,
    SUMMARY_CONTRACT_KEYS,
    SUMMARY_CONTRACT_OPTIONAL_KEYS,
)
from tarzan.contracts.validation import validate_columns


# --- Input schema ----------------------------------------------------------

class TestInputSchema:
    def test_order_schema_required_columns(self):
        req = sch.ORDER_LIST_SCHEMA.required_columns()
        assert req == {"date", "type", "isin", "quantity", "gross_eur", "net_eur"}

    def test_loader_constant_derives_from_schema(self):
        # Single source of truth: the loader constant IS the schema's required set.
        assert set(ORDER_REQUIRED_COLUMNS) == sch.ORDER_LIST_SCHEMA.required_columns()

    def test_schema_has_version_and_markdown(self):
        assert sch.SCHEMA_VERSION >= 1
        md = sch.ORDER_LIST_SCHEMA.to_markdown()
        assert "order_list.csv" in md and "date" in md and "Required" in md
        # full reference renders all files
        assert "targets_per_holding.csv" in sch.schemas_markdown()


# --- Boundary validation ---------------------------------------------------

class TestValidateColumns:
    def test_missing_required_is_fatal_both_modes(self):
        for strict in (False, True):
            fatal, _ = validate_columns(["date", "type", "isin"],
                                        sch.ORDER_LIST_SCHEMA, strict=strict)
            assert fatal and "missing required" in fatal

    def test_unknown_column_lenient_warns_not_fatal(self):
        cols = ["date", "type", "isin", "quantity", "gross_eur", "net_eur", "junk"]
        fatal, warnings = validate_columns(cols, sch.ORDER_LIST_SCHEMA, strict=False)
        assert fatal is None
        assert warnings and "junk" in warnings[0]

    def test_unknown_column_strict_is_fatal(self):
        cols = ["date", "type", "isin", "quantity", "gross_eur", "net_eur", "junk"]
        fatal, _ = validate_columns(cols, sch.ORDER_LIST_SCHEMA, strict=True)
        assert fatal and "junk" in fatal

    def test_all_known_columns_clean(self):
        cols = [c.name for c in sch.ORDER_LIST_SCHEMA.columns]
        fatal, warnings = validate_columns(cols, sch.ORDER_LIST_SCHEMA, strict=True)
        assert fatal is None and warnings == []


class TestLoaderStrictGate:
    def _csv(self, extra=""):
        head = "date,type,isin,quantity,gross_eur,net_eur" + extra + "\n"
        row = "2025-01-01,buy,IE00B4L5Y983,10,1000,-1000" + (",x" if extra else "") + "\n"
        return io.BytesIO((head + row).encode())

    def test_lenient_default_accepts_unknown_column(self):
        orders = load_orders(self._csv(",junk_col"), "x.csv")  # strict defaults False
        assert len(orders) == 1

    def test_strict_rejects_unknown_column(self):
        with pytest.raises(DataIngestionError):
            load_orders(self._csv(",junk_col"), "x.csv", strict=True)

    def test_strict_accepts_clean_file(self):
        orders = load_orders(self._csv(), "x.csv", strict=True)
        assert len(orders) == 1


# --- External output contract ----------------------------------------------

class TestSummaryContract:
    def test_base_keys_always_present(self):
        m = PortfolioMetrics()  # holdings-only defaults
        s = m.to_summary_dict()
        assert SUMMARY_CONTRACT_KEYS.issubset(s.keys())

    def test_optional_keys_only_on_order_path(self):
        m = PortfolioMetrics()  # xirr/twror all None
        s = m.to_summary_dict()
        assert not (SUMMARY_CONTRACT_OPTIONAL_KEYS & set(s.keys()))
        m2 = PortfolioMetrics()
        m2.xirr_pct = 12.3
        m2.twror_pct = 8.0
        m2.twror_annualized_pct = 15.0
        m2.returns_coverage_pct = 100.0
        s2 = m2.to_summary_dict()
        assert SUMMARY_CONTRACT_OPTIONAL_KEYS.issubset(s2.keys())

    def test_no_unexpected_top_level_keys(self):
        # Guards against an accidental new/renamed key silently entering the
        # external contract.
        m = PortfolioMetrics()
        m.xirr_pct = 1.0
        m.twror_pct = 1.0
        m.twror_annualized_pct = 1.0
        m.returns_coverage_pct = 1.0
        allowed = SUMMARY_CONTRACT_KEYS | SUMMARY_CONTRACT_OPTIONAL_KEYS
        assert set(m.to_summary_dict().keys()) == allowed

    def test_summary_is_strict_json(self):
        m = PortfolioMetrics()
        m.risk = {"sharpe": float("nan")}  # non-finite must serialize as null
        json.dumps(m.to_summary_dict(), allow_nan=False)  # must not raise

    def test_schema_version_in_summary(self):
        assert PortfolioMetrics().to_summary_dict()["schema_version"] >= 1
