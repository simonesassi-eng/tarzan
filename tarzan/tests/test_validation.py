"""Tests for the input-boundary validators and the data-quality report."""

from __future__ import annotations

import pytest

from tarzan.runtime import data_quality as dq
from tarzan.models.order import OrderType
from tarzan.contracts.validation import (
    check_order_sign,
    currency_is_known,
    isin_format_error,
    normalize_currency,
    normalize_isin,
)


class TestIsinFormat:
    def test_valid_isin_passes(self):
        assert isin_format_error("IE00B4L5Y983") is None

    def test_real_bond_isin_that_fails_mod10_still_passes_format(self):
        # US Treasury / BTP ISINs the review noted fail yfinance's check
        # digit must NOT be rejected — we validate format only.
        assert isin_format_error("US91282CGJ45") is None
        assert isin_format_error("IT0005542359") is None

    def test_wrong_length_rejected(self):
        assert isin_format_error("IE00B4L5") is not None
        assert isin_format_error("IE00B4L5Y9831") is not None

    def test_missing_country_prefix_rejected(self):
        assert isin_format_error("1200B4L5Y983") is not None

    def test_non_alphanumeric_rejected(self):
        assert isin_format_error("IE00B4L5Y9-3") is not None

    def test_empty_rejected(self):
        assert isin_format_error("") is not None

    def test_normalize_isin_uppercases_and_strips(self):
        assert normalize_isin("  ie00-b4l5 y983 ") == "IE00B4L5Y983"
        assert normalize_isin(None) == ""
        assert normalize_isin("nan") == ""


class TestCurrency:
    def test_known_iso_codes(self):
        for c in ("EUR", "USD", "GBP", "JPY", "ZAR"):
            assert currency_is_known(c)

    def test_minor_unit_codes_known(self):
        assert currency_is_known("GBp")
        assert currency_is_known("ZAc")

    def test_unknown_code_flagged(self):
        assert not currency_is_known("ZZZ")
        assert not currency_is_known("XYZ")

    def test_normalize_blank_defaults_eur(self):
        assert normalize_currency("") == "EUR"
        assert normalize_currency(None) == "EUR"
        assert normalize_currency("nan") == "EUR"

    def test_normalize_preserves_minor_unit_case(self):
        assert normalize_currency("GBp") == "GBp"
        assert normalize_currency("usd") == "USD"


class TestOrderSign:
    def test_buy_positive_unchanged(self):
        r = check_order_sign(OrderType.BUY, 10.0)
        assert r.quantity == 10.0
        assert r.message is None

    def test_sell_positive_is_corrected_to_negative(self):
        r = check_order_sign(OrderType.SELL, 30.0)
        assert r.quantity == -30.0
        assert r.message is not None

    def test_buy_negative_is_corrected_to_positive(self):
        r = check_order_sign(OrderType.BUY, -5.0)
        assert r.quantity == 5.0
        assert r.message is not None

    def test_transfer_out_positive_corrected(self):
        r = check_order_sign(OrderType.TRANSFER_OUT, 100.0)
        assert r.quantity == -100.0
        assert r.message is not None

    def test_coupon_zero_quantity_unchanged(self):
        r = check_order_sign(OrderType.COUPON, 0.0)
        assert r.quantity == 0.0
        assert r.message is None

    def test_sell_already_negative_unchanged(self):
        r = check_order_sign(OrderType.SELL, -30.0)
        assert r.quantity == -30.0
        assert r.message is None


class TestDataQualityReport:
    def test_clean_run_reports_no_issues(self):
        dq.reset()
        assert dq.issues() == []
        assert "no issues" in dq.summary_line().lower()
        assert "No issues this run" in dq.render()

    def test_records_and_counts(self):
        dq.reset()
        dq.warning("order_load", "skipped a bad row", context="row 3")
        dq.error("metrics", "computer failed")
        dq.info("returns", "coverage 90%")
        c = dq.counts()
        assert c == {"WARNING": 1, "ERROR": 1, "INFO": 1}
        assert len(dq.issues()) == 3

    def test_render_groups_by_source_and_is_greppable(self):
        dq.reset()
        dq.warning("order_load", "sign corrected", context="IE00X")
        dq.error("metrics", "boom")
        report = dq.render()
        # Greppable severity+source prefix.
        assert "[ERROR][metrics]" in report
        assert "[WARNING][order_load]" in report
        # Grouped section headers present.
        assert "## metrics" in report
        assert "## order_load" in report
        # Summary counts line.
        assert "SUMMARY:" in report

    def test_write_report_creates_file(self, tmp_path):
        dq.reset()
        dq.warning("order_load", "something", context="row 1")
        path = dq.write_report(str(tmp_path))
        assert path is not None
        content = (tmp_path / "data_quality.log").read_text()
        assert "TARZAN DATA-QUALITY REPORT" in content
        assert "something" in content

    def test_record_never_raises(self):
        dq.reset()
        # Bad args must not propagate — diagnostics can't break the pipeline.
        dq.record("WARNING", "src", "msg", context=None)
        assert len(dq.issues()) == 1
