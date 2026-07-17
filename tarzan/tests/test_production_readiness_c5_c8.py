"""Pre-fix exploration oracles for production-readiness conditions C5-C8.

These tests intentionally describe the required behavior. They must fail on the
unfixed revision and remain unchanged while the corresponding fixes are built.
All fixtures are deterministic, local, and non-secret.
"""

from __future__ import annotations

import io
import threading

import pytest

from tarzan.contracts.exceptions import DataIngestionError
from tarzan.data import price_cache
from tarzan.data.loader import load_orders, load_targets_per_holding
from tarzan.runtime import data_quality, report_html


def _status_name(value: object) -> str | None:
    status = getattr(value, "status", None)
    return getattr(status, "value", status)


# **Validates: Requirements 2.5**
def test_c5_duplicate_canonical_target_rows_invalidate_the_whole_target_set():
    """Equivalent/conflicting duplicates must retain rows and block planning.

    Counterexample: both rows canonicalize to the same ISIN. The unfixed loader
    overwrites the first row and returns a planning-ready mapping containing the
    second value, with no source-row diagnostic.
    """
    source = io.BytesIO(
        b"name,isin,ticker,target_equities,target_fixed_income\n"
        b"first,IE00B4L5Y983,IWDA,120,30\n"
        b"second,ie00b4l5y983,iwda,130,20\n"
    )

    outcome = load_targets_per_holding(source, "duplicate_targets.csv")
    errors = list(getattr(outcome, "errors", ()))
    duplicate = next(
        (
            error
            for error in errors
            if getattr(error, "code", None) == "DUPLICATE_TARGET_ROW"
        ),
        None,
    )
    source_rows = tuple(getattr(duplicate, "source_rows", ())) if duplicate else ()

    assert _status_name(outcome) == "INVALID", (
        "duplicate target rows were collapsed into a normal mapping instead of "
        f"invalidating planning: outcome={outcome!r}"
    )
    assert getattr(outcome, "planning_eligible", None) is False
    assert source_rows == (2, 3), (
        "duplicate diagnostics must preserve every original source row; "
        f"observed={source_rows!r}"
    )


# **Validates: Requirements 2.6**
def test_c6_strict_contract_rejects_a_partially_invalid_order_ledger():
    """Strict type enforcement must reject the input, not commit valid rows.

    Counterexample: one valid BUY plus one row with a non-numeric required
    quantity. The unfixed strict path validates columns but silently skips the
    malformed row and returns a partial financial ledger.
    """
    ledger = io.BytesIO(
        b"date,type,isin,quantity,gross_eur,net_eur,fees_eur\n"
        b"2025-01-01,buy,IE00B4L5Y983,10,1000,-1000,0\n"
        b"2025-01-02,buy,IE00B4L5Y983,not-a-number,500,-500,0\n"
    )

    with pytest.raises(DataIngestionError, match="row|quantity|numeric|finite"):
        load_orders(ledger, "partial_invalid.csv", strict=True)


# **Validates: Requirements 2.7**
def test_c7_fallback_report_has_complete_identity_lifecycle_and_impacts():
    """A selected fallback cannot be rendered as a coarse warning/clean run."""
    data_quality.reset()
    try:
        data_quality.warning(
            "market-data",
            "Primary quote failed; cached quote selected.",
            context="IE00B4L5Y983",
        )
        rendered = report_html.render("2025-06-30 00:00", [])
    finally:
        data_quality.reset()

    required_labels = {
        "Failure ID",
        "Original failure",
        "Remedies",
        "Selected fallback",
        "Provenance",
        "Availability",
        "Affected section",
        "Analytical impact",
        "Publication impact",
    }
    missing = sorted(label for label in required_labels if label not in rendered)
    false_clean = "every input parsed and priced cleanly" in rendered

    assert missing == [] and not false_clean, (
        "report.html does not preserve the complete fallback lifecycle; "
        f"missing={missing}, false_clean={false_clean}"
    )


# **Validates: Requirements 2.8**
def test_c8_cache_uses_a_non_code_executing_versioned_representation(
    monkeypatch, tmp_path,
):
    """Normal cache persistence must not emit executable pickle bytes."""
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)

    price_cache.store_resolution("IE00B4L5Y983", "IWDA.AS")
    payload = price_cache._resolution_path().read_bytes()

    assert payload.lstrip().startswith(b"{"), (
        "cache uses a code-executing/non-versioned binary representation; "
        f"prefix={payload[:8]!r}"
    )
    assert b'"schema_version"' in payload


# **Validates: Requirements 2.8**
def test_c8_overlapping_cache_updates_preserve_both_commits(monkeypatch, tmp_path):
    """A synchronized stale-writer schedule must not lose a committed key."""
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)

    real_load = price_cache._load_resolution_map
    writer_a_loaded = threading.Event()
    release_writer_a = threading.Event()

    def staged_load() -> dict:
        snapshot = real_load()
        if threading.current_thread().name == "c8-writer-a":
            writer_a_loaded.set()
            assert release_writer_a.wait(timeout=5)
        return snapshot

    def writer_a() -> None:
        price_cache.store_resolution("IE00AAAA0001", "AAAA.MI")

    with monkeypatch.context() as patch:
        patch.setattr(price_cache, "_load_resolution_map", staged_load)
        thread = threading.Thread(target=writer_a, name="c8-writer-a")
        thread.start()
        assert writer_a_loaded.wait(timeout=5)
        try:
            price_cache.store_resolution("IE00BBBB0002", "BBBB.MI")
        finally:
            release_writer_a.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    persisted = {
        "IE00AAAA0001": price_cache.load_resolution("IE00AAAA0001"),
        "IE00BBBB0002": price_cache.load_resolution("IE00BBBB0002"),
    }
    assert persisted == {
        "IE00AAAA0001": "AAAA.MI",
        "IE00BBBB0002": "BBBB.MI",
    }, f"overlapping cache update lost a committed key: {persisted}"
