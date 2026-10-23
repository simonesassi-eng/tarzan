"""A cache optimization needs authenticated coverage, not an import exit code."""

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tarzan.data import price_cache, protected_cache as protected, public_cache as public
from tarzan.runtime import sealed
from tarzan.runtime.io_utils import canonical_json_bytes


KEY = b"synthetic-key-is-not-a-secret-00"
NOW = datetime(2026, 10, 23, 17, tzinfo=timezone.utc)


@pytest.fixture
def locations(monkeypatch, tmp_path):
    private, staging = tmp_path / "private", tmp_path / "sealed"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(private))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    return private, staging


def prices(*, end="2026-10-22", offset=0):
    return pd.DataFrame(
        {"Close": [100.0 + offset, 101.0 + offset, 102.0 + offset],
         "Price Close": [100.0 + offset, 101.0 + offset, 102.0 + offset]},
        index=pd.date_range(end=end, periods=3, tz="UTC", name="Date"),
    )


def archive_macros(staging):
    for symbol in sorted(public.PUBLIC_SYMBOLS):
        price_cache.store_history(symbol, prices())
    assert protected.export_snapshot(staging, KEY) == len(public.PUBLIC_SYMBOLS)


def mutate_archive(staging, operation):
    path = staging / protected.FILENAME
    document = json.loads(sealed.unseal(path.read_bytes(), KEY, purpose=protected.PURPOSE))
    operation(document)
    sealed.write_snapshot(path, canonical_json_bytes(document), KEY, purpose=protected.PURPOSE)


def refreshed_import(staging, monkeypatch, tmp_path):
    destination = tmp_path / "restored"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(destination))
    return protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW,
    ), destination


def test_complete_fresh_authenticated_macros_can_skip_public_restore(locations, monkeypatch, tmp_path):
    private, staging = locations
    archive_macros(staging)
    originals = {path.name: path.read_bytes() for path in (private / "history").glob("*.json")}
    result, destination = refreshed_import(staging, monkeypatch, tmp_path)
    assert result.restored_files == len(public.PUBLIC_SYMBOLS)
    assert result.public_macro_complete is True
    assert {path.name: path.read_bytes() for path in (destination / "history").glob("*.json")} == originals
    # The optimization proves cache-head coverage, not a live market observation.
    assert price_cache.load_history(sorted(public.PUBLIC_SYMBOLS)[0]).index[-1] == pd.Timestamp(
        "2026-10-22", tz="UTC",
    )


@pytest.mark.parametrize("case", [
    "missing", "stale", "future", "missing-close", "all-close-null", "boolean",
    "duplicate-date", "invalid-date", "nonmonotonic", "series",
])
def test_partial_or_unusable_macro_evidence_cannot_authorize_skip(
    locations, monkeypatch, tmp_path, case,
):
    _, staging = locations
    archive_macros(staging)
    symbol = sorted(public.PUBLIC_SYMBOLS)[0]
    relative = f"history/{price_cache._safe(symbol)}.json"

    def corrupt(document):
        cached = document["files"][relative]
        entries = cached["entries"]
        rows = entries["table"]["data"]
        if case == "missing":
            del document["files"][relative]
            return
        if case == "stale":
            for row in rows:
                row["Date"] = row["Date"].replace("2026-10", "2026-09")
        elif case == "future":
            rows[-1]["Date"] = "2026-10-24T00:00:00.000Z"
        elif case == "missing-close":
            entries["table"]["schema"]["fields"] = [
                field for field in entries["table"]["schema"]["fields"]
                if field["name"] != "Close"
            ]
            for row in rows:
                del row["Close"]
        elif case == "all-close-null":
            for row in rows:
                row["Close"] = None
        elif case == "boolean":
            rows[-1]["Close"] = True
        elif case == "duplicate-date":
            rows[-1]["Date"] = rows[-2]["Date"]
        elif case == "invalid-date":
            rows[-1]["Date"] = "not-a-date"
        elif case == "nonmonotonic":
            rows.reverse()
        else:
            entries["kind"] = "series"
        document["files"][relative] = price_cache._envelope("history", entries)

    mutate_archive(staging, corrupt)
    result, _ = refreshed_import(staging, monkeypatch, tmp_path)
    assert result.public_macro_complete is False
    assert result.restored_files == len(public.PUBLIC_SYMBOLS) - (case == "missing")


@pytest.mark.parametrize("case,expected", [
    ("recent-padding", True), ("future-padding", False), ("stale-with-new-padding", False),
])
def test_missing_tail_price_uses_the_last_real_close_without_rejuvenating_it(
    locations, monkeypatch, tmp_path, case, expected,
):
    _, staging = locations
    archive_macros(staging)
    first = sorted(public.PUBLIC_SYMBOLS)[0]
    relative = f"history/{price_cache._safe(first)}.json"

    def pad(document):
        entries = document["files"][relative]["entries"]
        rows = entries["table"]["data"]
        rows[-1]["Close"] = None
        if case == "future-padding":
            rows[-1]["Date"] = "2026-10-24T00:00:00.000Z"
        elif case == "stale-with-new-padding":
            for row in rows[:-1]:
                row["Date"] = row["Date"].replace("2026-10", "2026-09")
        document["files"][relative] = price_cache._envelope("history", entries)

    mutate_archive(staging, pad)
    result, _ = refreshed_import(staging, monkeypatch, tmp_path)
    assert result.public_macro_complete is expected
    # The source's null row remains null; bootstrap cannot invent a quote.
    restored = price_cache.load_history(first)
    assert pd.isna(restored["Close"].iloc[-1])
    if case == "recent-padding":
        assert restored["Close"].dropna().index[-1] == pd.Timestamp("2026-10-21", tz="UTC")


def test_cache_disabled_retains_fallback_even_with_authenticated_complete_document(locations, monkeypatch):
    _, staging = locations
    archive_macros(staging)
    monkeypatch.setenv("TARZAN_DISABLE_CACHE", "1")
    result = protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW,
    )
    assert result.public_macro_complete is False


def test_macro_proof_refuses_naive_reference_clock(locations):
    _, staging = locations
    archive_macros(staging)
    result = protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW.replace(tzinfo=None),
    )
    assert result.public_macro_complete is False


def test_successful_empty_import_is_not_complete_macro_coverage(locations):
    _, staging = locations
    result = protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW,
    )
    assert result.restored_files == 0
    assert result.public_macro_complete is False


def test_already_present_different_local_history_is_not_authenticated_by_an_old_archive(locations):
    _, staging = locations
    archive_macros(staging)
    first = sorted(public.PUBLIC_SYMBOLS)[0]
    price_cache.store_history(first, prices(offset=10))
    result = protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW,
    )
    assert result.restored_files == 0
    assert result.public_macro_complete is False
    assert price_cache.load_history(first)["Close"].iloc[-1] == 112


def test_already_present_exact_authenticated_histories_do_not_need_a_nonzero_import_count(
    locations, monkeypatch, tmp_path,
):
    _, staging = locations
    archive_macros(staging)
    refreshed_import(staging, monkeypatch, tmp_path)
    result = protected._import_snapshot(
        staging, KEY, prove_public_macro=True, reference_time=NOW,
    )
    assert result.restored_files == 0
    assert result.public_macro_complete is True


@pytest.mark.parametrize("age,expected", [
    (timedelta(days=5), True),
    (timedelta(days=5, milliseconds=1), False),
    (timedelta(milliseconds=-1), False),
])
def test_bootstrap_age_is_finite_causal_and_does_not_extend_refresh_policy(
    locations, monkeypatch, tmp_path, age, expected,
):
    _, staging = locations
    end = NOW - age
    for symbol in public.PUBLIC_SYMBOLS:
        price_cache.store_history(symbol, prices(end=end))
    protected.export_snapshot(staging, KEY)
    result, _ = refreshed_import(staging, monkeypatch, tmp_path)
    assert result.public_macro_complete is expected
    assert price_cache.REFRESH_TAIL_DAYS == 5


@pytest.mark.parametrize("case", ["wrong-key", "tag", "schema", "repair", "checksum"])
def test_cli_failures_never_emit_macro_complete_or_restore_unauthenticated_data(
    locations, monkeypatch, tmp_path, capsys, case,
):
    _, staging = locations
    archive_macros(staging)
    if case in {"schema", "repair", "checksum"}:
        def poison(document):
            if case == "schema":
                document["schema_version"] = "unsupported"
            elif case == "repair":
                document["history_repair_version"] = "unsupported"
            else:
                next(iter(document["files"].values()))["checksum"] = "bad"
        mutate_archive(staging, poison)
    elif case == "tag":
        path = staging / protected.FILENAME
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    destination = tmp_path / "restored"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(destination))
    monkeypatch.setenv(sealed.KEY_ENV, base64.b64encode(b"x" * 32 if case == "wrong-key" else KEY).decode())
    output = tmp_path / "actions-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert protected.main(["import", "--directory", str(staging)]) == 1
    assert output.read_text() == "public_macro_complete=false\n"
    assert not list(destination.rglob("*.json"))
    assert "Close" not in capsys.readouterr().out


def test_complete_import_cli_emits_only_the_fixed_public_coverage_boolean(
    locations, monkeypatch, tmp_path, capsys,
):
    _, staging = locations
    archive_macros(staging)
    original = protected._authenticated_public_macro_complete
    monkeypatch.setattr(
        protected, "_authenticated_public_macro_complete",
        lambda documents, **kwargs: original(documents, reference_time=NOW),
    )
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "restored"))
    monkeypatch.setenv(sealed.KEY_ENV, base64.b64encode(KEY).decode())
    output = tmp_path / "actions-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert protected.main(["import", "--directory", str(staging)]) == 0
    assert output.read_text() == "public_macro_complete=false\npublic_macro_complete=true\n"
    console = capsys.readouterr().out
    assert all(symbol not in console for symbol in public.PUBLIC_SYMBOLS)


@pytest.mark.parametrize("key", ["", "invalid", base64.b64encode(b"too-short").decode()])
def test_missing_or_invalid_cli_key_keeps_the_independent_fallback_enabled(
    locations, monkeypatch, tmp_path, key,
):
    _, staging = locations
    output = tmp_path / "actions-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv(sealed.KEY_ENV, key)
    assert protected.main(["import", "--directory", str(staging)]) == 1
    assert output.read_text() == "public_macro_complete=false\n"


def test_public_export_does_not_parse_previous_snapshot_when_private_coverage_is_complete(
    locations, monkeypatch, tmp_path,
):
    _, staging = locations
    archive_macros(staging)
    monkeypatch.setattr(public, "_read_snapshot", lambda _: pytest.fail("redundant previous parse"))
    assert public.export_snapshot(tmp_path / "public") == len(public.PUBLIC_SYMBOLS)


@pytest.mark.parametrize("previous", [b"not JSON", b'{"schema_version":"wrong"}'])
def test_corrupt_previous_public_snapshot_cannot_block_new_verified_projection(
    locations, tmp_path, previous,
):
    _, _ = locations
    first = sorted(public.PUBLIC_SYMBOLS)[0]
    price_cache.store_history(first, prices())
    destination = tmp_path / "public"
    destination.mkdir()
    (destination / "snapshot.json").write_bytes(previous)
    assert public.export_snapshot(destination) == 1
    assert set(public._read_snapshot(destination / "snapshot.json")) == {first}


def test_public_update_is_retained_even_when_redundant_restore_is_skipped():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/newsletter.yml").read_text(), Loader=yaml.BaseLoader)
    steps = {step["name"]: step for step in workflow["jobs"]["send"]["steps"]}
    imported = steps["Validate and import encrypted market snapshot"]
    assert imported["id"] == "protected_import"
    for name in ("Restore public macro snapshot", "Validate and import public macro snapshot"):
        condition = steps[name]["if"]
        assert "steps.protected_import.outputs.public_macro_complete != 'true'" in condition
        assert "outcome" not in condition and "cache-hit" not in condition
    assert steps["Project public macro snapshot"]["if"] == "always() && !cancelled()"
    assert "public_macro_complete" not in steps["Save verified public macro snapshot"]["if"]
    assert "--only-missing" in steps["Validate and import public macro snapshot"]["run"]
