"""Only fixed public macro tables can leave the private runtime cache."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tarzan.data import price_cache, public_cache as public
from tarzan.runtime.io_utils import canonical_json_bytes


@pytest.fixture
def cache_locations(monkeypatch, tmp_path):
    private, staging = tmp_path / "private", tmp_path / "public"
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(private))
    monkeypatch.delenv("TARZAN_DISABLE_CACHE", raising=False)
    return private, staging


def prices(*, timezone="Europe/Rome", offset=0):
    index = pd.date_range("2026-10-19", periods=3, tz=timezone, name="private-account")
    frame = pd.DataFrame({
        "Close": [101.0 + offset, np.nan, 103.0 + offset],
        "Volume": [10, 20, 30], "owner@example.invalid": ["private"] * 3,
    }, index=index)
    frame.attrs["ledger"] = {"quantity": 12345, "secret": "never-public"}
    frame.columns.name = "private-customer"
    return frame


def save_document(staging, mutate):
    path = staging / "snapshot.json"
    document = json.loads(path.read_bytes())
    document.pop("checksum")
    mutate(document)
    document["checksum"] = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    path.write_bytes(canonical_json_bytes(document))


def test_projection_does_not_copy_private_names_attrs_queries_or_manual_sources(cache_locations):
    private, staging = cache_locations
    macro = sorted(public.PUBLIC_SYMBOLS)[0]
    price_cache.store_history(macro, prices())
    price_cache.store_history("PRIVATE-HOLDING", prices())
    price_cache.store_history("manual:licensed", prices())
    (private / "identity-and-inputs.json").write_text('{"owner":"private"}')
    assert public.export_snapshot(staging) == 1
    path = staging / "snapshot.json"
    data = path.read_text()
    assert set(json.loads(data)["tables"]) == {macro}
    for forbidden in (
        "owner@example.invalid", "private-account", "private-customer", "never-public",
        "PRIVATE-HOLDING", "manual:licensed", "quantity", "ledger", "identity-and-inputs",
    ):
        assert forbidden not in data
    assert list(staging.iterdir()) == [path]
    assert staging.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("timezone", [None, "UTC", "Europe/Rome", "America/New_York"])
def test_round_trip_numeric_prices_dates_and_timezone(cache_locations, monkeypatch, tmp_path, timezone):
    _, staging = cache_locations
    macro = sorted(public.PUBLIC_SYMBOLS)[0]
    original = prices(timezone=timezone)
    price_cache.store_history(macro, original)
    assert public.export_snapshot(staging) == 1
    monkeypatch.setenv("TARZAN_CACHE_DIR", str(tmp_path / "fresh-private"))
    assert public.import_snapshot(staging) == 1
    restored = price_cache.load_history(macro)
    expected = original[["Close", "Volume"]].astype(float)
    expected.index.name, expected.columns.name = None, None
    pd.testing.assert_frame_equal(restored, expected, check_freq=False)
    assert restored.attrs == {}


def test_new_run_exports_incremental_changes_and_retains_other_validated_macro_tables(cache_locations):
    _, staging = cache_locations
    first, second = sorted(public.PUBLIC_SYMBOLS)[:2]
    price_cache.store_history(first, prices())
    price_cache.store_history(second, prices(offset=5))
    assert public.export_snapshot(staging) == 2
    old = (staging / "snapshot.json").read_bytes()
    price_cache.store_history(first, prices(offset=10))
    price_cache._history_path(second).unlink()
    assert public.export_snapshot(staging) == 2
    new = (staging / "snapshot.json").read_bytes()
    assert old != new
    tables = public._read_snapshot(staging / "snapshot.json")
    assert tables[first]["Close"].iloc[-1] == 113
    assert tables[second]["Close"].iloc[-1] == 108


@pytest.mark.parametrize("case", [
    "schema", "policy", "repair", "checksum", "root-field", "private-symbol",
    "table-field", "private-column", "duplicates", "row-shape", "boolean",
    "invalid-date", "duplicate-date", "nonmonotonic", "naive-with-zone",
    "unexpected-zone", "huge-zone", "too-many-rows",
])
def test_invalid_snapshot_is_rejected_before_any_private_write(cache_locations, monkeypatch, case):
    _, staging = cache_locations
    first, second = sorted(public.PUBLIC_SYMBOLS)[:2]
    price_cache.store_history(first, prices())
    price_cache.store_history(second, prices())
    public.export_snapshot(staging)

    def poison(document):
        table = document["tables"][second]
        if case == "schema":
            document["schema_version"] = "future"
        elif case == "policy":
            document["policy_id"] = "copy-private-directory"
        elif case == "repair":
            document["history_repair_version"] = "old"
        elif case == "root-field":
            document["private_input"] = "secret"
        elif case == "private-symbol":
            document["tables"]["PRIVATE-HOLDING"] = table
        elif case == "table-field":
            table["attrs"] = {"private": "secret"}
        elif case == "private-column":
            table["columns"][0] = "quantity"
        elif case == "duplicates":
            table["columns"] = ["Close", "Close"]
        elif case == "row-shape":
            table["values"][-1] = [1]
        elif case == "boolean":
            table["values"][0][0] = True
        elif case == "invalid-date":
            table["dates"][0] = "private-context"
        elif case == "duplicate-date":
            table["dates"][1] = table["dates"][0]
        elif case == "nonmonotonic":
            table["dates"] = list(reversed(table["dates"]))
        elif case == "naive-with-zone":
            table["dates"][0] = "2026-10-19T00:00:00"
        elif case == "unexpected-zone":
            table["timezone"] = None
        elif case == "huge-zone":
            table["timezone"] = "x" * 65
        elif case == "too-many-rows":
            table["dates"] *= public.MAX_ROWS
            table["values"] *= public.MAX_ROWS

    save_document(staging, poison)
    if case == "checksum":
        path = staging / "snapshot.json"
        document = json.loads(path.read_bytes())
        document["checksum"] = "0" * 64
        path.write_bytes(canonical_json_bytes(document))
    stored = []
    monkeypatch.setattr(price_cache, "store_history", lambda *args: stored.append(args))
    with pytest.raises((ValueError, TypeError, KeyError)):
        public.import_snapshot(staging)
    assert stored == []


@pytest.mark.parametrize("case", ["private", "child", "parent", "symlink", "symlink-parent", "unrelated"])
def test_public_staging_cannot_alias_private_storage_or_unrelated_files(cache_locations, tmp_path, case):
    private, staging = cache_locations
    private.mkdir()
    if case == "private":
        target = private
    elif case == "child":
        target = private / "public"
    elif case == "parent":
        target = tmp_path
    elif case == "symlink":
        staging.symlink_to(private, target_is_directory=True)
        target = staging
    elif case == "symlink-parent":
        real = tmp_path / "real"
        real.mkdir()
        staging.symlink_to(real, target_is_directory=True)
        target = staging / "nested"
    else:
        staging.mkdir()
        (staging / "caller-file.txt").write_text("caller-owned")
        target = staging
    with pytest.raises(ValueError):
        public.export_snapshot(target)
    if case == "unrelated":
        assert (staging / "caller-file.txt").read_text() == "caller-owned"
    assert not (private / "snapshot.json").exists()


def test_original_symbol_evidence_is_required_even_for_fixed_macro_filename(cache_locations, monkeypatch):
    _, staging = cache_locations
    macro = sorted(public.PUBLIC_SYMBOLS)[0]
    monkeypatch.setattr(price_cache, "load_history", lambda symbol: prices() if symbol == macro else None)
    monkeypatch.setattr(price_cache, "_read_json", lambda *args: {"symbol": "PRIVATE-HOLDING"})
    with pytest.raises(ValueError, match="original-symbol"):
        public.export_snapshot(staging)
    assert not staging.exists()


def test_save_output_is_written_only_for_verified_nonempty_export(cache_locations, monkeypatch, tmp_path, capsys):
    _, staging = cache_locations
    output = tmp_path / "actions-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert public.main(["export", "--directory", str(staging)]) == 0
    assert output.read_text() == "nonempty=false\n"
    macro = sorted(public.PUBLIC_SYMBOLS)[0]
    price_cache.store_history(macro, prices())
    assert public.main(["export", "--directory", str(staging)]) == 0
    assert output.read_text().endswith("nonempty=true\n")
    before = output.read_bytes()
    (staging / "private-file").write_text("sensitive")
    assert public.main(["export", "--directory", str(staging)]) == 1
    assert output.read_bytes() == before
    assert "sensitive" not in capsys.readouterr().out


def test_snapshot_size_limit_precedes_parsing(cache_locations, monkeypatch):
    _, staging = cache_locations
    staging.mkdir()
    (staging / "snapshot.json").write_bytes(b"x" * 5)
    monkeypatch.setattr(public, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="snapshot file"):
        public.import_snapshot(staging)


def test_corrupt_private_cache_diagnostics_are_not_published(cache_locations, monkeypatch):
    import io
    import logging
    from tarzan import log_redaction

    _, staging = cache_locations
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    logging.getLogger().addHandler(console)
    monkeypatch.setenv("CI", "true")

    def corrupt_read(symbol):
        logging.getLogger("tarzan.data.price_cache").warning(
            "private-account quantity=12345 symbol=AAPL return=12.34%%"
        )
        return None

    monkeypatch.setattr(price_cache, "load_history", corrupt_read)
    try:
        assert public.main(["export", "--directory", str(staging)]) == 0
        assert "PRIVATE_DIAGNOSTIC" in stream.getvalue()
        assert "private-account" not in stream.getvalue()
        assert "12345" not in stream.getvalue()
    finally:
        logging.getLogger().removeHandler(console)
        log_redaction.install(enabled=False)


def test_workflow_caches_only_the_public_projection_and_updates_each_run():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/newsletter.yml").read_text()
    assert "queue: max" in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.count("path: ~/.cache/tarzan-public/snapshot.json") == 2
    assert "path: ~/.cache/tarzan\n" not in workflow
    assert "tarzan-market-cache-" not in workflow
    assert workflow.count("${{ github.run_id }}-${{ github.run_attempt }}") == 2
    assert "actions/cache/restore@" in workflow and "actions/cache/save@" in workflow
    assert "steps.cache_export.outcome == 'success'" in workflow
    assert "steps.cache_export.outputs.nonempty == 'true'" in workflow
    assert "python -m tarzan.data.public_cache import" in workflow
    assert "python -m tarzan.data.public_cache export" in workflow
    assert "upload-artifact" not in workflow
