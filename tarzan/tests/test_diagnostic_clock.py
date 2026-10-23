"""Optional native clock facts must not replace finance or private failure evidence."""

from dataclasses import replace
from datetime import date, datetime, timezone
import json
import os
import time
from types import SimpleNamespace

import pytest

from tarzan.runtime import diagnostics as diag
from tarzan.runtime.session import RunContext, RunMode
from tarzan.tests.test_encrypted_diagnostics import KEY, result


def document(context, *, original=None):
    original = result() if original is None else original
    owner = replace(original, session=SimpleNamespace(context=context))
    return diag.build_report(owner, operation="verify", exit_code=1, duration_seconds=2.5)


def context(*, mode=RunMode.LIVE, effective=None, captured=None):
    return RunContext(
        "synthetic-clock", mode, effective,
        captured or datetime(2026, 10, 21, 22, 14, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("zone", ["UTC", "Europe/Rome", "America/New_York"])
def test_native_clock_uses_actual_host_analysis_date_and_offset(monkeypatch, zone):
    if not hasattr(time, "tzset"):
        pytest.skip("host timezone fixture needs tzset")
    previous = os.environ.get("TZ")
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    try:
        owner = context()
        actual = document(owner)["run"]["clock"]
        assert actual == {
            "analysis_date": owner.analysis_date.isoformat(),
            "mode": "LIVE", "basis": "HOST_LOCAL",
            "captured_host_utc_offset_seconds": int(
                owner.captured_at.astimezone().utcoffset().total_seconds(),
            ),
        }
        assert set(actual) == diag._RUN_CLOCK_NAMES
        assert zone not in json.dumps(actual)
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.parametrize("mode", [RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE])
def test_native_effective_clock_retains_real_boundary_instead_of_host_day(mode):
    owner = context(mode=mode, effective=date(2025, 7, 1))
    clock = document(owner)["run"]["clock"]
    assert clock["analysis_date"] == "2025-07-01"
    assert clock["mode"] == mode.value
    assert clock["basis"] == "EFFECTIVE_DATE"


def test_live_context_with_explicit_effective_date_remains_supported():
    clock = document(context(mode=RunMode.LIVE, effective=date(2025, 7, 1)))["run"]["clock"]
    assert clock["mode"] == "LIVE"
    assert clock["basis"] == "EFFECTIVE_DATE"
    assert clock["analysis_date"] == "2025-07-01"


def test_naive_capture_does_not_invent_an_offset_or_timezone():
    owner = context(captured=datetime(2026, 10, 21, 22, 14))
    clock = document(owner)["run"]["clock"]
    assert clock["analysis_date"] == "2026-10-21"
    assert clock["captured_host_utc_offset_seconds"] is None


def test_generic_context_keeps_old_report_shape_and_does_not_read_new_getters():
    class LegacyContext:
        captured_at = datetime(2026, 10, 21, 10, tzinfo=timezone.utc)

        def __getattr__(self, name):
            pytest.fail(f"new generic-context getter evaluated: {name}")

    original = result()
    before = document(SimpleNamespace(captured_at=LegacyContext.captured_at), original=original)
    after = document(LegacyContext(), original=original)
    assert after == before
    assert "clock" not in after["run"]
    assert "clock_projection_error_type" not in after["run"]


def test_clock_is_optional_v1_and_roundtrip_preserves_primary_projection(tmp_path):
    from tarzan.runtime.io_utils import canonical_json_bytes

    original = result()
    old = document(SimpleNamespace(captured_at=context().captured_at), original=original)
    new = document(context(), original=original)
    assert set(old) == set(new) == diag._FIELDS
    assert old["schema_version"] == new["schema_version"] == "1"
    for name in ("quality", "summary", "kpis", "ledger", "provider_counts", "exit_code"):
        assert new[name] == old[name]
    for name, value in (("old", old), ("new", new)):
        diag.write_report(tmp_path / name, value, KEY)
        assert diag.read_report(tmp_path / name / diag.FILENAME, KEY) == value
        # The version-1 envelope and primary projections are unchanged.
        decoded = json.loads(canonical_json_bytes(value))
        assert set(decoded) == diag._FIELDS and isinstance(decoded["run"], dict)


@pytest.mark.parametrize("bad", [
    {}, {"analysis_date": "secret-canary"},
    {"analysis_date": "2026-10-21", "mode": "LIVE", "basis": "HOST_LOCAL",
     "captured_host_utc_offset_seconds": True},
    {"analysis_date": "2026-10-21", "mode": "PRIVATE-CANARY", "basis": "HOST_LOCAL",
     "captured_host_utc_offset_seconds": 0},
])
def test_corrupt_optional_clock_projection_keeps_encrypted_primary_ledger(
    monkeypatch, tmp_path, bad,
):
    monkeypatch.setattr(diag, "_native_run_clock", lambda _: bad)
    original = document(context())
    assert "clock" not in original["run"]
    assert original["run"]["clock_projection_error_type"] == "CLOCK_PROJECTION_UNAVAILABLE"
    assert original["ledger"][0]["payload"]["stable_code"] == "NEWSLETTER_SEMANTIC_INVARIANT_FAILED"
    diag.write_report(tmp_path / "sealed", original, KEY)
    assert diag.read_report(tmp_path / "sealed" / diag.FILENAME, KEY) == original
    assert "PRIVATE-CANARY" not in json.dumps(original)
    assert "secret-canary" not in json.dumps(original)


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_clock_observer_failure_does_not_replace_primary_outcome_or_capture(
    monkeypatch, tmp_path, error_type,
):
    def broken(_):
        raise error_type("synthetic-private-clock-canary")

    monkeypatch.setattr(diag, "_native_run_clock", broken)
    owner = replace(result(), session=SimpleNamespace(context=context()))
    monkeypatch.setenv(diag.DIRECTORY_ENV, str(tmp_path / "encrypted"))
    monkeypatch.setattr("tarzan.runtime.sealed.configured_key", lambda **kwargs: KEY)
    assert diag.capture_configured(owner, operation="verify", exit_code=1, duration_seconds=2.5)
    restored = diag.read_report(tmp_path / "encrypted" / diag.FILENAME, KEY)
    assert restored["exit_code"] == 1
    assert restored["run"]["clock_projection_error_type"] == "CLOCK_PROJECTION_UNAVAILABLE"
    assert restored["ledger"][0]["payload"]["stable_code"] == "NEWSLETTER_SEMANTIC_INVARIANT_FAILED"
    assert "synthetic-private-clock-canary" not in json.dumps(restored)


@pytest.mark.parametrize("field,value", [
    ("analysis_date", "not-a-date"), ("analysis_date", "2026-10-99"),
    ("mode", "PRIVATE"), ("mode", []), ("basis", "UTC-INVENTED"),
    ("captured_host_utc_offset_seconds", True),
    ("captured_host_utc_offset_seconds", 86400),
    ("captured_host_utc_offset_seconds", float("nan")),
])
def test_invalid_new_clock_fields_are_rejected_as_untrusted_metadata(tmp_path, field, value):
    report = document(context())
    report["run"]["clock"][field] = value
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "encrypted", report, KEY)
    assert not (tmp_path / "encrypted" / diag.FILENAME).exists()


def test_corrupt_but_well_formed_host_clock_uses_fixed_error_without_losing_primary_ledger(
    monkeypatch, tmp_path,
):
    bad = diag._native_run_clock(context())
    bad["analysis_date"] = "2025-01-01"
    monkeypatch.setattr(diag, "_native_run_clock", lambda _: bad)
    original = document(context())
    assert "clock" not in original["run"]
    assert original["run"]["clock_projection_error_type"] == "CLOCK_PROJECTION_UNAVAILABLE"
    diag.write_report(tmp_path / "sealed", original, KEY)
    restored = diag.read_report(tmp_path / "sealed" / diag.FILENAME, KEY)
    assert restored == original
    assert restored["ledger"][0]["payload"]["stable_code"] == "NEWSLETTER_SEMANTIC_INVARIANT_FAILED"


@pytest.mark.parametrize("captured,day,offset,valid", [
    ("2026-10-21T22:14:00+00:00", "2026-10-22", 7200, True),
    ("2026-10-21T22:14:00+00:00", "2026-10-21", -18000, True),
    ("2026-10-22T00:14:00+02:00", "2026-10-21", -18000, True),
    ("2026-10-21T22:14:00", "2026-10-21", None, True),
    ("2026-10-21T22:14:00+00:00", "2026-10-21", 7200, False),
    ("2026-10-21T22:14:00+00:00", "2026-10-22", -18000, False),
    ("2026-10-21T22:14:00", "2026-10-21", 7200, False),
    ("2026-10-21T22:14:00+00:00", "2026-10-21", None, False),
    (None, "2026-10-21", None, False),
])
def test_host_local_clock_validation_uses_captured_offset_not_decryption_host(
    tmp_path, captured, day, offset, valid,
):
    report = document(context())
    report["captured_at"] = captured
    report["run"]["clock"] = {
        "analysis_date": day, "mode": "LIVE", "basis": "HOST_LOCAL",
        "captured_host_utc_offset_seconds": offset,
    }
    if valid:
        diag.write_report(tmp_path / "sealed", report, KEY)
        assert diag.read_report(tmp_path / "sealed" / diag.FILENAME, KEY) == report
    else:
        with pytest.raises(ValueError):
            diag.write_report(tmp_path / "sealed", report, KEY)
        assert not (tmp_path / "sealed" / diag.FILENAME).exists()


@pytest.mark.parametrize("marker", [
    None, True, [], {}, "ValueError", "PRIVATE-CANARY",
])
def test_clock_error_status_is_closed_and_not_arbitrary_metadata(tmp_path, marker):
    report = document(context())
    report["run"].pop("clock")
    report["run"]["clock_projection_error_type"] = marker
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "sealed", report, KEY)
    assert not (tmp_path / "sealed" / diag.FILENAME).exists()


def test_clock_and_clock_error_cannot_both_claim_authority(tmp_path):
    report = document(context())
    report["run"]["clock_projection_error_type"] = "CLOCK_PROJECTION_UNAVAILABLE"
    with pytest.raises(ValueError):
        diag.write_report(tmp_path / "sealed", report, KEY)
    assert not (tmp_path / "sealed" / diag.FILENAME).exists()


def test_legacy_report_without_clock_keeps_existing_unconstrained_capture_contract(tmp_path):
    report = document(context())
    report["run"].pop("clock")
    report["captured_at"] = None
    diag.write_report(tmp_path / "sealed", report, KEY)
    assert diag.read_report(tmp_path / "sealed" / diag.FILENAME, KEY) == report
