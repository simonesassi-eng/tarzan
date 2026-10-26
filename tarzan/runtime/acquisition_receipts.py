"""Closed optional admission evidence at diagnostic capture, not a scale gate."""

from datetime import datetime, timezone
import math

from tarzan.runtime.workload import BoundedMeasure, MAX_RECORDED_INTEGER


FINAL_ACQUISITION_ERROR = "FINAL_ACQUISITION_UNAVAILABLE"
_NATIVE_COUNTS = (
    "capacity", "active", "active_optional", "essential_demand",
    "queued_essential", "queued_optional", "requested", "acquired",
    "released", "rejected", "peak",
)
_PROTECTED_COUNTS = (
    "groups_created", "activations", "active", "pending", "reserved_pending",
    "queued", "acquired", "released",
)
_MEASURE_FIELDS = frozenset({
    "value", "unit", "minimum", "maximum_recorded", "truncated",
})
_FIELDS = frozenset({
    "schema_version", "scope", "native_acquisition", "protected_acquisition",
    "admission_closed", "truncated", "descriptive_only", "affects_financial_decisions",
})
_SCOPE = "native_owner_at_diagnostic_capture"
_WINDOW_FIELDS = frozenset({"started_at", "completed_at"})


def _observation_instant():
    """Receipt time for this observer, never the analysis/as-of clock."""
    return datetime.now(timezone.utc)


def _observation_window(started, completed):
    if any(
        type(value) is not datetime or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
        for value in (started, completed)
    ) or completed < started:
        raise ValueError("invalid final acquisition observation clock")
    return {
        "started_at": started.isoformat(timespec="microseconds"),
        "completed_at": completed.isoformat(timespec="microseconds"),
    }


def _validate_observation_window(window):
    if type(window) is not dict or set(window) != _WINDOW_FIELDS:
        raise ValueError("invalid final acquisition observation window")
    instants = []
    for name in ("started_at", "completed_at"):
        value = window[name]
        if type(value) is not str or len(value) != 32:
            raise ValueError("invalid final acquisition observation timestamp")
        instant = datetime.fromisoformat(value)
        if (
            instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant)
            or instant.isoformat(timespec="microseconds") != value
        ):
            raise ValueError("invalid final acquisition observation timestamp")
        instants.append(instant)
    if instants[1] < instants[0]:
        raise ValueError("invalid final acquisition observation chronology")


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid native acquisition count")
    return BoundedMeasure(
        min(value, MAX_RECORDED_INTEGER), "count",
        truncated=value > MAX_RECORDED_INTEGER,
    ).to_dict()


def native_final_acquisition(result):
    """Never borrow an ambient owner or acquire/wait for a financial permit."""
    from tarzan.runtime.acquisition import RequestCoordinator
    from tarzan.runtime.session import matching_native_session
    from tarzan.runtime.transport import ProviderBudget

    session = matching_native_session(result)
    if session is None or type(session.transport) is not ProviderBudget:
        return None
    coordinator = session.transport.coordinator
    if type(coordinator) is not RequestCoordinator:
        return None
    started = _observation_instant()
    snapshot = coordinator.try_observation()
    if snapshot is None:
        raise ValueError("native acquisition observation contended")
    completed = _observation_instant()
    native, protected = snapshot["native_acquisition"], snapshot["protected_acquisition"]
    counts = {name: _count(native[name]) for name in _NATIVE_COUNTS}
    wait = native["queue_wait_seconds"]
    if type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0:
        raise ValueError("invalid native queue wait")
    maximum_seconds = MAX_RECORDED_INTEGER / 1000.0
    truncated_wait = wait > maximum_seconds
    wait_ms = (
        MAX_RECORDED_INTEGER if truncated_wait
        else min(round(wait * 1000, 3), MAX_RECORDED_INTEGER)
    )
    counts["queue_wait_ms"] = BoundedMeasure(
        wait_ms, "milliseconds", truncated=truncated_wait,
    ).to_dict()
    protected_counts = {name: _count(protected[name]) for name in _PROTECTED_COUNTS}
    observation = {
        "schema_version": "1",
        "scope": _SCOPE,
        "native_acquisition": counts,
        "protected_acquisition": protected_counts,
        "admission_closed": native["closed"],
        "truncated": any(
            measure["truncated"]
            for group in (counts, protected_counts) for measure in group.values()
        ),
        "descriptive_only": True,
        "affects_financial_decisions": False,
        "observation_window": _observation_window(started, completed),
    }
    validate_final_acquisition(observation)
    return observation


def validate_final_acquisition(observation):
    """Reject arbitrary labels, impossible fields and nonfinite cipher content."""
    if type(observation) is not dict or set(observation) not in (
        _FIELDS, _FIELDS | {"observation_window"},
    ):
        raise ValueError("invalid final acquisition fields")
    if (
        observation["schema_version"] != "1" or observation["scope"] != _SCOPE
        or observation["descriptive_only"] is not True
        or observation["affects_financial_decisions"] is not False
        or type(observation["admission_closed"]) is not bool
        or type(observation["truncated"]) is not bool
    ):
        raise ValueError("invalid final acquisition scope")
    if "observation_window" in observation:
        _validate_observation_window(observation["observation_window"])
    groups = (
        ("native_acquisition", {*_NATIVE_COUNTS, "queue_wait_ms"}),
        ("protected_acquisition", set(_PROTECTED_COUNTS)),
    )
    truncated = False
    for name, fields in groups:
        group = observation[name]
        if type(group) is not dict or set(group) != fields:
            raise ValueError("invalid final acquisition group")
        for field, measure in group.items():
            unit = "milliseconds" if field == "queue_wait_ms" else "count"
            if type(measure) is not dict or set(measure) != _MEASURE_FIELDS:
                raise ValueError("invalid final acquisition measure")
            value = measure["value"]
            valid_type = type(value) in (int, float) if unit == "milliseconds" else type(value) is int
            try:
                finite = valid_type and math.isfinite(value)
            except OverflowError:
                finite = False
            if (
                not finite or not 0 <= value <= MAX_RECORDED_INTEGER
                or measure["unit"] != unit
                or type(measure["minimum"]) is not int or measure["minimum"] != 0
                or type(measure["maximum_recorded"]) is not int
                or measure["maximum_recorded"] != MAX_RECORDED_INTEGER
                or type(measure["truncated"]) is not bool
            ):
                raise ValueError("invalid final acquisition value")
            truncated |= measure["truncated"]
    if observation["truncated"] is not truncated:
        raise ValueError("inconsistent final acquisition recording bound")
