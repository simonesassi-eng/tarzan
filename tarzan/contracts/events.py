"""Closed workflow outputs and explicit operational delivery freshness."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re


MAX_DELIVERY_AGE_SECONDS = 45 * 60
MAX_FUTURE_SKEW_SECONDS = 120


def timestamp(value):
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError("request timestamp is missing or invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("request timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError("request timestamp must include its timezone")
    return parsed.astimezone(timezone.utc)


def _optional_text(mapping, key, default):
    value = mapping.get(key)
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise ValueError("workflow text field has an invalid type")
    return value


class FreshnessState(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SKIPPED_STALE = "SKIPPED_STALE"
    BLOCKED_INVALID_REQUEST = "BLOCKED_INVALID_REQUEST"


@dataclass(frozen=True)
class FreshnessAssessment:
    state: FreshnessState
    reason: str
    age_seconds: float | None = None

    @property
    def eligible(self):
        return self.state is FreshnessState.ELIGIBLE


def assess_freshness(requested_at, run_created_at, *, now, automated=False):
    if not automated and not requested_at and not run_created_at:
        return FreshnessAssessment(FreshnessState.ELIGIBLE, "LOCAL_UNQUEUED_REQUEST")
    if automated and not run_created_at:
        return FreshnessAssessment(FreshnessState.BLOCKED_INVALID_REQUEST, "NATIVE_RUN_TIMESTAMP_REQUIRED")
    try:
        stamps = [timestamp(value) for value in (requested_at, run_created_at) if value]
    except ValueError:
        return FreshnessAssessment(FreshnessState.BLOCKED_INVALID_REQUEST, "INVALID_REQUEST_TIMESTAMP")
    if not stamps or now.tzinfo is None:
        return FreshnessAssessment(FreshnessState.BLOCKED_INVALID_REQUEST, "INVALID_OPERATIONAL_CLOCK")
    ages = [(now.astimezone(timezone.utc) - stamp).total_seconds() for stamp in stamps]
    if min(ages) < -MAX_FUTURE_SKEW_SECONDS:
        return FreshnessAssessment(FreshnessState.BLOCKED_INVALID_REQUEST, "FUTURE_REQUEST_TIMESTAMP")
    age = max(ages)
    if age >= MAX_DELIVERY_AGE_SECONDS:
        return FreshnessAssessment(FreshnessState.SKIPPED_STALE, "MAX_DELIVERY_AGE_EXCEEDED", age)
    return FreshnessAssessment(FreshnessState.ELIGIBLE, "WITHIN_DELIVERY_WINDOW", age)


def resolve_workflow_event(event, *, event_name, run_id, run_created_at):
    if not isinstance(event, dict) or not re.fullmatch(r"[0-9]{1,24}", str(run_id)):
        raise ValueError("invalid native workflow event")
    created = timestamp(run_created_at).isoformat()
    if event_name == "repository_dispatch":
        payload = event.get("client_payload", {})
        if not isinstance(payload, dict):
            raise ValueError("invalid repository dispatch payload")
        label = _optional_text(payload, "label", "on-demand")
        raw_id = _optional_text(payload, "event_id", f"workflow:{run_id}")
        if not isinstance(raw_id, str) or not 1 <= len(raw_id) <= 256 or any(c in raw_id for c in "\r\n"):
            raise ValueError("invalid stable event ID")
        # Existing native public protocol IDs keep their durable claim identity.
        # Arbitrary caller context is never copied verbatim to public outputs.
        native_id = re.fullmatch(
            r"(?:workflow:[0-9]{1,24}|update:[a-f0-9]{64}|scheduled:\d{4}-\d{2}-\d{2}:[a-z0-9-]{1,40})",
            raw_id,
        )
        stable_id = raw_id if native_id else "dispatch:" + hashlib.sha256(raw_id.encode()).hexdigest()
        requested = timestamp(_optional_text(payload, "timestamp", created)).isoformat()
    elif event_name == "workflow_dispatch":
        inputs = event.get("inputs")
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            raise ValueError("invalid workflow inputs")
        label = _optional_text(inputs, "trigger_label", "manual")
        stable_id, requested = f"workflow:{run_id}", created
    else:
        raise ValueError("unsupported newsletter event")
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_ .:/-]{0,79}", label):
        raise ValueError("trigger label must be a short single-line label")
    return {
        "label": label, "event_id": stable_id,
        "requested_at": requested, "run_created_at": created,
    }
