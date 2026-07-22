"""Append-only run evidence, typed availability, and failure projection."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Iterable, Mapping, Optional, TypeVar


LEDGER_SCHEMA_VERSION = "1.0"


class Availability(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


T = TypeVar("T")


@dataclass(frozen=True)
class SectionResult(Generic[T]):
    availability: Availability
    value: Optional[T]
    failure_refs: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.availability is Availability.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable section cannot carry a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability.value,
            "value": self.value,
            "failure_refs": list(self.failure_refs),
            "provenance_refs": list(self.provenance_refs),
        }


class LedgerEntryType(str, Enum):
    FAILURE_OPEN = "FAILURE_OPEN"
    REMEDY = "REMEDY"
    FAILURE_CLOSE = "FAILURE_CLOSE"
    PROVIDER_ATTEMPT = "PROVIDER_ATTEMPT"
    BOUNDARY = "BOUNDARY"
    CAPABILITY = "CAPABILITY"
    PLAN = "PLAN"
    STAGE = "STAGE"
    ARTIFACT = "ARTIFACT"
    PUBLICATION = "PUBLICATION"
    DELIVERY = "DELIVERY"
    STORAGE = "STORAGE"
    TELEMETRY = "TELEMETRY"


@dataclass(frozen=True)
class LedgerEntry:
    sequence: int
    entry_id: str
    entry_type: LedgerEntryType
    recorded_at: datetime
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "recorded_at": self.recorded_at.isoformat(),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    stage: str
    stable_code: str
    severity: str
    original_failure: Mapping[str, Any]
    remedies: tuple[Mapping[str, Any], ...]
    automatically_corrected: bool
    selected_resolution: Optional[str]
    provenance: tuple[str, ...]
    availability: Availability
    affected_outputs: tuple[str, ...]
    analytical_impact: str
    publication_impact: str
    closed: bool


class ErrorNormalizer:
    """Recursively sanitize exceptions/evidence before any persistent sink."""

    _SENSITIVE_KEYS = re.compile(
        r"(?i)(api[_-]?key|authorization|cookie|credential|password|prompt|payload|secret|token)"
    )
    _SECRET_VALUE = re.compile(
        r"(?i)(bearer\s+[a-z0-9._~+/-]+=*|api[_-]?key\s*[:=]\s*\S+|"
        r"password\s*[:=]\s*\S+|token\s*[:=]\s*\S+)"
    )
    _KEYED_URL = re.compile(r"([?&](?:key|token|signature|credential)=)[^&\s]+", re.I)

    @classmethod
    def normalize(cls, value: Any) -> Any:
        if isinstance(value, BaseException):
            return {
                "type": type(value).__name__,
                "message": cls._sanitize_text(str(value)),
            }
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, Mapping):
            return {
                str(key): "[REDACTED]" if cls._SENSITIVE_KEYS.search(str(key))
                else cls.normalize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls.normalize(item) for item in value]
        if isinstance(value, str):
            return cls._sanitize_text(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return cls._sanitize_text(repr(value))

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        text = cls._SECRET_VALUE.sub("[REDACTED]", text)
        return cls._KEYED_URL.sub(r"\1[REDACTED]", text)


class RunLedger:
    """Thread-safe append-only evidence authority for exactly one run."""

    def __init__(self, attempt_id: str) -> None:
        self.attempt_id = attempt_id
        self._lock = threading.RLock()
        self._entries: list[LedgerEntry] = []
        self._failure_ordinals: dict[tuple[str, str], int] = {}

    def append(self, entry_type: LedgerEntryType, payload: Mapping[str, Any]) -> LedgerEntry:
        normalized = ErrorNormalizer.normalize(payload)
        with self._lock:
            sequence = len(self._entries) + 1
            stable = json.dumps(
                {"type": entry_type.value, "payload": normalized, "sequence": sequence},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            )
            entry_id = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
            entry = LedgerEntry(
                sequence=sequence,
                entry_id=entry_id,
                entry_type=entry_type,
                recorded_at=datetime.now(timezone.utc),
                payload=normalized,
            )
            self._entries.append(entry)
            return entry

    def open_failure(
        self,
        *,
        stage: str,
        stable_code: str,
        severity: str,
        error: Any,
        affected_outputs: Iterable[str],
        analytical_impact: str,
        publication_impact: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> str:
        with self._lock:
            key = (stage, stable_code)
            ordinal = self._failure_ordinals.get(key, 0) + 1
            self._failure_ordinals[key] = ordinal
        normalized_error = ErrorNormalizer.normalize(error)
        if not isinstance(normalized_error, Mapping):
            normalized_error = {"message": normalized_error}
        raw = f"failure-v1|{stage}|{stable_code}|{ordinal}"
        failure_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        self.append(LedgerEntryType.FAILURE_OPEN, {
            "failure_id": failure_id,
            "stage": stage,
            "stable_code": stable_code,
            "severity": severity,
            "original_failure": normalized_error,
            "context": dict(context or {}),
            "affected_outputs": list(affected_outputs),
            "analytical_impact": analytical_impact,
            "publication_impact": publication_impact,
        })
        return failure_id

    def remedy(
        self,
        failure_id: str,
        *,
        remedy_id: str,
        action: str,
        outcome: str,
        availability: Availability,
        provenance: Iterable[str] = (),
    ) -> LedgerEntry:
        remedies = [
            entry for entry in self.entries
            if entry.entry_type is LedgerEntryType.REMEDY
            and entry.payload.get("failure_id") == failure_id
        ]
        return self.append(LedgerEntryType.REMEDY, {
            "failure_id": failure_id,
            "ordinal": len(remedies) + 1,
            "remedy_id": remedy_id,
            "action": action,
            "outcome": outcome,
            "availability": availability.value,
            "provenance": list(provenance),
        })

    def close_failure(
        self,
        failure_id: str,
        *,
        automatically_corrected: bool,
        selected_resolution: Optional[str],
        availability: Availability,
        provenance: Iterable[str] = (),
    ) -> LedgerEntry:
        return self.append(LedgerEntryType.FAILURE_CLOSE, {
            "failure_id": failure_id,
            "automatically_corrected": automatically_corrected,
            "selected_resolution": selected_resolution,
            "availability": availability.value,
            "provenance": list(provenance),
        })

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def failure_records(self) -> tuple[FailureRecord, ...]:
        opens = {
            entry.payload["failure_id"]: entry
            for entry in self.entries
            if entry.entry_type is LedgerEntryType.FAILURE_OPEN
        }
        records: list[FailureRecord] = []
        for failure_id, opened in opens.items():
            remedies = tuple(
                dict(entry.payload)
                for entry in self.entries
                if entry.entry_type is LedgerEntryType.REMEDY
                and entry.payload.get("failure_id") == failure_id
            )
            closure = next((
                entry for entry in reversed(self.entries)
                if entry.entry_type is LedgerEntryType.FAILURE_CLOSE
                and entry.payload.get("failure_id") == failure_id
            ), None)
            close_payload = closure.payload if closure else {}
            records.append(FailureRecord(
                failure_id=failure_id,
                stage=str(opened.payload["stage"]),
                stable_code=str(opened.payload["stable_code"]),
                severity=str(opened.payload["severity"]),
                original_failure=dict(opened.payload["original_failure"]),
                remedies=remedies,
                automatically_corrected=bool(close_payload.get("automatically_corrected", False)),
                selected_resolution=close_payload.get("selected_resolution"),
                provenance=tuple(close_payload.get("provenance", ())),
                availability=Availability(close_payload.get("availability", Availability.UNAVAILABLE.value)),
                affected_outputs=tuple(opened.payload.get("affected_outputs", ())),
                analytical_impact=str(opened.payload.get("analytical_impact", "")),
                publication_impact=str(opened.payload.get("publication_impact", "")),
                closed=closure is not None,
            ))
        return tuple(records)
