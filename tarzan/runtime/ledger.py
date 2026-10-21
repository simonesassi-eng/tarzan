"""Append-only run evidence, typed availability, and failure projection."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Iterable, Mapping, Optional, TypeVar

from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.contracts.immutable import freeze_json, thaw_json


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze_json(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "entry_id": self.entry_id,
            "entry_type": self.entry_type.value,
            "recorded_at": self.recorded_at.isoformat(),
            "payload": thaw_json(self.payload),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_failure", freeze_json(self.original_failure))
        object.__setattr__(self, "remedies", tuple(freeze_json(item) for item in self.remedies))


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
            value = {item.name: getattr(value, item.name) for item in fields(value)}
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
        self._remedy_ordinals: dict[str, int] = {}
        self._failure_revision = 0
        self._failure_cache_revision = -1
        self._failure_cache: tuple[FailureRecord, ...] = ()

    def append(self, entry_type: LedgerEntryType, payload: Mapping[str, Any]) -> LedgerEntry:
        normalized = ErrorNormalizer.normalize(payload)
        with self._lock:
            if entry_type is LedgerEntryType.REMEDY:
                failure_id = normalized["failure_id"]
                ordinal = self._remedy_ordinals.get(failure_id, 0) + 1
                self._remedy_ordinals[failure_id] = ordinal
                normalized["ordinal"] = ordinal
            sequence = len(self._entries) + 1
            stable = canonical_json_bytes(
                {"type": entry_type.value, "payload": normalized, "sequence": sequence},
                ascii_only=True,
                default=str,
            )
            entry_id = hashlib.sha256(stable).hexdigest()[:24]
            entry = LedgerEntry(
                sequence=sequence,
                entry_id=entry_id,
                entry_type=entry_type,
                recorded_at=datetime.now(timezone.utc),
                payload=normalized,
            )
            self._entries.append(entry)
            if entry_type in (
                LedgerEntryType.FAILURE_OPEN, LedgerEntryType.REMEDY, LedgerEntryType.FAILURE_CLOSE,
            ):
                self._failure_revision += 1
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
        return self.append(LedgerEntryType.REMEDY, {
            "failure_id": failure_id,
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
        # Take one consistent snapshot and index it once. The old projection
        # re-copied and re-scanned the complete log for every open failure.
        with self._lock:
            revision = self._failure_revision
            if self._failure_cache_revision == revision:
                return self._failure_cache
            entries = tuple(self._entries)
        opens: dict[str, LedgerEntry] = {}
        remedy_index: dict[str, list[Mapping[str, Any]]] = {}
        closures: dict[str, LedgerEntry] = {}
        for entry in entries:
            failure_id = entry.payload.get("failure_id")
            if entry.entry_type is LedgerEntryType.FAILURE_OPEN:
                opens[failure_id] = entry
            elif entry.entry_type is LedgerEntryType.REMEDY:
                remedy_index.setdefault(failure_id, []).append(entry.payload)
            elif entry.entry_type is LedgerEntryType.FAILURE_CLOSE:
                closures[failure_id] = entry
        records: list[FailureRecord] = []
        for failure_id, opened in opens.items():
            remedies = tuple(remedy_index.get(failure_id, ()))
            closure = closures.get(failure_id)
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
        result = tuple(records)
        with self._lock:
            if self._failure_revision == revision:
                self._failure_cache = result
                self._failure_cache_revision = revision
        return result
