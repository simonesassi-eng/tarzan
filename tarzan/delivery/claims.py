"""Durable, purpose-specific delivery identities and claim state machine."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DELIVERY_IDENTITY_SCHEMA_VERSION = "1.0"
DELIVERY_STATE_SCHEMA_VERSION = "1.0"


class DeliveryPurpose(str, Enum):
    NORMAL_NEWSLETTER = "NORMAL_NEWSLETTER"
    CRITICAL_FAILURE_NOTIFICATION = "CRITICAL_FAILURE_NOTIFICATION"


class DeliveryState(str, Enum):
    CLAIMED = "CLAIMED"
    SMTP_INVOCATION_STARTED = "SMTP_INVOCATION_STARTED"
    ACKNOWLEDGED_SUCCESS = "ACKNOWLEDGED_SUCCESS"
    DEFINITE_PRE_SEND_FAILURE = "DEFINITE_PRE_SEND_FAILURE"
    UNCERTAIN = "UNCERTAIN"


_ALLOWED_DELIVERY_TRANSITIONS: dict[DeliveryState, frozenset[DeliveryState]] = {
    DeliveryState.CLAIMED: frozenset({
        DeliveryState.SMTP_INVOCATION_STARTED,
        DeliveryState.DEFINITE_PRE_SEND_FAILURE,
    }),
    DeliveryState.SMTP_INVOCATION_STARTED: frozenset({
        DeliveryState.ACKNOWLEDGED_SUCCESS,
        DeliveryState.UNCERTAIN,
    }),
    DeliveryState.ACKNOWLEDGED_SUCCESS: frozenset(),
    DeliveryState.DEFINITE_PRE_SEND_FAILURE: frozenset(),
    DeliveryState.UNCERTAIN: frozenset(),
}


@dataclass(frozen=True)
class DeliveryIntent:
    stable_event_id: str
    purpose: DeliveryPurpose
    recipient_set_digest: str
    template_schema_version: str
    authorized_resend_token: Optional[str] = None

    @property
    def logical_id(self) -> str:
        canonical = json.dumps({
            "identity_schema_version": DELIVERY_IDENTITY_SCHEMA_VERSION,
            "stable_event_id": self.stable_event_id,
            "purpose": self.purpose.value,
            "recipient_set_digest": self.recipient_set_digest,
            "template_schema_version": self.template_schema_version,
            "authorized_resend_token": self.authorized_resend_token,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def intent_digest(self) -> str:
        # Deliberately contains no raw recipient, market value, or content.
        return hashlib.sha256(
            f"{self.purpose.value}|{self.recipient_set_digest}|{self.template_schema_version}".encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ClaimResult:
    created: bool
    duplicate: bool
    conflict: bool
    state: DeliveryState


class DeliveryClaimStore:
    @staticmethod
    def _validate_transition_request(
        expected: tuple[DeliveryState, ...],
        target: DeliveryState,
    ) -> None:
        """Reject any CAS request that could authorize an illegal edge.

        Validation belongs to the store contract rather than one adapter so a
        permissive or stale remote endpoint cannot bypass the local state
        machine. Every state accepted by the CAS request must legally reach
        the requested target.
        """
        if not expected or any(
            target not in _ALLOWED_DELIVERY_TRANSITIONS[state]
            for state in expected
        ):
            expected_names = ",".join(state.value for state in expected) or "<empty>"
            raise ValueError(
                f"invalid delivery transition {expected_names} -> {target.value}"
            )

    def claim(self, intent: DeliveryIntent) -> ClaimResult:
        raise NotImplementedError

    def transition(
        self,
        logical_id: str,
        expected: tuple[DeliveryState, ...],
        target: DeliveryState,
    ) -> DeliveryState:
        raise NotImplementedError


_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class LocalJsonDeliveryClaimStore(DeliveryClaimStore):
    """Transactional durable store for local/manual operation and tests."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self):
        key = str(self.path)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.RLock())
        with lock:
            lock_path = self.path.with_suffix(self.path.suffix + ".lock")
            with open(lock_path, "a+b") as handle:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                try:
                    yield
                finally:
                    try:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except (ImportError, OSError):
                        pass

    def _read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": DELIVERY_STATE_SCHEMA_VERSION, "claims": {}}
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema_version") != DELIVERY_STATE_SCHEMA_VERSION:
            raise ValueError("incompatible delivery claim schema")
        if not isinstance(document.get("claims"), dict):
            raise ValueError("invalid delivery claim store")
        return document

    def _write(self, document: dict) -> None:
        payload = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def claim(self, intent: DeliveryIntent) -> ClaimResult:
        with self._lock():
            document = self._read()
            existing = document["claims"].get(intent.logical_id)
            if existing:
                conflict = existing.get("intent_digest") != intent.intent_digest
                return ClaimResult(
                    created=False,
                    duplicate=not conflict,
                    conflict=conflict,
                    state=DeliveryState(existing["state"]),
                )
            document["claims"][intent.logical_id] = {
                "intent_digest": intent.intent_digest,
                "purpose": intent.purpose.value,
                "state": DeliveryState.CLAIMED.value,
            }
            self._write(document)
            return ClaimResult(True, False, False, DeliveryState.CLAIMED)

    def transition(
        self,
        logical_id: str,
        expected: tuple[DeliveryState, ...],
        target: DeliveryState,
    ) -> DeliveryState:
        self._validate_transition_request(expected, target)
        with self._lock():
            document = self._read()
            record = document["claims"].get(logical_id)
            if not record:
                raise KeyError(f"delivery claim not found: {logical_id}")
            current = DeliveryState(record["state"])
            if current not in expected:
                raise ValueError(f"invalid delivery transition {current.value} -> {target.value}")
            record["state"] = target.value
            self._write(document)
            return target


def recipient_set_digest(recipients: list[str]) -> str:
    canonical = "\n".join(sorted({item.strip().casefold() for item in recipients if item.strip()}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AppsScriptPropertiesDeliveryClaimStore(DeliveryClaimStore):
    """Authenticated HTTP adapter for lock-serialized Apps Script Properties."""

    def __init__(
        self,
        endpoint: str,
        auth_token: str,
        *,
        timeout_seconds: float = 15.0,
        minimum_retention_days: int = 30,
    ) -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("delivery claim endpoint must use HTTPS")
        if not auth_token:
            raise ValueError("delivery claim service credential is required")
        if timeout_seconds <= 0 or minimum_retention_days <= 0:
            raise ValueError("claim-service timeout and retention must be positive")
        self.endpoint = endpoint
        self._auth_token = auth_token
        self.timeout_seconds = float(timeout_seconds)
        self.minimum_retention_days = int(minimum_retention_days)

    def _call(self, request: dict) -> dict:
        # Apps Script web-app events do not reliably expose custom headers, so
        # the credential is carried only in the TLS-protected request body. It
        # is never persisted by the service or included in errors/evidence.
        wire = dict(request)
        wire["auth_token"] = self._auth_token
        payload = json.dumps(
            wire, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        http_request = Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = response.read(64 * 1024)
        except HTTPError as error:
            raise RuntimeError(
                f"delivery claim service rejected request with HTTP {error.code}"
            ) from None
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"delivery claim service request failed ({type(error).__name__})"
            ) from None
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("delivery claim service returned invalid JSON") from None
        if not isinstance(document, dict) or document.get("ok") is not True:
            code = document.get("error_code", "CLAIM_SERVICE_ERROR") if isinstance(document, dict) else "CLAIM_SERVICE_ERROR"
            raise RuntimeError(f"delivery claim service failed with {code}")
        retention = document.get("retention_days")
        if not isinstance(retention, int) or retention < self.minimum_retention_days:
            raise RuntimeError("delivery claim retention is below the reconciliation window")
        return document

    def claim(self, intent: DeliveryIntent) -> ClaimResult:
        document = self._call({
            "action": "claim",
            "state_schema_version": DELIVERY_STATE_SCHEMA_VERSION,
            "logical_id": intent.logical_id,
            "intent_digest": intent.intent_digest,
            "purpose": intent.purpose.value,
        })
        try:
            return ClaimResult(
                created=bool(document["created"]),
                duplicate=bool(document["duplicate"]),
                conflict=bool(document["conflict"]),
                state=DeliveryState(document["state"]),
            )
        except (KeyError, ValueError, TypeError):
            raise RuntimeError("delivery claim service returned an invalid claim result") from None

    def transition(
        self,
        logical_id: str,
        expected: tuple[DeliveryState, ...],
        target: DeliveryState,
    ) -> DeliveryState:
        self._validate_transition_request(expected, target)
        document = self._call({
            "action": "transition",
            "state_schema_version": DELIVERY_STATE_SCHEMA_VERSION,
            "logical_id": logical_id,
            "expected": [state.value for state in expected],
            "target": target.value,
        })
        try:
            returned = DeliveryState(document["state"])
        except (KeyError, ValueError, TypeError):
            raise RuntimeError("delivery claim service returned an invalid transition result") from None
        if returned is not target:
            raise RuntimeError(
                "delivery claim service did not apply the requested transition"
            )
        return returned
