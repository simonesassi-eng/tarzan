"""Durable, purpose-specific delivery identities and claim state machine."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tarzan.runtime.io_utils import atomic_write_bytes, canonical_json_bytes


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
        canonical = canonical_json_bytes({
            "identity_schema_version": DELIVERY_IDENTITY_SCHEMA_VERSION,
            "stable_event_id": self.stable_event_id,
            "purpose": self.purpose.value,
            "recipient_set_digest": self.recipient_set_digest,
            "template_schema_version": self.template_schema_version,
            "authorized_resend_token": self.authorized_resend_token,
        }, ascii_only=True)
        return hashlib.sha256(canonical).hexdigest()

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
        payload = canonical_json_bytes(document, ascii_only=True)
        atomic_write_bytes(self.path, payload)

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
                if current == target:
                    return target
                raise ValueError(f"invalid delivery transition {current.value} -> {target.value}")
            record["state"] = target.value
            self._write(document)
            return target


def recipient_set_digest(recipients: list[str]) -> str:
    canonical = "\n".join(sorted({item.strip().casefold() for item in recipients if item.strip()}))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_CLAIM_ATTEMPTS = 3
_CLAIM_BACKOFF_SECONDS = 2.0


class _TransientClaimError(RuntimeError):
    """A claim failure the service provably did not commit — safe to retry."""


class AppsScriptPropertiesDeliveryClaimStore(DeliveryClaimStore):
    """Authenticated HTTP adapter for lock-serialized Apps Script Properties."""

    def __init__(
        self,
        endpoint: str,
        auth_token: str,
        *,
        # Must exceed the service's own lock wait (tryLock(20000) in
        # scripts/apps_script/Code.gs) plus Apps Script cold start. A shorter
        # client timeout abandons a request the service is still serializing
        # behind the scheduler tick, turning routine contention into a blocked
        # publication.
        timeout_seconds: float = 35.0,
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
        """Post one claim action, retrying only knowably-transient failures.

        The service shares its script lock with the Gmail scheduler tick, and
        Google returns occasional 5xx on web apps, so a single unlucky attempt
        must not block an otherwise healthy publication. Retried failures are
        confined to ones the service could not have committed (lock contention),
        that never reached it (network/5xx), or where a 2xx response body was
        too garbled to read (the request likely reached the service, but a
        corrupted body says nothing about whether it committed -- see
        _call_once); a definitive rejection or short retention still fails
        closed on the first response.
        """
        last: Exception | None = None
        for attempt in range(1, _CLAIM_ATTEMPTS + 1):
            try:
                return self._call_once(request)
            except _TransientClaimError as error:
                last = error
                if attempt == _CLAIM_ATTEMPTS:
                    break
                time.sleep(_CLAIM_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"{last} after {_CLAIM_ATTEMPTS} attempts") from None

    def _call_once(self, request: dict) -> dict:
        # Apps Script web-app events do not reliably expose custom headers, so
        # the credential is carried only in the TLS-protected request body. It
        # is never persisted by the service or included in errors/evidence.
        wire = dict(request)
        wire["auth_token"] = self._auth_token
        payload = canonical_json_bytes(wire, ascii_only=True)
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
            message = f"delivery claim service rejected request with HTTP {error.code}"
            if error.code >= 500:
                raise _TransientClaimError(message) from None
            raise RuntimeError(message) from None
        except (URLError, TimeoutError, OSError) as error:
            raise _TransientClaimError(
                f"delivery claim service request failed ({type(error).__name__})"
            ) from None
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # The service returned a 2xx (an HTTPError would have been raised
            # above), so the request most likely reached it -- a garbled body
            # says nothing about whether it committed the claim before the
            # response was corrupted in transit. A retry is safe regardless:
            # intent.stable_event_id is the idempotency key the service's own
            # claim lookup is keyed on, so a retry either claims cleanly (the
            # first attempt did not commit) or comes back "duplicate" (it did)
            # -- never a double send. Observed as a real, transient failure
            # (2026-08-05, workflow:30989210047): the very next scheduled run
            # succeeded with no code change, consistent with a one-off glitch
            # rather than a persistent one this retry would not help.
            raise _TransientClaimError(
                "delivery claim service returned invalid JSON") from None
        if not isinstance(document, dict) or document.get("ok") is not True:
            code = document.get("error_code", "CLAIM_SERVICE_ERROR") if isinstance(document, dict) else "CLAIM_SERVICE_ERROR"
            message = f"delivery claim service failed with {code}"
            # Lock contention is rejected before the service touches state, so
            # a later attempt cannot duplicate an already-committed claim.
            if code == "LOCK_UNAVAILABLE":
                raise _TransientClaimError(message)
            raise RuntimeError(message)
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
