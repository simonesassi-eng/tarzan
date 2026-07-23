"""Run ownership, serialization, isolation ports, and stable identities."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import deque
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol

from tarzan.runtime.io_utils import canonical_json_bytes


class RunMode(str, Enum):
    LIVE = "LIVE"
    POINT_IN_TIME = "POINT_IN_TIME"
    REPRODUCIBLE = "REPRODUCIBLE"


@dataclass(frozen=True)
class RunAttemptEnvelope:
    attempt_id: str
    created_at: datetime
    invocation_source: str

    @classmethod
    def create(cls, invocation_source: str = "unknown") -> "RunAttemptEnvelope":
        return cls(
            attempt_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            invocation_source=invocation_source,
        )


@dataclass(frozen=True)
class RunContext:
    attempt_id: str
    mode: RunMode
    effective_date: Optional[date]
    captured_at: datetime
    invocation_source: str = "unknown"
    schema_versions: Mapping[str, str] = field(default_factory=dict)
    policy_versions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode in (RunMode.POINT_IN_TIME, RunMode.REPRODUCIBLE) and self.effective_date is None:
            raise ValueError(f"{self.mode.value} requires an effective date")
        object.__setattr__(self, "schema_versions", MappingProxyType(dict(self.schema_versions)))
        object.__setattr__(self, "policy_versions", MappingProxyType(dict(self.policy_versions)))

    @property
    def analysis_date(self) -> date:
        if self.effective_date is not None:
            return self.effective_date
        # Operational capture is stored in UTC, but LIVE analysis follows the
        # host-local calendar used by the established CLI/newsletter contract.
        # Convert before taking the date so a run near local midnight cannot
        # be attributed to yesterday's UTC date.
        if self.captured_at.tzinfo is not None:
            return self.captured_at.astimezone().date()
        return self.captured_at.date()

    @property
    def allows_live_transport(self) -> bool:
        return self.mode is RunMode.LIVE


@dataclass
class RunSession:
    """The only owner of mutable analysis state for one active run."""

    context: RunContext
    config_snapshot: Mapping[str, Any]
    ledger: Any
    diagnostics: list[Any] = field(default_factory=list)
    audit: list[Any] = field(default_factory=list)
    memo: dict[Any, Any] = field(default_factory=dict)
    analysis_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.config_snapshot = MappingProxyType(dict(self.config_snapshot))

    def bind_config(self, config: Mapping[str, Any]) -> None:
        """Bind the effective configuration once after validated loading."""
        if self.config_snapshot:
            raise RuntimeError("run session configuration is already bound")
        self.config_snapshot = MappingProxyType(dict(config))


@dataclass(frozen=True)
class RunResult:
    metrics: Any
    config: Any
    attempt_id: str
    analysis_id: str
    ledger: Any

    def compatibility_tuple(self) -> tuple[Any, Any]:
        return self.metrics, self.config


_current_session: ContextVar[Optional[RunSession]] = ContextVar(
    "tarzan_active_run_session", default=None
)
_last_run_result: ContextVar[Optional[RunResult]] = ContextVar(
    "tarzan_last_run_result", default=None
)


@contextmanager
def activate_session(session: RunSession):
    """Make *session* the context-local authority for one leased run."""
    if _current_session.get() is not None:
        raise RuntimeError("a run session is already active in this context")
    token = _current_session.set(session)
    try:
        yield session
    finally:
        _current_session.reset(token)


def current_session() -> Optional[RunSession]:
    return _current_session.get()


def record_last_run_result(result: RunResult) -> None:
    _last_run_result.set(result)


def last_run_result() -> Optional[RunResult]:
    return _last_run_result.get()


IsolationScope = Optional[str]


class TenantPolicyPort(Protocol):
    def resolve(self, scope: IsolationScope, invocation: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ScopedStoragePort(Protocol):
    def open(self, scope: IsolationScope, context: RunContext) -> Any: ...


class SingleUserPolicyPort:
    def resolve(self, scope: IsolationScope, invocation: Mapping[str, Any]) -> Mapping[str, Any]:
        if scope is not None:
            raise ValueError("current single-user policy accepts only scope=None")
        return MappingProxyType(dict(invocation))


class LocalStoragePort:
    def __init__(self, root) -> None:
        self.root = root

    def open(self, scope: IsolationScope, context: RunContext):
        if scope is not None:
            raise ValueError("current local storage accepts only scope=None")
        return self.root / context.attempt_id


def canonical_analysis_id(evidence: Mapping[str, Any]) -> str:
    """Hash canonical output-affecting evidence; operational noise is excluded."""
    ignored = {"attempt_id", "wall_time", "latency", "retry_delay", "diagnostic_prose"}
    canonical = {key: value for key, value in evidence.items() if key not in ignored}
    encoded = canonical_json_bytes(canonical, ascii_only=True, default=str)
    return hashlib.sha256(encoded).hexdigest()


class _GateLease(AbstractContextManager["_GateLease"]):
    def __init__(self, gate: "SerialExecutionGate", ticket: int, envelope: RunAttemptEnvelope) -> None:
        self._gate = gate
        self.ticket = ticket
        self.envelope = envelope
        self._released = False

    def __enter__(self) -> "_GateLease":
        return self

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._gate._release(self.ticket)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class SerialExecutionGate:
    """Process-local FIFO gate enforcing one active single-user run."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: deque[int] = deque()
        self._next_ticket = 0
        self._active_ticket: Optional[int] = None

    def acquire(
        self,
        envelope: RunAttemptEnvelope,
        timeout: Optional[float] = None,
    ) -> _GateLease:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._queue.append(ticket)
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._active_ticket is not None or self._queue[0] != ticket:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._queue.remove(ticket)
                    self._condition.notify_all()
                    raise TimeoutError(f"serial execution lease timed out for {envelope.attempt_id}")
                self._condition.wait(remaining)
            self._queue.popleft()
            self._active_ticket = ticket
            return _GateLease(self, ticket, envelope)

    def _release(self, ticket: int) -> None:
        with self._condition:
            if self._active_ticket != ticket:
                raise RuntimeError("attempted to release a lease that is not active")
            self._active_ticket = None
            self._condition.notify_all()

    @property
    def active_count(self) -> int:
        with self._condition:
            return int(self._active_ticket is not None)

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return len(self._queue)
