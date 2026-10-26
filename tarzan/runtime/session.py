"""Run ownership, serialization, isolation ports, and stable identities."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.memo import ProviderMemo
from tarzan.runtime.input_receipts import immutable_fingerprints


logger = logging.getLogger(__name__)


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
    providers: ProviderMemo = field(default_factory=ProviderMemo, repr=False)
    transport: Any = field(default=None, repr=False)
    log_records: list[dict] = field(default_factory=list, repr=False)
    gate: Optional["SerialExecutionGate"] = field(default=None, repr=False)
    analysis_id: Optional[str] = None
    _input_fingerprints: Mapping[str, str] = field(
        default_factory=lambda: immutable_fingerprints({}),
        init=False, repr=False, compare=False,
    )
    _input_fingerprints_bound: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.config_snapshot = MappingProxyType(dict(self.config_snapshot))

    def bind_config(self, config: Mapping[str, Any]) -> None:
        """Bind the effective configuration once after validated loading."""
        if self.config_snapshot:
            raise RuntimeError("run session configuration is already bound")
        self.config_snapshot = MappingProxyType(dict(config))

    @property
    def input_fingerprints(self) -> Mapping[str, str]:
        """Descriptive receipts only; excluded from canonical financial facts."""
        return self._input_fingerprints

    def bind_input_fingerprints(self, values) -> None:
        """Retain one immutable receipt on this fresh, completed input attempt."""
        if self._input_fingerprints_bound:
            raise RuntimeError("input fingerprints are already bound")
        receipt = immutable_fingerprints(values)
        self._input_fingerprints = receipt
        self._input_fingerprints_bound = True

    def close_transport(self) -> None:
        """Close owned clients, not the retained deadline/policy authority.

        A later output can acquire a new client under the same budget. Ordinary
        cleanup failures must not replace the analysis/output's primary result.
        """
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                logger.warning(
                    "Provider client cleanup failed; primary outcome retained.",
                    extra={"public_code": "PROVIDER_CLIENT_CLEANUP_FAILED"},
                )


@dataclass(frozen=True)
class RunResult:
    metrics: Any
    config: Any
    attempt_id: str
    analysis_id: str
    ledger: Any
    session: Optional[RunSession] = field(default=None, repr=False, compare=False)
    report_view: Any = field(default=None, repr=False, compare=False)

    def compatibility_tuple(self) -> tuple[Any, Any]:
        return self.metrics, self.config


def matching_native_session(result) -> Optional[RunSession]:
    """Read an exact attempt's diagnostic owner, never an ambient/prior run."""
    from tarzan.runtime.ledger import RunLedger

    if type(result) is not RunResult or type(result.session) is not RunSession:
        return None
    session = result.session
    if (
        type(session.context) is not RunContext
        or session.context.attempt_id != result.attempt_id
        or session.ledger is not result.ledger
        or type(session.ledger) is not RunLedger
        or session.ledger.attempt_id != result.attempt_id
    ):
        return None
    return session


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


def current_or_matching_session() -> Optional[RunSession]:
    """Read a completed session only when the ambient attempt still matches."""
    active = current_session()
    if active is not None:
        return active
    result = last_run_result()
    if result is None or result.session is None:
        return None
    from tarzan import runtime

    return result.session if runtime.context().attempt_id == result.attempt_id else None


@contextmanager
def completed_run_scope(result: RunResult):
    """Lease and bind the original owner through render/finalization/delivery."""
    session = result.session
    active = current_session()
    if session is None:
        # Compatibility for injected callers that have no production lifecycle.
        yield result
        return
    if active is session:
        yield result
        return
    if active is not None:
        raise RuntimeError("cannot render another run inside an active session")
    from tarzan import runtime

    envelope = RunAttemptEnvelope(
        result.attempt_id, session.context.captured_at, "completed-run-output",
    )
    lease = session.gate.acquire(envelope) if session.gate is not None else nullcontext()
    with lease, runtime.bind_context(session.context), activate_session(session):
        token = _last_run_result.set(result)
        try:
            yield result
        finally:
            try:
                # Optional completed-run work can create fresh native clients
                # after the orchestrator has closed its analysis connections.
                # Nested scopes leave them with the outer owner until it exits.
                session.close_transport()
            finally:
                _last_run_result.reset(token)


@dataclass(frozen=True)
class RunRenderContext:
    """Explicit completed owner for callers rendering older analyses."""

    result: RunResult

    def activate(self):
        return completed_run_scope(self.result)

    def project(self):
        if self.result.report_view is None:
            return self.result.compatibility_tuple()
        return self.result.report_view.project()


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
