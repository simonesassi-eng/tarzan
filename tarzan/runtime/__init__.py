"""Context-local run clock and mode compatibility facade.

The authoritative run lifecycle lives in :mod:`tarzan.runtime.session`. This
module keeps the established clock helpers while storing the same immutable
``RunContext`` object owned by the active ``RunSession``. Live, point-in-time,
and reproducible modes therefore cannot split clocks or transport policy.
"""

from __future__ import annotations

import datetime
from contextvars import ContextVar
from typing import Optional

from tarzan.runtime.session import RunContext, RunMode


_ctx: ContextVar[Optional[RunContext]] = ContextVar("tarzan_run_context", default=None)


def _build_context(
    *,
    deterministic: bool,
    as_of: Optional[datetime.date],
    attempt_id: str,
    invocation_source: str,
) -> RunContext:
    if deterministic and as_of is None:
        raise ValueError("reproducible mode requires an as_of effective date")
    mode_value = (
        RunMode.REPRODUCIBLE
        if deterministic
        else RunMode.POINT_IN_TIME
        if as_of is not None
        else RunMode.LIVE
    )
    return RunContext(
        attempt_id=attempt_id,
        mode=mode_value,
        effective_date=as_of,
        captured_at=datetime.datetime.now(datetime.timezone.utc),
        invocation_source=invocation_source,
        schema_versions={
            "input": "2",
            "summary": "2",
            "ledger": "1.0",
            "manifest": "1.0",
            "cache": "1",
            "exposure": "1.0",
            "capability": "1.0",
            "provider_policy": "1.0",
            "telemetry": "1.0",
            "delivery_identity": "1.0",
            "delivery_state": "1.0",
        },
        policy_versions={"release": "1.0"},
    )


def reset(
    *,
    attempt_id: str = "compatibility",
    invocation_source: str = "compatibility",
) -> RunContext:
    """Install a fresh live context for the current execution context."""
    value = _build_context(
        deterministic=False,
        as_of=None,
        attempt_id=attempt_id,
        invocation_source=invocation_source,
    )
    _ctx.set(value)
    return value


def configure(
    deterministic: bool = False,
    as_of: Optional[datetime.date] = None,
    *,
    attempt_id: str = "compatibility",
    invocation_source: str = "compatibility",
) -> RunContext:
    """Resolve one coherent mode before any provider or financial work."""
    value = _build_context(
        deterministic=deterministic,
        as_of=as_of,
        attempt_id=attempt_id,
        invocation_source=invocation_source,
    )
    _ctx.set(value)
    return value


def context() -> RunContext:
    value = _ctx.get()
    return value if value is not None else reset()


def mode() -> RunMode:
    return context().mode


def is_deterministic() -> bool:
    """Return the established reproducible-mode compatibility flag."""
    return mode() is RunMode.REPRODUCIBLE


def allows_live_transport() -> bool:
    return context().allows_live_transport


def as_of() -> Optional[datetime.date]:
    return context().effective_date


def today() -> datetime.date:
    """Return the one captured analysis date for this run."""
    return context().analysis_date


def now_stamp(fmt: str = "%d %b %Y, %H:%M") -> str:
    """Render a coherent artifact stamp from the run-owned clock."""
    value = context()
    stamp = (
        datetime.datetime.combine(value.effective_date, datetime.time.min)
        if value.effective_date is not None
        else value.captured_at.astimezone()
    )
    return stamp.strftime(fmt)
