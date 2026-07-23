"""Run-owned data-quality compatibility projection.

The authoritative evidence is appended to the active :class:`RunLedger` and
the same issue object is mirrored into :class:`RunSession.diagnostics`.
A context-local report preserves established helper APIs for renderers,
pre-session errors, and versioned consumers; it is not an independent
production artifact authority. Initialized CLI/email runs publish diagnostics
only through ``LocalArtifactWriter``.

Recording is best-effort so diagnostics cannot break financial execution.
"""

from __future__ import annotations

import logging
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Severities, ordered most→least serious for summary display.
ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


@dataclass
class Issue:
    """One data-quality event worth surfacing to the user."""

    severity: str
    source: str          # pipeline stage, e.g. "order_load", "enricher"
    message: str         # human-readable, actionable ("what + why + action")
    context: Optional[str] = None  # optional key (ISIN / ticker / row idx)


@dataclass
class _Report:
    issues: list[Issue] = field(default_factory=list)


# Context-local compatibility projection; the active RunSession also owns each
# issue and its ledger evidence.
_report: ContextVar[Optional[_Report]] = ContextVar("tarzan_data_quality", default=None)


def _current_report() -> _Report:
    report = _report.get()
    if report is None:
        report = _Report()
        _report.set(report)
    return report


def reset() -> None:
    """Start a fresh report in the current run context."""
    _report.set(_Report())


def record(
    severity: str,
    source: str,
    message: str,
    context: Optional[str] = None,
    *,
    accepted_resolution: Optional[str] = None,
    acceptance_provenance=(),
) -> None:
    """Record one issue. Best-effort — never raises into the caller.

    ``accepted_resolution`` is reserved for an explicit degraded-evidence
    policy. Generic warnings remain closed for lifecycle completeness but carry
    no selected resolution, so renderers continue to classify them as needing
    review instead of inferring acceptance from closure alone.
    """
    try:
        issue = Issue(severity=severity, source=source, message=message, context=context)
        _current_report().issues.append(issue)
        from tarzan.runtime.ledger import Availability, LedgerEntryType
        from tarzan.runtime.session import current_session
        session = current_session()
        if session is not None:
            session.diagnostics.append(issue)
            availability = (
                Availability.DEGRADED
                if severity == WARNING
                else Availability.UNAVAILABLE
                if severity == ERROR
                else Availability.AVAILABLE
            )
            session.ledger.append(LedgerEntryType.STAGE, {
                "stage": source,
                "severity": severity,
                "message": message,
                "context": context,
                "availability": availability.value,
            })
            if severity in (ERROR, WARNING):
                explicitly_accepted = bool(
                    severity == WARNING and accepted_resolution
                )
                failure_id = session.ledger.open_failure(
                    stage=source,
                    stable_code=f"DATA_QUALITY_{severity}",
                    severity=severity,
                    error={"message": message, "context": context},
                    affected_outputs=[source],
                    analytical_impact=(
                        "affected result is unavailable"
                        if severity == ERROR
                        else "affected result is degraded under an explicit fallback policy"
                        if explicitly_accepted
                        else "affected result is degraded and requires review"
                    ),
                    publication_impact=(
                        "DEGRADE" if severity == WARNING else "DEGRADE_OR_BLOCK_BY_POLICY"
                    ),
                )
                if severity == WARNING:
                    provenance = tuple(str(item) for item in acceptance_provenance)
                    if explicitly_accepted:
                        session.ledger.remedy(
                            failure_id,
                            remedy_id=str(accepted_resolution),
                            action="select explicit degraded-evidence policy",
                            outcome="ACCEPTED",
                            availability=Availability.DEGRADED,
                            provenance=provenance,
                        )
                    session.ledger.close_failure(
                        failure_id,
                        automatically_corrected=False,
                        selected_resolution=(
                            str(accepted_resolution)
                            if explicitly_accepted
                            else None
                        ),
                        availability=Availability.DEGRADED,
                        provenance=provenance,
                    )
    except Exception:  # noqa: BLE001 — a diagnostic must never break the pipeline
        pass


def warning(source: str, message: str, context: Optional[str] = None) -> None:
    record(WARNING, source, message, context)


def accepted_warning(
    source: str,
    message: str,
    context: Optional[str] = None,
    *,
    resolution: str,
    provenance=(),
) -> None:
    """Record a warning whose degraded fallback was explicitly accepted."""
    record(
        WARNING,
        source,
        message,
        context,
        accepted_resolution=resolution,
        acceptance_provenance=provenance,
    )


def error(source: str, message: str, context: Optional[str] = None) -> None:
    record(ERROR, source, message, context)


def info(source: str, message: str, context: Optional[str] = None) -> None:
    record(INFO, source, message, context)


def issues() -> list[Issue]:
    """The issues recorded so far this run (most callers just want counts)."""
    return list(_current_report().issues)


def counts() -> dict[str, int]:
    """Issue counts per severity, e.g. ``{"ERROR": 0, "WARNING": 3}``."""
    return Counter(i.severity for i in _current_report().issues)


def summary_line() -> str:
    """A one-line summary for the console / other logs."""
    c = counts()
    if not c:
        return "Data quality: no issues."
    parts = [f"{c[s]} {s.lower()}(s)" for s in (ERROR, WARNING, INFO) if c.get(s)]
    return "Data quality: " + ", ".join(parts) + "."
