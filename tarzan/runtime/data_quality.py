"""Per-run data-quality report.

A single, skimmable place for everything the pipeline *skipped, coerced, or
fell back on* during one run — the events that would otherwise be scattered
across the verbose ``analyzer.log`` (loader row skips, input-validation
failures, FX/price fallbacks, stale quotes, degraded metric computers).

Design
------
* Process-global collector, reset at the start of every ``orchestrator.run``
  (mirrors the config / geo-resolver cache resets) and written out by the
  CLI after the run.
* **Best-effort**: recording an issue or writing the report must never raise
  into the pipeline — a diagnostic that breaks the run defeats its purpose.
* The report is **always** produced, even on a clean run ("No issues"), so a
  user can trust that an absent/empty section means "nothing happened", not
  "the reporter silently failed".

Format (``output/data_quality.log``)
------------------------------------
A summary header with counts by severity and category, then the issues
grouped by ``source`` (the pipeline stage), one issue per line::

    [WARNING][order_load] row 42: ISIN 'IE00XYZ' invalid format ... — skipped

Lines are prefixed ``[SEVERITY][source]`` so the file is greppable
(``grep ERROR``, ``grep order_load``) as well as readable top-to-bottom.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Severities, ordered most→least serious for summary display.
ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"
_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


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


def record(severity: str, source: str, message: str,
           context: Optional[str] = None) -> None:
    """Record one issue. Best-effort — never raises into the caller."""
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
                failure_id = session.ledger.open_failure(
                    stage=source,
                    stable_code=f"DATA_QUALITY_{severity}",
                    severity=severity,
                    error={"message": message, "context": context},
                    affected_outputs=[source],
                    analytical_impact=(
                        "affected result is unavailable"
                        if severity == ERROR
                        else "affected result is degraded and requires review"
                    ),
                    publication_impact=(
                        "DEGRADE" if severity == WARNING else "DEGRADE_OR_BLOCK_BY_POLICY"
                    ),
                )
                if severity == WARNING:
                    session.ledger.close_failure(
                        failure_id,
                        automatically_corrected=False,
                        selected_resolution=None,
                        availability=Availability.DEGRADED,
                    )
    except Exception:  # noqa: BLE001 — a diagnostic must never break the pipeline
        pass


def warning(source: str, message: str, context: Optional[str] = None) -> None:
    record(WARNING, source, message, context)


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


def render() -> str:
    """Render the full report as a string (also what gets written to disk)."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TARZAN DATA-QUALITY REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "Everything the run skipped, coerced, or fell back on. "
        "This is NOT the full log (see analyzer.log) — only items worth a "
        "human's review."
    )
    lines.append("")

    all_issues = _current_report().issues
    if not all_issues:
        lines.append("No issues this run — every input parsed and priced cleanly. ✅")
        lines.append("")
        return "\n".join(lines)

    # Summary counts.
    c = counts()
    summary = ", ".join(
        f"{c[s]} {s}" for s in (ERROR, WARNING, INFO) if c.get(s)
    )
    lines.append(f"SUMMARY: {summary}")
    # Per-category (source) counts for a quick "where" read.
    by_source = Counter(i.source for i in all_issues)
    lines.append("BY SECTION: " + ", ".join(
        f"{src}={n}" for src, n in sorted(by_source.items())
    ))
    lines.append("")
    lines.append("-" * 72)
    lines.append("")

    # Issues grouped by source, most-serious severity first within each group.
    for src in sorted({i.source for i in all_issues}):
        group = [i for i in all_issues if i.source == src]
        group.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))
        lines.append(f"## {src}  ({len(group)})")
        for i in group:
            ctx = f" [{i.context}]" if i.context else ""
            lines.append(f"  [{i.severity}][{i.source}]{ctx} {i.message}")
        lines.append("")

    return "\n".join(lines)


def write_report(output_dir: str, filename: str = "data_quality.log") -> Optional[str]:
    """Write the report to ``output_dir/filename`` and return its path.

    Best-effort: on any I/O error we log and return None rather than break
    the run (the console summary line still gives the headline counts).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render())
        return path
    except Exception as e:  # noqa: BLE001
        logger.debug("Data-quality report write failed: %s", e)
        return None
