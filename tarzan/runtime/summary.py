"""Immutable, versioned machine summary shared by CLI and email entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.ledger import Availability, FailureRecord
from tarzan.runtime.publication import PublicationEvaluator, PublicationOutcome
from tarzan.runtime.session import RunResult


RUN_SUMMARY_SCHEMA_VERSION = "2"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class VersionedRunSummary:
    """A deeply immutable summary whose bytes never contain Attempt ID."""

    analysis_id: str
    publication_state: str
    sections: Mapping[str, Any]
    metrics: Mapping[str, Any]
    schema_version: str = RUN_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.analysis_id:
            raise ValueError("analysis_id is required")
        object.__setattr__(self, "sections", _freeze(self.sections))
        object.__setattr__(self, "metrics", _freeze(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "publication_state": self.publication_state,
            "sections": _thaw(self.sections),
            # This is the explicit compatibility projection of the established
            # PortfolioMetrics machine contract, not a second reconstruction.
            "metrics": _thaw(self.metrics),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


class SummaryProjector:
    """Project one ledger-derived machine contract for every entry point."""

    @staticmethod
    def _section(
        availability: Availability,
        value: Any,
        failures: tuple[FailureRecord, ...],
    ) -> dict[str, Any]:
        return {
            "availability": availability.value,
            "value": value if availability is not Availability.UNAVAILABLE else None,
            "failure_refs": [record.failure_id for record in failures],
        }

    @staticmethod
    def _affects(record: FailureRecord, names: set[str]) -> bool:
        affected = {str(item).casefold() for item in record.affected_outputs}
        return bool(affected & names)

    @classmethod
    def project(
        cls,
        result: RunResult,
        publication: Optional[PublicationOutcome] = None,
    ) -> VersionedRunSummary:
        failures = result.ledger.failure_records()
        publication = publication or PublicationEvaluator.evaluate(failures)
        metrics = result.metrics

        critical = tuple(
            record for record in failures
            if record.severity.upper() == "CRITICAL" and not record.automatically_corrected
        )
        portfolio_failures = tuple(
            record for record in failures
            if cls._affects(record, {"portfolio", "valuation", "total", "total_value"})
        )
        planning_failures = tuple(
            record for record in failures
            if cls._affects(record, {"planning", "plan", "rebalancing", "optimizer"})
        )

        if metrics is None or any(record in critical for record in portfolio_failures):
            portfolio_availability = Availability.UNAVAILABLE
            machine_metrics: dict[str, Any] = {}
        else:
            machine_metrics = metrics.to_summary_dict()
            portfolio_availability = (
                Availability.DEGRADED
                if failures or getattr(metrics, "degraded_computers", None)
                else Availability.AVAILABLE
            )

        planning_unavailable = metrics is None or any(
            record in critical for record in planning_failures
        )
        planning_availability = (
            Availability.UNAVAILABLE
            if planning_unavailable
            else Availability.DEGRADED
            if planning_failures
            else Availability.AVAILABLE
        )
        planning_value = None if metrics is None else {
            "actions": list(getattr(metrics, "rebalancing_suggestions", None) or []),
            "verifications": list(getattr(metrics, "rebalancing_verifications", None) or []),
        }

        portfolio_section = cls._section(
            portfolio_availability,
            machine_metrics,
            portfolio_failures,
        )
        portfolio_section["trustworthy_total_eur"] = getattr(
            metrics, "trustworthy_total_value_eur", None
        ) if metrics is not None else None
        portfolio_section["known_subtotal_eur"] = getattr(
            metrics, "known_valuation_subtotal_eur", None
        ) if metrics is not None else None
        sections = {
            "portfolio": portfolio_section,
            "planning": cls._section(
                planning_availability,
                planning_value,
                planning_failures,
            ),
            "publication": {
                "availability": Availability.AVAILABLE.value,
                "value": {
                    "decision": publication.decision.value,
                    "delivery_purpose": publication.delivery_purpose.value,
                    "critical_failure_refs": list(publication.critical_failure_ids),
                },
                "failure_refs": list(publication.critical_failure_ids),
            },
        }
        return VersionedRunSummary(
            analysis_id=result.analysis_id,
            publication_state=publication.decision.value,
            sections=sections,
            metrics=machine_metrics,
        )
