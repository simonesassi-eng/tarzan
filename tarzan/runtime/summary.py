"""Immutable, versioned machine summary shared by CLI and email entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from tarzan.contracts.immutable import freeze_json as _freeze, thaw_json as _thaw
from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.ledger import Availability, FailureRecord
from tarzan.runtime.publication import PublicationEvaluator, PublicationOutcome
from tarzan.runtime.session import RunResult


RUN_SUMMARY_SCHEMA_VERSION = "3"


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
        facts = (
            result.report_view.project_summary_facts()
            if result.report_view is not None else None
        )
        # A legacy caller's direct metrics contract is intentionally retained,
        # including its existing conversion/error boundaries. Prepared reports
        # use immutable narrow facts; neither path caches mutable ledger state.
        metrics = result.metrics if result.report_view is None else None
        metrics_present = facts is not None or metrics is not None

        def metric_fact(name, default=None):
            return getattr(facts if facts is not None else metrics, name, default)

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

        if not metrics_present or any(
            record in critical for record in portfolio_failures
        ):
            portfolio_availability = Availability.UNAVAILABLE
            machine_metrics: Mapping[str, Any] = {}
        else:
            if facts is None:
                machine_metrics = metrics.to_summary_dict()
            elif facts.machine_metrics is not None:
                machine_metrics = facts.machine_metrics
            else:
                # Retain the legacy error boundary: a critical unavailable
                # portfolio does not serialize its machine values at all.
                # Unprepared legacy views or malformed serializers fall back
                # to the original authority only when values are requested.
                projected_metrics = result.report_view.project()[0]
                machine_metrics = projected_metrics.to_summary_dict()
            portfolio_availability = (
                Availability.UNAVAILABLE
                if metric_fact("valuation_availability", "") == "UNAVAILABLE"
                else Availability.DEGRADED
                if failures or metric_fact("degraded_computers")
                else Availability.AVAILABLE
            )

        planning_unavailable = (
            not metrics_present
            or metric_fact("planning_availability", "") == "UNAVAILABLE"
            or any(record in critical for record in planning_failures)
        )
        planning_availability = (
            Availability.UNAVAILABLE
            if planning_unavailable
            else Availability.DEGRADED
            if planning_failures
            else Availability.AVAILABLE
        )
        if not metrics_present:
            planning_value = None
        elif facts is not None:
            planning_value = None if planning_unavailable else facts.planning_value
        else:
            planning_value = {
                "actions": list(getattr(metrics, "rebalancing_suggestions", None) or []),
                "verifications": list(
                    getattr(metrics, "rebalancing_verifications", None) or []
                ),
            }

        portfolio_section = cls._section(
            portfolio_availability,
            machine_metrics,
            portfolio_failures,
        )
        portfolio_section["trustworthy_total_eur"] = (
            metric_fact("trustworthy_total_value_eur") if metrics_present else None
        )
        portfolio_section["known_subtotal_eur"] = (
            metric_fact("known_valuation_subtotal_eur") if metrics_present else None
        )
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
