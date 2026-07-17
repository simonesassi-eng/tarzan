"""Explicit publication decisions derived from typed run evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from tarzan.runtime.ledger import FailureRecord


class PublicationDecision(str, Enum):
    SEND_NORMAL = "SEND_NORMAL"
    SEND_DEGRADED_NORMAL = "SEND_DEGRADED_NORMAL"
    BLOCK_NORMAL_AND_NOTIFY_FAILURE = "BLOCK_NORMAL_AND_NOTIFY_FAILURE"


class DeliveryPurpose(str, Enum):
    NORMAL_NEWSLETTER = "NORMAL_NEWSLETTER"
    CRITICAL_FAILURE_NOTIFICATION = "CRITICAL_FAILURE_NOTIFICATION"


@dataclass(frozen=True)
class PublicationOutcome:
    decision: PublicationDecision
    delivery_purpose: DeliveryPurpose
    critical_failure_ids: tuple[str, ...] = ()


class PublicationEvaluator:
    @staticmethod
    def evaluate(failures: Iterable[FailureRecord]) -> PublicationOutcome:
        records = tuple(failures)
        critical = tuple(
            record.failure_id for record in records
            if record.severity.upper() == "CRITICAL" and not record.automatically_corrected
        )
        if critical:
            return PublicationOutcome(
                PublicationDecision.BLOCK_NORMAL_AND_NOTIFY_FAILURE,
                DeliveryPurpose.CRITICAL_FAILURE_NOTIFICATION,
                critical,
            )
        if records:
            return PublicationOutcome(
                PublicationDecision.SEND_DEGRADED_NORMAL,
                DeliveryPurpose.NORMAL_NEWSLETTER,
            )
        return PublicationOutcome(
            PublicationDecision.SEND_NORMAL,
            DeliveryPurpose.NORMAL_NEWSLETTER,
        )
