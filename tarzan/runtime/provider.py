"""Structured provider/cache outcomes and validated quality policies."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

from tarzan.runtime.ledger import Availability


PROVIDER_POLICY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ProviderAttempt:
    source: str
    operation: str
    ordinal: int
    outcome: str
    fallback_rung: int
    observation_time: Optional[str] = None
    fetch_time: Optional[str] = None
    age_seconds: Optional[float] = None
    latency_ms: Optional[float] = None
    coverage_pct: Optional[float] = None
    failure_ref: Optional[str] = None


@dataclass(frozen=True)
class ProviderQualityPolicy:
    policy_id: str
    freshness_seconds: float
    minimum_coverage_pct: float
    timeout_seconds: float
    retry_budget: int
    allow_fallback: bool
    valuation_materiality_pct: float
    publication_materiality_pct: float

    def __post_init__(self) -> None:
        finite = {
            "freshness_seconds": self.freshness_seconds,
            "minimum_coverage_pct": self.minimum_coverage_pct,
            "timeout_seconds": self.timeout_seconds,
            "valuation_materiality_pct": self.valuation_materiality_pct,
            "publication_materiality_pct": self.publication_materiality_pct,
        }
        for name, value in finite.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.minimum_coverage_pct <= 100:
            raise ValueError("minimum_coverage_pct must be in [0, 100]")
        if not 0 <= self.valuation_materiality_pct <= 100:
            raise ValueError("valuation_materiality_pct must be in [0, 100]")
        if not 0 <= self.publication_materiality_pct <= 100:
            raise ValueError("publication_materiality_pct must be in [0, 100]")
        if self.retry_budget < 0:
            raise ValueError("retry_budget must be nonnegative")


K = TypeVar("K")
V = TypeVar("V")


class ProviderResult(Mapping[K, V], Generic[K, V]):
    """Mapping-compatible selected value plus complete provider evidence."""

    def __init__(
        self,
        value: Mapping[K, V],
        *,
        availability: Availability,
        attempts: tuple[ProviderAttempt, ...],
        policy: Mapping[str, object],
        selected_source: Optional[str] = None,
    ) -> None:
        self.value = dict(value)
        self.availability = availability
        self.attempts = attempts
        self.policy = dict(policy)
        self.selected_source = selected_source

    def __getitem__(self, key: K) -> V:
        return self.value[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    @property
    def degraded(self) -> bool:
        return self.availability is Availability.DEGRADED


class ValuationEvidenceState(str, Enum):
    PRIMARY = "PRIMARY"
    FALLBACK = "FALLBACK"
    STALE = "STALE"
    MISSING = "MISSING"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ValuationEvidence:
    instrument_key: str
    instrument_kind: str
    data_class: str
    state: ValuationEvidenceState
    value_eur: Optional[float]
    materiality_basis_eur: Optional[float]
    source: Optional[str]
    policy_id: str
    accepted_by_policy: bool
    age_seconds: Optional[float] = None


@dataclass(frozen=True)
class ValuationCompletenessAssessment:
    availability: Availability
    trustworthy_total_eur: Optional[float]
    known_subtotal_eur: float
    missing_materiality_pct: Optional[float]
    planning_eligible: bool
    evidence: tuple[ValuationEvidence, ...]
    failure_refs: tuple[str, ...] = ()


class ValuationCompletenessEvaluator:
    """Apply explicit kind/data-class policy before totals or planning are trusted."""

    DATA_CLASS = "current_valuation"

    def __init__(self, policies: Mapping[str, ProviderQualityPolicy]) -> None:
        self._policies = dict(policies)

    @staticmethod
    def _finite(value: object) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _instrument_key(holding: object) -> str:
        from tarzan.models.instrument_key import instrument_key

        return instrument_key(
            getattr(holding, "isin", None),
            getattr(holding, "ticker", None),
        ) or "UNKNOWN"

    @staticmethod
    def _kind(holding: object) -> str:
        from tarzan.instruments.registry import TypeEvidenceGateway

        resolution = TypeEvidenceGateway().resolve(
            getattr(holding, "security_type", None),
            getattr(holding, "instrument_type", None),
            *tuple(getattr(holding, "instrument_kind_evidence", ()) or ()),
        )
        return resolution.kind.value if resolution.kind is not None else "UNKNOWN"

    def evaluate(self, holdings: list[object], ledger) -> ValuationCompletenessAssessment:
        from datetime import datetime, timezone

        from tarzan.runtime.ledger import LedgerEntryType
        from tarzan.runtime.session import current_session

        session = current_session()
        captured_at = (
            session.context.captured_at
            if session is not None
            else datetime.now(timezone.utc)
        )
        rows: list[dict[str, object]] = []
        total_basis = 0.0
        known_subtotal = 0.0
        degraded_failure_refs: list[str] = []

        for holding in holdings:
            if getattr(holding, "is_seeded_target", False):
                continue
            key = self._instrument_key(holding)
            kind = self._kind(holding)
            policy_key = f"{kind}:{self.DATA_CLASS}"
            policy = self._policies.get(policy_key)
            if policy is None:
                policy = self._policies.get(f"UNKNOWN:{self.DATA_CLASS}")
            if policy is None:
                raise ValueError(f"missing required valuation policy for {policy_key}")

            current_value = self._finite(getattr(holding, "current_value", None))
            anchor_value = self._finite(getattr(holding, "market_value_eur", None))
            cost_basis = self._finite(getattr(holding, "cost_basis_eur", None))
            quantity_value = self._finite(getattr(holding, "quantity", None))
            quantity = abs(quantity_value) if quantity_value is not None else None
            current_price = self._finite(getattr(holding, "current_price", None))
            has_primary_price = current_price is not None

            # Zero storage anchors on a nonzero position do not prove zero
            # materiality. Order-derived holdings use 0.0 when exact mechanics
            # cannot produce a seed; if cost/current evidence is also absent,
            # the basis is indeterminate and must take the critical path.
            basis_candidates = (
                anchor_value,
                cost_basis,
                current_value if has_primary_price else None,
            )
            basis = next(
                (abs(value) for value in basis_candidates
                 if value is not None and value != 0.0),
                None,
            )
            if basis is None and quantity == 0.0:
                basis = 0.0
            if basis is not None:
                total_basis += basis

            age_seconds = None
            fetched = getattr(holding, "fetch_timestamp", None)
            if fetched is not None:
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age_seconds = max(0.0, (captured_at - fetched).total_seconds())
            stale = age_seconds is not None and age_seconds > policy.freshness_seconds

            if kind == "UNKNOWN":
                state = ValuationEvidenceState.UNSUPPORTED
                selected = None
                accepted = False
            elif has_primary_price and current_value is not None and not stale:
                state = ValuationEvidenceState.PRIMARY
                selected = current_value
                accepted = True
            elif stale and current_value is not None:
                state = ValuationEvidenceState.STALE
                selected = current_value if policy.allow_fallback else None
                accepted = policy.allow_fallback
            elif anchor_value is not None and (anchor_value != 0.0 or quantity == 0.0):
                state = ValuationEvidenceState.FALLBACK
                selected = anchor_value if policy.allow_fallback else None
                accepted = policy.allow_fallback
            else:
                state = ValuationEvidenceState.MISSING
                selected = None
                accepted = False

            evidence = ValuationEvidence(
                instrument_key=key,
                instrument_kind=kind,
                data_class=self.DATA_CLASS,
                state=state,
                value_eur=selected,
                materiality_basis_eur=basis,
                source=getattr(holding, "data_source", None),
                policy_id=policy.policy_id,
                accepted_by_policy=accepted,
                age_seconds=age_seconds,
            )
            rows.append({"evidence": evidence, "policy": policy})
            if selected is not None and accepted:
                known_subtotal += selected

            ledger.append(LedgerEntryType.CAPABILITY, {
                "instrument": key,
                "kind": kind,
                "capability": "PRICING_VALUATION",
                "state": state.value,
                "accepted_by_policy": accepted,
                "policy_id": policy.policy_id,
                "age_seconds": age_seconds,
            })
            if state in (ValuationEvidenceState.FALLBACK, ValuationEvidenceState.STALE) and accepted:
                failure_id = ledger.open_failure(
                    stage="valuation",
                    stable_code=f"{state.value}_VALUATION_SELECTED",
                    severity="WARNING",
                    error={"instrument": key, "state": state.value},
                    affected_outputs=["portfolio", "valuation"],
                    analytical_impact="valuation uses explicitly labeled non-primary evidence",
                    publication_impact="DEGRADE",
                    context={"policy_id": policy.policy_id},
                )
                ledger.remedy(
                    failure_id,
                    remedy_id=f"select-{state.value.casefold()}",
                    action="select policy-permitted non-primary valuation",
                    outcome="SUCCEEDED",
                    availability=Availability.DEGRADED,
                    provenance=[str(getattr(holding, "data_source", None) or state.value)],
                )
                ledger.close_failure(
                    failure_id,
                    automatically_corrected=True,
                    selected_resolution=f"select-{state.value.casefold()}",
                    availability=Availability.DEGRADED,
                    provenance=[str(getattr(holding, "data_source", None) or state.value)],
                )
                degraded_failure_refs.append(failure_id)

        missing_rows = [row for row in rows if not row["evidence"].accepted_by_policy]
        missing_failure_refs: list[str] = []
        for row in missing_rows:
            evidence = row["evidence"]
            policy = row["policy"]
            missing_failure_refs.append(ledger.open_failure(
                stage="valuation",
                stable_code=f"{evidence.state.value}_VALUATION_UNAVAILABLE",
                severity="WARNING",
                error={
                    "instrument": evidence.instrument_key,
                    "state": evidence.state.value,
                },
                affected_outputs=["portfolio", "valuation"],
                analytical_impact=(
                    "known subtotal excludes valuation evidence that did not "
                    "satisfy the applied policy"
                ),
                publication_impact="DEGRADE",
                context={
                    "policy_id": policy.policy_id,
                    "materiality_basis_eur": evidence.materiality_basis_eur,
                },
            ))
        known_missing_basis = [
            float(row["evidence"].materiality_basis_eur)
            for row in missing_rows
            if row["evidence"].materiality_basis_eur is not None
        ]
        missing_basis_indeterminate = any(
            row["evidence"].materiality_basis_eur is None
            for row in missing_rows
        )
        missing_basis = sum(known_missing_basis)
        missing_pct = (
            None
            if missing_basis_indeterminate
            else missing_basis / total_basis * 100.0
            if total_basis > 0
            else None
        )

        # Materiality is a portfolio-relative policy decision, but repeated
        # gaps governed by the same kind/policy must be assessed together.
        # Evaluating each row independently lets two 6% gaps evade a 10%
        # threshold even though that policy is missing 12% of the portfolio.
        # A missing materiality basis is never zero: it makes the group
        # indeterminate and therefore critical because safety cannot be
        # established against the configured threshold.
        missing_by_policy: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in missing_rows:
            evidence = row["evidence"]
            policy = row["policy"]
            missing_by_policy.setdefault(
                (evidence.instrument_kind, policy.policy_id), []
            ).append(row)

        material_groups: list[dict[str, object]] = []
        material_rows: list[dict[str, object]] = []
        for (kind, policy_id), group_rows in missing_by_policy.items():
            policy = group_rows[0]["policy"]
            group_basis_values = [
                row["evidence"].materiality_basis_eur
                for row in group_rows
            ]
            group_basis_indeterminate = any(
                value is None for value in group_basis_values
            )
            group_basis = sum(
                float(value) for value in group_basis_values if value is not None
            )
            group_pct = (
                None
                if group_basis_indeterminate
                else group_basis / total_basis * 100.0
                if total_basis > 0
                else None
            )
            if group_pct is None or group_pct > policy.valuation_materiality_pct:
                material_rows.extend(group_rows)
                material_group = {
                    "instrument_kind": kind,
                    "policy_id": policy_id,
                    "missing_count": len(group_rows),
                    "missing_basis_eur": group_basis,
                    "missing_materiality_pct": group_pct,
                    "threshold_pct": policy.valuation_materiality_pct,
                }
                if group_basis_indeterminate:
                    material_group["materiality_indeterminate"] = True
                material_groups.append(material_group)

        critical_ref = None
        if material_rows:
            critical_ref = ledger.open_failure(
                stage="valuation_completeness",
                stable_code="MATERIAL_VALUATION_GAP",
                severity="CRITICAL",
                error={
                    "missing_count": len(material_rows),
                    "states": [row["evidence"].state.value for row in material_rows],
                },
                affected_outputs=["portfolio", "total", "planning", "rebalancing", "publication"],
                analytical_impact=(
                    "trustworthy total is unavailable; known subtotal remains partial evidence; "
                    "optimization and rebalancing are suppressed"
                ),
                publication_impact="BLOCK_NORMAL_AND_NOTIFY_FAILURE",
                context={
                    "missing_materiality_pct": missing_pct,
                    "material_groups": material_groups,
                    "applied_policy_ids": list(dict.fromkeys(
                        row["policy"].policy_id for row in material_rows
                    )),
                },
            )

        all_evidence = tuple(row["evidence"] for row in rows)
        failure_refs = tuple(
            degraded_failure_refs
            + missing_failure_refs
            + ([critical_ref] if critical_ref else [])
        )
        if critical_ref:
            availability = Availability.UNAVAILABLE
        elif any(item.state is not ValuationEvidenceState.PRIMARY for item in all_evidence):
            availability = Availability.DEGRADED
        else:
            availability = Availability.AVAILABLE
        return ValuationCompletenessAssessment(
            availability=availability,
            trustworthy_total_eur=None if critical_ref else known_subtotal,
            known_subtotal_eur=known_subtotal,
            missing_materiality_pct=missing_pct,
            planning_eligible=critical_ref is None,
            evidence=all_evidence,
            failure_refs=failure_refs,
        )
