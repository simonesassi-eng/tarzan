"""No-guess two-axis registry for instrument mechanics and category semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Mapping, Optional, TypeVar

from tarzan.runtime.ledger import Availability


CAPABILITY_SCHEMA_VERSION = "1.0"


class InstrumentKind(str, Enum):
    STOCK = "STOCK"
    ETF = "ETF"
    BOND = "BOND"
    CASH = "CASH"


class InstrumentCapability(str, Enum):
    IDENTITY = "IDENTITY"
    PRICING_VALUATION = "PRICING_VALUATION"
    HISTORY_RETURNS = "HISTORY_RETURNS"
    INCOME = "INCOME"
    EXPOSURE_CLASSIFICATION = "EXPOSURE_CLASSIFICATION"
    SECTOR = "SECTOR"
    REBALANCING = "REBALANCING"


class SupportState(str, Enum):
    SUPPORTED = "SUPPORTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"


class TypeResolutionState(str, Enum):
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"


T = TypeVar("T")


@dataclass(frozen=True)
class CapabilityResult(Generic[T]):
    support: SupportState
    availability: Availability
    value: Optional[T]
    semantics_version: str = CAPABILITY_SCHEMA_VERSION
    provenance: tuple[str, ...] = ()
    failure_refs: tuple[str, ...] = ()
    analytical_impact: str = ""
    publication_impact: str = "NONE"

    def __post_init__(self) -> None:
        if self.support is not SupportState.SUPPORTED and self.value is not None:
            raise ValueError("unsupported/not-applicable capability cannot carry a value")


@dataclass(frozen=True)
class TypeResolution:
    state: TypeResolutionState
    kind: Optional[InstrumentKind]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class InstrumentAdapter:
    kind: InstrumentKind
    capabilities: Mapping[InstrumentCapability, SupportState]
    semantics_version: str = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        missing = set(InstrumentCapability) - set(self.capabilities)
        if missing:
            raise ValueError(f"{self.kind.value} missing capability declarations: {sorted(m.value for m in missing)}")
        object.__setattr__(self, "capabilities", MappingProxyType(dict(self.capabilities)))


@dataclass(frozen=True)
class TrackedCategoryProfile:
    name: str
    denominator: str
    sector_support: SupportState
    rebalancing_support: SupportState
    semantics_version: str = CAPABILITY_SCHEMA_VERSION


@dataclass(frozen=True)
class ResolvedInstrumentProfile:
    resolution: TypeResolution
    adapter: Optional[InstrumentAdapter]
    categories: tuple[TrackedCategoryProfile, ...]

    def capability(self, capability: InstrumentCapability) -> CapabilityResult[None]:
        if self.adapter is None:
            return CapabilityResult(
                support=SupportState.UNSUPPORTED,
                availability=Availability.UNAVAILABLE,
                value=None,
                provenance=self.resolution.evidence,
                analytical_impact=f"{capability.value} unavailable for unresolved instrument kind",
                publication_impact="DEGRADE",
            )
        support = self.adapter.capabilities[capability]
        availability = (
            Availability.AVAILABLE if support is SupportState.SUPPORTED
            else Availability.UNAVAILABLE
        )
        return CapabilityResult(
            support=support,
            availability=availability,
            value=None,
            provenance=self.resolution.evidence,
            analytical_impact="" if support is SupportState.SUPPORTED else f"{capability.value} unavailable",
        )


class TypeEvidenceGateway:
    """Resolve only explicit/version-mapped evidence; never inspect names/prices."""

    _MAPPING = {
        "STOCK": InstrumentKind.STOCK,
        "EQUITY": InstrumentKind.STOCK,
        "ETF": InstrumentKind.ETF,
        "BOND": InstrumentKind.BOND,
        "FIXEDINCOME": InstrumentKind.BOND,
        "CASH": InstrumentKind.CASH,
        "MONEYMARKET": InstrumentKind.CASH,
    }

    def resolve(self, *assertions: Optional[str]) -> TypeResolution:
        evidence = tuple(str(item).strip().upper() for item in assertions if str(item or "").strip())
        resolved = {self._MAPPING[item.replace("_", "").replace(" ", "")] for item in evidence
                    if item.replace("_", "").replace(" ", "") in self._MAPPING}
        if len(resolved) == 1:
            return TypeResolution(TypeResolutionState.RESOLVED, next(iter(resolved)), evidence)
        if len(resolved) > 1:
            return TypeResolution(TypeResolutionState.AMBIGUOUS, None, evidence)
        return TypeResolution(TypeResolutionState.UNKNOWN, None, evidence)


class InstrumentAdapterRegistry:
    def __init__(self, adapters: tuple[InstrumentAdapter, ...]) -> None:
        self._adapters = {adapter.kind: adapter for adapter in adapters}
        if len(self._adapters) != len(adapters):
            raise ValueError("duplicate instrument-kind registration")

    def resolve(self, resolution: TypeResolution) -> ResolvedInstrumentProfile:
        adapter = self._adapters.get(resolution.kind) if resolution.state is TypeResolutionState.RESOLVED else None
        return ResolvedInstrumentProfile(resolution, adapter, ())


class TrackedCategoryRegistry:
    def __init__(self, profiles: tuple[TrackedCategoryProfile, ...]) -> None:
        self._profiles = {profile.name: profile for profile in profiles}
        if len(self._profiles) != len(profiles):
            raise ValueError("duplicate tracked-category registration")

    def get(self, name: str) -> Optional[TrackedCategoryProfile]:
        return self._profiles.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._profiles)


def _all_supported(*, sector: SupportState = SupportState.SUPPORTED) -> Mapping[InstrumentCapability, SupportState]:
    return {
        InstrumentCapability.IDENTITY: SupportState.SUPPORTED,
        InstrumentCapability.PRICING_VALUATION: SupportState.SUPPORTED,
        InstrumentCapability.HISTORY_RETURNS: SupportState.SUPPORTED,
        InstrumentCapability.INCOME: SupportState.SUPPORTED,
        InstrumentCapability.EXPOSURE_CLASSIFICATION: SupportState.SUPPORTED,
        InstrumentCapability.SECTOR: sector,
        InstrumentCapability.REBALANCING: SupportState.SUPPORTED,
    }


def default_instrument_registry() -> InstrumentAdapterRegistry:
    return InstrumentAdapterRegistry((
        InstrumentAdapter(InstrumentKind.STOCK, _all_supported()),
        InstrumentAdapter(InstrumentKind.ETF, _all_supported()),
        InstrumentAdapter(InstrumentKind.BOND, _all_supported(sector=SupportState.NOT_APPLICABLE)),
        InstrumentAdapter(InstrumentKind.CASH, _all_supported(sector=SupportState.NOT_APPLICABLE)),
    ))


def default_tracked_category_registry() -> TrackedCategoryRegistry:
    from tarzan.models.holding import AssetClass

    profiles = []
    for category in AssetClass:
        sector = (
            SupportState.SUPPORTED if category is AssetClass.EQUITIES
            else SupportState.NOT_APPLICABLE
        )
        profiles.append(TrackedCategoryProfile(
            name=category.value,
            denominator="invested_capital" if category is not AssetClass.CASH_EQUIVALENTS else "total_capital",
            sector_support=sector,
            rebalancing_support=SupportState.SUPPORTED,
        ))
    return TrackedCategoryRegistry(tuple(profiles))
