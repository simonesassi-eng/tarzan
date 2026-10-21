"""Row-preserving executable contract for per-instrument targets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from types import MappingProxyType

from tarzan.contracts.immutable import freeze_json


class TargetSetStatus(str, Enum):
    ABSENT = "ABSENT"
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class TargetError:
    code: str
    message: str
    canonical_key: str
    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class ValidatedTargetRow:
    source_name: str
    source_row: int
    canonical_key: str
    value: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "value", freeze_json(self.value))

@dataclass(frozen=True, init=False, eq=False)
class TargetSetOutcome(Mapping[str, dict]):
    """Mapping-compatible target outcome with complete row/error evidence."""

    status: TargetSetStatus
    rows: tuple[ValidatedTargetRow, ...]
    errors: tuple[TargetError, ...]
    planning_eligible: bool
    _mapping: Mapping[str, Mapping[str, Any]]

    def __init__(
        self,
        *,
        status: TargetSetStatus,
        rows: tuple[ValidatedTargetRow, ...] = (),
        errors: tuple[TargetError, ...] = (),
        mapping: Mapping[str, dict] | None = None,
    ) -> None:
        status = TargetSetStatus(status)
        eligible = status in (TargetSetStatus.ABSENT, TargetSetStatus.VALID)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "errors", tuple(errors))
        object.__setattr__(self, "planning_eligible", eligible)
        # Invalid sets deliberately expose no planning mapping.
        # Preserve source-row alias identity, without collapsing equal but
        # distinct rows or allowing a source dict to mutate the plan later.
        frozen_rows = {}
        frozen_mapping = {}
        if eligible:
            for key, value in (mapping or {}).items():
                if id(value) not in frozen_rows:
                    frozen_rows[id(value)] = freeze_json(value)
                frozen_mapping[key] = frozen_rows[id(value)]
        object.__setattr__(self, "_mapping", MappingProxyType(frozen_mapping))

    def __getitem__(self, key: str) -> dict:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._mapping) == dict(other)
        return NotImplemented

    @classmethod
    def absent(cls) -> "TargetSetOutcome":
        return cls(status=TargetSetStatus.ABSENT)


TargetPlan = TargetSetOutcome
