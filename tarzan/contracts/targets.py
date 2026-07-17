"""Row-preserving executable contract for per-instrument targets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


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


class TargetSetOutcome(Mapping[str, dict]):
    """Mapping-compatible target outcome with complete row/error evidence."""

    def __init__(
        self,
        *,
        status: TargetSetStatus,
        rows: tuple[ValidatedTargetRow, ...] = (),
        errors: tuple[TargetError, ...] = (),
        mapping: Mapping[str, dict] | None = None,
    ) -> None:
        self.status = status
        self.rows = rows
        self.errors = errors
        self.planning_eligible = status in (TargetSetStatus.ABSENT, TargetSetStatus.VALID)
        # Invalid sets deliberately expose no planning mapping.
        self._mapping = dict(mapping or {}) if self.planning_eligible else {}

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
