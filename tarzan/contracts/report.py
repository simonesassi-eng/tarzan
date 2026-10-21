"""Immutable financial report facts, with defensive pandas projections.

No provider lookup or arithmetic belongs here. A report view snapshots the
already computed cube and reconstructs private presentation copies. Source or
renderer mutation cannot change a completed analysis's financial authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _MappingFact:
    items: tuple


@dataclass(frozen=True)
class _SequenceFact:
    items: tuple
    kind: str


@dataclass(frozen=True)
class _IndexFact:
    values: tuple
    names: tuple
    kind: str
    dtype: Any = None
    timezone: Any = None
    frequency: Any = None

    def project(self):
        if self.kind == "multi":
            return pd.MultiIndex.from_tuples(self.values, names=self.names)
        if self.kind == "datetime":
            return pd.DatetimeIndex(
                self.values, tz=self.timezone, name=self.names[0], freq=self.frequency,
            )
        return pd.Index(self.values, dtype=self.dtype, name=self.names[0], tupleize_cols=False)


@dataclass(frozen=True)
class _FrameFact:
    index: _IndexFact
    columns: _IndexFact
    rows: tuple
    dtypes: tuple
    attrs: Any


@dataclass(frozen=True)
class _SeriesFact:
    index: _IndexFact
    values: tuple
    name: Any
    dtype: Any
    attrs: Any


@dataclass(frozen=True)
class _RecordFact:
    record_type: type
    values: tuple


def _freeze(value):
    if isinstance(value, pd.DataFrame):
        return _FrameFact(
            _freeze(value.index), _freeze(value.columns),
            tuple(tuple(_freeze(item) for item in row) for row in value.values.tolist()),
            tuple(value.dtypes), _freeze(value.attrs),
        )
    if isinstance(value, pd.Series):
        return _SeriesFact(
            _freeze(value.index), tuple(_freeze(item) for item in value.tolist()),
            _freeze(value.name), value.dtype, _freeze(value.attrs),
        )
    if isinstance(value, pd.Index):
        kind = "multi" if isinstance(value, pd.MultiIndex) else (
            "datetime" if isinstance(value, pd.DatetimeIndex) else "index"
        )
        return _IndexFact(
            tuple(value.tolist()), tuple(value.names), kind, value.dtype,
            getattr(value, "tz", None), getattr(value, "freq", None),
        )
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _RecordFact(type(value), tuple(
            (item.name, _freeze(getattr(value, item.name))) for item in fields(value) if item.init
        ))
    if isinstance(value, Mapping):
        return _MappingFact(tuple((_freeze(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list, set, frozenset)):
        return _SequenceFact(tuple(_freeze(item) for item in value), type(value).__name__)
    if value is None or value is pd.NA or value is pd.NaT or isinstance(
        value, (str, int, float, bool, date, datetime, timedelta, Enum),
    ):
        return value
    raise TypeError(f"unsupported report fact: {type(value).__name__}")


def _project(value):
    if isinstance(value, _FrameFact):
        frame = pd.DataFrame(
            [[_project(item) for item in row] for row in value.rows],
            index=value.index.project(), columns=value.columns.project(),
        )
        for ordinal, dtype in enumerate(value.dtypes):
            frame.isetitem(ordinal, frame.iloc[:, ordinal].astype(dtype))
        frame.attrs.update(_project(value.attrs))
        return frame
    if isinstance(value, _SeriesFact):
        series = pd.Series(
            [_project(item) for item in value.values], index=value.index.project(),
            dtype=value.dtype, name=_project(value.name),
        )
        series.attrs.update(_project(value.attrs))
        return series
    if isinstance(value, _MappingFact):
        return {_project(key): _project(item) for key, item in value.items}
    if isinstance(value, _SequenceFact):
        constructors = {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset}
        return constructors[value.kind](_project(item) for item in value.items)
    if isinstance(value, _RecordFact):
        return value.record_type(**{name: _project(item) for name, item in value.values})
    if isinstance(value, _IndexFact):
        return value.project()
    return value


@dataclass(frozen=True)
class ReportView:
    """Stable output boundary; projections contain no shared mutable facts."""

    _metrics: Any
    _config: Any
    benchmark_alpha_beta: str | None
    benchmark_geo: str | None

    @classmethod
    def build(cls, metrics, config, *, benchmark_alpha_beta=None, benchmark_geo=None):
        return cls(_freeze(metrics), _freeze(config), benchmark_alpha_beta, benchmark_geo)

    def project(self):
        return _project(self._metrics), _project(self._config)
