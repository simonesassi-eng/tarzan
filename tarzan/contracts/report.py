"""Immutable financial report facts, with defensive pandas projections.

No provider lookup or new financial calculation belongs here. A report view
snapshots the computed cube and its native machine contract, then reconstructs
private presentation copies. Source or renderer mutation cannot change a
completed analysis's financial authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from tarzan.contracts.immutable import freeze_json


@dataclass(frozen=True)
class _MappingFact:
    items: tuple


@dataclass(frozen=True)
class _SequenceFact:
    items: tuple
    kind: str


@dataclass(frozen=True)
class _CategoricalDtypeFact:
    categories: Any
    ordered: bool


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
        return pd.Index(
            self.values, dtype=_project_dtype(self.dtype),
            name=self.names[0], tupleize_cols=False,
        )


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


def _freeze_dtype(dtype):
    if isinstance(dtype, pd.CategoricalDtype):
        return _CategoricalDtypeFact(_freeze(dtype.categories), dtype.ordered)
    return dtype


def _project_dtype(dtype):
    if isinstance(dtype, _CategoricalDtypeFact):
        return pd.CategoricalDtype(
            categories=_project(dtype.categories), ordered=dtype.ordered,
        )
    return dtype


def _freeze(value):
    if isinstance(value, pd.DataFrame):
        return _FrameFact(
            _freeze(value.index), _freeze(value.columns),
            tuple(
                tuple(_freeze(item) for item in row)
                for row in value.itertuples(index=False, name=None)
            ),
            tuple(_freeze_dtype(dtype) for dtype in value.dtypes), _freeze(value.attrs),
        )
    if isinstance(value, pd.Series):
        return _SeriesFact(
            _freeze(value.index), tuple(_freeze(item) for item in value.tolist()),
            _freeze(value.name), _freeze_dtype(value.dtype), _freeze(value.attrs),
        )
    if isinstance(value, pd.Index):
        kind = "multi" if isinstance(value, pd.MultiIndex) else (
            "datetime" if isinstance(value, pd.DatetimeIndex) else "index"
        )
        return _IndexFact(
            tuple(value.tolist()), tuple(value.names), kind, _freeze_dtype(value.dtype),
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
            frame.isetitem(ordinal, frame.iloc[:, ordinal].astype(_project_dtype(dtype)))
        frame.attrs.update(_project(value.attrs))
        return frame
    if isinstance(value, _SeriesFact):
        series = pd.Series(
            [_project(item) for item in value.values], index=value.index.project(),
            dtype=_project_dtype(value.dtype), name=_project(value.name),
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
class SummaryReportFacts:
    """Narrow immutable financial facts, never a ledger/publication snapshot."""

    machine_metrics: Mapping[str, Any] | None
    _planning_value: Any
    valuation_availability: str
    planning_availability: str
    degraded_computers: bool
    trustworthy_total_value_eur: Any
    known_valuation_subtotal_eur: Any

    def __post_init__(self):
        for name in (
            "machine_metrics", "trustworthy_total_value_eur",
            "known_valuation_subtotal_eur",
        ):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))
        object.__setattr__(self, "_planning_value", _freeze(self._planning_value))

    @property
    def planning_value(self):
        """Serialize raw frozen plans only when the planning section is used."""
        return freeze_json(_project(self._planning_value))

    @classmethod
    def from_metrics(
        cls, metrics, *, serialize_machine=True,
        defer_machine_failure=False,
    ):
        """Snapshot the native machine contract and raw planning facts."""
        if metrics is None:
            return None
        machine = None
        if serialize_machine:
            try:
                # Prior prepared reports normalized numpy/typed record values
                # before the native serializer. Preserve that JSON boundary
                # without reconstructing the full source cube.
                machine = _project(_freeze(metrics.to_summary_dict()))
            except Exception:
                if not defer_machine_failure:
                    raise

        def snapshot(value):
            return _project(_freeze(value))

        return cls(
            machine_metrics=machine,
            _planning_value={
                "actions": list(getattr(metrics, "rebalancing_suggestions", None) or []),
                "verifications": list(
                    getattr(metrics, "rebalancing_verifications", None) or []
                ),
            },
            valuation_availability=getattr(metrics, "valuation_availability", ""),
            planning_availability=getattr(metrics, "planning_availability", ""),
            degraded_computers=bool(getattr(metrics, "degraded_computers", None)),
            trustworthy_total_value_eur=snapshot(
                getattr(metrics, "trustworthy_total_value_eur", None),
            ),
            known_valuation_subtotal_eur=snapshot(
                getattr(metrics, "known_valuation_subtotal_eur", None),
            ),
        )


def _validated_selection(names, available):
    """Validate the complete selection before reconstructing any mutable fact."""
    if isinstance(names, (str, bytes)):
        raise ValueError("report selection requires an iterable of field names")
    try:
        selected = tuple(names)
    except TypeError:
        raise ValueError("report selection requires an iterable of field names") from None
    if any(not isinstance(name, str) for name in selected):
        raise ValueError("report selection contains an invalid field name")
    if len(set(selected)) != len(selected):
        raise ValueError("report selection contains duplicate field names")
    if any(name not in available for name in selected):
        raise ValueError("report selection contains an unknown field name")
    return selected


@dataclass(frozen=True)
class ReportView:
    """Stable output boundary; projections contain no shared mutable facts."""

    _metrics: Any
    _config: Any
    benchmark_alpha_beta: str | None
    benchmark_geo: str | None
    _summary_facts: SummaryReportFacts | None = None

    @classmethod
    def build(cls, metrics, config, *, benchmark_alpha_beta=None, benchmark_geo=None):
        frozen_metrics, frozen_config = _freeze(metrics), _freeze(config)
        try:
            summary = SummaryReportFacts.from_metrics(
                metrics, defer_machine_failure=True,
            )
        except Exception:
            # Generic dataclass reports and malformed machine contracts could
            # previously build a view. Keep that boundary: only a later
            # machine-value request propagates the serializer's error.
            summary = None
        return cls(
            frozen_metrics, frozen_config, benchmark_alpha_beta, benchmark_geo, summary,
        )

    def project(self):
        return _project(self._metrics), _project(self._config)

    @property
    def is_native_portfolio_metrics(self) -> bool:
        """Immutable authority metadata, with no copy or property evaluation.

        A subclass/custom record may need other constructor inputs or override
        a getter. Its legacy full projection must retain that exact contract.
        """
        from tarzan.models.portfolio import PortfolioMetrics

        return (
            isinstance(self._metrics, _RecordFact)
            and self._metrics.record_type is PortfolioMetrics
        )

    def project_summary_facts(self):
        """Return only immutable financial facts; publication remains dynamic."""
        if self._summary_facts is not None:
            return self._summary_facts
        # Compatibility for the original four-argument constructor and for
        # serializer failures retained at their original request boundary.
        return SummaryReportFacts.from_metrics(
            _project(self._metrics), defer_machine_failure=True,
        )

    def project_metrics_fields(self, names):
        """Defensively copy exactly requested dataclass-init metric fields."""
        available = (
            dict(self._metrics.values)
            if isinstance(self._metrics, _RecordFact) else {}
        )
        selected = _validated_selection(names, available)
        return {name: _project(available[name]) for name in selected}

    def project_metric_properties(self, names):
        """Read raw original getters from only their frozen dependencies.

        These properties need holdings and the total, not histories, config or
        the entire cube. Evaluate lazily: generic typed frames may legitimately
        have no cost basis, and requesting a property must preserve that error.
        """
        selected = _validated_selection(
            names, ("unrealized_pnl_eur", "unrealized_pnl_pct"),
        )
        if not selected:
            return {}
        dependencies = self.project_metrics_fields(("holdings_df", "total_value"))
        metrics = self._metrics.record_type(**dependencies)
        return {name: getattr(metrics, name) for name in selected}
