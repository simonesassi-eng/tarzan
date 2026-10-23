"""Complete canonical material facts; never pandas' truncated display repr."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import math

import numpy as np
import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics


MATERIAL_FACTS_VERSION = "3"
_OPERATIONAL_FIELDS = {
    "attempt_id", "recorded_at", "fetch_time", "fetch_timestamp",
    "latency_ms", "age_seconds", "freshness_age_seconds",
}


def _operational_field(name) -> bool:
    """Only declared operational field names, not dates or financial values."""
    name = getattr(name, "value", name)
    return isinstance(name, str) and name in _OPERATIONAL_FIELDS


def material_value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        positions = [
            position for position, name in enumerate(value.columns)
            if not _operational_field(name)
        ]
        # Column selection must not coerce mixed integer/float columns through
        # DataFrame.values: that can round exact integers above 2**53 before
        # hashing. Positional tuples also retain duplicate financial labels.
        rows = (
            [[material_value(row[position]) for position in positions]
             for row in value.itertuples(index=False, name=None)]
            if positions else [[] for _ in range(len(value.index))]
        )
        return {
            "kind": "table",
            "columns": material_value([value.columns[position] for position in positions]),
            "index": material_value(value.index.tolist()),
            "rows": rows,
        }
    if isinstance(value, pd.Series):
        return {
            "kind": "series", "name": material_value(value.name),
            "index": material_value(value.index.tolist()),
            "values": material_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return material_value(value.item())
    if is_dataclass(value):
        return {
            item.name: material_value(getattr(value, item.name))
            for item in fields(value) if not _operational_field(item.name)
        }
    if isinstance(value, Mapping):
        return {str(getattr(key, "value", key)): material_value(item)
                for key, item in value.items() if not _operational_field(key)}
    if isinstance(value, (list, tuple)):
        return [material_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None  # nonfinite values are published as Unavailable, never a number
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported material fact: {type(value).__name__}")


def material_metrics(metrics):
    return {
        item.name: material_value(getattr(metrics, item.name, None))
        for item in fields(PortfolioMetrics)
    }
