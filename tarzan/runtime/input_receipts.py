"""Immutable raw-byte fingerprints, never input contents or source authority."""

from collections.abc import Mapping
import re
from types import MappingProxyType


INPUT_FILENAMES = frozenset({
    "order_list.csv", "targets.csv", "targets_per_holding.csv",
})


def immutable_fingerprints(values):
    """Copy a closed, bounded receipt before attaching it to a native attempt."""
    if not isinstance(values, Mapping):
        raise ValueError("invalid input fingerprint names")
    copied = dict(values)
    if any(type(name) is not str or name not in INPUT_FILENAMES for name in copied):
        raise ValueError("invalid input fingerprint names")
    if any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in copied.values()):
        raise ValueError("invalid input fingerprint digest")
    return MappingProxyType(copied)
