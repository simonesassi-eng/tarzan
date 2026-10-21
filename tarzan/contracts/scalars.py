"""Closed scalar domains at configuration and trading-protection boundaries."""

from __future__ import annotations

import math


def parse_boolean(value, *, blank: bool = False) -> bool:
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return blank
    if text in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if text in {"false", "0", "no", "n", "f", "off"}:
        return False
    raise ValueError("expected an explicit true/false value")


def finite_nonnegative(value, *, maximum: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise ValueError("expected a finite nonnegative value within the field domain")
    return number
