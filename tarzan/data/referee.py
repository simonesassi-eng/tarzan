"""Pure validation for observations allowed to referee another market source.

A downloaded pair is not evidence about a return until its two dated observations
match the interval under examination. No provider calls or timestamp relaxation live
here.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd


CURRENT_SESSION_BASELINE_ATTR = "current_session_baseline"


def finite_number(value, *, positive: bool = False) -> Optional[float]:
    """A finite scalar number, never a JSON boolean masquerading as a price."""
    if (
        not pd.api.types.is_scalar(value) or pd.api.types.is_bool(value)
        or isinstance(value, (complex, np.complexfloating))
    ):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def observation_date(value) -> Optional[dt.date]:
    """An explicit calendar date; numeric epochs and missing timestamps are unknown.

    Preserve the observation's own local calendar date rather than shifting a daily
    close across midnight by converting it to UTC.
    """
    if not isinstance(value, (str, dt.date, dt.datetime, pd.Timestamp)):
        return None
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].+)?", value):
        return None  # "today"/"now" denote receipt time, not source observations
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if pd.isna(timestamp) else timestamp.date()


def quote_pair(quote, *, session, anchor) -> Optional[tuple[float, float]]:
    """A finite, positive EUR pair for exactly ``anchor -> session``, or None."""
    if not isinstance(quote, Mapping):
        return None
    session_date, anchor_date = observation_date(session), observation_date(anchor)
    if session_date is None or anchor_date is None or anchor_date >= session_date:
        return None
    if (
        observation_date(quote.get("date")) != session_date
        or observation_date(quote.get("prev_date")) != anchor_date
        or str(quote.get("currency") or "EUR").strip().upper() != "EUR"
    ):
        return None
    price = finite_number(quote.get("price"), positive=True)
    previous = finite_number(quote.get("prev_close"), positive=True)
    return None if price is None or previous is None else (price, previous)
