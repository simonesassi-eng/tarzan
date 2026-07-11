"""Per-run runtime context — the single clock and determinism switch.

Tarzan's output is normally perturbed by three live inputs: the wall clock
(``datetime.now()``), live intraday quotes (the broker-style 1D and the
Markets strip), and the Gemini AI narrative. That makes a run impossible to
reproduce offline and un-golden-testable end-to-end.

This module centralizes the clock and a ``deterministic`` switch so a run can
be pinned:

  * ``as_of`` pins "today" — the terminal valuation date for XIRR/TWROR and
    the daily series — so the same inputs + as_of always produce the same
    numbers.
  * ``deterministic`` additionally tells the live/AI surfaces to stand down
    (skip intraday quotes and the Gemini call), so nothing network-live or
    non-reproducible leaks into the output.

Design mirrors ``data_quality`` / ``audit``: a process-global context, reset
at the top of ``orchestrator.run``, best-effort accessors. **Default is OFF**
— ``today()`` returns ``datetime.now().date()`` and ``is_deterministic()`` is
False — so an ordinary run behaves exactly as before this module existed.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunContext:
    deterministic: bool = False
    # Pinned "today". When None, the live wall-clock date is used.
    as_of: Optional[datetime.date] = None
    # Pinned wall-clock stamp string for report headers in deterministic mode
    # (so "Generated ..." lines don't vary run-to-run). None → live now().
    stamp: Optional[str] = None


# Process-global, reset per run.
_ctx = RunContext()


def reset() -> None:
    """Clear any pinning — a fresh run starts live unless configured otherwise."""
    global _ctx
    _ctx = RunContext()


def configure(deterministic: bool = False, as_of: Optional[datetime.date] = None,
              stamp: Optional[str] = None) -> None:
    """Set the run context. ``as_of`` implies a pinned clock even if
    ``deterministic`` is False (an as-of valuation without silencing live
    quotes is a legitimate mode); ``deterministic`` additionally stands the
    live/AI surfaces down."""
    global _ctx
    _ctx = RunContext(deterministic=deterministic, as_of=as_of, stamp=stamp)


def is_deterministic() -> bool:
    return _ctx.deterministic


def as_of() -> Optional[datetime.date]:
    return _ctx.as_of


def today() -> datetime.date:
    """The run's "today": the pinned ``as_of`` when set, else the live date.

    This is the single source callers should use instead of
    ``datetime.now().date()`` for anything that anchors a valuation/return, so
    a pinned run is reproducible.
    """
    return _ctx.as_of if _ctx.as_of is not None else datetime.datetime.now().date()


def now_stamp(fmt: str = "%d %b %Y, %H:%M") -> str:
    """A wall-clock stamp for report headers. Pinned string in deterministic
    mode (falls back to the as_of date at midnight if no explicit stamp was
    given), else the live formatted now()."""
    if _ctx.stamp is not None:
        return _ctx.stamp
    if _ctx.deterministic and _ctx.as_of is not None:
        return datetime.datetime(
            _ctx.as_of.year, _ctx.as_of.month, _ctx.as_of.day
        ).strftime(fmt)
    return datetime.datetime.now().strftime(fmt)
