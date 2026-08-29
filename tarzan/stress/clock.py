"""Pinning the run instant — hour of day included.

FINDING F0: the hour of day cannot be controlled from outside the process.
``python -m tarzan.main`` has seven flags and none names an instant; no TARZAN_*
variable reads a clock; no time-faking library is installed; and
``runtime._build_context`` hardcodes ``captured_at=datetime.now(utc)`` as a
literal, so no supported API can move it. The bench is therefore an IN-PROCESS
driver, and that is a property of the code under test, not a choice.

FINDING F0-bis: ``--as_of`` cannot supply an hour either, and it disables the
code the bench exists to stress. Under POINT_IN_TIME/REPRODUCIBLE the market
seam ignores ``captured_at`` and returns the END of the effective date
(23:59:59.999999Z = 19:59 ET, 01:59 Rome), so every cash market reads CLOSED
forever; and ``allows_live_transport()`` goes False, after which the intraday and
official-quote fetchers return {} and stamping is refused outright. "EU pre-open",
"EU open / US closed" and "both open" are unreachable through ``--as_of`` by
construction.

So a pinned-hour scenario runs in LIVE mode with a pinned date, which RunContext
permits (its __post_init__ validates only the converse), and the network is closed
off separately by ``net.py`` rather than by the run mode.

Two seams, both verified end to end through ``main.main()``:

  S1  market_quotes._intraday_reference_now — the ONE reference instant behind
      every session predicate (market_open_now, market_status, futures_open_now,
      fx_open_now, session_day, session_span, _has_intraday, _clip_to_reference,
      intraday_feeds, fetch_market_quotes: 11 call sites). Documented in its own
      docstring as a monkeypatchable seam.
  S2' runtime._build_context — wrapped, not replaced, so ``configure()`` stores a
      PINNED context in the ContextVar and hands the SAME object to RunSession.
      This matters: patching ``runtime.context`` instead leaves the session
      holding a real-clock context, and provider.py:306 prefers
      ``session.context`` — the freshness classifier would silently keep reading
      the wall clock. Patching ``runtime.session.current_session`` to None closes
      that leak but breaks the run outright ("_run_once requires an active
      RunSession"), which is how this seam was chosen.

Plus delivery.now_local (subject-line hour, no seam of its own) and a quote-memo
reset, because that memo keys on time.monotonic() which no pin can move.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

ROME = ZoneInfo("Europe/Rome")
UTC = dt.timezone.utc

#: The pinned instants. Names are the scenario ids used by the matrix.
#: Verified predicate matrix (EU / US / futures / FX):
#:   C1 F/F/T/T   C2 T/F/T/T   C3 T/T/T/T   C4 F/T/T/T   C5 F/F/T/T   C6 F/F/F/F
INSTANTS = {
    # Wed 26 Aug 2026 is an ordinary session on every venue in the book.
    "C1": dt.datetime(2026, 8, 26, 7, 30, tzinfo=ROME),   # EU pre-open
    "C2": dt.datetime(2026, 8, 26, 10, 0, tzinfo=ROME),   # EU open, US closed
    "C3": dt.datetime(2026, 8, 26, 16, 0, tzinfo=ROME),   # both open
    "C4": dt.datetime(2026, 8, 26, 18, 0, tzinfo=ROME),   # EU closed, US open
    "C5": dt.datetime(2026, 8, 26, 23, 30, tzinfo=ROME),  # both closed, post-US
    "C6": dt.datetime(2026, 8, 29, 12, 0, tzinfo=ROME),   # Saturday
    # FINDING F1: a weekday exchange holiday is NOT expressible. market_open_now
    # never consults exchange_calendar (market_quotes.py:148 says so), so this
    # instant reads EU OPEN on Christmas Day. Kept in the matrix precisely to
    # record that, as check E10's pre-registered expected failure.
    # A PAST holiday, deliberately. Pinning a FUTURE effective date while in LIVE
    # mode makes the enricher compute a refresh window from the cache's last date
    # to a date months ahead and allocate without bound: the process is
    # SIGKILLed (exit 137, no traceback, no output). That state is not reachable
    # in production — --as_of forces POINT_IN_TIME, and PIT with a future date
    # runs clean — so it is a limit of this bench, not a defect of the product,
    # and it is recorded as such rather than reported as a finding.
    # 15 Aug 2025 is the Assumption: Borsa Italiana closed, NYSE open.
    "C7": dt.datetime(2025, 8, 15, 16, 0, tzinfo=ROME),   # EU holiday, US open
}


@dataclasses.dataclass(frozen=True)
class Pin:
    instant: dt.datetime
    effective_date: dt.date

    @property
    def label(self) -> str:
        return self.instant.strftime("%a %Y-%m-%d %H:%M %Z")


def pin_for(instant_id: str) -> Pin:
    inst = INSTANTS[instant_id]
    return Pin(instant=inst, effective_date=inst.date())


def install(pin: Pin) -> None:
    """Install S1, S2' and the delivery clock. Idempotent per process."""
    from tarzan import delivery, runtime
    from tarzan.data import market_quotes as mq
    from tarzan.runtime.session import RunMode

    original = getattr(runtime, "_stress_original_build_context", None)
    if original is None:
        original = runtime._build_context
        runtime._stress_original_build_context = original

    def _pinned_build_context(**kw):
        ctx = original(**kw)
        # mode=LIVE with effective_date set is legal and is the only combination
        # that gives a pinned DATE and an honoured pinned HOUR at once.
        return dataclasses.replace(
            ctx, mode=RunMode.LIVE, effective_date=pin.effective_date,
            captured_at=pin.instant.astimezone(UTC),
        )

    runtime._build_context = _pinned_build_context

    if not hasattr(mq, "_stress_original__intraday_reference_now"):
        mq._stress_original__intraday_reference_now = mq._intraday_reference_now
    if not hasattr(delivery, "_stress_original_now_local"):
        delivery._stress_original_now_local = delivery.now_local

    stamp = pd.Timestamp(pin.instant)
    mq._intraday_reference_now = (
        lambda tzinfo=None: stamp if tzinfo is None else stamp.tz_convert(tzinfo)
    )
    delivery.now_local = lambda: pin.instant

    # The quote memo keys on time.monotonic(), which no pin can move, so two
    # scenarios back to back in one process would share one answer.
    if hasattr(mq, "reset_quote_memo"):
        mq.reset_quote_memo()


def uninstall() -> None:
    """Remove the pin, restoring the real clock.

    Load-bearing, not housekeeping. The pin replaces ``runtime._build_context``
    with a wrapper that forces mode=LIVE and a fixed effective_date, so leaving it
    installed makes every LATER run ignore its own ``--as_of``: a POINT_IN_TIME
    cell that followed a LIVE cell silently ran at the previous cell's date and in
    the previous cell's mode. That is how the bench came to report a fully
    liquidated book (P09, all sells done by Feb 2026, asked at 29 Aug 2026) as
    still holding 31/67/30 units — the run had actually been pinned to 15 Aug
    2025, before any of the sells. Called before every non-LIVE run.
    """
    from tarzan import delivery, runtime
    from tarzan.data import market_quotes as mq

    original = getattr(runtime, "_stress_original_build_context", None)
    if original is not None:
        runtime._build_context = original
    for module, name in ((mq, "_intraday_reference_now"), (delivery, "now_local")):
        saved = getattr(module, "_stress_original_" + name, None)
        if saved is not None:
            setattr(module, name, saved)
    if hasattr(mq, "reset_quote_memo"):
        mq.reset_quote_memo()
