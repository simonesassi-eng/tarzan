"""Regenerate Tarzan's vendored exchange-holiday table.

The calendar is VENDORED, not computed at run time: a pinned ``--as_of`` run must
be byte-reproducible, and a calendar library's holiday data changes between
releases, so the calendar belongs to the repository rather than to whatever
happens to be installed. It also costs no runtime dependency — 18 venues over
nine years is under 4 KB gzipped.

``exchange_calendars`` is therefore a GENERATION-time dependency only and is
deliberately absent from requirements.in. Run this in a throwaway environment:

    python3 -m venv /tmp/cal && /tmp/cal/bin/pip install exchange_calendars pandas
    /tmp/cal/bin/python scripts/refresh_exchange_holidays.py

Then commit the regenerated ``tarzan/data/exchange_holidays.csv.gz`` together
with the test run, and note the library version in the module docstring of
``tarzan/data/exchange_calendar.py``.

When to re-run: the source calendars bound themselves about a year into the
future, so the table's horizon moves with them. Refresh when the horizon gets
close (``exchange_calendar.covers`` starts returning False for dates a run
actually asks about) or when an exchange announces a schedule change. Beyond the
horizon every lookup degrades to the Mon-Fri rule, which is correct-but-coarse
rather than wrong, so a late refresh loses precision and never correctness.

The MIC list must stay in step with ``exchange_calendar._SUFFIX_MIC`` and
``_GROUP_MIC``; ``test_exchange_calendar.TestEveryReachableVenueHasACalendar``
fails if a venue Tarzan can resolve is missing from the table.
"""

from __future__ import annotations

import pandas as pd

# Every exchange Tarzan can resolve a listing to. Keep in step with
# exchange_calendar._SUFFIX_MIC / _GROUP_MIC.
MICS = [
    "XMIL",   # Borsa Italiana (.MI, .ETLX)
    "XETR",   # Xetra + the German regional venues (.DE .F .MU .SG .BE .DU .HM .HA .TG)
    "XPAR", "XAMS", "XBRU", "XLIS", "XMAD",   # Euronext / Madrid
    "XWBO", "XSWX", "XHEL", "XDUB",           # Vienna, Zurich, Helsinki, Dublin
    "XLON",                                   # London (.L)
    "XNYS",                                   # US cash (indices, bare tickers)
    "XTKS", "XHKG", "XSHG", "XASX", "XKRX",   # Tokyo, Hong Kong, Shanghai, Sydney, Seoul
]
START, END = "2019-01-01", "2031-12-31"
OUT = "tarzan/data/exchange_holidays.csv.gz"


def _naive(value) -> pd.Timestamp:
    """A tz-naive, midnight-normalised Timestamp (calendars differ on tz)."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def main() -> None:
    import exchange_calendars as xc

    rows: list[tuple[str, str]] = []
    report: list[tuple[str, object, object, int]] = []
    for mic in MICS:
        calendar = xc.get_calendar(mic)
        low = max(_naive(START), _naive(calendar.first_session))
        high = min(_naive(END), _naive(calendar.last_session))
        sessions = {_naive(s) for s in calendar.sessions_in_range(low, high)}
        # Only WEEKDAYS are recorded: weekends are never sessions anywhere, so
        # listing them would quadruple the table for no information.
        closed = [d for d in pd.bdate_range(low, high) if _naive(d) not in sessions]
        rows.extend((mic, d.date().isoformat()) for d in closed)
        report.append((mic, low.date(), high.date(), len(closed)))

    frame = pd.DataFrame(rows, columns=["mic", "date"]).sort_values(["mic", "date"])
    frame.to_csv(OUT, index=False, compression="gzip")

    print(f"exchange_calendars {xc.__version__}  requested horizon {START}..{END}")
    print(f"{len(frame)} closed weekdays across {len(MICS)} venues -> {OUT}")
    for mic, low, high, count in report:
        print(f"   {mic:<6} {low} .. {high}  {count:>4} closed weekdays")
    print("\nUpdate the library version noted in "
          "tarzan/data/exchange_calendar.py, then run the suite.")


if __name__ == "__main__":
    main()
