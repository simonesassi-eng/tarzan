"""The covering array, as data.

Reduction criterion, declared: 2-way (all-pairs). The full cross product is
10 portfolios x 7 instants x 3 modes x 2 input modes = 420 runs. Every defect in
the project's own history was triggered by ONE factor or ONE pair of factors
(session x mode, currency x venue, window x weekend) — never by a three-factor
interaction — so the array covers every PAIR and samples triples.

One constraint shapes it, and it is a property of the code, not a choice
(see clock.py, F0-bis): instants C1-C7 are only meaningful in LIVE mode, because
--as_of forces the market reference to end-of-day and disables live transport. In
PIT/REPRO the array varies the effective DATE instead, and the cell records that.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Optional

PORTFOLIOS = [f"P{i:02d}" for i in range(1, 11)]
INSTANTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
MODES = ["LIVE", "PIT", "REPRO"]

#: Dates used when the mode cannot honour an hour: a weekday, a Saturday, and a
#: weekday exchange holiday.
PIT_DATES = {"weekday": dt.date(2026, 8, 26),
             "weekend": dt.date(2026, 8, 29),
             "holiday": dt.date(2026, 12, 25)}


@dataclasses.dataclass(frozen=True)
class Cell:
    cid: str
    portfolio: str
    instant: Optional[str]
    mode: str
    strict: bool
    as_of: Optional[dt.date] = None
    note: str = ""


def block_a() -> list:
    """Coverage: every (portfolio, mode), (portfolio, strict), (instant, mode) and
    (mode, strict) pair appears at least once."""
    cells, n = [], 0
    # LIVE x all seven instants, cycling the portfolios and the input mode.
    for i, inst in enumerate(INSTANTS):
        for j in range(2):
            p = PORTFOLIOS[(i * 2 + j) % len(PORTFOLIOS)]
            n += 1
            cells.append(Cell(f"A{n:02d}", p, inst, "LIVE", strict=bool((i + j) % 2),
                              note="pinned hour, live mode"))
    # PIT and REPRO over the three date kinds, cycling portfolios and strictness.
    for mode in ("PIT", "REPRO"):
        for k, (kind, day) in enumerate(PIT_DATES.items()):
            for j in range(2):
                p = PORTFOLIOS[(n + j) % len(PORTFOLIOS)]
                n += 1
                cells.append(Cell(f"A{n:02d}", p, None, mode,
                                  strict=bool((k + j) % 2), as_of=day,
                                  note=f"hour forced to end-of-day; date kind={kind}"))
    return cells


def block_b() -> list:
    """Determinism twins: four books, REPRO, each run twice."""
    return [Cell(f"B{i:02d}", p, None, "REPRO", strict=False,
                 as_of=PIT_DATES["weekday"], note="twin for D1")
            for i, p in enumerate(("P01", "P04", "P07", "P10"), start=1)]


def block_c() -> list:
    """Differential: permutation, truncation, time-of-day invariance."""
    out = []
    for i, p in enumerate(("P02", "P03", "P04", "P06", "P10"), start=1):
        out.append(Cell(f"C{i:02d}", p, None, "REPRO", strict=False,
                        as_of=PIT_DATES["weekday"], note="D3 permutation base"))
    for i, p in enumerate(("P02", "P04", "P05", "P06", "P10"), start=1):
        out.append(Cell(f"C1{i}", p, None, "PIT", strict=False,
                        as_of=dt.date(2025, 6, 30), note="D4 lookahead base"))
    # Both instants must fall on the SAME effective date, or the comparison is
    # not "a different hour" but "a different day": the first version paired
    # C5 (Wed 23:30) with C6 (Sat 12:00) and reported 5/5 failures that were
    # simply three days of market movement.
    for i, p in enumerate(("P01", "P03", "P04", "P07", "P10"), start=1):
        out.append(Cell(f"C2{i}", p, "C1", "LIVE", strict=False,
                        note="D2 instant A: Wed 07:30, pre-open (closed)"))
    return out


def block_d() -> list:
    """Degradation: provider dark, empty cache, zero value, claims."""
    return [
        Cell("D01", "P03", "C3", "LIVE", False, note="no quotes at all"),
        Cell("D02", "P02", "C3", "LIVE", False, note="empty cache: no tape anywhere"),
        Cell("D03", "P09", "C5", "LIVE", False, note="entirely liquidated book"),
        Cell("D04", "P10", "C3", "LIVE", True, note="pathological book, strict"),
        Cell("D05", "P08", "C1", "LIVE", False, note="one order, pre-open"),
        Cell("D06", "P04", "C6", "LIVE", False, note="bond-only book on a Saturday"),
        # P10 at C6 so its Saturday-dated order is IN scope: at C1-C5 the run's
        # effective date is 26 Aug and that order (29 Aug) is correctly excluded
        # as post-as_of, so the weekend-trade-date path was never reached.
        Cell("D07", "P10", "C6", "LIVE", False, note="weekend-dated order in scope"),
    ]


def block_e() -> list:
    """External verification: the one block allowed to touch the network."""
    return [Cell("E01", "P02", "C4", "LIVE", False, note="live tape comparison")]


def all_cells() -> list:
    return block_a() + block_b() + block_c() + block_d() + block_e()


def summarise() -> str:
    cells = all_cells()
    lines = [f"{len(cells)} cells: A={len(block_a())} B={len(block_b())} "
             f"C={len(block_c())} D={len(block_d())} E={len(block_e())}"]
    pairs = {("portfolio", "mode"): set(), ("instant", "mode"): set(),
             ("mode", "strict"): set(), ("portfolio", "strict"): set()}
    for c in cells:
        pairs[("portfolio", "mode")].add((c.portfolio, c.mode))
        pairs[("instant", "mode")].add((c.instant or "-", c.mode))
        pairs[("mode", "strict")].add((c.mode, c.strict))
        pairs[("portfolio", "strict")].add((c.portfolio, c.strict))
    for k, v in pairs.items():
        lines.append(f"  {k[0]:10s} x {k[1]:8s}: {len(v)} distinct pairs covered")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise())
    for c in all_cells():
        print(f"  {c.cid:4s} {c.portfolio} {c.instant or '-':3s} {c.mode:6s} "
              f"strict={str(c.strict):5s} {c.as_of or '':10} {c.note}")
