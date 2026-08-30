"""Execute the matrix and append one record per run.

    python -m tarzan.stress.run                # everything
    python -m tarzan.stress.run --only A B     # selected blocks
    python -m tarzan.stress.run --list         # the matrix, no runs

Every run is isolated in a temporary tree; the ledger and the fixtures are the
reusable corpus. Nothing here fixes a defect: a failing check is recorded with
its repro and the run continues, and a check that could not be executed is
recorded as SKIP rather than omitted.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
LEDGER = HERE / "ledger.jsonl"
CACHE_SNAPSHOT = Path(os.environ.get("STRESS_CACHE_SNAPSHOT",
                                     Path.home() / ".cache" / "tarzan"))
BUDGET = {"runs": 120, "network_calls": 60, "wall_clock_s": 90 * 60}


def _first_line(err) -> str | None:
    if not err:
        return None
    lines = [l for l in str(err).splitlines() if l.strip()]
    return lines[0][:300] if lines else str(err)[:300]


def _log(fh, record: dict) -> None:
    fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    fh.flush()


def _truth(portfolio: str, effective: dt.date | None = None) -> dict:
    """The generator's ground truth, truncated to ``effective``.

    Orders after the effective date are correctly excluded by the run, so the
    oracle must exclude them too — otherwise the no-lookahead behaviour working
    reads as a lost position."""
    truth = json.loads((HERE / "fixtures" / portfolio / "seed.json").read_text())["truth"]
    if effective is None:
        return truth
    qty: dict = {}
    cash = 0.0
    src = HERE / "fixtures" / portfolio / "order_list.csv"
    for row in csv.DictReader(src.open()):
        if dt.date.fromisoformat(row["date"]) > effective:
            continue
        qty[row["isin"]] = qty.get(row["isin"], 0.0) + float(row["quantity"])
        cash = round(cash + float(row["net_eur"]), 2)
    return {**truth, "quantity_by_isin": qty, "cash_from_flows": cash}


def _permuted_orders(portfolio: str, dest: Path, seed: int) -> str:
    src = HERE / "fixtures" / portfolio / "order_list.csv"
    rows = list(csv.DictReader(src.open()))
    random.Random(seed).shuffle(rows)
    out = dest / f"{portfolio}_permuted.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return str(out)


def _truncated_orders(portfolio: str, dest: Path, cutoff: dt.date) -> str:
    src = HERE / "fixtures" / portfolio / "order_list.csv"
    rows = [r for r in csv.DictReader(src.open())
            if dt.date.fromisoformat(r["date"]) <= cutoff]
    out = dest / f"{portfolio}_to_{cutoff}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv.DictReader(src.open()).fieldnames))
        w.writeheader(); w.writerows(rows)
    return str(out)


def _quantities(cell, cache):
    """Quantities for the conservation check, from a direct orchestrator call.

    No artifact carries per-holding quantity, so this is the only sound source —
    see driver.quantities_in_process for why the earlier global-capture version
    was withdrawn after it falsely accused a correctly liquidated book.
    """
    from tarzan.stress import driver
    try:
        return driver.quantities_in_process(portfolio=cell.portfolio,
                                            as_of=_effective(cell), cache_dir=cache)
    except Exception:                                  # noqa: BLE001 — recorded
        # None, not {}: an empty dict is indistinguishable from a book that really
        # holds nothing, and C5 would then certify the emptiness it failed to read.
        return None


def _effective(cell) -> dt.date | None:
    from tarzan.stress import clock
    if cell.as_of:
        return cell.as_of
    return clock.pin_for(cell.instant).effective_date if cell.instant else None


def _per_run_checks(res, portfolio: str, effective: dt.date | None = None,
                    *, quantities: dict | None = None) -> list:
    from tarzan.stress import checks
    truth = _truth(portfolio, effective)
    out = []
    # Propagate the oracle's None: guarding the assignment left C5's own
    # no-oracle branch unreachable, so a malfunctioning oracle read as an empty
    # book instead of as "nothing was verified".
    res.quantity_by_isin = quantities
    # The gate FIRST: every other check in this list can only make a claim about
    # an artifact that exists.
    out += checks.c0_run_rendered(res)
    out += checks.c5_quantity_and_cash(res, truth)
    out += checks.c6_weights_and_contributions(res)
    out += checks.c7_fx_and_native(res, truth)
    out += checks.c8_cross_artifact(res)
    out += checks.c14_every_holding_is_visible(res)
    out += checks.c16_share_percentages_are_shares(res)
    out += checks.g11_degradation(res)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="blocks to run: A B C D E")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(HERE.parents[1]))
    from tarzan.stress import checks, driver, matrix, net

    if args.list:
        print(matrix.summarise())
        return 0

    if args.quiet:
        logging.disable(logging.WARNING)

    net.install_socket_guard()
    session = Path(tempfile.mkdtemp(prefix="tarzan_stress_"))
    cache = driver.prepare_cache(CACHE_SNAPSHOT, session / "cache")
    started = time.time()
    runs = 0
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    fh = LEDGER.open("a")
    _log(fh, {"kind": "session_start", "session_dir": str(session),
              "cache_snapshot": str(CACHE_SNAPSHOT), "budget": BUDGET,
              "matrix": matrix.summarise()})

    blocks = args.only or ["A", "B", "C", "D", "E"]
    results = {}

    def execute(cell, **kw) -> object:
        nonlocal runs
        if runs >= BUDGET["runs"]:
            _log(fh, {"kind": "stop_rule", "why": "run budget exhausted"})
            raise SystemExit(2)
        if time.time() - started > BUDGET["wall_clock_s"]:
            _log(fh, {"kind": "stop_rule", "why": "wall-clock budget exhausted"})
            raise SystemExit(2)
        runs += 1
        t0 = time.time()
        res = driver.run_one(portfolio=cell.portfolio, instant=cell.instant,
                             mode=cell.mode, strict=cell.strict,
                             outroot=session / "out", cache_dir=cache,
                             run_id=kw.pop("run_id", cell.cid),
                             as_of_override=cell.as_of, **kw)
        _log(fh, {"kind": "run", "cell": cell.cid, "portfolio": cell.portfolio,
                  "instant": cell.instant, "mode": cell.mode, "strict": cell.strict,
                  "as_of": cell.as_of, "note": cell.note,
                  "argv": res.argv, "exit_code": res.exit_code,
                  "error": _first_line(res.error),
                  "network_attempts": len(res.network_attempts),
                  "network_hosts": sorted(set(res.network_attempts))[:6],
                  "seconds": round(time.time() - t0, 1),
                  "artifacts": sorted(p.name for p in Path(res.outdir).glob("*/*/*"))
                  if res.outdir else []})
        return res

    tally: dict = {}

    def record(cell_id: str, verdicts: list) -> None:
        for v in verdicts:
            tally[v.state()] = tally.get(v.state(), 0) + 1
            # v.state() is the ONE conversion. This copy of the ternary dropped
            # XFAIL/XPASS, so the ledger recorded four expected-failure passes as
            # ordinary PASSes and one expected failure as a real FAIL.
            _log(fh, {"kind": "check", "cell": cell_id, "check": v.check,
                      "type": v.kind, "verdict": v.state(), "void": v.void,
                      "expected_fail": v.expected_fail, "detail": v.detail})
            print(f"  {cell_id:5s} {v.line()}")

    try:
        # ---- A: coverage -------------------------------------------------- #
        if "A" in blocks:
            print("\n=== BLOCK A — coverage ===")
            for cell in matrix.block_a():
                res = execute(cell)
                results[cell.cid] = res
                qty = _quantities(cell, cache)
                record(cell.cid, _per_run_checks(res, cell.portfolio,
                                                 _effective(cell), quantities=qty))

        # ---- B: determinism twins ---------------------------------------- #
        if "B" in blocks:
            print("\n=== BLOCK B — determinism (D1, C13) ===")
            for cell in matrix.block_b():
                a = execute(cell, run_id=cell.cid + "a")
                b = execute(cell, run_id=cell.cid + "b")
                record(cell.cid, checks.d1_reproducible_identical(a, b))
                record(cell.cid, checks.c13_planning_determinism(a, b))
                record(cell.cid, _per_run_checks(a, cell.portfolio, _effective(cell),
                                                 quantities=_quantities(cell, cache)))

        # ---- C: differential --------------------------------------------- #
        if "C" in blocks:
            print("\n=== BLOCK C — differential (D3, D4, D2) ===")
            work = session / "work"; work.mkdir(exist_ok=True)
            for cell in matrix.block_c():
                if cell.cid.startswith("C0"):                 # D3 permutation
                    base = execute(cell, run_id=cell.cid + "base")
                    perm = execute(cell, run_id=cell.cid + "perm",
                                   orders_override=_permuted_orders(cell.portfolio, work, 7))
                    record(cell.cid, checks.d3_row_permutation(base, perm))
                elif cell.cid.startswith("C1"):               # D4 lookahead
                    full = execute(cell, run_id=cell.cid + "full")
                    trunc = execute(cell, run_id=cell.cid + "trunc",
                                    orders_override=_truncated_orders(
                                        cell.portfolio, work, cell.as_of))
                    record(cell.cid, checks.d4_no_lookahead(full, trunc))
                else:                                          # D2 invariance
                    a = execute(cell, run_id=cell.cid + "a")
                    b = execute(matrix.Cell(cell.cid, cell.portfolio, "C5", "LIVE",
                                            False,
                                            note="D2 instant B: Wed 23:30, post-close "
                                                 "(same day, also closed)"),
                                run_id=cell.cid + "b")
                    record(cell.cid, checks.d2_time_of_day_invariance(a, b))

        # ---- D: degradation ---------------------------------------------- #
        if "D" in blocks:
            print("\n=== BLOCK D — degradation (G11, G12) ===")
            for cell in matrix.block_d():
                kw = {}
                if cell.cid == "D02":
                    empty = session / "empty_cache"; empty.mkdir(exist_ok=True)
                    kw["run_id"] = cell.cid
                    res = driver.run_one(portfolio=cell.portfolio, instant=cell.instant,
                                         mode=cell.mode, strict=cell.strict,
                                         outroot=session / "out", cache_dir=empty,
                                         run_id=cell.cid)
                    runs += 1
                    _log(fh, {"kind": "run", "cell": cell.cid, "note": cell.note,
                              "exit_code": res.exit_code, "argv": res.argv,
                              "error": _first_line(res.error),
                              "network_attempts": len(res.network_attempts)})
                else:
                    res = execute(cell)
                results[cell.cid] = res
                qty = _quantities(cell, cache)
                record(cell.cid, _per_run_checks(res, cell.portfolio,
                                                 _effective(cell), quantities=qty))
            store = session / "claims.json"
            record("D07", checks.g12_duplicate_claim_suppresses(
                store, "stress-normal-2026-08-26", "digest-abc"))

        # ---- E: external -------------------------------------------------- #
        if "E" in blocks:
            print("\n=== BLOCK E — external verification (E9, E10) ===")
            record("E10", checks.e10_sessions_from_the_calendar([
                ("XMME.MI", dt.date(2026, 12, 25), "Christmas, Borsa Italiana"),
                ("XDEQ.MI", dt.date(2026, 1, 1), "New Year, Borsa Italiana"),
                ("SWDA.MI", dt.date(2026, 4, 3), "Good Friday, Borsa Italiana"),
            ]))
            from tarzan.stress import clock
            clock.install(clock.pin_for("C7"))
            # Name the date actually measured: the installed pin is C7, and the
            # verdict line used to print 2026-12-25 while evaluating C7's own day.
            record("E10", checks.e10_market_open_reads_the_calendar(
                clock.INSTANTS["C7"], "XMME.MI", clock.INSTANTS["C7"].date()))
            _log(fh, {"kind": "note", "cell": "E01",
                      "detail": "live tape comparison requires network; run separately "
                                "with STRESS_ALLOW_NETWORK=1"})
    except SystemExit as exc:
        print(f"\nSTOPPED by stop rule (exit {exc.code})")
    finally:
        _log(fh, {"kind": "session_end", "runs": runs, "tally": dict(tally),
                  "seconds": round(time.time() - started, 1),
                  "network_attempts_total": len(net.attempts())})
        fh.close()
        # VOID is printed apart from SKIP and NOT counted as coverage. Collapsing
        # the two is how twelve runs that produced nothing read as a clean session:
        # a SKIP means the check looked and the state was not there, a VOID means
        # there was nothing to look at.
        order = ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP", "NOTE", "VOID")
        line = " · ".join(f"{k} {tally[k]}" for k in order if tally.get(k))
        print(f"\n{sum(tally.values())} verdicts: {line}")
        if tally.get("VOID"):
            print(f"  {tally['VOID']} VOID = no artifact to judge. NOT coverage; "
                  "C0 names the runs that produced nothing.")
        print(f"{runs} runs in {round(time.time()-started)}s · "
              f"network attempts {len(net.attempts())} · ledger {LEDGER}")
        print(f"session tree {session} (delete when done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
