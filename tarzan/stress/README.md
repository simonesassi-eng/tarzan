# Structural stress bench

A reusable regression corpus for the whole pipeline: ten seeded synthetic books
run through the real orchestrator across three run modes and seven pinned
instants, with every assertion made against an oracle that is not the code under
test.

## Running it

```bash
python -m tarzan.stress.run                     # the whole matrix, offline
STRESS_ALLOW_NETWORK=1 python -m tarzan.stress.external   # the ONE networked block
python -m tarzan.stress.generate                # rewrite the fixtures from their seeds
```

The matrix takes ~4.5 minutes and 71 runs. It must report `network attempts 0`;
any other number means a fetch boundary escaped the guard and every result in
that session is suspect.

## What it guarantees about isolation

* Each session writes to its own temporary tree; `input/` and `output/` are never
  read or written.
* The market cache is a per-session **writable copy** of the snapshot, so a run
  that stores a row cannot contaminate the next session's baseline.
* No SMTP. Delivery is checked at the level of the publication decision and the
  claim store, never by sending.
* The guard covers Python sockets **and** `curl_cffi`, because yfinance 1.1.0
  drives libcurl and is invisible to a socket patch. It also replaces
  `yfinance.Ticker` wholesale rather than individual helpers.
* Every book and order list derives from an explicit seed. No real position, ISIN
  or size from the actual book appears anywhere in `fixtures/`.

## Reading a verdict

| verdict | meaning | coverage? |
|---|---|---|
| `PASS` | the oracle agreed | yes |
| `FAIL` | the oracle disagreed — a claim about the product | yes |
| `SKIP` | the state this check needs did not arise | yes — it looked |
| `NOTE` | observed and deliberately NOT asserted (no sound oracle) | no |
| `VOID` | there was no artefact to judge | **no** |
| `XFAIL` | a pre-registered expected failure | yes |
| `XPASS` | a pre-registered expected failure that passed | yes |

The distinction between `SKIP` and `VOID` is the one that matters, and conflating
them is how the bench ran past two defects seventy-one times.

A `SKIP` says the check looked and the state was not there — a EUR-only book has no
currency to convert, and that is a fact about the fixture. A `VOID` says there was
nothing to look at, and the reason there was nothing is usually the defect itself.
Twelve runs produced no newsletter at all; every check that needed the HTML returned
`SKIP`, and one cell reported "4 PASS, 5 SKIP, 0 FAIL" for a run that exited 1.
Worse, seven checks passed VACUOUSLY on those runs: `None == None` is true, so two
runs that wrote nothing were certified byte-identical, and "every failure record
carries an id" was true of zero records.

So `C0` now gates every run — exit 0 and a rendered newsletter, no exemptions — and
`VOID` is printed and counted apart from `SKIP`. If you see `VOID`, `C0` should be
failing beside it; if it is not, the bench has another blind spot.

**Known open item:** `E9` reports `VOID` for one instrument in the networked block.
Its row renders offline in the same configuration, so the gap is in live resolution,
not in the oracle. Left visible rather than reclassified.

## The thing this bench keeps teaching

Six of the fourteen defects the first sessions surfaced were in the bench, not in
the product — a row parser that could not match a label longer than eight
characters, a section scan with no end boundary, a fixture whose bond cash leg
ignored the per-100 convention, a clock pin that stayed installed across cells, a
cache still patched in during the networked block, an oracle off by one session.
Each one produced a confident, specific, entirely false accusation.

So: a failing assertion is a claim about the harness until the artefact itself has
been read. `C7`'s docstring records a check that was withdrawn for exactly this
reason after three attempts to save it.
