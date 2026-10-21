# Review implementation register

Review base: `b7b579c`. Changes are grouped into independently tested commits.
Actual portfolio inputs, captures and KPI deltas remain in ignored local
artifacts. This register contains no personal portfolio amounts.

| Findings | Scope | State |
|---|---|---|
| Verification | Frozen raw-provider capture, offline KPI comparison | Implemented; two baseline replays match, no network attempts or semantic errors |
| R01–R03, R06 | Target gates, closed booleans, cash-flow signs | Implemented; 64 new regressions, 1,617 full-suite passes, unchanged frozen KPIs |
| R04 | Instrument resolution and valuation provenance | Implemented; persistent negative mechanics evidence, explicit valuation origins, no frozen KPI change |
| R05 | Evidence-backed corporate-action repair | Implemented; no jump-only events, explicit provider/curated factors, separate price rulers and pre-repair storage |
| R07, R18 | Causal identity and validated cache enumeration | Implemented; shared join/seed catalog, boundary-aware profile/xref reads and immutable target aliases |
| R08, R20 | Single-flight and real provider budgets | Pending |
| R09, R13–R14, R19 | Render lifecycle, finalization outcomes, optional work, logging | Pending |
| R10–R12, R25 | Complete analysis identity, immutable orders/ledger, linear projection | Implemented; full financial facts hashed, defensive order facts, deeply immutable ledger, atomic remedy ordinals and linear/cached failure projection |
| R15–R17, R24 | Drive selection/lifecycle and authenticated Update messages | Pending |
| R21–R23 | Public-cache allowlist, bounded event queue, incremental snapshots | Pending |
| R26–R27 | Domain/report boundaries, shared identity catalog, consistent installation/docs | Pending |

Remote legacy-market cache cleanup was explicitly authorized. Only the eight
resolved `tarzan-market-cache-*` IDs were deleted; the pip cache was preserved.
Deleted snapshots cannot be recovered. The currently published legacy workflow
can recreate them until the revised workflow is pushed. No push or production
Apps Script change is part of this implementation turn.

Configuration paths and uploads now share one parser. Duplicate keys, missing
headers, malformed CSV and invalid financial scalar domains reject a present
input rather than selecting defaults. Optional blank values retain documented
defaults. Economic cash-sign errors reject the complete order input even in
lenient column mode; no row is silently removed and no amount is sign-flipped.
Legitimate equal broker fills remain distinct orders.

Provider mechanics evidence now survives taxonomy/display rewrites. Selected
market valuation is an immutable EUR-per-unit fact; an anchor is never primary
merely because an unvalued quote remains on the holding. Live stamps, returns
and tax consumers use the same retained mechanics authority. Existing stamp
fixtures now declare their ETF mechanics explicitly; return/price expectations
were preserved. MONEY's independently verified generic-EQUITY reconciliation
remains narrow, and specific provider conflicts still block valuation.

The analysis digest now covers all published financial tables and full series,
effective target rows, instrument-resolution evidence and the actually used
configuration. Operational clocks are excluded. The narrow external summary
schema remains version 3; its contents did not expand. Effective orders are
defensive immutable facts (including equal legitimate fills), and ledger
projections cannot mutate nested source evidence. Failure projection indexes
one consistent entry snapshot, caches immutable results, and never rescans the
whole log per failure. Remedy ordinals are assigned atomically at append.

Split repair no longer guesses events or round factors from persistent losses.
It requires a provider `Stock Splits` declaration or the exact curated listing
and effective date. CL2's existing 300 factor is retained; Euronext's public
Milan notice `MIL_20231004_03289_ETP` independently confirms the split, unchanged
ISIN `FR0010755611` and 10 October 2023 effective date. The public abstract does
not disclose the factor, which remains explicit prior curation corroborated by
the Paris price ruler; no paid notice was purchased or claimed to be read.
Local repair neither waits for nor consumes a future confirmation window.
Provider history is cached before local repair, so a later historical boundary
cannot inherit a future local adjustment. History-only cache version is now 3.

Target joins and seed checks now share `TargetIdentityCatalog`, retaining exact
ISIN compatibility and alias collision guards. Future profile/xref observations
cannot cross a historical boundary; unchanged xref refreshes preserve first
knowledge, and changed mappings retain a bounded observation history. Legacy
timeless xrefs remain readable only in LIVE mode; they cannot prove past
knowledge. Similar-name prefixes no longer suppress an explicit new target.
Validated target plans/rows are immutable while preserving aliases of one source
row. Manual-series listing uses current checksum/schema-validated JSON histories,
not obsolete pickle globs. Stored history now also carries its original symbol
to prevent sanitized filename collisions from supplying another ticker's data.
