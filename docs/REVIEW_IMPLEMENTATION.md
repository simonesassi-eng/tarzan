# Review implementation register

Review base: `b7b579c`. Changes are grouped into independently tested commits.
Actual portfolio inputs, captures and KPI deltas remain in ignored local
artifacts. This register contains no personal portfolio amounts.

| Findings | Scope | State |
|---|---|---|
| Verification | Frozen raw-provider capture, offline KPI comparison | Implemented; two baseline replays match, no network attempts or semantic errors |
| R01–R03, R06 | Target gates, closed booleans, cash-flow signs | Implemented; 64 new regressions, 1,617 full-suite passes, unchanged frozen KPIs |
| R04 | Instrument resolution and valuation provenance | Implemented; persistent negative mechanics evidence, explicit valuation origins, no frozen KPI change |
| R05 | Evidence-backed corporate-action repair | Pending |
| R07–R08, R18, R20 | Causal identity, single-flight, cache API, real provider budgets | Pending |
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
