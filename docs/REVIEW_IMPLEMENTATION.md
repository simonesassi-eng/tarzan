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
| R08 | Run-owned single-flight provider stores | Implemented; concurrent misses/negative answers coalesce per key, cancellation propagates, unrelated keys remain parallel |
| R20 | Real provider budgets and acceptance/publication policy | Implemented locally; separate transport policies, native capped I/O and bounded retries/deadlines; accepted valuation coverage and independent publication thresholds are enforced |
| R09, R13–R14, R19 | Render lifecycle, finalization outcomes, optional work, logging | Implemented; retained output lease/clock/ledger, explicit artifact outcome, opt-in isolated What-If and idempotent run-owned capture |
| R10–R12, R25 | Complete analysis identity, immutable orders/ledger, linear projection | Implemented; full financial facts hashed, defensive order facts, deeply immutable ledger, atomic remedy ordinals and linear/cached failure projection |
| R15–R16, R24 | Drive selection/lifecycle | Implemented; complete bounded listing, unique native file identity, private owned staging and cleanup on all exits |
| R17 | Authenticated Update messages | Implemented locally; native Gmail owner/message-level SENT verification, unquoted commands and bounded per-message idempotency; Gmail API service/authorization needed on deployment |
| R21–R23 | Public-cache allowlist, bounded event queue, incremental snapshots | Pending |
| R26 | Domain/report boundaries and shared identity catalog | Implemented; shared target identity catalog, defensive ReportView/RunRenderContext and immutable prepared display references; real section builders/semantic gate do not consult later global taxonomy |
| R27 | Consistent installation/docs | Pending |

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

Provider stores now belong to `RunSession`, including info, history, FX,
OpenFIGI, geography, benchmark history, quote pairs, justETF and bond prices.
Concurrent misses share a per-key Future; batched callers reserve only their
new keys. Misses are cached, exceptions/cancellation reach waiters, recursive
same-key fetches fail promptly, and returned data is defensive. No network work
holds the shared memo lock. justETF workers inherit the owning run context.
The configured history period and market strip also remain run-owned.
The pinned-history test now checks actual behavior instead of source indentation.

Completed results retain their session and immutable `ReportView`. Output
scopes reacquire the serial lease, bind the original clock and ledger, and restore
the caller on exit. Financial tables, nested values, series and configuration
are immutable facts with defensive typed pandas projections. Fixed market-context
quotes are prepared during analysis; newsletter sections and the AI digest do
not fetch them. AI attempts, diagnostics and logs stay with their original owner.

CLI finalization returns a typed outcome; exit success is decided only after
the core manifest commits. Rendering/semantic errors retain computed financial
evidence but block normal output. `--what-if` is now opt-in, runs after the
core commit and writes a separate `what-if/<attempt>/` artifact set; optional
failure cannot overwrite the core newsletter/manifest. Console setup reuses its
owned handler without removing foreign handlers, and capture buffers are
run/context-owned. Artifact names/attempt paths reject traversal; new attempt
directories are private. SMTP state semantics and resend authorization are not
changed by these CLI/output fixes.

Drive selection follows `nextPageToken`, rejects ambiguous names before any
download, bounds pages/files/bytes/chunks and validates available native size
and MD5 revision evidence. Google OAuth read-only credentials authorize the
actual listing; transport uses a real 20-second native HTTP timeout. No staging
is created before authenticated listing/required-file selection succeeds.
`DownloadedInputSet` owns a new private temporary directory even when a caller
provides a destination parent; caller files are not overwritten or removed.
Delivery retains that owner until its final exit. Partial download/listing
failure cleans staging, and incomplete Drive configuration cannot silently
select an older local input set. Original local inputs retain caller ownership.

Update requests now require agreement between the trigger's effective owner and
the native Gmail profile, then the individual message's native `SENT` label,
exact owner From header and recent internal timestamp. Received/automated mail,
quoted commands and unavailable native authorization fail closed. Per-message
markers and stable opaque event IDs allow later replies in the same thread
without resending a handled reply. Searches, verification and retained markers
are bounded; capacity pauses new requests rather than discarding recent markers.
Legacy handled threads migrate on the protocol upgrade. GitHub dispatch payloads
no longer contain email subjects, bodies or mailbox addresses. SMTP identifies
generated newsletters with standard automation and application headers.

The actual production Apps Script source is executed in Node with native-service
doubles, including spoofing, missing OAuth, quotation, pagination, migration,
locking, marker capacity and later same-thread replies. The existing delivery
claim/CAS protocol tests remain intact. No production script was changed; the
operator must add Gmail API and authorize `installTrigger()` for this upgrade.

`ReportReference` prepares the curated names, roles, benchmark identities,
watchlist and risk-rating bands once at the analysis boundary. Its mappings and
sequences are deeply immutable and part of the material analysis digest. The
completed `ReportView` retains them; all production section builders and the
semantic gate consume those same facts rather than consulting later global
configuration. Legacy unprepared cubes retain a single compatibility preparation
at the context boundary. The golden newsletter remains byte-identical; a new
regression forbids every global reference accessor during a completed render.

Provider transport policy is now separate from valuation acceptance. The
configured run deadline starts after validated configuration is bound; pacing,
HTTP retries and native I/O consume it. Yahoo uses its real `curl_cffi` session
port (including cookie/crumb traffic), without a second outer retry multiplier.
Owned intraday SDK batches run without internal threads so their operation
context is retained; independent outer enrichment/referee workers remain
parallel. The previous SDK default may impose a lower timeout than the configured
ceiling. OpenFIGI, Borsa, justETF, optional external proxies and Gemini pass their
configured timeout to the native request. Native clients close on analysis exits.
Before-input Drive authentication retains its separate bounded native timeout.
Deadlines are checked before/after calls and waits; this does not terminate
arbitrary Python/CPU work or promise a hard whole-newsletter wall-clock limit.

Accepted valuation coverage is weighted by materiality basis within each
kind/policy. Explicitly permitted anchors/stale fallbacks count as accepted
valuation, not as real market-history coverage. An insufficient or indeterminate
coverage makes the total/planning unavailable. Valuation and publication
materiality thresholds are assessed independently: a policy-permitted degraded
publication cannot label an unavailable total as complete. Unknown materiality
remains fail-closed. The narrow external summary remains schema 3.

Fake-clock and native-port tests cover changed timeout/retry settings, operation
and run exhaustion, pacing, cancellation, HTTP status classification, SDK retry
nonmultiplication and non-Yahoo adapters. Independent gate tests cover coverage,
fallback acceptance, separate publication thresholds and unavailable summaries.
