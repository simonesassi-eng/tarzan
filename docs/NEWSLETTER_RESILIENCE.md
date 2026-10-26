# Newsletter verification and recovery

The cache is an optimization, not financial authority. Cold, partial, absent or
invalid snapshots must fall back to bounded native requests and explicit
availability. Existing financial/identity/semantic gates remain enforced.

## Verify actual inputs without sending

Locally, use a fresh private output directory:

```bash
python scripts/verify_newsletter.py \
  --input-dir input \
  --private-output .private/verification-cold
```

This reads the same order/target inputs and runs the actual analyzer and
newsletter renderer. AI is disabled. It never enters `run_and_send`, obtains a
delivery claim or calls SMTP; mail credentials are unnecessary. Exit code zero
means normal/degraded publication is allowed, not that every market feed was
available. Inspect `report.json` quality, ledger and KPI projections.

On GitHub, run **Tarzan Newsletter** with the boolean **verify_only** enabled on
the intended branch. Its separate step receives only Drive input credentials
and the snapshot key, not SMTP, recipient, Gemini or claim credentials. Its
concurrency group is separate from normal deliveries. Apps Script and automatic
`repository_dispatch` sends need no changes and do not enable this mode.

## Private persistence on a public repository

`TARZAN_CACHE_KEY_B64` is an independent 256-bit base64 key held by native GitHub
Secrets. AES-GCM authenticates each encrypted snapshot under a purpose-specific
context, with a fresh nonce. Only these explicit ciphertext files leave the
runner:

- Cache: `market.sealed`, containing allowlisted market histories/resolutions,
  instrument profiles, provider maps, geography, TER and cross-references.
- Diagnostic artifact: `report.sealed`, containing owned ledger evidence,
  availability, native/adapter outcome counts, stage timings and financial
  projections. No environment dump, raw SDK capture or credential values.

No inputs, claims, cookies, output directories or manual/licensed series are
cached. Authentication and validation precede decompression/private writes.
Restoration preserves timestamps/checksums and does not renew TTLs. Existing
run-local files win; the public macro fallback only fills missing histories.
Every attempt saves under a unique run-ID/attempt/key-fingerprint key, avoiding
same-day immutable-key freezes. Native cache eviction/retention is best-effort.

The encrypted import can skip the redundant public **restore/import** only
after proving all fixed macro histories authenticated, schema-compatible and
usable within the existing five-calendar-day refresh-tail window. A successful
but empty/partial import, disabled cache, invalid key/tag, stale data or future
date keeps the fallback enabled. Null daily padding is not an observation: the
guard dates the last real close, exactly as the macro consumer does. This is
only a bootstrap reuse guard, not a relaxed quote policy; ordinary daily-tail
and intraday refreshes still run, and no timestamp or map TTL is renewed.
The public **export/save** continues on every attempt to keep the independent
fallback recoverable. A corrupt previous public projection cannot block
export of new validated macro tables; valid previous tables are reused only
when their private counterparts are missing.

The diagnostic upload requests 14-day retention. A killed/cancelled process,
missing key or storage outage can still prevent capture/upload. A normal send's
diagnostic failure cannot change its delivery outcome or authorize a resend.
An explicit verification fails if its requested encrypted capture is unavailable.
Repository/organization limits may shorten requested artifact retention:
check the upload warning and artifact expiration rather than assuming 14 days.

Normal Drive sends retain SHA-256 receipts for the known input filenames in
`input_sha256`, using the same downloaded buffers that passed size/checksum
validation. These immutable receipts survive staging cleanup and belong only
to that exact native attempt. They describe downloaded bytes, not a later
caller-modified file. Local, injected or unavailable input observations do not
invent hashes; explicit verification fingerprints keep their existing contract.
Receipt collection is optional and cannot change prices, Analysis ID or delivery.

`run.final_acquisition` independently observes the retained native coordinator
at diagnostic capture, including work after analysis such as rendering requests.
It does not rewrite the earlier analysis-scoped `TELEMETRY` ledger event.
Native and protected counts are observed atomically without waiting or acquiring
a provider permit; unavailable/contended observations are omitted or marked
`FINAL_ACQUISITION_UNAVAILABLE`, never replaced with zeros. Bounded measurements
retain truncation. `admission_closed` describes the actual coordinator state:
client cleanup does not itself close the shared admission authority. These
optional version-1 fields are descriptive, not financial or publication gates.

New native final observations also contain a UTC `observation_window` bracketing
the nonblocking coordinator read. This is the observer's receipt-time clock,
not the report's `captured_at`, which remains the immutable analysis-start
clock (including point-in-time runs). A backwards, unavailable or invalid
observer clock marks only this optional observation unavailable; it cannot
retry delivery or change portfolio facts. Older final observations without
the window remain readable, but do not acquire a fabricated final timestamp.

## Recover an exact failure privately

Download the run's `tarzan-diagnostics-<run-id>-<attempt>` artifact into a private
local directory. Its only file is `report.sealed`. With the owner-only recovery
key:

```bash
python -m tarzan.runtime.diagnostics decrypt \
  --snapshot .private/downloaded-diagnostics/report.sealed \
  --key-file .private/secrets/tarzan-cache-key.b64 \
  --output .private/downloaded-diagnostics/report.json
```

The command refuses an existing output or unauthenticated data. Never print the
key, commit the decrypted report, or paste financial evidence into public logs.
Keep old recovery keys if rotating: a new secret cannot decrypt old artifacts.
Changing the key naturally creates a new cache namespace; old cache data are
ignored and reacquired, not trusted with the wrong key.

The ledger's failure payload gives the exact critical/source/semantic cause.
`provider_counts.native` counts native HTTP outcomes, separately from SDK/adapter
attempts and log messages. Repeated “404” log records are not automatically that
many network requests. SMTP acknowledgment is server acceptance, not inbox receipt.

Yahoo native attempts may additionally contain a fixed `native_request_class`:
`CHART`, `QUOTE`, `QUOTE_SUMMARY`, `FUNDAMENTALS`, `COOKIE`, `CRUMB`, `CONSENT` or
`OTHER`. Only an allowlisted HTTPS host/path shape determines the category; no
address, query, ticker, cookie or token is retained by this observer or its URL
parser. Unknown shapes remain `OTHER` and still follow the same native request,
timeout and retry policy. These labels distinguish price endpoints from cookie
and authentication endpoints; they do not establish quote quality or imply that
each HTTP 404 is a missing instrument. The category is operational, not a new
financial fact or analysis-identity input: it is retained only in native-attempt
ledger entries, leaving the financial material-field contract unchanged.

Owned Yahoo responses admit raw OHLC/adjusted-close price cells before SDK
coercion. Booleans, nonscalar JSON values, malformed numeric strings and
nonfinite/overflowing prices become missing observations rather than invented
zero/one prices or exceptions that discard the valid neighbouring rows.
Valid finite numeric prices/strings, timestamps, volume, metadata flags, corporate
actions, native response bytes/errors and alternative-source recovery retain
their existing contracts. This local admission adds no provider request or
negative cache. It cannot reconstruct original types already erased in an
older cached observation, and is not evidence that real portfolio data contained
such a malformed price.

An optional `run.clock` in the encrypted version-1 report records the native
context's actual `analysis_date`, mode and date basis (`HOST_LOCAL` or
`EFFECTIVE_DATE`), with the host UTC offset at capture when known. It does not
invent an exchange timezone or change the analysis clock. Legacy reports and
generic contexts need no new clock fields. If this optional projection fails,
`run.clock_projection_error_type` is the fixed `CLOCK_PROJECTION_UNAVAILABLE`;
the primary ledger, financial projection and outcome remain exportable. Keep
this operational failure distinct from `projection_error_type`.

## Verify session dates against their actual return source

The order-derived return uses the original flow-adjusted daily NAV. Its short
carried tail can end before the analysis date; raw euro accounting and capital
flows must still reach the analysis date. They are different source calendars.
For example, Saturday accounting can retain Friday's value or a weekend
deposit without becoming a new measured NAV observation.

The internal `return_nav_series` preserves the exact NAV consumed by performance
before its business-day resample. The immutable report keeps this source
independent of the display's attribution packet. The semantic guard verifies
its native dates, observed business endpoint and endpoint values against the
return history, then checks euro steps against the complete original values
and flows. Missing/corrupt NAV authority blocks delivery. Genuine weekend
observations keep their native dates; no packet-selected clipping, forced
Friday endpoint, accounting truncation or new return formula is permitted.
This provenance is deliberately financial material and changes the Analysis ID
when added; it does not change the external schema-3 summary or financial math.

A same-SHA verification on Friday does not verify Saturday's source supports.
Inspect `run.clock`, native observation dates and the accounting/NAV tails.
The regression suite exercises the original builder, immutable report, renderer,
semantic validator, verification entrypoint and dry-run publisher on Friday,
Saturday, Sunday and Monday pre-open. Network, delivery claims and SMTP are
prohibited in these tests; date and source tampering must remain blocked.

## Measure useful performance

For cold/warm comparisons, use separate fresh SDK caches/processes, identical
input hashes and no overlapping provider probes. Measure the whole verification
step and individual metric stages; a fast intraday batch is not a fast newsletter.
Check required references, history/valuation availability, value-weighted return
coverage, intraday bars and macro availability alongside elapsed time.

Keep the measurement boundaries explicit. The private diagnostic's
`duration_seconds` measures the entrypoint, excluding dependency imports and
diagnostic projection/encryption. `run.timings.imports_seconds` measures module
initialization after the shim's standard-library imports; the additive
`imports_and_entrypoint_seconds` still excludes interpreter startup, GitHub
bootstrap and cleanup. Workflow creation-to-completion includes runner/setup
and artifact/cache work. Creation-to-SMTP-ACK measures server acceptance, not
inbox receipt. Verification deliberately omits AI, claims and SMTP, so it is
not a full-send benchmark.

Stage wall times can overlap or contain nested stages. Process CPU measures all
threads active in that interval, not exclusive CPU attributable to one stage;
thread CPU measures only that stage's caller. Do not sum overlapping or nested
wall/process-CPU observations. Frozen provider-tape calls include SDK
constructors and consumer boundaries, not just physical HTTP requests; use the
native transport ledger for native-request counts.

The configured limits are 540 seconds of shared provider budget, 120 seconds for
optional reference discovery and 90 seconds reserved for final evidence and
publication. Native timeout/retry/pacing budgets apply in copied worker contexts;
these limits do not terminate arbitrary Python CPU work.

The native owned LIVE engine can prepare closed-position history and reference
resolution in two joined branches. Required reference resolution precedes
canonical holding/target/reference and fixed-macro prefetch, which precedes
optional listing discovery; the independent closed-position history branch can
overlap that work. Injected, pinned and ineligible callers keep sequential
preparation. Both branches drain before publishing their context and share the
same aggregate native-request cap. Executor worker counts are not separate
native-request allowances; the aggregate cap is stricter than the former
independent pools. A declared essential workload reserves one slot from
optional requests while those branches are owned and joined.

Optional probes can consume a provider's rate quota even without exhausting
the time budget. Existing quote, baseline, currency, identity and freshness
selection still happens on the ordinary paths.
Per-symbol batch histories are memoized only within that run, with every native
history parameter in the key and defensive copies. Previously acquired facts
remain readable after I/O authority expires; missing symbols cannot start new
native requests. No disk TTL is renewed and no intraday snapshot is exported.

For a code-only financial comparison, use
[frozen KPI verification](FROZEN_KPI_VERIFICATION.md): same observations, input
hashes, initial cache and captured clock. Two live runs answering at different
times do not prove that code changed portfolio returns.
