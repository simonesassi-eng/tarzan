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
