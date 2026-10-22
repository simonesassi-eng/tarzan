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

The diagnostic upload requests 14-day retention. A killed/cancelled process,
missing key or storage outage can still prevent capture/upload. A normal send's
diagnostic failure cannot change its delivery outcome or authorize a resend.
An explicit verification fails if its requested encrypted capture is unavailable.

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

The configured limits are 540 seconds of shared provider budget, 120 seconds for
optional reference discovery and 90 seconds reserved for final evidence and
publication. Native timeout/retry/pacing budgets apply in copied worker contexts;
these limits do not terminate arbitrary Python CPU work.

For a code-only financial comparison, use
[frozen KPI verification](FROZEN_KPI_VERIFICATION.md): same observations, input
hashes, initial cache and captured clock. Two live runs answering at different
times do not prove that code changed portfolio returns.
