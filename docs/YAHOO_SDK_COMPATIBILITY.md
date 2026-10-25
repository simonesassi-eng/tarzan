# Yahoo SDK compatibility boundary

Tarzan pins yfinance 1.7.0 and curl-cffi 0.15.0 in its single immutable
Python 3.12 dependency lock. curl-cffi versions below 0.15.0 are affected by
GHSA-qw2m-4pqf-rmpp / CVE-2026-33752. Updating a dependency does not prove
that an application was exploitable or has been exploited.

The SDK stack update preserves all unrelated locked versions and hashes.
It introduces the native SDK/transport dependencies lxml, rich,
markdown-it-py, mdurl and pygments; frozendict is no longer required.
Runtime installation remains the exact projection of the same hash lock,
with dependency resolution and hash verification enabled.

## Metadata acquisition

`tarzan.data.yahoo_info` is a bounded adapter for the inspected native SDK,
not a replacement financial-data parser. An owned live caller delegates
quote-summary and quote parsing to the SDK and omits only its separate
complementary trailingPegRatio request, which enrichment does not consume.
It retains every noncomplementary field, including nested profile fields.

The adapter checks the native client/factory, ten inspected entry ports,
their source boundaries, the Quote constructor alias and the native curl
HTTP-error contract. Unsupported SDKs, alternative backends, injected
ports, public clients, direct calls and pinned callers keep full `info`.
The client created by the adapter never escapes. No SDK class or global
authentication/retry flag is patched.

The inspected SDK preserves a valid quote-summary when the quote endpoint
returns a native HTTP error, and preserves a valid quote when quote-summary
fails. A retained previous close is not invented into a timestamped live
quote. Both sources unavailable remain unavailable.

Raw-parser fault tests and native transport tests are different gates.
The latter retain the SDK's authentication recovery and Tarzan's native
retry/deadline policy. An exhausted native request budget still makes the
operation unavailable; the adapter does not bypass it to claim a price.

## Batch timezone contract

Owned Tarzan intraday batches retain UTC indices and the same instants;
`ignore_tz=True` retains venue-local labels. This is deliberate: newer native
SDK batches can retain a shared listing timezone. Shape/value/instant tests
compare the native result after explicit timezone conversion, including
different exchange timezones. Daily labels are not shifted through UTC.

## Release evidence and remaining limits

The update is gated by clean hash-locked installation, closed dependency-set
checks, platform wheel provenance, the complete suite and raw-HTTP frozen
comparisons using the actual SDK parsers. Frozen comparisons retain identical
inputs, clock, initial identity cache and observations. Personal portfolio
snapshots and deltas remain private and are not published here.

The SDK's process-global two-attempt metadata fallback for timezone lookup
remains unchanged. Concurrent scheduling can select different missing aliases;
verification captures real observed responses and fails closed on an
unrecorded query. It never invents an absent response or disables that fallback
to make a comparison pass.

The upgrade is not, by itself, a demonstrated production runtime improvement.
Offline replay, verification-only execution, creation-to-SMTP acceptance and
whole-workflow completion are distinct measurements. Missing/stale external
prices and quality warnings remain explicit.
