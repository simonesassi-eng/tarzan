# Benchmark acquisition capability

Benchmark construction consumes a selected listing's history and currency.
When the existing venue-currency policy determines the currency, it does
not consume quote/profile metadata. The owned native path can therefore
omit that metadata without replacing the financial-data parser or changing
the currency policy.

`_fetch_benchmark_data` makes this narrow request through the original
ticker-data and bare-alias resolver. Their default contract remains complete.
The private capability flag does not affect ordinary enrichment.

For a qualified listing, the same history helper retains cache merging,
corporate-action coherence, current-day sibling recovery and row provenance.
For a bare alias, the original resolver performs exactly one mapping lookup.
The same current positive mapping, root/TTL checks and visible-history
admission apply. An absent, expired, malformed or unusable mapping still
requires the original metadata-based multi-exchange discovery.

Listings whose currency is ambiguous retain metadata and the existing FX
conversion, including native-currency return evidence. No empty info result
is inserted into the required-info memo; later consumers can still acquire
the current quote/profile they actually need.

The boundary requires an owned live transport, the inspected native SDK
and unchanged source-certified acquisition ports. Pinned/unowned callers,
unsupported SDKs and injected ports keep the complete existing contract.
Native metadata-based timezone fallback, authentication recovery, deadlines
and retry policies remain unchanged.

## Verification and limits

Tests use the installed SDK with synthetic primitive provider responses.
They cover direct and cached identities, multiple exchanges, unknown currency,
FX instants, cache expiry/corruption, unavailable history, recovery on another
exchange, later-run recovery, essential/optional phases and injected ports.
An evolving-cache adversary requires the same single lookup and selected
identity as the original resolver.

Raw-HTTP frozen portfolio comparisons require identical inputs, clock,
initial cache and observations. Financial facts, plans, source/quality
evidence, native chart demand and rendered HTML must remain equivalent.
Personal snapshots stay private.

The SDK's process-global timezone fallback can change which missing aliases
receive metadata attempts. Net logical-query reduction is not a claim that
every query set is an exact subset, nor a count of physical HTTP/auth retries.
Offline replay, no-email verification, SMTP acceptance and whole-workflow
completion remain separate measurements. Live runtime savings must be
measured; they cannot be inferred from a smaller frozen query count.
