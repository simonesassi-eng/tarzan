# Native Yahoo ISIN source identity

The inspected, owned yfinance 1.7 constructor can accept an ISIN and internally
select a Yahoo listing. Previously Tarzan sometimes labelled the resulting
metadata/history with the requested ISIN rather than the listing actually
queried. Candidate ranking could consequently use the wrong venue identity.

Only a positively observed native constructor records a transient, immutable
requested/selected identity. The existing run owner retains successful facts,
not clients or cached misses. No additional constructor, search, request,
credential or persistent resolution cache is introduced. An arbitrary
metadata `symbol` field cannot create this authority.

Metadata retains its ordinary dictionary fields and JSON representation.
Defensive run-memo copies preserve the typed constructor evidence. Existing
positive identity remains usable when metadata fails or transport expires;
another run does not inherit this transient authority. Native SDK caches and
their existing recovery policies are not represented as perfect data sources.

Daily history, downstream selected-ticker evidence and candidate ranking use
the actual observed listing. Qualified explicit symbols remain qualified.
All configured candidate probes, including qualified ISIN-plus-suffix symbols,
remain: native constructor search is not proof that those alternatives lack
data or that they cannot rank better. Configured venue preferences and source
admission rules remain; correcting the candidate's identity can legitimately change its
rank and is a correctness change, not a policy-neutral speed optimization.

Legacy raw-ISIN cache rows do not identify their venue/ruler. They are neither
deleted nor merged into a newly observed listing. That listing uses its own
ordinary cache and refresh/revision/fallback rules. Unknown SDKs, unowned or
pinned callers and injected acquisition ports retain the original contract.

Synthetic native-parser tests cover search and cached mapping, several
venues/currencies, empty/failed metadata/history, source-copy isolation,
parallel single-flight, serialization, qualified/injected/pinned contracts
and incompatible raw versus canonical caches. They do not prove real-source
identity accuracy, physical HTTP/authentication recovery or email latency.
