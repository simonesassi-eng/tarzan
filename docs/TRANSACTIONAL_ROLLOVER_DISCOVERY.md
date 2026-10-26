# Transactional rollover discovery

Rollover discovery can join an unknown bond identifier to a verified bond
identifier only when the existing complete set of transaction, issue, currency,
source, quantity, price and date checks agrees. Names or matching prices alone
do not establish identity.

Per-instrument eligibility is now prepared before pair comparison:

- Only unknown or exact bond mechanics can enter a candidate pair.
- An incoming identifier must contain only positive incoming transfers.
- An outgoing identifier must contain only negative sales/outgoing transfers.
- Mixed directions, absent position movements and conflicting kinds cannot
  become candidates.

All original pair checks, numerical tolerances and lexical ordering remain.
Every qualifying pair still enters the complete ambiguity graph. An identifier
with two possible counterparts is not greedily joined to the first one.
Actual duplicate broker tranches, distributions, carrying basis and cash legs
are preserved; only effective-order copies receive established mechanics.

The change avoids quadratic pair checks for ordinary non-bond holdings and
buy-only ledgers. A genuinely dense population of eligible rollovers can still
require quadratic pair comparison; this is not a universal linearity claim.
It does not omit Yahoo requests, select another exchange, change price-source
priority or make a partial history complete.

Verification includes an independent exhaustive selection oracle, generated
ledgers, ambiguity, both mechanics directions, excluded/conflicting identities,
currency/source/date/quantity/price boundaries, duplicate tranches and immutable
effective-order basis. Release also requires full suites and matched cold/warm
portfolio, source and HTML comparisons. Discovery-only timing is not newsletter
delivery latency.
