# Native Yahoo timezone-cache readiness

An owned native client serializes the inspected yfinance 1.7.0 timezone
store's lazy initialization under its existing client-creation lock.
Initialization finishes before any Ticker created with that client can enter
concurrent timezone lookups. This is local SQLite preparation, not a market
request, timezone entry, negative cache, source selection or freshness proof.

The SDK has separate lazy instance/database/table initialization. Concurrent
first lookups on a cold store can raise an OperationalError; the native batch
then retains an unavailable history for that child. Real HTTP loopback probes
with synthetic prices reproduced the missing child without a provider outage.
This does not establish that a user's previous live issue suffered that race.

The preparation is limited to the inspected native Session factory, SDK,
cache getter, store and initializer. Unsupported SDKs and injected factories
or high cache ports keep their previous contract. It does not patch SDK
classes, clear caches, alter cookie/crumb recovery or move a request deadline.
Positive stored timezone identities survive client close/recreation.

An ordinary store failure does not block construction of an otherwise usable
client. Only its exception type enters the fixed warning; path and exception
text are excluded. Actual owner cancellation still propagates before client
construction. Preparation holds no native HTTP permit or child-future wait.

Tests use genuine temporary SQLite stores, concurrent client users, retained
positive entries, injected ports, unsupported versions, cache failures and
owner cancellation. Physical parser/memo/admission tests use local synthetic
responses; frozen financial/HTML comparisons are separate release gates.
Neither test count nor this preparation is proof of faster email delivery,
perfect external sources, universal cache thread safety, or a fix for
concurrent public SDK users outside Tarzan's owned single-analysis boundary.
