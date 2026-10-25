# Daily history cache coverage

A wide date span is not sufficient evidence that a cached daily history covers
the requested period. Tail-only reuse now requires valid, unique, ordered daily
dates, no future rows, finite positive Close and available Price Close values,
the existing period/slack rule and a row for every expected session between the
actual endpoints.

The fetcher supplies the selected symbol to the existing vendored calendar.
Unknown calendars or dates outside its horizon keep a full fetch. Generic
callers without a symbol cannot infer missing holidays; they conservatively
require all weekdays. An injected legacy predicate retains its original
two-argument call contract.

This does not invent prices, interpolate missing bars, suppress another venue
or change currency, source ranking, corporate-action repair or adjustment-scale
backfill. Price-only cache upgrades and fresh-tail behavior are preserved.
No new cache namespace, persistent metadata, codec or expiry is introduced.

Young histories still receive the complete requested-period acquisition.
A positive provider first-trade date establishes availability, not immutability:
an older individual price can be corrected while the recent overlap and
listing metadata remain unchanged. Skipping the original full validation
would therefore weaken its freshness contract and is not enabled.

Existing mature-history tail reuse still cannot detect every isolated older
provider correction; it keeps its original overlap-based revision policy.
This fix does not claim otherwise.

Regression tests cover distant endpoints, internal gaps, invalid prices,
future/malformed dates, venue holidays, unknown calendars, young histories
and legacy injected ports. Portfolio release gates compare both cold and
common warm-cache inputs without mail or unrecorded network access.
