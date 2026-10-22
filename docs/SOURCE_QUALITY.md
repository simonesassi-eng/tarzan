# Market-source quality

## Current-session prices

The selector prioritizes coherent observations from the newest observed
session. Venue priority applies within that session and its consensus window;
an older preferred venue, or an undated quote, cannot replace newer dated
evidence. Price-level, independent-baseline and abnormal-move checks remain
in force. No available qualifying observation means retaining the last close,
not manufacturing a point for today.

## Different-root listings

`tarzan.data.listing_aliases` is a deliberately small, immutable catalog.
An explicit alias requires a causal cached OpenFIGI `ID_ISIN` profile that
confirms its ISIN, ETF kind, both ticker roots and one exact share-class FIGI.
It is unavailable before its own verification or in a pinned analysis.
Missing or contradictory authority leaves the existing bounded same-root
candidates unchanged. Alias lookup opens no network connection, and its
memo belongs only to the active run.

Current-session recovery also checks the holding's ISIN, rejecting a
conflicting or absent identifier, and rejects a quote whose explicit currency
contradicts the verified listing. An alias supplements the existing candidate
batch; it does not start unbounded discovery.

The provider mapping key records the actual quote source in `data_source`.
The canonical ticker and its identity-resolution fields remain unchanged:
a validated quote fallback is not a new instrument or a replacement identity.

## Historical bond prices

A trade price is an observation on its own date, not a daily price history.
Carrying it forward remains an explicit approximation. Current-value coverage,
current-session coverage and full historical availability are different
measures; none should be used to imply another.

Any historical replacement needs verified ISIN, currency, observation dates
and price convention. Bond clean prices per 100 nominal must be converted
exactly once, with causal daily FX where required. Price-only history must
remain separate from booked coupons so income is not counted twice.
Manual or licensed data and credentials must not enter public artifacts.

## Verification

Use the private frozen-provider replay for a code-only KPI comparison, plus
the no-SMTP local verifier for current availability. Prices obtained at
different wall-clock times are not a code-only comparison. A successful test
suite establishes software behavior, not perfect external market data.
