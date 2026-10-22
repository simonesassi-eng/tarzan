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

`tarzan.data.bond_history` supplements historical-only holdings that have an
exact BOND kind and no usable price-only tape. It uses the public Borsa Italiana
chart's Euronext Milan history with:

- exact ISIN, exchange, BOND kind and matching native instrument currency;
- five-year daily `closePx`, without adjustment or an added last-trade point;
- completed prior-day observations, valid positive prices, unique dates and
  a daily-cadence check; conflicting rows or coarsely sampled charts are rejected;
- EUR-per-nominal prices, one `/100`, and positive daily FX at or before each
  price date. FX calendar labels are aligned explicitly across timezones;
  neither prices nor FX are backward-filled. Both observations have a
  seven-calendar-day carry limit.

Security and FX observation calendars are aligned independently. A new FX
observation revalues the last known native bond close even when its venue
is closed or no bond trade occurred. The original price date remains attached:
fresh FX cannot turn an overage security close into fresh market evidence.

The loader runs once before valuation, with at most three context-owned
workers and the existing optional-provider budget and publication reserve.
No provider I/O occurs inside the daily price resolver. Complete cached closed
intervals require no new transport; an explicit native no-data receipt is
reused for 24 hours. Timeout, authorization or malformed-response failures are
not cached as proof that market history does not exist.

Raw cache facts retain their real observation dates and receipt timestamps.
Only the strict credential-free namespace is eligible for the existing
authenticated encrypted cache. The publisher-provided anonymous chart access
descriptor is used in memory with fixed-origin requests and no redirects:
no access HTML/header/token enters the cache or frozen market tape.

Required date windows follow documented economic-equivalence groups, but
price authority stays exact-ISIN. A cum/ex closure is not authorization to
borrow another identifier's market prices. Existing market histories, canonical
tickers, current values, orders, tax cost basis and target inputs are not
replaced by this adapter. Unsupported or uncovered intervals retain the
explicit approximation warning and DEGRADED history state.

Incoming securities transfers are valued at market for contributed-capital
and return calculations. Better historical prices can therefore change that
capital, XIRR, TWR, historical P&L and risk even when the bonds are now closed.
This is a correction of the historical measurement, not a current cash loss.
The existing clean-price-plus-booked-coupon convention is unchanged; importing
these closes does not introduce a dirty-price/accrued-interest valuation model.

## Verification

Use the private frozen-provider replay for a code-only KPI comparison, plus
the no-SMTP local verifier for current availability. Prices obtained at
different wall-clock times are not a code-only comparison. A successful test
suite establishes software behavior, not perfect external market data.
The frozen tape records only raw identity/price fields for the official bond
port; unrecorded FX or provider inputs raise rather than falling back to live
network or silently treating a currency as EUR.
