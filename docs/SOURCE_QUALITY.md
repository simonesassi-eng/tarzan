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

FX observations must be finite, positive real scalars. Boolean, complex,
container and overflowing values are rejected before numeric coercion,
inverse-pair arithmetic or merging with a fresh cache tail can erase their
invalid type. Valid real native tapes retain the existing values, dtype,
index and provenance. Direct-pair priority, inverse fallback and usable-cache
fallback remain unchanged; absent FX never becomes a fabricated 1:1 rate.
Security and FX causal alignment, minor-unit scaling and observation-age
limits remain separate requirements, not evidence supplied by this scalar gate.

`tarzan.data.bond_history` supplements historical-only holdings that have an
exact BOND kind and no usable price-only tape. It uses the public Borsa Italiana
chart's Euronext MOT/EuroTLX history with:

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
intervals require no new transport. A missing MOT listing can fall back to the
same exact ISIN on EuroTLX, with independently verified exchange, currency and
daily cadence. Each full packet is acquired at most once per venue and run.
Partial windows from different venues are never spliced into one history.
No-data, timeout, authorization or malformed-response failures are not persisted
as proof that market history does not exist; legacy negative entries are ignored.

Raw cache facts retain their real observation dates and receipt timestamps.
Only the strict credential-free namespace is eligible for the existing
authenticated encrypted cache. The publisher-provided anonymous chart access
descriptor is used in memory with fixed-origin requests and no redirects:
no access HTML/header/token enters the cache or frozen market tape.

Required date windows follow documented economic-equivalence groups, but
price authority stays exact-ISIN unless a separately documented pricing link
authorizes a specific use. A cum/ex closure is not authorization to borrow
another identifier's market prices. Existing market histories, canonical
tickers, current values, orders, tax cost basis and target inputs are not
replaced by this adapter. Unsupported or uncovered intervals retain the
explicit approximation warning and DEGRADED history state.

`tarzan.data.bond_pricing_authority` contains only exact, directional
issuer-documented liquidation-price links. For the explicitly listed BTP Valore
issues, the MEF Information Memorandum names both ISINs and states that selling
the loyalty-coded security converts it to the negotiable code. The latter's
clean price measures liquidation value, not the value of keeping the loyalty
right. Native packets and their encrypted cache retain the source's real ISIN;
the derived price series separately records the target ISIN, native exchange,
policy ID and authority timestamp. A run earlier than the authority verification,
a wrong currency/source/policy, pre-issue dates or maturity dates cannot use it.
These links must never net positions, rewrite orders or tax basis, accrue a
loyalty premium, or be inferred from a name or a neighboring ISIN.

Incoming securities transfers are valued at market for contributed-capital
and return calculations. Better historical prices can therefore change that
capital, XIRR, TWR, historical P&L and risk even when the bonds are now closed.
This is a correction of the historical measurement, not a current cash loss.
The existing clean-price-plus-booked-coupon convention is unchanged; importing
these closes does not introduce a dirty-price/accrued-interest valuation model.

For an order-derived report, the newsletter's State section explicitly states
that Total P&L, TWR and MWR cover the tracked period. Security transfers enter
at market value on the operation date, or the disclosed order-price fallback
when market evidence is unavailable. Recorded carrying prices remain the
cost-basis reference; they are not overwritten by the market contribution.
The disclosure changes no financial calculation and is absent on the
holdings-only path, which cannot establish a tracked-period transfer return.

## Intraday evidence versus drawable charts

One positive, finite observation in the correct session is valid evidence,
even if an illiquid listing has no second trade. A chart's previous-close
reference or independently dated official endpoint can make that observation
drawable, but neither is counted as an additional native bar. A baseline alone
does not establish intraday availability. Invalid prices, conflicting duplicate
timestamps, observations after the capture and previous-session bars at the
start of a new session are rejected.

The existing bounded EUR sibling search and price-coherence guard remain in
force. A fresh qualifying sibling wins over a lagged canonical path; a lagged
same-session cash-market path can otherwise be shown as lagged, without changing
the freshness threshold or extending it to now using an undated quote. A closed
market's last session is labelled separately from a fresh active-market feed.
The audit and private diagnostics retain the real source ticker, actual last
bar instant, native bar count and availability state for every requested symbol,
including missing symbols. These are display facts, not portfolio valuation or
daily-return authority.

Counts of requested/drawable paths include watchlist instruments. Private
diagnostics separately report missing/lagged current holdings. The newsletter
states that chart counts are not portfolio-value coverage. Adding an alternative
intraday provider requires exact instrument/currency/session evidence and real
timestamped observations; a daily OHLC point cannot be relabelled as intraday.

## Verification

Use the private frozen-provider replay for a code-only KPI comparison, plus
the no-SMTP local verifier for current availability. Prices obtained at
different wall-clock times are not a code-only comparison. A successful test
suite establishes software behavior, not perfect external market data.
The frozen tape records only raw identity/price fields for the official bond
port; unrecorded FX or provider inputs raise rather than falling back to live
network or silently treating a currency as EUR.
