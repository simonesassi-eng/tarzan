# Causal daily history

Daily Yahoo data and retained cache rows are admitted before raw-request memo,
refresh-window selection, adjustment repair, source selection, valuation and FX
arithmetic. A positive price with a future date is not price evidence. `NaT`
cannot become the selected newest close. Direct and inverse FX retain their
existing source order; rejecting an invalid direct response does not suppress
the inverse source.

LIVE admission uses receipt time, not the run's earlier capture time. A real
trade acquired after capture remains usable. Timezone-aware dates are read on
their own clock; naive dates use an explicitly known venue, otherwise UTC.
Pinned daily histories retain their venue-local midnight date boundary.
Non-datetime injected ports retain their explicit undated contract.

## Overnight session labels

A daily currency or futures bar can legitimately have tomorrow's session
label. An exception requires the pinned native SDK's already-received chart:
matching symbol/kind, timezone, daily granularity, an overnight regular trading
period, and an actual regular trade inside that period and no later than
receipt. No lazy `history_metadata` or `tradingPeriods` property is accessed.
There is no blanket extra-day grace period.

Only bounded, validated primitive session evidence is persisted alongside such
rows. Its original trade time survives a cache round trip; a fresh read cannot
rejuvenate it. Fresh overlapping prices replace overlapping evidence too.
FX composite prices use the older real price/FX observation, so a future FX
label cannot be hidden behind an older quote.

Raw market and total-return columns are filtered together. Future split rows
cannot revise earlier prices. Provider cache precedes synthetic sibling fills;
those fills remain run-local and are not persisted as native prices.

This change does not improve the external provider's accuracy, remove required
venues, lower throttling, or claim a newsletter latency saving. Healthy frozen
data must retain all financial results, plans, provenance, warnings and rendered
HTML. Synthetic invalid data intentionally changes to the available causal
price or the existing explicit fallback, rather than the invalid future price.
