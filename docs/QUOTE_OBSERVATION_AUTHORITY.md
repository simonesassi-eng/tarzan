# Quote observation authority

A current regular quote needs a finite positive price and an admissible
observation instant. An explicitly malformed or future timestamp does not
become an undated quote by discarding its clock.

The same parsing and instant rule govern metadata candidate admission,
listing-market evidence, final enrichment price selection, listing freshness,
current-session quote choice and series stamping. Unix epochs and accepted
ISO/datetime metadata representations share an aware UTC clock. Invalid
numeric domains and unrepresentable timestamps cannot acquire price authority.
Relative strings such as `now`/`today` cannot manufacture a source observation
from the machine's receipt clock.

In live runs, the upper bound is the instant of evaluation/receipt, not the
run's capture instant: a legitimate trade may arrive while acquisition is in
progress. The established captured-calendar age rule remains. Pinned modes
use the effective UTC day's end; final pinned valuation still excludes all
current quotes and clips history to its existing reporting boundary.

A genuinely missing timestamp retains the explicit undated-fallback contract.
The provider's independent previous close also remains fallback-only.
Rejecting an invalid regular quote does not discard genuine history or reset
its observation time, cache origin, currency/unit ruler or carry age.
An older FX observation cannot conceal a future native quote.

No source priority, configured exchange discovery, authentication recovery,
network request, provider pacing, global capacity or deadline is removed.
Future or malformed evidence may legitimately yield a different selected
source or fallback on an adversarial feed; unchanged ordinary observations
retain their financial results.

Tests exercise the actual installed native SDK parsers over independent
synthetic curl responses, positive/missing/invalid clock controls, alternate
venue recovery, receipt-versus-capture timing, timezone/calendar boundaries,
foreign conversion and pinned-mode clipping. Differential financial gates
require identical inputs, receipt schedule, initial cache and recorded
observations, including full KPI/plan/quality/source/HTML equivalence.
Recorded-response replay does not prove physical HTTP/auth-failure behavior
or the accuracy of external market sources.
