# Optional income and cost rate admission

Provider yield and expense fields are optional metadata, not price evidence.
Only finite real scalar rates are admitted. Booleans, complex numbers,
containers, missing/nonfinite values and overflowing conversions do not
become financial rates or interrupt enrichment.

Provider field priority remains unchanged: use the first *valid* positive
rate, rather than the first truthy raw object. A rejected high-priority field
does not hide a valid lower-priority field. Curated taxonomy TER still
overrides the provider value.

The existing provider unit rule remains: positive values below one are
fractions; values at least one are interpreted as percentages and divided
by 100. This is a compatibility heuristic, not proof of an external field's
units. This change does not resolve ambiguity in that heuristic.

The income/cost consumer already receives fractions. It rejects malformed
optional fields independently but never repeats provider percentage
normalization. Finite negative direct fractions retain their existing
meaning. Ordinary floating-point series avoid extension/categorical dtype
failures while retaining row index and summation order.

Missing rates retain the existing zero contribution and total portfolio
weight denominator. A zero aggregate therefore does **not** prove that all
holdings have known zero yield or zero fees. Coverage policy is not changed.

Tests exercise provider fallback, curated priority, enrichment into the native
weighted consumer, invalid scalar domains and pandas nullable/categorical
dtypes. Frozen portfolio comparisons must preserve all KPI, source, plan,
warning, analysis and rendered HTML facts. Synthetic invalid-rate corrections
are expected to differ from the defective old output; they are not evidence
that a real portfolio previously contained invalid metadata.
