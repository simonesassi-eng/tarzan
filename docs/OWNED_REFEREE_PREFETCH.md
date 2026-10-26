# Owned early referee preparation

The current stamp consumes dated justETF quotes, whereas later reconciliation
consumes the long series. Native LIVE preparation submits both at the original
early epoch, in the original series-before-quotes order and six-worker pool.
It does not defer required histories until the later consumer.

The quote consumer joins only quotes. Until that join, requests retain the
established essential FIFO; slow series parsing cannot reserve otherwise free
native slots against those same early quotes. The owner then protects remaining
submitted series, including retry backoff, against work in subsequent phases.
This reservation holds no native permit and never expands the aggregate cap.
Already-running requests are counted once, not double-reserved.

Independent order history, benchmarks and native CPU planning may proceed.
Before their financial adoption, the owner joins the earlier referee work.
The later long-series consumer reuses its session memo. Successful and failed
analysis exits also drain accepted work before transport closure; nested scopes
reuse the owner rather than abandoning or replacing outstanding callbacks.

Cancellation from earlier referee work precedes later worker cancellation;
an actual interruption of the owner's wait or submission remains primary.
An ordinary series fault cannot invalidate a separately valid, dated current
quote. It does not certify that a long series is available: its consumer retains
the existing absence, validation and fallback policies.

Pinned/unowned runs, replaced referee/parser entry ports, alternative transport
owners and ambient optional work retain the original prefetch contract.
Listing searches, identity, currency, price rulers, source priority, provider
retry/authentication and financial formulas are not pruned or changed.

Admission observations contain only bounded numeric counts, including actual
reservation activations and remaining protected work. They are descriptive,
not financial authority or a claim that a particular run saved time.
Frozen native-parser comparison is separate from physical HTTP and delivery
latency; synthetic waits are not live-provider measurements.
