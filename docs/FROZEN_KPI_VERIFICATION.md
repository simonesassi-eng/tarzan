# Private, code-only KPI comparisons

The verification command captures public provider answers at the transport
boundary, not enriched holdings or previously computed metrics. A replay runs
the real loader, resolver, valuation, metrics engine and newsletter renderer
again against those same answers. It performs no SMTP delivery and disables AI.

```sh
python -m tarzan.verification.portfolio capture \
  --snapshot .private/my-review --label baseline --input-dir input
python -m tarzan.verification.portfolio replay \
  --snapshot .private/my-review --label candidate
python -m tarzan.verification.portfolio compare \
  .private/my-review/baseline.json .private/my-review/candidate.json
```

Before editing financial logic, replay the baseline and require no KPI
differences, no semantic-gate errors and no attempted network access. Repeat
the comparison before each commit and record its result. Keep actual amounts,
HTML, inputs and provider snapshots private: `.private/` is ignored by Git.
Snapshot directories are owner-only; new input/output files are owner-only.
Neither authenticated request headers nor credentials belong in a tape.

Inputs are copied and checksummed. Identity-cache evidence is copied separately;
history starts cold to retain the raw prices before any split repair. The
captured run boundary is reused, including cache TTL selection. Advancing the
host date cannot silently expire the copied identity evidence during replay;
the process-wide transport clock is not changed. Tape JSON is non-executable and checksummed.
An unknown provider request or an uncaptured network transport terminates
replay; it must never silently introduce a fresh quote.

The comparison includes total valuation, gross/net P&L, MWR, TWR, period
returns, risk, coverage, allocation, individual positions and rebalancing plans.
Operational IDs and AI prose are not financial KPIs. Tiny floating-point
roundoff is excluded, but zero and Unavailable are different outcomes.

A capture is not an atomic market snapshot: providers answer at different
instants during one run. The promise is that all compared code revisions
consume the **same captured observations**, not that every exchange was sampled
at one instant. Provider outages remain recorded outages. These comparisons
do not certify market prices, fiscal correctness or investment performance.

When completing a preliminary capture with `--seed-tape`, existing price
observations are retained and missing protocol observations are fetched before
the baseline is sealed. Do not compare code changes using different tapes.
Replay wall-clock timings are not live network-performance measurements.

Yahoo HTTP replay uses the installed `curl_cffi` response and error classes.
A recorded 404 must reach the same SDK fallback as a live 404; a generic
`requests.HTTPError` is not an interchangeable substitute. Other provider
ports retain their own response contracts. Missing server reason phrases,
cookies and authentication metadata are not invented or restored.

The default tape also records public SDK-level answers such as `Ticker.info`
and `Ticker.history`. That is useful for code-only portfolio comparisons,
but does not verify a replacement SDK's parsing. An SDK migration additionally
requires replaying recorded raw HTTP through the actual candidate SDK, with
the same input, observations and SDK wall clock, and with live network and
email delivery denied. Neither kind of replay alone establishes live
newsletter latency.
