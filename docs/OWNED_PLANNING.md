# Owned native planning preparation

The LIVE order-based engine can prepare both unchanged native rebalancing
searches while its existing historical-instrument and benchmark tasks run.
This is CPU overlap, not fewer provider requests or a different optimizer.

Only inspected native engine/solver ports and exact native input types qualify.
Pinned runs, unowned callers, custom engine implementations and injected
solver ports keep their established serial behavior. There is one additional
owned spawn process, no additional native HTTP permit and no worker past join.
The process receives builtins-only typed facts, has network/email disabled
and removes credential environment variables. It does not inherit model
pickle/reduction hooks, transport, ledger or audit state.
Known HTTP/socket/email ports are rejected before calculation; this is an
application-level guard, not an operating-system network sandbox.

Preparation copies only consumed current facts, configuration and seeds.
It does not copy historical tapes, dynamic external attributes or arbitrary
copy hooks. Typed keys retain holding/mapping order, float bits, every
configuration field, native search defaults and mechanics/tax selectors.
An input or dependency change rejects the prepared result.

The original `_rebalancing` stage remains planning authority. Target,
historical mechanics, valuation and classification gates run before either
result is adopted. Cost calculation and the two audit records remain there.
Preparation never publishes a plan or makes a portfolio unavailable.
Ordinary preparation/setup failures fall back to the original searches.
Cancellation drains accepted work and preserves owner/source precedence.

`metrics:_prepare_rebalancing` records the preparation's wall, caller-thread
and inclusive parent-process CPU, plus separate child compute/CPU observations.
The ordinary rebalancing stage measures adoption,
cost and audit. These overlap with source intervals: do not add their wall
or process CPU durations. Recorded-response replay is not SMTP delivery or
physical HTTP/authentication replay, and controlled source waits are not
measurements of production provider latency.
