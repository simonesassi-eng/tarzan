# Yahoo pacing observations

An existing native run owner now records the shared Yahoo pacer's calls,
completions and interruptions, time waiting for its lock, actual sleep
intervals, and requested sleep intervals. The configured minimum interval
remains 0.2 seconds. Provider retry, timeout, priority and request-capacity
policies do not change.

These are sums of caller intervals, not delivery or critical-path durations.
Parallel lock waiters overlap. Do not add their accumulated waits to an
analysis stage or subtract them from newsletter delivery to claim a speedup.
HTTP attempts and admission continue to have their own separate observations.

The observer runs after releasing the pacing lock and holds no HTTP permit.
Recording and snapshots are nonblocking; contention or invalid/failed optional
observations are explicitly incomplete, never a manufactured zero-wait proof.
Incomplete sampling is separate from exceeded recording bounds: it does not
relabel the independent native acquisition observations as truncated.
Interoperable recording bounds are descriptive, not supported workload limits.
No request, ticker, financial value, credential or exception text is recorded.

Only an existing native owned budget is observed. Direct/unowned callers do
not gain a budget, session, request, or new transport authority. Original
pacer/provider errors and caller cancellation retain their identity.

The additive `yahoo_spacing` workload evidence does not participate in
financial decisions or Analysis ID. It is a diagnostic prerequisite for
measuring further runtime changes, not itself a latency optimization.
