# Tarzan

**Portfolio analysis for investors who swing smart.**

Tarzan is a serialized, single-user portfolio analyzer with market-data enrichment, explicit instrument capabilities, reproducible point-in-time modes, and an HTML email newsletter for multi-asset portfolios.

## Features

- **Effective orders** — one immutable order snapshot drives holdings, costs, returns, tax, history, and planning. Pinned runs exclude post-boundary orders.
- **Market evidence** — provider/cache attempts and provenance are recorded with `AVAILABLE`, `DEGRADED`, or `UNAVAILABLE` status. A valid cache selected after a preferred source fails is degraded.
- **Risk and returns** — CAGR, Sharpe, Sortino, drawdown, VaR/CVaR, volatility, Beta/Alpha, XIRR, and TWR.
- **Second source** — eligible instrument returns are checked against available independent sources. Corroborated corrections are labeled; disagreement without corroboration remains flagged, not silently “fixed”.
- **Canonical exposure** — asset-class, geography, sector, optimization, and verification share one capital/notional projection. Legitimate notional exposure above 100% is preserved.
- **Explicit capabilities** — Stock, ETF, Bond, and Cash mechanics are resolved independently from Equities, Fixed Income, Cash & Cash Equivalents, Gold, Commodities, Crypto, and Alternative exposure categories. Unsupported behavior is Unavailable, never guessed or represented as zero.
- **Rebalancing** — deterministic local-search plans model protected/deployable cash, contributions, sales, tax, and fees. Only rounded final actions that pass exact funding and position proof are executable.

## Quickstart

Run from the repository root with Python 3.12. Use an isolated environment;
`requirements.txt` is the sole supported lock, including native Google clients.
The legacy `tarzan/requirements.txt` delegates to it, not to floating versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip check

python -m tarzan.main \
  --input_orders input/order_list.csv \
  --input_config input/targets.csv \
  --input_targets_per_holding input/targets_per_holding.csv \
  --output output/
```

Each initialized analysis finalizes a correlated local artifact set under:

```text
output/<effective-or-live-date>/<attempt-id>/
  manifest.json
  summary.json
  ledger.jsonl
  report.html
  newsletter.html       # when normal/degraded rendering is allowed
```

The manifest maps the operational Attempt ID to the deterministic Analysis ID and declares `storage_scope=local`, ephemerality, checksums, and `retention_guarantee=none`. Automation-local files remain ephemeral. The newsletter workflow separately retains an authenticated **encrypted** diagnostic projection for 14 days; it never uploads this plaintext artifact set or raw SDK logs. Recovery requires the dedicated snapshot key.

Optional candidate backtests/workbooks require `--what-if`. They run after the
core manifest commits, under `output/<date>/what-if/<attempt-id>/`, and cannot
overwrite or invalidate the core newsletter. Manual/licensed proxy ingestion
is an explicit local CLI workflow, never an automatic newsletter dependency.

## Run modes

- `LIVE` — captured current clock and eligible live market/Gemini transport.
- `POINT_IN_TIME` — `--as_of YYYY-MM-DD`; one effective boundary controls clocks and orders, and live transport is disabled.
- `REPRODUCIBLE` — `--deterministic --as_of YYYY-MM-DD`; requires the date and uses only replayable/cache evidence. Live market and Gemini transport are disabled.

```bash
python -m tarzan.main --input_orders input/order_list.csv --as_of 2026-06-30
python -m tarzan.main --input_orders input/order_list.csv --deterministic --as_of 2026-06-30
```

## Inputs

The order list is the single source of truth. Minimum case-insensitive columns:

| Column | Type | Required | Description |
|---|---|:---:|---|
| `date` | date | yes | Order date (`YYYY-MM-DD`) |
| `type` | string | yes | buy, sell, transfer, coupon, etc. |
| `isin` | string | yes | ISIN identifier |
| `quantity` | float | yes | Units traded |
| `gross_eur` | float | yes | Gross EUR amount |
| `net_eur` | float | yes | Net EUR cash flow |

Optional aliases and columns are defined by the executable contract in `tarzan/contracts/`. `--strict` rejects unknown columns; lenient mode records them.
Economic cash-sign errors reject the complete input in both modes; signs are
never auto-flipped. Equal legitimate broker fills remain separate orders.

`targets.csv` configures cash, allocation, and rebalancing. Invested class targets use a **notional** nonnegative domain and may total above 100%; equity-geography targets are a partition. Blank optional targets remain absent. `targets_per_holding.csv` preserves source rows: every equivalent or conflicting duplicate canonical instrument emits `DUPLICATE_TARGET_ROW`, invalidates the complete target set for planning, and leaves independent analytics available.
Absent, valid and invalid target plans are distinct. An unreadable present file
or an unknown protection boolean suppresses planning rather than removing a
constraint or creating an empty executable plan.

## Availability, valuation, and publication

A successful numeric zero is distinct from Unavailable. Every failure/fallback has a stable ledger identity, ordered remedies, selected resolution, provenance, affected outputs, analytical impact, and publication impact.

Provider transport policies in `tarzan/config/constants.yaml` enforce native
I/O timeouts, bounded retries and a shared run deadline. They are separate from
instrument/data-class acceptance policies for freshness, coverage and permitted
fallbacks. The deadline does not forcibly terminate arbitrary Python/CPU work.
Portfolio history (including closed positions) and required risk/geography
references precede optional watchlist discovery. Optional native work has its
own deadline and leaves a configured reserve for final evidence/publication.
Yahoo batches retain bounded per-symbol concurrency and the native run owner.

Primary, fallback, stale, missing and unsupported valuations remain labeled.
An anchor cannot become primary because display taxonomy changes. Accepted
valuation coverage is not market-history coverage: synthetic/carry-flat bond
series and unavailable independent prices remain explicit limitations.
Material or indeterminate valuation gaps withhold the trustworthy total and
planning, retaining only a labeled known subtotal.

Publication has an independent materiality policy. Decisions include
`SEND_NORMAL`, `SEND_DEGRADED_NORMAL`, `BLOCK_NORMAL_AND_NOTIFY_FAILURE`
and `BLOCK_ALL_DELIVERY`. Critical runs do not send the normal newsletter;
rendered semantic contradictions block SMTP entirely.

## Gemini boundary

Gemini receives only typed portfolio/analysis domain content and explicit instructions. The API credential is passed only to transport. Keys, prompts, payloads, environment data, arbitrary files, raw provider responses, and secret-bearing exceptions are never persisted. Grounded, non-grounded, failed, and deterministic Signals fallback attempts are recorded without payload content. Gemini is never called in `POINT_IN_TIME` or `REPRODUCIBLE` mode.

## Delivery safety

Each normal or failure-notification intent has a purpose-specific stable
identity. Delivery claims before SMTP, checkpoints local evidence and consumes
the claim by compare-and-swap immediately before `send_message`. A duplicate
still in `CLAIMED` may proceed only through that same CAS; consumed duplicates
and conflicts suppress SMTP. `UNCERTAIN` never authorizes an automatic resend.
A human-authorized resend requires a new audited identity.

Configured remote claim storage coordinates separate runners and fails closed
when unavailable. With neither endpoint nor credential configured, the
deliberate fallback is host-local: an ephemeral runner then has no cross-run
deduplication guarantee. Partial remote configuration rejects delivery. SMTP
acknowledgment means server acceptance, not inbox receipt or exactly-once delivery.

Apps Script owns the schedule and authenticated per-message Update requests.
Actions uses a bounded native queue and checks the oldest request/run age
through the final SMTP boundary; a request older than 45 minutes is skipped.
This is not a timing/receipt SLA. One fixed macro-price projection is publicly
cached. Queried market histories/identities can persist **only as authenticated
ciphertext** under `TARZAN_CACHE_KEY_B64`, never as a public plaintext directory.
Inputs, claims, SDK cookies and manual/licensed histories are excluded.
Restoration preserves original timestamps; it does not make stale data fresh.
Unique cache keys per attempt persist learned mappings even after a same-day hit.
Public console output contains fixed status codes, not financial prose or
tracebacks. Older public logs/history do not become private retroactively.

The standalone `Checks` workflow runs the full suite on pushes/pull requests.
Newsletter preflight validates native event metadata, compilation and installed
dependency consistency, without Drive/Gemini/claim/recipient/SMTP credentials.
Mail/Gemini/claim secrets are scoped to the publication step. The explicit
`verify_only` mode reads Drive inputs and renders with no mail/recipient/claim
credentials, no AI, no SMTP and no delivery claim. The snapshot key is bound
only to native consumers, not external actions. Actions are SHA-pinned;
installation enforces package hashes. A failed `Checks` run adds a notice rather
than itself withholding a digest; operational freshness, financial/semantic
gates, claims and SMTP can still block it. Dependency pins do not expire merely
because a calendar date passes.

See [automation setup](docs/setup-newsletter.md) for the actual schedule,
permissions and the separate Apps Script update/deployment steps.
See [newsletter recovery and no-email verification](docs/NEWSLETTER_RESILIENCE.md)
for encrypted diagnostics, key recovery and cold/warm checks.

## Development and evidence

The full suite also requires Node.js for the Apps Script harnesses (verified
with Node 22). Missing Node or native Google dependencies is a test failure,
not a reason to silently skip those boundaries.

```bash
node --version
python -m compileall -q tarzan scripts
python -m pip check
python -m pytest tarzan/tests -q
```

The ledger records bounded descriptive workload observations (orders, holdings, history points, provider/cache attempts, capabilities, stage outcomes/duration, and plan-search counts). They describe exercised fixtures; they are not a portfolio-size, latency, throughput, availability, or unlimited-scale claim.
See [the implementation register](docs/REVIEW_IMPLEMENTATION.md) and
[frozen KPI verification](docs/FROZEN_KPI_VERIFICATION.md). A before/after
financial comparison uses the same inputs, observations and time boundary;
two separate live runs are not evidence of a code-induced return.

See [`tarzan/README.md`](tarzan/README.md) for package details.

## License

Personal project. All rights reserved.
