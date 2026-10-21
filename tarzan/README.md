# Tarzan — Package Reference

Technical reference for the `tarzan` Python package. For the product overview
and quickstart, see the [root README](../README.md).

## Architecture

```text
tarzan/
├── __init__.py, version.py      # public package and application-version authority
├── main.py                      # CLI and local artifact finalization
├── orchestrator.py              # serialized run lifecycle and financial pipeline
├── contracts/
│   ├── schema.py                # executable input schema/version
│   ├── validation.py            # boundary validation
│   ├── targets.py               # typed target domains and duplicate outcomes
│   ├── report.py                # immutable defensive ReportView
│   ├── reference.py             # prepared immutable display references
│   ├── events.py                # closed event metadata and operational freshness
│   └── exceptions.py            # domain exception hierarchy
├── runtime/
│   ├── session.py               # RunAttemptEnvelope, RunContext, RunSession, FIFO gate
│   ├── ledger.py                # append-only failures, remedies, stages, and evidence
│   ├── summary.py               # immutable VersionedRunSummary shared by entry points
│   ├── provider.py              # quality policy and valuation completeness gate
│   ├── transport.py             # native timeout/retry ports and shared run deadline
│   ├── memo.py                  # per-key single-flight provider results
│   ├── workload.py              # bounded descriptive telemetry and local harness
│   ├── artifacts.py             # atomic local artifact set; checksum manifest last
│   ├── publication.py           # explicit publication decisions
│   ├── effective_orders.py      # immutable on/before-boundary order view
│   ├── report_html.py           # ledger-derived human report
│   ├── data_quality.py          # compatibility projection onto run-owned diagnostics
│   └── audit.py                 # compatibility projection onto run-owned plan evidence
├── instruments/
│   ├── registry.py              # explicit kinds, capabilities, and tracked categories
│   └── valuation.py             # retained type resolution and valuation origin
├── delivery/
│   ├── __init__.py              # publication intent, claim, checkpoint, and SMTP boundary
│   ├── claims.py                # durable purpose-specific claim state machine
│   └── drive_loader.py          # read-only private Drive input adapter
├── data/
│   ├── loader.py                # CSV/XLSX contracts and row-preserving target loading
│   ├── identity_catalog.py      # shared causal target identity catalog
│   ├── enricher.py              # explicit kind/capability-aware enrichment
│   ├── market_quotes.py         # structured market-provider results
│   ├── price_cache.py           # versioned checksummed JSON with atomic replacement
│   ├── public_cache.py          # fixed macro-only public snapshot projection
│   ├── geo_resolver.py          # geographic allocation resolver
│   ├── bond_fetcher.py          # fixed-income price fallback and value math
│   └── proxy_data.py            # configured asset-class simulation proxies
├── engine/
│   ├── allocations.py           # canonical exposure/denominator authority
│   ├── metrics.py               # metric-stage coordinator
│   ├── returns_builder.py       # order-derived value series, XIRR, and TWR
│   ├── rebalancer.py            # deterministic final-action optimizer and proof
│   ├── stats.py, tax.py         # return/risk math and estimated tax
│   ├── benchmarks.py            # benchmark-relative metrics
│   └── robustness.py            # rolling, stress, and bootstrap analysis
├── export/
│   ├── newsletter/              # in-memory HTML newsletter rendering
│   ├── ai_summary.py            # typed Gemini boundary and deterministic Signals fallback
│   └── whatif_excel.py          # local-only What-If workbook rendering
└── tests/                       # deterministic unit/property/integration evidence
```

`RunSession` is the sole owner of analysis state after the process-local FIFO
lease is acquired. Provider-wide rate limiting may remain process-shared as a
transport safeguard, but clocks, configuration, diagnostics, plan audit,
ledger, identity, and memoization are run-owned. Completed output reacquires the
original lease/clock/ledger and uses defensive `ReportView` facts. Display names,
roles, benchmark identities, watchlist and risk bands are prepared during analysis;
rendering never silently re-resolves them from later global taxonomy. Per-key
provider misses coalesce without serializing unrelated network keys. Owned
clients close at the end of analysis and outer completed-output scopes.

## Installation

Run from the repository root with Python 3.12 and an isolated environment.
The root lock contains exact transitive pins and SHA-256 artifact hashes,
including native Drive clients required by the full test suite.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip check
```

`tarzan/requirements.txt` is only a relative include of the root lock. It
supports legacy installation paths, not a different environment. Do not install
only a few floating packages or skip native SDK tests to mask missing dependencies.

## CLI

```bash
python -m tarzan.main --input_orders input/order_list.csv

python -m tarzan.main \
  --input_orders input/order_list.csv \
  --input_config input/targets.csv \
  --input_targets_per_holding input/targets_per_holding.csv \
  --output output/ \
  --strict
```

### Run modes

- `LIVE`: no boundary date; one captured clock and eligible live market/Gemini
  transport.
- `POINT_IN_TIME`: `--as_of YYYY-MM-DD`; one clock/order boundary and no live
  transport.
- `REPRODUCIBLE`: `--deterministic --as_of YYYY-MM-DD`; the effective date is
  mandatory and only replayable/cache evidence is admissible.

```bash
python -m tarzan.main --input_orders input/order_list.csv --as_of 2026-06-30
python -m tarzan.main --input_orders input/order_list.csv \
  --deterministic --as_of 2026-06-30
```

A deterministic Analysis ID covers effective orders/targets/configuration,
policy/schema versions, retained resolution/valuation evidence, every financial
table and full series, and prepared report references—not just the narrow
versioned summary. Attempt ID, wall time, latency, retry delay, diagnostic prose
and workload telemetry do not affect it.

## Inputs and executable contracts

### Order list

The order list is the single financial source of truth. Minimum columns are
case-insensitive:

| Column | Type | Required | Meaning |
|---|---|:---:|---|
| `date` | date | yes | Settlement/value date (`YYYY-MM-DD`) |
| `trade_date` | date | no | Order/market-exposure date; defaults to `date` when absent |
| `type` | string | yes | Buy, sell, transfer, coupon, etc. |
| `isin` | string | yes | Instrument identifier |
| `quantity` | finite number | yes | Units traded |
| `gross_eur` | finite number | yes | Gross EUR amount |
| `net_eur` | finite number | yes | Net EUR cash flow |
| `instrument_kind` | `STOCK`/`ETF`/`BOND`/`CASH` | no | Exact valuation mechanics when provider kind evidence is unavailable |

`instrument_kind` is independent of exposure category: a Fixed Income ETF is
kind `ETF` and uses unit pricing, while an individual `BOND` uses nominal clean
price per 100. Tarzan never selects this mechanic from category, name, ticker,
price, or quantity. If neither the order nor an exact provider declaration
resolves the kind, order-derived valuation is excluded rather than guessed.

Optional aliases/columns are defined in `tarzan/contracts/schema.py`. Lenient
mode records unknown columns; `--strict` rejects them and rejects a partially
invalid ledger rather than committing only its valid rows. Pinned runs build
one immutable effective-order snapshot and exclude every post-boundary order
from holdings, cost, returns, tax, history, targets, and planning.
Invalid economic cash-flow signs reject the entire ledger in both modes; they
are never auto-flipped. Identical legitimate broker fills retain distinct IDs.

### Portfolio and per-holding targets

`targets.csv` uses typed key suffixes:

- `_eur`: absolute EUR amount.
- `_pctg`: percentage-point value.
- `_date`: date string.
- no suffix: boolean value.

Invested class targets are finite nonnegative **notional** exposures; legitimate
individual values and totals above 100% are preserved. Equity-geography values
are a partition. Blank optional values stay absent rather than becoming zero.

`targets_per_holding.csv` joins by canonical ISIN. Source rows are retained.
Every equivalent or conflicting duplicate canonical key emits
`DUPLICATE_TARGET_ROW`, invalidates the complete target set for planning, and
leaves independent analytics available. No duplicate is silently overwritten,
normalized, or discarded.
An immutable `TargetPlan` distinguishes absence from invalid input. Read/parse
failures and unknown protection booleans cannot become an empty valid plan.
All planning is suppressed for an invalid set; target join and seeding share
one collision-aware identity catalog with a historical observation boundary.

## Financial authorities

### Exposure and capabilities

One versioned canonical projection supplies capital/invested denominators,
notional asset-class exposure, geography, sector, optimizer inputs, verifier
inputs, and explanations. Missing required denominators are Unavailable rather
than zero.

Instrument mechanics and exposure category are separate axes. Stock, ETF,
Bond, and Cash kinds explicitly declare identity, pricing/valuation,
history/returns, income, exposure/classification, sector, and rebalancing
capabilities. The seven tracked categories are Equities, Fixed Income, Cash &
Cash Equivalents, Gold, Commodities, Crypto, and Alternative. Unknown,
ambiguous, unsupported, and not-applicable behavior is typed and explained; no
ticker/name/price heuristic or default adapter supplies fabricated output.

Eligible equity sector values reconcile known sectors plus `Unknown` to 100
percentage points against the disclosed non-cash invested-capital denominator.
Unsupported sector capability or a missing denominator renders Unavailable.

### Returns, risk, and planning

Tarzan computes period returns, CAGR, XIRR, TWR, volatility, Sharpe, Sortino,
maximum drawdown, VaR/CVaR, Beta/Alpha, allocation history, and benchmark
comparisons when their evidence is available. Historical price provenance
labels primary, synthetic, carry-flat, and excluded paths rather than treating
them as equivalent.

Rebalancing models initial cash, protected cash, authorized deployable cash,
external contribution, gross sales, taxes, fees, purchases, ending cash, and
residual separately. Search candidates are never instructions. Only rounded,
thresholded final action records that pass exact funding, nonnegative-position,
protected-cash, frozen-position, no-oversell, and canonical post-plan checks are
`EXECUTABLE`; otherwise planning is explicitly non-executable or Unavailable.

## Provider, cache, and valuation evidence

Transport policy in `config/constants.yaml` governs native provider timeouts,
bounded retries and a shared deadline. It does not forcibly kill arbitrary
Python/CPU work. Acceptance policy separately governs freshness, within-kind
materiality-weighted coverage and permitted anchors/stale fallbacks.
Valuation and publication materiality thresholds are independent.

Provider/cache attempts record source, operation, times/age, ordinal, latency,
fallback rung, coverage, outcome, applied policy, and failure/remedy links. A
policy-valid primary cache can be Available; cache selected after a preferred
source failure is Degraded. The disk cache is versioned, checksummed,
non-code-executing JSON committed by synchronized temporary write, flush, and
atomic replacement. Corrupt, incompatible, malicious, or truncated content is
a recorded safe miss and is never executed.
Corporate-action repair requires a declared split or an exact curated event,
never a persistent price loss alone. Provider histories are stored before local
repair; identity profiles/xrefs retain causal observation boundaries. Checksums
detect corruption, not authenticity of the provider's price observations.

The runtime cache is private/ephemeral in Actions and never archived wholesale.
Only schema-validated fixed Markets-universe numeric price tables may enter the
single public macro snapshot. Identity profiles, holdings and manual/licensed
histories are excluded. New private directories/locks use private creation
permissions without chmodding existing caller-owned paths.

A material or indeterminate valuation gap makes the trustworthy total
Unavailable, preserves only a labeled known subtotal, suppresses optimizer and
rebalancing output, and creates critical publication evidence. A non-material
gap remains explicitly Degraded with its own failure record; it is never a
false clean run.
An accepted anchor is valuation evidence, not primary pricing or real market
history. Synthetic/carry-flat series and unresolved independent-source
disagreements are not certified merely because the software tests pass.

## Failure, summary, report, and publication

`RunLedger` is the append-only evidence authority. A failure/fallback record has
a stable identity, original normalized failure, ordered remedies/outcomes,
selected correction/fallback, provenance, correction state, availability,
affected outputs, analytical impact, and publication impact. Successful numeric
zero remains distinct from Unavailable/null.

`SummaryProjector` creates the one immutable, strict-JSON
`VersionedRunSummary` used by CLI and email paths. `report.html` renders the same
ledger lifecycle. Publication is one of:

- `SEND_NORMAL`
- `SEND_DEGRADED_NORMAL`
- `BLOCK_NORMAL_AND_NOTIFY_FAILURE`
- `BLOCK_ALL_DELIVERY`

An uncorrected critical failure never sends the normal newsletter; it creates a
separate sanitized critical-failure notification intent.
Semantic rendering contradictions use `BLOCK_ALL_DELIVERY`. Ledger entries,
failure records and order snapshots cannot be mutated through their projections;
failure projection indexes one consistent entry snapshot rather than rescanning
the whole log per failure.

## Local artifact lifecycle

Every initialized CLI/email attempt finalizes a correlated local set:

```text
output/<effective-or-live-date>/<attempt-id>/
  manifest.json
  summary.json
  ledger.jsonl
  report.html
  newsletter.html       # only when rendered
```

Files are rendered in memory, written to sibling temporary files, flushed,
atomically replaced, and listed in the checksum manifest written last. The
manifest maps Attempt ID to Analysis ID and declares `storage_scope=local`, the
execution environment, automation ephemerality, and
`retention_guarantee=none`. Automation-local artifacts are ephemeral: there is
no upload, remote recovery, or retention promise. The What-If workbook cannot
be attached to email or sent through normal, degraded, or failure publication.
`--what-if` is opt-in and runs after core finalization; its separate set lives
at `output/<date>/what-if/<attempt-id>/`, including its own manifest/report
and optional `what_if.xlsx`. Optional failure leaves the committed core intact.
CLI success requires successful core artifact finalization, not merely a render.

Drive lists all bounded pages, rejects ambiguous names before staging and
validates native size/checksum evidence before publishing private paths.
The staging owner survives the last CSV consumer and releases before SMTP;
cleanup failures remain retryable and privately recorded, never replacing a
primary error or an acknowledged delivery state. Caller-owned inputs are not deleted.
Newsletter text in `.html.j2` templates is autoescaped; prebuilt HTML is an
explicit builder boundary, not a blanket trust of broker/provider content.

## Gemini boundary

The request payload (`build_digest`) carries only typed parsed
portfolio/analysis domain content and explicit summary instructions.
Environment maps, arbitrary paths or files, credentials, authentication
objects, and machine metadata are not inputs. The API key reaches only the
final transport adapter. Keys, prompts, payloads, raw responses, environment
fragments, file contents, and secret-bearing exceptions are not persisted.

Grounded, non-grounded, failed, and deterministic Signals fallback attempts are
recorded as non-secret structured evidence. Gemini transport is prohibited in
`POINT_IN_TIME` and `REPRODUCIBLE` modes.

## Delivery safety

Normal newsletters and critical-failure notifications have different stable
logical identities. The delivery path must:

1. atomically persist a durable purpose-specific claim;
2. checkpoint local evidence;
3. transition to `SMTP_INVOCATION_STARTED` immediately before SMTP;
4. record `ACKNOWLEDGED_SUCCESS` only after definitive return.

Consumed duplicate/conflicting claims suppress SMTP. A duplicate in `CLAIMED`
may proceed only by the same transactional pre-invocation CAS.
A provable pre-invocation failure is
`DEFINITE_PRE_SEND_FAILURE`; an interruption after invocation begins is
`UNCERTAIN` and is never retried automatically. An authorized resend requires a
new audited token and therefore a distinct logical identity.
SMTP acknowledgment does not prove inbox receipt or exactly-once delivery.
Configured remote storage failures and incomplete endpoint/credential pairs
block SMTP without local fallback. With neither setting configured, the
deliberate host-local fallback has no cross-run guarantee on an ephemeral runner.

GitHub automation uses standalone push/PR Checks, native event preflight,
compilation/dependency consistency checks, full-SHA action pins,
hash-enforced Python installation, and one final step scoped to the
Drive, Gemini, claim-service, recipient, and SMTP secrets it consumes. The Apps
Script claim endpoint uses `LockService` plus a separate `delivery_claim:`
Properties namespace and retains hashed control records beyond the declared
reconciliation window.
Actions uses a native bounded queue of at most 100 pending runs. The oldest
request/native-run timestamp authorizes at most a 45-minute delivery window,
rechecked through the last SMTP boundary. Old/invalid requests cannot authorize
send; no punctual-delivery or durable queue SLA is claimed.

Update authorization uses the native Gmail profile and the individual recent
message's `SENT` label plus exact owner From, not a thread-level sent folder.
Unquoted literal commands and bounded per-message markers permit a later reply
in the same thread. Missing native authorization fails closed. GitHub dispatches
contain opaque IDs/times, not subjects or mailbox content.
Automation console formatting exposes fixed status codes while preserving
original records for private run evidence. Historical public logs are not erased.

## Bounded workload evidence

Each terminal run appends versioned descriptive workload counts for orders,
holdings, history points, provider/cache attempts, capability events, stage
outcomes/duration, plan searches, and final actions. JSON-safe recording bounds
and truncation flags are explicit. Telemetry is excluded from Analysis ID and
financial decisions and is not a portfolio-size, latency, throughput, memory,
availability, or unlimited-scale claim.

The deterministic harness exercises explicit kinds/categories without market,
Drive, Gemini, delivery, upload, or other network access:

```bash
python scripts/observe_workload.py
```

## Validation

The complete test suite also requires Node.js for the production Apps Script
harnesses; verification used Node 22. Missing Node/native Google clients is
reported as a failure rather than silently skipped.

```bash
node --version
python -m compileall -q tarzan scripts
python -m pip check
python -m pytest tarzan/tests -q
```

Exact dependency hashes, immutable workflow pins, credential placement, version
authorities and provider/model pins are asserted in
`tarzan/tests/test_delivery_gates.py`, so the test suite above covers them. The
standalone release validator and its manifest are gone: the manifest restated
constants that live in the code, and its positive-scope check validated the shape
of each declared path but never its existence — which is how `tarzan/backtest.py`
stayed declared long after the module became the `tarzan/backtest/` package.
See [the implementation register](../docs/REVIEW_IMPLEMENTATION.md) for the
tested review fixes and [frozen verification](../docs/FROZEN_KPI_VERIFICATION.md)
for private same-evidence KPI comparisons.
