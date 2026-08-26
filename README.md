# Tarzan

**Portfolio analysis for investors who swing smart.**

Tarzan is a serialized, single-user portfolio analyzer with market-data enrichment, explicit instrument capabilities, reproducible point-in-time modes, and an HTML email newsletter for multi-asset portfolios.

## Features

- **Effective orders** — one immutable order snapshot drives holdings, costs, returns, tax, history, and planning. Pinned runs exclude post-boundary orders.
- **Market evidence** — provider/cache attempts and provenance are recorded with `AVAILABLE`, `DEGRADED`, or `UNAVAILABLE` status. A valid cache selected after a preferred source fails is degraded.
- **Risk and returns** — CAGR, Sharpe, Sortino, drawdown, VaR/CVaR, volatility, Beta/Alpha, XIRR, and TWROR.
- **Canonical exposure** — asset-class, geography, sector, optimization, and verification share one capital/notional projection. Legitimate notional exposure above 100% is preserved.
- **Explicit capabilities** — Stock, ETF, Bond, and Cash mechanics are resolved independently from Equities, Fixed Income, Cash & Cash Equivalents, Gold, Commodities, Crypto, and Alternative exposure categories. Unsupported behavior is Unavailable, never guessed or represented as zero.
- **Rebalancing** — deterministic local-search plans model protected/deployable cash, contributions, sales, tax, and fees. Only rounded final actions that pass exact funding and position proof are executable.

## Quickstart

Tarzan targets Python 3.12. The generated lock pins every transitive dependency and artifact hash.

```bash
python -m pip install --require-hashes -r requirements.txt

python -m tarzan.main \
  --input_orders input/order_list.csv \
  --input_config input/targets.csv \
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
  what_if.xlsx          # optional, local-only
```

The manifest maps the operational Attempt ID to the deterministic Analysis ID and declares `storage_scope=local`, ephemerality, checksums, and `retention_guarantee=none`. Automation-local files are ephemeral; no workflow uploads them or promises remote recovery.

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

`targets.csv` configures cash, allocation, and rebalancing. Invested class targets use a **notional** nonnegative domain and may total above 100%; equity-geography targets are a partition. Blank optional targets remain absent. `targets_per_holding.csv` preserves source rows: every equivalent or conflicting duplicate canonical instrument emits `DUPLICATE_TARGET_ROW`, invalidates the complete target set for planning, and leaves independent analytics available.

## Availability, valuation, and publication

A successful numeric zero is distinct from Unavailable. Every failure/fallback has a stable ledger identity, ordered remedies, selected resolution, provenance, affected outputs, analytical impact, and publication impact.

Provider freshness, coverage, timeout, retry budget, fallback allowance, and valuation materiality are validated from explicit instrument/data-class policy in `tarzan/config/constants.yaml`. Primary, fallback, stale, missing, and unsupported valuation evidence remain labeled. A material or indeterminate gap makes the trustworthy total Unavailable, retains only a labeled known subtotal, suppresses optimization/rebalancing, and produces `BLOCK_NORMAL_AND_NOTIFY_FAILURE`.

Publication decisions are `SEND_NORMAL`, `SEND_DEGRADED_NORMAL`, or `BLOCK_NORMAL_AND_NOTIFY_FAILURE`. Critical runs never send the normal portfolio newsletter; they create a separate sanitized failure-notification intent.

## Gemini boundary

Gemini receives only typed portfolio/analysis domain content and explicit instructions. The API credential is passed only to transport. Keys, prompts, payloads, environment data, arbitrary files, raw provider responses, and secret-bearing exceptions are never persisted. Grounded, non-grounded, failed, and deterministic Signals fallback attempts are recorded without payload content. Gemini is never called in `POINT_IN_TIME` or `REPRODUCIBLE` mode.

## Delivery safety

Each normal or critical-failure intent has a purpose-specific stable logical identity. The final publication step must create a durable transactional claim before SMTP, checkpoint local evidence, and mark `SMTP_INVOCATION_STARTED` immediately before `send_message`. Duplicate claims suppress SMTP. A post-invocation interruption becomes `UNCERTAIN` and is never retried automatically. A human-authorized resend requires a new audited token and therefore a new identity.

GitHub validation is credential-free and must pass before the final publication step receives Drive, Gemini, claim-service, recipient, or SMTP secrets. Workflow actions use full commit SHAs; Python installation uses `--require-hashes`. Pins are refreshed for a reason — an applicable advisory or a dependency need — never on a clock that can expire between two digests. There is no release manifest and no declaration audit: the properties that mattered (credentials confined to the final publication step, action SHA pinning, dependency hash locking, one version authority) are assertions in the test suite, so they hold wherever the suite runs. What gates publication is behaviour — compile, `pip check`, and the full test suite. The `Checks` workflow runs the same behavioural gate on every push so a broken commit costs a red tick immediately; it is not a precondition for delivery.

## Development and evidence

```bash
python -m compileall -q tarzan scripts
python -m pytest tarzan/tests -q
```

The ledger records bounded descriptive workload observations (orders, holdings, history points, provider/cache attempts, capabilities, stage outcomes/duration, and plan-search counts). They describe exercised fixtures; they are not a portfolio-size, latency, throughput, availability, or unlimited-scale claim.

See [`tarzan/README.md`](tarzan/README.md) for package details.

## License

Personal project. All rights reserved.
