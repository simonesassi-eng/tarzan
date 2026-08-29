# Tarzan — Data Model Hardening & Global-Launch Readiness

**Status:** DRAFT for review · **Author:** (eng) · **Reviewers:** (Principal SDE)
**Target:** public/global launch in ~8 weeks · **Scope of this doc:** the data
structures, plus the launch-blocking work those structures must support
(multi-tenancy, privacy, correctness guarantees).

**Progress:** Track A (data-structure integrity: A1 order PK, A2 canonical
instrument key, A3 order→taxonomy referential check, A5 metrics
schema_version, A4-safe Holding guard, golden-master gate) is DONE and merged
to main. Track A′ (contracts — below) is DONE on branch
`feat/contracts-input-schema`. Tracks B (multi-tenancy) and C (privacy/GDPR)
remain the launch gate.

---

## 0. TL;DR (read this first)

Tarzan today is a **single-user, stateless, in-memory analytics pipeline**: it
reads one user's `order_list.csv` fresh each run, derives everything from it,
and emails a report. The engine is genuinely well-designed — a ledger-as-
source-of-truth star schema with real domain enums. But it is **not shaped for
a multi-tenant public product**, and three classes of gap are launch-blocking:

1. **Identity & integrity** — no primary key on the order ledger, ambiguous
   ISIN/ticker instrument keying, no referential check that an order's
   instrument is known. These allow *silent wrong numbers*.
2. **Multi-tenancy** — one Drive folder, one recipient, one set of secrets.
   There is no concept of a "user" anywhere in the code.
3. **Privacy/compliance** — order lists are financial PII; the target market
   is the EU (GDPR). Today PII flows through one person's Gmail + service
   account with no data-handling contract.

This doc proposes a phased plan. **The honest headline: a *correctness-and-
privacy-safe* launch in 8 weeks is achievable; a "no further changes needed to
scale to millions" architecture is not.** I recommend we scope the 8 weeks to
"safe for a bounded invite-only cohort in the EU," and stage the
horizontal-scale work behind that.

---

## 1. Current-state data model (accurate)

### Entities ("tables")

| Entity | Kind | Notes |
|---|---|---|
| `Order` (`models/order.py`) | append-only **fact ledger** | read fresh per run; no persistence; no key |
| `Holding` (`models/holding.py`) | **derived snapshot** | computed from Orders; ~30 nullable enriched fields |
| `instrument_taxonomy.csv` | **dimension** | 60 rows; keyed by ISIN *or* ticker; wide (repeating geo/exposure groups) |
| `InvestorConfig` (`models/investor_config.py`) | parameters | targets, tolerances, tax rates |
| `PortfolioMetrics` (`models/portfolio.py`) | **denormalized report cube** | ~50 fields; mutable; unversioned |
| `AssetClass`, `Geography`, `OrderType` | **domains** (enums) | good — real constraints |

### Data-flow (star schema, correctly shaped)
```
order_list.csv ──► [Order]*  (fact ledger, single source of truth)
                      │
     instrument_taxonomy.csv (dimension) ─┐
                      ▼                    ▼
              build_holdings_from_orders ──► [Holding]* (derived)
                      ▼
                MetricsEngine ──► PortfolioMetrics (report cube) ──► Excel / newsletter / report.html
```
This shape is a strength and should be preserved.

### Threat model correction
Orders are **not** accumulated across runs (verified: `load_orders` reads the
file fresh; there is no DB/append store). So integrity risks are *intra-file*
(duplicate rows, malformed rows, unknown instruments), not cross-run drift.

---

## 2. Findings (RDBMS lens) with severity

Severity = (financial-risk × likelihood at scale). L/M/S = fix effort.

| # | Finding | Principle | Sev | Effort |
|---|---|---|:--:|:--:|
| F1 | `Order` has **no primary key**; identical rows are indistinguishable → a duplicated block in an input file is silently double-counted | Entity integrity | **High** | S |
| F2 | **Ambiguous instrument key** (ISIN-else-ticker) repeated across loader/orchestrator/taxonomy/newsletter | Keys | **High** | M |
| F3 | **No referential integrity** order→taxonomy; unknown ISIN silently defaults (class=Alternative, geo=USA) | Referential integrity | **High** | S |
| F4 | **Money is `float`**, no money domain; sub-cent drift + NaN/Inf risk | Type/domain integrity | Med | L |
| F5 | `Holding` **conflates concerns** (input + enriched + target + rebalancer scratch) and is ~all-nullable; required-vs-enriched split lives in comments, not types | Normalization / NOT NULL | Med | M |
| F6 | `PortfolioMetrics` is **mutable + unversioned**; field rename silently breaks consumers | View schema stability | Med | S |
| F7 | Taxonomy CSV violates 1NF (repeating geo/exposure groups) | Normalization | **Low (WONTFIX)** | — |
| F8 | `class_breakdown`/`geo_breakdown` are embedded 1-to-many maps | Normalization | **Low (WONTFIX)** | — |

**Explicit WONTFIX (a Principal will want these justified, not silently
skipped):**
- **F7** — the taxonomy is a *hand-edited config dimension*. A wide row is far
  more maintainable than 3 joined child tables; the column set is fixed and
  small. Denormalization is the correct engineering choice here. (If the
  taxonomy ever becomes user-editable at scale, revisit → move to a real
  `instruments` table with `instrument_geo`/`instrument_exposure` children.)
- **F8** — embedded maps are ergonomic for the in-memory pipeline; splitting
  them buys nothing until data lives in a DB.

---

## 3. Proposed changes

### Track A — Data-structure integrity (the RDBMS work)

**A1. Primary key on `Order` + intra-file dedup guard (F1).**
- Add `order_id: str` — a deterministic hash of the natural key
  (`trade_date, isin, type, quantity, net_eur, source`) plus a disambiguating
  ordinal for genuine same-key repeats.
- In `load_orders`, detect *exact* duplicate rows (same natural key AND same
  ordinal position collision) → keep, but emit a data-quality WARNING with the
  count, so a double-pasted block is visible, not silent. (We do NOT auto-drop:
  two identical buys can be legitimate. We *surface*.)
- Rationale: entity integrity + the ledger stays append-only-honest.

**A2. Canonical instrument key + single resolver (F2).**
- Introduce `InstrumentKey` = normalized ISIN when present, else
  `TICKER:<upper>` sentinel. One `resolve_instrument(isin, ticker)` function;
  delete the scattered "ISIN first else ticker" branches
  (loader, orchestrator `_apply_per_holding_targets`, taxonomy lookups,
  newsletter resolution).
- Taxonomy: add a stable `key` column derived the same way, so lookups are a
  single dict hit, not a two-step fallback.

**A3. Referential-integrity check order→taxonomy (F3).**
- After load, compute the set of order ISINs with no taxonomy row AND no
  successful enrichment classification; emit a data-quality **WARNING** listing
  them ("3 instruments have no taxonomy/classification; defaulted to
  Alternative — verify"). Infrastructure already exists (`data_quality`).
- This turns a silent FK violation into a visible one.

**A4. Split `Holding` into typed lifecycle stages (F5).**
- `RawHolding` (from orders: isin, ticker, qty, cost, currency — all NOT NULL)
  → `EnrichedHolding` (adds market fields) → rebalancer reads a separate
  `RebalanceTarget` object rather than scratch fields on the holding.
- Freeze each stage (`frozen=True`) so a computed snapshot can't be mutated
  mid-pipeline. (This also closes the "immutable-ish" comment-vs-reality gap.)
- **Cost:** touches enricher + metrics + rebalancer signatures. Medium, and
  behavior-preserving (verify via golden metrics diff).

**A5. Version + freeze `PortfolioMetrics` (F6).**
- Add `schema_version: int`; make it `frozen`. Consumers assert the version.
  Cheap insurance against a silent field-rename breaking the newsletter/Excel.

**A6. Money domain (F4). — DEFERRED (evidence-based, 2026-07-13).**
- Would introduce a `Money` type (integer minor-units or `Decimal`) at the
  ledger boundary; keep `float` only inside vectorized numpy risk math (where
  Decimal is impractical), converting explicitly at the seam.
- **Measured before deciding** (real 144-order portfolio, on `main`):
  - `sum(net_eur)`: float vs exact-Decimal drift ≈ **4.4e-11 EUR** (~0.00000000004 ¢)
  - `sum(cost_basis)`: drift **0.0**
  So the reported `total_value` is already correct to the cent;
  the float error is ~11 decimal places down and rounds away at every display
  precision. Reason it's so small: Tarzan reads pre-computed EUR values from
  the CSV and mostly *adds* a few hundred numbers — float drift only reaches
  cent scale with millions of ops or long multiply/divide chains, not this
  workload.
- **Decision:** DEFER. This is L effort touching ~8 modules (loader, order,
  holding, returns, tax, rebalancer, export) with real regression risk, for a
  benefit that is currently unobservable. The Phase-1 NaN/Inf guards + output
  sanitization already eliminated the *acute* float dangers. Rewriting a
  money-critical hot path for a 1e-11 EUR gain would more likely introduce
  bugs than fix them. Revisit only if the workload changes materially
  (tick-level data, per-tenant aggregation over thousands of accounts). The
  golden-master test is the safety net for doing it later.

### Track A′ — Contracts (input schema / validation / output DTO) — DONE

The boundary half of the data work: with the *structures* sound, the
*contracts around them* were still shaped for one expert user. Implemented:

- **A′1 — Explicit, versioned input schema** (`tarzan/contracts/schema.py`): a
  dependency-free declarative `FileSchema`/`ColumnSpec` for `order_list.csv`
  and `targets_per_holding.csv` — one source of truth for the format, with a
  `SCHEMA_VERSION` and self-documenting `to_markdown()`. The loader's required
  columns now derive from it. (No pydantic on purpose — the CI newsletter
  runner needs no extra install.)
- **A′2 — Boundary validation with a strict gate** (`validate_columns`):
  missing-required is always fatal with an actionable message; an unknown
  column WARNS in the default *lenient* mode (recorded in the data-quality
  report) and is REJECTED in `--strict` / `TARZAN_STRICT_INPUT=1`. Strict
  rejections propagate (no silent degrade to "no orders"). Default OFF keeps
  the automated newsletter and existing files unaffected — the multi-tenant
  onboarding (Track B) should run strict.
- **A′3 — Pinned external output DTO**: `to_summary_dict` is documented as the
  narrow, versioned external surface (`SUMMARY_CONTRACT_KEYS` + optional
  order-path keys), distinct from the wide internal cube, with a test pinning
  the exact key set. This is the seam a future API/mobile client binds to.

**Deferred within A′** (needs Track B or external consumers to be worth it):
a formal OpenAPI/JSON-Schema contract at a web edge, and content-addressed
input files. No value before there is an API.

### Track B — Multi-tenancy (launch-blocking for "friends/global")

**B1. First-class `User`/`Tenant` entity.**
- A `tenants` registry (start as a private config file / small table):
  `tenant_id → {drive_subfolder, recipient_email, locale, created_at, consent_ts}`.
- Per-tenant Drive subfolders (`/tarzan/<tenant_id>/order_list.csv`), never a
  shared folder.
- `tarzan.delivery.run_and_send` becomes `for tenant in tenants: run(tenant)`
  (the delivery logic already lives in the package, not the CI shim); the Apps
  Script dispatches one job per tenant (tenant_id in payload).

**B2. Tenant isolation invariant.**
- Every run is scoped to exactly one tenant; no global mutable state may carry
  a tenant's data into another's run. Today's process-global collectors
  (`data_quality`, `audit`, `runtime`) already `reset()` per run — audit that
  this holds under a multi-tenant loop (reset between tenants), add a test.
- Cache is tenant-agnostic (market data only) — safe to share, already verified.

**B3. Concurrency & scale reality.**
- 8-week target: a **sequential loop over a bounded cohort** (tens of users) in
  CI is fine. This does NOT scale to thousands (GitHub Actions minutes, 12-min
  job cap, one service account). Horizontal scale = a real backend (queue +
  workers + object storage + a DB for tenants/consent). **Explicitly staged
  post-launch**; see §5.

### Track C — Privacy / compliance (launch-blocking in the EU)

**C1. Data classification & minimization.** Mark order lists / emails as PII;
document retention (how long inputs/outputs live in Drive/logs), and scrub PII
from `report.html`/logs shared outside the tenant.

**C2. Consent & lawful basis (GDPR).** A recorded consent timestamp per tenant;
a documented data-processing purpose; a deletion path ("forget me" → purge
Drive subfolder + caches + any stored outputs).

**C3. Secrets & trust boundary.** Today all tenants' data flows through one
Gmail + one service account. For "friends" this is acceptable *with disclosure*;
for "global" it is not — it needs per-tenant credential scoping or a backend
that never routes one user's data through another's infrastructure.

---

## 4. Phased roadmap (8 weeks)

Each phase is independently shippable, behavior-preserving where possible, and
verified (golden metrics diff + full suite + end-to-end), per the discipline
already used on Phases 1–4.

- **Wk 1–2 — Integrity (Track A1–A3, A5).** Order PK + dedup surfacing,
  canonical instrument key + resolver, order→taxonomy referential check,
  versioned/frozen `PortfolioMetrics`. *Low risk, high value, no UX change.*
- **Wk 3–4 — Holding lifecycle split (A4)** + a **golden-master test corpus**
  (a committed sample tenant whose full metrics are pinned) so all later
  refactors are provably safe.
- **Wk 4–6 — Multi-tenancy (Track B1–B2).** Tenant registry, per-tenant Drive
  subfolders, tenant loop, isolation test. This is the biggest *product* change.
- **Wk 6–7 — Privacy/compliance (Track C1–C3).** Consent record, retention +
  deletion path, PII scrubbing in shared artifacts, secrets review.
- **Wk 7–8 — Launch hardening.** Load/failure testing the tenant loop (one
  tenant's bad input must not break others), observability (per-tenant run
  status), docs, and a go/no-go review.

**Fast-follow (post-launch, staged):** Money domain (A6); real backend for
horizontal scale (queue/workers/DB); taxonomy → DB if it becomes user-editable.

---

## 5. What a Principal will push on (pre-empted)

1. **"Is this actually launch-scoped, or gold-plating?"** — Track A is
   correctness (launch-blocking). Track B/C are the *real* blockers for a
   public EU product. Money-domain and horizontal-scale are honestly
   *deferred*, with reasons. This doc does not pretend 8 weeks buys a
   planet-scale system.
2. **"Where can it silently produce a wrong number?"** — enumerated (F1/F3 are
   the silent ones; A1/A3 fix them by *surfacing*, not hiding).
3. **"Blast radius of one bad tenant input?"** — B2 isolation invariant + Wk7-8
   failure test: one tenant's malformed file must not abort the cohort.
4. **"GDPR story?"** — Track C. Non-negotiable for EU launch; must not be an
   afterthought.
5. **"Reversibility / verification?"** — golden-master corpus (Wk3-4) makes
   every structural change provably behavior-preserving; per-phase branch +
   full suite + end-to-end, as already practiced.

## 6. Recommendation

Approve **Track A now** (2 weeks, pure integrity, low risk) as it stands on its
own merits regardless of the launch. Treat **Tracks B + C as the launch
gate** and timebox them honestly. Explicitly defer **money-domain** and
**horizontal-scale backend** as documented fast-follows. Do **not** commit to
"global scale" in 8 weeks — commit to "correct, private, multi-tenant for a
bounded EU cohort," which is defensible and real.
