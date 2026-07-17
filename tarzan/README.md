# Tarzan — Package Reference

Technical reference for the `tarzan` Python package. For an overview and
quickstart, see the [root README](../README.md).

## Architecture

```
tarzan/
├── __init__.py                  # Package root, versioning
├── main.py                      # CLI entry point (writes side reports)
├── orchestrator.py              # Pipeline: load → enrich → compute
├── contracts/                   # Input/output boundary contracts
│   ├── schema.py                #   Declarative, versioned input-file schema
│   ├── validation.py            #   Input-boundary validators (ISIN/currency/order sign)
│   └── exceptions.py            #   Domain exception hierarchy (TarzanError).
│                                #     NOTE: only DataIngestionError is currently
│                                #     raised; other stages fail soft into the
│                                #     data-quality report instead.
├── runtime/                     # Per-run cross-cutting state (all reset each run)
│   ├── __init__.py              #   RunContext: the clock + determinism switch
│   ├── data_quality.py          #   Per-run data-quality collector (skips/coercions/fallbacks)
│   ├── audit.py                 #   Per-run rebalancing audit collector (why each trade)
│   └── report_html.py           #   The single run report → output/report.html
├── delivery/                    # Run + render + dispatch the newsletter
│   ├── __init__.py              #   Newsletter service: resolve inputs → run → render → email
│   └── drive_loader.py          #   Download input CSVs from a private Google Drive folder
├── config/
│   ├── __init__.py              # Configuration loader (YAML + taxonomy CSV → accessors)
│   ├── constants.yaml           # Tunable parameters (risk-free rate, classification, ...)
│   └── static.yaml              # Rarely-changed mappings (exchanges, FIGI, ...)
├── models/
│   ├── holding.py               # Holding dataclass, AssetClass / Geography enums
│   ├── order.py                 # Order dataclass + OrderType (the order-list row)
│   ├── investor_config.py       # InvestorConfig with CSV deserialization
│   └── portfolio.py             # PortfolioMetrics (output DTO)
├── data/
│   ├── loader.py                # CSV / XLSX → list[Order], config parsing
│   ├── enricher.py              # yfinance, FX, classification, backtest period
│   ├── market_quotes.py         # Live quotes + broker-style 1D (Markets strip)
│   ├── price_cache.py           # On-disk cache of immutable market data
│   ├── geo_resolver.py          # Geographic allocation resolver
│   ├── bond_fetcher.py          # Borsa Italiana bond fallback + bond value math
│   └── proxy_data.py            # Proxy series for asset-class simulation
├── engine/
│   ├── stats.py                 # Pure return/risk math (CAGR, Sharpe, XIRR, TWROR, ...)
│   ├── metrics.py               # MetricsEngine: orchestrates all computers
│   ├── returns_builder.py       # Order-derived value series + XIRR/TWROR
│   ├── benchmarks.py            # Benchmark series + benchmark-relative metrics
│   ├── rebalancer.py            # Local-search rebalancing optimizer
│   ├── tax.py                   # Estimated Italian capital-gains tax
│   ├── robustness.py            # Rolling / stress / bootstrap risk analysis
│   └── synthetic.py             # Synthetic-history helpers
├── export/
│   ├── newsletter/              # HTML email newsletter (the primary artifact):
│   │                            #   __init__ (orchestrator + public API),
│   │                            #   _constants, _format, _charts,
│   │                            #   _sections_alloc, _sections_perf
│   ├── _perf_series.py          # Pure performance/return series helpers (newsletter)
│   ├── _charts.py               # Inline SVG chart builders
│   ├── _format.py               # Shared formatting / palette helpers
│   ├── ai_summary.py            # Optional Gemini narrative summary
│   └── whatif_excel.py          # What-if scenario workbook (scripts/whatif.py only)
└── tests/                       # Pytest suite (~20 modules, run: pytest tarzan/tests/)
```

## Installation

```bash
python -m pip install --require-hashes -r requirements.txt
```

## Usage

```bash
# Minimal run (uses defaults for input_config and output)
python -m tarzan.main --input_orders input/order_list.csv

# Full CLI
python -m tarzan.main \
    --input_orders input/order_list.csv \
    --input_config input/targets.csv \
    --output       output/
```

Provide your own `order_list.csv` (see the Input section above); each run's
artifacts land in `output/<YYYY-MM-DD>/`.

### Reproducible / as-of runs

```bash
# Value the portfolio as of a past date (pins the terminal valuation date
# for XIRR/TWROR and the daily series):
python -m tarzan.main --input_orders input/order_list.csv --as_of 2026-06-30

# Fully deterministic: pin the clock AND stand down the live surfaces (skip
# live intraday quotes and the AI summary):
python -m tarzan.main --input_orders input/order_list.csv --deterministic --as_of 2026-06-30
```

The runtime resolves exactly one mode. `LIVE` admits eligible live providers;
`POINT_IN_TIME` is selected by `--as_of`; `REPRODUCIBLE` requires both
`--deterministic` and `--as_of`. Both pinned modes use one effective clock and
order boundary and prohibit live market/Gemini transport. Deterministic summary
bytes use the Analysis ID and exclude operational Attempt ID, wall time,
latency, retry delay, and diagnostic prose.

## Input

### Order list (required)

A `.csv` or `.xlsx` of your orders — the single source of truth. Tarzan
derives the current snapshot (net quantity, average-cost basis, market
value via live prices) and the historical value series from it. Minimum
columns (case-insensitive):

| Column      | Type  | Required | Description                          |
|-------------|-------|:--------:|--------------------------------------|
| `date`      | date  | ✓        | Order date (YYYY-MM-DD)              |
| `type`      | str   | ✓        | buy / sell / transfer_in / coupon …  |
| `isin`      | str   | ✓        | 12-character ISIN code               |
| `quantity`  | float | ✓        | Units traded (sign per direction)    |
| `gross_eur` | float | ✓        | Gross amount in EUR                  |
| `net_eur`   | float | ✓        | Net cash flow in EUR (− for buys)    |

Optional columns (`trade_date`, `name`, `ticker`, `currency`,
`price_native`, `fx_rate`, `fees_eur`, `source`) are used when present.

Geographic allocation is resolved automatically: first by ticker / ISIN
lookup in `input/instrument_taxonomy.csv`, then via yfinance fund composition data.

#### Optional: build the order list from a Fineco export

If you get a "Lista Titoli" movements export from Fineco, the
`scripts/preprocess_orders.py` script normalises it into the Tarzan
order-list schema:

```bash
python scripts/preprocess_orders.py
```

This step is **optional** — `python -m tarzan.main` only ever reads
`input/order_list.csv`, regardless of how it got there.

### Per-instrument targets (optional)

A `targets_per_holding.csv` (joined by ISIN) carries the rebalancer's
per-instrument `target_equities` / `target_fixed_income` weights and the
`no_buy_no_sell` flag — the order list itself has no target columns.

### Targets (optional)

A `targets.csv` with key / value pairs. Keys follow a typed-suffix
convention so the unit is unambiguous:

- `_eur` — absolute EUR amount
- `_pctg` — percentage value
- `_date` — free-form date string
- no suffix — boolean flag

**Rebalancing parameters**

| Key                                  | Default | Description                                             |
|--------------------------------------|---------|---------------------------------------------------------|
| `rebalancing_lump_sum_amount_eur`    | `0`     | Extra cash to deploy in a rebalance                     |
| `rebalancing_target_tolerance_pctg`  | `2.0`   | Tolerance band around every allocation target. The LP uses it as the hard ceiling and the dashboard uses it as the traffic-light threshold. |
| `rebalancing_no_sell`                | `false` | If true, solver can only buy                            |

**Cash buffer (absolute EUR, tracked separately from invested %)**

| Key                      | Default | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `target_cash_buffer_eur` | `0`     | Target cash amount; excess is invested by solver |

**Invested allocation (notional % of invested portfolio = total − cash; finite and nonnegative, totals above 100% are valid)**

| Key                                                | Description                     |
|----------------------------------------------------|---------------------------------|
| `target_invested_allocation_equities_pctg`         | Target weight for equities      |
| `target_invested_allocation_fixed_income_pctg`     | Target weight for fixed income  |
| `target_invested_allocation_gold_pctg`             | Target weight for gold          |
| `target_invested_allocation_commodities_pctg`      | Target weight for commodities   |
| `target_invested_allocation_crypto_pctg`           | Target weight for crypto        |
| `target_invested_allocation_alternative_pctg`      | Target weight for alternative   |

**Equity geography (% of equity portion, must sum to 100)**

| Key                                                    | Description     |
|--------------------------------------------------------|-----------------|
| `target_equity_geo_usa_pctg`                           | USA             |
| `target_equity_geo_japan_pctg`                         | Japan           |
| `target_equity_geo_eurozone_emu_pctg`                  | Eurozone        |
| `target_equity_geo_dev_ex_usa_ex_emu_ex_jp_pctg`       | Other developed |
| `target_equity_geo_emerging_markets_pctg`              | Emerging mkts   |

## Financial metrics

### Performance
- CAGR, YTD, periodic returns (1d to 5y), IRR

### Order-list returns — XIRR & TWROR (optional)

When an order list is supplied (`--input_orders input/order_list.csv`, or
`ORDERS_PATH` for the newsletter), the order list becomes the single
source of the portfolio's historical value series and Tarzan additionally
computes:

- **XIRR** (money-weighted return) — the annualized rate that zeroes the
  net present value of every external cash flow plus today's value.
  Sensitive to *when* you deposit/withdraw.
- **TWROR** (time-weighted return) — chained period returns, neutral to
  deposit timing; the market behaviour of the held portfolio. Reported
  cumulative and annualized.

Both surface in the Excel Dashboard (KPI rows) and the newsletter
Performance section, and are `None`/absent for a holdings-only run.

**Historical price fallback ladder.** Building a daily value series needs
a price for every held instrument on every date. When Yahoo Finance has
no usable history, Tarzan walks an explicit ladder and **records which
rung priced each instrument** so the figure is transparent:

1. `yfinance` — real daily series (preferred).
2. `synthetic` — linear interpolation between the order-list trade prices
   (the default fallback). Captures the trend between trades; understates
   intra-trade volatility.
3. `carry_flat` — a single known price held flat (zero volatility
   contribution for that name).
4. `excluded` — no price at all; the instrument drops out of the
   valuation on that date.

The output reports a **coverage %** (share of value priced by real market
data) and lists the fallback-priced instruments, in both Excel (a
Performance-tab footnote) and the newsletter (a muted sub-line).

**Known limitation.** A few fixed-income ISINs have no daily history on
either Yahoo Finance or the Borsa Italiana fallback — notably the US
Treasury `US91282CGJ45`, and intermittently the BTP/Eurobond lines
(`IT0005542359`, `XS2105803527`, `IT0005358806`). These are priced by the
`synthetic`/`carry_flat` rungs, so their contribution to TWROR is
approximate (trend-only). XIRR is unaffected by this — it depends only on
the cash flows and today's value, both of which are known exactly.
Foreign-currency bonds (e.g. ZAR/USD) are converted to EUR via the order
`fx_rate`, so they are valued correctly despite the missing history.

### Risk
- **Sharpe ratio** — risk-adjusted return (excess return / volatility)
- **Sortino ratio** — penalizes downside volatility only
- **Max drawdown** — largest peak-to-trough loss
- **VaR (95%)** — Value at Risk via historical simulation (non-parametric)
- **CVaR (95%)** — Expected Shortfall, the mean loss beyond VaR (a coherent
  risk measure, Artzner et al. 1999)
- **Realized volatility** — annualized rolling window
- **Beta / Alpha** — CAPM vs S&P 500

### Allocations
- By asset class, geography (equity only), and sector
- Multi-geography ETFs split proportionally
- Delta vs target with rebalancing suggestions

### Benchmarks
- Comparison against 20+ indexes (S&P 500, ACWI, VTI, AVUV, ...)

## Output

The HTML email newsletter `portfolio_digest_[YYYYMMDD_HHMM].html` — Tarzan's
primary artifact, a single 600px-wide inbox-optimised page. Sections:

1. **Hero** — Total / Invested / Cash values, Total Gain, unrealized P&L.
2. **Performance** — period returns + risk (1D…5Y, CAGR, Volatility, Sharpe,
   Sortino, Max DD, VaR/CVaR 95%, Alpha, Beta), XIRR/TWROR when an order list
   is supplied, and the money-moved charts.
3. **Allocation** — by asset class and equity geography, vs target, with the
   rebalancing status banner.
4. **Holdings** — enriched table grouped by asset class.
5. **Market context** — optional AI narrative summary (Gemini free tier).

Each run also writes `report.html` — the run log (data-quality summary + a
lean color-coded log table). Both land under `output/<YYYY-MM-DD>/`.

The what-if scenario tool (`scripts/whatif.py`) can additionally emit an Excel
workbook via `--excel`; that is the only remaining spreadsheet surface.

## Exception hierarchy

All domain errors inherit from `TarzanError`:

- `TarzanError` — base class
- `DataIngestionError` — input data cannot be loaded or parsed

## Testing

```bash
pytest tarzan/tests/
```
