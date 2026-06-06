# Tarzan — Package Reference

Technical reference for the `tarzan` Python package. For an overview and
quickstart, see the [root README](../README.md).

## Architecture

```
tarzan/
├── __init__.py                  # Package root, versioning
├── main.py                      # CLI entry point
├── orchestrator.py              # Pipeline: load → enrich → compute
├── exceptions.py                # Domain exception hierarchy (TarzanError)
├── config/
│   ├── __init__.py              # Configuration loader (YAML → typed accessors)
│   ├── constants.yaml           # Tunable parameters (risk-free rate, classification, ...)
│   └── static.yaml              # Rarely-changed mappings (exchanges, FIGI, ...)
├── models/
│   ├── holding.py               # Holding dataclass, AssetClass / Geography enums
│   ├── investor_config.py       # InvestorConfig with CSV deserialization
│   └── portfolio.py             # PortfolioMetrics (output DTO)
├── data/
│   ├── loader.py                # CSV / XLSX → list[Holding], config parsing
│   ├── enricher.py              # yfinance, FX, classification, backtest period
│   ├── geo_resolver.py          # Geographic allocation resolver
│   ├── bond_fetcher.py          # Borsa Italiana bond fallback scraper
│   └── cache.py                 # Local cache for enriched data
├── engine/
│   ├── metrics.py               # MetricsEngine: performance, risk, allocations
│   └── rebalancer.py            # Mixed-integer rebalancing optimizer
├── export/
│   └── excel.py                 # 5-sheet Excel report generator
└── tests/                       # Pytest suite
    ├── conftest.py
    ├── test_loader.py
    ├── test_metrics.py
    ├── test_enricher.py
    └── test_rebalancer.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Minimal run (uses defaults for input_config and output)
python -m tarzan.main --input_holdings input/sample/sample_holdings.csv

# Full CLI
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/
```

See [`input/sample/`](../input/sample/) for ready-to-use sample CSVs and
[`output/sample/`](../output/sample/) for a pre-generated Excel report.

## Input

### Holdings (required)

A `.csv` or `.xlsx` file with the following columns (case-insensitive):

| Column             | Type  | Required | Description                     |
|--------------------|-------|:--------:|---------------------------------|
| `isin`             | str   | ✓        | 12-character ISIN code          |
| `ticker`           | str   | ✓        | Yahoo Finance ticker            |
| `quantity`         | float | ✓        | Number of units (> 0)           |
| `cost_basis_eur`   | float | ✓        | Total cost in EUR               |
| `market_value_eur` | float | ✓        | Current market value in EUR     |
| `currency`         | str   | ✓        | Instrument currency             |

Geographic allocation is resolved automatically: first by ticker / ISIN
lookup in `input/indexes.csv`, then via yfinance fund composition data.

#### Optional: build holdings from a Fineco export

Curating `holdings.csv` by hand is fine but tedious if you already get a
"Portafoglio di sintesi" export from Fineco. The
`scripts/preprocess_fineco.py` script normalises that export into the
Tarzan schema and merges the per-holding targets you maintain in a
separate CSV (joined by ISIN).

Drop the two files into `input/fineco_raw/` and run:

```bash
python scripts/preprocess_fineco.py
```

Defaults read `input/fineco_raw/portafoglio-export.xls` and
`input/fineco_raw/targets_per_holding.csv`, and write
`input/holdings.csv` after timestamping a backup of any existing file.
Override paths with `--input`, `--targets`, `--output` if needed.

This step is **always optional** — `python -m tarzan.main` only ever
reads `input/holdings.csv`, regardless of how it got there.

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
| `portfolio_inception_date`           | `""`    | Inception date used by performance charts               |

**Cash buffer (absolute EUR, tracked separately from invested %)**

| Key                      | Default | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `target_cash_buffer_eur` | `0`     | Target cash amount; excess is invested by solver |

**Invested allocation (% of invested portfolio = total − cash, must sum to 100)**

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

Excel file `portfolio_dashboard_[YYYYMMDD_HHMM].xlsx` with 5 sheets:

1. **Dashboard** — Hero KPIs (Total / Invested / Cash values, Total Gain, RTD),
   invested allocation vs target (with cash buffer as an EUR row), geography
   breakdown, top 5 holdings, rebalancing alert.
2. **Optimizer** — Banner with traffic-light status, rebalancing actions
   (buy / sell / amount / % of portfolio / reason), consolidated allocation
   deviations grouped by type (invested asset classes incl. cash buffer in
   EUR, equity geography, per-holding equity and fixed income targets),
   and solver parameters.
3. **Holdings** — Full enriched table (ticker, ISIN, asset class, quantity,
   prices, values, % of portfolio, % of invested, % of asset class, gain,
   geography, data source).
4. **Performance** — Unified period returns + risk table (1D…5Y, CAGR,
   Volatility, Sharpe, Sortino, Max DD, VaR 95%, CVaR 95%, Alpha, Beta) for
   portfolio, holdings and benchmarks, plus a legend with investor-friendly
   descriptions and rating thresholds.
5. **Return Contribution** — Per-holding contribution to total return,
   breakdowns by asset class and by equity geography.

## Exception hierarchy

All domain errors inherit from `TarzanError`:

- `TarzanError` — base class
- `DataIngestionError` — input data cannot be loaded or parsed
- `DataEnrichmentError` — market data enrichment failed for a holding
- `InsufficientDataError` — not enough data to compute a metric
- `MetricCalculationError` — numerical error in a metric calculation
- `ClassificationError` — instrument cannot be classified
- `ConfigurationError` — invalid or missing configuration

## Testing

```bash
pytest tarzan/tests/
```
