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
│   ├── constants.yaml           # Tunable parameters (risk-free rate, benchmarks, ...)
│   └── static.yaml              # Rarely-changed mappings (exchanges, colors, ...)
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
│   └── excel.py                 # 8-sheet Excel dashboard generator
├── presentation/
│   ├── app.py                   # Streamlit entry point
│   ├── charts.py                # Plotly chart factories (donut, line, drawdown)
│   ├── formatters.py            # Number / currency / percent formatters
│   ├── assets/                  # Logo and static assets
│   └── views/                   # Per-page Streamlit views
│       ├── dashboard.py
│       ├── holdings.py
│       ├── optimizer.py
│       ├── performance.py
│       ├── contribution.py
│       └── documentation.py
└── tests/                       # Pytest suite
    ├── conftest.py
    ├── test_loader.py
    ├── test_metrics.py
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

# Streamlit dashboard
streamlit run tarzan/presentation/app.py
```

See [`input/sample/`](../input/sample/) for ready-to-use sample CSVs and
[`output/sample/`](../output/sample/) for a pre-generated Excel dashboard.

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

### Targets (optional)

A `targets.csv` with key / value pairs:

| Key                       | Default        | Description                              |
|---------------------------|----------------|------------------------------------------|
| `monthly_invest_capacity` | `0`            | Monthly investment budget in EUR         |
| `geo_exposure`            | 20% each       | JSON: target geographic allocation       |
| `allocation_targets`      | Equity 65% ... | JSON: target asset-class allocation      |
| `rebalancing_threshold`   | `5.0`          | Rebalancing deviation threshold (%)      |

## Financial metrics

### Performance
- CAGR, YTD, periodic returns (1d to 5y), IRR

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
- What-if analysis: hypothetical value if invested in each benchmark

## Output

Excel file `portfolio_dashboard_[YYYYMMDD_HHMM].xlsx` with 8 sheets:

1. **Dashboard** — KPIs (VaR / CVaR included), donut chart, top / bottom
   performers, goals
2. **Holdings** — Full enriched table with data sources and timestamps
3. **Allocations** — Pie / bar charts with actual vs target
4. **Performance** — Cumulative returns, per-period grids, holdings overlay
5. **Risk** — Full risk metrics, drawdown chart, risk-return scatter
6. **Multi-Purpose Analysis** — Return contribution, breakdowns, rebalancing
   actions
7. **Benchmark** — Comparison vs 20+ benchmarks, cumulative performance,
   what-if
8. **Documentation** — Description and formula for every metric

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
