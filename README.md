# Tarzan

**Portfolio analysis for investors who swing smart.**

Tarzan is a production-grade portfolio analyzer with live market data
enrichment, instrument classification, and a 5-sheet Excel report for
multi-asset portfolios.

---

## Features

- **Data enrichment** — Live prices, FX conversion, and instrument
  classification via yfinance, OpenFIGI, and Borsa Italiana fallback.
- **Risk metrics** — CAGR, Sharpe, Sortino, Max Drawdown, VaR / CVaR (95%),
  realized volatility, Beta / Alpha vs S&P 500.
- **Allocations** — By asset class (Equities, Fixed Income, Gold,
  Commodities, Crypto, Alternative), equity geography, and sector.
  Multi-geography ETFs are split proportionally. Delta vs target with
  rebalancing suggestions.
- **Benchmarks** — Comparison against 20+ indexes (S&P 500, ACWI, VTI,
  AVUV, ...).
- **Rebalancing engine** — Mixed-integer optimization (`scipy.milp`) for
  buy / sell / lump-sum suggestions that respect user constraints,
  including a separate absolute cash buffer target.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run the sample portfolio
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/
```

A ready-made report built from the sample data lives in
[`output/sample/portfolio_dashboard_sample.xlsx`](output/sample/).

## Input format

Holdings are a CSV or XLSX with the following columns (case-insensitive):

| Column             | Type  | Required | Description                     |
|--------------------|-------|:--------:|---------------------------------|
| `isin`             | str   | ✓        | 12-character ISIN code          |
| `ticker`           | str   | ✓        | Yahoo Finance ticker            |
| `quantity`         | float | ✓        | Number of units (> 0)           |
| `cost_basis_eur`   | float | ✓        | Total cost in EUR               |
| `market_value_eur` | float | ✓        | Current market value in EUR     |
| `currency`         | str   | ✓        | Instrument currency             |

Targets are an optional CSV with typed-suffix keys (`_eur`, `_pctg`,
`_date`) for cash buffer, invested-allocation targets, equity geography
targets and rebalancing parameters.

See [`tarzan/README.md`](tarzan/README.md) for the full configuration
reference and metric definitions.

## Project layout

```
Tarzan/
├── tarzan/                  # Python package
│   ├── main.py              # CLI entry point
│   ├── orchestrator.py      # Pipeline: load → enrich → compute
│   ├── config/              # YAML-driven configuration
│   ├── data/                # Loaders, enricher, cache, geo resolver
│   ├── engine/              # Metrics and rebalancer
│   ├── export/              # Excel report generator
│   ├── models/              # Holding, InvestorConfig, PortfolioMetrics
│   └── tests/               # Pytest suite
├── input/
│   └── sample/              # Sample holdings / targets CSVs (tracked)
├── output/
│   └── sample/              # Pre-generated sample Excel report (tracked)
└── requirements.txt
```

## Tech stack

- Python 3.12
- pandas, numpy, scipy (mixed-integer optimization)
- yfinance (market data)
- openpyxl (Excel export)

## Development

```bash
# Run the test suite
pytest tarzan/tests/
```

## License

Personal project. All rights reserved.
