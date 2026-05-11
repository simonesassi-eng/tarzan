# Tarzan

**Portfolio analysis for investors who swing smart.**

Tarzan is a production-grade portfolio analyzer with live market data enrichment,
professional Excel reporting, and an interactive Streamlit dashboard for
multi-asset portfolios.

<p align="center">
  <img src="tarzan/presentation/assets/tarzan_logo.png" alt="Tarzan logo" width="120"/>
</p>

---

## Features

- **Data enrichment** — Live prices, FX conversion, and instrument classification
  via yfinance, OpenFIGI, and Borsa Italiana fallback.
- **Risk metrics** — CAGR, Sharpe, Sortino, Max Drawdown, VaR / CVaR (95%),
  realized volatility, Beta / Alpha vs S&P 500.
- **Allocations** — By asset class, geography, and sector. Multi-geography ETFs
  are split proportionally. Delta vs target with rebalancing suggestions.
- **Benchmarks** — Comparison against 20+ indexes (S&P 500, ACWI, VTI, AVUV, ...).
- **Rebalancing engine** — Mixed-integer optimization (`scipy.milp`) for buy /
  sell / lump-sum suggestions that respect user constraints, including a
  separate absolute cash buffer target.
- **Two UIs** — Streamlit dashboard for interactive exploration, Excel export
  (5-sheet workbook) for offline reporting.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Try the sample portfolio
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/

# 3. Or launch the Streamlit app
streamlit run tarzan/presentation/app.py
```

The Streamlit app opens in your browser. Upload your own holdings CSV and
(optionally) a targets CSV, then click **Analyze Portfolio**. A ready-made
dashboard built from the sample data lives in
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

Targets are an optional CSV with key / value pairs for monthly investment
capacity, geographic exposure, asset-class targets, and rebalancing thresholds.

See [`tarzan/README.md`](tarzan/README.md) for the full configuration reference
and metric definitions.

## Project layout

```
Tarzan/
├── tarzan/                  # Python package (core engine + UI)
│   ├── main.py              # CLI entry point
│   ├── orchestrator.py      # Pipeline: load → enrich → compute
│   ├── config/              # YAML-driven configuration
│   ├── data/                # Loaders, enricher, cache, geo resolver
│   ├── engine/              # Metrics and rebalancer
│   ├── export/              # Excel report generator
│   ├── models/              # Holding, InvestorConfig, PortfolioMetrics
│   ├── presentation/        # Streamlit app and views
│   └── tests/               # Pytest suite
├── input/
│   └── sample/              # Sample holdings / targets CSVs (tracked)
├── output/
│   └── sample/              # Pre-generated sample Excel dashboard (tracked)
└── requirements.txt
```

## Tech stack

- Python 3.12
- pandas, numpy, scipy (mixed-integer optimization)
- yfinance (market data), openpyxl (Excel export)
- Streamlit and plotly (interactive UI)

## Development

```bash
# Run the test suite
pytest tarzan/tests/
```

## License

Personal project. All rights reserved.
