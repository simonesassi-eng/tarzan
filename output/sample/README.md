# Sample output

This folder contains a pre-generated Excel dashboard produced by running
Tarzan against the sample portfolio in [`input/sample/`](../../input/sample/).

## Files

- **`portfolio_dashboard_sample.xlsx`** — 8-sheet dashboard with KPIs,
  allocations (with traffic-light deviation colors), top holdings,
  performance, risk, return contribution, and documentation.
- **`sample_run.log`** — full execution log from the run that produced
  the dashboard.

## How to regenerate

From the project root:

```bash
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/
```

The CLI writes a timestamped `portfolio_dashboard_YYYYMMDD_HHMM.xlsx`.
Rename it to `portfolio_dashboard_sample.xlsx` if you want to overwrite
the tracked artifact.

## Notes

- Ticker prices come from Yahoo Finance at runtime, so numbers shift a
  little between runs.
- The `CASH-EUR` pseudo-ticker fails to resolve on Yahoo (by design); the
  pipeline falls back to `market_value_eur` from the CSV for that row.
- The sample uses a `rebalancing_threshold` of 2.5% to demonstrate the
  green / amber / red coloring on the dashboard allocation tables.
