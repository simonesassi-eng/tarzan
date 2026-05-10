# Sample input

Sample portfolio and target files used to showcase Tarzan without any
real personal data.

## Files

- **`sample_holdings.csv`** — 10 positions across equities, fixed income,
  gold, and cash. Uses real, liquid tickers (iShares, Vanguard,
  Xtrackers) for realistic live market data.
- **`sample_targets.csv`** — investor targets: 65% equities / 25% fixed
  income / 5% gold / 3% cash / 2% alternative, with geographic
  allocation tilted to USA (55%) and a rebalancing threshold of 2.5%.

## How to use

From the project root:

```bash
# CLI
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/

# Streamlit
streamlit run tarzan/presentation/app.py
# then upload the two CSVs from this folder in the sidebar
```

See the generated dashboard in [`output/sample/`](../../output/sample/).

## Notes

- `CASH-EUR` is a pseudo-ticker for the cash position; Yahoo Finance
  does not resolve it, which is expected. The pipeline falls back to the
  `market_value_eur` column for that row.
- All `quantity`, `cost_basis_eur` and `market_value_eur` values are
  fictitious and chosen to produce a portfolio of roughly €80,000.
