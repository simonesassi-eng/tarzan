# Sample input

Sample portfolio and target files used to showcase Tarzan without any
real personal data.

## Files

- **`sample_holdings.csv`** — 11 positions across equities, fixed income,
  gold, crypto and cash. Uses real, liquid tickers (iShares, Vanguard,
  Xtrackers, ETC Group) for realistic live market data.
- **`sample_targets.csv`** — investor targets using the typed-key schema:
  - Invested allocation (% of invested portfolio, excludes cash):
    66% equities · 25% fixed income · 5% gold · 3% crypto · 1% alternative.
  - `target_cash_buffer_eur = 3000` kept separately as an absolute
    buffer that the optimizer aims at.
  - Equity geography (% of equity portion): tilted to USA (55%).
  - `rebalancing_threshold_pctg = 2.5` drives the traffic-light colors.

## Target CSV key convention

Keys follow a typed-suffix convention so the unit is unambiguous:

- `_eur`  — absolute EUR amount (e.g. `target_cash_buffer_eur`)
- `_pctg` — percentage value (e.g. `target_invested_allocation_equities_pctg`)
- `_date` — free-form date string
- no suffix — boolean flag (`rebalancing_no_sell`)

## How to use

From the project root:

```bash
python -m tarzan.main \
    --input_holdings input/sample/sample_holdings.csv \
    --input_config   input/sample/sample_targets.csv \
    --output         output/sample/
```

See the generated report in [`output/sample/`](../../output/sample/).

## Notes

- `CASH-EUR` is a pseudo-ticker for the cash position; Yahoo Finance
  does not resolve it, which is expected. The pipeline falls back to the
  `market_value_eur` column for that row.
- All `quantity`, `cost_basis_eur` and `market_value_eur` values are
  fictitious and chosen to produce a portfolio of roughly €80,000.
