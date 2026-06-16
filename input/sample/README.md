# Sample input

Sample order list and target files used to showcase Tarzan without any
real personal data.

## Files

- **`sample_order_list.csv`** — a handful of buy orders across equities,
  fixed income and gold. Uses real, liquid tickers (iShares, Vanguard,
  Xtrackers, Invesco) so live market data resolves. Tarzan derives the
  current snapshot (net quantity, average-cost basis, market value) and
  the historical value series from these orders.
- **`sample_targets.csv`** — investor targets using the typed-key schema:
  - Invested allocation (% of invested portfolio, excludes cash):
    66% equities · 25% fixed income · 5% gold · 3% crypto · 1% alternative.
  - `target_cash_buffer_eur = 3000` kept separately as an absolute
    buffer that the optimizer aims at.
  - Equity geography (% of equity portion): tilted to USA (55%).
  - `rebalancing_target_tolerance_pctg = 2.5` drives both the LP solver ceiling and the traffic-light colors.

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
    --input_orders input/sample/sample_order_list.csv \
    --input_config input/sample/sample_targets.csv \
    --output       output/sample/
```

See the generated report in [`output/sample/`](../../output/sample/).

## Notes

- The order list is the single source of truth. Per-instrument
  rebalancing targets (the `target_equities` / `target_fixed_income`
  columns) live in an optional `targets_per_holding.csv`, joined by ISIN.
