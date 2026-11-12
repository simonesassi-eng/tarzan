# Backtest method — the one way Tarzan simulates history

Every simulated return in Tarzan — the what-if backtest (`tarzan.backtest`), the
Excel what-if, and the newsletter's target line (`tarzan.engine.target_backfill`)
— is built by the pipeline below. There is no second implementation. If a new
consumer needs a modeled sleeve, it calls these functions; it does not rebuild
them, and an analysis script does not monkeypatch them. When a script needs a
different treatment, the treatment is wrong or missing in the engine: fix it
here, with a test, and record it in this file.

## Pipeline

1. **Proxies** — `proxy_data.proxy_returns_for(keys)`
   - One long-history proxy per bucket (equity region, intermediate or long
     duration bonds, gold, commodities, trend, carry, cash), longest history
     first.
   - Fund proxies are grossed up by their own expense ratio
     (`PROXY_FUND_ER`), so the target ETF's TER is the only fee charged.
   - USD proxies are converted to EUR by the interval method
     (`_usd_returns_to_eur`): the FX move over each of the asset's own intervals,
     holidays included.
   - Cash and financing are at the ECB rate for a EUR reporter.
   - Managed futures before 2007: AQR TSMOM timing at the real funds' level,
     only when ingested locally (`manual_proxies ingest TSMOM`; copyrighted,
     never vendored); otherwise cash.

2. **Sleeve model** — `backtest.engine.instrument_base_returns(item, …)`
   - Exposures from the taxonomy composition (`instrument_exposures`, duration
     routing for bonds).
   - Leverage above 1× is financed at the short rate + 0.5%.
   - **Currency on leverage:**
     - A *futures overlay* — taxonomy roles Efficient Core, Trend Stacked, Gold
       Stacked and Carry Stacked — carries currency on its NAV only. A future's
       notional is never converted.
     - A daily-leveraged *swap* ETF on a EUR-converted index (role Equity
       Leveraged, e.g. CL2) levers the currency too. Verified on CL2:
       "2× the EUR return" fits monthly with corr 0.963, against 0.922 for
       "2× in USD, then FX".
   - A share class whose name says "EUR Hedged" strips the FX move and earns
     the EUR−USD rate differential.
   - TER is charged as a daily drag on the modeled base only.

3. **Real tape** — the fund's own history, extended back by its **twin**: another
   catalog share class of the same strategy (`target_backfill.twin_rows`),
   converted to this class's currency treatment. Examples: MFEH←DBMFE, and the
   UCITS MATE←the US MATE. The backtest records twins in `TWIN_REAL`; the
   target line uses the same rule.

4. **Splice** — the real tape always wins where it exists. Before it, modes are:
   - `factor`: Fama-French legs for factor funds, monthly;
   - `calibrated`: monthly beta and alpha fitted on the overlap, with alpha used
     only if |t| ≥ 2;
   - `naive`: the composition model as-is.

   Every splice is judged by `_check_backfill_realism` (synthetic vs real
   volatility ratio, band 0.7–1.4).

5. **Portfolio** — `synthetic.combine_returns(df, weights, rebalance)`; drift
   between rebalances.

6. **Robustness** — stress windows, rolling windows, joint block bootstrap and the
   conditional Monte Carlo from today's yields and valuations
   (`engine.robustness`).

## Rules

- **One reconstruction.** Change `instrument_base_returns` or the proxies, never
  a copy. `tests/test_single_sleeve_reconstruction.py` fails if the target line
  stops using it.
- **New instrument:**
  - give it a taxonomy row with ISIN, role and TER;
  - if it is levered, decide overlay vs swap through its role, not in code;
  - check its realism ratio and, if it has ≥ 1 year of real data, regress it
    on its components (see the validation list below).
- **Comparing allocations:**
  - quote numbers from one engine version only;
  - run at least the `factor`, `calibrated` and `naive` modes plus one financing
    sensitivity;
  - compare variants with a *paired* bootstrap of the difference;
  - treat Sharpe gaps under ~0.02 as noise: many variants have been tried
    (deflated Sharpe).
- **Engine changes that move results** go in the commit message with the size of
  the shift.

## Validation checks (real prices; re-run after engine changes)

| Sleeve | Check | Last result (2026-11) |
|---|---|---|
| NTSG | weekly regression on MSCI World (EUR) + Treasury excess | equity β 0.91, bond β 0.44–0.52, α ≈ 0 |
| CL2 | monthly fit of 2× EUR return − ECB financing | corr 0.963, residual drag ≈ 1.4%/yr (engine charges ≈ 1.0%) |
| MATE | US class vs S&P + trend − cash | 1.4–3.6%/yr below, TER included — stress MAET at +1/+2% |
| Trend (DBMFE/MFEH) | realism ratio | 0.5–0.6× — proxy too smooth, flattered |
| UEQC | realism ratio | ~0.4× — vendored carry index too smooth, flattered |

## Known limitations

- The managed-futures and commodity-carry proxies are about half as volatile as
  the real funds. Allocations that lean on them are flattered.
- Before 1990 there are no developed-market factor legs.
- A EUR-hedged class held by a USD reporter is not modeled.
