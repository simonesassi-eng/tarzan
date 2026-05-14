"""Smoke test: render the newsletter from the sample portfolio.

Run from the project root:
    python -m prototypes.newsletter.render_sample

Outputs an HTML file under output/sample/ and prints its path.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tarzan.export.newsletter import generate_newsletter  # noqa: E402
from tarzan.models.investor_config import InvestorConfig  # noqa: E402
from tarzan.models.portfolio import PortfolioMetrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _build_fake_metrics() -> PortfolioMetrics:
    """Build a PortfolioMetrics that mirrors the sample portfolio used in the
    static HTML mock-up. This avoids depending on yfinance at smoke-test time.
    """
    holdings = [
        # name, ticker, isin, asset_class, qty, cost, value, gain_pct
        ("Vanguard FTSE All-World", "VWCE.DE", "IE00B3RBWM25", "Equities", 80, 10240, 11520, 12.5),
        ("iShares Core S&P 500", "CSPX.L", "IE00B5BMR087", "Equities", 15, 8550, 9750, 14.0),
        ("iShares Core MSCI World", "EUNL.DE", "IE00B4L5Y983", "Equities", 50, 4900, 5450, 11.2),
        ("Xtrackers MSCI Emerging Markets", "XMME.DE", "LU0292107645", "Equities", 90, 5040, 5310, 5.4),
        ("Vanguard FTSE Developed Europe", "VERX.DE", "IE00B945VV12", "Equities", 120, 3960, 4320, 9.1),
        ("iShares Nikkei 225 UCITS", "CNKY.L", "IE00B52MJD48", "Equities", 35, 2240, 2450, 9.4),
        ("iShares Core Global Aggregate Bond", "IEAG.L", "IE00B3F81R35", "Fixed Income", 100, 5600, 5700, 1.8),
        ("iShares USD Treasury Bond 7-10y", "IBTM.L", "IE00B1FZS798", "Fixed Income", 40, 4200, 4250, 1.2),
        ("Cash EUR", "CASH-EUR", "CASH-EUR", "Cash & Cash Equivalents", 5000, 5000, 5000, 0.0),
        ("Invesco Physical Gold", "SGLD.L", "IE00B579F325", "Gold", 18, 4410, 5040, 14.3),
    ]
    rows = []
    total = sum(h[6] for h in holdings)
    for h in holdings:
        rows.append({
            "name": h[0], "ticker": h[1], "isin": h[2], "asset_class": h[3],
            "quantity": h[4], "cost_basis_eur": h[5], "current_value": h[6],
            "gain_pct": h[7], "gain_eur": h[6] - h[5],
            "weight_pct": h[6] / total * 100,
            "avg_purchase_price": h[5] / h[4] if h[4] > 0 else 0,
        })
    df = pd.DataFrame(rows)

    # Build allocation_by_class from the holdings (excluding cash for invested allocation)
    invested = df[df["asset_class"] != "Cash & Cash Equivalents"]
    invested_total = invested["current_value"].sum()
    alloc_class = (
        invested.groupby("asset_class")["current_value"].sum() / invested_total * 100
    ).reset_index()
    alloc_class.columns = ["category", "weight_pct"]

    # Geographic allocation (mirroring the static mock-up percentages of equity)
    geo = pd.DataFrame([
        ("USA", 53.1), ("Eurozone EMU", 14.2),
        ("Dev ex-USA ex-EMU ex-JP", 13.4), ("Emerging Markets", 11.6),
        ("Japan", 7.7),
    ], columns=["category", "weight_pct"])

    acwi_geo = {
        "USA": 63.85, "Eurozone EMU": 7.35,
        "Dev ex-USA ex-EMU ex-JP": 13.08, "Emerging Markets": 10.32,
        "Japan": 5.4,
    }

    # Goal deltas (used by smart insights to find largest drift)
    targets_class = {"Equities": 68.0, "Fixed Income": 26.0, "Gold": 5.0,
                     "Commodities": 0.0, "Alternative": 1.0}
    targets_geo = {"USA": 55.0, "Eurozone EMU": 14.0,
                   "Dev ex-USA ex-EMU ex-JP": 12.0, "Emerging Markets": 11.0,
                   "Japan": 8.0}
    goal_rows = []
    for _, r in alloc_class.iterrows():
        target = targets_class.get(r["category"])
        if target is not None:
            goal_rows.append({
                "type": "asset_class", "category": r["category"],
                "actual_pct": r["weight_pct"], "target_pct": target,
                "delta_pct": r["weight_pct"] - target,
            })
    for _, r in geo.iterrows():
        target = targets_geo.get(r["category"])
        if target is not None:
            goal_rows.append({
                "type": "geography (equity only)", "category": r["category"],
                "actual_pct": r["weight_pct"], "target_pct": target,
                "delta_pct": r["weight_pct"] - target,
            })
    goal_df = pd.DataFrame(goal_rows)

    # Synthesised history for the sparkline (30 days, +5% growth)
    end = datetime.now().date()
    dates = pd.date_range(end=end, periods=30, freq="D")
    history = pd.Series(
        [55940 + (i * 95) + (i % 5) * 30 for i in range(30)],
        index=dates,
    )

    # One rebalancing suggestion (BUY of Fixed Income to close the drift)
    rebal = [{
        "ticker": "IEAG.L", "name": "iShares Core Global Aggregate Bond",
        "isin": "IE00B3F81R35", "direction": "buy", "amount_eur": 1000.0,
        "reason": "Fixed Income is 9.1 pp below target. Deploying €1,000 brings the bucket from 16.9% to 18.6%, while keeping cash above the €3,000 buffer.",
    }]

    # Holding performance with 1W and other periods (subset)
    hp_rows = []
    weekly = {"VWCE.DE": 1.05, "CSPX.L": 2.40, "EUNL.DE": 1.18, "XMME.DE": -0.32,
              "VERX.DE": 0.55, "CNKY.L": 0.92, "IEAG.L": 0.17, "IBTM.L": -0.42,
              "SGLD.L": 1.85}
    for h in holdings:
        if h[1] == "CASH-EUR":
            continue
        hp_rows.append({
            "name": h[0], "ticker": h[1], "type": "In portfolio",
            "1w": weekly.get(h[1], 0.0), "1m": h[7] * 0.4, "ytd": h[7] * 0.6,
            "1y": h[7] * 1.1, "3y": h[7] * 0.9, "5y": h[7] * 1.0,
            "3m": h[7] * 0.3, "cagr": h[7] * 0.7,
        })
    # Add a few benchmark rows
    bench_rows = [
        {"name": "S&P 500", "ticker": "^GSPC", "type": "Benchmark index",
         "1w": 1.31, "1m": 3.10, "ytd": 6.90, "1y": 12.10, "3y": 10.85, "5y": 13.20, "3m": 4.92, "cagr": 10.85},
        {"name": "MSCI ACWI", "ticker": "^892400-USD-STRD", "type": "Benchmark index",
         "1w": 0.96, "1m": 2.42, "ytd": 4.80, "1y": 9.85, "3y": 8.42, "5y": 9.95, "3m": 3.75, "cagr": 8.42},
        {"name": "FTSE All-World", "ticker": "VWRL.L", "type": "Benchmark index",
         "1w": 0.92, "1m": 2.35, "ytd": 4.62, "1y": 9.40, "3y": 8.10, "5y": 9.55, "3m": 3.61, "cagr": 8.10},
        {"name": "Nasdaq 100", "ticker": "^NDX", "type": "Benchmark index",
         "1w": 1.85, "1m": 4.20, "ytd": 9.10, "1y": 18.42, "3y": 15.20, "5y": 18.55, "3m": 6.55, "cagr": 15.20},
    ]
    holding_perf = pd.DataFrame(hp_rows + bench_rows)

    # Benchmark comparison (used by smart insights to detect ACWI win)
    bench_cmp = pd.DataFrame([
        {"name": "S&P 500", "cagr": 10.85, "volatility": 15.8, "sharpe": 0.85},
        {"name": "MSCI ACWI", "cagr": 8.42, "volatility": 13.6, "sharpe": 0.81},
    ])

    return PortfolioMetrics(
        total_value=float(df["current_value"].sum()),
        invested_value=float(invested_total),
        cash_value=float(df.loc[df["asset_class"] == "Cash & Cash Equivalents", "current_value"].sum()),
        cash_target_eur=3000.0,
        holdings_df=df,
        allocation_by_class=alloc_class,
        allocation_by_geo=geo,
        acwi_geo=acwi_geo,
        performance={"1w": 1.05, "1m": 2.84, "ytd": 5.21, "1y": 9.40, "cagr": 8.59},
        performance_full={"1w": 1.05, "1m": 2.84, "3m": 4.12, "ytd": 5.21,
                          "1y": 9.40, "3y": 8.50, "5y": 9.10,
                          "cagr": 8.59, "volatility": 12.4, "sharpe": 0.92,
                          "sortino": 1.34, "max_drawdown": -8.7,
                          "var_95": -1.18, "cvar_95": -1.62, "alpha": 0.85, "beta": 0.78},
        risk={"volatility": 12.4, "sharpe": 0.92, "sortino": 1.34,
              "max_drawdown": -8.7, "var_95": -1.18, "cvar_95": -1.62,
              "alpha": 0.85, "beta": 0.78},
        goal_deltas=goal_df,
        rebalancing_suggestions=rebal,
        portfolio_history=history,
        benchmark_comparison=bench_cmp,
        holding_performance=holding_perf,
    )


def main() -> None:
    metrics = _build_fake_metrics()
    config = InvestorConfig.from_csv(str(ROOT / "input/sample/sample_targets.csv"))
    output_dir = ROOT / "output" / "sample"
    path = generate_newsletter(metrics, config, str(output_dir), issue_number=19)
    print(f"\n✓ Newsletter rendered to:\n  {path}\n")


if __name__ == "__main__":
    main()
