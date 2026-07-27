"""Multi-horizon rolling + Monte Carlo report for the candidate portfolios (CLI).

Thin wrapper, no compute of its own: runs the shared backtest
(:func:`tarzan.backtest.run_backtest`) and renders
:func:`tarzan.engine.robustness.multi_horizon` — rolling 1/3/5/10/15-year
annualised-return distributions plus block-bootstrap Monte Carlo (CAGR /
max-drawdown percentiles and P(loss)) per portfolio.

Run:  python -m scripts.horizon_analysis [--portfolios NAME ...]
      -> output/horizon_analysis_<date>.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.backtest import run_backtest                      # noqa: E402
from tarzan.engine import robustness as rob                   # noqa: E402
from tarzan.data import proxy_data                            # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

_ROLL_ROWS = [("p5", "p05"), ("p25", "p25"), ("p50", "median"),
              ("p75", "p75"), ("p95", "p95")]
_MC_ROWS = [("CAGR p5", "cagr", "p05"), ("CAGR p25", "cagr", "p25"),
            ("CAGR p50", "cagr", "median"), ("CAGR p75", "cagr", "p75"),
            ("CAGR p95", "cagr", "p95"), ("MaxDD p5", "max_drawdown", "p05"),
            ("MaxDD p50", "max_drawdown", "median"),
            ("MaxDD p95", "max_drawdown", "p95")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="input/portfolio_test.csv")
    ap.add_argument("--portfolios", nargs="*", default=None,
                    help="subset of portfolio names (default: all)")
    args = ap.parse_args()

    portfolios = run_backtest(ROOT / args.weights, currency="eur",
                              backfill="factor", rebalance="quarterly")
    if args.portfolios:
        keep = {p.lower() for p in args.portfolios}
        portfolios = [p for p in portfolios if p.name.lower() in keep]
    if not portfolios:
        print("No portfolios."); return 1

    w = next(p.window for p in portfolios if p.window)
    rf = proxy_data.risk_free_annual(w[0], w[1])
    mh = {p.name: rob.multi_horizon(p.nav, rf_annual=rf) for p in portfolios}

    names = [p.name for p in portfolios]
    colw = max(13, max(len(n) for n in names) + 2)
    lines: list[str] = []
    out = lines.append
    out(f"MULTI-HORIZON ANALYSIS — aligned history {w[0]:%Y-%m} → {w[1]:%Y-%m}, "
        f"EUR, quarterly rebalance, risk-free {rf:.2f}%")
    out("Rolling = every overlapping start date in the real history (sequence risk as it happened).")
    out("MC = 2000 block-bootstrap sims (21d blocks) per horizon: calendar reshuffled, fat tails kept.")

    def header():
        return "  " + f"{'':<24}" + "".join(f"{n[-colw + 2:]:>{colw}}" for n in names)

    def row(label, fn, fmt="{:.1f}%"):
        s = f"  {label:<24}"
        for n in names:
            v = fn(mh[n])
            s += (f"{fmt.format(v):>{colw}}" if v is not None else f"{'—':>{colw}}")
        return s

    for yrs in rob.HORIZON_YEARS:
        out("")
        out(f"ROLLING {yrs}Y ANNUALISED RETURN (overlapping daily windows)")
        out(header())
        for label, key in _ROLL_ROWS:
            def _roll(m, k=key, y=yrs):
                v = m[y]["rolling"].get(k) if m[y]["rolling"] else None
                return v * 100 if v is not None else None
            out(row(label, _roll))
        out(row("% windows positive",
                lambda m, y=yrs: m[y]["rolling"].get("pct_positive") if m[y]["rolling"] else None,
                fmt="{:.0f}%"))
        out(row("# windows",
                lambda m, y=yrs: m[y]["rolling"].get("n") if m[y]["rolling"] else None,
                fmt="{:.0f}"))

    for yrs in rob.HORIZON_YEARS:
        out("")
        out(f"MONTE CARLO {yrs}Y (block bootstrap, 2000 sims)")
        out(header())
        for label, key, pct in _MC_ROWS:
            out(row(label, lambda m, k=key, p=pct, y=yrs: (
                m[y]["mc"][k][p] if m[y]["mc"] else None)))
        out(row("P(loss at horizon)",
                lambda m, y=yrs: m[y]["mc"].get("prob_loss") if m[y]["mc"] else None))

    text = "\n".join(lines)
    print(text)
    stamp = dt.date.today().strftime("%Y%m%d")
    dest = ROOT / "output" / f"horizon_analysis_{stamp}.txt"
    dest.write_text(text, encoding="utf-8")
    print(f"\nSaved to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
