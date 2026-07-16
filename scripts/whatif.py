"""What-if analysis comparing one or more portfolios, side by side (CLI).

Thin console/CLI wrapper over :mod:`tarzan.backtest`: it evaluates every
portfolio in the weights CSV — portfolios are columns, metrics are rows —
reusing Tarzan's shared engine (enrichment, synthetic history, robustness,
allocations) with a finer, currency-matched, time-varying risk-free.

All compute lives in ``tarzan.backtest``; this module only parses arguments and
prints the console report. The Excel report is delegated to
``tarzan.export.whatif_excel``.

Usage:
    python -m scripts.whatif --weights input/portfolio_test.csv \
        --config input/targets.csv --excel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.backtest import (  # noqa: E402
    ASSET_ORDER, GEO_ORDER, backfill_label, instrument_ter, run_backtest,
    simulation_rows, testfol_instrument_map, testfol_lines,
)
from tarzan.data import proxy_data  # noqa: E402
from tarzan.data.loader import load_config  # noqa: E402
from tarzan.export._format import short_instrument_name  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("whatif")

_COLW = 13
_LABW = 26


# ---------------------------------------------------------------------------
# Console reporting (portfolios as columns)
# ---------------------------------------------------------------------------

def _present(order, portfolios, attr) -> list[str]:
    """Labels from ``order`` that are non-zero in at least one portfolio."""
    return [c for c in order
            if any(getattr(p, attr).get(c, 0.0) for p in portfolios)]


def _header(names, with_target=False) -> str:
    line = f"  {'':<{_LABW}}" + "".join(f"{n[:_COLW - 1]:>{_COLW}}" for n in names)
    if with_target:
        line += f"{'Target':>{_COLW}}"
    return line


def _matrix(title, labels, portfolios, attr, target=None) -> None:
    print("\n" + "-" * (2 + _LABW + _COLW * (len(portfolios) + (1 if target else 0))))
    print(title)
    names = [p.name for p in portfolios]
    print(_header(names, with_target=target is not None))
    for lbl in labels:
        row = f"  {lbl:<{_LABW}}"
        for p in portfolios:
            row += f"{getattr(p, attr).get(lbl, 0.0):>{_COLW - 1}.1f}%"
        if target is not None:
            row += f"{target.get(lbl, 0.0):>{_COLW - 1}.1f}%"
        print(row)


_METRIC_ROWS = [
    ("CAGR", "cagr", "%"), ("Volatility (ann.)", "volatility", "%"),
    ("Sharpe", "sharpe", ""), ("Sortino", "sortino", ""),
    ("Max Drawdown", "max_drawdown", "%"),
    ("VaR 95% (daily)", "var_95", "%"), ("CVaR 95% (daily)", "cvar_95", "%"),
    ("Beta vs S&P 500", "beta", ""), ("Alpha (ann.)", "alpha", "%"),
]


def _print_metrics_table(portfolios, attr: str, title: str) -> None:
    names = [p.name for p in portfolios]
    rf = next((getattr(p, attr, {}) or {}).get("risk_free")
              for p in portfolios if getattr(p, attr, None))
    rf_s = f" · risk-free {rf:.2f}%" if isinstance(rf, (int, float)) else ""
    print(f"\n{title}{rf_s}")
    print(_header(names))
    for label, key, unit in _METRIC_ROWS:
        row = f"  {label:<{_LABW}}"
        for p in portfolios:
            v = (getattr(p, attr, {}) or {}).get(key)
            s = "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.2f}{unit}"
            row += f"{s:>{_COLW}}"
        print(row)


def _print_metrics(portfolios) -> None:
    names = [p.name for p in portfolios]
    sep = "-" * (2 + _LABW + _COLW * len(names))
    w = next((p.window for p in portfolios if p.window), None)
    win = f"{w[0]:%Y-%m} → {w[1]:%Y-%m}" if w else "—"
    print("\n" + sep)
    print(f"PORTFOLIO METRICS — single aligned history ({win})")
    _print_metrics_table(portfolios, "metrics_aligned_eur", "EUR numeraire (unhedged)")
    _print_metrics_table(portfolios, "metrics_aligned_usd", "USD numeraire")


def _print_robustness(portfolios, rebalance: str) -> None:
    names = [p.name for p in portfolios]
    sep = "-" * (2 + _LABW + _COLW * len(names))

    def line(label, fn):
        print(f"  {label:<{_LABW}}" + "".join(f"{fn(p):>{_COLW}}" for p in portfolios))

    def roll(p, sub, key):
        v = p.rob.get(sub, {}).get(key)
        return "—" if v is None else f"{v * 100:.1f}%"

    def stress(p, scen, field):
        d = p.rob.get("stress", {}).get(scen, {})
        return "—" if not d.get("covered") else f"{d[field]:.1f}%"

    print("\n" + sep)
    print("ROBUSTNESS (same aligned history)")
    print(_header(names))
    line("Roll 1Y ret p05", lambda p: roll(p, "rolling1y", "p05"))
    line("Roll 1Y ret median", lambda p: roll(p, "rolling1y", "median"))
    line("Roll 1Y ret p95", lambda p: roll(p, "rolling1y", "p95"))
    line("1Y windows positive",
         lambda p: (lambda d: "—" if not d else f"{d['pct_positive']:.0f}%")(p.rob.get("rolling1y", {})))
    line("Roll 3Y ret p05", lambda p: roll(p, "rolling3y", "p05"))
    line("Roll 3Y ret median", lambda p: roll(p, "rolling3y", "median"))
    line("Roll 1Y Sharpe min–max",
         lambda p: (lambda d: "—" if not d else f"{d['min']:.2f}–{d['max']:.2f}")(p.rob.get("sharpe", {})))
    line("MC CAGR 1Y [p05/p95]",
         lambda p: (lambda d: "—" if not d else f"{d['p05']:.0f}/{d['p95']:.0f}%")(p.rob.get("bootstrap", {}).get("cagr", {})))
    line("MC MaxDD p05 (worst)",
         lambda p: (lambda d: "—" if not d else f"{d['p05']:.1f}%")(p.rob.get("bootstrap", {}).get("max_drawdown", {})))
    line("Dot-com maxDD", lambda p: stress(p, "Dot-com 2000-02", "max_drawdown"))
    line("GFC'08 maxDD", lambda p: stress(p, "GFC 2008", "max_drawdown"))
    line("COVID'20 maxDD", lambda p: stress(p, "COVID 2020", "max_drawdown"))
    line("2022 maxDD", lambda p: stress(p, "2022 rate shock", "max_drawdown"))
    ccy = getattr(proxy_data, "_TARGET_CCY", "EUR")
    print("  (single history = per-instrument splice: REAL fund returns where available,")
    print(f"   proxy-reconstructed before inception; {ccy}-based)")
    print(f"  (net of each fund's TER on the modeled base; {rebalance} rebalancing, costless)")


def _print_simulation_map(portfolios) -> None:
    ccy = getattr(proxy_data, "_TARGET_CCY", "EUR")
    print("\nSIMULATION MAP (per instrument — proxy used before the fund's real history)")
    print(f"  {'Ticker':<9}{'Real from':>10}{'Base from':>11}  Simulation base (proxy × exposure)")
    print("  " + "-" * 88)
    for r in simulation_rows(portfolios):
        print(f"  {r['ticker']:<9}{r['real_from']:>10}{r['base_from']:>11}  {r['base']}")


def _print_report(portfolios, asset_target, geo_target, anchor, rebalance) -> None:
    names = [p.name for p in portfolios]
    width = 2 + _LABW + _COLW * len(names)

    print("\n" + "=" * width)
    print("WHAT-IF PORTFOLIO COMPARISON")
    print("=" * width)
    print(f"  Notional anchor (display only): €{anchor:,.0f}")

    print("\nPORTFOLIO SPECS")
    print(_header(names))
    specs = [
        ("Instruments", lambda p: f"{len(p.items)}"),
        ("Gross exposure", lambda p: f"{p.gross:.0f}%"),
        ("Leverage", lambda p: f"{p.leverage:.2f}x"),
        ("Notional EUR", lambda p: f"{anchor * p.gross / 100.0:,.0f}"),
    ]
    for label, fn in specs:
        print(f"  {label:<{_LABW}}" + "".join(f"{fn(p):>{_COLW}}" for p in portfolios))

    print("\nINSTRUMENT WEIGHTS (%)")
    tickers = sorted({it.bare for p in portfolios for it in p.items})
    name_by: dict[str, str] = {}
    for p in portfolios:
        for it in p.items:
            name_by.setdefault(it.bare, short_instrument_name(it.holding.name or it.symbol, 44))
    _TW, _DW = 9, 46
    print(f"  {'Ticker':<{_TW}}{'Description':<{_DW}}"
          + "".join(f"{n[:_COLW - 1]:>{_COLW}}" for n in names))
    wmaps = [p.weights() for p in portfolios]
    for tk in tickers:
        row = f"  {tk:<{_TW}}{name_by.get(tk, ''):<{_DW}}"
        for wm in wmaps:
            row += (f"{wm[tk]:>{_COLW - 1}.1f}%" if tk in wm else f"{'·':>{_COLW}}")
        print(row)

    _print_simulation_map(portfolios)

    _matrix("ASSET ALLOCATION — FUNDED CAPITAL (incl. cash)",
            _present(ASSET_ORDER, portfolios, "cap"), portfolios, "cap", asset_target)
    _matrix("ASSET ALLOCATION — NOTIONAL EXPOSURE (mix, normalised)",
            _present(ASSET_ORDER, portfolios, "notl_mix"), portfolios, "notl_mix", asset_target)
    _matrix("ASSET ALLOCATION — NOTIONAL EXPOSURE (gross % of capital, leveraged)",
            _present(ASSET_ORDER, portfolios, "notl_gross"), portfolios, "notl_gross")
    _matrix("LEVERAGE BY CLASS (notional − funded, pp of capital)",
            _present(ASSET_ORDER, portfolios, "lev_by_class"), portfolios, "lev_by_class")
    _matrix("EQUITY GEOGRAPHY — NOTIONAL (% of equity sleeve)",
            _present(GEO_ORDER, portfolios, "geo_notl"), portfolios, "geo_notl", geo_target)
    _matrix("EQUITY GEOGRAPHY — CAPITAL (% of equity sleeve)",
            _present(GEO_ORDER, portfolios, "geo_cap"), portfolios, "geo_cap", geo_target)

    _print_metrics(portfolios)
    _print_robustness(portfolios, rebalance)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="What-if comparison of one or more portfolios")
    parser.add_argument("--weights", default="input/portfolio_test.csv")
    parser.add_argument("--config", default="input/targets.csv")
    parser.add_argument("--notional", type=float, default=100_000.0,
                        help="Nominal EUR anchor for the notional-exposure figures "
                             "(display only; does not affect any return/risk metric)")
    parser.add_argument("--excel", nargs="?", const="output/whatif.xlsx", default=None,
                        help="Write an Excel report (default path: output/whatif.xlsx)")
    parser.add_argument("--currency", choices=["eur", "usd"], default="eur",
                        help="Reporting currency for the synthetic backtest.")
    parser.add_argument("--backfill", choices=["naive", "calibrated", "factor"],
                        default="factor",
                        help="Pre-inception backfill (default 'factor').")
    parser.add_argument("--rebalance",
                        choices=["daily", "monthly", "quarterly", "semiannual",
                                 "annual", "none"],
                        default="quarterly",
                        help="Rebalance policy for the backtest (default 'quarterly').")
    args = parser.parse_args(argv)

    weights_path = ROOT / args.weights
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        return 1

    config = load_config(str(ROOT / args.config))
    asset_target = config.invested_allocation_targets_pctg
    geo_target = config.equity_geo_targets_pctg
    anchor = args.notional

    logger.info("Backtest basis: %s currency, %s backfill, %s rebalance, TER net",
                args.currency.upper(), args.backfill, args.rebalance)
    portfolios = run_backtest(weights_path, currency=args.currency,
                              backfill=args.backfill, rebalance=args.rebalance)
    if not portfolios:
        logger.error("No portfolios could be built from %s", weights_path)
        return 1

    _print_report(portfolios, asset_target, geo_target, anchor, args.rebalance)

    if args.excel:
        from tarzan.export.whatif_excel import export_whatif_excel
        out = ROOT / args.excel
        out.parent.mkdir(parents=True, exist_ok=True)
        export_whatif_excel(str(out), portfolios, asset_target, geo_target, anchor,
                            tolerance=config.rebalancing_target_tolerance_pctg,
                            sim_rows=simulation_rows(portfolios),
                            testfol={p.name: testfol_lines(p) for p in portfolios},
                            testfol_byinst={p.name: testfol_instrument_map(p)
                                            for p in portfolios})
        print(f"\n  Excel report: {out}")

    print("\n" + "=" * (2 + _LABW + _COLW * len(portfolios)))
    print("Done. Composition splits come from taxonomy exp_* — verify with the specs.")
    print("=" * (2 + _LABW + _COLW * len(portfolios)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
