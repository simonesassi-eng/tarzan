"""Standalone preview of the newsletter "Backtesting" section.

Thin wrapper: runs the real backtest (:func:`tarzan.backtest.run_backtest`) and
renders the SAME production section (``_sections_backtest._render``) into a
standalone HTML page — no compute or rendering is re-implemented here. Use it to
eyeball the section without sending a newsletter.

Run:  python -m scripts.backtesting_preview   ->  output/backtesting_preview.html
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tarzan.backtest import run_backtest                                # noqa: E402
from tarzan.data.loader import load_config                             # noqa: E402
from tarzan.export.newsletter._constants import PALETTE                # noqa: E402
from tarzan.export.newsletter import _sections_backtest as sec         # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(str(ROOT / "input" / "targets.csv"))
    portfolios = run_backtest(ROOT / "input" / "portfolio_test.csv",
                              currency="eur", backfill="factor", rebalance="quarterly")
    if not portfolios:
        print("No portfolios could be built.")
        return 1
    tol = float(getattr(cfg, "rebalancing_target_tolerance_pctg", 1.5) or 1.5)
    body = sec._render(portfolios,
                       cfg.invested_allocation_targets_pctg or {},
                       cfg.equity_geo_targets_pctg or {}, tol)
    P = PALETTE
    html = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Backtesting preview</title></head>'
        f'<body style="margin:0;background:{P["page"]};font-family:-apple-system,'
        f'Segoe UI,Roboto,Helvetica,Arial,sans-serif;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr><td align="center" style="padding:24px 12px;">'
        f'<table role="presentation" width="760" cellpadding="0" cellspacing="0" border="0" '
        f'style="max-width:760px;background:#FFFFFF;border:1px solid {P["border"]};border-radius:16px;">'
        f'<tr><td style="padding:24px 32px 32px 32px;">'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.1em;'
        f'color:{P["subtle"]};text-transform:uppercase;">Optimizer &middot; preview</div>'
        f'<div style="font-size:22px;font-weight:800;color:{P["ink"]};margin-top:2px;">Backtesting</div>'
        f'{body}'
        f'</td></tr></table></td></tr></table></body></html>'
    )
    out = ROOT / "output" / "backtesting_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nBacktesting preview written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
