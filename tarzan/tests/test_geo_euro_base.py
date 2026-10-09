"""The euro amount beside a geography weight must use the sleeve the weight
is a share of.

Geography weights are shares of the NOTIONAL equity sleeve (a 2x fund
contributes 200% of its value, a 90/60 efficient-core fund 90%), so their
inline euro must be that same notional sleeve. Multiplying the notional share
by the physical market value instead mixed two denominators and made a region
read fewer euros than its sole holding was worth on its own — Emerging Markets
showed less than XMME, a pure-EM fund, was worth by itself.

Network-free: a hand-built PortfolioMetrics with one leveraged holding.
"""

from __future__ import annotations

import re

import pandas as pd

from tarzan.export.newsletter._sections_alloc import _build_diversification
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _levered_metrics() -> PortfolioMetrics:
    """One 2x US fund (€100k market) + one pure-EM fund (€20k market).

    Notional equity = 100k×2 + 20k = €220k. EM is €20k of that = 9.09%.
    The physical market sleeve is €120k. Pre-fix the renderer printed
    9.09% × 120k = €10.9k for EM — less than the €20k EM fund itself.
    """
    df = pd.DataFrame([
        {"isin": "US2X", "ticker": "CL2", "name": "US 2x Levered",
         "asset_class": "Equities", "current_value": 100000.0,
         "cost_basis_eur": 100000.0, "weight_pct": 90.91, "gain_pct": 0.0,
         "quantity": 1.0, "avg_purchase_price": 100000.0, "pct_of_class": 83.33,
         "currency": "EUR"},
        {"isin": "EMXX", "ticker": "XMME", "name": "EM Broad",
         "asset_class": "Equities", "current_value": 20000.0,
         "cost_basis_eur": 20000.0, "weight_pct": 9.09, "gain_pct": 0.0,
         "quantity": 1.0, "avg_purchase_price": 20000.0, "pct_of_class": 16.67,
         "currency": "EUR"},
    ])
    m = PortfolioMetrics(
        total_value=120000.0, invested_value=120000.0, cash_value=0.0,
        holdings_df=df,
        # Notional equity exposure is 220k / 120k invested = 183.33%.
        allocation_by_class=pd.DataFrame([
            {"category": "Equities", "weight_pct": 183.33},
        ]),
        # Geo weights are shares of the NOTIONAL sleeve: EM = 20k/220k.
        allocation_by_geo=pd.DataFrame([
            {"category": "USA", "weight_pct": 90.91},
            {"category": "Emerging Markets", "weight_pct": 9.09},
        ]),
    )
    return m


def _eur_to_float(token: str) -> float:
    """'17.5k' / '€120k' / '1,234' -> float euros."""
    t = token.replace("€", "").replace(",", "").strip()
    mult = 1000.0 if t.endswith("k") else 1.0
    return float(t.rstrip("k")) * mult


def test_geo_euro_uses_the_notional_sleeve_not_the_market_value():
    ctx = _NewsletterContext(
        metrics=_levered_metrics(),
        config=InvestorConfig(),
        issue_number=1,
        benchmark_alpha_beta="S&P 500",
        benchmark_geo="MSCI ACWI",
    )
    html = _build_diversification(ctx)["html"]

    # The Emerging Markets row's euro today must be at least what the sole EM
    # holding is worth — the invariant the bug violated. In the geography card a
    # label line reads 1M, euros, weight, name, so it is the last euro before the
    # name.
    geo = html[html.index('aria-label="Equity geography'):]
    geo = geo[:geo.index("</svg>")]
    euros = re.findall(r">€([\d.,]+k?)<", geo[:geo.index(">Emerging<")])
    assert euros, "no Emerging Markets euro rendered"
    em_eur = _eur_to_float(euros[-1])
    assert em_eur >= 20000.0 - 500.0, (
        f"Emerging Markets shows €{em_eur:.0f}, below the €20k its only "
        f"holding (XMME) is worth — the notional share was applied to the "
        f"physical market sleeve"
    )
    # And it equals the notional EM sleeve (9.09% × €220k), not 9.09% × €120k.
    assert abs(em_eur - 20000.0) < 800.0, em_eur


def test_the_plan_side_is_sized_on_the_plans_own_sleeve():
    """The PLAN column is the plan's sleeve, not today's. With Equities at 183.33%
    today and 150% in the plan, a region the plan puts at 50% holds 50% of the PLAN's
    sleeve; 50% of today's sleeve is not a figure the plan contains, and the plan
    column then did not add up to the asset-class card. (Synthetic book, above.)"""
    cfg = InvestorConfig()
    cfg.invested_allocation_targets_pctg = {"Equities": 150.0}
    cfg.equity_geo_targets_pctg = {"USA": 50.0, "Emerging Markets": 50.0}
    ctx = _NewsletterContext(metrics=_levered_metrics(), config=cfg, issue_number=1,
                             benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI")
    html = _build_diversification(ctx)["html"]
    geo = html[html.index('aria-label="Equity geography'):]
    geo = geo[:geo.index("</svg>")]
    plan_side = geo[geo.index(">USA<", geo.index(">Emerging<")):]   # right-hand labels
    euros = [_eur_to_float(e) for e in re.findall(r">€([\d.,]+k?)<", plan_side)]
    assert euros and abs(euros[0] - 90000.0) < 1000.0, euros
    assert "€220k notional today, €180k in the plan" in html   # synthetic
