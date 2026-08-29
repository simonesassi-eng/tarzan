"""The STATE "Session" tile's euro figure must agree with the % beside it.

Three things have to hold at once, and each was wrong at some point:

1. The euro figure is derived from ``performance["1d"]`` itself, not from a
   separately-walked window. The original code used ``_window_money_pnl(..., 1)``,
   which spans a CALENDAR day, while ``session_pct`` is the last-TRADING-day
   change; across a weekend those are different spans, which is how a -0.18%
   session printed a four-figure euro loss on the 2026-07-29 issue.
2. The base excludes cash. ``session_pct`` is a price-only return over the
   priced holdings (metrics._portfolio_history sums price_history x quantity),
   so cash contributes no price move to it.
3. The percentage is de-compounded. The base is an END-of-session value while
   the percentage is measured against the session's START, so the move is
   ``base * p/(1+p)``, not ``base * p``.

Network-free: metrics are constructed directly.
"""

from __future__ import annotations

import re

from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_hero
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _session_caption(total: float, invested: float, pct: float) -> str:
    m = PortfolioMetrics(
        total_value=total,
        invested_value=invested,
        cash_value=total - invested,
    )
    m.performance = {"1d": pct, "cagr": None}
    hero = _build_hero(_NewsletterContext(
        metrics=m, config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI",
    ))
    tile = next(t for t in hero["tiles"] if t["label"] == "Session")
    return re.sub(r"<[^>]+>", "", tile["caption"])


def test_session_euro_excludes_cash_from_its_base():
    """A round book with the 4% cash weight and the -0.175119% session that
    exposed this on the 2026-07-29 issue: EUR100,000 total, EUR4,000 of it cash.

    -0.175119% of the EUR96,000 priced sleeve is -EUR168. Billing it against
    the cash-inclusive total gives -EUR175 -- a EUR7 overstatement that grows
    with the cash weight (at a 20% buffer the gap is five times larger).
    """
    caption = _session_caption(100_000.0, 96_000.0, -0.175119)
    assert "−€168" in caption, caption
    assert "−€175" not in caption, \
        "the euro figure must not be billed against a cash-inclusive total"


def test_session_euro_is_decompounded_not_applied_to_the_end_value():
    """A +10% session on an ending EUR11,000 sleeve moved EUR1,000, not
    EUR1,100: the percentage is measured against the EUR10,000 it started
    from. Applying it straight to the ending value overstates every move."""
    caption = _session_caption(11_000.0, 11_000.0, 10.0)
    assert "+€1.0k" in caption or "+€1,000" in caption, caption


def test_session_euro_tracks_the_percentage_sign():
    assert _session_caption(100_000.0, 100_000.0, 1.0).startswith("+€")
    assert _session_caption(100_000.0, 100_000.0, -1.0).startswith("−€")


def test_session_euro_falls_back_to_total_when_nothing_is_invested():
    """A cash-only portfolio has no priced sleeve to bill against; the total
    is the only base available, and the tile must still render."""
    caption = _session_caption(50_000.0, 0.0, 0.5)
    assert "€" in caption, caption
