"""A book with nothing left in it is a state, not an error.

Every effective order netting to zero used to be reported three different ways by
one run: the log called it bad input ("Order list produced no holdings. / No
portfolio value computed. Check input data."), the exit code said 1, no issue was
rendered — and summary.json still said ``SEND_NORMAL``. The realized gain on the
closed round trips, which is the only thing such a book has to report, went with it.

What these pin: the realized P&L reaches the issue, and no tile claims 0.00% on a
book that made money. The percentage is the HEADLINE of a P&L tile, so a coerced
zero put a prominent "0.00%" beside a caption reading "+€5.1k" — the big number
saying the book had made nothing while the small one said what it had really made.

Zero EFFECTIVE ORDERS is a different state and stays an input error.
"""

from __future__ import annotations

import pandas as pd

from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_hero
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _liquidated(pnl_eur=5050.34, twror=36.79) -> PortfolioMetrics:
    """A book holding nothing, with a real realized gain behind it."""
    m = PortfolioMetrics(total_value=0.0, invested_value=0.0, cash_value=0.0,
                         holdings_df=pd.DataFrame())
    m.pnl_eur = pnl_eur
    # None on purpose, and not a defect: pnl_pct is measured over NET capital
    # contributed, which is negative once everything has been withdrawn, so the
    # ratio is genuinely undefined. TWROR is the percentage that still means
    # something — the return over the periods the money was actually invested.
    m.pnl_pct = None
    m.twror_pct = twror
    m.inception_date = "2025-03-04"
    return m


def _tile(m: PortfolioMetrics, label: str) -> dict:
    """One STATE tile by label. Labels arrive HTML-escaped ("Total P&amp;L")."""
    from html import unescape

    tiles = _build_hero(_NewsletterContext(metrics=m, config=InvestorConfig()))["tiles"]
    for t in tiles:
        if unescape(str(t.get("label", ""))) == label:
            return t
    raise AssertionError(f"no {label!r} tile rendered; got "
                         f"{[unescape(str(t.get('label', ''))) for t in tiles]}")


class TestALiquidatedBookReportsWhatItMade:
    def test_the_realized_gain_is_the_headline(self):
        t = _tile(_liquidated(), "Total P&L")
        assert "5.1k" in t["value"], \
            f"the realized gain is not the headline; tile reads {t['value']!r}"

    def test_no_tile_claims_zero_percent(self):
        for label in ("Total P&L", "Unrealized P&L"):
            t = _tile(_liquidated(), label)
            assert "0.00%" not in t["value"], \
                f"{label} headlines 0.00% on a book that realized a gain"

    def test_twror_still_states_the_percentage_that_means_something(self):
        assert "+36.79%" in _tile(_liquidated(), "TWROR")["value"]

    def test_nothing_open_means_no_unrealized_percentage(self):
        """No cost basis, so the ratio is not applicable — the tile falls back to
        the euro figure rather than asserting break-even."""
        t = _tile(_liquidated(), "Unrealized P&L")
        assert "%" not in t["value"], f"tile reads {t['value']!r}"

    def test_a_normal_book_still_headlines_the_percentage(self):
        """Guards against over-correcting: dropping the ``or 0.0`` must not stop a
        book WITH a cost basis from leading with its percentage."""
        df = pd.DataFrame([{
            "isin": "US0000000001", "ticker": "AAA", "name": "Alpha",
            "asset_class": "Equities", "current_value": 6000.0,
            "cost_basis_eur": 5000.0, "weight_pct": 100.0, "gain_pct": 20.0,
            "gain_eur": 1000.0, "quantity": 100.0, "avg_purchase_price": 50.0,
            "pct_of_class": 100.0, "currency": "EUR",
        }])
        m = PortfolioMetrics(total_value=6000.0, invested_value=6000.0,
                             cash_value=0.0, holdings_df=df)
        m.pnl_eur = 1000.0
        m.pnl_pct = 20.0
        assert _tile(m, "Total P&L")["value"] == "+20.00%"
        assert "+20.00%" in _tile(m, "Unrealized P&L")["value"]


class TestZeroEffectiveOrdersIsStillAnInputError:
    """The line between the two states: the liquidated book has six real orders
    that net to zero; an empty or mis-dated order list has none, and that IS bad
    input — both ``if not orders`` branches in the orchestrator still return early.
    """

    def test_the_two_states_are_distinguishable_by_order_count(self):
        from tarzan.stress import generate

        book = {b.pid: b for b in generate.build_all()}["P09"]
        assert book.truth.order_count == 6, "the fixture is meant to have real orders"
        assert all(abs(q) < 1e-9 for q in book.truth.quantity_by_isin.values()), \
            "the fixture is meant to net to zero"
