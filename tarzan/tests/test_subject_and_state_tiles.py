"""The subject line's 1D, and the two P&L tiles leading with the percentage.

Two requests, one property between them: the headline figure a reader sees first
should be the one that says how the book is DOING, and it must be the same number
the body prints. The euro amount grows with the book; a euro P&L answers nothing
without the capital behind it.
"""

from __future__ import annotations

import html as H
import re

import pandas as pd
import pytest

from tarzan import delivery
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_alloc import _build_hero
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics

_FIXED = delivery.datetime(2026, 7, 13, 19, 35,
                           tzinfo=delivery.ZoneInfo("Europe/Rome"))


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setattr(delivery, "now_local", lambda: _FIXED)


def _metrics(**kw) -> PortfolioMetrics:
    m = PortfolioMetrics(total_value=100_000.0, invested_value=100_000.0,
                         holdings_df=pd.DataFrame({"cost_basis_eur": [90_000.0]}))
    for k, v in kw.items():
        setattr(m, k, v)
    return m


class TestTheSubjectCarriesTheOneDayMove:
    def test_it_prints_the_portfolios_1d(self):
        m = _metrics(performance={"1d": 0.4237})
        assert delivery.build_subject(m, "Portfolio Digest") == \
            "Portfolio Digest - 19:35 - 1D +0.42%"

    def test_a_down_day_uses_the_minus_SIGN(self):
        """U+2212, like every other negative figure in the issue — not a hyphen,
        which is also the subject's own separator."""
        m = _metrics(performance={"1d": -1.267})
        subject = delivery.build_subject(m, "Portfolio Digest")
        assert subject == "Portfolio Digest - 19:35 - 1D −1.27%"
        assert "-1.27" not in subject

    def test_it_is_the_same_expression_the_state_session_tile_reads(self):
        """One number, one source. The tile and the subject both read
        ``performance["1d"]``, so they cannot describe different days."""
        m = _metrics(performance={"1d": 0.4237, "cagr": 5.0})
        subject = delivery.build_subject(m, "P")
        tiles = _build_hero(_NewsletterContext(
            metrics=m, config=InvestorConfig()))["tiles"]
        session = next(t for t in tiles if t["label"] == "Session")
        assert H.unescape(session["value"]) == "+0.42%"
        assert "+0.42%" in subject

    def test_no_1d_falls_back_and_says_so(self):
        """A holdings-only run has no order-derived NAV, and a book younger than
        two sessions has no previous session to anchor on. The subject keeps a
        number but relabels it, rather than calling a lifetime figure "1D"."""
        for perf in ({}, {"1d": None}, {"1d": float("nan")}):
            s = delivery.build_subject(_metrics(performance=perf), "P")
            assert s == "P - 19:35 - uP&L +11.11%", (perf, s)

    def test_a_zero_day_is_still_a_day(self):
        m = _metrics(performance={"1d": 0.0})
        assert delivery.build_subject(m, "P") == "P - 19:35 - 1D +0.00%"


class TestLiveWhenOpenPreviousCloseWhenNot:
    """The subject needs no open/closed branch — the SERIES carries it.

    ``current_session`` stamps today's market point onto every price history
    before anything reads a price, so while a venue is open the NAV's terminal
    point IS the live valuation and 1D measures against the previous session.
    With every venue shut, nothing is stamped, the terminal point is the last
    close, and the same expression measures the last completed session.
    """

    @staticmethod
    def _nav(closes):
        idx = pd.bdate_range("2026-06-01", periods=len(closes))
        return pd.Series(closes, index=idx, dtype=float)

    def _subject_pct(self, nav):
        from tarzan.engine.stats import compute_period_return
        m = _metrics(performance={"1d": compute_period_return(nav, "1d")},
                     portfolio_history=nav)
        return delivery.build_subject(m, "P")

    def test_a_live_terminal_point_moves_the_subject(self):
        closed = self._nav([100.0] * 20 + [101.0])
        live = self._nav([100.0] * 20 + [101.5])       # same session, price moved

        assert self._subject_pct(closed) == "P - 19:35 - 1D +1.00%"
        assert self._subject_pct(live) == "P - 19:35 - 1D +1.50%"

    def test_with_the_market_shut_it_is_the_last_completed_session(self):
        """The terminal point is Friday's close; the subject reads Friday's move
        whatever day it is generated on, because the window anchors on the
        previous SESSION rather than on 'yesterday'."""
        nav = self._nav([100.0] * 20 + [100.75])
        assert nav.index[-1].weekday() < 5
        assert self._subject_pct(nav) == "P - 19:35 - 1D +0.75%"


class TestThePnlTilesLeadWithThePercentage:
    @staticmethod
    def _tiles(**kw):
        m = _metrics(**kw)
        return {t["label"]: t for t in _build_hero(_NewsletterContext(
            metrics=m, config=InvestorConfig()))["tiles"]}

    def test_total_pnl_headlines_the_percentage_and_captions_the_euros(self):
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Total P&amp;L"]
        assert H.unescape(t["value"]) == "+10.00%"
        cap = H.unescape(t["caption"])
        assert cap.startswith("+€12.5k"), cap
        assert cap.endswith("on contributed capital"), cap

    def test_unrealized_pnl_headlines_the_percentage_too(self):
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Unrealized P&amp;L"]
        assert H.unescape(t["value"]).endswith("%")
        assert "€" in H.unescape(t["caption"])

    def test_the_euro_amount_is_not_lost(self):
        """It moved, it did not go away — it is still the first thing on the
        caption line."""
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Total P&amp;L"]
        assert "€" not in H.unescape(t["value"])
        assert "€" in H.unescape(t["caption"])

    def test_the_colour_follows_the_headline(self):
        """The tone is drawn on the number it is next to. A percentage and a euro
        amount can disagree in sign when contributed capital is negative (more
        withdrawn than paid in), and then colouring by the euros would paint the
        headline the wrong way."""
        pos = self._tiles(pnl_eur=-100.0, pnl_pct=3.0)["Total P&amp;L"]
        assert pos["tone"] == "pos", pos
        neg = self._tiles(pnl_eur=100.0, pnl_pct=-3.0)["Total P&amp;L"]
        assert neg["tone"] == "neg", neg

    def test_a_nan_percentage_falls_back_to_the_euros(self):
        """Never headline a "—".

        ``_pct`` renders NaN as an em dash, and a tile whose big number is a dash
        while the euro amount it could have shown sits in small type below is
        strictly worse than the old layout. None cannot reach here (the builder
        defaults the percentage to 0.0), but NaN can, and it is the case that
        would print the dash.
        """
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=float("nan"))["Total P&amp;L"]
        assert "€" in H.unescape(t["value"]), t
        assert "%" not in H.unescape(t["value"]), t
        assert H.unescape(t["caption"]) == "on contributed capital", t

    def test_the_portfolio_tile_still_leads_with_euros(self):
        """Only the two P&L tiles changed. "Portfolio" is a level, not a return —
        a percentage there would have no denominator to mean anything against."""
        t = self._tiles()["Portfolio"]
        assert H.unescape(t["value"]).startswith("€")


class TestTheRenderedTileMarkup:
    def test_the_percentage_is_in_the_display_type_and_the_euros_below(self):
        """End to end through the template, since the swap is only real if the
        big type carries the percentage in the actual document."""
        from tarzan.tests.test_newsletter_golden_html import GOLDEN_PATH
        html = GOLDEN_PATH.read_text()
        block = html.split("Total P&amp;L")[1][:600]
        # The display-type div comes first, then the prose caption.
        big = re.search(r'font-size:22px[^>]*>([^<]+)</div>', block)
        small = re.search(r'line-height:1\.5[^>]*>([^<]+)</div>', block)
        assert big and big.group(1).strip().endswith("%"), block[:300]
        assert small and "€" in small.group(1), block[:300]
