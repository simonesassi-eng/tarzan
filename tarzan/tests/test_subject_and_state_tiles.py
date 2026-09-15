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


class TestTheSubjectNamesWhatItShows:
    """The FIGURE needs no open/closed branch; the LABEL does.

    ``current_session`` stamps today's market point onto every price history before
    anything reads a price, so while a venue is open the NAV's terminal point IS the
    live valuation and 1D measures against the previous session. With every venue
    shut, nothing is stamped and the same expression measures the last completed
    session. One expression, both cases — that part never needed a branch.

    What did need one is the word in front of it. There is a third state the old
    reasoning missed and it is the one the reader meets every morning: the venue is
    OPEN and no bar exists yet. At 09:12 on Tue 15 Sep 2026 every tape ended on Mon
    14 Sep, so "1D −0.59%" was Monday's completed session under today's name.
    """

    @staticmethod
    def _nav(closes):
        idx = pd.bdate_range("2026-06-01", periods=len(closes))
        return pd.Series(closes, index=idx, dtype=float)

    def _subject(self, nav, **perf):
        from tarzan.engine.stats import compute_period_return
        m = _metrics(performance={"1d": compute_period_return(nav, "1d"), **perf},
                     portfolio_history=nav)
        return delivery.build_subject(m, "P")

    def test_the_figure_follows_the_terminal_point(self):
        closed = self._nav([100.0] * 20 + [101.0])
        live = self._nav([100.0] * 20 + [101.5])       # same session, price moved
        assert "+1.00%" in self._subject(closed, **{"1d_intraday": True})
        assert "+1.50%" in self._subject(live, **{"1d_intraday": True})

    def test_an_intraday_figure_is_labelled_1d(self):
        nav = self._nav([100.0] * 20 + [101.0])
        assert self._subject(nav, **{"1d_intraday": True}) == "P - 19:35 - 1D +1.00%"

    def test_a_completed_session_is_labelled_with_its_DATE(self):
        """Not "1D". The fixture's tape ends Mon 29 Jun, so that is what the subject
        says — and a reader opening it on the 30th cannot mistake it for the 30th."""
        nav = self._nav([100.0] * 20 + [100.75])
        assert nav.index[-1].strftime("%d %b") == "29 Jun"
        assert self._subject(nav) == "P - 19:35 - 29 Jun +0.75%"

    def test_it_falls_back_to_1d_when_the_session_cannot_be_named(self):
        """No history to read a session date off. Better an unqualified "1D" than a
        date the series does not support."""
        m = _metrics(performance={"1d": 0.5})
        assert delivery.build_subject(m, "P") == "P - 19:35 - 1D +0.50%"

    def test_no_1d_at_all_still_relabels_to_upnl(self):
        """``unrealized_pnl_pct`` is a derived property, so the fixture sets what it
        is derived FROM: value 100k on a 90k cost basis is +11.11%."""
        m = _metrics(performance={"1d": None})
        assert delivery.build_subject(m, "P") == "P - 19:35 - uP&L +11.11%"


class TestThePnlTilesLeadWithTheEuros:
    """The money made is the headline; the rate it was made at is the caption.

    This ran the other way for a while, on the reasoning that a euro P&L answers
    nothing without the capital behind it while a percentage compares to everything
    else in the issue. Both are true and it is still the wrong way round for a P&L —
    and the issue is full of rates elsewhere: TWR, MWR and CAGR all lead with one.
    """

    @staticmethod
    def _tiles(**kw):
        m = _metrics(**kw)
        return {t["label"]: t for t in _build_hero(_NewsletterContext(
            metrics=m, config=InvestorConfig()))["tiles"]}

    def test_total_pnl_headlines_the_euros_and_captions_the_percentage(self):
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Total P&amp;L"]
        assert H.unescape(t["value"]) == "+€12.5k"   # synthetic
        cap = H.unescape(t["caption"])
        assert cap.startswith("+10.00%"), cap
        assert cap.endswith("on contributed capital"), cap

    def test_unrealized_pnl_headlines_the_euros_too(self):
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Unrealized P&amp;L"]
        assert "€" in H.unescape(t["value"])
        assert "%" in H.unescape(t["caption"])

    def test_the_percentage_is_not_lost(self):
        """It moved, it did not go away — it is still the first thing on the
        caption line."""
        t = self._tiles(pnl_eur=12_500.0, pnl_pct=10.0)["Total P&amp;L"]
        assert "%" not in H.unescape(t["value"])
        assert "%" in H.unescape(t["caption"])

    def test_the_colour_follows_the_headline(self):
        """The tone is drawn on the number it is next to. A percentage and a euro
        amount can disagree in sign when contributed capital is negative (more
        withdrawn than paid in), and now that the euros are the headline the colour
        follows THEM."""
        pos = self._tiles(pnl_eur=100.0, pnl_pct=-3.0)["Total P&amp;L"]
        assert pos["tone"] == "pos", pos
        neg = self._tiles(pnl_eur=-100.0, pnl_pct=3.0)["Total P&amp;L"]
        assert neg["tone"] == "neg", neg

    def test_a_nan_euro_amount_falls_back_to_the_percentage(self):
        """Never headline a "—".

        ``_eur_smart`` cannot render a NaN as a figure, and a tile whose big number is
        a dash while the percentage it could have shown sits in small type below is
        strictly worse than either layout.
        """
        t = self._tiles(pnl_eur=float("nan"), pnl_pct=10.0)["Total P&amp;L"]
        assert "%" in H.unescape(t["value"]), t
        assert H.unescape(t["caption"]) == "on contributed capital", t

    def test_the_portfolio_tile_still_leads_with_euros(self):
        """It always did: "Portfolio" is a level, not a return, and a percentage there
        would have no denominator to mean anything against."""
        t = self._tiles()["Portfolio"]
        assert H.unescape(t["value"]).startswith("€")


class TestTheRenderedTileMarkup:
    def test_the_euros_are_in_the_display_type_and_the_percentage_below(self):
        """End to end through the template, since the swap is only real if the big
        type carries the euro amount in the actual document."""
        from tarzan.tests.test_newsletter_golden_html import GOLDEN_PATH
        html = GOLDEN_PATH.read_text()
        block = html.split("Total P&amp;L")[1][:600]
        # The display-type div comes first, then the prose caption.
        big = re.search(r'font-size:22px[^>]*>([^<]+)</div>', block)
        small = re.search(r'line-height:1\.5[^>]*>([^<]+)</div>', block)
        assert big and "€" in big.group(1), block[:300]
        assert small and "%" in small.group(1), block[:300]


# ======================================================================
# The two return measures, each in both forms
# ======================================================================

def _return_metrics(**kw) -> PortfolioMetrics:
    """A book with both return measures and a sub-year span.

    262 calendar days, so the cumulative figure is SMALLER than the annualized one
    for both measures — which is the direction that catches a tile plotting the
    wrong form. All figures synthetic.
    """
    import datetime

    idx = pd.date_range("2025-12-23", "2026-09-11", freq="B")
    m = _metrics(**kw)
    m.portfolio_history = pd.Series(
        [100.0 + 10.89 * i / (len(idx) - 1) for i in range(len(idx))], index=idx)
    # ``_mwr_line`` re-solves the IRR at each point off the real euro value, so the
    # chart-vs-tile test below needs this series too.
    m.actual_value_series = pd.Series(
        [100.0 + 10.0 * i / (len(idx) - 1) for i in range(len(idx))], index=idx)
    m.twr_pct = 10.89
    m.twr_annualized_pct = 15.50
    m.xirr_pct = 12.45
    m.xirr_net_tax_pct = 11.20
    m.xirr_cashflows = [(datetime.date(2025, 12, 23), -100.0),
                        (datetime.date(2026, 9, 11), 110.0)]
    return m


def _tiles_by_label(m) -> dict:
    return {H.unescape(str(t["label"])): t for t in _build_hero(
        _NewsletterContext(metrics=m, config=InvestorConfig()))["tiles"]}


class TestBothReturnMeasuresAppearInBothForms:
    """Four cells, not two.

    These were two tiles that disagreed about which form leads: TWR headlined the
    cumulative figure with the annualized one in its caption, MWR headlined the
    annualized figure. So the two numbers a reader sees first answered different
    questions, and money-weighted-against-time-weighted — the one comparison that
    says whether the timing of contributions helped — could not be read at all.
    """

    def test_all_four_cells_exist(self):
        labels = _tiles_by_label(_return_metrics())
        for want in ("TWR since inception", "TWR annualized",
                     "MWR since inception", "MWR annualized"):
            assert want in labels, sorted(labels)

    def test_each_cell_headlines_its_own_form(self):
        t = _tiles_by_label(_return_metrics())
        assert H.unescape(t["TWR since inception"]["value"]) == "+10.89%"
        assert H.unescape(t["TWR annualized"]["value"]) == "+15.50%"
        assert H.unescape(t["MWR annualized"]["value"]) == "+12.45%"
        # The cumulative MWR is xirr_pct de-annualized over the 262-day span:
        # (1.1245 ** (262/365.25) - 1) = +8.78%.
        assert H.unescape(t["MWR since inception"]["value"]) == "+8.78%"

    def test_the_cumulative_figures_are_below_the_annualized_ones(self):
        """The book is younger than a year, so an annual RATE overstates what it
        actually made. A tile showing +12.45% under "since inception" would be the
        un-de-annualized bug this split exists to prevent."""
        t = _tiles_by_label(_return_metrics())

        def val(label):
            return float(H.unescape(t[label]["value"]).replace("+", "").rstrip("%"))

        assert val("TWR since inception") < val("TWR annualized")
        assert val("MWR since inception") < val("MWR annualized")

    def test_the_cumulative_mwr_is_the_figure_the_chart_line_ends_on(self):
        """One helper behind both, so the tile and the since-inception chart's MWR
        end label cannot state two numbers for one measure."""
        from tarzan.export._perf_series import _mwr_line, mwr_period_pct

        m = _return_metrics()
        tile = float(H.unescape(_tiles_by_label(m)["MWR since inception"]["value"])
                     .replace("+", "").rstrip("%"))
        line = _mwr_line(m, list(m.portfolio_history.index))
        assert line is not None
        assert round(line[-1], 2) == tile
        assert round(mwr_period_pct(m), 2) == tile

    def test_the_captions_name_the_measure_and_the_span(self):
        t = _tiles_by_label(_return_metrics())
        assert "262 days" in H.unescape(t["TWR since inception"]["caption"])
        assert "262 days" in H.unescape(t["MWR since inception"]["caption"])
        assert "per year" in H.unescape(t["TWR annualized"]["caption"])
        # Net of tax is an ANNUALIZED XIRR, so it belongs on the annualized cell.
        assert "+11.20% net of tax" in H.unescape(t["MWR annualized"]["caption"])
        # Symmetric pairs: same shape either side of the 2x2, so a caption cannot
        # grow past the column and wrap between a figure and its unit.
        assert H.unescape(t["TWR since inception"]["caption"]) == \
            "time-weighted \u00b7 262 days"
        assert H.unescape(t["MWR since inception"]["caption"]) == \
            "money-weighted \u00b7 262 days"


class TestCagrIsAFallbackNotADuplicate:
    def test_no_cagr_tile_when_the_annualized_twr_is_there(self):
        """They are the same number to the last float bit on the order path — both
        annualize one cumulative return read off one series (pinned end to end by
        ``test_golden_master.test_the_two_annualizations_agree``). Two cells for one
        fact, so the CAGR cell goes."""
        labels = _tiles_by_label(_return_metrics(performance={"cagr": 15.50}))
        assert "TWR annualized" in labels
        assert "CAGR" not in labels

    def test_cagr_appears_when_there_is_no_annualized_twr(self):
        """The holdings-only path: ``_returns`` is appended to the computer list ONLY
        when an order list is supplied, so TWR is None there while
        ``performance.cagr`` still computes off the fixed-basket history. Dropping the
        tile outright would leave that path with no annualized return at all.
        """
        m = _metrics(performance={"cagr": 7.25})
        labels = _tiles_by_label(m)
        assert "TWR annualized" not in labels
        assert H.unescape(labels["CAGR"]["value"]) == "+7.25%"
