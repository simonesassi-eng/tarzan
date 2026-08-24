"""The MARKETS section must show each instrument's trading hours and
whether its exchange is open right now, not just level/change/sparkline.

Before this, a reader saw a mix of markets on different session clocks
(US/Europe/Asia/futures/FX) with no stated hours, no way to tell whether a
level was live or from a closed session, and (for "Open"/"Closed" itself)
no stated day to pin the status to. Network-free: quotes are injected
directly, and market_status is monkeypatched so the open/closed/day
outcome does not depend on when the suite happens to run.
"""

from __future__ import annotations

from unittest import mock

import tarzan.data.market_quotes as mq
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_perf import _build_markets
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _ctx() -> _NewsletterContext:
    return _NewsletterContext(
        metrics=PortfolioMetrics(total_value=1.0, invested_value=1.0, cash_value=0.0),
        config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI",
    )


_SAMPLE = [
    {"name": "S&P 500", "symbol": "^GSPC", "category": "US",
     "value": 5000.0, "change": 10.0, "pct": 0.2,
     "spark": [4990.0, 5000.0], "baseline": 4990.0},
    {"name": "S&P 500 (FUT)", "symbol": "ES=F", "category": "US",
     "value": 5010.0, "change": 12.0, "pct": 0.24,
     "spark": [4998.0, 5010.0], "baseline": 4998.0},
]


def test_cash_index_shows_hours_open_badge_and_day():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           return_value=(True, "Tue")):
        html = _build_markets(_ctx())["html"]
    assert "09:30\u201316:00 ET" in html
    assert "&#9679;</span> Op. Tue" in html


def test_cash_index_shows_closed_badge_and_day():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           return_value=(False, "Fri")):
        html = _build_markets(_ctx())["html"]
    assert "&#9679;</span> Cl. Fri" in html


def test_chg_column_shows_only_percent_not_the_absolute_value():
    # Saves the width the day label now needs: +0.20% shows, the +10.0
    # points/€ that used to sit underneath it does not.
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status", return_value=(True, "Mon")):
        html = _build_markets(_ctx())["html"]
    assert "+0.20%" in html
    assert "+10.0" not in html
    assert "+12.0" not in html


def test_futures_show_open_status_and_day_not_the_cash_hours():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           side_effect=lambda sym, now=None: (
                               (True, "Tue") if sym == "^GSPC"
                               else (True, "Mon"))):
        html = _build_markets(_ctx())["html"]
    # "S&P 500 (FUT)" gets its own \u224824h line and its own day/status,
    # not the cash index's hours or its day, and the FUT tag is not
    # doubled by the render-time auto-suffix (name already carries it).
    fut_pos = html.index("S&P 500 (FUT)")
    fut_chunk = html[fut_pos:fut_pos + 250]
    assert "\u224824h" in fut_chunk
    assert "Op. Mon" in fut_chunk
    assert "09:30" not in fut_chunk  # not the cash index's hours
    assert "(FUT) (FUT)" not in html


def test_futures_show_closed_with_day_over_the_weekend():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           side_effect=lambda sym, now=None: (
                               (False, "Fri") if sym == "ES=F"
                               else (False, "Fri"))):
        html = _build_markets(_ctx())["html"]
    fut_pos = html.index("S&P 500 (FUT)")
    fut_chunk = html[fut_pos:fut_pos + 250]
    assert "Cl. Fri" in fut_chunk


def test_daily_close_fallback_chart_is_labelled_no_intraday():
    # No timestamped spark_series (as happens for some exchanges, mainland
    # China/Hong Kong among them, per a real Yahoo data gap) means _spark_for
    # falls back to the daily-close history. Drawn the same way as a real
    # session path, that used to be indistinguishable from one -- exactly
    # what read as "the market just opened but the chart is already full".
    # This is the one place the label survives: a POPULATED path is the thing a
    # dashed placeholder is not, so nothing but the caption says the 40 points
    # under it are daily closes. Hence the closed venue -- while it trades
    # there is no chart at all, only the dash.
    sample = [{"name": "SSE Composite", "symbol": "000001.SS",
               "category": "Asia", "value": 3200.0, "change": 5.0,
               "pct": 0.16, "spark": [3150.0 + i for i in range(40)],
               "baseline": 3195.0}]  # no "spark_series" key at all
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=sample), \
         mock.patch.object(mq, "market_status", return_value=(False, "Mon")):
        html = _build_markets(_ctx())["html"]
    assert "<polyline" in html      # the daily-closes chart really is drawn
    assert "no intraday" in html    # and it says what those closes are


def test_real_intraday_chart_is_not_labelled_no_intraday():
    import pandas as pd
    intra = pd.Series(
        [3190.0, 3193.0, 3196.0],
        index=pd.to_datetime(["2026-08-03 09:30", "2026-08-03 09:45",
                              "2026-08-03 10:00"]),
    )
    sample = [{"name": "Nikkei 225", "symbol": "^N225", "category": "Asia",
               "value": 3196.0, "change": 6.0, "pct": 0.19,
               "spark": [3190.0, 3193.0, 3196.0], "baseline": 3190.0,
               "spark_series": intra}]
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=sample), \
         mock.patch.object(mq, "market_status", return_value=(True, "Mon")):
        html = _build_markets(_ctx())["html"]
    assert "no intraday" not in html


class TestMoversIgnoreInstrumentsWithoutAWeek:
    """The best/worst cards must rank only instruments that HAVE a 5D.

    ``sort_values(..., na_position="last")`` puts the missing rows at the end,
    and "worst" is read off the end — so a holding with no 5D (a position opened
    this week, a feed with under two closes) became the worst performer. The
    figure then went through ``float(row.get("5d") or 0.0)``, and NaN is truthy,
    so the card rendered an em dash in red while the real worst performer was
    never shown.
    """

    def _movers(self, rows):
        import pandas as pd
        from tarzan.export.newsletter._sections_perf import _build_movers

        metrics = PortfolioMetrics(
            total_value=300.0, invested_value=300.0, cash_value=0.0,
            holdings_df=pd.DataFrame([
                {"ticker": r["ticker"], "asset_class": "Equities",
                 "current_value": 100.0} for r in rows]),
        )
        metrics.holding_performance = pd.DataFrame(rows)
        return _build_movers(_NewsletterContext(
            metrics=metrics, config=InvestorConfig()))

    def test_a_holding_with_no_five_day_is_not_the_worst_performer(self):
        out = self._movers([
            {"ticker": "AAA", "name": "Up", "type": "In portfolio", "5d": 3.0},
            {"ticker": "BBB", "name": "Down", "type": "In portfolio", "5d": -7.5},
            {"ticker": "CCC", "name": "New", "type": "In portfolio",
             "5d": float("nan")},
        ])
        assert out["available"] is True
        assert out["best"]["ticker"] == "AAA"
        assert out["worst"]["ticker"] == "BBB", "the real worst, not the NaN row"
        assert "7.5" in out["worst"]["pct"]
        assert out["worst"]["is_positive"] is False

    def test_no_holding_has_a_five_day_means_unavailable(self):
        out = self._movers([
            {"ticker": "AAA", "name": "New", "type": "In portfolio",
             "5d": float("nan")},
        ])
        assert out["available"] is False


class TestInstrumentNamesAreEscaped:
    """Provider/broker text becomes markup in ``uni_name``, so it escapes there.

    Ten curated names carry an ampersand — "iShares Core S&P 500", "L&G
    Multi-Strategy", the whole Return Stacked family — and they reached the
    document as a bare "&". Mail clients tolerate that, so it rendered fine and
    the invalid markup went unnoticed; a name with an angle bracket would not
    have. The digest template is ``.html.j2``, an extension
    ``select_autoescape()`` does not match, so Jinja escapes nothing either:
    this is the only place it can be done once for every table.
    """

    def test_an_ampersand_in_a_name_is_escaped(self):
        from tarzan.export.newsletter._constants import uni_name

        html = uni_name("iShares Core S&P 500 UCITS ETF", "SXR8")
        assert "S&amp;P" in html
        assert "S&P" not in html.replace("&amp;", "&amp;")

    def test_markup_in_a_name_cannot_break_out(self):
        from tarzan.export.newsletter._constants import uni_name

        html = uni_name("<script>x</script>", "AAA")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_the_ticker_and_tags_are_escaped_too(self):
        from tarzan.export.newsletter._constants import ticker_span, uni_name

        assert "&amp;" in ticker_span("A&B")
        tagged = uni_name("Name", "AAA", tags=(("α&β", "#000", "#fff"),))
        assert "α&amp;β" in tagged
