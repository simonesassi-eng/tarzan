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

import pathlib
import re
from unittest import mock

import tarzan.data.market_quotes as mq
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_perf import _build_markets
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _ctx(quotes=None) -> _NewsletterContext:
    return _NewsletterContext(
        metrics=PortfolioMetrics(
            total_value=1.0, invested_value=1.0, cash_value=0.0,
            market_quotes=tuple(_SAMPLE if quotes is None else quotes),
        ),
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
    import datetime as dt
    mq._memo = None
    winter = dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc)
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "_intraday_reference_now", lambda tzinfo=None: winter), \
         mock.patch.object(mq, "market_status",
                           return_value=(True, "Tue")):
        out = _build_markets(_ctx())
    # New York's session on the reader's clock (the zone is named once, in the
    # section subtitle). The status and its day are said once in the subtitle
    # when every row shares them, and on the row only when its own differs.
    assert "15:30\u201322:00" in out["html"]
    assert "all Op. Tue" in out["sub"]
    assert "Op.&nbsp;Tue" not in out["html"]


def test_cash_index_shows_closed_badge_and_day():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           return_value=(False, "Fri")):
        out = _build_markets(_ctx())
    assert "all Cl. Fri" in out["sub"]


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
    # The future (second "S&P 500" row) gets its own Globex window and its own
    # day/status, not the cash index's hours or its day; the board shows no
    # "(FUT)" tag (the heading says it).
    fut_pos = html.rindex("S&amp;P 500")
    fut_chunk = html[fut_pos:fut_pos + 500]
    assert "\u201323:00" in fut_chunk or "\u201322:00" in fut_chunk  # Globex, reader's clock
    # The cash index's "Op. Tue" is the status the subtitle states; the future's
    # own day differs, so its row says so.
    assert "Op.&nbsp;Mon" in fut_chunk
    assert "15:30" not in fut_chunk  # not the cash index's hours
    assert "(FUT)" not in html


def test_futures_show_closed_with_day_over_the_weekend():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           side_effect=lambda sym, now=None: (
                               (False, "Fri") if sym == "ES=F"
                               else (False, "Fri"))):
        out = _build_markets(_ctx())
    # Both rows closed since Friday: said once, for the board.
    assert "all Cl. Fri" in out["sub"]
    assert "Cl.&nbsp;Fri" not in out["html"]


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
        out = _build_markets(_ctx(sample))
    assert "<polyline" in out["html"]      # the daily-closes chart really is drawn
    # ...and it says what those closes are: on the row, or once in the subtitle
    # when every row (here the only one) carries the same caption.
    assert "no intraday" in out["html"] + out["sub"]


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
        html = _build_markets(_ctx(sample))["html"]
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


class TestEveryMarkupBoundaryEscapes:
    """Each builder that turns text into markup escapes it, fed text with an
    ampersand directly.

    The whole-document guard in test_newsletter_golden_html only has teeth on
    what its synthetic fixture happens to contain — it passed while the real
    digest still carried four raw ampersands, because that fixture has no cash
    class header and no rebalancing plan label. These probe the boundaries
    themselves, so coverage does not depend on the fixture's shape.

    "Cash & Cash Equivalents" is the AssetClass enum value and "Buy & sell
    (full rebalance)" a plan label: both are DATA elsewhere (Excel, the JSON
    summary), so they are escaped here, where they become HTML, not at source.
    """

    # Same allowlist as the whole-document guard in test_newsletter_golden_html.
    _RAW = re.compile(
        r"&(?!(?:amp|lt|gt|quot|apos|nbsp|middot|rsquo|lsquo|ndash|mdash|"
        r"minus|times|hellip|bull|deg|euro|copy|#\d+|#x[0-9a-fA-F]+);)"
    )

    def _clean(self, html: str) -> bool:
        return not self._RAW.search(html)

    def test_the_instrument_label_escapes(self):
        from tarzan.export.newsletter._constants import ticker_span, uni_name
        assert self._clean(uni_name("iShares Core S&P 500", "SXR8"))
        assert self._clean(ticker_span("A&B"))

    def test_the_table_group_header_escapes(self):
        """The class/role header carries the asset-class name verbatim."""
        from tarzan.export.newsletter._constants import (
            render_unified_table, uni_cell,
        )
        html = render_unified_table(
            "Instrument", [("1D", "right", 60)],
            [("Cash & Cash Equivalents", "#15803D",
              [("Cash / Money & Market",
                [{"name_html": "x", "cells": [uni_cell("1.0%")]}])])])
        assert self._clean(html), html[:200]

    def test_the_optimizer_plan_label_escapes(self):
        """Rendered by the template, so it carries an explicit |e filter."""
        from jinja2 import Environment
        tmpl = (pathlib.Path("tarzan/export/templates/portfolio_digest.html.j2")
                .read_text(encoding="utf-8"))
        assert "{{ plan.label|e }}" in tmpl, (
            "the explicit plan-label text boundary must remain escaped"
        )
        assert self._clean(Environment(autoescape=False)
                           .from_string("{{ v|e }}")
                           .render(v="Buy & sell (full rebalance)"))

    def test_the_allocation_bridge_label_escapes(self):
        from tarzan.export.newsletter._sections_alloc import _bridge_row, _bridge_svg
        from tarzan.models.investor_config import InvestorConfig as _Cfg
        row = _bridge_row(key="c", label="Cash & Cash Eq.", colour="#15803D",
                          now=50.0, target=50.0, base=1000.0)
        assert self._clean(_bridge_svg([row], [row], _Cfg(), two=False))

    def test_the_hero_tile_and_status_bar_escape(self):
        """Both expose typed escaped text, without a second template escape."""
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_alloc import _build_hero
        hero = _build_hero(_NewsletterContext(
            metrics=PortfolioMetrics(total_value=1000.0, invested_value=1000.0,
                                     cash_value=0.0),
            config=InvestorConfig(), benchmark_geo="MSCI ACWI & Friends"))
        for tile in hero["tiles"]:
            for field in ("label", "value", "caption"):
                assert self._clean(str(tile[field])), tile


def test_regions_run_in_opening_order_with_futures_on_their_own():
    """Futures first, then Asia, Europe, US by cash open, then bonds, then the
    round-the-clock groups; every index future comes from market_feeds, and
    every row a country owns has a flag."""
    from tarzan.data.market_feeds import FEED_KEYS, FEED_ROWS
    from tarzan.export.newsletter._constants import MARKET_FLAGS

    assert mq.CATEGORY_ORDER[:5] == ["Futures", "Asia", "Europe", "US", "Bonds"]
    cats = [c for _, _, c in mq.MARKETS]
    assert cats == sorted(cats, key=mq.CATEGORY_ORDER.index)
    # Every index future, the CME E-minis included, is a feed row.
    assert not {s for _, s, c in mq.MARKETS if c == "Futures"}
    assert {"fut:ES", "fut:NQ", "fut:YM", "fut:RTY"} <= set(FEED_KEYS)
    assert {r.category for r in FEED_ROWS} <= set(mq.CATEGORY_ORDER)
    assert [s for _, s, c in mq.MARKETS
            if c not in ("Commodities",) and s != "BTC-USD"
            and s not in MARKET_FLAGS] == []
    assert [k for k in FEED_KEYS if k not in MARKET_FLAGS] == []
    # Every new venue has a session, so its hours line and Op./Cl. badge show.
    for sym in ("^NSEI", "^STOXX", "FTSEMIB.MI"):
        assert mq.session_caption(sym), sym


def test_futures_heading_replaces_the_fut_tag_and_rows_carry_flags():
    rows = [
        {"name": "S&P 500 (FUT)", "symbol": "ES=F", "category": "Futures",
         "value": 5010.0, "change": 12.0, "pct": 0.24,
         "spark": [4998.0, 5010.0], "baseline": 4998.0},
        {"name": "FTSE MIB", "symbol": "FTSEMIB.MI", "category": "Europe",
         "value": 50000.0, "change": 50.0, "pct": 0.1,
         "spark": [49950.0, 50000.0], "baseline": 49950.0},
    ]
    with mock.patch.object(mq, "market_status", return_value=(True, "Mon")):
        html = _build_markets(_ctx(rows))["html"]
    assert "(FUT)" not in html
    assert "\U0001F1FA\U0001F1F8 S&amp;P 500<" in html
    assert "\U0001F1EE\U0001F1F9 FTSE MIB<" in html
    assert html.index("Futures") < html.index("Europe")


def test_the_board_declares_its_column_widths_so_a_long_name_cannot_push_the_line_out():
    """With fixed layout the first row (a region head spanning every column)
    decides the widths; without a colgroup the name got a quarter of the
    column and the inline lines vanished behind wrapped hour captions."""
    long_name = [{"name": "Nasdaq Composite", "symbol": "^IXIC", "category": "US",
                  "value": 24573.0, "change": 250.0, "pct": 1.06,
                  "spark": [24300.0, 24573.0], "baseline": 24323.0}]
    mq._memo = None
    with mock.patch.object(mq, "market_status", return_value=(False, "Fri")):
        html = _build_markets(_ctx(long_name))["html"]
    assert "table-layout:fixed;" in html
    assert re.search(r'<colgroup><col><col style="width:\d+px;"></colgroup>', html)
    assert "text-overflow:ellipsis" in html


def test_a_closed_future_is_labelled_with_the_readers_day_it_last_traded():
    """ICE's MSCI EM opens Monday's session on Sunday evening, never Friday's;
    Osaka's night session ends 06:00 Tokyo, still the reader's evening."""
    from datetime import datetime, timezone
    from tarzan.data import market_feeds as F

    rows = {r.key: r for r in F.FEED_ROWS}
    at = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    assert F.schedule(rows["fut:MME"], at("2026-11-14T12:00"))[1:] == (False, "Fri")  # Saturday
    assert F.schedule(rows["fut:MME"], at("2026-11-16T01:30"))[1] is True             # Sun 20:30 NY
    assert F.schedule(rows["fut:NK"], at("2026-11-11T23:20"))[1:] == (False, "Wed")


def _q(name, sym, cat, pct, **extra):
    v = 1000.0
    return dict({"name": name, "symbol": sym, "category": cat, "value": v,
                 "change": v * pct / 100, "pct": pct, "spark": [v, v], "baseline": v}, **extra)


class TestTheBoardIsBlocksOfCategories:
    """Each category is a full-width block (head over a rule, then its rows in
    two columns), so a category boundary is a horizontal line across the whole
    section. Two independent columns of whole categories put the boundaries at
    different heights on either side ("difficile capire dove finisce una
    categoria e comincia la successiva")."""

    def test_every_category_head_spans_the_column_and_precedes_its_rows(self):
        rows = [_q("A", "A1", "Asia", 1.0), _q("B", "B1", "Asia", -1.0), _q("C", "C1", "Asia", 0.5),
                _q("D", "D1", "Europe", 0.2), _q("E", "E1", "Europe", -0.4)]
        with mock.patch.object(mq, "market_status", return_value=(False, "Thu")):
            html = _build_markets(_ctx(rows))["html"]
        heads = [m.start() for m in re.finditer(r'border-bottom:1px solid #2A3442;[^"]*">(Asia|Europe)</div>', html)]
        assert len(heads) == 2
        # Asia's three rows all sit between the Asia head and the Europe head.
        asia, europe = heads
        for n in ("A<", "B<", "C<"):
            assert asia < html.index(">" + n) < europe, n
        assert html.index(">D<") > europe and html.index(">E<") > europe
        # The rows of a category flow down two columns: the first half left.
        block = html[asia:europe]
        assert block.count('<td valign="top">') == 2
        assert block.index(">A<") < block.index(">B<") < block.index('<td></td>') < block.index(">C<")

    def test_the_change_cell_is_tinted_on_its_own_category(self):
        """The grids' rule: background says which way and how far among the
        category's peers; the figure is ink. The best and worst of a category
        reach the ramp's ends whatever another category did."""
        from tarzan.export._heat import _GREEN_END, _RED_END
        rows = [_q("A", "A1", "Asia", 0.30), _q("B", "B1", "Asia", -0.10),
                _q("C", "C1", "Europe", 3.00), _q("D", "D1", "Europe", -2.00)]
        with mock.patch.object(mq, "market_status", return_value=(False, "Thu")):
            html = _build_markets(_ctx(rows))["html"]

        def cell(name):
            i = html.index(">" + name + "<")
            return re.search(r'<td align="right" style="padding:1px 3px;[^"]*">', html[i:]).group(0)
        assert f"background:{_GREEN_END};" in cell("A") and f"background:{_GREEN_END};" in cell("C")
        assert f"background:{_RED_END};" in cell("B") and f"background:{_RED_END};" in cell("D")
        assert "color:#EEF2F7;" in cell("A")
        # Mixed signs keep each side's own end: A at +0.30 is as green as C at +3.00.

    def test_a_falling_yield_is_the_good_end_of_the_bonds_ramp(self):
        from tarzan.export._heat import _GREEN_END, _RED_END
        rows = [dict(_q("US 2Y", "y:US2", "Bonds", 0.0), unit="yield", bp=+4.0, value=4.5, feed_row=True),
                dict(_q("Bund 2Y", "y:DE2", "Bonds", 0.0), unit="yield", bp=-3.0, value=2.1, feed_row=True)]
        with mock.patch.object(mq, "market_status", return_value=(False, "Thu")):
            html = _build_markets(_ctx(rows))["html"]
        i, j = html.index(">US 2Y<"), html.index(">Bund 2Y<")
        us = re.search(r'<td align="right" style="padding:1px 3px;[^"]*">', html[i:]).group(0)
        de = re.search(r'<td align="right" style="padding:1px 3px;[^"]*">', html[j:]).group(0)
        assert f"background:{_RED_END};" in us and f"background:{_GREEN_END};" in de

    def test_a_longer_hours_caption_weighs_more_in_the_column_split(self):
        """Four rows: three with two-session captions (three lines each) and
        one single-line. By count the split is 2/2; by height the left takes
        two tall rows and the right two... unless the tall ones cluster:
        three tall then one short → left two tall, right one tall + one short."""
        from tarzan.export.newsletter import _sections_perf as sp
        tall = {"feed_row": True, "hours_caption": "23:50\u201306:30, 07:10\u201321:00", "live": True,
                "session_day": "Thu"}
        rows = [dict(_q("A", "f:A", "Futures", 0.1), **tall), dict(_q("B", "f:B", "Futures", 0.2), **tall),
                dict(_q("C", "f:C", "Futures", 0.3), **tall),
                dict(_q("D", "f:D", "Futures", 0.4), feed_row=True, hours_caption="08:00\u201322:00",
                     live=True, session_day="Thu")]
        with mock.patch.object(mq, "market_status", return_value=(True, "Thu")):
            html = _build_markets(_ctx(rows))["html"]
        left = html[:html.index("<td></td>")]
        assert ">A<" in left and ">B<" in left and ">C<" not in left
