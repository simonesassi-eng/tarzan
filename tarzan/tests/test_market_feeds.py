"""Futures and yields from the three public feeds: parsing, voting, composition.

Network-free: every body below is shaped like the feed's real answer (recorded
2 Nov 2026) with synthetic figures, served through the module's one HTTP seam.
"""

from __future__ import annotations

import datetime
import json
from datetime import timedelta, timezone

import pandas as pd
import pytest

from tarzan.data import market_feeds as mf

NOW = datetime.datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc)   # Monday 17:00 CET
FRI = datetime.datetime(2026, 10, 30, tzinfo=timezone.utc)


def ms(t):
    return int(t.timestamp() * 1000)


# --- bodies shaped like each feed's real answer --------------------------------

def tv_body(rows):
    """rows: {symbol: (close, change_abs, update_time)}"""
    return {"totalCount": len(rows),
            "data": [{"s": s, "d": [c, ch, int(t.timestamp())]} for s, (c, ch, t) in rows.items()]}


def cnbc_body(rows):
    """rows: {symbol: (last, previous_day_closing, last_time)}"""
    return {"FormattedQuoteResult": {"FormattedQuote": [
        {"symbol": s, "code": 0, "last": last, "previous_day_closing": prev,
         "last_time": t.strftime("%Y-%m-%dT%H:%M:%S.000%z")}
        for s, (last, prev, t) in rows.items()]}}


def cnbc_chart_body(points):
    return {"barData": {"priceBars": [{"close": f"{v}", "tradeTimeinMills": ms(t)} for t, v in points]}}


def mw_body(points, *, prior=None, prior_at=FRI, session=None):
    extra = [{"Name": "MostRecentLast", "Date": ms(points[-1][0]), "Value": points[-1][1]}]
    if prior is not None:
        extra.append({"Name": "PriorClose", "Date": ms(prior_at), "Value": prior})
    sessions = ([{"Kind": "Trading", "Start": ms(session[0]), "End": ms(session[1])}]
                if session else [])
    return {"TimeInfo": {"Ticks": [ms(t) for t, _ in points]},
            "Series": [{"DataPoints": [[v] for _, v in points], "ExtraData": extra,
                        "MarketSessions": sessions}]}


def path(start, values, step_min=5):
    return [(start + timedelta(minutes=step_min * i), v) for i, v in enumerate(values)]


@pytest.fixture
def serve(monkeypatch):
    """Install bodies per feed; returns the list of requests made."""
    seen = []

    def install(tv=None, cnbc=None, cnbc_charts=None, mw=None):
        def fake(provider, method, url, **kw):
            seen.append(provider)
            if provider == mf.TV:
                return tv
            if provider == mf.CNBC:
                if "charts" in url:
                    frame = "1D" if "/1D." in url else "1M"
                    return (cnbc_charts or {}).get((kw["params"]["symbol"], frame))
                return cnbc
            if provider == mf.MW:
                key = url  # the series key is inside the encoded query
                for k, (intra, daily) in (mw or {}).items():
                    if k.replace("/", "%2F") in key:
                        return intra if "PT5M" in key else daily
            return None
        monkeypatch.setattr(mf, "_request_json", fake)
        return seen
    return install


DE10 = mf.FeedRow("Bund 10Y", "bond:DE10Y", "Bonds", "yield",
                  {mf.TV: "TVC:DE10Y", mf.CNBC: "DE10Y", mf.MW: "BOND/BX//TMBMKDE-10Y"},
                  ((mf.TV, mf.CNBC), (mf.MW, mf.MW), (mf.TV, mf.MW), (mf.TV, None)), 10.0)
IT10 = mf.FeedRow("BTP 10Y", "bond:IT10Y", "Bonds", "yield",
                  {mf.TV: "TVC:IT10Y", mf.CNBC: "IT10Y", mf.MW: "BOND/BX//TMBMKIT-10Y"},
                  DE10.recipes, 10.0)
DAX = mf.FeedRow("DAX", "fut:FDAX", "Futures", "index",
                 {mf.TV: "EUREX:FDAX1!", mf.MW: "FUTURE/DE/XEUR/DAX00"},
                 ((mf.MW, mf.MW), (mf.CNBC, mf.CNBC), (mf.TV, mf.MW), (mf.TV, mf.CNBC), (mf.TV, None)),
                 14.2)


# --- parsers ----------------------------------------------------------------------

class TestParsers:
    def test_tradingview_previous_close_is_close_minus_change(self, serve):
        serve(tv=tv_body({"TVC:DE10Y": (3.2928, -0.0313, NOW), "EUREX:FDAX1!": (-5, 1, NOW)}))
        got = mf._tradingview_quotes(["TVC:DE10Y", "EUREX:FDAX1!"],
                                     {"TVC:DE10Y": "yield", "EUREX:FDAX1!": "index"})
        assert got["TVC:DE10Y"].prev == pytest.approx(3.3241)
        assert got["TVC:DE10Y"].at == NOW
        assert "EUREX:FDAX1!" not in got  # a non-positive index level is no quote

    def test_cnbc_never_uses_a_bond_previous_close_but_keeps_a_futures_one(self, serve):
        cet = timezone(timedelta(hours=1))
        serve(cnbc=cnbc_body({"DE10Y": ("3.2929%", "3.2921%", NOW.astimezone(cet)),
                              "@FFI.1": ("10,792.50", "10,756.50", NOW)}))
        got = mf._cnbc_quotes(["DE10Y", "@FFI.1"], {"DE10Y": "yield", "@FFI.1": "index"})
        assert got["DE10Y"].last == pytest.approx(3.2929) and got["DE10Y"].prev is None
        assert got["@FFI.1"].prev == pytest.approx(10756.5)
        assert got["DE10Y"].at == NOW

    def test_marketwatch_prior_close_dated_the_same_day_is_discarded(self, serve):
        """The rolled-baseline case: a 'prior' close from the last print's own day."""
        bars = path(NOW - timedelta(hours=1), [3.30, 3.31, 3.31])
        serve(mw={"BOND/BX//TMBMKDE-10Y": (mw_body(bars, prior=3.30, prior_at=NOW), None)})
        obs, _chart = mf._marketwatch("BOND/BX//TMBMKDE-10Y", "yield")
        assert obs.last == pytest.approx(3.31) and obs.prev is None

    def test_marketwatch_reads_the_dated_prior_close_and_session_window(self, serve):
        bars = path(NOW - timedelta(hours=1), [26000, 26050, 26100])
        session = (NOW - timedelta(hours=9), NOW + timedelta(hours=4))
        serve(mw={"FUTURE/DE/XEUR/DAX00": (mw_body(bars, prior=25913, session=session), None)})
        obs, chart = mf._marketwatch("FUTURE/DE/XEUR/DAX00", "index")
        assert obs.prev == 25913 and obs.last == 26100
        assert chart.span[0] == pd.Timestamp(session[0]) and len(chart.intra) == 3

    @pytest.mark.parametrize("body", [None, {}, {"data": "x"}, {"data": [{"s": "A"}]}, []])
    def test_garbage_answers_yield_nothing_and_never_raise(self, serve, body):
        serve(tv=body, cnbc=body)
        assert mf._tradingview_quotes(["A"], {"A": "index"}) == {}
        assert mf._cnbc_quotes(["A"], {"A": "index"}) == {}


# --- voting -------------------------------------------------------------------------

def obs(feed, last, prev, at=NOW):
    return mf.Obs(feed, last, prev, at)


class TestVoting:
    def test_confirmed_level_and_change_is_verified(self):
        shown = obs(mf.TV, 3.29, 3.32)
        assert mf.verdict(shown, [obs(mf.MW, 3.31, 3.342)], "yield") == "verified"

    def test_a_rolled_baseline_is_disputed(self):
        """Same level, but one source measures the day from today's close."""
        shown = obs(mf.TV, 4.41, 4.49)            # -8 bp
        assert mf.verdict(shown, [obs(mf.MW, 4.41, 4.41)], "yield") == "disputed"

    def test_two_sources_on_different_days_compare_only_levels(self):
        """23:00 CET: MarketWatch has opened the Treasuries' next session (its
        change is ~0 from today's close) while TradingView still measures today."""
        today = mf.Obs(mf.TV, 4.88, 4.93, NOW, NOW.replace(hour=0))
        tomorrow = mf.Obs(mf.MW, 4.882, 4.881, NOW, NOW.replace(hour=22))
        assert mf.verdict(today, [tomorrow], "yield") == "verified"
        same_day = mf.Obs(mf.MW, 4.882, 4.881, NOW, NOW.replace(hour=0) - timedelta(hours=2))
        assert mf.verdict(today, [same_day], "yield") == "disputed"

    def test_level_only_confirmation_counts_when_no_one_else_has_a_baseline(self):
        assert mf.verdict(obs(mf.TV, 3.29, 3.32), [obs(mf.CNBC, 3.29, None)], "yield") == "verified"

    def test_alone_is_single(self):
        assert mf.verdict(obs(mf.TV, 3.29, 3.32), [], "yield") == "single"

    def test_an_outvoted_source_is_not_shown_when_two_others_agree(self):
        row = mf.FeedRow("FTSE 100", "fut:Z", "Futures", "index",
                         {mf.TV: "Z", mf.CNBC: "F", mf.MW: "M"},
                         ((mf.MW, mf.MW), (mf.CNBC, mf.CNBC), (mf.TV, None)))
        votes = {mf.MW: obs(mf.MW, 11800, 11700),          # wrong contract
                 mf.CNBC: obs(mf.CNBC, 10792, 10756),
                 mf.TV: obs(mf.TV, 10791, 10756)}
        shown, chart_feed, status = mf.choose(row, votes)
        assert shown.feed == mf.CNBC and chart_feed == mf.CNBC and status == "verified"

    def test_a_source_days_behind_does_not_vote(self):
        votes = {mf.TV: obs(mf.TV, 23576, 23365), mf.MW: obs(mf.MW, 23278, 23023, NOW - timedelta(days=3))}
        assert set(mf._fresh(votes)) == {mf.TV}

    def test_no_baseline_anywhere_means_no_row(self):
        assert mf.choose(DE10, {mf.CNBC: obs(mf.CNBC, 3.29, None)}) is None


# --- composition ----------------------------------------------------------------------

def _bond_world(serve, *, tv=True, cnbc=True, mw=True):
    de = path(NOW - timedelta(hours=3), [3.33, 3.31, 3.30, 3.294], step_min=60)
    it = path(NOW - timedelta(hours=3), [4.49, 4.45, 4.42, 4.413], step_min=60)
    # CNBC draws one-minute bars, plus one stamped after the run's instant.
    minutes = [(NOW - timedelta(minutes=480 - i), 3.30) for i in range(480)]
    after = [(NOW + timedelta(minutes=10), 3.50)]
    return serve(
        tv=tv_body({"TVC:DE10Y": (3.294, -0.030, NOW), "TVC:IT10Y": (4.413, -0.077, NOW)}) if tv else None,
        cnbc=cnbc_body({"DE10Y": ("3.294%", "3.294%", NOW), "IT10Y": ("4.413%", "4.413%", NOW)}) if cnbc else None,
        cnbc_charts={("DE10Y", "1D"): cnbc_chart_body(minutes + after),
                     ("IT10Y", "1D"): cnbc_chart_body(it)},
        mw={"BOND/BX//TMBMKDE-10Y": (mw_body(de, prior=3.342), mw_body(path(FRI, [3.37, 3.342], 1440))),
            "BOND/BX//TMBMKIT-10Y": (mw_body(it, prior=4.487), None)} if mw else None,
    )


class TestComposition:
    def test_bonds_and_their_spread(self, serve):
        _bond_world(serve)
        rows = {d["symbol"]: d for d in mf.fetch_feed_quotes(NOW, rows=(DE10, IT10))}
        de, it, spread = rows["bond:DE10Y"], rows["bond:IT10Y"], rows["spread:IT-DE10Y"]
        assert de["source"] == mf.TV and de["verdict"] == "verified"
        assert de["bp"] == pytest.approx(-3.0) and de["pct"] is None
        assert spread["value"] == pytest.approx((4.413 - 3.294) * 100)
        assert spread["bp"] == pytest.approx(it["bp"] - de["bp"])
        assert list(rows) == ["bond:DE10Y", "bond:IT10Y", "spread:IT-DE10Y"]

    def test_intraday_is_thinned_to_quarter_hours_and_never_after_the_run(self, serve):
        _bond_world(serve)
        de = {d["symbol"]: d for d in mf.fetch_feed_quotes(NOW, rows=(DE10,))}["bond:DE10Y"]
        ser = de["spark_series"]
        assert ser is not None and len(ser) <= 8 * 4 + 1
        assert ser.index.max() <= pd.Timestamp(NOW)
        # thinned to real observations, not bucket labels
        assert ser.index[-1] == pd.Timestamp(NOW - timedelta(minutes=1))

    def test_a_feed_down_still_shows_the_row(self, serve):
        _bond_world(serve, tv=False)
        de = {d["symbol"]: d for d in mf.fetch_feed_quotes(NOW, rows=(DE10,))}["bond:DE10Y"]
        assert de["source"] == mf.MW and de["bp"] == pytest.approx((3.294 - 3.342) * 100)

    def test_every_feed_down_shows_nothing(self, serve):
        _bond_world(serve, tv=False, cnbc=False, mw=False)
        assert mf.fetch_feed_quotes(NOW, rows=(DE10, IT10)) == []

    def test_a_point_in_time_run_makes_no_request(self, serve):
        from tarzan import runtime

        seen = _bond_world(serve)
        runtime.configure(as_of=datetime.date(2026, 6, 30))
        assert mf.fetch_feed_quotes(NOW, rows=(DE10,)) == []
        assert seen == []

    def test_a_chart_of_another_session_is_not_drawn(self, serve):
        old = path(NOW - timedelta(days=3), [26000, 26010, 26020])
        serve(tv=tv_body({"EUREX:FDAX1!": (26100, 187, NOW)}),
              mw={"FUTURE/DE/XEUR/DAX00": (mw_body(old, prior=25913, prior_at=FRI - timedelta(days=1)), None)})
        dax = mf.fetch_feed_quotes(NOW, rows=(_row("fut:FDAX"),))[0]
        assert dax["source"] == mf.TV and dax["spark_series"] is None

    def test_live_means_a_recent_print(self, serve):
        serve(tv=tv_body({"EUREX:FDAX1!": (26100, 187, NOW - timedelta(hours=2))}))
        assert mf.fetch_feed_quotes(NOW, rows=(DAX,))[0]["live"] is False

    def test_a_parser_crash_omits_rows_instead_of_raising(self, serve, monkeypatch):
        serve()
        monkeypatch.setattr(mf, "_tradingview_quotes", lambda *a: 1 / 0)
        assert mf.fetch_feed_quotes(NOW, rows=(DAX,)) == []


# --- rendering -------------------------------------------------------------------------

def test_the_strip_prints_yields_in_bp_and_names_a_dispute():
    from unittest import mock

    import tarzan.data.market_quotes as mq
    from tarzan.export.newsletter._constants import _NewsletterContext
    from tarzan.export.newsletter._sections_perf import _build_markets
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.portfolio import PortfolioMetrics

    base = {"feed_row": True, "category": "Bonds", "spark": [], "spark_series": None,
            "span": None, "session_hours": 10.0, "as_of": NOW, "live": True,
            "observed_day": NOW.date(), "stale_session": False, "pct": None, "change": 0.0}
    rows = [
        dict(base, name="Bund 10Y", symbol="bond:DE10Y", unit="yield", value=3.2941,
             bp=-3.13, baseline=3.325, verdict="verified", source=mf.TV, alternatives=[]),
        dict(base, name="BTP–Bund 10Y", symbol="spread:IT-DE10Y", unit="spread", value=111.9,
             bp=-4.7, baseline=116.6, verdict="verified", source=mf.TV, alternatives=[]),
        dict(base, name="Bonos 2Y", symbol="bond:ES2Y", unit="yield", value=3.2202,
             bp=4.3, baseline=3.177, verdict="disputed", source=mf.TV,
             alternatives=[(mf.CNBC, 3.22, None), (mf.MW, 3.173, -0.3)]),
    ]
    ctx = _NewsletterContext(
        metrics=PortfolioMetrics(total_value=1.0, invested_value=1.0, cash_value=0.0,
                                 market_quotes=tuple(rows)),
        config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI")
    with mock.patch.object(mq, "market_status", return_value=(True, "Mon")):
        html = _build_markets(ctx)["html"]
    assert "3.294%" in html and "−3.1 bp" in html
    assert "112 bp" in html
    assert "+4.3 bp?" in html
    assert "Bonos 2Y: TradingView +4.3 bp (shown); MarketWatch −0.3 bp" in html
    assert "CNBC n/a" not in html
    assert "\U0001F1EE\U0001F1F9\U0001F1E9\U0001F1EA BTP" in html
    assert "upd 17:00&nbsp;&middot;&nbsp;Op." in html   # a bond: the print's time on the reader's clock


def test_the_chart_axis_covers_every_bar(serve):
    """A lone print the evening before must not squeeze the day onto the edge."""
    evening = [(NOW - timedelta(hours=18), 5.29)]
    day = path(NOW - timedelta(hours=8), [5.24, 5.23, 5.22, 5.20], step_min=150)
    serve(tv=tv_body({"TVC:GB10Y": (5.20, -0.09, NOW)}),
          cnbc=cnbc_body({"UK10Y": ("5.20%", "5.20%", NOW)}),
          cnbc_charts={("UK10Y", "1D"): cnbc_chart_body(evening + day)})
    gilt = mf.FeedRow("Gilt 10Y", "bond:GB10Y", "Bonds", "yield",
                      {mf.TV: "TVC:GB10Y", mf.CNBC: "UK10Y"}, DE10.recipes, 10.0)
    d = mf.fetch_feed_quotes(NOW, rows=(gilt,))[0]
    lo, hi = d["span"]
    ser = d["spark_series"]
    assert lo <= ser.index[0] and hi >= ser.index[-1]
    # ...and the stray evening print is not part of the drawn session
    assert ser.index[0] == pd.Timestamp(day[0][0])


def test_the_ai_digest_reads_yields_in_basis_points():
    from tarzan.export import ai_summary
    from tarzan.models.investor_config import InvestorConfig
    from tarzan.models.portfolio import PortfolioMetrics

    rows = ({"name": "Bund 10Y", "unit": "yield", "value": 3.2941, "bp": -3.13, "pct": None},
            {"name": "BTP–Bund 10Y", "unit": "spread", "value": 111.94, "bp": -4.7, "pct": None},
            {"name": "DAX", "unit": "index", "value": 26107.0, "bp": None, "pct": 0.75})
    m = PortfolioMetrics(total_value=1.0, invested_value=1.0, cash_value=0.0, market_quotes=rows)
    got = ai_summary.build_digest(m, InvestorConfig())["markets_today"]
    assert got[0] == {"name": "Bund 10Y", "level": 3.294, "unit": "yield %", "change_bp": -3.1}
    assert got[1]["unit"] == "bp" and got[1]["change_bp"] == -4.7
    assert got[2] == {"name": "DAX", "level": 26107.0, "change_pct": 0.75}


# --- the cases seen live on 3 Nov 2026 ------------------------------------------------

def test_a_lone_bad_print_is_not_drawn(serve):
    """CNBC's US 2Y carried one 5.026% bar in a 4.59% day."""
    bars = path(NOW - timedelta(hours=3), [4.59, 4.60, 5.026, 4.60, 4.59, 4.598], step_min=30)
    serve(tv=tv_body({"TVC:DE10Y": (3.294, -0.03, NOW, )}),
          cnbc=cnbc_body({"DE10Y": ("3.294%", "3.294%", NOW)}),
          cnbc_charts={("DE10Y", "1D"): cnbc_chart_body([(t, v - 1.3) for t, v in bars])})
    ser = mf.fetch_feed_quotes(NOW, rows=(DE10,))[0]["spark_series"]
    assert ser.max() < 3.4 and len(ser) == 5


def test_the_chart_is_one_session_from_the_stated_start(serve):
    """CNBC's '1D' for Asian futures ran 36 hours; TradingView states the start."""
    start = NOW - timedelta(hours=5)
    bars = path(NOW - timedelta(hours=36), [8900 + i for i in range(73)], step_min=30)
    serve(tv={"data": [{"s": "ASX24:AP1!", "d": [8972, -18, int(NOW.timestamp()), int(start.timestamp())]}]},
          cnbc=cnbc_body({"@YAP.1": ("8,972", "8,990", NOW)}),
          cnbc_charts={("@YAP.1", "1D"): cnbc_chart_body(bars)})
    spi = mf.FeedRow("ASX SPI 200", "fut:SPI", "Futures", "index",
                     {mf.TV: "ASX24:AP1!", mf.CNBC: "@YAP.1"}, DAX.recipes)
    ser = mf.fetch_feed_quotes(NOW, rows=(spi,))[0]["spark_series"]
    assert ser.index[0] >= pd.Timestamp(start) - pd.Timedelta(minutes=1)


def test_marketwatch_without_bars_does_not_vote(serve):
    """A holiday: one point, stamped with the session's end."""
    body = mw_body([(NOW - timedelta(hours=2), 2.77)], prior=2.77)
    serve(mw={"BOND/BX//TMBMKJP-10Y": (body, None)})
    obs, chart = mf._marketwatch("BOND/BX//TMBMKJP-10Y", "yield")
    assert obs is None and chart.intra is None


def test_two_benchmark_bonds_are_compared_on_level_only():
    """LSEG's and Tullett's 2Y Bund were 4.6 bp apart with opposite moves."""
    lseg = mf.Obs(mf.TV, 3.0327, 3.0474, NOW)
    tullett = mf.Obs(mf.MW, 3.078, 3.0376, NOW)
    assert mf.verdict(lseg, [tullett], "yield") == "verified"


def test_treasuries_are_measured_from_the_new_york_close():
    row = next(r for r in mf.FEED_ROWS if r.key == "bond:US10Y")
    assert row.recipes[0] == (mf.MW, mf.MW)


# --- futures timetables ---------------------------------------------------------------

def _row(key):
    return next(r for r in mf.FEED_ROWS if r.key == key)


@pytest.mark.parametrize("when, is_open, day", [
    (datetime.datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc), True, "Thu"),    # 13:00 CET
    (datetime.datetime(2026, 11, 5, 23, 30, tzinfo=timezone.utc), False, "Thu"),  # 00:30 Fri, before 01:10
    (datetime.datetime(2026, 11, 7, 11, 0, tzinfo=timezone.utc), False, "Fri"),   # Saturday
    # Christmas Friday: Xetra is shut on the 24th too, so the last session is Wednesday.
    (datetime.datetime(2026, 12, 25, 11, 0, tzinfo=timezone.utc), False, "Wed"),
])
def test_eurex_hours_and_status(when, is_open, day):
    caption, open_now, label = mf.schedule(_row("fut:FDAX"), when)
    assert caption.startswith("01:10–22:00 CE")
    assert (open_now, label) == (is_open, day)


def test_osaka_night_session_belongs_to_the_day_it_starts():
    nk = _row("fut:NK")
    friday_night = datetime.datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)   # Sat 05:00 JST
    assert mf.schedule(nk, friday_night)[1:] == (True, "Fri")
    the_break = datetime.datetime(2026, 11, 9, 7, 30, tzinfo=timezone.utc)      # Mon 16:30 JST
    assert mf.schedule(nk, the_break)[1:] == (False, "Mon")
    culture_day = datetime.datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)     # Tue 10:00, holiday
    assert mf.schedule(nk, culture_day)[1] is False


def test_bonds_have_no_timetable_and_keep_the_print_time():
    assert mf.schedule(_row("bond:DE10Y"), NOW) is None


def test_every_future_has_a_timetable_and_singapore_a_zone_label():
    futures = [r for r in mf.FEED_ROWS if r.category == "Futures"]
    assert all(r.sessions and r.tz for r in futures)
    # Singapore 09:00-16:35, 17:15-05:15 on the reader's clock (CET in November).
    assert mf.schedule(_row("fut:A50"), NOW)[0] == "02:00–09:35, 10:15–22:15 CET"


# --- out-of-session behaviour (7 Nov 2026) -------------------------------------------

SAT = datetime.datetime(2026, 11, 7, 13, 0, tzinfo=timezone.utc)
FRI_UTC = datetime.datetime(2026, 11, 6, tzinfo=timezone.utc)


def test_a_closed_future_shows_the_settlement_and_confirms_the_baseline(serve):
    """TradingView prints Friday's settlement (25875), MarketWatch the 22:00 last
    trade (25754): both measure from Thursday's 26418, so they confirm each other."""
    bars = path(FRI_UTC.replace(hour=6, minute=40), [26100 - 6 * i for i in range(58)], step_min=15)
    serve(tv={"data": [{"s": "EUREX:FDAX1!", "d": [25875, 25875 - 26418,
                                                    int(SAT.replace(hour=9).timestamp()),
                                                    int(FRI_UTC.timestamp())]}]},
          mw={"FUTURE/DE/XEUR/DAX00": (mw_body(bars, prior=26418,
                                                prior_at=FRI_UTC - timedelta(days=1)), None)})
    dax = mf.fetch_feed_quotes(SAT, rows=(_row("fut:FDAX"),))[0]
    assert dax["source"] == mf.TV and dax["value"] == 25875
    assert dax["verdict"] == "verified"
    assert dax["pct"] == pytest.approx((25875 / 26418 - 1) * 100)
    assert dax["spark_series"] is not None          # Friday's session is drawn
    assert dax["live"] is False


def test_a_saturday_stamp_does_not_push_the_other_bond_sources_out(serve):
    """CNBC stamped the Bund with Saturday 08:31: TradingView and MarketWatch
    (Friday) were dropped as stale and the row disappeared."""
    fri_eve = FRI_UTC.replace(hour=20, minute=59)
    serve(tv=tv_body({"TVC:DE10Y": (3.2541, 0.0153, fri_eve)}),
          cnbc=cnbc_body({"DE10Y": ("3.2508%", "3.2508%", SAT.replace(hour=8, minute=31))}),
          mw={"BOND/BX//TMBMKDE-10Y": (mw_body(path(FRI_UTC.replace(hour=7), [3.25, 3.26, 3.2678], 300),
                                                prior=3.2611, prior_at=FRI_UTC - timedelta(days=1)), None)})
    rows = mf.fetch_feed_quotes(SAT, rows=(DE10,))
    assert rows and rows[0]["source"] == mf.TV and rows[0]["verdict"] == "verified"


def test_a_closed_futures_source_from_an_older_session_still_does_not_vote(serve):
    votes = {mf.TV: obs(mf.TV, 25875, 26418, SAT),
             mf.MW: obs(mf.MW, 26100, 26300, SAT - timedelta(days=3))}
    assert set(mf._fresh(votes, _row("fut:FDAX"), SAT)) == {mf.TV}


def test_the_eight_oclock_issue_draws_marketwatchs_pre_market(serve):
    """08:00 CET on a weekday: MarketWatch's regular day starts at 07:50, its
    pre-market at 07:30; with the pre-market bars the line has something to draw."""
    run = datetime.datetime(2026, 11, 5, 7, 3, tzinfo=timezone.utc)
    pre = (run.replace(hour=6, minute=30), run.replace(hour=6, minute=50))
    bars = path(run.replace(hour=6, minute=30), [26000, 26010, 26005, 26020, 26030, 26025], step_min=5)
    body = mw_body(bars, prior=25990, prior_at=run - timedelta(days=1), session=pre)
    body["Series"][0]["MarketSessions"] = [
        {"Kind": "PreMarket", "Start": ms(pre[0]), "End": ms(pre[1])},
        {"Kind": "Trading", "Start": ms(pre[1]), "End": ms(run.replace(hour=21))}]
    serve(tv=tv_body({"EUREX:FDAX1!": (26030, 40, run)}),
          mw={"FUTURE/DE/XEUR/DAX00": (body, None)})
    dax = mf.fetch_feed_quotes(run, rows=(_row("fut:FDAX"),))[0]
    assert dax["spark_series"] is not None and dax["spark_series"].index[0] == pd.Timestamp(pre[0])
    assert dax["live"] is True


def test_marketwatch_is_asked_for_its_pre_market_bars(monkeypatch):
    import json
    from urllib.parse import parse_qs, urlparse

    asked = []
    monkeypatch.setattr(mf, "_request_json", lambda provider, method, url, **kw: asked.append(url))
    mf._mw_series("FUTURE/DE/XEUR/DAX00", "PT5M", "D1")
    query = json.loads(parse_qs(urlparse(asked[0]).query)["json"][0])
    assert query["ShowPreMarket"] is True and query["IncludeMockTick"] is False


# --- TradingView chart socket --------------------------------------------------------

class FakeSocket:
    """Plays back server frames; records what the client sent."""

    def __init__(self, frames):
        self.frames, self.sent = list(frames), []

    def send(self, text):
        self.sent.append(text)

    def recv(self, timeout=None):
        if not self.frames:
            raise TimeoutError
        return self.frames.pop(0)


def _wire(payload):
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return f"~m~{len(text)}~m~{text}"


def _session_ids(sock):
    return [json.loads(f.split("~m~", 2)[2])["p"][0] for f in sock.sent
            if '"chart_create_session"' in f]


def test_the_socket_reader_collects_bars_echoes_heartbeats_and_skips_rejects():
    import itertools

    sock = FakeSocket([])
    # The reader creates its sessions first; answer them afterwards.
    orig_send = sock.send

    def send(text):
        orig_send(text)
        ids = _session_ids(sock)
        if len(ids) == 2 and not sock.frames and '"create_series"' in text:
            good, bad = ids
            sock.frames = [
                _wire("~h~7"),
                _wire({"m": "timescale_update", "p": [good, {"s": {"s": [
                    {"i": 0, "v": [1793926800, 1, 1, 1, 26100.0, 5]},
                    {"i": 1, "v": [1793927100, 1, 1, 1, 26110.0, 5]}]}}]}),
                _wire({"m": "symbol_error", "p": [bad, "sym", "invalid symbol"]}),
            ]
    sock.send = send
    clock = itertools.count(0.0, 0.01)
    bars = mf._tv_read_bars(sock, ["EUREX:FDAX1!", "BAD:XX"], deadline=5.0, clock=lambda: next(clock))
    assert bars == {"EUREX:FDAX1!": [(1793926800.0, 26100.0), (1793927100.0, 26110.0)]}
    assert "~m~4~m~~h~7" in sock.sent          # heartbeat echoed


def test_the_socket_reader_gives_up_at_its_deadline():
    sock = FakeSocket([])
    assert mf._tv_read_bars(sock, ["EUREX:FDAX1!"], deadline=0.0) == {}


def test_tradingview_charts_never_raise():
    """The suite's socket refuses to connect: no line, no exception."""
    assert mf._tradingview_charts(["EUREX:FDAX1!"], {}) == {}


def test_tradingview_charts_parse_the_socket_through_the_real_path(monkeypatch):
    import contextlib

    class Answering(FakeSocket):
        def send(self, text):
            super().send(text)
            if '"create_series"' in text:
                cs = _session_ids(self)[-1]
                self.frames.append(_wire({"m": "timescale_update", "p": [cs, {"s": {"s": [
                    {"i": 0, "v": [1793926800, 0, 0, 0, 26100.0, 1]},
                    {"i": 1, "v": [1793927100, 0, 0, 0, -5.0, 1]},          # implausible: dropped
                    {"i": 2, "v": [1793927400, 0, 0, 0, 26120.0, 1]}]}}]}))

    monkeypatch.setattr(mf, "_tv_connect", lambda timeout: contextlib.nullcontext(Answering([])))
    line = mf._tradingview_charts(["EUREX:FDAX1!"], {"EUREX:FDAX1!": "index"})["EUREX:FDAX1!"]
    assert list(line.values) == [26100.0, 26120.0] and str(line.index.tz) == "UTC"


def test_the_tradingview_line_draws_the_whole_eurex_day(serve, monkeypatch):
    """08:00 CET: MarketWatch has nothing yet, TradingView has bars since 01:10."""
    run = datetime.datetime(2026, 11, 5, 7, 5, tzinfo=timezone.utc)
    night = path(run.replace(hour=0, minute=15), [26000 + i for i in range(80)], step_min=5)
    line = pd.Series({t: v for t, v in night if t <= run})
    serve(tv={"data": [{"s": "EUREX:FDAX1!", "d": [26079, 79, int(run.timestamp()),
                                                    int(run.replace(hour=0, minute=0).timestamp())]}]})
    monkeypatch.setattr(mf, "_tradingview_charts", lambda syms, units: {"EUREX:FDAX1!": line})
    dax = mf.fetch_feed_quotes(run, rows=(_row("fut:FDAX"),))[0]
    ser = dax["spark_series"]
    assert ser is not None and ser.index[0] <= pd.Timestamp(run.replace(hour=0, minute=30))
    lo, hi = dax["span"]
    assert (f"{lo:%H:%M}", f"{hi:%H:%M}") == ("01:10", "22:00")   # Eurex, Frankfurt time


def test_a_tradingview_line_of_another_session_falls_back_to_the_recipes_chart(serve, monkeypatch):
    run = datetime.datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    old = pd.Series({run - timedelta(days=3, minutes=5 * i): 26000.0 + i for i in range(10)}).sort_index()
    today = path(run.replace(hour=7), [26100, 26110, 26120, 26130], step_min=60)
    serve(tv={"data": [{"s": "EUREX:FDAX1!", "d": [26130, 30, int(run.timestamp()),
                                                    int(run.replace(hour=0).timestamp())]}]},
          mw={"FUTURE/DE/XEUR/DAX00": (mw_body(today, prior=26100, prior_at=run - timedelta(days=1)), None)})
    monkeypatch.setattr(mf, "_tradingview_charts", lambda syms, units: {"EUREX:FDAX1!": old})
    dax = mf.fetch_feed_quotes(run, rows=(_row("fut:FDAX"),))[0]
    assert dax["spark_series"] is not None and dax["spark_series"].index[0].day == 5


def test_an_osaka_evening_session_belongs_to_the_next_trading_day():
    nk = _row("fut:NK")
    friday_evening = datetime.datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)   # Fri 21:00 JST
    caption, is_open, day, start, end, window = mf._timetable(nk, friday_evening)
    assert is_open and day == "Fri"
    # The trading day it opens is Monday's: from Friday 17:00 to Monday 15:45 JST...
    assert (start.strftime("%a %H:%M"), end.strftime("%a %H:%M")) == ("Fri 17:00", "Mon 15:45")
    # ...but the chart shows the session in progress.
    assert (window[0].strftime("%a %H:%M"), window[1].strftime("%a %H:%M")) == ("Fri 17:00", "Sat 06:00")
