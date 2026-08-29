"""A Tarzan return must be the figure the instrument's Yahoo page shows.

Two things had to change for that to hold, and each is pinned here.

1. The window edges are Yahoo's: its ranges are CALENDAR spans (1mo, 3mo, 6mo,
   1y, 3y, 5y) and its 5D spans five sessions. Tarzan measured fixed day counts,
   so 90 days anchored two sessions past three calendar months — verified
   against the site on 18 Aug 2026: EXUS.MI 3M read +6.45% where Yahoo showed
   +7.87%, from 20 May against Yahoo's 18 May.

2. The 1D is measured against the quote pair (``regularMarketPrice`` /
   ``regularMarketPreviousClose``), not against the daily chart. Same day, every
   Milan ETF came back with a NULL close for Monday 17 Aug in the chart feed
   while the quote endpoint carried the official 40.815 — so the chart alone
   could only measure from the Friday.

Network-free: fixture series and an explicit quote pair.
"""

from __future__ import annotations

import pandas as pd
import pytest

import tarzan.data.market_quotes as mq
from tarzan.engine.stats import compute_period_return, window_anchor


def _closes(pairs):
    return pd.Series([v for _, v in pairs],
                     index=pd.to_datetime([d for d, _ in pairs]))


def _business_series(start: str, end: str, step: float = 0.1):
    idx = pd.bdate_range(start, end)
    return pd.Series([100.0 + i * step for i in range(len(idx))], index=idx)


class TestTrailingWindowAnchor:
    @pytest.fixture(autouse=True)
    def _pin_today(self, monkeypatch):
        # Every fixture below ends 2026-08-18 and reasons about windows
        # measured back from it (see _window_end's docstring). Without this,
        # window_anchor reads the real run-owned clock (runtime.today()),
        # so a fixture written against one date silently drifts by a day
        # each day the suite runs after it - which is exactly what the
        # three tests below this one, before this fixture existed, did.
        import datetime
        monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2026, 8, 18))

    def test_a_month_snaps_to_the_last_session_on_or_before(self):
        """The bug the user caught: XMME.MI 1M read +1.13% (from Mon 20 Jul)
        where the published figure is +2.55% (from Fri 17 Jul). The window
        starts one calendar month back — 18 Jul, a Saturday — so the anchor is
        the last session AT OR BEFORE it (Friday), not the first one after.
        """
        s = _closes([
            ("2026-07-16", 78.53), ("2026-07-17", 76.78),   # window start = Fri
            ("2026-07-20", 77.86),                           # NOT this Monday
            ("2026-08-14", 80.12), ("2026-08-18", 78.74),
        ])
        assert window_anchor(s, "1m") == pd.Timestamp("2026-07-17")
        assert round(compute_period_return(s, "1m"), 2) == 2.55

    def test_three_months_and_five_years_follow_the_calendar(self):
        s = _business_series("2019-01-01", "2026-08-18")
        assert window_anchor(s, "3m") == pd.Timestamp("2026-05-18")
        assert window_anchor(s, "5y") == pd.Timestamp("2021-08-18")

    def test_five_days_is_five_days_of_change(self):
        """"5D" is five days of CHANGE, so it anchors five sessions back.

        Confirmed against Yahoo's own pages on 24 Aug 2026: AVWS.DE read -1.56%
        and RSSY -1.28%, and both are reproducible only from the 17 Aug close —
        five sessions before the current one. Anchoring four back (18 Aug) gave
        -1.04% and -1.39%. The 1D was already one step, which is why only the
        5D disagreed.
        """
        s = _business_series("2026-07-01", "2026-08-18")
        # 18 Aug (Tue) back five sessions → 11 Aug (Tue): six closes, five moves.
        assert window_anchor(s, "5d") == pd.Timestamp("2026-08-11")

    def test_a_feed_with_no_current_price_measures_the_window_it_has(self):
        """NTSG.MI on 18 Aug 2026: no close after the 14th, and no quote the
        sanity gate would accept either.

        Both edges of a window must be read off the same clock. This used to
        count the START from the run's calendar day while the ENDPOINT came from
        the series, which paired 18 Aug with a 14 Aug close and reported a
        five-session move over three. The window now ends on the last session
        the series actually observed, so it spans five real sessions (7->14 Aug)
        and says what it measured.

        When Yahoo DOES quote the instrument the stamp writes that quote, dated
        by its own ``regularMarketTime`` — the live intraday point while the
        venue trades, the completed close once it shuts — so this branch is
        reached only when there is genuinely no current price.
        """
        stale = _closes([
            ("2026-08-07", 28.8), ("2026-08-10", 29.0), ("2026-08-11", 29.1),
            ("2026-08-12", 29.2), ("2026-08-13", 29.4), ("2026-08-14", 29.6),
        ])
        assert window_anchor(stale, "5d") == pd.Timestamp("2026-08-07")
        assert round(compute_period_return(stale, "5d"), 2) == round((29.6 / 28.8 - 1) * 100, 2)

    def test_the_money_and_percent_columns_share_one_window(self):
        """The matrix row said "7 days" and mixed two spans: the euros walked
        seven CALENDAR days while the TWROR beside them measured five sessions.
        Both now read window_anchor, so 5D bills the same five sessions.
        """
        from tarzan.export._perf_series import _window_money_pnl

        idx = pd.date_range("2026-08-04", "2026-08-18", freq="D")  # ends Tuesday
        pnl = pd.Series(range(len(idx)), index=idx, dtype=float)    # +€1/day
        actual = pd.Series([1000.0] * len(idx), index=idx)

        gain, _pct = _window_money_pnl(pnl, actual, "5d")
        # Tue 18 Aug back five sessions is Tue 11 Aug: seven calendar days of
        # P&L. The point is that the euros and the percentage bill the SAME
        # window, whatever its length — they used to differ.
        assert gain == 7.0

    def test_one_day_is_unavailable_when_yesterday_is_missing(self):
        """1D names its span exactly, so it may not absorb a missing bar.

        The 18 Aug 2026 shape: the series HAS today, but Monday 17 Aug never
        printed. The anchor then slides back to Friday and a two-session move
        (+1.03% here) used to print under a "1D" label with nothing marking it.
        Longer buckets are unaffected — a month with a hole in it is still a
        month — so 1M still resolves on the same series.
        """
        gapped = _closes([
            ("2026-07-16", 78.0), ("2026-07-17", 78.2),
            ("2026-08-13", 77.5), ("2026-08-14", 77.8),   # Fri; Mon 17 absent
            ("2026-08-18", 78.6),                          # today
        ])
        assert window_anchor(gapped, "1d") is None
        assert compute_period_return(gapped, "1d") is None
        # The same series still reports 1M, measured from 17 Jul.
        assert window_anchor(gapped, "1m") == pd.Timestamp("2026-07-17")

    def test_one_day_is_the_previous_session_across_a_weekend(self, monkeypatch):
        """Friday→Monday is one session apart, so 1D stays available.

        The guard above must not fire on a Monday run: a weekend is not a
        missing session. Pins today to the Monday (overriding the class fixture,
        whose 18 Aug would make Monday the series end and collapse the anchor).
        """
        import datetime
        monkeypatch.setattr(
            "tarzan.runtime.today", lambda: datetime.date(2026, 8, 17))
        s = _closes([
            ("2026-08-12", 100.0), ("2026-08-13", 100.5),
            ("2026-08-14", 101.0),                          # Friday
            ("2026-08-17", 102.0),                          # Monday = today
        ])
        assert window_anchor(s, "1d") == pd.Timestamp("2026-08-14")
        assert round(compute_period_return(s, "1d"), 4) == round(
            (102.0 / 101.0 - 1) * 100, 4)

    def test_a_bucket_the_series_cannot_cover_is_unavailable(self):
        """Yahoo silently shows MAX for a range longer than the history (EXUS.MI
        reported the same +38.04% under 3y and 5y over 2.3 years of data).
        Tarzan reports the bucket as absent instead of relabelling a short
        window."""
        s = _business_series("2024-05-13", "2026-08-18")
        assert window_anchor(s, "5y") is None
        assert compute_period_return(s, "5y") is None


class TestWindowMatrixHasOneSource:
    """Every WINDOW row, including 1D, is derived from the one price series.

    The 1D row used to be overwritten by a separate live-quote move
    (``performance_full["1d"]``, de-compounded against invested_value), so the
    row could print a number the series never contained — a four-figure loss
    beside a +€11 Session tile on 19 Aug 2026. The override is gone: the row
    reads the same
    ``pnl_series`` / ``portfolio_history`` as the 5D and 1M rows, whose current
    point the engine has already stamped from the live valuation. A divergent
    ``performance_full`` must not be able to move it.
    """

    def _matrix(self, monkeypatch, injected_1d):
        import datetime
        import re

        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_performance30
        from tarzan.models.investor_config import InvestorConfig
        from tarzan.models.portfolio import PortfolioMetrics

        monkeypatch.setattr("tarzan.runtime.today", lambda: datetime.date(2026, 8, 19))
        idx = pd.date_range("2026-06-20", "2026-08-19", freq="D")
        values = []
        for day in idx:
            value = 100_000.0
            if day >= pd.Timestamp("2026-08-18"):
                value = 96_000.0          # yesterday dropped 4%
            if day == pd.Timestamp("2026-08-19"):
                value = 96_010.0          # today is flat: +€10
            values.append(value)
        actual = pd.Series(values, index=idx)
        pnl = actual - 90_000.0
        metrics = PortfolioMetrics(
            total_value=96_010.0, invested_value=96_010.0, cash_value=0.0,
            holdings_df=pd.DataFrame([{"cost_basis_eur": 90_000.0, "ticker": "AAA",
                                       "weight_pct": 100.0}]))
        metrics.actual_value_series = actual
        metrics.pnl_series = pnl
        metrics.unrealized_series = pnl
        metrics.portfolio_history = pd.Series([v / 1000 for v in values], index=idx)
        metrics.pnl_eur, metrics.pnl_pct, metrics.twror_pct = 6_010.0, 6.7, 6.5
        metrics.performance_full = {"1d": injected_1d}

        html = _build_performance30(_NewsletterContext(
            metrics=metrics, config=InvestorConfig()))["matrix_html"]
        cells = {}
        for label in ("1D", "5D", "1M"):
            block = re.search(rf">{label}<.*?</tr>", html, flags=re.S).group(0)
            cells[label] = re.findall(r">([+\-−]?€?[\d.,k]+%?)<", block)
        return cells

    def test_the_one_day_row_is_the_series_last_session_move(self, monkeypatch):
        # 19 Aug (Tue) vs 18 Aug (Mon): +€10, +0.01%. The injected +99% must
        # NOT appear — the row has no path to performance_full any more.
        cells = self._matrix(monkeypatch, injected_1d=99.0)
        joined = " ".join(cells["1D"])
        assert "+€10" in joined, joined
        assert "+0.01%" in joined, joined
        assert "99" not in joined, joined

    def test_the_five_day_row_is_series_derived(self, monkeypatch):
        cells = self._matrix(monkeypatch, injected_1d=99.0)
        assert "−3.99%" in " ".join(cells["5D"])


class TestPrevCloseEUR:
    """The previous-close baseline is scaled onto the same ruler as the
    valuation being stamped, so the 1D is the venue's published move for a Milan
    ETF, a US Treasury and a split-mismatched feed alike, and the FX/scale drops
    out of the percentage."""

    def _prev(self, quote, price_eur):
        from tarzan.data.current_session import prev_close_eur
        return prev_close_eur(quote, price_eur)

    def test_eur_listing_reproduces_the_quote_move(self):
        # EXUS.MI: valuation 40.82, quote 40.815/40.81. The stamped 1D
        # (valuation / prev_eur) must equal the quote's own move.
        prev_eur = self._prev({"price": 40.815, "prev_close": 40.81}, 40.82)
        assert round(40.82 / prev_eur - 1, 8) == round(40.815 / 40.81 - 1, 8)

    def test_non_eur_listing_converts_and_the_fx_cancels(self):
        # US Treasury in USD: native 98.60 vs prev 98.38. The EUR valuation is
        # native x 0.86 FX; the converted baseline reproduces the native move.
        native_now, native_prev, fx = 98.60, 98.38, 0.86
        price_eur = native_now * fx
        prev_eur = self._prev({"price": native_now, "prev_close": native_prev}, price_eur)
        assert round(price_eur / prev_eur - 1, 6) == round(native_now / native_prev - 1, 6)

    def test_a_scale_mismatch_does_not_fabricate_a_jump(self):
        # NTSG.MI: history ran at ~29.9, the quote pair at ~25.5 (a split in one
        # feed only). Pairing 29.9 against a raw 25.8 gave +16%; scaling makes
        # the 1D the quote's real -1.12% move instead.
        prev_eur = self._prev({"price": 25.515, "prev_close": 25.805}, 29.9)
        assert round(29.9 / prev_eur - 1, 6) == round(25.515 / 25.805 - 1, 6)

    def test_a_bond_without_a_quote_keeps_the_feed(self):
        # No Yahoo quote (the EIB ZAR bond is priced via Borsa/synthetic):
        # None tells the caller to leave the feed's own close in place.
        assert self._prev({}, 100.0) is None
        assert self._prev({"prev_close": 99.0}, 100.0) is None  # no price


class TestStampTodayWritesOnlyTheRealPreviousSession:
    """The published prev_close may only land on the session before today.

    ``regularMarketPreviousClose`` is, by definition, the close before the
    CURRENT session. The stamp used to write it onto whatever the last row
    before today happened to be — so when a venue's daily feed ran several
    sessions behind, a settled close was overwritten with a figure from a
    different day. That corrupts the series every window, the risk block and
    every chart reads, and the price-level gate in ``_pick_quote`` cannot see
    it because the levels are within tolerance.
    """

    def _stamp(self, series, today, value, quote):
        from tarzan.data.current_session import stamp_today
        return stamp_today(series, pd.Timestamp(today), value, quote)

    def test_an_adjacent_previous_session_is_repaired(self):
        # Mon 17 present, run on Tue 18: the vendor's null-close repair this
        # exists for. Monday's row takes the published close, scaled.
        s = _closes([("2026-08-14", 40.96), ("2026-08-17", 40.70)])
        out = self._stamp(s, "2026-08-18", 40.82,
                          {"price": 40.815, "prev_close": 40.81})
        assert round(float(out.loc[pd.Timestamp("2026-08-17")]), 4) == 40.8150
        assert float(out.loc[pd.Timestamp("2026-08-18")]) == 40.82
        # Friday, two sessions back, is untouched.
        assert float(out.loc[pd.Timestamp("2026-08-14")]) == 40.96

    def test_a_lagging_feed_keeps_its_own_settled_closes(self, monkeypatch):
        # This venue last printed Wed 19; the run is Fri 21. Thursday's
        # published close belongs to Thursday, so it is inserted there and
        # Wednesday's settled 102.00 is left alone (it used to be overwritten
        # with the 99.00, corrupting the series every window reads).
        #
        # window_anchor measures back from the run's own clock, so it is pinned
        # to the same Friday — without that this asserts against whatever day
        # the suite happens to run on (see TestTrailingWindowAnchor._pin_today).
        import datetime
        monkeypatch.setattr(
            "tarzan.runtime.today", lambda: datetime.date(2026, 8, 21))
        s = _closes([("2026-08-17", 100.0), ("2026-08-18", 101.0),
                     ("2026-08-19", 102.0)])
        out = self._stamp(s, "2026-08-21", 103.0,
                          {"price": 103.0, "prev_close": 99.0})
        assert float(out.loc[pd.Timestamp("2026-08-19")]) == 102.0, (
            "a settled close from another session must survive the stamp"
        )
        assert float(out.loc[pd.Timestamp("2026-08-20")]) == 99.0, (
            "the published previous close belongs on the previous session"
        )
        assert float(out.loc[pd.Timestamp("2026-08-21")]) == 103.0
        # Correctly dated, the repair also makes the 1D available and right.
        assert window_anchor(out, "1d") == pd.Timestamp("2026-08-20")
        assert round(compute_period_return(out, "1d"), 4) == round(
            (103.0 / 99.0 - 1) * 100, 4)

    def test_a_monday_run_repairs_the_friday_close(self):
        # The previous session across a weekend is Friday, not "yesterday".
        s = _closes([("2026-08-13", 50.0), ("2026-08-14", 51.0)])
        out = self._stamp(s, "2026-08-17", 52.0,
                          {"price": 52.0, "prev_close": 51.5})
        assert float(out.loc[pd.Timestamp("2026-08-14")]) == 51.5
        assert float(out.loc[pd.Timestamp("2026-08-13")]) == 50.0


class TestPickQuoteSanityGate:
    """_current_prices picks the quote whose level matches the instrument, the
    same sibling-aware resolution the intraday feed uses — so a corrupt canonical
    quote is skipped for a clean sibling instead of poisoning the baseline."""

    def _pick(self, symbols, quotes, ref):
        from tarzan.data.current_session import pick_quote
        return pick_quote(symbols, quotes, ref)

    def test_clean_canonical_is_used(self):
        q = self._pick(["EXUS.MI", "EXUS.DE"],
                       {"EXUS.MI": {"price": 40.3, "prev_close": 40.8}}, 40.31)
        assert q.get("prev_close") == 40.8

    def test_corrupt_canonical_falls_through_to_the_sibling(self):
        # NTSG.MI on 20 Aug 2026: quote 25.5 against a ~29.4 valuation → rejected;
        # the .DE sibling at 29.35 is within tolerance and supplies the close.
        q = self._pick(
            ["NTSG.MI", "NTSG.DE"],
            {"NTSG.MI": {"price": 25.515, "prev_close": 25.805},
             "NTSG.DE": {"price": 29.35, "prev_close": 29.45}},
            29.7)
        assert q.get("prev_close") == 29.45

    def test_all_off_scale_returns_empty(self):
        # Nothing agrees with the valuation → {}, so the caller keeps the feed.
        q = self._pick(["X.MI", "X.DE"],
                       {"X.MI": {"price": 10.0, "prev_close": 10.0},
                        "X.DE": {"price": 200.0, "prev_close": 200.0}},
                       29.7)
        assert q == {}


class TestOneDayIsTheQuotePairThroughTheSeries:
    """The 1D is still the published pair — it reaches the reader once, through
    the stamped price series, instead of being recomputed as a second answer.

    ``intraday_feeds`` used to also compute the percentage itself, and the value
    was discarded: every table and chart reads
    ``compute_period_return(price_series, "1d")``. Since the stamp writes BOTH
    endpoints of that window from the same quote pair, the series now yields the
    published move by construction — which is what these assert, on the series
    the reader actually sees.
    """

    def _intraday(self, values, day, start, tz="Europe/Rome", freq="15min"):
        idx = pd.date_range(f"{day} {start}", periods=len(values), freq=freq, tz=tz)
        return pd.Series([float(v) for v in values], index=idx)

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        # Tuesday 08:58 CEST: Milan shut, the session being measured is Monday's.
        pinned = pd.Timestamp("2026-08-18 08:58", tz="Europe/Rome")
        monkeypatch.setattr(mq, "_intraday_reference_now", lambda tzinfo=None: pinned)
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        monkeypatch.setattr(
            "tarzan.runtime.today",
            lambda: __import__("datetime").date(2026, 8, 18))

    def test_a_null_close_in_the_chart_still_yields_the_published_move(self):
        """The 18 Aug 2026 shape: Monday 17 absent from the daily feed, so the
        chart's own prior close is FRIDAY's and a close-to-close read could only
        measure two sessions. The published pair carries Monday's official
        40.81, which the stamp writes on Monday's own date — so the 1D is the
        +0.0123% the instrument's page showed, not a two-session move."""
        from tarzan.data.current_session import stamp_today

        chart = _closes([("2026-08-13", 40.96), ("2026-08-14", 40.81)])
        stamped = stamp_today(chart, pd.Timestamp("2026-08-18"), 40.815,
                              {"price": 40.815, "prev_close": 40.81})

        assert float(stamped.loc[pd.Timestamp("2026-08-17")]) == 40.81
        assert window_anchor(stamped, "1d") == pd.Timestamp("2026-08-17")
        assert round(compute_period_return(stamped, "1d"), 4) == round(
            (40.815 / 40.81 - 1) * 100, 4)

    def test_the_feed_resolver_carries_the_baseline_and_liveness(self, monkeypatch):
        """What the resolver still owns: the bars to draw, their 0% baseline and
        whether the venue is trading — never a percentage."""
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {
            "EXUS.MI": self._intraday([40.79, 40.825], "2026-08-17", "16:45")})
        monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda s: pd.DataFrame(
            {"Close": [40.96, 40.81, None]},
            index=pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-17"])))
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "EXUS.MI": {"price": 40.815, "prev_close": 40.81}})

        selected = mq.intraday_feeds(["EXUS.MI"])["EXUS.MI"]

        assert "pct" not in selected, "the resolver no longer answers the 1D"
        assert selected["intraday_baseline"] == 40.81
        assert selected["live"] is False          # Milan is not trading at 08:58
        assert selected["intraday_source_ticker"] == "EXUS.MI"

    def test_an_instrument_with_no_intraday_feed_still_carries_a_baseline(
            self, monkeypatch):
        """No bars to draw, but the published previous close is still the pill's
        reference — and its presence is what says a live quote exists."""
        monkeypatch.setattr(mq, "_fetch_intraday", lambda symbols: {})
        monkeypatch.setattr("tarzan.data.enricher._fetch_history", lambda s: None)
        monkeypatch.setattr(mq, "_fetch_official_quotes", lambda symbols: {
            "X710.MI": {"price": 100.5, "prev_close": 100.0}})

        selected = mq.intraday_feeds(["X710.MI"])["X710.MI"]

        assert selected["intraday_series"] is None
        assert selected["intraday_baseline"] == 100.0
        assert selected["live"] is False


class TestBenchmarkWeekendContamination:
    """A weekend-dated close in a benchmark series steals the window anchor.

    MSCI ACWI (ISAC.MI) is a weekday-traded ETF, so it can only have real
    closes Mon–Fri. When ``convert_to_eur`` (which aligns against a ~24/5 FX
    series) or a weekend "today" stamp injects a Saturday/Sunday row, that row
    sits nearer the one-month cutoff than the real close and ``window_anchor``
    picks it — the reported +1.7% against a real +0.08% on 22 Aug 2026.
    ``_drop_weekends`` removes such rows at the benchmark boundary.
    """

    def test_interior_weekend_point_steals_1m_anchor(self):
        from tarzan.engine.benchmarks import _drop_weekends

        # Real weekday closes flat at ~106; a spurious Saturday FX point at
        # 104.3 lands just after the ~1M cutoff.
        pairs = [(f"2026-07-{d:02d}", 106.0) for d in (20, 21, 22, 23, 24)]
        pairs += [(f"2026-08-{d:02d}", 106.1)
                  for d in (17, 18, 19, 20, 21)]  # Mon–Fri
        contaminated = _closes(pairs + [("2026-07-25", 104.3)]).sort_index()
        # 2026-07-25 is a Saturday.
        assert (contaminated.index.weekday >= 5).sum() == 1

        cleaned = _drop_weekends(contaminated)
        assert (cleaned.index.weekday >= 5).sum() == 0
        assert len(cleaned) == len(contaminated) - 1

    def test_drop_weekends_is_a_noop_on_clean_weekday_series(self):
        from tarzan.engine.benchmarks import _drop_weekends

        clean = _business_series("2026-05-01", "2026-08-21")
        out = _drop_weekends(clean)
        assert len(out) == len(clean)
        assert out.equals(clean)

    def test_tz_aware_friday_close_is_kept(self):
        # Yahoo stamps Milan daily bars tz-aware; a Friday bar must survive.
        from tarzan.engine.benchmarks import _drop_weekends

        s = pd.Series(
            [1.0, 2.0],
            index=pd.to_datetime(
                ["2026-08-14T00:00:00+00:00", "2026-08-15T00:00:00+00:00"]
            ),  # Fri, Sat (UTC)
        )
        out = _drop_weekends(s)
        assert list(out.index.strftime("%Y-%m-%d")) == ["2026-08-14"]


class TestWeekendWindowEnd:
    """A window run on a weekend must measure from the last SESSION, not the
    calendar day. On Sun 29 Jun 2025 the golden's 1D was blank (anchor == last
    close) and every window ran a day long; rolling 'today' back to the last
    business day makes a weekend run measure exactly what Yahoo shows from
    Friday's close."""

    def test_window_end_rolls_a_weekend_today_back_to_friday(self, monkeypatch):
        import tarzan.runtime as rt
        from tarzan.engine.stats import _window_end
        # Sunday 2026-08-23; the series' last real close is Friday 2026-08-21.
        monkeypatch.setattr(rt, "today", lambda: __import__("datetime").date(2026, 8, 23))
        series_end = pd.Timestamp("2026-08-21")
        assert _window_end(series_end) == pd.Timestamp("2026-08-21")  # Friday, not Sunday

    def test_the_window_ends_where_the_data_ends(self, monkeypatch):
        import tarzan.runtime as rt
        from tarzan.engine.stats import _window_end
        # Friday 2026-08-21, but this feed's last observation is Wednesday. The
        # window ends there: pairing a start counted from Friday with a Wednesday
        # endpoint is what made every window a session short of the published
        # figure. A quote for Friday would have been stamped and BE the last
        # observation, so this is the no-current-price case.
        monkeypatch.setattr(rt, "today", lambda: __import__("datetime").date(2026, 8, 21))
        assert _window_end(pd.Timestamp("2026-08-19")) == pd.Timestamp("2026-08-19")

    def test_a_carried_weekend_row_still_rolls_to_the_last_session(self, monkeypatch):
        """The order-derived portfolio NAV is calendar-daily, so its last row is
        routinely a Sunday carrying Friday's value."""
        import tarzan.runtime as rt
        from tarzan.engine.stats import _window_end
        monkeypatch.setattr(rt, "today", lambda: __import__("datetime").date(2026, 8, 23))
        assert _window_end(pd.Timestamp("2026-08-23")) == pd.Timestamp("2026-08-21")


class TestATzAwareSeriesKeepsItsOwnClock:
    """A daily bar is stamped at the VENUE's midnight, and both the weekend
    filter and the window cutoff have to respect that.

    Two bugs met here and both blanked a 1D in a live digest (24 Aug 2026):

    * ``_drop_weekends`` collapsed the index to UTC before reading the weekday,
      so a venue east of UTC moved back a day — Milan's Monday ``00:00+02:00``
      is Sunday ``22:00`` in UTC. It deleted EVERY MONDAY from every EU-listed
      benchmark series (256 of 1294 rows on XDEQ.MI, and 100% of the dropped
      rows were Mondays), which also left the series ending on a Friday so the
      Monday 1D collapsed onto the series end;
    * the session cutoff was rebuilt at midnight UTC while a US bar sits at
      ``04:00+00:00``, so the cutoff sorted BEFORE the session it named and the
      anchor slid one session early — the 1D then failed its own adjacency
      guard for every US listing.
    """

    def _us_series(self):
        """Daily closes as yfinance returns them for a US listing: local
        midnight, which is 04:00 in UTC."""
        idx = pd.DatetimeIndex([
            pd.Timestamp(f"2026-08-{d} 00:00", tz="America/New_York")
            for d in (18, 19, 20, 21, 24)
        ]).tz_convert("UTC")
        return pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)

    def _milan_series(self):
        idx = pd.DatetimeIndex([
            pd.Timestamp(f"2026-08-{d} 00:00", tz="Europe/Rome")
            for d in (18, 19, 20, 21, 24)
        ])
        return pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)

    def test_a_us_listing_reports_its_one_day(self, monkeypatch):
        import datetime
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 24))
        s = self._us_series()
        assert window_anchor(s, "1d", "RSST") is not None, (
            "a midnight-UTC cutoff sorts before a 04:00+00:00 session bar"
        )
        assert round(compute_period_return(s, "1d"), 4) == round(
            (104.0 / 103.0 - 1) * 100, 4)

    def test_the_weekend_filter_keeps_a_monday_east_of_utc(self):
        from tarzan.engine.benchmarks import _drop_weekends

        kept = _drop_weekends(self._milan_series())
        assert len(kept) == 5, "no real session may be dropped"
        assert pd.Timestamp("2026-08-24 00:00", tz="Europe/Rome") in kept.index

    def test_the_weekend_filter_still_drops_a_real_weekend_row(self):
        """The FX-conversion artifact it exists for: a Saturday-dated close."""
        from tarzan.engine.benchmarks import _drop_weekends

        s = self._milan_series()
        s.loc[pd.Timestamp("2026-08-22 00:00", tz="Europe/Rome")] = 99.0
        kept = _drop_weekends(s.sort_index())
        assert len(kept) == 5
        assert all(ts.weekday() < 5 for ts in kept.index)

    def test_a_milan_listing_reports_its_monday_one_day(self, monkeypatch):
        import datetime
        from tarzan.engine.benchmarks import _drop_weekends
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 24))
        s = _drop_weekends(self._milan_series())
        assert round(compute_period_return(s, "1d"), 4) == round(
            (104.0 / 103.0 - 1) * 100, 4)


class TestAWindowEdgeIsASessionDateNotATimestamp:
    """The same session must anchor identically however its bar is stamped.

    A daily bar carries its venue's midnight, and ``convert_to_eur`` restamps a
    US series in UTC — so one session is ``00:00-04:00`` natively and
    ``04:00+00:00`` converted. ``_window_end`` may also come from the run's own
    clock, at midnight. Comparing those timestamps put the cutoff hours before
    the session it named, so the anchor slid one session early and only for the
    converted series: RSSY's 5D then measured six sessions, and WTIP's EUR
    return reconciled against neither its native return nor its native return
    times the FX move. Reported from a live digest against WTIP's and RSSY's
    Yahoo pages.

    Verified on the real cache after the fix: for WTIP, RSSY, RSST, CTAP, GDE
    and RSSX across 5D/1M/3M/1Y, ``tarzan_eur == native x fx`` to 0.0000pp.
    """

    def _pair(self):
        """The same sessions, stamped natively and as converted to UTC.

        Six months of them, so the monthly buckets have real history to anchor
        in rather than falling into the short-series branch.
        """
        idx = pd.bdate_range("2026-03-02", "2026-08-24", tz="America/New_York")
        native = pd.Series(
            [100.0 + i * 0.1 for i in range(len(idx))], index=idx)
        converted = native.copy()
        converted.index = native.index.tz_convert("UTC")
        return native, converted

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        # A Tuesday run: ``_window_end`` comes from the clock at midnight, which
        # is the case that broke — the series' own last bar is the Monday.
        import datetime
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 25))

    @pytest.mark.parametrize("bucket", ["5d", "1m", "3m"])
    def test_both_stampings_anchor_on_the_same_session(self, bucket):
        native, converted = self._pair()
        a_native = window_anchor(native, bucket, "RSSY")
        a_conv = window_anchor(converted, bucket, "RSSY")
        assert a_native is not None and a_conv is not None
        assert a_native.date() == a_conv.date(), (
            f"{bucket}: native anchored {a_native.date()}, "
            f"converted anchored {a_conv.date()}"
        )

    @pytest.mark.parametrize("bucket", ["5d", "1m", "3m"])
    def test_both_stampings_return_the_same_percentage(self, bucket):
        native, converted = self._pair()
        assert round(compute_period_return(native, bucket), 9) == round(
            compute_period_return(converted, bucket), 9)

    def test_a_five_session_window_really_spans_five_sessions(self):
        """The bug made it six: 18..24 Aug instead of 19..24."""
        import datetime as _dt
        _native, converted = self._pair()
        # The series' last observation is Mon 24 Aug, and five sessions back
        # from it is Mon 17 Aug — the anchor Yahoo's own page uses.
        assert window_anchor(converted, "5d", "RSSY").date() == _dt.date(2026, 8, 17)
