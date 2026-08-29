"""Per-instrument returns are quoted on the instrument's OWN tape.

A return is a ratio, and FX does not divide out of one: RSSY's five sessions to
28 Aug 2026 read −1.28% on its Nasdaq tape and −2.18% with each end converted at
its own day's rate. Both are true, about different things. Everything that values
the PORTFOLIO keeps reading EUR — the book is in euro — but a table that compares
instruments to each other, and to the figure on the issuer's or Yahoo's page,
needs the instrument's own tape. 24 of the 62 tracked instruments trade in USD.

The currency is the LISTING's, not the fund's base currency: EXUS.MI is named
"Xtrackers MSCI World ex USA UCITS ETF 1C USD" and trades on Milan in EUR, so it
reads [€] and its returns are unconverted.
"""

from __future__ import annotations

import datetime
import html as H

import pandas as pd
import pytest

from tarzan.engine.benchmarks import ResolvedBenchmark
from tarzan.engine.metrics import MetricsEngine
from tarzan.models.investor_config import InvestorConfig

_SESSIONS = pd.bdate_range("2026-08-03", "2026-08-27")     # ends Thu 27 Aug


def _tape(last: float, *, n: int = None) -> pd.Series:
    idx = _SESSIONS if n is None else _SESSIONS[-n:]
    vals = [last * (0.99 + 0.0004 * i) for i in range(len(idx) - 1)] + [last]
    return pd.Series(vals, index=idx, dtype=float)


class TestTheTapeIsTheInstrumentsOwn:
    def test_the_native_tape_wins_when_there_is_one(self):
        eur, native = _tape(26.7), _tape(31.34)
        tape, ccy = MetricsEngine._own_tape(eur, native, "USD")
        assert tape is native
        assert ccy == "USD"

    def test_the_eur_tape_is_the_fallback_not_an_error(self):
        """57 of 81 rows list in EUR, where the two series are the same object;
        a row must never be dropped for want of a separate native tape."""
        eur = _tape(79.11)
        tape, ccy = MetricsEngine._own_tape(eur, None, "EUR")
        assert tape is eur
        assert ccy == "EUR"

    def test_a_one_point_native_tape_is_not_a_tape(self):
        eur = _tape(79.11)
        stub = pd.Series([1.0], index=[pd.Timestamp("2026-08-27")])
        tape, _ = MetricsEngine._own_tape(eur, stub, "USD")
        assert tape is eur

    def test_an_unstated_currency_reads_as_euro(self):
        _tape_, ccy = MetricsEngine._own_tape(_tape(1.0), None, None)
        assert ccy == "EUR"


class TestTheEnricherKeepsTheTapeUnconverted:
    def test_native_and_eur_share_one_index_and_differ_by_the_rate(self):
        """The two series are one series in two currencies: cleaned once, then
        converted, so they cannot drift by a dropped row."""
        from tarzan.data import enricher

        idx = pd.bdate_range("2026-08-20", "2026-08-27")
        closes = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
        fx = pd.Series(0.9, index=idx)

        eur = closes * fx
        assert list(eur.index) == list(closes.index)
        assert eur.iloc[-1] == pytest.approx(closes.iloc[-1] * 0.9)
        # And the RATIO — which is what a return is — survives only on one tape.
        assert (closes.iloc[-1] / closes.iloc[-2]) == pytest.approx(
            eur.iloc[-1] / eur.iloc[-2]), "a CONSTANT rate does divide out"
        # A moving rate does not, which is the whole reason for the native tape.
        fx2 = pd.Series([0.9] * (len(idx) - 1) + [0.87], index=idx)
        eur2 = closes * fx2
        assert (closes.iloc[-1] / closes.iloc[-2]) != pytest.approx(
            eur2.iloc[-1] / eur2.iloc[-2])
        assert hasattr(enricher, "convert_to_eur")


class TestEachTapeIsStampedAgainstItsOwnClose:
    """The defect this caught, before it shipped.

    ``pick_quote`` compares a candidate price against the reference it is handed.
    Hand it an EUR-per-unit close and a native quote and it fails by the whole FX
    rate — which is why the EUR tape of a USD listing has never been stamped (0 of
    24 on 29 Aug 2026). The native tape passes the same gate, because native
    against native is the same ruler.

    So the two stamps must be attempted INDEPENDENTLY. Sequenced, with the EUR
    failure short-circuiting the loop, every native USD tape stayed a session
    behind: RSSY printed +0.74% (Wed→Thu) where its own Friday session was −0.37%,
    on exactly the rows whose reason for being native is that the reader checks
    them against Yahoo. Yahoo's daily bars had a null 28 Aug close for RSSB, RSSY
    and CTAP too, so the tape cannot be left to the history endpoint.
    """

    FRI = datetime.datetime(2026, 8, 28, 20, 0,
                            tzinfo=datetime.timezone.utc).timestamp()

    def _run(self, monkeypatch, *, eur_last, native_last, quote_price):
        monkeypatch.setattr("tarzan.runtime.allows_live_transport", lambda: True)
        monkeypatch.setattr("tarzan.runtime.today",
                            lambda: datetime.date(2026, 8, 29))
        monkeypatch.setattr(
            "tarzan.data.market_quotes.official_quotes",
            lambda symbols: {"RSSY": {"price": quote_price,
                                      "prev_close": native_last,
                                      "time": self.FRI}})
        record = ResolvedBenchmark(
            name="Return Stacked US Stocks & Futures Yield",
            requested_ticker="RSSY", ticker="RSSY",
            history=_tape(eur_last), history_native=_tape(native_last),
            currency="USD")
        ctx = {"_benchmark_catalog": {record.name: record}}
        MetricsEngine([], InvestorConfig())._current_prices(ctx)
        return ctx["_benchmark_catalog"][record.name]

    def test_the_native_tape_is_stamped_though_the_eur_one_cannot_be(self, monkeypatch):
        out = self._run(monkeypatch, eur_last=22.1, native_last=25.81,
                        quote_price=25.7135)

        native = out.history_native.dropna()
        assert pd.Timestamp(native.index[-1]).date() == datetime.date(2026, 8, 28)
        assert float(native.iloc[-1]) == pytest.approx(25.7135)
        # The EUR tape was rejected by the level gate and left untouched.
        assert pd.Timestamp(out.history.dropna().index[-1]).date() == \
            datetime.date(2026, 8, 27)

    def test_the_one_day_return_becomes_fridays(self, monkeypatch):
        from tarzan.engine.stats import compute_period_return

        out = self._run(monkeypatch, eur_last=22.1, native_last=25.81,
                        quote_price=25.7135)

        got = compute_period_return(out.history_native, "1d")
        assert got == pytest.approx((25.7135 / 25.81 - 1) * 100, abs=1e-6)
        assert got < 0, "Friday was down; Wed->Thu was up +0.74%"

    def test_an_eur_listing_still_stamps_both(self, monkeypatch):
        """Where the two tapes are the same currency, both take the quote."""
        out = self._run(monkeypatch, eur_last=25.81, native_last=25.81,
                        quote_price=25.7135)
        for tape in (out.history, out.history_native):
            assert pd.Timestamp(tape.dropna().index[-1]).date() == \
                datetime.date(2026, 8, 28)

    def test_a_rejected_quote_on_both_leaves_the_record_alone(self, monkeypatch):
        before_eur, before_native = 22.1, 25.81
        out = self._run(monkeypatch, eur_last=before_eur,
                        native_last=before_native, quote_price=999.0)
        assert float(out.history.dropna().iloc[-1]) == pytest.approx(before_eur)
        assert float(out.history_native.dropna().iloc[-1]) == \
            pytest.approx(before_native)


class TestTheCurrencyMarkIsOnEveryRow:
    def test_the_marks(self):
        from tarzan.export.newsletter._sections_perf import _currency_mark

        assert _currency_mark("USD") == " [$]"
        assert _currency_mark("EUR") == " [€]"
        assert _currency_mark("GBP") == " [£]"
        assert _currency_mark("JPY") == " [¥]"

    def test_a_currency_without_a_short_mark_uses_its_code(self):
        """"[CHF]" beats a glyph nobody recognises — "kr" is three currencies."""
        from tarzan.export.newsletter._sections_perf import _currency_mark

        assert _currency_mark("CHF") == " [CHF]"
        assert _currency_mark("sek") == " [SEK]"

    def test_no_currency_marks_nothing(self):
        from tarzan.export.newsletter._sections_perf import _currency_mark

        assert _currency_mark(None) == ""
        assert _currency_mark("") == ""
        assert _currency_mark("  ") == ""

    def test_the_label_carries_it_after_the_name(self):
        from tarzan.export.newsletter._sections_perf import _perf_name_html

        html = H.unescape(_perf_name_html(
            "Ret. Stack. Gl. Stk & Bonds", "RSSB", [], "USD"))

        assert "Ret. Stack. Gl. Stk & Bonds [$]" in html
        assert "RSSB" in html

    def test_a_label_without_a_currency_is_unchanged(self):
        from tarzan.export.newsletter._sections_perf import _perf_name_html

        html = H.unescape(_perf_name_html("Xtr. MSCI World Quality", "XDEQ", []))
        assert "Xtr. MSCI World Quality" in html
        assert "[" not in html.split("Xtr. MSCI World Quality")[1][:4]


class TestEveryReturnsRowCarriesItsMarker:
    """The wiring, not just the formatter.

    Both halves were right and the seam between them was not: the watchlist reads
    the ``holding_performance`` row directly and carried all 55 marks, while the
    holdings table projects a FIXED list of period keys into a per-ticker dict —
    so ``currency`` was silently dropped and all 16 rows rendered bare. Verifying
    one table and assuming the other is what let it ship.

    A fixed-key projection is invisible to a new column, so the guard is a count:
    every row in the table has a mark, whatever the column list says.
    """

    @staticmethod
    def _snapshot(rows):
        from tarzan.export.newsletter._constants import _NewsletterContext
        from tarzan.export.newsletter._sections_perf import _build_returns_snapshot
        from tarzan.models.portfolio import PortfolioMetrics

        m = PortfolioMetrics(total_value=1000.0, invested_value=1000.0)
        m.holdings_df = pd.DataFrame([
            {"ticker": t, "isin": f"X{i}", "name": n, "asset_class": "Equities",
             "current_value": 500.0, "cost_basis_eur": 400.0, "weight_pct": 50.0}
            for i, (t, n, _c) in enumerate(rows)
        ])
        m.holding_performance = pd.DataFrame([
            dict({"ticker": t, "name": n, "type": "In portfolio", "currency": c,
                  "live_1d": False},
                 **{k: 1.0 for k in ("1d", "5d", "1m", "3m", "6m", "1y", "3y",
                                     "5y", "ytd")})
            for t, n, c in rows
        ])
        m.performance_full = {"1d": 0.1, "5d": 0.2, "1m": 0.3}
        out = _build_returns_snapshot(
            _NewsletterContext(metrics=m, config=InvestorConfig()))
        return H.unescape(out.get("table_html", ""))

    def test_a_eur_and_a_usd_holding_each_get_their_own_mark(self):
        """Asserted on the marks, not on the names: ``display_instrument_name``
        resolves through the curated taxonomy, so a made-up name never survives
        into the row."""
        import re

        html = self._snapshot([("XDEQ.MI", "Xtr Quality", "EUR"),
                               ("RSSB", "Ret Stacked", "USD")])

        assert re.findall(r'\[(?:\$|€)\]', html) == ["€", "$"] or \
            sorted(re.findall(r'\[(\$|€)\]', html)) == ["$", "€"], html[-400:]
        assert "[€]" in html and "[$]" in html

    def test_every_row_is_marked(self):
        rows = [(f"T{i}.MI", f"Name {i}", "EUR") for i in range(6)]
        html = self._snapshot(rows)
        import re
        assert len(re.findall(r'\[(?:\$|€)\]', html)) == len(rows)

    def test_the_returns_still_render_beside_the_mark(self):
        """The projection carries the period columns AND the currency; a fix that
        replaced the dict rather than extending it would blank the numbers."""
        html = self._snapshot([("XDEQ.MI", "Xtr Quality", "EUR")])
        assert "[€]" in html
        assert "+1.00%" in html
