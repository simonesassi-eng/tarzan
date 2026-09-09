"""The public run log must not carry the book.

Fixtures below are the REAL leaking lines, copied from a public
``newsletter.yml`` run's log, with the figures replaced by synthetic ones.
"""

from __future__ import annotations

import io
import logging

from tarzan.log_redaction import RedactingFilter, install


class TestTheLeakingLinesAreRedacted:
    @staticmethod
    def _r(text: str) -> str:
        return RedactingFilter().redact(text)

    def test_both_portfolio_total_lines_lose_the_figure(self):
        """The two call sites that published it, in their own formats. The
        trailing "EUR" goes with the number — it is part of the amount."""
        enricher = "Enrichment complete. Total portfolio value: 123456.78 EUR"  # synthetic
        orchestrator = "Total portfolio value: €123456.78"  # synthetic
        assert self._r(enricher) == "Enrichment complete. Total portfolio value: <amount>"
        assert self._r(orchestrator) == "Total portfolio value: <amount>"
        assert "9,850" not in self._r("priced and correct at EUR 9,850.00")

    def test_an_isin_becomes_a_token(self):
        out = self._r("Resolved ISIN IE00B4L5Y983 → EUNL.DE (from cache)")
        assert "IE00B4L5Y983" not in out
        assert "EUNL.DE" not in out
        assert "ISIN#" in out and "SYM#" in out

    def test_the_vendors_own_line_is_covered_too(self):
        """yfinance logs the failing symbol to its own logger, which is why the
        filter sits on the handler rather than at Tarzan's call sites."""
        out = self._r("$SGLD.L: possibly delisted; no price data found")
        assert "SGLD" not in out

    def test_a_failure_stays_followable(self):
        """One symbol keeps one token, so a multi-line diagnostic still reads."""
        f = RedactingFilter()
        first = f.redact("intraday fallback exhausted for CL2.MI (tried X25E.DE)")
        second = f.redact("CL2.MI resolved on retry")
        token = first.split("for ")[1].split(" ")[0]
        assert token.startswith("SYM#")
        assert second.startswith(token), (first, second)

    def test_two_runs_cannot_be_linked(self):
        """A bare hash of a ticker is guessable — the ETF universe is small — so
        the salt is per process and the tokens do not survive across runs."""
        a = RedactingFilter().redact("CL2.MI")
        b = RedactingFilter().redact("CL2.MI")
        assert a != b, "tokens must not be stable across processes"


class TestItLeavesTheRestOfTheLogAlone:
    @staticmethod
    def _r(text: str) -> str:
        return RedactingFilter().redact(text)

    def test_percentages_and_dates_survive(self):
        """A log with no percentages or timestamps is not worth keeping."""
        line = "1D +0.44% · TWROR +11.28% · beta 0.80 on 2026-09-08T15:37:45Z"
        assert self._r(line) == line

    def test_prose_and_module_names_are_not_symbols(self):
        line = "tarzan.data.enricher: see Fig.2 and the No.4 case"
        assert self._r(line) == line


class TestInstallation:
    def test_it_filters_records_that_reach_the_handler(self):
        """End to end: a child logger's record, through the root handler."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            install(enabled=True)
            logging.getLogger("tarzan.data.enricher").warning(
                "Total portfolio value: %.2f EUR", 123456.78)  # synthetic
            assert "123456" not in stream.getvalue(), stream.getvalue()
            assert "<amount>" in stream.getvalue()
        finally:
            root.removeHandler(handler)
            for h in root.handlers:
                h.filters = [f for f in h.filters
                             if not isinstance(f, RedactingFilter)]

    def test_off_locally_on_under_ci(self, monkeypatch):
        """Local runs keep their full detail; CI is the public one."""
        monkeypatch.delenv("CI", raising=False)
        assert install() is None
        monkeypatch.setenv("CI", "true")
        f = install()
        assert isinstance(f, RedactingFilter)
        for h in logging.getLogger().handlers:
            h.filters = [x for x in h.filters if x is not f]
