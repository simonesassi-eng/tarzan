"""The public run log must not carry the book.

Fixtures below are the REAL leaking lines, copied from a public
``newsletter.yml`` run's log, with the figures replaced by synthetic ones.
"""

from __future__ import annotations

import io
import logging

import pytest

from tarzan.log_redaction import RedactingFilter, RestrictedConsoleFormatter, install


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
        line = "1D +0.44% · TWR +11.28% · beta 0.80 on 2026-09-08T15:37:45Z"
        assert self._r(line) == line

    def test_prose_and_module_names_are_not_symbols(self):
        line = "tarzan.data.enricher: see Fig.2 and the No.4 case"
        assert self._r(line) == line


class TestInstallation:
    def test_it_restricts_records_that_reach_the_handler(self):
        """Public text is withheld, while the original record stays private."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            install(enabled=True)
            logging.getLogger("tarzan.data.enricher").warning(
                "Total portfolio value: %.2f EUR", 123456.78)  # synthetic
            assert "123456" not in stream.getvalue(), stream.getvalue()
            assert "PRIVATE_DIAGNOSTIC" in stream.getvalue()
        finally:
            root.removeHandler(handler)
            install(enabled=False)
            for h in root.handlers:
                h.filters = [f for f in h.filters
                             if not isinstance(f, RedactingFilter)]

    def test_off_locally_on_under_ci(self, monkeypatch):
        """Local runs keep their full detail; CI is the public one."""
        for key in ("CI", "GITHUB_ACTIONS", "TARZAN_RESTRICTED_CONSOLE"):
            monkeypatch.delenv(key, raising=False)
        assert install() is None
        monkeypatch.setenv("CI", "true")
        f = install()
        assert isinstance(f, RestrictedConsoleFormatter)
        install(enabled=False)
        for h in logging.getLogger().handlers:
            h.filters = [x for x in h.filters if x is not f]


def test_public_formatter_never_formats_private_arguments_and_preserves_original_record():
    class PrivateArgument:
        def __str__(self):
            raise AssertionError("public formatter touched private argument")

    argument = PrivateArgument()
    record = logging.LogRecord(
        "tarzan.data.owner@example.invalid", logging.ERROR, "/private/path", 1,
        "quantity=%s", (argument,), (ValueError, ValueError("private-secret"), None),
    )
    record.stack_info = "private-ledger owner@example.invalid"
    before = dict(record.__dict__)
    assert RestrictedConsoleFormatter().format(record) == "[ERROR] market-data: PRIVATE_DIAGNOSTIC"
    assert record.__dict__ == before


@pytest.mark.parametrize("name", ["root", "yfinance", "vendor.private-account", "tarzan.delivery.private"])
def test_public_console_withholds_bare_symbols_quantities_percentages_addresses_and_tracebacks(name):
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    private_records = []

    class PrivateCapture(logging.Handler):
        _tarzan_private_capture = True

        def emit(self, record):
            private_records.append(record)

    logger = logging.getLogger(name)
    original_handlers, original_propagate, original_level = logger.handlers[:], logger.propagate, logger.level
    capture = PrivateCapture()
    logger.handlers, logger.propagate, logger.level = [console, capture], False, logging.DEBUG
    try:
        install(enabled=True)
        try:
            raise ValueError("secret-token PRIVATE quantity=12345 return=12.34%")
        except ValueError:
            logger.exception(
                "AAPL quantity=%s return=12.34%% owner@example.invalid https://private.invalid/token",
                12345,
            )
        output = stream.getvalue()
        assert output.startswith("[ERROR]") and "PRIVATE_DIAGNOSTIC" in output
        for forbidden in ("AAPL", "12345", "12.34", "owner", "private.invalid", "secret", "Traceback"):
            assert forbidden not in output
        assert len(private_records) == 1
        record = private_records[0]
        assert record.args == (12345,) and "AAPL" in record.getMessage()
        assert record.exc_info[1].args == ("secret-token PRIVATE quantity=12345 return=12.34%",)
    finally:
        install(enabled=False)
        logger.handlers, logger.propagate, logger.level = original_handlers, original_propagate, original_level


def test_restricted_formatter_allows_only_fixed_status_codes_and_is_reversible():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    original = logging.Formatter("%(message)s")
    handler.setFormatter(original)
    logger = logging.getLogger("review.private-console")
    logger.addHandler(handler)
    try:
        install(enabled=True)
        first = handler.formatter
        install(enabled=True)
        assert handler.formatter is first
        record = logging.LogRecord("tarzan.delivery", logging.INFO, "", 1, "private", (), None)
        record.public_code = "SMTP_ACKNOWLEDGED"
        assert first.format(record) == "[INFO] delivery: SMTP_ACKNOWLEDGED"
        record.public_code = "owner@example.invalid"
        assert first.format(record) == "[INFO] delivery: PRIVATE_DIAGNOSTIC"
        install(enabled=False)
        assert handler.formatter is original
        assert handler.formatter.format(record) == "private"
    finally:
        logger.removeHandler(handler)
        install(enabled=False)


def test_entrypoint_exception_keeps_private_traceback_out_of_public_console(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    from tarzan.runtime.logging import LOG_CAPTURE

    monkeypatch.setenv("TARZAN_RESTRICTED_CONSOLE", "1")
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    original_streams = {
        handler: handler.stream for handler in original_handlers
        if isinstance(handler, logging.StreamHandler)
    }
    path = Path(__file__).resolve().parents[2] / "scripts/send_newsletter.py"
    spec = importlib.util.spec_from_file_location("review_private_send_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def unexpected():
        raise ValueError("owner@example.invalid private-ledger quantity=12345")

    monkeypatch.setattr(module, "run_and_send", unexpected)
    try:
        assert module.main() == 1
        output = capsys.readouterr()
        assert "NEWSLETTER_ENTRYPOINT_FAILED" in output.out
        assert "private-ledger" not in output.out + output.err
        assert "12345" not in output.out + output.err
        assert "Traceback" not in output.out + output.err
        assert any("private-ledger" in record["message"] for record in LOG_CAPTURE.records)
    finally:
        install(enabled=False)
        for handler in root.handlers[:]:
            if handler not in original_handlers:
                root.removeHandler(handler)
        for handler, stream in original_streams.items():
            handler.stream = stream
        root.setLevel(original_level)
