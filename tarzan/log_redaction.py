"""Keep the book out of a PUBLIC run's log.

This repo is public, and a GitHub Actions run's log is public with it — no
login, no fork, no clone needed. Nothing is committed, so git is not the
channel and rewriting history does not touch this: the runner captures the
process's stdout/stderr and GitHub serves the page.

Before this filter, one successful send published the portfolio total twice
(``tarzan.data.enricher`` at the end of enrichment, ``tarzan.orchestrator``
after metrics) plus every ISIN and venue-suffixed symbol it resolved. At
roughly nine sends a day against a 90-day retention that is a daily
net-worth time series, attached to the author's name.

Installed on the root HANDLER, not at the call sites, because the leak is not
one line's fault. Any future log line publishes whatever it formats, and so
does any vendor's own logger — ``yfinance`` writes the failing symbol to its
own. A filter at the handler covers the lines that exist and the ones nobody
has written yet. (A logger-level filter would not: filters on a logger run
only for records logged directly to it, never for records propagated up from
children.)

Amounts and ISINs are dropped outright. Symbols become a per-run pseudonym so
a failure stays followable across the lines that discuss it — "fallback
exhausted for SYM#a3f2, tried SYM#b1c8" still reads as a story — without
naming the holding. The salt is random per process, because a bare hash of a
ticker is not privacy: the universe of European ETF tickers is small enough to
enumerate in seconds. Random-per-run also means two runs' tokens cannot be
linked into a position history.

ponytail: message text only. ``exc_info`` tracebacks are formatted by the
handler after this filter runs, so a repr of a holding inside a traceback
would still print. Nothing logs one today; if that changes, redact in a
Formatter instead of a Filter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets

# Synthetic examples of the shapes this matches: "€123,456.78", "9876.54 EUR",
# "EUR 1,500.00". (Synthetic on purpose — an illustration is not worth leaking.)
_AMOUNT = re.compile(r"(?:€|EUR)\s?\d[\d.,]*\d|\d[\d.,]*\d\s?(?:€|EUR)\b|(?:€|EUR)\s?\d")

# ISO 6166: two-letter country, nine alphanumerics, one check digit.
_ISIN = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")

# Venue-suffixed tickers as the providers spell them. An explicit suffix list
# rather than \.[A-Z]+ so ordinary prose ("Fig.2", "cf.MI") is left alone.
_SYMBOL = re.compile(
    r"\b[A-Z0-9]{1,6}\.(?:MI|DE|L|AS|PA|SW|F|VI|MC|MU|SG|BE|DU|HM|HA|STU|"
    r"ETLX|TO|NE|NX|IR|BR|LS|CO|ST|HE|OL|WA|PR|BD|TI|MX|SA|JO|AX|NZ|T|HK)\b"
)


class RedactingFilter(logging.Filter):
    """Rewrite each record's rendered message with the book taken out."""

    def __init__(self) -> None:
        super().__init__()
        self._salt = secrets.token_bytes(16)
        self._seen: dict[str, str] = {}

    def _token(self, kind: str, value: str) -> str:
        """A stable-within-this-process pseudonym for one identifier."""
        key = f"{kind}:{value}"
        if key not in self._seen:
            digest = hashlib.blake2s(self._salt + key.encode(), digest_size=2)
            self._seen[key] = f"{kind}#{digest.hexdigest()}"
        return self._seen[key]

    def redact(self, text: str) -> str:
        # ISINs first: an "IE00B4L5Y983.MI" is an ISIN carrying a venue suffix,
        # and taking the ISIN out first leaves a bare ".MI" that names nothing.
        text = _ISIN.sub(lambda m: self._token("ISIN", m.group(0)), text)
        text = _SYMBOL.sub(lambda m: self._token("SYM", m.group(0)), text)
        return _AMOUNT.sub("<amount>", text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # a malformed %-format is the caller's bug, not ours
            return True
        redacted = self.redact(rendered)
        if redacted != rendered:
            record.msg, record.args = redacted, ()
        return True


def install(*, enabled: bool | None = None) -> RedactingFilter | None:
    """Attach the filter to every root handler. On by default under ``$CI``.

    Call AFTER ``basicConfig``/``setup_logging``: the filter goes on the
    handlers that exist, so a handler added later is not covered.

    Matches ``scripts/verify_returns_vs_yahoo.py``, which gates its own
    per-instrument redaction on ``$CI`` for the same reason.
    """
    if enabled is None:
        enabled = bool(os.environ.get("CI"))
    if not enabled:
        return None
    f = RedactingFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(f)
    if not root.handlers:  # nothing configured yet — cover the fallback path
        logging.lastResort.addFilter(f)
    return f
