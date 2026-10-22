"""Public console output is a fixed status projection, not financial prose.

The legacy text redactor remains for explicitly requested local use. Automation
uses a formatter that never formats the message, arguments or traceback. Bare
symbols, quantities, percentages, paths, provider bodies and addresses therefore
cannot evade a regex. Original records remain intact for owned private reports.
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


_PUBLIC_CODES = frozenset({
    "SMTP_ACKNOWLEDGED", "SKIPPED_STALE", "BLOCKED_INVALID_REQUEST",
    "NEWSLETTER_ENTRYPOINT_FAILED", "SEMANTIC_VALIDATION_FAILED",
    "CLAIM_STORE_UNAVAILABLE", "DELIVERY_INTENT_CONFLICT",
    "SUPPRESSED_CONCURRENT", "SUPPRESSED_DUPLICATE", "DRY_RUN",
    "PUBLICATION_BLOCKED", "PRE_DELIVERY_CHECKPOINT_FAILED",
    "UNCERTAIN", "DEFINITE_PRE_SEND_FAILURE", "DELIVERY_ACK_PERSISTENCE_FAILED",
    "ANALYSIS_STARTED", "ANALYSIS_COMPLETE",
    "PROVIDER_BUDGET_EXHAUSTED", "PROVIDER_RETRY_BUDGET_EXHAUSTED",
    "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED", "ESSENTIAL_BENCHMARK_UNAVAILABLE",
    "INPUT_STAGING_CLEANUP_FAILED", "DRIVE_CLIENT_CLEANUP_FAILED",
    "PROVIDER_CLIENT_CLEANUP_FAILED",
})


class RestrictedConsoleFormatter(logging.Formatter):
    def __init__(self, original=None):
        super().__init__()
        self.original = original

    def format(self, record):
        level = record.levelname if isinstance(record.levelname, str) and record.levelname in {
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        } else "LOG"
        name = record.name if isinstance(record.name, str) else ""
        if name.startswith("tarzan.delivery") or name == "tarzan.newsletter":
            stage = "delivery"
        elif name.startswith("tarzan.data.") or name == "yfinance":
            stage = "market-data"
        elif name.startswith("tarzan.engine."):
            stage = "metrics"
        elif name.startswith("tarzan.export."):
            stage = "render"
        elif name == "tarzan.orchestrator":
            stage = "analysis"
        else:
            stage = "application"
        code = getattr(record, "public_code", None)
        code = code if isinstance(code, str) and code in _PUBLIC_CODES else "PRIVATE_DIAGNOSTIC"
        return f"[{level}] {stage}: {code}"


def install(*, enabled: bool | None = None) -> RestrictedConsoleFormatter | None:
    """Restrict existing console handlers; never mutate private capture records."""
    if enabled is None:
        enabled = any(os.environ.get(key, "").lower() in {"1", "true", "yes"} for key in (
            "CI", "GITHUB_ACTIONS", "TARZAN_RESTRICTED_CONSOLE",
        ))
    root = logging.getLogger()
    handlers = list(root.handlers)
    for value in root.manager.loggerDict.values():
        if isinstance(value, logging.Logger):
            handlers.extend(value.handlers)
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    selected = None
    for handler in dict.fromkeys(handlers):
        if getattr(handler, "_tarzan_private_capture", False):
            continue
        if not isinstance(handler, logging.StreamHandler) or isinstance(handler, logging.FileHandler):
            continue
        # Remove only this module's obsolete in-place filter.
        handler.filters = [item for item in handler.filters if type(item) is not RedactingFilter]
        existing = handler.formatter
        if enabled:
            selected = existing if isinstance(existing, RestrictedConsoleFormatter) else RestrictedConsoleFormatter(existing)
            handler.setFormatter(selected)
        elif isinstance(existing, RestrictedConsoleFormatter):
            handler.setFormatter(existing.original)
    if enabled:
        logging.captureWarnings(True)
    return selected if enabled else None
