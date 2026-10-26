"""Keep native metadata from relabeling a different provider listing.

The SDK rewrites the first result's symbol before selecting a record. Admit
the completed request's identities first, retaining independent valid records
and native parsing, status, authentication and fallback behavior.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import parse_qs, unquote, urlsplit

from curl_cffi.requests.models import Request


logger = logging.getLogger(__name__)
_MISSING = object()


def _completed_metadata_source(response):
    request = response.request
    if request is not None:
        if type(request) is not Request or request.method != "GET":
            return None
        url = request.url
    else:
        url = response.url
    if type(url) is not str:
        return None
    try:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.hostname not in {
                "query1.finance.yahoo.com", "query2.finance.yahoo.com",
            }
        ):
            return None
        prefix = "/v10/finance/quoteSummary/"
        if parts.path.startswith(prefix):
            symbol = unquote(parts.path.removeprefix(prefix))
            if symbol and symbol == symbol.strip() and "," not in symbol and "/" not in symbol:
                return "quoteSummary", (symbol,)
        if parts.path == "/v7/finance/quote":
            values = parse_qs(parts.query, keep_blank_values=True).get("symbols")
            if type(values) is list and len(values) == 1:
                symbols = values[0].split(",")
                if all(symbol and symbol == symbol.strip() for symbol in symbols):
                    return "quoteResponse", tuple(dict.fromkeys(symbols))
    except ValueError:
        pass
    return None


def _summary_identity(record, symbol):
    # Real quoteSummary often omits the top-level symbol. Its single-symbol
    # native request remains authority when the provider declares no identity.
    # Explicit declarations in issuer modules must never contradict it.
    identities = [record.get("symbol", _MISSING)]
    for name in ("price", "quoteType"):
        module = record.get(name)
        if type(module) is dict:
            identities.append(module.get("symbol", _MISSING))
    return all(
        value is _MISSING or type(value) is str and value == symbol
        for value in identities
    )


def _duplicates_agree(records):
    if len(records) == 1:
        return True
    try:
        encoded = [
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for record in records
        ]
    except (TypeError, ValueError, OverflowError):
        return False
    # Strict JSON comparison also distinguishes bool from numeric 0/1.
    return all(value == encoded[0] for value in encoded[1:])


def admit_metadata_identity(document, response):
    """Admit qualified records before native flattening erases their identity.

    No fields, currencies or prices are invented. A rejected quote source
    cannot erase independently qualified summary/profile data, or vice versa.
    Unknown envelopes and sources keep their original native contract.
    """
    source = _completed_metadata_source(response)
    if source is None or type(document) is not dict:
        return document
    envelope, symbols = source
    container = document.get(envelope)
    if type(container) is not dict or type(container.get("result")) is not list:
        return document
    records = container["result"]
    # Custom-decoder dictionaries keep the caller's uninspected contract.
    if any(isinstance(record, dict) and type(record) is not dict for record in records):
        return document
    requested = set(symbols)
    groups = {}
    rejected = False
    for record in records:
        if type(record) is not dict:
            rejected = True
            continue
        if envelope == "quoteSummary":
            symbol = symbols[0]
            qualified = _summary_identity(record, symbol)
        else:
            symbol = record.get("symbol")
            qualified = type(symbol) is str and symbol in requested
        if not qualified:
            rejected = True
            continue
        groups.setdefault(symbol, []).append(record)
    retained = []
    for group in groups.values():
        if _duplicates_agree(group):
            retained.append(group[0])
        else:
            rejected = True
    container["result"] = retained
    if rejected and not getattr(response, "_tarzan_metadata_identity_reported", False):
        response._tarzan_metadata_identity_reported = True
        try:
            logger.warning(
                "Yahoo metadata contained unqualified or conflicting listing identities; "
                "independent qualified records were retained.",
                extra={"public_code": "YAHOO_METADATA_IDENTITY_REJECTED"},
            )
        except Exception:
            # Optional logging must not discard independent valid provider
            # data. Fatal cancellation still follows the original call path.
            pass
    return document
