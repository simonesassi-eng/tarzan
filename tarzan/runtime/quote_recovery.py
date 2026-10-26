"""Causal recovery for one native Yahoo quote-group acquisition.

Native HTTP attempts are never removed. Only retry exhaustion of the exact
group GET is held until its independently acquired native cores complete.
Cookies, summaries, other groups, owners, deadlines and cancellations retain
their original provider failure policy.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from threading import RLock
from urllib.parse import parse_qs, urlsplit


_group = ContextVar("tarzan_native_quote_recovery", default=None)
_request = ContextVar("tarzan_native_quote_recovery_request", default=None)
_RETRY_EXHAUSTED = "PROVIDER_RETRY_BUDGET_EXHAUSTED"


def _positive_json_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, ValueError):
        return False


@dataclass(eq=False)
class _QuoteRecovery:
    budget: object
    symbols: tuple[str, ...]
    completed: set[str] = field(default_factory=set)
    deferred: int = 0
    closed: bool = False
    lock: RLock = field(default_factory=RLock)

    def accepts_request(self, method, url, parameters):
        if method != "GET" or type(url) is not str:
            return False
        try:
            parts = urlsplit(url)
            if (
                parts.scheme != "https"
                or parts.hostname != "query1.finance.yahoo.com"
                or parts.path != "/v7/finance/quote"
            ):
                return False
            supplied = parse_qs(parts.query, keep_blank_values=True).get("symbols")
            if parameters is not None:
                if type(parameters) is not dict or supplied is not None:
                    return False
                value = parameters.get("symbols")
                supplied = [value] if type(value) is str else None
            return (
                type(supplied) is list and len(supplied) == 1
                and tuple(supplied[0].split(",")) == self.symbols
            )
        except (TypeError, ValueError):
            return False


def _matching_group(budget):
    from tarzan.runtime.session import current_session

    group = _group.get()
    session = current_session()
    if (
        type(group) is _QuoteRecovery and group.budget is budget
        and session is not None and session.transport is budget
        and not group.closed
    ):
        return group
    return None


@contextmanager
def request_scope(budget, method, url, parameters):
    """Bind only the physical group request, never its authentication calls."""
    group = _matching_group(budget)
    if group is None or not group.accepts_request(method, url, parameters):
        yield
        return
    token = _request.set(group)
    try:
        yield
    finally:
        _request.reset(token)


def defer_retry_failure(budget, provider, code):
    """Decline all failures except this exact group's native retry exhaustion."""
    group = _matching_group(budget)
    if (
        group is None or _request.get() is not group
        or provider != "yahoo" or code != _RETRY_EXHAUSTED
    ):
        return False
    with group.lock:
        if group.closed:
            return False
        group.deferred += 1
    return True


def completed_native_core(client):
    """Observe an actual private native parser result, not a caller boolean."""
    from tarzan.data import yahoo_info
    from tarzan.runtime.session import current_session

    session = current_session()
    group = _matching_group(session.transport) if session is not None else None
    if group is None or not yahoo_info._native_owned_core(client):
        return
    symbol = vars(client).get("ticker")
    quote = client._quote
    info = vars(quote).get("_info")
    if (
        type(symbol) is not str or symbol not in group.symbols
        or vars(quote).get("_symbol") != symbol
        or vars(quote).get("_already_fetched") is not True
        or type(info) is not dict
        or type(info.get("currency")) is not str or not info["currency"].strip()
        or not any(
            _positive_json_number(info.get(name))
            for name in ("regularMarketPrice", "previousClose", "regularMarketPreviousClose")
        )
    ):
        return
    with group.lock:
        if not group.closed:
            group.completed.add(symbol)


@contextmanager
def acquisition(budget, symbols):
    """Commit unresolved failure on every exit; retain recovered attempts."""
    from tarzan.runtime.ledger import LedgerEntryType
    from tarzan.runtime.session import current_session

    session = current_session()
    ordered = tuple(symbols)
    if (
        session is None or session.transport is not budget
        or not 2 <= len(ordered) <= 64 or len(set(ordered)) != len(ordered)
        or any(type(symbol) is not str or not symbol or symbol != symbol.strip()
               or "," in symbol for symbol in ordered)
        or _group.get() is not None
    ):
        raise ValueError("native quote recovery requires one exact owned group")
    group = _QuoteRecovery(budget, ordered)
    token = _group.set(group)
    successful_exit = False
    try:
        yield
        successful_exit = True
    finally:
        _group.reset(token)
        with group.lock:
            group.closed = True
            deferred = group.deferred
            recovered = successful_exit and group.completed == set(ordered)
        if deferred:
            if not recovered:
                budget._failure("yahoo", _RETRY_EXHAUSTED)
            elif budget.ledger is not None:
                budget.ledger.append(LedgerEntryType.TELEMETRY, {
                    "source": "yahoo", "operation": "native_quote_group",
                    "outcome": "RECOVERED",
                    "exhausted_group_requests": deferred,
                    "independently_completed_cores": len(ordered),
                })
