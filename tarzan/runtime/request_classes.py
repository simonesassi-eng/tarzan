"""Closed native-request observations; never retain request addresses or facts.

Classification observes a trusted Yahoo host/path shape, not its response,
instrument identity or financial usability. Unknown inputs remain OTHER.
Context variables contain only an owner, fixed provider/category and a boolean
annotation mask, never a request address or dynamic request identifier.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit


REQUEST_CLASSES = frozenset({
    "CHART", "QUOTE", "QUOTE_SUMMARY", "FUNDAMENTALS",
    "COOKIE", "CRUMB", "CONSENT", "OTHER",
})
_QUERY_HOSTS = frozenset({"query1.finance.yahoo.com", "query2.finance.yahoo.com"})
_CONSENT_HOSTS = frozenset({"guce.yahoo.com", "consent.yahoo.com"})
_scope = ContextVar("tarzan_native_request_class", default=None)
_annotation_allowed = ContextVar("tarzan_native_request_annotation_allowed", default=True)
# urlsplit's standard LRU would retain the full request, including credentials
# in the query. The pinned interpreter exposes its uncached implementation.
# If another interpreter does not, observe OTHER rather than cache addresses.
_UNCACHED_URLSPLIT = getattr(urlsplit, "__wrapped__", None)


def yahoo_request_class(url) -> str:
    """Project a fixed label without coercing objects or inspecting query facts."""
    if type(url) is not str or not url or len(url) > 16384:
        return "OTHER"
    # urlsplit silently removes some control characters. Such inputs must not
    # acquire a trusted host shape by normalization or invoke object getters.
    if url != url.strip() or any(ord(char) < 32 or ord(char) == 127 for char in url):
        return "OTHER"
    try:
        if _UNCACHED_URLSPLIT is None:
            return "OTHER"
        parsed = _UNCACHED_URLSPLIT(url)
        if (
            parsed.scheme != "https" or parsed.port not in (None, 443)
            or parsed.username is not None or parsed.password is not None
        ):
            return "OTHER"
        host, path = parsed.hostname, parsed.path
        if host == "fc.yahoo.com" and path in ("", "/"):
            return "COOKIE"
        if host in _CONSENT_HOSTS and path in {
            "/consent", "/copyConsent", "/v2/collectConsent",
        }:
            return "CONSENT"
        if host not in _QUERY_HOSTS:
            return "OTHER"
        if path == "/v1/test/getcrumb":
            return "CRUMB"
        if path in {"/v7/finance/quote", "/v6/finance/quote"}:
            return "QUOTE"
        if path.startswith("/v8/finance/chart/") and path.count("/") == 4 and not path.endswith("/"):
            return "CHART"
        if path.startswith("/v10/finance/quoteSummary/") and path.count("/") == 4 and not path.endswith("/"):
            return "QUOTE_SUMMARY"
        if path == "/ws/fundamentals-timeseries/v1/finance/timeseries" or (
            path.startswith("/ws/fundamentals-timeseries/v1/finance/timeseries/")
            and path.count("/") == 6 and not path.endswith("/")
        ):
            return "FUNDAMENTALS"
    except BaseException:
        # An observer cannot replace the real request's success, failure or
        # cancellation. Never stringify an error containing request details.
        return "OTHER"
    return "OTHER"


@contextmanager
def yahoo_request_boundary():
    """An unannotated inner native request cannot inherit its parent's label."""
    state = None
    neutral_state = None
    try:
        state = _annotation_allowed.set(False)
    except BaseException:
        # Independent fallback when the boolean observer itself is unavailable.
        try:
            neutral_state = _scope.set(None)
        except BaseException:
            pass
    try:
        yield
    finally:
        _restore_annotation_allowance(state)
        if neutral_state is not None:
            try:
                _scope.reset(neutral_state)
            except BaseException:
                try:
                    _scope.set(None)
                except BaseException:
                    pass


def _restore_annotation_allowance(state):
    if state is not None:
        try:
            _annotation_allowed.reset(state)
        except BaseException:
            try:
                _annotation_allowed.set(False)
            except BaseException:
                pass


@contextmanager
def yahoo_request_scope(owner, url):
    """Bind only this Yahoo request, including native retries, then restore."""
    try:
        category = yahoo_request_class(url)
        if type(category) is not str or category not in REQUEST_CLASSES:
            category = None
    except BaseException:
        category = None
    # Do not retain even the raw address in an inactive observer-generator
    # frame while the unchanged native Session legitimately uses its input.
    url = None
    token = None
    allowed_state = None
    try:
        token = _scope.set((owner, "yahoo", category))
    except BaseException:
        pass
    if token is not None:
        try:
            allowed_state = _annotation_allowed.set(True)
        except BaseException:
            try:
                _scope.set((owner, "yahoo", None))
            except BaseException:
                pass
    try:
        yield
    finally:
        _restore_annotation_allowance(allowed_state)
        if token is not None:
            try:
                _scope.reset(token)
            except BaseException:
                # Cleanup of optional telemetry cannot mask cancellation or
                # the response. A defensive clear avoids retaining a bad scope.
                try:
                    _scope.set(None)
                except BaseException:
                    pass


def request_class_for(owner, provider):
    """Do not inherit another owner/provider's annotation or dynamic labels."""
    try:
        if _annotation_allowed.get() is not True:
            return None
        observed = _scope.get()
        if (
            type(observed) is tuple and len(observed) == 3
            and observed[0] is owner and type(provider) is str and provider == "yahoo"
            and type(observed[1]) is str and observed[1] == provider
            and type(observed[2]) is str and observed[2] in REQUEST_CLASSES
        ):
            return observed[2]
    except BaseException:
        pass
    return None
