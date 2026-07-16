"""Shared yfinance network layer: anti-throttle spacing + bounded retry.

yfinance scrapes Yahoo's unofficial endpoints, which rate-limit (HTTP 429) on
bursts. This ONE module owns the spacing between calls and the exponential
backoff on transient errors, so every caller — the enricher, the backtest proxy
fetch (:mod:`tarzan.data.proxy_data`) and the geo/ISIN resolver
(:mod:`tarzan.data.geo_resolver`) — shares the same throttle discipline instead
of each rolling its own (or none). A single process-wide interval clock means
the spacing holds ACROSS modules, not just within one.
"""

from __future__ import annotations

import logging
import random
import threading
import time as _time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_YF_MIN_INTERVAL = 0.2           # min spacing between yfinance calls (anti-429)
_MAX_FETCH_ATTEMPTS = 3          # total tries per request before giving up
_BACKOFF_BASE_SECONDS = 0.75     # exponential backoff base

_lock = threading.Lock()
_last_call = [0.0]               # mutable single-cell monotonic timestamp


def reset() -> None:
    """Reset the spacing clock (called at the start of each run)."""
    with _lock:
        _last_call[0] = 0.0


def space_yf_call() -> None:
    """Enforce a minimum interval between yfinance calls across threads/modules."""
    with _lock:
        wait = _YF_MIN_INTERVAL - (_time.monotonic() - _last_call[0])
        if wait > 0:
            _time.sleep(wait)
        _last_call[0] = _time.monotonic()


def is_transient_error(exc: Exception) -> bool:
    """True when an exception looks like throttling/transient network trouble
    (worth retrying) rather than a definitive 'not found'."""
    msg = str(exc).lower()
    transient = ("429", "too many requests", "rate limit", "timed out",
                 "timeout", "connection", "temporarily", "503", "502")
    return any(t in msg for t in transient)


def retry(fn: Callable, *, what: str, log: Optional[logging.Logger] = None):
    """Run ``fn`` with bounded exponential backoff on transient errors.

    Returns ``fn()`` on success, or None if every attempt failed. A definitive
    (non-transient) error is not retried.
    """
    lg = log or logger
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — classified below
            if not is_transient_error(e) or attempt == _MAX_FETCH_ATTEMPTS:
                lg.debug("%s failed (attempt %d): %s", what, attempt, e)
                return None
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            lg.debug("%s throttled (attempt %d), backing off %.2fs", what, attempt, delay)
            _time.sleep(delay)
    return None


def fetch_yf(fn: Callable, *, what: str, log: Optional[logging.Logger] = None):
    """Spacing + retry wrapper: space the call, then run it under backoff."""
    def _spaced():
        space_yf_call()
        return fn()
    return retry(_spaced, what=what, log=log)
