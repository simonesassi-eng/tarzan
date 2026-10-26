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

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
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
    try:
        started = _time.monotonic()
    except BaseException:
        started = None
    locked = sleep_started = sleep_finished = None
    requested_sleep = 0.0
    completed = False
    try:
        with _lock:
            locked = _time.monotonic()
            wait = _YF_MIN_INTERVAL - (locked - _last_call[0])
            if wait > 0:
                from tarzan.runtime.transport import provider_wait
                requested_sleep = wait
                sleep_started = locked
                provider_wait(wait, "yahoo")
            _last_call[0] = _time.monotonic()
            if sleep_started is not None:
                sleep_finished = _last_call[0]
            completed = True
    finally:
        # This is outside the global pacing lock, holds no native permit and
        # never creates a budget. An optional observer cannot replace success,
        # the provider's deadline error or the caller's cancellation.
        try:
            from tarzan.runtime.workload import record_yahoo_spacing
            if started is None:
                raise ValueError("optional pacing observation clock unavailable")
            if locked is None or sleep_started is not None and sleep_finished is None:
                observed_end = _time.monotonic()
                if sleep_started is not None:
                    sleep_finished = observed_end
            else:
                observed_end = locked
            record_yahoo_spacing(
                lock_wait=max(
                    0.0, (locked if locked is not None else observed_end) - started,
                ),
                sleep_wait=(
                    max(0.0, sleep_finished - sleep_started)
                    if sleep_started is not None and sleep_finished is not None
                    else 0.0
                ),
                requested_sleep=requested_sleep,
                completed=completed,
            )
        except BaseException:
            try:
                from tarzan.runtime.workload import mark_yahoo_spacing_incomplete
                mark_yahoo_spacing_incomplete()
            except BaseException:
                pass


def is_transient_error(exc: Exception) -> bool:
    """True when an exception looks like throttling/transient network trouble
    (worth retrying) rather than a definitive 'not found'."""
    msg = str(exc).lower()
    transient = ("429", "too many requests", "rate limit", "timed out",
                 "timeout", "connection", "temporarily", "503", "502")
    return any(t in msg for t in transient)


def retry(fn: Callable, *, what: str, log: Optional[logging.Logger] = None,
          provider: str = "yahoo"):
    """Run ``fn`` with bounded exponential backoff on transient errors.

    Returns ``fn()`` on success, or None if every attempt failed. A definitive
    (non-transient) error is not retried.
    """
    lg = log or logger
    from tarzan.runtime.transport import current_budget
    budget = current_budget()
    if budget is not None:
        try:
            # HTTP retries belong to the native adapter, not this second layer.
            with budget.operation(provider):
                return fn()
        except Exception as error:
            lg.debug("%s unavailable under provider policy (%s)", what, type(error).__name__)
            return None
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


def ticker(symbol):
    """Use the actual yfinance session port, including cookie/crumb requests."""
    import yfinance as yf
    from tarzan.runtime.transport import current_budget
    budget = current_budget()
    return yf.Ticker(symbol) if budget is None else yf.Ticker(symbol, session=budget.yahoo_client())


def download(symbols, **kwargs):
    """Batch native Ticker.history calls with owned, bounded worker contexts.

    The SDK's batch coordinator has process-global result dictionaries and
    disables threads under DEBUG logging. Neither owns Tarzan's ContextVars.
    Owned runs therefore keep the SDK's history/HTTP ports, but coordinate
    them themselves; no shared SDK batch state is changed.
    """
    import yfinance as yf
    from tarzan.runtime.transport import current_budget
    budget = current_budget()
    kwargs["progress"] = False
    if budget is None:
        return yf.download(symbols, **kwargs)
    return _owned_download(symbols, **kwargs)


def _owned_download(symbols, **kwargs):
    import pandas as pd
    from yfinance.utils import empty_df
    from tarzan import config
    from tarzan.runtime.memo import memoized

    raw_symbols = symbols.replace(",", " ").split() if isinstance(symbols, str) else symbols
    ordered = sorted({str(symbol).upper() for symbol in raw_symbols})
    if not ordered:
        return pd.DataFrame()
    arguments = {
        "period": None, "interval": "1d", "start": None, "end": None,
        "prepost": False, "actions": False, "auto_adjust": True,
        "back_adjust": False, "repair": False, "rounding": False,
        "keepna": False, "timeout": 10,
    }
    group_by = kwargs.pop("group_by", "column")
    ignore_tz = kwargs.pop("ignore_tz", None)
    multi_level_index = kwargs.pop("multi_level_index", True)
    threads = kwargs.pop("threads", True)
    kwargs.pop("progress", None)
    kwargs.pop("session", None)  # a caller cannot replace the run-owned session
    if set(kwargs) - set(arguments):
        raise TypeError("unsupported Yahoo batch parameter")
    arguments.update(kwargs)
    if ignore_tz is None:
        ignore_tz = arguments["interval"][-1] not in {"m", "h"}
    maximum = config.max_workers()
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_workers must be a positive integer")
    if threads is True:
        workers = maximum
    elif threads is False or threads is None:
        workers = 1
    elif type(threads) is int and threads > 0:
        workers = min(maximum, threads)
    else:
        raise ValueError("Yahoo batch threads must be boolean or a positive integer")

    def fetch(symbol):
        def history():
            client = ticker(symbol)
            frame = client.history(**arguments)
            if arguments["interval"] == "1d":
                from tarzan.data import history_time

                frame = history_time.admit_history(
                    history_time.attach_native_session_evidence(frame, client), symbol,
                )
            return frame

        # Each history operation, including its cookie/crumb requests/retries,
        # owns its timeout. A large catalog is NOT one 20-second operation.
        frame = memoized(
            "yahoo.batch_history", (symbol, tuple(arguments.items())),
            lambda: retry(history, what="batch history"),
            legacy={},
        )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return empty_df()
        frame = frame.copy()
        if ignore_tz:
            frame.index = frame.index.tz_localize(None)
        return frame

    with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as pool:
        pending = {
            symbol: pool.submit(copy_context().run, fetch, symbol)
            for symbol in ordered
        }
        frames = {symbol: future.result() for symbol, future in pending.items()}
    # Preserve Tarzan's established batch shape and timezone contract.
    # Native SDK batches may now retain a shared listing timezone, but our
    # intraday consumers continue to receive the same instants in UTC.
    # ignore_tz=True retains venue-local labels, as before.
    frame = pd.concat(frames, axis=1, sort=True, names=["Ticker", "Price"])
    frame.index = pd.to_datetime(frame.index, utc=not ignore_tz)
    if group_by == "column":
        frame.columns = frame.columns.swaplevel(0, 1)
        frame.sort_index(level=0, axis=1, inplace=True)
    if not multi_level_index and len(ordered) == 1:
        frame = frame.droplevel(0 if group_by == "ticker" else 1, axis=1).rename_axis(None, axis=1)
    return frame


def fetch_download(symbols, *, what: str, log: Optional[logging.Logger] = None, **kwargs):
    """Pace a batch once; cap native I/O per child and by the global run budget."""
    from tarzan.runtime.transport import ProviderBudgetExceeded, current_budget
    budget = current_budget()
    if budget is None:
        return fetch_yf(lambda: download(symbols, **kwargs), what=what, log=log)
    try:
        try:
            space_yf_call()
        except ProviderBudgetExceeded:
            # Previously acquired, owned facts need no new I/O authority.
            # Missing children still enter retry's operation guard and cannot
            # start native requests after the deadline.
            pass
        return download(symbols, **kwargs)
    except Exception as error:
        (log or logger).debug("%s batch unavailable (%s)", what, type(error).__name__)
        return None


def yahoo_data():
    from yfinance.data import YfData
    from tarzan.runtime.transport import current_budget
    budget = current_budget()
    if budget is not None:
        return YfData(session=budget.yahoo_client())
    from curl_cffi.requests import Session
    return YfData(session=Session(impersonate="chrome"))


def fetch_yf(fn: Callable, *, what: str, log: Optional[logging.Logger] = None):
    """Spacing + retry wrapper: space the call, then run it under backoff."""
    def _spaced():
        space_yf_call()
        return fn()
    return retry(_spaced, what=what, log=log)
