"""Separately acquired justETF quote pairs and EUR histories for corroboration.

The primary ETF tape is not overwritten by a lone justETF answer. A quote may
corroborate a disputed daily baseline under current-session arbitration; a
printed long-window return may be corrected only under the existing settlement
rule with a qualified, comparable third-venue witness. Yahoo venue siblings
are not additional independent providers merely because their suffix differs.

Missing, malformed or nonfinite source facts stay unavailable, never verified.
Ordinary provider failures degrade corroboration; owner/source cancellation
still propagates after accepted work drains. Final publication remains the
run's typed availability/semantic policy, not this fetcher's decision.

The undocumented endpoints are used with run-owned memoization, bounded pools,
native timeouts and the existing retry/admission policy. No alias, venue or
history search is pruned here to obtain a faster answer.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import copy_context
from pathlib import Path
from typing import Optional

import pandas as pd
from tarzan.data.referee import finite_number, observation_date
from tarzan.runtime.memo import clear_provider_stores, memoized, provider_memo, provider_store

logger = logging.getLogger(__name__)

_BASE = "https://www.justetf.com/api/etfs/{isin}"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
                   "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"),
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT_S = 10
_WORKERS = 6

_series_memo: dict[str, Optional[pd.Series]] = {}
_quote_memo: dict[str, Optional[dict]] = {}
_OWNED_PREFETCH_KEY = "owned_current_referee_prefetch"


def reset_memo() -> None:
    with provider_memo().lock:
        provider_store("justetf.series", _series_memo).clear()
        provider_store("justetf.quote", _quote_memo).clear()
        clear_provider_stores("justetf.series", "justetf.quote")


def _get_json(url: str) -> Optional[dict]:
    """The decoded body, or None for anything that is not a JSON answer.

    An ISIN justETF does not cover comes back as its HTML page with HTTP 200, so the
    content type is checked rather than the status.
    """
    import requests
    from tarzan.runtime.transport import provider_call

    try:
        r = provider_call(
            "justetf", lambda timeout: requests.get(url, headers=_HEADERS, timeout=timeout),
            timeout=_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 — a referee must never break a run
        logger.info("justETF unreachable (%s): %s", type(e).__name__, url.split("?")[0])
        return None
    try:
        if not r.ok or "json" not in str(r.headers.get("content-type") or "").lower():
            return None
        body = r.json()
        return body if isinstance(body, dict) else None
    except ValueError:
        return None
    finally:
        # requests.get creates a short-lived session. Release a received response on
        # every exit, including an HTML fallback or malformed JSON answer.
        with suppress(Exception):
            r.close()


def _fetch_series(isin: str) -> Optional[pd.Series]:
    """Daily EUR closes, dividends reinvested — the same total-return basis as the
    engine's ``auto_adjust=True`` tape — indexed by session date."""
    body = _get_json(
        _BASE.format(isin=isin) + "/performance-chart?locale=en&currency=EUR"
        "&valuesType=MARKET_VALUE&reduceData=false&includeDividends=true")
    if not isinstance(body, dict):
        return None
    points = body.get("series")
    if not isinstance(points, (list, tuple)):
        return None
    from tarzan import runtime

    today = runtime.today()
    data = {}
    for p in points:
        try:
            day = observation_date(p["date"])
            value = finite_number(p["value"]["raw"], positive=True)
        except (KeyError, TypeError, ValueError):
            continue
        if day is not None and day <= today and value is not None:
            data[pd.Timestamp(day)] = value
    if not data:
        return None
    return pd.Series(data, dtype=float).sort_index()


def _fetch_quote(isin: str) -> Optional[dict]:
    """A finite, dated EUR market pair, or None; freshness is never invented."""
    body = _get_json(_BASE.format(isin=isin) + f"/quote?locale=en&currency=EUR&isin={isin}")
    if not isinstance(body, dict) or not _response_matches_quote_request(body, isin):
        return None
    try:
        price = finite_number(body["latestQuote"]["raw"], positive=True)
        previous = finite_number(body["previousQuote"]["raw"], positive=True)
        day = observation_date(body["latestQuoteDate"])
        previous_day = observation_date(body["previousQuoteDate"])
    except (KeyError, TypeError, ValueError):
        return None
    from tarzan import runtime

    if (
        price is None or previous is None or day is None or previous_day is None
        or previous_day >= day or day > runtime.today()
    ):
        return None
    return {
        "price": price, "prev_close": previous, "date": day, "prev_date": previous_day,
        "venue": body.get("quoteTradingVenue"), "currency": "EUR",
    }


def _response_matches_quote_request(body: dict, isin: str) -> bool:
    """Reject an explicit identity/response-currency contradiction.

    The observed quote schema supplies neither field; ISIN and EUR are the request's
    contract. Check exact optional scalar declarations if a response supplies them,
    never the fund's base currency or a venue's trading currency: those can legitimately
    differ from the requested EUR quote.
    """
    for field, expected in (("isin", str(isin).strip().upper()), ("currency", "EUR")):
        supplied = body.get(field)
        if supplied is None:
            continue
        if not isinstance(supplied, str) or supplied.strip().upper() not in ("", expected):
            return False
    return True


def prefetch(isins) -> None:
    """Fetch missing series/quotes and join accepted work before propagating faults."""
    from tarzan import runtime
    from tarzan.runtime.session import current_session

    finish_owned_prefetch(current_session())

    wanted = [i for i in dict.fromkeys(isins) if i]
    if not wanted or not runtime.allows_live_transport():
        return
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = [
            pool.submit(copy_context().run, fetch, isin)
            for fetch in (series, quote) for isin in wanted
        ]
        for future in futures:
            future.result()


class _OwnedPrefetch:
    """All early series and quotes are submitted and joined by one run owner."""

    def __init__(self, wanted, budget):
        self.budget = budget
        self.protection = budget.coordinator.protect(len(wanted), activate=False)
        self.pool = None
        self.series_futures = []
        self.quote_futures = []
        self.complete = False
        self.failure = None
        self.failure_reported = False
        try:
            self.pool = ThreadPoolExecutor(
                max_workers=_WORKERS, thread_name_prefix="tarzan-current-referee",
            )
            # Preserve the original submission epoch/order. Histories are not
            # first acquired in the later long-window consumer.
            for isin in wanted:
                future = self.pool.submit(copy_context().run, self._series, isin)
                self.series_futures.append(future)
                future.add_done_callback(lambda done: self.protection.finish())
            for isin in wanted:
                self.quote_futures.append(
                    self.pool.submit(copy_context().run, quote, isin),
                )
        except BaseException as primary:
            self.protection.finish(len(wanted) - len(self.series_futures))
            self.finish(primary=primary)

    def _series(self, isin):
        with self.budget.protected_requests(self.protection):
            return series(isin)

    @staticmethod
    def _worker_error(future, error):
        return (
            future.done() and not future.cancelled()
            and future.exception() is error
        )

    def quotes_ready(self):
        for future in self.quote_futures:
            try:
                future.result()
            except BaseException as error:
                if self._worker_error(future, error) or isinstance(error, Exception):
                    # The canonical original series-before-quotes interruption
                    # is established after all accepted callbacks drain.
                    self.finish()
                    raise
                self.finish(primary=error)
        # All quotes retain the original early essential FIFO and its usable
        # native capacity. Only subsequent phases must leave room for the
        # already-purchased histories that are still outstanding.
        self.protection.activate()

    def finish(self, primary=None):
        if not self.complete:
            try:
                for future in self.series_futures + self.quote_futures:
                    try:
                        future.result()
                    except BaseException as error:
                        if self._worker_error(future, error) or isinstance(error, Exception):
                            if (
                                self.failure is None
                                or isinstance(self.failure, Exception)
                                and not isinstance(error, Exception)
                            ):
                                self.failure = error
                        else:
                            # An owner-wait interruption precedes child errors.
                            if primary is None:
                                primary = error
            finally:
                if self.pool is not None:
                    self.pool.shutdown(wait=True)
                self.protection.close()
                self.complete = True
        if primary is not None:
            raise primary
        if self.failure is not None and not isinstance(self.failure, Exception):
            raise self.failure
        # Ordinary long-history absence/failure has no authority over a valid
        # independent current quote. It remains unavailable to its consumer.
        if self.failure is not None and not self.failure_reported:
            self.failure_reported = True
            try:
                logger.info(
                    "Early referee callback unavailable (%s); independent source evidence retained.",
                    type(self.failure).__name__,
                )
            except Exception:
                # A secondary log handler is not financial/source authority.
                pass


def finish_owned_prefetch(session):
    """Join native early work before long-window adoption or run finalization."""
    if session is None:
        return
    tickets = session.memo.get(_OWNED_PREFETCH_KEY)
    if type(tickets) is list:
        for ticket in tickets:
            ticket.finish()


@contextmanager
def owned_prefetch_scope():
    """The scope cannot return, fail or close its transport with live callbacks."""
    from tarzan.runtime.session import current_session, RunSession

    session = current_session()
    if type(session) is not RunSession or _OWNED_PREFETCH_KEY in session.memo:
        yield
        return
    session.memo[_OWNED_PREFETCH_KEY] = []
    try:
        try:
            yield
        except BaseException:
            try:
                finish_owned_prefetch(session)
            except BaseException:
                # The scope body's primary, not an ambient handled exception,
                # retains authority after all accepted callbacks have joined.
                pass
            raise
        else:
            finish_owned_prefetch(session)
    finally:
        session.memo.pop(_OWNED_PREFETCH_KEY, None)


def _owned_prefetch_available(session):
    from tarzan import runtime
    from tarzan.runtime.acquisition import RequestCoordinator
    from tarzan.runtime.session import RunSession
    from tarzan.runtime.transport import ProviderBudget

    ports = (
        prefetch, series, quote, _fetch_series, _fetch_quote, _get_json,
    )
    return (
        type(session) is RunSession
        and runtime.allows_live_transport()
        and type(session.memo.get(_OWNED_PREFETCH_KEY)) is list
        and type(session.transport) is ProviderBudget
        and type(session.transport.coordinator) is RequestCoordinator
        and session.transport.optional_work_token() is None
        and session.transport.protected_work_token() is None
        and type(_NATIVE_OWNED_PREFETCH_PORTS) is tuple
        and ports == _NATIVE_OWNED_PREFETCH_PORTS
        and all(Path(port.__code__.co_filename).resolve() == Path(__file__).resolve() for port in ports)
    )


def prefetch_current_quotes(isins):
    """Release the quote consumer while the same owner's early histories finish."""
    from tarzan.runtime.session import current_session

    wanted = [isin for isin in dict.fromkeys(isins) if isin]
    session = current_session()
    if not wanted or not _owned_prefetch_available(session):
        return prefetch(wanted)
    ticket = _OwnedPrefetch(wanted, session.transport)
    session.memo[_OWNED_PREFETCH_KEY].append(ticket)
    ticket.quotes_ready()


def series(isin: str) -> Optional[pd.Series]:
    from tarzan import runtime

    if not isin or not runtime.allows_live_transport():
        return None
    return memoized(
        "justetf.series", isin, lambda: _fetch_series(isin), legacy=_series_memo,
    )


def quote(isin: str) -> Optional[dict]:
    from tarzan import runtime

    if not isin or not runtime.allows_live_transport():
        return None
    return memoized(
        "justetf.quote", isin, lambda: _fetch_quote(isin), legacy=_quote_memo,
    )


# Audited consumer scheduling must not reorder injected/custom transport ports.
_NATIVE_REFEREE_PORTS = (prefetch, series, quote)
_NATIVE_OWNED_PREFETCH_PORTS = (
    prefetch, series, quote, _fetch_series, _fetch_quote, _get_json,
)
