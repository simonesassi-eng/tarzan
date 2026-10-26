"""Coalesce metadata that enrichment workers have actually demanded.

There is no ticker prefetch or venue shortcut. An owned pool gathers only
already-requested ordinary listing keys for at most three milliseconds;
each caller receives the complete native metadata for that exact key.
Different optional, protected and enclosing-operation authorities never join.
"""

from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Condition
import time


_active = ContextVar("tarzan_native_info_groups", default=None)
_executing = ContextVar("tarzan_native_info_group_execution", default=None)
_GATHER_SECONDS = .003


class _DemandGroups:
    def __init__(self, budget):
        self.budget = budget
        self.condition = Condition()
        self.pending = []
        self.driving = False
        self.closed = False

    def _authority(self):
        from tarzan.runtime.transport import _operation

        operation = _operation.get()
        return (
            self.budget.optional_work_token(), self.budget.protected_work_token(),
            operation if operation is not None and operation[0] is self.budget else None,
        )

    def get(self, symbol):
        return self.many([symbol])[symbol]

    def many(self, symbols):
        authority = self._authority()
        ordered = list(dict.fromkeys(symbols))
        if not ordered:
            return {}
        futures = {symbol: Future() for symbol in ordered}
        owned_waiters = set(futures.values())
        context = copy_context()
        with self.condition:
            if self.closed:
                raise RuntimeError("native metadata group scope is closed")
            self.pending.extend(
                (symbol, authority, futures[symbol], context) for symbol in ordered
            )
            self.condition.notify_all()
        while True:
            owns_driver = False
            chosen = []
            try:
                with self.condition:
                    if all(future.done() for future in futures.values()):
                        return {symbol: futures[symbol].result() for symbol in ordered}
                    if self.driving:
                        self.condition.wait(.05)
                        continue
                    self.driving = True
                    owns_driver = True
                    gather_until = time.monotonic() + _GATHER_SECONDS
                    while True:
                        remaining = gather_until - time.monotonic()
                        if remaining <= 0:
                            break
                        self.condition.wait(remaining)
                    remaining = []
                    for item in self.pending:
                        if item[1] == authority and len(chosen) < 64:
                            chosen.append(item)
                        else:
                            remaining.append(item)
                    self.pending = remaining
            except BaseException as error:
                with self.condition:
                    aborted = owned_waiters | {item[2] for item in chosen}
                    self.pending = [item for item in self.pending if item[2] not in aborted]
                    if owns_driver:
                        self.driving = False
                    self.condition.notify_all()
                for _symbol, _authority, waiter, _context in chosen:
                    if not waiter.done():
                        waiter.set_exception(error)
                raise
            try:
                from tarzan.data.yahoo_quote_batch import info_many

                def load():
                    token = _executing.set(self)
                    try:
                        return info_many([item[0] for item in chosen])
                    finally:
                        _executing.reset(token)

                values = context.run(load)
                if not isinstance(values, dict) or any(item[0] not in values for item in chosen):
                    raise ValueError("native metadata group omitted a demanded key")
            except BaseException as error:
                for _symbol, _authority, waiter, _context in chosen:
                    waiter.set_exception(error)
            else:
                for key, _authority, waiter, _context in chosen:
                    waiter.set_result(values[key])
            finally:
                with self.condition:
                    self.driving = False
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            if self.driving or self.pending:
                raise RuntimeError("native metadata group work did not drain")


@contextmanager
def metadata_group_scope():
    from tarzan.data.yahoo_info import _native_owned_contract
    from tarzan.runtime.session import current_session

    if _active.get() is not None or not _native_owned_contract():
        yield
        return
    groups = _DemandGroups(current_session().transport)
    token = _active.set(groups)
    try:
        yield
    finally:
        _active.reset(token)
        groups.close()


def demanded_info(symbol):
    """Return handled/value; declined callers retain the original full path."""
    from tarzan.data.yahoo_quote_batch import _ordinary
    handled, groups = _eligible_groups()
    if not handled or not _ordinary(symbol):
        return False, None
    return True, groups.get(symbol)


def demanded_many(symbols):
    """Handle an already declared full discovery plan, not speculative venues."""
    from tarzan.data.yahoo_quote_batch import _native_data, _ordinary, _owned_response_port
    from tarzan.data import _yf_net

    ordered = list(dict.fromkeys(symbols))
    if not ordered:
        return True, {}
    handled, groups = _eligible_groups()
    if not handled:
        return False, None
    data = _yf_net.yahoo_data()
    if not _native_data(data) or not _owned_response_port(data):
        return False, None
    if not all(_ordinary(symbol) for symbol in ordered):
        return False, None
    return True, groups.many(ordered)


def _eligible_groups():
    from tarzan.data.yahoo_info import _native_owned_contract
    from tarzan.runtime.session import current_session

    groups = _active.get()
    session = current_session()
    if (
        type(groups) is not _DemandGroups or session is None
        or session.transport is not groups.budget
        or _executing.get() is groups or not _native_owned_contract()
    ):
        return False, None
    return True, groups
