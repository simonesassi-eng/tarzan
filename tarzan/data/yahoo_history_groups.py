"""Pace already-demanded native daily histories as bounded owned cohorts.

No history request or result is combined: every participant retains its own
exact arguments, original operation deadline, copied context, native parser
and session/provenance evidence. This is scheduling only, not source reuse.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Condition
import time


_active = ContextVar("tarzan_native_history_groups", default=None)
_GATHER_SECONDS = .003
_MAX_HISTORY_WORKERS = 2


class _HistoryGroups:
    def __init__(self, budget):
        from tarzan import config

        maximum = config.max_workers()
        if type(maximum) is not int or maximum < 1:
            raise ValueError("max_workers must be a positive integer")
        self.budget = budget
        self.maximum = min(maximum, _MAX_HISTORY_WORKERS)
        self.condition = Condition()
        self.pending = []
        self.driving = False
        self.closed = False

    def get(self, load, pace):
        # A cohort shares only scheduling. Unlike a combined metadata HTTP
        # request, its children must keep their individual operation contexts.
        authority = (
            self.budget.optional_work_token(), self.budget.protected_work_token(),
        )
        waiter = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("native history cohort is closed")
            self.pending.append((authority, waiter, copy_context(), load, pace))
            self.condition.notify_all()
        while True:
            owns_driver = False
            chosen = []
            try:
                with self.condition:
                    if waiter.done():
                        return waiter.result()
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
                        if item[0] == authority and len(chosen) < self.maximum:
                            chosen.append(item)
                        else:
                            remaining.append(item)
                    self.pending = remaining
            except BaseException as error:
                # Gathering owns no transport. An interrupted follower drops
                # only its queued demand; it must not unlock another driver.
                with self.condition:
                    aborted = {waiter} | {item[1] for item in chosen}
                    self.pending = [item for item in self.pending if item[1] not in aborted]
                    if owns_driver:
                        self.driving = False
                    self.condition.notify_all()
                for item in chosen:
                    if not item[1].done():
                        item[1].set_exception(error)
                raise
            try:
                # No native transport permit is held during gathering/pacing.
                # Use an actual chosen child's authority, not whichever
                # waiting thread happened to become this cohort's driver.
                # A shorter child's expired pacing authority cannot erase an
                # independently healthy peer; peers retain scalar pacing.
                from tarzan.runtime.transport import ProviderBudgetExceeded
                independent_pacing = False
                try:
                    chosen[0][2].run(chosen[0][4])
                except ProviderBudgetExceeded as error:
                    chosen[0][1].set_exception(error)
                    chosen = chosen[1:]
                    independent_pacing = True

                def execute(item):
                    if independent_pacing:
                        item[4]()
                    return item[3]()

                if not chosen:
                    continue
                if len(chosen) == 1:
                    item = chosen[0]
                    try:
                        item[1].set_result(item[2].run(execute, item))
                    except BaseException as error:
                        item[1].set_exception(error)
                else:
                    with ThreadPoolExecutor(
                        max_workers=len(chosen), thread_name_prefix="tarzan-native-history",
                    ) as pool:
                        futures = [
                            (item, pool.submit(item[2].run, execute, item)) for item in chosen
                        ]
                        for item, future in futures:
                            try:
                                item[1].set_result(future.result())
                            except BaseException as error:
                                item[1].set_exception(error)
            except BaseException as error:
                for item in chosen:
                    if not item[1].done():
                        item[1].set_exception(error)
            finally:
                with self.condition:
                    self.driving = False
                    self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            if self.driving or self.pending:
                raise RuntimeError("native history cohort work did not drain")


@contextmanager
def history_group_scope():
    from tarzan.data.yahoo_info import _native_history_contract, _native_owned_contract
    from tarzan.runtime.session import current_session

    if (
        _active.get() is not None
        or not _native_owned_contract() or not _native_history_contract()
    ):
        yield
        return
    groups = _HistoryGroups(current_session().transport)
    token = _active.set(groups)
    try:
        yield
    finally:
        _active.reset(token)
        groups.close()


def demanded_history(symbol, arguments, load, pace):
    """Decline unknown ports and requests; callers keep the scalar contract."""
    from tarzan.data import _yf_net, yahoo_info
    from tarzan.data.yahoo_quote_batch import _native_data, _ordinary, _owned_response_port
    from tarzan.runtime.session import current_session

    groups = _active.get()
    session = current_session()
    if (
        type(groups) is not _HistoryGroups or session is None
        or session.transport is not groups.budget
        or not yahoo_info._native_owned_contract() or not _ordinary(symbol)
        or not yahoo_info._native_history_contract()
        or type(arguments) is not dict
        or arguments.get("interval") != "1d" or arguments.get("auto_adjust") is not False
        or set(arguments) not in (
            {"interval", "auto_adjust", "period"}, {"interval", "auto_adjust", "start"},
        )
    ):
        return False, None
    data = _yf_net.yahoo_data()
    if not _native_data(data) or not _owned_response_port(data):
        return False, None
    return True, groups.get(load, pace)
