"""Own independent row acquisitions; never select or adopt financial results.

The caller supplies audited read-only row callbacks. Each callback retains its
complete candidate order and listing/interval qualification. Only scheduling
between independent rows changes. Verdicts, dataframe writes and diagnostic
adoption stay with the caller in original row order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any


_NativeExecutor = ThreadPoolExecutor
_native_shutdown = ThreadPoolExecutor.shutdown


@dataclass
class ThirdSourceOutcome:
    """One callback's owned result, not a missing-price/identity certificate."""

    index: int
    result: Any = None
    error: BaseException | None = None
    diagnostics: list = field(default_factory=list)
    submitted: bool = False
    started: bool = False
    completed: bool = False


@dataclass(frozen=True)
class ThirdSourceBatch:
    """Returned only after accepted work drains; adopt in ``outcomes`` order.

    Ordinary callback errors belong to their row. The caller must propagate
    ``interruption`` after handling the canonical observed prefix, and must
    never interpret an unstarted/canceled outcome as an empty provider answer.
    """

    outcomes: tuple[ThirdSourceOutcome, ...]
    owner_error: BaseException | None = None

    @property
    def interruption(self) -> BaseException | None:
        if self.owner_error is not None and not isinstance(self.owner_error, Exception):
            return self.owner_error
        for outcome in self.outcomes:
            if outcome.error is not None and not isinstance(outcome.error, Exception):
                return outcome.error
        return self.owner_error


def _worker_limit(workers: int | None) -> int:
    from tarzan import config
    from tarzan.runtime.session import current_session
    from tarzan.runtime.transport import ProviderBudget

    configured = config.max_workers()
    if type(configured) is not int or configured < 1:
        return 0
    if workers is not None and (type(workers) is not int or workers < 1):
        return 0
    session = current_session()
    budget = getattr(session, "transport", None)
    if type(budget) is not ProviderBudget:
        return 0
    return min(configured, workers or configured, budget.coordinator.capacity)


def can_acquire_together(*, native: bool = False, workers: int | None = None) -> bool:
    """Fail closed unless an audited native caller has a matching LIVE owner.

    ``native=True`` is an explicit caller assertion, not callback introspection.
    Integration must keep custom/injected engines and ports on its original
    serial path. This helper does not mint transport/deadline authority.
    """
    if native is not True:
        return False
    from tarzan import runtime
    from tarzan.runtime.session import current_session

    session = current_session()
    return bool(
        session is not None
        and runtime.allows_live_transport()
        and runtime.context().attempt_id == session.context.attempt_id
        and _worker_limit(workers) >= 2
    )


def _observe(outcome: ThirdSourceOutcome, callback: Callable[[], Any]) -> ThirdSourceOutcome:
    from tarzan.runtime import data_quality as dq

    # A started callback proves acceptance even if a submit implementation
    # raised after enqueuing it. The executor still owns and drains that work.
    outcome.submitted = outcome.started = True
    try:
        # Override a copied parent buffer: each child owns a distinct list.
        dq.defer_into(outcome.diagnostics)
        outcome.result = callback()
    except BaseException as error:
        outcome.error = error
    finally:
        outcome.completed = True
    return outcome


def acquire_third_source_votes(
    callbacks: Iterable[Callable[[], Any]], *,
    parallel: bool = False, workers: int | None = None,
) -> ThirdSourceBatch:
    """Acquire once per row, with bounded submission and canonical outcomes.

    Default/injected, pinned, legacy and capacity-one use serial acquisition.
    Root should guard its complete two-pass integration with
    :func:`can_acquire_together` so serial fallback also preserves incremental
    row setup/adoption. Ordinary per-row errors are isolated. Source or owner
    cancellation stops new submissions, joins accepted callbacks and is
    exposed in ``batch.interruption``. Ordinary cleanup errors do not mask a
    canonical source cancellation; genuine owner BaseException cancellation,
    including during wait or shutdown, takes precedence over source errors.

    Callbacks must return owned facts and must not mutate financial outputs.
    This boundary introduces neither a fact/miss cache nor a new provider call,
    optional scope, deadline, client or native admission permit.
    """
    outcomes: list[ThirdSourceOutcome] = []
    owner_error: BaseException | None = None

    def owner_failure(error: BaseException) -> None:
        nonlocal owner_error
        if owner_error is None or (
            isinstance(owner_error, Exception) and not isinstance(error, Exception)
        ):
            owner_error = error

    try:
        iterator = iter(callbacks)
    except BaseException as error:
        return ThirdSourceBatch((), error)
    exhausted = False

    def next_callback():
        nonlocal exhausted
        try:
            callback = next(iterator)
            if not callable(callback):
                raise TypeError("third-source row acquisition must be callable")
        except StopIteration:
            exhausted = True
            return None
        except BaseException as error:
            owner_failure(error)
            exhausted = True
            return None
        outcome = ThirdSourceOutcome(len(outcomes))
        outcomes.append(outcome)
        return outcome, callback

    use_pool = can_acquire_together(native=parallel, workers=workers)
    if type(callbacks) in (list, tuple) and len(callbacks) < 2:
        use_pool = False
    if not use_pool:
        while not exhausted:
            item = next_callback()
            if item is None:
                break
            outcome, callback = item
            try:
                copy_context().run(_observe, outcome, callback)
            except BaseException as error:
                owner_failure(error)
                break
            if outcome.error is not None and not isinstance(outcome.error, Exception):
                break
        return ThirdSourceBatch(tuple(outcomes), owner_error)

    limit = _worker_limit(workers)
    pending = {}
    source_cancellation = False

    def source_interrupted() -> bool:
        # Inspect only the bounded active window, not every completed row on
        # each submission. The canonical error is selected from all outcomes
        # once, after drain, by ThirdSourceBatch.interruption.
        return source_cancellation or any(
            outcome.completed and outcome.error is not None
            and not isinstance(outcome.error, Exception)
            for outcome in pending.values()
        )

    def read_future(future, outcome: ThirdSourceOutcome) -> bool:
        """Return whether the accepted future is finished, retaining primaries."""
        try:
            observed = future.result()
        except CancelledError as error:
            if outcome.error is None:
                outcome.error = error
            # _observe captures ordinary callback CancelledError locally. A
            # canceled accepted Future instead means the callback did not run:
            # expose an owner fault, never a successful empty observation.
            owner_failure(error)
            return future.done()
        except BaseException as error:
            child_error = None
            if future.done() and not future.cancelled():
                try:
                    child_error = future.exception()
                except BaseException as observation_error:
                    owner_failure(observation_error)
            if child_error is error:
                if outcome.error is None or (
                    isinstance(outcome.error, Exception) and not isinstance(error, Exception)
                ):
                    outcome.error = error
                if not (outcome.started and outcome.completed):
                    # _observe captures callback errors and completes its
                    # record. An escaped failure before that boundary belongs
                    # to the executor/context owner, not to the price source.
                    owner_failure(error)
            else:
                # An interruption of the owner's wait is not a child failure.
                owner_failure(error)
            return future.done()
        if observed is not outcome or not (outcome.started and outcome.completed):
            error = RuntimeError("third-source future did not complete its owned observation")
            if outcome.error is None:
                outcome.error = error
            owner_failure(error)
        return True

    try:
        pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="tarzan-third-votes")
    except BaseException as error:
        return ThirdSourceBatch(tuple(outcomes), error)
    try:
        while not exhausted or pending:
            while (
                not exhausted and len(pending) < limit
                and owner_error is None and not source_interrupted()
            ):
                item = next_callback()
                if item is None:
                    break
                outcome, callback = item
                try:
                    future = pool.submit(copy_context().run, _observe, outcome, callback)
                except BaseException as error:
                    owner_failure(error)
                    break
                outcome.submitted = True
                pending[future] = outcome
            if owner_error is not None or source_interrupted() or not pending:
                break
            try:
                finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            except BaseException as error:
                owner_failure(error)
                break
            # Replenish after any completion, not only the first row's result.
            for future in sorted(finished, key=lambda value: pending[value].index):
                outcome = pending[future]
                if read_future(future, outcome):
                    if outcome.error is not None and not isinstance(outcome.error, Exception):
                        source_cancellation = True
                    del pending[future]
    except BaseException as error:
        owner_failure(error)
    finally:
        # Stop admission, not an accepted prefix: setup/iterator faults must
        # not cancel a queued earlier row whose result the caller must adopt.
        # A wait may itself be interrupted. Retry joining that accepted future
        # before releasing the owner; no worker may survive this return.
        for future, outcome in sorted(pending.items(), key=lambda item: item[1].index):
            while not read_future(future, outcome):
                pass
        try:
            pool.shutdown(wait=True)
        except BaseException as error:
            owner_failure(error)
            if isinstance(pool, _NativeExecutor):
                # A submit implementation can raise after enqueuing work and
                # hide its Future. Drain the actual executor too; interruption
                # of shutdown must not abandon such an accepted callback.
                while True:
                    try:
                        _native_shutdown(pool, wait=True)
                    except BaseException as cleanup_error:
                        owner_failure(cleanup_error)
                        if isinstance(cleanup_error, Exception):
                            break
                    else:
                        break
    return ThirdSourceBatch(tuple(outcomes), owner_error)
