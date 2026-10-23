"""Run-owned admission for native requests, not SDK coordinators or futures.

One permit encloses only a synchronous native I/O callback. Callers release it
before retry backoff, pacing or waiting for another task. No provider name,
request argument, response or exception text is retained in observations.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import time


class RequestCoordinatorClosed(RuntimeError):
    """The owner has stopped admission; active native calls may still finish."""


class RequestDeadlineExceeded(TimeoutError):
    """A remaining-time port returned a nonpositive admission allowance."""


@dataclass(eq=False)
class _Ticket:
    optional: bool


class RequestCoordinator:
    """Bound aggregate native concurrency and prioritize essential FIFO work.

    Each class is FIFO in queue-admission order, not executor submission order.
    Waiting essential requests precede optional requests. At capacity >= 2,
    one slot is reserved while essential demand is declared, queued or active.
    Optional-only work otherwise uses the full capacity. Declare independent
    essential work with ``essential_scope`` before submitting its tasks, and
    retain that logical scope until all those tasks have joined. The scope
    owns no native permit, so a coordinator may wait for its child futures.
    Capacity 1 has no unused reserve and both classes can make progress.

    ``remaining`` is called outside the condition lock, including once after
    admission immediately before the callback. Native deadline/cancellation
    exceptions pass through unchanged. A same-thread recursive lease fails
    closed instead of borrowing another permit or deadlocking at capacity 1.

    Closing wakes queued callers but does not kill arbitrary running Python.
    The owner joins/drains its tasks before reopening or closing native clients.
    """

    _WAIT_POLL_SECONDS = 0.05

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("native request capacity must be a positive Python integer")
        self._capacity = capacity
        self._condition = threading.Condition(threading.Lock())
        self._local = threading.local()
        self._essential: deque[_Ticket] = deque()
        self._optional: deque[_Ticket] = deque()
        self._active = 0
        self._active_optional = 0
        self._essential_demand = 0
        self._closed = False
        self._requested = 0
        self._acquired = 0
        self._released = 0
        self._rejected = 0
        self._peak = 0
        self._queue_wait_seconds = 0.0

    @property
    def capacity(self) -> int:
        """The immutable aggregate native cap for this coordinator's lifetime."""
        return self._capacity

    @staticmethod
    def _remaining_seconds(remaining: Callable[[], float]) -> float:
        value = remaining()  # Never called while holding the condition lock.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("remaining request time must be a finite number")
        try:
            seconds = float(value)
        except (OverflowError, ValueError):
            raise ValueError("remaining request time must be a finite number") from None
        if not math.isfinite(seconds):
            raise ValueError("remaining request time must be a finite number")
        if seconds <= 0:
            raise RequestDeadlineExceeded("native request admission deadline exhausted")
        return seconds

    def _eligible(self, ticket: _Ticket) -> bool:
        """Read queue/counters only while the caller owns the condition lock."""
        if self._active >= self.capacity:
            return False
        if not ticket.optional:
            return bool(self._essential) and self._essential[0] is ticket
        essential_expected = (
            self._essential_demand or self._essential
            or self._active > self._active_optional
        )
        optional_capacity = self.capacity - int(
            self.capacity >= 2 and bool(essential_expected)
        )
        return (
            not self._essential
            and bool(self._optional)
            and self._optional[0] is ticket
            and self._active_optional < optional_capacity
        )

    @contextmanager
    def essential_scope(self):
        """Declare logical essential demand without holding a native permit."""
        declared = False
        try:
            with self._condition:
                if self._closed:
                    raise RequestCoordinatorClosed("native request coordinator is closed")
                self._essential_demand += 1
                declared = True
                self._condition.notify_all()
            yield
        finally:
            if declared:
                with self._condition:
                    self._essential_demand -= 1
                    self._condition.notify_all()

    @contextmanager
    def lease(self, *, optional: bool, remaining: Callable[[], float]):
        """Own one native callback slot, preserving deadline exception identity."""
        if type(optional) is not bool:
            raise ValueError("request priority must be an explicit boolean")
        if not callable(remaining):
            raise TypeError("remaining request time must be callable")
        if getattr(self._local, "leasing", False):
            raise RuntimeError("recursive native request lease is not allowed")

        ticket = _Ticket(optional)
        queue = self._optional if optional else self._essential
        queued = acquired = registered = False
        elapsed = 0.0
        try:
            self._local.leasing = True
            started = time.monotonic()
            with self._condition:
                self._requested += 1
                registered = True
            while True:
                elapsed = max(0.0, time.monotonic() - started)
                seconds = self._remaining_seconds(remaining)
                with self._condition:
                    if self._closed:
                        raise RequestCoordinatorClosed("native request coordinator is closed")
                    if not queued:
                        queue.append(ticket)
                        queued = True
                    if self._eligible(ticket):
                        queue.popleft()
                        queued = False
                        self._active += 1
                        self._active_optional += int(optional)
                        self._acquired += 1
                        self._peak = max(self._peak, self._active)
                        acquired = True
                        self._queue_wait_seconds += elapsed
                        # Admit a second waiter if more capacity is now usable.
                        self._condition.notify_all()
                        break
                    self._condition.wait(timeout=min(seconds, self._WAIT_POLL_SECONDS))
            self._remaining_seconds(remaining)
            yield
        except BaseException:
            if registered and not acquired:
                with self._condition:
                    if queued:
                        queue.remove(ticket)
                    self._rejected += 1
                    self._queue_wait_seconds += elapsed
                    self._condition.notify_all()
            raise
        finally:
            try:
                if acquired:
                    with self._condition:
                        self._active -= 1
                        self._active_optional -= int(optional)
                        self._released += 1
                        self._condition.notify_all()
            finally:
                self._local.leasing = False

    def close(self) -> None:
        """Stop admission and wake waiters; running callbacks retain their slots."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reopen(self) -> None:
        """Reuse the same owner's counters only after all leases/waiters drain."""
        with self._condition:
            if (
                self._active or self._essential or self._optional
                or self._essential_demand
                or self._requested != self._released + self._rejected
            ):
                raise RuntimeError("native requests must drain before reopening")
            self._closed = False
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int | float | bool]:
        """Fresh descriptive counts only; no request identities or payloads."""
        with self._condition:
            return {
                "capacity": self.capacity,
                "active": self._active,
                "active_optional": self._active_optional,
                "essential_demand": self._essential_demand,
                "queued_essential": len(self._essential),
                "queued_optional": len(self._optional),
                "requested": self._requested,
                "acquired": self._acquired,
                "released": self._released,
                "rejected": self._rejected,
                "peak": self._peak,
                "queue_wait_seconds": self._queue_wait_seconds,
                "closed": self._closed,
            }
