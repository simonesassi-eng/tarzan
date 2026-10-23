"""Run-owned provider deadlines, native timeouts and bounded HTTP retries.

Deadlines are checked before/after calls and pacing. Native client timeouts
bound transport I/O; this is not a mechanism for killing arbitrary Python code.
Provider URLs, headers, bodies and instrument identifiers never enter telemetry.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import math
import threading
import time


_PROVIDERS = frozenset({"yahoo", "openfigi", "borsa", "justetf", "manual", "gemini"})
_operation = ContextVar("tarzan_provider_operation", default=None)
_optional_work = ContextVar("tarzan_optional_provider_work", default=None)
logger = logging.getLogger(__name__)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ProviderTransportPolicy:
    provider: str
    timeout_seconds: float
    retry_budget: int

    def __post_init__(self):
        if self.provider not in _PROVIDERS:
            raise ValueError("unknown provider transport")
        _positive(self.timeout_seconds, "timeout_seconds")
        if type(self.retry_budget) is not int or not 0 <= self.retry_budget <= 5:
            raise ValueError("retry_budget must be an integer in [0, 5]")


@dataclass(frozen=True)
class ProviderTransportSettings:
    run_budget_seconds: float
    providers: tuple[ProviderTransportPolicy, ...]
    optional_budget_seconds: float = 120
    publication_reserve_seconds: float = 90

    def __post_init__(self):
        _positive(self.run_budget_seconds, "run_budget_seconds")
        _positive(self.optional_budget_seconds, "optional_budget_seconds")
        _positive(self.publication_reserve_seconds, "publication_reserve_seconds")
        object.__setattr__(self, "providers", tuple(self.providers))
        if len(self.providers) != len(_PROVIDERS) or {p.provider for p in self.providers} != _PROVIDERS:
            raise ValueError("each provider transport must be declared exactly once")

    @classmethod
    def configured(cls):
        from tarzan import config
        raw = config.get("provider_transport")
        if not isinstance(raw, dict) or set(raw) != {
            "run_budget_seconds", "optional_budget_seconds", "publication_reserve_seconds", "providers",
        }:
            raise ValueError("provider_transport must declare run, optional and reserve budgets and providers")
        if not isinstance(raw["providers"], dict):
            raise ValueError("provider_transport providers must be a mapping")
        return cls(
            raw["run_budget_seconds"],
            tuple(ProviderTransportPolicy(provider=name, **values)
                  for name, values in sorted(raw["providers"].items())),
            raw["optional_budget_seconds"],
            raw["publication_reserve_seconds"],
        )


class ProviderBudgetExceeded(RuntimeError):
    """A deadline is exhausted; further retries cannot restore authority."""

class OptionalProviderBudgetExceeded(ProviderBudgetExceeded):
    """Optional discovery stopped; the essential run still owns its reserve."""



class ProviderRequestFailure(RuntimeError):
    """Bounded native request failure, without leaking its private request."""


def _transient(error):
    if isinstance(error, ProviderBudgetExceeded):
        return False
    from tarzan.data._yf_net import is_transient_error
    status = getattr(error, "code", None)
    return (
        isinstance(error, (TimeoutError, ConnectionError))
        or type(error).__name__ in {
            "Timeout", "ConnectTimeout", "ReadTimeout", "ConnectionError",
            "ChunkedEncodingError", "RemoteDisconnected",
        }
        or status in (429, 500, 502, 503, 504)
        or is_transient_error(error)
    )


@contextmanager
def _yahoo_request_observation(owner, url):
    """Optional observation cannot enter, suppress or replace a native error."""
    boundary = None
    observation = None
    try:
        from tarzan.runtime.request_classes import yahoo_request_boundary
        boundary = yahoo_request_boundary()
        boundary.__enter__()
    except BaseException:
        pass
    try:
        from tarzan.runtime.request_classes import yahoo_request_scope
        observation = yahoo_request_scope(owner, url)
        observation.__enter__()
    except BaseException:
        # Still attempt cleanup when an injected observer failed during enter.
        # Its optional failure must not prevent the real request from running.
        pass
    url = None
    try:
        yield
    finally:
        if observation is not None:
            try:
                # The observer needs no primary exception or response details.
                # Ignore suppression requests and errors, including cancellation.
                observation.__exit__(None, None, None)
            except BaseException:
                pass
        if boundary is not None:
            try:
                boundary.__exit__(None, None, None)
            except BaseException:
                pass


class ProviderBudget:
    def __init__(
        self, settings, ledger=None, *, clock=time.monotonic, sleep=time.sleep,
        capacity=None,
    ):
        from tarzan import config
        from tarzan.runtime.acquisition import RequestCoordinator

        self.settings = settings
        self.policies = {policy.provider: policy for policy in settings.providers}
        self.ledger = ledger
        self.clock = clock
        self.sleep = sleep
        self.deadline = clock() + settings.run_budget_seconds
        self._lock = threading.Lock()
        self._reported = set()
        self._yahoo_client = None
        self.coordinator = RequestCoordinator(
            config.max_workers() if capacity is None else capacity,
        )

    def optional_work_token(self):
        """Pending-work authority, not a key for successfully observed facts.

        Distinct optional deadlines must not lead essential callers or another
        optional phase through a shorter-lived single-flight operation.
        """
        stage = _optional_work.get()
        return stage if stage is not None and stage[0] is self else None

    @contextmanager
    def essential_work(self):
        """Declare independent essential tasks before submit, until joined."""
        with self.coordinator.essential_scope():
            yield

    def _limiting_deadline(self, provider):
        deadline = self.deadline
        optional = False
        active = _operation.get()
        if active is not None and active[0] is self and active[1] == provider:
            deadline = min(deadline, active[2])
        stage = _optional_work.get()
        if stage is not None and stage[0] is self and stage[2] <= deadline:
            deadline, optional = stage[2], True
        return deadline, optional

    def _exhausted(self, provider):
        _, optional = self._limiting_deadline(provider)
        code = "OPTIONAL_PROVIDER_BUDGET_EXHAUSTED" if optional else "PROVIDER_BUDGET_EXHAUSTED"
        self._failure(provider, code)
        error = OptionalProviderBudgetExceeded if optional else ProviderBudgetExceeded
        raise error(f"{provider} provider budget exhausted")

    def _remaining(self, provider):
        deadline, _ = self._limiting_deadline(provider)
        remaining = deadline - self.clock()
        if remaining <= 0:
            self._exhausted(provider)
        return remaining

    def _failure(self, provider, code):
        if type(provider) is not str or provider not in _PROVIDERS:
            raise ValueError("unknown provider transport")
        with self._lock:
            key = (provider, code)
            if key in self._reported:
                return
            self._reported.add(key)
        logger.warning("Provider evidence is bounded by its configured policy.", extra={"public_code": code})
        if self.ledger is None:
            return
        self.ledger.open_failure(
            stage="provider_transport", stable_code=code, severity="WARNING",
            error={"provider": provider, "code": code},
            affected_outputs=["market_data"], publication_impact="DEGRADE",
            analytical_impact="provider evidence unavailable; only explicit data-quality fallbacks remain",
            context={"provider": provider},
            identity_scope=f"provider:{provider}",
        )

    @contextmanager
    def optional_work(self, name):
        """Cap optional work without spending the final publication reserve.

        The scope is copied into provider workers with the other run context.
        It limits native timeouts, retries and pacing, not arbitrary Python CPU.
        """
        deadline = min(
            self.clock() + self.settings.optional_budget_seconds,
            self.deadline - self.settings.publication_reserve_seconds,
        )
        parent = _optional_work.get()
        if parent is not None and parent[0] is self:
            deadline = min(deadline, parent[2])
        token = _optional_work.set((self, name, deadline))
        try:
            yield
        finally:
            _optional_work.reset(token)

    def optional_request_allowed(self, provider):
        """Check before starting discovery, including work with cached leaves."""
        try:
            self._remaining(provider)
        except ProviderBudgetExceeded:
            return False
        return True

    @contextmanager
    def operation(self, provider):
        policy = self.policies[provider]
        self._remaining(provider)
        active = _operation.get()
        if active is not None and active[0] is self and active[1] == provider:
            yield
            self._remaining(provider)
            return
        token = _operation.set((self, provider, min(self.deadline, self.clock() + policy.timeout_seconds)))
        try:
            yield
            self._remaining(provider)
        finally:
            _operation.reset(token)

    def wait(self, seconds, provider):
        remaining = self._remaining(provider)
        if seconds >= remaining:
            self._exhausted(provider)
        if seconds > 0:
            self.sleep(seconds)
        self._remaining(provider)

    def call(self, provider, callback, *, requested_timeout=None):
        """Retry a native request, capped by one operation and the run deadline."""
        from tarzan.runtime.ledger import LedgerEntryType

        policy = self.policies[provider]
        with self.operation(provider):
            for ordinal in range(1, policy.retry_budget + 2):
                retryable = False
                # A permit encloses only native I/O. SDK coordination, cache
                # leaders/followers, pacing and retry backoff hold no permit.
                with self.coordinator.lease(
                    optional=self.optional_work_token() is not None,
                    remaining=lambda: self._remaining(provider),
                ):
                    # Queue time consumes the original operation/run budget.
                    # Do not pass a timeout calculated before admission.
                    cap = min(policy.timeout_seconds, self._remaining(provider))
                    if requested_timeout is not None:
                        requested = requested_timeout
                        if isinstance(requested, (tuple, list)):
                            requested = min(requested)
                        if isinstance(requested, (int, float)) and math.isfinite(requested) and requested > 0:
                            cap = min(cap, requested)
                    started = self.clock()
                    outcome = "CANCELLED"
                    value = None
                    try:
                        value = callback(cap)
                        self._remaining(provider)
                        status = getattr(value, "status_code", None)
                        retryable = status in (429, 500, 502, 503, 504)
                        outcome = f"HTTP_{status}" if status is not None else "SUCCEEDED"
                        if retryable:
                            close = getattr(value, "close", None)
                            if callable(close):
                                close()
                        else:
                            return value
                    except ProviderBudgetExceeded:
                        outcome = "BUDGET_EXHAUSTED"
                        # The native response never reaches its usual caller's
                        # finally block when the post-request deadline fails.
                        close = getattr(value, "close", None)
                        if callable(close):
                            try:
                                close()
                            except Exception:
                                logger.warning(
                                    "Provider response cleanup failed; deadline retained.",
                                    extra={"public_code": "PROVIDER_RESPONSE_CLEANUP_FAILED"},
                                )
                        raise
                    except Exception as error:
                        retryable = _transient(error)
                        outcome = type(error).__name__
                        if not retryable:
                            raise
                    finally:
                        if self.ledger is not None:
                            payload = {
                                "source": provider, "operation": "native_http",
                                "ordinal": ordinal, "outcome": outcome,
                                "timeout_seconds": cap, "retry_budget": policy.retry_budget,
                                "latency_ms": max(0.0, self.clock() - started) * 1000.0,
                            }
                            try:
                                from tarzan.runtime.request_classes import REQUEST_CLASSES, request_class_for
                                category = request_class_for(self, provider)
                                if type(category) is str and category in REQUEST_CLASSES:
                                    payload["native_request_class"] = category
                            except BaseException:
                                # Optional observations must not replace the
                                # original native outcome or cancellation.
                                pass
                            self.ledger.append(LedgerEntryType.PROVIDER_ATTEMPT, payload)
                if ordinal <= policy.retry_budget:
                    self.wait(0.25 * 2 ** (ordinal - 1), provider)
            self._failure(provider, "PROVIDER_RETRY_BUDGET_EXHAUSTED")
            # The SDK must not retry the same already-exhausted network exception.
            raise ProviderRequestFailure(f"{provider} native request retry budget exhausted") from None

    def yahoo_client(self):
        with self._lock:
            if self._yahoo_client is None:
                from curl_cffi.requests import Session
                owner = self

                class BudgetedSession(Session):
                    def request(self, method, url, **kwargs):
                        requested = kwargs.pop("timeout", None)
                        native_request = super().request
                        with _yahoo_request_observation(owner, url):
                            try:
                                return owner.call(
                                    "yahoo",
                                    lambda timeout: native_request(method, url, timeout=timeout, **kwargs),
                                    requested_timeout=requested,
                                )
                            except ProviderBudgetExceeded:
                                raise
                            except Exception:
                                # Prevent the SDK's own transient-error retry loop
                                # from multiplying already-budgeted native attempts.
                                raise ProviderRequestFailure("yahoo native request unavailable") from None

                self._yahoo_client = BudgetedSession(impersonate="chrome")
            return self._yahoo_client

    def close(self):
        # Close only the native client. A completed owner can still create a
        # fresh client for its output under this SAME deadline and coordinator.
        with self._lock:
            client, self._yahoo_client = self._yahoo_client, None
        if client is not None:
            client.close()


def current_budget():
    from tarzan.runtime.session import current_session
    session = current_session()
    if session is None:
        return None
    with session.providers.lock:
        if session.transport is None:
            raw = session.config_snapshot.get("provider_transport")
            settings = (
                ProviderTransportSettings(
                    raw["run_budget_seconds"],
                    tuple(ProviderTransportPolicy(**policy) for policy in raw["providers"]),
                    raw.get("optional_budget_seconds", 120),
                    raw.get("publication_reserve_seconds", 90),
                )
                if raw else ProviderTransportSettings.configured()
            )
            session.transport = ProviderBudget(settings, session.ledger)
    return session.transport


def provider_call(provider, callback, *, timeout):
    budget = current_budget()
    return callback(timeout) if budget is None else budget.call(
        provider, callback, requested_timeout=timeout,
    )


def provider_wait(seconds, provider):
    budget = current_budget()
    if budget is None:
        if seconds > 0:
            time.sleep(seconds)
    else:
        budget.wait(seconds, provider)
