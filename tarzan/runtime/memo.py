"""Run-owned provider caches with per-key, exception-safe single-flight."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import Any, Callable, Hashable, Mapping


class SingleFlight:
    """Coalesce concurrent work, without serializing unrelated keys.

    Only pending work is retained here. Completed values (including misses)
    live in the caller's cache, so clearing that cache really permits a refresh.
    A leader's exception, including cancellation, reaches all its waiters.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._pending: dict[Hashable, tuple[Future, int]] = {}

    def many(self, keys, loader: Callable[[list], Mapping]) -> dict:
        ordered = list(dict.fromkeys(keys))
        futures = {}
        leaders = []
        owner = get_ident()
        with self._lock:
            # Check recursion before reserving any keys, avoiding orphan work.
            if any(
                key in self._pending and self._pending[key][1] == owner
                for key in ordered
            ):
                raise RuntimeError("recursive single-flight request for the same key")
            for key in ordered:
                pending = self._pending.get(key)
                if pending is None:
                    pending = (Future(), owner)
                    self._pending[key] = pending
                    leaders.append(key)
                futures[key] = pending[0]
        if leaders:
            try:
                values = loader(leaders)
                if not isinstance(values, Mapping) or any(k not in values for k in leaders):
                    raise ValueError("single-flight loader omitted a reserved key")
            except BaseException as error:
                for key in leaders:
                    futures[key].set_exception(error)
            else:
                for key in leaders:
                    futures[key].set_result(values[key])
            finally:
                with self._lock:
                    for key in leaders:
                        self._pending.pop(key, None)
        return {key: futures[key].result() for key in ordered}

    def one(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        return self.many([key], lambda keys: {key: loader()})[key]


@dataclass
class ProviderMemo:
    """Mutable provider state owned by one run, not by a module."""

    flight: SingleFlight = field(default_factory=SingleFlight)
    lock: RLock = field(default_factory=RLock)
    stores: dict[str, dict] = field(default_factory=dict)
    optional_stores: dict[Hashable, dict] = field(default_factory=dict)
    generations: dict[str, int] = field(default_factory=dict)

    def store(self, namespace: str) -> dict:
        with self.lock:
            return self.stores.setdefault(namespace, {})

    def optional_store(self, namespace: str, authority: Hashable) -> dict:
        with self.lock:
            return self.optional_stores.setdefault((namespace, authority), {})


_compatibility = ProviderMemo()


def provider_memo() -> ProviderMemo:
    from tarzan.runtime.session import current_session

    session = current_session()
    return session.providers if session is not None else _compatibility


def provider_store(namespace: str, legacy: dict) -> dict:
    """Keep direct helper compatibility; production always uses run ownership."""
    from tarzan.runtime.session import current_session

    return provider_memo().store(namespace) if current_session() is not None else legacy


def _has_provider_payload(value) -> bool:
    """Distinguish established missing-result conventions, not market validity.

    A nonempty raw payload is reusable only as the same provider/request fact;
    the caller's decoder, ``cache_if`` and financial policies still decide
    identity, dates, currencies and usability. Numeric zero/False are legitimate
    scalar facts and must not be discarded by a generic truthiness check.
    """
    if value is None:
        return False
    import numpy as np
    import pandas as pd

    if value is pd.NA or value is pd.NaT:
        return False
    if isinstance(value, np.ndarray):
        return bool(value.size)
    if isinstance(value, (Mapping, list, tuple, set, frozenset, str, bytes, pd.Series, pd.DataFrame)):
        return bool(len(value))
    return True


def memoized_many(namespace: str, keys, loader, *, legacy: dict, cache_if=None) -> dict:
    """Coalesce missing keys and return defensive copies of provider facts.

    By default completed misses are retained, as before. A caller may opt into
    selective admission: rejected observations still reach concurrent waiters,
    but a later request can retry. Admission is validated before storing any
    member of a batch.

    Optional work may borrow completed essential facts, but it neither leads
    essential single-flight waiters nor publishes its misses into their cache.
    Its missing results and pending work belong to that exact run/optional
    deadline. Nonempty completed provider facts remain reusable without
    rescheduling I/O. Another optional deadline can retry independently. No market
    identity, freshness, absence or inception is inferred by this boundary.
    """
    owner = provider_memo()
    common = provider_store(namespace, legacy)
    from tarzan.runtime.session import current_session

    session = current_session()
    budget = getattr(session, "transport", None)
    optional_token = getattr(budget, "optional_work_token", None)
    work = optional_token() if callable(optional_token) else None
    authority = (budget, work[2]) if work is not None else None
    def cached(key):
        # A completed essential fact needs no fresh transport authority.
        return key in common or key in store

    def fact(key):
        return common[key] if key in common else store[key]

    def flight_key(key):
        return namespace, key, authority, generation

    ordered = list(dict.fromkeys(keys))
    with owner.lock:
        generation = owner.generations.get(namespace, 0)
        store = owner.optional_store(namespace, authority) if authority is not None else common
        # Preserve facts already read by this call even if a later explicit
        # reset clears their dictionary while another batch key is loading.
        initial = {key: fact(key) for key in ordered if cached(key)}
        missing = [key for key in ordered if key not in initial]

    def load_reserved(reserved):
        actual = [key[1] for key in reserved]
        with owner.lock:
            current_generation = owner.generations.get(namespace, 0) == generation
            established = {
                key: fact(key) for key in actual if current_generation and cached(key)
            }
            still_missing = [key for key in actual if key not in established]
        fetched = loader(still_missing) if still_missing else {}
        if any(key not in fetched for key in still_missing):
            raise ValueError("provider loader omitted a requested key")
        admitted = [
            key for key in still_missing
            if cache_if is None or cache_if(fetched[key])
        ]
        with owner.lock:
            current_generation = owner.generations.get(namespace, 0) == generation
            # Reset does not pretend to kill native I/O. An older caller may
            # receive its response, but its leader cannot republish facts into
            # the refreshed namespace or lead a post-reset single flight.
            for key in admitted if current_generation else ():
                value = deepcopy(fetched[key])
                if authority is None:
                    # An independently completed optional payload can arrive
                    # while an essential leader was fetching. Its existence
                    # must not be overwritten by a later missing result.
                    if key not in common or _has_provider_payload(value):
                        common[key] = value
                elif _has_provider_payload(value):
                    # First completed common fact wins; source selection stays
                    # with the ordinary caller, never this presence predicate.
                    if key not in common or not _has_provider_payload(common[key]):
                        common[key] = value
                else:
                    store[key] = value
            return {
                flight_key(key): (
                    fact(key) if current_generation and cached(key)
                    else fetched[key] if key in fetched else established[key]
                )
                for key in actual
            }

    resolved = {}
    if missing:
        resolved = owner.flight.many([flight_key(key) for key in missing], load_reserved)
    with owner.lock:
        return {
            key: deepcopy(
                resolved[flight_key(key)] if flight_key(key) in resolved else initial[key]
            )
            for key in ordered
        }


def memoized(namespace: str, key, loader, *, legacy: dict, cache_if=None):
    return memoized_many(
        namespace, [key], lambda keys: {key: loader()}, legacy=legacy, cache_if=cache_if,
    )[key]


def clear_provider_stores(*namespaces: str) -> None:
    """Refresh named namespaces without admitting an older pending leader."""
    owner = provider_memo()
    with owner.lock:
        for namespace in dict.fromkeys(namespaces):
            owner.generations[namespace] = owner.generations.get(namespace, 0) + 1
            owner.store(namespace).clear()
        for key in tuple(owner.optional_stores):
            if key[0] in namespaces:
                del owner.optional_stores[key]
