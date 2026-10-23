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

    def store(self, namespace: str) -> dict:
        with self.lock:
            return self.stores.setdefault(namespace, {})


_compatibility = ProviderMemo()


def provider_memo() -> ProviderMemo:
    from tarzan.runtime.session import current_session

    session = current_session()
    return session.providers if session is not None else _compatibility


def provider_store(namespace: str, legacy: dict) -> dict:
    """Keep direct helper compatibility; production always uses run ownership."""
    from tarzan.runtime.session import current_session

    return provider_memo().store(namespace) if current_session() is not None else legacy


def memoized_many(namespace: str, keys, loader, *, legacy: dict, cache_if=None) -> dict:
    """Coalesce missing keys and return defensive copies of provider facts.

    By default completed misses are retained, as before. A caller may opt into
    selective admission: rejected observations still reach concurrent waiters,
    but a later request can retry. Admission is validated before storing any
    member of a batch.
    """
    owner = provider_memo()
    store = provider_store(namespace, legacy)
    ordered = list(dict.fromkeys(keys))
    with owner.lock:
        missing = [key for key in ordered if key not in store]

    def load_reserved(reserved):
        actual = [key[1] for key in reserved]
        with owner.lock:
            still_missing = [key for key in actual if key not in store]
        fetched = loader(still_missing) if still_missing else {}
        if any(key not in fetched for key in still_missing):
            raise ValueError("provider loader omitted a requested key")
        admitted = [
            key for key in still_missing
            if cache_if is None or cache_if(fetched[key])
        ]
        with owner.lock:
            for key in admitted:
                store[key] = deepcopy(fetched[key])
            return {
                (namespace, key): store[key] if key in store else fetched[key]
                for key in actual
            }

    resolved = {}
    if missing:
        resolved = owner.flight.many([(namespace, key) for key in missing], load_reserved)
    with owner.lock:
        return {
            key: deepcopy(
                resolved[(namespace, key)] if (namespace, key) in resolved else store[key]
            )
            for key in ordered
        }


def memoized(namespace: str, key, loader, *, legacy: dict, cache_if=None):
    return memoized_many(
        namespace, [key], lambda keys: {key: loader()}, legacy=legacy, cache_if=cache_if,
    )[key]


def clear_provider_stores(*namespaces: str) -> None:
    owner = provider_memo()
    with owner.lock:
        for namespace in namespaces:
            owner.store(namespace).clear()
