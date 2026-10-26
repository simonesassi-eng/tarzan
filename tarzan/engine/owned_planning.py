"""Speculative native CPU preparation, never planning/publication authority."""

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import enum
import multiprocessing
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from struct import pack, unpack
from types import FunctionType

import numpy as np

from tarzan.engine import allocations, rebalancer, tax
from tarzan.instruments import registry, valuation
from tarzan.models.holding import AssetClass, Geography, Holding
from tarzan.models.investor_config import InvestorConfig
from tarzan.runtime.provider import ProviderQualityPolicy
from tarzan.runtime.transport import ProviderTransportPolicy, ProviderTransportSettings


_INPUT_DATACLASSES = (
    InvestorConfig, registry.TypeResolution, rebalancer.LSParams,
    ProviderQualityPolicy, ProviderTransportPolicy, ProviderTransportSettings,
)
_INPUT_ENUMS = (AssetClass, Geography, registry.InstrumentKind, registry.TypeResolutionState)
_FACT_TYPES = {
    f"{cls.__module__}.{cls.__qualname__}": cls
    for cls in (*_INPUT_DATACLASSES, *_INPUT_ENUMS)
}


# These are every Holding field read by the inspected native solver,
# allocation primitives and mechanics selector. Histories are not solver inputs.
HOLDING_FIELDS = (
    "isin", "ticker", "currency", "name", "instrument_type", "security_type",
    "instrument_kind_evidence", "instrument_provider_evidence",
    "instrument_resolution", "quantity", "cost_basis_eur",
    "market_value_eur", "current_value", "current_price", "gain_pct",
    "asset_class", "no_buy_no_sell", "geography", "geo_breakdown",
    "class_breakdown", "target_equities", "target_fixed_income",
    "target_portfolio", "is_seeded_target",
)


def _native_ports():
    ports = []
    for module in (rebalancer, allocations, registry, valuation):
        root = Path(module.__file__).resolve()
        for name, value in vars(module).items():
            if isinstance(value, FunctionType) and value.__module__ == module.__name__:
                ports.append((module, name, value, root))
            elif isinstance(value, type) and value.__module__ == module.__name__:
                ports.append((module, name, value, root))
                for method_name, method in vars(value).items():
                    if isinstance(method, (classmethod, staticmethod)):
                        method = method.__func__
                    if isinstance(method, FunctionType):
                        ports.append((value, method_name, method, root))
    ports.append((tax, "is_government_bond", tax.is_government_bond, Path(tax.__file__).resolve()))
    for cls in (*_INPUT_DATACLASSES, Holding):
        root = Path(sys.modules[cls.__module__].__file__).resolve()
        for name in ("__init__", "__post_init__"):
            method = vars(cls).get(name)
            if isinstance(method, FunctionType):
                ports.append((cls, name, method, root))
    return tuple(ports)


_NATIVE_PORTS = _native_ports()


def _known_origin(owner, name, original, root):
    filename = original.__code__.co_filename
    if Path(filename).resolve() == root:
        return True
    # Python generates these methods for native dataclasses/Enums. Their
    # identity is still checked on every admission; arbitrary foreign methods
    # (including replacements before this module is imported) remain serial.
    if isinstance(owner, type) and is_dataclass(owner):
        return (
            name in {"__init__", "__repr__", "__eq__", "__setattr__",
                     "__delattr__", "__hash__"}
            and (filename == "<string>"
                 or Path(filename).resolve() == Path(dataclasses.__file__).resolve())
        )
    if isinstance(owner, type) and issubclass(owner, Enum):
        return (
            name in {"_generate_next_value_", "__repr__", "__str__", "__format__", "__new__"}
            and Path(filename).resolve() == Path(enum.__file__).resolve()
        )
    return False


def native_solver_available():
    """Injected solver/allocation/mechanics ports keep the serial contract."""
    if type(_NATIVE_PORTS) is not tuple or not _NATIVE_PORTS:
        return False
    try:
        for owner, name, original, root in _NATIVE_PORTS:
            current = vars(owner).get(name)
            if isinstance(current, (classmethod, staticmethod)):
                current = current.__func__
            if current is not original:
                return False
            if not isinstance(original, type) and not _known_origin(owner, name, original, root):
                return False
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _freeze(value):
    """Typed exact keys; mapping insertion order and float bits are retained."""
    if value is None:
        return ("none",)
    if type(value) in (str, bool, int):
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float", pack(">d", value))
    if type(value) in _INPUT_ENUMS:
        return ("enum", f"{type(value).__module__}.{type(value).__qualname__}", _freeze(value.value))
    if isinstance(value, np.generic) and np.issubdtype(value.dtype, np.number):
        return ("numpy", value.dtype.str, value.tobytes())
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(_freeze(item) for item in value))
    if type(value) is dict:
        return ("dict", tuple((_freeze(key), _freeze(item)) for key, item in value.items()))
    if type(value) in _INPUT_DATACLASSES:
        # Read ordinary stored fields only, never arbitrary property access.
        data = vars(value)
        return ("dataclass", f"{type(value).__module__}.{type(value).__qualname__}", tuple(
            (field.name, _freeze(data[field.name])) for field in fields(value)
        ))
    raise TypeError("unsupported native planning input")


def _thaw(value):
    """Restore only declared native fact types from a builtins-only payload."""
    tag = value[0]
    if tag == "none":
        return None
    if tag in {"str", "bool", "int"}:
        return value[1]
    if tag == "float":
        return unpack(">d", value[1])[0]
    if tag == "numpy":
        return np.frombuffer(value[2], dtype=np.dtype(value[1]), count=1)[0]
    if tag in {"list", "tuple"}:
        return (list if tag == "list" else tuple)(_thaw(item) for item in value[1])
    if tag == "dict":
        return {_thaw(key): _thaw(item) for key, item in value[1]}
    cls = _FACT_TYPES[value[1]]
    if tag == "enum" and cls in _INPUT_ENUMS:
        return cls(_thaw(value[2]))
    if tag == "dataclass" and cls in _INPUT_DATACLASSES:
        data = dict(value[2])
        if set(data) != {field.name for field in fields(cls)}:
            raise ValueError("native fact fields changed")
        result = object.__new__(cls)
        vars(result).update({name: _thaw(item) for name, item in data.items()})
        return result
    raise TypeError("unsupported native fact payload")


def _signature(holdings, config, total_value, lump_sum):
    if type(config) is not InvestorConfig or any(type(h) is not Holding for h in holdings):
        raise TypeError("uninspected planning input")
    return (
        tuple(tuple((name, _freeze(vars(h)[name])) for name in HOLDING_FIELDS) for h in holdings),
        _freeze(config), _freeze(total_value), _freeze(lump_sum),
        _freeze(rebalancer.LSParams()),
        _freeze(registry.TypeEvidenceGateway._MAPPING),
        _freeze(tax._GOV_BOND_NAME.pattern), _freeze(tax._GOV_BOND_NAME.flags),
    )


def _copy_fact(value):
    """Copy inspected stored facts without executing arbitrary copy hooks."""
    if (
        value is None or type(value) in (str, bool, int, float)
        or type(value) in _INPUT_ENUMS or isinstance(value, np.generic)
    ):
        return value  # admitted immutable scalars
    if type(value) in (list, tuple):
        return type(value)(_copy_fact(item) for item in value)
    if type(value) is dict:
        return {_copy_fact(key): _copy_fact(item) for key, item in value.items()}
    if type(value) in _INPUT_DATACLASSES:
        data = vars(value)
        stored = {
            field.name: _copy_fact(data[field.name]) for field in fields(value)
        }
        # These exact native Python dataclasses were admitted above. Copy
        # their already-established stored facts, not constructors, reduction
        # hooks, external attributes or provider-policy validation callbacks.
        result = object.__new__(type(value))
        vars(result).update(stored)
        return result
    raise TypeError("unsupported native planning fact")


@dataclass(frozen=True)
class PlanningInputs:
    holdings: tuple[Holding, ...]
    config: InvestorConfig
    total_value: float
    lump_sum: float | None


def snapshot_inputs(holdings, config, total_value, lump_sum):
    """Copy only consumed current facts; no tape, transport, ledger or audit."""
    if not native_solver_available():
        return None
    try:
        _signature(holdings, config, total_value, lump_sum)
        subjects = tuple(Holding(**{
            "currency": h.currency,
            **{name: _copy_fact(vars(h)[name]) for name in HOLDING_FIELDS},
        }) for h in holdings)
        configuration = _copy_fact(config)
        assert _signature(subjects, configuration, total_value, lump_sum) == _signature(
            holdings, config, total_value, lump_sum,
        )
        return PlanningInputs(subjects, configuration, total_value, lump_sum)
    except Exception:
        # Preparation admission is optional. Ordinary defects, including a
        # recursive unsupported input, retain the established owner computer.
        return None


@dataclass
class PreparedPlans:
    """Owned results can be consumed once, after the owner's ordinary gates."""

    variants: dict
    timing: dict = field(default_factory=dict)
    worker_evidence: dict = field(default_factory=dict)

    def take(self, holdings, config, total_value, lump_sum):
        if not native_solver_available():
            return None
        try:
            signature = _signature(holdings, config, total_value, lump_sum)
        except Exception:
            return None
        prepared = self.variants.pop(config.rebalancing_no_sell, None)
        if prepared is None or prepared[0] != signature:
            return None
        return deepcopy(prepared[1])


def prepare_plans(inputs):
    """Run both unchanged native searches; no audit or public decision here."""
    if not native_solver_available():
        return None
    from tarzan.runtime.workload import StageClock

    started = StageClock.capture()
    variants = {}
    for no_sell in (True, False):
        config = replace(inputs.config, rebalancing_no_sell=no_sell)
        signature = _signature(inputs.holdings, config, inputs.total_value, inputs.lump_sum)
        result = rebalancer.compute_unified_rebalancing(
            list(inputs.holdings), config, inputs.total_value, lump_sum=inputs.lump_sum,
        )
        if signature != _signature(inputs.holdings, config, inputs.total_value, inputs.lump_sum):
            return None
        variants[no_sell] = signature, result
    # Timing is descriptive and cannot change authoritative financial results.
    try:
        timing = started.fields()
    except Exception:
        timing = {}
    return PreparedPlans(variants, timing)


def _deny_worker_io(*args, **kwargs):
    raise RuntimeError("native planning worker has no network or email authority")


def _initialize_worker():
    """Remove inherited credentials before math and reject network/email I/O."""
    for name in tuple(os.environ):
        if (
            name.startswith(("SMTP_", "GOOGLE_", "DRIVE_", "DELIVERY_CLAIM_"))
            or name in {
                "GEMINI_API_KEY", "OPENFIGI_API_KEY", "GITHUB_TOKEN", "GH_TOKEN",
                "RECIPIENT_EMAIL", "TARZAN_CACHE_KEY_B64", "AUTHORIZED_RESEND_TOKEN",
            }
        ):
            os.environ.pop(name, None)
    os.environ["TARZAN_DISABLE_AI"] = "1"
    os.environ["TARZAN_DISABLE_CACHE"] = "1"
    import http.client
    import smtplib
    import socket
    import urllib.request
    import requests
    from curl_cffi import requests as curl_requests

    socket.create_connection = _deny_worker_io
    socket.socket.connect = _deny_worker_io
    socket.socket.connect_ex = _deny_worker_io
    socket.socket.sendto = _deny_worker_io
    http.client.HTTPConnection.connect = _deny_worker_io
    urllib.request.urlopen = _deny_worker_io
    requests.sessions.Session.request = _deny_worker_io
    curl_requests.Session.request = _deny_worker_io
    curl_requests.AsyncSession.request = _deny_worker_io
    smtplib.SMTP = _deny_worker_io
    smtplib.SMTP_SSL = _deny_worker_io


def _execute_native_payload(payload, day, captured_at):
    from datetime import date, datetime
    from tarzan import runtime
    from tarzan.runtime.session import RunContext, RunMode

    # Only builtins cross the process pipe, never arbitrary model pickle or
    # reduction hooks, histories, clients, ledgers or an investor's secrets.
    rows, config, total, lump = payload
    inputs = PlanningInputs(
        tuple(Holding(**_thaw(row)) for row in rows),
        _thaw(config), _thaw(total), _thaw(lump),
    )
    context = RunContext(
        "native-planning-worker", RunMode.REPRODUCIBLE,
        date.fromisoformat(day), datetime.fromisoformat(captured_at),
    )
    with runtime.bind_context(context):
        prepared = prepare_plans(inputs)
    if prepared is None:
        return None
    return (
        {flag: (key, _freeze(result)) for flag, (key, result) in prepared.variants.items()},
        prepared.timing,
        {"pid": os.getpid(), "native_solver_invocations": len(prepared.variants)},
    )


def prepare_plans_owned(inputs):
    """Own one spawn process through join; startup failures retain serial math."""
    if not native_solver_available():
        return None
    from tarzan import runtime
    from tarzan.runtime.workload import StageClock

    started = StageClock.capture()
    payload = (
        tuple(_freeze({name: vars(h)[name] for name in HOLDING_FIELDS}) for h in inputs.holdings),
        _freeze(inputs.config), _freeze(inputs.total_value), _freeze(inputs.lump_sum),
    )
    context = runtime.context()
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
    ) as pool:
        result = pool.submit(
            _execute_native_payload, payload,
            context.analysis_date.isoformat(), context.captured_at.isoformat(),
        ).result()
    if result is None:
        return None
    variants, child_timing, evidence = result
    evidence["separate_process"] = evidence.pop("pid") != os.getpid()
    try:
        timing = {
            **started.fields(),
            "worker_process_cpu_ms": child_timing.get("process_cpu_ms", 0.0),
            "worker_compute_ms": child_timing.get("duration_ms", 0.0),
        }
    except Exception:
        timing = {}
    return PreparedPlans(
        {flag: (key, _thaw(plan)) for flag, (key, plan) in variants.items()},
        timing, evidence,
    )
