"""Request the metadata enrichment consumes, not unused complementary PEG.

The pinned SDK's native core parser preserves all quote-summary and quote
fields. Its separate complementary request adds only trailingPegRatio, which
Tarzan's enrichment does not consume. Unknown SDKs and injected/public callers
keep the full info contract. No SDK class, global flag or transport is patched.
"""

from dataclasses import dataclass
from pathlib import Path
import re

from curl_cffi.requests.exceptions import HTTPError as _NATIVE_HTTP_ERROR
import yfinance as yf
from yfinance.base import TickerBase as _SDK_BASE
from yfinance.scrapers.history import PriceHistory as _SDK_HISTORY
from yfinance.scrapers.quote import Quote as _SDK_QUOTE
from yfinance.ticker import Ticker as _SDK_TICKER

from tarzan.data import _yf_net


_NATIVE_FACTORY = _yf_net.ticker
_NATIVE_SDK_PORTS = (
    (_SDK_TICKER, "__init__", _SDK_TICKER.__init__),
    (_SDK_BASE, "__init__", _SDK_BASE.__init__),
    (_SDK_TICKER, "get_info", _SDK_TICKER.get_info),
    (_SDK_TICKER, "info", _SDK_TICKER.info),
    (_SDK_QUOTE, "__init__", _SDK_QUOTE.__init__),
    (_SDK_QUOTE, "info", _SDK_QUOTE.info),
    (_SDK_QUOTE, "_fetch_info", _SDK_QUOTE._fetch_info),
    (_SDK_QUOTE, "_fetch_complementary", _SDK_QUOTE._fetch_complementary),
    (_SDK_QUOTE, "_fetch", _SDK_QUOTE._fetch),
    (_SDK_QUOTE, "_fetch_additional_info", _SDK_QUOTE._fetch_additional_info),
)
_NATIVE_HISTORY_PORTS = (
    (_SDK_BASE, "history", _SDK_BASE.history),
    (_SDK_BASE, "_lazy_load_price_history", _SDK_BASE._lazy_load_price_history),
    (_SDK_BASE, "_get_ticker_tz", _SDK_BASE._get_ticker_tz),
    (_SDK_BASE, "_fetch_ticker_tz", _SDK_BASE._fetch_ticker_tz),
    (_SDK_HISTORY, "__init__", _SDK_HISTORY.__init__),
    (_SDK_HISTORY, "history", _SDK_HISTORY.history),
)
_ISIN_SHAPE = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")
_LISTING_FACTS = "yahoo.native_constructor_listing"


@dataclass(frozen=True)
class _NativeListingIdentity:
    requested: str
    selected: str


class _NativeListingInfo(dict):
    """Dictionary-compatible metadata plus transient constructor evidence.

    The marker is not a provider field or a persisted resolution. Run memo
    defensive copies preserve it; JSON and ordinary dictionary copies do not
    publish it. An arbitrary metadata ``symbol`` never creates this evidence.
    """

    __slots__ = ("_source_identity",)

    def __init__(self, payload, identity):
        super().__init__(payload)
        self._source_identity = identity


def source_symbol_for_info(requested: str, info: dict) -> str:
    """Recover only an inspected native ISIN constructor's actual listing."""
    if type(requested) is not str or type(info) is not _NativeListingInfo:
        return requested
    identity = getattr(info, "_source_identity", None)
    return _identity_source(requested, identity)


def _identity_source(requested, identity):
    if (
        type(requested) is str
        and type(identity) is _NativeListingIdentity
        and identity.requested == requested
        and _ISIN_SHAPE.fullmatch(requested.upper()) is not None
        and type(identity.selected) is str
        and identity.selected
        and identity.selected == identity.selected.strip()
        and identity.selected != requested
        and _ISIN_SHAPE.fullmatch(identity.selected.upper()) is None
    ):
        return identity.selected
    return requested


def _constructor_listing(client, symbol):
    """Observe the already-created native client; do not search a second time."""
    if type(symbol) is not str or _ISIN_SHAPE.fullmatch(symbol.upper()) is None:
        return None
    selected = vars(client).get("ticker")
    quote_symbol = vars(client._quote).get("_symbol")
    if (
        type(selected) is str and selected
        and selected == selected.strip()
        and selected != symbol
        and selected == quote_symbol
        and _ISIN_SHAPE.fullmatch(selected) is None
    ):
        return _NativeListingIdentity(symbol, selected)
    return None


def _native_owned_contract() -> bool:
    """A bounded adapter for the inspected SDK, not a generic private API."""
    from tarzan import runtime
    from tarzan.runtime.session import current_session
    from tarzan.runtime.transport import ProviderBudget

    session = current_session()
    if (
        session is None
        or type(session.transport) is not ProviderBudget
        or not runtime.allows_live_transport()
        or getattr(yf, "__version__", None) != "1.7.0"
        or yf.Ticker is not _SDK_TICKER
        or _yf_net.ticker is not _NATIVE_FACTORY
        or type(_NATIVE_SDK_PORTS) is not tuple
        or len(_NATIVE_SDK_PORTS) != 10
    ):
        return False
    try:
        sdk_root = Path(yf.__file__).resolve().parent
        if Path(_NATIVE_FACTORY.__code__.co_filename).resolve() != Path(_yf_net.__file__).resolve():
            return False  # reject an injected factory captured before import
        for owner, name, original in _NATIVE_SDK_PORTS:
            if getattr(owner, name) is not original:
                return False
            function = original.fget if isinstance(original, property) else original
            if not Path(function.__code__.co_filename).resolve().is_relative_to(sdk_root):
                return False
        if _SDK_BASE.__init__.__globals__.get("Quote") is not _SDK_QUOTE:
            return False  # a factory alias can inject per-instance behavior
        if _SDK_QUOTE._fetch.__globals__.get("HTTPError") is not _NATIVE_HTTP_ERROR:
            return False  # an alternate backend/exception contract is uninspected
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _native_owned_core(client) -> bool:
    return (
        type(client) is _SDK_TICKER
        and type(vars(client).get("_quote")) is _SDK_QUOTE
        and _native_owned_contract()
    )


def is_native_isin_request(symbol) -> bool:
    """Distinguish an inspected ISIN constructor path without creating a client."""
    return (
        type(symbol) is str
        and _ISIN_SHAPE.fullmatch(symbol.upper()) is not None
        and _native_owned_contract()
    )


def record_constructor_listing(client, symbol):
    """Retain only a completed positive identity in the existing run owner.

    This is the constructor already needed by info/history, not a speculative
    client, search, cached miss or authority for additional transport.
    """
    if (
        type(symbol) is not str
        or _ISIN_SHAPE.fullmatch(symbol.upper()) is None
        or not _native_owned_core(client)
    ):
        return None
    identity = _constructor_listing(client, symbol)
    if identity is None:
        return None
    from tarzan.runtime.session import current_session

    owner = current_session().providers
    with owner.lock:
        owner.stores.setdefault(_LISTING_FACTS, {})[symbol] = identity
    return identity


def completed_constructor_source(symbol):
    """Read an existing positive identity without I/O, lookup or flight joins."""
    if not is_native_isin_request(symbol):
        return symbol
    from tarzan.runtime.session import current_session

    owner = current_session().providers
    with owner.lock:
        identity = owner.stores.get(_LISTING_FACTS, {}).get(symbol)
    return _identity_source(symbol, identity)


def info_for_enrichment(symbol):
    """Keep required fields and native fallback/retry parsing intact.

    The client is created here and never escapes. No caller-supplied Ticker
    loses its complementary fields, and no cross-run metadata is cached.
    Errors still reach the existing enrichment retry/availability policy.
    """
    client = _yf_net.ticker(symbol)
    if not _native_owned_core(client):
        return client.info
    identity = record_constructor_listing(client, symbol)
    quote = client._quote
    quote._fetch_info()
    if identity is not None and isinstance(quote._info, dict):
        return _NativeListingInfo(quote._info, identity)
    return quote._info


def _native_history_contract() -> bool:
    """Reject unknown history methods or factories without creating a client."""
    if (
        type(_NATIVE_HISTORY_PORTS) is not tuple
        or len(_NATIVE_HISTORY_PORTS) != 6
    ):
        return False
    try:
        sdk_root = Path(yf.__file__).resolve().parent
        for owner, name, original in _NATIVE_HISTORY_PORTS:
            if getattr(owner, name) is not original:
                return False
            if not Path(original.__code__.co_filename).resolve().is_relative_to(sdk_root):
                return False
        return _SDK_BASE._lazy_load_price_history.__globals__.get("PriceHistory") is _SDK_HISTORY
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _owns_timezone_only_history(client) -> bool:
    """Inspect the native history path before declining unused PEG demand."""
    return (
        _native_owned_core(client)
        and vars(client).get("_price_history") is None
        and vars(client._quote).get("_already_fetched_complementary") is False
        and _native_history_contract()
    )


def history_for_enrichment(symbol, **arguments):
    """Keep native history's full core timezone fallback, without unused PEG.

    The inspected history entry uses info only for its two timezone keys.
    Mark complementary metadata unnecessary on this fresh private client,
    never on SDK classes, public clients or run-shared metadata. Required
    native quote parsing, timezone limits, retries and provenance remain.
    """
    client = _yf_net.ticker(symbol)
    if _owns_timezone_only_history(client):
        client._quote._already_fetched_complementary = True
    record_constructor_listing(client, symbol)
    frame = client.history(**arguments)
    from tarzan.data import history_time
    from tarzan.data.chart_actions import attach_rejected_chart_evidence

    frame = history_time.admit_history(
        history_time.attach_native_session_evidence(frame, client), symbol,
    )
    return attach_rejected_chart_evidence(frame, client)
