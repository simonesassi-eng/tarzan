"""Request the metadata enrichment consumes, not unused complementary PEG.

The pinned SDK's native core parser preserves all quote-summary and quote
fields. Its separate complementary request adds only trailingPegRatio, which
Tarzan's enrichment does not consume. Unknown SDKs and injected/public callers
keep the full info contract. No SDK class, global flag or transport is patched.
"""

from pathlib import Path

from curl_cffi.requests.exceptions import HTTPError as _NATIVE_HTTP_ERROR
import yfinance as yf
from yfinance.base import TickerBase as _SDK_BASE
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


def _native_owned_core(client) -> bool:
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
        or type(client) is not _SDK_TICKER
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
        return type(vars(client).get("_quote")) is _SDK_QUOTE
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def info_for_enrichment(symbol):
    """Keep required fields and native fallback/retry parsing intact.

    The client is created here and never escapes. No caller-supplied Ticker
    loses its complementary fields, and no cross-run metadata is cached.
    Errors still reach the existing enrichment retry/availability policy.
    """
    client = _yf_net.ticker(symbol)
    if not _native_owned_core(client):
        return client.info
    quote = client._quote
    quote._fetch_info()
    return quote._info
