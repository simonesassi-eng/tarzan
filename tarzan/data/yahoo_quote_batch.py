"""Share completed native quote groups without replacing native info parsing.

Each symbol still receives its independent full quote-summary/profile and the
SDK's exact flattening and precedence. Inconclusive members retain ordinary
individual acquisition. Facts are transient to this one acquisition, never
negative cache entries or authority to change listing resolution.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from inspect import unwrap
from pathlib import Path

from yfinance.config import YfConfig
from yfinance.data import YfData

from tarzan.data import _yf_net, yahoo_info
from tarzan.runtime.quote_recovery import acquisition, completed_native_core, _positive_json_number
from tarzan.runtime.yahoo_metadata import _completed_metadata_source


_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?"
_NATIVE_RAW_JSON = YfData.get_raw_json
_NATIVE_GET = YfData.get
_MAX_NATIVE_INFO_WORKERS = 2


def _parameters(symbols):
    return {
        "symbols": ",".join(symbols), "formatted": "false",
        "lang": YfConfig.locale.lang, "region": YfConfig.locale.region,
    }


@dataclass(frozen=True)
class _CompletedQuote:
    symbol: str
    parameters: dict
    record: dict


class _CompletedQuoteData:
    """Local view on one fresh private client; no shared SDK mutation."""

    def __init__(self, original, fact):
        self.original = original
        self.fact = fact
        self.reused = 0

    def get_raw_json(self, url, params=None, timeout=30):
        if (
            url == _QUOTE_URL and params == self.fact.parameters
            and type(timeout) in (int, float) and timeout == 30
        ):
            self.reused += 1
            return {"quoteResponse": {
                "result": [deepcopy(self.fact.record)], "error": None,
            }}
        return self.original.get_raw_json(url, params=params, timeout=timeout)


def _native_data(data):
    import yfinance.data as sdk_data

    try:
        return (
            type(data) is YfData and YfData.get_raw_json is _NATIVE_RAW_JSON
            and YfData.get is _NATIVE_GET
            and getattr(data.get_raw_json, "__func__", None) is _NATIVE_RAW_JSON
            and getattr(data.get, "__func__", None) is _NATIVE_GET
            and hasattr(_NATIVE_RAW_JSON, "__code__")
            and hasattr(_NATIVE_GET, "__code__")
            and Path(_NATIVE_RAW_JSON.__code__.co_filename).resolve()
            == Path(sdk_data.__file__).resolve()
            and Path(_NATIVE_GET.__code__.co_filename).resolve().is_relative_to(
                Path(sdk_data.__file__).resolve().parent,
            )
            and Path(unwrap(_NATIVE_GET).__code__.co_filename).resolve()
            == Path(sdk_data.__file__).resolve()
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _owned_response_port(data):
    from tarzan.runtime.session import current_session
    from tarzan.runtime.yahoo_response import YahooChartResponse

    session = current_session()
    return (
        session is not None
        and data._session is session.transport.yahoo_client()
        and getattr(data._session, "response_class", None) is YahooChartResponse
    )


def _ordinary(symbol):
    return (
        type(symbol) is str and bool(symbol) and symbol == symbol.strip()
        and symbol == symbol.upper()
        and "," not in symbol and not yahoo_info._ISIN_SHAPE.fullmatch(symbol.upper())
    )


def _positive(record):
    return type(record) is dict and any(
        _positive_json_number(record.get(name))
        for name in ("regularMarketPrice", "previousClose", "regularMarketPreviousClose")
    )


def _facts(symbols, data):
    parameters = _parameters(symbols)

    def fetch():
        response = data.get(_QUOTE_URL, params=parameters)
        response.raise_for_status()
        from tarzan.runtime.yahoo_response import YahooChartResponse
        if type(response) is not YahooChartResponse:
            return {}
        if _completed_metadata_source(response) != ("quoteResponse", tuple(symbols)):
            return {}
        document = response.json()
        envelope = document.get("quoteResponse") if type(document) is dict else None
        if type(envelope) is not dict or envelope.get("error") is not None:
            return {}
        records = envelope.get("result")
        if type(records) is not list:
            return {}
        result = {}
        for record in records:
            if (
                _positive(record) and type(record.get("symbol")) is str
                and record["symbol"] in symbols and record["symbol"] not in result
            ):
                symbol = record["symbol"]
                result[symbol] = _CompletedQuote(
                    symbol, {**parameters, "symbols": symbol}, deepcopy(record),
                )
        return result

    observed = _yf_net.fetch_yf(fetch, what="native quote group")
    return observed if type(observed) is dict else {}


def _core(symbol, fact):
    client = _yf_net.ticker(symbol)
    if not yahoo_info._native_owned_core(client):
        return client.info
    if not _native_data(client._quote._data):
        return yahoo_info.info_for_enrichment(symbol)
    if client.ticker != symbol:
        return yahoo_info.info_for_enrichment(symbol)
    original = client._quote._data
    view = _CompletedQuoteData(original, fact) if fact is not None else None
    try:
        if view is not None:
            client._quote._data = view
        client._quote._fetch_info()
        # A changed locale/parameter contract delegates to the full native
        # individual request, rather than relabeling the prior group's fact.
        if view is not None and view.reused > 1:
            raise RuntimeError("native core consumed its quote more than once")
        completed_native_core(client)
        return deepcopy(client._quote._info)
    finally:
        client._quote._data = original


def _individual(symbol):
    def load():
        _yf_net.space_yf_call()
        return (
            _core(symbol, None) if _ordinary(symbol)
            else yahoo_info.info_for_enrichment(symbol)
        )

    result = _yf_net.retry(load, what="native individual metadata")
    return result if isinstance(result, dict) else {}


def info_many(symbols):
    """Acquire exactly the demanded symbols, with bounded groups and recovery."""
    from tarzan.runtime.session import current_session

    ordered = list(dict.fromkeys(symbols))
    if not yahoo_info._native_owned_contract():
        return {
            symbol: _yf_net.fetch_yf(
                lambda symbol=symbol: yahoo_info.info_for_enrichment(symbol),
                what="metadata",
            ) or {}
            for symbol in ordered
        }
    data = _yf_net.yahoo_data()
    if not _native_data(data) or not _owned_response_port(data):
        return {symbol: _individual(symbol) for symbol in ordered}
    result = {}
    ordinary = [symbol for symbol in ordered if _ordinary(symbol)]
    for offset in range(0, len(ordinary), 64):
        chunk = ordinary[offset:offset + 64]
        if len(chunk) == 1:
            result[chunk[0]] = _individual(chunk[0])
            continue
        with acquisition(current_session().transport, chunk):
            facts = _facts(chunk, data)

            def load(symbol):
                if symbol not in facts:
                    return _individual(symbol)
                observed = _yf_net.retry(
                    lambda symbol=symbol: _core(symbol, facts[symbol]),
                    what="native grouped metadata",
                )
                return observed if isinstance(observed, dict) else {}

            from tarzan import config
            with ThreadPoolExecutor(
                # Grouped metadata must not create synchronized four-way
                # summary retries. Native I/O still belongs to the unchanged
                # global coordinator/capacity and timeout/pacing policy.
                max_workers=min(config.max_workers(), _MAX_NATIVE_INFO_WORKERS, len(chunk)),
                thread_name_prefix="tarzan-native-info",
            ) as pool:
                pending = {
                    symbol: pool.submit(copy_context().run, load, symbol)
                    for symbol in chunk
                }
                result.update((symbol, future.result()) for symbol, future in pending.items())
    for symbol in ordered:
        if symbol not in result:
            result[symbol] = _individual(symbol)
    return {symbol: result[symbol] for symbol in ordered}
