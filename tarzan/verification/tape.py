"""Record leaf provider answers, then fail closed on unrecorded offline I/O.

The tape freezes transport inputs, not enrichment decisions or financial
results. JSON is deliberately non-executable. Request headers and credentials
are never recorded. A caller owns the private snapshot directory.
"""

from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import io
import json
import math
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import ExitStack
from enum import Enum
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from tarzan.runtime.io_utils import canonical_json_bytes
from tarzan.runtime.ledger import ErrorNormalizer


class UnrecordedInput(BaseException):
    """A verification failure that provider fallback handlers cannot swallow."""


def encode(value):
    """Lossless JSON representation of the provider types used by Tarzan."""
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "table": json.loads(value.to_json(orient="table", date_format="iso", double_precision=15)),
            "attrs": encode(value.attrs),
            "frequency": getattr(value.index, "freqstr", None),
        }
    if isinstance(value, pd.Series):
        return {"type": "series", "frame": encode(value.to_frame("__value__")), "name": encode(value.name)}
    if isinstance(value, Mapping):
        return {"type": "mapping", "items": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": "sequence", "items": [encode(item) for item in value], "tuple": isinstance(value, tuple)}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return encode(value.value)
    if isinstance(value, dt.datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, dt.date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "nonfinite", "value": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"unrecordable provider type: {type(value).__name__}")


def decode(value):
    kind = value["type"]
    if kind == "dataframe":
        frame = pd.read_json(io.StringIO(json.dumps(value["table"])), orient="table")
        frame.attrs = decode(value["attrs"])
        frequency = value.get("frequency")
        if frequency and isinstance(frame.index, pd.DatetimeIndex):
            try:
                frame.index = pd.DatetimeIndex(frame.index, freq=frequency)
            except ValueError:
                pass
        return frame
    if kind == "series":
        series = decode(value["frame"]).iloc[:, 0]
        series.name = decode(value["name"])
        return series
    if kind == "mapping":
        return {decode(key): decode(item) for key, item in value["items"]}
    if kind == "sequence":
        items = [decode(item) for item in value["items"]]
        return tuple(items) if value["tuple"] else items
    if kind == "bytes":
        return base64.b64decode(value["value"])
    if kind == "datetime":
        return dt.datetime.fromisoformat(value["value"])
    if kind == "date":
        return dt.date.fromisoformat(value["value"])
    if kind == "nonfinite":
        return float(value["value"])
    if kind == "scalar":
        return value["value"]
    raise ValueError(f"unknown tape value type: {kind}")


class HTTPAnswer:
    """Minimal response shared by urllib, requests and Yahoo's quote endpoint."""

    def __init__(self, *, body: bytes, status: int = 200, url: str = "", encoding: str = "utf-8",
                 headers: Mapping | None = None):
        self.content = body
        self.status = self.status_code = status
        self.url = url
        self.encoding = encoding or "utf-8"
        self.headers = dict(headers or {})
        self._stream = io.BytesIO(body)

    @property
    def text(self):
        return self.content.decode(self.encoding, errors="replace")

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return json.loads(self.text)

    def read(self, *args):
        return self._stream.read(*args)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"recorded HTTP {self.status_code}", response=self)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _http_value(response, url):
    body = response.content if hasattr(response, "content") else response.read()
    return {
        "body": body,
        "status": getattr(response, "status_code", getattr(response, "status", 200)),
        "url": url,
        "encoding": getattr(response, "encoding", None) or "utf-8",
        # Only public protocol metadata needed by the adapters; never cookies,
        # authentication headers or the client's request headers.
        "headers": {"content-type": response.headers.get("content-type", "")}
        if hasattr(response, "headers") else {},
    }


def _semantic_arguments(kwargs):
    # Timeout/session/retry controls do not change a requested market observation.
    # All data-affecting parameters (period, interval, adjustment, date) stay.
    arguments = {
        key: value for key, value in kwargs.items()
        if key not in {"timeout", "session", "headers", "verify", "proxies"}
    }
    # This facility records public market responses, not authenticated API
    # requests. Reject credentials rather than storing a redacted request that
    # could accidentally alias another observation on replay.
    def reject_credentials(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if ErrorNormalizer._SENSITIVE_KEYS.search(str(key)):
                    raise ValueError("credential-bearing request is not a market observation")
                reject_credentials(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_credentials(item)
        elif isinstance(value, str) and ErrorNormalizer._sanitize_text(value) != value:
            raise ValueError("credential-bearing request is not a market observation")

    reject_credentials(arguments)
    return arguments


def _request_key(operation, arguments):
    """Canonicalize maps and symbol batches, never market observations."""
    safe = _semantic_arguments(arguments)
    if operation == "yf.http" and isinstance(safe.get("params"), Mapping):
        params = dict(safe["params"])
        if isinstance(params.get("symbols"), str):
            params["symbols"] = ",".join(sorted(params["symbols"].split(",")))
        safe = {**safe, "params": params}

    def canonical(value):
        if isinstance(value, Mapping):
            return {key: canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return type(value)(canonical(item) for item in value)
        return value

    safe = canonical(safe)
    key = hashlib.sha256(canonical_json_bytes(
        {"operation": operation, "arguments": encode(safe)},
    )).hexdigest()
    return key, safe


class ProviderTape:
    """Thread-safe transport tape with isolated copies and an offline guard."""

    SCHEMA_VERSION = "1"

    def __init__(self, *, records=None, replay=False):
        self.replay = replay
        self.records = {}
        for record in copy.deepcopy(records or {}).values():
            key, arguments = _request_key(record["operation"], decode(record["arguments"]))
            if key in self.records and self.records[key].get("value") != record.get("value") and replay:
                raise ValueError("ambiguous frozen answers for one semantic request")
            record["arguments"] = encode(arguments)
            self.records.setdefault(key, record)
        self._lock = threading.RLock()
        self._stack = ExitStack()
        self._pace = None
        self.calls = 0
        self.misses = []
        self.network_attempts = []

    def call(self, operation, arguments, producer):
        key, safe_arguments = _request_key(operation, arguments)
        with self._lock:
            self.calls += 1
            existing = self.records.get(key)
            if existing is not None:
                return self._answer(existing)
            if self.replay:
                self.misses.append({"operation": operation, "arguments": safe_arguments})
                raise UnrecordedInput(f"unrecorded {operation}; no network fallback permitted")
        try:
            if self._pace is not None and operation.startswith("yf.") and operation != "yf.ticker":
                self._pace()
            packed = {"operation": operation, "arguments": encode(safe_arguments), "value": encode(producer())}
        except Exception as error:
            packed = {
                "operation": operation, "arguments": encode(safe_arguments),
                "error": {"type": type(error).__name__, "message": ErrorNormalizer.normalize(error)["message"]},
            }
        with self._lock:
            # Single frozen answer per semantic request. Concurrent capture misses
            # may reach the provider, but all consumers select the same winner.
            packed = self.records.setdefault(key, packed)
        return self._answer(packed)

    @staticmethod
    def _answer(record):
        error = record.get("error")
        if error:
            name, message = error["type"], error["message"]
            if "Timeout" in name or name == "TimeoutError":
                raise requests.Timeout(message)
            if name in {"HTTPError", "URLError"}:
                raise requests.RequestException(message)
            raise RuntimeError(message)
        return decode(record["value"])

    def _block(self, *args, **kwargs):
        self.network_attempts.append("unrecorded network transport")
        raise UnrecordedInput("offline verification attempted an unrecorded network transport")

    def __enter__(self):
        import yfinance as yf
        import yfinance.multi as yf_multi
        from yfinance.data import YfData
        from tarzan.data import bond_fetcher, bond_history, enricher
        from tarzan.data import _yf_net

        # Throttle actual captures, not already recorded responses. Recording
        # misses call the original pacer; replay cannot reach this branch.
        self._pace = _yf_net.space_yf_call
        self._stack.enter_context(patch.object(_yf_net, "space_yf_call", lambda: None))

        tape = self
        original_ticker = yf.Ticker

        class Funds:
            def __init__(self, ticker):
                self.ticker = ticker

            def __getattr__(self, name):
                return tape.call("yf.funds." + name, {"symbol": self.ticker.symbol},
                                 lambda: getattr(self.ticker.actual().funds_data, name))

        class Ticker:
            def __init__(self, symbol, *args, **kwargs):
                self.symbol = str(symbol)
                self._actual = None
                self._init_args, self._init_kwargs = args, kwargs
                self.ticker = tape.call("yf.ticker", {"symbol": self.symbol},
                                       lambda: self.actual().ticker)

            def actual(self):
                if tape.replay:
                    raise UnrecordedInput("offline tape cannot construct a live provider client")
                if self._actual is None:
                    self._actual = original_ticker(self.symbol, *self._init_args, **self._init_kwargs)
                return self._actual

            @property
            def info(self):
                return tape.call("yf.info", {"symbol": self.symbol}, lambda: self.actual().info)

            def get_info(self):
                return self.info

            def history(self, *args, **kwargs):
                return tape.call("yf.history", {"symbol": self.symbol, "args": args, **kwargs},
                                 lambda: self.actual().history(*args, **kwargs))

            @property
            def isin(self):
                return tape.call("yf.isin", {"symbol": self.symbol}, lambda: self.actual().isin)

            @property
            def fast_info(self):
                return tape.call("yf.fast_info", {"symbol": self.symbol}, lambda: dict(self.actual().fast_info))

            @property
            def funds_data(self):
                return Funds(self)

        original_get = YfData.get

        def yahoo_get(instance, url, *args, **kwargs):
            value = tape.call("yf.http", {"url": url, "args": args, **kwargs},
                              lambda: _http_value(original_get(instance, url, *args, **kwargs), url))
            return HTTPAnswer(**value)

        original_request = requests.Session.request

        def request(instance, method, url, **kwargs):
            value = tape.call("requests", {"method": method, "url": url, **kwargs},
                              lambda: _http_value(original_request(instance, method, url, **kwargs), url))
            return HTTPAnswer(**value)

        original_urlopen = urllib.request.urlopen

        def urlopen(request, *args, **kwargs):
            url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
            body = request.data if isinstance(request, urllib.request.Request) else None
            value = tape.call("urllib", {"url": url, "body": body},
                              lambda: _http_value(original_urlopen(request, *args, **kwargs), url))
            return HTTPAnswer(**value)

        original_bond_history = bond_history._request_raw

        def official_bond_history(isin, *, market="MOTX"):
            # Freeze only raw market fields. The publisher's anonymous public
            # view HTML/access descriptor is never a verification artifact.
            return tape.call(
                "borsa.bond_history",
                {"isin": isin, "exchange": bond_history._MARKETS.get(market), "period": "5Y",
                 "adjustment": False, "add-last-price": False},
                lambda: (
                    original_bond_history(isin) if market == "MOTX"
                    else original_bond_history(isin, market=market)
                ),
            )

        self._stack.enter_context(patch.object(yf, "Ticker", Ticker))
        self._stack.enter_context(patch.object(yf_multi, "Ticker", Ticker))
        self._stack.enter_context(patch.object(YfData, "get", yahoo_get))
        self._stack.enter_context(patch.object(requests.Session, "request", request))
        self._stack.enter_context(patch.object(urllib.request, "urlopen", urlopen))
        self._stack.enter_context(patch.object(enricher, "urlopen", urlopen))
        self._stack.enter_context(patch.object(bond_fetcher, "urlopen", urlopen))
        self._stack.enter_context(patch.object(bond_history, "_request_raw", official_bond_history))
        if self.replay:
            self._stack.enter_context(patch.object(socket.socket, "connect", self._block))
            self._stack.enter_context(patch.object(socket, "create_connection", self._block))
            from curl_cffi import requests as curl_requests
            self._stack.enter_context(patch.object(curl_requests.Session, "request", self._block))
        return self

    def __exit__(self, *args):
        return self._stack.__exit__(*args)
