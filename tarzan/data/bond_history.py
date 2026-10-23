"""Official, unadjusted bond closes for historical-only holdings.

The public Borsa Italiana chart delegates to Euronext's Milan history API.
Its public-view access descriptor is used only in memory, never logged or
cached. Exact ISIN, BOND kind, currency and non-adjustment evidence are
required. There is no cross-ISIN substitution or generated price history.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

import pandas as pd


NAMESPACE = "bond_history_v1"
SOURCE = "borsa_italiana"
SOURCE_ATTR = "_tarzan_bond_history_source"
BASIS_ATTR = "_tarzan_bond_history_basis"
ISIN_ATTR = "_tarzan_bond_history_isin"
OBSERVATIONS_ATTR = "_tarzan_bond_history_observations"
SCHEMA_VERSION = "1"
MAX_POINTS = 5000
MAX_CARRY_DAYS = 7
_BASE = "https://grafici.borsaitaliana.it/"
_EXCHANGE = "XMIL"
_KNOWN_AT = datetime(2026, 10, 22, 12, 27, tzinfo=timezone.utc)
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]\Z")
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass(frozen=True)
class _PublicView:
    isin: str
    exchange: str
    token: str = field(repr=False)


class _ViewParser(HTMLParser):
    def __init__(self, isin):
        super().__init__()
        self.isin = isin
        self.views = []

    def handle_starttag(self, tag, attrs):
        if tag != "chart-allinone":
            return
        values = dict(attrs)
        if len(values) != len(attrs):
            return
        token = values.get("token") or ""
        if (
            values.get("code") == self.isin
            and values.get("exchcode") == _EXCHANGE
            and values.get("url") == _BASE + "api/"
            and 0 < len(token) <= 4096
            and all(0x21 <= ord(character) <= 0x7E for character in token)
        ):
            self.views.append(_PublicView(self.isin, _EXCHANGE, token))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # No access header may be forwarded to another origin or endpoint.
        return None


def _read_body(opener, url, *, headers=None, limit=4 * 1024 * 1024):
    from tarzan.runtime.transport import provider_call

    def read(timeout):
        with opener.open(Request(url, headers=headers or _HEADERS), timeout=timeout) as response:
            content = response.read(limit + 1)
            if len(content) > limit:
                raise ValueError("public market response exceeds its byte limit")
            return content

    return provider_call("borsa", read, timeout=8)


def _public_market_payload(body):
    """Raw price fields only; never return access/error/SDK credential fields."""
    if not isinstance(body, dict):
        return None
    result = {name: body[name] for name in (
        "status", "entityID", "adjustement", "addDayLastPrice",
        "beginningDate", "endingDate", "period", "transco",
    ) if name in body}
    transco = result.get("transco")
    if isinstance(transco, dict):
        result["transco"] = {name: transco[name] for name in (
            "code", "codification", "exchCode",
        ) if name in transco}
    history = body.get("history")
    if isinstance(history, dict):
        result["history"] = {name: history[name] for name in (
            "accuracy", "currency", "instrType",
        ) if name in history}
        points = history.get("historyDt")
        if isinstance(points, list):
            if len(points) > MAX_POINTS:
                raise ValueError("public history exceeds its row limit")
            fields = (
                "dt", "openPx", "closePx", "highPx", "lowPx", "lastPx",
                "qty", "volNbTrade", "volCap", "setPx", "vwap",
            )
            result["history"]["historyDt"] = [
                {name: point[name] for name in fields if name in point}
                if isinstance(point, dict) else None
                for point in points
            ]
    return result


def _contains_descriptor(value, token):
    """A malformed provider echo must not cross the raw market-recording port."""
    if isinstance(value, str):
        return token in value
    if isinstance(value, dict):
        return any(
            _contains_descriptor(key, token) or _contains_descriptor(item, token)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_descriptor(item, token) for item in value)
    return False


def _request_raw(isin: str):
    """Credential-free market-input port used by the frozen verification tape."""
    if not _ISIN.fullmatch(str(isin)):
        return None
    opener = build_opener(_NoRedirect())
    view_html = _read_body(
        opener, _BASE + f"interactive-chart/{isin}-MOTX?lang=it", limit=100_000,
    ).decode("utf-8")
    parser = _ViewParser(isin)
    parser.feed(view_html)
    if len(parser.views) != 1:
        return None
    view = parser.views[0]
    query = urlencode({
        "period": "5Y", "adjustment": "false", "add-last-price": "false",
    })
    raw = _read_body(
        opener, _BASE + f"api/instruments/{isin},{view.exchange},ISIN/history/period?{query}",
        headers={**_HEADERS, "Accept": "application/json",
                 "Authorization": "Bearer " + view.token},
    )
    market_payload = _public_market_payload(json.loads(raw))
    if _contains_descriptor(market_payload, view.token):
        return None
    return {
        "exchange": view.exchange,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "payload": market_payload,
    }


def _instant(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return result.astimezone(timezone.utc) if result.tzinfo is not None else None


def _positive_price(value):
    if type(value) not in (int, float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def cache_entry_valid(entry, isin: str) -> bool:
    """Strict, credential-free schema shared by readers and encrypted export."""
    if not isinstance(entry, dict) or not _ISIN.fullmatch(str(isin)):
        return False
    common = {"schema_version", "source", "isin", "exchange", "fetched_at", "state"}
    if (
        entry.get("schema_version") != SCHEMA_VERSION
        or entry.get("source") != SOURCE or entry.get("isin") != isin
        or entry.get("exchange") != _EXCHANGE
        or _instant(entry.get("fetched_at")) is None
    ):
        return False
    if entry.get("state") == "UNAVAILABLE":
        return (
            set(entry) == common | {"provider_code"}
            and type(entry["provider_code"]) is int and entry["provider_code"] == 2304
        )
    if (
        entry.get("state") != "AVAILABLE"
        or set(entry) != common | {"currency", "basis", "prices"}
        or entry.get("basis") != "NATIVE_CLEAN_PER_100"
        or not _CURRENCY.fullmatch(str(entry.get("currency") or ""))
    ):
        return False
    prices = entry.get("prices")
    if not isinstance(prices, list) or not 0 < len(prices) <= MAX_POINTS:
        return False
    previous = None
    gaps = []
    settled_before = _instant(entry["fetched_at"]).astimezone(ZoneInfo("Europe/Rome")).date()
    for point in prices:
        if not isinstance(point, dict) or set(point) != {"date", "close"}:
            return False
        text = point.get("date")
        if not isinstance(text, str) or not _DATE.fullmatch(text):
            return False
        try:
            day = date.fromisoformat(text)
        except ValueError:
            return False
        price = point["close"]
        if (
            _positive_price(price) is None
            or (previous is not None and day <= previous)
            or day >= settled_before
        ):
            return False
        if previous is not None:
            gaps.append((day - previous).days)
        previous = day
    # Aggregated monthly/weekly chart closes can be labelled at the start of
    # their bucket. They cannot be imported as daily, causal observations.
    return len(prices) >= 3 and statistics.median(gaps) <= 4


def parse_packet(packet, isin: str, *, closed_before: date | None = None):
    """Validate identity/units and retain the actual acquisition timestamp.

    A live download may finish after the run began, even across local midnight.
    Its receipt is not the market date: an optional run cutoff excludes closes
    that were not completed on the run's Milan calendar.
    """
    if not isinstance(packet, dict) or packet.get("exchange") != _EXCHANGE:
        return None
    observed = _instant(packet.get("fetched_at"))
    body = packet.get("payload")
    if observed is None or not isinstance(body, dict) or not _ISIN.fullmatch(isin):
        return None
    settled_before = observed.astimezone(ZoneInfo("Europe/Rome")).date()
    if closed_before is not None:
        settled_before = min(settled_before, closed_before)
    common = {
        "schema_version": SCHEMA_VERSION, "source": SOURCE, "isin": isin,
        "exchange": _EXCHANGE, "fetched_at": observed.isoformat(),
    }
    if (
        body.get("entityID") != "euronext" or body.get("period") != "5Y"
        or body.get("adjustement") is not False or body.get("addDayLastPrice") is not False
    ):
        return None
    if type(body.get("status")) is int and body["status"] == 2304:
        return {**common, "state": "UNAVAILABLE", "provider_code": 2304}
    transco, history = body.get("transco"), body.get("history")
    if (
        type(body.get("status")) is not int or body["status"] != 0
        or not isinstance(transco, dict)
        or transco.get("code") != isin or transco.get("codification") != "ISIN"
        or transco.get("exchCode") != _EXCHANGE
        or not isinstance(history, dict) or history.get("instrType") != "BOND"
        or not _CURRENCY.fullmatch(str(history.get("currency") or ""))
    ):
        return None
    points = history.get("historyDt")
    if not isinstance(points, list) or not 0 < len(points) <= MAX_POINTS:
        return None
    prices = {}
    for point in points:
        if not isinstance(point, dict):
            continue
        text, price = point.get("dt"), point.get("closePx")
        if not isinstance(text, str) or not re.fullmatch(r"[0-9]{8}", text):
            continue
        try:
            day = datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            continue
        price = _positive_price(price)
        if price is None:
            continue
        if day >= settled_before:
            continue
        if day in prices and prices[day] != price:
            return None
        prices[day] = price
    entry = {
        **common, "state": "AVAILABLE", "currency": history["currency"],
        "basis": "NATIVE_CLEAN_PER_100",
        "prices": [{"date": day.isoformat(), "close": price}
                   for day, price in sorted(prices.items())],
    }
    return entry if cache_entry_valid(entry, isin) else None


def _native_identity_authorized(isin, captured_at):
    from tarzan.data import price_cache

    profile = price_cache.load_instrument_profile(isin, as_of=captured_at, allow_stale=True)
    provenance = profile.get("provenance") if isinstance(profile, dict) else None
    observed = _instant(profile.get("observed_at")) if isinstance(profile, dict) else None
    return bool(
        isinstance(profile, dict) and profile.get("isin") == isin
        and profile.get("kind") == "BOND" and profile.get("status") == "VERIFIED"
        and profile.get("source") == "OpenFIGI" and profile.get("confidence") == "HIGH"
        and observed is not None and observed <= captured_at and isinstance(provenance, dict)
        and provenance.get("provider") == "OpenFIGI" and provenance.get("idType") == "ID_ISIN"
    )


def raw_history(isin: str, since: date, until: date):
    """Reuse positive history and exact run-owned packets, never old absences.

    The native port always returns the same full five-year packet, regardless
    of the caller's window. Coalesce that packet per ISIN/captured run, so a
    completed absence can be shared within one run without suppressing recovery
    in the next. Transport errors and malformed packets remain retryable.
    """
    from tarzan import runtime
    from tarzan.data import price_cache
    from tarzan.runtime.memo import memoized
    from tarzan.runtime.session import current_session
    from tarzan.runtime.transport import current_budget

    context = runtime.context()
    if context.captured_at.tzinfo is None or not _ISIN.fullmatch(str(isin)):
        return None

    def load():
        try:
            cached = price_cache.load_timed(NAMESPACE, isin, ttl_days=36525)
        except (OSError, TypeError, ValueError):
            cached = None
        if not cache_entry_valid(cached, isin):
            cached = None
        if cached is not None and _instant(cached["fetched_at"]) > context.captured_at:
            cached = None
        if cached is not None:
            if cached["state"] == "UNAVAILABLE":
                # An old structured absence is not current price evidence.
                cached = None
            elif (
                date.fromisoformat(cached["prices"][0]["date"]) <= since
                and date.fromisoformat(cached["prices"][-1]["date"]) >= until
            ):
                return cached
        if (
            not runtime.allows_live_transport() or not context.allows_live_transport
            or context.captured_at < _KNOWN_AT or current_session() is None
            or not _native_identity_authorized(isin, context.captured_at)
        ):
            return cached
        def fetch_packet():
            budget = current_budget()
            if budget is None or not budget.optional_request_allowed("borsa"):
                return None
            try:
                with budget.operation("borsa"):
                    packet = _request_raw(isin)
                    acquired_at = datetime.now(timezone.utc)
                    result = parse_packet(
                        packet, isin,
                        closed_before=context.captured_at.astimezone(ZoneInfo("Europe/Rome")).date(),
                    )
                # Normal downloads are observed after run start. Reject only
                # impossible future receipts, never backdate real observations.
                if result is not None and _instant(result["fetched_at"]) > acquired_at:
                    return None
                return result
            except Exception:
                return None  # no raw SDK/access/error text enters diagnostics

        fresh = memoized(
            "bond_history.packet", (isin, context.captured_at), fetch_packet,
            legacy={}, cache_if=lambda value: value is not None,
        )
        if fresh is None:
            return cached
        if fresh["state"] == "UNAVAILABLE":
            return cached
        if cached is not None:
            if cached["currency"] != fresh["currency"]:
                return cached  # native rulers must never be merged across currencies
            points = {point["date"]: point["close"] for point in cached["prices"]}
            points.update({point["date"]: point["close"] for point in fresh["prices"]})
            fresh["prices"] = [
                {"date": day, "close": price} for day, price in sorted(points.items())
            ]
            if not cache_entry_valid(fresh, isin):
                return cached
        price_cache.store_timed(NAMESPACE, isin, fresh)
        return fresh

    if current_session() is None:
        return load()
    return memoized(
        "bond_history.raw", (isin, since, until, context.captured_at), load, legacy={},
        cache_if=lambda value: value is not None,
    )


def eur_per_nominal(entry, *, as_of: date, fx=None) -> pd.Series:
    """Daily label alignment, causal FX, and precisely one per-100 conversion."""
    if not isinstance(entry, dict) or not cache_entry_valid(entry, entry.get("isin", "")):
        return pd.Series(dtype=float)
    if entry["state"] != "AVAILABLE":
        return pd.Series(dtype=float)
    points = entry["prices"]
    prices = pd.Series(
        [point["close"] for point in points],
        index=pd.to_datetime([point["date"] for point in points]), dtype=float,
    )
    prices = prices[prices.index.date <= as_of]
    if prices.empty:
        return pd.Series(dtype=float)
    observations = pd.Series(prices.index, index=prices.index)
    if entry["currency"] != "EUR":
        from tarzan.data import enricher

        fx = enricher._usable_fx_series(
            enricher._get_fx_series(entry["currency"]) if fx is None else fx,
        ).copy()
        if fx.empty or not isinstance(fx.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        fx.index = fx.index.tz_localize(None).normalize()
        fx = fx[fx.index.date <= as_of].sort_index()
        fx = fx[~fx.index.duplicated(keep="last")]
        if fx.empty:
            return pd.Series(dtype=float)
        # Security and currency calendars are independent. A new FX rate
        # revalues the last known native close even on a venue holiday or
        # a day without a bond trade; it must not renew that close's age.
        days = prices.index.union(fx.index).sort_values()
        days = days[days >= prices.index[0]]
        dated = pd.Series(days, index=days)
        native_observations = observations.reindex(days, method="ffill")
        fx_observations = pd.Series(fx.index, index=fx.index).reindex(days, method="ffill")
        native_age = (dated - native_observations).dt.days
        fx_age = (dated - fx_observations).dt.days
        valid = native_age.between(0, MAX_CARRY_DAYS) & fx_age.between(0, MAX_CARRY_DAYS)
        prices = (prices.reindex(days, method="ffill") * fx.reindex(days, method="ffill"))[valid].dropna()
        observations = native_observations.where(
            native_observations <= fx_observations, fx_observations,
        ).reindex(prices.index)
    prices = prices / 100.0
    prices = prices[prices.map(lambda value: math.isfinite(value) and value > 0)]
    prices.attrs = {
        SOURCE_ATTR: SOURCE, BASIS_ATTR: "EUR_PER_NOMINAL", ISIN_ATTR: entry["isin"],
        OBSERVATIONS_ATTR: {
            index.date().isoformat(): pd.Timestamp(observations.loc[index]).date().isoformat()
            for index in prices.index
        },
    }
    return prices


def price_on(series, isin: str, day: date):
    """No future/backfilled observation and no arbitrarily stale market carry."""
    if (
        series is None or series.empty or not isinstance(series.index, pd.DatetimeIndex)
        or not series.index.is_monotonic_increasing or not series.index.is_unique
        or series.attrs.get(SOURCE_ATTR) != SOURCE
        or series.attrs.get(BASIS_ATTR) != "EUR_PER_NOMINAL"
        or series.attrs.get(ISIN_ATTR) != isin
    ):
        return None
    timestamp = pd.Timestamp(day)
    if series.index.tz is not None:
        timestamp = timestamp.tz_localize(series.index.tz)
    position = series.index.searchsorted(timestamp, side="right") - 1
    if position < 0:
        return None
    index = series.index[position]
    observations = series.attrs.get(OBSERVATIONS_ATTR)
    if not isinstance(observations, dict):
        return None
    try:
        observed = date.fromisoformat(observations[index.date().isoformat()])
        value = float(series.iloc[position])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if (
        math.isfinite(value) and value > 0 and observed <= index.date()
        and 0 <= (day - observed).days <= MAX_CARRY_DAYS
    ) else None


def hydrate_historical_bonds(holdings, windows, *, as_of: date) -> tuple[str, ...]:
    """One preprocessing boundary; no current prices, identities or orders change."""
    from tarzan import runtime
    from tarzan.instruments.registry import InstrumentKind
    from tarzan.instruments.valuation import holding_kind_resolution
    from tarzan.runtime.transport import current_budget

    captured_at = runtime.context().captured_at
    if captured_at.tzinfo is None or captured_at < _KNOWN_AT:
        return ()
    wanted = []
    for holding in holdings:
        if (
            not holding.is_historical_only or holding.isin not in windows
            or holding_kind_resolution(holding).kind is not InstrumentKind.BOND
            or not _ISIN.fullmatch(str(holding.isin))
        ):
            continue
        existing = holding.price_history_price_only
        if existing is not None and not existing.empty:
            numeric = pd.to_numeric(existing, errors="coerce")
            if (numeric.gt(0) & numeric.lt(float("inf"))).any():
                continue
        wanted.append(holding)
    if not wanted:
        return ()

    def prepare(holding):
        try:
            raw = raw_history(holding.isin, *windows[holding.isin])
            if raw is None or raw.get("state") != "AVAILABLE" or raw["currency"] != holding.currency:
                return holding, None
            prices = eur_per_nominal(raw, as_of=as_of)
            return holding, prices if not prices.empty else None
        except Exception:
            # A supplementary provider must not erase otherwise computable
            # history. Offline UnrecordedInput is a BaseException and still
            # escapes, so missing frozen evidence cannot be hidden here.
            return holding, None

    from contextlib import nullcontext

    budget = current_budget()
    scope = budget.optional_work("historical-bond-prices") if budget is not None else nullcontext()
    with scope, ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(copy_context().run, prepare, holding) for holding in wanted]
        prepared = [future.result() for future in futures]
    selected = []
    for holding, prices in prepared:
        if prices is not None:
            holding.price_history_price_only = prices
            selected.append(holding.isin)
    return tuple(selected)
