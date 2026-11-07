"""Index futures and government-bond yields Yahoo does not carry.

Yahoo only serves CME index futures and US Treasury yields. For everything else
(Eurex/ICE/Osaka/SGX/ASX futures, European and Japanese yields) the strip reads
three public, keyless feeds — the same endpoints that power their own websites:

* TradingView: its scanner (one POST for every quote, with a previous close
  that is always the prior trading day's; futures: the prior settlement) and
  its chart socket (five-minute bars of every session whole, a Eurex future's
  night hours included), which draws every line first.
* CNBC (Reuters data): quotes plus a one-day intraday chart. Its quoted
  previous close for BONDS rolls to the same day's close after the European
  close, so it is never used as a bond baseline, and its daily history is
  mis-dated by a day for some venues, so that is only drawn, never measured.
* MarketWatch (Dow Jones; bonds from Tullett Prebon): a 5-minute intraday
  series with the session's real window and a DATED prior close, plus a
  correctly dated daily history. Its benchmark bonds are not always the ones
  TradingView/CNBC quote (Spain 2Y differed by 5.6 bp on 2 Nov 2026), so its
  bond levels are compared with a wider tolerance.

None of them is official, any can change or disappear, and TradingView and CNBC
carry the SAME vendor's bond yields (LSEG), so for bonds only MarketWatch is an
independent check. Hence every row is VOTED: the figures shown come from one
source and are confirmed against the others; a row no other source confirms is
marked, and a source that is down simply stops voting. Nothing here can stop
an issue: on any failure the row is omitted and the run goes on.

Measured on 2 Nov 2026 (every row, every source, same instant): futures levels
agree within 0.05% and previous settlements exactly; LSEG yields agree to
0.1 bp; Tullett vs LSEG within 2 bp on 10Y levels and 3 bp on the day's change,
except Spain 2Y (5.6 bp apart: different benchmark bonds). A rolled baseline,
the error this exists to catch, is off by the whole day's move. The tolerances
below sit between the two.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

TV, CNBC, MW = "tradingview", "cnbc", "marketwatch"
FEED_LABEL = {TV: "TradingView", CNBC: "CNBC", MW: "MarketWatch"}

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
_TIMEOUT_S = 10
# MarketWatch's own public website key (shipped in its page JavaScript), not a
# credential of ours.
_MW_KEY, _MW_TOKEN = "cecc4267a0", "cecc4267a0194af89ca343805a3e57af"

# A source whose last print is this much older than the freshest one is not
# reporting the same market (MarketWatch's Hang Seng sat three days behind).
_STALE_BEHIND = timedelta(hours=3)
# Out of session: a print older than a long weekend is from a dead feed
# (MarketWatch's Hang Seng sat three days behind).
_ANCIENT = timedelta(days=4)
# A row is "trading now" when its shown print is at most this old. Covers the
# feeds' 10-15 minute delay with room to spare.
_LIVE_WITHIN = timedelta(minutes=30)
# A drawn bar this far from both the median and the level is a bad print:
# basis points for a yield, percent for an index.
_SPIKE = {"yield": 30.0, "index": 4.5}
# Intraday series longer than this are thinned to one point per 15 minutes.
_THIN_ABOVE = 96
# A gap this long inside one feed's intraday bars separates two sessions.
_SESSION_GAP = timedelta(hours=4)
# Closed futures: settlement and last trade may differ by an evening's move.
_CLOSED_LEVEL = 2.0
# A drawn intraday line must end near the quote it sits under.
_CHART_BEHIND = timedelta(hours=2)
# Session starts further apart than this are different trading days. The same
# day's starts differ by at most a couple of hours between feeds (US 10Y: 00:00
# vs 22:00 the evening before); the next day's by about a day.
_OTHER_SESSION = timedelta(hours=12)
# (level, change) agreement. Index: % of level and percentage points of the
# day's change. Yield: basis points for both — the level band is wide because
# MarketWatch's benchmark bonds can differ from LSEG's, the change band tight
# because both measure the same day's move.
_TOLERANCE = {"index": (0.3, 0.15), "yield": (8.0, 4.0)}


@dataclass(frozen=True)
class FeedRow:
    name: str
    key: str            # stable strip key, also the flag key
    category: str       # "Futures" | "Bonds"
    unit: str           # "index" | "yield"
    symbols: dict       # feed -> that feed's symbol
    # (quote feed, chart feed) in preference order: the quote feed supplies the
    # level and the previous close, the chart feed the drawn line. The first
    # recipe whose figures another source confirms is shown.
    recipes: tuple = ()
    hours: float = 22.0  # session length, for the elapsed-time chart axis
    # The exchange's trading windows, local wall clock ("HH:MM", "HH:MM"); an end
    # before its start runs into the next morning. With ``tz``, they caption the
    # row and decide open/closed. ``calendar`` is a cash ticker on the same
    # exchange calendar, for holidays.
    sessions: tuple = ()
    tz: str = ""
    calendar: str = ""


def _future(name, key, hours=22.0, *, sessions=(), tz="", calendar="", **symbols):
    return FeedRow(name, key, "Futures", "index", symbols,
                   ((MW, MW), (CNBC, CNBC), (TV, MW), (TV, CNBC), (TV, None)), hours,
                   sessions, tz, calendar)


# Official trading hours, exchange-local.
_EUREX = dict(sessions=(("01:10", "22:00"),), tz="Europe/Berlin", calendar="^GDAXI")


def _bond(name, key, hours=10.0, **symbols):
    # LSEG levels first (what CNBC/Investing/Bloomberg-style pages print),
    # drawn with CNBC's chart of the same feed; MarketWatch's own series when
    # TradingView is unavailable.
    return FeedRow(name, key, "Bonds", "yield", symbols,
                   ((TV, CNBC), (MW, MW), (TV, MW), (TV, None)), hours)


def _treasury(name, key, **symbols):
    # TradingView starts a Treasury's day at 00:00 UTC, so its "change" is
    # since midnight, not since the New York close every other site measures
    # from. MarketWatch's dated prior close is the New York close.
    return FeedRow(name, key, "Bonds", "yield", symbols,
                   ((MW, MW), (TV, CNBC), (TV, MW), (TV, None)), 24.0)


# Display order inside each category follows the cash-open order (Asia,
# Europe, US), as the rest of the strip does.
FEED_ROWS: tuple[FeedRow, ...] = (
    # Asian night sessions open the NEXT trading day (JPX, ASX24 and SGX all
    # count them that way, settlement at the day session's close): "-" marks a
    # window that starts the evening of the previous trading day.
    _future("ASX SPI 200", "fut:SPI", sessions=(("-17:10", "07:00"), ("09:50", "16:30")),
            tz="Australia/Sydney", calendar="^AXJO", **{TV: "ASX24:AP1!", CNBC: "@YAP.1"}),
    _future("Nikkei 225", "fut:NK", sessions=(("-17:00", "06:00"), ("08:45", "15:45")),
            tz="Asia/Tokyo", calendar="^N225", **{TV: "OSE:NK2251!", CNBC: "@JNI.1"}),
    _future("FTSE China A50", "fut:A50", sessions=(("-17:15", "05:15"), ("09:00", "16:35")),
            tz="Asia/Singapore", **{TV: "SGX:CN1!", CNBC: "@SFC.1"}),
    _future("Euro Stoxx 50", "fut:FESX", hours=14.2, **_EUREX,
            **{TV: "EUREX:FESX1!", MW: "FUTURE/DE/XEUR/FESX00"}),
    _future("Stoxx 600", "fut:FXXP", hours=14.2, **_EUREX,
            **{TV: "EUREX:FXXP1!", MW: "FUTURE/DE/XEUR/FXXP00"}),
    _future("DAX", "fut:FDAX", hours=14.2, **_EUREX,
            **{TV: "EUREX:FDAX1!", MW: "FUTURE/DE/XEUR/DAX00"}),
    _future("CAC 40", "fut:FCE", hours=14.0, sessions=(("08:00", "22:00"),),
            tz="Europe/Paris", calendar="^FCHI",
            **{TV: "EURONEXT:FCE1!", MW: "FUTURE/FR//FCE00"}),
    _future("FTSE 100", "fut:Z", hours=20.0, sessions=(("01:00", "21:00"),),
            tz="Europe/London", calendar="^FTSE",
            **{TV: "ICEEUR:Z1!", CNBC: "@FFI.1", MW: "FUTURE/UK/IFEU/Z00"}),
    _treasury("US 2Y", "bond:US2Y",
              **{TV: "TVC:US02Y", CNBC: "US2Y", MW: "BOND/BX//TMUBMUSD02Y"}),
    _treasury("US 10Y", "bond:US10Y",
              **{TV: "TVC:US10Y", CNBC: "US10Y", MW: "BOND/BX//TMUBMUSD10Y"}),
    _treasury("US 30Y", "bond:US30Y",
              **{TV: "TVC:US30Y", CNBC: "US30Y", MW: "BOND/BX//TMUBMUSD30Y"}),
    _bond("Bund 2Y", "bond:DE2Y",
          **{TV: "TVC:DE02Y", CNBC: "DE2Y", MW: "BOND/BX//TMBMKDE-02Y"}),
    _bond("Bund 10Y", "bond:DE10Y",
          **{TV: "TVC:DE10Y", CNBC: "DE10Y", MW: "BOND/BX//TMBMKDE-10Y"}),
    _bond("BTP 2Y", "bond:IT2Y",
          **{TV: "TVC:IT02Y", CNBC: "IT2Y", MW: "BOND/BX//TMBMKIT-02Y"}),
    _bond("BTP 10Y", "bond:IT10Y",
          **{TV: "TVC:IT10Y", CNBC: "IT10Y", MW: "BOND/BX//TMBMKIT-10Y"}),
    _bond("OAT 10Y", "bond:FR10Y",
          **{TV: "TVC:FR10Y", CNBC: "FR10Y", MW: "BOND/BX//TMBMKFR-10Y"}),
    _bond("Bonos 2Y", "bond:ES2Y",
          **{TV: "TVC:ES02Y", CNBC: "ES2Y", MW: "BOND/BX//TMBMKES-02Y"}),
    _bond("Bonos 10Y", "bond:ES10Y",
          **{TV: "TVC:ES10Y", CNBC: "ES10Y", MW: "BOND/BX//TMBMKES-10Y"}),
    _bond("Gilt 2Y", "bond:GB2Y",
          **{TV: "TVC:GB02Y", CNBC: "UK2Y", MW: "BOND/BX//TMBMKGB-02Y"}),
    _bond("Gilt 10Y", "bond:GB10Y",
          **{TV: "TVC:GB10Y", CNBC: "UK10Y", MW: "BOND/BX//TMBMKGB-10Y"}),
    _bond("JGB 10Y", "bond:JP10Y", hours=12.0,
          **{TV: "TVC:JP10Y", CNBC: "JP10Y", MW: "BOND/BX//TMBMKJP-10Y"}),
)

# (name, key, minuend row, subtrahend row): a spread in basis points, built
# from the two rows AS SHOWN, so it can never disagree with them.
SPREADS = (("BTP–Bund 10Y", "spread:IT-DE10Y", "bond:IT10Y", "bond:DE10Y"),)

# The order the strip lists the feed rows in: each spread right after the row
# it is quoted against.
FEED_KEYS = tuple(k for r in FEED_ROWS
                  for k in (r.key, *(s[1] for s in SPREADS if s[2] == r.key)))


@dataclass(frozen=True)
class Obs:
    """One source's quote: last print, the close before it, and when.

    ``session`` is when the session this quote's day belongs to began, where
    the feed says (TradingView's daily bar, MarketWatch's trading window). Two
    feeds roll to the next session at different hours — MarketWatch opens the
    Treasuries' next day at 22:00 UTC, TradingView at midnight — and in between
    they are measuring different days, both correctly.
    """
    feed: str
    last: float
    prev: Optional[float]
    at: datetime  # tz-aware
    session: Optional[datetime] = None


@dataclass
class Chart:
    intra: object = None   # pd.Series, UTC index
    daily: object = None   # pd.Series of closes, date-like index
    span: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _request_json(provider: str, method: str, url: str, **kwargs):
    """Decoded JSON body or None; never raises (a feed must never break a run)."""
    import requests
    from tarzan.runtime.transport import provider_call

    headers = {"User-Agent": _UA, "Accept": "application/json", **kwargs.pop("headers", {})}
    try:
        r = provider_call(
            provider,
            lambda timeout: requests.request(method, url, headers=headers, timeout=timeout, **kwargs),
            timeout=_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("%s unreachable (%s)", FEED_LABEL.get(provider, provider), type(e).__name__)
        return None
    try:
        if not r.ok:
            return None
        return r.json()
    except ValueError:
        return None
    finally:
        with suppress(Exception):
            r.close()


def _number(value) -> Optional[float]:
    """A finite float from a number or a formatted string ("4.41%", "10,794.50")."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "").rstrip("%")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _plausible(value: Optional[float], unit: str) -> Optional[float]:
    """Positive for an index level; a yield may be zero or negative."""
    if value is None:
        return None
    if unit == "index":
        return value if value > 0 else None
    return value if -5.0 < value < 50.0 else None


# ---------------------------------------------------------------------------
# Parsers (one per feed). Each returns {symbol: Obs} or a Chart, never raises.
# ---------------------------------------------------------------------------

def _tradingview_quotes(symbols: list[str], units: dict) -> dict:
    if not symbols:
        return {}
    body = _request_json(TV, "POST", "https://scanner.tradingview.com/global/scan",
                         data=json.dumps({"symbols": {"tickers": symbols},
                                          "columns": ["close", "change_abs", "update_time", "time"]}),
                         headers={"Content-Type": "application/json"})
    out = {}
    try:
        for row in body["data"]:
            sym, vals = row["s"], row["d"]
            unit = units.get(sym, "index")
            last = _plausible(_number(vals[0]), unit)
            chg, ts = _number(vals[1]), _number(vals[2])
            if last is None or chg is None or ts is None:
                continue
            prev = _plausible(last - chg, unit)
            bar = _number(vals[3]) if len(vals) > 3 else None
            out[sym] = Obs(TV, last, prev, datetime.fromtimestamp(ts, timezone.utc),
                           datetime.fromtimestamp(bar, timezone.utc) if bar else None)
    except Exception as e:  # noqa: BLE001
        logger.info("TradingView answer unreadable (%s)", type(e).__name__)
    return out


def _cnbc_quotes(symbols: list[str], units: dict) -> dict:
    if not symbols:
        return {}
    from urllib.parse import urlencode

    url = ("https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?"
           + urlencode({"symbols": "|".join(symbols), "requestMethod": "itv", "noform": 1,
                        "partnerId": 2, "fund": 1, "exthrs": 1, "output": "json"}))
    body = _request_json(CNBC, "GET", url)
    out = {}
    try:
        quotes = body["FormattedQuoteResult"]["FormattedQuote"]
        for q in quotes if isinstance(quotes, list) else [quotes]:
            sym, unit = q.get("symbol"), units.get(q.get("symbol"), "index")
            if q.get("code") != 0 or sym not in units:
                continue
            last = _plausible(_number(q.get("last")), unit)
            try:
                at = datetime.fromisoformat(str(q.get("last_time")))
            except ValueError:
                continue
            if last is None or at.tzinfo is None:
                continue
            # CNBC rolls a bond's previous close to the same day's close after
            # the European close (Bund 10Y read +0.08 bp at 18:08 CET on a
            # -3.1 bp day), so it is no baseline for yields. Its futures
            # previous close is the prior settlement, matching every source.
            prev = (None if unit == "yield"
                    else _plausible(_number(q.get("previous_day_closing")), unit))
            out[sym] = Obs(CNBC, last, prev, at)
    except Exception as e:  # noqa: BLE001
        logger.info("CNBC answer unreadable (%s)", type(e).__name__)
    return out


def _cnbc_chart(symbol: str, unit: str) -> Chart:
    import pandas as pd

    chart = Chart()
    for frame in ("1D", "1M"):
        body = _request_json(CNBC, "GET",
                             f"https://ts-api.cnbc.com/harmony/app/charts/{frame}.json",
                             params={"symbol": symbol})
        try:
            bars = body["barData"]["priceBars"]
            pts = {}
            for b in bars:
                close = _plausible(_number(b.get("close")), unit)
                ms = _number(b.get("tradeTimeinMills"))
                if close is not None and ms is not None:
                    pts[pd.Timestamp(int(ms), unit="ms", tz="UTC")] = close
            if len(pts) < 2:
                continue
            ser = pd.Series(pts).sort_index()
            if frame == "1D":
                chart.intra = ser
            else:
                # Drawn only: CNBC dates some venues' daily bars a day late.
                chart.daily = ser
        except Exception:  # noqa: BLE001
            continue
    return chart


# TradingView's chart feed is a websocket, the one its own charts read: the only
# free intraday source holding a Eurex future's whole day. MarketWatch's starts
# at 07:30 CET, CNBC carries no Eurex contract, so every 08:00 issue drew the DAX
# from nothing although the contract had traded since 01:10.
_TV_WS = "wss://data.tradingview.com/socket.io/websocket"
_TV_BARS = 300  # five-minute bars: 25 hours, a whole session even on a Saturday


def _tv_frame(payload: dict) -> str:
    text = json.dumps(payload, separators=(",", ":"))
    return f"~m~{len(text)}~m~{text}"


def _tv_split(raw: str) -> list[str]:
    """The socket.io-style envelope: ``~m~<length>~m~<payload>`` repeated."""
    import re

    out = []
    while raw:
        m = re.match(r"~m~(\d+)~m~", raw)
        if not m:
            break
        n, start = int(m.group(1)), m.end()
        out.append(raw[start:start + n])
        raw = raw[start + n:]
    return out


def _tv_read_bars(ws, symbols: list[str], deadline: float, clock=None) -> dict:
    """``{symbol: [(epoch seconds, close), ...]}`` read from an open chart socket.

    One chart session per symbol. Heartbeats (``~h~N``) are echoed or the server
    hangs up. A symbol the feed rejects is simply absent. Stops when every
    symbol answered or at ``deadline`` (``clock()`` seconds).
    """
    import random
    import string
    import time

    clock = clock or time.monotonic
    sessions = {}
    ws.send(_tv_frame({"m": "set_auth_token", "p": ["unauthorized_user_token"]}))
    for i, sym in enumerate(symbols):
        cs = "cs_" + "".join(random.choices(string.ascii_lowercase, k=12))
        sessions[cs] = sym
        ws.send(_tv_frame({"m": "chart_create_session", "p": [cs, ""]}))
        ws.send(_tv_frame({"m": "resolve_symbol", "p": [
            cs, "sym", "=" + json.dumps({"symbol": sym, "adjustment": "splits",
                                         "session": "extended"})]}))
        ws.send(_tv_frame({"m": "create_series", "p": [cs, "s", "s", "sym", "5", _TV_BARS, ""]}))
    bars, done = {}, set()
    while len(done) < len(sessions):
        left = deadline - clock()
        if left <= 0:
            break
        try:
            raw = ws.recv(timeout=left)
        except TimeoutError:
            break
        for frame in _tv_split(raw if isinstance(raw, str) else raw.decode()):
            if frame.startswith("~h~"):
                ws.send(f"~m~{len(frame)}~m~{frame}")
                continue
            try:
                msg = json.loads(frame)
            except ValueError:
                continue
            head, args = msg.get("m"), msg.get("p") or []
            cs = args[0] if args else None
            if cs not in sessions:
                continue
            if head in ("symbol_error", "series_error", "critical_error"):
                done.add(cs)
            elif head == "timescale_update":
                try:
                    rows = args[1]["s"]["s"]
                    bars[sessions[cs]] = [(float(r["v"][0]), r["v"][4]) for r in rows]
                except (KeyError, IndexError, TypeError):
                    pass
                done.add(cs)
    return bars


def _tv_connect(timeout: float):
    """The one network seam of the chart socket (the suite stubs it)."""
    from websockets.sync.client import connect

    return connect(_TV_WS, origin="https://www.tradingview.com", user_agent_header=_UA,
                   open_timeout=timeout, close_timeout=2)


def _tradingview_charts(symbols: list[str], units: dict) -> dict:
    """``{symbol: pd.Series of closes (UTC)}`` for ``symbols``, one socket for
    all of them, bounded by the provider budget. Empty on any failure."""
    import time

    import pandas as pd
    from tarzan.runtime.transport import provider_call

    if not symbols:
        return {}

    def fetch(timeout):
        with _tv_connect(timeout) as ws:
            return _tv_read_bars(ws, symbols, time.monotonic() + timeout)

    try:
        raw = provider_call(TV, fetch, timeout=_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — a chart must never break a run
        logger.info("TradingView charts unreachable (%s)", type(e).__name__)
        return {}
    out = {}
    for sym, rows in (raw or {}).items():
        pts = {}
        for ts, close in rows:
            value = _plausible(_number(close), units.get(sym, "index"))
            if value is not None and math.isfinite(ts):
                pts[pd.Timestamp(int(ts), unit="s", tz="UTC")] = value
        if len(pts) >= 2:
            out[sym] = pd.Series(pts).sort_index()
    return out


def _mw_series(key: str, step: str, frame: str):
    from urllib.parse import urlencode

    query = {"Step": step, "TimeFrame": frame, "EntitlementToken": _MW_TOKEN,
             "IncludeMockTick": False, "FilterNullSlots": False, "FilterClosedPoints": True,
             "IncludeClosedSlots": False, "IncludeOfficialClose": True, "InjectOpen": False,
             # Pre-market: Eurex trades from 01:10 CET but MarketWatch's regular
             # day starts at 07:50; its pre-market bars from 07:30 are all an 08:00
             # issue can draw (the night hours are on none of the three feeds).
             "ShowPreMarket": True, "ShowAfterHours": False, "UseExtendedTimeFrame": True,
             "WantPriorClose": True, "IncludeCurrentQuotes": False,
             "ResetTodaysAfterHoursPercentChange": False,
             "Series": [{"Key": key, "Dialect": "Charting", "Kind": "Ticker", "SeriesId": "s1",
                         "DataTypes": ["Last"], "Indicators": []}]}
    url = ("https://api.wsj.net/api/michelangelo/timeseries/history?"
           + urlencode({"json": json.dumps(query), "ckey": _MW_KEY}))
    return _request_json(MW, "GET", url, headers={"Dylan2010.EntitlementToken": _MW_TOKEN})


def _mw_points(body, unit: str):
    """The series' (UTC timestamp, value) points and its Series record."""
    import pandas as pd

    series = body["Series"][0]
    ticks = body["TimeInfo"]["Ticks"]
    pts = {}
    for ms, point in zip(ticks, series["DataPoints"]):
        value = _plausible(_number(point[0] if point else None), unit)
        if value is not None and _number(ms) is not None:
            pts[pd.Timestamp(int(ms), unit="ms", tz="UTC")] = value
    return (pd.Series(pts).sort_index() if pts else None), series


def _marketwatch(key: str, unit: str) -> tuple[Optional[Obs], Chart]:
    """MarketWatch's quote, read from its own intraday series, plus the chart.

    The quote API's net change rolls at the settlement for some futures (FTSE,
    CAC read ~0% on a +0.3%/+1.2% day), but the series carries a DATED prior
    close, which is used only when it is dated before the last print's day.
    """
    import pandas as pd

    chart, obs = Chart(), None
    try:
        intra, series = _mw_points(_mw_series(key, "PT5M", "D1"), unit)
        extra = {e.get("Name"): e for e in series.get("ExtraData") or [] if isinstance(e, dict)}
        windows = [s for s in series.get("MarketSessions") or []
                   if isinstance(s, dict) and s.get("Kind") in ("PreMarket", "Trading")
                   and _number(s.get("Start")) and _number(s.get("End"))]
        if windows:
            chart.span = (pd.Timestamp(int(windows[0]["Start"]), unit="ms", tz="UTC"),
                          pd.Timestamp(int(windows[-1]["End"]), unit="ms", tz="UTC"))
        # Only real bars date a print. On a holiday (the JGB on 3 Nov 2026)
        # the series is one point stamped at the session's END, which
        # outdated the other feeds' real prints and got them discarded as
        # stale; a session is at least two bars.
        last = at = None
        if intra is not None and len(intra) >= 2:
            last, at = float(intra.iloc[-1]), intra.index[-1].to_pydatetime()
        prior = extra.get("PriorClose") or {}
        prev = _plausible(_number(prior.get("Value")), unit)
        prior_ms = _number(prior.get("Date"))
        if prev is not None and (prior_ms is None or at is None
                                 or datetime.fromtimestamp(prior_ms / 1000, timezone.utc).date()
                                 >= at.date()):
            prev = None
        if last is not None and at is not None:
            obs = Obs(MW, last, prev, at,
                      chart.span[0].to_pydatetime() if chart.span else None)
        chart.intra = intra if intra is not None and len(intra) >= 2 else None
    except Exception as e:  # noqa: BLE001
        logger.info("MarketWatch answer unreadable (%s)", type(e).__name__)
    if obs is not None:
        try:
            closes, _ = _mw_points(_mw_series(key, "P1D", "P3M"), unit)
            chart.daily = closes if closes is not None and len(closes) >= 2 else None
        except Exception:  # noqa: BLE001
            pass
    return obs, chart


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _change(o: Obs, unit: str) -> Optional[float]:
    if o.prev is None:
        return None
    if unit == "yield":
        return (o.last - o.prev) * 100.0
    return (o.last / o.prev - 1.0) * 100.0 if o.prev else None


def _agreement(a: Obs, b: Obs, unit: str, closed: bool = False) -> tuple[bool, Optional[bool]]:
    """(levels agree, changes agree — None when either has no baseline).

    ``closed``: a future whose session is over. Its official close is the
    settlement (TradingView prints it), while MarketWatch and CNBC keep the
    last trade, which for Eurex runs on from the 17:30 settlement to 22:00, so
    the two legitimately differ by the evening's trading. Then the BASELINE is
    what must match (both measure from the same previous settlement), and the
    level only within a session's range."""
    if closed and unit == "index":
        if a.prev is None or b.prev is None:
            return abs(a.last / b.last - 1.0) * 100.0 <= _CLOSED_LEVEL, None
        return (abs(a.last / b.last - 1.0) * 100.0 <= _CLOSED_LEVEL,
                abs(a.prev / b.prev - 1.0) * 100.0 <= _TOLERANCE[unit][1])
    tol_level, tol_change = _TOLERANCE[unit]
    if unit == "yield":
        level = abs(a.last - b.last) * 100.0 <= tol_level
    else:
        level = abs(a.last / b.last - 1.0) * 100.0 <= tol_level
    ca, cb = _change(a, unit), _change(b, unit)
    if ca is None or cb is None:
        return level, None
    if unit == "yield" and abs(a.last - b.last) * 100.0 > tol_change:
        # Levels this far apart are two different benchmark bonds (LSEG's and
        # Tullett's 2Y Bund were 4.6 bp apart on 3 Nov 2026); their moves are
        # two bonds' moves, not one figure to confirm.
        return level, None
    if a.session and b.session and abs(a.session - b.session) > _OTHER_SESSION:
        return level, None  # different days' moves: nothing to compare
    return level, abs(ca - cb) <= tol_change


def verdict(shown: Obs, others: list[Obs], unit: str, closed: bool = False) -> str:
    """"verified": another source confirms the level and, where it can, the
    day's change (closed futures: the baseline); "disputed": other sources
    answered but none confirms; "single": nobody else answered."""
    if not others:
        return "single"
    checks = [_agreement(shown, o, unit, closed) for o in others]
    if not any(level for level, _ in checks):
        return "disputed"
    changes = [chg for _, chg in checks if chg is not None]
    return "verified" if not changes or any(changes) else "disputed"


def _fresh(obs: dict, row: Optional[FeedRow] = None, now: Optional[datetime] = None) -> dict:
    """Drop sources not reporting the current market.

    Print times are only evidence while a future's exchange is trading: then a
    source three hours behind the freshest one is stuck. Out of session they
    mean nothing — CNBC stamps a closed market with the current time (or, for
    the FTSE, tonight's 22:00), TradingView with the settlement's publication —
    and comparing them dropped TradingView and MarketWatch from every European
    bond on Saturdays, leaving CNBC alone with no usable baseline and the rows
    gone. Then only a print older than a long weekend is stale; a stuck feed
    otherwise shows in its LEVEL, which the vote weighs.
    """
    if not obs:
        return obs
    if now is None:
        newest = max(o.at for o in obs.values())
        return {f: o for f, o in obs.items() if o.at >= newest - _STALE_BEHIND}
    timetable = _timetable(row, now) if row is not None else None
    at = {f: min(o.at, now) for f, o in obs.items()}
    if timetable is not None and timetable[1]:
        newest = max(at.values())
        return {f: o for f, o in obs.items() if at[f] >= newest - _STALE_BEHIND}
    if timetable is not None and timetable[3] is not None:
        # A closed future: current if it printed during the session just ended.
        return {f: o for f, o in obs.items() if at[f] >= timetable[3]}
    return {f: o for f, o in obs.items() if o.at >= now - _ANCIENT}


def choose(row: FeedRow, obs: dict, closed: bool = False) -> Optional[tuple]:
    """(quote Obs, chart feed, verdict) for the first recipe another source
    confirms, else the first that can be shown at all; None when no recipe has
    a level AND a previous close.

    ``closed`` (a future after its session): TradingView's settlement-to-
    settlement figure first, the exchange's official day."""
    first = None
    recipes = (sorted(row.recipes, key=lambda r: r[0] != TV)
               if closed and row.unit == "index" else row.recipes)
    for quote_feed, chart_feed in recipes:
        shown = obs.get(quote_feed)
        if shown is None or shown.prev is None:
            continue
        if chart_feed is not None and chart_feed not in row.symbols:
            continue
        if chart_feed == MW and MW not in obs:
            continue  # MarketWatch had no bars to draw (FTSE on a Saturday)
        others = [o for f, o in obs.items() if f != quote_feed]
        v = verdict(shown, others, row.unit, closed)
        if v != "disputed":
            return shown, chart_feed, v
        first = first or (shown, chart_feed, v)
    return first


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _parallel(calls: list):
    """Run (fn, *args) calls on a small pool, each in its own copy of the run
    context (the provider budget and run clock live in context variables)."""
    if not calls:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
        futures = [pool.submit(copy_context().run, fn, *args) for fn, *args in calls]
        out = []
        for fut in futures:
            try:
                out.append(fut.result())
            except Exception:  # noqa: BLE001
                out.append(None)
        return out


def _session_only(ser, start, last, unit):
    """One session's bars, without isolated bad prints.

    ``start`` is the session start a feed states. Without it, a continuous
    market's "1D" chart from CNBC runs from the previous session (36 hours of
    Asian futures, two days of Treasuries). A bar far from BOTH the series'
    median and the shown level is a bad print, not a move: CNBC's US 2Y
    carried one 5.026% bar in a 4.59% day, which flattened the whole line.
    """
    if ser is None:
        return None
    if start is not None:
        ser = ser[ser.index >= start - timedelta(minutes=1)]
    if len(ser) < 2:
        return None
    far = _SPIKE[unit]
    scale = 100.0 if unit == "yield" else 100.0 / max(abs(float(ser.median())), 1e-9)
    med = float(ser.median())
    keep = ((ser - med).abs() * scale <= far) | ((ser - last).abs() * scale <= far)
    ser = ser[keep]
    return ser if len(ser) >= 2 else None


def _clip(ser, now, step=None, anchor=None):
    """Nothing after the run's reference instant; optionally one point per
    ``step`` (CNBC draws one-minute bars: a thousand points, four times over in
    the SVG, would cost ~40 KB a row of an email Gmail clips at 102 KB)."""
    if ser is None:
        return None
    try:
        import pandas as pd

        ser = ser[ser.index <= pd.Timestamp(now)]
        if step is not None:
            # One session: whatever precedes the last overnight-sized hole is
            # another day's stray print (CNBC opens a gilt's day at 22:33 the
            # evening before), which would draw the empty night as a flat line.
            # Of the stretches between such holes, the one the shown quote
            # belongs to: the last that began by ``anchor``. On a Saturday
            # CNBC's "1D" runs from Thursday to a few stray Saturday prints,
            # and the last stretch alone was too short to draw.
            holes = ser.index.to_series().diff() > _SESSION_GAP
            starts = [ser.index[0], *holes[holes].index]
            if anchor is not None:
                began = [t for t in starts if t <= pd.Timestamp(anchor)]
                first = began[-1] if began else starts[0]
            else:
                first = starts[-1]
            later = [t for t in starts if t > first]
            ser = ser[(ser.index >= first) & ((ser.index < later[0]) if later else True)]
            # The last REAL observation of each bucket, at its own time: a
            # bucket label could post-date the run's reference instant. Only a
            # long series is thinned: half an hour of five-minute bars at 08:00
            # would otherwise shrink to two points.
            if len(ser) > _THIN_ABOVE:
                ser = ser.groupby(ser.index.floor(step)).tail(1)
        return ser if len(ser) >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def schedule(row: FeedRow, now: datetime):
    """``(caption, open now, session weekday)`` from the row's trading windows,
    or None when the row has none (bonds). A window belongs to the day it
    starts on: an evening session that runs past midnight is that day's.
    Holidays come from ``row.calendar``'s exchange; without one (SGX) only
    weekends close it."""
    timetable = _timetable(row, now)
    return timetable[:3] if timetable else None


def _timetable(row: FeedRow, now: datetime):
    """:func:`schedule` plus the start and end of the trading day the latest
    (or current) window belongs to, and that window itself.

    A trading day is a weekday its exchange is open (``row.calendar``). Its
    windows are on that date, except one marked "-", which opens the evening of
    the PREVIOUS trading day (Osaka's Friday-evening session belongs to Monday).
    The weekday shown is the calendar day the current or latest window began.
    """
    if not row.sessions or not row.tz:
        return None
    from datetime import time as dtime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(row.tz)
    local = now.astimezone(tz)

    def trading_day(day):
        if day.weekday() >= 5:
            return False
        if not row.calendar:
            return True
        from tarzan.data.exchange_calendar import is_session
        return is_session(row.calendar, day)

    def previous_trading_day(day):
        day -= timedelta(days=1)
        while not trading_day(day):
            day -= timedelta(days=1)
        return day

    def windows(day):
        out = []
        for start, end in row.sessions:
            base = previous_trading_day(day) if start.startswith("-") else day
            a = datetime.combine(base, dtime.fromisoformat(start.lstrip("-")), tzinfo=tz)
            b = datetime.combine(base, dtime.fromisoformat(end), tzinfo=tz)
            out.append((a, b if b > a else b + timedelta(days=1)))
        return out

    from tarzan.data.market_quotes import reader_hours

    by_clock = sorted(row.sessions, key=lambda w: w[0].lstrip("-"))
    caption = reader_hours(row.tz, [
        tuple(tuple(int(x) for x in t.lstrip("-").split(":")) for t in w) for w in by_clock
    ], day=local.date())
    latest = None  # (window start, its trading day) of the latest window begun
    for ahead in range(4, -9, -1):
        day = local.date() + timedelta(days=ahead)
        if not trading_day(day):
            continue
        spans = windows(day)
        for a, b in spans:
            if a <= local < b:
                return (caption, True, f"{a:%a}",
                        min(x for x, _ in spans), max(y for _, y in spans), (a, b))
            if a <= local and (latest is None or a > latest[0]):
                latest = (a, day, b)
    if latest is None:
        return caption, False, "", None, None, None
    spans = windows(latest[1])
    return (caption, False, f"{latest[0]:%a}",
            min(x for x, _ in spans), max(y for _, y in spans), (latest[0], latest[2]))


def _row_dict(row: FeedRow, shown: Obs, chart: Chart, status: str, others: list, now) -> dict:
    import pandas as pd

    unit = row.unit
    timetable = _timetable(row, now)
    window = timetable[5] if timetable else None
    # A future draws the session in progress, or the one just ended: Osaka's
    # trading day spans Friday evening to Monday afternoon, an axis no line
    # can be read on. Its % is still the trading day's, from the settlement.
    start = (window[0] if window else
             shown.session or next((o.session for o in others if o.session), None))
    if start is None and chart.span:
        start = chart.span[0].to_pydatetime()
    intra = _session_only(_clip(chart.intra, now), start, shown.last, unit)
    intra = _clip(intra, now, "15min", anchor=shown.at)
    if intra is not None:
        last_bar = intra.index[-1].to_pydatetime()
        if window is not None and not timetable[1]:
            # Closed: the shown print may carry the settlement's publication
            # time, hours after the last trade; the chart only has to be of the
            # session that just ended.
            wrong_session = last_bar < window[0]
        elif (now - shown.at) <= _LIVE_WITHIN:
            wrong_session = abs(last_bar - shown.at) > _CHART_BEHIND
        else:
            # Out of session (a bond on a Saturday): the line ends at the last
            # trade, the quote may be stamped hours later (Bund 2Y: 18:00 and
            # 21:00); it only has to be the same trading day.
            wrong_session = abs(last_bar - shown.at) > _OTHER_SESSION
        if wrong_session:
            # The chart is not of the quote's session (a feed rolled early or
            # late): draw nothing rather than another session under this level.
            intra = None
    daily = _clip(chart.daily, now)
    spark = ([float(x) for x in intra.values] if intra is not None
             else [float(x) for x in daily.values[-40:]] if daily is not None else [])
    delta = _change(shown, unit)
    live = timetable[1] if timetable else (now - shown.at) <= _LIVE_WITHIN
    span = chart.span
    if window is not None:
        # The exchange's own session (01:10-22:00 for Eurex), so the line
        # grows across the real session whichever feed drew it.
        span = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
    if intra is not None and (span is None or intra.index[0] < span[0] or intra.index[-1] > span[1]):
        # No session window from the feed (or one the bars overrun): the axis
        # runs from the first bar for one session length while the market
        # trades, and over exactly the bars once it is closed, but never ends
        # before the last bar. CNBC's gilt day opens with a lone print at 22:33
        # the evening before; a fixed-length axis from there piled the whole
        # day's moves onto the right edge as a cliff.
        t0, t1 = intra.index[0], intra.index[-1]
        span = (t0, max(t0 + pd.Timedelta(hours=row.hours), t1) if live else t1)
    return {
        "name": row.name, "symbol": row.key, "category": row.category,
        "feed_row": True, "unit": unit, "verdict": status, "source": shown.feed,
        "value": shown.last, "change": shown.last - shown.prev,
        "pct": None if unit == "yield" else delta,
        "bp": delta if unit == "yield" else None,
        "baseline": shown.prev, "spark": spark, "spark_series": intra,
        "span": span, "session_hours": row.hours,
        "as_of": shown.at, "live": live,
        "hours_caption": timetable[0] if timetable else None,
        "session_day": timetable[2] if timetable else None,
        "observed_day": shown.at.date(), "stale_session": False,
        "alternatives": [(o.feed, o.last, _change(o, unit)) for o in others],
    }


def _spread(name: str, key: str, a: dict, b: dict) -> Optional[dict]:
    """Minuend minus subtrahend, in basis points, from the two rows as shown."""
    import pandas as pd

    if a is None or b is None:
        return None
    level = (a["value"] - b["value"]) * 100.0
    prev = (a["baseline"] - b["baseline"]) * 100.0
    intra = None
    sa, sb = a.get("spark_series"), b.get("spark_series")
    if sa is not None and sb is not None:
        joined = pd.concat([sa, sb], axis=1, keys=["a", "b"]).sort_index().ffill().dropna()
        if len(joined) >= 2:
            intra = (joined["a"] - joined["b"]) * 100.0
    status = ("disputed" if "disputed" in (a["verdict"], b["verdict"])
              else "verified" if a["verdict"] == b["verdict"] == "verified" else "single")
    return {
        "name": name, "symbol": key, "category": "Bonds", "feed_row": True,
        "unit": "spread", "verdict": status, "source": a["source"],
        "value": level, "change": level - prev, "pct": None, "bp": level - prev,
        "baseline": prev, "spark": [float(x) for x in intra.values] if intra is not None else [],
        "spark_series": intra, "span": a.get("span") if a.get("span") == b.get("span") else None,
        "session_hours": a.get("session_hours"),
        "as_of": min(a["as_of"], b["as_of"]), "live": a["live"] and b["live"],
        "observed_day": a["observed_day"], "stale_session": False, "alternatives": [],
    }


def fetch_feed_quotes(now: Optional[datetime] = None, rows=FEED_ROWS) -> list[dict]:
    """One strip dict per row that could be shown, in ``FEED_KEYS`` order.

    Empty when the run may not use live transport (a pinned ``--as_of`` run):
    these feeds have no point-in-time history, and a figure from today must
    not appear in an issue dated earlier. Never raises.
    """
    from tarzan import runtime

    try:
        if not runtime.allows_live_transport():
            return []
        if now is None:
            from tarzan.data.market_quotes import _intraday_reference_now
            now = _intraday_reference_now()
        units = {feed: {} for feed in (TV, CNBC)}
        for r in rows:
            for feed in (TV, CNBC):
                if feed in r.symbols:
                    units[feed][r.symbols[feed]] = r.unit
        # Round 1: every quote. Two batch calls plus one MarketWatch series per row.
        mw_rows = [r for r in rows if MW in r.symbols]
        results = _parallel([(_tradingview_quotes, list(units[TV]), units[TV]),
                             (_cnbc_quotes, list(units[CNBC]), units[CNBC])]
                            + [(_marketwatch, r.symbols[MW], r.unit) for r in mw_rows])
        tv, cnbc = results[0] or {}, results[1] or {}
        mw = {r.key: res for r, res in zip(mw_rows, results[2:]) if res}

        chosen = {}
        for r in rows:
            obs = {}
            if TV in r.symbols and r.symbols[TV] in tv:
                obs[TV] = tv[r.symbols[TV]]
            if CNBC in r.symbols and r.symbols[CNBC] in cnbc:
                obs[CNBC] = cnbc[r.symbols[CNBC]]
            if r.key in mw and mw[r.key][0] is not None:
                obs[MW] = mw[r.key][0]
            obs = _fresh(obs, r, now)
            timetable = _timetable(r, now)
            pick = choose(r, obs, closed=bool(timetable) and not timetable[1])
            if pick is not None:
                chosen[r.key] = (r, obs, *pick)

        # Round 2: the CNBC charts the chosen recipes draw (MarketWatch's came
        # with its quote).
        cnbc_charts = [(r.key, r.symbols[CNBC], r.unit) for r, _o, _s, feed, _v in chosen.values()
                       if feed == CNBC]
        tv_wanted = {r.symbols[TV]: r.unit for r, *_ in chosen.values() if TV in r.symbols}
        drawn = _parallel([(_tradingview_charts, list(tv_wanted), tv_wanted)]
                          + [(_cnbc_chart, sym, unit) for _k, sym, unit in cnbc_charts])
        tv_lines = drawn[0] or {}
        charts = {k: c for (k, _s, _u), c in zip(cnbc_charts, drawn[1:]) if c is not None}

        out = {}
        for key, (r, obs, shown, chart_feed, status) in chosen.items():
            fallback = (charts.get(key) if chart_feed == CNBC
                        else mw[key][1] if chart_feed == MW and key in mw else Chart()) or Chart()
            others = [o for f, o in obs.items() if f != shown.feed]
            # TradingView's line first: it holds every session whole (a Eurex
            # future's night hours included) and is the same feed as most shown
            # levels. The recipe's own chart when TradingView has none, or one
            # of the wrong session.
            line = tv_lines.get(r.symbols.get(TV))
            row = None
            if line is not None:
                row = _row_dict(r, shown, Chart(intra=line, daily=fallback.daily,
                                                span=fallback.span),
                                status, others, now)
                if row["spark_series"] is None:
                    row = None
            out[key] = row or _row_dict(r, shown, fallback, status, others, now)
        for name, key, a, b in SPREADS:
            spread = _spread(name, key, out.get(a), out.get(b))
            if spread is not None:
                out[key] = spread
        return [out[k] for k in FEED_KEYS if k in out]
    except Exception as e:  # noqa: BLE001 — the strip must never break a run
        logger.info("market feeds unavailable (%s)", type(e).__name__)
        return []
