"""Live quotes for the newsletter "Markets" strip (yfinance-style).

A curated set mirroring the yfinance markets bar (US / Europe / Asia /
Crypto / Rates / Commodities / Currencies). For each instrument we return
the latest level, the change versus the previous close, and an intraday
"day" path (rebased to the previous close) for a two-tone sparkline.

Data sources, both best-effort and graceful:
  * level + previous close: the enricher's cached, throttled daily-history
    helper (reuses the on-disk price cache; no cold re-download on warm
    runs);
  * intraday day path: a single batched ``yfinance`` download (one request
    for all symbols) — if it fails or a symbol is missing, the daily
    history is used as the spark fallback.

Each quote: ``{name, symbol, category, value, change, pct, spark,
baseline}`` where ``baseline`` is the previous close (the 0% line the
sparkline shades green above / red below).
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, time as dtime
from typing import Optional

from tarzan.runtime.ledger import Availability
from tarzan.runtime.provider import ProviderAttempt, ProviderResult

logger = logging.getLogger(__name__)

# A market is treated as "trading now" (live) when its latest intraday bar is
# no older than this. yfinance intraday lags ~15-30 min, so this keeps a live
# session marked live while flipping to "previous day" within ~1h of the close
# (so an evening newsletter doesn't mislabel a closed market as live).
_MARKET_OPEN_MAX_LAG_MIN = 60

# Regular cash-session hours per exchange group, keyed by a normalized code.
# (IANA timezone, (open_h, open_m), (close_h, close_m)). Holidays and lunch
# breaks are not modelled (best-effort); weekends are handled separately.
_SESSIONS: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "EU": ("Europe/Rome", (9, 0), (17, 30)),      # Milan/Xetra/Paris/Amsterdam
    "L":  ("Europe/London", (8, 0), (16, 30)),    # London
    "US": ("America/New_York", (9, 30), (16, 0)),  # US cash session
    "JP": ("Asia/Tokyo", (9, 0), (15, 0)),        # Tokyo
    "HK": ("Asia/Hong_Kong", (9, 30), (16, 0)),   # Hong Kong
    "CN": ("Asia/Shanghai", (9, 30), (15, 0)),    # Shanghai
    "AU": ("Australia/Sydney", (10, 0), (16, 0)),  # Sydney
    "KR": ("Asia/Seoul", (9, 0), (15, 30)),       # Seoul
}

# Index / bare symbols → exchange group (symbols without a Yahoo suffix).
_INDEX_EXCHANGE: dict[str, str] = {
    "^GSPC": "US", "^DJI": "US", "^IXIC": "US", "^RUT": "US", "^RUI": "US",
    "^VIX": "US", "^NDX": "US", "^SPXEW": "US",
    "^IRX": "US", "^FVX": "US", "^TNX": "US", "^TYX": "US",
    "^FTSE": "L",
    "^FCHI": "EU", "^GDAXI": "EU", "^STOXX50E": "EU", "^N100": "EU",
    "^N225": "JP", "^HSI": "HK", "^AXJO": "AU", "^KS11": "KR",
}

# Yahoo listing suffix → exchange group.
_SUFFIX_EXCHANGE: dict[str, str] = {
    "MI": "EU", "DE": "EU", "PA": "EU", "AS": "EU", "F": "EU",
    "L": "L",
    "SS": "CN", "SZ": "CN", "HK": "HK", "T": "JP",
    "AX": "AU", "KS": "KR",
}


def _exchange_for(ticker: str) -> Optional[str]:
    """Map a verified Yahoo listing to its exchange-session group.

    A suffixless symbol is not inherently American. Curated taxonomy evidence
    may promote a bare input to a full listing; otherwise ``None`` deliberately
    delegates freshness to wall-clock age instead of fabricating a venue.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return None
    if t.endswith("-USD") or t.endswith("=X") or t.endswith("=F"):
        return None
    if t.startswith("^"):
        return _INDEX_EXCHANGE.get(t)
    if "." in t:
        return _SUFFIX_EXCHANGE.get(t.rsplit(".", 1)[1])

    try:
        from tarzan import config as cfg

        _, resolved_ticker = cfg.resolve_taxonomy_identity("", t)
        resolved = str(resolved_ticker or "").strip().upper()
        if resolved != t and "." in resolved:
            return _SUFFIX_EXCHANGE.get(resolved.rsplit(".", 1)[1])
    except Exception:  # noqa: BLE001 - freshness must fail conservatively
        pass
    return None


def is_continuous_market(ticker: str) -> bool:
    """Whether the instrument trades ~around the clock with no bounded cash
    session: commodity/index futures (``=F``), FX pairs (``=X``) and crypto
    (``-USD``). These have no equity-style open→close, so their intraday
    sparkline should be drawn full-width (stretched) rather than "growing
    through the session"."""
    t = (ticker or "").upper()
    return t.endswith("=F") or t.endswith("=X") or t.endswith("-USD")


def market_open_now(ticker: str, now: Optional[datetime] = None) -> Optional[bool]:
    """Whether the instrument's primary exchange is in its regular trading
    session right now, judged by exchange hours (NOT bar recency).

    Returns True/False for instruments on a known cash exchange (mapped by
    Yahoo suffix or index symbol); True for 24/7 crypto; and None when the
    session concept doesn't cleanly apply (FX and futures trade nearly around
    the clock) so callers can fall back to recency. Weekends are closed;
    holidays are not modelled. Never raises."""
    t = (ticker or "").upper()
    if t.endswith("-USD"):
        return True  # crypto trades 24/7
    if t.endswith("=X") or t.endswith("=F"):
        return None  # FX / futures → let the caller decide by recency
    ex = _exchange_for(t)
    if ex is None:
        return None
    tzname, (oh, om), (ch, cm) = _SESSIONS[ex]
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
        n = now.astimezone(tz) if now is not None else datetime.now(tz)
    except Exception:  # noqa: BLE001
        return None
    if n.weekday() >= 5:  # Saturday / Sunday
        return False
    return dtime(oh, om) <= n.time() <= dtime(ch, cm)


# Short zone label per exchange group, for a compact human-readable hours
# caption ("09:30\u201316:00 ET") rather than a raw IANA zone name.
_SESSION_LABEL: dict[str, str] = {
    "EU": "CET", "L": "GMT", "US": "ET", "JP": "JST",
    "HK": "HKT", "CN": "CST", "AU": "AEST", "KR": "KST",
}


def session_caption(ticker: str) -> str:
    """A short, human caption for an instrument's trading session.

    A bounded cash session gets its local hours and zone abbreviation, e.g.
    "09:30\u201316:00 ET" \u2014 always in the exchange's own local time, since that
    is how every financial site quotes session hours and the reader already
    holds several exchanges' worth of them side by side. A continuously
    traded instrument (futures/FX/crypto, per :func:`is_continuous_market`)
    has no single bounded session to state, so it gets "\u224824h" instead of a
    fabricated or misleading open/close pair. Empty string when the exchange
    is not one of the modelled groups. Never raises.
    """
    if is_continuous_market(ticker):
        return "\u224824h"
    ex = _exchange_for(ticker)
    if ex is None or ex not in _SESSIONS:
        return ""
    tzname, (oh, om), (ch, cm) = _SESSIONS[ex]
    label = _SESSION_LABEL.get(ex, "")
    hours = f"{oh:02d}:{om:02d}\u2013{ch:02d}:{cm:02d}"
    return f"{hours} {label}".strip()


def futures_open_now(now: Optional[datetime] = None) -> bool:
    """Whether CME/CBOT equity index futures (E-mini S&P/Nasdaq/Dow/Russell
    \u2014 ticker suffix "=F") are inside their Globex trading window right now.

    "\u224824h" (per :func:`session_caption`) means nearly continuous, not
    literally 24/7: closed over the weekend (Friday 17:00 ET to Sunday
    18:00 ET) and for a one-hour daily maintenance halt (17:00\u201318:00 ET,
    Monday\u2013Thursday). All times ET. Does not model CME holiday closures.
    Defaults to closed (the conservative reading) if local time cannot be
    computed. Never raises.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_SESSIONS["US"][0])
        n = now.astimezone(tz) if now is not None else datetime.now(tz)
    except Exception:  # noqa: BLE001
        return False
    wd, t = n.weekday(), n.time()
    if wd == 5:  # Saturday: closed all day
        return False
    if wd == 6:  # Sunday: the week reopens at 18:00 ET
        return t >= dtime(18, 0)
    if wd == 4:  # Friday: closes at 17:00 ET, no reopen the same day
        return t < dtime(17, 0)
    # Monday-Thursday: closed only for the 17:00-18:00 ET maintenance break.
    return not (dtime(17, 0) <= t < dtime(18, 0))


_WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _cash_session_day(n: datetime, oh: int, om: int) -> str:
    """The plain calendar weekday (exchange-local) whose cash session is
    live right now, or was the most recently completed one: today once
    today's own session has started, otherwise the previous trading day
    (walking back over a weekend one weekday further)."""
    from datetime import timedelta
    d = n
    if not (d.weekday() < 5 and d.time() >= dtime(oh, om)):
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
    return _WD[d.weekday()]


def fx_open_now(now: Optional[datetime] = None) -> bool:
    """Whether FX spot (Yahoo ticker suffix "=X") is trading right now.

    24/5, not 24/7: closed from Friday 17:00 ET (New York close) to Sunday
    17:00 ET (the conventional "Sydney open" reference, stated in New York
    time so it moves with New York's own DST the same way the close does
    -- this matches 22:00 UTC in winter / 21:00 UTC in summer for both
    edges). Unlike CME futures, FX has no daily maintenance halt. All times
    ET. Defaults to closed if local time cannot be computed. Never raises.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_SESSIONS["US"][0])
        n = now.astimezone(tz) if now is not None else datetime.now(tz)
    except Exception:  # noqa: BLE001
        return False
    wd, t = n.weekday(), n.time()
    if wd == 5:
        return False
    if wd == 6:
        return t >= dtime(17, 0)
    if wd == 4:
        return t < dtime(17, 0)
    return True


def market_status(ticker: str, now: Optional[datetime] = None) -> tuple:
    """(is_open, weekday_label) for the MARKETS caption: whether trading is
    live right now, and the plain calendar day that status refers to.

    Unlike :func:`market_open_now`, this always resolves a real yes/no for
    a CME/CBOT future ("=F") or FX pair ("=X") rather than deferring to
    recency, since both schedules are well known (:func:`futures_open_now`,
    :func:`fx_open_now`). Crypto ("-USD") is continuously open with no
    weekly closure worth stating, so it gets ``(True, "")`` \u2014 no day
    needed when there is never a different one to point to. ``(None, "")``
    when no schedule is modelled for the ticker. Never raises.
    """
    t = (ticker or "").upper()
    if t.endswith("-USD"):
        return True, ""
    if t.endswith("=F") or t.endswith("=X"):
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(_SESSIONS["US"][0])
            n = now.astimezone(tz) if now is not None else datetime.now(tz)
        except Exception:  # noqa: BLE001
            return None, ""
        is_open = futures_open_now(n) if t.endswith("=F") else fx_open_now(n)
        # Today's calendar day once open, or closed only for today's brief
        # maintenance break; Friday specifically during the weekend closure
        # (Saturday, or Sunday before the 18:00 ET reopen), since that is
        # the day whose session most recently ended.
        day = _WD[n.weekday()] if (is_open or n.weekday() < 5) else "Fri"
        return is_open, day
    ex = _exchange_for(t)
    if ex is None or ex not in _SESSIONS:
        return None, ""
    tzname, (oh, om), (ch, cm) = _SESSIONS[ex]
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
        n = now.astimezone(tz) if now is not None else datetime.now(tz)
    except Exception:  # noqa: BLE001
        return None, ""
    return market_open_now(ticker, now), _cash_session_day(n, oh, om)


def market_session_age_seconds(
    ticker: str,
    observed_at: datetime,
    captured_at: datetime,
) -> Optional[float]:
    """Return freshness age measured in completed cash-market sessions.

    Daily market bars are date-labelled (normally at midnight), so the bar for
    Friday represents Friday's completed close. Closed weekend hours must not
    age it. Every later completed weekday session counts as one policy day
    (86,400 seconds); an in-progress current session contributes only elapsed
    open time. Same-day evidence uses ordinary elapsed time so stale intraday
    data remains detectable. ``None`` delegates to wall-clock freshness for
    continuous or unknown markets. Holidays are not modelled because Tarzan
    currently has no authoritative exchange calendar.
    """
    from datetime import timedelta

    exchange = _exchange_for(ticker)
    if exchange is None:
        return None
    # Daily bars retain a midnight date label even when timezone conversion
    # moves that instant away from 00:00 in the venue timezone.
    observed_is_date_label = (
        observed_at.hour == 0
        and observed_at.minute == 0
        and observed_at.second == 0
        and observed_at.microsecond == 0
    )
    # ISIN placeholders and cash labels are not exchange symbols. Treating
    # them as bare US tickers would fabricate a session authority.
    normalized = str(ticker or "").strip().upper()
    if (
        len(normalized.replace("-", "")) == 12
        and normalized.replace("-", "")[:2].isalpha()
        and normalized.replace("-", "").isalnum()
    ):
        return None

    tzname, (open_hour, open_minute), (close_hour, close_minute) = _SESSIONS[exchange]
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(tzname)
        # A naive daily-bar timestamp is a venue-local date label. Aware
        # timestamps (for example regularMarketTime) carry a real instant.
        observed = (
            observed_at.replace(tzinfo=zone)
            if observed_at.tzinfo is None
            else observed_at.astimezone(zone)
        )
        captured = (
            captured_at.replace(tzinfo=zone)
            if captured_at.tzinfo is None
            else captured_at.astimezone(zone)
        )
    except Exception:  # noqa: BLE001
        return None

    if captured <= observed:
        return 0.0
    if captured.date() == observed.date():
        return max(0.0, (captured - observed).total_seconds())

    age = 0.0
    # A non-date-label observation is intraday. If its own session
    # subsequently completed, a newer close exists and consumes one policy day.
    if not observed_is_date_label and observed.weekday() < 5:
        observed_close = observed.replace(
            hour=close_hour,
            minute=close_minute,
            second=0,
            microsecond=0,
        )
        if captured >= observed_close and observed < observed_close:
            age += 86400.0

    day = observed.date() + timedelta(days=1)
    while day <= captured.date():
        if day.weekday() < 5:
            session_open = datetime.combine(
                day,
                dtime(open_hour, open_minute),
                tzinfo=zone,
            )
            session_close = datetime.combine(
                day,
                dtime(close_hour, close_minute),
                tzinfo=zone,
            )
            if captured >= session_close:
                age += 86400.0
            elif captured > session_open:
                age += (captured - session_open).total_seconds()
        day += timedelta(days=1)
    return max(0.0, age)

# (display name, yfinance symbol, category), in display order. The strip
# shows at most 2 rows per category (the newsletter caps it).
MARKETS: list[tuple[str, str, str]] = [
    # US — equity indices + US Treasury yields (^IRX/^FVX/^TNX/^TYX), grouped
    # together since they are all US-market references.
    ("S&P 500", "^GSPC", "US"),
    ("Dow 30", "^DJI", "US"),
    ("Nasdaq Composite", "^IXIC", "US"),
    ("Nasdaq 100", "^NDX", "US"),
    ("Russell 2000", "^RUT", "US"),
    ("VIX", "^VIX", "US"),
    ("US 13-Wk", "^IRX", "US"),
    ("US 5-Yr", "^FVX", "US"),
    ("US 10-Yr", "^TNX", "US"),
    ("US 30-Yr", "^TYX", "US"),
    # US index futures (CME/CBOT E-mini contracts), right after their cash
    # index. Their own name already carries "(FUT)" -- not left to the
    # render-time auto-suffix in _row() -- because otherwise this name would
    # collide with the cash index's ("S&P 500" listed twice), which breaks
    # any lookup keyed by name (fetch_market_quotes results included). Trade
    # nearly around the clock but not literally 24/7 (see
    # futures_open_now()/market_status() for the real Globex weekly + daily
    # schedule) rather than the cash session above them.
    # NQ=F tracks the Nasdaq-100 specifically, which now sits above as its
    # own cash entry (distinct from the Composite, ^IXIC) -- named
    # "Nasdaq 100 (FUT)" to pair with it directly.
    ("S&P 500 (FUT)", "ES=F", "US"),     # E-mini S&P
    ("Dow 30 (FUT)", "YM=F", "US"),      # E-mini Dow
    ("Nasdaq 100 (FUT)", "NQ=F", "US"),  # E-mini Nasdaq
    ("Russell 2000 (FUT)", "RTY=F", "US"),  # E-mini Russell
    # Europe — equity indices + a German 10Y reference. Yahoo exposes no
    # German 10Y yield ticker (à la ^TNX), so "Bund 10Y" is a German
    # government-bond ETF proxy: iShares eb.rexx Government Germany 5.5-10.5yr
    # (EXHD.DE), a EUR PRICE centered on the ~10Y segment (moves inverse to
    # yield), not a yield.
    ("FTSE 100", "^FTSE", "Europe"),
    ("CAC 40", "^FCHI", "Europe"),
    ("DAX", "^GDAXI", "Europe"),
    ("Euronext 100", "^N100", "Europe"),
    ("Euro Stoxx 50", "^STOXX50E", "Europe"),
    ("Bund 10Y", "EXHD.DE", "Europe"),
    # Asia
    ("SSE Composite", "000001.SS", "Asia"),
    ("Nikkei 225", "^N225", "Asia"),
    ("Hang Seng", "^HSI", "Asia"),
    ("ASX 200", "^AXJO", "Asia"),
    ("KOSPI", "^KS11", "Asia"),
    # Commodities. Two global crude benchmarks are shown side by side: WTI
    # (CL=F, NYMEX/CME — the US reference) and Brent (BZ=F, ICE Futures Europe
    # — the international reference used for most of the world incl. Europe).
    ("WTI Crude", "CL=F", "Commodities"),
    ("Gold", "GC=F", "Commodities"),
    ("Silver", "SI=F", "Commodities"),
    ("Copper", "HG=F", "Commodities"),
    ("Natural Gas", "NG=F", "Commodities"),
    ("Brent Crude", "BZ=F", "Commodities"),
    ("Platinum", "PL=F", "Commodities"),
    # Currencies (fiat FX pairs + Bitcoin, both quoted vs USD and traded
    # ~around the clock).
    ("EUR/USD", "EURUSD=X", "Currencies"),
    ("USD/JPY", "JPY=X", "Currencies"),
    ("USD/GBP", "GBP=X", "Currencies"),
    ("Bitcoin", "BTC-USD", "Currencies"),
]

CATEGORY_ORDER = ["US", "Europe", "Asia", "Commodities", "Currencies"]

_memo: Optional[list[dict]] = None
_memo_at: float = 0.0  # monotonic timestamp the memo was filled
# Live quotes go stale within a session. Memoizing forever means a
# long-running process (a persistent worker / server, not the one-shot CLI)
# serves the same quotes across the market close and into the next session.
# A short TTL bounds that: within one CLI run every call is still served from
# the memo, but a process that outlives the TTL re-fetches.
_MEMO_TTL_SECONDS = 900  # 15 minutes (≈ yfinance intraday lag)


# Intraday-only sibling fallback. When a EUR listing has no intraday feed
# (the classic Borsa Italiana ``.MI`` case, where Yahoo's Milan feed is often
# stale/empty), we may borrow the intraday series from a same-root candidate on
# another EUR venue (Xetra/Euronext), but only after the canonical listing was
# attempted and its daily close passes the price-coherence guard below. Only
# EUR venues are used so the "vs previous close" % stays a faithful EUR proxy
# — London (.L, often USD/GBP) is intentionally excluded to avoid FX-contaminated
# returns. This affects ONLY the intraday sparkline / broker-1D path; EOD/daily
# history (valuation, returns, risk) always stays on the canonical listing.
_SIBLING_SUFFIXES: dict[str, tuple[str, ...]] = {
    "MI": ("DE", "PA", "AS", "F"),   # Milan  → Xetra, Paris, Amsterdam, Frankfurt
    "PA": ("DE", "MI", "AS", "F"),   # Paris  → Xetra, Milan, ...
    "AS": ("DE", "MI", "PA", "F"),   # Amsterdam
    "F":  ("DE", "MI", "PA", "AS"),  # Frankfurt floor → Xetra, ...
    "DE": ("MI", "PA", "AS", "F"),   # Xetra   → Milan, Paris, ...
}

# A venue candidate is accepted only when its latest intraday price is within
# this fraction of the canonical listing's last known close. A wider gap
# signals that the same-root ticker on another exchange may be a different
# instrument and is rejected by the collision guard.
_SIBLING_PRICE_TOLERANCE = 0.10



# A 15m-interval feed that has truly gone stale (Yahoo's Borsa Italiana
# ``.MI`` intraday feed is the known offender — see the sibling-fallback
# note above) still satisfies ``len(ser) >= 2`` on its last couple of
# pre-market/open ticks and never advances again. That let a stale-but-
# nonempty primary series silently short-circuit the sibling fallback that
# exists specifically to route around it, freezing the sparkline and the
# broker-1D % at whatever those first two ticks were for the rest of the
# session. Two intervals of slack (30 min) covers a missed bar without
# flagging a feed that is merely early in the session as stale.
#
# This staleness-by-recency check only means something while a session
# should be actively advancing. Once the relevant market is closed, its
# last print being wall-clock "old" is simply correct (see
# _fetch_intraday's own-day filter, which already narrowed the series to
# one real session) — rejecting it here as "stale" sent every closed-market
# symbol through an unnecessary, and usually unsuccessful, sibling search.
# _has_intraday takes the ticker so it can tell the two cases apart via
# market_open_now(); for FX/futures (market_open_now returns None — no
# clean open/close concept) the recency check still applies, unchanged.
#
# The threshold is 3h, not the 30 min (2x the 15m interval) it started at.
# Yahoo's free 15m feed for small European ETF listings lags far more than
# its nominal interval: measured on a live run, NTSG.PA was 36 min behind
# and NTSG.DE 246 min, while both carried a full session of real bars priced
# within 0.25% of the primary's own close. The 30 min rule rejected both, so
# the sibling fallback found perfectly usable data and threw it away — the
# holding then showed the previous day's close while its market was open,
# which is exactly what the fallback exists to prevent. 3h still catches a
# feed frozen at the open (the .MI failure this guard was built for, which
# stops advancing for the whole session) while accepting one that is merely
# behind. A lagging-but-advancing feed is worth far more than yesterday.
_INTRADAY_STALE_AFTER_SECONDS = 10800  # 3h — Yahoo's EU ETF 15m feed lags


def _intraday_reference_now(tzinfo=None):
    """Real current time, tz-aware. A thin, monkeypatchable seam (mirrors
    the ``now=`` pattern used by ``market_open_now`` elsewhere in this
    module) so tests can pin "now" without threading a parameter through
    the whole intraday call chain."""
    from datetime import timezone as _tz
    return datetime.now(tzinfo or _tz.utc)


def _no_trading_day_skipped(ticker: str, last_ts, now) -> bool:
    """Whether every calendar day strictly between ``last_ts`` and ``now``
    was a non-trading day for ``ticker`` (weekend, per market_open_now's
    weekday gate) - i.e. the market had no opportunity to print anything
    fresher in between. Probed once at midday per intervening day; coarse
    but sufficient to separate "just closed" from "stuck behind a session
    that did happen". Conservative on any error: reports a skip (False),
    so a real staleness check still runs rather than silently trusting
    data that couldn't be verified.

    The probe -- and the day range it walks -- is built in the EXCHANGE's own
    timezone, not the series'. yfinance indexes intraday bars in UTC, and
    midday UTC falls outside most exchanges' local sessions (08:00 in New
    York, before the 09:30 open; the small hours in Tokyo/Hong Kong/Sydney).
    Probing at 12:00 UTC therefore reported "closed" for every intervening
    weekday on every non-European exchange, so a feed genuinely stuck behind
    a completed session read as "nothing fresher could exist" and was
    accepted as that market's last session -- which is exactly the stale-day
    anchoring bug the caller's guard exists to prevent. 12:00 local is inside
    the regular session of every exchange in _SESSIONS.
    """
    from datetime import timedelta
    try:
        ex = _exchange_for((ticker or "").upper())
        if ex is None or ex not in _SESSIONS:
            return False
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(_SESSIONS[ex][0])
        d = last_ts.astimezone(tz).date() + timedelta(days=1)
        end_date = now.astimezone(tz).date()
        while d < end_date:
            probe = datetime.combine(d, dtime(12, 0), tzinfo=tz)
            if market_open_now(ticker, now=probe):
                return False
            d += timedelta(days=1)
    except Exception:  # noqa: BLE001
        return False
    return True


def _has_intraday(ser, ticker: Optional[str] = None) -> bool:
    if ser is None or len(ser) < 2:
        return False
    try:
        last_ts = ser.index[-1]
        now = _intraday_reference_now(getattr(last_ts, "tzinfo", None))
        if (ticker is not None
                and market_open_now(ticker, now=now) is False
                and _no_trading_day_skipped(ticker, last_ts, now)):
            # Closed market, and nothing fresher could plausibly exist yet:
            # an "old" last print is expected and correct, not a sign of a
            # stuck feed. See the comment on _INTRADAY_STALE_AFTER_SECONDS
            # above. If a valid trading day WAS skipped in between, fall
            # through to the normal age check instead - the sibling search
            # exists precisely to route around a primary stuck behind a
            # session that did happen.
            return True
        age_seconds = (now - last_ts.to_pydatetime()).total_seconds()
    except Exception:  # noqa: BLE001
        # Unexpected index type: don't newly reject on a shape we can't
        # evaluate — fall back to the pre-existing length-only behavior.
        return True
    return age_seconds <= _INTRADAY_STALE_AFTER_SECONDS


def _official_and_prev(fetch_history, ticker: str, iday):
    """From a ticker's daily history, return ``(official_close_on_iday,
    previous_close_before_iday)`` as floats (or ``None`` each when missing).

    ``official_close_on_iday`` is the exchange's settled daily close for the
    session dated ``iday`` (it incorporates the closing auction). Best-effort:
    returns ``(None, None)`` on any error."""
    try:
        hist = fetch_history(ticker)
        if hist is None or not len(hist) or "Close" not in getattr(hist, "columns", []):
            return None, None
        dclose = hist["Close"].dropna()
        same = dclose[[ts.date() == iday for ts in dclose.index]]
        prior = dclose[[ts.date() < iday for ts in dclose.index]]
        oc = float(same.iloc[-1]) if len(same) else None
        pv = float(prior.iloc[-1]) if len(prior) else None
        return oc, pv
    except Exception:  # noqa: BLE001
        return None, None


def _sibling_symbols(ticker: str) -> list[str]:
    """Candidate sibling listings (same root, alternate EUR venue) for a
    ticker, in priority order. Empty for indices (^...), FX/futures (=X/=F),
    crypto (-USD), suffixless US tickers, and non-EUR venues."""
    t = (ticker or "").upper()
    if not t or "." not in t or t.startswith("^") or "=" in t or "-" in t:
        return []
    root, suf = t.rsplit(".", 1)
    return [f"{root}.{s}" for s in _SIBLING_SUFFIXES.get(suf, ())]


def _resolve_intraday(
    symbols: list[str],
    *,
    allow_sibling_fallback: bool = True,
) -> dict:
    """Fetch intraday series for exact symbols, optionally trying siblings.

    The portfolio pipeline enables sibling discovery only at this run-scoped
    preprocessing boundary. Downstream analytics and presentation consume the
    selected series and provenance without resolving another venue.
    """
    prim = _fetch_intraday(symbols)
    out: dict = {s: (prim[s], s) for s in symbols if _has_intraday(prim.get(s), s)}
    missing = [s for s in symbols if s not in out]
    if missing:
        stale = [s for s in missing if prim.get(s) is not None]
        empty = [s for s in missing if prim.get(s) is None]
        if stale:
            logger.info(
                "intraday primary rejected as stale (>%ds old), trying siblings: %s",
                _INTRADAY_STALE_AFTER_SECONDS,
                ", ".join(stale),
            )
        if empty:
            logger.info(
                "intraday primary returned no data, trying siblings: %s",
                ", ".join(empty),
            )
    if not missing or not allow_sibling_fallback:
        return out

    cand_map = {s: _sibling_symbols(s) for s in missing}
    all_cands = list(dict.fromkeys(c for cs in cand_map.values() for c in cs))
    if not all_cands:
        return out

    sib = _fetch_intraday(all_cands)
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception:  # noqa: BLE001
        _fetch_history = None  # type: ignore

    for s in missing:
        # The canonical listing's last known close is mandatory evidence for
        # the collision guard. An equal ticker root alone does not establish
        # that another venue serves the same instrument.
        prim_close = None
        if _fetch_history is not None:
            try:
                h = _fetch_history(s)
                if h is not None and len(h) and "Close" in getattr(h, "columns", []):
                    cl = h["Close"].dropna()
                    if len(cl):
                        prim_close = float(cl.iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        if not prim_close:
            logger.info(
                "intraday fallback %s rejected (no canonical close for "
                "price-coherence guard)",
                s,
            )
            continue
        rejected_candidates: list[str] = []
        stale_candidates: list[str] = []
        for c in cand_map[s]:
            ser = sib.get(c)
            if not _has_intraday(ser, c):
                # Absent and stale are different diagnoses, and conflating
                # them is actively misleading: a sibling holding a full
                # session of bars that merely lag the staleness threshold
                # used to be reported as "no data", which reads as "this
                # venue does not exist" and hides the fact that the
                # threshold — not the provider — is what dropped it.
                if ser is not None:
                    stale_candidates.append(c)
                continue
            dev = abs(float(ser.iloc[-1]) / prim_close - 1.0)
            if dev > _SIBLING_PRICE_TOLERANCE:
                logger.info(
                    "intraday fallback %s→%s rejected (%.1f%% off primary close)",
                    s,
                    c,
                    dev * 100,
                )
                rejected_candidates.append(c)
                continue
            out[s] = (ser, c)
            logger.info(
                "intraday fallback: %s → %s (price-coherent EUR venue candidate)",
                s,
                c,
            )
            break
        else:
            tried = cand_map[s]
            accounted = set(rejected_candidates) | set(stale_candidates)
            no_data = [c for c in tried if c not in accounted]
            logger.info(
                "intraday fallback exhausted for %s — no usable venue "
                "(tried %s; no data: %s; stale >%ds: %s; price-mismatched: %s)",
                s,
                ", ".join(tried) or "none",
                ", ".join(no_data) or "none",
                _INTRADAY_STALE_AFTER_SECONDS,
                ", ".join(stale_candidates) or "none",
                ", ".join(rejected_candidates) or "none",
            )
    return out


def _fetch_intraday(symbols: list[str]) -> dict:
    """One batched intraday download → ``{symbol: Close series}``. Empty on
    any failure (the caller falls back to the daily history)."""
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return {}

    out: dict = {}
    try:
        import warnings
        import yfinance as yf
        from tarzan.data import _yf_net

        def _download():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 5d, not 1d: a market closed for a long weekend or a
                # holiday needs more than one day of lookback to still find
                # its last completed session. The per-symbol filter below
                # then keeps exactly one day out of whatever comes back -
                # today's if the market is open, otherwise the most recent
                # one present - so a wider window never means a multi-day
                # chart, only a better chance of finding a single real one.
                return yf.download(symbols, period="5d", interval="15m",
                                   group_by="ticker", progress=False, threads=True)
        # Shared spacing+retry so the intraday batch survives a 429 burst.
        raw = _yf_net.fetch_yf(_download, what="intraday batch", log=logger)
        if raw is None or len(raw) == 0:
            return {}
        level0 = set(raw.columns.get_level_values(0)) if hasattr(raw.columns, "get_level_values") else set()
        for s in symbols:
            try:
                if s in level0 and "Close" in raw[s].columns:
                    cl = raw[s]["Close"].dropna()
                    if len(cl) >= 2:
                        last_day = cl.index[-1].date()
                        same_day = cl[[ts.date() == last_day for ts in cl.index]]
                        if len(same_day) >= 2:
                            out[s] = same_day
            except Exception:  # noqa: BLE001
                continue
        missing = [s for s in symbols if s not in out]
        if missing:
            # Visible at INFO (not DEBUG) on purpose: a symbol silently
            # missing intraday data falls back to the daily-close chart
            # with a "no intraday" label in the newsletter, which reads as
            # a routine one-off Yahoo gap unless the pattern is visible in
            # the logs. If the same symbols show up here run after run,
            # that is a real gap in the batch response, not flakiness.
            logger.info("intraday batch: %d/%d symbols missing from response: %s",
                        len(missing), len(symbols), ", ".join(missing))
    except Exception as e:  # noqa: BLE001
        logger.debug("intraday batch failed: %s", e)
    return out


def _fetch_official_prev_closes(symbols: list[str]) -> dict:
    """One batched Yahoo v7 quote call → ``{symbol:
    regularMarketPreviousClose}`` (positive floats only).

    This is the OFFICIAL prior settlement/close Yahoo shows behind its headline
    change — the authoritative daily-% baseline. Needed because yfinance's
    *daily-history* ``Close`` is unreliable for some futures (see ``_quote``):
    the v7 quote endpoint carries the correct settlement where the chart API
    does not. ONE authenticated batched request for the whole strip (not one
    ``.info`` per symbol — that is the slowest, most 429-prone yfinance path,
    and its lighter ``fast_info`` cousin returns the wrong value for exactly the
    futures this fixes). Empty on any failure so the caller falls back to the
    daily-history close. Never raises."""
    from tarzan import runtime

    if not runtime.allows_live_transport() or not symbols:
        return {}
    out: dict = {}
    try:
        from yfinance.data import YfData
        from tarzan.data import _yf_net

        data = YfData()

        def _get():
            resp = data.get(
                "https://query2.finance.yahoo.com/v7/finance/quote",
                params={"symbols": ",".join(symbols)},
            )
            resp.raise_for_status()
            return resp.json()

        payload = _yf_net.fetch_yf(_get, what="markets prev-close batch", log=logger)
        if not payload:
            return {}
        for q in payload.get("quoteResponse", {}).get("result", []):
            sym = q.get("symbol")
            pc = q.get("regularMarketPreviousClose")
            try:
                if sym and pc is not None and float(pc) > 0:
                    out[sym] = float(pc)
            except (TypeError, ValueError):
                continue
        missing = [s for s in symbols if s not in out]
        if missing:
            # INFO, not DEBUG: a symbol absent here silently falls back to the
            # (possibly unreliable) daily-history close for its baseline. A
            # persistent gap for the same symbol is a real regression in the
            # quote response, not one-off flakiness — see _fetch_intraday's
            # matching rationale.
            logger.info("prev-close batch: %d/%d symbols missing: %s",
                        len(missing), len(symbols), ", ".join(missing))
    except Exception as e:  # noqa: BLE001
        logger.debug("prev-close batch failed: %s", e)
    return out


def _quote(dclose, intra, spark_points: int = 40,
           official_prev: Optional[float] = None) -> Optional[dict]:
    """Assemble one quote from the daily close series and (optional)
    intraday close series. Returns None when there is not enough data.

    ``official_prev`` is Yahoo's ``regularMarketPreviousClose`` — the official
    prior settlement/close behind its headline change. When present it is the
    baseline, in preference to the daily-history previous close, because
    yfinance's daily ``Close`` is unreliable for some futures: measured, GC=F's
    daily close sat ~1.3% below the true prior settlement (4361.8 vs 4419.7),
    which inflated the strip's daily % by that much versus every site quoting
    Yahoo's headline. Falls back to the daily-history close, then the first
    intraday tick, when the official close is absent (offline runs, or a symbol
    the batch quote did not return)."""
    spark_series = None
    official = official_prev if (official_prev and official_prev > 0) else None
    if intra is not None and len(intra) >= 2:
        cur = float(intra.iloc[-1])
        iday = intra.index[-1].date()
        prev = official
        if prev is None and dclose is not None and len(dclose):
            prior = dclose[[ts.date() < iday for ts in dclose.index]]
            if len(prior):
                prev = float(prior.iloc[-1])
        if prev is None:
            prev = float(intra.iloc[0])
        spark = [float(x) for x in intra.values]
        baseline = prev
        # Keep the timestamped intraday series so the newsletter can draw it
        # on a full-session time axis (line grows through the day).
        spark_series = intra
    elif dclose is not None and len(dclose) >= 2:
        cur = float(dclose.iloc[-1])
        prev = official if official is not None else float(dclose.iloc[-2])
        spark = [float(x) for x in dclose.iloc[-spark_points:].values]
        baseline = spark[0]
    else:
        return None
    change = cur - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return {"value": cur, "change": change, "pct": pct,
            "spark": spark, "baseline": baseline, "spark_series": spark_series}


def broker_1d(
    tickers: list[str],
    *,
    allow_sibling_fallback: bool = True,
) -> dict:
    """Broker-style 1D return per ticker: the latest intraday price vs the
    previous official close, in the instrument's listing currency.

    The portfolio pipeline enables sibling fallback here, once, and retains
    the selected series plus provenance in its run-scoped metrics result.
    ``allow_sibling_fallback=False`` remains available to exact-feed callers.
    This is the "since previous close" figure a broker shows live during the
    session (and the last completed session's change once closed). Returns
    ``{ticker: {"pct": float, "live": bool}}`` only for tickers with a usable
    intraday series (>=2 points); callers fall back to the end-of-day close
    return for the rest. ``live`` is True only when the market is trading
    *now* — i.e. the latest intraday bar is recent. A same-day bar from a
    session that has already closed (e.g. viewed in the evening) is NOT live.
    Best-effort and currency-consistent: both the live price and the previous
    close come from the same native yfinance feed, so for a EUR-listed ETF
    the % is the EUR daily move. Never raises."""
    import pandas as pd
    uniq = [t for t in {t for t in tickers if t}]
    policy = {"policy_id": "broker_1d-v1", "requested_tickers": len(uniq)}
    if not uniq:
        return ProviderResult(
            {}, availability=Availability.AVAILABLE, attempts=(), policy=policy
        )
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception as exc:  # noqa: BLE001
        return ProviderResult(
            {},
            availability=Availability.UNAVAILABLE,
            attempts=(ProviderAttempt(
                source="yfinance",
                operation="broker_1d",
                ordinal=1,
                outcome=f"FAILED:{type(exc).__name__}",
                fallback_rung=0,
            ),),
            policy=policy,
        )
    # Resolve intraday canonical-first with guarded EUR venue fallback. A
    # ``.MI`` holding can use a price-coherent same-root candidate only after
    # its canonical close is available for the collision guard. ``src`` is the
    # listing the series came from, so the previous close is pulled from that
    # SAME feed — keeping ``cur`` and ``prev`` currency-consistent.
    resolved = _resolve_intraday(
        uniq,
        allow_sibling_fallback=allow_sibling_fallback,
    )
    out: dict = {}
    for tk, (intra, src) in resolved.items():
        if intra is None or len(intra) < 2:
            continue
        cur = float(intra.iloc[-1])
        last_ts = intra.index[-1]
        try:
            intraday_observation = pd.Timestamp(last_ts)
            if intraday_observation.tzinfo is None:
                intraday_observation = intraday_observation.tz_localize("UTC")
            else:
                intraday_observation = intraday_observation.tz_convert("UTC")
            intraday_observation_timestamp = (
                intraday_observation.to_pydatetime()
            )
        except (TypeError, ValueError, OverflowError):
            intraday_observation_timestamp = None
        iday = last_ts.date()
        # "live" = the source listing's exchange is in its regular session
        # right now, judged by EXCHANGE HOURS (not bar recency). Uses ``src``
        # so a Milan holding served by its Xetra twin is judged by the venue
        # that actually produced the bars. FX/futures/crypto have no fixed
        # session → market_open_now returns None and we fall back to recency.
        # "now" goes through _intraday_reference_now — the same pinnable seam
        # _has_intraday uses — instead of a bare market_open_now(src) (no
        # now=) or Timestamp.now(): those read the REAL wall clock, so
        # is_live silently depended on what moment this happened to run,
        # not on the series' own reference time. A test built around a
        # fixed historical "now" only gets a deterministic result once this
        # reads through the same pinnable seam.
        now_ref = _intraday_reference_now()
        mkt_open = market_open_now(src, now=now_ref)
        if mkt_open is None:
            try:
                lt = (last_ts.tz_convert("UTC") if getattr(last_ts, "tzinfo", None)
                      else last_ts.tz_localize("UTC"))
                age_min = (now_ref - lt).total_seconds() / 60.0
                is_live = age_min <= _MARKET_OPEN_MAX_LAG_MIN
            except Exception:  # noqa: BLE001
                is_live = False
        else:
            is_live = bool(mkt_open)

        # Preserve the exact series and its own previous-close baseline. The
        # renderer consumes these values directly, so it never has to fetch or
        # re-resolve a venue and cannot drift from the 1D calculation.
        source_official, source_previous_close = _official_and_prev(
            _fetch_history, src, iday
        )
        intraday_baseline = (
            source_previous_close
            if source_previous_close
            else float(intra.iloc[0])
        )

        # --- Closed session: the authoritative 1D move is the instrument's
        # OWN primary-listing official daily close (which includes the closing
        # auction) vs its previous close — exactly what a broker shows for the
        # held position. The sibling series remains the sparkline source. ---
        if not is_live:
            for cand in dict.fromkeys((tk, src)):
                if cand == src:
                    oc, pv = source_official, source_previous_close
                else:
                    oc, pv = _official_and_prev(_fetch_history, cand, iday)
                if oc is not None and pv:
                    out[tk] = {
                        "pct": (oc / pv - 1.0) * 100.0,
                        "live": False,
                        "source_ticker": cand,
                        "intraday_source_ticker": src,
                        "intraday_series": intra,
                        "intraday_baseline": intraday_baseline,
                        "intraday_observation_timestamp": (
                            intraday_observation_timestamp
                        ),
                        "source_reason": (
                            "canonical official close"
                            if cand == tk
                            else "price-coherent EUR venue official-close fallback"
                        ),
                    }
                    break
            if tk in out:
                continue

        # --- Live session (or no official close available yet): the latest
        # intraday tick vs the previous close from the SAME feed, so ``cur``
        # and ``prev`` are currency-consistent. ---
        prev = source_previous_close or float(intra.iloc[0])
        if prev:
            out[tk] = {
                "pct": (cur / prev - 1.0) * 100.0,
                "live": bool(is_live),
                "source_ticker": src,
                "intraday_source_ticker": src,
                "intraday_series": intra,
                "intraday_baseline": prev,
                "intraday_observation_timestamp": (
                    intraday_observation_timestamp
                ),
                "source_reason": (
                    "canonical intraday feed"
                    if src == tk
                    else "price-coherent EUR venue intraday fallback"
                ),
            }
    coverage = len(out) / len(uniq) * 100.0 if uniq else 100.0
    availability = (
        Availability.AVAILABLE if len(out) == len(uniq)
        else Availability.DEGRADED if out
        else Availability.UNAVAILABLE
    )
    return ProviderResult(
        out,
        availability=availability,
        attempts=(ProviderAttempt(
            source="yfinance",
            operation="broker_1d",
            ordinal=1,
            outcome="SUCCEEDED" if out else "UNAVAILABLE",
            fallback_rung=0,
            coverage_pct=coverage,
        ),),
        policy=policy,
        selected_source="yfinance" if out else None,
    )


def fetch_market_quotes(force: bool = False) -> list[dict]:
    """Fetch the curated market quotes (memoised per process). Best-effort:
    returns whatever could be fetched; never raises."""
    global _memo, _memo_at
    if _memo is not None and not force and (_time.monotonic() - _memo_at) < _MEMO_TTL_SECONDS:
        return _memo
    try:
        from tarzan.data.enricher import _fetch_history
    except Exception as e:  # noqa: BLE001
        logger.debug("market quotes unavailable (%s)", e)
        return []

    symbols = [s for _, s, _ in MARKETS]
    intraday = _fetch_intraday(symbols)
    prev_closes = _fetch_official_prev_closes(symbols)
    out: list[dict] = []
    for name, symbol, category in MARKETS:
        try:
            hist = _fetch_history(symbol)
            dclose = (hist["Close"].dropna()
                      if hist is not None and len(hist) and "Close" in getattr(hist, "columns", [])
                      else None)
            intra = intraday.get(symbol)
            # A strip sparkline must show ONE real session, never a prior one
            # dressed up as today's. At (or just after) the open the 15m feed
            # has no today bars yet, so _fetch_intraday collapses to yesterday's
            # full session — which would render full-width next to a live "Op."
            # badge. The same freshness gate the holdings path uses (broker_1d /
            # _resolve_intraday) rejects a session that's stale while the market
            # is open, so _quote falls back to the daily-close chart ("no
            # intraday"). Fresh intraday (mid-session) and a genuinely completed
            # last session (market closed) both still pass.
            if intra is not None and not _has_intraday(intra, symbol):
                intra = None
            q = _quote(dclose, intra, official_prev=prev_closes.get(symbol))
            if q is None:
                continue
            out.append({"name": name, "symbol": symbol, "category": category, **q})
        except Exception as e:  # noqa: BLE001
            logger.debug("market quote %s failed: %s", symbol, e)
            continue

    _memo = out
    _memo_at = _time.monotonic()
    return out
