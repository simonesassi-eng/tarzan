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
# This table is also the calibration knob for session_day(), so it decides
# which observations are usable, not just which caption is printed: modelling
# EU as 09:00 means a German venue's genuine 08:00-09:00 pre-market bars count
# towards the PREVIOUS session on a pre-09:00 run. Conservative and
# self-consistent — widen the open here if pre-market should count.
_SESSIONS: dict[str, tuple[str, tuple[int, int], tuple[int, int]]] = {
    "EU": ("Europe/Rome", (9, 0), (17, 30)),      # Milan/Xetra/Paris/Amsterdam
    "L":  ("Europe/London", (8, 0), (16, 30)),    # London
    "US": ("America/New_York", (9, 30), (16, 0)),  # US cash session
    "JP": ("Asia/Tokyo", (9, 0), (15, 0)),        # Tokyo
    "HK": ("Asia/Hong_Kong", (9, 30), (16, 0)),   # Hong Kong
    "CN": ("Asia/Shanghai", (9, 30), (15, 0)),    # Shanghai
    "AU": ("Australia/Sydney", (10, 0), (16, 0)),  # Sydney
    "KR": ("Asia/Seoul", (9, 0), (15, 30)),       # Seoul
    # German regional/retail venues (Munich, Stuttgart, Berlin, Düsseldorf,
    # Hamburg, Hanover, Tradegate), which really do quote 08:00–22:00 — an ETF
    # whose only Yahoo listing is one of these prints hours before and after
    # Xetra, and calling that "the previous session" would discard most of it.
    "DE_REG": ("Europe/Berlin", (8, 0), (22, 0)),
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

# Yahoo listing suffix → exchange group. This must cover EVERY venue the ISIN
# resolver can settle on (config's isin_exchange_suffixes) and every sibling it
# can fall back to: a suffix missing from here has no session at all, so
# session_day() returns None and the instrument silently reverts to the
# "whatever day the data ends on" convention this module exists to replace.
# That gap is how one Munich-listed ETF (IS39.MU) carried the previous
# session's bars into the 09:09 issue and flipped a whole column to "Intraday".
# test_market_session_alignment asserts the coverage so the lists cannot drift.
_SUFFIX_EXCHANGE: dict[str, str] = {
    "MI": "EU", "DE": "EU", "PA": "EU", "AS": "EU", "F": "EU",
    "ETLX": "EU", "BR": "EU", "LS": "EU", "MC": "EU", "VI": "EU", "SW": "EU",
    "MU": "DE_REG", "SG": "DE_REG", "BE": "DE_REG", "DU": "DE_REG",
    "HM": "DE_REG", "HA": "DE_REG", "TG": "DE_REG",
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
        n = (now if now is not None else _intraday_reference_now()).astimezone(tz)
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
        n = (now if now is not None else _intraday_reference_now()).astimezone(tz)
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


def _session_date(n: datetime, oh: int, om: int):
    """The exchange-local DATE whose cash session is live at ``n``, or was the
    most recently started one: today once today's own session has opened,
    otherwise the previous trading day (walking back over a weekend one
    weekday further). ``n`` must already be in exchange-local time."""
    from datetime import timedelta
    d = n
    if not (d.weekday() < 5 and d.time() >= dtime(oh, om)):
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d = d - timedelta(days=1)
    return d.date()


def _cash_session_day(n: datetime, oh: int, om: int) -> str:
    """The weekday label of :func:`_session_date` — the caption form."""
    return _WD[_session_date(n, oh, om).weekday()]


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
        n = (now if now is not None else _intraday_reference_now()).astimezone(tz)
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
            n = (now if now is not None else _intraday_reference_now()).astimezone(tz)
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
        n = (now if now is not None else _intraday_reference_now()).astimezone(tz)
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
    """The ONE reference instant for every market-data freshness/session
    decision in a run, tz-aware. A thin, monkeypatchable seam (mirrors the
    ``now=`` pattern used by ``market_open_now`` elsewhere in this module) so
    tests can pin "now" without threading a parameter through the whole
    intraday call chain.

    It reads the run clock, not the wall clock: LIVE runs use the run's own
    capture instant, so every section of one issue judges "is this market open"
    and "is this bar current" against the same moment instead of each drifting
    to whenever it happened to execute. A POINT_IN_TIME / REPRODUCIBLE run gets
    the END of its effective date, which is what makes the no-observation-later-
    than-``as_of`` invariant expressible as a single comparison. Falls back to
    the wall clock only if the run context cannot be read.
    """
    from datetime import timezone as _tz
    tz = tzinfo or _tz.utc
    try:
        from tarzan import runtime
        ctx = runtime.context()
        if ctx.effective_date is not None:
            return datetime.combine(
                ctx.effective_date, dtime.max, tzinfo=_tz.utc).astimezone(tz)
        return ctx.captured_at.astimezone(tz)
    except Exception:  # noqa: BLE001 — a clock must never break a render
        return datetime.now(tz)


def _exchange_tz(ticker: str):
    """The IANA zone of the instrument's exchange, or None when no cash
    session is modelled for it (futures/FX/crypto/unknown venue)."""
    ex = _exchange_for((ticker or "").upper())
    if ex is None or ex not in _SESSIONS:
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(_SESSIONS[ex][0])
    except Exception:  # noqa: BLE001
        return None


def session_day(ticker: str, now: Optional[datetime] = None):
    """The exchange-local DATE of the cash session that is live right now, or
    of the most recently started one.

    This is the authority for "which session do today's figures belong to",
    and it is deliberately separate from :func:`market_open_now` ("is the
    venue trading"). Minutes after an open the two disagree in the way that
    matters: the venue is open, but no observation from that session exists
    yet, and a figure carried over from the previous session must not be
    presented as this one's. ``None`` when no cash session is modelled
    (futures/FX/crypto), where the caller keeps its own day convention.
    """
    ex = _exchange_for((ticker or "").upper())
    if ex is None or ex not in _SESSIONS:
        return None
    tzname, (oh, om), _close = _SESSIONS[ex]
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
        n = (now or _intraday_reference_now()).astimezone(tz)
    except Exception:  # noqa: BLE001
        return None
    return _session_date(n, oh, om)


def session_span(ticker: str, now: Optional[datetime] = None):
    """``(open, close)`` instants of the session :func:`session_day` names,
    tz-aware, or ``None`` when no cash session is modelled.

    This is the x-axis a session chart must be drawn on. Spreading N bars evenly
    across the full width instead — which is what a sparkline does by default —
    says "this is a whole session" no matter how little of one the bars cover:
    three prints from an illiquid ETF's first hour came out shaped like a
    completed day. Placing each bar at its real offset in [open, close] makes a
    quiet morning look quiet and a finished session fill the width, from the
    same code path."""
    ex = _exchange_for((ticker or "").upper())
    if ex is None or ex not in _SESSIONS:
        return None
    tzname, (oh, om), (ch, cm) = _SESSIONS[ex]
    day = session_day(ticker, now)
    if day is None:
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname)
        return (datetime.combine(day, dtime(oh, om), tzinfo=tz),
                datetime.combine(day, dtime(ch, cm), tzinfo=tz))
    except Exception:  # noqa: BLE001
        return None


def _clip_to_reference(ser, now, tz=None):
    """Drop observations later than ``now`` — the run's reference instant.

    The invariant is that nothing an issue prints can post-date its own
    ``as_of``. A vendor window is requested by period ("5d"), not by bound, so
    under ``--as_of`` it happily returns bars from after the effective date;
    clipping here, at the one place intraday series enter the system, is what
    keeps every downstream section on the same side of the boundary. A tz-naive
    index is read as exchange-local time (``tz``), which is what such an index
    means when the vendor or a fixture supplies one."""
    try:
        import pandas as pd

        ref = pd.Timestamp(now)
        if getattr(ser.index, "tz", None) is None:
            if ref.tzinfo is not None:
                ref = ref.tz_convert(tz) if tz is not None else ref.tz_convert("UTC")
                ref = ref.tz_localize(None)
        elif ref.tzinfo is None:
            ref = ref.tz_localize("UTC")
        return ser[ser.index <= ref]
    except Exception:  # noqa: BLE001 — never drop a series over a clip
        return ser


def _select_session_series(ser, ticker: str, now: Optional[datetime] = None):
    """Narrow an intraday close series to ONE session: the current one.

    ``ser`` arrives as several days of bars (the vendor is asked for a window
    wide enough to survive a long weekend). Picking "the last calendar day
    present" — which is what this used to do, in UTC — silently returns the
    PREVIOUS session whenever the current one has not printed yet, which is
    every run in the first ~half hour after an open. Every consumer then treats
    yesterday's completed session as today's, so a close-to-close return gets
    rendered under an "Intraday" heading. Selecting by the exchange's own
    session date instead means "not yet" comes back as absent, which callers
    already handle honestly.

    Bars are also matched in the EXCHANGE's timezone, not UTC: a session that
    straddles UTC midnight (Sydney on summer time) would otherwise be cut in
    half. Returns an empty series when the current session has fewer than two
    bars. Continuous markets (futures/FX/crypto) have no cash session, so they
    keep the last-day-present convention."""
    now = now or _intraday_reference_now()
    tz = _exchange_tz(ticker)
    ser = _clip_to_reference(ser, now, tz)
    if ser is None or len(ser) < 2:
        return ser[:0] if ser is not None else ser
    # No cash session modelled (futures/FX/crypto) → session_day is None and
    # this falls back to the last day present, their own day convention.
    day = session_day(ticker, now) or _observed_day(ser.index[-1], tz)
    return ser[[_observed_day(ts, tz) == day for ts in ser.index]]


def _observed_day(ts, tz=None):
    """The session date an observation belongs to, in the exchange's own
    timezone.

    A bar timestamped in UTC lands on the PREVIOUS UTC date for any venue far
    enough east — Sydney on summer time opens at 23:00 UTC — so reading its
    plain ``.date()`` dates a live session one day early, and every comparison
    against :func:`session_day` (also exchange-local) then calls it stale. Every
    reader of "which session is this observation from" goes through here so the
    two can only ever disagree about a real difference."""
    try:
        return (ts.astimezone(tz) if tz is not None and ts.tzinfo is not None
                else ts).date()
    except Exception:  # noqa: BLE001
        return ts.date()


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
                # its last completed session. _select_session_series then
                # keeps exactly the CURRENT session out of whatever comes
                # back, so a wider window never means a multi-day chart,
                # only a better chance of finding a single real one.
                return yf.download(symbols, period="5d", interval="15m",
                                   group_by="ticker", progress=False, threads=True)
        # Shared spacing+retry so the intraday batch survives a 429 burst.
        raw = _yf_net.fetch_yf(_download, what="intraday batch", log=logger)
        if raw is None or len(raw) == 0:
            return {}
        # One instant for the whole batch: every symbol's session and freshness
        # is judged against the same moment, so two strips in one issue cannot
        # disagree about what "now" is.
        now = _intraday_reference_now()
        level0 = set(raw.columns.get_level_values(0)) if hasattr(raw.columns, "get_level_values") else set()
        for s in symbols:
            try:
                if s in level0 and "Close" in raw[s].columns:
                    cl = _select_session_series(raw[s]["Close"].dropna(), s, now)
                    if cl is not None and len(cl) >= 2:
                        out[s] = cl
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
    """``{symbol: regularMarketPreviousClose}`` — see ``_fetch_official_quotes``."""
    return {
        symbol: quote["prev_close"]
        for symbol, quote in _fetch_official_quotes(symbols).items()
        if quote.get("prev_close")
    }


_quote_memo: dict[str, dict] = {}


def reset_quote_memo() -> None:
    """Drop the run-scoped quote pairs (mirrors ``enricher.reset_run_caches``)."""
    _quote_memo.clear()


def official_quotes(symbols: list[str]) -> dict:
    """``{symbol: {"price", "prev_close"}}`` for ``symbols``, fetched once per run.

    The published pair is the ONE authority for an instrument's current price
    and the close before it, so the current point of a price series, the 1D of
    that series and the portfolio's own valuation cannot come from three
    different snapshots of two different feeds. Batched: only symbols not
    already memoized reach the network.
    """
    missing = [s for s in dict.fromkeys(symbols) if s and s not in _quote_memo]
    if missing:
        fetched = _fetch_official_quotes(missing)
        for symbol in missing:
            # Cache the miss too: one attempt per symbol per run.
            _quote_memo[symbol] = fetched.get(symbol, {})
    return {s: _quote_memo[s] for s in dict.fromkeys(symbols)
            if s and _quote_memo.get(s)}


def _fetch_official_quotes(symbols: list[str]) -> dict:
    """One batched Yahoo v7 quote call → ``{symbol: {"price", "prev_close"}}``
    (positive floats only; either key may be absent).

    This pair IS the instrument's headline change on its own Yahoo page, and it
    comes from the quote pipeline rather than the daily chart — which for some
    listings drops a whole session: on 18 Aug 2026 the chart had a null close
    for Monday 17 on every Milan ETF while this endpoint carried the official
    40.815 for EXUS.MI. Measuring the 1D against it is therefore both the
    robust choice (a second, independent vendor path) and the aligned one.

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

        payload = _yf_net.fetch_yf(_get, what="markets quote batch", log=logger)
        if not payload:
            return {}
        for q in payload.get("quoteResponse", {}).get("result", []):
            sym = q.get("symbol")
            if not sym:
                continue
            fields: dict = {}
            for key, name in (("price", "regularMarketPrice"),
                              ("prev_close", "regularMarketPreviousClose")):
                try:
                    value = q.get(name)
                    if value is not None and float(value) > 0:
                        fields[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if fields:
                out[sym] = fields
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
           official_prev: Optional[float] = None,
           current_session_day=None, session_tz=None) -> Optional[dict]:
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
    the batch quote did not return).

    ``current_session_day`` is :func:`session_day` for the symbol — the session
    the run is IN. It is what makes ``official_prev`` safe to use: that field is
    the close before the CURRENT session, so pairing it with a level from an
    earlier session measures nothing. At the London open with no Thursday bars
    yet, Wednesday's close paired with Wednesday's ``previousClose`` printed
    "+0.00%"; the same level paired with Tuesday's close prints Wednesday's
    real move, which is what the figure actually is. ``None`` for continuously
    traded instruments (futures/FX/crypto), which have no cash session and keep
    ``official_prev`` unconditionally. ``session_tz`` is that exchange's zone,
    so the level is dated in the same timezone as ``current_session_day`` and
    the two are comparable. The returned ``observed_day`` states which session
    the level belongs to, so callers can label it honestly instead of assuming
    it is today's."""
    spark_series = None
    official = official_prev if (official_prev and official_prev > 0) else None

    def _close_before(day):
        if dclose is None or not len(dclose) or day is None:
            return None
        prior = dclose[[ts.date() < day for ts in dclose.index]]
        return float(prior.iloc[-1]) if len(prior) else None

    if intra is not None and len(intra) >= 2:
        cur = float(intra.iloc[-1])
        observed_day = _observed_day(intra.index[-1], session_tz)
        spark = [float(x) for x in intra.values]
        # Keep the timestamped intraday series so the newsletter can draw it
        # on a full-session time axis (line grows through the day).
        spark_series = intra
        fallback = float(intra.iloc[0])
    elif dclose is not None and len(dclose) >= 2:
        cur = float(dclose.iloc[-1])
        observed_day = dclose.index[-1].date()
        spark = [float(x) for x in dclose.iloc[-spark_points:].values]
        fallback = float(dclose.iloc[-2])
    else:
        return None

    # ``official_prev`` belongs to the current session, so a level from an
    # earlier one is measured against the close before ITS OWN session instead.
    stale_session = bool(current_session_day is not None
                         and observed_day < current_session_day)
    prev = None if stale_session else official
    if prev is None:
        prev = _close_before(observed_day)
    if prev is None:
        prev = fallback
    baseline = prev if spark_series is not None else spark[0]
    change = cur - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return {"value": cur, "change": change, "pct": pct,
            "spark": spark, "baseline": baseline, "spark_series": spark_series,
            "observed_day": observed_day, "stale_session": stale_session}


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
    # Yahoo's own headline pair for every requested ticker, batched once. It is
    # the authority for the 1D: same two numbers the instrument's page shows,
    # and sourced from the quote pipeline rather than the daily chart, which can
    # be missing the very session being measured.
    official = _fetch_official_quotes(uniq)
    out: dict = {}
    for tk in uniq:
        intra, src = resolved.get(tk, (None, tk))
        quote = official.get(tk) or {}
        q_price, q_prev = quote.get("price"), quote.get("prev_close")
        if intra is None or len(intra) < 2:
            # No usable intraday series: the quote pair alone still yields the
            # published 1D (live during the session, the last completed
            # session's move outside it), where before the row fell back to a
            # close-to-close read of a chart that may have a hole in it.
            if q_price and q_prev:
                is_open = market_open_now(tk, now=_intraday_reference_now())
                out[tk] = {
                    "pct": (q_price / q_prev - 1.0) * 100.0,
                    "live": bool(is_open),
                    "source_ticker": tk,
                    "intraday_source_ticker": tk,
                    "intraday_series": None,
                    "intraday_baseline": q_prev,
                    "intraday_observation_timestamp": None,
                    "source_reason": "yahoo official quote (no intraday feed)",
                }
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
        # Dated in the SOURCE VENUE's timezone: the daily history this indexes
        # into is keyed by exchange-local dates, so a UTC read would look up
        # the wrong session's official close for any venue east of UTC.
        iday = _observed_day(last_ts, _exchange_tz(src))
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

        # Yahoo's published pair wins for the percentage: for the canonical
        # listing it IS the figure on the instrument's page, in both session
        # states, and it survives a daily chart that lost the session being
        # measured (17 Aug 2026 on every Milan ETF). The intraday series stays
        # as resolved so the sparkline still comes from the venue that produced
        # the bars; its baseline follows the same close as the percentage
        # whenever the bars are the canonical listing's own.
        if q_price and q_prev:
            out[tk] = {
                "pct": (q_price / q_prev - 1.0) * 100.0,
                "live": bool(is_live),
                "source_ticker": tk,
                "intraday_source_ticker": src,
                "intraday_series": intra,
                "intraday_baseline": q_prev if src == tk else intraday_baseline,
                "intraday_observation_timestamp": intraday_observation_timestamp,
                "source_reason": "yahoo official quote (price vs previous close)",
            }
            continue

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
    # One instant for the whole strip, so two rows cannot disagree about which
    # session is current.
    now = _intraday_reference_now()
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
            q = _quote(dclose, intra, official_prev=prev_closes.get(symbol),
                       current_session_day=session_day(symbol, now),
                       session_tz=_exchange_tz(symbol))
            if q is None:
                continue
            out.append({"name": name, "symbol": symbol, "category": category, **q})
        except Exception as e:  # noqa: BLE001
            logger.debug("market quote %s failed: %s", symbol, e)
            continue

    _memo = out
    _memo_at = _time.monotonic()
    return out
