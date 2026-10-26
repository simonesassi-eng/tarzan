"""The ONE place today's market point is written.

Every "today" figure in Tarzan reads a price series: the hero chart and the
value/P&L series through ``PriceResolver``, the window matrix and the Returns
tables through ``compute_period_return``. The portfolio's own valuation, though,
is selected by the capability policy from whatever price enrichment settled on —
so "today" existed twice, and the two disagreed whenever one of them fell back.

Measured on a live run (24 Aug 2026, 16 holdings): fourteen agreed to four
decimals, and the two that did not were both fallbacks. MONEY.MI's valuation had
dropped to its 10.0920 order price while the market quoted 10.1840 (+0.91%,
€88); NTSG.MI's valuation sat at 29.2843 against a validated 29.0950 (−0.65%,
€203). Net effect: the hero printed a portfolio total beside P&L tiles that
implied one €115 lower, from a silent stale order price.

So there is one rule, applied once, here:

* today's point is the clean market QUOTE, validated against the series' OWN
  last real close — never a valuation that may have fallen back to an order
  price (stamping that put a 10.09 endpoint against a 10.18 anchor and printed
  a spurious −0.92% 5D);
* the previous session's close is the published ``prev_close``, scaled onto that
  same ruler (:func:`prev_close_eur`) and written ONLY onto that session's row
  (:func:`stamp_today`), so a vendor's missing bar is repaired without
  fabricating a close on a date it does not belong to;
* the holding's ``current_price``/``current_value`` are updated from the same
  number, so the valuation policy that runs next judges the price the series
  actually ends on.

Ordering is the point. This runs in the data layer, BEFORE
``ValuationCompletenessEvaluator``: the policy that decides whether a price is
trustworthy has to see the final price, not the one that gets overwritten
afterwards. When this lived in ``MetricsEngine`` it ran after both the policy
and ``_valuation``, which is why ``total_value`` and the series terminal could
not agree by construction.

Pinned runs (``--as_of`` / ``--deterministic``) stamp nothing: no live
observation may enter a reproducible run.

Weekend runs DO stamp. What may never be written is a point on a date that is not
a session — appending a Saturday-dated point slides ``window_anchor`` onto the day
a month before the WEEKEND rather than the last session — and that is now checked
per venue against the exchange calendar (:func:`stamp_date`) instead of by a
weekday test on the run's own clock. Refusing the whole weekend also refused a
FRIDAY close that the quote endpoint was carrying while the daily-bar feed was
not, which is how the Sat 29 Aug 2026 digest reported Thursday's session: Yahoo
held a 28 Aug row with a null close for all sixteen holdings, and every quote
carried the real Friday close. On a weekday that gap self-heals through the stamp,
so it surfaced only on the Saturday issue.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timezone
from typing import Optional

import pandas as pd

from tarzan.data.referee import (
    CURRENT_SESSION_BASELINE_ATTR, finite_number, observation_date, quote_pair,
)
from tarzan.models.currency import major_code, to_major

logger = logging.getLogger(__name__)


#: How old a quote's OWN observation may be and still be read as current.
#:
#: The level gate below cannot see time, and that is not a theoretical gap. NTSG.MI's
#: quote endpoint has been serving a price observed on 10 Oct 2025 — on 17 Sep 2026 that
#: is 342 days old. It was caught for a year only because the stale 25.515 sat >10% away
#: from the fund's real level; as the real price drifted down to 28.345 the deviation
#: became 9.98% and the quote passed a 10% tolerance by 0.02pp. What it would have priced
#: is the CURRENT VALUATION (``quantity * price``), ~10% under, and stamped a point on a
#: session eleven months back.
#:
#: Seven days, not one: a thinly traded listing can go a long weekend plus a holiday
#: without a print, and rejecting those would stop stamping instruments whose quote is
#: real and merely infrequent. It is two orders of magnitude short of the failure it is
#: here to catch, which is the margin that matters.
_MAX_QUOTE_AGE_DAYS = 7


#: A venue observed this long before the freshest venue of the same instrument is
#: superseded: its quote is older information about the same price. On 8 Oct 2026 the
#: off-market Xetra print was observed at 09:21 while Paris had been trading since;
#: on a thin line the gap between two venues' last trades is routinely an hour.
_CONSENSUS_WINDOW_MINUTES = 60

#: Two venues' 1D for the same instrument and session may differ by this much (in pp)
#: before the figure is DISPUTED: the larger of a 1pp floor and 1.5 of the instrument's
#: own daily sigmas. Honest cross-venue noise sits well inside it — over 462 shared
#: sessions one fund's Milan and Xetra closes differed by a sd of 0.35% — and the three
#: disputes measured on 8 Oct 2026 sat far outside it (4.31, 1.55 and 2.73pp).
_MOVE_TOLERANCE_FLOOR_PP = 1.0
_MOVE_TOLERANCE_SIGMAS = 1.5

#: An uncorroborated move beyond this many of the instrument's OWN daily standard
#: deviations is treated as a bad print rather than as the market. Five, so a real
#: crash day on a diversified fund still passes (5 sigmas of a 14%-vol fund is -4.4%),
#: and a single fat-finger on a thin venue does not.
_MAX_UNCONFIRMED_SIGMAS = 5.0


def _daily_sigma(history) -> Optional[float]:
    """The instrument's own daily return standard deviation (a fraction), or None
    when there is too little history — under a month — to measure one."""
    if history is None:
        return None
    try:
        r = pd.Series(history).dropna().astype(float).pct_change().dropna().tail(252)
    except Exception:  # noqa: BLE001 — a sanity check must never break a stamp
        return None
    if len(r) < 20:
        return None
    sigma = float(r.std())
    return sigma if math.isfinite(sigma) and sigma > 0 else None


def _move(quote: dict) -> Optional[float]:
    price = finite_number(quote.get("price"), positive=True)
    prev = finite_number(quote.get("prev_close"), positive=True)
    if price is None or prev is None:
        return None
    return finite_number(price / prev - 1.0)


#: How far a venue's published previous close may sit from an INDEPENDENT previous close
#: of the same session before it stops being a valid 1D baseline. Honest cross-venue
#: noise on closes has a sd of 0.35%; the fictitious Xetra close of 8 Oct 2026 sat 1.19%
#: from the traded one.
_BASELINE_TOLERANCE = 0.0075


def pick_quote(symbols: list[str], quotes: dict, reference_price: float,
               history=None, referee: Optional[dict] = None) -> dict:
    """The quote that today's valuation, stamp and 1D are read from — or ``{}``.

    Four gates, in order, and every one of them was added after a real issue went out
    wrong:

    1. **Recent** — its own observation is within ``_MAX_QUOTE_AGE_DAYS``. A dead feed
       served a price observed 342 days earlier and, once the real price drifted close
       enough, it arrived as today's valuation.
    2. **Coherent** — its level agrees with ``reference_price`` (the instrument's own
       last real close) within the sibling tolerance. Rejects a corrupt feed priced on
       a different scale.
    3. **Consensus** — among the venues observed in the newest session, one observed
       more than ``_CONSENSUS_WINDOW_MINUTES`` before the freshest is superseded. If the
       rest disagree on the 1D beyond the move tolerance, the figure is disputed: with
       three or more venues the MEDIAN mover is kept, with two the venue that has
       TRADED more today (``_better_evidenced``). A genuine move shows on every venue
       and never trips this.
    4. **Corroborated** — with ``history`` given, a move beyond
       ``_MAX_UNCONFIRMED_SIGMAS`` of the instrument's own daily returns is accepted
       only if a second same-session venue confirms it. Uncorroborated, it is rejected:
       the holding keeps its last close and the issue's "priced today" coverage says
       so, an honest gap instead of a wrong number.

    Without a dispute the caller's priority order decides among the newest
    observed session's peers. An older or undated preferred venue cannot
    supersede a dated newer session. Undated quotes remain usable on level
    alone only when no coherent dated quote is available.

    Returns ``{}`` when nothing qualifies, so the caller keeps the feed's own close.

    Both quote and reference must be on the same currency/unit ruler. The
    holdings adapter converts provider quotes before calling this selector;
    proximity of USD and EUR levels is never evidence of equal units.
    """
    from tarzan.data.market_quotes import _SIBLING_PRICE_TOLERANCE

    ref = finite_number(reference_price, positive=True)
    if ref is None:
        return {}
    eligible: list[tuple[str, dict]] = []
    for symbol in dict.fromkeys(symbols):
        quote = quotes.get(symbol) or {}
        native = finite_number(quote.get("price"), positive=True)
        if native is None or quote_is_stale(quote):
            continue
        if abs(native / ref - 1.0) <= _SIBLING_PRICE_TOLERANCE:
            eligible.append((symbol, quote))
    if not eligible:
        return {}

    observed = {sym: quote_observed_at(q) for sym, q in eligible}
    newest_day = max((o.date() for o in observed.values() if o is not None), default=None)
    same_session = [(sym, q) for sym, q in eligible
                    if observed[sym] is not None and observed[sym].date() == newest_day]
    freshest = max((observed[sym] for sym, _q in same_session), default=None)
    peers = [(sym, q) for sym, q in same_session
             if (freshest - observed[sym]).total_seconds() / 60.0
             <= _CONSENSUS_WINDOW_MINUTES]
    current_venues = {sym for sym, _q in peers}

    # Priority applies within one observed session, not across vintages. A
    # thin primary venue may have no trade today while a sibling does; keeping
    # yesterday's primary first would silently discard that current evidence.
    # Unknown-time compatibility quotes cannot prove a newer session either.
    ranked = [
        (sym, q) for sym, q in eligible
        if not current_venues or sym in current_venues
    ]

    # An independent previous close for the same session judges every venue's 1D
    # BASELINE, not only in a dispute. A fictitious close inside the move tolerance
    # passed silently: on 9 Oct 2026 Xetra's published close was 29.28 for a session
    # whose last trade was 28.995 (the independent one: 28.94), the 1D stayed within
    # 1.1pp of the other venue's, and the issue measured from a number nobody dealt at.
    # A NAV is not a market price, so a referee that is one does not judge baselines.
    baselines = {
        sym: _independent_baseline(
            referee, newest_day, _native_previous_date(q, sym, newest_day)
        )
        for sym, q in ranked
    }
    baseline = next((value for value in baselines.values() if value is not None), None)
    if baseline is not None:
        sound = [(s, q) for s, q in ranked
                 if baselines[s] is None
                 or finite_number(q.get("prev_close"), positive=True) is None
                 or abs(float(q["prev_close"]) / baselines[s] - 1.0) <= _BASELINE_TOLERANCE]
        if sound and len(sound) < len(ranked):
            logger.warning(
                "Baseline of %s rejected: its previous close disagrees with the "
                "independent %.4f", ", ".join(s for s, _q in ranked if (s, _q) not in sound),
                baseline)
            ranked = sound
            peers = [(s, q) for s, q in peers if (s, q) in sound]
        elif not sound:
            # The only venue's baseline is the broken one. Keep its price — it is the
            # market — and measure from the independent close instead.
            logger.warning("Baseline of %s replaced by the independent %.4f",
                           ranked[0][0], baseline)
            ranked = [(ranked[0][0], dict(ranked[0][1], prev_close=baseline))]
            peers = [(s, q) for s, q in peers if s != ranked[0][0]] + [ranked[0]]
    chosen_sym, chosen = ranked[0]

    sigma = _daily_sigma(history)
    tolerance = max(_MOVE_TOLERANCE_FLOOR_PP / 100.0,
                    _MOVE_TOLERANCE_SIGMAS * sigma if sigma else 0.0)
    moves = {sym: _move(q) for sym, q in peers}
    voting = [(sym, q) for sym, q in peers if moves[sym] is not None]
    if len(voting) >= 2:
        spread = max(moves[s] for s, _q in voting) - min(moves[s] for s, _q in voting)
        if spread > tolerance:
            if len(voting) >= 3:
                ordered = sorted(voting, key=lambda sq: moves[sq[0]])
                chosen_sym, chosen = ordered[(len(ordered) - 1) // 2]
            else:
                chosen_sym, chosen = _better_evidenced(voting, moves, referee, newest_day)
            logger.warning(
                "Venues dispute the 1D of %s (%s; spread %.2fpp > %.2fpp); kept %s",
                symbols[0] if symbols else "?",
                ", ".join(f"{s} {moves[s] * 100:+.2f}%" for s, _q in voting),
                spread * 100, tolerance * 100, chosen_sym)

    if sigma is not None and not _move_is_plausible(
            chosen, chosen_sym, voting, moves, sigma, tolerance):
        return {}
    return chosen


def _native_previous_date(quote: dict, symbol: str, session):
    """The previous close's native observation, never the run's receipt time."""
    day = observation_date(session)
    if day is None:
        return None
    if "prev_date" in quote:
        previous = observation_date(quote["prev_date"])
        return previous if previous is not None and previous < day else None
    # Yahoo's published previous close names the previous trading session. Use its
    # own calendar only for a dated quote on a venue the bundled calendar knows.
    if quote_observed_at(quote) is None:
        return None
    from tarzan.data.exchange_calendar import covers, is_session, previous_session

    if not covers(symbol, day) or not is_session(symbol, day):
        return None
    return previous_session(symbol, day)


def _independent_baseline(referee: Optional[dict], session, anchor=None) -> Optional[float]:
    """The referee's previous close, when it can judge this session's baselines.

    Only a MARKET quote for the same session can: justETF serves the issuer's NAV for a
    fund that does not trade on Xetra (``venue`` "NAV", dated days back), and a NAV is a
    different measure from any venue's close.
    """
    r = referee or {}
    if str(r.get("venue") or "").strip().upper() == "NAV":
        return None
    pair = quote_pair(r, session=session, anchor=anchor)
    return None if pair is None else pair[1]


def _better_evidenced(voting, moves: dict, referee: Optional[dict] = None,
                      session=None):
    """The better-evidenced of two venues that dispute an instrument's 1D.

    ``pick_quote`` has already filtered a previous close against a comparable
    independently dated baseline. This helper chooses among the remaining venues;
    ``referee`` and ``session`` remain in its signature for caller compatibility.
    The filter must precede volume: later on 8 Oct 2026 the venue with the fictitious
    29.355 close had out-traded the other, so "busier" picked the broken baseline
    (-1.23% against a real -0.28%) — while the referee's traded close was 29.01,
    exactly the other venue's.

    The one that traded more TODAY. Either end of a venue's 1D can be wrong on a thin
    line, and the three disputes measured on 8 Oct 2026 had their error at different
    ends — which is why neither "the smaller move" nor "the more liquid venue on
    average" survives all three, while today's volume does:

    * 11:08, a single off-market trade at 09:21 on one venue (-4.38%) against a venue
      that had kept trading (-0.07%): the busier venue is right.
    * 11:18, the quieter venue's PREVIOUS CLOSE was not a traded price — 29.355 against
      a session that traded only between 28.985 and 29.160 — so its -1.67% was
      measured from a number nobody dealt at; the other venue (473 vs 403 traded)
      said -0.12%. The busier venue is right.
    * a second fund, the other way round: the venue showing the SMALLER move (-0.92%)
      had a previous close made of ONE share at the open, while the larger mover
      (-3.65%) had traded 898 units through its whole previous session. The busier
      venue (188 vs 175 today) is right, and "the smaller move" would have been wrong.

    Volume missing on either side falls back to the smaller move, the weaker rule, so
    the choice is still deterministic. Ties keep the caller's priority order.
    """
    vols = [q.get("volume") for _s, q in voting]
    if all(v is not None for v in vols) and vols[0] != vols[1]:
        return max(voting, key=lambda sq: float(sq[1]["volume"]))
    return min(voting, key=lambda sq: abs(moves[sq[0]]))


def _move_is_plausible(quote: dict, symbol: str, voting, moves: dict,
                       sigma: float, tolerance: float) -> bool:
    """Gate 4 of :func:`pick_quote`: is this quote's own published move believable?

    Believable when it is within ``_MAX_UNCONFIRMED_SIGMAS`` of the instrument's own
    daily returns, OR when another same-session venue's move confirms it within the
    move tolerance. Never blocks on missing evidence: no previous close reads as
    plausible — this gate exists to stop a bad print, not to stop stamping.
    """
    move = _move(quote)
    if move is None or abs(move) <= _MAX_UNCONFIRMED_SIGMAS * sigma:
        return True
    if any(s != symbol and abs(moves[s] - move) <= tolerance for s, _q in voting):
        return True
    logger.warning(
        "Rejected an unconfirmed %+.2f%% print on %s (%.1f sigmas, no second venue "
        "agrees); the holding keeps its last close", move * 100, symbol, abs(move) / sigma)
    return False


def quote_is_stale(quote: dict) -> bool:
    """Whether a dated quote is too old or not yet observable as current.

    False when the quote carries no timestamp — unknown age is not evidence of age,
    and the caller's own checks still have to pass. A malformed present timestamp
    is rejected, not stripped to create unknown-age evidence. Dated against ``runtime.today()``
    rather than the wall clock so a pinned run stays reproducible.

    Public because two layers need the SAME rule: this module's level gate, which
    decides what today's price is, and the enricher's ISIN ranking, which decides
    which listing an instrument is read from at all. One threshold, one definition.
    """
    observed = quote_observed_at(quote)
    if observed is None:
        return quote.get("time") is not None
    if quote_observation_is_future(observed):
        return True
    try:
        from tarzan import runtime

        return (runtime.today() - observed.date()).days > _MAX_QUOTE_AGE_DAYS
    except Exception:  # noqa: BLE001 — a clock must never break the gate
        return False


def observation_time_limit() -> datetime:
    """One receipt/as-of upper bound for quotes and native daily histories."""
    from tarzan import runtime

    effective_date = runtime.as_of()
    return (
        datetime.combine(effective_date, time.max, tzinfo=timezone.utc)
        if effective_date is not None
        else datetime.now(timezone.utc)
    )


def quote_observation_is_future(observed: Optional[datetime]) -> bool:
    """Reject a price observed after the instant it can actually be received.

    LIVE providers can publish a genuine trade after the run's capture instant,
    while acquisition is in progress. Use the receipt-time wall clock for this
    upper bound, not ``captured_at``; the captured calendar still governs the
    existing age rule. Pinned runs use the end of their effective UTC date.
    Missing timestamps retain the caller's explicit undated-fallback contract.
    """
    if observed is None:
        return False
    limit = observation_time_limit()
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return observed > limit


def prev_close_eur(quote: dict, price_eur: float) -> Optional[float]:
    """The published previous close, on the SAME ruler as the price being
    stamped as today's point.

    Scale it by this instrument's own valuation-vs-native ratio
    (``price_eur / quote_price``): ~1 for a EUR listing, the FX for a USD/ZAR
    one, so the 1D equals the venue's own published move
    (EUR_now/EUR_prev == native_now/native_prev). Crucially the ratio also
    absorbs a scale mismatch between the two feeds: NTSG.MI's history ran at
    ~29.9 while its quote pair sat at ~25.5 (a split reflected in one feed and
    not the other), and pairing 29.9 (today) with a raw 25.8 prev_close printed
    a +16% one-day move. Scaling keeps both points on one ruler.

    Returns ``None`` when the quote carries no price or previous close (bonds
    Yahoo does not quote) so the caller keeps the feed's own close.
    """
    prev = quote.get("prev_close")
    native_price = quote.get("price")
    if not prev or not native_price:
        return None
    return float(prev) * (float(price_eur) / float(native_price))


def quote_observed_at(quote: dict) -> Optional[datetime]:
    """The instant the quote was observed, as an aware UTC datetime, or None.

    The valuation policy dates freshness on the observation rather than on when
    the request ran, so a stamped price needs its own timestamp to be booked as
    primary evidence instead of as an undated (fallback) quote.
    """
    observed = quote.get("time")
    if observed is None:
        return None
    epoch = finite_number(observed)
    if epoch is not None:
        try:
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    # Info metadata also admits ISO/datetime observations. Keep their actual
    # instants on the same UTC ruler rather than ranking them as undated.
    if not isinstance(observed, (str, date, pd.Timestamp)):
        return None
    if isinstance(observed, str) and observation_date(observed) is None:
        return None  # Relative "now"/"today" is receipt time, not source evidence.
    try:
        parsed = pd.Timestamp(observed)
        if pd.isna(parsed):
            return None
        parsed = (
            parsed.tz_localize("UTC")
            if parsed.tzinfo is None else parsed.tz_convert("UTC")
        )
        return parsed.to_pydatetime()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def stamp_date(quote: dict, today, ticker: str = "") -> Optional[pd.Timestamp]:
    """The SESSION date this quote's price belongs on, or None when it belongs on
    no session at all.

    Two rules, in order. The point belongs to the session the quote was OBSERVED
    in, read on the venue's own clock — not to the run's calendar day. And that
    date must be a real session for the venue, per the vendored exchange
    calendar: a price series may only ever carry session dates.

    The second rule is what used to be a weekday test in
    :func:`stamping_allowed`, and it is strictly stronger. Rejecting Saturday and
    Sunday outright also refused to write a FRIDAY close that the quote endpoint
    was carrying and the history endpoint was not, which is how the Sat 29 Aug
    2026 08:02 digest reported Thursday's session: Yahoo's daily bars held a
    28 Aug row with a null close for all sixteen holdings, while every quote
    carried the real Friday close (XDEQ.MI 79.73, observed Fri 16:20, against a
    series terminating at its own prev_close of 79.11). Dating from the
    observation puts that on Friday, where it belongs; asking the calendar
    whether the resolved date is a session is what keeps a Saturday-dated quote
    out — and covers exchange holidays, which the weekday rule never did.
    """
    observed = quote_observed_at(quote)
    if (
        observed is None and quote.get("time") is not None
    ) or quote_observation_is_future(observed):
        return None
    day = pd.Timestamp(today)
    if observed is not None:
        venue_day = observed
        try:
            from tarzan.data.market_quotes import _exchange_tz

            tz = _exchange_tz(ticker)
            if tz is not None:
                venue_day = observed.astimezone(tz)
        except Exception:  # noqa: BLE001 — a clock must never break a stamp
            pass
        day = pd.Timestamp(venue_day.date())
    if day is None or pd.isna(day):
        return None
    from tarzan.data.exchange_calendar import is_session

    return day if is_session(ticker, day.date()) else None


def stamp_today(series: pd.Series, today, today_value: float,
                quote: dict, ticker: str = "") -> Optional[pd.Series]:
    """Return ``series`` with today's point set to ``today_value`` and the prior
    session's close reconciled onto the same ruler via ``quote``.

    Preserves the series' ``name`` and ``attrs`` (the benchmark identity the
    semantic gate checks), and appends today after the last close so no re-sort
    is needed.

    ``regularMarketPreviousClose`` IS the close of the session immediately
    before today, so it is written on THAT date — inserted when the vendor
    dropped the bar (every Milan ETF came back with a null close for Monday
    17 Aug 2026 while the quote endpoint carried the official 40.815), and never
    written onto whatever older row happens to be last. Writing it onto the last
    row regardless of its date fabricated a price the venue never printed, on a
    date it does not belong to, in the series every window (5D/1M/YTD), the
    volatility/beta/drawdown block and every chart then read: a feed three
    sessions behind had a settled 102.00 close silently replaced by the previous
    day's 99.00. :func:`pick_quote` cannot catch that — it validates the price
    LEVEL, never the date.

    Dating it correctly repairs the gap AND leaves settled history alone, so the
    1D stays available and correct instead of measuring two sessions under a
    one-session label.

    Returns None when the quote belongs on no session (see :func:`stamp_date`),
    so a caller writes neither the series nor the price it would have paired with.
    """
    # Before Xetra opens on a Tuesday, ``regularMarketPrice`` is still Monday's
    # closing quote (``regularMarketTime`` says so), and dating it Tuesday moved
    # the whole window one session forward: AVWS.DE's 5D then anchored on 19 Aug
    # and read -0.55% where its own five sessions ending on the observed one
    # anchor 18 Aug and read -1.04%.
    day = stamp_date(quote, today, ticker or str(series.name or ""))
    if day is None:
        return None
    out = series.copy()
    stamp = day.tz_localize(series.index.tz) if series.index.tz else day
    prev_eur = prev_close_eur(quote, today_value)
    previous_row = None
    # Once today's session has opened, Yahoo ROLLS ``regularMarketPreviousClose`` onto
    # the last session that traded — even for an instrument that has not printed today.
    # Its quote is then the previous session's (observed yesterday), with price and
    # "previous close" both that session's close. Writing it on the session BEFORE the
    # observed one overwrote a real close with the next day's: on Fri 9 Oct 2026 at 09:28
    # a thin fund untraded since Thursday had its Wednesday close (114.94, a full session
    # of trades between 114.50 and 116.00) replaced by Thursday's 110.98, so Thursday's
    # -3.4% session vanished from its tape, and the same happened, by smaller amounts, to
    # every holding untraded at that minute. Pre-open the field is not rolled yet (price
    # and previous close differ), so that case keeps its repair.
    observed_prior_session = pd.Timestamp(day).date() < pd.Timestamp(today).date()
    rolled = (observed_prior_session and prev_eur is not None
              and abs(prev_eur / float(today_value) - 1.0) < 1e-9)
    if prev_eur is not None and not rolled:
        # Dated on the venue's OWN calendar: the session before the Tuesday
        # after Easter Monday is the Thursday before it, and a Mon-Fri rule
        # would have written the published close onto the closed Monday.
        from tarzan.data.exchange_calendar import previous_session as _prev

        prev = pd.Timestamp(_prev(ticker or str(series.name or ""), stamp.date()))
        if stamp.tz is not None:
            prev = prev.tz_localize(stamp.tz)
        if len(out.index) and prev >= out.index[0].normalize():
            out.loc[prev] = prev_eur
            previous_row = prev
    out.loc[stamp] = float(today_value)
    out = out.sort_index()
    out.name = series.name
    out.attrs = dict(getattr(series, "attrs", {}) or {})
    if previous_row is not None:
        source_symbol = str(quote.get("symbol") or ticker or series.name or "")
        native_previous = _native_previous_date(quote, source_symbol, day.date())
        out.attrs[CURRENT_SESSION_BASELINE_ATTR] = {
            "at": day.date().isoformat(),
            "anchor": previous_row.date().isoformat(),
            "observation_date": (
                native_previous.isoformat() if native_previous is not None else None
            ),
            "symbol": source_symbol,
        }
    else:
        # A rolled pair or an unavailable previous close supplies no new baseline.
        # Do not retain metadata that appears to authorise this newer point.
        out.attrs.pop(CURRENT_SESSION_BASELINE_ATTR, None)
    return out


def stamping_allowed() -> tuple[bool, Optional[pd.Timestamp]]:
    """``(allowed, today)`` for this run — the one gate both callers share.

    ``allowed`` is False only for a pinned/reproducible run: no live observation
    may enter one.

    It used to also refuse Saturdays and Sundays, on the reasoning that there is
    no live session to stamp at the weekend. True, and it conflated two things —
    "no session is in progress" with "there is no newer close to write". When the
    daily-bar feed is a session behind, the quote endpoint still carries that
    session's official close, and :func:`stamp_date` dates it onto its own venue
    day. Refusing the whole weekend is what made the Sat 29 Aug 2026 digest report
    THURSDAY: Yahoo's history held a 28 Aug row with a null close for all sixteen
    holdings while every quote carried the real Friday close. On a weekday the
    same gap self-heals through the stamp, which is why this only ever surfaced on
    the Saturday issue.

    The weekend is now rejected where it belongs — per venue, by the exchange
    calendar, on the date the observation actually falls on — which also covers
    the exchange holidays the old weekday rule admitted it did not model.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return False, None
    return True, pd.Timestamp(runtime.today())


def _candidates(tickers, *, isin_by_ticker=None) -> dict[str, list[str]]:
    """Each ticker plus its sibling venues, in priority order.

    Resolved the same way the intraday feed resolves it: a ``.MI`` quote can be
    corrupt while the fund's ``.DE`` line is clean (NTSG.MI returned 25.5
    against a real ~29.4), so all of them are fetched and the sanity gate picks
    the one whose level matches this instrument.
    """
    from tarzan.data.market_quotes import _sibling_symbols

    return {
        tk: [tk, *_sibling_symbols(
            tk, isin=isin_by_ticker.get(tk, "") if isin_by_ticker is not None else None,
        )]
        for tk in tickers if tk
    }


def _fx_on(fx, day):
    """Return a causal daily FX rate and its observation evidence."""
    from tarzan.data import enricher

    if fx is None or fx.empty or not isinstance(fx.index, pd.DatetimeIndex):
        return None
    threshold = pd.Timestamp(day)
    if fx.index.tz is not None:
        threshold = threshold.tz_localize(fx.index.tz)
    position = fx.index.searchsorted(threshold, side="right") - 1
    if position < 0:
        return None
    index = fx.index[position]
    origin = fx.attrs.get(enricher._HISTORY_ORIGINS_ATTR, {}).get(
        enricher._history_timestamp_key(index), enricher._HISTORY_ORIGIN_PRIMARY,
    )
    return float(fx.iloc[position]), {
        "observation_time": pd.Timestamp(index).to_pydatetime(),
        "is_fallback": origin == enricher._HISTORY_ORIGIN_CACHE,
        "source": "price_cache:FX" if origin == enricher._HISTORY_ORIGIN_CACHE else None,
    }


def _holding_quotes(holding, symbols, quotes, today):
    """Prepare independent native and EUR-per-unit quote maps for one holding."""
    from tarzan.data import enricher
    from tarzan.data.exchange_calendar import previous_session
    from tarzan.instruments.registry import InstrumentKind
    from tarzan.instruments.valuation import holding_kind_resolution

    native_quotes, eur_quotes = {}, {}
    own_currency = major_code(
        holding.price_currency or holding.price_quote_currency or holding.currency or "EUR"
    ).upper()
    kind = holding_kind_resolution(holding).kind
    if kind is None:
        return native_quotes, eur_quotes
    for symbol in symbols:
        original = quotes.get(symbol) or {}
        from tarzan.data.listing_aliases import alias_currency

        expected_currency = alias_currency(
            str(holding.ticker), symbol, isin=str(holding.isin or ""),
        )
        supplied_currency = original.get("currency")
        if (
            expected_currency is not None and supplied_currency
            and major_code(str(supplied_currency).strip()).upper() != expected_currency
        ):
            continue  # a venue suffix cannot override contradictory quote units
        currency = (
            enricher.venue_currency(symbol) or original.get("currency")
            or holding.price_quote_currency or holding.price_currency
            or holding.currency or "EUR"
        )
        currency = str(currency).strip()
        # The requested mapping key is the provider fact. A missing or
        # contradictory payload symbol must not erase the selected venue.
        quote = dict(original, symbol=symbol)
        for key in ("price", "prev_close"):
            try:
                value = float(quote.get(key))
            except (TypeError, ValueError, OverflowError):
                quote.pop(key, None)
                continue
            if not math.isfinite(value) or value <= 0:
                quote.pop(key, None)
            else:
                quote[key] = to_major(value, currency)
        if not quote.get("price"):
            continue
        normalized_currency = major_code(currency).upper()
        quote["currency"] = normalized_currency
        if normalized_currency == own_currency:
            native_quotes[symbol] = quote
        eur = dict(quote)
        if normalized_currency != "EUR":
            day = stamp_date(quote, today, ticker=symbol)
            if day is None:
                continue
            fx = enricher._usable_fx_series(
                enricher._get_fx_series(normalized_currency)
            ).sort_index()
            fx = fx[~fx.index.duplicated(keep="last")]
            current = _fx_on(fx, day)
            if current is None:
                continue  # unknown FX can never license a native-as-EUR stamp
            rate, evidence = current
            eur["price"] *= rate
            eur["_fx_evidence"] = evidence
            if eur.get("prev_close"):
                rolled = (
                    day.date() < pd.Timestamp(today).date()
                    and abs(quote["prev_close"] / quote["price"] - 1) < 1e-9
                )
                prior_day = day if rolled else previous_session(symbol, day.date())
                previous = _fx_on(fx, prior_day)
                if previous is None:
                    eur.pop("prev_close", None)
                else:
                    eur["prev_close"] *= previous[0]
        if kind is InstrumentKind.BOND:
            for key in ("price", "prev_close"):
                if key in eur:
                    eur[key] /= 100
        eur["currency"] = "EUR"
        if eur.get("prev_close") and not math.isfinite(eur["prev_close"]):
            eur.pop("prev_close", None)
        if math.isfinite(eur["price"]) and eur["price"] > 0:
            eur_quotes[symbol] = eur
    return native_quotes, eur_quotes


def _quotes_not_older_than(quotes, date, today, ticker):
    """A recent quote still cannot supersede a newer dated market close."""
    return {
        symbol: quote for symbol, quote in quotes.items()
        if (day := stamp_date(quote, today, ticker=ticker)) is not None
        and day.date() >= date
    }


def apply_to_holdings(holdings: list) -> tuple[str, ...]:
    """Stamp the current session onto every holding's series AND price.

    Returns the tickers stamped. Runs in the data layer, before the valuation
    policy, so ``current_price``/``current_value`` and the series terminal are
    the same number and the policy judges that one.
    """
    from tarzan.data.market_quotes import official_quotes

    allowed, today = stamping_allowed()
    if not allowed:
        return ()

    identities: dict[str, set[str]] = {}
    for holding in holdings:
        if holding.ticker:
            identities.setdefault(str(holding.ticker), set()).add(
                str(holding.isin or "").strip().upper()
            )
    isin_by_ticker = {
        ticker: next(iter(isins)) if len(isins) == 1 else ""
        for ticker, isins in identities.items()
    }
    candidates = _candidates(identities, isin_by_ticker=isin_by_ticker)
    if not candidates:
        return ()
    quotes = official_quotes(
        sorted({s for group in candidates.values() for s in group})
    )
    if not quotes:
        return ()

    # An independent previous close per holding, to arbitrate a disputed 1D baseline
    # (see ``_better_evidenced``). One justETF quote per ISIN, fetched in the same
    # pass; when justETF is unavailable the referee is simply absent.
    referees: dict[str, dict] = {}
    try:
        from tarzan.data import justetf

        by_isin = {str(h.ticker): h.isin for h in holdings if h.ticker and h.isin}
        justetf.prefetch_current_quotes(by_isin.values())
        referees = {tk: q for tk, isin in by_isin.items()
                    if (q := justetf.quote(isin))}
    except Exception as e:  # noqa: BLE001 — a referee must never block a stamp
        logger.info("No independent baseline this run: %s", e)

    stamped: list[str] = []
    for holding in holdings:
        from tarzan.instruments.valuation import (
            holding_kind_resolution, select_holding_valuation,
        )

        if holding_kind_resolution(holding).kind is None:
            continue
        price_only = getattr(holding, "price_history_price_only", None)
        total_return = getattr(holding, "price_history", None)
        symbols = candidates.get(str(holding.ticker), [])
        native_quotes, eur_quotes = _holding_quotes(holding, symbols, quotes, today)

        def _total_return_quote(quote):
            """Put a raw previous close onto the existing adjusted ruler."""
            if price_only is None or total_return is None:
                return quote
            day = stamp_date(quote, today, ticker=str(holding.ticker))
            if day is None or not quote.get("prev_close") or not quote.get("price"):
                return quote
            # Preserve stamp_today's protection against a rolled previous close.
            if (day.date() < pd.Timestamp(today).date()
                    and abs(float(quote["prev_close"]) / float(quote["price"]) - 1) < 1e-9):
                return quote
            from tarzan.data.exchange_calendar import previous_session

            previous = pd.Timestamp(previous_session(str(holding.ticker), day.date()))
            raw_index = previous.tz_localize(price_only.index.tz) if price_only.index.tz else previous
            adjusted_index = previous.tz_localize(total_return.index.tz) if total_return.index.tz else previous
            raw = price_only.get(raw_index)
            adjusted = total_return.get(adjusted_index)
            if raw is None or adjusted is None or pd.isna(raw) or pd.isna(adjusted) or raw <= 0:
                # An absent factor is not evidence that the two rulers coincide.
                return dict(quote, prev_close=None)
            return dict(quote, prev_close=float(quote["prev_close"]) * float(adjusted / raw))

        def _stamp(tape):
            """Stamp one tape against ITS OWN last close, or ``(None, None)``.

            Each tape resolves its own quote, because the level gate compares a
            candidate against the reference it is handed. Two tapes exist per
            instrument — the EUR one every portfolio figure reads, and the native
            one the per-instrument return columns read — and stamping only the
            first left every return column a session behind: AVEM.DE printed
            +0.02% (Wed→Thu, 27.625/27.62) where its own Friday session was
            +1.16% (27.945 against a 27.625 previous close). All 19 rows were
            affected, EUR listings included, because the two tapes are separate
            OBJECTS even where they hold identical numbers.
            """
            clean = None if tape is None else tape.dropna()
            if clean is None or len(clean) == 0:
                return None, None
            current_quotes = _quotes_not_older_than(
                native_quotes, pd.Timestamp(clean.index[-1]).date(), today,
                str(holding.ticker),
            )
            q = pick_quote(symbols, current_quotes,
                           float(clean.iloc[-1]), history=clean,
                           referee=(
                               referees.get(str(holding.ticker))
                               if not holding.price_currency or holding.price_currency == "EUR"
                               else None
                           ))
            p = q.get("price")
            if not p:
                return None, None
            return stamp_today(tape, today, float(p),
                               _total_return_quote(q), ticker=str(holding.ticker)), q

        native_stamped, _nq = _stamp(getattr(holding, "price_history_native", None))
        if native_stamped is not None:
            holding.price_history_native = native_stamped

        series = getattr(holding, "price_history", None)
        if series is None or series.empty:
            continue
        reference = price_only if price_only is not None and not price_only.empty else series
        usable = None if reference is None else reference.dropna()
        if usable is None or len(usable) == 0:
            continue
        current_quotes = _quotes_not_older_than(
            eur_quotes, pd.Timestamp(usable.index[-1]).date(), today,
            str(holding.ticker),
        )
        quote = pick_quote(
            symbols, current_quotes,
            float(usable.iloc[-1]), history=usable,
            referee=referees.get(str(holding.ticker)),
        )
        price = quote.get("price")
        if not price:
            continue
        # ``pick_quote`` has already established that this price sits on the
        # series' own ruler (within the sibling tolerance of its last real
        # close), which is exactly what ``current_price`` means — so the value
        # is ``quantity * price`` for every instrument kind, with no second
        # application of the bond per-100 convention.
        stamped_price = float(price)
        history = stamp_today(
            series, today, stamped_price, _total_return_quote(quote), ticker=str(holding.ticker))
        if history is None:
            # The quote belongs on no session for this venue, so neither the
            # series nor the price it would pair with may be written: they are
            # one observation and must not diverge.
            continue
        holding.price_history = history
        if price_only is not None and not price_only.empty:
            holding.price_history_price_only = stamp_today(
                price_only, today, stamped_price, quote, ticker=str(holding.ticker))
        holding.current_price = stamped_price
        quantity = float(getattr(holding, "quantity", 0.0) or 0.0)
        holding.current_value = quantity * stamped_price
        # Keep the older observation governing a converted price. A primary
        # security quote cannot erase the provenance of fallback FX evidence.
        observed = quote_observed_at(quote)
        fx_evidence = quote.get("_fx_evidence")
        if fx_evidence is not None:
            from tarzan.data.enricher import _composite_observation_time

            observed = _composite_observation_time(observed, fx_evidence["observation_time"])
        holding.price_is_fallback = bool(
            observed is None
            or (fx_evidence is not None and fx_evidence["is_fallback"])
        )
        holding.price_observation_timestamp = observed
        if fx_evidence is not None and holding.price_is_fallback and fx_evidence.get("source"):
            holding.data_source = fx_evidence["source"]
        else:
            holding.data_source = f"yfinance:{quote.get('symbol') or holding.ticker}:current_session"
        holding.valuation_result = select_holding_valuation(holding)
        stamped.append(str(holding.ticker))

    if stamped:
        logger.info(
            "Stamped the current session onto %d holding series and price(s)",
            len(stamped),
        )
    _log_tape_vintage(holdings)
    return tuple(stamped)


def _log_tape_vintage(holdings: list) -> None:
    """Record, per holding, the date its tape actually ends on.

    Nothing used to. Twice in one day a printed figure looked wrong and could not be
    diagnosed after the fact — a 1M of -0.69% against a recomputed -1.61%, which is the
    same arithmetic on a tape one session behind — because the run that produced it kept
    no record of what data it held. The run's log is retained; this puts the answer in
    it, so the next such question is a grep rather than an afternoon.

    Logged as the newest date plus only the holdings BEHIND it. A list of twenty
    identical dates is noise; the ones that lag are the whole signal, and a figure
    computed off a lagging tape is not wrong so much as older than it looks.
    """
    ends: dict[str, str] = {}
    for holding in holdings or []:
        series = getattr(holding, "price_history", None)
        if series is None:
            continue
        clean = series.dropna()
        if clean.empty:
            continue
        ends[str(getattr(holding, "ticker", "") or "?")] = str(
            pd.Timestamp(clean.index[-1]).date())
    if not ends:
        return
    newest = max(ends.values())
    behind = sorted((t, d) for t, d in ends.items() if d < newest)
    if behind:
        logger.info(
            "Tape vintage: newest close %s; %d holding(s) behind it -> %s",
            newest, len(behind),
            ", ".join(f"{t}@{d}" for t, d in behind),
        )
    else:
        logger.info("Tape vintage: all %d holdings current to %s",
                    len(ends), newest)
