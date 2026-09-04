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
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def pick_quote(symbols: list[str], quotes: dict, reference_price: float) -> dict:
    """The first candidate quote whose price agrees with ``reference_price``
    (the instrument's own last real close) within the sibling tolerance.

    This is the sanity gate that rejects a corrupt feed: NTSG.MI's quote priced
    the fund at 25.5 while its own series and its ``.DE`` sibling sat at ~29.4,
    so the canonical is skipped and the clean sibling supplies the close
    instead. Returns ``{}`` when nothing agrees, so the caller keeps the feed's
    own close rather than stamp from bad data.

    ponytail: the reference is an EUR-per-unit close while the quote is in the
    venue's native units, so a non-EUR listing fails the tolerance by the FX
    rate and is simply not stamped. Comparing in native units (i.e. stamping
    before the FX/bond conversion) is what would make this reachable for them.
    """
    from tarzan.data.market_quotes import _SIBLING_PRICE_TOLERANCE

    for symbol in symbols:
        quote = quotes.get(symbol) or {}
        native = quote.get("price")
        if not native:
            continue
        if abs(float(native) / float(reference_price) - 1.0) <= _SIBLING_PRICE_TOLERANCE:
            return quote
    return {}


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
    try:
        return datetime.fromtimestamp(int(observed), tz=timezone.utc)
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
    if prev_eur is not None:
        # Dated on the venue's OWN calendar: the session before the Tuesday
        # after Easter Monday is the Thursday before it, and a Mon-Fri rule
        # would have written the published close onto the closed Monday.
        from tarzan.data.exchange_calendar import previous_session as _prev

        prev = pd.Timestamp(_prev(ticker or str(series.name or ""), stamp.date()))
        if stamp.tz is not None:
            prev = prev.tz_localize(stamp.tz)
        if len(out.index) and prev >= out.index[0].normalize():
            out.loc[prev] = prev_eur
    out.loc[stamp] = float(today_value)
    out = out.sort_index()
    out.name = series.name
    out.attrs = dict(getattr(series, "attrs", {}) or {})
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


def _candidates(tickers) -> dict[str, list[str]]:
    """Each ticker plus its sibling venues, in priority order.

    Resolved the same way the intraday feed resolves it: a ``.MI`` quote can be
    corrupt while the fund's ``.DE`` line is clean (NTSG.MI returned 25.5
    against a real ~29.4), so all of them are fetched and the sanity gate picks
    the one whose level matches this instrument.
    """
    from tarzan.data.market_quotes import _sibling_symbols

    return {tk: [tk, *_sibling_symbols(tk)] for tk in tickers if tk}


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

    candidates = _candidates({str(h.ticker) for h in holdings if h.ticker})
    if not candidates:
        return ()
    quotes = official_quotes(
        sorted({s for group in candidates.values() for s in group})
    )
    if not quotes:
        return ()

    stamped: list[str] = []
    for holding in holdings:

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
            q = pick_quote(candidates.get(str(holding.ticker), []), quotes,
                           float(clean.iloc[-1]))
            p = q.get("price")
            if not p:
                return None, None
            return stamp_today(tape, today, float(p),
                               q, ticker=str(holding.ticker)), q

        native_stamped, _nq = _stamp(getattr(holding, "price_history_native", None))
        if native_stamped is not None:
            holding.price_history_native = native_stamped

        series = getattr(holding, "price_history", None)
        usable = None if series is None else series.dropna()
        if usable is None or len(usable) == 0:
            continue
        quote = pick_quote(
            candidates.get(str(holding.ticker), []), quotes,
            float(usable.iloc[-1]),
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
            series, today, stamped_price, quote, ticker=str(holding.ticker))
        if history is None:
            # The quote belongs on no session for this venue, so neither the
            # series nor the price it would pair with may be written: they are
            # one observation and must not diverge.
            continue
        holding.price_history = history
        holding.current_price = stamped_price
        quantity = float(getattr(holding, "quantity", 0.0) or 0.0)
        holding.current_value = quantity * stamped_price
        # A level-validated published quote is primary market evidence, so it
        # clears any fallback flag enrichment set from a staler rung. Its
        # observation time comes from the quote itself when the provider dated
        # it; the policy falls back to the fetch clock for a non-fallback quote
        # with no distinct timestamp.
        holding.price_is_fallback = False
        holding.price_observation_timestamp = quote_observed_at(quote)
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
