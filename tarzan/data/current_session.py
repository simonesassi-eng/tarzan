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
€203). Net effect: the hero printed a €242,224.03 portfolio beside P&L tiles
that implied €242,108.85 — €115 apart, from a silent stale order price.

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
observation may enter a reproducible run. Weekend runs stamp nothing either —
the last real close IS the current value (Yahoo shows exactly that on a closed
market), and appending a weekend-dated point slides ``window_anchor`` onto the
day a month before the WEEKEND rather than the last session.
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


def stamp_today(series: pd.Series, today, today_value: float,
                quote: dict, ticker: str = "") -> pd.Series:
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
    """
    out = series.copy()
    stamp = today.tz_localize(series.index.tz) if series.index.tz else today
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

    ``allowed`` is False for a pinned/reproducible run (no live observation may
    enter one) and on a weekend/holiday-less Saturday or Sunday, where there is
    no live session to stamp.
    """
    from tarzan import runtime

    if not runtime.allows_live_transport():
        return False, None
    today = pd.Timestamp(runtime.today())
    # ponytail: weekends only, holidays not modelled (see stats._window_end).
    if today.weekday() >= 5:
        return False, today
    return True, today


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
        holding.price_history = stamp_today(
            series, today, stamped_price, quote, ticker=str(holding.ticker))
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
    return tuple(stamped)
