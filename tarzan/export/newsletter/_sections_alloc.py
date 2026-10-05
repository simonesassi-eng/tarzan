"""Allocation / holdings / hero / optimizer section builders."""

from __future__ import annotations

from datetime import datetime, time
from html import escape as _esc
from typing import Any, Optional

import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.export._format import (
    display_instrument_name,
    eur_smart as _eur_smart,
    short_instrument_name,
)
from tarzan.export._charts import (
    waterfall as _wf_chart,
)
from tarzan.export._perf_series import (
    _norm_series,
    benchmark_gap_history,
    benchmark_gap_pp,
    mwr_period_pct,
    _perf_window,
    _window_money_pnl,
)
from tarzan.export.newsletter._constants import (
    ASSET_CLASS_ORDER,
    ASSET_COLORS,
    GEO_COLORS,
    PALETTE,
    TYPE,
    TYPE_PX,
    FONT_STACK,
    _NewsletterContext,
    _ordered,
    class_key,
    group_by_class_role,
    render_unified_table,
    role_for,
    uni_cell,
    uni_name,
    geo_label,
)
from tarzan.export.newsletter._format import (
    _display_ticker,
    _eur,
    _pct,
    _pct_smart,
    _signed_pp,
    is_missing,
)
from tarzan.export.newsletter._charts import (
    _hero_chart_legend,
    _hero_flow_chips,
    _hero_value_chart,
    _prev_session_label,
    session_span_labels,
    _spark,
    _timeline_vals,
    bullet as _bullet,
)

def _market_is_open(perf: Optional[dict]) -> bool:
    """Whether a venue the portfolio holds is TRADING, per exchange hours.

    ``market_open`` is the engine's exchange-hours fact; ``1d_intraday`` is a
    different one — that the 1D figures are intraday rather than close-to-close.
    Reading the latter for this caption is what printed "market CLOSED" at 09:09
    with Milan and London both trading: minutes after an open the venue is open
    but no intraday bar exists yet. Falls back to ``1d_intraday`` only when the
    engine did not state it (no live transport, or an older projection) — a weaker
    proxy, since a book with a today point implies a venue printed one."""
    p = perf or {}
    open_now = p.get("market_open")
    return bool(p.get("1d_intraday")) if open_now is None else bool(open_now)


def _session_basis(perf: Optional[dict], m) -> str:
    """What the Session figure IS, in the caption's own words.

    ``1d_intraday`` false means the number is a completed session's close-to-close
    move. Captioning that "market open" — which it legitimately can be, minutes
    before the holdings' own venues open — invites reading a finished session as
    today's, which is how a Tuesday 08:58 digest showed Monday's +0.41% as the
    live session. Name the basis instead, the way the Markets strip prints
    "Cl. Mon" rather than a bare percentage.

    This branch was unreachable in exactly the window it was written for. The flag it
    reads was set from exchange hours, so at 09:12 on Tue 15 Sep 2026 it was true with
    zero holdings priced, and the tile headlined Monday's −0.59% as "market open". It
    now comes from the tape (``1d_intraday``), so an open venue with no bars prints
    "14 Sep session, close to close" — which is what the figure is."""
    p = perf or {}
    if bool(p.get("1d_intraday")):
        return f'market {"open" if _market_is_open(p) else "closed"}'
    # Name the session the figure DESCRIBES, not the date on the other side of it.
    # "close-to-close vs 28 Aug" read as though 28 Aug were the baseline, while it
    # was the endpoint: on Sat 29 Aug 2026 the +0.45% spanned Thu 27 → Fri 28.
    _base, end = session_span_labels(m, "%d %b")
    return f"{end} session, close to close" if end else "close-to-close"


def _priced_today_note(perf: Optional[dict]) -> str:
    """How much of the book the live figure actually covers, when not all of it.

    A holding the vendor has not priced today is carried forward at its previous
    close, so it contributes zero to the day's move while its full value stays in
    the denominator. On 27 Aug 2026 at 09:12 that made two early ticks — CL2
    (7.7%) and UEQC (4.6%), up 0.74% and 0.15% — render as a portfolio "1D
    +0.06%"; by 10:11, with 91% priced, the same book read +0.18%.

    Stated rather than corrected: the figure is the NAV's real move and every
    table in the issue uses the same one, so silently substituting a different
    number would split the document. Absent at full coverage, where it would be
    noise, and absent when the basis is a completed session, which
    ``_session_basis`` already names.
    """
    p = perf or {}
    coverage = p.get("1d_coverage_pct")
    if coverage is None or not bool(p.get("1d_intraday")):
        return ""
    if coverage >= 99.5:
        return ""
    return f"{coverage:.0f}% of the book priced today"


def _funding_verification(verifications) -> Optional[dict[str, Any]]:
    """Return the final serialized-action funding proof, when available."""
    return next(
        (
            verification
            for verification in (verifications or [])
            if verification.get("kind") == "funding"
        ),
        None,
    )


def _build_headline(ctx: _NewsletterContext, hero: dict) -> dict:
    """Build the TL;DR headline shown above the Hero.

    Synthesizes the week into a single narrative sentence: ``how the
    portfolio moved + what to do next``. Designed to give the inbox
    reader the pugno-nello-stomaco answer in 5 seconds.
    """
    m = ctx.metrics
    perf_full = m.performance_full or {}
    week_return = perf_full.get("5d")

    parts: list[str] = []

    # Movement clause
    if is_missing(week_return):
        parts.append("Your portfolio is steady this week")
    else:
        wk_eur = m.total_value * float(week_return) / 100
        if abs(float(week_return)) < 0.1:
            parts.append("Your portfolio is essentially flat this week")
        elif float(week_return) >= 0:
            parts.append(
                f"Your portfolio gained {_eur_smart(wk_eur)} "
                f"({_pct(float(week_return), signed=True)}) this week"
            )
        else:
            parts.append(
                f"Your portfolio lost {_eur_smart(abs(wk_eur))} "
                f"({_pct(float(week_return), signed=True)}) this week"
            )

    # Action clause. A failed funding proof takes precedence over the mere
    # presence of draft actions: those trades must never read as instructions.
    suggestions = list(m.rebalancing_suggestions or [])
    funding = _funding_verification(m.rebalancing_verifications)
    if funding and funding.get("status") == "NON_EXECUTABLE":
        parts.append("and the draft rebalance is not executable until cash funding is resolved")
    elif suggestions:
        parts.append("and the optimizer suggests rebalancing below")
    else:
        parts.append("and your allocation is on target")

    return {
        "text": ", ".join(parts) + ".",
        "is_positive": (
            week_return is None
            or (isinstance(week_return, float) and pd.isna(week_return))
            or float(week_return) >= 0
        ),
    }

#: Beyond this many days, a green verdict is reported as "last verified" rather than
#: as "verified": the scheduled check runs every weekday, so a verdict older than a
#: long weekend means the check itself stopped running — and GitHub disables scheduled
#: workflows after a stretch of repository inactivity, which would otherwise freeze a
#: reassuring badge in place forever.
_VERIFY_FRESH_DAYS = 4


def _verify_verdict() -> Optional[dict]:
    """The last independent return check, or None when there is nothing to report.

    ``scripts/verify_returns_vs_yahoo.py`` recomputes every printed return against a
    raw Yahoo pull and runs as its own scheduled workflow, deliberately not as a step
    in the send (its own full pipeline run, against a render that is already
    throttle-bound). The cost of that separation is that a red verdict lands in a
    workflow log the reader of the digest never sees — so the digest carries it.

    Read from the environment, not fetched here. The workflow does the one API call it
    takes and passes the answer in, which keeps the network out of render, keeps the
    figures independent of whether GitHub answered, and means a local run or a pinned
    replay simply shows nothing.

    Returns ``{"ok", "text", "url"}``. ``ok`` False is the case worth seeing: the
    figures in the issue may be the ones the oracle disagreed with.
    """
    import os

    from tarzan import runtime as _runtime

    conclusion = (os.environ.get("VERIFY_CONCLUSION") or "").strip().lower()
    if not conclusion:
        return None
    raw_at = (os.environ.get("VERIFY_AT") or "").strip()
    when, age_days = "", None
    if raw_at:
        try:
            stamp = datetime.fromisoformat(raw_at.replace("Z", "+00:00"))
            when = stamp.strftime("%d %b %H:%M")
            age_days = (_runtime.today() - stamp.date()).days
        except ValueError:
            when = ""
    ok = conclusion == "success"
    stale = age_days is not None and age_days > _VERIFY_FRESH_DAYS
    if ok:
        # "last verified" states the same fact without implying it is current.
        lead = "last verified" if stale else "verified"
        text = f"{lead} {when}".strip() if when else "verified"
    else:
        text = (f"RETURN CHECK FAILED {when}".strip() if when
                else "RETURN CHECK FAILED")
    return {"ok": ok, "text": text,
            "url": (os.environ.get("VERIFY_URL") or "").strip()}


def _build_header(ctx: _NewsletterContext) -> dict:
    """Build the header strip metadata.

    The portfolio inception date is taken automatically from the order
    list (``metrics.inception_date``, the first order).

    Issue number is computed dynamically: weeks since inception when
    available, otherwise the ISO week of the current year. The explicit
    ``ctx.issue_number`` value overrides this only when greater than 1,
    so callers wishing to pin a specific number still can.
    """
    # The run-owned clock, not datetime.now(): under --as_of every other figure
    # in the newsletter is measured at the effective date, so a masthead stamped
    # with the wall clock dates the issue to a day the numbers do not describe.
    # It also made the markup golden fail on any day but the one it was
    # regenerated on, which is a test that expires rather than a gate.
    from tarzan import runtime as _runtime

    now = datetime.combine(_runtime.today(), time.min)
    inception_date = ctx.metrics.inception_date or ""
    issue_number = ctx.issue_number
    if issue_number <= 1 and inception_date:
        try:
            inception = pd.to_datetime(inception_date)
            weeks = max(1, int((now - inception.to_pydatetime()).days // 7) + 1)
            issue_number = weeks
        except Exception:
            issue_number = now.isocalendar().week
    elif issue_number <= 1:
        issue_number = now.isocalendar().week
    # ── The status bar ──────────────────────────────────────────────────────
    # A single line above the masthead carrying the figures a reader checks
    # before deciding whether to read at all. Terse by design: label, value,
    # nothing else.
    m = ctx.metrics
    risk = ((m.historical_risk or {}).get("portfolio") or {}).get("metrics") or {}
    perf = getattr(m, "performance", None) or {}

    def _bar(label, value, tone="flat"):
        # Same boundary as _tile: a benchmark name reaches the status bar via
        # "VS {name}", and names carry ampersands.
        return {"label": _esc(str(label)), "value": _esc(str(value)),
                "tone": tone}

    def _tone(v):
        if v is None:
            return "flat"
        return "pos" if float(v) >= 0 else "neg"

    status_bar = [_bar("NAV", _eur(m.total_value, decimals=0))]
    if perf.get("1d") is not None:
        status_bar.append(_bar("1D", _pct(perf["1d"], signed=True),
                               _tone(perf["1d"])))
    if m.twr_pct is not None:
        status_bar.append(_bar("TWR", _pct(m.twr_pct, signed=True),
                               _tone(m.twr_pct)))
    # The gap against the geography benchmark. An earlier pass left this out on
    # the grounds that the engine computes no such delta -- wrong: both terms are
    # computed and already printed side by side in the since-inception chart's
    # legend, so the entry is their difference, not a new estimate.
    gap = benchmark_gap_pp(m, ctx.benchmark_geo)
    if gap is not None:
        status_bar.append(_bar(
            f"VS {_bench_short(ctx.benchmark_geo)}",
            f'{"+" if gap > 0 else ("\u2212" if gap < 0 else "")}'
            f'{abs(gap):.2f}pp', _tone(gap)))
    if risk.get("beta") is not None:
        status_bar.append(_bar("\u03b2", f'{float(risk["beta"]):.2f}'))
    if risk.get("sharpe") is not None:
        status_bar.append(_bar("SHARPE", f'{float(risk["sharpe"]):.2f}'))

    # Data stamp: the issue date, the close the figures below rest on, and
    # whether a session is open. All three change what the numbers mean.
    #
    # Which close that is depends on the session, and one phrase cannot claim
    # both. With a session open the terminal point is LIVE, so the meaningful
    # close is the one it is measured from \u2014 the 1D's baseline. With every
    # session closed the terminal point IS a close, and that is what the issue is
    # valued at. This printed "close {last completed session}" under a comment
    # claiming it was the 1D baseline, which on Sat 29 Aug 2026 named 28 Aug \u2014 the
    # endpoint of the +0.45%, whose baseline was the 27th.
    base_label, end_label = session_span_labels(m, "%d %b")
    live = bool((perf or {}).get("1d_intraday"))
    stamp = now.strftime("%a, %d %b %Y")
    if live and base_label:
        stamp += f" \u00b7 vs {base_label} close"
    elif not live and end_label:
        stamp += f" \u00b7 at the {end_label} close"
    stamp += f' \u00b7 market {"OPEN" if _market_is_open(perf) else "CLOSED"}'
    # The verdict belongs on this line and nowhere else: the stamp already carries the
    # three facts that change what the numbers MEAN, and "were these numbers checked
    # against an outside source" is the fourth. A pass rides in the stamp; a failure
    # gets its own strip, because a reader who skims the header must not be able to
    # miss it.
    verify = _verify_verdict()
    if verify and verify["ok"]:
        stamp += f' \u00b7 {verify["text"]}'
    return {
        "date_short": now.strftime("%a, %d %b %Y"),
        "stamp": stamp,
        "issue_number": issue_number,
        "inception_date": inception_date,
        "status_bar": tuple(status_bar),
        "verify_alert": (None if not verify or verify["ok"] else {
            "text": _esc(verify["text"]), "url": _esc(verify["url"])}),
    }


def _bench_short(name: Optional[str]) -> str:
    """The benchmark's last word, upper-cased, for a label that has to fit a
    status-bar cell: "iShares MSCI ACWI" -> "ACWI"."""
    parts = [w for w in str(name or "").replace("-", " ").split() if w]
    return (parts[-1] if parts else "BENCHMARK").upper()

def _build_hero(ctx: _NewsletterContext) -> dict:
    m = ctx.metrics
    cfg = ctx.config

    # Two distinct P&L views at portfolio level:
    #   * Unrealized PnL — the snapshot gain on the positions held *today*
    #     vs their cost basis (= the Excel "RTD"). Numerator/denominator
    #     are current-holdings only; realized gains and income are excluded.
    #   * Total PnL — the lifetime, all-in gain (realized + unrealized +
    #     coupons/dividends) from the order list, expressed over the *net*
    #     capital contributed (current_value − Total PnL). It answers "how
    #     much have I actually made on the money I put in".
    unrealized_eur = m.unrealized_pnl_eur
    # NOT ``or 0.0``. ``unrealized_pnl_pct`` returns None precisely when there is
    # no cost basis to divide by, and 0.00% reads as "break-even" rather than "not
    # applicable" — the numeric-zero-is-not-unavailable rule the property's own
    # docstring states. Coercing it defeated ``_pnl_tile``'s ``is_missing``
    # fallback, so a fully liquidated book headlined "Total P&L 0.00%" beside a
    # caption reading "+€3.0k": the big number said the book had made nothing while
    # the small one said what it had really made.
    unrealized_pct = m.unrealized_pnl_pct

    has_total_pnl = m.pnl_eur is not None
    total_pnl_eur = m.pnl_eur if has_total_pnl else unrealized_eur
    total_pnl_pct = (
        m.pnl_pct if (has_total_pnl and m.pnl_pct is not None) else unrealized_pct
    )
    twr_pct = m.twr_pct

    # "Since inception" caption with the precise month/year of the first
    # order (derived automatically; falls back to empty when unknown).
    inception_label = ""
    if m.inception_date:
        try:
            inception_label = pd.to_datetime(m.inception_date).strftime("%b %Y")
        except Exception:
            inception_label = ""

    invested_pct = (m.invested_value / m.total_value * 100) if m.total_value > 0 else 0.0

    # ── The nine state tiles ────────────────────────────────────────────────
    # Every headline figure at the top, in three rows of three: what the
    # portfolio is worth and the profit in it, then the three return measures,
    # then the context each is judged against. These lived scattered across the
    # Performance section, which meant the opening screen answered "how much do
    # I have" but not "how am I doing".
    #
    # Each tile is (label, value, caption, tone). ``tone`` is 'pos'/'neg'/'flat'
    # and the template maps it to a colour, so no palette lookup leaks in here.
    def _tone(value) -> str:
        if value is None:
            return "flat"
        return "pos" if float(value) >= 0 else "neg"

    perf = getattr(m, "performance", None) or {}
    cagr_pct = perf.get("cagr")
    session_pct = perf.get("1d")
    # The session move in euros, for the tile caption: derived from the very
    # percentage shown beside it, so the two can never describe different
    # windows. (_window_money_pnl below walks a CALENDAR-day window, while
    # session_pct is the last-TRADING-day change — across a weekend those are
    # different spans, which is how a -0.18% session came to print a
    # four-figure euro loss.)
    #
    # The base is invested_value, not total_value: session_pct is a price-only
    # return over the priced holdings (see metrics._portfolio_history, which
    # sums price_history x quantity), and cash contributes no price move to
    # it. Applying it to a total that includes cash inflates the euro figure
    # by the cash weight.
    # base is an END-of-session value, while session_pct is measured against
    # the session's START, so the percentage is de-compounded rather than
    # applied directly: end - end/(1+p) == end * p/(1+p).
    session_eur = None
    if not is_missing(session_pct):
        base = m.invested_value if m.invested_value > 0 else m.total_value
        pct_val = float(session_pct) / 100.0
        if pct_val != -1.0:
            session_eur = base * pct_val / (1.0 + pct_val)
    if session_eur is None and m.pnl_series is not None and m.actual_value_series is not None:
        pair = _window_money_pnl(m.pnl_series, m.actual_value_series, "1d")
        if pair and pair[0] is not None:
            session_eur = float(pair[0])
    def _tile(label, value, caption, tone="flat"):
        # Escaped here because the digest template is ``.html.j2``, an extension
        # select_autoescape() does not match, so nothing escapes on the way out.
        # Asset-class names carry an ampersand ("Cash & Cash Equivalents") and so
        # do the P&L labels and any benchmark name, and they are DATA elsewhere
        # (Excel, the JSON summary) — so they are escaped where they become
        # markup, not at the source.
        return {"label": _esc(str(label)), "value": _esc(str(value)),
                "caption": _esc(str(caption)), "tone": tone}

    def _pnl_tile(label, eur, pct, denominator):
        """A P&L tile with the EUROS as the headline and the percentage beneath.

        This led with the percentage for a while, on the reasoning that a euro P&L
        answers nothing without the capital behind it while a percentage is comparable
        to every other figure in the issue. Both are true and it is still the wrong way
        round for a P&L: the money made is the thing, and the rate it was made at is how
        to judge it. The percentage keeps its place, first on the caption line, and the
        issue is full of percentages elsewhere -- TWR, MWR and CAGR are all rates and
        all still lead with one.

        The tone follows the headline, so the colour belongs to the number it is drawn
        on. Both figures share a sign, so nothing changes about which colour that is. If
        the euros are unavailable the tile falls back to leading with the percentage
        rather than headlining a "—".
        """
        if is_missing(eur):
            return _tile(label, _pct(pct, signed=True), denominator, _tone(pct))
        caption = denominator if is_missing(pct) else (
            f"{_pct(pct, signed=True)} · {denominator}")
        return _tile(label, _eur_smart(eur, signed=True), caption, _tone(eur))

    state_tiles = [
        _tile("Portfolio", _eur(m.total_value, decimals=0),
              f"invested {_eur_smart(m.invested_value)}"),
        _pnl_tile("Total P&L", total_pnl_eur, total_pnl_pct,
                  "on contributed capital"),
        _pnl_tile("Unrealized P&L", unrealized_eur, unrealized_pct,
                  "on open positions"),
    ]
    # ── The two return measures, each in BOTH forms, four cells ──────────────
    #
    # These were two tiles, and they disagreed about which form leads: TWR showed
    # the cumulative figure with the annualized one in its caption, MWR showed the
    # annualized figure with net-of-tax in its caption. So the two headline numbers
    # on the same row answered different questions, and the one comparison a reader
    # actually wants -- money-weighted against time-weighted, over the same span --
    # could not be read off them at all.
    #
    # Four cells, laid out as a 2x2 across two rows of the grid: cumulative in the
    # left column, annualized in the middle. The distance between the two
    # cumulative figures is what the timing of contributions cost or earned
    # (-2.11pp on the reference book), which no single tile here states and which
    # the since-inception chart draws.
    #
    # "since inception" and "annualized" rather than "cum."/"ann.": the labels are
    # uppercased in CSS, and the issue already calls the lifetime panel
    # "RETURN · SINCE INCEPTION", so this is the vocabulary the reader has.
    span_days = None
    ph = getattr(m, "portfolio_history", None)
    if ph is not None and len(ph) >= 2:
        span_days = (ph.index[-1].date() - ph.index[0].date()).days
    # The four captions are deliberately symmetric -- "<measure> · <span>" on the
    # cumulative pair, "<measure> · per year" on the annualized pair -- so a reader
    # sees a 2x2, not four unrelated lines. "(XIRR)" used to sit on the MWR cell and
    # pushed its caption over the column, wrapping between "266" and "days": a figure
    # split from its unit across two lines reads worse than a name the measure does
    # not need, since "money-weighted" already says which of the two it is.
    over = f" \u00b7 {span_days} days" if span_days else ""

    if twr_pct is not None:
        state_tiles.append(_tile(
            "TWR since inception", _pct(twr_pct, signed=True),
            f"time-weighted{over}", _tone(twr_pct)))
    twr_ann = m.twr_annualized_pct
    if twr_ann is not None:
        state_tiles.append(_tile(
            "TWR annualized", _pct(twr_ann, signed=True),
            "time-weighted \u00b7 per year", _tone(twr_ann)))
    elif cagr_pct is not None:
        # CAGR only when the annualized TWR is missing, which is the holdings-only
        # path: ``_returns`` is appended to the computer list ONLY when an order
        # list is supplied, so there TWR is None while ``performance.cagr`` still
        # computes off the fixed-basket history. On the order path the two are the
        # same number to the last float bit -- both annualize one cumulative return
        # read off one series -- so showing both was one fact in two cells.
        state_tiles.append(_tile("CAGR", _pct(cagr_pct, signed=True),
                                 "compound annual growth", _tone(cagr_pct)))
    # The gap against the geography benchmark, and how it got there. It sits on the
    # TWR row because that is what it IS: cumulative TWR minus the benchmark's own
    # cumulative return. An earlier pass left this tile out on the grounds that the
    # engine computes no such delta; both series are computed and drawn side by side
    # in the since-inception chart, so the gap is the distance between two lines the
    # reader can already see.
    gap = benchmark_gap_history(m, ctx.benchmark_geo)
    if gap is not None:
        now_pp = gap["now_pp"]
        sign = "+" if now_pp > 0 else ("\u2212" if now_pp < 0 else "")
        caption = (f'peak {"+" if gap["peak_pp"] >= 0 else "\u2212"}'
                   f'{abs(gap["peak_pp"]):.1f}pp {gap["peak_when"]}')
        if gap["turn_when"]:
            caption += f' \u00b7 turned {gap["turn_when"]}'
        state_tiles.append(_tile(
            f"vs {ctx.benchmark_geo}", f"{sign}{abs(now_pp):.2f}pp",
            caption, _tone(now_pp)))
    # MWR, same two forms. The cumulative one is ``xirr_pct`` de-annualized through
    # the shared helper, so it is the figure the since-inception chart's MWR line
    # ends on rather than a second estimate of it.
    mwr_cum = mwr_period_pct(m)
    if mwr_cum is not None:
        state_tiles.append(_tile(
            "MWR since inception", _pct(mwr_cum, signed=True),
            f"money-weighted{over}", _tone(mwr_cum)))
    if m.xirr_pct is not None:
        net = getattr(m, "xirr_net_tax_pct", None)
        state_tiles.append(_tile(
            "MWR annualized", _pct(m.xirr_pct, signed=True),
            "money-weighted \u00b7 per year"
            + (f" \u00b7 {_pct(net, signed=True)} net of tax"
               if net is not None else ""),
            _tone(m.xirr_pct)))
    if session_pct is not None:
        # What the session was worth and which session it was: a percentage
        # alone does not say either.
        parts = []
        if session_eur is not None:
            parts.append(_eur_smart(session_eur, signed=True))
        parts.append(_session_basis(perf, m))
        # How much of the book that figure covers, when it is not all of it.
        note = _priced_today_note(perf)
        if note:
            parts.append(note)
        state_tiles.append(_tile(
            "Session", _pct(session_pct, signed=True),
            " \u00b7 ".join(parts), _tone(session_pct)))
    if m.avg_ter is not None:
        # avg_ter arrives already in percent (metrics.py multiplies the stored
        # fractions by 100), so it must not be scaled again here. The synthetic
        # fixture has a zero TER, which hid this: a real run printed 23.106%.
        state_tiles.append(_tile("TER", f"{float(m.avg_ter):.3f}%",
                                 "weighted average, annual"))


    # Dual-axis hero chart: 30-day portfolio value (€, left) + both P&L measures
    # (€, right), with cash-flow triangles. Empty string when the order-derived
    # series are unavailable (holdings-only path).
    value_chart_html = ""
    hero_chart_legend = ""
    hero_flow_chips = ""
    win = _perf_window(m, 30)
    if (win and win.get("value") and len(win["value"]) >= 2
            and m.unrealized_series is not None and m.actual_value_series is not None):
        dts = win["dates"]
        idx = pd.DatetimeIndex(dts)

        def _pnl_eur(source):
            """A P&L series in EUROS, on the chart's own index.

            It used to be expressed as a percentage of its cost basis, which put a
            percentage axis beside a euro one and left the reader converting between
            them to see how much of the value line was profit. Both axes are money now,
            so the right one is a share of the left and the comparison is direct.
            """
            if source is None:
                return None
            s = _norm_series(source).reindex(idx, method="ffill").bfill()
            return list(s.astype(float).values)

        unreal_series = _pnl_eur(m.unrealized_series)
        total_series = _pnl_eur(m.pnl_series)
        value_chart_html = _hero_value_chart(
            win["value"], unreal_series, dts, win["flows"],
            total_eur=total_series,
        )
        if value_chart_html:
            hero_chart_legend = _hero_chart_legend(
                has_total=total_series is not None)
        hero_flow_chips = _hero_flow_chips(win["flows"])

    # This-week figures, mirroring the since-inception group:
    #   * Total PnL — the real money gained over the last five sessions, net
    #     of any contributions in them (delta of the cumulative P&L series);
    #   * TWR — the 5-session time-weighted return (performance_full['5d']),
    #     the same series the Returns tables use.
    perf_full = m.performance_full or {}
    week_twr = perf_full.get("5d")
    try:
        week_twr = float(week_twr) if week_twr is not None else None
        if week_twr != week_twr:  # NaN
            week_twr = None
    except (TypeError, ValueError):
        week_twr = None
    week_pnl_eur, week_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, "5d")
    # Last-30-days money P&L (net of contributions) for the scoreboard's
    # "Last 30 days" row, mirroring the chart window.
    month_pnl_eur, month_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, "1m")
    month_twr = perf_full.get("1m")
    try:
        month_twr = float(month_twr) if month_twr is not None else None
        if month_twr != month_twr:  # NaN
            month_twr = None
    except (TypeError, ValueError):
        month_twr = None

    # Cash KPI: show only the amount (no "above/below/on target" message).
    cash_msg, cash_msg_color = "", PALETTE["muted"]

    # Rebalance status: traffic-light derived from the largest non-cash
    # drift in goal_deltas. Mirrors the banner shown in the Excel
    # Optimizer tab so the two outputs agree.
    tol = float(cfg.rebalancing_target_tolerance_pctg or 0.0)
    max_abs_delta = 0.0
    if m.goal_deltas is not None and not m.goal_deltas.empty:
        non_cash = m.goal_deltas[m.goal_deltas["type"] != "cash"]
        if not non_cash.empty:
            max_abs_delta = float(non_cash["delta_pct"].abs().max())
    n_actions = len(m.rebalancing_suggestions or [])

    # The engine flags every verification entry with no_solution=True
    # when the LP returned 0 actions because no plan was feasible at
    # the configured tolerance ceiling (distinct from "already
    # aligned"). A serialized-action funding proof is a separate final
    # authority: draft actions are not actionable unless it is executable.
    rebal_infeasible = bool(
        m.rebalancing_verifications
        and any(v.get("no_solution") for v in m.rebalancing_verifications)
    )
    funding = _funding_verification(m.rebalancing_verifications)
    funding_non_executable = bool(
        funding and funding.get("status") == "NON_EXECUTABLE"
    )

    if rebal_infeasible:
        rebal_label = "Infeasible"
        rebal_sublabel = "no feasible plan"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif funding_non_executable:
        rebal_label = "Not executable"
        rebal_sublabel = "cash funding unresolved"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif n_actions == 0:
        # Solved cleanly with no trades (inside tolerance, or pinned by
        # locked positions / auto-relax). Nothing for the user to do.
        rebal_label = "Aligned"
        rebal_sublabel = "no action needed"
        rebal_color = PALETTE["green"]
        rebal_bg = PALETTE["green_bg"]
    else:
        # Actions to take. Communicate only that action is needed — no
        # count, no sublabel. The Optimizer section below carries specifics.
        rebal_label = "Action"
        rebal_sublabel = ""
        # Amber for a moderate plan, red when drift is well beyond tol.
        if max_abs_delta > 2 * tol:
            rebal_color, rebal_bg = PALETTE["red"], PALETTE["red_bg"]
        else:
            rebal_color, rebal_bg = PALETTE["amber"], PALETTE["amber_bg"]

    # The ten risk figures close STATE. They were section [11] RISK, and the
    # question they answer -- what shape was the ride -- is a property of the state
    # rather than a topic of its own, so they sit with the value and the return
    # measures instead of eleven sections later. The weak/fair/strong gauge that
    # used to sit beside each figure is gone: the configured bands name themselves
    # per metric, so the band is the caption and its colour is the figure's.
    from tarzan.export.newsletter._risk_tiles import risk_tiles as _risk_tiles

    state_tiles.extend(_risk_tiles(m, _tile))

    return {
        # Hero big number, rounded to whole euros (e.g. €XXX,XXX): decimals add
        # visual noise to the largest figure in the Status section.
        "total_value": _eur(m.total_value, decimals=0),
        # Total PnL — lifetime, realized + unrealized (order path).
        "total_pnl_eur": _eur_smart(total_pnl_eur, signed=True),
        "total_pnl_pct": _pct(total_pnl_pct, signed=True),
        "total_pnl_is_positive": total_pnl_eur >= 0,
        # Unrealized PnL — snapshot gain on current holdings (= Excel RTD).
        "unrealized_eur": _eur_smart(unrealized_eur, signed=True),
        "unrealized_pct": _pct(unrealized_pct, signed=True),
        "unrealized_is_positive": unrealized_eur >= 0,
        # Whether the lifetime Total PnL is a real (order-derived) figure
        # distinct from Unrealized; False on the holdings-only path.
        "has_total_pnl": has_total_pnl,
        # "Since inception · Mon YYYY" caption (empty when inception unknown).
        "inception_label": inception_label,
        # The nine state tiles, in display order. Optional ones are absent
        # rather than blank, so the grid never shows an empty box.
        "tiles": tuple(state_tiles),
        # Kept for the preheader/back-compat: the headline % is Total PnL.
        "gain_pct": _pct(total_pnl_pct, signed=True),
        # Cumulative time-weighted return since inception (order path only).
        "twr_pct": _pct(twr_pct, signed=True) if twr_pct is not None else None,
        "twr_is_positive": (twr_pct or 0.0) >= 0,
        # Dual-axis hero chart (value € + Unrealized PnL %) and cash-flow
        # chips, pre-rendered as safe HTML (empty on the holdings-only path).
        "value_chart": value_chart_html or None,
        "value_chart_legend": hero_chart_legend or None,
        "flow_chips_html": hero_flow_chips or None,
        "invested_value": _eur_smart(m.invested_value),
        "invested_pct": _pct(invested_pct, decimals=1),
        "cash_value": _eur_smart(m.cash_value),
        "cash_msg": cash_msg,
        "cash_msg_color": cash_msg_color,
        # This week: real money P&L (€ + %, net of contributions) and the
        # 1-week TWR — both clearly labeled in the template.
        "week_pnl_eur": _eur_smart(week_pnl_eur, signed=True) if week_pnl_eur is not None else None,
        "week_pnl_pct": _pct(week_pnl_pct, signed=True) if week_pnl_pct is not None else None,
        "week_pnl_is_positive": (week_pnl_eur or 0.0) >= 0,
        "week_twr_pct": _pct(week_twr, signed=True) if week_twr is not None else None,
        "week_twr_is_positive": (week_twr or 0.0) >= 0,
        # Last 30 days (mirrors the chart window): money P&L + TWR.
        "month_pnl_eur": _eur_smart(month_pnl_eur, signed=True) if month_pnl_eur is not None else None,
        "month_pnl_pct": _pct(month_pnl_pct, signed=True) if month_pnl_pct is not None else None,
        "month_pnl_is_positive": (month_pnl_eur or 0.0) >= 0,
        "month_twr_pct": _pct(month_twr, signed=True) if month_twr is not None else None,
        "month_twr_is_positive": (month_twr or 0.0) >= 0,
        # Annualized returns (card subtitles): money-weighted XIRR vs
        # time-weighted TWR-annualized.
        "xirr_pct": _pct(m.xirr_pct, signed=True) if m.xirr_pct is not None else None,
        "twr_annualized_pct": _pct(m.twr_annualized_pct, signed=True) if m.twr_annualized_pct is not None else None,
        # Net-of-tax ESTIMATE (order path only). Shown as a small line under
        # the since-inception PnL and as a sub-line in the XIRR annualized
        # card; the gross figures above are never altered. `has_net_tax` is
        # False (all keys None) when no CGT was estimated.
        "has_net_tax": (m.estimated_cgt_eur is not None and m.estimated_cgt_eur > 0),
        "cgt_eur": _eur_smart(-m.estimated_cgt_eur, signed=True) if m.estimated_cgt_eur else None,
        "pnl_net_tax_eur": _eur_smart(m.pnl_eur_net_tax, signed=True) if m.pnl_eur_net_tax is not None else None,
        "pnl_net_tax_pct": _pct(m.pnl_pct_net_tax, signed=True) if m.pnl_pct_net_tax is not None else None,
        "pnl_net_tax_is_positive": (m.pnl_eur_net_tax or 0.0) >= 0,
        "xirr_net_tax_pct": _pct(m.xirr_net_tax_pct, signed=True) if m.xirr_net_tax_pct is not None else None,
        # Rebalance status KPI replaces the "This Week" KPI which was
        # already covered by the TL;DR headline above.
        "rebal_label": rebal_label,
        "rebal_sublabel": rebal_sublabel,
        "rebal_color": rebal_color,
        "rebal_bg": rebal_bg,
        "rebal_n_actions": n_actions,
        "rebal_executable": not rebal_infeasible and not funding_non_executable,
    }

def _build_tax_note(ctx: _NewsletterContext) -> dict:
    """Bottom-of-newsletter net-of-tax estimate: the net figures plus the
    methodology disclaimer, moved out of the Performance card so it does not
    crowd the headline numbers. Renders only when a CGT estimate exists."""
    m = ctx.metrics
    P = PALETTE
    if not (m.estimated_cgt_eur and m.estimated_cgt_eur > 0):
        return {"available": False, "html": ""}

    def _sgn(v):
        return P["green"] if (v is not None and v >= 0) else P["red"]

    figs = []
    if m.xirr_net_tax_pct is not None:
        figs.append(f'XIRR net of tax <strong style="color:{_sgn(m.xirr_net_tax_pct)};">'
                    f'{_pct(m.xirr_net_tax_pct, signed=True)}</strong>')
    if m.pnl_eur_net_tax is not None:
        figs.append(f'P&amp;L net of tax <strong style="color:{_sgn(m.pnl_eur_net_tax)};">'
                    f'{_eur_smart(m.pnl_eur_net_tax, signed=True)}</strong>')
    figs_html = (" &nbsp;·&nbsp; ".join(figs)) if figs else ""
    # Rates from config, not hardcoded: the note only renders when a CGT
    # estimate exists (rates configured), so this states the rates actually
    # applied — no drift if the user runs a non-26%/12.5% jurisdiction.
    cfg = ctx.config
    std = float(cfg.rebalancing_capital_gains_tax_standard_pctg or 0.0)
    gov = float(cfg.rebalancing_capital_gains_tax_government_pctg or 0.0)
    rate_txt = f"{std:g}% / {gov:g}% on government bonds"
    html = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:{P["card_alt"]};border:1px solid {P["border"]};border-radius:10px;'
        f'border-collapse:separate;border-spacing:0;"><tr><td style="padding:12px 14px;">'
        f'<div style="{TYPE["prose"]}color:{P["muted"]};">'
        f'<span style="font-weight:700;color:{P["ink"]};">Net-of-tax estimate</span>'
        + (f' &nbsp;{figs_html}' if figs_html else "")
        + f'<div style="margin-top:4px;{TYPE["prose"]}color:{P["subtle"]};">'
        f'Estimate only: average-cost basis, '
        f'{rate_txt}, realized losses offset later gains where Italian rules allow '
        f'(ETF/fund gains are not offsettable). Excludes coupon/dividend withholding and the cost basis of '
        f'transferred-in positions. TWR is gross of tax.</div>'
        f'</div></td></tr></table>'
    )
    return {"available": True, "html": html}

# Presence bands, in the order the table lists them: why this instrument is in
# this issue at all. Also the sort's primary key — within a band rows are
# alphabetical by ticker, and there is no class/role grouping: this is a
# reference table looked up by symbol, not a view of the portfolio's shape.
_PRESENCE_BANDS = ("Portfolio", "Watchlist", "Hist. Portfolio only")

# Holding-resolution presence -> band. "Rebalance target" (a seeded optimizer
# target, never held) has no band and keeps its own label, sorted last: a target
# that is also a watchlist instrument is upgraded to Watchlist below, so what
# survives this map is genuinely neither held nor tracked.
_PRESENCE_BAND_OF = {
    "Current + Historical": "Portfolio",
    "Current": "Portfolio",
    "Historical only": "Hist. Portfolio only",
}


def _presence_rank(presence: str) -> int:
    return (_PRESENCE_BANDS.index(presence) if presence in _PRESENCE_BANDS
            else len(_PRESENCE_BANDS))


def _build_ticker_sources(ctx: _NewsletterContext) -> dict:
    """Appendix feed audit: every instrument this issue names, with the exact
    provider listings behind its daily bars and its intraday series.

    Two sources, because neither alone covers the issue. ``ticker_resolutions``
    holds every carrier the portfolio has ever owned — including one with no
    feed at all (a BTP), which the performance frame drops for having under two
    price rows. The benchmark catalog holds the watchlist, which is not a
    holding and so has no resolution record. They meet on ISIN, then on bare
    ticker (a rebalance target carries a symbol but no ISIN), so an instrument
    that is both is one row: held wins, exactly as the Watchlist table drops a
    benchmark the portfolio owns.
    """
    from tarzan import config as _cfg
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    m = ctx.metrics
    rows: list[dict] = []
    by_identity: dict[str, dict] = {}

    def _add(row: dict, *keys: str) -> None:
        rows.append(row)
        for key in keys:
            if key:
                by_identity.setdefault(key, row)

    for record in getattr(m, "ticker_resolutions", ()) or ():
        canonical = str(record.get("canonical_ticker") or "")
        isin = normalize_isin(record.get("isin"))
        presence = str(record.get("portfolio_presence") or "")
        _add({
            "ticker": _esc(_display_ticker(canonical) or ""),
            # Curated names carry ampersands ("iShares Core S&P 500", the Return
            # Stacked family); the template does not escape (.html.j2), so the
            # appendix escapes at its own boundary like every other builder.
            "name": _esc(_cfg.name_for(isin, canonical)
                         or str(record.get("name") or "")),
            "isin": isin,
            # The listing the daily bars came from; the selected symbol when the
            # instrument resolved but never returned history.
            "hist_ric": str(record.get("history_ticker")
                            or record.get("current_ticker") or canonical),
            "intr_ric": str(record.get("intraday_effective_ticker") or ""),
            "presence": _PRESENCE_BAND_OF.get(presence, presence),
        }, isin, normalize_ticker(canonical))

    resolved = getattr(m, "benchmark_tickers", {}) or {}
    quotes = getattr(m, "intraday_quotes", {}) or {}
    for name, requested, isin in _cfg.benchmark_identities():
        hist = str(resolved.get(name) or "")
        known = next((by_identity[k] for k in (isin, normalize_ticker(hist or requested))
                      if k and k in by_identity), None)
        if known is not None:
            if _presence_rank(known["presence"]) > _PRESENCE_BANDS.index("Watchlist"):
                known["presence"] = "Watchlist"
            known["isin"] = known["isin"] or isin
            continue
        quote = quotes.get(hist)
        quote = quote if isinstance(quote, dict) else {}
        _add({
            "ticker": _esc(_display_ticker(requested) or requested),
            "name": _esc(name),
            "isin": isin,
            "hist_ric": hist,
            # A benchmark's intraday series can come from a sibling venue, and
            # it is the catalog — not the request — that records which one.
            "intr_ric": str(quote.get("intraday_source_ticker")
                            or quote.get("source_ticker") or ""),
            "presence": "Watchlist",
        }, isin, normalize_ticker(hist or requested))

    rows.sort(key=lambda r: (_presence_rank(r["presence"]),
                             (r["ticker"] or r["name"]).upper()))
    return {"available": bool(rows), "rows": tuple(rows)}


def _build_allocation(ctx: _NewsletterContext) -> dict:
    """Build asset-class allocation rows (Excel Dashboard pattern)."""
    m = ctx.metrics
    cfg = ctx.config
    targets = cfg.invested_allocation_targets_pctg or {}
    alloc_df = m.allocation_by_class

    timeline = m.allocation_timeline or {}
    asset_series = timeline.get("asset")

    # Physical capital per class (sum of the market value of holdings whose
    # PRIMARY class is that class) — the denominator for the per-class
    # leverage = notional exposure / physical capital. >1 means the class is
    # partly synthetic (e.g. an efficient-core bond overlay).
    inv_val = float(getattr(m, "invested_value", 0.0) or 0.0)
    phys_by_class: dict[str, float] = {}
    hdf = m.holdings_df
    if hdf is not None and not hdf.empty and "asset_class" in hdf.columns:
        for cls, grp in hdf.groupby("asset_class"):
            phys_by_class[str(cls)] = float(grp["current_value"].sum())

    rows = []
    for klass in ASSET_CLASS_ORDER:
        match = (alloc_df[alloc_df["category"] == klass]
                 if not alloc_df.empty else alloc_df)
        has_holding = match is not None and not match.empty
        target = targets.get(klass)
        # Show a class if it is held OR it carries a (non-zero) target, so a
        # targeted-but-not-yet-held class appears as Now 0% vs its target.
        if not has_holding and not (target and target > 0):
            continue
        actual = float(match["weight_pct"].iloc[0]) if has_holding else 0.0
        delta = actual - target if target is not None else None
        color = ASSET_COLORS.get(klass, PALETTE["accent"])
        spark_vals = _timeline_vals(asset_series, klass)
        notional_eur = actual / 100.0 * inv_val
        phys = phys_by_class.get(klass, 0.0)
        leverage = (notional_eur / phys) if phys > 0 else None
        rows.append({
            "name": klass,
            "color": color,
            "actual_pct": _pct_smart(actual),
            "actual_pct_raw": actual,
            "target_pct": _pct_smart(target) if target is not None else None,
            "target_left": (
                min(max(float(target), 0), 100)
                if target is not None else None
            ),
            "delta": _signed_pp(delta) if delta is not None else None,
            "delta_color": _band_colour(delta, target, cfg),
            "bar_width": min(max(actual, 1), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
            "leverage": leverage,
        })

    # Cash buffer (EUR-based, appended after invested classes).
    # The bar width is scaled as % of total portfolio so cash visually
    # matches the other rows (it would otherwise dominate the bar
    # because target_cash_buffer_eur is small relative to invested
    # value). Status colour: inside ±allocation_band_rel_pctg of the cash
    # target is green, outside is red.
    if cfg.target_cash_buffer_eur > 0:
        cash_actual = m.cash_value
        cash_tgt = cfg.target_cash_buffer_eur
        delta_eur = cash_actual - cash_tgt
        cash_band = float(getattr(cfg, "allocation_band_rel_pctg", 25.0)) / 100.0 * cash_tgt
        cash_pct_of_total = (cash_actual / m.total_value * 100) if m.total_value > 0 else 0
        rows.append({
            # Shorter label only inside the Diversification block where
            # horizontal space is critical; other sections (Holdings,
            # Optimizer, Insights) keep the full "Cash & Cash
            # Equivalents" string.
            "name": "Cash & Cash Eq.",
            "color": ASSET_COLORS["Cash & Cash Equivalents"],
            "actual_pct": _eur_smart(cash_actual),
            "actual_pct_raw": cash_pct_of_total,
            "target_pct": _eur_smart(cash_tgt),
            "delta": _eur_smart(delta_eur, signed=True),
            "delta_color": (PALETTE["green"] if abs(delta_eur) <= cash_band + 1e-9
                            else PALETTE["red"]),
            "bar_width": min(max(cash_pct_of_total, 1), 100),
            "is_eur": True,
            # Raw EUR figures so the diversification table can show cash as a
            # normal row without it participating in the invested base.
            "cash_actual_eur": cash_actual,
            "cash_target_eur": cash_tgt,
            "cash_delta_eur": delta_eur,
        })

    return {
        "rows": rows,
        "has_timeline": any(r.get("spark") for r in rows),
    }

def _build_geography(ctx: _NewsletterContext) -> dict:
    """Build geographic equity rows with target & ACWI ticks."""
    m = ctx.metrics
    cfg = ctx.config
    targets = cfg.equity_geo_targets_pctg or {}
    geo_df = m.allocation_by_geo
    acwi = m.acwi_geo or {}

    timeline = m.allocation_timeline or {}
    geo_series = timeline.get("geo")

    # Actual equity-geo weights, plus any region that only has a target (so a
    # targeted-but-absent region still shows as Now 0% vs its target).
    actual_by_region: dict[str, float] = {}
    if not geo_df.empty:
        for _, r in geo_df.iterrows():
            actual_by_region[str(r["category"])] = float(r["weight_pct"])
    regions = list(actual_by_region.keys())
    for region in targets:
        if region not in actual_by_region and (targets.get(region) or 0) > 0:
            regions.append(region)
    # Order by actual descending (target-only regions, actual 0, sort last).
    regions.sort(key=lambda rg: -actual_by_region.get(rg, 0.0))

    rows = []
    for region in regions:
        actual = actual_by_region.get(region, 0.0)
        target = targets.get(region)
        acwi_v = acwi.get(region)
        delta_target = actual - target if target is not None else None
        color = GEO_COLORS.get(region, PALETTE["accent"])
        spark_vals = _timeline_vals(geo_series, region)
        rows.append({
            "name": region,
            "color": color,
            "actual_pct": _pct_smart(actual),
            "actual_pct_raw": actual,
            "target_pct": _pct_smart(target) if target is not None else "—",
            "acwi_pct": _pct_smart(acwi_v) if acwi_v is not None else "—",
            "delta": _signed_pp(delta_target) if delta_target is not None else "—",
            "delta_color": _band_colour(delta_target, target, cfg),
            "bar_width": min(max(actual, 1), 100),
            "target_left": min(max(target or 0, 0), 100),
            "acwi_left": min(max(acwi_v or 0, 0), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
        })

    return {
        "rows": rows,
        "benchmark_name": ctx.benchmark_geo,
        "has_timeline": any(r.get("spark") for r in rows),
    }

def _recent_timeline(series: Optional[list], dates: Optional[list],
                     days: int = 31) -> Optional[list]:
    """Slice a timeline series to its last ``days`` (the 1-month trend window).
    Falls back to the last 5 buckets so a sparkline always has ≥2 points."""
    if not series or not dates:
        return series
    cutoff = pd.Timestamp(dates[-1]) - pd.Timedelta(days=days)
    keep = [i for i, d in enumerate(dates) if pd.Timestamp(d) >= cutoff]
    if len(keep) < 2:
        keep = list(range(max(0, len(dates) - 5), len(dates)))
    return [series[i] for i in keep]

def _signed_eur(value) -> str:
    """A signed euro gap: "+EUR2.5k". Cash is held as an amount, not a share, so its
    row's drift cannot be points."""
    if value is None:
        return ""
    amount = float(value)
    return f'{"+" if amount >= 0 else "\u2212"}{_eur_smart(abs(amount))}'


# ── Allocation: the bridge ───────────────────────────────────────────────────
#
# Each block is ONE graphic, and the graphic is the table. Two stacked columns,
# today and the plan, each slice joined to itself: a band that widens is weight to
# add, one that narrows is weight to take out. Lines the plan sells funnel into one
# point; lines the plan buys from nothing grow as a wedge out of it.
#
# The figures sit in columns of the graphic itself, each said once:
#   left of TODAY   name, weight now          /  1M move, euros now
#   right of PLAN   plan weight, gap (points) /  plan euros
# A name is written on the side where the line has weight: a line held today is
# named on the left, a line only the plan holds is named on its own wedge.
#
# Drawn in a 350-unit viewBox (a phone's content width) at width:100%, so it is
# 1:1 on a phone and scales with the column anywhere wider.

_BW = 350.0
_FS, _FS2 = 12, 11               # label line, its second line
_CW, _CW2 = 7.25, 6.65           # advance of one monospace character at each
_SLOT = 32.0                     # vertical room per two-line label
_BAR, _LEAD, _COLGAP = 10, 6, 9
_TOP, _GAP = 22.0, 2.0
#: Successive lines of one asset class step through these, so two names of the
#: same class stay apart while keeping the class's hue.
_SHADE_STEPS = (("base", 0.0), ("card", 0.62), ("ink", 0.45), ("card", 0.40),
                ("ink", 0.22))
_LEGACY_SHADES = (0.95, 0.62, 0.84, 0.52, 0.74, 0.90, 0.57, 0.80, 0.66, 0.48)
_ASSET_SHORT = {"Fixed Income": "Fixed inc."}
_GEO_SHORT = {"Eurozone EMU": "Eurozone", "Dev ex-USA ex-EMU ex-JP": "Other dev",
              "Emerging Markets": "Emerging"}


def _mix(fg: str, bg: str, a: float) -> str:
    """``fg`` over ``bg`` at ``a``, as a flat hex (an emailed rgba() is unreliable)."""
    a = max(0.0, min(1.0, a))
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(f[i] * a + b[i] * (1 - a))):02x}" for i in range(3))


def _lift(colour: Optional[str]) -> str:
    """The class colours were tuned for a light page; lifted toward white they hold
    on the dark card without changing hue."""
    if not colour or not str(colour).startswith("#") or len(str(colour)) != 7:
        return PALETTE["accent"]
    return _mix("#FFFFFF", str(colour), 0.28)


def _shade(base: str, n: int) -> str:
    toward, a = _SHADE_STEPS[n % len(_SHADE_STEPS)]
    if toward == "base":
        return base
    if toward == "card":
        return _mix(base, PALETTE["card"], a)
    return _mix("#FFFFFF", base, a)


def _alloc_band(target: Optional[float], cfg) -> float:
    """The display band around a target: the NARROWER of ±``allocation_band_abs_pp``
    points and ±``allocation_band_rel_pctg`` % of the target (the 5/25 rule).

    Display only. The optimizer keeps ``rebalancing_target_tolerance_pctg``: it is the
    solver's constraint for the day the owner rebalances, a different question from
    "is this line out of band".
    """
    abs_pp = float(getattr(cfg, "allocation_band_abs_pp", 5.0) or 0.0)
    rel = float(getattr(cfg, "allocation_band_rel_pctg", 25.0) or 0.0) / 100.0
    return min(abs_pp, rel * abs(float(target or 0.0)))


def _band_colour(drift: Optional[float], target: Optional[float], cfg) -> str:
    """Green inside the band, red outside it. Two states: outside the band is the
    rebalancing trigger, and there is no third answer to that question."""
    if drift is None or is_missing(drift) or target is None:
        return PALETTE["muted"]
    return (PALETTE["green"] if abs(float(drift)) <= _alloc_band(target, cfg) + 1e-9
            else PALETTE["red"])


def _dodge(ys: list, gap: float, lo: float, hi: float) -> list:
    """Push label centres apart to at least ``gap``, inside [lo, hi], keeping order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for _ in range(80):
        moved = False
        for a, b in zip(order, order[1:]):
            if out[b] - out[a] < gap - 1e-6:
                push = (gap - (out[b] - out[a])) / 2
                out[a] -= push
                out[b] += push
                moved = True
        for i in order:
            out[i] = min(max(out[i], lo), hi)
        if not moved:
            break
    return out


def _bt(x: float, y: float, s: str, *, fill: str, weight: int = 400,
        anchor: str = "start", size: int = _FS) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{_esc(str(s))}</text>')


def _pc1(v: float) -> str:
    return f"{float(v):.1f}%"


def _move_text(trend: Optional[list]) -> str:
    tr = [float(x) for x in (trend or []) if x is not None]
    return f"1M {_signed_pp(tr[-1] - tr[0])}" if len(tr) >= 2 else ""


def _bridge_row(*, key: str, label: str, colour: str, now: float,
                target: Optional[float], base: float, trend=None,
                legacy: bool = False) -> dict:
    now = float(now or 0.0)
    tgt = None if target is None else float(target)
    return {"key": key, "label": label, "colour": colour, "now": now, "target": tgt,
            "drift": None if tgt is None else now - tgt,
            "eur_now": now / 100.0 * base, "eur_tgt": (tgt or 0.0) / 100.0 * base,
            "trend": trend, "legacy": legacy}


def _bridge_svg(left: list, right: list, stack_h: float, cfg, *, tail=(),
                rule100: bool = False, total_now: Optional[float] = None,
                total_tgt: Optional[float] = None,
                aria: str = "Allocation, today against the plan") -> str:
    """One bridge block. ``left``/``right`` are rows in stack order; ``tail`` rows
    sit under the stacks in the same columns:
    (name, 1M text, now text, now euros, plan text, plan euros, gap text, colour)."""
    P = PALETTE
    l_rows = [r for r in left if r["now"] > 0]
    r_rows = [r for r in right if (r["target"] or 0) > 0]
    if not l_rows and not r_rows:
        return ""
    gap_txt = lambda r: _signed_pp(r["drift"]) if r["drift"] is not None else "\u2014"  # noqa: E731
    w1 = lambda xs: max([len(s) for s in xs] + [0]) * _CW   # noqa: E731
    w2 = lambda xs: max([len(s) for s in xs] + [0]) * _CW2  # noqa: E731
    name_w = max(w1([r["label"] for r in l_rows] + [x[0] for x in tail]),
                 w2([_move_text(r["trend"]) for r in l_rows] + [x[1] for x in tail]))
    now_w = max(w1([_pc1(r["now"]) for r in l_rows] + [x[2] for x in tail] + ["TODAY"]),
                w2([_eur_smart(r["eur_now"]) for r in l_rows] + [x[3] for x in tail]))
    plan_w = max(w1([_pc1(r["target"]) for r in r_rows] + [x[4] for x in tail] + ["PLAN"]),
                 w2([_eur_smart(r["eur_tgt"]) for r in r_rows] + [x[5] for x in tail]))
    gap_w = w1([gap_txt(r) for r in r_rows] + [x[6] for x in tail] + ["GAP"])
    now_end = name_w + _COLGAP + now_w
    LX0 = now_end + _LEAD
    LX1 = LX0 + _BAR
    gap_end = _BW
    plan_end = gap_end - gap_w - _COLGAP
    RX1 = plan_end - plan_w - _LEAD
    RX0 = RX1 - _BAR
    flow = RX0 - LX1

    tn = total_now if total_now is not None else sum(r["now"] for r in l_rows)
    tt = total_tgt if total_tgt is not None else sum(r["target"] for r in r_rows)
    span = max(tn, tt, 1e-9)
    k = (stack_h - (max(len(l_rows), len(r_rows), 1) - 1) * _GAP) / span

    def stack(rows, key):
        y, out = _TOP, {}
        for r in rows:
            v = r[key] or 0.0
            out[r["key"]] = (y, v * k)
            y += v * k + (_GAP if v > 0 else 0)
        return out

    L, R = stack(left, "now"), stack(right, "target")
    bottom = _TOP + stack_h
    hdr = dict(fill=P["subtle"], weight=700, size=10)
    g = [_bt(now_end, 11, "TODAY", anchor="end", **hdr),
         _bt(plan_end, 11, "PLAN", anchor="end", **hdr),
         _bt(gap_end, 11, "GAP", anchor="end", **hdr)]

    leg = [r for r in left if r["legacy"] and r["now"] > 0]
    fan = None
    if leg:
        y0 = L[leg[0]["key"]][0]
        y1 = L[leg[-1]["key"]][0] + L[leg[-1]["key"]][1]
        fan = (LX1 + 0.56 * flow, (y0 + y1) / 2)
    by_key = {r["key"]: r for r in right + left}
    mid = (LX1 + RX0) / 2
    flows, bars, inks = [], [], []
    for key in dict.fromkeys([r["key"] for r in left] + [r["key"] for r in right]):
        r = by_key[key]
        ly, lh = L.get(key, (_TOP, 0.0))
        ry, rh = R.get(key, (_TOP, 0.0))
        c = r["colour"]
        if lh > 0 and rh > 0:
            d = (f'M{LX1:.1f},{ly:.1f} C{mid:.1f},{ly:.1f} {mid:.1f},{ry:.1f} {RX0:.1f},{ry:.1f} '
                 f'L{RX0:.1f},{ry + rh:.1f} C{mid:.1f},{ry + rh:.1f} {mid:.1f},{ly + lh:.1f} '
                 f'{LX1:.1f},{ly + lh:.1f} Z')
        elif lh > 0:
            # Sold: every legacy line funnels into ONE point, the sale being one
            # decision. A line with no target at all just tapers off.
            tx, ty = fan if (fan and r["legacy"]) else (mid, ly + lh / 2)
            cx = (LX1 + tx) / 2
            d = (f'M{LX1:.1f},{ly:.1f} C{cx:.1f},{ly:.1f} {tx - 12:.1f},{ty:.1f} {tx:.1f},{ty:.1f} '
                 f'C{tx - 12:.1f},{ty:.1f} {cx:.1f},{ly + lh:.1f} {LX1:.1f},{ly + lh:.1f} Z')
        elif rh > 0:
            # Bought from zero: a wedge that starts just past the sale point.
            tx = (fan[0] + 10) if fan else (RX0 - 0.5 * flow)
            ty = ry + rh / 2
            cx = (tx + RX0) / 2
            d = (f'M{RX0:.1f},{ry:.1f} C{cx:.1f},{ry:.1f} {tx + 12:.1f},{ty:.1f} {tx:.1f},{ty:.1f} '
                 f'C{tx + 12:.1f},{ty:.1f} {cx:.1f},{ry + rh:.1f} {RX0:.1f},{ry + rh:.1f} Z')
            # Its name on the wedge when the wedge can hold it, else just before
            # the tip, where nothing else is drawn.
            if rh >= 13:
                inks.append(_bt(RX0 - 5, ty + 4, r["label"], fill=P["ink"], weight=700,
                                anchor="end", size=11))
            else:
                inks.append(_bt(tx - 4, ty + 4, r["label"], fill=P["ink"], weight=700,
                                anchor="end", size=11))
        else:
            continue
        flows.append(f'<path d="{d}" fill="{c}" fill-opacity="0.30"/>')
        if lh > 0:
            bars.append(f'<rect x="{LX0:.1f}" y="{ly:.1f}" width="{_BAR}" height="{lh:.1f}" '
                        f'fill="{c}"/>')
        if rh > 0:
            bars.append(f'<rect x="{RX0:.1f}" y="{ry:.1f}" width="{_BAR}" height="{rh:.1f}" '
                        f'fill="{c}"/>')
    g += flows + bars + inks
    if fan:
        sx = LX1 + 7
        g += [f'<circle cx="{fan[0]:.1f}" cy="{fan[1]:.1f}" r="2.6" fill="{P["red"]}"/>',
              _bt(sx, fan[1] - 10, "sell", fill=P["ink"], weight=700, size=11),
              _bt(sx, fan[1] + 4, _pc1(sum(r["now"] for r in leg)), fill=P["ink"],
                  weight=700, size=11),
              _bt(sx, fan[1] + 18, _eur_smart(sum(r["eur_now"] for r in leg)),
                  fill=P["ink"], weight=700, size=11)]
    if rule100:
        y100 = _TOP + 100 * k + _GAP * 3
        g.append(f'<line x1="{LX0 - 3:.1f}" y1="{y100:.1f}" x2="{RX1 + 3:.1f}" y2="{y100:.1f}" '
                 f'stroke="{P["ink"]}" stroke-opacity="0.75" stroke-dasharray="3,3"/>'
                 f'<rect x="{mid - 19:.1f}" y="{y100 - 7.5:.1f}" width="38" height="15" rx="3" '
                 f'fill="{P["card"]}"/>'
                 + _bt(mid, y100 + 4, "100%", fill=P["ink"], weight=700, anchor="middle",
                       size=10))

    def place(rows, stk):
        y0s = [stk[r["key"]][0] + stk[r["key"]][1] / 2 for r in rows]
        return y0s, _dodge(y0s, _SLOT, _TOP + 10, bottom + 4)

    low = 0.0
    y0s, ys = place(l_rows, L)
    for r, y0, y in zip(l_rows, y0s, ys):
        if abs(y - y0) > 1.0:
            g.append(f'<path d="M{now_end + 3:.1f},{y:.1f} L{LX0 - 2:.1f},{y0:.1f}" '
                     f'stroke="{P["subtle"]}" stroke-width="0.8" fill="none"/>')
        g += [_bt(0, y - 1, r["label"], fill=P["red"] if r["legacy"] else P["ink"], weight=600),
              _bt(now_end, y - 1, _pc1(r["now"]), fill=P["ink"], weight=700, anchor="end"),
              _bt(0, y + 11.5, _move_text(r["trend"]), fill=P["muted"], size=_FS2),
              _bt(now_end, y + 11.5, _eur_smart(r["eur_now"]), fill=P["muted"],
                  anchor="end", size=_FS2)]
        low = max(low, y)
    y0s, ys = place(r_rows, R)
    for r, y0, y in zip(r_rows, y0s, ys):
        if abs(y - y0) > 1.0:
            g.append(f'<path d="M{RX1 + 2:.1f},{y0:.1f} L{RX1 + _LEAD - 3:.1f},{y:.1f}" '
                     f'stroke="{P["subtle"]}" stroke-width="0.8" fill="none"/>')
        g += [_bt(plan_end, y - 1, _pc1(r["target"]), fill=P["ink"], weight=700, anchor="end"),
              _bt(gap_end, y - 1, gap_txt(r), fill=_band_colour(r["drift"], r["target"], cfg),
                  weight=700, anchor="end"),
              _bt(plan_end, y + 11.5, _eur_smart(r["eur_tgt"]), fill=P["muted"],
                  anchor="end", size=_FS2)]
        low = max(low, y)

    H = max(bottom + 6, low + 18)
    if tail:
        g.append(f'<line x1="0" y1="{H + 2:.1f}" x2="{_BW:.0f}" y2="{H + 2:.1f}" '
                 f'stroke="{P["border"]}"/>')
        y = H + 18
        for nm, mv, npc, neu, ppc, peu, gp, gc in tail:
            g += [_bt(0, y, nm, fill=P["ink"], weight=600),
                  _bt(now_end, y, npc, fill=P["ink"], weight=700, anchor="end"),
                  _bt(plan_end, y, ppc, fill=P["ink"], weight=700, anchor="end"),
                  _bt(gap_end, y, gp, fill=gc, weight=700, anchor="end")]
            if mv or neu:
                g += [_bt(0, y + 13, mv, fill=P["muted"], size=_FS2),
                      _bt(now_end, y + 13, neu, fill=P["muted"], anchor="end", size=_FS2),
                      _bt(plan_end, y + 13, peu, fill=P["muted"], anchor="end", size=_FS2)]
                y += 32
            else:
                y += 18
        H = y - 6
    H = int(H) + 2
    return (f'<svg width="100%" viewBox="0 0 {_BW:.0f} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" aria-label="{_esc(aria)}" style="display:block;margin-top:10px;" '
            f'font-family="{FONT_STACK}">{"".join(g)}</svg>')


def _alloc_head(title: str, note: str, top: int) -> str:
    P = PALETTE
    return (f'<div style="margin-top:{top}px;font-size:11px;font-weight:700;'
            f'letter-spacing:0.06em;text-transform:uppercase;color:{P["ink"]};">'
            f'{_esc(title)}</div><div style="margin-top:2px;font-size:12px;'
            f'line-height:1.45;color:{P["subtle"]};">{_esc(note)}</div>')


def _alloc_note(text: str, top: int = 8) -> str:
    return (f'<div style="margin-top:{top}px;font-size:12px;line-height:1.45;'
            f'color:{PALETTE["muted"]};">{_esc(text)}</div>')


def _holding_bridge_rows(ctx: _NewsletterContext, items: list, *, base: float,
                         weights: dict, series: Optional[list],
                         series_key: str) -> tuple[list, list, list]:
    """(kept, new, sold) bridge rows for one per-holding verification.

    ``weights`` maps an item's ISIN to its CURRENT weight (% of the block's base),
    from the snapshot, not the verification's post-trade ``actual_pct``. ``series``
    is the 1-month timeline for the block, keyed by ISIN or ticker (``series_key``).
    Each line takes its asset class's hue, stepped so two names of a class differ;
    lines the plan sells are shades of red.
    """
    m = ctx.metrics
    df = getattr(m, "holdings_df", None)
    isin_of: dict[str, str] = {}
    class_of: dict[str, str] = {}
    ticker_of: dict[str, str] = {}
    name_of: dict[str, str] = {}
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            isin = str(row.get("isin", "") or "").strip()
            if not isin:
                continue
            for key in (row.get("isin"), row.get("ticker"), row.get("name")):
                if key and str(key).strip():
                    isin_of[str(key).strip().upper()] = isin
            tkr = str(row.get("ticker", "") or "").strip()
            if tkr:
                isin_of[tkr.upper().split(".")[0]] = isin
            class_of[isin] = str(row.get("asset_class", "") or "")
            ticker_of[isin] = tkr
            name_of[isin] = str(row.get("name", "") or "")

    def isin_for(it) -> str:
        for key in (it.get("isin"), it.get("ticker"), it.get("category")):
            if not key:
                continue
            k = str(key).strip().upper()
            if k in isin_of:
                return isin_of[k]
            if k.split(".")[0] in isin_of:
                return isin_of[k.split(".")[0]]
        return ""

    def trend_for(it, isin) -> Optional[list]:
        if not series:
            return None
        key = isin if series_key == "isin" else str(it.get("ticker") or "")
        if not key:
            return None
        xs = [float(pt.get(key, 0.0)) for pt in series]
        return xs if any(x > 0 for x in xs) and len(xs) >= 2 else None

    kept, new, sold = [], [], []
    for it in items:
        isin = isin_for(it)
        tgt = float(it.get("target_pct", 0.0) or 0.0)
        now = (weights.get(isin, 0.0) if weights is not None
               else float(it.get("actual_pct", 0.0) or 0.0))
        if tgt <= 0 and now <= 0.05:
            continue
        label = _display_ticker(it.get("ticker") or "") or short_instrument_name(
            it.get("category") or "", 8)
        row = dict(item=it, isin=isin, now=now, tgt=tgt, label=label,
                   cls=class_of.get(isin, "Equities"))
        (sold if tgt <= 0 else (new if now <= 0.05 else kept)).append(row)
    if weights is not None:
        # Every line held today is in TODAY, listed by the check or not. The check
        # is post-trade, so a line a full rebalance sells outright drops out of it
        # while the book still holds it -- and by the plan's own rule an unlisted
        # holding is targeted to zero. Cash is not invested capital.
        seen = {r["isin"] for r in kept + new + sold if r["isin"]}
        for isin, w in weights.items():
            if (isin in seen or w <= 0.05
                    or class_of.get(isin, "").lower().startswith("cash")):
                continue
            tk = ticker_of.get(isin, "")
            sold.append(dict(item={"ticker": tk, "isin": isin}, isin=isin, now=w,
                             tgt=0.0, cls=class_of.get(isin, "Equities"),
                             label=_display_ticker(tk) or short_instrument_name(
                                 name_of.get(isin) or isin, 8)))
    kept.sort(key=lambda r: (-r["tgt"], r["label"]))
    new.sort(key=lambda r: (-r["tgt"], r["label"]))
    sold.sort(key=lambda r: (-r["now"], r["label"]))

    steps: dict[str, int] = {}

    def colour(cls: str) -> str:
        n = steps.get(cls, 0)
        steps[cls] = n + 1
        return _shade(_lift(ASSET_COLORS.get(cls) or PALETTE["accent"]), n)

    out_kept = [_bridge_row(key=r["isin"] or r["label"], label=r["label"],
                            colour=colour(r["cls"]), now=r["now"], target=r["tgt"],
                            base=base, trend=trend_for(r["item"], r["isin"]))
                for r in kept + new]
    out_sold = [_bridge_row(key="sell:" + (r["isin"] or r["label"]), label=r["label"],
                            colour=_mix(PALETTE["red"], PALETTE["card"],
                                        _LEGACY_SHADES[i % len(_LEGACY_SHADES)]),
                            now=r["now"], target=0.0, base=base,
                            trend=trend_for(r["item"], r["isin"]), legacy=True)
                for i, r in enumerate(sold)]
    return out_kept[:len(kept)], out_kept[len(kept):], out_sold


def _build_diversification(ctx: _NewsletterContext) -> dict:
    """Pre-render the Allocation section: three bridges (asset class, equity
    geography, per-holding targets), each with its figures built in.

    Reuses :func:`_build_allocation` / :func:`_build_geography` for the numbers,
    the rebalancer's per-holding checks for the instrument lines and the
    allocation timeline for the month's move -- a presentational layer, not a
    second source of truth.
    """
    P = PALETTE
    cfg = ctx.config
    m = ctx.metrics
    alloc = _build_allocation(ctx)
    geo = _build_geography(ctx)
    if not (alloc.get("rows") or geo.get("rows")):
        return {"available": False, "html": ""}

    invested_base = float(getattr(m, "invested_value", 0.0) or 0.0)
    if invested_base <= 0:
        invested_base = float(getattr(m, "total_value", 0.0) or 0.0)
    # Geography and per-sleeve rows are shares of the NOTIONAL sleeve
    # (``_compute_geo_allocation`` distributes each holding's notional exposure), so
    # their euro base is that same sleeve: the class weight times invested capital.
    # Multiplying a notional share by the physical market value mixes two
    # denominators -- it made Emerging Markets read fewer euros than XMME, its only
    # holding, was worth.
    byclass = getattr(m, "allocation_by_class", None)

    def notional_sleeve_eur(klass: str) -> float:
        if byclass is None or byclass.empty:
            return 0.0
        row = byclass[byclass["category"] == klass]
        return 0.0 if row.empty else float(row["weight_pct"].iloc[0]) / 100.0 * invested_base

    equity_base = notional_sleeve_eur("Equities")
    fi_base = notional_sleeve_eur("Fixed Income")

    tl = m.allocation_timeline or {}
    dates = tl.get("dates") or []
    asset_series = _recent_timeline(tl.get("asset"), dates)
    geo_series = _recent_timeline(tl.get("geo"), dates)
    hold_series = _recent_timeline(tl.get("holding"), dates)
    hold_inv_series = _recent_timeline(tl.get("holding_invested"), dates)

    # The plan's own leverage per class. Only meaningful because the class targets
    # are DERIVED from the per-instrument plan; without a plan there is none to show.
    target_lev: dict = {}
    try:
        from tarzan.engine.target_derivation import derive_target_leverage, plan_weights

        plan, _ = plan_weights(getattr(m, "target_rows", None) or {})
        if not plan:
            plan = dict(getattr(m, "target_weights", {}) or {})
        if plan:
            target_lev = derive_target_leverage(plan)
    except Exception:  # noqa: BLE001 -- a missing factor must not cost the section
        target_lev = {}

    html: list[str] = []
    abs_pp = float(getattr(cfg, "allocation_band_abs_pp", 5.0))
    rel = float(getattr(cfg, "allocation_band_rel_pctg", 25.0))

    # ── Asset class ──
    asset_rows, cash, levs = [], None, []
    for r in alloc.get("rows") or []:
        if r.get("is_eur"):
            cash = r
            continue
        asset_rows.append(_bridge_row(
            key=r["name"], label=_ASSET_SHORT.get(r["name"], r["name"]),
            colour=_lift(r.get("color")), now=r.get("actual_pct_raw") or 0.0,
            target=r.get("target_left"), base=invested_base,
            trend=_timeline_vals(asset_series, r["name"])))
        lev = r.get("leverage")
        has_plan = bool(target_lev) and (r.get("target_left") or 0) > 0
        tlev = target_lev.get(r["name"]) if has_plan else None
        if lev is not None and (abs(lev - 1) > 0.005
                                or (has_plan and (tlev is None or abs(tlev - 1) > 0.005))):
            part = f"{_ASSET_SHORT.get(r['name'], r['name']).lower()} {lev:.2f}\u00d7"
            if has_plan:
                part += " \u2192 " + ("overlay only" if tlev is None else f"{tlev:.2f}\u00d7")
            levs.append(part)
    if asset_rows:
        tn = sum(r["now"] for r in asset_rows)
        tt = sum(r["target"] or 0.0 for r in asset_rows)
        ttrend = ([sum(float(x) for x in b.values()) for b in asset_series]
                  if asset_series and len(asset_series) >= 2 else None)
        tail = [("Total", _move_text(ttrend), _pc1(tn), _eur_smart(tn / 100 * invested_base),
                 _pc1(tt), _eur_smart(tt / 100 * invested_base), _signed_pp(tn - tt),
                 _band_colour(tn - tt, tt, cfg))]
        if cash is not None:
            # Cash is an amount outside invested capital, so it has no weight: its
            # row states the two amounts and their gap in euros, banded at
            # ±allocation_band_rel_pctg of its target.
            c_now = float(cash.get("cash_actual_eur") or 0.0)
            c_tgt = float(cash.get("cash_target_eur") or 0.0)
            c_ok = abs(c_now - c_tgt) <= rel / 100.0 * c_tgt + 1e-9
            tail.append(("Cash", "", _eur_smart(c_now), "", _eur_smart(c_tgt), "",
                         _signed_eur(c_now - c_tgt), P["green"] if c_ok else P["red"]))
        levered = max(tn, tt) > 100.5
        html.append(_alloc_head("Asset class",
                                f"notional, % of invested capital \u00b7 "
                                f"{_eur_smart(invested_base)}", top=6))
        html.append(_bridge_svg(asset_rows, asset_rows, 270, cfg, tail=tail,
                                rule100=levered, total_now=tn, total_tgt=tt,
                                aria="Asset class allocation, today against the plan"))
        note = []
        if levered:
            note.append(f"Past the dashed 100% line is futures overlay: {tn / 100:.2f}\u00d7 "
                        f"today, {tt / 100:.2f}\u00d7 in the plan.")
        if levs:
            note.append(f"By class: {', '.join(levs)}; the rest 1.00\u00d7.")
        if cash is not None:
            note.append("Cash is outside invested capital.")
        if note:
            html.append(_alloc_note(" ".join(note)))

    # ── Equity geography ──
    geo_rows = [_bridge_row(
        key=r["name"], label=_GEO_SHORT.get(r["name"], geo_label(r["name"])),
        colour=_lift(r.get("color")), now=r.get("actual_pct_raw") or 0.0,
        target=r.get("target_left"), base=equity_base,
        trend=_timeline_vals(geo_series, r["name"])) for r in geo.get("rows") or []]
    if geo_rows:
        html.append(_alloc_head("Equity geography",
                                f"% of the equity sleeve \u00b7 {_eur_smart(equity_base)} "
                                f"notional", top=30))
        html.append(_bridge_svg(geo_rows, geo_rows, 200, cfg,
                                aria="Equity geography, today against the plan"))

    # ── Per-holding targets ──
    blocks = []
    verifs = {v.get("kind"): v for v in (m.rebalancing_verifications or [])}
    if getattr(cfg, "target_use_per_holding_only", False):
        df = getattr(m, "holdings_df", None)
        weights: dict[str, float] = {}
        if df is not None and not df.empty and invested_base > 0:
            for _, row in df.iterrows():
                isin = str(row.get("isin", "") or "").strip()
                if isin:
                    weights[isin] = (weights.get(isin, 0.0)
                                     + float(row.get("current_value", 0.0) or 0.0)
                                     / invested_base * 100.0)
        v = verifs.get("per_holding_portfolio")
        if v:
            blocks.append(("Per-holding target",
                           f"% of invested capital \u00b7 {_eur_smart(invested_base)}",
                           _holding_bridge_rows(ctx, v.get("items") or [], base=invested_base,
                                                weights=weights, series=hold_inv_series,
                                                series_key="isin")))
    else:
        for kind, title, base in (("per_holding_equity", "Equities holdings", equity_base),
                                  ("per_holding_fi", "Fixed income holdings", fi_base)):
            v = verifs.get(kind)
            if v and v.get("items"):
                blocks.append((title, f"% of the sleeve \u00b7 {_eur_smart(base)} notional",
                               _holding_bridge_rows(ctx, v.get("items") or [], base=base,
                                                    weights=None, series=hold_series,
                                                    series_key="ticker")))
    for title, note, (kept, new, sold) in blocks:
        n_left = len(kept) + len(sold)
        if not (kept or new or sold):
            continue
        html.append(_alloc_head(title, note, top=30))
        html.append(_bridge_svg(kept + sold, kept + new,
                                max(200.0, _SLOT * n_left + 12), cfg,
                                aria=f"{title}, today against the plan"))

    html.append(_alloc_note(
        f"1M: change in weight over the last month, in points. Gap: now minus plan, "
        f"in points; green inside the band, red outside. Band: the narrower of "
        f"\u00b1{abs_pp:g} pts and \u00b1{rel:g}% of the target.", top=14))
    return {"available": True, "html": "".join(html)}

def _build_holdings(ctx: _NewsletterContext) -> dict:
    """Build holdings grouped by asset class + role, rendered through the
    shared unified-table renderer (identical shell to Returns / Performance /
    Risk / Optimizer)."""
    m = ctx.metrics
    df = m.holdings_df
    if df.empty:
        return {"summary": [], "table_html": "", "total_count": 0}

    # Curated taxonomy for the shared role categorizer (role sub-grouping).
    from tarzan import config as _cfg
    _tax = _cfg.instrument_taxonomy()

    # Class totals and counts, keyed by the SAME normaliser as the row lookup
    # below and as the grouping engine that writes the headers. Keyed on the raw
    # column, a holding with no class was in neither dict — ``groupby`` drops
    # None/NaN outright — so its "% Class" divided its euro value by the ``.get``
    # default of 1, and the chips never counted it at all.
    class_keys = df["asset_class"].map(class_key)
    class_totals = df.groupby(class_keys)["current_value"].sum().to_dict()
    class_counts = df.groupby(class_keys).size().to_dict()

    summary = []
    # Canonical classes in their canonical order, then any residual class
    # appended — the same "present first, extras never dropped" helper the group
    # headers are ordered with. Iterating ASSET_CLASS_ORDER and skipping what was
    # not in it meant a rendered group could have no chip, and the chip counts did
    # not add up to the number of holdings.
    for klass in _ordered(list(class_counts), ASSET_CLASS_ORDER):
        summary.append({
            "name": klass,
            "color": ASSET_COLORS.get(klass, PALETTE["accent"]),
            "count": int(class_counts[klass]),
            "label": "positions" if class_counts[klass] != 1 else "position",
        })

    # One row item per holding; the shared engine groups them by class → role.
    invested_base = m.invested_value if m.invested_value > 0 else 0.0
    row_items = []
    for _, h in df.iterrows():
        klass = class_key(h.get("asset_class"))
        value = float(h["current_value"])
        # Same normaliser on both sides of the lookup, so the keys cannot
        # disagree, and a default of 0.0 — "no total", as phys_by_class.get does
        # at line 910 — never a money amount standing in for one.
        cls_total = class_totals.get(klass, 0.0)
        gain_pct = h.get("gain_pct")
        gain_eur = h.get("gain_eur")
        has_gain = gain_pct is not None and not pd.isna(gain_pct)
        # % of invested value; "—" for cash (undefined) or no invested base.
        if klass == "Cash & Cash Equivalents" or invested_base <= 0:
            weight_str = "—"
        else:
            weight_str = _pct(value / invested_base * 100, decimals=1)
        # A class whose holdings sum to exactly zero has no share to state — the
        # em-dash, the way the cash weight above does it, never a division by a
        # stand-in value. A NEGATIVE total still yields a real share (−300 of
        # −500 is 60%), so only zero is withheld.
        class_str = (_pct(value / cls_total * 100, decimals=1)
                     if cls_total != 0 else "—")
        gain_color = (PALETTE["green"] if (gain_pct or 0) >= 0
                      else PALETTE["red"]) if has_gain else PALETTE["muted"]
        row_items.append({
            "_ac": klass,
            "_isin": h.get("isin", ""),
            "_ticker": h.get("ticker", ""),
            "name_html": uni_name(
                display_instrument_name(h.get("isin"), h.get("ticker"),
                                        h.get("name", ""), 40),
                _display_ticker(h.get("ticker")) or "",
            ),
            "cells": [
                uni_cell(_eur(value, 2), width=78),
                uni_cell(weight_str, width=50),
                # The class share is the faintest figure in the row: the class
                # itself is named in the group header above, in its colour, so
                # repeating that colour on every cell said it a second time.
                uni_cell(class_str,
                         color=PALETTE["subtle"], width=52),
                uni_cell(_pct(gain_pct, signed=True) if has_gain else "—",
                         color=gain_color, weight=700, width=62),
                uni_cell(_eur_smart(gain_eur, signed=True)
                         if (gain_eur is not None and not pd.isna(gain_eur)) else "—",
                         color=gain_color, weight=700, width=58),
            ],
        })
    groups = group_by_class_role(
        row_items, asset_class=lambda r: r["_ac"],
        isin=lambda r: r["_isin"], ticker=lambda r: r["_ticker"], taxonomy=_tax)

    table_html = render_unified_table(
        "Holding",
        # The concept's widths, now that the content box is 580px: the value and
        # the two gains get the room, the two shares get only what a percentage
        # needs, and what is left goes to the name.
        [("Value \u20ac", "right", 78), ("% Inv.", "right", 50),
         ("% Class", "right", 52), ("Gain %", "right", 62),
         ("Gain \u20ac", "right", 58)],
        [(cls, col, [(role, [{"name_html": it["name_html"], "cells": it["cells"]}
                             for it in items])
                     for role, items in role_list])
         for cls, col, role_list in groups])

    # Class chips above the table: a swatch, the class name and how many
    # instruments are in it. Inline runs, not the boxed six-cell grid this used
    # to be -- the grid claimed as much height as three table rows to say what
    # fits on one line, and the swatch is what ties a class to its colour in the
    # group headers below.
    chips = " ".join(
        f'<span style="display:inline-block;margin:0 10px 6px 0;'
        f'{TYPE["prose"]}color:{PALETTE["muted"]};"><span style="display:inline-block;width:9px;'
        f'height:9px;border-radius:2px;background:{it["color"]};'
        f'vertical-align:middle;margin-right:5px;"></span>{_esc(str(it["name"]))} '
        f'<b style="color:{PALETTE["ink"]};">{it["count"]}</b></span>'
        for it in summary)
    subtitle = ""
    return {"summary": summary, "chips_html": chips,
            "table_html": table_html,
            "subtitle": subtitle, "total_count": int(len(df))}

def _optimizer_plan_ctx(m: PortfolioMetrics, suggestions: list, taxonomy=None) -> dict:
    """Build one optimizer plan's render context (actions + totals) from a
    list of rebalancing suggestions.

    Flat, largest trade first, like the concept. The other instrument tables
    group by asset class because the reader is asking a question about the
    shape of the portfolio; here the question is "what do I do, and does it
    matter", which the trade size answers and the class headers only
    interrupted -- six header rows above nine trades.
    """
    df = m.holdings_df
    taxonomy = taxonomy or {}
    total_buy = sum(float(s["amount_eur"]) for s in suggestions
                    if s["direction"].lower() == "buy")
    total_sell = sum(float(s["amount_eur"]) for s in suggestions
                     if s["direction"].lower() == "sell")

    # One cell showing a share as "% (bold) over € (muted)", the compact
    # Diversification-cell style, so four value columns fit at 600px.
    def _pct_eur_cell(pct, eur, *, color=None, weight=700):
        """A "% (bold) over € (muted)" value cell for the unified renderer."""
        return uni_cell(_pct(pct, decimals=1) if pct is not None else "—",
                        color=color or PALETTE["ink"], weight=weight,
                        sub=(_eur_smart(eur) if eur is not None else ""))

    def _pill(direction):
        c = PALETTE["green"] if direction == "BUY" else PALETTE["red"]
        return (f'<span style="display:inline-block;padding:1px 6px;'
                f'background:{PALETTE["card"]};'
                f'color:{c};border:1px solid {c}33;border-radius:999px;'
                f'font-weight:700;font-size:{TYPE_PX["label"]}px;'
                f'vertical-align:middle;'
                f'margin-right:5px;">{direction}</span>')

    actions = []
    for s in sorted(suggestions, key=lambda s: -float(s["amount_eur"])):
        direction = s["direction"].upper()
        amount = float(s["amount_eur"])
        ticker = s.get("ticker", "")
        isin = s.get("isin", "")
        signed_amount = amount if direction == "BUY" else -amount
        # Suggestions already carry the full ticker selected during
        # preprocessing; presentation must not re-resolve it from ISIN/cache.
        tk = _display_ticker(ticker) or ""
        # Whole-row tint by action: light green for BUY, light red for SELL,
        # so the proposed action reads at a glance across all columns.
        row_bg = PALETTE["green_tint"] if direction == "BUY" else PALETTE["red_tint"]
        dir_color = PALETTE["green"] if direction == "BUY" else PALETTE["red"]
        actions.append({
            "direction": direction,
            # No "asset_class" here: nothing read it, and the only way to fill it
            # was a default of "Equities" for a suggestion whose ticker is not in
            # the book — which is every BUY of something not yet held.
            "role": role_for(isin, ticker, taxonomy),
            "_row_bg": row_bg,
            # Name cell: action pill + ticker pin + shortened name, via the
            # shared uni_name so it matches every other table.
            "name_html": uni_name(
                display_instrument_name(isin, ticker, s.get("name", "")), tk,
                pill=_pill(direction)),
            # Trade -> Now -> After -> Target, each abs + %. The trade leads
            # because it is what the table is for, and After sits next to
            # Target so "does this trade get me there?" is a glance rather
            # than a comparison across two intervening columns.
            "cells": [
                uni_cell(_eur_smart(signed_amount, signed=True),
                         color=dir_color, weight=700),
                _pct_eur_cell(s.get("current_pct"), s.get("current_eur"),
                              color=PALETTE["muted"]),
                _pct_eur_cell(s.get("after_pct"), s.get("after_eur")),
                _pct_eur_cell(s.get("target_pct"), s.get("target_eur"),
                              color=PALETTE["muted"]),
            ],
        })

    # One flat block, still sorted largest trade first. Every column keeps its
    # percentage over its euro amount: the percentage says whether the trade
    # matters to the allocation, the euro says what to type into the broker.
    table_html = render_unified_table(
        "Action",
        [("Trade", "right", 70), ("Now", "right", 62),
         ("After", "right", 62), ("Target", "right", 62)],
        [(None, None, [(None, [{"name_html": a["name_html"],
                                "cells": a["cells"],
                                "row_bg": a["_row_bg"]} for a in actions])])],
        zebra=False)

    n_total = len(suggestions)
    n_buy = sum(1 for s in suggestions if s["direction"].lower() == "buy")
    return {
        "actions": actions,
        "table_html": table_html,
        "n_total": n_total,
        "n_buy": n_buy,
        "n_sell": n_total - n_buy,
        "total_buy": _eur_smart(total_buy),
        "total_sell": _eur_smart(total_sell),
    }

def _build_optimizer(ctx: _NewsletterContext) -> dict:
    """Build both rebalancing plans with their final funding proof.

    Reads ``metrics.rebalancing_plans`` (always computed by the engine); falls
    back to the single ``rebalancing_suggestions`` set for back-compat. Draft
    actions remain visible for diagnosis, but are explicitly marked as such
    whenever the serialized-action proof says they are not executable.
    """
    m = ctx.metrics
    from tarzan import config as _cfg
    _tax = _cfg.instrument_taxonomy()
    plans_src = getattr(m, "rebalancing_plans", None)

    def _attach_cost(pc: dict, p: dict) -> None:
        # Estimated execution cost, shown atop the table: CGT on the sells and
        # fixed commission fees (from the engine's plan_cost, same tax/fee model
        # the optimizer solved for). Only render when non-zero.
        cgt = float(p.get("cgt_eur") or 0.0)
        fees = float(p.get("fees_eur") or 0.0)
        pc["cgt_eur"] = _eur_smart(cgt) if cgt else None
        pc["fees_eur"] = _eur_smart(fees) if fees else None
        pc["cost_total_eur"] = _eur_smart(cgt + fees) if (cgt or fees) else None

    def _attach_execution(pc: dict, verifications) -> None:
        funding = _funding_verification(verifications)
        if funding is None:
            pc["execution_status"] = None
            pc["executable"] = None
            return

        status = str(funding.get("status") or "UNKNOWN").upper()
        residual = float(funding.get("residual_eur") or 0.0)
        pc.update({
            "execution_status": status,
            "executable": status == "EXECUTABLE",
            "funding_initial_cash_eur": _eur_smart(
                float(funding.get("initial_cash_eur") or 0.0)),
            "funding_external_contribution_eur": _eur_smart(
                float(funding.get("external_contribution_eur") or 0.0)),
            "funding_ending_cash_eur": _eur_smart(
                float(funding.get("ending_cash_eur") or 0.0)),
            "funding_protected_cash_eur": _eur_smart(
                float(funding.get("protected_cash_eur") or 0.0)),
            "funding_residual_eur": _eur_smart(residual, signed=True),
            "funding_shortfall_eur": (
                _eur_smart(abs(residual)) if residual < -0.005 else None
            ),
        })

    if plans_src:
        plans = []
        for p in plans_src:
            pc = _optimizer_plan_ctx(m, list(p.get("suggestions") or []), _tax)
            pc["label"] = p.get("label", "")
            pc["no_sell"] = p.get("no_sell")
            # Which plan the configuration actually runs. Both are computed
            # every run and both can be executable, so without this the reader
            # has two trade lists and no way to tell which one is the proposal.
            pc["active"] = bool(p.get("no_sell")) == bool(
                getattr(ctx.config, "rebalancing_no_sell", False))
            _attach_cost(pc, p)
            _attach_execution(pc, p.get("verifications"))
            plans.append(pc)
        if not any(pc["actions"] for pc in plans):
            return {"available": False}
        return {"available": True, "plans": plans,
                "subtitle": _optimizer_subtitle(plans)}

    # Back-compat: single plan.
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions:
        return {"available": False}
    pc = _optimizer_plan_ctx(m, suggestions, _tax)
    pc["label"] = "Suggested actions"
    pc["no_sell"] = None
    pc["active"] = True
    _attach_execution(pc, m.rebalancing_verifications)
    return {"available": True, "plans": [pc],
            "subtitle": _optimizer_subtitle([pc])}

def _optimizer_subtitle(plans: list[dict]) -> str:
    """One line saying how many plans there are and which one is live.

    A reader who sees two trade lists needs to know whether both are proposals
    before reading either. The wording is derived from the plans themselves, so
    it cannot claim a plan is executable when the funding proof says otherwise.
    """
    n = len(plans)
    if n == 1:
        one = plans[0]
        state = ("executable as it stands" if one.get("executable")
                 else "a draft until cash funding is resolved")
        return f"One plan, {state}."
    live = [p for p in plans if p.get("executable")]
    if len(live) == n:
        return (f"{n} plans, both executable as they stand. The first buys "
                f"only, because selling is switched off in the configuration.")
    if not live:
        return (f"{n} plans, neither executable until cash funding is "
                f"resolved. Do not treat either list as instructions.")
    return (f"{n} plans, {len(live)} executable as it stands. The first buys "
            f"only, because selling is switched off in the configuration.")


def _wf_label(row: dict) -> str:
    """Bar label for the waterfall: the resolved ticker when there is one.

    Exchange suffixes are kept -- ``_display_ticker`` treats them as part of
    the instrument's identity -- with the name as fallback, clipped to what
    fits under a 46px bar.
    """
    tick = _display_ticker(row.get("ticker"))
    if tick:
        return tick
    return str(row.get("name") or "")[:9]

def _build_preheader(ctx: _NewsletterContext, hero: dict) -> str:
    """Preview text shown in inbox preview."""
    m = ctx.metrics
    n_actions = len(m.rebalancing_suggestions or [])
    funding = _funding_verification(m.rebalancing_verifications)
    parts = [f"Portfolio at {hero['total_value']} ({hero['gain_pct']} since inception)"]
    if funding and funding.get("status") == "NON_EXECUTABLE":
        parts.append("rebalance draft not executable")
    elif n_actions > 0:
        parts.append("rebalancing suggested")
    parts.append(f"{len(m.holdings_df)} holdings tracked")
    return " · ".join(parts)

