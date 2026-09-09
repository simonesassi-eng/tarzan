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
    _NewsletterContext,
    _ordered,
    class_key,
    group_by_class_role,
    render_unified_table,
    role_for,
    ticker_span,
    uni_cell,
    uni_name,
    geo_label,
)
from tarzan.export.newsletter._format import (
    _display_ticker,
    _eur,
    _pct,
    _pct_smart,
    _semaphore,
    _semaphore_color,
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

    ``market_open`` is the engine's exchange-hours fact; ``1d_live`` is a
    different one — that the 1D figures are intraday rather than close-to-close.
    Reading the latter for this caption is what printed "market CLOSED" at 09:09
    with Milan and London both trading: minutes after an open the venue is open
    but no intraday bar exists yet. Falls back to ``1d_live`` only when the
    engine did not state it (no live transport, or an older projection)."""
    p = perf or {}
    open_now = p.get("market_open")
    return bool(p.get("1d_live")) if open_now is None else bool(open_now)


def _session_basis(perf: Optional[dict], m) -> str:
    """What the Session figure IS, in the caption's own words.

    ``1d_live`` false means the number is a completed session's close-to-close
    move. Captioning that "market open" — which it legitimately can be, minutes
    before the holdings' own venues open — invites reading a finished session as
    today's, which is how a Tuesday 08:58 digest showed Monday's +0.41% as the
    live session. Name the basis instead, the way the Markets strip prints
    "Cl. Mon" rather than a bare percentage."""
    p = perf or {}
    if bool(p.get("1d_live")):
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
    if coverage is None or not bool(p.get("1d_live")):
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
    if m.twror_pct is not None:
        status_bar.append(_bar("TWROR", _pct(m.twror_pct, signed=True),
                               _tone(m.twror_pct)))
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
    live = bool((perf or {}).get("1d_live"))
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
    twror_pct = m.twror_pct

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
        """A P&L tile with the PERCENTAGE as the headline and the euros beneath.

        These two led with the euro amount, which is the figure that grows with
        the book rather than the one that says how the book is doing: a euro P&L
        answers nothing without the capital behind it, while +8.62% is the answer
        and is comparable to every other percentage in the issue. The euro amount
        keeps its place, first on the caption line.

        The tone follows the headline, so the colour belongs to the number it is
        drawn on. If the percentage is unavailable the tile falls back to leading
        with the euros rather than headlining a "—".
        """
        if is_missing(pct):
            return _tile(label, _eur_smart(eur, signed=True), denominator,
                         _tone(eur))
        caption = denominator if is_missing(eur) else (
            f"{_eur_smart(eur, signed=True)} · {denominator}")
        return _tile(label, _pct(pct, signed=True), caption, _tone(pct))

    state_tiles = [
        _tile("Portfolio", _eur(m.total_value, decimals=0),
              f"invested {_eur_smart(m.invested_value)}"),
        _pnl_tile("Total P&L", total_pnl_eur, total_pnl_pct,
                  "on contributed capital"),
        _pnl_tile("Unrealized P&L", unrealized_eur, unrealized_pct,
                  "on open positions"),
    ]
    if twror_pct is not None:
        ann = m.twror_annualized_pct
        state_tiles.append(_tile(
            "TWROR", _pct(twror_pct, signed=True),
            "time-weighted"
            + (f" \u00b7 {_pct(ann, signed=True)} annualized"
               if ann is not None else ""),
            _tone(twror_pct)))
    if m.xirr_pct is not None:
        net = getattr(m, "xirr_net_tax_pct", None)
        state_tiles.append(_tile(
            "MWR", _pct(m.xirr_pct, signed=True),
            "money-weighted (XIRR)"
            + (f" \u00b7 {_pct(net, signed=True)} net of tax"
               if net is not None else ""),
            _tone(m.xirr_pct)))
    if cagr_pct is not None:
        state_tiles.append(_tile("CAGR", _pct(cagr_pct, signed=True),
                                 "compound annual growth", _tone(cagr_pct)))
    # The gap against the geography benchmark, and how it got there. An earlier
    # pass left this tile out on the grounds that the engine computes no such
    # delta; both cumulative series are computed and drawn side by side in the
    # since-inception chart, so the gap is the distance between two lines the
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


    # Dual-axis hero chart: 30-day portfolio value (€, left) + both P&L
    # measures as % (right, flow-adjusted via the daily cost-basis series),
    # with cash-flow triangles. Empty string when the order-derived series are
    # unavailable (holdings-only path).
    value_chart_html = ""
    hero_chart_legend = ""
    hero_flow_chips = ""
    win = _perf_window(m, 30)
    if (win and win.get("value") and len(win["value"]) >= 2
            and m.unrealized_series is not None and m.actual_value_series is not None):
        dts = win["dates"]
        idx = pd.DatetimeIndex(dts)
        av = _norm_series(m.actual_value_series).reindex(idx, method="ffill").bfill()

        def _cost_basis_pct(source):
            """A P&L series as % of its own cost basis (value − that P&L).

            Both lines use this one definition, so Total and Unrealized are
            directly comparable on the shared right axis — and it is the same
            definition the STATE tile captions state.
            """
            if source is None:
                return None
            s = _norm_series(source).reindex(idx, method="ffill").bfill()
            return list(((s / (av - s).replace(0, float("nan"))) * 100.0)
                        .bfill().values.astype(float))

        unreal_series = _cost_basis_pct(m.unrealized_series)
        total_series = _cost_basis_pct(m.pnl_series)
        value_chart_html = _hero_value_chart(
            win["value"], unreal_series, dts, win["flows"],
            total_pct=total_series,
        )
        if value_chart_html:
            hero_chart_legend = _hero_chart_legend(
                has_total=total_series is not None)
        hero_flow_chips = _hero_flow_chips(win["flows"])

    # This-week figures, mirroring the since-inception group:
    #   * Total PnL — the real money gained over the last five sessions, net
    #     of any contributions in them (delta of the cumulative P&L series);
    #   * TWROR — the 5-session time-weighted return (performance_full['5d']),
    #     the same series the Returns tables use.
    perf_full = m.performance_full or {}
    week_twror = perf_full.get("5d")
    try:
        week_twror = float(week_twror) if week_twror is not None else None
        if week_twror != week_twror:  # NaN
            week_twror = None
    except (TypeError, ValueError):
        week_twror = None
    week_pnl_eur, week_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, "5d")
    # Last-30-days money P&L (net of contributions) for the scoreboard's
    # "Last 30 days" row, mirroring the chart window.
    month_pnl_eur, month_pnl_pct = _window_money_pnl(m.pnl_series, m.actual_value_series, "1m")
    month_twror = perf_full.get("1m")
    try:
        month_twror = float(month_twror) if month_twror is not None else None
        if month_twror != month_twror:  # NaN
            month_twror = None
    except (TypeError, ValueError):
        month_twror = None

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
        "twror_pct": _pct(twror_pct, signed=True) if twror_pct is not None else None,
        "twror_is_positive": (twror_pct or 0.0) >= 0,
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
        # 1-week TWROR — both clearly labeled in the template.
        "week_pnl_eur": _eur_smart(week_pnl_eur, signed=True) if week_pnl_eur is not None else None,
        "week_pnl_pct": _pct(week_pnl_pct, signed=True) if week_pnl_pct is not None else None,
        "week_pnl_is_positive": (week_pnl_eur or 0.0) >= 0,
        "week_twror_pct": _pct(week_twror, signed=True) if week_twror is not None else None,
        "week_twror_is_positive": (week_twror or 0.0) >= 0,
        # Last 30 days (mirrors the chart window): money P&L + TWROR.
        "month_pnl_eur": _eur_smart(month_pnl_eur, signed=True) if month_pnl_eur is not None else None,
        "month_pnl_pct": _pct(month_pnl_pct, signed=True) if month_pnl_pct is not None else None,
        "month_pnl_is_positive": (month_pnl_eur or 0.0) >= 0,
        "month_twror_pct": _pct(month_twror, signed=True) if month_twror is not None else None,
        "month_twror_is_positive": (month_twror or 0.0) >= 0,
        # Annualized returns (card subtitles): money-weighted XIRR vs
        # time-weighted TWROR-annualized.
        "xirr_pct": _pct(m.xirr_pct, signed=True) if m.xirr_pct is not None else None,
        "twror_annualized_pct": _pct(m.twror_annualized_pct, signed=True) if m.twror_annualized_pct is not None else None,
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
        f'transferred-in positions. TWROR is gross of tax.</div>'
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
    tol = cfg.rebalancing_target_tolerance_pctg

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
        sema = _semaphore(delta, tol)
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
            "delta_color": _semaphore_color(sema),
            "bar_width": min(max(actual, 1), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
            "leverage": leverage,
        })

    # Cash buffer (EUR-based, appended after invested classes).
    # The bar width is scaled as % of total portfolio so cash visually
    # matches the other rows (it would otherwise dominate the bar
    # because target_cash_buffer_eur is small relative to invested
    # value). Status color is still driven by the relative deviation
    # vs the cash target via _semaphore.
    if cfg.target_cash_buffer_eur > 0:
        cash_actual = m.cash_value
        cash_tgt = cfg.target_cash_buffer_eur
        rel_dev = (cash_actual - cash_tgt) / cash_tgt * 100 if cash_tgt > 0 else 0
        sema = _semaphore(rel_dev, tol)
        delta_eur = cash_actual - cash_tgt
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
            "delta_color": _semaphore_color(sema),
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
        "tolerance": _pct(tol, decimals=1).rstrip("%") + "%",
        "has_timeline": any(r.get("spark") for r in rows),
    }

def _build_geography(ctx: _NewsletterContext) -> dict:
    """Build geographic equity rows with target & ACWI ticks."""
    m = ctx.metrics
    cfg = ctx.config
    tol = cfg.rebalancing_target_tolerance_pctg

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
        sema = _semaphore(delta_target, tol)
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
            "delta_color": _semaphore_color(sema),
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

def _div_label(name: str, color: Optional[str] = None,
               ticker: Optional[str] = None) -> str:
    """Row label for the diversification tables: an optional colour swatch, an
    optional ticker pin, then the name.

    The swatch is only drawn when it keys something: in the asset-class and
    geography tables it is the colour of that row's trend line. In the
    per-holding tables every row was the same accent blue, so nine identical
    squares keyed nothing and took the width the name needed.
    """
    P = PALETTE
    sw = (f'<span style="display:inline-block;width:9px;height:9px;'
          f'border-radius:2px;background:{color};vertical-align:middle;'
          f'margin-right:6px;"></span>') if color else ""
    return (f'{sw}{ticker_span(ticker or "")}'
            f'<span style="color:{P["ink"]};">{_esc(str(name))}</span>')

def _div_table(rows: list[dict], tol: float, base: Optional[float] = None,
               show_leverage: bool = False, first_label: str = "Name",
               subs: bool = True, value_subs: Optional[bool] = None) -> str:
    """The allocation table: one slice per ROW, one line per row.

    Every column the section has always carried — the weight now, the target, the two
    against each other on a shared axis, the month's trend and the drift — in about
    22px instead of 50. The height came from two habits, not from the column count:
    the percentage stacked over its euro amount, and a 40px sparkline. On one line and
    at 15px the three tables together take roughly half the page they did.

    The track is a bullet graph: a pale band for the ±tolerance around the target, the
    weight as a bar over it, the target as a tick. A bar inside its band needs nothing
    doing, which the reader sees without reading a figure.

    ``show_leverage`` puts each class's notional-per-euro factor INSIDE the Now and
    Target cells rather than in a column of its own — actual beside the actual weight,
    the plan's beside the plan's, which is where each belongs. The target factor is
    ``None`` for a class the plan holds no physical capital in ("synth"): fixed income
    here is entirely an efficient core's bond overlay, and printing a ratio over a zero
    denominator would invent one.

    ``subs`` and ``value_subs`` are kept for call-site compatibility and no longer gate
    a second line, because there is no second line: ``base`` alone decides whether a
    euro amount appears beside a percentage.
    """
    if not rows:
        return ""
    P, FS = PALETTE, TYPE_PX["data"]
    GUT = 8

    # ONE width table for every block, so the three tables' columns line up down the
    # section. Two of them (one for the leverage variant, one without) put Now, Target
    # and Trend at different x in each block, and three tables whose columns do not
    # agree read as three unrelated things.
    #
    # Sized on the widest content each column must hold, at ~6px a character in the
    # 580px content box, minus its 8px gutter:
    #   name    131px  "NTSG WT Gl. Eff. Core" = 21 chars
    #   track    50px  a bar and a tick need no more
    #   now     120px  "114.7% EUR120.0k 1.15x" = 20 chars
    #   target  108px  "125.5% EUR140k 1.25x"   = 18 chars, and its gutter is 14 so the
    #                  figures do not touch the sparkline that follows
    #   trend    73px  27px of sparkline + 4 + "-10.8pp"
    #   drift    44px  "-10.8pp"
    #   drift  48px  "-10.8pp" is 42 and its box was exactly 42, so the first rounding
    #                 pushed the figure outside the table -- fixed layout clips nothing
    #                 unless told to, and clipping a number is worse than making room.
    W = {"name": 20, "track": 10, "now": 23, "target": 22, "trend": 14,
         "drift": 11}
    #: Inset from the table's own border, on the two columns that touch it. The header
    #: band and the row tints are backgrounds now, so a label flush against the border
    #: reads as a rendering slip rather than a column.
    EDGE = 12
    #: The gutter after Target, wider than the rest: to its right sits a graphic, and
    #: 8px between a figure and a sparkline reads as no gap at all.
    GUT_TARGET = 14
    #: Trend is the one left-aligned column between two right-aligned ones, so it needs
    #: its own leading inset or its sparkline starts where Target's figures end.
    TREND_LEAD = 8

    # The axis every track shares, scaled on the SLICES only. The total row draws no
    # bar -- it is a sum, not a slice -- but its 114.7% was in this maximum, so every
    # real class was squashed into the left eighth of its track and Gold, Commodities
    # and Alternative read as slivers.
    _scaled = [r for r in rows if not r.get("is_total") and not r.get("eur_row")]
    top = max([float(r.get("now") or 0.0) for r in _scaled]
              + [float(r.get("target") or 0.0) for r in _scaled] + [1.0]) * 1.08

    def _fig(pct: Optional[float], lev=None, *, bold: bool, dp: int) -> str:
        if pct is None:
            return f'<span style="color:{P["subtle"]};">\u2014</span>'
        colour = P["ink"] if bold else P["muted"]
        weight = "700" if bold else "400"
        out = (f'<span style="color:{colour};font-weight:{weight};">'
               f'{_pct_smart(float(pct))}</span>')
        if base:
            out += (f'<span style="color:{P["subtle"]};"> '
                    f'{_eur_smart(float(pct) / 100.0 * float(base))}</span>')
        if show_leverage and lev is not _NO_LEV:
            out += (f'<span style="color:{P["subtle"]};"> '
                    f'{"synth" if lev is None else f"{float(lev):.2f}\u00d7"}</span>')
        return out

    def _track(now: float, target: Optional[float], colour: str) -> str:
        w_now = max(1, int(round(now / top * 100)))
        if target is None:
            band = '<td style="font-size:0;line-height:0;">&nbsp;</td>'
            tick = ""
        else:
            lo = max(0, min(100, int(round((target - tol) / top * 100))))
            hi = max(0, min(100, int(round((target + tol) / top * 100))))
            x = max(0, min(100, int(round(target / top * 100))))
            band = (f'<td width="{lo}%" style="font-size:0;line-height:0;">&nbsp;</td>'
                    f'<td width="{max(1, hi - lo)}%" style="background:'
                    f'{_tint(P["accent"], P["card"], 0.20)};font-size:0;'
                    f'line-height:0;">&nbsp;</td>'
                    f'<td style="font-size:0;line-height:0;">&nbsp;</td>')
            tick = (f'<table role="presentation" width="100%" cellpadding="0" '
                    f'cellspacing="0" border="0"><tr>'
                    f'<td width="{x}%" style="font-size:0;line-height:0;">&nbsp;</td>'
                    f'<td width="1" style="height:13px;background:{P["ink"]};'
                    f'font-size:0;line-height:0;">&nbsp;</td>'
                    f'<td style="font-size:0;line-height:0;">&nbsp;</td>'
                    f'</tr></table>')
        return (
            f'<div><table role="presentation" width="100%" cellpadding="0" '
            f'cellspacing="0" border="0" style="height:9px;background:'
            f'{P["group_bg"]};border-radius:2px;"><tr style="height:9px;">'
            f'{band}</tr></table>'
            f'<div style="margin-top:-7px;"><table role="presentation" width="100%" '
            f'cellpadding="0" cellspacing="0" border="0"><tr>'
            f'<td width="{w_now}%" style="height:5px;background:{colour};'
            f'border-radius:2px;font-size:0;line-height:0;">&nbsp;</td>'
            f'<td style="font-size:0;line-height:0;">&nbsp;</td></tr></table></div>'
            f'<div style="margin-top:-9px;">{tick}</div></div>')

    def _th(key: str, label: str, align: str = "left") -> str:
        gut = GUT_TARGET if key == "target" else GUT
        left = EDGE if key == "name" else (TREND_LEAD if key == "trend" else 0)
        right = EDGE if key == "drift" else gut
        return (f'<td width="{W[key]}%" align="{align}" style="{TYPE["label"]}'
                f'color:{P["muted"]};padding:5px {right}px 5px {left}px;'
                f'white-space:nowrap;">{label}</td>')

    # The shell PORTFOLIO MOVERS uses, so the two sections read as one table language:
    # a rounded border, a header band on ``head_bg``, zebra rows.
    out = [f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
           f'border="0" style="width:100%;table-layout:fixed;margin-top:10px;'
           f'border:1px solid {P["border"]};border-radius:8px;'
           f'border-collapse:separate;border-spacing:0;overflow:hidden;">',
           f'<tr style="background:{P["head_bg"]};">'
           + _th("name", _esc(first_label)) + _th("track", "Vs target")
           + _th("now", "Now", "right") + _th("target", "Target", "right")
           + _th("trend", "Trend") + _th("drift", "Drift", "right") + '</tr>']

    for i, r in enumerate(rows):
        zebra = P["zebra"] if i % 2 else P["card"]
        pad = f'background:{zebra};padding:3px {GUT}px 3px 0;'
        if r.get("eur_row"):
            # Cash: an amount, not a share of the invested base, so no track and no
            # percentage — the row states the two figures and their gap.
            out.append(
                f'<tr><td style="{TYPE["data"]}color:{P["ink"]};{pad}'
                f'white-space:nowrap;overflow:hidden;border-top:1px solid '
                f'{P["row_rule"]};">{r.get("label_html", "")}</td>'
                f'<td style="{pad}border-top:1px solid {P["row_rule"]};">&nbsp;</td>'
                f'<td align="right" style="{TYPE["data"]}color:{P["ink"]};{pad}'
                f'font-variant-numeric:tabular-nums;white-space:nowrap;'
                f'border-top:1px solid {P["row_rule"]};">'
                f'{_eur_smart(float(r.get("now_eur") or 0.0))}</td>'
                f'<td align="right" style="{TYPE["data"]}color:{P["muted"]};{pad}'
                f'font-variant-numeric:tabular-nums;white-space:nowrap;'
                f'border-top:1px solid {P["row_rule"]};">'
                f'{_eur_smart(float(r.get("target_eur") or 0.0))}</td>'
                f'<td style="{pad}border-top:1px solid {P["row_rule"]};">&nbsp;</td>'
                f'<td align="right" style="{TYPE["data"]}'
                f'color:{r.get("delta_color") or P["muted"]};font-weight:700;'
                f'background:{zebra};padding:3px {EDGE}px 3px 0;'
                f'font-variant-numeric:tabular-nums;white-space:nowrap;'
                f'border-top:1px solid {P["row_rule"]};">'
                f'{_signed_eur(r.get("delta_eur"))}</td></tr>')
            continue

        now = float(r.get("now") or 0.0)
        target = r.get("target")
        target = None if target is None else float(target)
        drift = None if target is None else now - target
        dcol = _semaphore_color(_semaphore(drift, tol)) if drift is not None else P["muted"]
        colour = r.get("color") or P["accent"]
        is_total = bool(r.get("is_total"))
        rule = (f'border-top:1px solid {P["border"]};' if is_total
                else f'border-top:1px solid {P["row_rule"]};')
        name_colour = P["accent"] if is_total else P["ink"]
        lev = r.get("leverage", _NO_LEV) if show_leverage else _NO_LEV
        tlev = r.get("target_leverage", _NO_LEV) if show_leverage else _NO_LEV

        out.append(
            f'<tr><td style="{TYPE["data"]}color:{name_colour};'
            f'{"font-weight:700;" if is_total else ""}{pad}{rule}'
            f'white-space:nowrap;overflow:hidden;">{r.get("label_html", "")}</td>'
            f'<td style="{pad}{rule}">'
            f'{"&nbsp;" if is_total else _track(now, target, dcol)}</td>'
            f'<td align="right" style="{TYPE["data"]}{pad}{rule}'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">'
            f'{_fig(now, lev, bold=True, dp=1)}</td>'
            f'<td align="right" style="{TYPE["data"]}{pad}{rule}'
            f'font-variant-numeric:tabular-nums;white-space:nowrap;">'
            f'{_fig(target, tlev, bold=False, dp=0)}</td>'
            f'<td style="{pad}{rule}white-space:nowrap;overflow:hidden;">'
            f'{_trend_cell(r.get("spark_vals"), target, colour)}</td>'
            f'<td align="right" style="{TYPE["data"]}color:{dcol};font-weight:700;'
            f'padding:3px 0;{rule}font-variant-numeric:tabular-nums;'
            f'white-space:nowrap;">'
            f'{(_signed_pp(drift) + "pp") if drift is not None else "\u2014"}</td>'
            f'</tr>')

    out.append("</table>")
    return "".join(out)


def _signed_eur(value) -> str:
    """A signed euro gap: "+EUR2.5k". Cash is held as an amount, not a share, so its
    row's drift cannot be points."""
    if value is None:
        return ""
    amount = float(value)
    return f'{"+" if amount >= 0 else "\u2212"}{_eur_smart(abs(amount))}'


#: Sentinel: a row that carries no leverage key at all, as against one that carries
#: ``None`` to mean "synthetic". The two must render differently -- nothing versus the
#: word -- and ``None`` cannot express both.
_NO_LEV = object()


def _tint(fg: str, bg: str, alpha: float) -> str:
    """``fg`` over ``bg`` at ``alpha``, as a flat hex.

    Resolved here because an emailed ``rgba()`` is unreliable: several clients drop the
    declaration entirely and the band would vanish.
    """
    a = max(0.0, min(1.0, alpha))
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(f[i] * a + b[i] * (1 - a))):02x}" for i in range(3))


def _trend_cell(vals, target, colour) -> str:
    """A 15px sparkline of the last month against its target, plus the move in points.

    15px rather than the 40 this used to draw. The shape is what the column is for and
    it survives the smaller box; the 25px per row it returns is half of what made the
    section a thousand pixels tall.
    """
    series = [float(v) for v in (vals or []) if v is not None]
    if len(series) < 2:
        return f'<span style="{TYPE["prose"]}color:{PALETTE["subtle"]};">no series</span>'
    P = PALETTE
    w, h = 27, 15
    pool = series + ([float(target)] if target is not None else [])
    lo, hi = min(pool), max(pool)
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0

    def y(v: float) -> float:
        return h - 1.5 - (v - lo) / (hi - lo) * (h - 3)

    step = (w - 2) / (len(series) - 1)
    pts = " ".join(f"{1 + i * step:.1f},{y(v):.1f}" for i, v in enumerate(series))
    dash = ("" if target is None else
            f'<line x1="1" y1="{y(float(target)):.1f}" x2="{w - 1}" '
            f'y2="{y(float(target)):.1f}" stroke="{P["subtle"]}" stroke-width="0.7" '
            f'stroke-dasharray="2,2" stroke-opacity="0.8"/>')
    move = series[-1] - series[0]
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:inline-block;'
            f'vertical-align:middle;">{dash}'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.3" stroke-linejoin="round"/>'
            f'<circle cx="{1 + (len(series) - 1) * step:.1f}" '
            f'cy="{y(series[-1]):.1f}" r="1.6" fill="{colour}"/></svg>'
            f'<span style="{TYPE["prose"]}color:{P["subtle"]};padding-left:4px;">'
            f'{_signed_pp(move)}</span>')

def _ph_target_rows(ctx: _NewsletterContext, tol: float,
                    hold_inv_series: Optional[list]) -> tuple[list[dict], str]:
    """Rows (for :func:`_div_table`) for per-holding portfolio targets in
    ``target_use_per_holding_only`` mode: each targeted instrument's CURRENT
    weight (% of invested, straight from the rebalancer verification — so a
    not-yet-held target reads 0%), its target, and a 1-month weight trend on
    the SAME % of invested basis. Also returns a short "to exit" note for
    0%-target holdings still held. ``(rows, note)``.

    ``hold_inv_series`` is the timeline's ``holding_invested`` series (keyed by
    ISIN). Verification items carry a ticker/name, so we resolve each to its
    ISIN via the snapshot to look up the right trend line.
    """
    P = PALETTE
    m = ctx.metrics
    df = getattr(m, "holdings_df", None)

    # Resolver: ticker (bare + full) and name → ISIN, so a verification item
    # (which carries h.ticker like "NTSG.DE" and the name) can find its
    # ISIN-keyed trend line.
    invested = float(getattr(m, "invested_value", 0.0) or 0.0)
    if invested <= 0:
        invested = float(getattr(m, "total_value", 0.0) or 0.0)
    isin_of: dict[str, str] = {}
    cur_by_isin: dict[str, float] = {}  # CURRENT weight (% of invested) by ISIN
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            _isin = str(row.get("isin", "") or "").strip()
            if not _isin:
                continue
            for key in (row.get("isin"), row.get("ticker"), row.get("name")):
                if key and str(key).strip():
                    isin_of[str(key).strip().upper()] = _isin
            # Also index the exchange-stripped ticker (NTSG.DE → NTSG).
            tkr = str(row.get("ticker", "") or "").strip()
            if tkr:
                isin_of[tkr.upper().split(".")[0]] = _isin
            val = float(row.get("current_value", 0.0) or 0.0)
            cur_by_isin[_isin] = (val / invested * 100.0) if invested > 0 else 0.0

    def _isin_for(it) -> str:
        for key in (it.get("ticker"), it.get("category")):
            if not key:
                continue
            k = str(key).strip().upper()
            if k in isin_of:
                return isin_of[k]
            if k.split(".")[0] in isin_of:
                return isin_of[k.split(".")[0]]
        return ""

    def _trend_for(it) -> Optional[list]:
        if not hold_inv_series:
            return None
        isin = _isin_for(it)
        if not isin:
            return None
        xs = [float(pt.get(isin, 0.0)) for pt in hold_inv_series]
        if any(x > 0 for x in xs) and len(xs) >= 2:
            return xs
        return None

    items = []
    for v in (m.rebalancing_verifications or []):
        if v.get("kind") == "per_holding_portfolio":
            items = v.get("items", []) or []
            break

    targeted, exits = [], []
    for it in items:
        tgt = float(it.get("target_pct", 0.0) or 0.0)
        (targeted if tgt > 0 else exits).append(it)
    targeted.sort(key=lambda it: -float(it.get("target_pct", 0.0) or 0.0))

    rows = []
    for it in targeted:
        key = it.get("ticker") or it.get("category") or ""
        tk = _display_ticker(key) or ""
        # "Now" = the CURRENT weight (% of invested) from the snapshot, so it
        # matches the By-asset-class table exactly. The verification's
        # actual_pct is POST-trade (it reflects the plan's buys), which would
        # not equal the current weight — a not-yet-held target reads 0%.
        now = cur_by_isin.get(_isin_for(it), 0.0)
        rows.append({
            # No swatch: every per-holding row would carry the same accent
            # square, keying nothing and taking the width the name needs.
            "label_html": _div_label(
                # 15, not 42. The name column holds 21 characters and the ticker takes
                # the first five of them; at 42 the name ran straight into the track
                # beside it ("AVWS Avant. Gl. Sm Cap Value" overlapping its own bar).
                display_instrument_name(_isin_for(it), key,
                                        it.get("category") or key, 10),
                ticker=tk),
            "now": now,
            "target": float(it.get("target_pct", 0.0) or 0.0),
            "spark_vals": _trend_for(it),
            "color": P["accent"],
        })

    note = ""
    held_exits = [it for it in exits if cur_by_isin.get(_isin_for(it), 0.0) > 0.05]
    if held_exits:
        # How much money the instruction actually moves, not just which tickers
        # it names: a list of seven symbols does not say whether this is a
        # rounding trim or half the book.
        #
        # cur_by_isin holds each position's WEIGHT as a percent of invested
        # capital, not its euro value (see where it is built above), so the
        # share is the sum of those weights and the amount is derived from it.
        # Summing it as euros produced "7 positions worth €54".
        share = sum(cur_by_isin.get(_isin_for(it), 0.0) for it in held_exits)
        invested = float(getattr(ctx.metrics, "invested_value", 0.0) or 0.0)
        exit_eur = invested * share / 100.0 if invested > 0 else None
        pills = " ".join(
            f'<span style="display:inline-block;margin:0 4px 4px 0;'
            f'padding:1px 6px;border-radius:5px;background:{P["red_bg"]};'
            f'color:{P["red"]};font-size:{TYPE_PX["label"]}px;font-weight:700;'
            f'white-space:nowrap;">'
            f'{(_display_ticker(it.get("ticker") or "") or short_instrument_name(it.get("category") or "", 16))}'
            f'</span>'
            for it in held_exits)
        note = (
            f'<div style="margin-top:8px;{TYPE["prose"]}color:{P["muted"]};">'
            f'<b style="color:{P["red"]};">Targeted to 0%:</b> '
            f'{len(held_exits)} position{"s" if len(held_exits) != 1 else ""} '
            + (f'worth <b style="color:{P["ink"]};">{_eur_smart(exit_eur)}</b>, '
               if exit_eur is not None else "")
            + f'{share:.1f}% of invested capital'
            + f'.<div style="margin-top:5px;">{pills}</div></div>')
    return rows, note

def _holding_verif_rows(ctx: _NewsletterContext, tol: float,
                        hold_series: Optional[list], kind: str,
                        color: str) -> list[dict]:
    """Rows (for :func:`_div_table`) from an equity/FI per-holding verification
    (weight is % of the sleeve), used when NOT in per-holding-only mode."""
    rows = []
    for v in (ctx.metrics.rebalancing_verifications or []):
        if v.get("kind") != kind:
            continue
        # Sort by weight desc, with ticker as a STABLE tie-break: two holdings
        # at the same weight would otherwise keep their input order, which
        # traces back to a set() of open ISINs (hash-randomized per process) —
        # making the rendered order vary run-to-run and breaking reproducibility.
        items = sorted(v.get("items", []) or [],
                       key=lambda it: (-float(it.get("actual_pct", 0.0)),
                                       str(it.get("ticker", "")),
                                       str(it.get("category", ""))))
        for it in items:
            ticker = it.get("ticker", "")
            tk = _display_ticker(ticker) or ""
            vals = None
            if hold_series:
                xs = [float(pt.get(ticker, 0.0)) for pt in hold_series]
                if any(x > 0 for x in xs) and len(xs) >= 2:
                    vals = xs
            rows.append({
                "label_html": _div_label(
                    display_instrument_name(it.get("isin"), ticker,
                                            it.get("category") or ticker, 42),
                    ticker=tk),
                "now": float(it.get("actual_pct", 0.0)),
                "target": float(it.get("target_pct", 0.0)),
                "spark_vals": vals,
                "color": color,
            })
    return rows

def _build_diversification(ctx: _NewsletterContext) -> dict:
    """Pre-render the Diversification section (tile dashboard) as HTML.

    Reuses :func:`_build_allocation` / :func:`_build_geography` for the
    numbers and semaphore logic, the rebalancer's per-holding checks for the
    by-holding tiles, ``short_instrument_name`` for labels, the price-cache
    ISIN→symbol map for clean tickers, and :func:`_spark` for the trends — so
    this is a presentational layer, not a second source of truth.
    """
    P = PALETTE
    alloc = _build_allocation(ctx)
    geo = _build_geography(ctx)
    tol = ctx.config.rebalancing_target_tolerance_pctg

    # EUR bases (value of 100%) for the inline absolute amounts: invested
    # value for asset-class/per-holding-portfolio rows, and the equity/FI
    # sleeve totals for geography/per-holding-equity/per-holding-FI rows —
    # matching exactly what each row's % is already a share of.
    m = ctx.metrics
    invested_base = float(getattr(m, "invested_value", 0.0) or 0.0)
    if invested_base <= 0:
        invested_base = float(getattr(m, "total_value", 0.0) or 0.0)
    # Geography and per-sleeve rows are shares of the NOTIONAL sleeve
    # (``_compute_geo_allocation`` distributes each holding's notional equity
    # exposure), so their euro base must be that same notional sleeve — the
    # class weight × invested capital, exactly as the asset-class table's own
    # leverage math uses. Multiplying a notional share by the physical market
    # value instead (Σ current_value) mixes two different denominators: it made
    # Emerging Markets read fewer euros than its sole holding, XMME, was
    # worth on its own.
    byclass = getattr(m, "allocation_by_class", None)

    def _notional_sleeve_eur(klass: str) -> float:
        if byclass is None or byclass.empty:
            return 0.0
        row = byclass[byclass["category"] == klass]
        if row.empty:
            return 0.0
        return float(row["weight_pct"].iloc[0]) / 100.0 * invested_base

    equity_base = _notional_sleeve_eur("Equities")

    tl = ctx.metrics.allocation_timeline or {}
    dates = tl.get("dates") or []
    asset_series = _recent_timeline(tl.get("asset"), dates)
    geo_series = _recent_timeline(tl.get("geo"), dates)
    hold_series = _recent_timeline(tl.get("holding"), dates)
    hold_inv_series = _recent_timeline(tl.get("holding_invested"), dates)

    available = bool(alloc.get("rows") or geo.get("rows"))
    if not available:
        return {"available": False, "html": ""}

    def swatch(color, sz=10):
        return (f'<span style="display:inline-block;width:{sz}px;height:{sz}px;'
                f'border-radius:2px;background:{color};vertical-align:middle;"></span>')

    # The plan's own leverage per class, for the Target cell beside each class's
    # target weight. Only meaningful because the class targets are DERIVED from the
    # per-instrument plan: while they came from a separate file the ratio mixed two
    # sources and Alternative read 11/15 = 0.73x, which is not a leverage but the
    # distance between two lists that disagreed. From one plan it reads 15/15 = 1.00x.
    _target_lev: dict = {}
    try:
        from tarzan.engine.target_derivation import (
            derive_target_leverage, plan_weights)

        _plan, _ = plan_weights(getattr(ctx.metrics, "target_rows", None) or {})
        if not _plan:
            # The metrics object does not carry the raw rows; rebuild the plan from the
            # weights the engine already deduplicated for its own target line.
            _plan = dict(getattr(ctx.metrics, "target_weights", {}) or {})
        if _plan:
            _target_lev = derive_target_leverage(_plan)
    except Exception:  # noqa: BLE001 — a missing factor must not cost the section
        _target_lev = {}

    # ── Asset-class rows (cash folded in as a normal, EUR-native row that
    #    does NOT participate in the invested base) ──
    asset_rows = []
    for r in alloc["rows"]:
        if r.get("is_eur"):
            asset_rows.append({
                "label_html": _div_label(r["name"], r["color"]),
                "eur_row": True,
                "now_eur": r.get("cash_actual_eur", 0.0),
                "target_eur": r.get("cash_target_eur", 0.0),
                "delta_eur": r.get("cash_delta_eur", 0.0),
                "delta_color": r.get("delta_color", P["muted"]),
            })
            continue
        asset_rows.append({
            "target_leverage": _target_lev.get(r["name"]),
            "label_html": _div_label(r["name"], r["color"]),
            "now": r.get("actual_pct_raw"),
            "target": r.get("target_left"),
            "spark_vals": _timeline_vals(asset_series, r["name"]),
            "color": r["color"],
            "leverage": r.get("leverage"),
        })

    # ── "Invested Portfolio" summary row (notional totals + leverage),
    # placed BELOW the invested classes and ABOVE cash — cash is a separate
    # accounting entity that does not participate in the invested base. ──
    _cls_rows = [r for r in asset_rows if not r.get("eur_row")]
    _cash_rows = [r for r in asset_rows if r.get("eur_row")]
    if _cls_rows:
        _tnow = sum(float(r.get("now") or 0.0) for r in _cls_rows)
        _ttgt = sum(float(r.get("target") or 0.0) for r in _cls_rows)
        _ttrend = None
        if asset_series and len(asset_series) >= 2:
            _ttrend = [sum(float(x) for x in b.values()) for b in asset_series]
        total_row = {
            "is_total": True,
            "label_html": "\u2605 Total notional",
            "now": _tnow,
            "target": _ttgt,
            "leverage": (_tnow / 100.0) if _tnow else None,
            # The plan's own leverage: its notional over 100% of capital, the same
            # ratio the actual side of this row states.
            "target_leverage": (_ttgt / 100.0) if _ttgt else None,
            "spark_vals": _ttrend,
            "color": P["accent"],
        }
        # Reorder: classes → Invested Portfolio total → cash.
        asset_rows = _cls_rows + [total_row] + _cash_rows

    # ── Geography rows ──
    # geo_label shortens the display form only; the long name stays the
    # configuration key it is, and _timeline_vals still looks up by that key.
    geo_rows = [{
        "label_html": _div_label(geo_label(r["name"]), r["color"]),
        "now": r.get("actual_pct_raw"),
        "target": r.get("target_left"),
        "spark_vals": _timeline_vals(geo_series, r["name"]),
        "color": r["color"],
    } for r in geo["rows"]]

    # ── By-holding rows: per-holding-only → portfolio targets (current
    # weight); otherwise the equity/FI sleeve tables. ──
    per_holding_only = getattr(ctx.config, "target_use_per_holding_only", False)
    holding_rows, exits_note, eq_rows, fi_rows = [], "", [], []
    if per_holding_only:
        holding_rows, exits_note = _ph_target_rows(ctx, tol, hold_inv_series)
    else:
        eq_rows = _holding_verif_rows(ctx, tol, hold_series, "per_holding_equity",
                                      ASSET_COLORS.get("Equities", P["accent"]))
        fi_rows = _holding_verif_rows(ctx, tol, hold_series, "per_holding_fi",
                                      ASSET_COLORS.get("Fixed Income", P["accent"]))

    # The section kicker is the template's job: baked in here it bypassed the
    # ordinal counter, so this section printed an unnumbered header among
    # numbered ones.
    html: list[str] = []
    if asset_rows:
        html.append(_div_table(asset_rows, tol, base=invested_base,
                               show_leverage=True, first_label="Asset class"))
        # What the table's own marks mean, in one line: the band, the tick, the
        # 100% rule and which way a trend colour reads. The sum past 100% needs
        # no separate sentence -- the total row states it and the x factors in
        # the drift column say where it comes from.
        # No caption. The marks it described -- the tolerance band, the target tick,
        # the leverage factor, "synth" -- are now in a table that reads like the rest of
        # the issue, and six lines of legend under a seven-row table is more page than
        # the reader was spending on it.
    if geo_rows:
        html.append(_div_table(geo_rows, tol, base=equity_base,
                               first_label="Equity geography"))
        html.append(
            f'<div style="margin-top:8px;{TYPE["prose"]}color:{P["muted"]};">'
            f'Geography targets partition the equity sleeve only, so they '
            f'total 100%.</div>'
        )
    if holding_rows:
        # Now/Target show the euro amount under the percentage, same style
        # as the asset-class and equity-geography tables above — same
        # invested-value base, since these rows' percentages are on that
        # same basis (see _ph_target_rows). The trend sub-line stays off:
        # this table is a list of targets, not a trend view.
        html.append(_div_table(holding_rows, tol, base=invested_base,
                               first_label="Per-holding target", subs=False,
                               value_subs=True))
        if exits_note:
            html.append(exits_note)
    if eq_rows:
        html.append(_div_table(eq_rows, tol, base=None,
                               first_label="Equities holding", subs=False))
    if fi_rows:
        html.append(_div_table(fi_rows, tol, base=None,
                               first_label="Fixed Income holding", subs=False))
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

