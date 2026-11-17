"""Allocation / holdings / hero / optimizer section builders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime, time
from html import escape as _esc
from numbers import Integral, Real
from typing import Any, Optional

import pandas as pd

from tarzan.models.portfolio import PortfolioMetrics
from tarzan.contracts.session_move import SessionMoveEvidence
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
from tarzan.export._palette import text_width
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
    MONO_STACK,
)
from tarzan.export.newsletter._format import (
    _display_ticker,
    _escaped_text,
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
    # was the endpoint: on Sat 29 Aug 2026 the 1D figure spanned Thu 27 → Fri 28.
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


def _session_move_display(metrics, percent):
    """Project paired source evidence; never infer euros from today's wealth."""
    proof = getattr(metrics, "session_move_evidence", None)
    receipt = {
        "availability": "UNAVAILABLE", "percent": str(float(percent)),
        "profit": "", "basis": "UNPREPARED",
        "anchor": "", "end": "", "source_anchor": "", "source_end": "",
    }
    if type(proof) is not SessionMoveEvidence:
        return None, receipt
    receipt["basis"] = str(proof.source_basis)
    valid_dates = all(
        type(value) is date for value in (
            proof.anchor_date, proof.end_date, proof.source_anchor_date, proof.source_end_date,
        )
    )
    values = (proof.return_pct, proof.profit_eur, proof.external_flow_eur)
    try:
        valid_values = all(
            isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            for value in values
        )
    except (TypeError, ValueError, OverflowError):
        valid_values = False
    if not (
        proof.availability in ("AVAILABLE", "DEGRADED")
        and proof.source_basis in ("HOLDINGS_PRICE_AGGREGATE", "ORDER_FLOW_ADJUSTED_NAV")
        and proof.source_basis == getattr(metrics, "history_source_basis", "UNPREPARED")
        and getattr(metrics, "history_availability", "AVAILABLE") in ("AVAILABLE", "DEGRADED")
        and not (
            getattr(metrics, "history_availability", "AVAILABLE") == "DEGRADED"
            and proof.availability != "DEGRADED"
        )
        and valid_dates and valid_values
        and proof.anchor_date < proof.end_date
        and proof.source_anchor_date < proof.source_end_date
        and math.isclose(float(proof.return_pct), float(percent), rel_tol=1e-12, abs_tol=1e-10)
    ):
        return None, receipt
    receipt.update(
        availability=proof.availability, profit=str(float(proof.profit_eur)),
        anchor=proof.anchor_date.isoformat(), end=proof.end_date.isoformat(),
        source_anchor=proof.source_anchor_date.isoformat(), source_end=proof.source_end_date.isoformat(),
    )
    return float(proof.profit_eur), receipt


_FUNDING_AMOUNT_FIELDS = (
    "initial_settlement_cash_eur", "ending_settlement_cash_eur",
    "cash_equivalent_securities_before_eur", "cash_equivalent_securities_after_eur",
    "initial_cash_equivalents_eur", "ending_cash_equivalents_eur",
    "cash_equivalents_target_eur", "protected_cash_eur",
    "external_contribution_eur", "gross_sales_eur", "gross_purchases_eur",
    "estimated_tax_eur", "fees_eur", "residual_eur",
)
_FUNDING_INVARIANTS = (
    "position_invariants_satisfied", "protected_cash_satisfied",
    "settlement_cash_satisfied", "frozen_positions_satisfied",
    "no_sell_satisfied", "serialized_actions_satisfied",
    "action_identity_satisfied", "units_invariants_satisfied",
    "wealth_invariants_satisfied", "cash_mechanics_satisfied",
    "non_cost_only_progress_satisfied",
)
_LOCKED_FUNDING_FIELDS = (
    "locked_settlement_cash_eur", "initial_free_settlement_cash_eur",
    "ending_free_settlement_cash_eur",
)
_TARGET_CHECK_KINDS = frozenset({
    "asset", "geography", "per_holding_equity", "per_holding_fi",
    "per_holding_portfolio", "cash",
})


def _funding_verification(verifications) -> Optional[Mapping[str, Any]]:
    """Return the final serialized-action funding proof, when available."""
    final = None
    for verification in verifications or []:
        if (
            isinstance(verification, Mapping)
            and isinstance(verification.get("kind"), str)
            and verification["kind"] == "funding"
        ):
            final = verification
    return final


def _funding_amount(value):
    """Keep a finite numeric source fact; missing/invalid is not numeric zero."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        if not math.isfinite(value):
            return None
        # Numpy's abs(int64.min) overflows; convert integers without a float
        # roundtrip so the raw marker and cent display remain lossless.
        return int(value) if isinstance(value, Integral) else value
    except (OverflowError, TypeError, ValueError):
        return None


def _funding_status(funding: Optional[Mapping[str, Any]]) -> str:
    """The engine's model proof is not evidence of broker-order execution."""
    if funding is None:
        return "UNKNOWN"
    status = funding.get("status")
    if not isinstance(status, str):
        return "UNKNOWN"
    if status == "NON_EXECUTABLE":
        return status
    if status != "EXECUTABLE":
        return "UNKNOWN"
    fields = _FUNDING_AMOUNT_FIELDS
    invariants = _FUNDING_INVARIANTS
    if (
        "locked_cash_balances_satisfied" in funding
        or any(key in funding for key in _LOCKED_FUNDING_FIELDS)
    ):
        fields += _LOCKED_FUNDING_FIELDS
        invariants += ("locked_cash_balances_satisfied",)
    if (
        any(_funding_amount(funding.get(key)) is None for key in fields)
        or any(funding.get(key) is not True for key in invariants)
    ):
        return "UNKNOWN"
    return "EXECUTABLE"


def _targets_in_tolerance(verifications) -> bool:
    """Only nonempty native target evidence can support an alignment claim."""
    checks = []
    for verification in verifications or []:
        if not isinstance(verification, Mapping):
            return False
        no_solution = verification.get("no_solution")
        if no_solution is not None and no_solution is not False:
            return False
        if (
            isinstance(verification.get("kind"), str)
            and verification["kind"] in _TARGET_CHECK_KINDS
        ):
            items = verification.get("items")
            if not isinstance(items, (list, tuple)):
                return False
            fields = ("actual_eur", "target_eur") if verification["kind"] == "cash" else (
                "actual_pct", "target_pct")
            if any(
                not isinstance(item, Mapping)
                or any(_funding_amount(item.get(field)) is None for field in fields)
                for item in items
            ):
                return False
            checks.append(verification)
    return bool(
        checks
        and any(isinstance(check.get("items"), (list, tuple)) and check["items"]
                for check in checks)
        and all(isinstance(check.get("status"), str) and check["status"] == "✓ OK"
                for check in checks)
    )


def _rebalance_evidence(verifications, n_actions: int) -> tuple[str, bool, bool]:
    status = _funding_status(_funding_verification(verifications))
    infeasible = any(
        isinstance(verification, Mapping)
        and verification.get("no_solution") is not None
        and verification.get("no_solution") is not False
        for verification in verifications or []
    )
    aligned = bool(
        n_actions == 0 and status == "EXECUTABLE"
        and _targets_in_tolerance(verifications)
    )
    return status, bool(infeasible), aligned


def _funding_cell(funding, key: str) -> dict:
    value = _funding_amount(funding.get(key)) if funding is not None else None
    # Round display only. The data marker retains the unrounded source fact,
    # and neither funding success nor a cash deficit is inferred from the text.
    sign = "−" if value is not None and round(value, 2) < 0 else ""
    if value is None:
        text = "—"
    elif isinstance(value, Integral):
        text = f"{sign}€{abs(value):,}.00"
    else:
        text = f"{sign}€{abs(value):,.2f}"
    return {
        "field": key,
        "raw": str(value) if value is not None else "",
        "text": text,
    }


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
    status, infeasible, aligned = _rebalance_evidence(
        m.rebalancing_verifications, len(suggestions))
    if status == "NON_EXECUTABLE":
        parts.append("and the draft rebalance is not executable until funding or policy checks are resolved")
    elif infeasible:
        parts.append("and no feasible rebalance was produced")
    elif status != "EXECUTABLE":
        parts.append("and any rebalance remains a draft with unverified funding")
    elif suggestions:
        parts.append("and the optimizer shows a modeled rebalance below")
    elif aligned:
        parts.append("and checked targets are within tolerance")
    else:
        parts.append("and no trades were proposed; target alignment is unproven")

    return {
        "text": ", ".join(parts) + ".",
        "is_positive": (
            week_return is None
            or (isinstance(week_return, float) and pd.isna(week_return))
            or float(week_return) >= 0
        ),
    }

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
        # Explicit escaped text, shared with STATE; safe only because the
        # text escape runs first, not because the source is trusted.
        return {"label": _escaped_text(label), "value": _escaped_text(value),
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
        gap = round(gap, 2) + 0.0  # the sign and tone of what is PRINTED: never "+0.00"
        status_bar.append(_bar(
            f"VS {_bench_short(ctx.benchmark_geo)}",
            f'{"+" if gap > 0 else ("\u2212" if gap < 0 else "")}'
            f'{abs(gap):.2f}pp', _tone(gap)))
    # A NaN is not a value: the risk tiles skip it, and so does the bar.
    if not is_missing(risk.get("beta")):
        status_bar.append(_bar("\u03b2", f'{float(risk["beta"]):.2f}'))
    if not is_missing(risk.get("sharpe")):
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
    market_open = _market_is_open(perf)
    stamp += f' \u00b7 market {"OPEN" if market_open else "CLOSED"}'
    basis = (f"vs {base_label} close" if live and base_label
             else (f"at the {end_label} close" if (not live and end_label) else ""))
    return {
        "date_short": now.strftime("%a, %d %b %Y"),
        "stamp": stamp,
        # The statement masthead sets the three facts separately: the date, the
        # close the figures rest on, and the market state as a stamp.
        "basis": basis,
        "market_open": market_open,
        "issue_number": issue_number,
        "inception_date": inception_date,
        "status_bar": tuple(status_bar),
    }


def _bench_short(name: Optional[str]) -> str:
    """The benchmark's index word, upper-cased, for a label that has to fit a
    status-bar cell: "iShares MSCI ACWI" -> "ACWI", "SPDR MSCI ACWI IMI Acc" ->
    "ACWI IMI" (share-class words dropped; IMI kept with what it qualifies)."""
    share_class = {"acc", "dist", "ucits", "etf", "eur", "usd", "hedged", "hdg"}
    parts = [w for w in str(name or "").replace("-", " ").split()
             if w and w.casefold() not in share_class]
    if not parts:
        return "BENCHMARK"
    if parts[-1].upper() == "IMI" and len(parts) > 1:
        return f"{parts[-2]} {parts[-1]}".upper()
    return parts[-1].upper()

def _build_hero(ctx: _NewsletterContext) -> dict:
    m = ctx.metrics
    cfg = ctx.config

    # Two distinct P&L views at portfolio level:
    #   * Unrealized PnL — the snapshot gain on the positions held *today*
    #     vs their cost basis (= the Excel "RTD"). Numerator/denominator
    #     are current-holdings only; realized gains and income are excluded.
    #   * Total PnL — the tracked-period gain (realized + unrealized +
    #     coupons/dividends), over net contributed capital. Incoming securities
    #     enter at dated market value when available, not their carrying price:
    #     gains before that contribution are outside the tracked period.
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
        # A move that prints as 0.00% is flat, not a gain.
        if value is not None and not is_missing(value) and abs(round(float(value), 2)) < 1e-9:
            return "flat"
        if value is None:
            return "flat"
        return "pos" if float(value) >= 0 else "neg"

    perf = getattr(m, "performance", None) or {}
    cagr_pct = perf.get("cagr")
    session_pct = perf.get("1d")
    try:
        known_session_pct = (
            isinstance(session_pct, Real) and not isinstance(session_pct, bool)
            and math.isfinite(session_pct)
        )
    except (TypeError, ValueError, OverflowError):
        known_session_pct = False
    if not known_session_pct:
        session_pct = None
    session_eur, session_receipt = (
        _session_move_display(m, session_pct) if session_pct is not None else (None, None)
    )
    def _tile(label, value, caption, tone="flat"):
        # The context API retains escaped text; its type prevents a second
        # escape at the now-enabled .html.j2 template boundary.
        return {"label": _escaped_text(label), "value": _escaped_text(value),
                "caption": _escaped_text(caption), "tone": tone}

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
    # cumulative figures is what the timing of contributions cost or earned,
    # which no single tile here states and which
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
        now_pp = round(gap["now_pp"], 2) + 0.0  # sign of the printed value
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
            parts.append(_eur(session_eur, signed=True))
            parts.append(
                "flow-adjusted NAV" if session_receipt["basis"] == "ORDER_FLOW_ADJUSTED_NAV"
                else "historical holdings basket"
            )
            if session_receipt["availability"] == "DEGRADED":
                parts.append("estimated historical data")
            if (
                session_receipt["source_anchor"] != session_receipt["anchor"]
                or session_receipt["source_end"] != session_receipt["end"]
            ):
                parts.append(
                    f'source {session_receipt["source_anchor"]} to {session_receipt["source_end"]}'
                )
        else:
            parts.append("EUR attribution unavailable")
        parts.append(_session_basis(perf, m))
        # How much of the book that figure covers, when it is not all of it.
        note = _priced_today_note(perf)
        if note:
            parts.append(note)
        session_tile = _tile(
            "Session", _pct(session_pct, signed=True),
            " \u00b7 ".join(parts), _tone(session_pct))
        session_tile["session_proof"] = session_receipt
        state_tiles.append(session_tile)
    # ── The statement's State block, structured ─────────────────────────────
    # The same facts as the tiles above, as fields the template lays out: the
    # value with the session pill beside it, invested capital, and the four
    # ledger cells (Total P&L, Unrealized P&L, TWR, MWR). The benchmark gap left
    # this block on the owner's decision (2026-11-15); it stays in the KPI
    # matrix and the since-inception chart.
    def _fig(value, fmt, **kw):
        return None if is_missing(value) else fmt(value, **kw)
    session_day = ""
    if session_pct is not None:
        try:
            session_day = _prev_session_label(m, "%d %b") or ""
        except Exception:  # noqa: BLE001 — a caption, never a reason to fail the issue
            session_day = ""
    state = {
        "value": _eur(m.total_value, decimals=0),
        "invested": _eur_smart(m.invested_value),
        "days": span_days,
        "session": None if session_pct is None else {
            "pct": _pct(session_pct, signed=True),
            # Whole euros in the pill; the caption beneath states the cents.
            "eur": _fig(session_eur, _eur, decimals=0, signed=True),
            "day": session_day,
            "tone": _tone(session_pct),
            "proof": session_receipt,
            # The engine's own caption (euros to the cent, basis, window), which
            # the semantic gate reads back word for word.
            "caption": session_tile["caption"],
        },
        "pnl": {"eur": _fig(total_pnl_eur, _eur_smart, signed=True),
                "pct": _fig(total_pnl_pct, _pct, signed=True),
                "caption": "on contributed capital",
                "tone": _tone(total_pnl_eur if not is_missing(total_pnl_eur) else total_pnl_pct)},
        "unrealized": {"eur": _fig(unrealized_eur, _eur_smart, signed=True),
                       "pct": _fig(unrealized_pct, _pct, signed=True),
                       "caption": "on open positions",
                       "tone": _tone(unrealized_eur if not is_missing(unrealized_eur) else unrealized_pct)},
        "twr": None if twr_pct is None else {
            "si": _pct(twr_pct, signed=True), "tone": _tone(twr_pct),
            "ann": _fig(twr_ann, _pct, signed=True),
            "cagr": _fig(cagr_pct, _pct, signed=True) if twr_ann is None else None},
        "mwr": None if mwr_cum is None else {
            "si": _pct(mwr_cum, signed=True), "tone": _tone(mwr_cum),
            "ann": _fig(m.xirr_pct, _pct, signed=True),
            "net": _fig(getattr(m, "xirr_net_tax_pct", None), _pct, signed=True)},
    }
    # STATE ends with Session. TER and the risk figures live in the PORTFOLIO
    # KPI matrix, per window and beside the target and the benchmark.


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
            total_eur=total_series, w=354, h=190,
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
    funding_status, rebal_infeasible, rebal_aligned = _rebalance_evidence(
        m.rebalancing_verifications, n_actions)

    if funding_status == "NON_EXECUTABLE":
        rebal_label = "Not executable"
        rebal_sublabel = "funding or policy checks failed"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif rebal_infeasible:
        rebal_label = "Infeasible"
        rebal_sublabel = "no feasible plan"
        rebal_color = PALETTE["red"]
        rebal_bg = PALETTE["red_bg"]
    elif rebal_aligned:
        rebal_label = "Aligned"
        rebal_sublabel = "checked targets within tolerance"
        rebal_color = PALETTE["green"]
        rebal_bg = PALETTE["green_bg"]
    elif funding_status != "EXECUTABLE":
        rebal_label = "Unverified"
        rebal_sublabel = "funding proof unavailable or incomplete"
        rebal_color = PALETTE["amber"]
        rebal_bg = PALETTE["amber_bg"]
    elif n_actions == 0:
        rebal_label = "No trades"
        rebal_sublabel = "target alignment unproven"
        rebal_color = PALETTE["amber"]
        rebal_bg = PALETTE["amber_bg"]
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
        # Whether tracked-period Total PnL is a real (order-derived) figure
        # distinct from Unrealized; False on the holdings-only path.
        "has_total_pnl": has_total_pnl,
        "return_basis_note": _escaped_text(
            "Total P&L, TWR and MWR cover the tracked period. "
            "Securities transfers are contributions at market value on their "
            "operation date (order-price fallback if unavailable). "
            "Carrying prices remain the cost-basis reference."
        ) if has_total_pnl else "",
        # "Since inception · Mon YYYY" caption (empty when inception unknown).
        "inception_label": inception_label,
        # The nine state tiles, in display order. Optional ones are absent
        # rather than blank, so the grid never shows an empty box.
        "tiles": tuple(state_tiles),
        "state": state,
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
        "rebal_executable": funding_status == "EXECUTABLE" and not rebal_infeasible,
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
    # A statement block, not a card: the heading in ink with the figures beside
    # it, the method as a note beneath.
    html = (
        f'<div style="{TYPE["prose"]}color:{P["muted"]};">'
        f'<span style="font-weight:700;color:{P["ink"]};">Net-of-tax estimate</span>'
        + (f' &nbsp;{figs_html}' if figs_html else "")
        + f'<div style="margin-top:3px;{TYPE["note"]}color:{P["subtle"]};">'
        f'Estimate only: average-cost basis, '
        f'{rate_txt}, realized losses offset later gains where Italian rules allow '
        f'(ETF/fund gains are not offsettable). Excludes coupon/dividend withholding and the cost basis of '
        f'transferred-in positions. TWR is gross of tax.</div>'
        f'</div>'
    )
    return {"available": True, "html": html}

# Presence bands, in the order the table lists them: why this instrument is in
# this issue at all. Also the sort's primary key — within a band rows are
# alphabetical by ticker, and there is no class/role grouping: this is a
# reference table looked up by symbol, not a view of the portfolio's shape.
_PRESENCE_BANDS = ("Portfolio", "Watchlist", "Hist. Portfolio only")
# The data-sources table (Appendix). Bands whose instruments the issue names in
# full elsewhere (Holdings, Watchlist) print no name here; a closed position does.
_SOURCES_NAMED_ELSEWHERE = frozenset({"Portfolio", "Watchlist"})
_SOURCES_ISIN_PX = 80                      # 12 mono glyphs at 9.5px (~71px) + air
_MONO_EM = 0.62                            # a monospace glyph's advance, em
_SOURCES_LINE_PX = 354.0 - 6.0 - _SOURCES_ISIN_PX   # ticker + feeds on one line

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
    from tarzan.models.instrument_key import normalize_isin, normalize_ticker

    m = ctx.metrics
    rows: list[dict] = []
    by_identity: dict[str, dict] = {}
    evidence = getattr(m, "intraday_evidence", {}) or {}

    def intraday_note(ticker):
        fact = evidence.get(ticker, {})
        if not fact:
            return ""
        if fact.get("state") == "MISSING":
            return "No current-session bars"
        observed = str(fact.get("observed_at") or "")
        instant = (observed[11:16] + " UTC") if observed.endswith("+00:00") else ""
        label = (
            "Lagged bars" if fact.get("state") == "LAGGED_CURRENT_SESSION"
            else "Last session" if fact.get("state") == "LAST_COMPLETED_SESSION"
            else "Single bar" if fact.get("native_bar_count") == 1 else ""
        )
        return _escaped_text(" · ".join(item for item in (label, instant) if item))

    def _add(row: dict, *keys: str) -> None:
        rows.append(row)
        for key in keys:
            if key:
                by_identity.setdefault(key, row)

    for record in getattr(m, "ticker_resolutions", ()) or ():
        canonical = str(record.get("canonical_ticker") or "")
        isin = normalize_isin(record.get("isin"))
        presence = str(record.get("portfolio_presence") or "")
        native = record.get("historical_price_sources") or ()
        history_note = ""
        history_feed = str(record.get("history_ticker") or record.get("current_ticker") or canonical)
        if len(native) == 1:
            native_isin, venue, link = native[0]
            history_feed = f"{native_isin} · {venue}"
            history_note = "Official clean liquidation price" if link else "Official clean price"
        _add({
            "ticker": _escaped_text(_display_ticker(canonical) or ""),
            "name": _escaped_text(ctx.reference.name_for(isin, canonical)
                                 or record.get("name") or ""),
            "isin": isin,
            # The listing the daily bars came from; the selected symbol when the
            # instrument resolved but never returned history.
            "hist_ric": history_feed,
            "history_note": history_note,
            "intr_ric": str(record.get("intraday_effective_ticker") or ""),
            "intraday_note": intraday_note(canonical),
            "presence": _PRESENCE_BAND_OF.get(presence, presence),
        }, isin, normalize_ticker(canonical))

    resolved = getattr(m, "benchmark_tickers", {}) or {}
    quotes = getattr(m, "intraday_quotes", {}) or {}
    for name, requested, isin in ctx.reference.benchmark_identities:
        hist = str(resolved.get(name) or "")
        # A shared symbol (MATE: Nasdaq and UCITS) must not merge two funds:
        # a row matched by symbol whose ISIN differs is another instrument.
        known = next((by_identity[k] for k in (isin, normalize_ticker(hist or requested))
                      if k and k in by_identity
                      and by_identity[k]["isin"] in ("", isin or by_identity[k]["isin"])), None)
        if known is not None:
            if _presence_rank(known["presence"]) > _PRESENCE_BANDS.index("Watchlist"):
                known["presence"] = "Watchlist"
            known["isin"] = known["isin"] or isin
            continue
        quote = quotes.get(hist)
        quote = quote if isinstance(quote, dict) else {}
        _add({
            "ticker": _escaped_text(_display_ticker(requested) or requested),
            "name": _escaped_text(name),
            "isin": isin,
            "hist_ric": hist,
            "history_note": "",
            # A benchmark's intraday series can come from a sibling venue, and
            # it is the catalog — not the request — that records which one.
            "intr_ric": str(quote.get("intraday_source_ticker")
                            or quote.get("source_ticker") or ""),
            "intraday_note": intraday_note(hist),
            "presence": "Watchlist",
        }, isin, normalize_ticker(hist or requested))

    rows.sort(key=lambda r: (_presence_rank(r["presence"]),
                             (r["ticker"] or r["name"]).upper()))
    requested = tuple(getattr(m, "intraday_requested_tickers", ()) or ())
    quotes = getattr(m, "intraday_quotes", {}) or {}
    quality_note = ""
    if evidence and requested:
        lagged = sum(fact.get("state") == "LAGGED_CURRENT_SESSION" for fact in evidence.values())
        missing = len(set(requested) - set(quotes))
        quality_note = (
            f"Intraday paths: {len(quotes)}/{len(requested)} requested listings "
            f"(portfolio and watchlist); {lagged} lagged, {missing} without a drawable path. "
            "These counts are not portfolio valuation coverage. Bar times remain the actual last observations."
        )
    return {"available": bool(rows), "rows": tuple(rows), "quality_note": quality_note,
            "table_html": _sources_table_html(rows, ctx),
            "intraday_note": _sources_intraday_note(rows)}


def _sources_intraday_note(rows: list) -> str:
    """One sentence for the whole table when no row has an intraday feed: the
    column it would fill was a hundred dashes."""
    if rows and not any(str(r.get("intr_ric") or "").strip() for r in rows):
        return "No instrument has an intraday feed in this issue."
    return ""


def _sources_table_html(rows: list, ctx: _NewsletterContext) -> str:
    """The data sources grouped by presence (Portfolio, Watchlist, historical
    only), two lines a row: the instrument with its ISIN at the right, then the
    feeds on one mono line (daily listing, then "↗" and the intraday listing,
    then the notes on either). Each row carries the rail of its class when the
    taxonomy knows it.

    A three-column table put the feed in 66px, where "CL2.MI · ↗ CL2.MI · Last
    session · 15:00 UTC" wrapped to five lines and a hundred rows ran to seven
    thousand pixels. The ticker and name arrive escaped (``_escaped_text`` in
    ``ticker_sources``); the listings and the ISIN do not."""
    P = PALETTE
    if not rows:
        return ""
    tax = ctx.reference.taxonomy
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(str(r.get("presence") or "\u2014"), []).append(r)
    mono = f'font-family:{MONO_STACK};font-size:9.5px;line-height:1.3;'
    out = []
    for presence, rs in groups.items():
        # The name is printed here only where the issue prints it nowhere else:
        # a closed position. A held or watched instrument has its name in the
        # Holdings or Watchlist grid, and without it the ticker and the feeds
        # share one line when they fit (owner's choice, Nov 2026: ~1,300px).
        named = presence not in _SOURCES_NAMED_ELSEWHERE
        body = ""
        for r in rs:
            rail = ASSET_COLORS.get(_class_of_action(r.get("isin"), r.get("ticker"), tax) or "",
                                    P["row_rule"])
            feed = [f'<span style="color:{P["muted"]};">{_esc(str(r.get("hist_ric") or "\u2014"))}</span>']
            plain = [str(r.get("hist_ric") or "\u2014")]
            if r.get("history_note"):
                feed.append(str(r["history_note"]))
                plain.append(str(r["history_note"]))
            if str(r.get("intr_ric") or "").strip():
                feed.append(f'<span style="color:{P["muted"]};">\u2197 {_esc(str(r["intr_ric"]))}</span>')
                plain.append(f'\u2197 {r["intr_ric"]}')
            if r.get("intraday_note"):
                feed.append(str(r["intraday_note"]))
                plain.append(str(r["intraday_note"]))
            tk = str(r.get("ticker") or "")
            name = str(r.get("name") or "\u2014")
            isin_cell = (f'<td align="right" style="padding:{{top}}px 0 {{bot}}px 8px;vertical-align:top;'
                         f'{mono}color:{P["muted"]};white-space:nowrap;{{rule}}">'
                         f'{_esc(str(r.get("isin") or "\u2014"))}</td>')
            rule = f'border-bottom:1px solid {P["row_rule"]};'
            one_line = (not named and text_width(tk, TYPE_PX["data"], 700) + 6.0
                        + len(" \u00b7 ".join(plain)) * _MONO_EM * 9.5 <= _SOURCES_LINE_PX)
            if one_line:
                body += (f'<tr><td style="padding:5px 0 5px 6px;border-left:2px solid {rail};{rule}'
                         f'font-size:{TYPE_PX["data"]}px;line-height:1.25;color:{P["ink"]};">'
                         + (f'<b>{tk}</b> ' if tk else "")
                         + f'<span style="{mono}color:{P["faint"]};">' + " \u00b7 ".join(feed)
                         + '</span></td>'
                         + isin_cell.format(top=6, bot=5, rule=rule) + '</tr>')
                continue
            body += (f'<tr><td style="padding:5px 0 1px 6px;border-left:2px solid {rail};'
                     f'font-size:{TYPE_PX["data"]}px;line-height:1.25;color:{P["ink"]};">'
                     + (f'<b>{tk}</b> ' if tk else "")
                     + (f'<span style="color:{P["muted"]};">{name}</span>' if named else "")
                     + '</td>' + isin_cell.format(top=6, bot=1, rule="") + '</tr>'
                     f'<tr><td colspan="2" style="padding:0 0 5px 6px;border-left:2px solid {rail};'
                     f'{rule}{mono}color:{P["faint"]};">'
                     + " \u00b7 ".join(feed) + '</td></tr>')
        out.append(
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="table-layout:fixed;border-collapse:collapse;margin-top:8px;font-family:{FONT_STACK};">'
            f'<colgroup><col><col style="width:{_SOURCES_ISIN_PX}px;"></colgroup>'
            f'<tr><td colspan="2" style="padding:10px 0 3px 0;border-bottom:1px solid {P["border"]};'
            f'{TYPE["label"]}color:{P["muted"]};">{_esc(presence)} \u00b7 {len(rs)}</td></tr>'
            f'<tr><td style="padding:3px 0 3px 6px;{TYPE["label"]}font-size:9.5px;color:{P["faint"]};">'
            f'Instrument \u00b7 feeds</td>'
            f'<td align="right" style="padding:3px 0;{TYPE["label"]}font-size:9.5px;color:{P["faint"]};">ISIN</td></tr>'
            f'{body}</table>')
    return "".join(out)


def _build_allocation(ctx: _NewsletterContext) -> dict:
    """Build asset-class allocation rows (Excel Dashboard pattern)."""
    m = ctx.metrics
    cfg = ctx.config
    targets = cfg.invested_allocation_targets_pctg or {}
    alloc_df = m.allocation_by_class

    timeline = m.allocation_timeline or {}
    asset_series = timeline.get("asset")

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
        rows.append({
            "name": klass,
            "color": color,
            "actual_pct": _pct_smart(actual),
            "actual_pct_raw": actual,
            "target_pct": _pct_smart(target) if target is not None else None,
            # Notional: a class the plan runs past 100% of capital (a book heavy
            # in 2x trackers) keeps its figure. The old 100% bar clipped it here.
            "target_left": max(float(target), 0.0) if target is not None else None,
            "delta": _signed_pp(delta) if delta is not None else None,
            "delta_color": _band_colour(delta, target, cfg),
            "bar_width": min(max(actual, 1), 100),
            "spark": _spark(spark_vals, target, color) if spark_vals else None,
        })

    # Strategic Cash & Cash Equivalents reserve, not spendable bank cash.
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
            "name": "Strategic reserve",
            "color": ASSET_COLORS["Cash & Cash Equivalents"],
            "actual_pct": _eur_smart(cash_actual),
            "actual_pct_raw": cash_pct_of_total,
            "target_pct": _eur_smart(cash_tgt),
            "delta": _eur_smart(delta_eur, signed=True),
            "delta_color": (PALETTE["green"] if abs(delta_eur) <= cash_band + 1e-9
                            else PALETTE["red"]),
            "bar_width": min(max(cash_pct_of_total, 1), 100),
            "is_eur": True,
            # Raw EUR figures so the diversification table can show reserve as a
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


# ── Allocation: the butterfly ────────────────────────────────────────────────
#
# Each block is one SVG on the ground: a row per line, the name in a centre
# column, TODAY as a bar growing left from it and PLAN as a bar growing right,
# both on one scale (the block's largest weight), the plan's bar at half
# opacity. The figures stand in fixed outer columns, aligned down the block:
# 1M · today % · today € on the left, plan % · plan € · gap on the right. Every
# column is measured on the block's own text (text_width), so class names,
# regions and tickers all take the same grammar on one line, nothing is
# abbreviated and nothing wraps. The owner chose this over the stacked
# bridge (Nov 2026: "amo questo"), asking for it compact.
#
# A line the plan sells has a red name and no right bar; a line bought from
# nothing has no left bar. The sum of the sold lines is a row under the block.

_BW = 354.0                      # the content column every chart is drawn for (phone)
_ROW = 18.0                      # one line per row
_TOP = 13.0                      # the column heads
_BAR_H = 9.0
_FS, _NFS = 9.5, 10.0            # figures, names
_ASSET_SHORT: dict = {}          # every class name prints whole
_GEO_SHORT = {"Eurozone EMU": "Eurozone", "Dev ex-USA ex-EMU ex-JP": "Other dev",
              "Emerging Markets": "Emerging"}


def _mix(fg: str, bg: str, a: float) -> str:
    """``fg`` over ``bg`` at ``a``, as a flat hex (an emailed rgba() is unreliable)."""
    a = max(0.0, min(1.0, a))
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(f[i] * a + b[i] * (1 - a))):02x}" for i in range(3))


def _lift(colour: Optional[str]) -> str:
    """A colour tuned for a light page, lifted toward white so it holds on the dark
    ground without changing hue (the geography set)."""
    if not colour or not str(colour).startswith("#") or len(str(colour)) != 7:
        return PALETTE["accent"]
    return _mix("#FFFFFF", str(colour), 0.28)


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


def _bt(x: float, y: float, s: str, *, fill: str, weight: int = 400,
        anchor: str = "start", size: float = 10.0, label: bool = False) -> str:
    """One SVG text. ``label`` adds the LABEL role's tracking; its capitals are
    written as such, since text-transform does not reach SVG text."""
    track = ' letter-spacing="0.6"' if label else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{track}>{s}</text>')


def _pc1(v: float) -> str:
    return f"{float(v):.1f}%"


def _move_text(trend: Optional[list]) -> str:
    """The month's change in weight, in points; blank without a series."""
    tr = [float(x) for x in (trend or []) if x is not None]
    return _signed_pp(tr[-1] - tr[0]) if len(tr) >= 2 else ""


def _move_colour(trend: Optional[list]) -> str:
    """Green for a weight that rose over the month, red for one that fell; muted
    for a move that prints as 0.0."""
    tr = [float(x) for x in (trend or []) if x is not None]
    if len(tr) < 2 or abs(tr[-1] - tr[0]) < 0.05:
        return PALETTE["muted"]
    return PALETTE["green"] if tr[-1] > tr[0] else PALETTE["red"]


def _gap_text(r: dict) -> str:
    return _signed_pp(r["drift"]) if r["drift"] is not None else "\u2014"


def _bridge_row(*, key: str, label: str, colour: str, now: float,
                target: Optional[float], base: float, trend=None,
                legacy: bool = False, plan_base: Optional[float] = None) -> dict:
    """``plan_base`` is the euro base of the PLAN side when it differs from today's:
    a sleeve the plan sizes differently from what is held."""
    now = float(now or 0.0)
    plan_base = base if plan_base is None else plan_base
    tgt = None if target is None else float(target)
    return {"key": key, "label": label, "colour": colour, "now": now, "target": tgt,
            "drift": None if tgt is None else now - tgt,
            "eur_now": now / 100.0 * base, "eur_tgt": (tgt or 0.0) / 100.0 * plan_base,
            "trend": trend, "legacy": legacy}


def _sleeve_note(what: str, today: float, plan: float, tail: str = ".") -> str:
    """The "% of the sleeve" caption, naming the plan's euro base when it differs."""
    bases = f"{_eur_smart(today)} notional"
    if abs(plan - today) >= 0.5:
        bases += f" today, {_eur_smart(plan)} in the plan"
    return f"% of {what} ({bases}){tail}"


def _bridge_svg(left: list, right: list, cfg, *, two: bool = False, tail=(),
                aria: str = "Allocation, today against the plan") -> str:
    """One block's figure, then the ``tail`` rows (Total, Strategic reserve,
    sold lines) as a table under it. ``two`` is accepted for the callers that
    still pass it and ignored: every block is one line per row now."""
    P = PALETTE
    by_key: dict = {}
    for r in left:
        if r["now"] > 0:
            by_key[r["key"]] = dict(r, tgt=float(r["target"] or 0.0))
    for r in right:
        if (r["target"] or 0) <= 0:
            continue
        if r["key"] in by_key:
            by_key[r["key"]]["tgt"] = float(r["target"])
            by_key[r["key"]]["eur_tgt"] = r["eur_tgt"]
        else:
            by_key[r["key"]] = dict(r, now=0.0, tgt=float(r["target"]))
    rows = sorted(by_key.values(), key=lambda r: -max(r["now"], r["tgt"]))
    if not rows:
        return ""
    mx = max(max(r["now"] for r in rows), max(r["tgt"] for r in rows), 1.0)

    def w(txt, weight=400, size=_FS):
        return text_width(txt, size, weight)

    def figs(r, side):
        if side == "now":
            return f'{_pc1(r["now"])} {_eur_smart(r["eur_now"])}'
        return f'{_pc1(r["tgt"])} {_eur_smart(r["eur_tgt"])}'

    # Columns, measured on this block's own text.
    l_mv = max([w(_move_text(r["trend"])) for r in rows] + [0.0]) + 4.0
    l_fig = max([w(figs(r, "now")) for r in rows if r["now"] > 0] + [0.0])
    r_fig = max([w(figs(r, "tgt")) for r in rows if r["tgt"] > 0] + [0.0])
    r_gap = max([w(_gap_text(r), 700) for r in rows] + [0.0]) + 4.0
    cw = max([w(r["label"], 700, _NFS) for r in rows] + [0.0]) + 10.0
    lf = l_mv + l_fig                       # right edge of the TODAY figures
    rf = _BW - r_gap - r_fig                # left edge of the PLAN figures
    lane = max((rf - lf - cw) / 2.0, 12.0)  # one bar lane each side of the name
    cx0, cx1 = lf + lane, lf + lane + cw
    barmax = lane - 6.0
    lab = {"fill": P["faint"], "weight": 700, "size": 8.5, "label": True}
    g = [_bt(lf, 9.0, "1M \u00b7 TODAY \u00b7 \u20ac", anchor="end", **lab),
         _bt(rf, 9.0, "PLAN \u00b7 \u20ac \u00b7 GAP", **lab)]
    y = _TOP
    for r in rows:
        yc = y + _ROW / 2.0
        g.append(f'<line x1="0" x2="{_BW:g}" y1="{y + _ROW:.1f}" y2="{y + _ROW:.1f}" '
                 f'stroke="{P["row_rule"]}" stroke-width="1"/>')
        g.append(_bt((cx0 + cx1) / 2.0, yc + 3.5, _esc(r["label"]),
                     fill=P["red"] if r.get("legacy") else P["ink"], weight=700, size=_NFS,
                     anchor="middle"))
        if r["now"] > 0:
            bw = max(1.5, r["now"] / mx * barmax)
            g.append(f'<rect x="{cx0 - 3 - bw:.1f}" y="{yc - _BAR_H / 2:.1f}" width="{bw:.1f}" '
                     f'height="{_BAR_H:g}" fill="{r["colour"]}"/>')
            g.append(_bt(lf, yc + 3.5, f'<tspan font-weight="600">{_esc(_pc1(r["now"]))}</tspan> '
                         f'<tspan fill="{P["muted"]}">{_esc(_eur_smart(r["eur_now"]))}</tspan>',
                         fill=P["ink"], size=_FS, anchor="end"))
            if _move_text(r["trend"]):
                g.append(_bt(l_mv - 4.0, yc + 3.5, _esc(_move_text(r["trend"])),
                             fill=_move_colour(r["trend"]), size=_FS, anchor="end"))
        if r["tgt"] > 0:
            bw = max(1.5, r["tgt"] / mx * barmax)
            g.append(f'<rect x="{cx1 + 3:.1f}" y="{yc - _BAR_H / 2:.1f}" width="{bw:.1f}" '
                     f'height="{_BAR_H:g}" fill="{r["colour"]}" fill-opacity="0.55"/>')
            g.append(_bt(rf, yc + 3.5, f'<tspan font-weight="600">{_esc(_pc1(r["tgt"]))}</tspan> '
                         f'<tspan fill="{P["muted"]}">{_esc(_eur_smart(r["eur_tgt"]))}</tspan>',
                         fill=P["ink"], size=_FS))
            g.append(_bt(_BW, yc + 3.5, _esc(_gap_text(r)),
                         fill=_band_colour(r["drift"], r["tgt"], cfg), weight=700, size=_FS,
                         anchor="end"))
        y += _ROW
    svg = (f'<svg width="100%" viewBox="0 0 {_BW:g} {int(y) + 2}" preserveAspectRatio="xMidYMid meet" '
           f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(aria)}" '
           f'style="display:block;width:100%;" font-family="{FONT_STACK}">{"".join(g)}</svg>')
    return svg + _bridge_tail(tail)


def _bridge_tail(tail) -> str:
    """The rows under the stacks (Total, Strategic reserve): name, today's weight
    and euros, an arrow, the plan's, the gap."""
    if not tail:
        return ""
    P = PALETTE
    cell = f'padding:4px 0;border-top:1px solid {P["row_rule"]};{TYPE["data"]}font-weight:400;'
    rows = ""
    for t in tail:
        now = (f'{_esc(t["now_pc"])} ' if t.get("now_pc") else "") + f'<span style="color:{P["muted"]};">{_esc(t["now_eur"])}</span>'
        plan = (f'{_esc(t["plan_pc"])} ' if t.get("plan_pc") else "") + f'<span style="color:{P["muted"]};">{_esc(t["plan_eur"])}</span>'
        rows += (f'<tr><td style="{cell}color:{P["ink"]};">{_esc(t["name"])}</td>'
                 f'<td style="{cell}color:{P["ink"]};white-space:nowrap;">{now}</td>'
                 f'<td style="{cell}color:{P["faint"]};">\u2192</td>'
                 f'<td style="{cell}color:{P["ink"]};white-space:nowrap;">{plan}</td>'
                 f'<td align="right" style="{cell}font-weight:700;color:{t["colour"]};white-space:nowrap;">'
                 f'{_esc(t["gap"])}</td></tr>')
    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="table-layout:fixed;border-collapse:collapse;margin-top:2px;font-family:{FONT_STACK};'
            f'font-variant-numeric:tabular-nums;"><colgroup><col><col style="width:96px;">'
            f'<col style="width:18px;"><col style="width:96px;"><col style="width:44px;"></colgroup>'
            f'{rows}</table>')


def _bridge_card(svg: str) -> str:
    """The figure on the ground: no shell. The statement draws rules, not boxes."""
    if not svg:
        return ""
    return f'<div style="margin-top:4px;font-size:0;line-height:0;">{svg}</div>'


def _bridge_title(title: str) -> str:
    """A block's title: small caps over a rule, the statement's block head."""
    P = PALETTE
    return (f'<div style="margin-top:10px;padding-bottom:3px;border-bottom:1px solid {P["border"]};'
            f'{TYPE["label"]}color:{P["muted"]};">{_esc(title)}</div>')


def _alloc_note(text: str, top: int = 6) -> str:
    return (f'<div style="margin-top:{top}px;{TYPE["note"]}color:{PALETTE["muted"]};">'
            f'{_esc(text)}</div>')


def _holding_bridge_rows(ctx: _NewsletterContext, items: list, *, base: float,
                         plan_base: Optional[float] = None,
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
    root_isins: dict[str, set] = {}
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
                root_isins.setdefault(tkr.upper().split(".")[0], set()).add(isin)
            class_of[isin] = str(row.get("asset_class", "") or "")
            ticker_of[isin] = tkr
            name_of[isin] = str(row.get("name", "") or "")

    def isin_for(it) -> str:
        # The item's own ISIN is its identity; a symbol root only where exactly
        # one held fund has it (MATE.MI must not borrow a held Nasdaq MATE).
        own = str(it.get("isin") or "").strip().upper()
        if own:
            return own
        for key in (it.get("ticker"), it.get("category")):
            if not key:
                continue
            k = str(key).strip().upper()
            if k in isin_of:
                return isin_of[k]
            owners = root_isins.get(k.split(".")[0], ())
            if len(owners) == 1:
                return next(iter(owners))
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

    def colour(cls: str) -> str:
        # The class hue, as the band and rail beside the same instrument carry it.
        return ASSET_COLORS.get(cls) or PALETTE["accent"]

    out_kept = [_bridge_row(key=r["isin"] or r["label"], label=r["label"],
                            colour=colour(r["cls"]), now=r["now"], target=r["tgt"],
                            base=base, plan_base=plan_base,
                            trend=trend_for(r["item"], r["isin"]))
                for r in kept + new]
    # A sold line keeps its class colour on the slice; its band is the sale's red.
    out_sold = [_bridge_row(key="sell:" + (r["isin"] or r["label"]), label=r["label"],
                            colour=colour(r["cls"]),
                            now=r["now"], target=0.0, base=base,
                            trend=trend_for(r["item"], r["isin"]), legacy=True)
                for r in sold]
    return out_kept[:len(kept)], out_kept[len(kept):], out_sold


def _financing(total_pct: float) -> str:
    """The implicit cash position behind a notional total: −16% for 116%."""
    cash = int(round(100.0 - float(total_pct)))
    if cash == 0:
        return "0%"
    return f"{'\u2212' if cash < 0 else '+'}{abs(cash)}%"


def _stacked_funds(ctx: _NewsletterContext) -> list[str]:
    """The funds that carry more than one euro of exposure per euro invested, as
    the return-stacking literature states them — "NTSG 1.5× (0.9 equities + 0.6
    fixed income)" — held today, in the plan, or both ("plan only" / "today only").

    Today's funds come from the book; the plan's from the per-instrument plan the
    class targets were derived from (``config.target_plan_weights``, a shared
    symbol ISIN-qualified). The two meet on the bare symbol.
    """
    from tarzan import config as taxonomy
    from tarzan.engine.target_derivation import _primary_class

    found: dict[str, dict] = {}

    def add(symbol: str, breakdown: dict, side: str) -> None:
        total = sum(float(v) for v in breakdown.values())
        if total <= 100.5 or not symbol:
            return
        entry = found.setdefault(symbol, {"breakdown": breakdown, "total": total, "sides": set()})
        entry["sides"].add(side)

    hdf = ctx.metrics.holdings_df
    if hdf is not None and not hdf.empty:
        for _, h in hdf.iterrows():
            symbol = _display_ticker(str(h.get("ticker") or "")) or ""
            bd = taxonomy.class_breakdown_for(h.get("isin"), h.get("ticker"), h.get("asset_class")) or {}
            add(symbol, bd, "today")
    for key, weight in (getattr(ctx.config, "target_plan_weights", None) or {}).items():
        if float(weight or 0.0) <= 0:
            continue
        bd = taxonomy.class_breakdown_for(None, key, _primary_class(key)) or {}
        add(_display_ticker(str(key)) or "", bd, "plan")

    out = []
    # Held funds first (both sides, then today only), the plan's newcomers last;
    # the largest stack first within each.
    def order(kv):
        sides = kv[1]["sides"]
        rank = 0 if len(sides) == 2 else (1 if "today" in sides else 2)
        return (rank, -kv[1]["total"], kv[0])

    for symbol, e in sorted(found.items(), key=order):
        parts = [(str(getattr(k, "value", k)).lower(), float(v)) for k, v in e["breakdown"].items() if float(v) > 0]
        if len(parts) == 1:
            body = parts[0][0]
        else:
            body = " + ".join(f"{v / 100:g} {k}" for k, v in parts)
        side = "" if len(e["sides"]) == 2 else (", plan only" if "plan" in e["sides"] else ", today only")
        out.append(f"{symbol} {e['total'] / 100:g}\u00d7 ({body}{side})")
    return out


def _build_diversification(ctx: _NewsletterContext) -> dict:
    """Pre-render the Allocation section: three bridge cards (asset class, equity
    geography, per-holding targets), each with its figures built in, their columns
    shared so they line up down the section.

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
    # The PLAN side of a sleeve is the plan's sleeve, not today's: a region's plan
    # weight times TODAY's equity sleeve is not a figure the plan contains, and the
    # plan column then did not add up to the asset-class card's own Equities plan.
    class_targets = cfg.invested_allocation_targets_pctg or {}

    def plan_sleeve_eur(klass: str, today: float) -> float:
        t = class_targets.get(klass)
        return today if t is None else float(t) / 100.0 * invested_base

    equity_plan_base = plan_sleeve_eur("Equities", equity_base)
    fi_plan_base = plan_sleeve_eur("Fixed Income", fi_base)

    tl = m.allocation_timeline or {}
    dates = tl.get("dates") or []
    asset_series = _recent_timeline(tl.get("asset"), dates)
    geo_series = _recent_timeline(tl.get("geo"), dates)
    hold_series = _recent_timeline(tl.get("holding"), dates)
    hold_inv_series = _recent_timeline(tl.get("holding_invested"), dates)

    abs_pp = float(getattr(cfg, "allocation_band_abs_pp", 5.0))
    rel = float(getattr(cfg, "allocation_band_rel_pctg", 25.0))
    specs: list[dict] = []

    # ── Asset class ──
    asset_rows, cash = [], None
    for r in alloc.get("rows") or []:
        if r.get("is_eur"):
            cash = r
            continue
        asset_rows.append(_bridge_row(
            key=r["name"], label=_ASSET_SHORT.get(r["name"], r["name"]),
            colour=r.get("color") or PALETTE["accent"], now=r.get("actual_pct_raw") or 0.0,
            target=r.get("target_left"), base=invested_base,
            trend=_timeline_vals(asset_series, r["name"])))
    if asset_rows:
        tn = sum(r["now"] for r in asset_rows)
        tt = sum(r["target"] or 0.0 for r in asset_rows)
        ttrend = ([sum(float(x) for x in b.values()) for b in asset_series]
                  if asset_series and len(asset_series) >= 2 else None)
        tail = [{"name": "Total", "trend": ttrend, "move": _move_text(ttrend),
                 "now_pc": _pc1(tn), "now_eur": _eur_smart(tn / 100 * invested_base),
                 "plan_pc": _pc1(tt), "plan_eur": _eur_smart(tt / 100 * invested_base),
                 "gap": _signed_pp(tn - tt), "colour": _band_colour(tn - tt, tt, cfg)}]
        if cash is not None:
            # The strategic reserve is outside the invested allocation base.
            # It may contain securities, not just settlement cash. Its row
            # states the amounts and their gap in euros, banded at
            # ±allocation_band_rel_pctg of its target.
            c_now = float(cash.get("cash_actual_eur") or 0.0)
            c_tgt = float(cash.get("cash_target_eur") or 0.0)
            c_ok = abs(c_now - c_tgt) <= rel / 100.0 * c_tgt + 1e-9
            # To the euro: rounded to thousands, both sides printed the same
            # figure and the gap beside them read as a contradiction.
            tail.append({"name": "Strategic reserve", "move": "", "now_pc": "",
                         "now_eur": _eur(c_now, 0), "plan_pc": "",
                         "plan_eur": _eur(c_tgt, 0), "gap": _signed_eur(c_now - c_tgt),
                         "colour": P["green"] if c_ok else P["red"]})
        levered = max(tn, tt) > 100.5
        stacks = _stacked_funds(ctx)
        note = [f"Notional, % of invested capital ({_eur_smart(invested_base)})."]
        if levered:
            # The return-stacking statement of leverage: gross exposure per euro
            # of capital, the excess as an implicit negative cash position
            # financed at the cash rate, and the funds that stack it. Capital is
            # attributed to no class, so there is no per-class ratio to print.
            note.append(f"Totals past 100% are leverage: {tn / 100:.2f}\u00d7 today, "
                        f"{tt / 100:.2f}\u00d7 plan; the excess is implicit cash borrowed at "
                        f"the cash rate ({_financing(tn)} today, {_financing(tt)} plan).")
        if stacks:
            note.append(f"Stacked funds: {', '.join(stacks)}.")
        if cash is not None:
            note.append("Strategic reserve sits outside this base; not bank cash available for trades.")
        specs.append({"title": "Asset class", "left": asset_rows, "right": asset_rows,
                      "tail": tail, "two": True, "note": " ".join(note),
                      "aria": "Asset class allocation, today against the plan"})

    # ── Equity geography ──
    geo_rows = [_bridge_row(
        key=r["name"], label=_GEO_SHORT.get(r["name"], geo_label(r["name"])),
        colour=_lift(r.get("color")), now=r.get("actual_pct_raw") or 0.0,
        target=r.get("target_left"), base=equity_base, plan_base=equity_plan_base,
        trend=_timeline_vals(geo_series, r["name"])) for r in geo.get("rows") or []]
    if geo_rows:
        specs.append({"title": "Equity geography", "left": geo_rows, "right": geo_rows,
                      "two": True,
                      "note": _sleeve_note("the equity sleeve", equity_base,
                                           equity_plan_base, ", so each side totals 100%."),
                      "aria": "Equity geography, today against the plan"})

    # ── Per-holding targets ──
    verifs = {v.get("kind"): v for v in (m.rebalancing_verifications or [])}
    holding_blocks = []
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
            holding_blocks.append((
                "Per-holding target", "Market value, % of invested capital.",
                _holding_bridge_rows(ctx, v.get("items") or [], base=invested_base,
                                     weights=weights, series=hold_inv_series,
                                     series_key="isin")))
    else:
        for kind, title, base, plan_base in (
                ("per_holding_equity", "Equities holding", equity_base, equity_plan_base),
                ("per_holding_fi", "Fixed income holding", fi_base, fi_plan_base)):
            v = verifs.get(kind)
            if v and v.get("items"):
                holding_blocks.append((
                    title, _sleeve_note("the sleeve", base, plan_base),
                    _holding_bridge_rows(ctx, v.get("items") or [], base=base,
                                         plan_base=plan_base,
                                         weights=None, series=hold_series,
                                         series_key="ticker")))
    for title, note, (kept, new, sold) in holding_blocks:
        if not (kept or new or sold):
            continue
        tail = []
        if sold:
            # The lines the plan sells, summed: what the rebalance raises.
            s_now = sum(r["now"] for r in sold)
            tail.append({"name": "To sell", "move": "",
                         "now_pc": _pc1(s_now), "now_eur": _eur_smart(sum(r["eur_now"] for r in sold)),
                         "plan_pc": "0.0%", "plan_eur": "\u20ac0",
                         "gap": _signed_pp(s_now), "colour": P["red"]})
        specs.append({"title": title, "left": kept + sold, "right": kept + new, "tail": tail,
                      "two": False, "note": note, "aria": f"{title}, today against the plan"})

    legend = (f"1M: weight change over the month. Gap: now \u2212 plan, in pts; green "
              f"within \u00b1{abs_pp:g} pts or \u00b1{rel:g}% of the target, whichever is narrower.")
    html: list[str] = []
    for i, sp in enumerate(specs):
        # Class and region names take two label lines at phone width; tickers one.
        html.append(_bridge_title(sp["title"]))
        html.append(_bridge_card(_bridge_svg(
            sp["left"], sp["right"], cfg, two=sp["two"], tail=sp.get("tail") or (),
            aria=sp["aria"])))
        # The first block's caption carries the legend every block shares: the
        # reader meets the red/green gaps there first.
        text = f'{sp["note"]} {legend}' if i == 0 else sp["note"]
        html.append(_alloc_note(text))
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
    _tax = ctx.reference.taxonomy

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
                                        h.get("name", ""), 40, names=ctx.reference.names),
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

def _class_of_action(isin, ticker, taxonomy) -> str:
    """The asset class of a trade's instrument from the curated taxonomy, for its
    rail colour; "" when the catalog does not know it (the rail is then a rule)."""
    try:
        from tarzan import config as cfg
        hit = cfg.catalog_lookup(taxonomy, isin, ticker)
        # The taxonomy map holds ``(asset_class, role)`` per instrument; a
        # hand-built map may hold a dict. Read either.
        if isinstance(hit, tuple):
            ac = hit[0] if hit else None
        else:
            ac = (hit or {}).get("asset_class") if hasattr(hit, "get") else None
        return class_key(ac) if ac else ""
    except Exception:  # noqa: BLE001 — a rail colour, never a reason to fail the issue
        return ""


def _trade_rows_html(actions: list) -> str:
    """The plan's trades as statement rows: BUY/SELL, ticker and name with the
    signed amount at the right; beneath, Now → After · Target as % over €. Each
    row carries the 2px rail of its instrument's class."""
    P = PALETTE
    def _pe(pair, colour):
        pct, eur = pair
        a = _pct(pct, decimals=1) if pct is not None else "\u2014"
        b = _eur_smart(eur) if eur is not None else ""
        return (f'<b style="color:{colour};">{a}</b>'
                + (f' <span style="color:{P["muted"]};">{b}</span>' if b else ""))
    rows = []
    for a in actions:
        rail = ASSET_COLORS.get(a.get("_class") or "", P["border"])
        rows.append(
            f'<div style="margin-top:6px;border-left:2px solid {rail};padding:0 0 4px 6px;'
            f'border-bottom:1px solid {P["row_rule"]};">'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="table-layout:fixed;"><colgroup><col><col style="width:64px;"></colgroup><tr>'
            f'<td style="padding:0;vertical-align:baseline;overflow:hidden;">{a["name_html"]}</td>'
            f'<td align="right" style="padding:0;vertical-align:baseline;{TYPE["data"]}font-weight:700;'
            f'color:{a["_dir_color"]};white-space:nowrap;font-variant-numeric:tabular-nums;">'
            f'{a["_trade"]}</td></tr></table>'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
            f'style="margin-top:2px;table-layout:fixed;{TYPE["data"]}font-weight:400;color:{P["muted"]};'
            f'font-variant-numeric:tabular-nums;"><colgroup><col style="width:33%;"><col style="width:37%;">'
            f'<col style="width:30%;"></colgroup><tr>'
            f'<td style="padding:0;white-space:nowrap;">now {_pe(a["_now"], P["ink"])}</td>'
            f'<td style="padding:0;white-space:nowrap;">\u2192 after {_pe(a["_after"], P["ink"])}</td>'
            f'<td align="right" style="padding:0;white-space:nowrap;">target {_pe(a["_target"], P["ink"])}</td>'
            f'</tr></table></div>')
    return f'<div style="margin-top:6px;">{"".join(rows)}</div>'


def _optimizer_plan_ctx(m: PortfolioMetrics, suggestions: list, taxonomy=None,
                        reference=None) -> dict:
    """Build one optimizer plan's render context (actions + totals) from a
    list of rebalancing suggestions.

    Flat, largest trade first, like the concept. The other instrument tables
    group by asset class because the reader is asking a question about the
    shape of the portfolio; here the question is "what do I do, and does it
    matter", which the trade size answers and the class headers only
    interrupted -- six header rows above nine trades.
    """
    df = m.holdings_df
    if reference is None:
        from tarzan.contracts.reference import reference_for
        reference = reference_for(m)
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
        return (f'<span data-tarzan-funding-action="true" '
                f'style="display:inline-block;padding:1px 5px;'
                f'background:{PALETTE["shade"]};'
                f'color:{c};border-radius:2px;letter-spacing:0.06em;'
                f'font-weight:700;font-size:9.5px;'
                f'vertical-align:middle;'
                f'margin-right:6px;">{_esc(direction)}</span>')

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
            "_class": _class_of_action(isin, ticker, taxonomy),
            "_trade": _eur_smart(signed_amount, signed=True),
            "_dir_color": dir_color,
            "_now": (s.get("current_pct"), s.get("current_eur")),
            "_after": (s.get("after_pct"), s.get("after_eur")),
            "_target": (s.get("target_pct"), s.get("target_eur")),
            # Name cell: action pill + ticker pin + shortened name, via the
            # shared uni_name so it matches every other table.
            "name_html": uni_name(
                display_instrument_name(
                    isin, ticker, s.get("name", ""), names=reference.names), tk,
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
    table_html = _trade_rows_html(actions)

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
    _tax = ctx.reference.taxonomy
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
        status, infeasible, aligned = _rebalance_evidence(verifications, pc["n_total"])
        pairs = [
            ("Bank cash", "initial_settlement_cash_eur", "ending_settlement_cash_eur"),
            ("Cash-equivalent securities", "cash_equivalent_securities_before_eur",
             "cash_equivalent_securities_after_eur"),
            ("Strategic reserve (Cash & Cash Equivalents)",
             "initial_cash_equivalents_eur", "ending_cash_equivalents_eur"),
        ]
        singles = [
            ("Strategic reserve target", "cash_equivalents_target_eur"),
            ("Protected bank cash", "protected_cash_eur"),
            ("External contribution", "external_contribution_eur"),
            ("Gross security sales", "gross_sales_eur"),
            ("Gross security purchases", "gross_purchases_eur"),
            ("Estimated sale tax", "estimated_tax_eur"),
            ("Estimated fees", "fees_eur"),
            ("Bank cash above protected amount", "residual_eur"),
        ]
        if funding is not None and (
            "locked_cash_balances_satisfied" in funding
            or any(key in funding for key in _LOCKED_FUNDING_FIELDS)
        ):
            pairs.append(("Free bank cash", "initial_free_settlement_cash_eur",
                          "ending_free_settlement_cash_eur"))
            singles.insert(0, ("Locked bank cash", "locked_settlement_cash_eur"))
        pc.update({
            "execution_status": status,
            "executable": status == "EXECUTABLE",
            "funding_available": funding is not None,
            "infeasible": infeasible,
            "aligned": aligned,
            "funding_pairs": [
                {"label": label, "before": _funding_cell(funding, before),
                 "after": _funding_cell(funding, after)}
                for label, before, after in pairs
            ],
            "funding_details": [
                {"label": label, "amount": _funding_cell(funding, key)}
                for label, key in singles
            ],
        })

    if plans_src:
        plans = []
        for p in plans_src:
            pc = _optimizer_plan_ctx(
                m, list(p.get("suggestions") or []), _tax, ctx.reference)
            # Strip a caller's Markup subclass: a plan label is untrusted text,
            # not HTML, even when it was marked safe upstream.
            label = p.get("label")
            pc["label"] = str(label) if label is not None else ""
            pc["no_sell"] = p.get("no_sell")
            # Which plan the configuration actually runs. Both are computed
            # every run and both can be executable, so without this the reader
            # has two trade lists and no way to tell which one is the proposal.
            pc["active"] = bool(p.get("no_sell")) == bool(
                getattr(ctx.config, "rebalancing_no_sell", False))
            _attach_cost(pc, p)
            _attach_execution(pc, p.get("verifications"))
            plans.append(pc)
        return {"available": True, "plans": plans,
                "subtitle": _optimizer_subtitle(plans)}

    # Back-compat: single plan.
    suggestions = list(m.rebalancing_suggestions or [])
    if not suggestions and not m.rebalancing_verifications:
        return {"available": False}
    pc = _optimizer_plan_ctx(m, suggestions, _tax, ctx.reference)
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
    checked = sum(plan.get("executable") is True for plan in plans)
    noun = "modeled proposal" if n == 1 else "modeled proposals"
    return (
        f"{n} {noun}; {checked} passed funding, position and policy checks at "
        "selected prices. ACTIVE marks the configured proposal, not a broker order."
    )


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
    status, infeasible, aligned = _rebalance_evidence(
        m.rebalancing_verifications, n_actions)
    parts = [f"Portfolio at {hero['total_value']} ({hero['gain_pct']} since inception)"]
    if status == "NON_EXECUTABLE":
        parts.append("rebalance draft not executable")
    elif infeasible:
        parts.append("no feasible rebalance")
    elif status != "EXECUTABLE":
        parts.append("rebalance draft, funding unverified")
    elif n_actions > 0:
        parts.append("modeled rebalance")
    elif aligned:
        parts.append("checked targets within tolerance")
    else:
        parts.append("no trades proposed, target alignment unproven")
    parts.append(f"{len(m.holdings_df)} holdings tracked")
    return " · ".join(parts)

