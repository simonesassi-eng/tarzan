"""Every return the issue prints, checked against a second, independent source.

The engine prices off Yahoo, so before this nothing in a run could tell a Yahoo error
from a market move: the oracle proved Tarzan's arithmetic matched Yahoo, never that Yahoo
was right. Here each holding's figures are compared with justETF (FactSet/Xetra) and a
figure the two sources disagree on is MARKED in the issue as not verified. Nothing is
replaced: the referee decides whether a number is corroborated, it never supplies one.

Two parts, which together cover the whole printed span:

* 5D -> 5Y are compared up to the last close BOTH sources carry (yesterday, mid-session):
  justETF's daily series ends there, and comparing a live endpoint with a closed one
  manufactures a gap that is the clock. Each window is anchored by ``window_anchor``, the
  one authority on window edges, on the engine's own tape, and justETF is read at that
  same date.
* 1D is the engine's live move against justETF's quote for the same session.

The threshold is measured, not chosen. Across 96 window comparisons on the reference book
the honest gap between the sources had a median of 0.2-0.4pp and grows with the size of
the return (venue noise at both ends is multiplicative): the largest was 1.20 x (1 + |r|).
A figure is DIVERGED beyond 1.5 x (1 + |r|) pp. The one fund that crossed it, by 2.7-4.0,
was the thin line whose Yahoo prices had already been shown wrong at both ends of its 1D.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from tarzan.data.referee import (
    CURRENT_SESSION_BASELINE_ATTR, finite_number, observation_date, quote_pair,
)

#: The long windows checked against the second source's daily series.
LONG_WINDOWS = ("5d", "1m", "3m", "ytd", "1y", "3y", "5y")

#: Diverged beyond this many pp per unit of (1 + |r|). See the module docstring.
_RELATIVE_TOLERANCE_PP = 1.5

#: The 1D allowance: minutes separate the two quotes, so it scales with the instrument's
#: own daily volatility, floored at 1pp — the same rule as the oracle's intraday check.
_ONE_DAY_FLOOR_PP = 1.0
_ONE_DAY_SIGMAS = 1.5

OK, DIVERGED, UNAVAILABLE = "ok", "diverged", "na"
#: A disagreement settled by a third source in the engine's favour: printed as is.
CONFIRMED = "confirmed"
#: A disagreement settled AGAINST the engine: the printed figure is replaced.
CORRECTED = "corrected"


def _naive_days(s: pd.Series) -> pd.Series:
    """Valid positive closes on their native dates, preserving listing metadata."""
    empty = pd.Series(dtype=float)
    if not isinstance(s, pd.Series) or pd.api.types.is_bool_dtype(s.dtype):
        return empty
    if pd.api.types.is_complex_dtype(s.dtype):
        return empty
    try:
        if not isinstance(s.index, pd.DatetimeIndex) and pd.api.types.is_numeric_dtype(s.index):
            return empty  # an integer row number is not a dated market observation
        idx = (
            s.index if isinstance(s.index, pd.DatetimeIndex)
            else pd.DatetimeIndex([observation_date(value) for value in s.index])
        )
        values = s
        if pd.api.types.is_object_dtype(s.dtype):
            invalid_scalars = s.map(
                lambda value: pd.api.types.is_bool(value)
                or isinstance(value, (complex, np.complexfloating))
            )
            values = s.mask(invalid_scalars)
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError):
        return empty
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    valid = np.isfinite(numbers) & (numbers > 0) & ~idx.isna()
    out = pd.Series(numbers[valid], index=idx[valid].normalize(), name=s.name)
    out.attrs = dict(s.attrs)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _anchor(tape: pd.Series, bucket: str, ticker: Optional[str]):
    if bucket == "ytd":
        prior = tape[tape.index.year < tape.index[-1].year]
        return prior.index[-1] if len(prior) else None
    from tarzan.engine.stats import window_anchor

    return window_anchor(tape, bucket, ticker)


#: Extra allowance (pp) when the second source is the issuer's NAV rather than a market
#: price. justETF serves a NAV for a fund that does not trade on Xetra, and a market
#: return differs from a NAV return by the change in the premium/discount, which this
#: comparison cannot separate from an error: one managed-futures fund's 1Y differed by
#: 2.4pp on that account alone. Against a NAV the check can only catch a GROSS error.
NAV_PREMIUM_ALLOWANCE_PP = 3.0


def check_windows(tape: pd.Series, ref: Optional[pd.Series],
                  ticker: Optional[str] = None, extra_pp: float = 0.0) -> dict:
    """``{bucket: {"status", "alt", "ours", "at"}}`` for every long window.

    ``ours`` and ``alt`` are the two sources' returns (%) over the SAME span, ending on
    ``at`` — the last close both carry. ``na`` when there is no second source, no common
    close, or the second source does not reach back to the window's anchor.
    """
    out = {w: {"status": UNAVAILABLE, "alt": None, "ours": None, "at": None}
           for w in LONG_WINDOWS}
    allowance = finite_number(extra_pp)
    if tape is None or ref is None or allowance is None or allowance < 0:
        return out
    t, r = _naive_days(tape), _naive_days(ref)
    common = t.index.intersection(r.index)
    if not len(common):
        return out
    end = common[-1]
    t, r = t[t.index <= end], r[r.index <= end]
    for w in LONG_WINDOWS:
        a = _anchor(t, w, ticker)
        if a is None:
            continue
        ref_at = r[r.index <= a]
        if not len(ref_at) or r.index[0] > a + pd.Timedelta(days=7):
            continue
        ours = finite_number((float(t.iloc[-1]) / float(t.loc[a]) - 1.0) * 100.0)
        alt = finite_number((float(r.iloc[-1]) / float(ref_at.iloc[-1]) - 1.0) * 100.0)
        if ours is None or alt is None:
            continue
        limit = _RELATIVE_TOLERANCE_PP * (1.0 + abs(ours) / 100.0) + allowance
        out[w] = {"status": DIVERGED if abs(ours - alt) > limit else OK,
                  "alt": alt, "ours": ours, "at": end.date(), "anchor": a,
                  "alt_anchor": ref_at.index[-1]}
    return out


def _native_baseline_at(tape: pd.Series, day: _dt.date) -> bool:
    provenance = tape.attrs.get(CURRENT_SESSION_BASELINE_ATTR)
    return not (
        isinstance(provenance, Mapping)
        and observation_date(provenance.get("anchor")) == day
        and observation_date(provenance.get("observation_date")) != day
    )


def qualify_sibling(
    tape: pd.Series, sibling: pd.Series, *, ticker: str, symbol: str, isin: str,
    primary_identity=None, sibling_identity=None, ambiguous_signatures=(),
    verified_alias: bool = False, verified_currency: Optional[str] = None,
    at=None, level_tolerance: float = 0.10,
) -> Optional[pd.Series]:
    """Admit a native-EUR vote for a qualified listing over a common close.

    Same-root guesses need the exact full-name/type/currency signature already
    received for both listings. Different roots need a causal verified alias.
    This function is pure: it never discovers an identity or fetches a quote.
    """
    from tarzan.data.listing_identity import QuoteIdentity, qualified_symbol, relative_signature

    primary, candidate = qualified_symbol(ticker), qualified_symbol(symbol)
    if not primary or not candidate or primary == candidate:
        return None
    if not isinstance(isin, str) or not isin.strip():
        return None
    expected = isin.strip().upper()
    same_root = primary.rsplit(".", 1)[0] == candidate.rsplit(".", 1)[0]
    trusted_alias = verified_alias is True
    if not same_root and not trusted_alias:
        return None
    for identity, requested in ((primary_identity, primary), (sibling_identity, candidate)):
        if identity is None:
            continue
        if (
            not isinstance(identity, QuoteIdentity)
            or identity.symbol != requested
            or identity.malformed or identity.conflicting
            or identity.isin is not None and identity.isin != expected
            or identity.currency is not None and identity.currency != "EUR"
        ):
            return None
    if trusted_alias:
        if verified_currency != "EUR":
            return None
        # The verified catalog names an ETF share class; an explicit contrary
        # native type cannot be overridden by its cached alias authority.
        if any(
            identity is not None and identity.quote_type not in (None, "ETF")
            for identity in (primary_identity, sibling_identity)
        ):
            return None
        basis = "causal_verified_alias"
    else:
        signature = relative_signature(primary_identity)
        if (
            signature is None
            or signature != relative_signature(sibling_identity)
            or signature in (ambiguous_signatures or ())
        ):
            return None
        basis = "relative_quote_signature"
    day, tolerance = observation_date(at), finite_number(level_tolerance)
    if day is None or tolerance is None or tolerance < 0:
        return None
    own, other = _naive_days(tape), _naive_days(sibling)
    end = pd.Timestamp(day)
    if end not in own.index or end not in other.index:
        return None
    if not _native_baseline_at(own, day) or not _native_baseline_at(other, day):
        return None
    ratio = finite_number(float(other.loc[end]) / float(own.loc[end]))
    if ratio is None or abs(ratio - 1.0) > tolerance:
        return None
    out = other.copy()
    out.attrs["third_source_listing"] = {
        "primary_symbol": primary, "symbol": candidate, "requested_isin": expected,
        "identity_basis": basis, "currency": "EUR", "common_close": day.isoformat(),
    }
    return out


def _return_between(series: pd.Series, anchor, end) -> Optional[float]:
    s = _naive_days(series)
    anchor_date, end_date = observation_date(anchor), observation_date(end)
    if anchor_date is None or end_date is None or anchor_date >= end_date:
        return None
    a, e = pd.Timestamp(anchor_date), pd.Timestamp(end_date)
    if a not in s.index or e not in s.index:
        return None
    if not _native_baseline_at(s, anchor_date) or not _native_baseline_at(s, end_date):
        return None
    return finite_number((float(s.loc[e]) / float(s.loc[a]) - 1.0) * 100.0)


def settle(verdict: dict, tape: pd.Series, third_sources) -> dict:
    """Settle a DIVERGED window with third sources, by majority. Returns the verdict
    updated: ``confirmed`` (the engine agrees with a third source), ``corrected`` with
    ``corrected_value`` (the two outside sources agree with each other and not with the
    engine), or still ``diverged`` (nobody agrees: no right number exists to print).

    The third sources are the same instrument on OTHER venues, read over the very span
    the verdict was reached on (same anchor, same last close). A correction keeps the
    engine's own move from that close to today, so only the disputed part is replaced:
    ``(1 + agreed) x (tape today / tape at the shared close) - 1``.
    """
    if verdict.get("status") != DIVERGED:
        return verdict
    ours, alt, anchor, at = (verdict.get(k) for k in ("ours", "alt", "anchor", "at"))
    ours, alt = finite_number(ours), finite_number(alt)
    at_date = observation_date(at)
    anchor_date = observation_date(anchor)
    if ours is None or alt is None or anchor_date is None or at_date is None:
        return verdict
    # A source's last observation before an anchor is not that anchor. Such a
    # second-source comparison can remain a warning, but cannot justify replacing
    # or confirming a return over a different interval.
    if "alt_anchor" in verdict and observation_date(verdict["alt_anchor"]) != anchor_date:
        return dict(verdict, third=[])
    t = _naive_days(tape)
    if (
        pd.Timestamp(anchor_date) not in t.index or pd.Timestamp(at_date) not in t.index
        or not _native_baseline_at(t, anchor_date) or not _native_baseline_at(t, at_date)
    ):
        return dict(verdict, third=[])
    limit = _RELATIVE_TOLERANCE_PP * (1.0 + abs(ours) / 100.0)
    votes = [v for v in (_return_between(src, anchor, at) for src in third_sources or ())
             if v is not None]
    out = dict(verdict, third=votes)
    if any(abs(v - ours) <= limit for v in votes):
        out["status"] = CONFIRMED
        return out
    agreeing = [v for v in votes if abs(v - alt) <= limit]
    if not agreeing:
        return out
    count = 1 + len(agreeing)
    # Keep the exact existing arithmetic for ordinary prices. Only an overflowing
    # sum needs the numerically safe alternative; normal financial rounding is not
    # a reason to change a previously computed return.
    agreed = finite_number((alt + sum(agreeing)) / count)
    if agreed is None:
        agreed = finite_number(alt / count + sum(v / count for v in agreeing))
    if agreed is None:
        return out
    shared = t[t.index <= pd.Timestamp(at_date)]
    if not len(shared):
        return out
    tail = float(t.iloc[-1]) / float(shared.iloc[-1])
    corrected = finite_number(((1.0 + agreed / 100.0) * tail - 1.0) * 100.0)
    if corrected is None:
        return out
    out.update(status=CORRECTED,
               corrected_value=corrected,
               agreed=agreed)
    return out


def check_one_day(ours_1d, quote: Optional[dict], today: _dt.date,
                  tape: Optional[pd.Series] = None,
                  ticker: Optional[str] = None) -> dict:
    """The live 1D against an independently dated pair over the SAME interval."""
    unavailable = {"status": UNAVAILABLE, "alt": None}
    ours = finite_number(ours_1d)
    day = observation_date(today)
    if ours is None or not quote or day is None:
        return unavailable
    clean = _naive_days(tape) if tape is not None else None
    if clean is not None:
        if clean.empty or clean.index[-1].date() != day:
            return unavailable
        from tarzan.engine.stats import window_anchor

        anchor = window_anchor(clean, "1d", ticker)
        if anchor is None:
            return unavailable
        prior = observation_date(anchor)
        # A sibling's native previous close may have been written on the canonical
        # venue's row. An equal row date alone must not make unequal intervals agree.
        provenance = clean.attrs.get(CURRENT_SESSION_BASELINE_ATTR)
        if isinstance(provenance, dict) and observation_date(provenance.get("at")) == day:
            if (
                observation_date(provenance.get("anchor")) != prior
                or observation_date(provenance.get("observation_date")) != prior
            ):
                return unavailable
    else:
        # Compatibility callers without a tape must still name a known venue. A
        # missing previous observation is never licensed by the run's current time.
        from tarzan.data.exchange_calendar import covers, previous_session

        source_ticker = ticker or {"XETRA": "REFEREE.DE", "XETR": "REFEREE.DE"}.get(
            str(quote.get("venue") or "").strip().upper()
        )
        if not source_ticker or not covers(source_ticker, day):
            return unavailable
        prior = previous_session(source_ticker, day)
    pair = quote_pair(quote, session=day, anchor=prior)
    if pair is None:
        return unavailable
    alt = finite_number((pair[0] / pair[1] - 1.0) * 100.0)
    if alt is None:
        return unavailable
    sigma_pp = None
    if clean is not None:
        r = clean.pct_change().dropna().tail(252)
        r = r[np.isfinite(r)]
        if len(r) >= 20:
            sigma_pp = finite_number(float(r.std()) * 100.0)
    allowance = max(_ONE_DAY_FLOOR_PP, _ONE_DAY_SIGMAS * sigma_pp if sigma_pp else 0.0)
    return {"status": DIVERGED if abs(ours - alt) > allowance else OK,
            "alt": alt}
