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
from typing import Optional

import pandas as pd

#: The long windows checked against the second source's daily series.
LONG_WINDOWS = ("5d", "1m", "3m", "ytd", "1y", "3y", "5y")

#: Diverged beyond this many pp per unit of (1 + |r|). See the module docstring.
_RELATIVE_TOLERANCE_PP = 1.5

#: The 1D allowance: minutes separate the two quotes, so it scales with the instrument's
#: own daily volatility, floored at 1pp — the same rule as the oracle's intraday check.
_ONE_DAY_FLOOR_PP = 1.0
_ONE_DAY_SIGMAS = 1.5

OK, DIVERGED, UNAVAILABLE = "ok", "diverged", "na"


def _naive_days(s: pd.Series) -> pd.Series:
    s = s.dropna()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(s.values.astype(float), index=idx.normalize())
    return out[~out.index.duplicated(keep="last")].sort_index()


def _anchor(tape: pd.Series, bucket: str, ticker: Optional[str]):
    if bucket == "ytd":
        prior = tape[tape.index.year < tape.index[-1].year]
        return prior.index[-1] if len(prior) else None
    from tarzan.engine.stats import window_anchor

    return window_anchor(tape, bucket, ticker)


def check_windows(tape: pd.Series, ref: Optional[pd.Series],
                  ticker: Optional[str] = None) -> dict:
    """``{bucket: {"status", "alt", "ours", "at"}}`` for every long window.

    ``ours`` and ``alt`` are the two sources' returns (%) over the SAME span, ending on
    ``at`` — the last close both carry. ``na`` when there is no second source, no common
    close, or the second source does not reach back to the window's anchor.
    """
    out = {w: {"status": UNAVAILABLE, "alt": None, "ours": None, "at": None}
           for w in LONG_WINDOWS}
    if tape is None or ref is None:
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
        ours = (t.iloc[-1] / t.loc[a] - 1.0) * 100.0
        alt = (r.iloc[-1] / ref_at.iloc[-1] - 1.0) * 100.0
        limit = _RELATIVE_TOLERANCE_PP * (1.0 + abs(ours) / 100.0)
        out[w] = {"status": DIVERGED if abs(ours - alt) > limit else OK,
                  "alt": alt, "ours": ours, "at": end.date()}
    return out


def check_one_day(ours_1d, quote: Optional[dict], today: _dt.date,
                  tape: Optional[pd.Series] = None) -> dict:
    """The live 1D against the second source's quote for the same session."""
    if ours_1d is None or ours_1d != ours_1d or not quote:
        return {"status": UNAVAILABLE, "alt": None}
    if quote.get("date") != today or not quote.get("prev_close"):
        return {"status": UNAVAILABLE, "alt": None}
    alt = (float(quote["price"]) / float(quote["prev_close"]) - 1.0) * 100.0
    sigma_pp = None
    if tape is not None:
        r = _naive_days(tape).pct_change().dropna().tail(252)
        if len(r) >= 20:
            sigma_pp = float(r.std()) * 100.0
    allowance = max(_ONE_DAY_FLOOR_PP, _ONE_DAY_SIGMAS * sigma_pp if sigma_pp else 0.0)
    return {"status": DIVERGED if abs(float(ours_1d) - alt) > allowance else OK,
            "alt": alt}
