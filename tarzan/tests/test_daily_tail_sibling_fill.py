"""A venue that hasn't printed today's close borrows a sibling's RETURN.

Yahoo publishes some European listings' daily bar late (Borsa Italiana
``.MI`` most often) while the same fund's Xetra/Paris listing already has
today. Left alone, that holding's 1w/1m/YTD, TWROR, XIRR and risk are all
measured to yesterday while every other holding is measured to today.

The borrowed quantity is a RETURN, never a price: both endpoints come from
the sibling, so the venue basis cancels in the division, and the result is
applied to this venue's own last real close. Splicing the sibling's raw
price instead would embed the venue basis as a fake return — measured at
0.31% sd on NTSG against a 0.70% typical daily move, i.e. mostly noise, and
permanently inflating volatility.

Network-free: the sibling fetch seam is monkeypatched.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from tarzan.data import enricher as en


TODAY = datetime.date(2026, 8, 11)      # a Tuesday
YESTERDAY = datetime.date(2026, 8, 10)  # the Monday before it


def _frame(pairs) -> pd.DataFrame:
    """A daily OHLC-ish frame with just the Close column populated."""
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.DataFrame({"Close": [float(v) for _, v in pairs]}, index=idx)


def _series(pairs) -> pd.Series:
    idx = pd.to_datetime([d for d, _ in pairs])
    return pd.Series([float(v) for _, v in pairs], index=idx)


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setattr(en, "_today_local", lambda: TODAY)


def _sibling(monkeypatch, mapping):
    """Serve canned close series for sibling lookups."""
    monkeypatch.setattr(en, "_sibling_close_series",
                        lambda sym: mapping.get(sym))


# --- the core behaviour ----------------------------------------------------

def test_borrows_the_siblings_return_not_its_price(monkeypatch):
    """The real NTSG case from 2026-08-11.

    Milan's last close is 29.8250 (10 Aug) and it has no 11 Aug bar. Xetra
    printed 29.8500 on the 10th and 29.8600 on the 11th.

      Xetra's own move : 29.8600/29.8500 - 1 = +0.033501%
      applied to Milan : 29.8250 * 1.00033501 = 29.834993

    Splicing Xetra's PRICE would instead imply 29.8600/29.8250 - 1 =
    +0.1174% — 0.084pp of which is pure venue basis, not a market move.
    """
    own = _frame([(YESTERDAY, 29.8250)])
    own = pd.concat([_frame([(datetime.date(2026, 8, 7), 29.6950)]), own])
    _sibling(monkeypatch, {"NTSG.DE": _series([(YESTERDAY, 29.8500),
                                              (TODAY, 29.8600)])})

    out = en._fill_today_from_sibling("NTSG.MI", own)
    closes = out["Close"].dropna()
    assert [pd.Timestamp(i).date() for i in closes.index][-1] == TODAY

    sibling_return = 29.8600 / 29.8500 - 1.0
    assert closes.iloc[-1] == pytest.approx(29.8250 * (1 + sibling_return), abs=1e-9)

    # The implied return is the sibling's OWN move, with no venue basis in it.
    implied = closes.iloc[-1] / 29.8250 - 1.0
    assert implied == pytest.approx(sibling_return, abs=1e-12)
    # And it is NOT the price-splice figure.
    assert implied != pytest.approx(29.8600 / 29.8250 - 1.0, abs=1e-9)


def test_records_provenance_for_the_synthesized_close(monkeypatch):
    own = _frame([(datetime.date(2026, 8, 7), 29.6950), (YESTERDAY, 29.8250)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(YESTERDAY, 29.8500),
                                              (TODAY, 29.8600)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)

    key = en._history_timestamp_key(pd.Timestamp(TODAY))
    assert out.attrs[en._HISTORY_ORIGINS_ATTR][key] == \
        en._HISTORY_ORIGIN_SIBLING_RETURN
    note = out.attrs[en._HISTORY_SYNTHETIC_ATTR][key]
    assert note["source"] == "NTSG.DE"
    assert note["own_last_close"] == pytest.approx(29.8250)
    assert note["sibling_return_pct"] == pytest.approx(
        (29.8600 / 29.8500 - 1) * 100, abs=1e-9)


# --- the guards ------------------------------------------------------------

def test_no_fill_when_the_venue_is_already_current(monkeypatch):
    own = _frame([(YESTERDAY, 29.8250), (TODAY, 29.9000)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(YESTERDAY, 29.8500),
                                              (TODAY, 29.8600)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)
    assert out is own, "a venue that printed today must be left untouched"


def test_no_fill_when_more_than_one_session_is_missing(monkeypatch):
    """Several missing sessions is a deeper outage than a late bar; chaining
    borrowed returns across it is not something to do silently."""
    stale = datetime.date(2026, 8, 5)  # the Wednesday — 3 sessions back
    own = _frame([(datetime.date(2026, 8, 4), 29.5), (stale, 29.6050)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(stale, 29.5900),
                                              (TODAY, 29.8600)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)
    assert out is own


def test_no_fill_when_the_sibling_also_lacks_today(monkeypatch):
    own = _frame([(datetime.date(2026, 8, 7), 29.6950), (YESTERDAY, 29.8250)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(datetime.date(2026, 8, 7), 29.77),
                                              (YESTERDAY, 29.8500)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)
    assert out is own


def test_no_fill_when_the_siblings_previous_close_is_a_different_day(monkeypatch):
    """The borrowed return must span exactly the missing step. If the
    sibling's own previous bar isn't on this venue's last close date, the
    return would cover a longer span than the one gap being filled."""
    own = _frame([(datetime.date(2026, 8, 7), 29.6950), (YESTERDAY, 29.8250)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(datetime.date(2026, 8, 6), 29.60),
                                              (TODAY, 29.8600)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)
    assert out is own


def test_no_fill_when_the_sibling_is_a_different_instrument(monkeypatch):
    """Equal ticker root on another exchange is not proof of the same fund.
    A level far off this venue's own close fails the collision guard."""
    own = _frame([(datetime.date(2026, 8, 7), 29.6950), (YESTERDAY, 29.8250)])
    _sibling(monkeypatch, {"NTSG.DE": _series([(YESTERDAY, 250.0),
                                              (TODAY, 251.0)])})
    out = en._fill_today_from_sibling("NTSG.MI", own)
    assert out is own, "a 740%-off level must not be treated as the same fund"


def test_no_fill_without_a_sibling_venue(monkeypatch):
    """Indices, FX, futures and bare tickers have no EUR-venue siblings."""
    own = _frame([(datetime.date(2026, 8, 7), 5000.0), (YESTERDAY, 5010.0)])
    _sibling(monkeypatch, {})
    for sym in ("^GSPC", "EURUSD=X", "CL=F", "BTC-USD", "AAPL"):
        assert en._fill_today_from_sibling(sym, own) is own


def test_a_synthesized_close_is_never_derived_from_another(monkeypatch):
    """The sibling lookup must not route through _fetch_history, or one
    venue's outage would propagate reconstructed prices across every
    listing of the fund (and could recurse).

    Asserted behaviourally: if _sibling_close_series reached _fetch_history,
    this canary would fire.
    """
    called: list[str] = []
    monkeypatch.setattr(en, "_fetch_history",
                        lambda s: called.append(s) or pd.DataFrame())
    monkeypatch.setattr(en.price_cache, "load_history", lambda s: None)
    monkeypatch.setattr(en.price_cache, "refresh_start", lambda c: None)
    monkeypatch.setattr(en, "_retry", lambda fn, what=None: pd.DataFrame())
    monkeypatch.setattr(en.price_cache, "merge_history", lambda a, b: pd.DataFrame())

    en._sibling_close_series("NTSG.DE")
    assert called == [], \
        f"_sibling_close_series reached _fetch_history for {called}"


def test_single_missing_session_detection_skips_weekends():
    """Monday's fill looks back to Friday, not to Sunday."""
    monday = datetime.date(2026, 8, 10)
    friday = datetime.date(2026, 8, 7)
    assert en._is_previous_trading_day("NTSG.MI", friday, monday) is True
    assert en._is_previous_trading_day(
        "NTSG.MI", datetime.date(2026, 8, 6), monday) is False


# --- run-mode and cache invariants ----------------------------------------

def test_pinned_runs_never_fill(monkeypatch, tmp_path):
    """A reproducible/point-in-time run must reflect only what its own venue
    printed. Otherwise the same as_of yields different history depending on
    which venues happened to answer, and the golden/lookahead guarantees go
    with it. _fetch_history gates the fill on `not pinned`."""
    import inspect
    src = inspect.getsource(en._fetch_history)
    fill_line = next(ln for ln in src.splitlines()
                     if "_fill_today_from_sibling" in ln)
    indent = len(fill_line) - len(fill_line.lstrip())
    guard = next(ln for ln in src.splitlines()
                 if "if not pinned" in ln)
    guard_indent = len(guard) - len(guard.lstrip())
    assert guard_indent < indent, \
        "the fill must sit inside the `if not pinned` branch"


def test_the_synthetic_close_is_not_persisted_to_the_cache(monkeypatch):
    """store_history must run BEFORE the fill: only closes the venue really
    printed belong in its own cache. Persisting a reconstruction would let a
    later run read it back as genuine — and then derive another synthetic
    close from it."""
    stored: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(en.price_cache, "store_history",
                        lambda sym, frame: stored.update({sym: frame.copy()}))
    monkeypatch.setattr(en.price_cache, "load_history", lambda s: None)
    monkeypatch.setattr(en.price_cache, "refresh_start", lambda c: None)
    monkeypatch.setattr(en.price_cache, "merge_history", lambda a, b: b)
    monkeypatch.setattr(en.price_cache, "repair_split_jumps", lambda f: f)
    own = _frame([(datetime.date(2026, 8, 7), 29.6950), (YESTERDAY, 29.8250)])
    monkeypatch.setattr(en, "_retry", lambda fn, what=None: own)
    _sibling(monkeypatch, {"NTSG.DE": _series([(YESTERDAY, 29.8500),
                                              (TODAY, 29.8600)])})
    with en._net_lock:
        en._history_memo.pop("NTSG.MI", None)

    out = en._fetch_history("NTSG.MI")

    # The returned frame carries today; the cached frame must not.
    assert [pd.Timestamp(i).date() for i in out["Close"].dropna().index][-1] == TODAY
    cached_dates = [pd.Timestamp(i).date()
                    for i in stored["NTSG.MI"]["Close"].dropna().index]
    assert TODAY not in cached_dates, \
        "a synthesized close must never be written to the venue's own cache"
