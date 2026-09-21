"""A daily bar's date is its VENUE's session date, not an instant.

Converting to UTC before dropping the timezone slides every European series one
session into the past: a Milan bar stamped 2026-09-21 00:00+02:00 becomes
2026-09-20 22:00 UTC and normalizes to the 20th. Four call sites each re-derived
this rule and three got it wrong, so the proxy series the backtest splices were
dated a day before the funds' own real returns — attenuating every cross-sleeve
correlation and therefore overstating diversification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.stats import normalize_index, normalize_session_index


def _milan_bars(n=5, start="2026-09-15"):
    """Daily bars stamped midnight Europe/Rome, as yfinance returns them."""
    idx = pd.date_range(start, periods=n, freq="B", tz="Europe/Rome")
    return pd.Series(np.arange(float(n)), index=idx)


def _new_york_bars(n=5, start="2026-09-15"):
    idx = pd.date_range(start, periods=n, freq="B", tz="America/New_York")
    return pd.Series(np.arange(float(n)), index=idx)


def test_european_session_date_survives():
    """The failure that started this: a UTC trip moves the date back a day."""
    s = _milan_bars()
    out = normalize_session_index(s.index)
    assert [d.date().isoformat() for d in out] == [
        d.date().isoformat() for d in s.index], "session date must not move"
    # And the wrong way round, to pin what we are guarding against.
    wrong = s.index.tz_convert("UTC").tz_localize(None).normalize()
    assert wrong[0].date() < s.index[0].date(), "fixture must reproduce the slide"


def test_us_session_date_also_survives():
    """A midnight-04:00 stamp survives a UTC trip by luck, so it must not be the
    only venue the rule is tested on."""
    s = _new_york_bars()
    out = normalize_session_index(s.index)
    assert [d.date() for d in out] == [d.date() for d in s.index]


def test_venues_in_different_zones_align_on_the_same_day():
    """The whole point: Milan and New York bars for the same session must land
    on one key, or a correlation between them is measured across a day."""
    mi = normalize_session_index(_milan_bars().index)
    ny = normalize_session_index(_new_york_bars().index)
    assert list(mi) == list(ny)


def test_naive_index_is_untouched():
    idx = pd.date_range("2026-09-15", periods=4, freq="B")
    assert list(normalize_session_index(idx)) == list(idx)


def test_intraday_stamps_collapse_to_their_own_day():
    idx = pd.DatetimeIndex(["2026-09-21 09:05", "2026-09-21 17:30"], tz="Europe/Rome")
    out = normalize_session_index(idx)
    assert out[0] == out[1] == pd.Timestamp("2026-09-21")


def test_idempotent():
    once = normalize_session_index(_milan_bars().index)
    assert list(normalize_session_index(once)) == list(once)


def test_normalize_index_delegates_and_keeps_values():
    """normalize_index is the Series-level wrapper; it must not diverge."""
    s = _milan_bars()
    out = normalize_index(s)
    assert list(out.index) == list(normalize_session_index(s.index))
    assert list(out.values) == list(s.values)


def test_drop_duplicates_keeps_the_last_observation():
    idx = pd.DatetimeIndex(["2026-09-21 09:05", "2026-09-21 17:30"], tz="Europe/Rome")
    s = pd.Series([1.0, 2.0], index=idx)
    out = normalize_index(s, drop_duplicates=True)
    assert len(out) == 1 and out.iloc[0] == 2.0


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
