"""A continuous instrument (futures/FX) an hour into its ~23-24h trading
window must not render as if the session were almost over: the 6.5h/8.5h
cash-market heuristic in _intraday_spark clamped anything past that to the
right edge, so a market that had just reopened looked nearly complete.
"""

from __future__ import annotations

import re

import pandas as pd

from tarzan.export.newsletter._charts import _intraday_spark


def _bars(n: int, start: str) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.Series([100.0 + i * 0.1 for i in range(n)], index=idx)


def _last_x(svg: str) -> float:
    xs = [float(x) for x in re.findall(r'points="[^"]*?([\d.]+),[\d.]+"', svg)]
    # Fallback: pull every x from the points attribute directly.
    m = re.search(r'points="([^"]+)"', svg)
    pts = [p.split(",") for p in m.group(1).strip().split(" ") if p]
    return max(float(p[0]) for p in pts)


def test_without_session_hours_partial_elapsed_time_clamps_to_full_width():
    # 33 bars (~8h) exceeds the assumed 6.5h cash session -> clamped to the
    # full width, the exact behaviour session_hours exists to fix for a
    # continuous instrument only ~8/23 of the way through its real window.
    svg = _intraday_spark(_bars(33, "2024-01-08 14:00"), 100.0, w=44,
                          in_progress=True)
    assert _last_x(svg) > 40  # clamped to near the 44px right edge


def test_with_session_hours_the_same_elapsed_time_stays_partial():
    # Same ~8h of bars, but told the real window is ~23h -> should sit at
    # roughly 8/23 of the width, not clamped to full.
    svg = _intraday_spark(_bars(33, "2024-01-08 14:00"), 100.0, w=44,
                          in_progress=True, session_hours=23.0)
    last_x = _last_x(svg)
    assert 10 < last_x < 20, last_x  # 44 * 8h/23h ≈ 15.3px
