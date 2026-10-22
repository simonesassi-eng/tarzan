"""Daily data decisions cannot change calendar halfway through one run."""

from dataclasses import replace
from datetime import date, datetime, timezone

import pandas as pd

from tarzan import runtime
from tarzan.data import enricher as E


def test_live_tail_fill_uses_the_captured_analysis_day_not_the_host_clock():
    context = replace(
        runtime.context(), captured_at=datetime(2025, 1, 15, 12, tzinfo=timezone.utc),
    )
    with runtime.bind_context(context):
        assert E._today_local() == context.analysis_date == date(2025, 1, 15)


def test_pinned_daily_clock_stays_at_the_effective_boundary():
    context = runtime.configure(as_of=date(2024, 3, 12))
    with runtime.bind_context(context):
        assert E._today_local() == date(2024, 3, 12)


def test_stale_history_age_is_measured_at_capture_not_a_later_wall_clock(monkeypatch):
    context = replace(
        runtime.context(), captured_at=datetime(2025, 1, 15, 12, tzinfo=timezone.utc),
    )
    cached = pd.DataFrame(
        {"Close": [10.0, 11.0], "Price Close": [10.0, 11.0]},
        index=pd.date_range("2025-01-13", periods=2, tz="UTC"),
    )
    warnings = []
    monkeypatch.setattr(E.price_cache, "load_history", lambda symbol: cached)
    monkeypatch.setattr(E.price_cache, "refresh_start", lambda frame: None)
    monkeypatch.setattr(E.price_cache, "store_history", lambda *args: None)
    monkeypatch.setattr(E.price_cache, "repair_split_jumps", lambda frame, **kwargs: frame)
    monkeypatch.setattr(E, "_retry", lambda *args, **kwargs: None)
    monkeypatch.setattr(E, "_fill_today_from_sibling", lambda symbol, frame: frame)
    monkeypatch.setattr(E.dq, "warning", lambda *args, **kwargs: warnings.append(args))
    with runtime.bind_context(context):
        E._fetch_history_uncached("SYNTHETIC.X")
    assert warnings == []
