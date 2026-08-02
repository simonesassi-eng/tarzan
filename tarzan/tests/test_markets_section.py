"""The MARKETS section must show each instrument's trading hours and
whether its exchange is open right now, not just level/change/sparkline.

Before this, a reader saw a mix of markets on different session clocks
(US/Europe/Asia/futures/FX) with no stated hours, no way to tell whether a
level was live or from a closed session, and (for "Open"/"Closed" itself)
no stated day to pin the status to. Network-free: quotes are injected
directly, and market_status is monkeypatched so the open/closed/day
outcome does not depend on when the suite happens to run.
"""

from __future__ import annotations

from unittest import mock

import tarzan.data.market_quotes as mq
from tarzan.export.newsletter._constants import _NewsletterContext
from tarzan.export.newsletter._sections_perf import _build_markets
from tarzan.models.investor_config import InvestorConfig
from tarzan.models.portfolio import PortfolioMetrics


def _ctx() -> _NewsletterContext:
    return _NewsletterContext(
        metrics=PortfolioMetrics(total_value=1.0, invested_value=1.0, cash_value=0.0),
        config=InvestorConfig(), issue_number=1,
        benchmark_alpha_beta="S&P 500", benchmark_geo="MSCI ACWI",
    )


_SAMPLE = [
    {"name": "S&P 500", "symbol": "^GSPC", "category": "US",
     "value": 5000.0, "change": 10.0, "pct": 0.2,
     "spark": [4990.0, 5000.0], "baseline": 4990.0},
    {"name": "S&P 500 (FUT)", "symbol": "ES=F", "category": "US",
     "value": 5010.0, "change": 12.0, "pct": 0.24,
     "spark": [4998.0, 5010.0], "baseline": 4998.0},
]


def test_cash_index_shows_hours_open_badge_and_day():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           return_value=(True, "Tue")):
        html = _build_markets(_ctx())["html"]
    assert "09:30\u201316:00 ET" in html
    assert "&#9679;</span> Open Tue" in html


def test_cash_index_shows_closed_badge_and_day():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           return_value=(False, "Fri")):
        html = _build_markets(_ctx())["html"]
    assert "&#9679;</span> Closed Fri" in html


def test_chg_column_shows_only_percent_not_the_absolute_value():
    # Saves the width the day label now needs: +0.20% shows, the +10.0
    # points/€ that used to sit underneath it does not.
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status", return_value=(True, "Mon")):
        html = _build_markets(_ctx())["html"]
    assert "+0.20%" in html
    assert "+10.0" not in html
    assert "+12.0" not in html


def test_futures_show_open_status_and_day_not_the_cash_hours():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           side_effect=lambda sym, now=None: (
                               (True, "Tue") if sym == "^GSPC"
                               else (True, "Mon"))):
        html = _build_markets(_ctx())["html"]
    # "S&P 500 (FUT)" gets its own \u224824h line and its own day/status,
    # not the cash index's hours or its day, and the FUT tag is not
    # doubled by the render-time auto-suffix (name already carries it).
    fut_pos = html.index("S&P 500 (FUT)")
    fut_chunk = html[fut_pos:fut_pos + 250]
    assert "\u224824h" in fut_chunk
    assert "Open Mon" in fut_chunk
    assert "09:30" not in fut_chunk  # not the cash index's hours
    assert "(FUT) (FUT)" not in html


def test_futures_show_closed_with_day_over_the_weekend():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_status",
                           side_effect=lambda sym, now=None: (
                               (False, "Fri") if sym == "ES=F"
                               else (False, "Fri"))):
        html = _build_markets(_ctx())["html"]
    fut_pos = html.index("S&P 500 (FUT)")
    fut_chunk = html[fut_pos:fut_pos + 250]
    assert "Closed Fri" in fut_chunk
