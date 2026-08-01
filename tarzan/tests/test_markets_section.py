"""The MARKETS section must show each instrument's trading hours and
whether its exchange is open right now, not just level/change/sparkline.

Before this, a reader saw a mix of markets on different session clocks
(US/Europe/Asia/futures/FX) with no stated hours and no way to tell
whether a level was live or from a closed session. Network-free: quotes
are injected directly, and market_open_now is monkeypatched so the open/
closed outcome does not depend on when the suite happens to run.
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


def test_cash_index_shows_hours_and_open_badge():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_open_now", return_value=True):
        html = _build_markets(_ctx())["html"]
    assert "09:30\u201316:00 ET" in html
    assert "&#9679;</span> Open" in html


def test_cash_index_shows_closed_badge_when_market_open_now_is_false():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_open_now", return_value=False):
        html = _build_markets(_ctx())["html"]
    assert "&#9679;</span> Closed" in html


def test_futures_show_approx_24h_with_no_open_closed_badge():
    mq._memo = None
    with mock.patch.object(mq, "fetch_market_quotes", return_value=_SAMPLE), \
         mock.patch.object(mq, "market_open_now", return_value=True):
        html = _build_markets(_ctx())["html"]
    # "S&P 500 (FUT)" is rendered with its own \u224824h line, not the cash
    # index's hours/status, and the FUT tag is not doubled by the
    # render-time auto-suffix (name already carries it in MARKETS).
    fut_pos = html.index("S&P 500 (FUT)")
    fut_chunk = html[fut_pos:fut_pos + 200]
    assert "\u224824h" in fut_chunk
    assert "Open" not in fut_chunk and "Closed" not in fut_chunk
    assert "(FUT) (FUT)" not in html
