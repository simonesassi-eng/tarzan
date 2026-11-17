"""The fixed-mix line: the book's current weights held constant over its life.

Its weights are the book's at the PREVIOUS close, so over the last session on the
tape it walks exactly as the book did. Weighted on end-of-session values instead,
the lines that fell today weighed too little and the line read a basis point or
two better than the book on a day with nothing traded (the owner saw -0.80 beside
-0.81 and asked which was right).

Network-free: hand-built holdings with synthetic price tapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine.metrics import MetricsEngine
from tarzan.models.holding import Holding
from tarzan.models.investor_config import InvestorConfig

_IDX = pd.date_range("2026-01-05", periods=40, freq="B")


def _holding(ticker, prices, quantity):
    h = Holding(isin=f"IE{ticker:0<10}", ticker=ticker, quantity=quantity,
                cost_basis_eur=1.0, market_value_eur=1.0, currency="EUR")
    h.price_history = pd.Series(prices, index=_IDX)
    h.current_value = float(prices[-1]) * quantity
    h.current_price = float(prices[-1])
    return h


def _static(holdings):
    engine = MetricsEngine(list(holdings), InvestorConfig())
    out: dict = {}
    engine._static_history(out)
    return out.get("static_history")


def _book_nav(holdings):
    return sum(h.price_history * h.quantity for h in holdings)


class TestTheLastSessionIsTheBooks:
    def test_on_a_day_with_no_trades_the_last_step_equals_the_books(self):
        """Two lines, one falls 2.6% today and one rises 0.9%: the book's day is
        the start-of-day-weighted sum, and so must the fixed mix's be."""
        rng = np.random.default_rng(7)
        a = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, len(_IDX)))
        b = 50.0 * np.cumprod(1 + rng.normal(0, 0.01, len(_IDX)))
        a[-1], b[-1] = a[-2] * (1 - 0.026), b[-2] * (1 + 0.009)
        holdings = [_holding("AAA", a, 300.0), _holding("BBB", b, 500.0)]
        nav = _book_nav(holdings)
        book_day = nav.iloc[-1] / nav.iloc[-2] - 1.0
        line = _static(holdings)
        assert line is not None
        line_day = line.iloc[-1] / line.iloc[-2] - 1.0
        assert line_day == pytest.approx(book_day, abs=1e-12)
        # And it is NOT the end-of-session-weighted figure, which is the one
        # that read better than the book.
        r = pd.Series({"AAA": a[-1] / a[-2] - 1, "BBB": b[-1] / b[-2] - 1})
        w_end = pd.Series({"AAA": a[-1] * 300.0, "BBB": b[-1] * 500.0})
        end_weighted = float((w_end / w_end.sum() * r).sum())
        assert abs(end_weighted - book_day) > 1e-5
        assert line_day != pytest.approx(end_weighted, abs=1e-6)

    def test_the_weights_are_still_the_books_mix_to_within_the_days_move(self):
        """The previous-close weights differ from today's by the day's moves only;
        over the rest of the tape the line is the constant mix the owner asked for."""
        a = np.linspace(100.0, 120.0, len(_IDX))
        b = np.linspace(50.0, 45.0, len(_IDX))
        holdings = [_holding("AAA", a, 10.0), _holding("BBB", b, 10.0)]
        line = _static(holdings)
        rets = pd.DataFrame({"AAA": pd.Series(a, _IDX).pct_change(),
                             "BBB": pd.Series(b, _IDX).pct_change()}).dropna()
        w_prev = pd.Series({"AAA": a[-2] * 10.0, "BBB": b[-2] * 10.0})
        w_prev = w_prev / w_prev.sum()
        expect = (1.0 + (rets * w_prev).sum(axis=1)).cumprod() * 100.0
        pd.testing.assert_series_equal(line.iloc[1:], expect, check_names=False,
                                       check_freq=False, atol=1e-9)

    def test_a_line_without_a_tape_withholds_the_whole_series(self):
        a = np.linspace(100.0, 110.0, len(_IDX))
        held = _holding("AAA", a, 10.0)
        blind = _holding("BBB", a, 10.0)
        blind.price_history = None
        assert _static([held, blind]) is None
