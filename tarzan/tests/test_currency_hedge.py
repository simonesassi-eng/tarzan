"""A EUR-hedged share class does not earn the currency move.

The proxies arrive already converted into the reporting currency, so a hedged
class has to have that conversion divided back out and the interest differential
added in its place — covered interest parity. Getting this wrong gave the hedged
trend sleeve six points of volatility it does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.synthetic import replicate_portfolio_returns


def _legs(n=2000, seed=3, fx_vol=0.007, carry=-0.76):
    """FX return path and a constant daily rate differential (annual percent)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    fx = pd.Series(rng.normal(0.0, fx_vol, n), index=idx)
    diff = pd.Series(carry / 100.0 / 252.0, index=idx)
    return idx, fx, diff


def test_hedging_strips_the_currency_variance():
    """The whole point: an unhedged EUR class carries FX vol, a hedged one does not."""
    idx, fx, diff = _legs()
    rng = np.random.default_rng(11)
    usd = pd.Series(rng.normal(0.0002, 0.005, len(idx)), index=idx)
    # What the pipeline hands over: the USD strategy already converted to EUR.
    eur = (1.0 + usd) * (1.0 + fx) - 1.0

    unhedged = replicate_portfolio_returns({"X": 1.0}, {"X": eur})
    hedged = replicate_portfolio_returns({"X": 1.0}, {"X": eur},
                                        hedge_fx=fx, hedge_carry=diff)
    ann = lambda s: float(s.std()) * np.sqrt(252) * 100
    assert ann(unhedged) > 1.5 * ann(hedged), (
        f"unhedged {ann(unhedged):.2f}% should carry the FX vol that hedged "
        f"{ann(hedged):.2f}% does not")
    # And it must land back on the underlying strategy's own volatility.
    assert abs(ann(hedged) - ann(usd)) < 0.3


def test_hedging_charges_the_rate_differential():
    """Return-wise it is not free: the hedge pays (or earns) the differential."""
    idx, fx, diff = _legs(carry=-2.0)
    flat = pd.Series(0.0, index=idx)          # no strategy return, no FX move
    out = replicate_portfolio_returns({"X": 1.0}, {"X": flat},
                                      hedge_fx=flat, hedge_carry=diff)
    realised = ((1.0 + out).prod() ** (252 / len(out)) - 1.0) * 100
    assert realised < -1.8, f"a -2%/yr differential must show up, got {realised:.2f}"


def test_one_leg_alone_is_ignored():
    """Half a hedge would be a wrong hedge, so both legs are required."""
    idx, fx, diff = _legs()
    r = pd.Series(0.0003, index=idx)
    base = replicate_portfolio_returns({"X": 1.0}, {"X": r})
    assert replicate_portfolio_returns({"X": 1.0}, {"X": r}, hedge_fx=fx).equals(base)
    assert replicate_portfolio_returns({"X": 1.0}, {"X": r}, hedge_carry=diff).equals(base)


def test_hedge_applies_after_the_financing_charge():
    """Order matters: leverage is financed on the unhedged exposure, then the
    share class is hedged — not the other way round."""
    idx, fx, diff = _legs()
    r = pd.Series(0.0004, index=idx)
    fin = pd.Series(0.0001, index=idx)
    lev = replicate_portfolio_returns({"X": 2.0}, {"X": r}, financing_daily=fin,
                                      hedge_fx=fx, hedge_carry=diff)
    unlev = replicate_portfolio_returns({"X": 1.0}, {"X": r}, financing_daily=fin,
                                        hedge_fx=fx, hedge_carry=diff)
    assert lev.mean() > unlev.mean()


def test_the_hedged_list_is_explicit_not_sniffed():
    """The provider calls MFEH '...R EUR HP UCITS ETF' — no 'EUR Hedged' in it —
    so a name heuristic silently matches nothing and the list must be explicit."""
    from tarzan.backtest.engine import _HEDGED_TICKERS
    assert "MFEH" in _HEDGED_TICKERS
    assert "DBMFE" not in _HEDGED_TICKERS


class TestWeightText:
    """A printed weight recap has to add up: unlike a numeric cell, text cannot
    be un-rounded later."""

    def test_half_weights_survive(self):
        """The bug this exists for: Python rounds half to EVEN, so "%.0f" turned a
        7.5/7.5 trend split into "8 · 8" and the recap line read 101 for a
        portfolio that sums to 100."""
        from tarzan.export.whatif_excel import _wtxt
        assert _wtxt(7.5) == "7.5"
        assert _wtxt(8.5) == "8.5"

    def test_whole_weights_stay_clean(self):
        from tarzan.export.whatif_excel import _wtxt
        assert _wtxt(35.0) == "35"
        assert _wtxt(5.0) == "5"

    def test_a_real_book_adds_up(self):
        from tarzan.export.whatif_excel import _wtxt
        book = {"NTSG": 35.0, "AVWC": 10.0, "AVWS": 10.0, "SGLD": 10.0,
                "CL2": 8.0, "DBMFE": 7.5, "MFEH": 7.5, "AVEM": 7.0, "UEQC": 5.0}
        assert sum(float(_wtxt(w)) for w in book.values()) == 100.0
