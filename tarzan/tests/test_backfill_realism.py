"""A backfill drives most of a 20+ year line off a few years of real overlap, so
a reconstruction that is too SMOOTH manufactures a risk-adjusted advantage the
fund never had. The guard exists because three such sleeves were found by hand,
each after it had already reversed an allocation conclusion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tarzan.backtest.engine import (
    BACKFILL_REALISM, _MIN_REAL_DAYS, _REALISM_BAND, _check_backfill_realism,
)


def _series(n, vol, start, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(0.0, vol, n), index=idx)


def _spliced(synth_vol, real_vol, n_synth=1500, n_real=600):
    """A spliced series whose synthetic tail and real head have chosen vols."""
    real = _series(n_real, real_vol, "2021-01-04", seed=5)
    tail = _series(n_synth, synth_vol, "2015-01-01", seed=6)
    tail = tail.loc[tail.index < real.index.min()]
    return pd.concat([tail, real]).sort_index(), real


def setup_function(_):
    BACKFILL_REALISM.clear()


def test_too_smooth_backfill_warns_and_records(caplog):
    spliced, real = _spliced(synth_vol=0.004, real_vol=0.010)   # ratio ~0.4
    with caplog.at_level(logging.WARNING, logger="backtest.engine"):
        _check_backfill_realism("XSMOOTH", spliced, real)
    rec = BACKFILL_REALISM["XSMOOTH"]
    assert rec["ratio"] < _REALISM_BAND[0]
    assert any("too SMOOTH" in r.message for r in caplog.records), caplog.text
    # The annualised numbers are what a reader acts on, so they must be right.
    assert abs(rec["vol_synth_pct"] - 0.004 * (252 ** 0.5) * 100) < 1.5
    assert abs(rec["vol_real_pct"] - 0.010 * (252 ** 0.5) * 100) < 1.5


def test_realistic_backfill_records_without_warning(caplog):
    spliced, real = _spliced(synth_vol=0.0102, real_vol=0.010)
    with caplog.at_level(logging.WARNING, logger="backtest.engine"):
        _check_backfill_realism("XOK", spliced, real)
    r = BACKFILL_REALISM["XOK"]["ratio"]
    assert _REALISM_BAND[0] <= r <= _REALISM_BAND[1], r
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING]


def test_overstated_risk_does_not_warn():
    """A synthetic tail MORE volatile than the fund hides nothing — the backtest
    is then pessimistic, which is the safe direction."""
    spliced, real = _spliced(synth_vol=0.020, real_vol=0.010)
    _check_backfill_realism("XLOUD", spliced, real)
    assert BACKFILL_REALISM["XLOUD"]["ratio"] > _REALISM_BAND[1]


def test_short_real_history_is_not_judged():
    """Under ~1 year of real data the real vol is too noisy to accuse a backfill."""
    spliced, real = _spliced(synth_vol=0.004, real_vol=0.010,
                             n_real=_MIN_REAL_DAYS - 50)
    _check_backfill_realism("XNEW", spliced, real)
    assert "XNEW" not in BACKFILL_REALISM


def test_no_synthetic_tail_is_not_judged():
    """An instrument whose real history covers the whole window has nothing to
    check — it must not be recorded as if it had passed."""
    real = _series(600, 0.010, "2021-01-04")
    _check_backfill_realism("XALLREAL", real.copy(), real)
    assert "XALLREAL" not in BACKFILL_REALISM


def test_missing_real_history_is_not_judged():
    spliced, _ = _spliced(synth_vol=0.004, real_vol=0.010)
    _check_backfill_realism("XNONE", spliced, None)
    _check_backfill_realism("XEMPTY", spliced, pd.Series(dtype=float))
    assert not BACKFILL_REALISM


if __name__ == "__main__":
    for fn in (test_too_smooth_backfill_warns_and_records,):
        pass
    print("run with pytest (needs caplog)")
