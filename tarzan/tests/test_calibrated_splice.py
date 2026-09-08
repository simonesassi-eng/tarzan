"""The calibrated backfill drives two decades of synthetic history off one beta,
so an attenuated beta is not a rounding error — it rebuilds that history at a
fraction of the true volatility and lets the intercept absorb the shortfall as
fabricated alpha. Non-synchronous closes (a EUR-listed fund at 17:30 CET vs a
US-dominated proxy at 22:00 CET) attenuate a DAILY fit badly, which is why the
calibration runs on monthly returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tarzan.engine.synthetic import calibrated_splice

TRADING_DAYS = 252


def _nonsynchronous(n: int = 1500, beta: float = 1.0, seed: int = 7):
    """A proxy and a fund that share one beta but observe it a day apart.

    The fund sees HALF of each day's proxy move on the day and the other half on
    the next — the timing split a 17:30 close sees of a 22:00-close basket. Over
    a month the split washes out, so the monthly beta is the true one while the
    daily beta is attenuated toward beta/2.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    proxy = pd.Series(rng.normal(0.0003, 0.012, n), index=idx)
    shifted = proxy.shift(1).fillna(0.0)
    fund = beta * (0.5 * proxy + 0.5 * shifted)
    return proxy, fund


def test_monthly_fit_recovers_beta_daily_would_halve_it():
    proxy, fund = _nonsynchronous(beta=1.0)
    # Reference: what a daily OLS reads off this overlap.
    x, y = proxy.values, fund.values
    b_daily = ((x - x.mean()) * (y - y.mean())).mean() / x.var()
    assert b_daily < 0.7, f"fixture must attenuate the daily beta, got {b_daily:.3f}"

    # The splice reconstructs the pre-inception tail; its volatility reveals the
    # beta actually used. A daily fit would rebuild it at ~b_daily × proxy vol.
    real = fund.iloc[len(fund) // 2:]
    out = calibrated_splice(proxy, real)
    pre = out.loc[out.index < real.index.min()]
    ratio = pre.std() / proxy.loc[pre.index].std()
    assert 0.85 <= ratio <= 1.15, f"backfill vol {ratio:.2f}x proxy, expected ~1x"
    assert ratio > b_daily + 0.15, "monthly fit must beat the attenuated daily one"


def test_intercept_is_not_inflated_into_fabricated_alpha():
    """A zero-alpha fund must not backfill with a large drift. The daily fit's
    shrunken beta forces the intercept to absorb the missing return; the monthly
    fit leaves it near zero."""
    proxy, fund = _nonsynchronous(beta=1.0)
    real = fund.iloc[len(fund) // 2:]
    out = calibrated_splice(proxy, real)
    pre = out.loc[out.index < real.index.min()]
    drift = (pre.mean() - proxy.loc[pre.index].mean()) * TRADING_DAYS
    assert abs(drift) < 0.02, f"backfill invents {drift * 100:.2f}%/yr of alpha"


def test_real_returns_are_untouched():
    proxy, fund = _nonsynchronous()
    real = fund.iloc[len(fund) // 2:]
    out = calibrated_splice(proxy, real)
    pd.testing.assert_series_equal(out.loc[real.index], real, check_names=False)


def test_falls_back_when_overlap_too_short_in_months():
    """Enough days but too few months → the naive splice, not a fit on noise."""
    proxy, fund = _nonsynchronous(n=800)
    real = fund.iloc[-300:]                      # ~14 months, over 252 days
    out = calibrated_splice(proxy, real, min_overlap=252, min_overlap_months=24)
    pre = out.loc[out.index < real.index.min()]
    pd.testing.assert_series_equal(pre, proxy.loc[pre.index], check_names=False)


def test_genuine_low_beta_survives():
    """The fix must not force beta to 1 — a true half-beta fund still reads ~0.5."""
    proxy, fund = _nonsynchronous(beta=0.5)
    real = fund.iloc[len(fund) // 2:]
    out = calibrated_splice(proxy, real)
    pre = out.loc[out.index < real.index.min()]
    ratio = pre.std() / proxy.loc[pre.index].std()
    assert 0.40 <= ratio <= 0.62, f"expected ~0.5x proxy vol, got {ratio:.2f}"


if __name__ == "__main__":
    test_monthly_fit_recovers_beta_daily_would_halve_it()
    test_intercept_is_not_inflated_into_fabricated_alpha()
    test_real_returns_are_untouched()
    test_falls_back_when_overlap_too_short_in_months()
    test_genuine_low_beta_survives()
    print("ok")
