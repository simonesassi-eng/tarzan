"""Risk-free wiring guards for the backtest robustness sub-metrics.

The headline backtest Sharpe (``full_metrics``) already used the time-varying
risk-free path; ``rolling_sharpe_range`` and ``block_bootstrap`` used to hardcode
the flat 4% ``RISK_FREE_RATE``. These check that (a) the scalar fallback is
mathematically identical to the old flat-rate form, and (b) a real risk-free
path (e.g. ZIRP ~0%) actually moves the numbers — proving the wiring is live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarzan.engine import robustness as rob
from tarzan.engine.stats import RISK_FREE_RATE, TRADING_DAYS


def _nav(seed: int = 1, n: int = 600, mu: float = 0.0005, sd: float = 0.011):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(100.0 * np.cumprod(1 + rng.normal(mu, sd, n)), index=idx)


class TestRollingSharpeRange:
    def test_none_matches_old_flat_rate_formula(self):
        """rf_daily=None must reproduce the pre-wiring flat-rate Sharpe exactly."""
        nav = _nav()
        new = rob.rolling_sharpe_range(nav, 252)
        r = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        roll = r.rolling(252)
        ann_ret = roll.mean() * TRADING_DAYS * 100.0
        ann_vol = roll.std() * np.sqrt(TRADING_DAYS) * 100.0
        old = ((ann_ret - RISK_FREE_RATE) / ann_vol
               ).replace([np.inf, -np.inf], np.nan).dropna()
        for k, v in {"min": old.min(), "median": np.median(old.values),
                     "max": old.max()}.items():
            assert new[k] == pytest.approx(float(v))

    def test_zirp_path_lifts_sharpe(self):
        """A ~0% real risk-free (vs flat 4%) subtracts less → higher Sharpe."""
        nav = _nav()
        r = nav.pct_change().dropna()
        rf0 = pd.Series(0.0, index=r.index)
        flat = rob.rolling_sharpe_range(nav, 252)
        tv = rob.rolling_sharpe_range(nav, 252, rf_daily=rf0)
        assert tv["median"] > flat["median"]


class TestBlockBootstrap:
    def test_rf_annual_none_defaults_to_scalar(self):
        nav = _nav()
        a = rob.block_bootstrap(nav, n_sims=400)
        b = rob.block_bootstrap(nav, n_sims=400, rf_annual=RISK_FREE_RATE)
        # Same seed + same rate → identical CIs.
        assert a["sharpe"] == b["sharpe"]

    def test_zirp_rate_lifts_bootstrap_sharpe(self):
        nav = _nav()
        flat = rob.block_bootstrap(nav, n_sims=400)
        zirp = rob.block_bootstrap(nav, n_sims=400, rf_annual=0.0)
        assert zirp["sharpe"]["median"] > flat["sharpe"]["median"]

    def test_horizon_band_is_stable_in_block_length(self):
        """The whole point of resampling monthly: no arbitrary knob left.

        On daily returns the 15y band swung from 14.4 to 10.6 points as the
        block grew from 1 to 126 days, so the answer hung on the block size.
        Monthly aggregation removes the short-run noise that drove that, so the
        band must now be essentially flat in the block length.
        """
        nav = _nav(n=3000)
        bands = []
        for months in (1, 3, 6, 12):
            d = rob.block_bootstrap(nav, n_sims=1500, block_months=months,
                                    horizon_days=10 * TRADING_DAYS)
            bands.append(d["cagr"]["p95"] - d["cagr"]["p05"])
        spread = max(bands) - min(bands)
        assert spread < 0.15 * max(bands), f"band not stable in block length: {bands}"

    def test_daily_block_days_still_accepted(self):
        """Legacy callers pass block_days; it must map onto whole months."""
        nav = _nav(n=3000)
        legacy = rob.block_bootstrap(nav, n_sims=300, block_days=21)
        native = rob.block_bootstrap(nav, n_sims=300, block_months=1)
        assert legacy["cagr"] == native["cagr"]

    def test_too_short_history_returns_empty(self):
        assert rob.block_bootstrap(_nav(n=200), n_sims=100) == {}


class TestVolatilityFrequency:
    """Vol/Sharpe/Sortino read MONTHLY returns; the reconstruction's daily
    frequency carries non-synchronous pricing noise that inflates variance and
    cancels on aggregation."""

    @staticmethod
    def _noisy_nav(n=3000, seed=5, noise=0.004):
        """A clean series plus iid daily measurement noise that cancels monthly.

        Adding and removing the same shock on consecutive days leaves month-end
        levels untouched while inflating daily variance — exactly the signature
        of a stale/non-synchronous price.
        """
        rng = np.random.default_rng(seed)
        true_r = rng.normal(0.0004, 0.006, n)
        e = rng.normal(0.0, noise, n)
        obs = true_r + e - np.concatenate([[0.0], e[:-1]])
        idx = pd.bdate_range("2005-01-03", periods=n)
        return (pd.Series(1.0 + obs, index=idx).cumprod(),
                pd.Series(1.0 + true_r, index=idx).cumprod())

    def test_monthly_vol_rejects_noise_daily_vol_absorbs_it(self):
        noisy, clean = self._noisy_nav()

        def _daily_vol(nav):
            return nav.pct_change().dropna().std() * np.sqrt(TRADING_DAYS) * 100

        daily_vol = _daily_vol(noisy)
        reported = rob.full_metrics(noisy)["volatility"]
        truth = rob.full_metrics(clean)["volatility"]
        assert daily_vol > 1.3 * _daily_vol(clean), (
            "test setup must inject visible daily noise")
        assert abs(reported - truth) < 0.15 * truth, (
            f"monthly vol {reported:.2f} should track the noise-free {truth:.2f}, "
            f"not the daily {daily_vol:.2f}")

    def test_sharpe_is_not_depressed_by_daily_noise(self):
        noisy, clean = self._noisy_nav()
        assert rob.full_metrics(noisy)["sharpe"] > 0.8 * rob.full_metrics(clean)["sharpe"]

    def test_max_drawdown_stays_on_the_daily_path(self):
        """MaxDD must keep daily granularity: a month-end version would erase
        real intra-month crashes."""
        idx = pd.bdate_range("2020-01-01", periods=400)
        r = np.full(400, 0.0004)
        r[100:105] = -0.06                      # a crash fully inside one month
        r[105:110] = +0.065                     # recovered before month end
        nav = pd.Series(1.0 + r, index=idx).cumprod()
        assert rob.full_metrics(nav)["max_drawdown"] < -20.0

    def test_short_window_keeps_the_daily_estimate(self):
        nav = _nav(n=150)                        # ~7 months: too few for monthly
        assert rob.full_metrics(nav)["volatility"] == pytest.approx(
            nav.pct_change().dropna().std() * np.sqrt(TRADING_DAYS) * 100)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
