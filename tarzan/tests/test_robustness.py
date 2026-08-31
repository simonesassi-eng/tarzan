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


class TestGarchStress:
    """The stress generator exists to reach drawdowns the bootstrap cannot."""

    @staticmethod
    def _garch_nav(n=3000, seed=11, omega=2e-6, alpha=0.12, beta=0.85):
        """A series drawn FROM a GARCH(1,1) process, so the fit is well specified."""
        rng = np.random.default_rng(seed)
        s2 = omega / (1 - alpha - beta)
        r = np.empty(n)
        e = 0.0
        for t in range(n):
            s2 = omega + alpha * e ** 2 + beta * s2
            e = np.sqrt(s2) * rng.normal()
            r[t] = 0.0004 + e
        return pd.Series(1.0 + r, index=pd.bdate_range("2005-01-03", periods=n)).cumprod()

    def test_finds_a_deeper_drawdown_tail_than_the_bootstrap(self):
        """The whole reason it exists: the bootstrap draws months independently
        and cannot build the persistent high-vol runs that deep drawdowns are
        made of."""
        nav = self._garch_nav()
        boot = rob.block_bootstrap(nav, n_sims=1500, horizon_days=15 * TRADING_DAYS)
        stress = rob.garch_stress(nav, n_sims=1500)
        assert stress, "fit must succeed on a GARCH series"
        assert stress["max_drawdown"]["p05"] < boot["max_drawdown"]["p05"], (
            f"stress tail {stress['max_drawdown']['p05']:.1f} should be deeper than "
            f"bootstrap {boot['max_drawdown']['p05']:.1f}")

    def test_leaves_the_return_distribution_alone(self):
        """It is a drawdown stress, not a competing view of expected return: the
        CAGR fields must stay in the bootstrap's neighbourhood or the framing is
        wrong.

        The bound is loose (20% relative, fixed seed for determinism) because on
        a single synthetic realisation the sampling noise dominates — it ranged
        0.3% to 15% across seeds. The substantive evidence is the real
        portfolios, where the two medians agree to 0.02-0.14 percentage points
        (8.41 vs 8.43 on the lead target). A small negative gap is EXPECTED and
        not a defect: clustering raises the variance of the cumulative return,
        and more variance at the same mean means a lower median compound return.
        """
        nav = self._garch_nav()
        boot = rob.block_bootstrap(nav, n_sims=2000, horizon_days=15 * TRADING_DAYS)
        stress = rob.garch_stress(nav, n_sims=2000)
        ref = abs(boot["cagr"]["median"])
        assert abs(stress["cagr"]["median"] - boot["cagr"]["median"]) < 0.20 * ref

    def test_a_price_never_goes_negative(self):
        """A resampled shock can exceed -100%; the month floors at total loss."""
        nav = self._garch_nav(omega=4e-5, alpha=0.20, beta=0.70)   # violent vol
        out = rob.garch_stress(nav, n_sims=1000)
        assert out["cagr"]["p05"] >= -100.0
        assert out["max_drawdown"]["p05"] >= -100.0

    def test_fit_is_stationary_and_reported(self):
        p = rob.garch_stress(self._garch_nav(), n_sims=500)["params"]
        assert 0.0 < p["persistence"] < 0.995
        assert p["alpha"] > 0 and p["beta"] > 0

    def test_short_history_returns_empty(self):
        assert rob.garch_stress(_nav(n=400), n_sims=100) == {}


class TestJointBootstrap:
    """Sleeve-level simulation: rebalancing becomes real, correlations become a knob."""

    @staticmethod
    def _sleeves(n=3000, seed=7, rho=0.2):
        """Three sleeves with a known pairwise correlation and DIFFERENT vols.

        Heterogeneity is the point: with interchangeable sleeves rebalancing has
        nothing to do, and the whole question is what weight drift costs.
        """
        rng = np.random.default_rng(seed)
        c = np.full((3, 3), rho); np.fill_diagonal(c, 1.0)
        z = rng.multivariate_normal(np.zeros(3), c, n)
        z = z * np.array([0.020, 0.009, 0.004]) + 0.0004      # racy / mid / calm
        idx = pd.bdate_range("2005-01-03", periods=n)
        df = pd.DataFrame(z, columns=["A", "B", "C"], index=idx)
        return df, pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    def test_rebalancing_narrows_the_outcome_of_unequal_sleeves(self):
        """Left alone, the raciest sleeve compounds into a bigger and bigger share
        of the portfolio, so the 15y outcome disperses more. Rebalancing keeps the
        risk profile it was designed with — the effect the blended-NAV bootstrap
        cannot show at all, since there are no sleeves left in it to drift."""
        df, w = self._sleeves()
        hold = rob.joint_bootstrap(df, w, rebalance="none", n_sims=400)
        rebal = rob.joint_bootstrap(df, w, rebalance="annual", n_sims=400)

        def band(d):
            return d["cagr"]["p95"] - d["cagr"]["p05"]

        assert band(rebal) < band(hold), f"{band(rebal):.2f} vs {band(hold):.2f}"
        assert rebal["max_drawdown"]["p05"] > hold["max_drawdown"]["p05"]

    def test_breaking_correlations_deepens_the_drawdown_tail(self):
        df, w = self._sleeves()
        base = rob.joint_bootstrap(df, w, n_sims=300)
        broken = rob.joint_bootstrap(df, w, n_sims=300, corr_shift=1.0)
        assert broken["mean_pair_corr"] > base["mean_pair_corr"] + 0.4
        assert broken["max_drawdown"]["p05"] < base["max_drawdown"]["p05"]

    def test_copula_at_zero_shift_reproduces_historical_correlation(self):
        """The knob must be calibrated: shift 0 has to land on the sample."""
        df, w = self._sleeves(rho=0.35)
        target = ((1 + df).resample("ME").prod() - 1).corr().values
        target = target[np.triu_indices(3, 1)].mean()
        out = rob.joint_bootstrap(df, w, n_sims=300, corr_shift=0.0)
        assert abs(out["mean_pair_corr"] - target) < 0.08

    def test_needs_at_least_two_sleeves_and_enough_months(self):
        df, w = self._sleeves()
        assert rob.joint_bootstrap(df[["A"]], w[["A"]], n_sims=50) == {}
        assert rob.joint_bootstrap(df.iloc[:200], w, n_sims=50) == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
