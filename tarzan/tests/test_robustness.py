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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
